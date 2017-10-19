from keras.models import Sequential
from keras.layers import Input,Dense,MaxPooling2D,Flatten
from keras.layers.convolutional import Conv2D
import numpy as np
import matplotlib.image as image
import random
from skimage import transform
from keras.optimizers import RMSprop
from collections import deque
import socket
#state is define as [Y_image,Depth_image]
#settings
debug = True

input_shape=(198,198,8)
outpur_shape=4
n_s=[16,32,64,256] #number of filters
fil_size=[9,5] #filter size
st=[3,1] #strides
p_s=2 #pool_size,pool_strides
r_len=8#recording times per second 8

class DQNAgent:
    Y_model_path='models/ymodel'#+episode.h
    D_model_path='models/dmodel'
    memory_path='memory/'
    reward_file=memory_path+'reward.dat'
    action_file=memory_path+'action.dat'
    ep_reward_file=memory_path+'ep_reward.dat'
    batch_size = 25
    epsilon = 1
    epsilon_decay = 0.99
    epsilon_final = 0.1
    epsilon_endtime = 30000
    action_size = 4
    discount_factor = 0.99
    n_replay = 1 #replay per learning step
    learn_start = 3000
    replay_memory = 30000
    memory = deque(maxlen=replay_memory)
    clip_delta = 1
    def __init__(self):
        self.Y_model = self.build_model()
        self.targer_Y_model = self.build_model()
        self.D_model = self.build_model()
        self.targer_D_model = self.build_model()
        # self.load_model()
        # self.load_memory()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(filters=n_s[0],kernel_size=(fil_size[0],fil_size[0]),strides=(st[0],st[0]),padding='valid',input_shape=input_shape,activation='relu'))
        model.add(MaxPooling2D(pool_size=(p_s,p_s),strides=(p_s,p_s),padding='same'))
        model.add(Conv2D(filters=n_s[1],kernel_size=(fil_size[1],fil_size[1]),strides=(st[1],st[1]),padding='valid',activation='relu'))
        model.add(MaxPooling2D(pool_size=(p_s,p_s),strides=(p_s,p_s),padding='same'))
        model.add(Conv2D(filters=n_s[2],kernel_size=(fil_size[1],fil_size[1]),strides=(st[1],st[1]),padding='valid',activation='relu'))
        model.add(MaxPooling2D(pool_size=(p_s,p_s),strides=(p_s,p_s),padding='same'))
        model.add(Flatten())
        model.add(Dense(n_s[3],activation='relu'))
        model.add(Dense(outpur_shape))
        model.compile(loss='mse',optimizer='rmsprop')
        return model

    def update_targer_model(self):
        self.Y_model.set = self.targer_Y_model
        self.D_model = self.targer_D_model

    def get_action(self,state):
        '''get action without epsilon greedy'''
        res1 = self.Y_model.predict(state[:1])
        res2 = self.D_model.predict(state[1:])
        q = res1 + res2
        return np.argmax(q[0])+1

    def e_get_action(self,state):
        if self.epsilon > self.epsilon_final:
            self.epsilon = self.epsilon-(1-self.epsilon_final)/self.epsilon_endtime
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)+1
        return get_action(state)


    def get_data(self,episode,step):
        path = '../dataset'
        shape=[1,]
        shape.extend(input_shape)
        images = np.zeros(shape)
        depths = np.zeros(shape)
        # for step in range(0,steps):
        for i in range(1,r_len+1):
            y_image_path = path+'/RGB/ep'+str(episode)+'/image_' + str(step+1) + '_' + str(i) + '.png'
            d_image_path = path+'/Depth/ep'+str(episode)+'/depth_' + str(step+1) +'_' + str(i) +'.png'
            y_image = image.imread(y_image_path)
            d_image = image.imread(d_image_path)
            y_image = transform.resize(y_image,(198,198),mode='reflect')
            d_image = transform.resize(d_image,(198,198),mode='reflect')
            images[0,:,:,i-1]=y_image
            depths[0,:,:,i-1]=d_image
        return images,depths

    def memorize(self,state,action,reward,n_state):
        self.memory.add(state,action,reward,n_state)

    def memory_replay(self):
        batch = min(len(self.memory),self.batch_size)
        for i in range(0,len(self.memory),self.batch_size):
            mini_batch = random.sample(list(self.memory),batch)
            self.train_Y_model(mini_batch)
            self.train_D_model(mini_batch)

    def train_Y_model(self,mini_batch):
        batch_size = len(mini_batch)
        shape=[batch_size,]
        shape.extend(input_shape)
        update_input = np.zeros(shape)
        update_target = np.zeros((batch_size, self.action_size))
        for i in range(batch_size):
            state,action,reward,n_state,terminal = mini_batch[i]
            target = self.Y_model.predict(state[:1])[0]
            action = int(action-1)
            # target = np.zeros(self.action_size)
            if terminal:
                target[action] = reward
            else:
                q_2 = np.amax(self.targer_Y_model.predict(n_state[:1])[0])
                target[action] = reward + self.discount_factor*q_2
            if self.clip_delta:
                if target[action]> self.clip_delta:
                    target[action] = self.clip_delta
                elif target[action] < -self.clip_delta:
                    target[action] = -self.clip_delta
            update_input[i]=state[0]
            update_target[i] = target

        self.Y_model.fit(update_input,update_target,batch_size=self.batch_size,epochs=5)

    def train_D_model(self,mini_batch):
        batch_size = len(mini_batch)
        shape=[batch_size,]
        shape.extend(input_shape)
        update_input = np.zeros(shape)
        update_target = np.zeros((batch_size, self.action_size))
        for i in range(batch_size):
            state,action,reward,n_state,terminal = mini_batch[i]
            action = int(action-1)
            target = self.D_model.predict(state[1:])[0]
            # target = np.zeros(self.action_size)

            if terminal:
                target[action] = reward
            else:
                target[action] = reward + self.discount_factor*np.amax(self.targer_D_model.predict(n_state[1:])[0])
            if self.clip_delta:
                if target[action]> self.clip_delta:
                    target[action] = self.clip_delta
                elif target[action] < -self.clip_delta:
                    target[action] = -self.clip_delta
            update_input[i]=state[1]
            update_target[i] = target

        self.D_model.fit(update_input,update_target,batch_size=self.batch_size,epochs=5)

    def load_model(self,episode):
        if episode==0:
            return
        episode=str(episode)
        self.Y_model.load_weights(self.Y_model_path+episode+'.h5')
        self.D_model.load_weights(self.D_model_path+episode+'.h5')
        self.update_targer_model()

    def save_model(self,episode):
        episode=str(episode)
        self.Y_model.save_weights(self.Y_model_path+episode+'.h5')
        self.D_model.save_weights(self.D_model_path+episode+'.h5')

    def save_memory(self):
        pass

    def memorize(self,s,a,r,n,t):
        self.memory.append((s,a,r,n,t))

    def load_memory_of_episode(self,episode):
        #load images rewards actions terminals
        reward = read_dat_file(self.reward_file)
        action = read_dat_file(self.action_file)
        # if len(reward.shape) == 1:
        #     steps = reward.shape[0]
        # else:
        steps = len(reward[episode-1])
        for step in range(steps):
            y,d = self.get_data(episode,step+1)
            state = np.concatenate((y,d),axis=0)
            terminal=True
            if step < steps-1:
                y_,d_ = self.get_data(episode,step+1)
                next_state=np.concatenate((y_,d_),axis=0)
                terminal=False
            self.memorize(state,action[episode-1][step],reward[episode-1][step],next_state,terminal)
            debug_print('memory of episode %d step %d loaded'%(episode,step+1))


def read_dat_file(path):
    f = open(path)
    lines = f.readlines()
    x = []
    for i in range(len(lines)):
        xi = lines[i]
        xi = xi[:-1].split(',')
        xi = list(map(int,xi))
        x.append(xi)
    return x
def debug_print(word):
    if debug:
        print(word)

def main():
    pass

if __name__ == '__main__':
    main()
