import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.image as image
import random
from skimage import transform
from collections import deque
import socket
#state is define as [Y_image,Depth_image]
#settings
debug = True

input_shape=(8,198,198)
output_shape=4
n_s=[16,32,64,256] #number of filters
fil_size=[9,5] #filter size
fsl=5#final side length
st=[3,1] #strides
p_s=2 #pool_size,pool_strides
r_len=8#recording times per second 8
cuda=torch.cuda.is_available()
loss_func = nn.MSELoss()
class DQN(nn.Module):
    def __init__(self,):
        super(DQN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(r_len,n_s[0],kernel_size=fil_size[0],stride = st[0]),
            nn.ReLU(),
            nn.MaxPool2d(p_s,p_s),
        )
        self.conv2 =nn.Sequential(
            nn.Conv2d(n_s[0],n_s[1],kernel_size=fil_size[1],stride = st[1]),
            nn.ReLU(),
            nn.MaxPool2d(p_s,p_s),
        )
        self.conv3 =nn.Sequential(
            nn.Conv2d(n_s[1],n_s[2],kernel_size=fil_size[1],stride = st[1]),
            nn.ReLU(),
            nn.MaxPool2d(p_s,p_s),
        )
        self.L1 = nn.Linear(n_s[2]*fsl*fsl,n_s[3])
        # self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(n_s[3],output_shape)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.L1(x))
        # x = self.drop(x)
        return self.out(x)



class DQNAgent:
    Y_model_path='models/ymodel'
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
    discount_factor = 0.7
    n_replay = 1 #replay per learning step
    learn_start = 3000
    replay_memory = 30000
    memory = deque(maxlen=replay_memory)
    clip_delta = 0

    def __init__(self,episode=0):
        if episode==0:
            self.Y_model = self.build_model()
            self.target_Y_model = self.build_model()
            # self.D_model = self.build_model()
            # self.target_D_model = self.build_model()
        else:
            self.load_model(episode)
        self.Y_model.double()
        # self.D_model.double()
        self.target_Y_model.double()
        # self.target_D_model.double()
        if cuda:
            self.Y_model.cuda()
            self.target_Y_model.cuda()
            # self.D_model.cuda()
            # self.target_D_model.cuda()
        self.Y_optimizer = torch.optim.RMSprop(self.Y_model.parameters(),1e-4)
        # self.D_optimizer = torch.optim.RMSprop(self.D_model.parameters(),1e-4)


    def build_model(self):
        return DQN()

    def update_target_model(self):
        self.target_Y_model.load_state_dict(self.Y_model.state_dict())
        # self.target_D_model.load_state_dict(self.D_model.state_dict())
    def get_action(self,state):
        '''get action without epsilon greedy'''
        if cuda:
            ystate = Variable(torch.from_numpy(state[:1]).cuda())
            dstate = Variable(torch.from_numpy(state[1:]).cuda())
        else:
            ystate = Variable(torch.from_numpy(state[:1]))
            dstate = Variable(torch.from_numpy(state[1:]))
        res1 = self.Y_model.forward(ystate)
        # res2 = self.D_model.forward(dstate)
        q = res1 #+ res2
        if cuda:
            q=q[0].cpu().data.numpy()
        else:
            q=q[0].data.numpy()
        print("wait:%f look toward:%f hello:%f shake hand: %f"%(q[0],q[1],q[2],q[3]))
        act = np.argmax(q)
        return act+1

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
            images[0,i-1,:,:]=y_image
            depths[0,i-1,:,:]=d_image
        return images,depths

    def memorize(self,state,action,reward,n_state):
        self.memory.add(state,action,reward,n_state)

    def memory_replay(self):
        # batch = min(len(self.memory),self.batch_size)
        # mini_batch = random.sample(list(self.memory),batch)
        qy,yloss=self.train_Y_model(self.sample_memory())
        # qd=self.train_D_model(mini_batch)
        qd=0
        return qy,qd,yloss

    def sample_memory(self):
        return random.sample(self.memory,self.batch_size)

    def train_Y_model(self,batchMem):
        memsize=len(batchMem)
        batch_size = self.batch_size
        shape=[memsize,]
        shape.extend(input_shape)
        update_input = np.zeros(shape)
        update_target = np.zeros((memsize, self.action_size))
        for i in range(memsize):
            state,action,reward,n_state,terminal = batchMem[i]
            if cuda:
                ystate = Variable(torch.from_numpy(state[:1]).cuda())
                nstate = Variable(torch.from_numpy(n_state[:1]).cuda())
            else:
                ystate = Variable(torch.from_numpy(state[:1]))
                nstate = Variable(torch.from_numpy(n_state[:1]))
            target = self.Y_model.forward(ystate).cpu().data.numpy()[0]
            print('action: ',action,' reward: ',reward )
            print('target before:',target)
            action = int(action)-1
            # target = np.zeros(self.action_size)
            if terminal:
                target[action] = reward
            else:
                q_2 =self.target_Y_model.forward(nstate)
                q_2 = torch.max(q_2).cpu().data.numpy()
                target[action] = reward + self.discount_factor*q_2
            print('max Q_n:'q_2)
            print('target after:',target)
            if self.clip_delta:
                if target[action]> self.clip_delta:
                    target[action] = self.clip_delta
                elif target[action] < -self.clip_delta:
                    target[action] = -self.clip_delta
            update_input[i]=state[0]
            update_target[i] = target
        if cuda:
            update_input=Variable(torch.from_numpy(update_input).cuda())
            update_target=Variable(torch.from_numpy(update_target).cuda())
        else:
            update_input=Variable(torch.from_numpy(update_input))
            update_target=Variable(torch.from_numpy(update_target))
        prediction=self.Y_model.forward(update_input)
        loss = loss_func(prediction,update_target)

        self.Y_optimizer.zero_grad()
        loss.backward()
        self.Y_optimizer.step()
        return np.mean(update_target.cpu().data.numpy()),np.mean(loss.cpu().data.numpy())

    def train_D_model(self,batchMem):#todo cuda capablity
        batch_size = self.batch_size
        memsize=len(batchMem)
        shape=[memsize,]
        shape.extend(input_shape)
        update_input = np.zeros(shape)
        update_target = np.zeros((memsize, self.action_size))
        for i in range(memsize):
            state,action,reward,n_state,terminal = batchMem[i]
            action = int(action)-1
            dstate = Variable(torch.from_numpy(state[1:]))
            target = self.D_model.forward(dstate).data.numpy()[0]
            # target = np.zeros(self.action_size)

            if terminal:
                target[action] = reward
            else:
                nstate = Variable(torch.from_numpy(n_state[1:]))
                target[action] = reward + self.discount_factor*torch.max(self.target_D_model.forward(nstate).data)
            if self.clip_delta:
                if target[action]> self.clip_delta:
                    target[action] = self.clip_delta
                elif target[action] < -self.clip_delta:
                    target[action] = -self.clip_delta
            update_input[i]=state[1]
            update_target[i] = target
        update_input=Variable(torch.from_numpy(update_input))
        update_target=Variable(torch.from_numpy(update_target))
        prediction=self.D_model.forward(update_input)
        loss = loss_func(prediction,update_target)
        print(np.mean(loss.data.numpy()))
        self.D_optimizer.zero_grad()
        loss.backward()
        self.D_optimizer.step()
        return np.mean(update_target.data.numpy())


    def load_model(self,episode):
        if episode==0:
            return
        episode=str(episode)
        if cuda:
            self.Y_model=torch.load(self.Y_model_path+episode+'_gpu.pkl')
            self.target_Y_model=torch.load(self.Y_model_path+episode+'_gpu.pkl')
            # self.D_model=torch.load(self.D_model_path+episode+'.pkl')
        else:
            self.Y_model=torch.load(self.Y_model_path+episode+'.pkl')
            self.target_Y_model=torch.load(self.Y_model_path+episode+'.pkl')
        self.update_target_model()

    def save_model(self,episode):
        episode=str(episode)
        if cuda:
            torch.save(self.Y_model,self.Y_model_path+episode+'_gpu.pkl')
            # torch.save(self.D_model,self.D_model_path+episode+'_gpu.pkl')
        torch.save(self.Y_model.cpu(),self.Y_model_path+episode+'.pkl')

    def save_memory(self):
        pass

    def memorize(self,s,a,r,n,t):
        self.memory.append((s,a,r,n,t))

    def load_memory_of_episode(self,episode):
        #load images rewards actions terminals
        self.memory = deque(maxlen=self.replay_memory)
        reward = read_dat_file(self.reward_file)
        action = read_dat_file(self.action_file)
        # if len(reward.shape) == 1:
        #     steps = reward.shape[0]
        # else:
        steps = len(reward[episode-1])
        debug_print("step:"+str(steps))
        for step in range(steps):
            y,d = self.get_data(episode,step)
            state = np.concatenate((y,d),axis=0)
            terminal=True
            if step < steps-1:
                y_,d_ = self.get_data(episode,step+1)
                next_state=np.concatenate((y_,d_),axis=0)
                terminal=False
            r = reward[episode-1][step]
            if(r>3):
                r = 1
                # terminal = True
            elif (r<0):
                r = -0.1
            else:
                r = 0
            self.memorize(state,action[episode-1][step],r,next_state,terminal)
            debug_print('memory of episode %d step %d loaded'%(episode,step+1))


def read_dat_file(path):
    f = open(path)
    lines = f.readlines()
    x = []
    f.close()
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
