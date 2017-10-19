from agent import DQNAgent
import numpy as np
import socket
import sys
import time

t_steps = 20
host ='localhost'
port=12375

memory_path='memory/'
reward_file=memory_path+'reward.dat'
action_file=memory_path+'action.dat'



def send_action(action):
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect((host,port))
    s.send(str(action).encode())
    print(time.strftime('%Y/%m/%d-%H:%M:%S',time.localtime(time.time())),' send action: ',action)
    data = s.recv(1024).decode()
    print(time.strftime('%Y /%m/%d-%H:%M:%S',time.localtime(time.time())),' recieve reward: ',data)
    s.close()
    return data[0]
def main():
    if len(sys.argv) >1:
        host = sys.argv[1]
    ep_reward_file=memory_path+'ep_reward.dat'
    epi_file=open('../files/episode.txt')
    episode = epi_file.read(1)
    epi_file.close()
    qagent=DQNAgent()
    data = 'x'
    while(data!='9'):
        data = send_action(9)
    ys,ds=qagent.get_data(episode,1)
    state = np.concatenate((ys,ds),axis=0)
    actions=[]
    rewards=[]
    for step in range(1,t_steps+1):
        action = qagent.e_get_action(state)
        # action = qagent.get_action(state)
        reward = send_action(action)
        ys,ds = qagent.get_data(episode,step)
        n_state = np.concatenate((ys,ds),axis=0)
        actions.append(action)
        rewards.append(reward)
        state = n_state
    #save-actions，rewards
    actions = map(str,actions)
    rewards = map(str,rewards)
    r_file = open(reward_file,'a')
    a_file = open(action_file,'a')
    r_str = ','.join(rewards)
    a_str = ','.join(actions)
    r_file.write(r_str+'\n')
    a_file.write(a_str+'\n')
    r_file.close()
    a_file.close()
    print("episode : ",episode," finished.")
    episode=str(int(episode)+1)
    epi_file=open('../files/episode.txt','w')
    epi_file.write(episode)
    epi_file.close()
if __name__ == '__main__':
    main()
