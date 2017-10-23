from agent import DQNAgent
import numpy as np
import socket
import sys
import time
t_steps = 40
host ='localhost'
port=12375
actiong_dict={1:"wait",2:"look toward human",3:"hello",4:"shake hand",9:"start"}

memory_path='memory/'
reward_file=memory_path+'reward.dat'
action_file=memory_path+'action.dat'



def send_action(action):
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect((host,port))
    s.send(str(action).encode())
    print(time.strftime('%Y/%m/%d-%H:%M:%S',time.localtime(time.time())),' send action: ',actiong_dict[int(action)])
    data = s.recv(1024).decode()
    print(time.strftime('%Y/%m/%d-%H:%M:%S',time.localtime(time.time())),' recieve reward: ',data)
    s.close()
    return data[0]
def main():
    if len(sys.argv) >1:
        host = sys.argv[1]
    epi_file=open('../files/episode.txt')
    episode = epi_file.readline()
    epi_file.close()
    qagent=DQNAgent(2)
    data = 'x'
    while(data!='9'):
        data = send_action(9)
    ys,ds=qagent.get_data(episode,0)
    state = np.concatenate((ys,ds),axis=0)

    for step in range(1,t_steps+1):
        action = qagent.get_action(state)
        # action = qagent.get_action(state)
        reward = send_action(action)
        ys,ds = qagent.get_data(episode,step)
        n_state = np.concatenate((ys,ds),axis=0)
        state = n_state
    #save-actions，rewards
if __name__ == '__main__':
    main()
