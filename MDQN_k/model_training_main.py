from agent import DQNAgent
import numpy as np
import time
#memory structure
#___dataset
# |   |_Depth
# |   |_RGB
# |-files
# |   |_episode.txt
# |_ MDQN_k
#     |_memory
#         |_action.dat
#         |_reward.dat
forward = True
def main():
    epi_file=open('../files/episode.txt')
    episode = epi_file.readline()
    epi_file.close()
    episode = int(episode)-1
    qagent = DQNAgent(0)#episode-1)
    qagent.load_memory_of_episode(episode)
    qys=[]
    qds=[]
    for k in range(5):
        for j in range(2):
            # for i in range(0,len(qagent.memory),qagent.batch_size):
            qy,qd=qagent.memory_replay()
        qagent.update_targer_model()
        qys.append(qy)
        qds.append(qd)
    qagent.save_model(episode)
    res = time.strftime('%Y/%m/%d-%H:%M:%S',time.localtime(time.time()))+"Average of episode: %d Q_y: %f Q_d: %f"%(episode,np.mean(qys),np.mean(qds))
    epi_file=open('../files/avg_Q.txt','a')
    epi_file.write(res+'\n')
    epi_file.close()

    if forward:
        epi_file=open('../files/episode.txt','w')
        epi_file.write(str(episode+2))
        epi_file.close()

if __name__ == '__main__':
    #set memory path
    for i in range(14):
        epi_file=open('../files/episode.txt')
        episode = epi_file.readline()
        epi_file.close()
        print("episode %d start"%(int(episode)-1))
        if int(episode)>15:
            break;
        main()
