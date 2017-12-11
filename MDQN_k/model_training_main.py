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
    qagent = DQNAgent(episode-1)
    qys=[]
    qds=[]
    res_before = qagent.evalutate_4()
    if episode != 9:
        qagent.load_memory_of_episode(episode)
        for k in range(50):
            for j in range(10):
                for i in range(0,len(qagent.memory),qagent.batch_size):
                    qy,qd,yloss=qagent.memory_replay()
                print("Iteration %d-%d loss:%f"%(k,j,yloss))
            qagent.update_target_model()
            qys.append(qy)
            qds.append(qd)
    qagent.save_model(episode)
    res_after=qagent.evalutate_4()
    res = time.strftime('%Y/%m/%d-%H:%M:%S',time.localtime(time.time()))+"Average of episode: %d Q_y: %f Q_d: %f,accuracy before: %f,accuracy after: %f"%(episode,np.mean(qys),np.mean(qds),res_before,res_after)
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
