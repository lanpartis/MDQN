from agent import DQNAgent
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
    episode = epi_file.read(1)
    epi_file.close()
    episode = int(episode)-1
    qagent = DQNAgent()
    qagent.load_model(episode-1)
    qagent.load_memory_of_episode(episode)
    for i in range(0,len(qagent.memory),qagent.batch_size):
        qagent.memory_replay()
        if i%10==0:
            qagent.update_targer_model()
    qagent.update_targer_model()
    qagent.save_model(episode)
    if forward:
        epi_file=open('../files/episode.txt'，'w')
        epi_file.write(str(episode+1))
        epi_file.close()

if __name__ == '__main__':
    #select what to train
    #set memory path
    main()
