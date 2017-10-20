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
    episode = epi_file.readline()
    epi_file.close()
    episode = int(episode)-1
    qagent = DQNAgent(episode-1)
    qagent.load_memory_of_episode(episode)
    for k in range(50):
        for j in range(10):
            for i in range(0,len(qagent.memory),qagent.batch_size):
                qagent.memory_replay()
        qagent.update_targer_model()
    qagent.update_targer_model()
    qagent.save_model(episode)
    print("Training finished")
    if forward:
        epi_file=open('../files/episode.txt','w')
        epi_file.write(str(episode+2))
        epi_file.close()

if __name__ == '__main__':
    #select what to train
    #set memory path
    main()
