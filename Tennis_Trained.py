from unityagents import UnityEnvironment
import numpy as np

from ddpg_agent import Agent
import torch


env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]

num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
print('The state for the second agent looks like:', states[1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent_list = []

for i in range(num_agents):
    agent_list.append(Agent(state_size, action_size, num_agents,random_seed=0))

agent_list[0].actor_local.load_state_dict(torch.load('actor_0.pth'))
agent_list[1].actor_local.load_state_dict(torch.load('actor_1.pth'))


scores = np.zeros(num_agents)
NUM_GAMES = 100
for ii in range(NUM_GAMES):
    while True:
        actions = [agent_list[i].act(states[i]) for i in range(num_agents)]

        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        states = next_states
        scores += rewards

        print('\rScores: {:.2f}\t{:.2f}'
              .format(scores[0], scores[1]), end="")

        if np.any(dones):
            break

print("\nScores: {}".format(scores/NUM_GAMES))

