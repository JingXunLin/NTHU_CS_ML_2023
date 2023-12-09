from TetrisBattle.envs.tetris_env import TetrisSingleEnv

import math
import random
import matplotlib
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="human")

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0) #output_shape=(16,24,24)
        self.relu1 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0) #output_shape=(32,8,8)
        self.relu2 = nn.ReLU() # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) 
        # Fully connected 1 ,#input_shape=(32*4*4)
        self.fc1 = nn.Linear(96, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.n_actions)

    def forward(self, state):
        # Convolution 1
        out = self.cnn1(state)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)
        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2 
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = out.flatten()
        # Linear function (readout)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 0.01

# Get number of actions from gym action space
n_actions = 10
# Get the number of state observations
state = env.reset()
state = np.expand_dims(state[:,:10], axis=0)
n_observations = 200

policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
#policy_net.load_state_dict(torch.load('policy_net_state_dict'))
#target_net.load_state_dict(torch.load('target_net_state_dict'))

optimizer = optim.Adam(policy_net.parameters(), lr=LR, betas=(0.9, 0.999))
memory = ReplayMemory(30000)

steps_done = 81463

def select_action(state, mode):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return np.argmax(policy_net(state))
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)




def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 300

action_mapping = [[0], [2]]
for i in range(4):
    kk = []
    for j in range(i):
        kk.append(3)
    ttkk = kk[:]
    ttkk.append(5)
    action_mapping.append(ttkk)
    ttkk = kk[:]
    ttkk.append(6)
    action_mapping.append(ttkk)

mode = 'train'
steps_done = 0 #381834
if mode == 'train':
    last_scores = 1
    for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
        state = env.reset()
        state = np.expand_dims(state[:,:10], axis=0)
        state = torch.tensor(state, dtype=torch.float32, device=device)
        rewards = []
        cleared = 0
        last_action = -1
        score_now = 1
        for t in count():
            action = select_action(state, mode)
            if action == last_action:
                reward -= last_scores / 10
            sum_reward = 0
            fallen = 0
            for v in action_mapping[action]:
                env.take_turns()
                observation, reward, done, infos = env.step(v)
                sum_reward += reward
                if infos['is_fallen']:
                    cleared += infos['cleared']
                    fallen = 1
            
            if infos['is_fallen']:
                observation = np.expand_dims(observation[:,:10], axis=0)
                reward = torch.tensor([sum_reward], device=device)
                rewards.append(reward)

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)
                # Move to the next state
                state = next_state
                # Perform one step of the optimization (on the policy network)
                optimize_model()

                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    torch.save(target_net_state_dict, 'target_net_state_dict')
                    torch.save(policy_net_state_dict, 'policy_net_state_dict')
                    print(len(rewards))
                    print(f"avg reward = {sum(rewards)/len(rewards)}, steps = {steps_done}, clear = {cleared}")
                    break
        last_scores = score_now
    print('Complete')
else:
    
    policy_net.eval()
    target_net.eval()
    state = env.reset()
    state = state[:][:17].flatten()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    rewards = []
    while True:
        action = select_action(state, mode)
        observation, reward, terminated, _ = env.step(action.item())
        observation = observation[:][:17].flatten()
        reward = torch.tensor([reward], device=device)
        done = terminated
        rewards.append(reward)

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device)

        # Move to the next state
        state = next_state

        if done:
            print(f"avg reward = {sum(rewards)/len(rewards)}")
            break