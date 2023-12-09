from TetrisBattle.envs.tetris_env import TetrisSingleEnv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm


class ActorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.fc3 = nn.Linear(32, 10)

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
        out = out.view(out.size(0), -1).flatten()
        # Linear function (readout)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return F.softmax(out, dim=-1)
    
class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.fc3 = nn.Linear(32, 1)

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
        return torch.squeeze(out, 0)

from torch.optim.lr_scheduler import StepLR
class ActorCriticAgent():

    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic
        #self.actor_opt = optim.Adam(self.actor.parameters(), lr=0.01, betas=(0.9, 0.999))
        #self.critic_opt = optim.Adam(self.critic.parameters(), lr=0.01, betas=(0.9, 0.999))
        self.actor_opt = optim.RMSprop(self.actor.parameters(), lr=7e-7, alpha=0.95, eps=1e-08)
        self.critic_opt = optim.RMSprop(self.critic.parameters(), lr=7e-7, alpha=0.95, eps=1e-08)

        self.actor_loss = []
        self.critic_loss = []

    def forward(self, state):
        return self.actor(state), self.critic(state)
    
    def learn(self, log_probs, rewards, values):
        advantages = rewards - values

        loss_actor = (-log_probs * advantages).sum()
        loss_critic = F.smooth_l1_loss(values, rewards).sum()

        self.actor_opt.zero_grad()
        loss_actor.backward(retain_graph=True)
        self.actor_opt.step()

        self.critic_opt.zero_grad()
        loss_critic.backward()
        self.critic_opt.step()

    def sample(self, state):
        action_prob = self.actor(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

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
def do_group_action(action, env):
    ret_reward = 0
    ret_done = False
    for v in action_mapping[action]:
        env.take_turns()
        observation, reward, done, infos = env.step(v)
        ret_reward += reward
        ret_done = ret_done or done
    return observation, ret_reward, ret_done, infos

actor = ActorNetwork()
actor.load_state_dict(torch.load('actor_model'))
critic = CriticNetwork()
critic.load_state_dict(torch.load('critic_model'))
agent = ActorCriticAgent(actor, critic)

# training  single mode
env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="human")

agent.actor.train()  # Switch network into training mode
agent.critic.train()
EPISODE_PER_BATCH = 6
NUM_BATCH = 300 

avg_total_rewards= []
for batch in range(NUM_BATCH):

    log_probs, rewards = [], []
    total_rewards= []
    values = []
    # collect trajectory
    have_clear = 0
    last_scores = 1

    for episode in range(EPISODE_PER_BATCH):
        state = env.reset()
        state = np.expand_dims(state[:,:10], axis=0)
        total_reward, total_step = 0, 0
        seq_rewards = []
        last_action = -1
        score_now = 1
        while True:
            action, log_prob = agent.sample(state) # at, log(at|st)
            next_state, reward, done, _ = do_group_action(action, env)
            
            reward -= 0.01
            last_action = action
            if _['is_fallen']:
                have_clear += _['cleared']
                score_now += _['cleared']

            next_state = np.expand_dims(next_state[:,:10], axis=0)
            log_probs.append(log_prob) # [log(a1|s1), log(a2|s2), ...., log(at|st)]
            values.append(agent.critic(torch.FloatTensor(state)))
            state = next_state

            total_reward += reward
            total_step += 1
            rewards.append(reward)

            if done:
                print(f"scores = {total_reward}")
                total_rewards.append(total_reward)
                break

        for i in range(total_step):
          rewards[len(rewards)-2 - i] = rewards[len(rewards)-2 - i] + rewards[len(rewards)-1 - i] * 0.99
    print(f"batch now = {batch} ", end =' ')
        
    # record training process
    if(have_clear or 1):
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_total_rewards.append(avg_total_reward)
        print(avg_total_reward)
        # update agent

        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # normalize the reward
        rewards = np.array(rewards, dtype='float32')
        agent.learn(torch.stack(log_probs), torch.from_numpy(rewards), torch.stack(values))
        
        torch.save(agent.actor.state_dict(), 'actor_model')
        torch.save(agent.critic.state_dict(), 'critic_model')
        print(f"cleared {have_clear}, and saved")