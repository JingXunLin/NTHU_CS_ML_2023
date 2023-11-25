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
        self.fc1 = nn.Linear(578, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 8)

    def forward(self, state):
        hid = F.relu(self.fc1(state))
        hid = F.relu(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)
    
class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(578, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        hid = F.relu(self.fc1(state))
        hid = F.relu(self.fc2(hid))
        return torch.squeeze(self.fc3(hid), 0)
from torch.optim.lr_scheduler import StepLR
class ActorCriticAgent():

    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=0.01, betas=(0.9, 0.999))
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=0.01, betas=(0.9, 0.999))

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
actor = ActorNetwork()
#actor.load_state_dict(torch.load('actor_model'))
critic = CriticNetwork()
#critic.load_state_dict(torch.load('critic_model'))
agent = ActorCriticAgent(actor, critic)

# training  single mode
env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="human")

agent.actor.train()  # Switch network into training mode
agent.critic.train()
EPISODE_PER_BATCH = 2 
NUM_BATCH = 50 

avg_total_rewards= []

prg_bar = tqdm(range(NUM_BATCH))
for batch in prg_bar:

    log_probs, rewards = [], []
    total_rewards= []
    values = []
    # collect trajectory
    for episode in range(EPISODE_PER_BATCH):

        state = env.reset()[:][:17].flatten()
        total_reward, total_step = 0, 0
        seq_rewards = []
        while True:
            action, log_prob = agent.sample(state) # at, log(at|st)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[:][:17].flatten()

            log_probs.append(log_prob) # [log(a1|s1), log(a2|s2), ...., log(at|st)]
            values.append(agent.critic(torch.FloatTensor(state)))
            state = next_state

            total_reward += reward
            total_step += 1
            rewards.append(reward)

            if done:
                total_rewards.append(total_reward)
                break
        for i in range(total_step):
          rewards[len(rewards)-2 - i] = rewards[len(rewards)-2 - i] + rewards[len(rewards)-1 - i] * 0.99
        print(f"episode : {episode}")
    # record training process
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_total_rewards.append(avg_total_reward)
    prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}")
    print(avg_total_reward)
    # update agent

    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # normalize the reward
    rewards = np.array(rewards, dtype='float32')
    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards), torch.stack(values))
    
torch.save(agent.actor.state_dict(), 'actor_model')
torch.save(agent.critic.state_dict(), 'critic_model')
print("saved")