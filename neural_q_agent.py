import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from environment import Environment

BUFFER_SIZE = int(1e5)  #replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed,
                 fc1_unit=64, fc2_unit=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience",
                                      field_names=["state",
                                                   "action",
                                                   "reward",
                                                   "next_state",
                                                   "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        # Randomly sample a batch of experiences from memory
        experiences = random.sample(self.memory, k=self.batch_size)

        #  make sure the num types are matched
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return(states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NeuralQAgent():

    def __init__(self, state_size, action_size, seed):
        self.env = Environment()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(self.state_size, self.action_size, seed).to(device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)
        # time t
        self.t_step = 0

    def step(self, state, action, reward, next_step, done):
        self.memory.add(state, action, reward, next_step, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)

    def act(self, state, eps=0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # epsilon greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        criterion = torch.nn.MSELoss()
        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        predicted_targets = self.qnetwork_local(states).gather(1, actions)
        with torch.no_grad():
            labels_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        labels = rewards + (gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def train(self, n_episodes=200, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.996):
        # scores for each episode
        scores = []
        # last 100 scores
        scores_window = deque(maxlen=100)
        eps = eps_start
        for i_episode in range(1, n_episodes+1):
            state = self.env.reset_vector()
            score = 0
            for t in range(max_t):
                action = self.act(state, eps)
                next_state, reward, done = self.env.step_vector(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break

                scores_window.append(score)
                scores.append(score)

                eps = max(eps*eps_decay, eps_end)
                print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
                if i_episode % 100 == 0:
                    print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

                if np.mean(scores_window) >= 200.0:
                    print('\nEnvironment solve in {:d} episodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                                np.mean(scores_window)))
                    torch.save(self.qnetwork_local.state_dict(), 'checkpoint.pth')
                    break
        return scores
