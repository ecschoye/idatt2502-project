import pickle
import random

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from ..model.dqn import DQN
from collections import deque
from ..utils.replay_buffer import ReplayBuffer


class DDQNAgent:
    def __init__(self, env, state_space, action_space, memory_size=20000, batch_size=32, lr=0.00025, gamma=0.90, epsilon=1.0, epsilon_min=0.01, epsilon_decay=10**5, copy=5000, pretrained_path=None):
        # Environment
        self.env = env
        self.state_space = state_space
        self.action_space = action_space

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, running on CPU.")

        # Hyperparameters
        self.batch_size = batch_size
        self.copy = copy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_max = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.lr = lr
        self.memory_size = memory_size

        # Memory and models
        self.memory = ReplayBuffer(state_space, self.memory_size)
        self.local_model = DQN(self.state_space, self.action_space).to(self.device)
        self.target_model = DQN(self.state_space, self.action_space).to(self.device)
        self.ending_position = 0
        self.num_in_queue = 0
        self.steps = 0

        # Load pretrained model
        self.pretrained_path = pretrained_path
        if self.pretrained_path is not None:
            self.local_model.load(self.device, self.pretrained_path)
            self.target_model.load(self.device, self.pretrained_path)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.lr, eps=1e-4)
        self.loss = nn.SmoothL1Loss().to(self.device)

        self.steps = 0
        self.train_loss = []
        self.reward_history = []

    def update_epsilon(self):
        #self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)'
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-1 * ((self.steps + 1) / self.epsilon_decay))

    def increment_steps(self):
        self.steps += 1

    def best_action(self, state):
        return torch.argmax(self.local_model(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

    def act(self, state):
        #print("State shape in act:", state.shape)
        # Epsilon-greedy action selection
        if random.random() <= self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            action_values = self.local_model(torch.tensor(state, dtype=torch.float32, device=self.device))
            action = torch.argmax(action_values).item()

        self.increment_steps()

        return action

    def remember(self, state, action, reward, next_state, done):
        self.ending_position = (self.ending_position + 1) % self.memory_size
        self.num_in_queue = min(self.num_in_queue + 1, self.memory_size)
        self.memory.add(state, action, reward, next_state, done, self.ending_position)

    def update_q_value(self, reward, next_state, done):
        return reward + torch.mul((self.gamma * self.target_model(next_state).max(1).values.unsqueeze(1)), 1 - done)

    def update_target_model(self):
        self.target_model.load_state_dict(self.local_model.state_dict())

    def experience_replay(self):
        if self.steps % self.copy == 0:
            self.update_target_model()

        if self.num_in_queue < self.batch_size:
            return

        state, action, reward, next_state, done_flag = self.memory.sample(
            self.num_in_queue, self.batch_size, self.device
        )

        # target = REWARD + torch.mul((self.gamma * self.target_net(STATE2).max(1).values.unsqueeze(1)),  1 - DONE)
        self.optimizer.zero_grad()
        target = self.update_q_value(reward, next_state, done_flag)
        current = self.local_model(state).gather(1, action.long())
        loss = self.loss(current, target)
        loss.backward()
        self.optimizer.step()

    def save(self):
        self.local_model.save()
        self.target_model.save()
        self.memory.save()

    def load(self):
        self.local_model.load(self.device)
        self.target_model.load(self.device)
        self.memory.load()
