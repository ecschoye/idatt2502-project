import pickle

from ..model.dqn import DQN
from collections import deque
import torch
import torch.nn as nn
import numpy as np

from ..utils.replay_buffer import ReplayBuffer


class DDQNAgent:
    def __init__(self, env, state_space, action_space, memory_size=10000, batch_size=64, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, copy=5000, pretrained_path=None):
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
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.lr = lr
        self.memory_size = memory_size

        # Memory and models
        self.memory = ReplayBuffer(state_space, self.memory_size)
        self.local_model = DQN(self.state_space, self.action_space).to(self.device)
        self.target_model = DQN(self.state_space, self.action_space).to(self.device)

        # Load pretrained model
        self.pretrained_path = pretrained_path
        if self.pretrained_path is not None:
            self.local_model.load(self.device, self.pretrained_path)
            self.target_model.load(self.device, self.pretrained_path)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.steps = 0
        self.train_loss = []
        self.reward_history = []

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def increment_steps(self):
        self.steps += 1

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            action = self.local_model(state).argmax(dim=1).item()

        self.increment_steps()
        self.update_epsilon()

        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)


    def update_target_model(self):
        self.target_model.load_state_dict(self.local_model.state_dict())

    def learn(self):
        if self.steps % self.copy == 0:
            self.update_target_model()
        if self.memory.counter < self.batch_size:
            return

        # Sample from memory
        states, actions, rewards, next_states, done_flags = self.memory.sample(self.batch_size, self.device)

        next_actions = self.target_model(next_states)
        local_action_values = self.local_model(states).gather(1, actions.unsqueeze(-1))

        max_next_action_values = self.local_model(next_states).argmax(1)
        next_best_action_values = next_actions.gather(1, max_next_action_values.unsqueeze(-1))

        #expected_action_values = rewards + self.gamma * next_best_action_values * (1.0 - done_flags.float())
        expected_action_values = rewards + torch.mul(self.gamma * next_best_action_values,
                                                     (1.0 - done_flags.float()))

        loss = self.loss(local_action_values, expected_action_values)
        self.train_loss.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_epsilon()

    def save(self):
        self.local_model.save()
        self.target_model.save()
        self.memory.save()

    def load(self):
        self.local_model.load(self.device)
        self.target_model.load(self.device)
        self.memory.load()

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.learn()
                state = next_state
                episode_reward += reward

            self.reward_history.append(episode_reward)

            if episode % self.copy == 0:
                self.update_target_model()