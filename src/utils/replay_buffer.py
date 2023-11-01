import os

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_space, memory_size):
        self.state_space = state_space
        self.counter = 0

        # Initialize storage arrays
        self.states = np.zeros((memory_size, *self.state_space), dtype=np.float32)
        self.next_states = np.zeros((memory_size, *self.state_space), dtype=np.float32)
        self.actions = np.zeros(memory_size, dtype=np.int64)
        self.rewards = np.zeros(memory_size, dtype=np.float32)
        self.done_flags = np.zeros(memory_size, dtype=bool)

    def add(self, state, action, reward, next_state, done):
        index = self.counter % self.states.shape[0]
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.done_flags[index] = done

        self.counter += 1

    def sample(self, batch_size, device):
        max_index = min(self.counter, self.states.shape[0])
        indices = np.random.choice(max_index, batch_size, replace=False)
        states = torch.from_numpy(self.states[indices]).to(device)
        actions = torch.from_numpy(self.actions[indices]).to(device)
        rewards = torch.from_numpy(self.rewards[indices]).to(device)
        next_states = torch.from_numpy(self.next_states[indices]).to(device)
        done_flags = torch.from_numpy(self.done_flags[indices]).to(device)

        return states, actions, rewards, next_states, done_flags

    def save(self):
        dir_path = "../replay_buffer_data"
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.states, os.path.join(dir_path, 'states.pt'))
        torch.save(self.actions, os.path.join(dir_path, 'actions.pt'))
        torch.save(self.rewards, os.path.join(dir_path, 'rewards.pt'))
        torch.save(self.next_states, os.path.join(dir_path, 'next_states.pt'))
        torch.save(self.done_flags, os.path.join(dir_path, 'done_flags.pt'))

    def load(self):
        dir_path = "../replay_buffer_data"
        os.makedirs(dir_path, exist_ok=True)
        self.states = torch.load(os.path.join(dir_path, 'states.pt'))
        self.actions = torch.load(os.path.join(dir_path, 'actions.pt'))
        self.rewards = torch.load(os.path.join(dir_path, 'rewards.pt'))
        self.next_states = torch.load(os.path.join(dir_path, 'next_states.pt'))
        self.done_flags = torch.load(os.path.join(dir_path, 'done_flags.pt'))
        self.counter = self.states.shape[0]
