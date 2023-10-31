import os
import random

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_space, memory_size):
        self.state_space = state_space

        # Initialize storage arrays
        self.states = torch.zeros((memory_size, *self.state_space))
        self.next_states = torch.zeros((memory_size, *self.state_space))
        self.actions = torch.zeros(memory_size, 1)
        self.rewards = torch.zeros(memory_size, 1)
        self.done_flags = torch.zeros(memory_size, 1)

    def add(self, state, action, reward, next_state, done, index):
        self.states[index] = state.float()
        self.actions[index] = action.float()
        self.rewards[index] = reward.float()
        self.next_states[index] = next_state.float()
        self.done_flags[index] = done.float()

    def sample(self, num_in_queue, batch_size, device):
        index = random.choices(range(num_in_queue), k=batch_size)

        states = torch.tensor(self.states[index]).float().to(device)
        actions = torch.tensor(self.actions[index]).long().to(device)
        rewards = torch.tensor(self.rewards[index]).float().to(device)
        next_states = torch.tensor(self.next_states[index]).float().to(device)
        done_flags = torch.tensor(self.done_flags[index]).float().to(device)

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
