import os
import pickle
import random

import torch


class ExperienceReplayBuffer:
    def __init__(self, state_space, memory_size):
        """
        Initialize a replay buffer.
        """
        self.state_space = state_space
        self.capacity = memory_size
        self.current_position = 0
        self.size = 0

        # Initialize storage tensors
        self.state_buffer = torch.zeros((memory_size, *self.state_space))
        self.next_state_buffer = torch.zeros((memory_size, *self.state_space))
        self.action_buffer = torch.zeros(memory_size, 1)
        self.reward_buffer = torch.zeros(memory_size, 1)
        self.done_buffer = torch.zeros(memory_size, 1)

    def add_experience(self, state, action, reward, next_state, done):
        """
        Add a new experience to the replay buffer.
        """
        index = self.current_position
        self.state_buffer[index] = state.float()
        self.action_buffer[index] = action.float()
        self.reward_buffer[index] = reward.float()
        self.next_state_buffer[index] = next_state.float()
        self.done_buffer[index] = done.float()

        self.current_position = (self.current_position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size, device):
        """
        Sample a batch of experiences from the replay buffer.
        """
        indices = random.choices(range(self.size), k=batch_size)

        state_sample = self.state_buffer[indices].clone().detach().float().to(device)
        action_sample = self.action_buffer[indices].clone().detach().long().to(device)
        reward_sample = self.reward_buffer[indices].clone().detach().float().to(device)
        next_state_sample = (
            self.next_state_buffer[indices].clone().detach().float().to(device)
        )
        done_sample = self.done_buffer[indices].clone().detach().float().to(device)

        return (
            state_sample,
            action_sample,
            reward_sample,
            next_state_sample,
            done_sample,
        )

    def save(self):
        """
        Save the current replay buffer.
        """
        dir_path = "DDQN/experience_replay_buffer_data/"
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.state_buffer, os.path.join(dir_path, "states.pt"))
        torch.save(self.action_buffer, os.path.join(dir_path, "actions.pt"))
        torch.save(self.reward_buffer, os.path.join(dir_path, "rewards.pt"))
        torch.save(self.next_state_buffer, os.path.join(dir_path, "next_states.pt"))
        torch.save(self.done_buffer, os.path.join(dir_path, "done_flags.pt"))

        with open(os.path.join(dir_path, "counters.pkl"), "wb") as f:
            pickle.dump(
                {
                    "current_position": self.current_position,
                    "size": self.size,
                },
                f,
            )

    def load(self):
        """
        Load the state of the buffer from disk.
        """
        dir_path = "DDQN/experience_replay_buffer_data/"
        os.makedirs(dir_path, exist_ok=True)
        self.state_buffer = torch.load(os.path.join(dir_path, "states.pt"))
        self.action_buffer = torch.load(os.path.join(dir_path, "actions.pt"))
        self.reward_buffer = torch.load(os.path.join(dir_path, "rewards.pt"))
        self.next_state_buffer = torch.load(os.path.join(dir_path, "next_states.pt"))
        self.done_buffer = torch.load(os.path.join(dir_path, "done_flags.pt"))
        with open(os.path.join(dir_path, "counters.pkl"), "rb") as f:
            counters = pickle.load(f)
            self.current_position = counters["current_position"]
            self.size = counters["size"]
