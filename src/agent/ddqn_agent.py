import math
import random

import numpy as np
import torch
import torch.nn as nn

from ..model.dqn import DQN
from ..utils.replay_buffer import ReplayBuffer


class DDQNAgent:
    def __init__(
        self,
        env,
        state_space,
        action_space,
        memory_size=20000,
        batch_size=32,
        lr=0.00025,
        gamma=0.90,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=10**5,
        copy=5000,
        pretrained_path=None,
    ):
        # Environment
        self.env = env
        self.state_space = state_space
        self.action_space = action_space

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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
        self.local_model = DQN(self.state_space, self.action_space).to(
            self.device
        )
        self.target_model = DQN(self.state_space, self.action_space).to(
            self.device
        )
        self.ending_position = 0
        self.num_in_queue = 0
        self.steps = 0

        # Load pretrained model
        self.pretrained_path = pretrained_path
        if self.pretrained_path is not None:
            self.local_model.load(self.device)
            self.target_model.load(self.device, True)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.local_model.parameters(), lr=self.lr, eps=1e-4
        )
        self.loss = nn.SmoothL1Loss().to(self.device)

        self.train_loss = []
        self.reward_history = []

    def update_epsilon(self):
        """
        Adjusts the epsilon value for the epsilon-greedy
        policy using exponential decay.

        This encourages the agent to gradually shift from exploring
        the environment to exploiting the learned policy as it gains
        more experience (indicated by the number of steps taken).
        """
        self.epsilon = self.epsilon_min + (
            self.epsilon_max - self.epsilon_min
        ) * math.exp(-1 * ((self.steps + 1) / self.epsilon_decay))

    def increment_steps(self):
        """
        Increments the number of steps taken by the agent.
        """
        self.steps += 1

    def best_action(self, state):
        """
        Identifies the action with the highest Q-value for the given state.

        :return: A tensor containing the action with the highest Q-value.
        """
        return (
            torch.argmax(self.local_model(state.to(self.device)))
            .unsqueeze(0)
            .unsqueeze(0)
            .cpu()
        )

    def act(self, state):
        """
        Chooses an action based on the current
        state using the epsilon-greedy policy.

        With a probability of epsilon, a random action
        is chosen (exploration).
        Otherwise, the action with the highest predicted
        Q-value is chosen (exploitation).

        Also increments the number of steps taken by the agent.

        :param state: The current state of the environment.
        :return: The chosen action.
        """
        # Epsilon-greedy action selection
        if random.random() <= self.epsilon:
            # Random action
            action = np.random.randint(self.action_space)
        else:
            # Greedy action
            action_values = self.local_model(
                torch.tensor(state, dtype=torch.float32, device=self.device)
            )
            action = torch.argmax(action_values).item()

        self.increment_steps()

        return action

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores the experience in the replay buffer.
        """
        self.memory.add(state, action, reward, next_state, done)

    def update_q_value(self, reward, next_state, done):
        """
        Computes the updated Q-value for the current state
        using the reward and the next state.

        This function calculates the target Q-value for the
        current state based on the Bellman equation, taking into
        account whether the next state is terminal (done).
        """
        return reward + torch.mul(
            (
                self.gamma
                * self.target_model(next_state).max(1).values.unsqueeze(1)
            ),
            1 - done,
        )

    def update_target_model(self):
        """
        Updates the weights of the target network with
        the weights of the local network.

        This method is used to periodically update the target
        network's weights, thereby stabilizing training.
        """
        self.target_model.load_state_dict(self.local_model.state_dict())

    def experience_replay(self):
        """
        Performs a single step of the experience replay algorithm.

        This method conducts the learning process for the agent by
        using a batch of experiences sampled from memory. It updates
        the local model by minimizing the difference between predicted
        Q-values and target Q-values, calculated using the Bellman equation.
        """
        # Check if it is time to update the target model
        # based on the copy parameter
        if self.steps % self.copy == 0:
            # Update the target model
            self.update_target_model()

        # Check if there are enough samples in memory
        if self.memory.num_in_queue < self.batch_size:
            return

        # Sample a batch of experiences from the memory
        state, action, reward, next_state, done_flag = self.memory.sample(
            self.batch_size, self.device
        )

        # Zero the gradients
        self.optimizer.zero_grad()

        # Calculate the target Q values for the next state
        target = self.update_q_value(reward, next_state, done_flag)

        # Predict the current Q values from the local model
        # using the sampled states and actions
        current = self.local_model(state).gather(1, action.long())

        # Compute the loss
        loss = self.loss(current, target)

        # Backpropagate the loss
        loss.backward()

        # Perform an optimization step
        self.optimizer.step()

    def save(self):
        """
        Saves the model and the memory.
        """
        self.local_model.save()
        self.target_model.save(True)
        self.memory.save()

    def load(self):
        """
        Loads the model and the memory.
        """
        self.local_model.load(self.device)
        self.target_model.load(self.device)
        self.memory.load()
