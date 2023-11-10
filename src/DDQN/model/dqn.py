import os

import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q Network class that defines the neural network architecture.
    """

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        """Convulation output size calculation"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """Forward pass of the neural network"""
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    def save(self, target: bool = False):
        """Save the model"""
        dir_path = (
            "DDQN/trained_model/target" if target else "DDQN/trained_model/current"
        )
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        torch.save(self.state_dict(), dir_path + "_ddqn_model.pt")

    def load(self, device, target: bool = False):
        """Load the model"""
        dir_path = (
            "DDQN/trained_model/target" if target else "DDQN/trained_model/current"
        )
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        self.load_state_dict(
            torch.load(dir_path + "_ddqn_model.pt", map_location=device)
        )
