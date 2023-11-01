import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Setting up a basic Feed Forward Neural Network

class DiscreteActorCriticNN(nn.Module): 
  def __init__(self, in_dim, out_dim): 
    super(DiscreteActorCriticNN, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(in_dim[0], 32, kernel_size=8, stride=4),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=2, stride=1),
      nn.ReLU()
    )

    conv_out_size = self._get_conv_out(in_dim)

    self.fc = nn.Sequential(
      nn.Linear(conv_out_size, 512),
      nn.ReLU(),
      nn.Linear(512, out_dim)
    )

  # Forward function to do a forward pass on our network. 
  # Uses ReLU for activation
  def _get_conv_out(self, shape):
    o = self.conv(torch.zeros(1, *shape))
    return int(np.prod(o.size()))

  def forward(self, state):
    x = torch.tensor(state, dtype=torch.float).unsqueeze(0)
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    return F.softmax(self.fc(x), dim=0)