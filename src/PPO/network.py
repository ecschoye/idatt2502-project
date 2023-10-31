import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Setting up a basic Feed Forward Neural Network

class FeedForwardNN(nn.Module): 
  def __init__(self, in_dim, out_dim): 
    super(FeedForwardNN, self).__init__()

    #Adding neural network layers using basic nn.Linear layers
    self.layer1 = nn.Linear(in_dim, 64)
    self.layer2 = nn.Linear(64, 64)
    self.layer3 = nn.Linear(64, out_dim)

  # Forward function to do a forward pass on our network. 
  # Uses ReLU for activation
  def forward(self, obs): 
    #Convert observation to tensor if it's a numpy array
    if isinstance(obs, np.ndarray): 
      obs = torch.tensor(obs, dtype=torch.float)
    
    activation1 = F.relu(self.layer1(obs))
    activation2 = F.relu(self.layer2(activation1))
    output = self.layer3(activation2)

    return output