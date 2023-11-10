import torch
from torch import nn


class DiscreteActorCriticNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DiscreteActorCriticNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(in_dim)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, out_dim)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(o.view(1, -1).size(1))

    def forward(self, state):
        x = state.clone().detach().to(dtype=torch.float, device=self.device)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
