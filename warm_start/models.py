import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(256, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return F.relu(self.fc(x.flatten(1)))


class DuelingHead(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, num_actions)
        )

    def forward(self, features):
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingQNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.encoder = Encoder()
        self.head = DuelingHead(num_actions)

    def forward(self, x):
        return self.head(self.encoder(x))

    def freeze_encoder(self):
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
