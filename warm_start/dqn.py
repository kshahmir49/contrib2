import copy
import random
from collections import deque

import numpy as np
import torch
from torch.nn import functional as F


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DuelingDoubleDQN:
    def __init__(self, network, device, lr=5e-5, gamma=0.99, target_update_steps=1000):
        self.device = device
        self.online = network.to(device)
        self.target = copy.deepcopy(self.online).to(device)
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            (p for p in self.online.parameters() if p.requires_grad), lr=lr
        )
        self.gamma = gamma
        self.target_update_steps = target_update_steps
        self.steps = 0

    def act(self, observation, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.online.head.advantage[-1].out_features)
        with torch.no_grad():
            state = torch.as_tensor(observation, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            return int(self.online(state).argmax(dim=1).item())

    def update(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.as_tensor(np.stack(states), dtype=torch.float32,
                                 device=self.device)
        next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32,
                                      device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long,
                                  device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        values = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_values = self.target(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * (1 - dones) * next_values

        loss = F.smooth_l1_loss(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.online.parameters() if p.requires_grad), 10.0
        )
        self.optimizer.step()
        return loss.item()

    def advance_step(self):
        self.steps += 1
        if self.steps % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
