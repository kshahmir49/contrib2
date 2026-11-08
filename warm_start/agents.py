from __future__ import annotations

import copy
import random
from collections import deque
from typing import List, Tuple

import torch
import torch.nn as nn

Transition = Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, bool]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.data.append((state.detach().cpu(), int(action), torch.tensor(float(reward), dtype=torch.float32), next_state.detach().cpu(), bool(done)))

    def sample(self, batch_size: int):
        return random.sample(self.data, min(len(self.data), batch_size))

    def __len__(self):
        return len(self.data)


class DQNAgent:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device,
                 gamma: float = 0.99, epsilon: float = 1.0, final_epsilon: float = 0.1,
                 epsilon_decay: float = 0.0012, target_update_steps: int = 10000,
                 repeat_action: int = 4):
        self.model = model
        self.target_model = copy.deepcopy(model).to(device)
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.target_update_steps = target_update_steps
        self.loss_fn = nn.MSELoss()
        self.training_step_counter = 0
        self.action_counter = 0
        self.previous_action = None
        self.repeat_action = repeat_action

    def select_action(self, num_actions: int, state: torch.Tensor, greedy: bool = False) -> int:
        if (not greedy) and self.repeat_action > 1 and self.previous_action is not None and self.action_counter % self.repeat_action != 0:
            self.action_counter += 1
            return int(self.previous_action)
        if (not greedy) and random.random() < self.epsilon:
            action = random.randrange(num_actions)
        else:
            with torch.no_grad():
                q = self.model(state.to(self.device))
                action = int(torch.argmax(q, dim=1).item())
        self.previous_action = action
        self.action_counter += 1
        return action

    def train_step(self, minibatch):
        if not minibatch:
            return None
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.stack(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        q_values = self.model(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_model(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q * (~dones)
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        self.training_step_counter += 1
        if self.training_step_counter >= self.target_update_steps:
            self.target_model = copy.deepcopy(self.model).to(self.device)
            self.training_step_counter = 0
        return float(loss.detach().cpu())

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
