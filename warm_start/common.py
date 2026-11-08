import csv
import os
import random

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for_training():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(network, action_space, seed, episodes=20, max_steps=200, device="cpu"):
    from environment import TargetSearch

    env = TargetSearch(action_space, seed=seed, max_steps=max_steps)
    network.eval()
    successes = []
    steps = []
    returns = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        for t in range(max_steps):
            with torch.no_grad():
                x = torch.from_numpy(state).unsqueeze(0).to(device)
                action = int(network(x).argmax(dim=1).item())
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        successes.append(int(info["success"]))
        steps.append(t + 1)
        returns.append(total_reward)
    return {
        "success_rate": float(np.mean(successes)),
        "average_steps": float(np.mean(steps)),
        "median_steps": float(np.median(steps)),
        "mean_return": float(np.mean(returns)),
    }


def save_row(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
