from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "envs"))
from simple_ma_env_allagents import Simple_MA_Env_Centralized_AllAgents

from models import DQN, TransferDQN, load_anchor_encoder
from utils import get_device, make_state_from_obs, set_seed, write_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent1-ckpt", type=str, default=None, help="Orthogonal/anchor checkpoint. If absent and --agent1-random not set, random actions are used.")
    p.add_argument("--agent2-ckpt", type=str, default=None, help="Diagonal checkpoint.")
    p.add_argument("--agent3-ckpt", type=str, default=None, help="Irregular checkpoint.")
    p.add_argument("--agent1-random", action="store_true")
    p.add_argument("--agent2-random", action="store_true")
    p.add_argument("--agent3-random", action="store_true")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=0.2, help="Evaluation exploration rate, matching original notebook's 20% random actions.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="runs/eval_all_agents.csv")
    return p.parse_args()


def _safe_load_model(path: Optional[str], num_actions: int, device: torch.device):
    if path is None:
        return None
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        model = ckpt.to(device)
        model.eval()
        return model
    args = ckpt.get("args", {})
    method = ckpt.get("method") or args.get("method")
    if method is None:
        # Anchor DQN checkpoint.
        model = DQN(num_actions=num_actions).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model
    if method != "anchor_only":
        raise ValueError("Only proposed anchor_only transfer checkpoints are supported")
    anchor_ckpt = args.get("anchor_ckpt") or args.get("anchor-ckpt")
    if not anchor_ckpt or not os.path.exists(anchor_ckpt):
        raise ValueError(f"Anchor checkpoint not found for {path}: {anchor_ckpt}")
    anchor_encoder = load_anchor_encoder(anchor_ckpt, device)
    model = TransferDQN(
        num_actions=num_actions, method="anchor_only", anchor_encoder=anchor_encoder
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _act(model, obs_state: torch.Tensor, n: int, random_flag: bool, epsilon: float, device: torch.device) -> int:
    if random_flag or model is None or random.random() < epsilon:
        return random.randrange(n)
    with torch.no_grad():
        return int(torch.argmax(model(obs_state.unsqueeze(0).to(device)), dim=1).item())


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    env = Simple_MA_Env_Centralized_AllAgents(100)
    models = [
        _safe_load_model(args.agent1_ckpt, env.action_space_agent1.n, device),
        _safe_load_model(args.agent2_ckpt, env.action_space_agent2.n, device),
        _safe_load_model(args.agent3_ckpt, env.action_space_agent3.n, device),
    ]
    random_flags = [args.agent1_random, args.agent2_random, args.agent3_random]
    rows = []
    ucrs = []
    steps_list = []
    success = []
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        state = torch.zeros((4, 50, 50), dtype=torch.float32, device=device)
        state = make_state_from_obs(obs, state, device)
        done = False
        steps = 0
        lap1 = lap2 = lap3 = None
        people_count = 0
        ret = 0.0
        while not done and steps < args.max_steps:
            a1 = _act(models[0], state, env.action_space_agent1.n, random_flags[0], args.epsilon, device)
            a2 = _act(models[1], state, env.action_space_agent2.n, random_flags[1], args.epsilon, device)
            a3 = _act(models[2], state, env.action_space_agent3.n, random_flags[2], args.epsilon, device)
            obs, r1, r2, r3, done, people_count, lap1, lap2, lap3, env_steps = env.step(a1, a2, a3)
            ret += float(r1 + r2 + r3)
            state = make_state_from_obs(obs, state, device)
            steps += 1
        union = lap1 + lap2 + lap3
        overlap = lap1 * lap2 * lap3
        denom = max(1, int(np.count_nonzero(union)))
        ucr = (int(np.count_nonzero(union)) - int(np.count_nonzero(overlap))) / denom
        rows.append({
            "episode": ep,
            "seed": args.seed,
            "steps": steps,
            "success": int(people_count > 0),
            "people_count": int(people_count),
            "return": ret,
            "ucr": ucr,
        })
        ucrs.append(ucr)
        steps_list.append(steps)
        success.append(int(people_count > 0))
        print(f"[eval] ep={ep}/{args.episodes} steps={steps} success={success[-1]} ucr={ucr:.4f}")
    write_csv(args.out, rows)
    print("Summary:")
    print(json.dumps({
        "episodes": args.episodes,
        "mean_ucr": float(np.mean(ucrs)),
        "std_ucr": float(np.std(ucrs)),
        "mean_steps": float(np.mean(steps_list)),
        "success_rate": float(np.mean(success)),
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
