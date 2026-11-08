from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "envs"))
from simple_ma_env_ugv_newreward import Simple_MA_Env_Centralized_UGV

from agents import DQNAgent, ReplayBuffer
from models import DQN, trainable_parameter_count, total_parameter_count
from utils import append_csv, ensure_dir, get_device, make_state_from_obs, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="runs/anchor")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--buffer-size", type=int, default=50000)
    p.add_argument("--learning-starts", type=int, default=5000)
    p.add_argument("--epsilon-decay", type=float, default=0.0012)
    p.add_argument("--final-epsilon", type=float, default=0.1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save-every", type=int, default=0, help="Save intermediate anchors every N episodes; 0 disables.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "checkpoints"))

    env = Simple_MA_Env_Centralized_UGV(100)
    model = DQN(num_actions=env.action_space.n).to(device)
    optimizer = torch.optim.RMSprop((p for p in model.parameters() if p.requires_grad), lr=args.lr, momentum=0.95)
    agent = DQNAgent(model, optimizer, device, gamma=args.gamma, epsilon=1.0,
                     final_epsilon=args.final_epsilon, epsilon_decay=args.epsilon_decay)
    rb = ReplayBuffer(args.buffer_size)

    meta = vars(args).copy()
    meta.update({
        "device": str(device),
        "trainable_params": trainable_parameter_count(model),
        "total_params": total_parameter_count(model),
        "num_actions": env.action_space.n,
    })
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log_path = os.path.join(args.out_dir, "episode_log.csv")
    global_steps = 0
    t0 = time.time()
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        state = torch.zeros((4, 50, 50), dtype=torch.float32, device=device)
        state = make_state_from_obs(obs, state, device)
        done = False
        ep_reward = 0.0
        loss_val = None
        steps = 0
        while not done and steps < args.max_steps:
            prev_state = state.clone()
            action = agent.select_action(env.action_space.n, prev_state.unsqueeze(0), greedy=False)
            obs, reward, done, info = env.step(action)
            state = make_state_from_obs(obs, state, device)
            rb.push(prev_state, action, reward, state, done)
            if len(rb) >= args.learning_starts:
                loss_val = agent.train_step(rb.sample(args.batch_size))
            ep_reward += float(reward)
            steps += 1
            global_steps += 1
        agent.decay_epsilon()
        row = {
            "episode": ep,
            "seed": args.seed,
            "steps": steps,
            "reward": ep_reward,
            "epsilon": agent.epsilon,
            "loss": loss_val if loss_val is not None else "",
            "global_steps": global_steps,
            "elapsed_sec": round(time.time() - t0, 3),
        }
        append_csv(log_path, row)
        print(f"[anchor] ep={ep}/{args.episodes} steps={steps} reward={ep_reward:.3f} eps={agent.epsilon:.3f} loss={row['loss']}")
        if args.save_every and ep % args.save_every == 0:
            ckpt = os.path.join(args.out_dir, "checkpoints", f"anchor_ep{ep}_seed{args.seed}.pt")
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "episode": ep}, ckpt)

    final_path = os.path.join(args.out_dir, "checkpoints", "anchor_final.pt")
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "episode": args.episodes}, final_path)
    print(f"Saved {final_path}")


if __name__ == "__main__":
    main()
