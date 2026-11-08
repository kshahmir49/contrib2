from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "envs"))
from simple_ma_env_ugv_newreward import Simple_MA_Env_Centralized_UGV as OrthogonalEnv
from simple_ma_env_ugv_newreward_diagonal import Simple_MA_Env_Centralized_UGV as DiagonalEnv
from simple_ma_env_ugv_newreward_irregular import Simple_MA_Env_Centralized_UGV as IrregularEnv

from agents import DQNAgent, ReplayBuffer
from models import TransferDQN, load_anchor_encoder, trainable_parameter_count, total_parameter_count
from utils import append_csv, ensure_dir, get_device, make_state_from_obs, set_seed

ENV_BY_AGENT = {
    "orthogonal": OrthogonalEnv,
    "diagonal": DiagonalEnv,
    "irregular": IrregularEnv,
}

METHODS = ["anchor_only"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, default="anchor_only", choices=METHODS)
    p.add_argument("--agent", type=str, required=True, choices=list(ENV_BY_AGENT))
    p.add_argument("--anchor-ckpt", type=str, default=None)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="runs/transfer")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--buffer-size", type=int, default=50000)
    p.add_argument("--learning-starts", type=int, default=2000, help="Lower default than anchor because transfer heads are small.")
    p.add_argument("--epsilon-decay", type=float, default=0.0012)
    p.add_argument("--final-epsilon", type=float, default=0.1)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.anchor_ckpt:
        raise SystemExit("--anchor-ckpt is required for the proposed warm-start transfer")

    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "checkpoints"))

    env = ENV_BY_AGENT[args.agent](100)
    anchor_encoder = load_anchor_encoder(args.anchor_ckpt, device)
    model = TransferDQN(
        num_actions=env.action_space.n,
        method="anchor_only",
        anchor_encoder=anchor_encoder,
    ).to(device)
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
            "method": args.method,
            "agent": args.agent,
            "steps": steps,
            "reward": ep_reward,
            "epsilon": agent.epsilon,
            "loss": loss_val if loss_val is not None else "",
            "global_steps": global_steps,
            "elapsed_sec": round(time.time() - t0, 3),
        }
        append_csv(log_path, row)
        print(f"[{args.method}:{args.agent}] ep={ep}/{args.episodes} steps={steps} reward={ep_reward:.3f} eps={agent.epsilon:.3f} loss={row['loss']}")

    final_path = os.path.join(args.out_dir, "checkpoints", f"{args.agent}_{args.method}_final.pt")
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "episode": args.episodes,
                "method": args.method, "agent": args.agent}, final_path)
    print(f"Saved {final_path}")


if __name__ == "__main__":
    main()
