import argparse
import os

import torch

from common import device_for_training, evaluate, save_row, seed_all
from dqn import DuelingDoubleDQN, ReplayBuffer
from environment import TargetSearch
from models import DuelingQNetwork


def main():
    parser = argparse.ArgumentParser(description="Train a dueling head with a frozen anchor encoder")
    parser.add_argument("--agent", choices=["orthogonal", "diagonal", "irregular"], required=True)
    parser.add_argument("--anchor-ckpt", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--out-dir", default="runs/transfer")
    args = parser.parse_args()

    seed_all(args.seed)
    device = device_for_training()
    env = TargetSearch(args.agent, seed=args.seed, max_steps=args.max_steps)
    checkpoint = torch.load(args.anchor_ckpt, map_location="cpu", weights_only=True)
    source_agent = checkpoint["agent"]
    source = DuelingQNetwork(TargetSearch(source_agent).num_actions)
    source.load_state_dict(checkpoint["model"])

    model = DuelingQNetwork(env.num_actions)
    model.encoder.load_state_dict(source.encoder.state_dict())
    model.freeze_encoder()
    learner = DuelingDoubleDQN(model, device, lr=args.lr)
    replay = ReplayBuffer()
    os.makedirs(args.out_dir, exist_ok=True)

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        epsilon = max(0.05, 1.0 - (1.0 - 0.05) * (episode - 1) / max(args.episodes - 1, 1))
        episode_return = 0.0
        for _ in range(args.max_steps):
            action = learner.act(state, epsilon)
            next_state, reward, done, info = env.step(action)
            replay.push(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            learner.advance_step()
            if learner.steps >= args.learning_starts and len(replay) >= args.batch_size:
                learner.update(replay.sample(args.batch_size))
            if done:
                break
        save_row(os.path.join(args.out_dir, "training.csv"), {
            "episode": episode, "return": episode_return,
            "steps": env.steps, "success": int(info["success"]), "epsilon": epsilon
        })

    torch.save({"model": model.state_dict(), "agent": args.agent,
                "anchor": source_agent, "seed": args.seed},
               os.path.join(args.out_dir, "transfer_final.pt"))
    metrics = evaluate(model, args.agent, args.seed + 200000,
                       episodes=100, max_steps=args.max_steps, device=device)
    print(f"Evaluation after {args.episodes} episodes | {metrics}")
    save_row(os.path.join(args.out_dir, "evaluation.csv"), metrics)


if __name__ == "__main__":
    main()
