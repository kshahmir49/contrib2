import argparse
import os

import torch

from common import device_for_training, evaluate, save_row, seed_all
from dqn import DuelingDoubleDQN, ReplayBuffer
from environment import TargetSearch
from models import DuelingQNetwork


def main():
    parser = argparse.ArgumentParser(description="Train the warm-start anchor")
    parser.add_argument("--agent", choices=["irregular", "orthogonal", "diagonal"], default="irregular")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--out-dir", default="runs/anchor")
    args = parser.parse_args()

    seed_all(args.seed)
    device = device_for_training()
    env = TargetSearch(args.agent, seed=args.seed, max_steps=args.max_steps)
    model = DuelingQNetwork(env.num_actions)
    learner = DuelingDoubleDQN(model, device, lr=args.lr)
    replay = ReplayBuffer()
    os.makedirs(args.out_dir, exist_ok=True)
    best = (-1.0, float("inf"))

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        epsilon = max(0.02, 1.0 - (1.0 - 0.02) * (episode - 1) / max(args.episodes - 1, 1))
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

        if args.eval_every and (episode % args.eval_every == 0 or episode == args.episodes):
            metrics = evaluate(model, args.agent, seed=args.seed + 100000,
                               episodes=args.eval_episodes, max_steps=args.max_steps,
                               device=device)
            save_row(os.path.join(args.out_dir, "validation.csv"), {
                "episode": episode, **metrics
            })
            key = (metrics["success_rate"], -metrics["average_steps"])
            if key > (best[0], -best[1]):
                best = (metrics["success_rate"], metrics["average_steps"])
                torch.save({"model": model.state_dict(), "agent": args.agent,
                            "episode": episode, "seed": args.seed},
                           os.path.join(args.out_dir, "anchor_best.pt"))
            print(f"Episode {episode} | return {episode_return:.2f} | val success {metrics['success_rate']:.2f}")

    torch.save({"model": model.state_dict(), "agent": args.agent,
                "episode": args.episodes, "seed": args.seed},
               os.path.join(args.out_dir, "anchor_final.pt"))


if __name__ == "__main__":
    main()
