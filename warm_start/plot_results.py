from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="runs")
    p.add_argument("--out-dir", type=str, default="figures")
    p.add_argument("--smooth", type=int, default=25)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logs = glob.glob(os.path.join(args.root, "**", "episode_log.csv"), recursive=True)
    dfs = []
    for log in logs:
        df = pd.read_csv(log)
        if df.empty:
            continue
        df["run_dir"] = os.path.dirname(log)
        if "method" not in df:
            df["method"] = "anchor_train"
        if "agent" not in df:
            df["agent"] = "orthogonal"
        dfs.append(df)
    if not dfs:
        print("No logs found")
        return
    data = pd.concat(dfs, ignore_index=True)
    for agent, sub in data.groupby("agent"):
        plt.figure()
        for method, mdf in sub.groupby("method"):
            grouped = mdf.groupby("episode")["reward"].mean().rolling(args.smooth, min_periods=1).mean()
            plt.plot(grouped.index, grouped.values, label=method)
        plt.xlabel("Episode")
        plt.ylabel("Mean reward")
        plt.title(f"Target-search learning curves: {agent}")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(args.out_dir, f"learning_curve_{agent}.png")
        plt.savefig(out, dpi=200)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
