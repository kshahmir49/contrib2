from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="runs")
    p.add_argument("--out", type=str, default="runs/summary.csv")
    p.add_argument("--last-n", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    rows: List[dict] = []
    for log in glob.glob(os.path.join(args.root, "**", "episode_log.csv"), recursive=True):
        try:
            df = pd.read_csv(log)
        except Exception:
            continue
        if df.empty:
            continue
        meta_path = os.path.join(os.path.dirname(log), "metadata.json")
        meta = {}
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path))
            except Exception:
                pass
        tail = df.tail(args.last_n)
        row = {
            "run_dir": os.path.dirname(log),
            "method": meta.get("method", "anchor_train"),
            "agent": meta.get("agent", "orthogonal"),
            "seed": meta.get("seed", df.get("seed", pd.Series([None])).iloc[0]),
            "episodes": len(df),
            "mean_reward_last_n": tail["reward"].mean() if "reward" in tail else None,
            "std_reward_last_n": tail["reward"].std() if "reward" in tail else None,
            "mean_steps_last_n": tail["steps"].mean() if "steps" in tail else None,
            "trainable_params": meta.get("trainable_params"),
            "total_params": meta.get("total_params"),
            "final_gate_alpha": tail["gate_alpha"].dropna().iloc[-1] if "gate_alpha" in tail and tail["gate_alpha"].dropna().size else None,
            "elapsed_sec": df["elapsed_sec"].iloc[-1] if "elapsed_sec" in df else None,
        }
        rows.append(row)
    out_df = pd.DataFrame(rows).sort_values(["method", "agent", "seed"], na_position="last") if rows else pd.DataFrame()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(out_df)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
