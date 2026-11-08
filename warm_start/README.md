# Warm-start

Train the in-domain anchor, then transfer its frozen encoder to a target action head.

```bash
python warm_start/train_anchor.py --episodes 1000 --out-dir runs/anchor
python warm_start/train_transfer.py --agent orthogonal --anchor-ckpt runs/anchor/checkpoints/anchor_final.pt --episodes 1000 --out-dir runs/orthogonal
python warm_start/train_transfer.py --agent diagonal --anchor-ckpt runs/anchor/checkpoints/anchor_final.pt --episodes 1000 --out-dir runs/diagonal
```

`--seed`, `--episodes`, `--max-steps`, and other training flags are supported.
The anchor uses irregular movement. The target agents use orthogonal and diagonal action spaces.

The evaluation script takes explicit checkpoint paths. Example

```bash
python warm_start/evaluate_all_agents.py --agent1-ckpt runs/orthogonal/checkpoints/orthogonal_anchor_only_final.pt --agent2-ckpt runs/diagonal/checkpoints/diagonal_anchor_only_final.pt --agent3-random --episodes 20
```

The example uses a random third agent. Supply a third target checkpoint when evaluating a trained three-agent configuration.
