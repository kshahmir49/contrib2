# Warm-start target search

The anchor learns a convolutional encoder and a dueling Q head with Double DQN. A target agent reuses the frozen encoder and trains new value and advantage streams for its own action set.

The environment uses a 100 × 100 search region and a three-channel 50 × 50 semantic observation. The reward and convolutional architecture follow Sections 4.1.1 and 4.1.4 of the paper.

From the repository root, train the irregular anchor and transfer to each four-action target.

```bash
python warm_start/train_anchor.py --agent irregular --episodes 3000 --seed 0 --out-dir runs/anchor_seed0
python warm_start/train_transfer.py --agent orthogonal --anchor-ckpt runs/anchor_seed0/anchor_best.pt --episodes 1000 --seed 0 --out-dir runs/orthogonal_seed0
python warm_start/train_transfer.py --agent diagonal --anchor-ckpt runs/anchor_seed0/anchor_best.pt --episodes 1000 --seed 0 --out-dir runs/diagonal_seed0
```

Repeat with the other two seeds. To evaluate the 3000-episode transfer horizon, use `--episodes 3000` and a different output directory.

The anchor is selected by validation success rate, breaking ties using average steps. The validation interval, validation set size, random seed values and optimizer selection are implementation choices not fully specified in the paper; confirm them against the original experiment setup before claiming exact numerical reproduction.

The release includes only the proposed method. The fully distributed DDQN comparison is not included.
