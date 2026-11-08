# Decoupling Perception and Control for Parameter-Efficient Heterogeneous Reinforcement Learning

This repository contains the proposed warm-start and cold-start methods from the paper.

## Install

Use Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The cold-start tasks require MuJoCo and Gymnasium Robotics. Set `MUJOCO_GL=egl` for headless GPU rendering where required.

## Warm-start target search

An irregular-action anchor is trained with dueling Double DQN. Its convolutional encoder is then frozen and reused by the orthogonal and diagonal agents. Each target trains its own dueling action head.

```bash
python warm_start/train_anchor.py --agent irregular --episodes 3000 --seed 0 --out-dir runs/anchor_seed0
python warm_start/train_transfer.py --agent orthogonal --anchor-ckpt runs/anchor_seed0/anchor_best.pt --episodes 1000 --seed 0 --out-dir runs/orthogonal_seed0
python warm_start/train_transfer.py --agent diagonal --anchor-ckpt runs/anchor_seed0/anchor_best.pt --episodes 1000 --seed 0 --out-dir runs/diagonal_seed0
```

For the longer transfer horizon, set `--episodes 3000`. The [warm-start README](warm_start/README.md) describes the observation, reward and evaluation procedure.

## Cold-start visual control

Each task uses one frozen ImageNet-pretrained MobileNetV3-Small encoder to process the shared RGB frame. The encoder features are sent to separate 32-dimensional residual adapters and SAC actors. Critics use proprioceptive state-action inputs.

```bash
python cold_start/humanoid/proposed.py --steps 1000000 --seed 0 --out-dir runs/humanoid_seed0
python cold_start/standup/proposed.py --steps 1000000 --seed 0 --out-dir runs/standup_seed0
python cold_start/pusher/proposed.py --steps 500000 --seed 0 --out-dir runs/pusher_seed0
```

The default cold-start settings include 128 × 128 RGB frames, batch size 256, replay capacity 500,000, 120,000 warm-up steps, and 1,000 maximum steps per episode. Use five seeds for the evaluation protocol.


This repository releases **only the proposed methods**. It does not include distributed DDQN, distributed SAC or HATRPO implementations. Residual adapters are used only in the proposed cold-start method.

The code follows the methods and explicitly documented settings in the manuscript. Some details, including the warm-start optimizer and validation frequency, are not fully specified in the paper. Those implementation choices should be checked against the original experimental setup before claiming exact numerical reproducibility. No previously reported results have been rerun or independently verified.
