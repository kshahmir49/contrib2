# Decoupling Perception and Control for Heterogeneous Reinforcement Learning

Code for the proposed warm-start and cold-start methods from the paper
*Decoupling Perception and Control for Parameter-Efficient Heterogeneous Reinforcement Learning*.

## Methods

- **Warm-start** trains an anchor encoder and reuses its frozen features with new action heads for target navigation tasks.
- **Cold-start** uses one frozen MobileNetV3-Small encoder shared across control modules. Each module has its own residual adapter and SAC policy.

This repository contains **only the proposed methods**. Comparison baseline implementations are not included.

## Install

Use Python 3.10 or newer, ideally in a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

MuJoCo tasks require a working MuJoCo and rendering setup. On headless Linux, you may need to set `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` before running.

## Warm-start

From the repository root, train an anchor and then train a target head.

```bash
python warm_start/train_anchor.py --episodes 1000 --out-dir runs/anchor
python warm_start/train_transfer.py --agent orthogonal --anchor-ckpt runs/anchor/checkpoints/anchor_final.pt --episodes 1000 --out-dir runs/orthogonal
```

Use `--agent diagonal` for diagonal movement. The transfer script freezes the anchor encoder and trains the target action head. See [warm-start instructions](warm_start/README.md).

## Cold-start

Run the proposed method for any of the three MuJoCo tasks.

```bash
python cold_start/humanoid/proposed.py --steps 1000000 --out-dir runs/humanoid
python cold_start/standup/proposed.py --steps 1000000 --out-dir runs/standup
python cold_start/pusher/proposed.py --steps 500000 --out-dir runs/pusher
```

Use `--help` to see available settings. Models and return histories are saved under the selected output directory. Pretrained MobileNet weights are downloaded by torchvision on the first run.

## Quick checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q warm_start cold_start
```

The warm-start code has a local lightweight environment. The cold-start code requires Gymnasium Robotics/MaMuJoCo and MuJoCo.

## Notes

Residual adapters are used only in the proposed **cold-start** method. This repository does not include distributed SAC, HATRPO or distributed DDQN code and cannot reproduce the baseline comparison runs. The original training environment, package versions, and evaluation configuration should be confirmed before citing this repository as a complete reproducibility release.
