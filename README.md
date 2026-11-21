# Decoupling Perception and Control for Parameter-Efficient Heterogeneous Reinforcement Learning

Official implementation of the proposed warm-start and cold-start shared-perception approaches described in **Decoupling Perception and Control for Parameter-Efficient Heterogeneous Reinforcement Learning**.

## Abstract

Heterogeneous reinforcement learning systems often contain agents or control modules that observe the same environment but act through different control interfaces. Fully distributed policies preserve this heterogeneity, but they also duplicate perception for every agent. This creates perception-side parameter redundancy when separate encoders are learned from the same shared observation. This paper studies a parameter-efficient perception-control decoupling formulation that shares the observation encoder while keeping the downstream action-producing module agent-specific. Depending on how the shared encoder is obtained, two approaches are considered. In the warm-start case, an in-domain anchor agent is trained end-to-end, its encoder is frozen, and downstream agents learn only compact dueling action heads. In the cold-start case, a frozen pretrained visual backbone is shared, and lightweight residual adapters specialize the latent representation for each agent. Warm-start is evaluated in target search, where agents share a semantic spatial observation but differ in discrete movement primitives. Cold-start is evaluated in MuJoCo visual control, where heterogeneous control modules share rendered image observations but control different actuator groups. In target search, frozen anchor transfer improves early adaptation and evaluation performance while reducing downstream trainable parameters by 76.37%. In MuJoCo, shared pretrained perception achieves higher return and stability over the implemented distributed visual-control baselines with reductions of 51.23% and 64.01% in counted encoder plus visual-policy parameters for the two- and three-module cases, respectively; these complete-configuration reductions combine encoder sharing and policy-size differences. These results indicate that shared perception can reduce redundant encoder learning while preserving heterogeneous control interfaces under shared observations.

## Proposed methods

**Proposed warm-start**

An irregular-action anchor is trained using dueling Double DQN. Its convolutional encoder is then frozen and reused by orthogonal and diagonal agents, each of which trains an action-specific dueling head.

**Proposed shared PVR**

A single frozen ImageNet-pretrained MobileNetV3-Small encoder processes the shared RGB observation. Each agent uses a residual adapter with a 32-dimensional bottleneck and its own SAC actor. Critics use proprioceptive state-action inputs.

## Repository structure

```text
warm_start/
  README.md
  environment.py
  models.py
  dqn.py
  common.py
  train_anchor.py
  train_transfer.py
cold_start/
  humanoid/proposed.py
  standup/proposed.py
  pusher/proposed.py
requirements.txt
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Cold-start experiments require MuJoCo and Gymnasium Robotics. For headless GPU rendering on supported Linux systems, configure `MUJOCO_GL=egl` if needed.

The pretrained MobileNetV3-Small weights are obtained through Torchvision when first used.

## Warm-start target search

Train the irregular anchor agent.

```bash
python warm_start/train_anchor.py --agent irregular --episodes 3000 --seed 0 --out-dir runs/anchor_seed0
```

Train the orthogonal and diagonal agents using its frozen encoder.

```bash
python warm_start/train_transfer.py --agent orthogonal --anchor-ckpt runs/anchor_seed0/anchor_best.pt --episodes 1000 --seed 0 --out-dir runs/orthogonal_seed0
python warm_start/train_transfer.py --agent diagonal --anchor-ckpt runs/anchor_seed0/anchor_best.pt --episodes 1000 --seed 0 --out-dir runs/diagonal_seed0
```

For the 3,000-episode downstream experiments, set `--episodes 3000` and use separate output directories. The [warm-start guide](warm_start/README.md) describes the environment, action spaces, and evaluation procedure.

The scripts save training logs, the selected anchor checkpoint, downstream model checkpoints, and evaluation summaries in the specified output directories.

## Cold-start MuJoCo visual control

Run the proposed shared-PVR method in each environment.

```bash
python cold_start/humanoid/proposed.py --steps 1000000 --seed 0 --out-dir runs/humanoid_seed0
python cold_start/standup/proposed.py --steps 1000000 --seed 0 --out-dir runs/standup_seed0
python cold_start/pusher/proposed.py --steps 500000 --seed 0 --out-dir runs/pusher_seed0
```

Humanoid and Humanoid Standup use the `9|8` continuous action-space split. Pusher uses the `3p` MaMuJoCo configuration, corresponding to a `3|1|3` split.

The cold-start scripts use 128 × 128 RGB inputs, batch size 256, replay capacity 500,000, and 120,000 initial exploration steps. The scripts save episode returns in `returns.npy` together with the trained actor and critic weights.

## Citation

If you use this implementation, please cite **Decoupling Perception and Control for Parameter-Efficient Heterogeneous Reinforcement Learning**.
