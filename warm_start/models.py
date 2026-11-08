from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorEncoder(nn.Module):
    """CNN feature extractor matching the original IT-DQN architecture up to fc1."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(256, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc1(x))


class DQN(nn.Module):
    """Original IT-DQN model: encoder + action head."""

    def __init__(self, num_actions: int, latent_dim: int = 128):
        super().__init__()
        self.encoder = AnchorEncoder(latent_dim=latent_dim)
        self.head = nn.Linear(latent_dim, num_actions)

    @property
    def conv1(self): return self.encoder.conv1
    @property
    def conv2(self): return self.encoder.conv2
    @property
    def conv3(self): return self.encoder.conv3
    @property
    def fc1(self): return self.encoder.fc1
    @property
    def fc2(self): return self.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class TransferDQN(nn.Module):
    """A frozen anchor encoder with a trainable action head."""
    def __init__(self, num_actions: int, method: str = "anchor_only",
                 anchor_encoder: Optional[nn.Module] = None, latent_dim: int = 128):
        super().__init__()
        if method != "anchor_only":
            raise ValueError("This proposed-only release supports anchor_only")
        if anchor_encoder is None:
            raise ValueError("anchor_encoder is required")
        self.method = method
        self.encoder = anchor_encoder
        freeze_module(self.encoder)
        self.head = nn.Linear(latent_dim, num_actions)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_anchor_encoder(checkpoint_path: str, device: torch.device, latent_dim: int = 128) -> AnchorEncoder:
    """Load the feature encoder from an anchor checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder = AnchorEncoder(latent_dim=latent_dim).to(device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        old_model = ckpt
        state = old_model.state_dict()

    enc_state = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            enc_state[k.replace("encoder.", "")] = v
        elif k in {"conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "conv3.weight", "conv3.bias", "fc1.weight", "fc1.bias"}:
            enc_state[k] = v
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing:
        raise ValueError(f"Could not load complete anchor encoder from {checkpoint_path}. Missing: {missing}")
    freeze_module(encoder)
    return encoder
