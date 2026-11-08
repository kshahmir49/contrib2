from __future__ import annotations

import csv
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def downscale_image(image: np.ndarray, target_size: Tuple[int, int] = (50, 50)) -> np.ndarray:
    resized = cv2.resize(image.astype(np.uint8), target_size)
    return rgb2gray(resized).astype(np.float32)


def update_frame_stack(stack: torch.Tensor, new_frame: torch.Tensor) -> torch.Tensor:
    new_stack = torch.empty_like(stack)
    new_stack[0, :, :] = new_frame
    new_stack[1:, :, :] = stack[:-1, :, :]
    return new_stack


def make_state_from_obs(obs: np.ndarray, stack: torch.Tensor, device: torch.device) -> torch.Tensor:
    frame = torch.tensor(downscale_image(obs), dtype=torch.float32, device=device)
    return update_frame_stack(stack, frame)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_csv(path: str, row: Dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: str, rows: List[Dict]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
