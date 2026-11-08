"""Minimal Gym interfaces used by the navigation environments."""
import random
import numpy as np

class Env:
    metadata = {}
    def reset(self): raise NotImplementedError
    def step(self, action): raise NotImplementedError
    def render(self, mode="human"): pass
    def close(self): pass

class _Discrete:
    def __init__(self, n, *args, **kwargs):
        self.n = int(n)
    def sample(self):
        return random.randrange(self.n)

class _Box:
    def __init__(self, low, high, shape, dtype=np.float32, *args, **kwargs):
        self.low = low; self.high = high; self.shape = shape; self.dtype = dtype
    def sample(self):
        return np.random.randint(self.low, self.high + 1, self.shape).astype(self.dtype)

class spaces:
    Discrete = _Discrete
    Box = _Box
