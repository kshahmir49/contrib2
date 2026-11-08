import argparse
import os

# Use MUJOCO_GL=egl or MUJOCO_GL=osmesa if your system needs it.

import cv2
import time
import random
import numpy as np
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


device = (
    "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
)
print("Using device:", device)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_env(render_mode="rgb_array", seed: int | None = None):
    try:
        from gymnasium_robotics.mamujoco_v1 import parallel_env as mamujoco_parallel_env
        env = mamujoco_parallel_env("HumanoidStandup", "9|8", render_mode=render_mode)
    except ModuleNotFoundError:
        # Fallback import path (older distro)
        from gymnasium_robotics.envs.multiagent_mujoco.mamujoco_v1 import parallel_env as mamujoco_parallel_env
        env = mamujoco_parallel_env(scenario="HumanoidStandup", agent_conf="9|8", render_mode=render_mode)
    env.reset(seed=seed)
    return env


INPUT_SIZE = 128  # speed/quality trade-off


def preprocess_rgb(frame_rgb: np.ndarray) -> torch.Tensor:
    """RGB uint8 HxWx3 -> float tensor [1,3,H,W] in [0,1]."""
    img = cv2.resize(frame_rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    t = torch.from_numpy(img).unsqueeze(0)  # [1,3,H,W]
    return t


class FrozenMobileNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        m = mobilenet_v3_small(weights=weights)
        self.backbone = m.features  # conv body
        for p in self.backbone.parameters():
            p.requires_grad = False
        # imagenet norm
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("im_mean", mean)
        self.register_buffer("im_std", std)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # infer feat dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
            f = self.pool(self.backbone(dummy)).flatten(1)
            self.feat_dim = f.shape[1]

    @torch.no_grad()
    def encode(self, imgs_01: torch.Tensor) -> torch.Tensor:
        """imgs_01: [B,3,H,W] in [0,1] -> features [B,C]."""
        x = (imgs_01 - self.im_mean) / self.im_std
        z = self.backbone(x)
        z = self.pool(z).flatten(1)
        return z


class RunningNorm:
    def __init__(self, eps: float = 1e-5):
        self.mean = None; self.var = None; self.count = eps
    def update(self, x: np.ndarray):
        bmean = x.mean(axis=0); bvar = x.var(axis=0); n = x.shape[0]
        if self.mean is None:
            self.mean, self.var, self.count = bmean, bvar, n
        else:
            delta = bmean - self.mean
            tot = self.count + n
            new_mean = self.mean + delta * n / tot
            m_a = self.var * self.count
            m_b = bvar * n
            M2 = m_a + m_b + delta**2 * self.count * n / tot
            self.mean, self.var, self.count = new_mean, M2 / tot, tot
    def normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std  = torch.tensor(np.sqrt(self.var + 1e-8), dtype=x.dtype, device=x.device)
        return (x - mean) / (std + 1e-6)


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(dim, bottleneck)
        self.fc2 = nn.Linear(bottleneck, dim)
        self.act = nn.ReLU()
    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


class GaussianPolicyHead(nn.Module):
    """Features -> Gaussian policy (mean/log_std), tanh-squashed to [-1, 1]."""
    def __init__(self, feat_dim: int, action_dim: int, hidden: int = 128, bottleneck: int = 32):
        super().__init__()
        self.adapter = ResidualAdapter(feat_dim, bottleneck)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        self.LOG_STD_MIN = -5.0
        self.LOG_STD_MAX = 2.0

    def forward(self, f: torch.Tensor):
        z = self.adapter(f)
        h = self.net(z)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, f: torch.Tensor):
        mu, log_std = self(f)
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        u = mu + std * eps                 # reparameterization
        a = torch.tanh(u)
        # log prob with tanh correction
        log_prob = (-0.5 * ((u - mu) / (std + 1e-8))**2 - log_std - 0.5*np.log(2*np.pi)).sum(dim=1)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=1)
        return a, log_prob, torch.tanh(mu)


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.q1(x).squeeze(1), self.q2(x).squeeze(1)

# Replay buffer

class ReplayBuffer:
    def __init__(self, capacity=300_000):
        self.buf = deque(maxlen=capacity)
    def push(self, f, a, r, f2, d_term, s, s2):
        self.buf.append((f, a, r, f2, d_term, s, s2))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        f, a, r, f2, d, s, s2 = zip(*batch)
        return (
            np.stack(f), np.stack(a), np.stack(r), np.stack(f2),
            np.stack(d).astype(np.float32), np.stack(s), np.stack(s2)
        )
    def __len__(self):
        return len(self.buf)

# SAC Agent (per MaMuJoCo agent) with shared encoder

@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.01
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy_coef: float = 1.0
    grad_clip: float = 1.0


class SACAgent:
    def __init__(self, action_space, state_dim: int, feat_dim: int, shared_encoder: FrozenMobileNetEncoder, cfg=SACConfig()):
        self.device = device
        self.enc = shared_encoder
        self.cfg = cfg
        self.alow = torch.tensor(action_space.low, dtype=torch.float32, device=self.device)
        self.ahi  = torch.tensor(action_space.high, dtype=torch.float32, device=self.device)
        self.adim = int(action_space.shape[0])

        self.policy = GaussianPolicyHead(feat_dim, self.adim).to(self.device)
        self.q = TwinQ(state_dim, self.adim).to(self.device)
        self.q_targ = TwinQ(state_dim, self.adim).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict())

        self.opt_pi = optim.Adam(self.policy.parameters(), lr=cfg.actor_lr)
        self.opt_q  = optim.Adam(self.q.parameters(),      lr=cfg.critic_lr)

        # Entropy temperature (auto-alpha)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        # Target entropy = -|A|
        self.target_entropy = -cfg.target_entropy_coef * float(self.adim)
        self.opt_alpha = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        self.state_norm = RunningNorm()

    def _scale_to_env(self, a_tanh: torch.Tensor) -> torch.Tensor:
        return self.alow + (a_tanh + 1.0) / 2.0 * (self.ahi - self.alow)

    @torch.no_grad()
    def act(self, fvec: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            _, _, a_mu = self.policy.sample(fvec)  # a_mu is tanh(mean)
            a = self._scale_to_env(a_mu)
        else:
            a_tanh, _, _ = self.policy.sample(fvec)
            a = self._scale_to_env(a_tanh)
        a = torch.clamp(a, self.alow, self.ahi)
        return a.cpu().numpy().flatten().astype(np.float32)

    def update(self, batch):
        cfg = self.cfg
        f, a, r, f2, d, s, s2 = batch
        f  = torch.tensor(f,  dtype=torch.float32, device=self.device)
        f2 = torch.tensor(f2, dtype=torch.float32, device=self.device)
        a  = torch.tensor(a,  dtype=torch.float32, device=self.device)
        r  = torch.tensor(r,  dtype=torch.float32, device=self.device)
        d  = torch.tensor(d,  dtype=torch.float32, device=self.device)
        s  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        # normalize states for critic
        s  = self.state_norm.normalize_tensor(s)
        s2 = self.state_norm.normalize_tensor(s2)

        # ----- Critic update -----
        with torch.no_grad():
            a2_tanh, logp2, _ = self.policy.sample(f2)
            a2 = self._scale_to_env(a2_tanh)
            q1_t, q2_t = self.q_targ(s2, a2)
            q_targ = torch.min(q1_t, q2_t) - torch.exp(self.log_alpha) * logp2
            y = r + cfg.gamma * (1 - d) * q_targ

        q1, q2 = self.q(s, a)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_q.zero_grad(); loss_q.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), cfg.grad_clip)
        self.opt_q.step()

        # ----- Actor & alpha update -----
        a_tanh, logp, _ = self.policy.sample(f)
        a_pi = self._scale_to_env(a_tanh)
        q1_pi, q2_pi = self.q(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        alpha = torch.exp(self.log_alpha.detach())
        loss_pi = (alpha * logp - q_pi).mean()
        self.opt_pi.zero_grad(); loss_pi.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
        self.opt_pi.step()

        # temperature (alpha)
        loss_alpha = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad(); loss_alpha.backward(); self.opt_alpha.step()

        # ----- Target update -----
        with torch.no_grad():
            for tp, p in zip(self.q_targ.parameters(), self.q.parameters()):
                tp.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)

        return float(loss_q.item()), float(loss_pi.item()), float(torch.exp(self.log_alpha).item())


def train_sac(total_steps_target=10000, episodes=200, max_steps=1000, seed=0, batch_size=256, warmup_steps=10_000,
              updates_per_step=1, replay_capacity=300_000, fps=None):
    set_global_seed(seed)
    env = make_env(render_mode="rgb_array", seed=seed)
    agents = list(env.agents)
    action_spaces = {a: env.action_space(a) for a in agents}
    obs_spaces    = {a: env.observation_space(a) for a in agents}

    # Shared visual encoder
    encoder = FrozenMobileNetEncoder().to(device)
    feat_dim = encoder.feat_dim

    # Per-agent SAC + buffers
    sac = {}
    buffers = {}
    for a in agents:
        state_dim = int(np.prod(obs_spaces[a].shape))
        sac[a] = SACAgent(action_spaces[a], state_dim, feat_dim, encoder)
        buffers[a] = ReplayBuffer(capacity=replay_capacity)

    # Print hetero action dims
    dims = {a: action_spaces[a].shape[0] for a in agents}
    print("Action dims per agent:", dims)

    total_steps = 0
    returns_hist = []
    ep = 0

    # for ep in range(1, episodes + 1):
    while total_steps <= total_steps_target:
        ep += 1
        _reset = env.reset(seed=seed + ep)
        obs = _reset[0] if isinstance(_reset, tuple) else _reset
        done = {a: False for a in agents}
        ep_ret = {a: 0.0 for a in agents}

        # Initial feature from rendered frame
        frame = env.render()
        img_t = preprocess_rgb(frame).to(device)
        fvec = encoder.encode(img_t)  # [1, C]

        step = 0
        while not all(done.values()) and step < max_steps:
            actions = {}
            for a in agents:
                if done[a]:
                    actions[a] = None
                else:
                    if total_steps < warmup_steps:
                        act = action_spaces[a].sample()
                        actions[a] = np.asarray(act, dtype=np.float32)
                    else:
                        actions[a] = sac[a].act(fvec, deterministic=False)

            next_obs, rewards, term, trunc, infos = env.step(actions)
            frame_next = env.render()
            img_tp1 = preprocess_rgb(frame_next).to(device)
            fvec_next = encoder.encode(img_tp1)

            # Store transitions per-agent (use terminal-only flag for bootstrap)
            for a in agents:
                r = float(rewards.get(a, 0.0))
                term_flag  = float(term.get(a, False))
                trunc_flag = float(trunc.get(a, False))
                if not done[a]:
                    s  = np.asarray(obs[a],      dtype=np.float32).reshape(-1)
                    s2 = np.asarray(next_obs[a], dtype=np.float32).reshape(-1)
                    sac[a].state_norm.update(np.stack([s, s2], axis=0))
                    buffers[a].push(
                        fvec.cpu().numpy()[0], actions[a], r,
                        fvec_next.cpu().numpy()[0], term_flag, s, s2
                    )
                    ep_ret[a] += r
                done[a] = bool(term_flag or trunc_flag)

            # Learn
            if total_steps >= warmup_steps:
                for a in agents:
                    if len(buffers[a]) >= batch_size:
                        for _ in range(updates_per_step):
                            batch = buffers[a].sample(batch_size)
                            loss_q, loss_pi, alpha = sac[a].update(batch)

            obs = next_obs
            fvec = fvec_next
            step += 1
            total_steps += 1

            if fps is not None:
                cv2.imshow("SAC MultiAgent (RGB)", cv2.cvtColor(frame_next, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(max(1, int(1000 / max(1, fps)))) & 0xFF in (27, ord('q')):
                    env.close(); cv2.destroyAllWindows(); return

        avg_ret = float(np.mean(list(ep_ret.values())))
        returns_hist.append(avg_ret)
        print(f"Episode {ep:04d} | steps {step} | avg return {avg_ret:.2f} | total_steps: {total_steps}")

    env.close()
    if fps is not None:
        cv2.destroyAllWindows()
    return returns_hist, sac


def main():
    parser = argparse.ArgumentParser(description="Train the proposed shared-perception SAC method")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=60000)
    parser.add_argument("--replay-capacity", type=int, default=500000)
    parser.add_argument("--out-dir", default="runs/standup")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    returns, agents = train_sac(
        total_steps_target=args.steps,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        replay_capacity=args.replay_capacity,
    )
    for name, agent in agents.items():
        torch.save(agent.policy.state_dict(), os.path.join(args.out_dir, f"actor_{name}.pt"))
        torch.save(agent.q.state_dict(), os.path.join(args.out_dir, f"critic_{name}.pt"))
    np.save(os.path.join(args.out_dir, "returns.npy"), np.asarray(returns))


if __name__ == "__main__":
    main()
