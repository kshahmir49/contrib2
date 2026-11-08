import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "warm_start"))

from dqn import DuelingDoubleDQN
from environment import TargetSearch
from models import DuelingQNetwork


class WarmStartTests(unittest.TestCase):
    def test_state_and_actions(self):
        for name, actions in [("irregular", 12), ("orthogonal", 4), ("diagonal", 4)]:
            env = TargetSearch(name, seed=1)
            state = env.reset()
            self.assertEqual(state.shape, (3, 50, 50))
            self.assertEqual(env.num_actions, actions)
            self.assertEqual(state.dtype, np.float32)

    def test_dueling_network(self):
        model = DuelingQNetwork(4)
        output = model(torch.zeros(2, 3, 50, 50))
        self.assertEqual(output.shape, (2, 4))
        self.assertEqual(model.encoder.conv1.in_channels, 3)
        self.assertEqual(model.encoder.fc.out_features, 128)
        self.assertEqual(model.head.value[-1].out_features, 1)
        self.assertEqual(model.head.advantage[-1].out_features, 4)

    def test_frozen_encoder(self):
        model = DuelingQNetwork(4)
        model.freeze_encoder()
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))

    def test_double_dqn_huber(self):
        model = DuelingQNetwork(4)
        learner = DuelingDoubleDQN(model, torch.device("cpu"), target_update_steps=2)
        self.assertEqual(learner.target_update_steps, 2)
        state = np.zeros((3, 50, 50), dtype=np.float32)
        transition = [(state, 0, 0.5, state.copy(), False)]
        loss = learner.update(transition)
        self.assertTrue(np.isfinite(loss))
        learner.advance_step()
        learner.advance_step()
        self.assertEqual(learner.steps, 2)

    def test_reward_formula(self):
        env = TargetSearch("orthogonal", seed=1)
        env.reset()
        env.position = (20, 20)
        env.target = (80, 20)
        env.distance = 60.0
        _, reward, done, _ = env.step(2)
        self.assertAlmostEqual(reward, -0.01 + 0.2 * (60.0 - 58.0) / 100.0)
        self.assertFalse(done)


class ColdStartTests(unittest.TestCase):
    def test_proposed_architectures(self):
        for name in ("humanoid", "standup", "pusher"):
            src = (ROOT / "cold_start" / name / "proposed.py").read_text()
            self.assertIn("class ResidualAdapter", src)
            self.assertIn("return x +", src)
            self.assertIn('default=120000', src)
            self.assertIn('default=500000', src)
            self.assertIn('default=1000', src)
            self.assertNotIn("DistributedSAC", src)


if __name__ == "__main__":
    unittest.main()
