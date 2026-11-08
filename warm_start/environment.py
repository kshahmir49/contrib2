import random

import numpy as np


MOVES = {
    "orthogonal": [(0, 2), (0, -2), (2, 0), (-2, 0)],
    "diagonal": [(2, 2), (2, -2), (-2, 2), (-2, -2)],
    "irregular": [
        (2, 2), (2, -2), (-2, 2), (-2, -2),
        (2, 1), (1, 2), (2, -1), (1, -2),
        (-1, -2), (-2, -1), (-1, 2), (-2, 1),
    ],
}


class TargetSearch:
    def __init__(self, agent="irregular", seed=0, size=100, max_steps=200):
        if agent not in MOVES:
            raise ValueError(f"Unknown action space {agent}")
        self.agent = agent
        self.moves = MOVES[agent]
        self.size = size
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.steps = 0
        self.position = (0, 0)
        self.target = (0, 0)
        self.distance = 0.0

    @property
    def num_actions(self):
        return len(self.moves)

    def reset(self):
        self.steps = 0
        self.position = (self.rng.randrange(self.size), self.rng.randrange(self.size))
        self.target = (self.rng.randrange(self.size), self.rng.randrange(self.size))
        while self._distance() <= 5:
            self.target = (self.rng.randrange(self.size), self.rng.randrange(self.size))
        self.distance = self._distance()
        return self.observation()

    def _distance(self):
        return float(np.hypot(self.target[0] - self.position[0],
                              self.target[1] - self.position[1]))

    def observation(self):
        state = np.zeros((3, 50, 50), dtype=np.float32)
        state[2, 10:40, 10:40] = 0.25
        self._mark(state[0], self.position)
        self._mark(state[1], self.target)
        return state

    def _mark(self, plane, position):
        x = round(position[0] * 49 / (self.size - 1))
        y = round(position[1] * 49 / (self.size - 1))
        for i in range(max(0, x - 2), min(50, x + 3)):
            for j in range(max(0, y - 2), min(50, y + 3)):
                if (i - x) ** 2 + (j - y) ** 2 <= 4:
                    plane[i, j] = 1.0

    def step(self, action):
        if not 0 <= action < self.num_actions:
            raise ValueError(f"Invalid action {action}")
        dx, dy = self.moves[action]
        x, y = self.position
        self.position = (
            min(self.size - 1, max(0, x + dx)),
            min(self.size - 1, max(0, y + dy)),
        )
        self.steps += 1
        new_distance = self._distance()
        reward = -0.01 + 0.2 * (self.distance - new_distance) / self.size
        self.distance = new_distance
        success = new_distance <= 5
        timeout = self.steps >= self.max_steps
        if success:
            reward = 20.0
        elif timeout:
            reward -= 5.0
        return self.observation(), reward, success or timeout, {"success": success}
