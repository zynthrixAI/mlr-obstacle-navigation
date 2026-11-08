"""Gymnasium navigation env on the shared simulator.

obs  = 36 LiDAR rays / 5 (dropout -> 1.0) + goal distance / 10 + sin, cos(goal angle)   (39 dims)
act  = {0 forward 0.1 m, 1 rotate +15 deg, 2 rotate -15 deg}
r    = 10*(d_goal[t-1]-d_goal[t]) - 0.05 - 0.5*[min_ray < 0.30] + 100*[goal] - 100*[collision]
ends : goal (< 0.2 m), collision, or 400 steps (truncation)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from sim.robot import MAX_RANGE, goal_polar, lidar, step_pose
from sim.seeds import RL_TRAIN
from sim.world import make_world

MAX_STEPS = 400
GOAL_TOL = 0.2


def start_goal_for(seed: int):
    """Deterministic start pose / goal for a map seed (used for held-out evaluation)."""
    w = make_world(seed)
    rng = np.random.default_rng(seed + 1)
    s, g = w.sample_start_goal(rng)
    return (float(s[0]), float(s[1]), float(rng.uniform(-np.pi, np.pi))), g


class NavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed_pool=RL_TRAIN):
        self.seed_pool = np.asarray(list(seed_pool))
        self.observation_space = gym.spaces.Box(-1.0, 1.5, shape=(39,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

    # ------------------------------------------------------------------
    def _obs(self):
        r, _ = lidar(self.world, self.pose, 36, self.np_random)
        self.last_rays = r
        d, a = goal_polar(self.pose, self.goal)
        rays = np.nan_to_num(r, nan=MAX_RANGE) / MAX_RANGE
        return np.r_[rays, min(d / 10.0, 1.5), np.sin(a), np.cos(a)].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        map_seed = int(options.get("map_seed", self.np_random.choice(self.seed_pool)))
        self.world = make_world(map_seed)
        if "map_seed" in options:                                   # held-out evaluation episode
            self.pose, self.goal = start_goal_for(map_seed)
        else:
            s, g = self.world.sample_start_goal(self.np_random)
            self.pose, self.goal = (s[0], s[1], float(self.np_random.uniform(-np.pi, np.pi))), g
        self.t = 0
        self.d_prev = goal_polar(self.pose, self.goal)[0]
        self.start_dist = self.d_prev
        self.path = 0.0
        self.min_clear = self.world.clearance(self.pose[:2])
        self.near_steps = 0
        return self._obs(), {"map_seed": map_seed, "map_type": self.world.map_type}

    def step(self, action):
        self.t += 1
        self.pose, collided = step_pose(self.world, self.pose, int(action))
        if int(action) == 0 and not collided:
            self.path += 0.1
        clear = self.world.clearance(self.pose[:2])
        self.min_clear = min(self.min_clear, clear)
        self.near_steps += clear < 0.30
        obs = self._obs()
        d = goal_polar(self.pose, self.goal)[0]
        min_ray = np.nanmin(self.last_rays) if not np.all(np.isnan(self.last_rays)) else MAX_RANGE
        r = 10.0 * (self.d_prev - d) - 0.05 - 0.5 * (min_ray < 0.30)
        self.d_prev = d
        success = d < GOAL_TOL
        terminated = bool(success or collided)
        if success:
            r += 100.0
        if collided:
            r -= 100.0
        truncated = self.t >= MAX_STEPS and not terminated
        info = {}
        if terminated or truncated:
            info = {"success": bool(success), "collision": bool(collided), "steps": self.t,
                    "path": self.path, "start_dist": self.start_dist, "min_clear": self.min_clear,
                    "near_frac": self.near_steps / self.t, "map_type": self.world.map_type}
        return obs, float(r), terminated, truncated, info
