"""Tabular Q-learning on a 216-state discretisation.

state = 3 sectors (front +-30, left 30..90, right -30..-90 deg) x 3 distance bins (<0.3, 0.3-1.0, >1.0 m)
        x 8 goal-angle bins  -> 27 * 8 = 216 states
alpha = 0.1, gamma = 0.99, epsilon linear 1.0 -> 0.05 over the first 70 % of 5000 episodes.
"""
from __future__ import annotations

import numpy as np

from sim.robot import sectors36

N_STATES, N_ACTIONS = 216, 3


def obs_to_state(obs):
    rays = obs[:36] * 5.0
    sec = sectors36(rays)
    bins = [0 if s < 0.3 else (1 if s < 1.0 else 2) for s in sec]
    ang = np.arctan2(obs[37], obs[38])
    ab = int(((ang + np.pi) / (2 * np.pi) * 8)) % 8
    return ((bins[0] * 3 + bins[1]) * 3 + bins[2]) * 8 + ab


def train(env, episodes=5000, seed=0, alpha=0.1, gamma=0.99, log=None):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    hist = []
    obs, _ = env.reset(seed=seed)
    for ep in range(episodes):
        eps = max(0.05, 1.0 - 0.95 * ep / (0.7 * episodes))
        obs, _ = env.reset()
        s = obs_to_state(obs)
        ret, done = 0.0, False
        while not done:
            a = int(rng.integers(3)) if rng.random() < eps else int(np.argmax(Q[s]))
            obs, r, term, trunc, info = env.step(a)
            s2 = obs_to_state(obs)
            target = r if term else r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            s, ret, done = s2, ret + r, term or trunc
        hist.append((ep, ret, info["success"], info["collision"]))
    return Q, np.array(hist)


class QPolicy:
    def __init__(self, Q):
        self.Q = Q

    def act(self, obs):
        return int(np.argmax(self.Q[obs_to_state(obs)]))
