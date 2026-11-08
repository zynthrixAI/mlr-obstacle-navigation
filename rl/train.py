"""Train tabular Q-learning (3 seeds) and PPO (3 seeds) in parallel processes.

Run:  python -m rl.train            (env PPO_STEPS overrides 300_000)
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")
PPO_STEPS = int(os.environ.get("PPO_STEPS", 300_000))
Q_EPISODES = int(os.environ.get("Q_EPISODES", 5000))
SEEDS = (0, 1, 2)


def train_q(seed):
    from rl.env import NavEnv
    from rl.qlearning import train
    t0 = time.time()
    Q, hist = train(NavEnv(), episodes=Q_EPISODES, seed=seed)
    np.save(os.path.join(MODELS, f"q_seed{seed}.npy"), Q)
    np.save(os.path.join(MODELS, f"q_seed{seed}_hist.npy"), hist)
    return f"Q seed {seed}: {time.time() - t0:.0f}s, last-500 success {hist[-500:, 2].mean():.2f}"


def train_ppo(seed):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from rl.env import NavEnv
    t0 = time.time()
    mon = os.path.join(MODELS, f"ppo_seed{seed}")

    def make():
        return Monitor(NavEnv(), mon, info_keywords=("success", "collision"))

    venv = VecNormalize(DummyVecEnv([make]), norm_obs=True, norm_reward=True)
    venv.seed(seed)
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, device="cpu")
    model.learn(total_timesteps=PPO_STEPS)
    model.save(os.path.join(MODELS, f"ppo_seed{seed}.zip"))
    venv.save(os.path.join(MODELS, f"ppo_seed{seed}_vecnorm.pkl"))
    return f"PPO seed {seed}: {time.time() - t0:.0f}s"


def _job(j):
    kind, seed = j
    return train_q(seed) if kind == "q" else train_ppo(seed)


def main():
    os.makedirs(MODELS, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = []
    if which in ("all", "ppo"):
        jobs += [("ppo", s) for s in SEEDS]
    if which in ("all", "q"):
        jobs += [("q", s) for s in SEEDS]
    with Pool(min(len(jobs), os.cpu_count())) as p:
        for msg in p.imap_unordered(_job, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
