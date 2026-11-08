"""PPO-v2: pre-specified ablation of the narrow-corridor failure of PPO (written before any v2 training).

Hypothesis
    The original PPO (rl/train.py; 300k steps, 3 seeds) reaches 0.93 success on open maps but only 0.15 on
    narrow-corridor maps, mostly by timing out (0.67) or colliding (0.17). A post-hoc runtime shield removes
    most collisions but does not raise narrow-map success (0.17). Hypothesis H-v2: the narrow failure is caused
    by (a) a myopic reward that under-penalises contact relative to the progress term, and (b) the memoryless
    single-frame observation (no way to tell "I just turned away from this wall" from "I am facing a new wall").
    If H-v2 holds, PPO-v2 should raise narrow-map success clearly above PPO (and above PPO+shield) on the same
    held-out maps, without giving up open/cluttered performance.

Changes vs. PPO (everything else identical: NavEnv dynamics/observation, RL_TRAIN map pool, SB3 PPO defaults,
MlpPolicy, VecNormalize(norm_obs=True, norm_reward=True), n_envs=1, seeds 0/1/2, torch 1 thread per process)
    1. Reward v2 (NavEnv options, defaults untouched for the original agents):
         r = 10*(d_goal[t-1]-d_goal[t]) - 0.05 - 1.0*[min_ray < 0.30] + 100*[goal] - 200*[collision]
       i.e. collision -200 (was -100), proximity -1.0/step (was -0.5). Progress term and step cost unchanged.
       No exploration / novelty term is added.
    2. Memory: last 4 observations stacked with SB3 VecFrameStack(n_stack=4), applied ON TOP of VecNormalize
       (DummyVecEnv -> VecNormalize -> VecFrameStack), so normalisation statistics are per 39-dim frame and the
       policy sees 4 normalised frames (156 dims, oldest first, newest last; zero-padded after reset).
       Evaluation reproduces this exactly: frozen VecNormalize stats normalise each raw obs, and a per-episode
       stack is reset to zeros at episode start with the first normalised obs placed in the newest slot.
    3. Budget: 1,000,000 env steps per seed, 3 seeds trained in 3 parallel processes. Fallback rule, decided
       before seeing any evaluation result: if a short timing pilot (3 parallel processes, training maps only)
       projects > 75 min wall-clock for 1M steps, train 600k steps instead. The decision and the pilot
       throughput are written to models/train_v2_log.txt.
    4. Evaluation: rl/evaluate.py agents "ppo_v2" and "ppo_v2_shield" (the unchanged post-hoc `shield`) on the
       same RL_EVAL maps, evaluated once, after training. Reported against the AC8 thresholds as rows prefixed
       "[ablation v2]" (descriptive; the original AC8 rows are unchanged). No hyper-parameter is tuned on
       RL_EVAL; nothing about this design is changed after seeing evaluation results.

Decision rule (stated in advance)
    H-v2 is "supported" if mean narrow success of ppo_v2 is >= ppo + 0.15 (absolute) AND the worst ppo_v2 seed
    beats the best ppo seed on narrow maps; "partially supported" if only narrow collisions fall (or success
    rises by less than that); "not supported" otherwise.

Outputs: models/ppo_v2_seed{s}.zip, models/ppo_v2_seed{s}_vecnorm.pkl, models/ppo_v2_seed{s}.monitor.csv,
         models/train_v2_log.txt
Run:  python -m rl.train_v2          (env PPO_V2_STEPS overrides the step budget; PPO_V2_OUT the output dir)
"""
from __future__ import annotations

import os
import time
from multiprocessing import Pool

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS = os.environ.get("PPO_V2_OUT", os.path.join(ROOT, "models"))
PPO_V2_STEPS = int(os.environ.get("PPO_V2_STEPS", 1_000_000))
SEEDS = (0, 1, 2)
N_STACK = 4
N_ENVS = 1
REWARD_V2 = {"collision_penalty": 200.0, "proximity_penalty": 1.0}
LOG = os.path.join(MODELS, "train_v2_log.txt")


def make_env_v2(**kw):
    from rl.env import NavEnv
    return NavEnv(**REWARD_V2, **kw)


def train_ppo_v2(seed):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
    t0 = time.time()
    mon = os.path.join(MODELS, f"ppo_v2_seed{seed}")

    def make():
        return Monitor(make_env_v2(), mon, info_keywords=("success", "collision"))

    vn = VecNormalize(DummyVecEnv([make] * N_ENVS), norm_obs=True, norm_reward=True)
    venv = VecFrameStack(vn, n_stack=N_STACK)
    venv.seed(seed)
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, device="cpu")
    model.learn(total_timesteps=PPO_V2_STEPS)
    model.save(os.path.join(MODELS, f"ppo_v2_seed{seed}.zip"))
    vn.save(os.path.join(MODELS, f"ppo_v2_seed{seed}_vecnorm.pkl"))
    return f"PPO-v2 seed {seed}: {PPO_V2_STEPS} steps, {time.time() - t0:.0f}s"


def main():
    os.makedirs(MODELS, exist_ok=True)
    t0 = time.time()
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] start PPO-v2: steps={PPO_V2_STEPS} seeds={SEEDS} "
                f"n_envs={N_ENVS} n_stack={N_STACK} reward={REWARD_V2} processes={len(SEEDS)}\n")
    with Pool(len(SEEDS)) as p:
        for msg in p.imap_unordered(train_ppo_v2, SEEDS):
            print(msg, flush=True)
            with open(LOG, "a") as f:
                f.write(msg + "\n")
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] done, wall-clock {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()
