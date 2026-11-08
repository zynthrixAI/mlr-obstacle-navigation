"""Evaluate random / reactive / Q-learning / PPO / PPO+safety shield on the held-out RL_EVAL maps (used once, after training).

Outputs: results/T5_rl.csv, results/T5_rl_episodes.csv, results/T5_acceptance.csv,
         results/F6_rl_success.png, results/F7_rl_learning.png
Run:  python -m rl.evaluate
"""
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl.env import NavEnv
from rl.qlearning import QPolicy
from rl.train import MODELS, RESULTS, SEEDS
from sim.robot import reactive_action, sectors36
from sim.seeds import MAP_TYPES, RL_EVAL, RL_TRAIN

assert not set(RL_EVAL) & set(RL_TRAIN)

# Safety shield (post-hoc, fixed before evaluation; no retraining). A forward move is vetoed when any
# LiDAR ray in the front half-plane hits inside a corridor the robot body would sweep:
# lateral offset < SHIELD_HALF_W and forward distance < SHIELD_AHEAD. The veto replaces "forward"
# with the reactive rule's turn (towards the more open side). Dropped-out rays (NaN) count as free.
SHIELD_HALF_W = 0.20      # robot radius 0.15 m + 0.05 m margin
SHIELD_AHEAD = 0.35       # robot radius 0.15 m + one 0.1 m step + 0.10 m margin
_ANG = np.deg2rad(np.arange(36) * 10.0)


def shield(a, rays):
    if a != 0:
        return a
    r = np.nan_to_num(rays, nan=1e6)                     # dropout = free
    fwd, lat = r * np.cos(_ANG), np.abs(r * np.sin(_ANG))
    if np.any((fwd > 0) & (fwd < SHIELD_AHEAD) & (lat < SHIELD_HALF_W)):
        _, l, rr = sectors36(rays)
        return 1 if l >= rr else 2
    return a


def run_policy(name, seed, act_fn, obs_fn=None):
    env = NavEnv(seed_pool=RL_EVAL)
    env.reset(seed=1000 + seed)              # seeds the sensor-noise RNG only
    rows = []
    rng = np.random.default_rng(seed)
    for ms in RL_EVAL:
        obs, inf = env.reset(options={"map_seed": ms})
        done = False
        while not done:
            a = act_fn(obs, env, rng)
            obs, r, te, tr, info = env.step(a)
            done = te or tr
        rows.append({"agent": name, "seed": seed, "map_seed": ms, **info,
                     "timeout": not (info["success"] or info["collision"]),
                     "path_ratio": info["path"] / info["start_dist"] if info["success"] else np.nan})
    return rows


def main():
    rows = []
    rows += run_policy("random", 0, lambda o, e, g: int(g.integers(3)))
    rows += run_policy("reactive", 0, lambda o, e, g: reactive_action(e.last_rays, np.arctan2(o[37], o[38])))
    for s in SEEDS:
        q = QPolicy(np.load(os.path.join(MODELS, f"q_seed{s}.npy")))
        rows += run_policy("qlearning", s, lambda o, e, g: q.act(o))
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    for s in SEEDS:
        model = PPO.load(os.path.join(MODELS, f"ppo_seed{s}.zip"), device="cpu")
        vn = VecNormalize.load(os.path.join(MODELS, f"ppo_seed{s}_vecnorm.pkl"), DummyVecEnv([NavEnv]))
        vn.training, vn.norm_reward = False, False          # frozen statistics
        act = lambda o, e, g: int(model.predict(vn.normalize_obs(o[None]), deterministic=True)[0][0])
        rows += run_policy("ppo", s, act)
        rows += run_policy("ppo_shield", s, lambda o, e, g: shield(act(o, e, g), e.last_rays))

    ep = pd.DataFrame(rows)
    ep.to_csv(os.path.join(RESULTS, "T5_rl_episodes.csv"), index=False)
    metrics = ["success", "collision", "timeout", "path_ratio", "min_clear", "near_frac"]
    per_seed = ep.groupby(["agent", "seed", "map_type"])[metrics].mean().reset_index()
    summ = per_seed.groupby(["agent", "map_type"])[metrics].agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(os.path.join(RESULTS, "T5_rl.csv"), index=False)
    print(summ[["agent", "map_type", "success_mean", "success_std", "collision_mean", "path_ratio_mean",
                "min_clear_mean", "near_frac_mean"]].round(3).to_string(index=False))

    # AC8
    acc = []
    thr = {"open": 0.80, "narrow": 0.60, "cluttered": 0.60}
    for mt in MAP_TYPES:
        p = summ[(summ.agent == "ppo") & (summ.map_type == mt)].iloc[0]
        rx = summ[(summ.agent == "reactive") & (summ.map_type == mt)].iloc[0]
        acc.append({"criterion": f"PPO success >= {thr[mt]:.0%} ({mt})", "value": round(p.success_mean, 3),
                    "pass": bool(p.success_mean >= thr[mt])})
        acc.append({"criterion": f"PPO collision <= reactive ({mt})",
                    "value": f"{p.collision_mean:.3f} vs {rx.collision_mean:.3f}",
                    "pass": bool(p.collision_mean <= rx.collision_mean)})
    for mt in MAP_TYPES:                                   # descriptive, post-hoc (not pre-specified AC8)
        p = summ[(summ.agent == "ppo_shield") & (summ.map_type == mt)].iloc[0]
        rx = summ[(summ.agent == "reactive") & (summ.map_type == mt)].iloc[0]
        acc.append({"criterion": f"[post-hoc] PPO+shield success >= {thr[mt]:.0%} ({mt})",
                    "value": round(p.success_mean, 3), "pass": bool(p.success_mean >= thr[mt])})
        acc.append({"criterion": f"[post-hoc] PPO+shield collision <= reactive ({mt})",
                    "value": f"{p.collision_mean:.3f} vs {rx.collision_mean:.3f}",
                    "pass": bool(p.collision_mean <= rx.collision_mean)})
    pd.DataFrame(acc).to_csv(os.path.join(RESULTS, "T5_acceptance.csv"), index=False)
    print(pd.DataFrame(acc).to_string(index=False))

    # F6 success / collision by map type
    agents = ["random", "reactive", "qlearning", "ppo", "ppo_shield"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axes, ["success", "collision"]):
        x = np.arange(len(MAP_TYPES))
        for i, a in enumerate(agents):
            d = summ[summ.agent == a].set_index("map_type").loc[list(MAP_TYPES)]
            ax.bar(x + (i - 2) * 0.16, d[f"{m}_mean"], 0.16, yerr=d[f"{m}_std"].fillna(0), label=a, capsize=3)
        ax.set_xticks(x, MAP_TYPES)
        ax.set_ylabel(f"{m} rate")
        ax.set_ylim(0, 1)
        ax.set_title(f"{m.capitalize()} rate on 50 held-out maps/type (mean ± std over seeds)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "F6_rl_success.png"), dpi=150)

    # F7 learning curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for s in SEEDS:
        f = glob.glob(os.path.join(MODELS, f"ppo_seed{s}*.monitor.csv"))
        if f:
            m = pd.read_csv(f[0], skiprows=1)
            axes[0].plot(m.l.cumsum(), m.success.rolling(200, min_periods=20).mean(), label=f"PPO seed {s}")
        h = np.load(os.path.join(MODELS, f"q_seed{s}_hist.npy"))
        axes[1].plot(h[:, 0], pd.Series(h[:, 2]).rolling(200, min_periods=20).mean(), label=f"Q seed {s}")
    axes[0].set_xlabel("environment steps")
    axes[1].set_xlabel("episode")
    for ax, t in zip(axes, ["PPO training success (rolling 200 ep.)", "Q-learning training success (rolling 200 ep.)"]):
        ax.set_title(t)
        ax.set_ylabel("success rate")
        ax.set_ylim(0, 1)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "F7_rl_learning.png"), dpi=150)


if __name__ == "__main__":
    main()
