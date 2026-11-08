"""Task 2d: robustness of the T2c temporal-tracking gain to odometry (dead-reckoning) noise.

T2c links segments across scans with EXACT simulator poses. Here each episode's pose is replaced by a
dead-reckoning estimate: the first scan is anchored at the true pose, then every scan-to-scan increment
(10 steps) is the true increment in the previous robot frame plus N(0, sigma_t) in x and y and N(0, sigma_h)
in heading, composed onto the previous ESTIMATED pose, so the error is a random walk that accumulates
within the episode. Levels are pre-specified; the RNG is seeded per (episode_id, level) via crc32, so the
noise is reproducible and independent of labels. Linking (world-frame centroids) and t_view_spread_deg
use the estimated poses (tasks.temporal.temporal_features is called on a copy of the table whose pose_*
columns hold the estimate); segment shape features are robot-frame and unchanged.

To save time the RF uses the T2/T2c-selected hyper-parameters FIXED (min_samples_leaf=5, max_features='sqrt';
the inner grid chose these in every outer fold of both T2 and T2c) instead of the inner grid; the
engineered-only reference and the 'none' level use the same fixed parameters, so all rows are like-for-like.
Same 60k subsample, same outer GroupKFold(5) by map, map-bootstrap CIs (1000 resamples)."""
import os, time, zlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from tasks.common import *
from tasks.segment_classification import N_SUB
from tasks.temporal import temporal_features, evaluate, cm_rows

LEVELS = [("none", 0.0, 0.0), ("low", 0.02, 1.0), ("medium", 0.05, 3.0), ("high", 0.10, 6.0)]  # (name, sigma_t m, sigma_h deg)
RF_JOBS = 2  # machine shared; RF results do not depend on n_jobs (fixed random_state)


def fixed_rf():
    return RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample", random_state=SEED,
                                  n_jobs=RF_JOBS, min_samples_leaf=5, max_features="sqrt")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def noisy_poses(seg, level, sig_t, sig_h_deg):
    """Dead-reckoning pose estimate per row. Uses only pose_* / episode_id / step (never labels)."""
    sc = seg[["episode_id", "step", "pose_x", "pose_y", "pose_theta"]].drop_duplicates(["episode_id", "step"])
    sc = sc.sort_values(["episode_id", "step"], kind="stable")
    est = {}
    sig_h = np.deg2rad(sig_h_deg)
    for ep, d in sc.groupby("episode_id", sort=False):
        x, y, th = d.pose_x.values, d.pose_y.values, d.pose_theta.values
        n = len(d)
        rng = np.random.default_rng(zlib.crc32(f"{ep}|{level}".encode()))
        nt = rng.normal(0.0, 1.0, (n, 2)) * sig_t
        nh = rng.normal(0.0, 1.0, n) * sig_h
        ex, ey, eth = np.empty(n), np.empty(n), np.empty(n)
        ex[0], ey[0], eth[0] = x[0], y[0], th[0]           # anchored at the first scan
        for k in range(1, n):
            c, s = np.cos(th[k - 1]), np.sin(th[k - 1])
            dx, dy = x[k] - x[k - 1], y[k] - y[k - 1]
            lx, ly = c * dx + s * dy, -s * dx + c * dy       # true increment in previous robot frame
            lx, ly = lx + nt[k, 0], ly + nt[k, 1]
            dth = wrap(th[k] - th[k - 1]) + nh[k]
            ce, se = np.cos(eth[k - 1]), np.sin(eth[k - 1])
            ex[k] = ex[k - 1] + ce * lx - se * ly
            ey[k] = ey[k - 1] + se * lx + ce * ly
            eth[k] = wrap(eth[k - 1] + dth)
        for k, st in enumerate(d.step.values):
            est[(ep, st)] = (ex[k], ey[k], eth[k])
    keys = list(zip(seg.episode_id.values, seg.step.values))
    P = np.array([est[k] for k in keys])
    out = seg.copy()
    out["pose_x"], out["pose_y"], out["pose_theta"] = P[:, 0], P[:, 1], P[:, 2]
    if sig_t == 0 and sig_h_deg == 0:
        assert np.allclose(P, seg[["pose_x", "pose_y", "pose_theta"]].values, atol=1e-9)
    return out


def plot_f11(df):
    lv = df[df.level != "engineered-only"]
    ref = df[df.level == "engineered-only"].iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(lv))
    ax.axhspan(ref.pooled_ci_low, ref.pooled_ci_high, color="#9aa5b1", alpha=0.25, lw=0)
    ax.axhline(ref.pooled_macro_f1, c="#6b7785", lw=1.2, ls="--",
               label=f"RF engineered only ({ref.pooled_macro_f1:.3f})")
    ax.errorbar(x, lv.pooled_macro_f1, yerr=[lv.pooled_macro_f1 - lv.pooled_ci_low, lv.pooled_ci_high - lv.pooled_macro_f1],
                fmt="o-", c="#2a6fdb", capsize=4, lw=1.5, label="RF engineered + temporal")
    for xi, v in zip(x, lv.pooled_macro_f1):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xticks(x, [f"{r.level}\n(σt={r.sigma_t_m:g} m, σh={r.sigma_h_deg:g}°)" for r in lv.itertuples()], fontsize=8)
    ax.set_xlabel("odometry noise per scan interval (10 steps), random walk within episode")
    ax.set_ylabel("pooled OOF macro-F1 (map-bootstrap 95% CI)")
    lo = min(lv.pooled_ci_low.min(), ref.pooled_ci_low) - 0.02
    hi = max(lv.pooled_ci_high.max(), ref.pooled_ci_high) + 0.03
    ax.set_ylim(lo, hi)
    ax.set_title("Temporal-tracking gain vs odometry noise (fixed-param RF)")
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "F11_odometry.png"), dpi=150); plt.close(fig)


def main():
    t0 = time.time()
    full = pd.read_csv(os.path.join(DATA, "segments.csv"))
    sub_idx = full.sample(n=N_SUB, random_state=SEED).index.values   # identical subsample to T2/T2c
    seg0 = full.loc[sub_idx].reset_index(drop=True)
    y = seg0.label.values; g = seg0.map_seed.values
    folds = outer_folds(g)
    fcols = [c for c in full.columns if c.startswith("f_")]
    lab = full.label.values

    def run_rf(X):
        pred, _, _ = oof_predict(make_pipe(fixed_rf()), {}, X, y, g, "f1_macro", folds)
        return pred

    t1 = time.time()
    p_eng = run_rf(seg0[fcols].values)
    ms_e = per_map_stats(g, cm_rows(y, p_eng))
    v, l, h, _ = bootstrap_maps(ms_e, macro_f1_from_cm)
    rows = [dict(level="engineered-only", sigma_t_m=np.nan, sigma_h_deg=np.nan, link_rate=np.nan,
                 link_label_agreement=np.nan, **evaluate("RandomForest", "engineered", p_eng, y, folds),
                 pooled_ci_low=l, pooled_ci_high=h, gain_vs_eng=np.nan, gain_ci_low=np.nan, gain_ci_high=np.nan,
                 seconds=time.time() - t1)]
    print(f"[T2d] RF engineered (fixed params) pooled={v:.4f} [{l:.4f},{h:.4f}] ({time.time()-t1:.0f}s)", flush=True)

    for level, st, sh in LEVELS:
        t1 = time.time()
        est = noisy_poses(full, level, st, sh)
        tf, prev = temporal_features(est)
        ok = prev >= 0
        link_rate = ok.mean()
        agree = (lab[ok] == lab[prev[ok]]).mean()   # diagnostic only, never a feature
        # pose drift diagnostic: per-scan position/heading error vs truth
        perr = np.hypot(est.pose_x.values - full.pose_x.values, est.pose_y.values - full.pose_y.values)
        herr = np.degrees(np.abs(wrap(est.pose_theta.values - full.pose_theta.values)))
        seg = pd.concat([seg0[fcols], tf.loc[sub_idx].reset_index(drop=True)], axis=1)
        p = run_rf(seg.values)
        ms = per_map_stats(g, np.hstack([cm_rows(y, p_eng), cm_rows(y, p)]))
        v, l, h, _ = bootstrap_maps(ms, lambda u: macro_f1_from_cm(u[9:]))
        vd, ld, hd, _ = bootstrap_maps(ms, lambda u: macro_f1_from_cm(u[9:]) - macro_f1_from_cm(u[:9]))
        tl = tf.track_len.values
        rows.append(dict(level=level, sigma_t_m=st, sigma_h_deg=sh, link_rate=link_rate, link_label_agreement=agree,
                         **evaluate("RandomForest", "engineered+temporal", p, y, folds),
                         pooled_ci_low=l, pooled_ci_high=h, gain_vs_eng=vd, gain_ci_low=ld, gain_ci_high=hd,
                         frac_track_len_4plus=(tl >= 4).mean(), mean_view_spread_deg=np.nanmean(tf.t_view_spread_deg),
                         median_pos_err_m=np.median(perr), p95_pos_err_m=np.percentile(perr, 95),
                         median_head_err_deg=np.median(herr), seconds=time.time() - t1))
        print(f"[T2d] {level:6s} link={link_rate:.3f} agree={agree:.4f} pooled={v:.4f} [{l:.4f},{h:.4f}] "
              f"gain={vd:+.4f} [{ld:+.4f},{hd:+.4f}] ({time.time()-t1:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.insert(0, "note", "RF fixed params min_samples_leaf=5, max_features=sqrt (no inner grid); "
                         "pooled CI and gain CI = map bootstrap 1000; link_label_agreement diagnostic only")
    df.to_csv(os.path.join(RES, "T2d_odometry.csv"), index=False)
    print(df.drop(columns=["note"]).round(4).to_string(index=False))
    plot_f11(df)
    print(f"[T2d] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
