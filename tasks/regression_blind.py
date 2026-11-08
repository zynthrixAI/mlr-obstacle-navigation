"""Post-hoc 'truly blind' stratum for Task 2 (methodology review MUST-FIX 1).

truly_blind = noiseless min(ultrasonic) - d_nearest_true > 0.05 m, i.e. no cone actually sees the nearest point.
RF (engineered features, fixed n_estimators=300) is re-run with the same GroupKFold(5)-by-map OOF protocol.
Output: results/T3b_truly_blind.csv
Run:  python -m tasks.regression_blind
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from sim.robot import ultrasonic
from sim.world import make_world
from tasks.regression import baseline, engineered

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "frames.csv"))
    clean = np.array([ultrasonic(make_world(int(s)), (x, y, t), noisy=False).min()
                      for s, x, y, t in df[["map_seed", "x", "y", "theta"]].itertuples(index=False)])
    df["truly_blind"] = (clean - df.d_nearest_true) > 0.05
    X, y, g = engineered(df), df.d_nearest_true.values, df.map_seed.values
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(X, y, g):
        assert not set(g[tr]) & set(g[te])
        m = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=4, random_state=0))
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    base = baseline(df)
    rows = []
    for name, mask in [("truly_blind", df.truly_blind.values), ("seen", ~df.truly_blind.values),
                       ("between_cones (observable=False)", ~df.observable.values), ("all", np.ones(len(df), bool))]:
        near = mask & (y < 0.5)
        rows.append({"stratum": name, "n": int(mask.sum()),
                     "baseline_mae": np.abs(base[mask] - y[mask]).mean(), "rf_eng_mae": np.abs(oof[mask] - y[mask]).mean(),
                     "baseline_near_mae": np.abs(base[near] - y[near]).mean() if near.any() else np.nan,
                     "rf_eng_near_mae": np.abs(oof[near] - y[near]).mean() if near.any() else np.nan})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(os.path.join(ROOT, "results", "T3b_truly_blind.csv"), index=False)
    # F3 (overwrites the run_all version): MAE by distance bin for three strata
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 5.01)]
    strata = [("seen by a cone", ~df.truly_blind.values & df.observable.values, "tab:blue"),
              ("between cones (seen by cone edge)", ~df.truly_blind.values & ~df.observable.values, "tab:green"),
              ("truly blind", df.truly_blind.values, "tab:red")]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = np.arange(len(bins))
    for nm, p, ls in [("baseline (min US)", base, "--"), ("RF-eng", oof, "-")]:
        for sn, m, c in strata:
            v = [np.abs(p - y)[m & (y >= a) & (y < b)].mean() if (m & (y >= a) & (y < b)).sum() >= 10 else np.nan
                 for a, b in bins]
            ax.plot(xs, v, ls, marker="o", color=c, label=f"{nm} / {sn}")
    ax.set_xticks(xs, ["0-0.5", "0.5-1", "1-2", "2-5"])
    ax.set_xlabel("true nearest distance bin (m)")
    ax.set_ylabel("MAE (m)")
    ax.set_title("Distance MAE by bin and stratum (pooled OOF)")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "F3_regression_mae_by_bin.png"), dpi=150)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
