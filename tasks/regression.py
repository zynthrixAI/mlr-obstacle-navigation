"""Task 2: nearest-obstacle distance regression from 8 ultrasonics. PRIMARY model = RandomForest."""
import os, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from tasks.common import *

US = [f"us_{i}" for i in range(8)]
NEAR = 0.5
BINS = [(0, .5), (.5, 1), (1, 2), (2, 5.0001)]


def engineered(df):
    u = df[US].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = df[US].copy()
        out["us_min"], out["us_mean"], out["us_std"] = np.nanmin(u, 1), np.nanmean(u, 1), np.nanstd(u, 1)
        out["front_min"] = np.nanmin(u[:, [7, 0]], 1)   # bearings -45, 0
        out["side_min"] = np.nanmin(u[:, [2, 6]], 1)    # bearings +90, -90
        out["back_min"] = np.nanmin(u[:, [3, 5]], 1)    # bearings +135, -135
    out["n_dropout"] = np.isnan(u).sum(1)
    out["n_lt_1m"] = (u < 1.0).sum(1)  # NaN < 1 is False
    return out


def baseline(df):
    u = df[US].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = np.nanmin(u, 1)
    return np.where(np.isnan(b), 5.0, b)


def metrics(y, p):
    if len(y) == 0:
        return dict(n=0, MAE=np.nan, RMSE=np.nan, R2=np.nan, near_n=0, near_MAE=np.nan)
    e = p - y; near = y < NEAR
    ss = ((y - y.mean()) ** 2).sum()
    return dict(n=len(y), MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                R2=1 - (e ** 2).sum() / ss if ss > 0 else np.nan, near_n=int(near.sum()),
                near_MAE=np.abs(e[near]).mean() if near.any() else np.nan)


def models():
    return {
        "Ridge": (Ridge(), {"model__alpha": [0.1, 10, 100]}, ()),
        "Poly2+Ridge": (Ridge(), {"model__alpha": [1, 10, 100]}, (("poly", PolynomialFeatures(2, include_bias=False)),)),
        "KNN": (KNeighborsRegressor(), {"model__n_neighbors": [5, 15, 40]}, ()),
        "RandomForest": (RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=1),
                         {"model__min_samples_leaf": [1, 5, 15], "model__max_features": [1.0 / 3, 1.0]}, ()),
        "XGBoost": (XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=SEED, n_jobs=1,
                                 tree_method="hist", verbosity=0), {"model__max_depth": [3, 6]}, ()),
        "MLP": (MLPRegressor(max_iter=300, early_stopping=True, random_state=SEED),
                {"model__hidden_layer_sizes": [(64,), (64, 32)], "model__alpha": [1e-4, 1e-2]}, ()),
    }


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(DATA, "frames.csv"))
    y = df.d_nearest_true.values; g = df.map_seed.values; obs = df.observable.values.astype(bool)
    feats = {"raw": df[US], "engineered": engineered(df)}
    folds = outer_folds(g)
    preds = {("Baseline(min us)", "none"): baseline(df)}
    for fs, X in feats.items():
        for name, (est, grid, extra) in models().items():
            t1 = time.time()
            p, _, _ = oof_predict(make_pipe(est, extra), grid, X.values, y, g, "neg_mean_absolute_error", folds)
            preds[(name, fs)] = p
            print(f"[T3] {name:12s} {fs:10s} MAE={np.abs(p-y).mean():.4f} ({time.time()-t1:.0f}s)", flush=True)
    rows = []
    for (name, fs), p in preds.items():
        for sn, m in [("all", np.ones(len(y), bool)), ("observable", obs), ("blind_spot", ~obs)]:
            rows.append(dict(model=name, features=fs, stratum=sn, primary=(name == "RandomForest"), **metrics(y[m], p[m])))
    res = pd.DataFrame(rows); res.to_csv(os.path.join(RES, "T3_regression.csv"), index=False)
    print(res.round(4).to_string(index=False))

    # ---- AC5: observable rows only ----
    rf, base = preds[("RandomForest", "engineered")], preds[("Baseline(min us)", "none")]
    nm = obs & (y < NEAR)
    st = np.c_[np.abs(rf - y) * nm, np.abs(base - y) * nm, nm.astype(float)]
    ms = per_map_stats(g, st)
    mae_rf, lo1, hi1, _ = bootstrap_maps(ms, lambda v: v[0] / v[2])
    mae_b, lo0, hi0, _ = bootstrap_maps(ms, lambda v: v[1] / v[2])
    rel, lo, hi, _ = bootstrap_maps(ms, lambda v: 1 - v[0] / v[1])
    pa, pb = mae_rf <= 0.10, (rel >= 0.20) and (lo > 0)
    acc = [dict(criterion="AC5a: RF-eng near-range (<0.5m) MAE, observable rows, <= 0.10 m", value=mae_rf, ci_low=lo1, ci_high=hi1, **{"pass": bool(pa)}),
           dict(criterion="info: baseline near-range MAE, observable rows", value=mae_b, ci_low=lo0, ci_high=hi0, **{"pass": "NA"}),
           dict(criterion="AC5b: relative near-MAE reduction vs baseline >= 20% and CI low > 0", value=rel, ci_low=lo, ci_high=hi, **{"pass": bool(pb)}),
           dict(criterion="AC5 overall (a AND b)", value=np.nan, ci_low=np.nan, ci_high=np.nan, **{"pass": bool(pa and pb)})]
    pd.DataFrame(acc).to_csv(os.path.join(RES, "T3_acceptance.csv"), index=False)
    print(pd.DataFrame(acc).round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(BINS)); labs = [f"{a:g}-{min(b,5):g}" for a, b in BINS]
    for (nm_, p, ls) in [("baseline", base, "--"), ("RF-eng", rf, "-")]:
        for sn, m, c in [("observable", obs, "tab:blue"), ("between cones", ~obs, "tab:red")]:
            v = [np.abs(p - y)[m & (y >= a) & (y < b)].mean() if (m & (y >= a) & (y < b)).any() else np.nan for a, b in BINS]
            ax.plot(xs, v, ls, marker="o", color=c, label=f"{nm_} / {sn}")
    ax.set_xticks(xs, labs); ax.set_xlabel("true nearest distance bin (m)"); ax.set_ylabel("MAE (m)")
    ax.set_title("Distance MAE by bin (pooled OOF)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "F3_regression_mae_by_bin.png"), dpi=150)
    print(f"[T3] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
