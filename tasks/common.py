"""Shared protocol helpers: group-by-map outer CV, leakage-safe inner tuning, map bootstrap."""
import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
N_JOBS = 4
MAX_CONFIGS = 6


def outer_folds(groups, n_splits=5):
    """GroupKFold(5), groups = map_seed: every map entirely in one fold."""
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def make_pipe(model, extra_steps=()):
    """All preprocessing lives in the pipeline -> fit on training folds only."""
    steps = [("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("sc", StandardScaler())]
    steps += list(extra_steps)
    steps.append(("model", model))
    return Pipeline(steps)


def oof_predict(pipe, grid, X, y, groups, scoring, folds=None, fit_params_fn=None):
    """Out-of-fold predictions. Inner tuning: GridSearchCV(GroupKFold(3)) on the training part
    with groups= passed to fit. Returns (oof_pred, fold_id, best_params_list)."""
    X = np.asarray(X, float); y = np.asarray(y); groups = np.asarray(groups)
    folds = folds or outer_folds(groups)
    n_cfg = int(np.prod([len(v) for v in grid.values()])) if grid else 1
    assert n_cfg <= MAX_CONFIGS, f"{n_cfg} configs > {MAX_CONFIGS}"
    oof = np.full(len(y), np.nan); fid = np.full(len(y), -1); best = []
    for k, (tr, te) in enumerate(folds):
        assert not set(groups[tr]) & set(groups[te])
        if grid:
            gs = GridSearchCV(clone(pipe), grid, cv=GroupKFold(3), scoring=scoring,
                              n_jobs=N_JOBS, refit=True)
            gs.fit(X[tr], y[tr], groups=groups[tr])
            est, bp = gs.best_estimator_, gs.best_params_
        else:
            est = clone(pipe).fit(X[tr], y[tr]); bp = {}
        oof[te] = est.predict(X[te]); fid[te] = k; best.append(bp)
    assert (fid >= 0).all()
    return oof, fid, best


def per_map_stats(groups, stat_matrix):
    """Sum an (n_rows x d) matrix of additive per-row statistics within each map."""
    g = pd.factorize(np.asarray(groups))[0]
    out = np.zeros((g.max() + 1, stat_matrix.shape[1]))
    np.add.at(out, g, stat_matrix)
    return out


def bootstrap_maps(map_stats, stat_fn, n_boot=1000, seed=SEED):
    """Resample maps with replacement; per resample sum additive per-map stats and apply
    stat_fn(sum_vector) -> scalar (e.g. pooled-metric difference between two models).
    Returns (point_estimate, ci_low, ci_high, boot_array)."""
    rng = np.random.RandomState(seed)
    m = len(map_stats)
    point = stat_fn(map_stats.sum(0))
    vals = np.array([stat_fn(map_stats[rng.randint(0, m, m)].sum(0)) for _ in range(n_boot)])
    return point, *np.nanpercentile(vals, [2.5, 97.5]), vals


def macro_f1_from_cm(cm):
    cm = cm.reshape(int(np.sqrt(cm.size)), -1)
    tp = np.diag(cm); p = cm.sum(0); r = cm.sum(1)
    f1 = np.where(p + r > 0, 2 * tp / np.maximum(p + r, 1e-12), 0.0)
    return f1.mean()
