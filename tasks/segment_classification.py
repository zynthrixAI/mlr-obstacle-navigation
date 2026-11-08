"""Task 1: LiDAR segment classification (wall / obstacle / target). PRIMARY model = RandomForest."""
import os, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from tasks.common import *

N_SUB = 60000
CLASSES = ["wall", "obstacle", "target"]


def models():
    return {
        "Dummy": (DummyClassifier(strategy="most_frequent"), {}),
        "LogReg": (LogisticRegression(class_weight="balanced", max_iter=2000),
                   {"model__C": [0.1, 1, 10]}),
        "KNN": (KNeighborsClassifier(), {"model__n_neighbors": [5, 15, 40]}),
        "RandomForest": (RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample",
                                                random_state=SEED, n_jobs=1),
                         {"model__min_samples_leaf": [1, 5, 15], "model__max_features": ["sqrt"]}),
        "XGBoost": (XGBClassifier(n_estimators=150, learning_rate=0.1, random_state=SEED, n_jobs=1,
                                  tree_method="hist", verbosity=0),
                    {"model__max_depth": [3, 6]}),
        "MLP": (MLPClassifier(max_iter=150, early_stopping=True, random_state=SEED),
                {"model__hidden_layer_sizes": [(64,), (64, 32)], "model__alpha": [1e-4, 1e-2]}),
    }


def main():
    t0 = time.time()
    seg = pd.read_csv(os.path.join(DATA, "segments.csv"))
    n_full = len(seg)
    seg = seg.sample(n=N_SUB, random_state=SEED).reset_index(drop=True)
    print(f"[T2] NOTE: uniform random subsample of {N_SUB} of {n_full} segments (seed {SEED}); "
          f"folds still GroupKFold(5) by map_seed ({seg.map_seed.nunique()} maps)")
    y = seg.label.values; g = seg.map_seed.values
    fsets = {"raw": [c for c in seg.columns if c.startswith("raw_")],
             "engineered": [c for c in seg.columns if c.startswith("f_")]}
    folds = outer_folds(g)
    oof, rows, fold_rows = {}, [], []
    for fs, cols in fsets.items():
        X = seg[cols].values
        for name, (est, grid) in models().items():
            t1 = time.time()
            pipe = make_pipe(est)
            pred, fid, best = oof_predict(pipe, grid, X, y, g, "f1_macro", folds)
            oof[(name, fs)] = pred
            f1s, accs = [], []
            for k, (tr, te) in enumerate(folds):
                f1 = f1_score(y[te], pred[te], average="macro"); ac = accuracy_score(y[te], pred[te])
                rec = recall_score(y[te], pred[te], average=None, labels=[0, 1, 2], zero_division=0)
                f1s.append(f1); accs.append(ac)
                fold_rows.append(dict(model=name, features=fs, fold=k, n_test=len(te), macro_f1=f1,
                                      accuracy=ac, **{f"recall_{c}": r for c, r in zip(CLASSES, rec)},
                                      best_params=str(best[k])))
            rec = recall_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0)
            rows.append(dict(model=name, features=fs, macro_f1_mean=np.mean(f1s), macro_f1_std=np.std(f1s, ddof=1),
                             accuracy_mean=np.mean(accs), accuracy_std=np.std(accs, ddof=1),
                             pooled_macro_f1=f1_score(y, pred, average="macro"),
                             **{f"recall_{c}": r for c, r in zip(CLASSES, rec)},
                             n_segments=N_SUB, primary=(name == "RandomForest")))
            print(f"[T2] {name:13s} {fs:10s} macroF1={rows[-1]['macro_f1_mean']:.4f}±{rows[-1]['macro_f1_std']:.4f} "
                  f"acc={rows[-1]['accuracy_mean']:.4f} ({time.time()-t1:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "T2_segment_classification.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(RES, "T2_segment_folds.csv"), index=False)
    print(res[["model", "features", "macro_f1_mean", "macro_f1_std", "accuracy_mean", "accuracy_std"]].round(4).to_string(index=False))

    # ---- acceptance (RF only) ----
    pa, pb = oof[("RandomForest", "engineered")], oof[("LogReg", "raw")]
    cm_a = np.zeros((N_SUB, 9)); cm_b = np.zeros((N_SUB, 9))
    cm_a[np.arange(N_SUB), y * 3 + pa.astype(int)] = 1
    cm_b[np.arange(N_SUB), y * 3 + pb.astype(int)] = 1
    ms = per_map_stats(g, np.hstack([cm_a, cm_b]))
    f_a = lambda v: macro_f1_from_cm(v[:9]); f_d = lambda v: macro_f1_from_cm(v[:9]) - macro_f1_from_cm(v[9:])
    v1, l1, h1, _ = bootstrap_maps(ms, f_a)
    v2, l2, h2, _ = bootstrap_maps(ms, f_d)
    p1, p2 = v1 >= 0.85, (v2 >= 0.05) and (l2 > 0)
    acc = [dict(criterion="AC1a: RF-engineered pooled OOF macro-F1 >= 0.85", value=v1, ci_low=l1, ci_high=h1, **{"pass": p1}),
           dict(criterion="AC1b: RF-eng minus LogReg-raw macro-F1 >= +0.05 and bootstrap CI low > 0", value=v2, ci_low=l2, ci_high=h2, **{"pass": p2}),
           dict(criterion="AC1 overall (a AND b)", value=np.nan, ci_low=np.nan, ci_high=np.nan, **{"pass": bool(p1 and p2)})]
    # ceiling audit
    fn = seg["f_n"].values
    acc.append(dict(criterion="audit: fraction of segments with f_n < 4", value=(fn < 4).mean(), ci_low=np.nan, ci_high=np.nan, **{"pass": "NA"}))
    for nm, mask in [("f_n < 6", fn < 6), ("f_n >= 6", fn >= 6)]:
        acc.append(dict(criterion=f"audit: RF-eng macro-F1 on segments with {nm} (n={mask.sum()})",
                        value=f1_score(y[mask], pa[mask], average="macro"), ci_low=np.nan, ci_high=np.nan, **{"pass": "NA"}))
    pd.DataFrame(acc).to_csv(os.path.join(RES, "T2_acceptance.csv"), index=False)
    print(pd.DataFrame(acc).round(4).to_string(index=False))

    cm = confusion_matrix(y, pa, labels=[0, 1, 2], normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > .5 else "black")
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("RF (engineered), pooled OOF, row-normalised")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(RES, "F2_segment_confusion.png"), dpi=150)
    print(f"[T2] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
