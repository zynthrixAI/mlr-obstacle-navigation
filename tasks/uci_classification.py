"""UCI SCITOS G5 (4-sensor) classification with leakage-safe contiguous block CV."""
import os, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "uci", "sensor_readings_4.csv")
RES = os.path.join(ROOT, "results")
N_OUTER, GAP, EPS = 5, 50, 1e-3
SENS = ["front", "left", "right", "back"]


def load():
    df = pd.read_csv(DATA, header=None, names=SENS + ["label"])
    return df


def build_features(df):
    raw = df[SENS].copy()
    eng = raw.copy()
    s = raw.values
    eng["min"], eng["max"] = s.min(1), s.max(1)
    eng["mean"], eng["std"] = s.mean(1), s.std(1)
    eng["front_left"] = raw.front - raw.left
    eng["front_right_ratio"] = raw.front / (raw.right + EPS)
    eng["left_right"] = raw.left - raw.right
    # lag-1 deltas, previous row only (causal); first row -> 0
    for c in SENS:
        eng[f"d_{c}"] = raw[c].diff().fillna(0.0)
    return {"raw": raw, "engineered": eng}


def models():
    """name -> (pipeline, grid, needs_sample_weight)"""
    sc = lambda: StandardScaler()
    return {
        "Majority": (Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]), {}, False),
        "LogReg": (Pipeline([("sc", sc()), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))]),
                   {"clf__C": [0.1, 1, 10]}, False),
        "KNN": (Pipeline([("sc", sc()), ("clf", KNeighborsClassifier())]),
                {"clf__n_neighbors": [5, 15, 31], "clf__weights": ["uniform", "distance"]}, False),
        "RandomForest": (Pipeline([("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=4, random_state=SEED))]),
                         {"clf__max_depth": [None, 8], "clf__min_samples_leaf": [1, 5, 10]}, False),
        "XGBoost": (Pipeline([("clf", XGBClassifier(n_estimators=150, learning_rate=0.1, n_jobs=4, random_state=SEED, verbosity=0))]),
                    {"clf__max_depth": [3, 6], "clf__min_child_weight": [1, 5]}, True),
        "MLP": (Pipeline([("sc", sc()), ("clf", MLPClassifier(max_iter=300, early_stopping=True, random_state=SEED))]),
                {"clf__hidden_layer_sizes": [(32,), (64, 32)], "clf__alpha": [1e-4, 1e-2]}, False),
    }


def main():
    t0 = time.time()
    df = load()
    n = len(df)
    le = LabelEncoder().fit(df.label)
    y = le.transform(df.label)
    feats = build_features(df)
    edges = np.linspace(0, n, N_OUTER + 1).astype(int)
    rows, oof_rf = [], np.full(n, -1)
    inner = KFold(3, shuffle=False)
    for fi in range(N_OUTER):
        lo, hi = edges[fi], edges[fi + 1]
        te = np.arange(lo, hi)
        tr = np.r_[0:max(0, lo - GAP), min(n, hi + GAP):n].astype(int)
        for fname, X in feats.items():
            Xv = X.values
            for mname, (pipe, grid, sw) in models().items():
                yt = y[tr]
                if grid:
                    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner, n_jobs=1 if mname in ("RandomForest", "XGBoost") else 4,
                                      error_score=np.nan, refit=True)
                    kw = {"clf__sample_weight": compute_sample_weight("balanced", yt)} if sw else {}
                    gs.fit(Xv[tr], yt, **kw)
                    est = gs.best_estimator_
                else:
                    est = pipe.fit(Xv[tr], yt)
                pred = est.predict(Xv[te])
                f1 = f1_score(y[te], pred, average="macro")
                acc = accuracy_score(y[te], pred)
                rows.append(dict(model=mname, features=fname, fold=fi, macro_f1=f1, accuracy=acc))
                if mname == "RandomForest" and fname == "engineered":
                    oof_rf[te] = pred
                print(f"fold{fi} {fname:10s} {mname:12s} f1={f1:.3f} acc={acc:.3f} [{time.time()-t0:.0f}s]", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "uci_classification.csv"), index=False)
    order = ["Majority", "LogReg", "KNN", "RandomForest", "XGBoost", "MLP"]
    summ = res.groupby(["model", "features"]).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std")).reset_index()
    summ["model"] = pd.Categorical(summ["model"], order)
    summ = summ.sort_values(["model", "features"]).reset_index(drop=True)
    summ.to_csv(os.path.join(RES, "uci_summary.csv"), index=False)

    # confusion matrix (pooled OOF, RF engineered)
    cm = confusion_matrix(y, oof_rf, normalize="true")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title("RandomForest (engineered), pooled out-of-fold, row-normalised")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "uci_confusion_rf.png"), dpi=150); plt.close(fig)

    # bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.38
    for i, (fname, col) in enumerate([("raw", "#7a9cc6"), ("engineered", "#d9822b")]):
        s = summ[summ.features == fname]
        ax.bar(np.arange(len(order)) + (i - 0.5) * w, s.macro_f1_mean, w, yerr=s.macro_f1_std, capsize=3, label=fname, color=col)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_ylabel("macro-F1 (mean ± std over 5 folds)"); ax.legend(); ax.set_title("UCI sensor_readings_4: blocked CV")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "uci_f1_bar.png"), dpi=150); plt.close(fig)
    print(summ.to_string())
    print(f"Total runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
