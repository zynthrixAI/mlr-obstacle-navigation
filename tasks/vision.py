"""Task 6: camera crops (CNN) vs LiDAR RF vs late fusion.

`python -m tasks.vision` (default, --appearance easy) -> T6/F8/F9, cache data/crops.npz.
`python -m tasks.vision --appearance hard` -> T6b_vision_hard.csv, T6b_acceptance_hard.csv,
F12_vision_hard_confusion.png, F13_crops_hard.png, cache data/crops_hard.npz. Hard = pre-specified ambiguous
appearance (sim.world.hard_appearance) + partial occlusion (sim.camera.render_crop occ_rng); same 60k segments,
outer folds, models, fusion protocol and AC definitions (descriptive only). Env T6_THREADS sets torch/RF/Pool
threads (default 4)."""
import os, sys, time, random, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from multiprocessing import Pool
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tasks.common import *
from tasks.segment_classification import N_SUB, CLASSES

N_THREADS = int(os.environ.get("T6_THREADS", 4))
torch.set_num_threads(N_THREADS)
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
MAX_EPOCHS, PATIENCE, BATCH = 8, 3, 256
CACHE = os.path.join(DATA, "crops.npz")
MODE = "easy"
OUT = {"easy": dict(cache="crops.npz", table="T6_vision.csv", acc="T6_acceptance.csv", conf="F8_vision_confusion.png",
                    crops="F9_crop_examples.png", tag=""),
       "hard": dict(cache="crops_hard.npz", table="T6b_vision_hard.csv", acc="T6b_acceptance_hard.csv",
                    conf="F12_vision_hard_confusion.png", crops="F13_crops_hard.png", tag="[hard appearance] ")}
OCC_SEED = 0x0CC1  # hard-mode occlusion rng = default_rng([row, OCC_SEED]); separate from the noise rng (row)


def _render_group(args):
    from sim.world import make_world
    from sim.camera import render_crop
    ms, idxs, b0, b1, px, py, pt = args[:7]
    mode = args[7] if len(args) > 7 else "easy"
    w = make_world(int(ms))
    out = np.empty((len(idxs), 32, 32, 3), np.uint8)
    if mode == "hard":
        from sim.world import hard_appearance
        app = hard_appearance(int(ms))
        for k, i in enumerate(idxs):
            out[k] = render_crop(w, (px[k], py[k], pt[k]), b0[k], b1[k], np.random.default_rng(int(i)),
                                 appearance=app, occ_rng=np.random.default_rng([int(i), OCC_SEED]))
        return idxs, out
    for k, i in enumerate(idxs):
        out[k] = render_crop(w, (px[k], py[k], pt[k]), b0[k], b1[k], np.random.default_rng(int(i)))
    return idxs, out


def load_seg():
    seg = pd.read_csv(os.path.join(DATA, "segments.csv"))
    return seg.sample(n=N_SUB, random_state=SEED).reset_index(drop=True)


def get_crops(seg):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if z["X"].shape == (len(seg), 32, 32, 3) and np.array_equal(z["map_seed"], seg.map_seed.values) \
                and np.array_equal(z["y"], seg.label.values):
            print("[T6] crops loaded from cache"); return z["X"]
    t = time.time()
    jobs = []
    for ms, d in seg.groupby("map_seed"):
        jobs.append((ms, d.index.values, d.bearing_start_deg.values, d.bearing_end_deg.values,
                     d.pose_x.values, d.pose_y.values, d.pose_theta.values) + ((MODE,) if MODE != "easy" else ()))
    X = np.empty((len(seg), 32, 32, 3), np.uint8)
    with Pool(N_THREADS) as p:
        for idxs, out in p.imap_unordered(_render_group, jobs):
            X[idxs] = out
    np.savez_compressed(CACHE, X=X, y=seg.label.values, map_seed=seg.map_seed.values, row=np.arange(len(seg)))
    print(f"[T6] rendered {len(seg)} crops in {time.time()-t:.0f}s")
    return X


def hist_feats(X):
    c = X[:, :, 8:24, :]  # central 50% columns
    f = []
    for ch in range(3):
        v = c[..., ch].reshape(len(X), -1)
        f.append(np.stack([np.bincount(r // 32, minlength=8) for r in v]) / v.shape[1])
    Xf = X.astype(np.float32)
    f.append(Xf.mean((1, 2))); f.append(Xf.std((1, 2)))
    return np.hstack(f)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        L, c = [], 3
        for o in (16, 32, 64):
            L += [nn.Conv2d(c, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]; c = o
        self.f = nn.Sequential(*L); self.fc = nn.Linear(64, 3)

    def forward(self, x):
        return self.fc(self.f(x).mean((2, 3)))


def predict_proba(m, X):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            out.append(torch.softmax(m(X[i:i + 1024]), 1).numpy())
    return np.vstack(out)


def train_cnn(Xtr, ytr, Xva, yva, Xte, fold):
    mean = Xtr.reshape(-1, 3).mean(0) / 255; std = Xtr.reshape(-1, 3).std(0) / 255
    prep = lambda a: torch.tensor(((a / 255.0 - mean) / std).transpose(0, 3, 1, 2), dtype=torch.float32)
    # brightness is applied in raw [0,1] space then normalised
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1); std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    Ttr = torch.tensor((Xtr / 255.0).transpose(0, 3, 1, 2), dtype=torch.float32)
    Tva, Tte = prep(Xva), prep(Xte)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    cnt = np.bincount(ytr, minlength=3)
    wts = torch.tensor(len(ytr) / (3 * cnt), dtype=torch.float32)
    torch.manual_seed(SEED + fold)
    m = CNN(); opt = torch.optim.Adam(m.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss(weight=wts)
    best, best_state, bad, ep_used, times = -1, None, 0, 0, []
    g = torch.Generator().manual_seed(SEED + fold)
    for ep in range(MAX_EPOCHS):
        t = time.time(); m.train()
        perm = torch.randperm(len(Ttr), generator=g)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]; xb = Ttr[idx]
            flip = torch.rand(len(idx), generator=g) < 0.5
            xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(3), xb)
            xb = xb * (0.8 + 0.4 * torch.rand(len(idx), 1, 1, 1, generator=g))
            xb = (xb - mean_t) / std_t
            opt.zero_grad(); lossf(m(xb), ytr_t[idx]).backward(); opt.step()
        f1 = f1_score(yva, predict_proba(m, Tva).argmax(1), average="macro")
        times.append(time.time() - t)
        print(f"[T6]   fold {fold} epoch {ep+1} val-F1={f1:.4f} ({times[-1]:.0f}s)", flush=True)
        if f1 > best:
            best, bad, ep_used = f1, 0, ep + 1
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    m.load_state_dict(best_state)
    return predict_proba(m, Tva), predict_proba(m, Tte), ep_used, len(times), sum(times)


def colour_rule(X):
    """Hand-written 3-rule colour heuristic (no fitting), per methodology review: central 16x16 region.
    wall if mean saturation < 0.25; target if a bright low-saturation stripe row or red-orange saturated hue;
    else obstacle."""
    import colorsys
    out = np.empty(len(X), int)
    for i, im in enumerate(X):
        c = im[8:24, 8:24].astype(float) / 255
        hsv = np.array([colorsys.rgb_to_hsv(*px) for px in c.reshape(-1, 3)]).reshape(16, 16, 3)
        s_mean = hsv[..., 1].mean()
        rows_v, rows_s = hsv[..., 2].mean(1), hsv[..., 1].mean(1)
        stripe = bool(((rows_v > 0.7) & (rows_s < 0.25)).any())
        hue = hsv[..., 0][hsv[..., 1] > 0.4]
        red = hue.size > 20 and np.mean((hue < 0.10) | (hue > 0.95)) > 0.6 and s_mean > 0.45
        out[i] = 0 if s_mean < 0.25 else (2 if (stripe or red) else 1)
    return out


def main():
    global MODE, CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--appearance", choices=["easy", "hard"], default=os.environ.get("T6_APPEARANCE", "easy"))
    MODE = ap.parse_args().appearance
    O = OUT[MODE]; TAG = O["tag"]
    CACHE = os.path.join(DATA, O["cache"])
    print(f"[T6] appearance={MODE} threads={N_THREADS}", flush=True)
    t0 = time.time()
    seg = load_seg()
    X = get_crops(seg)
    y = seg.label.values; g = seg.map_seed.values
    fcols = [c for c in seg.columns if c.startswith("f_")]
    F = seg[fcols].values.astype(float)
    folds = outer_folds(g)
    names = ["Majority", "ColourRule (hand, 3 rules)", "ColorHist-LogReg", "CNN", "LiDAR-RF", "Fusion"]
    oof = {n: np.full(len(y), -1) for n in names}
    oof_p = {n: np.zeros((len(y), 3)) for n in ("CNN", "LiDAR-RF", "Fusion")}
    ws, eps, cnn_t = [], [], 0.0
    oof["ColourRule (hand, 3 rules)"][:] = colour_rule(X)          # no fitting, so no fold dependence
    for k, (tr, te) in enumerate(folds):
        assert not set(g[tr]) & set(g[te])
        itr_, iva_ = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(tr, groups=g[tr]))
        itr, iva = tr[itr_], tr[iva_]
        assert not set(g[itr]) & set(g[iva])
        oof["Majority"][te] = DummyClassifier(strategy="most_frequent").fit(F[itr], y[itr]).predict(F[te])
        hp = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))])
        oof["ColorHist-LogReg"][te] = hp.fit(hist_feats(X[itr]), y[itr]).predict(hist_feats(X[te]))
        rf = Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                       ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                     min_samples_leaf=5, n_jobs=N_THREADS, random_state=0))]).fit(F[itr], y[itr])
        prf_va, prf_te = rf.predict_proba(F[iva]), rf.predict_proba(F[te])
        pc_va, pc_te, ep_used, ep_run, tt = train_cnn(X[itr], y[itr], X[iva], y[iva], X[te], k)
        cnn_t += tt
        grid = np.round(np.arange(0, 1.01, 0.1), 1)
        sc = [f1_score(y[iva], (w * pc_va + (1 - w) * prf_va).argmax(1), average="macro") for w in grid]
        w = float(grid[int(np.argmax(sc))])
        pf_te = w * pc_te + (1 - w) * prf_te
        for n, p in (("CNN", pc_te), ("LiDAR-RF", prf_te), ("Fusion", pf_te)):
            oof_p[n][te] = p; oof[n][te] = p.argmax(1)
        ws.append(w); eps.append(ep_used)
        print(f"[T6] fold {k}: w={w} cnn_best_epoch={ep_used}/{ep_run} cnn_time={tt:.0f}s "
              f"F1 hist={f1_score(y[te], oof['ColorHist-LogReg'][te], average='macro'):.3f} "
              f"cnn={f1_score(y[te], oof['CNN'][te], average='macro'):.3f} "
              f"rf={f1_score(y[te], oof['LiDAR-RF'][te], average='macro'):.3f} "
              f"fusion={f1_score(y[te], oof['Fusion'][te], average='macro'):.3f}", flush=True)

    rows = []
    for n in names:
        p = oof[n]; assert (p >= 0).all()
        f1s = [f1_score(y[te], p[te], average="macro") for _, te in folds]
        rec = recall_score(y, p, average=None, labels=[0, 1, 2], zero_division=0)
        rows.append(dict(model=n, macro_f1_pooled=f1_score(y, p, average="macro"), macro_f1_mean=np.mean(f1s),
                         macro_f1_std=np.std(f1s, ddof=1), accuracy=accuracy_score(y, p),
                         **{f"recall_{c}": r for c, r in zip(CLASSES, rec)}, n=N_SUB,
                         fusion_w_per_fold=";".join(map(str, ws)) if n == "Fusion" else "",
                         cnn_epochs_per_fold=";".join(map(str, eps)) if n == "CNN" else "",
                         note=("RF with T2-selected min_samples_leaf=5, fit on inner-train maps (80% of each outer-train fold)" if n == "LiDAR-RF" else "no fitting; hand rule on central pixels" if n.startswith("ColourRule") else "")))
    res = pd.DataFrame(rows); res.to_csv(os.path.join(RES, O["table"]), index=False)
    print(res.round(4).to_string(index=False))

    def onehot(p):
        a = np.zeros((N_SUB, 9)); a[np.arange(N_SUB), y * 3 + p] = 1; return a
    ms = per_map_stats(g, np.hstack([onehot(oof["CNN"]), onehot(oof["LiDAR-RF"]), onehot(oof["Fusion"])]))
    f1c = lambda v, i: macro_f1_from_cm(v[9 * i:9 * i + 9])
    v2, l2, h2, _ = bootstrap_maps(ms, lambda v: f1c(v, 0))
    v3, l3, h3, _ = bootstrap_maps(ms, lambda v: f1c(v, 2) - f1c(v, 1))
    v4, l4, h4, _ = bootstrap_maps(ms, lambda v: f1c(v, 0) - f1c(v, 1))
    acc = [dict(criterion=TAG + "AC2: CNN pooled OOF macro-F1 >= 0.80", value=v2, ci_low=l2, ci_high=h2, **{"pass": bool(v2 >= 0.80)}),
           dict(criterion=TAG + "AC3: Fusion minus LiDAR-RF macro-F1 >= +0.02 and bootstrap CI low > 0", value=v3, ci_low=l3, ci_high=h3,
                **{"pass": bool(v3 >= 0.02 and l3 > 0)}),
           dict(criterion=TAG + "descriptive: CNN minus LiDAR-RF macro-F1", value=v4, ci_low=l4, ci_high=h4, **{"pass": "NA"})]
    if MODE == "easy":
        acc.append(dict(criterion="note: LiDAR-RF uses T2-selected min_samples_leaf=5; appearance model is authored (upper bound, see reviews/vision.md)",
                        value=np.nan, ci_low=np.nan, ci_high=np.nan, **{"pass": "NA"}))
    else:
        v5, l5, h5, _ = bootstrap_maps(ms, lambda v: f1c(v, 2) - f1c(v, 0))
        acc += [dict(criterion=TAG + "descriptive: Fusion minus CNN macro-F1", value=v5, ci_low=l5, ci_high=h5, **{"pass": "NA"}),
                dict(criterion=TAG + "note: descriptive only (AC thresholds were set for the easy variant); pre-specified hard "
                     "appearance in sim.world.hard_appearance + occlusion in sim.camera.render_crop; LiDAR-RF min_samples_leaf=5; "
                     "pooled OOF, 1000 map-bootstrap 95% CI", value=np.nan, ci_low=np.nan, ci_high=np.nan, **{"pass": "NA"})]
    pd.DataFrame(acc).to_csv(os.path.join(RES, O["acc"]), index=False)
    print(pd.DataFrame(acc).round(4).to_string(index=False))

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, n in zip(axs, ["LiDAR-RF", "CNN", "Fusion"]):
        cm = confusion_matrix(y, oof[n], labels=[0, 1, 2], normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > .5 else "black")
        ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{TAG}{n} (pooled OOF)")
    fig.colorbar(im, ax=axs, shrink=0.8); fig.savefig(os.path.join(RES, O["conf"]), dpi=150, bbox_inches="tight")

    rng = np.random.RandomState(SEED)
    fig, axs = plt.subplots(3, 10, figsize=(15, 5))
    for r in range(3):
        idx = rng.choice(np.where(y == r)[0], 10, replace=False)
        for c, i in enumerate(idx):
            axs[r, c].imshow(X[i]); axs[r, c].axis("off")
            axs[r, c].set_title(CLASSES[r], fontsize=8)
    fig.suptitle(TAG + "Example crops (rows: wall / obstacle / target)")
    fig.savefig(os.path.join(RES, O["crops"]), dpi=130, bbox_inches="tight")
    print(f"[T6] CNN epochs/fold={eps} total CNN time={cnn_t:.0f}s; total runtime={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
