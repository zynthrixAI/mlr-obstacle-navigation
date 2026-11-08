"""Task 6: camera crops (CNN) vs LiDAR RF vs late fusion."""
import os, time, random
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

torch.set_num_threads(4)
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
MAX_EPOCHS, PATIENCE, BATCH = 8, 2, 256
CACHE = os.path.join(DATA, "crops.npz")


def _render_group(args):
    from sim.world import make_world
    from sim.camera import render_crop
    ms, idxs, b0, b1, px, py, pt = args
    w = make_world(int(ms))
    out = np.empty((len(idxs), 32, 32, 3), np.uint8)
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
                     d.pose_x.values, d.pose_y.values, d.pose_theta.values))
    X = np.empty((len(seg), 32, 32, 3), np.uint8)
    with Pool(4) as p:
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


def main():
    t0 = time.time()
    seg = load_seg()
    X = get_crops(seg)
    y = seg.label.values; g = seg.map_seed.values
    fcols = [c for c in seg.columns if c.startswith("f_")]
    F = seg[fcols].values.astype(float)
    folds = outer_folds(g)
    names = ["Majority", "ColorHist-LogReg", "CNN", "LiDAR-RF", "Fusion"]
    oof = {n: np.full(len(y), -1) for n in names}
    oof_p = {n: np.zeros((len(y), 3)) for n in ("CNN", "LiDAR-RF", "Fusion")}
    ws, eps, cnn_t = [], [], 0.0
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
                                                     n_jobs=4, random_state=0))]).fit(F[itr], y[itr])
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
                         note="fixed-HP RF on 80% of train maps (differs from T2 tuned RF)" if n == "LiDAR-RF" else ""))
    res = pd.DataFrame(rows); res.to_csv(os.path.join(RES, "T6_vision.csv"), index=False)
    print(res.round(4).to_string(index=False))

    def onehot(p):
        a = np.zeros((N_SUB, 9)); a[np.arange(N_SUB), y * 3 + p] = 1; return a
    ms = per_map_stats(g, np.hstack([onehot(oof["CNN"]), onehot(oof["LiDAR-RF"]), onehot(oof["Fusion"])]))
    f1c = lambda v, i: macro_f1_from_cm(v[9 * i:9 * i + 9])
    v2, l2, h2, _ = bootstrap_maps(ms, lambda v: f1c(v, 0))
    v3, l3, h3, _ = bootstrap_maps(ms, lambda v: f1c(v, 2) - f1c(v, 1))
    v4, l4, h4, _ = bootstrap_maps(ms, lambda v: f1c(v, 0) - f1c(v, 1))
    acc = [dict(criterion="AC2: CNN pooled OOF macro-F1 >= 0.80", value=v2, ci_low=l2, ci_high=h2, **{"pass": bool(v2 >= 0.80)}),
           dict(criterion="AC3: Fusion minus LiDAR-RF macro-F1 >= +0.02 and bootstrap CI low > 0", value=v3, ci_low=l3, ci_high=h3,
                **{"pass": bool(v3 >= 0.02 and l3 > 0)}),
           dict(criterion="descriptive: CNN minus LiDAR-RF macro-F1", value=v4, ci_low=l4, ci_high=h4, **{"pass": "NA"}),
           dict(criterion="note: LiDAR-RF uses fixed hyper-parameters and 80% of training maps; differs from T2 tuned RF",
                value=np.nan, ci_low=np.nan, ci_high=np.nan, **{"pass": "NA"})]
    pd.DataFrame(acc).to_csv(os.path.join(RES, "T6_acceptance.csv"), index=False)
    print(pd.DataFrame(acc).round(4).to_string(index=False))

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, n in zip(axs, ["LiDAR-RF", "CNN", "Fusion"]):
        cm = confusion_matrix(y, oof[n], labels=[0, 1, 2], normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > .5 else "black")
        ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{n} (pooled OOF)")
    fig.colorbar(im, ax=axs, shrink=0.8); fig.savefig(os.path.join(RES, "F8_vision_confusion.png"), dpi=150, bbox_inches="tight")

    rng = np.random.RandomState(SEED)
    fig, axs = plt.subplots(3, 10, figsize=(15, 5))
    for r in range(3):
        idx = rng.choice(np.where(y == r)[0], 10, replace=False)
        for c, i in enumerate(idx):
            axs[r, c].imshow(X[i]); axs[r, c].axis("off")
            axs[r, c].set_title(CLASSES[r], fontsize=8)
    fig.suptitle("Example crops (rows: wall / obstacle / target)")
    fig.savefig(os.path.join(RES, "F9_crop_examples.png"), dpi=130, bbox_inches="tight")
    print(f"[T6] CNN epochs/fold={eps} total CNN time={cnn_t:.0f}s; total runtime={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
