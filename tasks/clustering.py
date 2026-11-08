"""Task 3: unsupervised clustering of 10-frame LiDAR windows vs hidden map_type.
NOTE: the three map generators are distinct by design, so recovering map_type is a sanity check
on separability, not evidence of a hard problem."""
import os, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from tasks.common import *

WIN = 10
SECT = {"front": list(range(0, 4)) + [33, 34, 35], "left": list(range(3, 10)),
        "right": list(range(27, 34)), "back": list(range(15, 22))}


def windows(df):
    L = df[[f"lidar_{i:02d}" for i in range(36)]].fillna(5.0).values
    df = df.assign(_i=np.arange(len(df))).sort_values(["episode_id", "step"])
    F, meta = [], []
    for ep, d in df.groupby("episode_id", sort=False):
        idx = d._i.values
        for s in range(0, len(idx) - WIN + 1, WIN):
            w = L[idx[s:s + WIN]]
            fmin = w.min(1)
            jumps = (np.abs(w - np.roll(w, -1, axis=1)) > 0.5).sum(1)
            feat = list(w.mean(0)) + [fmin.mean(), fmin.std(), (w >= 4.9).mean()]
            feat += [w[:, ix].min(1).mean() for ix in SECT.values()] + [jumps.mean()]
            F.append(feat); meta.append((d.map_seed.iloc[0], d.map_type.iloc[0], ep))
    names = [f"mean_r{i:02d}" for i in range(36)] + ["min_mean", "min_std", "free_frac"] + \
            [f"sect_{k}" for k in SECT] + ["jumps"]
    return pd.DataFrame(F, columns=names), pd.DataFrame(meta, columns=["map_seed", "map_type", "episode_id"])


def main():
    t0 = time.time()
    print("[T4] NOTE: map generators (open/narrow/cluttered) are distinct by design; ARI measures separability only.")
    df = pd.read_csv(os.path.join(DATA, "frames.csv"))
    W, M = windows(df)
    print(f"[T4] {len(W)} windows of {WIN} frames, {W.shape[1]} features, {M.map_seed.nunique()} maps")
    Z = StandardScaler().fit_transform(W.values)   # unsupervised, label-free standardisation
    types = M.map_type.values; g = M.map_seed.values
    rows, labs = [], {}
    for algo in ["kmeans", "gmm"]:
        for k in range(2, 7):
            if algo == "kmeans":
                m = KMeans(k, n_init=10, random_state=SEED).fit(Z); lab = m.labels_; bic = np.nan
            else:
                m = GaussianMixture(k, covariance_type="full", n_init=3, random_state=SEED, reg_covar=1e-4).fit(Z)
                lab = m.predict(Z); bic = m.bic(Z)
            labs[(algo, k)] = lab
            rows.append(dict(algorithm=algo, k=k, silhouette=silhouette_score(Z, lab) if len(set(lab)) > 1 else np.nan,
                             db=davies_bouldin_score(Z, lab) if len(set(lab)) > 1 else np.nan,
                             ari=adjusted_rand_score(types, lab), nmi=normalized_mutual_info_score(types, lab), bic=bic))
    res = pd.DataFrame(rows)
    ks = res[res.algorithm == "kmeans"].set_index("k").silhouette.idxmax()
    kg = res[res.algorithm == "gmm"].set_index("k").bic.idxmin()
    res["selected"] = ""
    res.loc[(res.algorithm == "kmeans") & (res.k == ks), "selected"] = "silhouette-best"
    res.loc[(res.algorithm == "gmm") & (res.k == kg), "selected"] = "BIC-best"
    res.loc[res.k == 3, "selected"] += " k=3 (reported)"
    res.to_csv(os.path.join(RES, "T4_clustering.csv"), index=False)
    print(res.round(4).to_string(index=False)); print(f"[T4] selected: KMeans k={ks} (silhouette), GMM k={kg} (BIC)")

    # map-level permutation null (k=3 K-Means) + map bootstrap of ARI
    lab3 = labs[("kmeans", 3)]
    obs_ari = adjusted_rand_score(types, lab3)
    maps = pd.DataFrame({"g": g, "t": types}).drop_duplicates("g").reset_index(drop=True)
    gi = pd.Series(range(len(maps)), index=maps.g)
    widx = gi.loc[g].values
    rng = np.random.RandomState(SEED)
    null = np.empty(500)
    for i in range(500):
        perm = rng.permutation(maps.t.values)   # permute labels across maps; windows inherit their map's label
        null[i] = adjusted_rand_score(perm[widx], lab3)
    p = (1 + (null >= obs_ari).sum()) / (500 + 1)
    # bootstrap over maps (cluster assignment fixed; resample maps' windows)
    by_map = [np.where(widx == j)[0] for j in range(len(maps))]
    boots = []
    for _ in range(500):
        sel = rng.randint(0, len(maps), len(maps))
        ix = np.concatenate([by_map[j] for j in sel])
        boots.append(adjusted_rand_score(types[ix], lab3[ix]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    perm_df = pd.DataFrame([dict(observed_ari=obs_ari, null_mean=null.mean(), null_p99=np.percentile(null, 99),
                                 null_max=null.max(), p_value=p, n_perm=500, boot_ci_low=lo, boot_ci_high=hi, n_boot=500)])
    perm_df.to_csv(os.path.join(RES, "T4_permutation.csv"), index=False)
    print(perm_df.round(4).to_string(index=False))

    # figures
    P = PCA(2, random_state=SEED).fit_transform(Z)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(P[:, 0], P[:, 1], c=lab3, cmap="tab10", s=6, vmin=0, vmax=9); ax[0].set_title("K-Means k=3 clusters")
    for i, t in enumerate(sorted(set(types))):
        m = types == t; ax[1].scatter(P[m, 0], P[m, 1], s=6, color=f"C{i}", label=t)
    ax[1].legend(markerscale=3); ax[1].set_title("true map_type (hidden label)")
    for a in ax: a.set_xlabel("PC1"); a.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "F4_cluster_pca.png"), dpi=150)

    cols = ["sect_front", "sect_left", "sect_right", "sect_back", "free_frac"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [4, 1.3]})
    wd = 0.25
    for c in range(3):
        m = lab3 == c
        maj = pd.Series(types[m]).value_counts(); pur = maj.iloc[0] / m.sum()
        lab = f"cluster {c} (n={m.sum()}, majority {maj.index[0]}, {pur:.0%})"
        ax[0].bar(np.arange(4) + c * wd, W.loc[m, cols[:4]].mean().values, wd, label=lab)
        ax[1].bar(c, W.loc[m, "free_frac"].mean(), color=f"C{c}")
    ax[0].set_xticks(np.arange(4) + wd, ["front", "left", "right", "back"]); ax[0].set_ylabel("mean sector min range (m)")
    ax[0].legend(fontsize=7); ax[0].set_title("K-Means k=3 centroid profiles")
    ax[1].set_xticks(range(3), [f"c{c}" for c in range(3)]); ax[1].set_title("free-ray fraction")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "F5_cluster_profiles.png"), dpi=150)
    print(f"[T4] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
