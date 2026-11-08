"""Task 2c: does causal temporal tracking of LiDAR segments improve segment classification?

Tracks are built on the FULL segments.csv (per episode, scans ordered by step): each segment is linked
to the nearest segment of the immediately previous scan (step - SEG_EVERY) of the same episode whose
world-frame centroid lies within LINK_DIST, greedy by distance, one-to-one. Labels are never used.
Temporal features use only the segment itself and earlier scans of its own track (last HIST scans).
Then the T2 subsample / folds / RF are reused to compare engineered vs engineered+temporal."""
import os, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score
from tasks.common import *
from tasks.segment_classification import N_SUB, CLASSES, models
from sim.generate import SEG_EVERY

LINK_DIST = 0.3
HIST = 5
AGG_MEANMAX = ["f_n", "f_width_deg", "f_chord", "f_circ_r", "f_lin_rms", "f_circ_rms", "f_bulge"]
AGG_STD = ["f_chord", "f_circ_r"]


def centroids(seg):
    s, e = seg.bearing_start_deg.values, seg.bearing_end_deg.values
    mid = np.deg2rad(s + ((e - s) % 360) / 2.0)
    ang = seg.pose_theta.values + mid
    r = seg.f_mean_r.values
    return seg.pose_x.values + r * np.cos(ang), seg.pose_y.values + r * np.sin(ang)


def link_tracks(seg):
    """Return prev[i] (row index of predecessor in previous scan, -1 if none). Uses only geometry."""
    cx, cy = centroids(seg)
    prev = np.full(len(seg), -1, dtype=np.int64)
    for _, ep in seg.groupby("episode_id", sort=False).groups.items():
        ep = np.asarray(ep)
        steps = seg.step.values[ep]
        by_step = {st: ep[steps == st] for st in np.unique(steps)}
        for st, cur in by_step.items():
            old = by_step.get(st - SEG_EVERY)          # immediately previous scan only
            if old is None or len(old) == 0:
                continue
            d = np.hypot(cx[cur][:, None] - cx[old][None, :], cy[cur][:, None] - cy[old][None, :])
            ii, jj = np.where(d <= LINK_DIST)
            order = np.argsort(d[ii, jj], kind="stable")
            used_c, used_o = set(), set()
            for k in order:
                a, b = ii[k], jj[k]
                if a in used_c or b in used_o:
                    continue
                used_c.add(a); used_o.add(b)
                prev[cur[a]] = old[b]
    return prev, cx, cy


def temporal_features(seg):
    prev, cx, cy = link_tracks(seg)
    n = len(seg)
    # causal: earlier scans have smaller step; prev always points to step - SEG_EVERY
    ok = prev >= 0
    assert (seg.step.values[prev[ok]] == seg.step.values[ok] - SEG_EVERY).all()
    assert (seg.episode_id.values[prev[ok]] == seg.episode_id.values[ok]).all()
    # full (uncapped) track length up to and including self, processed in step order
    tl = np.ones(n, dtype=np.int64)
    for i in np.argsort(seg.step.values, kind="stable"):
        if prev[i] >= 0:
            tl[i] = tl[prev[i]] + 1
    # history index matrix (self, prev, prev-prev, ...), capped at HIST
    H = np.full((n, HIST), -1, dtype=np.int64); H[:, 0] = np.arange(n)
    for k in range(1, HIST):
        p = H[:, k - 1]
        H[:, k] = np.where(p >= 0, prev[np.maximum(p, 0)], -1)
    M = H >= 0
    Hc = np.maximum(H, 0)
    out = {"track_len": tl, "t_hist_n": M.sum(1)}

    def gather(col):
        v = seg[col].values.astype(float)[Hc]; v[~M] = np.nan; return v
    with np.errstate(all="ignore"):
        for c in AGG_MEANMAX:
            v = gather(c)
            out[f"t_mean_{c[2:]}"] = np.nanmean(v, 1)
            out[f"t_max_{c[2:]}"] = np.nanmax(v, 1)
        for c in AGG_STD:
            out[f"t_std_{c[2:]}"] = np.nanstd(gather(c), 1)
        out["t_mean_mean_r"] = np.nanmean(gather("f_mean_r"), 1)
        # viewing angle: bearing from object centroid to robot (world frame), spread over history
        va = np.arctan2(seg.pose_y.values - cy, seg.pose_x.values - cx)
        v = va[Hc]
        rel = np.degrees(np.angle(np.exp(1j * (v - va[:, None]))))  # relative to current view, [-180,180]
        rel[~M] = np.nan
        out["t_view_spread_deg"] = np.nanmax(rel, 1) - np.nanmin(rel, 1)
    return pd.DataFrame(out, index=seg.index), prev


def evaluate(name, fs, pred, y, folds):
    f1s, accs = [], []
    for tr, te in folds:
        f1s.append(f1_score(y[te], pred[te], average="macro")); accs.append(accuracy_score(y[te], pred[te]))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0)
    return dict(model=name, features=fs, macro_f1_mean=np.mean(f1s), macro_f1_std=np.std(f1s, ddof=1),
                accuracy_mean=np.mean(accs), accuracy_std=np.std(accs, ddof=1),
                pooled_macro_f1=f1_score(y, pred, average="macro"),
                **{f"recall_{c}": r for c, r in zip(CLASSES, rec)}, n_segments=len(y))


def cm_rows(y, p):
    cm = np.zeros((len(y), 9)); cm[np.arange(len(y)), y * 3 + p.astype(int)] = 1; return cm


def plot_f10(bdf=None):
    """F10: per-track-length macro-F1 bars; redrawable from results/T2c_by_track_len.csv."""
    if bdf is None:
        bdf = pd.read_csv(os.path.join(RES, "T2c_by_track_len.csv"))
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(bdf)); w = 0.38
    for k, (col, lab, colr) in enumerate([("RandomForest|engineered", "RF engineered", "#9aa5b1"),
                                          ("RandomForest|engineered+temporal", "RF engineered + temporal", "#2a6fdb")]):
        b = ax.bar(x + (k - 0.5) * w, bdf[col], w, label=lab, color=colr)
        ax.bar_label(b, fmt="%.3f", fontsize=8)
    ax.axhline(0.85, ls="--", c="k", lw=0.8, label="AC1 target 0.85")
    ax.set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(bdf.bucket, bdf.n)])
    ax.set_xlabel("track length (scans, causal)"); ax.set_ylabel("pooled OOF macro-F1")
    # headroom above 1.0 so the legend sits above the bars and the 0.85 line, never on them
    ax.set_ylim(0, 1.13); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title("Segment classification by track length")
    ax.legend(fontsize=8, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "F10_temporal_gain.png"), dpi=150); plt.close(fig)


def main():
    t0 = time.time()
    full = pd.read_csv(os.path.join(DATA, "segments.csv"))
    tf, prev = temporal_features(full)
    tcols = list(tf.columns)
    t_track = time.time() - t0
    print(f"[T2c] tracking on {len(full)} segments: {t_track:.0f}s; linked {np.mean(prev >= 0):.3f}; "
          f"track_len dist: {pd.Series(tf.track_len).clip(upper=6).value_counts().sort_index().to_dict()}", flush=True)
    full = pd.concat([full, tf], axis=1)

    # identical subsample to T2 (same call, same seed, same row count -> same rows)
    ref = pd.read_csv(os.path.join(DATA, "segments.csv")).sample(n=N_SUB, random_state=SEED)
    seg = full.sample(n=N_SUB, random_state=SEED)
    assert (seg.index.values == ref.index.values).all()
    fcols = [c for c in ref.columns if c.startswith("f_")]
    assert np.allclose(seg[fcols].values, ref[fcols].values) and (seg.label.values == ref.label.values).all()
    seg = seg.reset_index(drop=True); del ref

    y = seg.label.values; g = seg.map_seed.values
    folds = outer_folds(g)
    fsets = {"engineered": fcols, "engineered+temporal": fcols + tcols}
    oof, rows = {}, []
    mdl = models()
    runs = [("RandomForest", "engineered"), ("RandomForest", "engineered+temporal"), ("XGBoost", "engineered+temporal")]
    t2 = pd.read_csv(os.path.join(RES, "T2_segment_classification.csv"))
    ref_rf = t2.query("model=='RandomForest' and features=='engineered'").macro_f1_mean.iloc[0]
    for name, fs in runs:
        t1 = time.time()
        est, grid = mdl[name]
        pred, _, best = oof_predict(make_pipe(est), grid, seg[fsets[fs]].values, y, g, "f1_macro", folds)
        oof[(name, fs)] = pred
        rows.append({**evaluate(name, fs, pred, y, folds), "best_params": str(best), "seconds": time.time() - t1})
        print(f"[T2c] {name:12s} {fs:20s} macroF1={rows[-1]['macro_f1_mean']:.4f}±{rows[-1]['macro_f1_std']:.4f} "
              f"pooled={rows[-1]['pooled_macro_f1']:.4f} ({time.time()-t1:.0f}s)", flush=True)
        if (name, fs) == ("RandomForest", "engineered"):
            diff = abs(rows[-1]["macro_f1_mean"] - ref_rf)
            assert diff <= 0.005, f"RF engineered {rows[-1]['macro_f1_mean']:.4f} does not reproduce T2 {ref_rf:.4f}"
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "T2c_temporal.csv"), index=False)

    # bootstrap over maps
    pa, pb = oof[("RandomForest", "engineered")], oof[("RandomForest", "engineered+temporal")]
    ms = per_map_stats(g, np.hstack([cm_rows(y, pa), cm_rows(y, pb)]))
    vb, lb, hb, _ = bootstrap_maps(ms, lambda v: macro_f1_from_cm(v[9:]))
    vd, ld, hd, _ = bootstrap_maps(ms, lambda v: macro_f1_from_cm(v[9:]) - macro_f1_from_cm(v[:9]))
    acc = [dict(criterion="AC1 re-check: RF eng+temporal pooled OOF macro-F1 >= 0.85", value=vb, ci_low=lb, ci_high=hb,
                **{"pass": bool(vb >= 0.85)}),
           dict(criterion="gain: RF (eng+temporal) minus RF (eng) pooled macro-F1, map-bootstrap 95% CI",
                value=vd, ci_low=ld, ci_high=hd, **{"pass": bool(ld > 0)})]
    # by track_len bucket
    tl = seg.track_len.values
    buckets = [("1", tl == 1), ("2-3", (tl >= 2) & (tl <= 3)), ("4+", tl >= 4)]
    brows = []
    for bn, m in buckets:
        r = dict(bucket=bn, n=int(m.sum()))
        for (name, fs), p in oof.items():
            r[f"{name}|{fs}"] = f1_score(y[m], p[m], average="macro")
        brows.append(r)
        acc.append(dict(criterion=f"descriptive: gain (RF eng+temporal minus RF eng) macro-F1, track_len {bn} (n={m.sum()})",
                        value=r["RandomForest|engineered+temporal"] - r["RandomForest|engineered"],
                        ci_low=np.nan, ci_high=np.nan, **{"pass": "NA"}))
    acc_df = pd.DataFrame(acc)
    acc_df.to_csv(os.path.join(RES, "T2c_acceptance.csv"), index=False)
    bdf = pd.DataFrame(brows)
    bdf.to_csv(os.path.join(RES, "T2c_by_track_len.csv"), index=False)
    print(res.drop(columns=["best_params"]).round(4).to_string(index=False))
    print(acc_df.round(4).to_string(index=False))
    print(bdf.round(4).to_string(index=False))

    plot_f10(bdf)
    print(f"[T2c] done in {time.time()-t0:.0f}s (tracking {t_track:.0f}s)")


if __name__ == "__main__":
    main()
