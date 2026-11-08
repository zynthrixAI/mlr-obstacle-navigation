"""Generate the supervised datasets from the simulator.

Outputs (data/):
  frames.csv    one row per recorded frame (every 5 steps): 36-ray LiDAR, 8 ultrasonics, ground truth
  segments.csv  one row per LiDAR object segment (360-ray scan, every 10 steps) with raw + engineered features

Run:  python -m sim.generate
"""
from __future__ import annotations

import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .robot import (MAX_RANGE, goal_polar, lidar, observable, reactive_action, step_pose, ultrasonic, wrap)
from .seeds import SUPERVISED
from .world import make_world

EPISODES, STEPS, FRAME_EVERY, SEG_EVERY = 3, 150, 5, 10
EPS_RANDOM = 0.2
JUMP = 0.15            # segmentation range-jump threshold (m)
PURITY = 0.90
MIN_POINTS = 3
RAW_RESAMPLE = 8
OUT = os.path.join(os.path.dirname(__file__), "..", "data")


# ---------------------------------------------------------------------- segmentation
def segment_scan(r, cls):
    """Split a 360-ray scan into segments of consecutive valid hits with |dr| <= JUMP.
    Returns list of index arrays (handles wrap-around)."""
    n = len(r)
    valid = ~np.isnan(r) & (r < MAX_RANGE - 0.05)
    brk = np.ones(n, bool)                   # brk[i]: a segment starts at i
    for i in range(n):
        j = i - 1
        brk[i] = not (valid[i] and valid[j] and abs(r[i] - r[j]) <= JUMP)
    if not brk.any():                        # one closed loop (rare)
        return [np.arange(n)] if valid.all() else []
    start = int(np.argmax(brk))
    order = np.roll(np.arange(n), -start)
    segs, cur = [], []
    for i in order:
        if brk[i] and cur:
            segs.append(np.array(cur))
            cur = []
        if valid[i]:
            cur.append(i)
    if cur:
        segs.append(np.array(cur))
    return segs


def segment_features(r, idx, n_rays=360):
    rr = r[idx]
    ang = 2 * np.pi * idx / n_rays
    pts = np.stack([rr * np.cos(ang), rr * np.sin(ang)], 1)
    n = len(idx)
    # raw: point count, mean range, ranges resampled to fixed length
    raw = {"raw_n": n, "raw_mean_r": rr.mean()}
    res = np.interp(np.linspace(0, n - 1, RAW_RESAMPLE), np.arange(n), rr)
    for k, v in enumerate(res):
        raw[f"raw_r{k}"] = v
    # engineered geometry
    chord = np.linalg.norm(pts[-1] - pts[0])
    path = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    c = pts - pts.mean(0)
    _, sv, _ = np.linalg.svd(c, full_matrices=False)
    lin_rms = sv[-1] / np.sqrt(n) if n > 2 else 0.0
    # algebraic (Kasa) circle fit
    A = np.c_[2 * pts, np.ones(n)]
    b = (pts ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, cc = sol
    rad = np.sqrt(max(cc + cx ** 2 + cy ** 2, 0.0))
    circ_rms = np.sqrt(np.mean((np.linalg.norm(pts - [cx, cy], axis=1) - rad) ** 2))
    # convexity: is the arc bulging toward the robot? (centre farther than the points)
    bulge = np.linalg.norm([cx, cy]) - rr.mean()
    eng = {
        "f_n": n, "f_width_deg": n * 360 / n_rays, "f_mean_r": rr.mean(), "f_min_r": rr.min(),
        "f_std_r": rr.std(), "f_chord": chord, "f_path": path, "f_path_chord": path / max(chord, 1e-6),
        "f_lin_rms": lin_rms, "f_circ_r": min(rad, 10.0), "f_circ_rms": circ_rms,
        "f_bulge": np.clip(bulge, -10, 10), "f_end_dr": rr[-1] - rr[0],
    }
    return raw, eng


# ---------------------------------------------------------------------- episodes
def run_map(seed: int):
    w = make_world(seed)
    rng = np.random.default_rng(seed + 7)
    frames, segs = [], []
    stats = {"seg_total": 0, "seg_small": 0, "seg_impure": 0}
    for ep in range(EPISODES):
        s, g = w.sample_start_goal(rng)
        pose = (s[0], s[1], rng.uniform(-np.pi, np.pi))
        for t in range(STEPS):
            r36, _ = lidar(w, pose, 36, rng)
            dist, ang = goal_polar(pose, g)
            if dist < 0.3:                                  # new goal when reached
                g = w.sample_free(rng, 0.35)
                dist, ang = goal_polar(pose, g)
            if t % FRAME_EVERY == 0:
                us = ultrasonic(w, pose, rng)
                dn, bearing = w.nearest(np.array(pose[:2]))
                rel = wrap(bearing - pose[2])
                row = {"map_seed": seed, "map_type": w.map_type, "episode_id": f"{seed}-{ep}", "step": t,
                       "x": pose[0], "y": pose[1], "theta": pose[2],
                       "d_nearest_true": dn, "nearest_bearing": rel, "observable": observable(rel)}
                row.update({f"lidar_{i:02d}": v for i, v in enumerate(r36)})
                row.update({f"us_{i}": v for i, v in enumerate(us)})
                frames.append(row)
            if t % SEG_EVERY == 0:
                r360, c360 = lidar(w, pose, 360, rng)
                for idx in segment_scan(r360, c360):
                    stats["seg_total"] += 1
                    if len(idx) < MIN_POINTS:
                        stats["seg_small"] += 1
                        continue
                    lab = c360[idx]
                    vals, cnt = np.unique(lab, return_counts=True)
                    k = np.argmax(cnt)
                    if vals[k] < 0 or cnt[k] / len(idx) < PURITY:
                        stats["seg_impure"] += 1
                        continue
                    raw, eng = segment_features(r360, idx)
                    segs.append({"map_seed": seed, "map_type": w.map_type, "episode_id": f"{seed}-{ep}",
                                 "step": t, "label": int(vals[k]), "purity": cnt[k] / len(idx), **raw, **eng,
                                 "bearing_start_deg": int(idx[0]), "bearing_end_deg": int(idx[-1]),
                                 "pose_x": pose[0], "pose_y": pose[1], "pose_theta": pose[2]})
            a = int(rng.integers(3)) if rng.random() < EPS_RANDOM else reactive_action(r36, ang)
            pose, _ = step_pose(w, pose, a)
    return frames, segs, stats


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    with Pool(os.cpu_count()) as p:
        res = p.map(run_map, list(SUPERVISED), chunksize=4)
    frames = pd.DataFrame([r for f, _, _ in res for r in f])
    segs = pd.DataFrame([r for _, s, _ in res for r in s])
    st = pd.DataFrame([s for _, _, s in res]).sum()
    frames.to_csv(os.path.join(OUT, "frames.csv"), index=False)
    segs.to_csv(os.path.join(OUT, "segments.csv"), index=False)
    summary = pd.DataFrame({
        "item": ["maps", "episodes", "frames", "segments_total", "segments_dropped_small(<3pts)",
                 "segments_dropped_impure(<90%)", "segments_kept", "frames_observable_frac"],
        "value": [frames.map_seed.nunique(), frames.episode_id.nunique(), len(frames), st.seg_total,
                  st.seg_small, st.seg_impure, len(segs), round(frames.observable.mean(), 3)]})
    os.makedirs(os.path.join(OUT, "..", "results"), exist_ok=True)
    summary.to_csv(os.path.join(OUT, "..", "results", "T1_dataset.csv"), index=False)
    print(summary.to_string(index=False))
    print("label counts:", segs.label.value_counts().to_dict())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
