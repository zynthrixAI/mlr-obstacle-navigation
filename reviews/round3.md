# Methodology review, round 3 (commits a693457, 8fda616)

**Verdict: APPROVE** on methodology. One presentation must-fix (F12) before the figure goes into the report; two should-fixes.

## 1. T2d odometry noise (tasks/temporal_odometry.py, T2d_odometry.csv, F11)

Checked, all OK:
- Noise model is causal and correctly composed: true scan-to-scan increment is expressed in the *previous true* robot frame, perturbed by N(0, sigma_t) in x/y and N(0, sigma_h) in heading, then composed onto the *previous estimated* pose (rotation by estimated heading, heading wrapped). First scan anchored at truth. Verified on a synthetic curved path that the body-frame increment residuals are ~N(0, sigma) and the error accumulates as a random walk. Zero-noise level asserts equality with true poses.
- RNG: `default_rng(crc32(f"{episode_id}|{level}"))`, drawn per scan from pose/step columns only. Confirmed label-independent (same estimates after permuting labels) and level-dependent.
- No label use: `noisy_poses` touches only episode_id/step/pose_*; `link_label_agreement` is computed but used only as a diagnostic column (never a feature).
- Same 60k subsample (`full.sample(n=N_SUB, random_state=SEED)`, same as T2/T2c), same `outer_folds(map_seed)`; temporal features computed on the full table with estimated poses and then `.loc[sub_idx]`, as in T2c. Shape features remain robot-frame and untouched.
- CIs: pooled and gain CIs are map-bootstrapped (1000 resamples, `bootstrap_maps`) from per-map confusion sums. Correct.
- `none` row reproduces T2c exactly: pooled 0.7132886639 (T2c 0.7132886639), gain +0.07792 [0.07097, 0.08441] identical to T2c_acceptance. Engineered-only row 0.6353670952 identical to T2c RF engineered, which confirms the fixed params (min_samples_leaf=5, max_features=sqrt) equal what the inner grid picked in all 5 folds of T2 and T2c (checked T2_segment_folds.csv and T2c best_params).
- Commit-message numbers match the CSV: gain +0.078/+0.078/+0.068/+0.047, all CI lows > 0 (0.071, 0.071, 0.062, 0.042).
- `tasks/temporal.py` diff is plot-only: the F10 block moved into `plot_f10(bdf)` with ylim 1.13 and legend moved to the top; no change to data, folds, features or CSVs. F10 looks fine (legend clear of bars and the 0.85 line). `run_all.py` only adds `temporal_odometry` to the module list.

Non-blocking note: sigma is per scan interval regardless of distance travelled (a stationary robot still drifts). Pre-specified and documented in the docstring and the F11 x-label; say so in the report when citing T2d, and update the "exact simulator poses" caveat in reports/discussion.md to point at T2d.

## 2. Hard camera appearance (sim/world.py, sim/camera.py, tasks/vision.py --appearance hard; T6b_*, F12, F13)

Checked, all OK:
- Docstring spec matches code line by line: walls 15% saturated random colour (sat U(0.6,0.95), val U(0.5,0.9)) else easy grey/beige formula (same ranges as the easy code: g U(0.55,0.75), g*U(0.95,1.0), g*U(0.85,0.95)); obstacles 30% target-like colour + height U(0.4,0.6), else easy obstacle distribution, stripe 20%; targets height U(0.4,0.6), 80% target hue N(0.03,0.03) else hue U(0.12,0.94), stripe 40%. Fixed 7 draws per object regardless of branch. Occlusion: p=0.25, width round(U(0.20,0.45)*32), uniform position, random unstriped occluder (hue U(0,1), sat U(0.3,0.95), val U(0.4,0.95), height U(0.3,1.2)), depth U(0.5,0.9) x nearest band depth; always 8 draws.
- Separate RNGs: appearance `default_rng(seed + 31_337_771)` vs easy `seed + 7_919_993`; occlusion `default_rng([row, 0x0CC1])` vs crop noise `default_rng(row)`. Appearance arrays are new read-only arrays; the cached World is never mutated.
- Easy outputs untouched: `git diff d6ad7e0 HEAD -- results/T6_vision.csv results/T6_acceptance.csv results/F8_* results/F9_*` is empty; easy render path (`appearance=None, occ_rng=None`) is byte-for-byte the old code path. LiDAR-RF row in T6b is identical to T6 (0.6365582456), which also confirms the same subsample and outer folds.
- Label leakage: occlusion probability, band position/width, occluder colour/height depend only on the row index (rows are a random, label-independent permutation of the subsample) and the band's geometry, never on class. Appearance branch probabilities depend on class by design (that *is* the appearance model, applied at the object level by map seed) and are not available to the classifier except through pixels. Height remains a legitimate cue (walls 1.0 m, targets 0.4-0.6 m, 70% of obstacles 0.3-1.2 m), as specified.
- Fusion weight chosen on the inner validation maps only (`GroupShuffleSplit` by map inside each outer-train fold; grid over w on `pc_va`/`prf_va`; CNN early stopping also on that split). Test maps untouched until prediction.
- Numbers match claims: CNN 0.7545 (0.755), Fusion 0.8073 (0.807), LiDAR-RF 0.6366 (0.637), ColourRule 0.5586 (0.559), Fusion - CNN +0.0528 [0.0466, 0.0599] (+0.053 [0.047, 0.060]).
- F13: crops show the intended ambiguity (target-coloured obstacles, striped obstacles, unstriped/odd-hue targets, coloured walls, occluders).

### Must-fix
1. **F12 titles overlap.** The "[hard appearance] " prefix on each of the three panel titles makes them run into each other ("...(pooled O[hard appearance] CNN (pooled OO[h]ard..."). Put the tag in `fig.suptitle` and keep per-axis titles as `f"{n} (pooled OOF)"`. Since OOF predictions are not persisted, redrawing needs a rerun of the hard pipeline; while doing so, save the OOF probabilities/preds (e.g. `results/T6b_oof.npz`) so figures are redrawable without retraining.

### Should-fix (not blocking)
2. T6b_acceptance_hard.csv: the AC2/AC3 rows still carry `pass` False/True even though the note row says thresholds are easy-variant only. Prefix those criteria with "descriptive:" or set `pass` to "NA (descriptive)" so a reader cannot count "hard AC3" as a pass.
3. ColourRule was hand-written against the easy appearance spec; in the hard table label it as such (it is a fair no-fit baseline, but not a rule designed for the hard variant).
