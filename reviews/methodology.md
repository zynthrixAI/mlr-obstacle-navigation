# Methodology review

**Verdict: APPROVE** (no reported number found to be wrong; two claims need rewording, see MUST-FIX).

## Checklist

1. **Grouped, out-of-fold CV — PASS.** `tasks/common.py: outer_folds` = GroupKFold(5) on `map_seed`; `oof_predict` asserts group disjointness, runs `GridSearchCV(cv=GroupKFold(3), groups=groups[tr])` on the training part only, predicts the held-out fold. Imputer/scaler/poly sit in `make_pipe` Pipelines; class weights (`balanced`, `balanced_subsample`, UCI `compute_sample_weight(yt)`) come from training rows at fit time. AC values in `T2/T3_acceptance.csv` match the pooled-OOF rows.
2. **Seed ranges — PASS.** `sim/seeds.py` asserts pairwise disjointness at import; `rl/evaluate.py` re-asserts; eval uses `NavEnv(seed_pool=RL_EVAL)` + `options={"map_seed": ms}`.
3. **Purity filter — PASS.** `sim/generate.py: run_map` drops majority share < 0.90 (or majority no-hit) and counts; T1 arithmetic holds (434 940 − 218 636 − 3 479 = 212 825); `segments.csv` min purity 0.900.
4. **Regression strata — PASS (computation); see MUST-FIX 1.** `World.nearest` → `wrap(bearing − θ)` → `robot.observable` (≤ 15° from one of 8 bearings); recomputed from `nearest_bearing`: 100 % agreement. AC5 masks `obs & (y < 0.5)`. *Meaningfulness:* the gap is 15° wide, so the nearest cone edge is ≤ 7.5° from the nearest point, and all objects are extended (walls, boxes ≥ 0.3 m, cylinders r ≥ 0.12 m); an edge sub-ray hits the same surface at ≈ d/cos 7.5° ≤ 1.009 d. Noiseless check (4 000 frames): min-US − d_true in blind rows has median 1.5 mm, MAE 16 mm, > 5 cm in 5.2 %, > 20 cm in 1.4 %. The 0.024 m baseline MAE is thus dominated by the 3 % multiplicative noise at median d = 0.35 m (σ ≈ 1 cm) and min-of-8 bias, not blindness. Legitimate only as "nearest point between cones", not "object unobserved".
5. **Clustering null / bootstrap — PASS.** `clustering.py`: `rng.permutation(maps.t)` permutes map_type over 300 maps, windows inherit; bootstrap resamples map indices; p = (1+#≥)/(N+1).
6. **RL — PASS.** `evaluate.py`: `vn.training=False; vn.norm_reward=False`, `vn.normalize_obs`, `deterministic=True`, held-out seeds. `env.py: step` implements exactly 10(d_prev−d) − 0.05 − 0.5[min_ray<0.30] + 100[goal] − 100[collision] (min_ray from the noisy scan, dropout → 5 m). `sectors36` matches its docstring; `obs_to_state` = 3×3×3×8 = 216; ε linear 1→0.05 over 70 %. **path_ratio < 1 is not a bug**: episodes end at d < 0.2 m, so path ≥ start_dist − 0.2; all 416 successes satisfy path_ratio ≥ 1 − 0.2/start_dist (min 0.929); path is an exact multiple of 0.1.
7. **Primary model — PASS.** README and `primary=` flags fix Random Forest; all AC rows use RF; XGBoost's higher T2 accuracy is not used for any claim.
8. **UCI — PASS.** Contiguous fifths, train = rows outside [lo−50, hi+50). Lag-1 deltas are causal and the only cross-boundary neighbour is a gap row; 0.0 % of test rows have an exact 4-sensor duplicate in train. Inner `KFold(3, shuffle=False)` has no gap → tuning optimism only; outer estimates unaffected.

## MUST-FIX (claims)

1. **Do not call the `observable=False` stratum a sensor blind spot** (`sim/robot.py: observable`, report Task 2). Rename to "nearest point between cones" and, if a blind-spot claim is wanted, add a post-hoc stratum in `tasks/regression.py` from stored `(x, y, theta, map_seed)`: truly_blind = noiseless `ultrasonic(...).min() − d_nearest_true > 0.05` (~5 % of current blind rows, ≈ 440 frames) and report RF vs baseline MAE there.
2. **State the T2 subsample.** `segment_classification.py` evaluates a uniform 60 000-row subsample of 212 825 segments (`N_SUB`); `T2_segment_classification.csv` reports `n_segments=60000` while T1 reports 212 825 kept. The report must say so wherever T2 numbers appear.

## SHOULD-NOTE (Limitations)

- AC5a (≤ 0.10 m) is passed by the trivial min-US baseline (0.0147 m), so it carries no evidence; only AC5b (fails, 10.6 % reduction) is informative. 74 % of frames have d < 0.5 m.
- T1 "segments_total/dropped_small" are inflated by noise-induced fragments: ~20 % of no-hit rays fall below the 4.95 m validity cut and form 1–2-point pseudo-segments; "impure" also counts majority-no-hit segments.
- RL `path_ratio` is conditional on success (per-agent selection bias); reactive/random have one seed (no std).
- Training success in `train_log.txt` pools map types; eval is per type with deterministic start/goal.
- UCI inner CV lacks a gap (tuning optimism); labels are near-deterministic (ceiling ≈ 1.0).
- Clustering: map generators are distinct by design; silhouette prefers k=2, k=3 is reported by prior choice.
