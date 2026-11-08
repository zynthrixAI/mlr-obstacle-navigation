# Vision task (T6) methodology review

**Verdict: CHANGES** — protocol is sound; the LiDAR-RF baseline is understated by an avoidable choice and the headline needs re-framing.

## 1. Leakage / protocol — PASS

- `load_seg` = `seg.sample(60000, random_state=0)` as in T2; `crops.npz` labels/map_seeds verified identical to the T2 subsample (300 maps).
- `outer_folds` (GroupKFold by map_seed) reused; inner `GroupShuffleSplit(groups=g[tr])`; disjointness asserted.
- CNN mean/std from `Xtr`; class weights from `ytr`; flip/brightness augmentation on training batches only; early stopping on validation F1; `Xte` only predicted. RF/LogReg/Dummy fit on `itr`; fusion weight grid scored on `iva`. Bootstrap resamples maps.
- Per-crop noise rng seeded by row index (label-independent).

## 2. Label leakage via rendering — PASS (no hidden channel), but the designed task is near-trivial

- Appearance rng is separate from geometry.
- Walls: saturation ≤ 0.15; obstacles ≥ 0.30 — zero overlap. Central-pixel mean saturation > 0.25 separates wall vs non-wall at 99.1 %.
- Obstacles inside the full target envelope (hue, sat ≥ 0.6, val ≥ 0.6, height 0.4–0.6): 27/2493 = 1.1 %; 59 % of targets carry a stripe no obstacle has.
- A 3-line colour rule scores macro-F1 0.854; a depth-3 tree on four hand features 0.869 OOF. Bayes ceiling ≈ 0.99, so AC2's 0.80 is undemanding. Framing leaks width only weakly (r = 0.32 with `f_width_deg`), which LiDAR has anyway.
- ColorHist-LogReg (0.704) is weaker than the hand rule, so not a strong "simple" baseline.

## 3. CNN vs LiDAR-RF fairness — FAIL as stated

Re-fitting the T6 RF with T2's selected `min_samples_leaf=5` on the same 80 % inner-train maps gives macro-F1 **0.637** (reported 0.585; 100 % of train maps also 0.637). The deficit is entirely the untuned `min_samples_leaf=1`; the CSV note blaming "80 % of training maps" is wrong.

## MUST-FIX

1. Use `min_samples_leaf=5` (or T2's 3-value grid on `iva`) for the T6 RF, rerun, correct the note. Expect LiDAR-RF ≈ 0.64, AC3 margin ≈ 0.31 (still passes).
2. Present AC2/AC3 as passes against authored appearance, not as evidence that vision beats LiDAR in general; cite the hand-rule number beside the CNN.

## Recommended wording

*Interpretation:* "On synthetic crops the CNN reaches macro-F1 0.94 and late fusion 0.95, against 0.64 for the LiDAR Random Forest on the same folds. The gap is expected: we designed the appearance model so that colour and a stripe almost fully disambiguate the classes (a three-rule colour heuristic already reaches 0.85), whereas 2D geometry cannot. The result shows that the pipeline, fusion and protocol work, not that a camera would deliver this margin in practice."

*Limitations:* "Class appearance (hue, saturation, height, stripe) was authored by us with little overlap, so the vision scores are an upper bound tied to our rendering choices and do not transfer to real imagery."

Minor: CNN best epochs vary 1–7 of 8; consider raising `PATIENCE`.
