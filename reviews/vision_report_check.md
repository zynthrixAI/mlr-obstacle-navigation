# Vision text check — reports/report.docx (8 Nov 2026)

**Verdict: APPROVE**

Every number in the new vision-related text matches the result files and code. No mismatches found.

## Checked (all match)

- **Table 12** vs `results/T6_vision.csv`: all six rows (Majority, ColourRule, ColorHist-LogReg, CNN, LiDAR-RF, Fusion) — mean ± std, pooled macro-F1, accuracy, per-class recall — round correctly to 3 decimals (e.g. CNN 0.948 ± 0.013 / 0.948 / 0.980 / 0.995 / 0.962 / 0.949; LiDAR-RF 0.636 ± 0.013 / 0.637; Fusion 0.959 ± 0.011 / 0.959 / 0.985 / 0.997 / 0.974 / 0.937).
- **Table 13** vs `results/T6_acceptance.csv`: AC2 0.948 [0.939, 0.957] Yes; AC3 0.323 [0.313, 0.334] Yes; descriptive CNN−RF 0.312 [0.301, 0.324]; note row text identical to the CSV.
- **Text numbers** (§6.5 interpretation, Discussion finding 5, suitability row, Conclusion): 0.948, 0.959, 0.637, 0.837, CI 0.939–0.957, +0.323 CI 0.313–0.334 — all correct. Fusion − CNN = 0.9593 − 0.9481 = +0.011 ✓.
- **Per-fold w** 0.5, 0.5, 0.8, 0.6, 0.5 = `fusion_w_per_fold` ✓.
- **CNN facts** vs `tasks/vision.py`: 3 conv blocks 16/32/64 ✓, Adam lr 1e-3 ✓, batch 256 ✓, class-weighted CE ✓, horizontal-flip + brightness (0.8–1.2) augmentation ✓, MAX_EPOCHS 8 ✓, PATIENCE 3 on val macro-F1 ✓, normalisation mean/std from inner-training crops only ✓, inner GroupShuffleSplit 80/20 by map ✓, same `outer_folds` GroupKFold(5) and N_SUB = 60 000 ✓, bootstrap by resampling maps ✓.
- **LiDAR-RF** min_samples_leaf = 5 ✓ (`vision.py`; T2 selected 5 in all five folds per `T2_segment_folds.csv`), fit on inner-train maps ✓.
- **Camera parameters** vs `sim/camera.py`: 32×32 RGB, height 0.30 m, 40° vertical FOV, ±3° padding, min HFOV 8°, distance shading, brightness 0.6–1.4, ±5 % per-channel tint, Gaussian σ = 8/255 ✓.
- **Appearance** vs `sim/world.py::_assign_appearance`: walls grey/beige 1.0 m; targets red–orange, 0.4–0.6 m, stripe 60 %; obstacles uniform random hue, 0.3–1.2 m ✓.
- **0.637 vs 0.635**: the text says "0.637 … on the same folds" (T6) and "Best macro-F1 (0.635)" (T2) without asserting a cause, so no contradiction.
- **Numbering/cross-refs**: §6.5 exists and is referenced from §3.1; Tables 12–13 then Table 14; Figures 9–10 follow Figure 8; Figure 3 (line "confusion matrix (Figure 3)" in AC1 discussion) → "Segment classification confusion matrix (RF, engineered)"; Figure 6 → "Cluster centroid profiles" ✓. Discussion "(finding 5)" → item 5 "Vision & fusion" ✓.
- **README.md** vision lines (32×32 crop, colour-rule / colour-hist LogReg / small CNN, late fusion, 60k segments, GroupKFold by map, T6 / F8 / F9 files, upper-bound caveat) consistent with code and results ✓.

## Non-blocking remarks (no change required)

1. §6.5 method: "Four image models and the LiDAR baseline" — Table 12 has three image-only models (ColourRule, ColorHist-LogReg, CNN) plus Majority, LiDAR-RF and Fusion. Wording nit only.
2. Figure 10 is produced from `results/F8_vision_confusion.png` and Figure 9 from `F9_crop_examples.png`; the figure numbers in the report are sequential and correct, just not aligned with the file prefixes (not an error).
