# Final report review — reports/report.docx

**Verdict: CHANGES** (minor, but one terminology conflict is visible in a figure).

## Number check

Every value in Tables 1–12 matches the corresponding `results/*.csv` file (T1, uci_summary, T2, T2_acceptance, T3, T3_acceptance, T3b, T4, T4_permutation, T5, T5_acceptance). All prose constants (sensor noise/dropout, cone geometry, map/episode/step counts, segmentation threshold, window length, 60,000 subsample, 50-row UCI gap, GroupKFold 5/3, ≤ 6 configs, seed ranges, reward, 400 steps, 216 states, Q-learning schedule, 300k steps × 3 seeds, 500 permutations/bootstraps) match `sim/`, `tasks/`, `rl/`. Derived figures (1.9% = 521/27,000; −34%; 10.6%) are correct; §7–9 are verbatim from `discussion.md`. **No numeric mismatches found.**

Claims: RF is the only model recommended as primary; AC1 fail, AC5 fail on AC5b only, AC8 open/cluttered success pass, narrow fail, all collision criteria fail — all reported correctly; the 60,000-row subsample is stated in §6.1, Table 3 and §9; the six references are real works, nothing fabricated.

## MUST-FIX

1. **Figure 4 legend calls the observable = False stratum "blind-spot"** ("baseline / blind-spot", "RF-eng / blind-spot"), contradicting the report's own wording and methodology MUST-FIX 1. Preferred: change the label in `tasks/regression.py` line 105 from `"blind-spot"` to `"between cones"` and regenerate `F3_regression_mae_by_bin.png`. If the figure is not regenerated, replace the Figure 4 caption with: *"Figure 4: Regression MAE by true-distance bin. 'blind-spot' in the legend denotes the observable = False stratum (nearest point between cones), not the truly blind stratum of Table 7."*
2. **Dangling figure cross-references in §7** (copied from discussion.md file names). Replace "The confusion matrix (F2) shows" with "The confusion matrix (Figure 3) shows", and "The centroid profiles (F5) are interpretable" with "The centroid profiles (Figure 6) are interpretable".

## Minor notes (layout, not blocking)

- Table 5 breaks across pages 7–8 mid-row: "(primary)" of the RandomForest-raw row is orphaned at the top of page 8 and the header row is not repeated. Set "repeat header row" and "keep row together" (or move the table), as readers of page 8 see unlabeled columns.
- Table 8 "Note" column wraps "k=3(reported)" leaving ")" on its own line; widen the column or shorten to "k=3 (reported)".
- Figure 3's embedded title is clipped ("row-normalise"); cosmetic, from the PNG itself.
- Table 12 "null 0.02" rounds the null 99th percentile (0.017); acceptable but 0.017 would match Table 9.
