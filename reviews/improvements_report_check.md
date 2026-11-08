# Report check: safety shield and temporal tracking (commit fd0336f)

**Verdict: APPROVE.** Every number in the new text of `reports/report.docx` and `README.md` matches the result files; must-fixes 1–4 of `reviews/improvements.md` are addressed; pre-existing numbers are unchanged.

## Must-fixes from improvements.md

1. **Build crash** — fixed. `build_report.py:377/380` map `ppo_shield` -> "PPO + safety shield (post-hoc)"; `report.docx` (14:47) post-dates T5 (14:01) and T2c (14:16).
2. **Shield described and labelled** — done. §6.4 paragraph states rule, 0.20 m / 0.35 m constants (= `SHIELD_HALF_W`, `SHIELD_AHEAD`), 36 rays, dropout = free, reactive turn to the more open side (`1 if l >= rr else 2`), no retraining, designed after seeing collisions, second use of the held-out set, not part of AC8, narrow-map 10° blind-spot limitation. Table 13 holds exactly the 6 pre-specified rows; Table 14 holds the 6 `[post-hoc]` rows under a separate "not pre-specified" caption (`a5.criterion.str.startswith("[post-hoc]")`).
3. **T2c in the report** — done. §6.1 "Exploratory: temporal tracking" with Tables 5–6 and Figure 3a (F10), plus the assumptions paragraph (exact poses, 0.3 m one-to-one greedy link, links never cross objects, purity-filtered file only). Labelled post-hoc / not an AC.
4. **Discussion and README** — done. AC1 tracking bullet, AC8 shield bullet, both suitability rows, "Post-hoc analyses" limitation; README task list mentions T2c (`tasks/temporal.py`, run via `tasks.run_all`, confirmed at `run_all.py:22`) and the `ppo_shield` agent; Outputs line mentions T2c and F10.

## Number checks (all match)

- §4 post-hoc sentence: no numbers; consistent with Tables 13/14.
- §6.1 text vs `tasks/temporal.py`: 0.3 m (`LINK_DIST`), previous scan = step − 10 (`SEG_EVERY = 10`), up to 5 views (`HIST`), mean/max of n, width, chord, circle radius, linearity and circle residuals, bulge (`AGG_MEANMAX`), std of chord and radius (`AGG_STD`), view spread, track length; labels unused in `link_tracks`.
- Table 5 vs `T2c_temporal.csv`: 0.635 ± 0.012 / 0.635 / 0.747 / 0.832 / 0.655 / 0.457; 0.713 ± 0.014 / 0.713 / 0.817 / 0.894 / 0.737 / 0.531; XGB 0.701 ± 0.012 / 0.702 / 0.834 / 0.933 / 0.756 / 0.334.
- Table 6 vs `T2c_acceptance.csv`: 0.713 [0.702, 0.724] No; 0.078 [0.071, 0.084] Yes; 0.003 (n=25252), 0.063 (n=17841), 0.126 (n=16907). Figure 3a bars = `T2c_by_track_len.csv` (0.552/0.555, 0.627/0.690, 0.653/0.779).
- Table 12 `ppo_shield` rows vs `T5_rl.csv`: open 0.960 ± 0.035, 0.000, 1.280 ± 0.033, 0.486 ± 0.021, 0.034 ± 0.006; narrow 0.167 ± 0.012, 0.033 ± 0.012, 0.995 ± 0.005, 0.347 ± 0.014, 0.166 ± 0.022; cluttered 0.773 ± 0.076, 0.000, 1.165 ± 0.017, 0.240 ± 0.006, 0.256 ± 0.030.
- Tables 13/14 vs `T5_acceptance.csv`: identical values and pass flags, row for row.
- Discussion: +0.078, CI 0.071–0.084, +0.003/+0.063/+0.126, 0.779, 42 % with no predecessor (25252/60000 = 42.1 %), 0 % open/cluttered from 4 %/31 %, success 96 %/77 %, narrow 17 % -> 3.3 %, success 17 %; suitability "+0.078 (0.713)", "96 % / 77 %, 0 % collisions", "4–31 %".
- Conclusion: 4 % -> 0 %, 31 % -> 0 %, success 0.960 / 0.773, 3.3 % narrow, 0.635 -> 0.713 (computed from the CSVs in `build_report.py:478-479`).
- README: 0.635 -> 0.713, "up to 5 past views".

## Cross-references and renumbering

- 17 table captions, in order T1 dataset, T2 UCI, T3/T4 segments, T5/T6 temporal, T7–T9 regression, T10–T11 clustering, T12–T14 RL, T15–T16 vision, T17 suitability; every in-text "Table N" (§6.2 7/8/9, Figure 4 caption "Table 9", §6.3 10/11, §6.4 12/13/14, §6.5 15/16) points at the right caption.
- Figures 1–10 plus 3a (`label_override` does not advance the counter, so Figure 4 follows 3a); "Figure 3" in the discussion is still the confusion matrix; Figure 7 (F6, 14:01) includes `ppo_shield` as its caption says.
- `diff` of old (661c7ff) vs new docx text shows only the additions above and the Table 5->7 … 14->17 renumbering; no pre-existing value changed.

## Should-note (not blocking)

- Table 14 shows "0.96" where the other rows use three decimals; it is copied verbatim from `T5_acceptance.csv`.
- Discussion/§7 "success stayed at 17 %" in narrow maps: plain PPO is 15.3 %, shield 16.7 %; "rose slightly from 15 % to 17 %" would be more precise.
- §6.1 feature list omits two features the code also adds (`t_hist_n` history count and `t_mean_mean_r` mean range); harmless.
