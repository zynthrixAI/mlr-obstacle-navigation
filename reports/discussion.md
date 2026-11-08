# Discussion (author: Opus; to be inserted verbatim into the report)

## Cross-task findings

**1. Feature engineering helps most where the raw signal is ambiguous.**
- In LiDAR segment classification, the geometric features (chord length, circle-fit radius and residual, linearity, bulge) raised Random Forest macro-F1 from 0.609 to 0.635. They raised logistic regression from 0.440 to 0.547, and XGBoost from 0.504 to 0.593.
- On the UCI data the 4 raw distances already almost determine the label, so the engineered features add nothing for tree models.
- In distance regression the min-of-ultrasonics feature is the dominant signal, and the extra features matter only where that minimum is wrong.

**2. Ensembles of trees are the most reliable default.** Random Forest was pre-specified as the primary model, and it was the best or close to the best on every supervised task:
- macro-F1 0.999 on UCI;
- macro-F1 0.635 on segments, where the best alternative was XGBoost at 0.593;
- R² 0.97 on distance.

Linear models were competitive only after feature engineering. k-NN was sensitive to feature scaling and to the choice of features (it got worse on engineered UCI features).

**3. The acceptance criteria failed for informative reasons.**
- **AC1, segment classification (0.85 target; RF reached 0.635):**
  - The miss is about what a single 2D scan can observe. Targets were deliberately small cylinders whose size overlaps the small obstacle cylinders. The confusion matrix (Figure 3) shows that most errors are between obstacle and target, and between short wall fragments and box faces.
  - Short segments (fewer than 6 points) reach only 0.48 macro-F1, against 0.69 for longer ones.
  - This is largely a sensing limit of a single scan. Telling these objects apart needs more views (temporal tracking) or appearance (a camera). We tested both.
  - **Tracking (exploratory, T2c):** linking each segment causally to the matching segment in the previous scan, and adding features aggregated over up to 5 past views, raised RF macro-F1 from 0.635 to 0.713 (+0.078, 95% CI 0.071–0.084). AC1 still fails. The gain grows with track length: +0.003 for segments seen once, +0.063 for 2–3 views and +0.126 for 4 or more (0.779). About 42% of segments have no predecessor, which caps the overall gain. Assumptions: exact simulator poses (perfect odometry), and objects spaced far enough apart that links are always correct, so this is an optimistic estimate.
  - **Camera:** see the vision & fusion experiment (finding 5).
- **AC5, distance regression (20% improvement over the min-ultrasonic baseline not met; RF achieved 10.6% on observable near-range rows):**
  - With 30° cones at 45° spacing, the nearest point is almost always inside, or within 7.5° of, a cone. So the raw minimum reading is already an excellent estimator (MAE about 2 cm).
  - Learning pays off where the sensor model genuinely fails. On the **truly blind frames** (1.9% of frames, where no cone sees the nearest point), RF cut the baseline MAE from 0.212 m to 0.140 m (−34%), and the near-range MAE from 0.315 m to 0.192 m.
  - In a safety setting those are the frames that matter. Learned correction is therefore most valuable as a targeted complement to the raw minimum reading, not as a replacement for it.
- **AC8, RL navigation (partially met):**
  - PPO met the success thresholds in open maps (93%) and cluttered maps (61%), against 60% and 18% for the reactive controller.
  - It failed in narrow serpentine corridors (15%, close to the reactive controller's 16%).
  - It failed the safety criterion: PPO collided in 4%, 17% and 31% of episodes (open, narrow, cluttered), against 0–2% for the reactive rule.
  - PPO has learned an aggressive goal-seeking policy. A −100 collision penalty was not enough to outweigh the dense progress reward over 300k steps.
  - Narrow corridors need long detours away from the goal, which a myopic progress reward actively discourages. This is the classic reward-shaping trap.
  - **Safety shield (post-hoc, T5):** a run-time veto replaces PPO's "forward" with the reactive turn whenever a LiDAR ray lies in the corridor the robot body would sweep (lateral < 0.20 m, ahead < 0.35 m). Without retraining, collisions fell to 0% in open and cluttered maps (from 4% and 31%), and success rose to 96% and 77%. In narrow maps collisions fell from 17% to 3.3% but success stayed at 17%. The remaining collisions come from thin wall ends that fall between the 10° LiDAR rays, a simulator artefact no LiDAR-only shield can catch. The shield was designed after seeing PPO's collisions and is a second use of the held-out maps, so these are post-hoc results reported separately from the pre-specified AC8.
  - Tabular Q-learning (216 states) stayed safe (collisions ≤ 2%) but was weak outside open maps (8% success). The coarse state cannot represent corridor structure.

**4. Clustering recovers environment type, with caveats.**
- K-Means with k=3 on windowed scan statistics reached ARI 0.71 (95% CI 0.67–0.75), far above a map-level permutation null (99th percentile 0.017, p = 0.002). GMM chosen by BIC reached ARI 0.83.
- The centroid profiles (Figure 6) are interpretable: an open cluster with a high free-ray fraction, a narrow cluster with low side distances and high front/back distances, and a cluttered cluster with many range jumps.
- Because our generators produce distinct environment families by design, this shows that the features separate the families. It does not show discovery of naturally occurring structure.
- Silhouette preferred k=2 (open vs non-open), a reminder that unsupervised "environment types" depend on resolution.

**5. Vision & fusion: a camera resolves the ambiguity in our setting, but the margin is an upper bound.**
- On the same 60,000 segments and folds, a small CNN on simulated 32×32 camera crops reached macro-F1 0.948 (95% CI 0.939–0.957). Late fusion with the LiDAR Random Forest reached 0.959, against 0.637 for the LiDAR Random Forest alone.
- AC2 (CNN ≥ 0.80) and AC3 (fusion gain ≥ +0.02, CI lower bound > 0; observed +0.323, CI 0.313–0.334) both pass.
- Fusion adds only about +0.011 over the CNN alone: almost all the gain comes from appearance.
- The gap is expected. We designed the appearance model so that colour and a stripe almost fully disambiguate the classes: a hand-written three-rule colour heuristic with no learning already reaches 0.837, whereas 2D geometry cannot.
- The result shows that the camera pipeline, fusion and protocol work. It does not show that a real camera would deliver this margin.

## Model-suitability recommendations (proposal objective 6)

| Robotics need | Recommended approach | Evidence | Caveat |
|---|---|---|---|
| Real-time obstacle class from LiDAR | Random Forest on engineered segment geometry | Best macro-F1 (0.635); fast inference; interpretable features | Ceiling limited by 2D geometry; tracking over past scans adds +0.078 (0.713) |
| Obstacle class when a camera is available | CNN on camera crops, late-fused with LiDAR Random Forest | Fusion 0.959, CNN 0.948 vs LiDAR 0.637 on the same folds | Synthetic appearance designed by us: an upper bound, not a real-image result |
| Distance to nearest obstacle | Raw min-ultrasonic reading, plus a learned (RF) correction for blind configurations | Baseline MAE about 2 cm; RF −34% error on truly blind frames | Learned model alone is not better on ordinary frames |
| Recognising environment type | K-Means/GMM on windowed scan statistics | ARI 0.71–0.83 vs null 99th percentile 0.017 | Validated on synthetic families only |
| Point-to-point navigation, open or cluttered | PPO with a LiDAR safety shield | 96% / 77% success, 0% collisions (post-hoc) | Shield tuned after seeing PPO failures; plain PPO collides 4–31% |
| Navigation in corridors / safety-critical | Reactive rule or a planner; RL needs better reward/curriculum | Reactive collisions ≤ 2%; PPO 15% success in narrow maps | Reactive success is low |
| Interpretable, low-data control | Tabular Q-learning | Safe (≤ 2% collisions) | Coarse state, low success beyond open maps |

## Limitations
- **Simulation only for three of the four tasks.**
  - 2D geometry, idealised Gaussian noise and dropout.
  - No multipath, specular reflection or motion blur.
  - The sim-to-real gap was not quantified.
- **The UCI dataset** is the 4-sensor version (5,456 rows, licensed CC BY 4.0). Its labels come from a hand-coded wall-following rule, so classification is near-ceiling by construction (RF macro-F1 0.999). Inner tuning splits had no time gap, which can make the tuning slightly optimistic.
- **Segment classification** was evaluated on a uniform 60,000-row subsample of the 212,825 pure segments, with folds by map. The dataset counts of total and dropped segments include noise-induced 1–2-point fragments.
- **The "observable" flag in regression** means "the nearest point lies within an ultrasonic cone". Points between cones are usually still seen by a cone edge, so the meaningful blind stratum is the post-hoc truly-blind one (T3b).
- **AC5's absolute threshold** (≤ 0.10 m) is also met by the trivial baseline, so only the relative-improvement criterion carries evidence.
- **RL:**
  - 3 seeds and 300k steps; one held-out episode per map (50 maps per type).
  - Path ratio is measured against straight-line distance and only on successful episodes. It can be slightly below 1 because of the 0.2 m goal tolerance.
  - The reactive and random baselines are deterministic or single-seed.
- **Post-hoc analyses.** The safety shield and the tracking features were added after the main results. The shield reuses the held-out RL maps; tracking uses exact simulator poses and links that are always correct by construction. Both are reported as exploratory, not as acceptance-criterion passes. Narrow-map collisions that remain come from wall ends thinner than the 10° LiDAR ray spacing.
- **Synthetic camera appearance.** Class appearance (hue, saturation, height, stripe) was authored by us with little overlap. The vision scores are therefore an upper bound tied to our rendering choices and do not transfer to real imagery.
- **Cut for time** (future work):
  - real camera images and learned (early) fusion;
  - curriculum learning, DQN, and A* optimal-path references;
  - nested tuning beyond small grids;
  - multiple-comparison correction;
  - sim-validity tests against real ultrasonic statistics;
  - effective-sample-size analysis.
