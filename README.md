# Intelligent Obstacle Detection and Navigation Analysis for Autonomous Robots Using Machine Learning

Machine Learning for Robotics (Section 9A), instructor: Mr. Sudais

**Group:** Syeda Hamna Uraizee (22k-5112), Ayesha Wahab (22k-5087), Hamdan (22k-5077)

## Overview

This project studies how machine learning can support obstacle detection and navigation for a mobile robot. It combines:

- **A 2D simulator** with a 360-ray LiDAR (downsampled to 36 rays for frame features and RL), 8 wide-cone ultrasonic sensors that have blind spots, and a simulated pan camera aimed at LiDAR detections. Three map types are supported: open, narrow and cluttered.
- **Real data:** the UCI Wall-Following Robot Navigation dataset (4-sensor version, 5,456 rows; CC BY 4.0; Freire et al., 2009).

## Tasks

1. **Classification.** LiDAR object-segment classification (wall / obstacle / target), plus movement classification on the UCI data.
   - Exploratory: temporal tracking — each segment is linked causally to the matching segment in the previous scan; features aggregated over up to 5 past views raise Random Forest macro-F1 from 0.635 to 0.713 (`tasks/temporal.py`, run inside `tasks.run_all`).
2. **Regression.** Nearest-obstacle distance estimated from ultrasonic readings only, reported separately for observable and blind-spot strata.
3. **Clustering.** Environment clustering, evaluated against a map-level permutation null.
4. **Reinforcement learning.** Navigation comparing a reactive baseline, tabular Q-learning and PPO.
   - Post-hoc safety shield: a LiDAR veto on PPO's forward action (agent `ppo_shield` in `rl.evaluate`) removes collisions in open and cluttered maps.
5. **Vision & fusion.** A simulated pan camera renders a 32×32 RGB crop of each LiDAR segment; colour-rule, colour-histogram LogReg and a small CNN classify wall/obstacle/target; late fusion with the LiDAR Random Forest (same 60k segments, GroupKFold by map).

## Evaluation Protocol

- Cross-validation uses `GroupKFold` grouped by `map_seed`, and only out-of-fold predictions are reported.
- All preprocessing sits inside scikit-learn `Pipeline`s, so nothing is fit on held-out data.
- The primary model is specified in advance: Random Forest.
- Seed ranges are disjoint: supervised 100000–100299, RL train 0–9999, RL eval 200000–200149.

## Layout

```
sim/       simulator and data generation
tasks/     supervised, regression and clustering tasks
rl/        Q-learning, PPO training and evaluation
data/      generated and UCI data
models/    saved models
results/   CSV tables and PNG figures
reviews/   review notes
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run from the repository root, in this order:

```bash
python -m sim.generate     # ~0.5 min (measured 32 s)
python -m tasks.run_all    # ~34 min (UCI, segments, temporal tracking, regression, clustering)
python -m tasks.regression_blind  # ~1 min, truly-blind regression stratum (T3b)
python -m tasks.vision           # ~7 min, camera crops + CNN + LiDAR-camera fusion (T6)
python -m rl.train         # ~20-40 min; env vars PPO_STEPS, Q_EPISODES
python -m rl.evaluate      # ~2.5 min (uses the committed models/)
```

## Outputs

- `results/`: tables T1–T6 (CSV) including the vision table (T6) and acceptance results, the temporal tracking table (T2c, with acceptance and by-track-length results), and T5, which now includes the `ppo_shield` agent; figures F* (PNG) including the vision confusion matrices (F8), example crops (F9), and the gain by track length (F10); and `uci_*` files for the UCI data.
- `models/`: trained model files.
- `data/crops.npz`: cached camera crops. This is a regenerable cache and is not tracked in git.

## Limitations

The sensors are simulated, and the environment is 2D. Camera appearance is synthetic and authored by us (colour/height/stripe), so vision scores are an upper bound that does not transfer to real images. The UCI labels are near-deterministic, so they make classification easier than real-world sensing would be.

## Reproducibility check
On 2026-11-08 the full pipeline (`sim.generate` → `tasks.run_all` → `tasks.regression_blind` → `tasks.vision` → `rl.evaluate`, about 44 min on the session's cloud machine) was rerun from a fresh clone. All 19 result CSVs matched the committed ones exactly, except the wall-clock `seconds` column of `T2c_temporal.csv`. `rl.train` was not rerun; evaluation uses the committed models.
