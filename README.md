# Intelligent Obstacle Detection and Navigation Analysis for Autonomous Robots Using Machine Learning

Machine Learning for Robotics (Section 9A), instructor: Mr. Sudais

**Group:** Syeda Hamna Uraizee (22k-5112), Ayesha Wahab (22k-5087), Hamdan (22k-5077)

## Overview

This project studies how machine learning can support obstacle detection and navigation for a mobile robot. It combines:

- **A 2D simulator** with a 360-ray LiDAR (downsampled to 36 rays for frame features and RL), and 8 wide-cone ultrasonic sensors that have blind spots. Three map types are supported: open, narrow and cluttered.
- **Real data:** the UCI Wall-Following Robot Navigation dataset (4-sensor version, 5,456 rows; CC BY 4.0; Freire et al., 2009).

## Tasks

1. **Classification.** LiDAR object-segment classification (wall / obstacle / target), plus movement classification on the UCI data.
2. **Regression.** Nearest-obstacle distance estimated from ultrasonic readings only, reported separately for observable and blind-spot strata.
3. **Clustering.** Environment clustering, evaluated against a map-level permutation null.
4. **Reinforcement learning.** Navigation comparing a reactive baseline, tabular Q-learning and PPO.

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
python -m sim.generate     # ~1 min
python -m tasks.run_all    # ~30 min (UCI, segments, regression, clustering)
python -m tasks.regression_blind  # ~1 min, truly-blind regression stratum (T3b)
python -m rl.train         # ~20-40 min; env vars PPO_STEPS, Q_EPISODES
python -m rl.evaluate      # ~5 min
```

## Outputs

- `results/`: tables T1–T5 (CSV), figures F* (PNG), and `uci_*` files for the UCI data.
- `models/`: trained model files.

## Limitations

The sensors are simulated, and the environment is 2D. There is no camera or vision input. The UCI labels are near-deterministic, so they make classification easier than real-world sensing would be.
