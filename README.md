# RAAS-VO

Implementation of **Reference-anchored ambiguity sets for metric visual odometry in repetitive scenes.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run. CVXPY/CLARABEL runtime dependencies were not installed in this inspection environment.

## Data and preprocessing

Input NPZ arrays are `positions: (H,T,3)` and `rotations: (H,T,3,3)` for complete causal frontend histories; `anchors: (T,3)` with NaN at missing references; `velocity: (T,3)`; `visual_bound: (T,)`; `time`; and evaluation-only `truth_position,truth_rotation`. The monocular frontend must generate histories without test trajectory alignment. Coordinates use the fixed supplied metric registration. Optional `cal_residual,cal_velocity,cal_bound` transfer a separate bias/timing calibration domain. Sparse references can be exported from the permitted trajectory source or from a finite controlled ambiguity experiment.

```bash
python -m raas_vo prepare --input data/branches.npz --config config.json --output runs/full
```

## Run

```bash
python -m raas_vo run --config config.json --output runs/full
```

`config.json` records the method settings and seed `13`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m raas_vo evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `joint`, `independent`, `bias-only`, `inflated-radius`, `single`, `ignore-truncation`.

The run command evaluates these controls together.

## Outputs

All main outputs are written below `runs/full/`.

`predictions.csv` reports accepted/unresolved/inconsistent status, branch count, scale span, diameter, representative radius, lineage resets and charged solves. `metrics.csv` reports translation/rotation error, availability, joint/conditional false commitment, tracking loss, heuristic deletion and cost.

```bash
python -m raas_vo plot --config config.json --output runs/full
```
