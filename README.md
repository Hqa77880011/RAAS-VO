# RAAS-VO

Implementation of **Reference-anchored ambiguity sets for metric visual odometry in repetitive scenes.**

## Install

Use Python 3.11 in an isolated environment, then run from the repository root:

```bash
python -m pip install -e .
```

## Data

Input NPZ arrays are `positions: (H,T,3)` and `rotations: (H,T,3,3)` for complete causal frontend histories; `anchors: (T,3)` with NaN at missing references; `velocity: (T,3)`; `visual_bound: (T,)`; `time`; and evaluation-only `truth_position,truth_rotation`. The monocular frontend must generate histories without test trajectory alignment. Coordinates use the fixed supplied metric registration. Optional `cal_residual,cal_velocity,cal_bound` transfer a separate bias/timing calibration domain. Sparse references can be exported from the permitted trajectory source or from a finite controlled ambiguity experiment.

## Workflow

```bash
python -m raas_vo prepare --input data/branches.npz
python -m raas_vo run
python -m raas_vo evaluate
python -m raas_vo plot
```

Main commands accept `--config config.json` and `--output runs/default`. Keep each seed/condition in its own output directory. The resolved configuration is written to `settings.json`.

## Configuration

`scale_min,scale_max` define metric scale. `origin` is the fixed translation registration. `bias_bound,delay_bound` define shared reference nuisance bounds. `reference_bound,velocity_error,acceleration_bound` bound observation/time-model error. `capacity` is the retained-history budget. `position_tolerance,rotation_tolerance_deg` qualify the actual representative. `solver_padding` widens numerical scale endpoints. Training epochs, batch size, learning rate and seed use the defaults shown in `config.json` where applicable.

## Outputs

`predictions.csv` reports accepted/unresolved/inconsistent status, branch count, scale span, diameter, representative radius, lineage resets and charged solves. `metrics.csv` reports translation/rotation error, availability, joint/conditional false commitment, tracking loss, heuristic deletion and cost.

## Implementation and controls

CLARABEL solves the joint convex scale/shared-bias/shared-delay domains. A feasible witness alone is not used as an exclusion proof. Solver-uncertain branches remain unresolved. Branch capacity deletion persists as truncation and prevents qualified output unless explicitly ignored by a diagnostic control. Position radius checks both scale endpoints from every retained history; orientation spread is checked separately. Frontend trajectories and enclosures are explicit inputs rather than inferred from ground truth.

Controls: `joint`, `independent`, `bias-only`, `inflated-radius`, `single`, `ignore-truncation`. The run command evaluates the listed controls together.

```bash
python -m raas_vo.compare --inputs "runs/*/metrics.csv" --reference joint --metrics translation_rmse availability joint_false_commit --unit seed
```

The comparison command averages repeated rows within the specified independent unit, bootstraps units and applies paired Wilcoxon/Holm comparisons per metric.

