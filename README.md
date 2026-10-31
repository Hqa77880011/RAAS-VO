# RAAS-VO

Implementation of **Reference-Anchored Ambiguity Sets for Metric Visual Odometry in Repetitive Scenes**. This repository implements the numerical scale/bias/delay compatibility and qualified reporting layer on supplied causal trajectory histories. It is separate from the manuscript's experiment records. It does not run an image frontend or reproduce the published TUM/KITTI scores.

## Install

Python >=3.10: `python -m pip install -e .`. CVXPY and CLARABEL are required at runtime. A CPU check was performed in a project-local virtual environment using Python3.13.5, CVXPY1.9.3 and CLARABEL0.11.1; the global environment was not changed.

## Inputs and acquisition contract

Numerical NPZ arrays:

- `positions[H,T,3]`, `rotations[H,T,3,3]`: complete causal frontend histories in the supplied initial metric-frame registration, before scale recovery.
- `anchors[T,3]`: permitted sparse reference measurements, NaN at unavailable times; `velocity[T,3]`: causal nominal velocity in the same metric frame.
- `visual_bound[T]`, `time[T]`: nonnegative uniform visual enclosure radius and strictly increasing timestamps.
- Optional separate-calibration `cal_residual[J,3]`, `cal_velocity[J,3]`, `cal_bound[J]`: nuisance-domain constraints frozen before testing.
- Evaluation only: `truth_position[T,3]`, `truth_rotation[T,3,3]`.

Acquire TUM RGB-D/KITTI images and permitted references from their dataset providers and run the declared monocular frontend externally. Depth/stereo/inertial exposure is not silently added. Supplied histories must not use full-test alignment. Initial registration is fixed equally for all controls. The present API expects positions/orientations already rotated by the declared initial rotation; `origin` supplies initial translation. The same nuisance bias/delay explains every causal anchor of a branch. Runtime does not require reference trajectory labels or use them for candidate selection.

```bash
python -m raas_vo prepare --input data/branches.npz --config config.json --output runs/full
python -m raas_vo run --config config.json --output runs/full
python -m raas_vo evaluate --config config.json --output runs/full
python -m raas_vo plot --config config.json --output runs/full
```

`evaluate` requires truth arrays; `run` accepts an unlabeled file. Configuration records scale prior, nuisance bounds, observation/remainder bounds, capacity, acceptance tolerances, solver padding and seed. Reference/visual bounds must be supplied from the permitted calibration protocol, not chosen using test errors. Defaults implement the finite diagnostic tolerance of .25m/2deg and the stated .02m coordinate-jitter Euclidean bound, with explicit velocity/acceleration terms.

## Mapping and controls

| Paper item | Entry | Command/configuration |
| --- | --- | --- |
| Common bias/time-offset SOC domain and min/max scale | `raas_vo/__main__.py:domain` | `run`; CLARABEL |
| Separate calibration-domain transfer | `domain` | optional cal_* arrays |
| Causal branch retention, sticky truncation, lineage continuity | `run` | `capacity`; six controls below |
| Supported representative, positional radius/diameter and orientation gate | `run` | position/rotation tolerances |
| Translation RMSE, mean rotation error, availability, joint/conditional false commitment | `evaluate` | fixed reference frame, all reporting opportunities retained |
| Positional set coverage, invalid-output/tracking-loss rates, branch/scale/radius/deletion/work diagnostics | `evaluate` | `metrics.csv` |
| Saved radius timeline | `plot` | `ambiguity.png` |
| Paired bootstrap/Wilcoxon-Holm and effect summaries | `raas_vo/compare.py` | command below |

`run` evaluates `joint`, `independent` (zero bias/delay), `bias-only`, `inflated-radius`, `single` and `ignore-truncation`. All receive the same supplied histories. Resource deletion remains flagged for the entire sequence. A later agreeing anchor does not restore omitted support. Representative selection follows the previous surviving lineage or the first active branch, without oracle branch choice.

Numerical endpoint padding is clipped to the prior interval. Nonoptimal or failed solver runs retain the full prior scale range and suppress acceptance. Numerical infeasibility removes that supplied branch; successful outputs remain **empirical**, not certified outer bounds. The code's fixed-center SOCP cannot certify a nonlinear image frontend's enclosure assumptions.

## Outputs and metric conventions

`sequence.npz` and `settings.json` store prepared input/configuration. `predictions.csv` records causal representative position/orientation, acceptance status, branch count, scale span, diameter/radius, sticky truncation, resets, charged endpoint solves and host latency. If truth is provided, positional union containment is logged as an evaluator diagnostic. `metrics.csv` reports available-output error and all-opportunity availability/false-commitment/tracking-loss/invalid-output rates. Conditional false commitment is NaN when nothing is accepted. Positional set coverage checks whether the truth lies in any retained branch's line-segment-plus-visual-radius support; it is not joint calibration-domain coverage or a formal retention guarantee. Rotation error is assessed separately. RMSE/mean error exclude unavailable numeric poses; `invalid_output` reports their proportion so those samples remain visible.

```bash
python -m raas_vo.compare --inputs 'runs/*/metrics.csv' --metrics availability joint_false_commit translation_rmse --unit seed --reference joint --bootstrap 10000 --seed 13 --output runs/comparison.csv
```

Use independently produced sequence/run records and appropriate `--unit` columns for real paired statistics. A single synthetic check cannot establish uncertainty intervals.

## Verification and reproduction boundary

```bash
python -m raas_vo.validate
```

The small finite replay checks a known scale-two interval and inconsistent reference constraints, sticky capacity truncation versus ignore-truncation acceptance, hand-computed coverage/availability/false-commitment rates, label-free causal outputs, file-backed preparation/run/evaluation and plotting. No training is applicable to this optimization-only layer.

Remaining scope includes causal frontend/history generation, finite diagnostic mask/event suites, geometry excitation/separation diagnostics, residual-ranked/point-penalty/matched-work controls, full TUM/KITTI manifests and runs, and real sequence-level statistical comparisons. Host solve latency and operation counts are not Raspberry Pi/GPU energy or full-frontend latency measurements. No reported paper number is inferred from the functional replay.
