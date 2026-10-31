"""Small finite replay of nuisance feasibility, truncation and reporting."""
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from .__main__ import domain, prepare, run, evaluate, plot


def main():
    config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    config.update(bias_bound=0., delay_bound=0., reference_bound=1e-5, velocity_error=0., acceleration_bound=0., solver_padding=1e-6, capacity=1)
    visual = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    anchors = visual*2
    velocity = np.zeros_like(visual)
    interval, status, solves = domain(visual, anchors, velocity, np.full(3, 1e-5), config, "joint")
    assert status == "empirical" and solves == 2
    assert interval[0] <= 2 <= interval[1] and interval[1]-interval[0] < 1e-3
    impossible = anchors.copy()
    impossible[0, 1] = 1
    assert domain(visual, impossible, velocity, np.full(3, 1e-5), config, "joint")[0] is None
    root = Path("runs/check")
    root.mkdir(parents=True, exist_ok=True)
    data = dict(positions=np.stack([visual, visual]), rotations=np.tile(np.eye(3), (2, 3, 1, 1)), anchors=anchors, velocity=velocity, visual_bound=np.zeros(3), time=np.arange(3, dtype=float), truth_position=anchors, truth_rotation=np.tile(np.eye(3), (3, 1, 1)))
    np.savez_compressed(root/"input.npz", **data)
    prepare(config, root, root/"input.npz")
    run(config, root)
    first = pd.read_csv(root/"predictions.csv")
    joint = first[first.method == "joint"]
    ignored = first[first.method == "ignore-truncation"]
    assert not joint.accepted.any()
    assert bool(ignored.accepted.iloc[-1])
    evaluate(config, root)
    values = pd.read_csv(root/"metrics.csv").set_index("method")
    assert values.loc["joint", "availability"] == 0
    assert values.loc["joint", "joint_false_commit"] == 0
    assert values.loc["joint", "set_coverage"] == 1
    assert values.loc["ignore-truncation", "translation_rmse"] < 1e-3
    plot(root)
    # Runtime operates without reference trajectory labels; no oracle enters acceptance.
    unlabeled = {key: value for key, value in data.items() if not key.startswith("truth_")}
    np.savez_compressed(root/"unlabeled.npz", **unlabeled)
    prepare(config, root, root/"unlabeled.npz")
    run(config, root)
    second = pd.read_csv(root/"predictions.csv")
    for key in ["accepted", "branches", "radius", "diameter", "scale_span", "position_x", "position_y", "position_z"]:
        np.testing.assert_allclose(first[key], second[key], atol=1e-8, equal_nan=True)
    # Restore the labeled replay so metrics can be inspected and regenerated.
    prepare(config, root, root/"input.npz")
    run(config, root)
    evaluate(config, root)
    print(json.dumps({"cvxpy": cp.__version__, "solvers": cp.installed_solvers(), "scale_interval": interval, "known_scale_and_infeasibility": "passed", "truncation_and_hand_rates": "passed", "label_free_causal_output": "passed", "prepare_run_evaluate_plot": "passed"}))


if __name__ == "__main__":
    main()
