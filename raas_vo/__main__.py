import argparse
import json
import time
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def domain(v, z, q, epsilon, c, mode, calibration=None):
    scale = cp.Variable()
    bias = cp.Variable(3)
    delay = cp.Variable()
    constraints = [scale >= c['scale_min'], scale <= c['scale_max']]
    constraints += [cp.abs(bias) <= c['bias_bound'], cp.abs(delay) <= c['delay_bound']]
    if mode == 'independent':
        constraints += [bias == 0, delay == 0]
    elif mode == 'bias-only':
        constraints += [delay == 0]
    for vk, zk, qk, ek in zip(v, z, q, epsilon):
        constraints += [cp.norm(zk - c['origin'] - scale * vk - bias - qk * delay) <= ek]
    if calibration is not None:
        for residual, velocity, bound in zip(calibration['residual'], calibration['velocity'], calibration['bound']):
            constraints += [cp.norm(residual - bias - velocity * delay) <= bound]
    limits = []
    solves = 0
    for sign in [1, -1]:
        problem = cp.Problem(cp.Minimize(sign * scale), constraints)
        try:
            problem.solve(solver='CLARABEL')
        except cp.error.SolverError:
            return [c['scale_min'], c['scale_max']], 'unresolved', solves + 1
        solves += 1
        if problem.status == 'infeasible':
            return None, 'infeasible', solves
        if problem.status != 'optimal':
            return [c['scale_min'], c['scale_max']], 'unresolved', solves
        limits.append(float(scale.value))
    return [max(c['scale_min'], limits[0]-c['solver_padding']), min(c['scale_max'], limits[1]+c['solver_padding'])], 'empirical', solves


def prepare(c, out, input_path):
    raw = np.load(input_path, allow_pickle=False)
    required = {'positions', 'rotations', 'anchors', 'velocity', 'visual_bound', 'time'}
    if not required <= set(raw.files):
        raise ValueError('Input requires ' + ', '.join(sorted(required)))
    h, t, d = raw['positions'].shape
    if d != 3 or raw['rotations'].shape != (h, t, 3, 3):
        raise ValueError('positions must be H,T,3 and rotations H,T,3,3')
    if raw['anchors'].shape != (t, 3) or raw['velocity'].shape != (t, 3) or raw['visual_bound'].shape != (t,) or raw['time'].shape != (t,):
        raise ValueError('anchors/velocity must be T,3; visual_bound/time must be T')
    if not np.all(np.diff(raw['time']) > 0) or np.any(raw['visual_bound'] < 0):
        raise ValueError('Require increasing timestamps and nonnegative visual bounds')
    np.savez_compressed(out / 'sequence.npz', **{key: raw[key] for key in raw.files})


def run(c, out):
    raw = np.load(out / 'sequence.npz', allow_pickle=False)
    paths = raw['positions']
    rotations = raw['rotations']
    records = []
    calibration = None
    if {'cal_residual', 'cal_velocity', 'cal_bound'} <= set(raw.files):
        calibration = {key: raw['cal_' + key] for key in ['residual', 'velocity', 'bound']}
    for method in ['joint', 'independent', 'bias-only', 'inflated-radius', 'single', 'ignore-truncation', 'point-penalty', 'matched-work-point']:
        active = list(range(paths.shape[0]))
        effective_capacity = 1 if method == 'single' else c['capacity']
        truncated = len(active) > effective_capacity
        active = active[:effective_capacity]
        previous = None
        previous_lineage = None
        score_cache = {}
        for t in range(paths.shape[1]):
            start = time.perf_counter()
            dynamic = "anchor_positions" in raw.files
            if dynamic:
                active = np.flatnonzero(raw["alive"][:,t]).tolist()
                truncated |= bool(raw["frontend_truncated"][t])
                if len(active)>effective_capacity:truncated=True;active=active[:effective_capacity]
            point_budget = None
            if method == "matched-work-point":
                point_budget=next(record["work"] for record in records if record["method"]=="joint" and record["frame"]==t)
                active=sorted(active,key=lambda h:score_cache.get(h,float("inf")))[:point_budget]
            remaining_budget=point_budget
            anchors = np.flatnonzero(np.isfinite(raw['anchors'][:t + 1]).all(1))
            limits = []
            point_scores = []
            kept = []
            unresolved = False
            charged_solves = 0
            mode = 'independent' if method == 'inflated-radius' else method
            mode = mode if mode in ['independent', 'bias-only'] else 'joint'
            for ordinal,h in enumerate(active):
                error_bound = c['reference_bound'] + raw['visual_bound'][anchors]
                error_bound = error_bound + c['velocity_error'] * c['delay_bound'] + .5 * c['acceleration_bound'] * c['delay_bound'] ** 2
                if method == 'inflated-radius':
                    error_bound = error_bound + np.sqrt(3) * c['bias_bound'] + c['speed_bound'] * c['delay_bound']
                visual_centers = raw['anchor_positions'][h,t,:len(anchors)] if dynamic else paths[h,anchors]
                if method in {'point-penalty','matched-work-point'}:
                    from .point_controls import fit
                    repeats=min(2,max(1,remaining_budget-(len(active)-ordinal-1))) if method=='matched-work-point' else 1
                    value,point_score,solves=fit(visual_centers,raw['anchors'][anchors],raw['velocity'][anchors],error_bound,c,repeats)
                    if remaining_budget is not None:remaining_budget-=solves
                    score_cache[h]=point_score if point_score is not None else float('inf')
                    scale_range=[value,value] if value is not None else [c['scale_min'],c['scale_max']]
                    status='empirical' if value is not None else 'unresolved'
                    point_scores.append(point_score if point_score is not None else float('inf'))
                else:
                    scale_range, status, solves = domain(visual_centers, raw['anchors'][anchors],
                                             raw['velocity'][anchors], error_bound, c, mode, calibration)
                charged_solves += solves
                if scale_range is not None:
                    kept.append(h)
                    limits.append(scale_range)
                    unresolved |= status == 'unresolved'
            active = kept
            if not active:
                records.append(dict(method=method, frame=t, time=raw['time'][t], accepted=False, status='inconsistent',
                                    branches=0, translation_error=np.nan, rotation_error=np.nan,
                                    radius=np.nan, diameter=np.nan, scale_span=np.nan, reset=False, covered=False, **{key: np.nan for key in ['position_x','position_y','position_z']+['rotation_'+str(i)+str(j) for i in range(3) for j in range(3)]},
                                    truncated=truncated, work=charged_solves, latency_ms=1000 * (time.perf_counter() - start)))
                continue
            same = [h for h in active if dynamic and raw["lineage"][h,t] == previous_lineage]
            selected = same[0] if same else (previous if not dynamic and previous in active else active[0])
            if method in {"point-penalty","matched-work-point"}:selected=active[int(np.argmin(point_scores))]
            current_lineage = int(raw["lineage"][selected,t]) if dynamic else selected
            idx = active.index(selected)
            point = np.asarray(c['origin']) + np.mean(limits[idx]) * paths[selected, t]
            endpoints = np.array([np.asarray(c['origin']) + s * paths[h, t] for h, interval in zip(active, limits) for s in interval])
            radius = np.linalg.norm(endpoints - point, axis=1).max() + raw['visual_bound'][t]
            diameter = np.linalg.norm(endpoints[:, None] - endpoints[None, :], axis=2).max() + 2 * raw['visual_bound'][t]
            angular = [Rotation.from_matrix(rotations[selected, t].T @ rotations[h, t]).magnitude() for h in active]
            rotation_radius = np.degrees(max(angular)) + c['rotation_bound_deg']
            accepted = radius <= c['position_tolerance'] and rotation_radius <= c['rotation_tolerance_deg'] and not unresolved
            if method in {"point-penalty","matched-work-point"}:
                accepted=not unresolved and min(point_scores)<=c.get("point_acceptance",1.)
            else:
                accepted &= not truncated or method == 'ignore-truncation'
            position_fields = dict(zip(['position_x','position_y','position_z'], point))
            rotation_fields = {'rotation_'+str(i)+str(j): rotations[selected,t,i,j] for i in range(3) for j in range(3)}
            covered = np.nan
            if 'truth_position' in raw.files:
                covered = False
                for h, interval in zip(active, limits):
                    vector = paths[h,t]
                    scalar = np.dot(raw['truth_position'][t]-c['origin'],vector)/max(np.dot(vector,vector),1e-12)
                    projected = np.asarray(c['origin'])+np.clip(scalar,*interval)*vector
                    covered |= np.linalg.norm(projected-raw['truth_position'][t]) <= raw['visual_bound'][t]+1e-7
            records.append(dict(method=method, frame=t, time=raw['time'][t], accepted=bool(accepted), status='unresolved' if unresolved else 'empirical',
                                branches=len(active), **position_fields, **rotation_fields, covered=covered,
                                radius=radius, diameter=diameter, scale_span=max(v[1] for v in limits) - min(v[0] for v in limits),
                                reset=previous_lineage is not None and current_lineage != previous_lineage, truncated=truncated,
                                work=charged_solves, latency_ms=1000 * (time.perf_counter() - start)))
            previous = selected
            previous_lineage = current_lineage
    pd.DataFrame(records).to_csv(out / 'predictions.csv', index=False)


def evaluate(c, out):
    data = pd.read_csv(out / 'predictions.csv')
    raw = np.load(out / 'sequence.npz', allow_pickle=False)
    if not {'truth_position','truth_rotation'} <= set(raw.files):
        raise ValueError('Evaluation requires truth_position and truth_rotation; run itself is label-free')
    rows = []
    for method, g in data.groupby('method'):
        indices = g.frame.to_numpy(dtype=int)
        points = g[['position_x','position_y','position_z']].to_numpy()
        orientations = g[['rotation_'+str(i)+str(j) for i in range(3) for j in range(3)]].to_numpy().reshape(-1,3,3)
        g = g.copy()
        g['translation_error'] = np.linalg.norm(points-raw['truth_position'][indices],axis=-1)
        g['rotation_error'] = [np.degrees(Rotation.from_matrix(r.T@reference).magnitude()) if np.isfinite(r).all() else np.nan for r,reference in zip(orientations,raw['truth_rotation'][indices])]
        wrong = (g.translation_error > c['position_tolerance']) | (g.rotation_error > c['rotation_tolerance_deg'])
        accepted = g[g.accepted]
        rows.append(dict(method=method, translation_rmse=np.sqrt((g.translation_error ** 2).mean()),
                         rotation_mean_deg=g.rotation_error.mean(), invalid_output=float((~np.isfinite(g.translation_error)).mean()), set_coverage=float(g.covered.fillna(False).mean()), availability=g.accepted.mean(),
                         joint_false_commit=float((g.accepted & wrong).mean()),
                         conditional_false_commit=float(wrong[g.accepted].mean()) if len(accepted) else np.nan,
                         tracking_loss=float((g.branches == 0).mean()), branches=g.branches.mean(),
                         radius=g.radius.mean(), diameter=g.diameter.mean(), scale_span=g.scale_span.mean(),
                         heuristic_deletion=g.truncated.mean(), lineage_resets=g.reset.sum(), work=g.work.sum(),
                         latency_ms=g.latency_ms.mean()))
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'predictions.csv')
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, g in d.groupby('method'):
        ax.plot(g.time, g.radius, label=method)
    ax.set(xlabel='Time', ylabel='Representative positional radius (m)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'ambiguity.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'run', 'evaluate', 'plot'])
    p.add_argument('--input')
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'prepare':
        if not args.input:
            p.error('--input is required')
        prepare(c, out, args.input)
    elif args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
