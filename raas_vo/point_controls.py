"""Residual point estimates with explicit work-matched repeat solves."""
import cvxpy as cp
import numpy as np


def fit(v,z,q,bounds,config,repeats=1):
    scale=cp.Variable();bias=cp.Variable(3);delay=cp.Variable()
    constraints=[scale>=config["scale_min"],scale<=config["scale_max"],cp.abs(bias)<=config["bias_bound"],cp.abs(delay)<=config["delay_bound"]]
    if not len(v):return (config["scale_min"]+config["scale_max"])/2,0.,0
    residual=cp.vstack([(zk-np.asarray(config["origin"])-scale*vk-bias-qk*delay)/max(float(epsilon),1e-8) for vk,zk,qk,epsilon in zip(v,z,q,bounds)])
    objective=cp.sum_squares(residual)/len(v)+config.get("point_penalty",.01)*(cp.sum_squares(bias)+cp.square(delay))
    work=0
    for _ in range(repeats):
        problem=cp.Problem(cp.Minimize(objective),constraints);problem.solve(solver="CLARABEL",warm_start=bool(work));work+=1
        if problem.status!="optimal":return None,None,work
    return float(scale.value),float(problem.value),work
