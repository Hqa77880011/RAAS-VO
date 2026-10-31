"""Scale excitation after nuisance projection and support-separation margins."""
import argparse
import json
from pathlib import Path
import cvxpy as cp
import numpy as np
import pandas as pd


def scale_information(v,q,weights=None):
    v=np.asarray(v).reshape(-1);q=np.asarray(q)
    nuisance=np.concatenate([np.tile(np.eye(3),(len(q),1)),q.reshape(-1,1)],axis=1)
    weight=np.repeat(np.ones(len(q)) if weights is None else weights,3)**.5
    vv=weight*v;aa=weight[:,None]*nuisance
    residual=vv-aa@np.linalg.pinv(aa)@vv
    return float(residual@residual)


def separation(first,second,past_first,past_second,observations,past_velocity,current_velocity,bounds,current_bounds,config):
    """Distance between two current-reference supports constrained by all past anchors."""
    supports=[];constraints=[]
    for vector,history in [(first,past_first),(second,past_second)]:
        scale=cp.Variable();bias=cp.Variable(3);delay=cp.Variable()
        constraints += [scale>=config["scale_min"],scale<=config["scale_max"],cp.abs(bias)<=config["bias_bound"],cp.abs(delay)<=config["delay_bound"]]
        for v,z,q,epsilon in zip(history,observations,past_velocity,bounds):constraints.append(cp.norm(z-np.asarray(config["origin"])-scale*v-bias-q*delay)<=epsilon)
        supports.append(scale*vector+bias+current_velocity*delay)
    problem=cp.Problem(cp.Minimize(cp.norm(supports[0]-supports[1])),constraints);problem.solve(solver="CLARABEL")
    if problem.status!="optimal":return None
    return float(problem.value-sum(current_bounds))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--input",required=True);parser.add_argument("--config",default="config.json");parser.add_argument("--output",required=True);args=parser.parse_args()
    data=np.load(args.input,allow_pickle=False);config=json.loads(Path(args.config).read_text(encoding="utf-8"));rows=[]
    frames=np.flatnonzero(np.isfinite(data["anchors"]).all(1))
    for count,frame in enumerate(frames):
        past=frames[:count]
        remainder=config["velocity_error"]*config["delay_bound"]+.5*config["acceleration_bound"]*config["delay_bound"]**2
        bounds=config["reference_bound"]+data["visual_bound"][past]+remainder
        dynamic="anchor_positions" in data.files
        active=np.flatnonzero(data["alive"][:,frame]) if dynamic else np.arange(len(data["positions"]))
        for index,first in enumerate(active):
            first_history=data["anchor_positions"][first,frame,:count+1] if dynamic else data["positions"][first,frames[:count+1]]
            info=scale_information(first_history,data["velocity"][frames[:count+1]])
            for second in active[index+1:]:
                second_history=data["anchor_positions"][second,frame,:count] if dynamic else data["positions"][second,past]
                margin=separation(data["positions"][first,frame],data["positions"][second,frame],first_history[:count],second_history,data["anchors"][past],data["velocity"][past],data["velocity"][frame],bounds,[config["reference_bound"]+data["visual_bound"][frame]+remainder]*2,config)
                rows.append(dict(frame=int(frame),first=first,second=second,scale_information=info,separation_margin=margin))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(output,index=False)


if __name__=="__main__":main()
