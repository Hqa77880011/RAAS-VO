"""Reproducible finite corridor diagnostics, independent of manuscript results."""
import argparse
import json
from pathlib import Path
import numpy as np
from .__main__ import prepare,run,evaluate,plot


def sequence(seed,scenario,opportunities=50):
    rng=np.random.default_rng(seed);time=np.arange(opportunities)*.1
    speed=.2+.6*np.sin(time*.7)**2;speed[:1]=0
    truth=np.c_[np.cumsum(speed)*.1,.08*np.sin(time),np.zeros(opportunities)]
    if scenario%5==0:truth*=.02
    velocity=np.zeros_like(truth);velocity[1:]=np.diff(truth,axis=0)/.1
    scale=2.;paths=[]
    for branch in range(8):
        alias=truth/scale
        if branch:alias=alias+np.c_[np.maximum(time-.3,0)*0+np.where(time>.3,(branch-4)*.3,0),np.zeros(opportunities),np.zeros(opportunities)]
        if scenario%5==4:alias=alias*(1+.1*time[:,None])
        paths.append(alias)
    anchors=np.full_like(truth,np.nan);frames=np.unique(np.rint(np.linspace(0,opportunities-1,4)).astype(int))
    delay=.08 if scenario%5==3 else .04;bias=np.array([.08,0,0])
    anchors[frames]=truth[frames]+bias+velocity[frames]*delay+rng.uniform(-.02,.02,(len(frames),3))
    rotations=np.tile(np.eye(3),(8,opportunities,1,1))
    return dict(positions=np.array(paths),rotations=rotations,anchors=anchors,velocity=velocity,visual_bound=np.full(opportunities,.02),time=time,truth_position=truth,truth_rotation=np.tile(np.eye(3),(opportunities,1,1)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);parser.add_argument("--config",default="config.json");parser.add_argument("--seed",type=int,default=13);parser.add_argument("--scenarios",type=int,default=20);parser.add_argument("--opportunities",type=int,default=50);parser.add_argument("--run",action="store_true");args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding="utf-8"));root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    for index in range(args.scenarios):
        output=root/f"scenario_{index:02d}";output.mkdir(exist_ok=True);np.savez_compressed(output/"input.npz",**sequence(args.seed+index,index,args.opportunities))
        (output/"settings.json").write_text(json.dumps(dict(config,seed=args.seed+index,scenario=index,synthetic_functional_suite=True),indent=2),encoding="utf-8")
        if args.run:prepare(config,output,output/"input.npz");run(config,output);evaluate(config,output);plot(output)


if __name__=="__main__":main()
