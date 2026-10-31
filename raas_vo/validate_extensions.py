"""Image frontend, geometry, point estimates and a reduced finite replay."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from .frontend import generate
from .geometry import scale_information,separation
from .point_controls import fit
from .finite_suite import sequence
from .__main__ import prepare,run,evaluate,plot


def images():
    rng=np.random.default_rng(13);intrinsics=np.array([[240.,0,160.],[0,240.,120.],[0,0,1.]])
    points=rng.uniform([-2,-1.4,4],[2,1.4,10],(300,3));patches=rng.integers(0,2,(300,13,13),dtype=np.uint8)*220+20
    output=[]
    for camera in [0,.08,.16,.24]:
        image=np.full((240,320),30,dtype=np.uint8)
        for index in np.argsort(points[:,2])[::-1]:
            point=points[index]-np.array([camera,0,0]);pixel=intrinsics@point;left,top=np.rint(pixel[:2]/pixel[2]).astype(int)-6
            if left>=0 and top>=0 and left+13<320 and top+13<240:image[top:top+13,left:left+13]=patches[index]
        output.append(image)
    return output,intrinsics


def main():
    config=json.loads(Path("config.json").read_text(encoding="utf-8"))
    v=np.array([[0.,0,0],[1,0,0],[2,0,0]]);q=np.zeros_like(v)
    np.testing.assert_allclose(scale_information(v,q),2.,atol=1e-8)
    assert scale_information(np.ones_like(v),q)<1e-10
    fixed=dict(config,scale_min=1.,scale_max=1.,bias_bound=0.,delay_bound=0.)
    margin=separation(np.zeros(3),np.array([2.,0,0]),np.empty((0,3)),np.empty((0,3)),np.empty((0,3)),np.empty((0,3)),np.zeros(3),[],[.2,.2],fixed)
    np.testing.assert_allclose(margin,1.6,atol=1e-6)
    point_config=dict(config,bias_bound=0.,delay_bound=0.)
    scale,score,work=fit(v,2*v,q,np.full(3,.01),point_config,repeats=2)
    assert abs(scale-2)<1e-5 and score<1e-5 and work==2
    frames,intrinsics=images();reference=np.full((4,3),np.nan)
    output=generate(frames,intrinsics,reference,np.arange(4,dtype=float),capacity=2,window=8,seed=13)
    assert output["alive"].any(0).all() and np.isfinite(output["positions"][output["alive"]]).all()
    assert np.linalg.norm(output["positions"][0,-1])>0
    # The prefix of a longer video equals a separately processed short causal video.
    prefix=generate(frames[:3],intrinsics,reference[:3],np.arange(3,dtype=float),capacity=2,window=8,seed=13)
    np.testing.assert_allclose(prefix["positions"],output["positions"][:,:3],atol=1e-7,equal_nan=True)
    root=Path("runs/extensions");root.mkdir(parents=True,exist_ok=True);np.savez_compressed(root/"frontend.npz",**output)
    np.savez_compressed(root/"finite.npz",**sequence(13,1,6))
    prepare(config,root,root/"finite.npz");run(config,root);evaluate(config,root);plot(root)
    metrics=pd.read_csv(root/"metrics.csv")
    assert {"point-penalty","matched-work-point"} <= set(metrics.method)
    records=pd.read_csv(root/"predictions.csv")
    budget=records[records.method=="joint"].set_index("frame").work
    spent=records[records.method=="matched-work-point"].set_index("frame").work
    assert (spent<=budget).all()
    print(json.dumps({"scale_information":2.,"separation_margin":margin,"point_scale":scale,"actual_point_solves":work,"orb_triangulation_pnp_causal_prefix":True,"frontend_active_frames":output["alive"].sum(0).tolist(),"reduced_finite_methods":metrics.method.tolist()}))


if __name__=="__main__":main()
