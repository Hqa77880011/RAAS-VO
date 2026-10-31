"""Causal monocular ORB/triangulation/PnP beam frontend in a fixed initial gauge."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def triangulate(first,second,pose_a,pose_b,intrinsics):
    projection_a=intrinsics@pose_a;projection_b=intrinsics@pose_b
    homogeneous=cv2.triangulatePoints(projection_a,projection_b,first.T.astype(float),second.T.astype(float))
    xyz=(homogeneous[:3]/homogeneous[3]).T
    depth_a=(pose_a[:,:3]@xyz.T+pose_a[:,3:]).T[:,2]
    depth_b=(pose_b[:,:3]@xyz.T+pose_b[:,3:]).T[:,2]
    valid=np.isfinite(xyz).all(1)&(depth_a>.01)&(depth_b>.01)&(np.linalg.norm(xyz,axis=1)<1000)
    return xyz,valid


def matches(descriptors,current,alternative=False):
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors,current,k=2)
    result=[]
    for pair in pairs:
        if len(pair)<2:continue
        first,second=pair
        chosen=second if alternative and second.distance<1.3*max(first.distance,1) else first
        if chosen.distance<64:result.append(chosen)
    return result


def generate(images,intrinsics,references,timestamps,capacity=8,window=8,seed=13,visual_radius=.05):
    """Each output column is the beam available at that time, never a future beam."""
    cv2.setNumThreads(1);cv2.setRNGSeed(seed)
    orb=cv2.ORB_create(nfeatures=2000)
    features=[orb.detectAndCompute(image,None) for image in images]
    if any(descriptor is None for _,descriptor in features):raise ValueError("Every frame needs observable ORB features")
    count=len(images);anchor_frames=np.flatnonzero(np.isfinite(references).all(1));slots=capacity
    positions=np.full((slots,count,3),np.nan);rotations=np.tile(np.eye(3),(slots,count,1,1));alive=np.zeros((slots,count),dtype=bool)
    anchored=np.full((slots,count,len(anchor_frames),3),np.nan);lineage=np.full((slots,count),-1,dtype=int);truncated=np.zeros(count,dtype=bool)
    positions[0,0]=0;alive[0,0]=True;lineage[0,0]=0
    for index,frame in enumerate(anchor_frames):
        if frame==0:anchored[0,0,index]=0
    pose0=np.c_[np.eye(3),np.zeros(3)];beam=[];next_id=1;ever_truncated=False
    for alternative in [False,True]:
        correspondence=matches(features[0][1],features[1][1],alternative)
        if len(correspondence)<8:continue
        a=np.array([features[0][0][m.queryIdx].pt for m in correspondence]);b=np.array([features[1][0][m.trainIdx].pt for m in correspondence])
        essential,mask=cv2.findEssentialMat(a,b,intrinsics,method=cv2.RANSAC,prob=.999,threshold=1.)
        if essential is None:continue
        _,rotation,translation,inliers=cv2.recoverPose(essential[:3],a,b,intrinsics,mask=mask)
        pose=np.c_[rotation,translation];xyz,valid=triangulate(a,b,pose0,pose,intrinsics);valid&=inliers.ravel()>0
        if valid.sum()<6:continue
        descriptors=features[1][1][[m.trainIdx for m in correspondence]][valid]
        beam.append(dict(id=0 if not beam else next_id,pose=pose,xyz=xyz[valid],descriptors=descriptors,born=np.ones(valid.sum(),dtype=int),history=[np.zeros(3),-rotation.T@translation[:,0]],score=int(valid.sum())))
        next_id+=1
    if not beam:raise RuntimeError("Two-view initialization failed; no trajectory is fabricated")
    for tick in range(1,count):
        if tick>1:
            children=[]
            for parent in beam:
                for alternative in [False,True]:
                    correspondence=matches(parent["descriptors"],features[tick][1],alternative)
                    if len(correspondence)<6:continue
                    xyz=parent["xyz"][[m.queryIdx for m in correspondence]];pixel=np.array([features[tick][0][m.trainIdx].pt for m in correspondence])
                    success,rvec,tvec,inliers=cv2.solvePnPRansac(xyz,pixel,intrinsics,None,iterationsCount=100,reprojectionError=2.,confidence=.999)
                    if not success or inliers is None or len(inliers)<6:continue
                    rvec,tvec=cv2.solvePnPRefineLM(xyz[inliers[:,0]],pixel[inliers[:,0]],intrinsics,None,rvec,tvec,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,10,1e-6))
                    rotation=cv2.Rodrigues(rvec)[0];pose=np.c_[rotation,tvec]
                    temporal=matches(features[tick-1][1],features[tick][1],False)
                    first=np.array([features[tick-1][0][m.queryIdx].pt for m in temporal]);second=np.array([features[tick][0][m.trainIdx].pt for m in temporal])
                    extra,valid=triangulate(first,second,parent["pose"],pose,intrinsics) if len(temporal) else (np.empty((0,3)),np.zeros(0,dtype=bool))
                    keep=parent["born"]>=tick-window
                    new_descriptors=features[tick][1][[m.trainIdx for m in temporal]][valid] if temporal else np.empty((0,32),dtype=np.uint8)
                    children.append(dict(id=parent["id"] if not alternative else next_id,pose=pose,xyz=np.r_[parent["xyz"][keep],extra[valid]],descriptors=np.r_[parent["descriptors"][keep],new_descriptors],born=np.r_[parent["born"][keep],np.full(valid.sum(),tick)],history=parent["history"]+[-rotation.T@tvec[:,0]],score=len(inliers)))
                    if alternative:next_id+=1
            if not children:raise RuntimeError(f"Visual tracking lost at frame {tick}; no pose is fabricated")
            beam=children
        beam=sorted(beam,key=lambda row:-row["score"])
        ever_truncated |= len(beam)>capacity
        beam=beam[:capacity];truncated[tick]=ever_truncated
        for slot,branch in enumerate(beam):
            alive[slot,tick]=True;positions[slot,tick]=branch["history"][-1];rotations[slot,tick]=branch["pose"][:,:3].T;lineage[slot,tick]=branch["id"]
            for index,frame in enumerate(anchor_frames):
                if frame<=tick:anchored[slot,tick,index]=branch["history"][frame]
    velocity=np.zeros((count,3))
    for tick in range(count):
        observed=anchor_frames[anchor_frames<=tick]
        if len(observed)>=2:
            first,last=observed[-2:]
            velocity[tick]=(references[last]-references[first])/(timestamps[last]-timestamps[first])
    return dict(positions=positions,rotations=rotations,anchors=references,velocity=velocity,visual_bound=np.full(count,visual_radius),time=timestamps,alive=alive,anchor_positions=anchored,anchor_frames=anchor_frames,lineage=lineage,frontend_truncated=truncated)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--frames",required=True);parser.add_argument("--references",required=True,help="NPZ anchors[T,3], time[T]");parser.add_argument("--intrinsics",required=True,help="JSON3x3 K");parser.add_argument("--output",required=True);parser.add_argument("--capacity",type=int,default=8);parser.add_argument("--window",type=int,default=8);parser.add_argument("--seed",type=int,default=13);parser.add_argument("--visual-radius",type=float,default=.05);args=parser.parse_args()
    paths=sorted(p for p in Path(args.frames).iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg"});images=[cv2.imread(str(path),cv2.IMREAD_GRAYSCALE) for path in paths]
    reference=np.load(args.references,allow_pickle=False)
    result=generate(images,np.asarray(json.loads(Path(args.intrinsics).read_text(encoding="utf-8"))),reference["anchors"],reference["time"],args.capacity,args.window,args.seed,args.visual_radius)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**result)


if __name__=="__main__":main()
