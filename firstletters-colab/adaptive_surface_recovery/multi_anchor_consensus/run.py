#!/usr/bin/env python3
"""Sparse additional anchors + path-symmetric waveform priors, frozen controls."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,hashlib,importlib.util,itertools,json,pathlib,resource,sys,time
import numpy as np
import requests
from scipy import ndimage as ndi
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import anchored_cut as ac
b=ac.base
old=b.old
data=b.data
CUBES=("00768_02152_03536","01744_02512_03024","09168_02512_02768","10192_03792_02512")
CENTER=(16,16)
EXTRA=((8,8),(8,24),(24,8),(24,24))
KEYS=("W","Walt","J","K","H","D","A","JM","M")

class Tracks:
    """Cache directed correspondences, never fit to the GT outside anchors."""
    def __init__(self,volume):
        self.f=b.features(volume);self.Y,self.X=volume.shape[1:];self.maps={}
    def get(self,a,c):
        k=(*a,*c)
        if k not in self.maps:self.maps[k]=old.dtw_map(self.f[:,a[0],a[1]],self.f[:,c[0],c[1]])
        return self.maps[k]
    def track(self,seed,anchor,transpose=False):
        seed=np.asarray(seed,np.int64)
        w=np.empty((len(seed),self.Y,self.X),np.int64);e=np.zeros_like(w)
        ay,ax=anchor;w[:,ay,ax]=seed
        def move(y,x,py,px):
            fw=self.get((py,px),(y,x));bw=self.get((y,x),(py,px))
            z=w[:,py,px];p=fw[z];w[:,y,x]=p;e[:,y,x]=e[:,py,px]+np.abs(bw[p]-z)
        if not transpose:
            for sign in (-1,1):
                for x in range(ax+sign,-1 if sign<0 else self.X,sign):move(ay,x,ay,x-sign)
            for x in range(self.X):
                for sign in (-1,1):
                    for y in range(ay+sign,-1 if sign<0 else self.Y,sign):move(y,x,y-sign,x)
        else:
            for sign in (-1,1):
                for y in range(ay+sign,-1 if sign<0 else self.Y,sign):move(y,ax,y-sign,ax)
            for y in range(self.Y):
                for sign in (-1,1):
                    for x in range(ax+sign,-1 if sign<0 else self.X,sign):move(y,x,y,x-sign)
        return w,e

def prior(paths):
    """Integer sum of clipped distances, invariant to path order."""
    z=np.arange(b.DLO,b.DHI,dtype=np.int64)
    s=np.zeros(paths[0].shape+(len(z),),np.int64)
    for p in paths:s+=np.minimum(np.abs(p[...,None]-z),12)
    return np.rint(150*s/len(paths)).astype(np.int64)

def self_test():
    rng=np.random.default_rng(6101);c=rng.integers(0,12,(2,2,2,4));seed=np.array([1,2]);a=(0,0)
    x,di=b.cut3d(c,seed,a,lam=3,slope=2,gap=1)
    y,dj=ac.cut3d(c,seed,a,lam=3,slope=2,gap=1)
    extra=[(0,1,1,0),(1,1,1,3)]
    p,dp=ac.cut3d(c,seed,a,lam=3,slope=2,gap=1,extra=extra)
    candidates=np.indices((4,)*8,dtype=np.int8).reshape(8,-1).T.reshape(-1,2,2,2)
    mask=np.all(candidates[:,:,0,0]==seed[None],axis=1)&np.all(np.diff(candidates,axis=1)>=1,axis=(1,2,3))
    for axis in (2,3):mask&=np.all(np.abs(np.diff(candidates,axis=axis))<=2,axis=(1,2,3))
    for s,y0,x0,z in extra:mask&=candidates[:,s,y0,x0]==z
    oracle=min(b.energy(c,z,3,(1,2)) for z in candidates[mask])
    conflict=False
    try:ac.cut3d(c,seed,a,lam=3,slope=2,gap=1,extra=[(0,0,0,0)])
    except (ValueError,ArithmeticError):conflict=True
    # Smooth translated waveform, both path orders and exact original cache parity.
    t=np.arange(96,dtype=float)
    s=np.exp(-((t-28)/2.5)**2)+.7*np.exp(-((t-53)/4)**2)-.45*np.exp(-((t-70)/2)**2)+.01*np.sin(t*.37)
    shift=np.arange(3)[:,None]+np.arange(3)[None,:]-2
    v=np.stack([np.interp(t-k,t,s) for k in shift.ravel()],1).reshape(96,3,3)
    cache=Tracks(v);W,E=cache.track([28,53],(1,1));Alt,AE=cache.track([28,53],(1,1),True)
    W0,E0=b.track(v,np.array([28,53]),(1,1));A0,EA0=b.track(v,np.array([28,53]),(1,1),True)
    gates={"singleton_exact":np.array_equal(x,y) and di["energy"]==dj["energy"],
      "multi_anchor_oracle":dp["energy"]==oracle and all(p[s,y0,x0]==z for s,y0,x0,z in extra),
      "infeasible_conflict_rejected":conflict,"path_swap_exact":np.array_equal(prior([W,Alt]),prior([Alt,W])),
      "cached_DTW_exact":all(np.array_equal(a,b0) for a,b0 in ((W,W0),(E,E0),(Alt,A0),(AE,EA0)))}
    return {"gates":{k:bool(v) for k,v in gates.items()},"oracle_energy":oracle,"decoded_energy":dp["energy"],"pass":all(gates.values())}

def run_anchor(ray,gid):
    ix=np.flatnonzero(ray==gid)
    if len(ix)<2 or not np.all(np.diff(ix)==1):return None
    mid=(int(ix[0])+int(ix[-1]))//2
    return mid if b.DLO<=mid<b.DHI else None

def scores_subset(gt,anchors,p,truth,mask):
    n=int(mask.sum())
    if not n:return {"pairs":0}
    ids=np.array([g for g,_ in anchors])[:,None,None]
    yy,xx=np.indices(gt.shape[1:]);pick=gt[p,yy,xx];d=np.abs(p[mask]-truth[mask])
    right=int(((pick==ids)&mask).sum());other=int(((pick>0)&(pick!=ids)&mask).sum())
    return {"pairs":n,"within2":float((d<=2).mean()),"correct_id":right/n,"wrong_id":other/n,"MAE_vox":float(d.mean())}

def case(ct,gt,normal):
    t=time.monotonic()
    v=np.moveaxis(ct,normal,0)[:,32:64,32:64].copy()
    g=np.moveaxis(gt,normal,0)[:,32:64,32:64].copy()
    a=old.anchor_ids(g[:,CENTER[0],CENTER[1]])
    r={"normal_axis":normal,"shape":list(v.shape),"anchor_count":len(a),"CT_sha256":old.arrsha(v),"label_sha256":old.arrsha(g),"evaluable":False}
    if not a:r["reason"]="No central seed";return r
    seed=np.array([z for _,z in a],np.int64)
    C=b.unary(v,seed,CENTER);S=len(seed)
    extra=[];by_ray=[];all_anchor_ids=set(gid for gid,_ in a);counts=np.zeros(S,np.int64)
    for y,x in EXTRA:
        ray=g[:,y,x];all_anchor_ids.update(gid for gid,_ in old.anchor_ids(ray))
        hits=[]
        for s,(gid,_) in enumerate(a):
            z=run_anchor(ray,gid)
            if z is not None:
                extra.append((s,y,x,z-b.DLO));hits.append((s,z));counts[s]+=1
        by_ray.append((y,x,hits))
    ids,ns=np.unique(g[g>0],return_counts=True);present={int(i) for i,n in zip(ids,ns) if n>=64}
    r["anchors"]={"extra_observations":len(extra),"extra_per_central_sheet":counts.tolist(),"all_sheets_have_extra":bool((counts>0).all()),
      "rays_with_extra":sum(bool(hits) for _,_,hits in by_ray),"manual_ids_ge64":len(present),"central_ids_ge64":len(present&{g for g,_ in a}),
      "union_anchor_ids_ge64":len(present&all_anchor_ids)}
    yy,xx=np.indices((32,32));allowed=np.ones((32,32),bool)
    for y,x in (CENTER,)+EXTRA:allowed&=(np.abs(yy-y)>2)|(np.abs(xx-x)>2)
    truth=b.truth_for(g,a,allowed);m=truth>=0;n=int(m.sum());planned=S*int(allowed.sum())
    r.update(eligible=n,planned=planned,eligible_fraction=n/planned)
    if n<500 or n/planned<.5:r["reason"]="Insufficient frozen evaluation population";return r
    cache=Tracks(v);W,E=cache.track(seed,CENTER);Alt,AE=cache.track(seed,CENTER,True)
    pred={"W":W,"Walt":Alt};diag={}
    path_by_sheet=[[W[s],Alt[s]] for s in range(S)]
    for y,x,hits in by_ray:
        if not hits:continue
        z=np.array([z for s,z in hits],np.int64)
        p,_=cache.track(z,(y,x));q,_=cache.track(z,(y,x),True)
        for j,(s,_) in enumerate(hits):path_by_sheet[s]+=[p[j],q[j]]
    cap=np.minimum(np.abs(np.arange(b.DLO,b.DHI)[None,None,None,:]-W[...,None]),12)
    H=C+np.rint(150*cap/(1+E[...,None]/4)).astype(np.int64)
    D=C+prior([W,Alt])
    M=C+np.stack([prior(paths) for paths in path_by_sheet],axis=0)
    # Report exact permutation invariance of the constructed objective.
    if not np.array_equal(prior([Alt,W]),D-C) or not all(np.array_equal(prior(paths),prior(paths[::-1])) for paths in path_by_sheet):raise ArithmeticError("Path-order cost mismatch")
    for key,cost,anchors in (("J",C,()),("K",C+150*cap,()),("H",H,()),("D",D,()),("A",D,extra),("JM",C,extra),("M",M,extra)):
        if anchors:
            p,z=ac.cut3d(cost,seed-b.DLO,CENTER,lam=100,slope=4,gap=2,extra=anchors)
        else:p,z=b.cut3d(cost,seed-b.DLO,CENTER,lam=100,slope=4,gap=2)
        pred[key]=p+b.DLO;diag[key]=z
    r["arms"]={key:b.score(g,a,pred[key],truth,allowed) for key in KEYS};r["cut_diagnostics"]=diag
    r["path"]={"cached_directed_edges":len(cache.maps),"central_disagreement_fraction":float((W!=Alt).mean()),
       "central_depth_difference_p50_p95":np.quantile(np.abs(W-Alt),[.5,.95]).tolist(),
       "paths_per_sheet":[len(p) for p in path_by_sheet],"prior_permutation_exact":True}
    spread=np.stack([np.ptp(np.stack(p),axis=0) for p in path_by_sheet],axis=0)
    ids=np.flatnonzero(m.ravel());order=ids[np.argsort(spread.ravel()[ids],kind="stable")]
    r["selective"]=[]
    for fraction in (1.,.8,.5):
        k=int(np.floor(fraction*n));mask=np.zeros(m.size,bool);mask[order[:k]]=True;mask=mask.reshape(m.shape)
        r["selective"].append({"requested_coverage":fraction,"actual_coverage":k/n,
            "spread_max":int(spread[mask].max()) if k else None,"M":scores_subset(g,a,pred["M"],truth,mask),"W":scores_subset(g,a,W,truth,mask)})
    r["uncertainty_gate"]=r["selective"][1]["M"]["wrong_id"]<=r["selective"][0]["M"]["wrong_id"] and r["selective"][1]["M"]["MAE_vox"]<r["selective"][0]["M"]["MAE_vox"]
    nearest=np.full(truth.shape,np.inf)
    for s in range(S):
        for j in range(S):
            if j!=s:
                both=m[s]&m[j]
                nearest[s,both]=np.minimum(nearest[s,both],np.abs(truth[s,both]-truth[j,both]))
    r["gap_strata"]={}
    for name,mask in (("lt8",m&(nearest<8)),("8to16",m&(nearest>=8)&(nearest<16)),("ge16",m&(nearest>=16)&np.isfinite(nearest)),("unclassified",m&~np.isfinite(nearest))):
        r["gap_strata"][name]={key:scores_subset(g,a,pred[key],truth,mask) for key in KEYS}
    w,j,k,z=[r["arms"][key] for key in ("W","JM","K","M")]
    exact=all(pred["M"][s,y,x]==d+b.DLO for s,y,x,d in extra)
    r["gates"]={"gap":z["gap_violations"]==0,"slope":z["slope_violations_axes"]==[0,0],"anchors_exact":bool(exact),
       "id_vs_W":z["correct_id_rate"]>=w["correct_id_rate"]-.03,"wrong_vs_W":z["wrong_id_rate"]<=w["wrong_id_rate"]+.03,
       "near2_vs_K":z["within2_rate"]>=k["within2_rate"]-.02,"near2_vs_JM":z["within2_rate"]>=j["within2_rate"]-.02}
    r["gain_vs_K"]=z["within2_rate"]-k["within2_rate"];r["gain_vs_JM"]=z["within2_rate"]-j["within2_rate"]
    r["evaluable"]=True;r["wall_seconds"]=time.monotonic()-t
    return r

def run(a):
    t=time.monotonic()
    r={"schema":"adaptive.multi_anchor_consensus.v1","script_sha256":b.digest(__file__),"protocol_sha256":b.digest(HERE/"protocol.md"),
      "anchor_extension_sha256":b.digest(HERE/"anchored_cut.py"),"base_solver_sha256":b.digest(HERE.parent/"dtw_order_3d"/"run.py"),
      "sources":{},"source_bytes":0,"cases":[],"pass":False}
    try:
        r["synthetic"]=self_test()
        if not r["synthetic"]["pass"]:raise ValueError("Synthetic contracts failed")
        data.CAP=160*1024**2;a.cache.mkdir(parents=True,exist_ok=False)
        with requests.Session() as session:
            for cube in CUBES:
                try:
                    ct,gt=data.read_cube(session,a.cache,cube,r)
                    deriv=[float(np.mean(ndi.gaussian_filter(ct.astype(np.float32),1,order=tuple(int(i==j) for i in range(3)))**2)) for j in range(3)]
                    q=case(ct,gt,int(np.argmax(deriv)));q["gradient_energy"]=deriv
                except Exception as e:q={"evaluable":False,"failure":f"{type(e).__name__}: {str(e)[:400]}"}
                q["cube"]=cube;r["cases"].append(q);print("CASE_JSON="+json.dumps(q,separators=(",",":"),allow_nan=False),flush=True)
        rows=[q for q in r["cases"] if q.get("evaluable")]
        r["gates"]={"all_four_assessable":len(rows)==4,"at_least_two_multisheet":sum(q["anchor_count"]>=2 for q in rows)>=2,
                    "at_least_two_extra_complete":sum(q["anchors"]["all_sheets_have_extra"] for q in rows)>=2,
                    "all_M_gates":len(rows)==4 and all(all(q["gates"].values()) for q in rows),
                    "some_gain_vs_K_and_JM":any(q["gain_vs_K"]>=.03 and q["gain_vs_JM"]>=.03 for q in rows)}
        r["pass"]=all(r["gates"].values())
        r["uncertainty_pass"]=len(rows)==4 and sum(q["uncertainty_gate"] for q in rows)>=3
        r["macro"]={k:{m:float(np.mean([q["arms"][k][m] for q in rows])) for m in ("within2_rate","correct_id_rate","wrong_id_rate","MAE_vox")} for k in KEYS} if rows else {}
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["wall_seconds"]=time.monotonic()-t;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("cache","output"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
