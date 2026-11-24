#!/usr/bin/env python3
"""Fresh-cube evaluation of exact ordered projection and reciprocal-DTW priors."""
from __future__ import annotations
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,hashlib,importlib.metadata,importlib.util,json,pathlib,resource,sys,time
import numpy as np
import requests
from scipy import ndimage as ndi

HERE=pathlib.Path(__file__).resolve().parent
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
old=module("ordered054",HERE.parent/"ordered_tracking"/"run.py")
data=module("instances053",HERE.parent/"orthogonal_instances"/"run.py")
CUBES=("07376_03024_04048","10448_02512_02512")
KEYS=("P_local","I_independent_cut","J_joint_cut","W_DTW","R_projection","K_fixed_prior","H_reciprocal")
DLO,DHI=old.DLO,old.DHI
AT=old.ANCHOR

def reciprocal(plane,seed,at):
    f=old.dtw_features(plane);s=len(seed);w=plane.shape[1]
    x=np.empty((s,w),np.int64);err=np.zeros_like(x);x[:,at]=seed
    for step in (-1,1):
        for i in range(at+step,-1 if step<0 else w,step):
            j=i-step
            forward=old.dtw_map(f[:,j],f[:,i]);back=old.dtw_map(f[:,i],f[:,j])
            q=x[:,j];p=forward[q]
            x[:,i]=p
            err[:,i]=err[:,j]+np.abs(back[p]-q)
    return x,err

def prior_costs(cost,w,err):
    dist=np.abs(np.arange(DLO,DHI,dtype=np.int64)[None,None,:]-w[:,:,None])
    cap=np.minimum(dist,12)
    return 1000*dist,cost+150*cap,cost+np.rint(150*cap/(1+err[:,:,None]/4.)).astype(np.int64)

def solve_hybrids(cost,w,err,seed,at):
    cs=prior_costs(cost,w,err)
    out={};diag={}
    for key,c,lam in zip(KEYS[4:],cs,(0,old.LAM,old.LAM)):
        q,d=old.cut(c,seed-DLO,at,lam=lam,slope=old.SLOPE,gap=old.GAP)
        out[key]=q+DLO;diag[key]=d
    return out,diag

def self_test():
    t=np.arange(96,dtype=float)
    s=np.exp(-((t-28)/2.5)**2)+.7*np.exp(-((t-53)/4)**2)-.45*np.exp(-((t-70)/2)**2)+.01*np.sin(t*.37)
    shifts=np.arange(-2,3)
    plane=np.stack([np.interp(t-j,t,s) for j in shifts],1)
    seed=np.array([28,53]);p,e=reciprocal(plane,seed,2)
    exact=old.propagate(plane,seed,2)
    clean=np.max(np.abs(p-(seed[:,None]+shifts)))<=1
    w=np.array([[24,24,20,24,24],[24,24,26,24,24]])
    c=np.zeros((2,5,DHI-DLO),np.int64)
    R,K,H=prior_costs(c,w,np.zeros_like(w))
    q,d=solve_hybrids(c,w,np.zeros_like(w),np.array([20,26]),2)
    gates={"clean_shifted_wave":bool(clean),"same_forward_as_W":bool(np.array_equal(p,exact)),
           "zero_error_cost_exact":bool(np.array_equal(K,H)),
           "forced_collision_removed":bool((np.diff(w,axis=0)<2).any() and all((np.diff(x,axis=0)>=2).all() for x in q.values())),
           "all_exact_seed":bool(all(np.array_equal(x[:,2],[20,26]) for x in q.values()))}
    return {"gates":gates,"pass":all(gates.values()),"clean_max_error":int(np.max(np.abs(p-(seed[:,None]+shifts)))),
            "clean_roundtrip_error_max":int(e.max()),"collision_input":w.tolist(),"collision_projection":q["R_projection"].tolist()}

def section(plane,gt,axis,index):
    t=time.monotonic();a=old.anchor_ids(gt[:,AT])
    r={"fixed_axis":axis,"slice":index,"anchor_count":len(a),"CT_sha256":old.arrsha(plane),"label_sha256":old.arrsha(gt),"evaluable":False}
    if len(a)<2:r["reason"]="fewer than two unique seed runs";return r
    seed=np.array([z for g,z in a],np.int64);c=old.unary(plane,seed)
    P=c.argmin(2)+DLO;P[:,AT]=seed
    I,i=old.cut(c,seed-DLO,AT,lam=old.LAM,slope=old.SLOPE,gap=0)
    J,j=old.cut(c,seed-DLO,AT,lam=old.LAM,slope=old.SLOPE,gap=old.GAP)
    W,E=reciprocal(plane,seed,AT)
    if not np.array_equal(W,old.propagate(plane,seed,AT)):raise ArithmeticError("Forward W parity failed")
    hy,di=solve_hybrids(c,W,E,seed,AT)
    preds={"P_local":P,"I_independent_cut":I+DLO,"J_joint_cut":J+DLO,"W_DTW":W,**hy}
    truth,allowed=old.evaluation_set(gt,a)
    r["arms"]={k:old.score(gt,a,p,truth,allowed) for k,p in preds.items()}
    r["cut_diagnostics"]={"I":i,"J":j,**di}
    r["reciprocity"]={"W_forward_exact":True,"cumulative_error_p50_p95_max":np.quantile(E[:,allowed],[.5,.95,1]).tolist(),
        "zero_error_fraction":float((E[:,allowed]==0).mean()),"changed_by_hybrids":{k:float((p[:,allowed]!=W[:,allowed]).mean()) for k,p in hy.items()}}
    v=r["arms"]["P_local"];r["evaluable"]=v["eligible"]>=30 and v["eligible_fraction"]>=.5
    if not r["evaluable"]:r["reason"]="insufficient labeled evaluation pairs"
    r["wall_seconds"]=time.monotonic()-t
    return r

def aggregate(rows):
    out=old.aggregate(rows);rr=[r for r in rows if r["evaluable"]]
    for key in KEYS[4:]:
        vals=[r["arms"][key] for r in rr]
        s={k:sum(v[k] for v in vals) for k in ("eligible","planned","near2","correct_id","other_id","background","absolute_error_sum","gap_violations","strict_crossings","adjacent_pairs")}
        n=s["eligible"];p=s["adjacent_pairs"]
        out["arms"][key]={**s,"within2_rate":s["near2"]/n if n else None,"correct_id_rate":s["correct_id"]/n if n else None,
                          "wrong_id_rate":s["other_id"]/n if n else None,"MAE_vox":s["absolute_error_sum"]/n if n else None,
                          "crossing_fraction":s["strict_crossings"]/p if p else None,"gap_fraction":s["gap_violations"]/p if p else None}
    if out["population_pass"]:
        W=out["arms"]["W_DTW"];H=out["arms"]["H_reciprocal"];J=out["arms"]["J_joint_cut"]
        out["H_gates"]={"zero_gap_violations":H["gap_violations"]==0,
              "within2_nonharm_W":H["within2_rate"]>=W["within2_rate"]-.02,
              "correct_id_nonharm_W":H["correct_id_rate"]>=W["correct_id_rate"]-.03,
              "other_id_nonharm_W":H["wrong_id_rate"]<=W["wrong_id_rate"]+.03,
              "within2_nonharm_J":H["within2_rate"]>=J["within2_rate"]-.02}
        out["H_within2_delta_W"]=H["within2_rate"]-W["within2_rate"]
    else:out["H_gates"]={"population":False};out["H_within2_delta_W"]=None
    return out

def run(a):
    t=time.monotonic()
    r={"schema":"adaptive.dtw_order_hybrid.v1","script_sha256":old.sha(__file__),"protocol_sha256":old.sha(HERE/"protocol.md"),
       "source_reader_sha256":old.sha(HERE.parent/"orthogonal_instances"/"run.py"),
       "solver_054_sha256":old.sha(HERE.parent/"ordered_tracking"/"run.py"),
       "sources":{},"source_bytes":0,"cases":[],"pass":False,
       "packages":{k:importlib.metadata.version(k) for k in ("numpy","scipy","dtw-python","pynrrd")}}
    try:
        r["synthetic"]=self_test()
        if not r["synthetic"]["pass"]:raise ValueError("Frozen hybrid controls failed")
        data.CAP=120*1024**2;a.cache.mkdir(parents=True,exist_ok=False)
        with requests.Session() as session:
            for cube in CUBES:
                ct,gt=data.read_cube(session,a.cache,cube,r)
                deriv=[float(np.mean(ndi.gaussian_filter(ct.astype(np.float32),1,order=tuple(int(i==j) for i in range(3)))**2)) for j in range(3)]
                normal=int(np.argmax(deriv));rows=[]
                for axis in range(3):
                    if axis==normal:continue
                    remain=[j for j in range(3) if j!=axis];naxis=remain.index(normal)
                    for index in (24,48,72):
                        plane=np.take(ct,index,axis=axis);labels=np.take(gt,index,axis=axis)
                        if naxis==1:plane=plane.T;labels=labels.T
                        try:row=section(plane,labels,axis,index)
                        except Exception as e:row={"fixed_axis":axis,"slice":index,"evaluable":False,"failure":f"{type(e).__name__}: {str(e)[:300]}"}
                        rows.append(row)
                rec={"cube":cube,"dominant_CT_normal_axis":normal,"gradient_energy":deriv,"sections":rows,"aggregate":aggregate(rows)}
                r["cases"].append(rec);print("CASE_JSON="+json.dumps(rec,separators=(",",":"),allow_nan=False),flush=True)
        r["complete"]=len(r["cases"])==2 and all(len(x["sections"])==6 and not any("failure" in q for q in x["sections"]) for x in r["cases"])
        deltas=[x["aggregate"]["H_within2_delta_W"] for x in r["cases"]]
        r["at_least_one_gain_ge03"]=any(x is not None and x>=.03 for x in deltas)
        r["pass"]=r["complete"] and r["at_least_one_gain_ge03"] and all(all(x["aggregate"]["H_gates"].values()) for x in r["cases"])
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["wall_seconds"]=time.monotonic()-t;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("cache","output"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
