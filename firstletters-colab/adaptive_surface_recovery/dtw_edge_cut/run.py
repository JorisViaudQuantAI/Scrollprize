#!/usr/bin/env python3
"""Path-free ordered optimization of all local DTW correspondences."""
from __future__ import annotations
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,importlib.util,json,pathlib,resource,sys,time
import numpy as np
import requests
from scipy import ndimage as ndi
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import edge_solver as es
sp=importlib.util.spec_from_file_location("consensus061",HERE.parent/"multi_anchor_consensus"/"run.py")
q=importlib.util.module_from_spec(sp);sys.modules[sp.name]=q;sp.loader.exec_module(q)
b=q.b;old=q.old;data=q.data
CUBES=("01744_02256_04304","07632_03024_04304","10192_03024_05072","10192_04304_03024")
KEYS=("W","Walt","J","K","H","D","G","G0","JM","GM")
WEIGHT=50

def transpose_maps(maps):
    return [(3-axis,reverse,mp.transpose(1,0,2).copy()) for axis,reverse,mp in maps]

def synthetic():
    inherited=q.self_test()
    rng=np.random.default_rng(6201);allx=np.indices((4,)*8,dtype=np.int8).reshape(8,-1).T.reshape(-1,2,2,2)
    seed=np.array([1,2]);anchor=(0,0)
    valid=np.all(allx[:,:,0,0]==seed[None],axis=1)&np.all(np.diff(allx,axis=1)>=1,axis=(1,2,3))
    for axis in (2,3):valid&=np.all(np.abs(np.diff(allx,axis=axis))<=2,axis=(1,2,3))
    rows=[]
    for test in range(2):
        c=rng.integers(0,13,(2,2,2,4),dtype=np.int64);maps=[]
        for axis in (1,2):
            shape=(1,2,4) if axis==1 else (2,1,4)
            for rev in (False,True):
                m=np.sort(rng.integers(0,4,shape),axis=-1)
                # Explicit constant-map endpoints and many-to-one correspondences.
                if test==0:m.reshape(-1,4)[0]=0 if rev else 3
                maps.append((axis,rev,m))
        def oracle_energy(p):
            total=b.energy(c,p,3,(1,2))
            for axis,rev,m in maps:
                for y in range(m.shape[0]):
                    for x in range(m.shape[1]):
                        a=(y,x);z=(y+1,x) if axis==1 else (y,x+1)
                        if rev:a,z=z,a
                        for s in range(2):total+=5*abs(int(p[s,z[0],z[1]])-int(m[y,x,p[s,a[0],a[1]]]))
            return int(total)
        best=min(oracle_energy(p) for p in allx[valid])
        p,d=es.cut3d(c,seed,anchor,lam=3,slope=2,gap=1,maps=maps,weight=5)
        pt,dt=es.cut3d(c.transpose(0,2,1,3),seed,anchor[::-1],lam=3,slope=2,gap=1,maps=transpose_maps(maps),weight=5)
        rows.append({"oracle":best,"cut":d["energy"],"direct_oracle_at_cut":oracle_energy(p),"transpose":dt["energy"],
                     "pass":best==d["energy"]==oracle_energy(p)==dt["energy"]})
    c=rng.integers(0,17,(2,2,2,4));maps=[]
    for axis in (1,2):
        shape=(1,2,4) if axis==1 else (2,1,4)
        for rev in (False,True):maps.append((axis,rev,np.broadcast_to(np.arange(4),shape).copy()))
    p,d=es.cut3d(c,seed,anchor,lam=3,slope=2,gap=1,maps=maps,weight=5)
    z,d0=b.cut3d(c,seed,anchor,lam=13,slope=2,gap=1)
    gates={"inherited061":inherited["pass"],"bruteforce_and_transposition":all(r["pass"] for r in rows),
           "identity_reduces_to_TV":d["energy"]==d0["energy"] and np.array_equal(p,z)}
    return {"gates":gates,"oracle_cases":rows,"identity_energy":d["energy"],"identity_TV_energy":d0["energy"],"pass":all(gates.values())}

def maps_from_cache(cache):
    maps=[];clipped=total=0
    for axis in (1,2):
        shape=(cache.Y-1,cache.X) if axis==1 else (cache.Y,cache.X-1)
        for reverse in (False,True):
            m=np.empty(shape+(b.DHI-b.DLO,),np.int64)
            for y,x in np.ndindex(shape):
                a=(y,x);c=(y+1,x) if axis==1 else (y,x+1)
                if reverse:a,c=c,a
                z=cache.get(a,c)[b.DLO:b.DHI]-b.DLO
                clipped+=int(((z<0)|(z>=b.DHI-b.DLO)).sum());total+=len(z)
                m[y,x]=np.clip(z,0,b.DHI-b.DLO-1)
            if (np.diff(m,axis=-1)<0).any():raise ValueError("Nonmonotone directed map")
            maps.append((axis,reverse,m))
    return maps,{"directed_neighbor_maps":sum(int(np.prod(m.shape[:-1])) for _,_,m in maps),"clipped_states":clipped,"all_states":total,"clipped_fraction":clipped/total}

def case(ct,gt,normal,check_transpose):
    t=time.monotonic();v=np.moveaxis(ct,normal,0)[:,32:64,32:64].copy();g=np.moveaxis(gt,normal,0)[:,32:64,32:64].copy()
    a=old.anchor_ids(g[:,q.CENTER[0],q.CENTER[1]])
    r={"normal_axis":normal,"shape":list(v.shape),"anchor_count":len(a),"CT_sha256":old.arrsha(v),"label_sha256":old.arrsha(g),"evaluable":False}
    if not a:r["reason"]="No central seed";return r
    seed=np.array([z for _,z in a],np.int64);S=len(seed);C=b.unary(v,seed,q.CENTER)
    extra=[];counts=np.zeros(S,np.int64)
    for y,x in q.EXTRA:
        for s,(gid,_) in enumerate(a):
            z=q.run_anchor(g[:,y,x],gid)
            if z is not None:extra.append((s,y,x,z-b.DLO));counts[s]+=1
    r["anchors"]={"extra_observations":len(extra),"extra_per_central_sheet":counts.tolist(),"all_sheets_have_extra":bool((counts>0).all())}
    yy,xx=np.indices((32,32));allowed=np.ones((32,32),bool)
    for y,x in (q.CENTER,)+q.EXTRA:allowed&=(np.abs(yy-y)>2)|(np.abs(xx-x)>2)
    truth=b.truth_for(g,a,allowed);m=truth>=0;n=int(m.sum());planned=S*int(allowed.sum())
    r.update(eligible=n,planned=planned,eligible_fraction=n/planned)
    if n<500 or n/planned<.5:r["reason"]="Insufficient frozen evaluation population";return r
    cache=q.Tracks(v);W,E=cache.track(seed,q.CENTER);Alt,_=cache.track(seed,q.CENTER,True)
    maps,receipt=maps_from_cache(cache)
    cap=np.minimum(np.abs(np.arange(b.DLO,b.DHI)[None,None,None,:]-W[...,None]),12)
    H=C+np.rint(150*cap/(1+E[...,None]/4)).astype(np.int64);D=C+q.prior([W,Alt])
    pred={"W":W,"Walt":Alt};diag={}
    for key,c in (("J",C),("K",C+150*cap),("H",H),("D",D)):
        p,z=b.cut3d(c,seed-b.DLO,q.CENTER,lam=100,slope=4,gap=2)
        pred[key]=p+b.DLO;diag[key]=z
    for key,lam,ex,mm in (("G",100,(),maps),("G0",0,(),maps),("JM",100,extra,()),("GM",100,extra,maps)):
        p,z=es.cut3d(C,seed-b.DLO,q.CENTER,lam=lam,slope=4,gap=2,extra=ex,maps=mm,weight=WEIGHT)
        pred[key]=p+b.DLO;diag[key]=z
    if check_transpose:
        x,z=es.cut3d(C.transpose(0,2,1,3),seed-b.DLO,q.CENTER[::-1],lam=100,slope=4,gap=2,maps=transpose_maps(maps),weight=WEIGHT)
        if z["energy"]!=diag["G"]["energy"]:raise ArithmeticError("Actual transposed energy differs")
        r["transpose_check"]={"energy":diag["G"]["energy"],"transposed_energy":z["energy"],
                              "decoded_difference_fraction":float((pred["G"]!=x.transpose(0,2,1)+b.DLO).mean())}
    r["arms"]={key:b.score(g,a,pred[key],truth,allowed) for key in KEYS};r["cut_diagnostics"]=diag
    r["maps"]={**receipt,"central_tree_path_disagreement_fraction":float((W!=Alt).mean()),"graph_uses_no_tree":True}
    nearest=np.full(truth.shape,np.inf)
    for s in range(S):
        for j in range(S):
            if j!=s:
                both=m[s]&m[j];nearest[s,both]=np.minimum(nearest[s,both],np.abs(truth[s,both]-truth[j,both]))
    r["gap_strata"]={}
    for name,mask in (("lt8",m&(nearest<8)),("8to16",m&(nearest>=8)&(nearest<16)),("ge16",m&(nearest>=16)&np.isfinite(nearest)),("unclassified",m&~np.isfinite(nearest))):
        r["gap_strata"][name]={k:q.scores_subset(g,a,pred[k],truth,mask) for k in KEYS}
    w,j,k,z,jm,gm=[r["arms"][key] for key in ("W","J","K","G","JM","GM")]
    r["G_gates"]={"gap":z["gap_violations"]==0,"slope":z["slope_violations_axes"]==[0,0],
      "id_W":z["correct_id_rate"]>=w["correct_id_rate"]-.03,"id_K":z["correct_id_rate"]>=k["correct_id_rate"]-.03,
      "wrong_W":z["wrong_id_rate"]<=w["wrong_id_rate"]+.03,"wrong_J":z["wrong_id_rate"]<=j["wrong_id_rate"]+.03,
      "near2_K":z["within2_rate"]>=k["within2_rate"]-.02,"near2_J":z["within2_rate"]>=j["within2_rate"]-.02}
    r["GM_gates"]={"gap":gm["gap_violations"]==0,"slope":gm["slope_violations_axes"]==[0,0],
      "id_JM":gm["correct_id_rate"]>=jm["correct_id_rate"]-.03,"wrong_JM":gm["wrong_id_rate"]<=jm["wrong_id_rate"]+.03,
      "near2_JM":gm["within2_rate"]>=jm["within2_rate"]-.02,
      "extra_exact":bool(all(pred["GM"][s,y,x]==z+b.DLO for s,y,x,z in extra))}
    r["gain_G_K"]=z["within2_rate"]-k["within2_rate"];r["gain_G_J"]=z["within2_rate"]-j["within2_rate"];r["gain_GM_JM"]=gm["within2_rate"]-jm["within2_rate"]
    r["evaluable"]=True;r["wall_seconds"]=time.monotonic()-t
    return r

def run(a):
    t=time.monotonic();r={"schema":"adaptive.dtw_edge_cut.v1","script_sha256":b.digest(__file__),"solver_sha256":b.digest(HERE/"edge_solver.py"),
      "protocol_sha256":b.digest(HERE/"protocol.md"),"consensus061_sha256":b.digest(HERE.parent/"multi_anchor_consensus"/"run.py"),
      "sources":{},"source_bytes":0,"cases":[],"pass":False}
    try:
        r["synthetic"]=synthetic()
        if not r["synthetic"]["pass"]:raise ValueError("Frozen graph contracts failed")
        data.CAP=180*1024**2;a.cache.mkdir(parents=True,exist_ok=False);checked=False
        with requests.Session() as session:
            for cube in CUBES:
                try:
                    ct,gt=data.read_cube(session,a.cache,cube,r)
                    deriv=[float(np.mean(ndi.gaussian_filter(ct.astype(np.float32),1,order=tuple(int(i==j) for i in range(3)))**2)) for j in range(3)]
                    row=case(ct,gt,int(np.argmax(deriv)),not checked);row["gradient_energy"]=deriv
                    checked|=bool(row.get("transpose_check"))
                except Exception as e:row={"evaluable":False,"failure":f"{type(e).__name__}: {str(e)[:500]}"}
                row["cube"]=cube;r["cases"].append(row);print("CASE_JSON="+json.dumps(row,separators=(",",":"),allow_nan=False),flush=True)
        rows=[z for z in r["cases"] if z.get("evaluable")]
        population=len(rows)==4 and sum(z["anchor_count"]>=2 for z in rows)>=2
        r["gates"]={"population":population,"actual_transpose_energy_exact":checked,"all_G_gates":len(rows)==4 and all(all(z["G_gates"].values()) for z in rows),
                    "some_G_gain_vs_K_and_J":any(z["gain_G_K"]>=.03 and z["gain_G_J"]>=.03 for z in rows)}
        r["pass"]=all(r["gates"].values())
        r["GM_gates"]={"population":population,"extra_anchor_population":sum(z["anchors"]["all_sheets_have_extra"] for z in rows)>=2,
           "all_GM_gates":len(rows)==4 and all(all(z["GM_gates"].values()) for z in rows),"some_GM_gain":any(z["gain_GM_JM"]>=.03 for z in rows)}
        r["GM_pass"]=all(r["GM_gates"].values())
        r["macro"]={k:{m:float(np.mean([z["arms"][k][m] for z in rows])) for m in ("within2_rate","correct_id_rate","wrong_id_rate","MAE_vox")} for k in KEYS} if rows else {}
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["wall_seconds"]=time.monotonic()-t;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("cache","output"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
