#!/usr/bin/env python3
"""True two-lateral-axis ordered cuts, reusing 054/055 DTW and unary costs."""
from __future__ import annotations
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,hashlib,importlib.util,json,pathlib,resource,sys,time
import numpy as np
import requests
from scipy import ndimage as ndi,sparse
from scipy.sparse.csgraph import maximum_flow,breadth_first_order

HERE=pathlib.Path(__file__).resolve().parent
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
hy=module("hybrid055",HERE.parent/"dtw_order_hybrid"/"run.py")
old=hy.old;data=hy.data
CUBES=("02256_02000_02000","10192_03792_03792","10192_04304_04048")
DLO,DHI=old.DLO,old.DHI
KEYS=("P_local","I_3D","J_3D","W_DTW","R_projection","K_fixed","H_reciprocal","Hrow")
MAX_NODES=1_000_000
MAX_ARCS=12_000_000

def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def energy(c,x,lam,axes):
    z=int(np.take_along_axis(c,x[...,None],axis=-1).sum(dtype=np.int64))
    return z+sum(lam*int(np.abs(np.diff(x,axis=a)).sum(dtype=np.int64)) for a in axes)

def feasible(x,seed,anchor,slope,gap,axes):
    return bool(np.array_equal(x[:,anchor[0],anchor[1]],seed)
           and all((np.abs(np.diff(x,axis=a))<=slope).all() for a in axes)
           and (not gap or (np.diff(x,axis=0)>=gap).all()))

def cut3d(c,seed,anchor,*,lam=100,slope=4,gap=2,axes=(1,2)):
    c=np.asarray(c,np.int64);seed=np.asarray(seed,np.int64)
    S,Y,X,D=c.shape;K=D-1
    if D<2 or S<1 or min(Y,X)<1 or seed.shape!=(S,) or (seed<0).any() or (seed>=D).any() or not 0<=anchor[0]<Y or not 0<=anchor[1]<X:raise ValueError("Invalid cut inputs")
    if slope<0 or slope>=D or gap<0 or gap>=D or lam<0 or (gap and (np.diff(seed)<gap).any()) or any(a not in (1,2) for a in axes):raise ValueError("Infeasible cut contract")
    n=S*Y*X*K
    if n>MAX_NODES:raise ValueError("Threshold node cap")
    nodes=np.arange(n,dtype=np.int64).reshape(S,Y,X,K);src=n;snk=n+1
    diff=np.diff(c,axis=-1)
    finite=int(np.abs(diff).sum(dtype=np.int64))
    for a in axes:finite+=2*lam*K*(n//K//nodes.shape[a])*(nodes.shape[a]-1)
    H=finite+1
    if H>=2**30:raise ValueError("Integer capacity cap")
    us=[];vs=[];ws=[];count=0
    def add(a,b,z):
        nonlocal count
        a,b,z=np.broadcast_arrays(a,b,z);a=a.ravel();b=b.ravel();z=z.ravel();good=z>0
        if not good.any():return
        count+=int(good.sum())
        if count>MAX_ARCS:raise ValueError("Directed arc cap")
        us.append(a[good]);vs.append(b[good]);ws.append(z[good].astype(np.int64))
    add(nodes,snk,np.maximum(diff,0));add(src,nodes,np.maximum(-diff,0))
    add(nodes[...,1:],nodes[...,:-1],H)
    for ax in axes:
        lo=[slice(None)]*4;hi=lo.copy();lo[ax]=slice(None,-1);hi[ax]=slice(1,None)
        a,b=nodes[tuple(lo)],nodes[tuple(hi)]
        if lam:add(a,b,lam);add(b,a,lam)
        if slope<K:
            add(a[...,slope:],b[...,:K-slope],H)
            add(b[...,slope:],a[...,:K-slope],H)
    if gap and S>1:
        add(src,nodes[1:,:,:,gap-1],H)
        add(nodes[:-1,:,:,K-gap],snk,H)
        if gap<K:add(nodes[:-1,:,:,:K-gap],nodes[1:,:,:,gap:],H)
    for s,z in enumerate(seed):
        a=nodes[s,anchor[0],anchor[1]]
        if z>0:add(src,a[z-1],H)
        if z<K:add(a[z],snk,H)
    g=sparse.coo_matrix((np.concatenate(ws),(np.concatenate(us),np.concatenate(vs))),shape=(n+2,n+2)).tocsr()
    del us,vs,ws
    if g.data.max(initial=0)>=2**31:raise ValueError("Parallel arc overflow")
    f=maximum_flow(g,src,snk,method="dinic")
    if int(f.flow_value)>=H:raise ArithmeticError("Hard constraint cut")
    residual=(g-f.flow).tocsr()
    residual.data=(residual.data>0).astype(np.int8);residual.eliminate_zeros()
    reach=breadth_first_order(residual,src,directed=True,return_predecessors=False)
    if snk in reach:raise ArithmeticError("Residual connects source and sink")
    m=np.zeros(n+2,bool);m[reach]=True;lev=m[nodes]
    if (lev[...,1:]&~lev[...,:-1]).any():raise ArithmeticError("Nonmonotone thresholds")
    x=lev.sum(-1,dtype=np.int64)
    if not feasible(x,seed,anchor,slope,gap,axes):raise ArithmeticError("Decoded infeasible")
    E=energy(c,x,lam,axes);constant=int(c[...,0].sum(dtype=np.int64)+np.minimum(diff,0).sum(dtype=np.int64))
    if E!=int(f.flow_value)+constant:raise ArithmeticError("Exact objective mismatch")
    return x,{"energy":E,"flow":int(f.flow_value),"constant":constant,"threshold_nodes":n,"arcs":int(g.nnz),"hard_capacity":H,"axes":list(axes)}

def features(ct):
    a=ndi.gaussian_filter1d(ct.astype(np.float64),1,axis=0)
    a=(a-a.mean(0))/np.maximum(a.std(0),1e-6)
    return np.stack((a,.5*np.gradient(a,axis=0)),axis=-1)

def track(ct,seed,anchor,transpose=False):
    # Deterministic spanning tree: row of anchor, then all columns.
    if transpose:
        x,e=track(ct.transpose(0,2,1),seed,anchor[::-1],False)
        return x.transpose(0,2,1),e.transpose(0,2,1)
    f=features(ct);S=len(seed);Y,X=ct.shape[1:];ay,ax=anchor
    w=np.empty((S,Y,X),np.int64);e=np.zeros_like(w);w[:,ay,ax]=seed
    def move(y,x,py,px):
        fw=old.dtw_map(f[:,py,px],f[:,y,x])
        bw=old.dtw_map(f[:,y,x],f[:,py,px])
        q=w[:,py,px];p=fw[q];w[:,y,x]=p
        e[:,y,x]=e[:,py,px]+np.abs(bw[p]-q)
    for step in (-1,1):
        for x in range(ax+step,-1 if step<0 else X,step):move(ay,x,ay,x-step)
    for x in range(X):
        for step in (-1,1):
            for y in range(ay+step,-1 if step<0 else Y,step):move(y,x,y-step,x)
    return w,e

def unary(ct,seed,anchor):
    w=np.lib.stride_tricks.sliding_window_view(ct.astype(np.float64),9,axis=0)
    w=w-w.mean(-1,keepdims=True);norm=np.linalg.norm(w,axis=-1,keepdims=True)
    w=np.divide(w,norm,out=np.zeros_like(w),where=norm>1e-10)
    tpl=w[seed-DLO,anchor[0],anchor[1]]
    sim=np.einsum("sk,dyxk->syxd",tpl,w,optimize=True)
    return np.rint(1000*(1-np.clip(sim,-1,1)+.002*np.abs(np.arange(DLO,DHI)[None,None,None,:]-seed[:,None,None,None]))).astype(np.int64)

def self_test():
    rng=np.random.default_rng(5701);rows=[]
    allx=np.indices((4,)*8,dtype=np.int8).reshape(8,-1).T.reshape(-1,2,2,2)
    seed=np.array([1,2]);anchor=(0,0)
    for i in range(2):
        c=rng.integers(0,13,(2,2,2,4),dtype=np.int64)
        for axes in ((1,2),(2,)):
            q,diag=cut3d(c,seed,anchor,lam=3,slope=2,gap=1,axes=axes)
            valid=np.all(allx[:,:,0,0]==seed[None],axis=1)&np.all(np.diff(allx,axis=1)>=1,axis=(1,2,3))
            for ax in axes:valid&=np.all(np.abs(np.diff(allx,axis=ax+1))<=2,axis=(1,2,3))
            opt=min(energy(c,x,3,axes) for x in allx[valid])
            tr=None
            if axes==(1,2):
                _,r=cut3d(c.transpose(0,2,1,3),seed,anchor[::-1],lam=3,slope=2,gap=1);tr=r["energy"]
            rows.append({"axes":list(axes),"oracle":opt,"cut":diag["energy"],"transposed":tr,"pass":opt==diag["energy"] and (tr is None or tr==opt)})
    c=np.full((2,2,2,4),20,np.int64);c[0,...,2]=0;c[1,...,1]=0;c[0,0,0,1]=0;c[1,0,0,2]=0
    i,_=cut3d(c,seed,anchor,lam=0,slope=3,gap=0);j,_=cut3d(c,seed,anchor,lam=0,slope=3,gap=1)
    t=np.arange(96,dtype=float);s=np.exp(-((t-28)/2.5)**2)+.7*np.exp(-((t-53)/4)**2)-.45*np.exp(-((t-70)/2)**2)+.01*np.sin(t*.37)
    shift=np.arange(3)[:,None]+np.arange(3)[None,:]-2
    v=np.stack([np.interp(t-a,t,s) for a in shift.ravel()],1).reshape(96,3,3)
    w,e=track(v,np.array([28,53]),(1,1));z,ee=track(v,np.array([28,53]),(1,1),True)
    err=int(np.abs(w-(np.array([28,53])[:,None,None]+shift)).max())
    gates={"inherited_055":hy.self_test()["pass"],"exact_bruteforce_and_transpose":all(q["pass"] for q in rows),
           "ordered_blocks_crossing":bool((np.diff(i,axis=0)<0).any() and (np.diff(j,axis=0)>=1).all()),
           "two_axis_shift":err<=1 and int(np.abs(z-w).max())<=1}
    return {"gates":gates,"oracle_cases":rows,"DTW_shift_max_error":err,"roundtrip_max":int(e.max()),"pass":all(gates.values())}

def truth_for(gt,anchors,allowed):
    S=len(anchors);Y,X=gt.shape[1:]
    truth=np.full((S,Y,X),-1,np.int64)
    for s,(gid,_) in enumerate(anchors):
        for y,x in zip(*np.nonzero(allowed)):
            d=np.flatnonzero(gt[:,y,x]==gid)
            if len(d) and (len(d)==1 or np.all(np.diff(d)==1)):truth[s,y,x]=(int(d[0])+int(d[-1]))//2
    return truth

def score(gt,anchors,p,truth,allowed):
    if p.shape!=truth.shape or (p<0).any() or (p>=gt.shape[0]).any():raise ValueError("Invalid prediction")
    S,Y,X=p.shape;v=truth>=0;n=int(v.sum());total=S*int(allowed.sum())
    ids=np.array([g for g,_ in anchors])[:,None,None]
    yy,xx=np.indices((Y,X));pick=gt[p,yy,xx];d=np.abs(p[v]-truth[v])
    near=int((d<=2).sum());right=int(((pick==ids)&v).sum());wrong=int(((pick>0)&(pick!=ids)&v).sum());bg=int(((pick==0)&v).sum())
    assert right+wrong+bg==n
    gaps=np.diff(p,axis=0)
    slopes=[np.abs(np.diff(p,axis=a)) for a in (1,2)]
    return {"eligible":n,"planned":total,"eligible_fraction":n/total,"near2":near,"correct_id":right,"other_id":wrong,"background":bg,
            "within2_rate":near/n if n else None,"correct_id_rate":right/n if n else None,"wrong_id_rate":wrong/n if n else None,
            "MAE_vox":float(d.mean()) if n else None,"absolute_error_sum":int(d.sum()),
            "gap_violations":int((gaps<2).sum()),"strict_crossings":int((gaps<0).sum()),"adjacent_pairs":gaps.size,
            "slope_violations_axes":[int((a>old.SLOPE).sum()) for a in slopes],
            "prediction_sha256":old.arrsha(p)}

def case(ct,gt,normal):
    t=time.monotonic()
    v=np.moveaxis(ct,normal,0)[:,32:64,32:64];g=np.moveaxis(gt,normal,0)[:,32:64,32:64];anchor=(16,16)
    a=old.anchor_ids(g[:,anchor[0],anchor[1]])
    r={"dominant_CT_normal_axis":normal,"shape_zyx":list(v.shape),"anchor_count":len(a),"CT_sha256":old.arrsha(v),"label_sha256":old.arrsha(g),"evaluable":False}
    if len(a)<2:r["reason"]="fewer than two seed runs";return r
    seed=np.array([z for _,z in a]);C=unary(v,seed,anchor)
    P=C.argmin(-1)+DLO;P[:,anchor[0],anchor[1]]=seed
    W,e=track(v,seed,anchor);alt,ae=track(v,seed,anchor,True)
    dist=np.abs(np.arange(DLO,DHI)[None,None,None,:]-W[...,None]);cap=np.minimum(dist,12)
    R=1000*dist;K=C+150*cap;H=C+np.rint(150*cap/(1+e[...,None]/4)).astype(np.int64)
    pred={"P_local":P,"W_DTW":W};di={}
    for key,c,lam,gap,axes in (("I_3D",C,100,0,(1,2)),("J_3D",C,100,2,(1,2)),("R_projection",R,0,2,(1,2)),("K_fixed",K,100,2,(1,2)),("H_reciprocal",H,100,2,(1,2)),("Hrow",H,100,2,(2,))):
        p,receipt=cut3d(c,seed-DLO,anchor,lam=lam,slope=old.SLOPE,gap=gap,axes=axes)
        pred[key]=p+DLO;di[key]=receipt
    yy,xx=np.indices((32,32));allowed=(np.abs(yy-anchor[0])>2)|(np.abs(xx-anchor[1])>2)
    truth=truth_for(g,a,allowed)
    r["arms"]={key:score(g,a,pred[key],truth,allowed) for key in KEYS}
    r["cut_diagnostics"]=di
    delta=np.abs(W-alt)
    r["path_dependence"]={"depth_disagreement_p50_p95_max":np.quantile(delta[:,allowed],[.5,.95,1]).tolist(),
           "nonidentical_fraction":float((delta[:,allowed]>0).mean()),"alternate_DTW":score(g,a,alt,truth,allowed),
           "reciprocity_p50_p95_max":np.quantile(e[:,allowed],[.5,.95,1]).tolist()}
    ids,counts=np.unique(g[g>0],return_counts=True);allids=[int(i) for i,n in zip(ids,counts) if n>=64]
    unseed=[i for i in allids if i not in [q[0] for q in a]]
    u=truth_for(g,[(i,0) for i in unseed],allowed)
    r["coverage"]={"manual_ids_ge64_vox":len(allids),"seeded_ids":len(a),"unseeded_ids_ge64_vox":len(unseed),
                   "unseeded_unique_run_pairs":int((u>=0).sum()),"seeded_unique_run_pairs":int((truth>=0).sum())}
    r["coverage"]["unseeded_pair_fraction"]=int((u>=0).sum())/max(1,int((u>=0).sum()+(truth>=0).sum()))
    Wm=r["arms"]["W_DTW"];Hm=r["arms"]["H_reciprocal"];Jm=r["arms"]["J_3D"];Rm=r["arms"]["Hrow"]
    r["evaluable"]=Wm["eligible"]>=500 and Wm["eligible_fraction"]>=.5
    r["gates"]={"population":r["evaluable"]}
    if r["evaluable"]:
        r["gates"].update(gap=Hm["gap_violations"]==0,slope=Hm["slope_violations_axes"]==[0,0],
            near2_vs_W=Hm["within2_rate"]>=Wm["within2_rate"]-.02,id_vs_W=Hm["correct_id_rate"]>=Wm["correct_id_rate"]-.03,
            wrong_vs_W=Hm["wrong_id_rate"]<=Wm["wrong_id_rate"]+.03,near2_vs_J=Hm["within2_rate"]>=Jm["within2_rate"]-.02,
            near2_vs_Hrow=Hm["within2_rate"]>=Rm["within2_rate"]-.02,id_vs_Hrow=Hm["correct_id_rate"]>=Rm["correct_id_rate"]-.03)
        r["delta_near2_W"]=Hm["within2_rate"]-Wm["within2_rate"];r["delta_near2_Hrow"]=Hm["within2_rate"]-Rm["within2_rate"]
    r["wall_seconds"]=time.monotonic()-t
    return r

def run(a):
    t=time.monotonic();r={"schema":"adaptive.dtw_order_3d.v1","script_sha256":digest(__file__),"protocol_sha256":digest(HERE/"protocol.md"),
        "solver_054_sha256":digest(HERE.parent/"ordered_tracking"/"run.py"),"hybrid_055_sha256":digest(HERE.parent/"dtw_order_hybrid"/"run.py"),
        "sources":{},"source_bytes":0,"cases":[],"pass":False}
    try:
        r["synthetic"]=self_test()
        if not r["synthetic"]["pass"]:raise ValueError("Frozen 3D synthetic controls failed")
        data.CAP=120*1024**2;a.cache.mkdir(parents=True,exist_ok=False)
        with requests.Session() as session:
            for cube in CUBES:
                try:
                    ct,gt=data.read_cube(session,a.cache,cube,r)
                    deriv=[float(np.mean(ndi.gaussian_filter(ct.astype(np.float32),1,order=tuple(int(i==j) for i in range(3)))**2)) for j in range(3)]
                    q=case(ct,gt,int(np.argmax(deriv)));q["gradient_energy"]=deriv
                except Exception as e:q={"evaluable":False,"failure":f"{type(e).__name__}: {str(e)[:400]}"}
                q["cube"]=cube;r["cases"].append(q);print("CASE_JSON="+json.dumps(q,separators=(",",":"),allow_nan=False),flush=True)
        r["complete"]=len(r["cases"])==len(CUBES) and all(q.get("evaluable",False) for q in r["cases"])
        r["gates"]={"all_cube_gates":r["complete"] and all(all(q["gates"].values()) for q in r["cases"]),
                    "some_H_gain_vs_W":any(q.get("delta_near2_W",-1)>=.03 for q in r["cases"]),
                    "some_H_gain_vs_Hrow":any(q.get("delta_near2_Hrow",-1)>=.03 for q in r["cases"])}
        r["pass"]=r["gates"]["all_cube_gates"] and r["gates"]["some_H_gain_vs_W"]
        r["cross_row_benefit_pass"]=r["pass"] and r["gates"]["some_H_gain_vs_Hrow"]
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["wall_seconds"]=time.monotonic()-t;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("cache","output"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
