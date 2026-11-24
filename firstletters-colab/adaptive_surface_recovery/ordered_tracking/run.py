#!/usr/bin/env python3
"""Exact discrete ordered surface cut and library DTW on manually annotated sections.

Only the anchor ray is used by any predictor. Evaluation labels elsewhere are
never used to set the data cost, regularizer, orientation, or solver.
"""
from __future__ import annotations
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,hashlib,importlib.metadata,importlib.util,itertools,json,pathlib,resource,sys,time
import numpy as np
import requests
from scipy import ndimage as ndi,sparse
from scipy.sparse.csgraph import maximum_flow,breadth_first_order
from dtw import dtw

HERE=pathlib.Path(__file__).resolve().parent
DLO=4
DHI=92
LAM=100
SLOPE=4
GAP=2
ANCHOR=48

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def arrsha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def energy(cost,x,lam):
    s,w,d=cost.shape
    return int(cost[np.arange(s)[:,None],np.arange(w)[None,:],x].sum(dtype=np.int64)+lam*np.abs(np.diff(x,axis=1)).sum(dtype=np.int64))

def cut(cost,seed,at,lam=100,slope=4,gap=0):
    """Ishikawa-style threshold encoding of unary + L1 TV + ordered hard arcs.

    gap=0 disables inter-surface constraints; it is NOT a gap-zero ordering.
    Verify feasibility and integer energy by independent direct arithmetic.
    """
    c=np.asarray(cost,np.int64);seed=np.asarray(seed,np.int64)
    S,W,D=c.shape;K=D-1
    if D<2 or W<2 or S<1 or seed.shape!=(S,) or not 0<=at<W or (seed<0).any() or (seed>=D).any():raise ValueError("Invalid cut inputs")
    if slope<0 or slope>=D or gap<0 or gap>=D or lam<0 or (gap and (np.diff(seed)<gap).any()):raise ValueError("Infeasible declared constraints")
    nodes=np.arange(S*W*K,dtype=np.int64).reshape(S,W,K);n=nodes.size;src=n;snk=n+1
    diff=np.diff(c,axis=2)
    finite=int(np.abs(diff).sum(dtype=np.int64))+2*lam*S*(W-1)*K
    H=finite+1
    if H>=2**30:raise ValueError("Integer capacity cap")
    us=[];vs=[];ws=[]
    def add(a,b,z):
        a,b,z=np.broadcast_arrays(a,b,z);a=a.ravel();b=b.ravel();z=z.ravel()
        good=z>0
        if good.any():us.append(a[good]);vs.append(b[good]);ws.append(z[good].astype(np.int64))
    add(nodes, snk, np.maximum(diff,0))
    add(src,nodes,np.maximum(-diff,0))
    add(nodes[:,:,1:],nodes[:,:,:-1],H)
    if lam:
        add(nodes[:,:-1],nodes[:,1:],lam)
        add(nodes[:,1:],nodes[:,:-1],lam)
    if slope<K:
        add(nodes[:,:-1,slope:],nodes[:,1:,:K-slope],H)
        add(nodes[:,1:,slope:],nodes[:,:-1,:K-slope],H)
    if gap and S>1:
        add(src,nodes[1:,:,gap-1],H)
        add(nodes[:-1,:,K-gap],snk,H)
        if gap<K:add(nodes[:-1,:,:K-gap],nodes[1:,:,gap:],H)
    for k,z in enumerate(seed):
        if z>0:add(src,nodes[k,at,z-1],H)
        if z<K:add(nodes[k,at,z],snk,H)
    g=sparse.coo_matrix((np.concatenate(ws),(np.concatenate(us),np.concatenate(vs))),shape=(n+2,n+2)).tocsr()
    if g.data.max(initial=0)>=2**31:raise ValueError("Parallel edge capacity overflow")
    flow=maximum_flow(g,src,snk,method="dinic")
    if flow.flow_value>=H:raise ArithmeticError("Hard cut arc violated")
    residual=(g-flow.flow).tocsr();residual.data=(residual.data>0).astype(np.int8);residual.eliminate_zeros()
    reachable=breadth_first_order(residual,src,directed=True,return_predecessors=False)
    if snk in reachable:raise ArithmeticError("Residual still connects source to sink")
    used=np.zeros(n+2,bool);used[reachable]=True
    levels=used[nodes]
    if (levels[:,:,1:]&~levels[:,:,:-1]).any():raise ArithmeticError("Nonmonotone thresholds")
    x=levels.sum(2,dtype=np.int64)
    if not np.array_equal(x[:,at],seed) or (np.abs(np.diff(x,axis=1))>slope).any() or (gap and (np.diff(x,axis=0)<gap).any()):raise ArithmeticError("Decoded cut infeasible")
    val=energy(c,x,lam)
    offset=int(c[:,:,0].sum(dtype=np.int64)+np.minimum(diff,0).sum(dtype=np.int64))
    if val!=int(flow.flow_value)+offset:raise ArithmeticError("Decoded objective != flow plus constant")
    return x,{"integer_energy":val,"flow":int(flow.flow_value),"constant":offset,"nodes":n+2,"arcs":int(g.nnz),"hard_capacity":H}

def dtw_map(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    z=dtw(a,b,dist_method="sqeuclidean",step_pattern="symmetric2",
          window_type="sakoechiba",window_args={"window_size":8},keep_internals=False)
    i=np.asarray(z.index1);j=np.asarray(z.index2);m=np.empty(len(a),np.int64)
    for k in range(len(a)):
        q=j[i==k]
        if not len(q):raise ArithmeticError("DTW source index absent from closed alignment")
        m[k]=int(np.floor(np.median(q)+.5))
    if (np.diff(m)<0).any() or (np.abs(m-np.arange(len(m)))>8).any():raise ArithmeticError("DTW path violated contract")
    return m

def dtw_features(plane):
    v=ndi.gaussian_filter1d(plane.astype(np.float64),1,axis=0)
    v=(v-v.mean(0))/np.maximum(v.std(0),1e-6)
    return np.stack((v,.5*np.gradient(v,axis=0)),axis=-1)

def propagate(plane,seed,at):
    f=dtw_features(plane);S=len(seed);W=plane.shape[1];x=np.empty((S,W),np.int64);x[:,at]=seed
    for step in (-1,1):
        for i in range(at+step,-1 if step<0 else W,step):
            j=i-step;m=dtw_map(f[:,j],f[:,i]);x[:,i]=m[x[:,j]]
    return x

def self_test():
    rng=np.random.default_rng(5401);rows=[]
    for i in range(4):
        c=rng.integers(0,13,(2,3,5));seed=np.array([1,3])
        x,r=cut(c,seed,1,lam=3,slope=2,gap=1)
        best=min(energy(c,np.array(a).reshape(2,3),3) for a in itertools.product(range(5),repeat=6)
                 if a[1]==1 and a[4]==3 and min(a[3]-a[0],a[4]-a[1],a[5]-a[2])>=1 and max(abs(a[1]-a[0]),abs(a[2]-a[1]),abs(a[4]-a[3]),abs(a[5]-a[4]))<=2)
        y,r2=cut(c[:,::-1],seed,1,lam=3,slope=2,gap=1)
        rows.append({"oracle_energy":best,"cut_energy":r["integer_energy"],"reflected_energy":r2["integer_energy"],"pass":best==r["integer_energy"]==r2["integer_energy"]})
    c=np.full((2,3,5),20,np.int64);c[0,[0,2],3]=0;c[1,[0,2],1]=0;c[0,1,1]=0;c[1,1,3]=0
    I,_=cut(c,[1,3],1,lam=0,slope=4,gap=0)
    J,_=cut(c,[1,3],1,lam=0,slope=4,gap=1)
    t=np.arange(96,dtype=float)
    s=np.exp(-((t-28)/2.5)**2)+.7*np.exp(-((t-53)/4)**2)-.45*np.exp(-((t-70)/2)**2)+.01*np.sin(t*.37)
    moved=np.interp(t-3,t,s)
    feats=lambda a:np.column_stack([a,.5*np.gradient(a)])
    d=dtw_map(feats(s),feats(s));q=dtw_map(feats(s),feats(moved))
    gates={"brute_force_and_reflection":all(x["pass"] for x in rows),"hard_order_blocks_crossing":bool((np.diff(I,axis=0)<0).any() and (np.diff(J,axis=0)>=1).all()),
           "dtw_identity":bool(np.array_equal(d,np.arange(96))),"dtw_shift":bool(abs(int(q[28])-31)<=1 and abs(int(q[53])-56)<=1)}
    return {"gates":gates,"tiny_cases":rows,"crossing_independent":I.tolist(),"crossing_joint":J.tolist(),
            "dtw_shift_mapped":[int(q[28]),int(q[53])],"pass":all(gates.values())}

def anchor_ids(ray):
    ray=np.asarray(ray);runs=[]
    b=np.r_[0,np.flatnonzero(ray[1:]!=ray[:-1])+1,len(ray)]
    for lo,hi in zip(b[:-1],b[1:]):
        if ray[lo]>0:runs.append((int(ray[lo]),int(lo),int(hi)))
    out=[]
    for gid,lo,hi in runs:
        if hi-lo>=2 and sum(x[0]==gid for x in runs)==1:
            mid=(lo+hi-1)//2
            if DLO<=mid<DHI:out.append((gid,mid))
    bad=set()
    for i in range(len(out)-1):
        if out[i+1][1]-out[i][1]<GAP:bad.update((i,i+1))
    return [(g,z) for i,(g,z) in enumerate(out) if i not in bad]

def unary(plane,seed):
    # Every query is a complete nine-point native column waveform.
    windows=np.lib.stride_tricks.sliding_window_view(plane.astype(np.float64),9,axis=0)
    # (88,W,9) centers 4..91
    a=windows-windows.mean(-1,keepdims=True);norm=np.linalg.norm(a,axis=-1,keepdims=True)
    a=np.divide(a,norm,out=np.zeros_like(a),where=norm>1e-10)
    template=a[np.asarray(seed)-DLO,ANCHOR]
    sim=np.einsum("s k,d w k->s w d",template,a,optimize=True)
    sim=np.clip(sim,-1,1)
    cost=1-sim+.002*np.abs(np.arange(DLO,DHI)[None,None,:]-np.asarray(seed)[:,None,None])
    return np.rint(1000*cost).astype(np.int64)

def evaluation_set(gt,anchors):
    S=len(anchors);W=gt.shape[1]
    truth=np.full((S,W),-1,np.int64)
    allowed=np.abs(np.arange(W)-ANCHOR)>2
    for s,(gid,mid) in enumerate(anchors):
        for w in np.flatnonzero(allowed):
            iz=np.flatnonzero(gt[:,w]==gid)
            if len(iz) and (len(iz)==1 or np.all(np.diff(iz)==1)):truth[s,w]=(int(iz[0])+int(iz[-1]))//2
    return truth,allowed

def score(gt,anchors,pred,truth,allowed):
    if pred.shape!=truth.shape or (pred<0).any() or (pred>=gt.shape[0]).any():raise ValueError("Prediction shape or bounds")
    valid=truth>=0;n=int(valid.sum());total=int(len(anchors)*allowed.sum())
    gid=np.array([x[0] for x in anchors])[:,None]
    picked=gt[pred,np.arange(gt.shape[1])[None,:]]
    err=np.abs(pred[valid]-truth[valid]);match=(picked==gid)&valid;other=(picked>0)&(picked!=gid)&valid
    bkg=(picked==0)&valid
    diff=np.diff(pred,axis=0)[:,allowed];npairs=diff.size
    return {"eligible":n,"planned":total,"eligible_fraction":n/total if total else 0.,
            "near2":int((err<=2).sum()),"correct_id":int(match.sum()),"other_id":int(other.sum()),"background":int(bkg.sum()),
            "absolute_error_sum":int(err.sum()),"within2_rate":float((err<=2).mean()) if n else None,
            "correct_id_rate":int(match.sum())/n if n else None,"wrong_id_rate":int(other.sum())/n if n else None,
            "MAE_vox":float(err.mean()) if n else None,"gap_violations":int((diff<GAP).sum()),"strict_crossings":int((diff<0).sum()),
            "adjacent_pairs":npairs,"prediction_sha256":arrsha(pred)}

def section(plane,gt,axis,index):
    t=time.monotonic();a=anchor_ids(gt[:,ANCHOR])
    out={"fixed_axis":axis,"slice":index,"anchor_count":len(a),"CT_sha256":arrsha(plane),"label_sha256":arrsha(gt),"evaluable":False}
    if len(a)<2:out["reason"]="fewer than two unique seed runs";return out
    seed=np.array([z for g,z in a],np.int64);cost=unary(plane,seed)
    P=cost.argmin(axis=2)+DLO;P[:,ANCHOR]=seed
    I,i=cut(cost,seed-DLO,ANCHOR,lam=LAM,slope=SLOPE,gap=0)
    J,j=cut(cost,seed-DLO,ANCHOR,lam=LAM,slope=SLOPE,gap=GAP)
    W=propagate(plane,seed,ANCHOR)
    truth,allowed=evaluation_set(gt,a)
    out["arms"]={k:score(gt,a,p,truth,allowed) for k,p in (("P_local",P),("I_independent_cut",I+DLO),("J_joint_cut",J+DLO),("W_DTW",W))}
    out["cut_diagnostics"]={"I":i,"J":j}
    v=out["arms"]["P_local"]
    out["evaluable"]=v["eligible"]>=30 and v["eligible_fraction"]>=.5
    if not out["evaluable"]:out["reason"]="insufficient labeled evaluation pairs"
    out["wall_seconds"]=time.monotonic()-t
    return out

def aggregate(rows):
    rr=[r for r in rows if r["evaluable"]]
    keys=("P_local","I_independent_cut","J_joint_cut","W_DTW")
    out={"planned_sections":len(rows),"evaluable_sections":len(rr),"arms":{}}
    for key in keys:
        vals=[r["arms"][key] for r in rr]
        sums={k:sum(v[k] for v in vals) for k in ("eligible","planned","near2","correct_id","other_id","background","absolute_error_sum","gap_violations","strict_crossings","adjacent_pairs")}
        n=sums["eligible"];p=sums["adjacent_pairs"]
        out["arms"][key]={**sums,"within2_rate":sums["near2"]/n if n else None,"correct_id_rate":sums["correct_id"]/n if n else None,
           "wrong_id_rate":sums["other_id"]/n if n else None,"MAE_vox":sums["absolute_error_sum"]/n if n else None,
           "crossing_fraction":sums["strict_crossings"]/p if p else None,"gap_fraction":sums["gap_violations"]/p if p else None}
    out["population_pass"]=len(rr)>=2 and out["arms"]["P_local"]["eligible"]>=500
    if out["population_pass"]:
        P,I,J,W=[out["arms"][k] for k in keys]
        out["J_gates"]={"gain_vs_local":J["within2_rate"]>=P["within2_rate"]+.05,
          "incremental_gain_vs_independent":J["within2_rate"]>=I["within2_rate"]+.03 or J["wrong_id_rate"]<=I["wrong_id_rate"]-.05,
          "correct_id_nonharm":J["correct_id_rate"]>=I["correct_id_rate"]-.03,"zero_gap_violations":J["gap_violations"]==0}
        out["W_gates"]={"gain_vs_local":W["within2_rate"]>=P["within2_rate"]+.05,
          "correct_id_nonharm":W["correct_id_rate"]>=P["correct_id_rate"]-.03,"crossing_fraction":W["crossing_fraction"]<=.01}
    else:out["J_gates"]={"population":False};out["W_gates"]={"population":False}
    return out

def run(a):
    start=time.monotonic()
    r={"schema":"adaptive.ordered_tracking.v1","script_sha256":sha(__file__),"protocol_sha256":sha(HERE/"protocol.md"),
       "prior_053_sha256":sha(a.prior),"sources":{},"source_bytes":0,"cases":[],"pass":False,
       "packages":{k:importlib.metadata.version(k) for k in ("numpy","scipy","dtw-python","pynrrd")}}
    try:
        r["synthetic"]=self_test()
        if not r["synthetic"]["pass"]:raise ValueError("Synthetic solver contracts failed")
        old=module("orthogonal053",HERE.parent/"orthogonal_instances"/"run.py")
        prior=json.loads(a.prior.read_text());old.CAP=100*1024**2
        a.cache.mkdir(parents=True,exist_ok=False)
        with requests.Session() as session:
            for cube in old.CUBES[1:]:
                ct,gt=old.read_cube(session,a.cache,cube,r)
                for kind in ("volume","mask"):
                    if r["sources"][cube][kind]["sha256"]!=prior["sources"][cube][kind]["sha256"]:raise ValueError("053 source hash mismatch")
                if r["sources"][cube]["crop"]["normalized_CT_sha256"]!=prior["sources"][cube]["crop"]["normalized_CT_sha256"]:raise ValueError("053 normalized CT mismatch")
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
        r["joint_pass"]=r["complete"] and all(all(x["aggregate"]["J_gates"].values()) for x in r["cases"])
        r["dtw_pass"]=r["complete"] and all(all(x["aggregate"]["W_gates"].values()) for x in r["cases"])
        r["pass"]=r["joint_pass"] or r["dtw_pass"]
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["wall_seconds"]=time.monotonic()-start;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("cache","output","prior"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
