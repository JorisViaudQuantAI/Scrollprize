#!/usr/bin/env python3
"""Actual Gemini RGT on manual instance IDs, matched anchors, observed pitch."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,hashlib,importlib.util,inspect,json,pathlib,resource,sys,time
import numpy as np
import requests
from scipy import ndimage as ndi
HERE=pathlib.Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location("consensus061",HERE.parent/"multi_anchor_consensus"/"run.py")
co=importlib.util.module_from_spec(sp);sys.modules[sp.name]=co;sp.loader.exec_module(co)
b=co.b;old=co.old;data=co.data
CENTER=co.CENTER;EXTRA=co.EXTRA
KEYS=("JM","M","R0","R5","RC")
def instrument_pitch(rgt):
    src=inspect.getsource(rgt.estimate_local_wrap_wavenumber_3d)
    needle="    lam = np.where(good, lam, med)"
    end="    return np.ascontiguousarray(k), info"
    if src.count(needle)!=1 or src.count(end)!=1:raise ValueError("Unexpected upstream pitch implementation")
    src=src.replace(needle,"    raw_spacing_observed = lam.copy()\n"+needle)
    src=src.replace(end,end+", {'raw': np.moveaxis(raw_spacing_observed, 0, int(normal_axis)), 'good': np.moveaxis(good, 0, int(normal_axis)), 'conf': np.moveaxis(conf, 0, int(normal_axis))}")
    scope=dict(vars(rgt));exec(compile(src,"<observed upstream pitch>","exec"),scope)
    return scope["estimate_local_wrap_wavenumber_3d"]

def allowed_mask():
    y,x=np.indices((32,32));m=np.ones((32,32),bool)
    for yy,xx in (CENTER,)+EXTRA:m&=(np.abs(y-yy)>2)|(np.abs(x-xx)>2)
    return m

def safe_div(a,b):return float(a/b) if b else None

def assign(dist,ids):
    d=np.where(np.isfinite(dist),dist,np.inf)
    j=np.argsort(d,axis=0,kind="stable");first=np.take_along_axis(d,j[:1],axis=0)[0]
    sec=np.take_along_axis(d,j[1:2],axis=0)[0] if len(ids)>1 else np.full_like(first,np.inf)
    accept=np.isfinite(first)&(first<=3)&((sec>=first+2)|~np.isfinite(sec))
    labels=np.where(accept,np.asarray(ids)[j[0]],0)
    return labels,accept,np.isfinite(d).sum(0)

def assignment_metrics(dist,ids,gt):
    labels,accept,nfinite=assign(dist,ids)
    seeded=np.isin(gt,ids);correct=accept&(labels==gt);wrong=accept&~correct
    return {"queries":len(gt),"seeded":int(seeded.sum()),"unseeded":int((~seeded).sum()),"accepted":int(accept.sum()),"rejected":int((~accept).sum()),
        "seeded_accepted":int((seeded&accept).sum()),"seeded_correct":int((seeded&correct).sum()),
        "seeded_correct_fraction":safe_div((seeded&correct).sum(),seeded.sum()),
        "seeded_coverage":safe_div((seeded&accept).sum(),seeded.sum()),
        "wrong_accepted":int(wrong.sum()),"wrong_among_accepted":safe_div(wrong.sum(),accept.sum()),
        "unseeded_false_accept":int((~seeded&accept).sum()),"unseeded_false_accept_fraction":safe_div((~seeded&accept).sum(),(~seeded).sum()),
        "queries_one_finite":int((nfinite==1).sum()),"queries_zero_finite":int((nfinite==0).sum())}

def score_height(gt,anchors,h,truth):
    m=truth>=0;n=int(m.sum());ok=np.isfinite(h)&(h>=0)&(h<=95);v=m&ok
    vals=np.where(ok,np.rint(h),0).astype(np.int64)
    yy,xx=np.indices(gt.shape[1:]);got=gt[vals,yy,xx];ids=np.array([g for g,_ in anchors])[:,None,None]
    e=np.abs(h-truth);near=int((v&(e<=2)).sum())
    right=int((v&(got==ids)).sum());other=int((v&(got>0)&(got!=ids)).sum());bg=int((v&(got==0)).sum());nvalid=int(v.sum())
    return {"eligible":n,"valid":nvalid,"invalid":n-nvalid,"coverage":safe_div(nvalid,n),"near2":near,"within2_rate":safe_div(near,n),
      "correct_id":right,"other_id":other,"background":bg,"correct_id_rate":safe_div(right,n),"wrong_id_rate":safe_div(other,n),
      "MAE_vox_conditional":float(e[v].mean()) if nvalid else None,"absolute_error_sum_conditional":float(e[v].sum()),
      "slope_violations_axes":[int((np.abs(np.diff(h,axis=a))>4).sum()) for a in (1,2)],
      "gap_violations":int((np.diff(h,axis=0)<2).sum()),"prediction_sha256":old.arrsha(h)}

def runs(ray):
    ends=np.r_[0,np.flatnonzero(np.diff(ray.astype(np.int64))!=0)+1,len(ray)]
    rr=[(int(ray[i]),int(i),int(j)) for i,j in zip(ends[:-1],ends[1:]) if ray[i]>0]
    count={}
    for gid,_,_ in rr:count[gid]=count.get(gid,0)+1
    return [(gid,i,j,(i+j-1)//2,count[gid]==1 and j-i>=2) for gid,i,j in rr]

def observations(gt,allowed,ids):
    counts={int(k):int(v) for k,v in zip(*np.unique(gt[gt>0],return_counts=True))}
    points=[];offset=[];pairs=[]
    for y,x in zip(*np.nonzero(allowed)):
        rr=runs(gt[:,y,x])
        for gid,lo,hi,z,good in rr:
            if not good or counts[gid]<64:continue
            points.append((z,y,x,gid))
            for zz in (z-1,z+1):
                if lo<=zz<hi:offset.append((zz,y,x,gid))
        for a,c in zip(rr[:-1],rr[1:]):
            if a[4] and c[4] and a[0] in ids and c[0] in ids:
                pairs.append(((a[3]+c[3])//2,y,x,c[3]-a[3],ids.index(a[0]),ids.index(c[0])))
    return np.asarray(points,np.int64).reshape(-1,4),np.asarray(offset,np.int64).reshape(-1,4),np.asarray(pairs,np.int64).reshape(-1,6)

def pitch_metrics(pred,valid,truth):
    v=valid&np.isfinite(pred);close=truth<8;wide=~close;hit=v&(pred<8);safe=v&(pred>=8)
    tp=int((close&hit).sum());tn=int((wide&safe).sum());fp=int((wide&hit).sum());fs=int((close&safe).sum())
    n=int(v.sum());nclose=int(close.sum());nwide=int(wide.sum())
    return {"pairs":len(truth),"true_close":nclose,"true_wide":nwide,"valid":n,"unknown_close":int((close&~v).sum()),"unknown_wide":int((wide&~v).sum()),
      "coverage":safe_div(n,len(truth)),"close_coverage":safe_div((close&v).sum(),nclose),
      "tp":tp,"tn":tn,"fp":fp,"false_safe":fs,"sensitivity_all":safe_div(tp,nclose),"specificity_all":safe_div(tn,nwide),"precision":safe_div(tp,tp+fp),
      "absolute_error_sum_clipped":float(np.abs(pred[v]-np.clip(truth[v],4,20)).sum()),"absolute_error_sum_unclipped":float(np.abs(pred[v]-truth[v]).sum()),
      "MAE_clipped_conditional":float(np.abs(pred[v]-np.clip(truth[v],4,20)).mean()) if n else None,
      "MAE_unclipped_conditional":float(np.abs(pred[v]-truth[v]).mean()) if n else None}

def pitch_pool(rows):
    keys=("pairs","true_close","true_wide","valid","unknown_close","unknown_wide","tp","tn","fp","false_safe","absolute_error_sum_clipped","absolute_error_sum_unclipped")
    s={k:sum(r[k] for r in rows) for k in keys}
    s.update(coverage=safe_div(s["valid"],s["pairs"]),close_coverage=safe_div(s["true_close"]-s["unknown_close"],s["true_close"]),
       sensitivity_all=safe_div(s["tp"],s["true_close"]),specificity_all=safe_div(s["tn"],s["true_wide"]),precision=safe_div(s["tp"],s["tp"]+s["fp"]),
       MAE_clipped_conditional=safe_div(s["absolute_error_sum_clipped"],s["valid"]),MAE_unclipped_conditional=safe_div(s["absolute_error_sum_unclipped"],s["valid"]))
    return s

def synthetic(rgt,observed):
    z=np.arange(48)[:,None,None];y,x=np.indices((12,12))
    v=np.broadcast_to(100+25*np.cos(2*np.pi*(z+.1*y)/8),(48,12,12)).astype(np.float32)
    k,info=rgt.estimate_local_wrap_wavenumber_3d(v,0)
    k2,i2,aux=observed(v,0)
    ph=np.broadcast_to(np.arange(48)[:,None,None]*.5,(48,12,12))
    h=rgt._heights_at_phase(ph,12.)
    distances=np.array([[0.,1.,4.,np.nan,0.],[9.,1.5,0.,np.nan,4.]])
    y,a,n=assign(distances,[1,2])
    p=pitch_metrics(np.array([5.,12.,5.,np.nan]),np.array([True,True,True,False]),np.array([5,5,12,5]))
    gates={"pitch_instrumentation_exact":np.array_equal(k,k2) and info==i2,
       "plane_phase_inversion":bool(np.allclose(h,24.)),"identity_rejection":np.array_equal(y,[1,0,2,0,1]),
       "unknown_pitch_denominator":p["tp"]==1 and p["false_safe"]==1 and p["unknown_close"]==1 and p["fp"]==1,
       "inherited061":co.self_test()["pass"]}
    return {"gates":{k:bool(v) for k,v in gates.items()},"pass":all(gates.values())}

def case(ct,gt,prior,rgt,observed):
    t=time.monotonic()
    normal=int(prior["normal_axis"])
    v=np.moveaxis(ct,normal,0)[:,32:64,32:64].copy();g=np.moveaxis(gt,normal,0)[:,32:64,32:64].copy()
    if old.arrsha(v)!=prior["CT_sha256"] or old.arrsha(g)!=prior["label_sha256"]:raise ValueError("061 array hash mismatch")
    a=old.anchor_ids(g[:,CENTER[0],CENTER[1]]);seed=np.array([z for _,z in a],np.int64);ids=[int(i) for i,_ in a]
    C=b.unary(v,seed,CENTER);extra=[];byray=[];anchors=[[(*CENTER,int(z))] for z in seed]
    for y,x in EXTRA:
        hits=[]
        for s,(gid,_) in enumerate(a):
            z=co.run_anchor(g[:,y,x],gid)
            if z is not None:extra.append((s,y,x,z-b.DLO));hits.append((s,z));anchors[s].append((y,x,z))
        byray.append((y,x,hits))
    allowed=allowed_mask();truth=b.truth_for(g,a,allowed);m=truth>=0
    if len(a)!=prior["anchor_count"] or len(extra)!=prior["anchors"]["extra_observations"] or int(m.sum())!=prior["eligible"]:raise ValueError("061 population mismatch")
    # Reuse exact 061 implementation to reconstruct only the matched anchor arms.
    cache=co.Tracks(v);W,E=cache.track(seed,CENTER);Alt,_=cache.track(seed,CENTER,True)
    ps=[[W[i],Alt[i]] for i in range(len(seed))]
    for y,x,hits in byray:
        if not hits:continue
        zz=np.array([z for _,z in hits],np.int64)
        p,_=cache.track(zz,(y,x));q,_=cache.track(zz,(y,x),True)
        for j,(i,_) in enumerate(hits):ps[i]+=[p[j],q[j]]
    prior_M=np.stack([co.prior(p) for p in ps])
    preds={};diagnostics={}
    def cut(key,cost):
        p,di=co.ac.cut3d(cost,seed-b.DLO,CENTER,lam=100,slope=4,gap=2,extra=extra)
        preds[key]=p+b.DLO;diagnostics[key]=di
    cut("JM",C);cut("M",C+prior_M)
    for k in ("JM","M"):
        if old.arrsha(preds[k])!=prior["arms"][k]["prediction_sha256"]:raise ValueError("061 prediction hash mismatch: "+k)
    nrm,plan,preferred=rgt.estimate_structure_tensor_dip_3d(v)
    nrm=nrm*np.where(nrm[...,0]<0,-1.,1.)[...,None]
    k,pitch_info,raw=observed(v,0);spacing=2*np.pi/k
    solved=rgt.solve_dip_guided_rgt_3d(v,nrm,plan,k,normal_axis=0)
    ph=solved["phi"]
    if not np.isfinite(ph).all() or (np.diff(ph,axis=0)<=0).any():raise ValueError("RGT phase not finite strictly monotone")
    levels0=ph[seed,CENTER[0],CENTER[1]]
    levels5=np.array([np.median([ph[z,y,x] for y,x,z in loc]) for loc in anchors])
    residuals=np.array([abs(float(ph[z,y,x])-levels5[i]) for i,loc in enumerate(anchors) for y,x,z in loc])
    for name,levels in (("R0",levels0),("R5",levels5)):
        preds[name]=np.stack([rgt._heights_at_phase(ph,float(z)) for z in levels])
    hh=preds["R5"];dd=np.arange(b.DLO,b.DHI,dtype=np.float64)
    diff=np.abs(hh[...,None]-dd)
    penalty=np.rint(150*np.where(np.isfinite(diff),np.minimum(diff,12),0)).astype(np.int64)
    cut("RC",C+penalty)
    arms={k:score_height(g,a,preds[k],truth) for k in KEYS}
    # Exact integer baselines use same score arithmetic, with no candidate-specific masks.
    for k0 in ("JM","M"):
        for key in ("eligible","near2","correct_id","other_id","gap_violations"):
            if arms[k0][key]!=prior["arms"][k0][key]:raise ValueError("061 metric mismatch")
    pts,offset,pairs=observations(g,allowed,ids)
    assignments={}
    for name,pop in (("centers",pts),("offset1",offset)):
        z,y,x,gtid=pop.T
        q={}
        for key in KEYS:
            q[key]=assignment_metrics(np.abs(preds[key][:,y,x]-z[None]),ids,gtid)
        for key,levels in (("phase0",levels0),("phase5",levels5)):
            dist=np.abs(ph[z,y,x][None]-levels[:,None])/np.maximum(k[z,y,x][None],1e-9)
            q[key]=assignment_metrics(dist,ids,gtid)
        assignments[name]=q
    zz,yy,xx,gtgap,i,j=pairs.T
    pitches={
      "raw":pitch_metrics(raw["raw"][zz,yy,xx],raw["good"][zz,yy,xx],gtgap),
      "filled":pitch_metrics(spacing[zz,yy,xx],np.ones(len(pairs),bool),gtgap),
      "JM":pitch_metrics(preds["JM"][j,yy,xx]-preds["JM"][i,yy,xx],np.ones(len(pairs),bool),gtgap)}
    rc,ma,jm,r5=[arms[k] for k in ("RC","M","JM","R5")]
    gates={"CG_converged":bool(solved["info"]["cg_converged"]),"R5_coverage":r5["coverage"]>=.95,
      "R5_levels_ordered":bool((np.diff(levels5)>0).all()),"RC_within_M":rc["within2_rate"]>=ma["within2_rate"]-.02,
      "RC_within_JM":rc["within2_rate"]>=jm["within2_rate"]-.02,"RC_ID_M":rc["correct_id_rate"]>=ma["correct_id_rate"]-.03,
      "RC_wrong_M":rc["wrong_id_rate"]<=ma["wrong_id_rate"]+.03,
      "RC_constraints":rc["gap_violations"]==0 and rc["slope_violations_axes"]==[0,0] and all(preds["RC"][i,y,x]==z+b.DLO for i,y,x,z in extra)}
    qa,qm=assignments["centers"]["RC"],assignments["centers"]["M"]
    def diff_ok(x,y,tol,larger=False):return x is not None and y is not None and (x>=y-tol if larger else x<=y+tol)
    labelg={"seeded_correct":diff_ok(qa["seeded_correct_fraction"],qm["seeded_correct_fraction"],.03,True),
       "wrong_accepted":diff_ok(qa["wrong_among_accepted"],qm["wrong_among_accepted"],.02),
       "unseeded_false_accept":diff_ok(qa["unseeded_false_accept_fraction"],qm["unseeded_false_accept_fraction"],.03)}
    return {"CT_sha256":old.arrsha(v),"label_sha256":old.arrsha(g),"evaluable":True,"seed_count":len(seed),"extra_count":len(extra),
       "pitch_info":pitch_info,"RGT_preferred_axis":preferred,"forced_depth_axis":0,"normal_planarity_median":float(np.median(plan)),
       "rgt_info":solved["info"],"phase_levels0":levels0.tolist(),"phase_levels5":levels5.tolist(),
       "anchor_phase_abs_residual_p50_p95_max":np.quantile(residuals,[.5,.95,1]).tolist(),"arms":arms,"cut_diagnostics":diagnostics,
       "assignment":assignments,"pitch":pitches,"gates":gates,"label_gates":labelg,"RC_gain_M":rc["within2_rate"]-ma["within2_rate"],
       "wall_seconds":time.monotonic()-t}

def run(a):
    t=time.monotonic()
    r={"schema":"adaptive.rgt_identity_bridge.v1","script_sha256":b.digest(__file__),"protocol_sha256":b.digest(HERE/"protocol.md"),
       "prior061_sha256":b.digest(a.prior),"sources":{},"source_bytes":0,"cases":[],"pass":False}
    sys.path.insert(0,str(a.gemini));import seismic_rgt_unwrapper as rgt
    r["upstream_sha256"]=b.digest(a.gemini/"seismic_rgt_unwrapper.py")
    observed=instrument_pitch(rgt)
    try:
        r["synthetic"]=synthetic(rgt,observed)
        if not r["synthetic"]["pass"]:raise ValueError("Frozen synthetic checks failed")
        prev=json.loads(a.prior.read_text());data.CAP=160*1024**2;a.cache.mkdir(parents=True,exist_ok=False)
        with requests.Session() as session:
            for cube in co.CUBES:
                try:
                    ct,gt=data.read_cube(session,a.cache,cube,r)
                    if any(r["sources"][cube][kind]["sha256"]!=prev["sources"][cube][kind]["sha256"] for kind in ("volume","mask")):raise ValueError("061 input source mismatch")
                    prior=next(c for c in prev["cases"] if c["cube"]==cube)
                    row=case(ct,gt,prior,rgt,observed)
                except Exception as e:row={"evaluable":False,"failure":f"{type(e).__name__}: {str(e)[:400]}"}
                row["cube"]=cube;r["cases"].append(row);print("CASE_JSON="+json.dumps(row,separators=(",",":"),allow_nan=False),flush=True)
        ok=[q for q in r["cases"] if q.get("evaluable")]
        r["gates"]={"complete":len(ok)==4,"all_geometry":len(ok)==4 and all(all(q["gates"].values()) for q in ok),
          "some_RC_gain":any(q["RC_gain_M"]>=.03 for q in ok)}
        r["pass"]=all(r["gates"].values())
        r["label_pass"]=len(ok)==4 and all(all(q["label_gates"].values()) for q in ok)
        r["pitch_pool"]={k:pitch_pool([q["pitch"][k] for q in ok]) for k in ("raw","filled","JM")}
        p=r["pitch_pool"]["raw"]
        r["pitch_pass"]=len(ok)==4 and p["true_close"]>=100 and p["coverage"]>=.7 and p["close_coverage"]>=.7 and p["sensitivity_all"]>=.7 and p["specificity_all"]>=.7
        r["macro"]={k:{m:float(np.mean([q["arms"][k][m] for q in ok])) for m in ("coverage","within2_rate","correct_id_rate","wrong_id_rate","MAE_vox_conditional")} for k in KEYS} if ok else {}
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["wall_seconds"]=time.monotonic()-t;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("cache","output","prior","gemini"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
