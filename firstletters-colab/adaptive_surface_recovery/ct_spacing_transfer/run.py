#!/usr/bin/env python3
"""Train CT compression evidence on Paris4 and test fixed predicted-midpoint queries on 0211/0343."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,importlib.util,json,pathlib,resource,sys,time
import numpy as np
import requests
from scipy import ndimage as ndi,signal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score
HERE=pathlib.Path(__file__).resolve().parent
def mod(n,p):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
r64=mod("identity064",HERE.parent/"rgt_identity_bridge"/"run.py")
g63=mod("gap063",HERE.parent/"gap_aware_transfer"/"run.py")
co=r64.co;b=r64.b;old=r64.old;data=r64.data;cross=g63.cross
NAMES=["filled_pitch","observed_pitch_or_zero","pitch_good","pitch_conf","planarity","abs_normal_depth"]+[f"ac_lag_{d}" for d in range(3,21)]+["gradient_energy","second_derivative_energy","peak_count","min_peak_gap","median_peak_gap"]
def waveform_features(p):
    p=np.asarray(p,np.float64)
    v=(p-p.mean(-1,keepdims=True))/(p.std(-1,keepdims=True)+1e-6)
    s=ndi.gaussian_filter1d(v,1,axis=-1,mode="reflect")
    h=s-ndi.gaussian_filter1d(s,4,axis=-1,mode="reflect")
    out=[]
    for lag in range(3,21):
        a=h[...,:-lag];c=h[...,lag:]
        out.append((a*c).sum(-1)/(np.sqrt((a*a).sum(-1)*(c*c).sum(-1))+1e-8))
    en=(s*s).mean(-1)+1e-8
    out.extend([(np.diff(s,axis=-1)**2).mean(-1)/en,(np.diff(s,n=2,axis=-1)**2).mean(-1)/en])
    peaks=[]
    for row in s:
        pk,_=signal.find_peaks(row,prominence=.5);gap=np.diff(pk)
        peaks.append([len(pk),float(gap.min()) if len(gap) else 0.,float(np.median(gap)) if len(gap) else 0.])
    return np.column_stack(out+list(np.asarray(peaks).T))
def features(vol,points,rgt,observed):
    z,y,x=points.T
    nrm,plan,pref=rgt.estimate_structure_tensor_dip_3d(vol)
    k,info,aux=observed(vol,0)
    spacing=2*np.pi/k;good=aux["good"][z,y,x]
    local=ndi.uniform_filter(vol.astype(np.float64),size=(1,3,3),mode="nearest")
    pad=np.pad(local,((24,24),(0,0),(0,0)),mode="reflect")
    profile=pad[z[:,None]+np.arange(49)[None],y[:,None],x[:,None]]
    X=np.column_stack([spacing[z,y,x],np.where(good,aux["raw"][z,y,x],0),good.astype(float),
        aux["conf"][z,y,x],plan[z,y,x],np.abs(nrm[z,y,x,0]),waveform_features(profile)])
    if X.shape!=(len(points),len(NAMES)) or not np.isfinite(X).all():raise ValueError("Feature shape/finite mismatch")
    return X,{"preferred_axis":int(pref),"upstream_pitch":info}
def jm(vol,seed,extra):
    C=b.unary(vol,seed,co.CENTER)
    p,diag=co.ac.cut3d(C,seed-b.DLO,co.CENTER,lam=100,slope=4,gap=2,extra=extra)
    return p+b.DLO,diag
def train_case(ct,gt,prior,rgt,observed):
    ax=int(prior["normal_axis"]);v=np.moveaxis(ct,ax,0)[:,32:64,32:64].copy();g=np.moveaxis(gt,ax,0)[:,32:64,32:64].copy()
    if old.arrsha(v)!=prior["CT_sha256"] or old.arrsha(g)!=prior["label_sha256"]:raise ValueError("061 crop mismatch")
    a=old.anchor_ids(g[:,co.CENTER[0],co.CENTER[1]]);seed=np.array([z for _,z in a],np.int64);ids=[int(i) for i,_ in a]
    extra=[]
    for y,x in co.EXTRA:
        for s,(gid,_) in enumerate(a):
            z=co.run_anchor(g[:,y,x],gid)
            if z is not None:extra.append((s,y,x,z-b.DLO))
    pred,_=jm(v,seed,extra)
    if old.arrsha(pred)!=prior["arms"]["JM"]["prediction_sha256"]:raise ValueError("061 JM hash mismatch")
    _,_,pairs=r64.observations(g,r64.allowed_mask(),ids)
    if not len(pairs):return None,{"pairs":0,"reason":"no adjacent uniquely labeled seeded pairs"}
    _,y,x,truth,i,j=pairs.T
    gap=pred[j,y,x]-pred[i,y,x]
    z=(pred[j,y,x]+pred[i,y,x])//2
    X,info=features(v,np.column_stack([z,y,x]),rgt,observed)
    return (X,gap,truth),{"pairs":len(truth),"close":int((truth<8).sum()),"feature_sha256":old.arrsha(X),"predicted_gap_sha256":old.arrsha(gap),"pitch_info":info}
def summary(y,p,valid=None):
    y=np.asarray(y,bool);p=np.asarray(p,float)
    ok=np.isfinite(p) if valid is None else valid&np.isfinite(p)
    close=ok&(p>=.5);wide=ok&(p<.5)
    tp=int((y&close).sum());tn=int((~y&wide).sum());fp=int((~y&close).sum());fs=int((y&wide).sum())
    n=int(y.sum());nw=int((~y).sum())
    div=r64.safe_div
    return {"pairs":len(y),"true_close":n,"true_wide":nw,"valid":int(ok.sum()),"coverage":div(ok.sum(),len(y)),
       "tp":tp,"tn":tn,"fp":fp,"false_safe":fs,"unknown_close":int((y&~ok).sum()),"unknown_wide":int((~y&~ok).sum()),
       "sensitivity":div(tp,n),"specificity":div(tn,nw),"precision":div(tp,tp+fp),
       "AUC":float(roc_auc_score(y[ok],p[ok])) if np.unique(y[ok]).size==2 else None,
       "AP":float(average_precision_score(y[ok],p[ok])) if np.unique(y[ok]).size==2 else None,
       "brier":float(((p[ok]-y[ok])**2).mean()) if ok.any() else None}
def predictions(X,gap,models):
    a={"JM":(gap<8).astype(float),"filled":(X[:,0]<8).astype(float),"raw":np.where(X[:,2]>0,(X[:,1]<8).astype(float),np.nan)}
    for key,(sc,lr) in models.items():
        q=X if key=="L" else np.column_stack([X,gap/20])
        a[key]=lr.predict_proba(sc.transform(q))[:,1]
    return a
def fit(X,gap,y):
    models={};meta={}
    for key,q in (("L",X),("LC",np.column_stack([X,gap/20]))):
        sc=StandardScaler().fit(q);lr=LogisticRegression(C=1,class_weight="balanced",random_state=6501,max_iter=500).fit(sc.transform(q),y)
        models[key]=(sc,lr)
        meta[key]={"mean":sc.mean_.tolist(),"scale":sc.scale_.tolist(),"coef":lr.coef_.ravel().tolist(),"intercept":lr.intercept_.tolist(),"iterations":lr.n_iter_.tolist()}
    return models,meta
def target_case(reader,prep,oldrow,models,rgt,observed):
    s=prep["selection"];c=s["center"];ax=s["axis"];lat=s["lateral"];lo=c-48;hi=c+48
    if (lo<0).any() or (hi>np.asarray(reader.shape)[::-1]).any():raise ValueError("CT box out of bounds")
    ct,org,sup=reader.read_box(lo[::-1],hi[::-1],return_support=True)
    if tuple(org)!=tuple(lo[::-1]) or ct.shape!=(96,96,96) or not sup.all():raise ValueError("Native support incomplete")
    if old.arrsha(ct)!=oldrow["CT_sha256"] or old.arrsha(np.where(prep["valid"],prep["truth"],np.nan))!=oldrow["reference_sha256"]:raise ValueError("063 CT/reference hash mismatch")
    mn,mx=np.percentile(ct,[1,99])
    if mx<=mn:raise ValueError("No contrast")
    v=np.rint(np.clip((ct.astype(np.float32)-mn)*255/(mx-mn),0,255)).astype(np.uint8).transpose(2-ax,2-lat[1],2-lat[0])[:,32:64,32:64].copy()
    p,_=jm(v,prep["seed"],prep["extra"])
    if old.arrsha(p)!=oldrow["arms"]["JM"]["prediction_sha256"]:raise ValueError("063 JM hash mismatch")
    y,x=np.indices((32,32));z=(p[0]+p[1])//2;gap=(p[1]-p[0]).ravel()
    X,info=features(v,np.column_stack([z.ravel(),y.ravel(),x.ravel()]),rgt,observed)
    allp=predictions(X,gap,models)  # predictions computed before reading the reference gap
    mask=prep["common"].ravel()
    truth=prep["truegap"].ravel()[mask]<8
    if len(truth)*2!=oldrow["arms"]["JM"]["eligible_pairs"]:raise ValueError("063 population mismatch")
    ps={k:v[mask] for k,v in allp.items()}
    return truth,ps,{"pairs":len(truth),"close":int(truth.sum()),"query_sha256":old.arrsha(z),"feature_sha256":old.arrsha(X),"pitch_info":info,
        "arms":{k:summary(truth,v) for k,v in ps.items()}}
def synthetic(rgt,obs):
    z=np.arange(48)[:,None,None];y,x=np.indices((12,12))
    v=np.broadcast_to(100+25*np.cos(2*np.pi*(z+.1*y)/8),(48,12,12)).astype(np.float32)
    k,i=rgt.estimate_local_wrap_wavenumber_3d(v,0);q,j,_=obs(v,0)
    p=np.stack([np.sin(np.arange(49)/1.4)*20,np.cos(np.arange(49)/2.2)*13])
    a=waveform_features(p);c=waveform_features(3*p+50)
    test=summary(np.array([True,True,False,False]),np.array([1,np.nan,0,1.]))
    gates={"pitch_exact":np.array_equal(k,q) and i==j,"affine_profile":bool(np.allclose(a,c,atol=1e-5,rtol=0)),
        "unknown_denominator":test["tp"]==1 and test["tn"]==1 and test["fp"]==1 and test["unknown_close"]==1}
    return {"gates":gates,"max_affine_diff":float(np.max(np.abs(a-c))),"pass":all(gates.values())}
def run(a):
    start=time.monotonic()
    r={"schema":"adaptive.ct_spacing_transfer.v1","script_sha256":b.digest(__file__),"protocol_sha256":b.digest(HERE/"protocol.md"),
       "prior061_sha256":b.digest(a.prior061),"prior063_sha256":b.digest(a.prior063),"sources":{},"source_bytes":0,"train":[],"cases":[],"pass":False,"feature_names":NAMES}
    sys.path.insert(0,str(a.gemini));import seismic_rgt_unwrapper as rgt;import grow_m7_patch as gm
    obs=r64.instrument_pitch(rgt);r["upstream_sha256"]=b.digest(a.gemini/"seismic_rgt_unwrapper.py")
    budget=gm.ByteBudget(180*1024**2)
    try:
        r["synthetic"]=synthetic(rgt,obs)
        if not r["synthetic"]["pass"]:raise ValueError("Synthetic contract failed")
        p61=json.loads(a.prior061.read_text());p63=json.loads(a.prior063.read_text())
        data.CAP=180*1024**2;a.cache.mkdir(parents=True,exist_ok=False)
        train=[]
        with requests.Session() as session:
            for cube in co.CUBES:
                ct,gt=data.read_cube(session,a.cache,cube,r)
                if any(r["sources"][cube][k]["sha256"]!=p61["sources"][cube][k]["sha256"] for k in ("volume","mask")):raise ValueError("061 source mismatch")
                row=next(q for q in p61["cases"] if q["cube"]==cube)
                q,meta=train_case(ct,gt,row,rgt,obs);r["train"].append({"cube":cube,**meta})
                if q is not None:train.append(q)
            X=np.concatenate([q[0] for q in train]);gap=np.concatenate([q[1] for q in train]);y=np.concatenate([q[2] for q in train])<8
            models,r["model"]=fit(X,gap,y)
            r["training_counts"]={"pairs":len(y),"close":int(y.sum()),"wide":int((~y).sum())}
            r["train_descriptive"]={k:summary(y,v) for k,v in predictions(X,gap,models).items()}
            r["train_feature_sha256"]=old.arrsha(X)
            budget.cap-=r["source_bytes"]
            if budget.cap<=0:raise ValueError("Combined byte budget exhausted")
            scored=[]
            for name,vid,um in cross.SCROLLS:
                prev=next(q for q in p63["cases"] if q["scroll"]==name)
                P,uv,xyz,f,mesh=cross.mesh(session,name)
                if any(v["sha256"]!=prev["mesh"]["objects"][n]["sha256"] for n,v in mesh["objects"].items()):raise ValueError("063 mesh mismatch")
                oldsel=cross.select(P,f,xyz);fresh,_=g63.select_fresh(P,xyz,f,oldsel)
                if [s["quantile"] for s in fresh]!=[q["quantile"] for q in prev["fresh"]]:raise ValueError("063 selection mismatch")
                root,meta=cross.find_volume(session,name,vid,um,budget)
                if any(meta[k]!=prev["CT_source"][k] for k in ("root","zattrs_sha256","zarray_sha256")):raise ValueError("063 CT metadata mismatch")
                with gm.BudgetedRemoteZarrV2(root,level="0",budget=budget,workers=1,cache_mb=64,max_retries=1,timeout=25) as reader:
                    for group,ss in (("fresh",fresh),("repeated",[s for s in oldsel if s["quantile"] in g63.ref.TARGETS[name]])):
                        for s in ss:
                            out={"scroll":name,"group":group,"quantile":s["quantile"]}
                            try:
                                prep,me=g63.prepare_reference(xyz,f,s)
                                if prep is None:raise ValueError("063 reference gate")
                                o=next(q for q in prev[group] if q["quantile"]==s["quantile"])
                                truth,ps,m=target_case(reader,prep,o,models,rgt,obs)
                                out.update(m);scored.append((name,group,truth,ps));out["complete"]=True
                            except Exception as e:out["failure"]=f"{type(e).__name__}: {str(e)[:400]}"
                            r["cases"].append(out);print("CASE_JSON="+json.dumps(out,separators=(",",":"),allow_nan=False),flush=True)
        def pool(rows):
            y=np.concatenate([q[2] for q in rows])
            return {k:summary(y,np.concatenate([q[3][k] for q in rows])) for k in rows[0][3]}
        for group in ("fresh","repeated"):
            rows=[q for q in scored if q[1]==group]
            r[group+"_pool"]=pool(rows) if rows else {}
        r["fresh_scroll"]={n:pool([q for q in scored if q[1]=="fresh" and q[0]==n]) for n in {q[0] for q in scored if q[1]=="fresh"}}
        F=r["fresh_pool"];R=r["repeated_pool"]
        if F and R:
            L=F["L"];Lr=R["L"]
            r["gates"]={"complete":len(scored)==6 and len(r["train"])==4,"close_population":L["true_close"]>=100,
              "fresh_sensitivity":L["sensitivity"] is not None and L["sensitivity"]>=.7,
              "fresh_specificity":L["specificity"] is not None and L["specificity"]>=.7,
              "each_scroll_wide_specificity":len(r["fresh_scroll"])==2 and all(q["L"]["specificity"] is not None and q["L"]["specificity"]>=.7 for q in r["fresh_scroll"].values()),
              "false_safe_half_JM":L["false_safe"]<=.5*F["JM"]["false_safe"],
              "repeated_sensitivity":Lr["sensitivity"] is not None and Lr["sensitivity"]>=.6,
              "repeated_specificity":Lr["specificity"] is not None and Lr["specificity"]>=.6}
            r["pass"]=all(r["gates"].values())
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["CT_budget"]={"bytes":budget.used,"requests":budget.requests,"cap_remaining_after_training":budget.cap};r["wall_seconds"]=time.monotonic()-start;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("cache","output","prior061","prior063","gemini"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
