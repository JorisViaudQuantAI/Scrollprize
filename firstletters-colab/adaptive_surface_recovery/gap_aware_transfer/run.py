#!/usr/bin/env python3
"""Gap-aware path consensus; prospective reference-only selection on two public winners."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,importlib.util,json,pathlib,resource,sys,time
import numpy as np
import requests
from scipy import ndimage as ndi
HERE=pathlib.Path(__file__).resolve().parent
def mod(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
co=mod("consensus061",HERE.parent/"multi_anchor_consensus"/"run.py")
ref=mod("component059",HERE.parent/"dtw_component_transfer"/"run.py")
cross=ref.prev;b=co.b
KEYS=("W","J","K","D","JM","M","S")
CENTER=co.CENTER;EXTRA=co.EXTRA

def attenuate(pr,gap):
    alpha=np.clip((np.asarray(gap,float)-2)/14,0,1)
    return np.rint(pr*alpha[None,:,:,None]).astype(np.int64)

def synthetic():
    h=co.self_test()
    p=np.array([[0,0,30],[31,0,30],[0,31,30],[31,31,30]],float)
    q=np.vstack([p,p+[0,0,12]]);f=np.array([[0,1,3],[0,3,2],[4,5,7],[4,7,6]])
    d,v,rows,counts,n=ref.project_components(q,f,[0,1],2,[0,0],0,f[0])
    d=d[np.argsort(d[:,16,16])]
    err=float(np.max(np.abs(d-np.array([30.,42.])[:,None,None])))
    rng=np.random.default_rng(6301);pr=rng.integers(0,1801,(2,4,4,8),dtype=np.int64)
    gap=rng.integers(2,20,(4,4))
    g={"inherited061":h["pass"],"separate_plane":n==2 and bool(v.all()) and err<=1e-10 and counts["ambiguous_pixels"]==1024,
       "weight_zero":bool((attenuate(pr,np.full((4,4),2))==0).all()),
       "weight_one":np.array_equal(attenuate(pr,np.full((4,4),16)),pr),
       "transpose":np.array_equal(attenuate(pr,gap).transpose(0,2,1,3),attenuate(pr.transpose(0,2,1,3),gap.T))}
    return {"gates":g,"plane_max_error":err,"pass":all(g.values())}

def allowed_mask():
    yy,xx=np.indices((32,32));m=np.ones((32,32),bool)
    for y,x in (CENTER,)+EXTRA:m&=(np.abs(yy-y)>2)|(np.abs(xx-x)>2)
    return m

def prepare_reference(x,f,s):
    d,v,rows,counts,n=ref.reference(x,f,s)
    rec={"components":n,"combined_reference_counts":counts,"component_counts":rows}
    if n!=2 or not v[:,16,16].all():
        rec["reason"]="not two unique central component hits";return None,rec
    order=np.argsort(d[:,16,16]);d=d[order];v=v[order]
    common=v.all(0)&allowed_mask();nr=int(common.sum())
    seed=np.rint(d[:,16,16]).astype(np.int64)
    rec.update(common_rays=nr,common_fraction=nr/int(allowed_mask().sum()))
    if rec["common_fraction"]<.75 or not ((seed>=b.DLO)&(seed<b.DHI)).all() or seed[1]-seed[0]<2:
        rec["reason"]="common support or center depth gate";return None,rec
    truegap=d[1]-d[0]
    if np.any(truegap[common]<2):
        rec["reason"]="reference violates ordered minimum gap";return None,rec
    extra=[];counts=np.zeros(2,np.int64);byray=[]
    for y,z in EXTRA:
        hits=[]
        for i in range(2):
            if v[i,y,z]:
                k=int(np.rint(d[i,y,z]))
                if b.DLO<=k<b.DHI:
                    extra.append((i,y,z,k-b.DLO));counts[i]+=1;hits.append((i,k))
        byray.append((y,z,hits))
    rec["extra_per_component"]=counts.tolist()
    if (counts<2).any():
        rec["reason"]="fewer than two additional anchors per component";return None,rec
    tight=int((truegap[common]<8).sum());wide=int((truegap[common]>=16).sum())
    rec.update(min_gap=float(truegap[common].min()),gap_lt8_rays=tight,gap_ge16_rays=wide,
               stratum="tight" if tight>=100 else ("wide" if tight==0 and wide>=.9*nr else "other"))
    return {"truth":d,"valid":v,"common":common,"truegap":truegap,"seed":seed,"extra":extra,"byray":byray,"selection":s},rec

def new_candidates(P,x,f):
    valid=np.isfinite(P).all(2)&(P>=0).all(2)&(P!=0).any(2)
    cell=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
    interior=ndi.binary_erosion(cell,structure=np.ones((3,3),bool))
    ids=np.flatnonzero(interior[cell])
    if len(ids)<3 or len(f)!=2*cell.sum():raise ValueError("058 face order mismatch")
    for j in range(1,64):
        qt=j/64;fid=int(ids[int(np.floor(qt*(len(ids)-1)))])
        tri=x[f[fid]];n=np.cross(tri[1]-tri[0],tri[2]-tri[0])
        if np.linalg.norm(n)<=1e-12:
            yield qt,None;continue
        axis=int(np.argmax(np.abs(n)))
        yield qt,{"quantile":qt,"face":fid,"center":np.rint(tri.mean(0)).astype(np.int64),"axis":axis,"lateral":[a for a in range(3) if a!=axis]}

def overlaps(s,t):
    return bool((np.abs(s["center"]-t["center"])<96).all())

def select_fresh(P,x,f,previous):
    picked=[];reports=[];seen=set()
    for qt,s in new_candidates(P,x,f):
        row={"quantile":qt,"selected":False}
        if s is None:row["reason"]="degenerate anchor face"
        elif any(overlaps(s,t) for t in previous+picked):row["reason"]="overlaps previously reserved full CT cube"
        else:
            try:
                prep,meta=prepare_reference(x,f,s)
                row["reference"]={k:val for k,val in meta.items() if k not in ("component_counts","combined_reference_counts")}
                if prep is None:row["reason"]=meta["reason"]
                elif meta["stratum"] not in ("tight","wide"):row["reason"]="other spacing stratum"
                elif meta["stratum"] in seen:row["reason"]="stratum already filled"
                else:
                    seen.add(meta["stratum"]);picked.append(s);row["selected"]=True
            except Exception as e:row["reason"]=f"{type(e).__name__}: {str(e)[:200]}"
        reports.append(row)
    return picked,reports

def metrics(pred,d,mask,um):
    if not mask.any():return {"eligible_pairs":0}
    r=ref.score(pred,d,mask,um)
    # Avoid accidentally treating whole-domain slopes as stratum-local slopes.
    return {k:v for k,v in r.items() if k not in ("prediction_sha256","gap_lt2","slopes_gt4")}

def evaluate(reader,prep,meta,um):
    t=time.monotonic()
    s=prep["selection"];c=s["center"];axis=s["axis"];lat=s["lateral"];lo=c-48;hi=c+48
    if (lo<0).any() or (hi>np.asarray(reader.shape)[::-1]).any():raise ValueError("CT cube outside source")
    ct,org,support=reader.read_box(lo[::-1],hi[::-1],return_support=True)
    if tuple(org)!=tuple(lo[::-1]) or ct.shape!=(96,96,96) or not support.all():raise ValueError("Incomplete native support")
    mn,mx=np.percentile(ct,[1,99])
    if mx<=mn:raise ValueError("No CT contrast")
    ct_norm=np.rint(np.clip((ct.astype(np.float32)-mn)*255/(mx-mn),0,255)).astype(np.uint8)
    vol=ct_norm.transpose(2-axis,2-lat[1],2-lat[0])[:,32:64,32:64].copy()
    seed=prep["seed"];extra=prep["extra"];C=b.unary(vol,seed,CENTER)
    cache=co.Tracks(vol);W,_=cache.track(seed,CENTER);Alt,_=cache.track(seed,CENTER,True)
    ps=[[W[i],Alt[i]] for i in range(2)]
    for y,x,hits in prep["byray"]:
        if not hits:continue
        a=np.array([z for _,z in hits],np.int64)
        p,_=cache.track(a,(y,x));q,_=cache.track(a,(y,x),True)
        for j,(i,_) in enumerate(hits):ps[i]+=[p[j],q[j]]
    prior=np.stack([co.prior(paths) for paths in ps],axis=0)
    pred={"W":W};di={}
    def cut(key,cost,ex=()):
        p,z=co.ac.cut3d(cost,seed-b.DLO,CENTER,lam=100,slope=4,gap=2,extra=ex)
        pred[key]=p+b.DLO;di[key]=z
    cut("J",C);cut("K",C+co.prior([W]));cut("D",C+co.prior([W,Alt]))
    cut("JM",C,extra);cut("M",C+prior,extra)
    g=pred["JM"][1]-pred["JM"][0]
    cut("S",C+attenuate(prior,g),extra)
    d=prep["truth"];m=prep["common"];truthgap=prep["truegap"]
    arms={key:ref.score(pred[key],d,m,um) for key in KEYS}
    strata={}
    for name,mask in (("lt8",m&(truthgap<8)),("8to16",m&(truthgap>=8)&(truthgap<16)),("ge16",m&(truthgap>=16))):
        strata[name]={key:metrics(pred[key],d,mask,um) for key in KEYS}
    actual=truthgap[m]<8;hat=g[m]<8;tp=int((actual&hat).sum());fp=int((~actual&hat).sum());fn=int((actual&~hat).sum())
    z=arms["S"];M=arms["M"];jm=arms["JM"]
    r={"quantile":s["quantile"],"pass_population":True,"reference":meta,"arms":arms,"gap_strata":strata,
       "CT_sha256":b.old.arrsha(ct),"normalized_CT_sha256":b.old.arrsha(vol),"support_sha256":b.old.arrsha(support),
       "reference_sha256":b.old.arrsha(np.where(prep["valid"],d,np.nan)),"cut_diagnostics":di,
       "gap_predictor":{"MAE_vox":float(np.abs(g[m]-truthgap[m]).mean()),"true_close":int(actual.sum()),"predicted_close":int(hat.sum()),
         "tp":tp,"fp":fp,"fn_false_safe":fn,"precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,
         "mean_attenuation":float(np.clip((g[m]-2)/14,0,1).mean())},
       "gates":{"within2_M":z["within2"]>=M["within2"]-.02,"within2_JM":z["within2"]>=jm["within2"]-.02,
         "wrong_M":z["wrong_reference_fraction"]<=M["wrong_reference_fraction"]+.03,
         "wrong_JM":z["wrong_reference_fraction"]<=jm["wrong_reference_fraction"]+.03,
         "gap_slope":z["gap_lt2"]==0 and z["slopes_gt4"]==[0,0],
         "extra_exact":bool(all(pred["S"][i,y,x]==z0+b.DLO for i,y,x,z0 in extra))},
       "gain_M":z["within2"]-M["within2"],"gain_without_more_wrong":z["within2"]>=M["within2"]+.03 and z["wrong_reference_fraction"]<=M["wrong_reference_fraction"],
       "wall_seconds":time.monotonic()-t}
    return r

def summarize(rows):
    ok=[r for r in rows if r.get("pass_population")]
    if not ok:return {"assessable":0}
    macro={k:{m:float(np.mean([r["arms"][k][m] for r in ok])) for m in ("within2","MAE_vox","wrong_reference_fraction")} for k in KEYS}
    strata={}
    for name in ("lt8","8to16","ge16"):
        sub={}
        for k in KEYS:
            a=[r["gap_strata"][name][k] for r in ok];n=sum(q["eligible_pairs"] for q in a)
            sub[k]={"pairs":n}
            if n:sub[k].update(within2=sum(q["near2"] for q in a if q["eligible_pairs"])/n,wrong_reference_fraction=sum(q["wrong_reference_hits"] for q in a if q["eligible_pairs"])/n,MAE_vox=sum(q["MAE_vox"]*q["eligible_pairs"] for q in a if q["eligible_pairs"])/n)
        strata[name]=sub
    return {"assessable":len(ok),"macro":macro,"pooled_gap_strata":strata}

def run(a):
    t=time.monotonic()
    r={"schema":"adaptive.gap_aware_transfer.v1","script_sha256":b.digest(__file__),"protocol_sha256":b.digest(HERE/"protocol.md"),
       "consensus061_sha256":b.digest(HERE.parent/"multi_anchor_consensus"/"run.py"),"prior058_sha256":b.digest(a.prior),"cases":[],"pass":False}
    sys.path.insert(0,str(a.gemini));import grow_m7_patch as gm
    budget=gm.ByteBudget(200*1024**2)
    try:
        r["synthetic"]=synthetic()
        if not r["synthetic"]["pass"]:raise ValueError("Synthetic assertion failure")
        before=json.loads(a.prior.read_text())
        with requests.Session() as session:
            for name,vid,um in cross.SCROLLS:
                row={"scroll":name,"fresh":[],"repeated":[]}
                try:
                    p=next(c for c in before["cases"] if c["scroll"]==name)
                    P,uv,x,f,row["mesh"]=cross.mesh(session,name)
                    for n,v in row["mesh"]["objects"].items():
                        if v["sha256"]!=p["mesh"]["objects"][n]["sha256"]:raise ValueError("058 mesh changed")
                    previous=cross.select(P,f,x)
                    fresh,reports=select_fresh(P,x,f,previous);row["selection_receipt"]=reports
                    row["fresh_selected"]=len(fresh)
                    root,row["CT_source"]=cross.find_volume(session,name,vid,um,budget)
                    if any(row["CT_source"][k]!=p["CT_source"][k] for k in ("root","zattrs_sha256","zarray_sha256")):raise ValueError("058 CT source changed")
                    with gm.BudgetedRemoteZarrV2(root,level="0",budget=budget,workers=1,cache_mb=64,max_retries=1,timeout=25) as reader:
                        for group,ss in (("repeated",[s for s in previous if s["quantile"] in ref.TARGETS[name]]),("fresh",fresh)):
                            for s in ss:
                                try:
                                    prep,meta=prepare_reference(x,f,s)
                                    if prep is None:z={"quantile":s["quantile"],"pass_population":False,"reference":meta}
                                    else:z=evaluate(reader,prep,meta,um)
                                except Exception as e:z={"quantile":s["quantile"],"pass_population":False,"failure":f"{type(e).__name__}: {str(e)[:300]}"}
                                row[group].append(z)
                                print("ROI_JSON="+json.dumps({"scroll":name,"group":group,**z},separators=(",",":"),allow_nan=False),flush=True)
                except Exception as e:row["failure"]=f"{type(e).__name__}: {str(e)[:400]}"
                r["cases"].append(row)
        fresh=[{**z,"scroll":q["scroll"]} for q in r["cases"] for z in q["fresh"]]
        repeated=[{**z,"scroll":q["scroll"]} for q in r["cases"] for z in q["repeated"]]
        ok=[z for z in fresh if z.get("pass_population")]
        r["fresh_summary"]=summarize(fresh);r["repeated_summary"]=summarize(repeated)
        tight=[z for z in ok if z["reference"]["stratum"]=="tight"];wide=[z for z in ok if z["reference"]["stratum"]=="wide"]
        st=r["fresh_summary"].get("pooled_gap_strata",{}).get("lt8",{})
        tightgate=bool(st.get("S",{}).get("pairs",0) and st["S"]["within2"]>=st["M"]["within2"]+.03 and st["S"]["within2"]>=st["JM"]["within2"]-.02)
        r["gates"]={"both_scrolls":len({z["scroll"] for z in ok})==2,"two_fresh_tight":len(tight)>=2,"two_fresh_wide":len(wide)>=2,
          "all_fresh_complete":len(fresh)==4 and len(ok)==4,"all_roi_gates":len(ok)==4 and all(all(z["gates"].values()) for z in ok),
          "some_tight_gain":any(z["gain_without_more_wrong"] for z in tight),"pooled_fresh_tight_gate":tightgate}
        r["pass"]=all(r["gates"].values())
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["CT_budget"]={"bytes":budget.used,"requests":budget.requests,"cap":budget.cap};r["wall_seconds"]=time.monotonic()-t;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("output","gemini","prior"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
