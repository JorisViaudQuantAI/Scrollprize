#!/usr/bin/env python3
"""Two native local crossings, original mesh incidence reference; frozen 057 solver."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,hashlib,importlib.util,json,pathlib,resource,sys,time
import numpy as np
import requests
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("cross058",HERE.parent/"dtw_cross_scroll"/"run.py")
prev=importlib.util.module_from_spec(spec);sys.modules[spec.name]=prev;spec.loader.exec_module(prev)
solver=prev.solver;top=prev.top
KEYS=("P_local","I_3D","J_3D","W_DTW","R_projection","K_fixed","H_reciprocal","Hrow","W_alternate")
TARGETS={"PHerc0211":(.5,),"PHerc0343":(.2,.5)}

def score(p,truth,mask,um):
    e=np.abs(p[:,mask]-truth[:,mask])
    wrong=np.abs(p[:,mask]-truth[::-1,mask])
    d=e.ravel();n=d.size
    if not n:raise ValueError("No eligible reference pairs")
    return {"eligible_pairs":n,"within2":float((e<=2).mean()),"near2":int((e<=2).sum()),
      "MAE_vox":float(d.mean()),"MAE_um":float(d.mean()*um),"p95_vox":float(np.quantile(d,.95)),
      "wrong_reference_hits":int(((wrong<e)&(wrong<=2)).sum()),"wrong_reference_fraction":float(((wrong<e)&(wrong<=2)).mean()),
      "other_nearer":int((wrong<e).sum()),"other_nearer_fraction":float((wrong<e).mean()),
      "no_near_reference":int(((e>2)&(wrong>2)).sum()),
      "gap_lt2":int((np.diff(p,axis=0)<2).sum()),"slopes_gt4":[int((np.abs(np.diff(p,axis=a))>4).sum()) for a in (1,2)],
      "prediction_sha256":solver.old.arrsha(p)}
def project_components(x,faces,lat,axis,latlo,depthlo,seed_face):
    # Global orientation is determined by the original anchor face, never by component.
    a=x[seed_face][:,lat]
    det=np.linalg.det(np.stack((a[1]-a[0],a[2]-a[0])))
    if abs(det)<=1e-12:raise ValueError("Degenerate projected seed")
    faces=faces.copy()
    if det<0:faces=faces[:,[0,2,1]]
    used=np.unique(faces);mp=np.full(len(x),-1,np.int64);mp[used]=np.arange(len(used))
    combined=top.rasterize_triangles(x[used][:,lat],x[used],mp[faces],(32,32),origin_xy=latlo)
    labels,nc=top._face_components(mp[faces],len(used))
    rows=[];depth=[];valid=[]
    for k in range(nc):
        f=faces[labels==k];u=np.unique(f);m=np.full(len(x),-1,np.int64);m[u]=np.arange(len(u))
        r=top.rasterize_triangles(x[u][:,lat],x[u],m[f],(32,32),origin_xy=latlo)
        d=r["xyz"][...,axis]-depthlo
        v=r["valid"]&np.isfinite(d)&(d>=0)&(d<=95)
        depth.append(d);valid.append(v);rows.append({"faces":len(f),"counts":r["counts"],"central_valid":bool(v[16,16])})
    return np.array(depth),np.array(valid),rows,combined["counts"],nc
def reference(x,f,s):
    c=s["center"];axis=s["axis"];lat=s["lateral"];tri=x[f]
    lo=c-48;hi=c+47;latlo=c[lat]-16;lathi=c[lat]+15
    take=((tri.max(1)>=lo)&(tri.min(1)<=hi)).all(1)&((tri[:,:,lat].max(1)>=latlo)&(tri[:,:,lat].min(1)<=lathi)).all(1)
    if not take.any():raise ValueError("No preselected faces")
    return project_components(x,f[take],lat,axis,latlo,lo[axis],f[s["face"]])
def self_test():
    p=np.array([[0,0,30],[31,0,30],[0,31,30],[31,31,30]],float)
    q=np.vstack([p,p+[0,0,12]]);f=np.array([[0,1,3],[0,3,2],[4,5,7],[4,7,6]])
    d,v,rows,counts,n=project_components(q,f,[0,1],2,[0,0],0,f[0])
    order=np.argsort(d[:,16,16]);d=d[order];v=v[order];m=v.all(0)
    truth=np.rint(d).astype(int);a=score(truth,d,m,1.);b=score(truth[::-1],d,m,1.)
    oracle=solver.self_test()
    g={"two_components":n==2,"all_combined_ambiguous":counts["ambiguous_pixels"]==1024 and counts["valid_pixels"]==0,
       "separate_exact_depths":bool(m.all() and np.all(d[0]==30) and np.all(d[1]==42)),
       "correct_vs_swapped":a["within2"]==1 and a["wrong_reference_fraction"]==0 and b["within2"]==0 and b["wrong_reference_fraction"]==1,
       "solver":oracle["pass"]}
    return {"gates":g,"solver057":oracle,"pass":all(g.values())}
def evaluate(reader,x,f,s,um,oldref):
    t=time.monotonic()
    d,v,rows,counts,n=reference(x,f,s)
    r={"quantile":s["quantile"],"component_count":n,"components":rows,"combined_reference_counts":counts,"pass_population":False}
    if counts!=oldref["reference_counts"]:raise ValueError("058 reference counts changed")
    if n!=2 or not v[:,16,16].all():
        r["reason"]="Exactly two unique central component hits required";return r
    order=np.argsort(d[:,16,16]);d=d[order];v=v[order]
    yy,xx=np.indices((32,32));allowed=(np.abs(yy-16)>2)|(np.abs(xx-16)>2)
    m=v.all(0)&allowed
    n=int(m.sum());seed=np.rint(d[:,16,16]).astype(np.int64)
    r.update(reference_rays=n,reference_pairs=2*n,reference_fraction=n/int(allowed.sum()),
             component_order=order.tolist(),per_component_eligible=[int((a&allowed).sum()) for a in v])
    if n:r["minimum_reference_gap_vox"]=float(np.min(d[1,m]-d[0,m]))
    if n/int(allowed.sum())<.75 or not ((seed>=solver.DLO)&(seed<solver.DHI)).all() or seed[1]-seed[0]<2 or (n and np.any(d[1,m]-d[0,m]<2)):
        r["reason"]="Fixed common reference/depth/order gate failed";return r
    c=s["center"];axis=s["axis"];lat=s["lateral"];lo=c-48;hi=c+48
    if (lo<0).any() or (hi>np.asarray(reader.shape)[::-1]).any():raise ValueError("CT cube out of bounds")
    ct,org,support=reader.read_box(lo[::-1],hi[::-1],return_support=True)
    if tuple(org)!=tuple(lo[::-1]) or ct.shape!=(96,96,96) or not support.all():raise ValueError("Incomplete stored CT")
    mn,mx=np.percentile(ct,[1,99])
    if mx<=mn:raise ValueError("No CT contrast")
    ct_norm=np.rint(np.clip((ct.astype(np.float32)-mn)*255/(mx-mn),0,255)).astype(np.uint8)
    vol=ct_norm.transpose(2-axis,2-lat[1],2-lat[0])[:,32:64,32:64].copy()
    anchor=(16,16);C=solver.unary(vol,seed,anchor)
    P=C.argmin(-1)+solver.DLO;P[:,16,16]=seed
    W,e=solver.track(vol,seed,anchor);alt,_=solver.track(vol,seed,anchor,True)
    dist=np.abs(np.arange(solver.DLO,solver.DHI)[None,None,None,:]-W[...,None]);cap=np.minimum(dist,12)
    H=C+np.rint(150*cap/(1+e[...,None]/4)).astype(np.int64)
    pred={"P_local":P,"W_DTW":W,"W_alternate":alt};diag={}
    for k,cost,lam,gap,axes in (("I_3D",C,100,0,(1,2)),("J_3D",C,100,2,(1,2)),("R_projection",1000*dist,0,2,(1,2)),("K_fixed",C+150*cap,100,2,(1,2)),("H_reciprocal",H,100,2,(1,2)),("Hrow",H,100,2,(2,))):
        p,q=solver.cut3d(cost,seed-solver.DLO,anchor,lam=lam,slope=4,gap=gap,axes=axes)
        pred[k]=p+solver.DLO;diag[k]=q
    arms={k:score(pred[k],d,m,um) for k in KEYS};w=arms["W_DTW"];j=arms["J_3D"];h=arms["H_reciprocal"]
    r.update(pass_population=True,CT_sha256=solver.old.arrsha(ct),normalized_sha256=solver.old.arrsha(vol),
        support_sha256=solver.old.arrsha(support),reference_depth_sha256=solver.old.arrsha(np.where(v,d,np.nan)),reference_valid_sha256=solver.old.arrsha(v),
        percentiles=[float(mn),float(mx)],arms=arms,cut_diagnostics=diag,
        path_dependence={"changed_fraction":float((W!=alt).mean()),"depth_delta_p50_p95_max":np.quantile(np.abs(W-alt),[.5,.95,1]).tolist()},
        gates={"gap":h["gap_lt2"]==0,"slope":h["slopes_gt4"]==[0,0],"within2_W":h["within2"]>=w["within2"]-.02,
               "within2_J":h["within2"]>=j["within2"]-.02,"MAE_W":h["MAE_vox"]<=w["MAE_vox"]+.25,
               "wrong_W":h["wrong_reference_fraction"]<=w["wrong_reference_fraction"]+.03,
               "wrong_J":h["wrong_reference_fraction"]<=j["wrong_reference_fraction"]+.03},
        H_gain_W=h["within2"]-w["within2"],H_gain_J=h["within2"]-j["within2"],wall_seconds=time.monotonic()-t)
    return r
def macro(rows):
    rows=[r for r in rows if r.get("pass_population")]
    if not rows:return {}
    return {k:{m:float(np.mean([r["arms"][k][m] for r in rows])) for m in ("within2","MAE_vox","MAE_um","wrong_reference_fraction","other_nearer_fraction")} for k in KEYS}
def run(a):
    t=time.monotonic()
    r={"schema":"adaptive.dtw_component_transfer.v1","script_sha256":solver.digest(__file__),"protocol_sha256":solver.digest(HERE/"protocol.md"),
       "cross058_sha256":solver.digest(HERE.parent/"dtw_cross_scroll"/"run.py"),"prior058_sha256":solver.digest(a.prior),"cases":[],"pass":False}
    sys.path.insert(0,str(a.gemini));import grow_m7_patch as gm
    budget=gm.ByteBudget(120*1024**2)
    try:
        r["synthetic"]=self_test()
        if not r["synthetic"]["pass"]:raise ValueError("Frozen synthetic gate failed")
        before=json.loads(a.prior.read_text())
        with requests.Session() as session:
            for name,vid,um in prev.SCROLLS:
                q={"scroll":name,"ROIs":[]}
                try:
                    b=next(c for c in before["cases"] if c["scroll"]==name)
                    P,uv,x,f,q["mesh"]=prev.mesh(session,name)
                    for n,obj in q["mesh"]["objects"].items():
                        if obj["sha256"]!=b["mesh"]["objects"][n]["sha256"]:raise ValueError("058 mesh hash mismatch")
                    ss=[s for s in prev.select(P,f,x) if s["quantile"] in TARGETS[name]]
                    root,q["CT_source"]=prev.find_volume(session,name,vid,um,budget)
                    if any(q["CT_source"][k]!=b["CT_source"][k] for k in ("root","zattrs_sha256","zarray_sha256")):raise ValueError("058 CT metadata mismatch")
                    with gm.BudgetedRemoteZarrV2(root,level="0",budget=budget,workers=1,cache_mb=64,max_retries=1,timeout=25) as reader:
                        for s in ss:
                            old=next(z for z in b["ROIs"] if z["quantile"]==s["quantile"])
                            if old.get("pass_population") or "CT_sha256" in old:raise ValueError("Target already scored in 058")
                            try:row=evaluate(reader,x,f,s,um,old)
                            except Exception as e:row={"quantile":s["quantile"],"pass_population":False,"failure":f"{type(e).__name__}: {str(e)[:400]}"}
                            q["ROIs"].append(row)
                    q["descriptive_macro"]=macro(q["ROIs"])
                except Exception as e:q["failure"]=f"{type(e).__name__}: {str(e)[:400]}"
                r["cases"].append(q);print("CASE_JSON="+json.dumps(q,separators=(",",":"),allow_nan=False),flush=True)
        rows=[q for c in r["cases"] for q in c["ROIs"]]
        r["complete"]=len(rows)==3 and all(q.get("pass_population",False) for q in rows)
        r["gates"]={"all_population":r["complete"],"all_H_gates":r["complete"] and all(all(q["gates"].values()) for q in rows),
                    "some_gain_vs_both":any(q.get("H_gain_W",-1)>=.03 and q.get("H_gain_J",-1)>=.03 for q in rows)}
        r["pass"]=all(r["gates"].values())
        r["descriptive_macro"]=macro(rows)
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["CT_budget"]={"bytes":budget.used,"requests":budget.requests,"cap":budget.cap};r["wall_seconds"]=time.monotonic()-t;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("output","gemini","prior"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
