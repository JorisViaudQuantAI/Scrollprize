#!/usr/bin/env python3
"""Sparse-anchor target-sheet recovery and selective public fitted-carrier mosaic."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,importlib.util,json,pathlib,resource,sys,time
import numpy as np
import requests
from scipy import ndimage as ndi
HERE=pathlib.Path(__file__).resolve().parent
def mod(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
atlas=mod("atlas067",HERE.parent/"claude_carrier_atlas"/"run.py")
gap=mod("gap063",HERE.parent/"gap_aware_transfer"/"run.py")
co=gap.co;b=co.b;cross=gap.cross;top=cross.top
CENTER=co.CENTER;EXTRA=co.EXTRA
KEYS=("C0","P","JM","M")
def div(a,b):return float(a/b) if b else None
def mask():
    y,x=np.indices((32,32));m=np.ones((32,32),bool)
    for yy,xx in (CENTER,)+EXTRA:m&=(np.abs(y-yy)>2)|(np.abs(x-xx)>2)
    return m
def project_fit(mesh,s):
    c=s["center"];ax=s["axis"];lat=s["lateral"];lo=c-48;hi=c+47;latlo=c[lat]-16;lathi=c[lat]+15
    tri=mesh.xyz[mesh.faces]
    take=((tri.max(1)>=lo)&(tri.min(1)<=hi)).all(1)&((tri[:,:,lat].max(1)>=latlo)&(tri[:,:,lat].min(1)<=lathi)).all(1)
    f=mesh.faces[take].copy()
    if not len(f):return np.full((32,32),np.nan),{"faces":0,"valid":0,"reason":"no candidate projection"}
    uv=mesh.xyz[f][:,:,lat];u=uv[:,1]-uv[:,0];v=uv[:,2]-uv[:,0];sign=float(np.sum(u[:,0]*v[:,1]-u[:,1]*v[:,0]))
    if abs(sign)<=1e-12:return np.full((32,32),np.nan),{"faces":len(f),"valid":0,"reason":"indeterminate orientation"}
    if sign<0:f=f[:,[0,2,1]]
    used=np.unique(f);m=np.full(len(mesh.xyz),-1,np.int64);m[used]=np.arange(len(used))
    r=top.rasterize_triangles(mesh.xyz[used][:,lat],mesh.xyz[used],m[f],(32,32),origin_xy=latlo)
    d=r["xyz"][...,ax]-lo[ax];v=r["valid"]&np.isfinite(d)&(d>=0)&(d<=95)
    return np.where(v,d,np.nan),{"faces":len(f),"valid":int(v.sum()),"global_projection_sign":1 if sign>0 else -1,"counts":r["counts"]}
def select(w,meshes):
    chosen=[];receipts=[]
    for q,s in gap.new_candidates(w.P,w.xyz,w.faces):
        row={"quantile":q,"selected":False}
        if len(chosen)>=6:row["reason"]="six predeclared eligible fields reserved"
        elif s is None:row["reason"]="invalid anchor face"
        elif any(gap.overlaps(s,t["sel"]) for t in chosen):row["reason"]="overlap with reserved full cube"
        else:
            try:
                d,v,st,nf=cross.reference(w.xyz,w.faces,s)
                common=v&mask();n=int(common.sum())
                row["reference_fraction"]=n/int(mask().sum())
                xy=(CENTER,)+EXTRA
                if row["reference_fraction"]<.90 or not all(v[y,x] and b.DLO<=int(np.rint(d[y,x]))<b.DHI for y,x in xy):
                    row["reason"]="unique reference/anchor support gate"
                else:
                    candidates=[];pr=[]
                    for m in meshes:
                        z,r=project_fit(m,s);candidates.append(z);pr.append(r)
                    H=np.stack(candidates)
                    row["candidate_projection_counts"]=pr
                    row["two_candidate_fraction"]=float((np.isfinite(H).sum(0)[common]>=2).mean())
                    if row["two_candidate_fraction"]<.75:row["reason"]="insufficient common multi-carrier support"
                    else:
                        row["selected"]=True
                        chosen.append({"sel":s,"truth":d,"valid":v,"common":common,"candidate":H,
                                       "seed":np.array([int(np.rint(d[CENTER]))]),"extra":[(0,y,x,int(np.rint(d[y,x]))-b.DLO) for y,x in EXTRA],
                                       "reference_counts":st})
            except Exception as e:row["reason"]=f"{type(e).__name__}: {str(e)[:300]}"
        receipts.append(row)
    return chosen,receipts
def choose(pred,candidates):
    dist=np.where(np.isfinite(candidates),np.abs(candidates-pred[None]),np.inf)
    order=np.argsort(dist,axis=0,kind="stable")
    first=np.take_along_axis(dist,order[:1],axis=0)[0];second=np.take_along_axis(dist,order[1:2],axis=0)[0]
    accept=np.isfinite(first)&(first<=3)&((second>=first+2)|~np.isfinite(second))
    label=np.where(accept,order[0]+1,0).astype(np.uint8)
    h=np.where(accept,np.take_along_axis(candidates,order[:1],axis=0)[0],np.nan)
    return label,h
def plane(seed,extra):
    pts=[(CENTER[0],CENTER[1],int(seed[0]))]+[(y,x,z+b.DLO) for _,y,x,z in extra]
    q=np.array([[1,y,x] for y,x,_ in pts]);z=np.array([z for _,_,z in pts])
    coef=np.linalg.lstsq(q,z,rcond=None)[0];y,x=np.indices((32,32))
    out=np.clip(np.rint(coef[0]+coef[1]*y+coef[2]*x),b.DLO,b.DHI-1).astype(np.int64)
    for yy,xx,v in pts:out[yy,xx]=v
    return out
def predict(v,seed,extra):
    C=b.unary(v,seed,CENTER);cache=co.Tracks(v);W,_=cache.track(seed,CENTER);Alt,_=cache.track(seed,CENTER,True)
    paths=[W[0],Alt[0]]
    for _,y,x,z in extra:
        p,_=cache.track(np.array([z+b.DLO]),(y,x));q,_=cache.track(np.array([z+b.DLO]),(y,x),True);paths.extend([p[0],q[0]])
    M=co.prior(paths)[None]
    out={"C0":np.full((32,32),int(seed[0]),np.int64),"P":plane(seed,extra)};di={}
    for k,cost in (("JM",C),("M",C+M)):
        p,r=co.ac.cut3d(cost,seed-b.DLO,CENTER,lam=100,slope=4,gap=0,extra=extra);out[k]=p[0]+b.DLO;di[k]=r
    for k in ("JM","M"):
        if any(out[k][y,x]!=z+b.DLO for _,y,x,z in extra) or out[k][CENTER]!=seed[0]:raise ArithmeticError("Anchor violated")
    # Constant comparator uses central only; force all five anchors for equal supplied inputs.
    for _,y,x,z in extra:out["C0"][y,x]=z+b.DLO
    return out,di
def oracle(c,d,m):
    delta=np.where(np.isfinite(c)&m[None],np.abs(c-d[None]),np.inf)
    q=np.argsort(delta,axis=0,kind="stable")
    a=np.take_along_axis(delta,q[:1],axis=0)[0];sec=np.take_along_axis(delta,q[1:2],axis=0)[0]
    near=m&(a<=3);unique=near&((sec>=a+2)|~np.isfinite(sec))
    return near,np.where(unique,q[0]+1,0)
def score(pred,label,h,c,d,m,onear,olab):
    n=int(m.sum());a=m&(label>0);near=a&(np.abs(h-d)<=3);far=a&(np.abs(h-d)>=6);mid=a&~near&~far
    same=m&(olab>0)&(label>0);oracle_n=int(onear.sum());nacc=int(a.sum())
    correct=int(near.sum());jumps=[];sameedges=0;slope=[0,0]
    lab=np.where(m,label,0)
    for ax in (0,1):
        l=[slice(None)]*2;r=l.copy();l[ax]=slice(None,-1);r[ax]=slice(1,None);l=tuple(l);r=tuple(r)
        valid=(lab[l]>0)&(lab[r]>0);diff=np.abs(h[l]-h[r])
        seam=valid&(lab[l]!=lab[r]);jumps.extend(diff[seam].tolist())
        sameedges+=int((valid&~seam).sum());slope[ax]=int((valid&(diff>4)).sum())
    cell=np.stack([lab[:-1,:-1],lab[1:,:-1],lab[:-1,1:],lab[1:,1:]])
    full=(cell>0).all(0);samecell=full&(cell==cell[:1]).all(0)
    comp={}
    for k in range(1,5):
        ll,nc=ndi.label(lab==k);ss=np.bincount(ll.ravel())[1:]
        comp[atlas.WINDS[k-1]]={"selected":int((lab==k).sum()),"components":int(nc),"largest":int(ss.max(initial=0))}
    return {"reference_rays":n,"oracle_near":oracle_n,"accepted":nacc,"near":correct,"far":int(far.sum()),"intermediate":int(mid.sum()),
            "accepted_coverage":nacc/n,"near_coverage":correct/n,"near_precision":div(correct,nacc),"far_fraction":div(far.sum(),nacc),
            "oracle_recovery":div(correct,oracle_n),"unique_oracle_comparable":int(same.sum()),"unique_oracle_agreement":div(((label==olab)&same).sum(),same.sum()),
            "target_within2":float((np.abs(pred[m]-d[m])<=2).mean()),"target_MAE_vox":float(np.abs(pred[m]-d[m]).mean()),
            "chosen_depth_MAE_vox":float(np.abs(h[a]-d[a]).mean()) if nacc else None,
            "predicted_slope_violations":[int((np.abs(np.diff(pred,axis=ax))>4).sum()) for ax in (0,1)],
            "mosaic_slope_violations":slope,"cross_carrier_seam_edges":len(jumps),"seam_step_q50_q95_max":np.quantile(jumps,[.5,.95,1]).tolist() if jumps else None,
            "same_carrier_neighbor_edges":sameedges,"complete_accepted_cells":int(full.sum()),"same_carrier_complete_cells":int(samecell.sum()),"components":comp,
            "selected_map":atlas.pack(lab)}
def vertex_labels(mesh,s,p,d,valid):
    c=s["center"];ax=s["axis"];lat=s["lateral"];lo=c-48
    ids=mesh.used;x=mesh.xyz[ids]
    # rows are second lateral axis; columns are first; 32 native rays.
    col=x[:,lat[0]]-(c[lat[0]]-16);row=x[:,lat[1]]-(c[lat[1]]-16);z=x[:,ax]-lo[ax]
    keep=(row>=0)&(row<31)&(col>=0)&(col<31)&(z>=0)&(z<=95)
    ids=ids[keep];row=row[keep];col=col[keep];z=z[keep]
    y0=np.floor(row).astype(int);x0=np.floor(col).astype(int);fy=row-y0;fx=col-x0
    # Every interpolation corner must be common, held-out reference support.
    ok=valid[y0,x0]&valid[y0+1,x0]&valid[y0,x0+1]&valid[y0+1,x0+1]
    ids=ids[ok];y0=y0[ok];x0=x0[ok];fy=fy[ok];fx=fx[ok];z=z[ok]
    def interp(a):
        return (1-fy)*((1-fx)*a[y0,x0]+fx*a[y0,x0+1])+fy*((1-fx)*a[y0+1,x0]+fx*a[y0+1,x0+1])
    actual=np.abs(z-interp(d))<=3;est=np.abs(z-interp(p))<=3;far=np.abs(z-interp(d))>=6
    label=np.zeros(len(mesh.xyz),np.uint8);label[ids]=np.where(est,2,1).astype(np.uint8)
    return {"eligible_vertices":len(ids),"predicted_near":int(est.sum()),"reference_near":int(actual.sum()),"true_near":int((est&actual).sum()),
       "false_near_far_reference":int((est&far).sum()),"precision":div((est&actual).sum(),est.sum()),"recall":div((est&actual).sum(),actual.sum()),
       "map":atlas.pack(label.reshape(mesh.shape))},label.reshape(mesh.shape)
def synthetic():
    c=np.full((4,1,5),np.nan);c[0,0]=[10,10,14,10,np.nan];c[1,0]=[15,11,np.nan,20,np.nan]
    p=np.full((1,5),10.);l,h=choose(p,c)
    pp=plane(np.array([20]),[(0,y,x,20-b.DLO) for y,x in EXTRA])
    g={"ray_intersection":atlas.self_test()["pass"],"anchor_solver":co.self_test()["pass"],
      "selector_near_tie_missing_and_far":np.array_equal(l,[[1,0,0,1,0]]),"constant_plane":bool((pp==20).all())}
    return {"gates":{k:bool(v) for k,v in g.items()},"pass":all(g.values())}
def run(a):
    t=time.monotonic();r={"schema":"adaptive.claude_ct_mosaic.v1","script_sha256":b.digest(__file__),"protocol_sha256":b.digest(HERE/"protocol.md"),
     "atlas067_code_sha256":b.digest(HERE.parent/"claude_carrier_atlas"/"run.py"),"source_commit":atlas.PIN,"cases":[],"pass":False}
    sys.path.insert(0,str(a.gemini));import grow_m7_patch as gm
    budget=gm.ByteBudget(120*1024**2);arr={}
    try:
        r["synthetic"]=synthetic()
        if not r["synthetic"]["pass"]:raise ValueError("Frozen synthetic checks failed")
        w,src=atlas.load_winner();r["winner_source"]=src
        meshes=[];r["fit_sources"]={}
        for name in atlas.WINDS:
            m,_,rec=atlas.load_fit(a.claude,"flat",name);meshes.append(m);r["fit_sources"][name]=rec
        selected,recs=select(w,meshes);r["selection_receipt"]=recs;r["selected"]=len(selected)
        print("SELECTION_JSON="+json.dumps({"selected":len(selected),"quantiles":[q["sel"]["quantile"] for q in selected]},separators=(",",":")),flush=True)
        with requests.Session() as session:
            root,r["CT_source"]=cross.find_volume(session,"PHerc0211","20250821151803",9.362,budget)
            prior=json.loads((HERE.parent/"dtw_cross_scroll"/"result.json").read_text());p=next(q for q in prior["cases"] if q["scroll"]=="PHerc0211")
            if any(r["CT_source"][k]!=p["CT_source"][k] for k in ("root","zarray_sha256","zattrs_sha256")):raise ValueError("058 native CT source changed")
            with gm.BudgetedRemoteZarrV2(root,level="0",budget=budget,workers=1,cache_mb=64,max_retries=1,timeout=25) as reader:
                for i,prep in enumerate(selected):
                    s=prep["sel"];q={"quantile":s["quantile"],"normal_axis_xyz":s["axis"],"complete":False}
                    try:
                        c=s["center"];ax=s["axis"];lat=s["lateral"];lo=c-48;hi=c+48
                        if (lo<0).any() or (hi>np.asarray(reader.shape)[::-1]).any():raise ValueError("CT cube out of bounds")
                        ct,org,sup=reader.read_box(lo[::-1],hi[::-1],return_support=True)
                        if tuple(org)!=tuple(lo[::-1]) or ct.shape!=(96,96,96) or not sup.all():raise ValueError("Incomplete CT support")
                        mn,mx=np.percentile(ct,[1,99])
                        if mx<=mn:raise ValueError("No CT contrast")
                        v=np.rint(np.clip((ct.astype(np.float32)-mn)*255/(mx-mn),0,255)).astype(np.uint8).transpose(2-ax,2-lat[1],2-lat[0])[:,32:64,32:64].copy()
                        pred,diag=predict(v,prep["seed"],prep["extra"])
                        d=prep["truth"];m=prep["common"];cand=prep["candidate"];onear,olab=oracle(cand,d,m)
                        q.update(CT_sha256=b.old.arrsha(ct),reference_sha256=b.old.arrsha(np.where(prep["valid"],d,np.nan)),candidate_sha256=b.old.arrsha(cand),
                                 reference_rays=int(m.sum()),oracle_near=int(onear.sum()),cut_diagnostics=diag,arms={},vertices={})
                        for k in KEYS:
                            lab,h=choose(pred[k],cand)
                            q["arms"][k]=score(pred[k],lab,h,cand,d,m,onear,olab)
                            arr[f"roi{i}_{k}_labels"]=np.where(m,lab,0);arr[f"roi{i}_{k}_height"]=np.where(m,h,np.nan)
                        for k in ("JM","M"):
                            q["vertices"][k]={}
                            for name,mesh in zip(atlas.WINDS,meshes):
                                z,mp=vertex_labels(mesh,s,pred[k],d,m);q["vertices"][k][name]=z
                                arr[f"roi{i}_{k}_{name}_vertex"]=mp
                        for k,z in (("reference",d),("support",m),("carrier_heights",cand),("center_xyz",c),("lateral_axes",np.array(lat)),("normal_axis",np.array(ax))):arr[f"roi{i}_{k}"]=z
                        M=q["arms"]["M"];J=q["arms"]["JM"]
                        q["gates"]={"accepted_population":M["accepted"]>=30 and M["oracle_near"]>=30,
                           "precision":M["near_precision"] is not None and M["near_precision"]>=.90,
                           "oracle_recovery":M["oracle_recovery"] is not None and M["oracle_recovery"]>=.70,
                           "far_fraction":M["far_fraction"] is not None and M["far_fraction"]<=.03,
                           "slopes":M["predicted_slope_violations"]==[0,0]}
                        q["M_gain_vs_JM"]=M["near_coverage"]-J["near_coverage"]
                        q["gain_nonharm"]=bool(q["M_gain_vs_JM"]>=.05 and M["near_precision"] is not None and J["near_precision"] is not None and M["near_precision"]>=J["near_precision"]-.02)
                        q["complete"]=True
                    except Exception as e:q["failure"]=f"{type(e).__name__}: {str(e)[:400]}"
                    r["cases"].append(q)
                    print("CASE_JSON="+json.dumps({k:z for k,z in q.items() if k!="vertices" and k!="arms"}|{"arms":{k:{j:v for j,v in z.items() if j!="selected_map"} for k,z in q.get("arms",{}).items()}},separators=(",",":"),allow_nan=False),flush=True)
        ok=[q for q in r["cases"] if q["complete"]]
        r["gates"]={"all_six":len(ok)==6,"all_M_gates":len(ok)==6 and all(all(q["gates"].values()) for q in ok),"some_incremental_gain":any(q["gain_nonharm"] for q in ok)}
        r["pass"]=all(r["gates"].values());r["pooled"]={}
        for k in KEYS:
            n=sum(q["reference_rays"] for q in ok);o=sum(q["oracle_near"] for q in ok);a0=sum(q["arms"][k]["accepted"] for q in ok);near=sum(q["arms"][k]["near"] for q in ok);far=sum(q["arms"][k]["far"] for q in ok)
            if n:r["pooled"][k]={"reference_rays":n,"oracle_near":o,"accepted":a0,"near":near,"far":far,"near_precision":div(near,a0),"near_coverage":near/n,"oracle_recovery":div(near,o),"far_fraction":div(far,a0)}
        if arr:
            np.savez_compressed(a.maps,**arr);r["maps_sha256"]=b.digest(a.maps)
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["CT_budget"]={"bytes":budget.used,"requests":budget.requests,"cap":budget.cap};r["wall_seconds"]=time.monotonic()-t;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("claude","gemini","output","maps"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
