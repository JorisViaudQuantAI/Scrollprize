#!/usr/bin/env python3
"""Wider-chart sparse geometry and exact bounded residual CT correction."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,importlib.util,json,pathlib,resource,sys,time
import numpy as np
import requests
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator
HERE=pathlib.Path(__file__).resolve().parent
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m68=mod("mosaic068",HERE.parent/"claude_ct_mosaic"/"run.py")
atlas=m68.atlas;gap=m68.gap;cross=m68.cross;co=m68.co;b=co.b;top=cross.top
N=96;DEPTH=128;RADIUS=3
A3=((16,16),(16,80),(80,48));A5=A3+((48,48),(80,80))
KEYS=("P3","P5","T5","B3","B5")
def heldout():
    y,x=np.indices((N,N));m=np.ones((N,N),bool)
    for a,c in A5:m&=(np.abs(y-a)>2)|(np.abs(x-c)>2)
    return m
def candidates(w):
    cell=w.valid[:-1,:-1]&w.valid[:-1,1:]&w.valid[1:,:-1]&w.valid[1:,1:]
    interior=ndi.binary_erosion(cell,structure=np.ones((3,3),bool));ids=np.flatnonzero(interior[cell])
    if len(w.faces)!=2*cell.sum() or len(ids)<3:raise ValueError("Original complete-face order mismatch")
    for j in range(1,128):
        q=j/128;fid=int(ids[int(np.floor(q*(len(ids)-1)))])
        tri=w.xyz[w.faces[fid]];n=np.cross(tri[1]-tri[0],tri[2]-tri[0])
        if np.linalg.norm(n)<=1e-12:continue
        a=int(np.argmax(np.abs(n)))
        yield {"quantile":q,"face":fid,"center":np.rint(tri.mean(0)).astype(np.int64),"axis":a,"lateral":[i for i in range(3) if i!=a]}
def project(mesh,s,isref=False):
    c=s["center"];ax=s["axis"];lat=s["lateral"];lo=c-64;hi=c+63;latlo=c[lat]-48;lathi=c[lat]+47
    tri=mesh.xyz[mesh.faces]
    take=((tri.max(1)>=lo)&(tri.min(1)<=hi)).all(1)&((tri[:,:,lat].max(1)>=latlo)&(tri[:,:,lat].min(1)<=lathi)).all(1)
    f=mesh.faces[take].copy()
    if not len(f):return np.full((N,N),np.nan),{"faces":0,"valid":0}
    if isref:
        uv=mesh.xyz[mesh.faces[s["face"]]][:,lat];u=uv[1]-uv[0];v=uv[2]-uv[0];sign=float(u[0]*v[1]-u[1]*v[0])
    else:
        uv=mesh.xyz[f][:,:,lat];u=uv[:,1]-uv[:,0];v=uv[:,2]-uv[:,0];sign=float(np.sum(u[:,0]*v[:,1]-u[:,1]*v[:,0]))
    if abs(sign)<=1e-12:return np.full((N,N),np.nan),{"faces":len(f),"valid":0,"reason":"indeterminate orientation"}
    if sign<0:f=f[:,[0,2,1]]
    used=np.unique(f);mp=np.full(len(mesh.xyz),-1,np.int64);mp[used]=np.arange(len(used))
    r=top.rasterize_triangles(mesh.xyz[used][:,lat],mesh.xyz[used],mp[f],(N,N),origin_xy=latlo)
    d=r["xyz"][...,ax]-lo[ax];v=r["valid"]&np.isfinite(d)&(d>=0)&(d<=DEPTH-1)
    return np.where(v,d,np.nan),{"faces":len(f),"valid":int(v.sum()),"counts":r["counts"]}
def affine(shape,anchors):
    xy=np.array([[1,y,x] for y,x,z in anchors],float);z=np.array([z for _,_,z in anchors],float)
    if np.linalg.matrix_rank(xy)!=3:raise ValueError("Collinear anchors")
    c=np.linalg.lstsq(xy,z,rcond=None)[0];y,x=np.indices(shape);p=np.rint(c[0]+c[1]*y+c[2]*x).astype(np.int64)
    for a,b,v in anchors:p[a,b]=v
    return p
def tps(shape,anchors):
    xy=np.array([[y,x] for y,x,z in anchors],float);z=np.array([z for _,_,z in anchors],float)
    yy,xx=np.indices(shape);q=RBFInterpolator(xy,z,kernel="thin_plate_spline",degree=1,smoothing=0.)(np.column_stack([yy.ravel(),xx.ravel()]))
    if not np.isfinite(q).all():raise ValueError("Nonfinite spline")
    p=np.clip(np.rint(q.reshape(shape)),4,DEPTH-5).astype(np.int64)
    for y,x,z in anchors:p[y,x]=z
    return p
def admissible(p):
    return bool(p.min()>=7 and p.max()<=DEPTH-8 and all((np.abs(np.diff(p,axis=a))<=2).all() for a in (0,1)))
def normalize(w):
    a=w-w.mean(-1,keepdims=True);n=np.linalg.norm(a,axis=-1,keepdims=True)
    return np.divide(a,n,out=np.zeros_like(a),where=n>1e-10)
def unary(v,p,anchors):
    # v shape depth,y,x; candidate depths from base +-3, nine point window.
    win=np.lib.stride_tricks.sliding_window_view(v.astype(np.float64),9,axis=0)
    templates=normalize(np.array([win[z-4,y,x] for y,x,z in anchors]))
    yy,xx=np.indices(p.shape)
    residual=np.arange(-3,4,dtype=np.int64)
    depths=p[...,None]+residual
    cand=normalize(win[depths-4,yy[...,None],xx[...,None]])
    corr=np.clip(np.einsum("yxdn,an->yxda",cand,templates).mean(-1),-1,1)
    return np.rint(1000*(1-corr)).astype(np.int64)+150*np.abs(residual)
def bounded(v,p,anchors):
    if not admissible(p):raise ValueError("Base outside frozen residual contract")
    C=unary(v,p,anchors);at=anchors[0][:2]
    q,di=co.ac.cut3d(C[None],np.array([3]),at,lam=100,slope=2,gap=0,extra=[(0,y,x,3) for y,x,z in anchors[1:]])
    residual=q[0]-3;out=p+residual
    if np.max(np.abs(residual))>3 or any(out[y,x]!=z for y,x,z in anchors) or any((np.abs(np.diff(out,axis=a))>4).any() for a in (0,1)):
        raise ArithmeticError("Residual physical constraint violation")
    di.update(changed_fraction=float((residual!=0).mean()),residual_abs_mean=float(np.abs(residual).mean()),residual_counts=np.bincount(q[0].ravel(),minlength=7).tolist(),max_physical_step=[int(np.abs(np.diff(out,axis=a)).max()) for a in (0,1)])
    return out,di
def previous_centers(w):
    root=HERE.parent
    q58=json.loads((root/"dtw_cross_scroll"/"result.json").read_text())
    q63=json.loads((root/"gap_aware_transfer"/"result.json").read_text())
    q68=json.loads((root/"claude_ct_mosaic"/"result.json").read_text())
    old=cross.select(w.P,w.faces,w.xyz)
    # 058's helper expects compact indexing but full original mesh is equivalent.
    needed={q["quantile"] for q in q68["selection_receipt"] if q["selected"]}
    c63=next(q for q in q63["cases"] if q["scroll"]=="PHerc0211")
    needed.update(q["quantile"] for q in c63["fresh"])
    needed.update(q["quantile"] for q in c63["repeated"] if q["quantile"] not in [s["quantile"] for s in old])
    by={q:s for q,s in gap.new_candidates(w.P,w.xyz,w.faces)}
    if not all(q in by and by[q] is not None for q in needed):raise ValueError("Prior selection cannot be reconstructed")
    return old+[by[q] for q in sorted(needed)]
def select(w,meshes):
    old=previous_centers(w);sel=[];rows=[];m=heldout()
    for s in candidates(w):
        rec={"quantile":s["quantile"],"selected":False}
        if len(sel)>=6:rec["reason"]="six frozen eligible fields reserved"
        elif any((np.abs(s["center"]-q["center"])<112).all() for q in old):rec["reason"]="overlaps earlier full CT cube"
        elif any((np.abs(s["center"]-q["sel"]["center"])<128).all() for q in sel):rec["reason"]="overlaps reserved field"
        else:
            try:
                d,ref=project(w,s,True);v=np.isfinite(d);common=v&m
                rec["reference_fraction"]=float(common.sum()/m.sum())
                if rec["reference_fraction"]<.9 or not all(v[y,x] and 7<=int(np.rint(d[y,x]))<=121 for y,x in A5):rec["reason"]="reference or anchor support"
                else:
                    a=[(y,x,int(np.rint(d[y,x]))) for y,x in A5];p3=affine((N,N),a[:3]);p5=affine((N,N),a)
                    if not admissible(p3) or not admissible(p5):rec["reason"]="anchor-only affine slope/depth gate"
                    else:
                        c=[];pr=[]
                        for mesh in meshes:
                            z,zr=project(mesh,s);c.append(z);pr.append(zr)
                        c=np.stack(c);rec["multi_carrier_fraction"]=float((np.isfinite(c).sum(0)[common]>=2).mean())
                        rec["carrier_projection"]=pr
                        if rec["multi_carrier_fraction"]<.75:rec["reason"]="insufficient carrier support"
                        else:
                            rec["selected"]=True;sel.append({"sel":s,"truth":d,"valid":v,"common":common,"candidates":c,"anchors":a,"P3":p3,"P5":p5,"reference":ref})
            except Exception as e:rec["reason"]=f"{type(e).__name__}: {str(e)[:300]}"
        rows.append(rec)
    return sel,rows,len(old)
def synthetic():
    # Waveform with nonperiodic narrow features. A translated interior is two vox deeper.
    z=np.arange(128,dtype=float);profile=90+70*np.exp(-((z-62)/.8)**2)-40*np.exp(-((z-64)/.9)**2)+23*np.exp(-((z-59)/.7)**2)
    v=np.broadcast_to(profile[:,None,None],(128,16,16)).copy()
    v[:,4:12,4:12]=np.interp(z-2,z,profile)[:,None,None]
    a=[(1,1,62),(1,14,62),(14,8,62)];p=affine((16,16),a);r,diag=bounded(v,p,a)
    C=unary(v,p,a);Y,X=8,8
    t=np.array([v[depth-4:depth+5,y,x] for y,x,depth in a])
    t=normalize(t);direct=[]
    for d in range(59,66):
        c=normalize(v[d-4:d+5,Y,X]);direct.append(int(np.rint(1000*(1-np.mean(t@c))))+150*abs(d-62))
    plane=np.array([[[0,0,30],[127,0,30]],[[0,127,30],[127,127,30]]],float)
    # native centre=(48,48,64), depth coordinate 30; exact full projected grid.
    s={"center":np.array([48,48,64]),"axis":2,"lateral":[0,1],"face":0}
    mesh=atlas.Mesh(plane);d,_=project(mesh,s,True)
    plane2=np.concatenate([plane, np.full((1,2,3),np.nan),plane+np.array([0,0,12])],axis=0)
    dm,_=project(atlas.Mesh(plane2),s,True)
    gates={"ray":atlas.self_test()["pass"],"solver":co.self_test()["pass"],"affine_anchor_exact":bool((p==62).all()),
      "wave_recovery":abs(int(r[8,8])-64)<=1,"bounded_constraints":diag["max_physical_step"][0]<=4 and diag["max_physical_step"][1]<=4,
      "direct_unary":np.array_equal(C[Y,X],direct),"projection_plane":np.isfinite(d).all() and np.allclose(d,30,atol=1e-10,rtol=0),
      "projection_multiple_unknown":not np.isfinite(dm).any()}
    return {"gates":{k:bool(v) for k,v in gates.items()},"center_residual":int(r[8,8]-62),"direct_cost":direct,"pass":all(gates.values())}
def evaluate(v,prep):
    a=prep["anchors"];p3=prep["P3"];p5=prep["P5"]
    P={"P3":p3,"P5":p5,"T5":tps((N,N),a)}
    P["B3"],di3=bounded(v,p3,a[:3]);P["B5"],di5=bounded(v,p5,a)
    d=prep["truth"];m=prep["common"];cand=prep["candidates"];onear,olab=m68.oracle(cand,d,m)
    arms={};arrays={}
    for k in KEYS:
        l,h=m68.choose(P[k],cand)
        arms[k]=m68.score(P[k],l,h,cand,d,m,onear,olab)
        arrays[k+"_labels"]=np.where(m,l,0);arrays[k+"_pred"]=P[k];arrays[k+"_height"]=np.where(m,h,np.nan)
    base=arms["P3"];z=arms["B3"]
    gates={"within2":z["target_within2"]>=base["target_within2"]-.02,"near_coverage":z["near_coverage"]>=base["near_coverage"]-.02,
      "precision":not (z["accepted"]>=30 and base["accepted"]>=30) or z["near_precision"]>=base["near_precision"]-.02,
      "far_fraction":z["far_fraction"] is None or z["far_fraction"]<=.03,
      "slope":z["predicted_slope_violations"]==[0,0]}
    return {"reference_rays":int(m.sum()),"oracle_near":int(onear.sum()),"reference_sha256":b.old.arrsha(d),"candidate_sha256":b.old.arrsha(cand),
        "arms":arms,"cut":{"B3":di3,"B5":di5},"gates":gates,"gain":all(gates.values()) and (z["target_within2"]>=base["target_within2"]+.03 or z["near_coverage"]>=base["near_coverage"]+.03)},arrays
def run(a):
    start=time.monotonic();r={"schema":"adaptive.wide_anchor_trust.v1","protocol_sha256":b.digest(HERE/"protocol.md"),"script_sha256":b.digest(__file__),"source_commit":atlas.PIN,"cases":[],"pass":False}
    sys.path.insert(0,str(a.gemini));import grow_m7_patch as gm
    budget=gm.ByteBudget(200*1024**2);arr={}
    try:
        r["synthetic"]=synthetic()
        if not r["synthetic"]["pass"]:raise ValueError("Frozen synthetic checks failed")
        w,r["winner_source"]=atlas.load_winner();meshes=[];r["fit_sources"]={}
        for name in atlas.WINDS:
            m,_,rec=atlas.load_fit(a.claude,"flat",name);meshes.append(m);r["fit_sources"][name]=rec
        chosen,r["selection_receipt"],r["prior_excluded_centers"]=select(w,meshes);r["selected"]=len(chosen)
        print("SELECTION_JSON="+json.dumps({"selected":len(chosen),"quantiles":[q["sel"]["quantile"] for q in chosen]},separators=(",",":")),flush=True)
        with requests.Session() as session:
            root,r["CT_source"]=cross.find_volume(session,"PHerc0211","20250821151803",9.362,budget)
            before=json.loads((HERE.parent/"dtw_cross_scroll"/"result.json").read_text())
            p=next(q for q in before["cases"] if q["scroll"]=="PHerc0211")
            if any(r["CT_source"][k]!=p["CT_source"][k] for k in ("root","zattrs_sha256","zarray_sha256")):raise ValueError("CT metadata changed")
            with gm.BudgetedRemoteZarrV2(root,level="0",budget=budget,workers=1,cache_mb=128,max_retries=1,timeout=25) as reader:
                for i,prep in enumerate(chosen):
                    s=prep["sel"];row={"quantile":s["quantile"],"complete":False}
                    try:
                        c=s["center"];ax=s["axis"];lat=s["lateral"];lo=c-64;hi=c+64
                        if (lo<0).any() or (hi>np.asarray(reader.shape)[::-1]).any():raise ValueError("CT out of bounds")
                        ct,org,sup=reader.read_box(lo[::-1],hi[::-1],return_support=True)
                        if tuple(org)!=tuple(lo[::-1]) or ct.shape!=(128,128,128) or not sup.all():raise ValueError("Incomplete CT support")
                        mn,mx=np.percentile(ct,[1,99])
                        if mx<=mn:raise ValueError("No CT contrast")
                        v=np.rint(np.clip((ct.astype(np.float32)-mn)*255/(mx-mn),0,255)).astype(np.uint8).transpose(2-ax,2-lat[1],2-lat[0])[:,16:112,16:112].copy()
                        rec,maps=evaluate(v,prep);row.update(rec,complete=True,CT_sha256=b.old.arrsha(ct),normalized_sha256=b.old.arrsha(v),support_sha256=b.old.arrsha(sup),percentiles=[float(mn),float(mx)])
                        for k,z in maps.items():arr[f"roi{i}_{k}"]=z
                        for k,z in (("reference",prep["truth"]),("common",prep["common"]),("carriers",prep["candidates"]),("center_xyz",c),("axis",np.array(ax)),("lateral_axes",np.array(lat))):arr[f"roi{i}_{k}"]=z
                    except Exception as e:row["failure"]=f"{type(e).__name__}: {str(e)[:400]}"
                    r["cases"].append(row)
                    print("CASE_JSON="+json.dumps({**{k:v for k,v in row.items() if k!="arms"},"arms":{k:{j:v for j,v in z.items() if j!="selected_map"} for k,z in row.get("arms",{}).items()}},separators=(",",":"),allow_nan=False),flush=True)
        ok=[q for q in r["cases"] if q["complete"]];r["pooled"]={}
        for k in KEYS:
            n=sum(q["reference_rays"] for q in ok);o=sum(q["oracle_near"] for q in ok);a0=sum(q["arms"][k]["accepted"] for q in ok);near=sum(q["arms"][k]["near"] for q in ok);far=sum(q["arms"][k]["far"] for q in ok)
            if n:r["pooled"][k]={"reference_rays":n,"oracle_near":o,"accepted":a0,"near":near,"far":far,"near_precision":m68.div(near,a0),
                "near_coverage":near/n,"oracle_recovery":m68.div(near,o),"far_fraction":m68.div(far,a0),
                "within2":sum(q["reference_rays"]*q["arms"][k]["target_within2"] for q in ok)/n,
                "MAE_vox":sum(q["reference_rays"]*q["arms"][k]["target_MAE_vox"] for q in ok)/n}
        r["gates"]={"six_complete":len(ok)==6,"all_local":len(ok)==6 and all(all(q["gates"].values()) for q in ok),
            "two_gains":sum(bool(q["gain"]) for q in ok)>=2,
            "pooled_MAE":bool(ok and r["pooled"]["B3"]["MAE_vox"]<r["pooled"]["P3"]["MAE_vox"])}
        r["pass"]=all(r["gates"].values())
        if arr:np.savez_compressed(a.maps,**arr);r["maps_sha256"]=b.digest(a.maps)
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["CT_budget"]={"bytes":budget.used,"requests":budget.requests,"cap":budget.cap};r["wall_seconds"]=time.monotonic()-start;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("claude","gemini","output","maps"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
