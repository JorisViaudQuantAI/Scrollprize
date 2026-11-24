#!/usr/bin/env python3
"""Independent-manuscript single-sheet CT placement, using frozen 057 solvers."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,hashlib,importlib.util,io,json,pathlib,resource,sys,time
import numpy as np
import requests,tifffile
from scipy import ndimage as ndi
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parents[1]
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
solver=module("solver057",HERE.parent/"dtw_order_3d"/"run.py")
winner=module("winner052",HERE.parent/"winner_geometry"/"run.py")
top=module("exact_reference_topology",BASE/"metric_flatten"/"topology.py")
BUCKET=winner.BUCKET
QUANTILES=(.2,.5,.8)
SCROLLS=(("PHerc0211","20250821151803",9.362),("PHerc0343","20250521140437",8.64))
GH="https://raw.githubusercontent.com/Nieuwlaar/pherc343-first-letters/e8370f1f88a211166056df62ea6fcfffe4a1063e/outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz/"
BLOBS={"x.tif":"153bebd7411cfdcbda185c204e3da446267e57f3","y.tif":"2109f714df38ab7432190a8733f8416e6ce293b4","z.tif":"1bff0872a45c822208dd69731eb907d9006a0338","meta.json":"a6541758e46edfa23c4cddf43ec5ae571070db17"}
NAMES=("x.tif","y.tif","z.tif","meta.json")
KEYS=("P_local","I_3D","W_DTW","R_projection","K_fixed","H_reciprocal","Hrow","W_alternate")
def sha(b):return hashlib.sha256(b).hexdigest()
def mesh(session,name):
    objects={};bodies={}
    if name=="PHerc0211":
        prior=json.loads((HERE.parent/"winner_geometry"/"layout_result.json").read_text())
        for n in NAMES:
            r=prior["source"]["objects"][n]
            b=winner.fetch(session,BUCKET+"/"+r["key"],2*1024**2,expected=r)
            if sha(b)!=r["sha256"]:raise ValueError("052 reference mesh hash mismatch")
            objects[n]={"source":r["key"],"sha256":sha(b),"bytes":len(b)}
            bodies[n]=b
    else:
        for n in NAMES:
            b=winner.fetch(session,GH+n,2*1024**2)
            blob=hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest()
            if blob!=BLOBS[n]:raise ValueError("0343 pinned git blob mismatch")
            objects[n]={"source":GH+n,"sha256":sha(b),"git_blob":blob,"bytes":len(b)};bodies[n]=b
    meta=json.loads(bodies["meta.json"])
    scale=np.asarray(meta["scale"],float)
    if not np.isfinite(scale).all() or not np.allclose(scale,.05,rtol=0,atol=1e-8):raise ValueError("Unexpected published carrier scale")
    coords=[tifffile.imread(io.BytesIO(bodies[n])).astype(np.float64) for n in NAMES[:3]]
    if any(c.ndim!=2 or c.shape!=coords[0].shape for c in coords):raise ValueError("Invalid reference grid")
    P=np.stack(coords,axis=-1)
    uv,x,f,grid=winner.grid_faces(P)
    return P,uv,x,f,{"objects":objects,"grid":grid,"source_mesh_bytes":sum(len(b) for b in bodies.values()),"scale":scale.tolist()}
def find_volume(session,scroll,vid,um,budget):
    # Official identifier and unit are checked against the actual directory name.
    f,d,listing=winner.listing(session,scroll+"/volumes/"+vid)
    hit=[]
    for p in d:
        nm=p.rstrip("/").split("/")[-1]
        if not nm.startswith(vid+"-") or not nm.endswith("-masked.zarr"):continue
        try:vu=float(nm.split("-")[1].removesuffix("um"))
        except Exception:continue
        if abs(vu-um)<=1e-6:hit.append(p.rstrip("/"))
    if len(hit)!=1:raise ValueError("Expected one masked native CT volume; found "+str(len(hit)))
    root=BUCKET+"/"+hit[0]
    attrs=winner.fetch(session,root+"/.zattrs",1024**2);budget.charge(len(attrs))
    zarray=winner.fetch(session,root+"/0/.zarray",1024**2);budget.charge(len(zarray))
    m=json.loads(zarray)
    if np.dtype(m["dtype"])!=np.dtype("uint8") or m["zarr_format"]!=2 or len(m["shape"])!=3:raise ValueError("Unexpected L0 CT format")
    a=json.loads(attrs);ms=a.get("multiscales",[])
    if ms:
        names=[v["name"] if isinstance(v,dict) else v for v in ms[0].get("axes",[])]
        if names and names!=["z","y","x"]:raise ValueError("Unexpected axes")
        if not any(str(d["path"])=="0" for d in ms[0].get("datasets",[])):raise ValueError("Missing native level in metadata")
    return root,{"listing":listing,"zattrs_sha256":sha(attrs),"zarray_sha256":sha(zarray),"shape_zyx":m["shape"],"chunks_zyx":m["chunks"],"dtype":m["dtype"],"source_voxel_um":um,"root":root}
def select(P,f,x):
    valid=np.isfinite(P).all(2)&(P>=0).all(2)&(P!=0).any(2)
    cell=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
    interior=ndi.binary_erosion(cell,structure=np.ones((3,3),bool))
    ids=np.flatnonzero(interior[cell])
    if len(ids)<3:raise ValueError("Insufficient interior complete cells")
    if len(f)!=2*cell.sum():raise ValueError("052 grid face order mismatch")
    out=[]
    for q in QUANTILES:
        fid=int(ids[int(np.floor(q*(len(ids)-1)))])
        tri=x[f[fid]];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0])
        if np.linalg.norm(normal)<=1e-12:raise ValueError("Degenerate selected anchor face")
        center=np.rint(tri.mean(0)).astype(np.int64);axis=int(np.argmax(np.abs(normal)));lateral=[a for a in range(3) if a!=axis]
        out.append({"quantile":q,"face":fid,"center":center,"axis":axis,"lateral":lateral})
    # Disjoint entire normalization cubes; no post hoc reselection.
    for i in range(len(out)):
        for j in range(i):
            if (np.abs(out[i]["center"]-out[j]["center"])<96).all():raise ValueError("Fixed full cubes overlap")
    return out
def reference(x,f,sel):
    c=sel["center"];axis=sel["axis"];lat=sel["lateral"];tri=x[f]
    lo=c-48;hi=c+47;latlo=c[lat]-16;lathi=c[lat]+15
    take=((tri.max(1)>=lo)&(tri.min(1)<=hi)).all(1)&((tri[:,:,lat].max(1)>=latlo)&(tri[:,:,lat].min(1)<=lathi)).all(1)
    faces=f[take].copy()
    if not len(faces):raise ValueError("No reference faces")
    a=x[f[sel["face"]]][:,lat]
    det=np.linalg.det(np.stack((a[1]-a[0],a[2]-a[0])))
    if abs(det)<=1e-12:raise ValueError("Degenerate projected seed")
    if det<0:faces=faces[:,[0,2,1]]
    used=np.unique(faces);mp=np.full(len(x),-1,np.int64);mp[used]=np.arange(len(used))
    r=top.rasterize_triangles(x[used][:,lat],x[used],mp[faces],(32,32),origin_xy=latlo)
    truth=r["xyz"][...,axis]-lo[axis]
    valid=r["valid"]&np.isfinite(truth)&(truth>=0)&(truth<=95)
    return truth,valid,r["counts"],int(take.sum())
def self_test():
    # Positive plane with known continuous depth in the integer native rays.
    xx=np.array([[0,0,5],[31,0,5],[0,31,5],[31,31,5]],float)
    ff=np.array([[0,1,3],[0,3,2]])
    r=top.rasterize_triangles(xx[:,:2],xx,ff,(32,32))
    a=bool(r["valid"].all() and np.allclose(r["xyz"][...,2],5))
    yy=np.vstack([xx,xx+[0,0,3]])
    rr=top.rasterize_triangles(yy[:,:2],yy,np.vstack([ff,ff+4]),(32,32))
    b=bool((~rr["valid"]).all() and rr["ambiguity_mask"].all())
    old=solver.self_test()
    return {"plane_exact":a,"double_layer_rejected":b,"solver057":old,"pass":a and b and old["pass"]}
def score(p,truth,valid,um):
    d=np.abs(p[0][valid]-truth[valid]);n=len(d)
    if not n:raise ValueError("No score support")
    return {"eligible":n,"within2":float((d<=2).mean()),"near2":int((d<=2).sum()),
            "MAE_vox":float(d.mean()),"MAE_um":float(d.mean()*um),"median_vox":float(np.median(d)),
            "p95_vox":float(np.quantile(d,.95)),"sum_absolute_error_vox":float(d.sum()),
            "slopes_gt4":[int((np.abs(np.diff(p,axis=a))>4).sum()) for a in (1,2)],"prediction_sha256":solver.old.arrsha(p)}
def evaluate(reader,x,f,sel,um):
    t=time.monotonic()
    c=sel["center"];axis=sel["axis"];lat=sel["lateral"]
    truth,valid,stats,nf=reference(x,f,sel)
    yy,xx=np.indices((32,32));allowed=(np.abs(yy-16)>2)|(np.abs(xx-16)>2)
    eligible=valid&allowed;n=int(eligible.sum())
    r={"quantile":sel["quantile"],"normal_axis_xyz":axis,"reference_faces":nf,"reference_counts":stats,"reference_eligible":n,"reference_fraction":n/int(allowed.sum()),"pass_population":False}
    if not valid[16,16] or not solver.DLO<=int(np.rint(truth[16,16]))<solver.DHI or r["reference_fraction"]<.75:
        r["reason"]="Fixed reference projection/seed population gate failed";return r
    lo=c-48;hi=c+48
    if (lo<0).any() or (hi>np.asarray(reader.shape)[::-1]).any():raise ValueError("Fixed CT cube outside source")
    block,org,sup=reader.read_box(lo[::-1],hi[::-1],return_support=True)
    if tuple(org)!=tuple(lo[::-1]) or block.shape!=(96,96,96) or not sup.all():raise ValueError("Incomplete stored CT")
    mn,mx=np.percentile(block,[1,99])
    if mx<=mn:raise ValueError("No CT contrast")
    norm=np.rint(np.clip((block.astype(np.float32)-mn)*255/(mx-mn),0,255)).astype(np.uint8)
    # CT array is ZYX. Native reference rows = lateral[1]; columns = lateral[0].
    v=norm.transpose(2-axis,2-lat[1],2-lat[0])[:,32:64,32:64].copy()
    seed=np.array([int(np.rint(truth[16,16]))],np.int64);anchor=(16,16)
    C=solver.unary(v,seed,anchor)
    P=C.argmin(-1)+solver.DLO;P[:,16,16]=seed
    W,e=solver.track(v,seed,anchor);alt,_=solver.track(v,seed,anchor,True)
    d=np.abs(np.arange(solver.DLO,solver.DHI)[None,None,None,:]-W[...,None]);cap=np.minimum(d,12)
    H=C+np.rint(150*cap/(1+e[...,None]/4)).astype(np.int64)
    pred={"P_local":P,"W_DTW":W,"W_alternate":alt};diag={}
    for k,cost,lam,axes in (("I_3D",C,100,(1,2)),("R_projection",1000*d,0,(1,2)),("K_fixed",C+150*cap,100,(1,2)),("H_reciprocal",H,100,(1,2)),("Hrow",H,100,(2,))):
        p,q=solver.cut3d(cost,seed-solver.DLO,anchor,lam=lam,slope=4,gap=0,axes=axes)
        pred[k]=p+solver.DLO;diag[k]=q
    r.update(pass_population=True,CT_shape=list(block.shape),CT_sha256=solver.old.arrsha(block),normalized_sha256=solver.old.arrsha(v),stored_support_sha256=solver.old.arrsha(sup),reference_depth_sha256=solver.old.arrsha(np.where(valid,truth,np.nan)),reference_valid_sha256=solver.old.arrsha(valid),percentiles=[float(mn),float(mx)],arms={k:score(pred[k],truth,eligible,um) for k in KEYS},cut_diagnostics=diag,
             path_dependence={"changed_fraction":float((W!=alt).mean()),"depth_delta_p50_p95_max":np.quantile(np.abs(W-alt),[.5,.95,1]).tolist()},wall_seconds=time.monotonic()-t)
    return r
def aggregate(rows):
    complete=len(rows)==3 and all(q.get("pass_population",False) for q in rows)
    if not complete:return {"complete":False}
    macro={k:{m:float(np.mean([r["arms"][k][m] for r in rows])) for m in ("within2","MAE_vox","MAE_um","p95_vox")} for k in KEYS}
    W=macro["W_DTW"];I=macro["I_3D"];H=macro["H_reciprocal"];R=macro["Hrow"]
    return {"complete":True,"macro":macro,"gates":{"within2_W":H["within2"]>=W["within2"]-.02,"within2_I":H["within2"]>=I["within2"]-.02,
        "MAE_W":H["MAE_vox"]<=W["MAE_vox"]+.25,"slopes":all(r["arms"]["H_reciprocal"]["slopes_gt4"]==[0,0] for r in rows)},
        "H_gain_W":H["within2"]-W["within2"],"H_gain_Hrow":H["within2"]-R["within2"]}
def main(a):
    t=time.monotonic();r={"schema":"adaptive.dtw_cross_scroll.v1","script_sha256":solver.digest(__file__),"protocol_sha256":solver.digest(HERE/"protocol.md"),
      "solver057_sha256":solver.digest(HERE.parent/"dtw_order_3d"/"run.py"),"cases":[],"pass":False}
    sys.path.insert(0,str(a.gemini));import grow_m7_patch as gm
    budget=gm.ByteBudget(160*1024**2)
    try:
        r["synthetic"]=self_test()
        if not r["synthetic"]["pass"]:raise ValueError("Frozen synthetic gate failed")
        with requests.Session() as session:
            for name,vid,um in SCROLLS:
                q={"scroll":name,"ROIs":[]}
                try:
                    P,uv,x,f,q["mesh"]=mesh(session,name)
                    selections=select(P,f,x)
                    root,q["CT_source"]=find_volume(session,name,vid,um,budget)
                    with gm.BudgetedRemoteZarrV2(root,level="0",budget=budget,workers=1,cache_mb=64,max_retries=1,timeout=25) as reader:
                        if tuple(reader.shape)!=tuple(q["CT_source"]["shape_zyx"]) or reader.dtype!=np.dtype("uint8"):raise ValueError("CT metadata changed")
                        for s in selections:
                            try:row=evaluate(reader,x,f,s,um)
                            except Exception as e:row={"quantile":s["quantile"],"pass_population":False,"failure":f"{type(e).__name__}: {str(e)[:400]}"}
                            q["ROIs"].append(row)
                    q["aggregate"]=aggregate(q["ROIs"])
                except Exception as e:q["failure"]=f"{type(e).__name__}: {str(e)[:400]}"
                r["cases"].append(q);print("CASE_JSON="+json.dumps(q,separators=(",",":"),allow_nan=False),flush=True)
        aa=[q.get("aggregate",{"complete":False}) for q in r["cases"]]
        r["complete"]=len(aa)==2 and all(q["complete"] for q in aa)
        r["pass"]=r["complete"] and all(all(q["gates"].values()) for q in aa) and any(q["H_gain_W"]>=.03 for q in aa)
        r["cross_row_benefit_pass"]=r["pass"] and all(q["H_gain_Hrow"]>=-.02 for q in aa) and any(q["H_gain_Hrow"]>=.03 for q in aa)
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["CT_budget"]={"bytes":budget.used,"requests":budget.requests,"cap":budget.cap}
    r["wall_seconds"]=time.monotonic()-t;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("output","gemini"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    main(p.parse_args())
