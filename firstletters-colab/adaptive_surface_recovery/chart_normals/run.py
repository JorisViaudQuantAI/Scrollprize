#!/usr/bin/env python3
"""Crease-aware chart-local original-face normals, using the existing renderer."""
from __future__ import annotations
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="4"
import argparse,hashlib,importlib.util,json,pathlib,resource,sys,time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parents[1]
COS30=np.cos(np.pi/6)
DEPTHS=np.arange(-12,12,dtype=float)

def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def sha(p):
    with pathlib.Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def angle(a,b):
    return np.degrees(2*np.arcsin(np.clip(np.linalg.norm(a-b,axis=-1)/2,0,1)))

def normals(xyz,faces):
    """Corner fan graph: only two-face original edges, within one chart."""
    x=np.asarray(xyz,float);f=np.asarray(faces,np.int64);nf=len(f)
    q=x[f];cross=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);length=np.linalg.norm(cross,axis=1)
    if not np.isfinite(length).all() or (length<=0).any():raise ValueError("Degenerate physical face")
    unit=cross/length[:,None]
    # Per face, create three edges, with corner IDs sorted by original vertex ID.
    corner_ids=np.arange(3*nf).reshape(nf,3)
    v=f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    cp=corner_ids[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    reorder=np.argsort(v,axis=1,kind="stable")
    v=np.take_along_axis(v,reorder,axis=1);cp=np.take_along_axis(cp,reorder,axis=1)
    ix=np.lexsort((v[:,1],v[:,0]));v=v[ix];cp=cp[ix]
    bounds=np.r_[0,np.flatnonzero(np.any(v[1:]!=v[:-1],axis=1))+1,len(v)]
    count=np.diff(bounds);pair_start=bounds[:-1][count==2]
    p0=cp[pair_start];p1=cp[pair_start+1]
    f0=p0[:,0]//3;f1=p1[:,0]//3
    dots=(unit[f0]*unit[f1]).sum(1)
    elig=dots>=COS30-1e-12;aa=p0[elig];bb=p1[elig]
    aa=aa.ravel();bb=bb.ravel()
    graph=coo_matrix((np.ones(len(aa),np.uint8),(aa,bb)),shape=(3*nf,3*nf)).tocsr()
    ng,groups=connected_components(graph,directed=False)
    vertex=f.ravel();lo=np.full(ng,len(x),np.int64);hi=np.full(ng,-1,np.int64)
    np.minimum.at(lo,groups,vertex);np.maximum.at(hi,groups,vertex)
    if not np.array_equal(lo,hi):raise ArithmeticError("Fan connects different original vertices")
    total=np.zeros((ng,3),float)
    np.add.at(total,groups,np.repeat(cross,3,axis=0))
    norm=np.linalg.norm(total,axis=1);safe=norm>0
    avg=np.zeros_like(total);avg[safe]=total[safe]/norm[safe,None]
    corner_face_unit=np.repeat(unit,3,axis=0)
    check=(avg[groups]*corner_face_unit).sum(1)>=COS30-1e-12
    safe[np.unique(groups[~check])]=False
    corner=np.where(safe[groups,None],avg[groups],corner_face_unit)
    # All eligible original edges count, including fallback unsafe fans.
    eligible_pairs0=p0[elig];eligible_pairs1=p1[elig]
    safe_edge=safe[groups[eligible_pairs0]].all(1)&safe[groups[eligible_pairs1]].all(1)
    old_jump=angle(unit[f0[elig]],unit[f1[elig]])
    new_jump=angle(corner[eligible_pairs0],corner[eligible_pairs1]).ravel()
    old_med=float(np.median(old_jump)) if len(old_jump) else 0.
    new_med=float(np.median(new_jump)) if len(new_jump) else 0.
    frac=float(safe_edge.mean()) if len(safe_edge) else 1.
    gain=1-new_med/old_med if old_med>1e-12 else (1. if new_med<1e-12 else -1.)
    metrics={"faces":nf,"original_edges":int(len(count)),"boundary_edges":int((count==1).sum()),"nonmanifold_edges":int((count>2).sum()),
             "two_face_edges":int((count==2).sum()),"eligible_smooth_edges":int(elig.sum()),
             "safe_fan_smooth_edges":int(safe_edge.sum()),"safe_edge_fraction":frac,"fan_groups":int(ng),
             "unsafe_fan_groups":int((~safe).sum()),"max_corner_deviation_deg":float(angle(corner,corner_face_unit).max()),
             "median_old_edge_jump_deg":old_med,"median_new_endpoint_jump_deg":new_med,
             "median_reduction_fraction":float(gain),"max_new_endpoint_jump_deg":float(new_jump.max()) if len(new_jump) else 0.,
             "no_cross_vertex_fans":True}
    return corner.reshape(nf,3,3),unit,groups.reshape(nf,3),metrics

class SmoothChart:
    def __init__(self,base):
        self.base=base;self.corner,self.unit,self.groups,self.metrics=normals(base.xyz,base.faces)
        self.look={int(v):i for i,v in enumerate(base.original_ids)}
        self.global_to_local=np.full(int(base.original_ids.max())+1,-1,np.int64)
        self.global_to_local[base.original_ids]=np.arange(len(base.original_ids))
        self.max_deviation=0.;self.rejected=0;self.max_unit_error=0.
    def footprint_support(self,*a,**kw):return self.base.footprint_support(*a,**kw)
    def sample_grid(self,*a,**kw):
        r=self.base.sample_grid(*a,**kw);v=r["valid"];ids=r["original_face_id"][v]
        ii=self.global_to_local[ids]
        if (ii<0).any():raise ValueError("Owner not in selected chart")
        w=r["barycentric_weights"][v]
        n=np.einsum("ni,nij->nj",w,self.corner[ii]);len_n=np.linalg.norm(n,axis=1)
        ok=np.isfinite(len_n)&(len_n>0);n[ok]/=len_n[ok,None]
        d=(n*self.unit[ii]).sum(1)
        ok&=d>0
        self.rejected+=int((~ok).sum())
        if not ok.all():raise ValueError("Invalid interpolated normal, no support removal allowed")
        if len(n):
            self.max_deviation=max(self.max_deviation,float(angle(n,self.unit[ii]).max()))
            self.max_unit_error=max(self.max_unit_error,float(np.max(np.abs(np.linalg.norm(n,axis=1)-1))))
        r["normals"][v]=n
        return r

def self_test():
    x=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[0,-1,0],[0,-1,1]],float)
    a=np.array([[0,1,2],[1,3,2]])
    n,_,g,s=normals(x,a);assert np.allclose(n,[0,0,1]) and s["safe_edge_fraction"]==1
    # 90-degree crease, only source edge (0,1) is shared.
    b=np.array([[0,1,2],[1,0,4]])
    n,u,g,s=normals(x,b);assert s["eligible_smooth_edges"]==0 and np.allclose(n,np.broadcast_to(u[:,None,:],n.shape))
    # Sharing only vertex 0 cannot connect disconnected fans.
    b=np.array([[0,1,2],[0,5,6]])
    n,u,g,s=normals(x,b);assert g[0,0]!=g[1,0]
    # Separate chart subsets have no internal cross-chart edge.
    _,_,_,s=normals(x,a[:1]);assert s["two_face_edges"]==0
    return 4

def run(a):
    t=time.monotonic()
    report={"schema":"adaptive.chart_normals.v1","unit_checks":self_test(),"script_sha256":sha(__file__),
            "protocol_sha256":sha(HERE/"protocol.md"),"cases":[],"pass":False,"CNN_forwards":0}
    det_report=json.loads(a.detector_result.read_text())
    cal=det_report.get("calibration",{})
    report["detector_040_calibration"]=cal
    report["detector_040_result_sha256"]=sha(a.detector_result)
    if not cal.get("pass"):
        report["failure"]="040 CT-frame calibration was not established; frozen 041 cannot compare against the published render."
    else:
        bridge=loadmod("adaptive_bridge",HERE.parent/"shared_renderer_bridge"/"run.py")
        detector=loadmod("adaptive_detector",HERE.parent/"detector_bridge"/"run.py")
        surface=loadmod("shared_original_surface",a.shared/"surface.py")
        renderer=loadmod("shared_footprint_renderer",a.shared/"render.py")
        sys.path.insert(0,str(a.gemini));import grow_m7_patch as gm
        sys.path.insert(0,str(BASE/"tile_phase"))
        common=loadmod("common",BASE/"tile_phase"/"common.py")
        prep=loadmod("existing_tile_prepare",BASE/"tile_phase"/"prepare.py")
        budget=gm.ByteBudget(80*1024**2);rel=None
        try:
            atlas=bridge.load_atlas(a.atlas,bridge.PIN)
            pri=bridge.ChartBridge(atlas,0,surface.OriginalSurface);fall=bridge.ChartBridge(atlas,1,surface.OriginalSurface)
            smooth=SmoothChart(pri);fsmooth=SmoothChart(fall)
            report["edge_metrics"]={"primary":smooth.metrics,"fallback":fsmooth.metrics}
            a.cache.mkdir(parents=True,exist_ok=False)
            rel=detector.ReleasedRender(a.cache,prep.fetch,common.HF_DATA)
            with gm.BudgetedRemoteZarrV2(bridge.CT_URL,level="0",budget=budget,workers=1,cache_mb=128,max_retries=1,timeout=15) as reader:
                depths=cal["chosen_sign"]*DEPTHS
                for i,center in enumerate([(1800,2800),(2200,4600),(2360,6400),(2800,6800)]):
                    yx=detector.nearest_origin(center)
                    b=rel.tile(yx);vol,org=detector.source_crop(pri,reader,yx)
                    p=detector.render_native(renderer,pri,vol,org,yx,depths,1)
                    s=detector.render_native(renderer,smooth,vol,org,yx,depths,1)
                    # Point support, owner IDs and XYZ must remain exactly unchanged.
                    r0=pri.sample_grid(detector.uv(yx),(.05,.05),(64,64));r1=smooth.sample_grid(detector.uv(yx),(.05,.05),(64,64))
                    same=all(np.array_equal(r0[k],r1[k],equal_nan=True) for k in ("xyz","valid","original_face_id","barycentric_weights","reason_bits"))
                    row={"kind":"primary","index":i,"calibration_excluded":i==2,"yx":list(yx),
                         "B_P":detector.stats(b,p),"B_S":detector.stats(b,s),"P_S":detector.stats(p,s),
                         "geometry_exact":same,"supported_pixels":4096}
                    report["cases"].append(row)
                # Fallback UV positions differ from primary; compare its two arms, no B reference.
                mid=len(fall.faces)//2;center=np.floor(fall.uv[fall.faces[mid]].mean(0));origin=center-.8+.0125
                lo,hi=fall.dense_crop_bounds(origin,(.05,.05),(32,32),depths)
                block,org,support=reader.read_box(lo[::-1],hi[::-1],return_support=True)
                if tuple(org)!=tuple(lo[::-1]) or tuple(block.shape)!=tuple((hi-lo)[::-1]) or not support.all():raise ValueError("Fallback source support")
                p,mp,_=renderer.render(block.astype(np.float32),np.asarray(org[::-1],float),fall,origin,(.05,.05),(32,32),depths,samples_per_axis=1)
                s,ms,_=renderer.render(block.astype(np.float32),np.asarray(org[::-1],float),fsmooth,origin,(.05,.05),(32,32),depths,samples_per_axis=1)
                if not np.array_equal(mp,ms) or not mp.any():raise ValueError("Fallback geometry/CT support changed")
                report["cases"].append({"kind":"fallback","same_support":True,"valid_pixels":int(mp.sum()),"P_S":detector.stats(p[:,mp],s[:,mp])})
            test=[r for r in report["cases"] if r["kind"]=="primary" and not r["calibration_excluded"]]
            bm=float(np.mean([r["B_P"]["mae"] for r in test]));sm=float(np.mean([r["B_S"]["mae"] for r in test]))
            report["mean_CT_MAE"]={"piecewise":bm,"smooth":sm,"relative_reduction":1-sm/bm}
            report["sample_normals"]={k:{"max_deviation_deg":c.max_deviation,"max_unit_error":c.max_unit_error,"rejected":c.rejected} for k,c in (("primary",smooth),("fallback",fsmooth))}
            gates={"all_geometry_exact":all(r.get("geometry_exact",r.get("same_support",False)) for r in report["cases"]) and len(test)==3,
                   "all_normal_contracts":all(c.max_deviation<=30+1e-6 and c.max_unit_error<=1e-10 and c.rejected==0 for c in (smooth,fsmooth)),
                   "safe_edges_ge_95pct":all(c.metrics["safe_edge_fraction"]>=.95 for c in (smooth,fsmooth)),
                   "median_edge_jump_reduction_ge90pct":all(c.metrics["median_reduction_fraction"]>=.9 for c in (smooth,fsmooth)),
                   "real_CT_MAE_reduction_ge10pct":sm<=.9*bm,
                   "no_individual_CT_MAE_worse_than5pct":all(r["B_S"]["mae"]<=1.05*r["B_P"]["mae"] for r in test)}
            report["gates"]=gates;report["pass"]=all(gates.values())
        except Exception as e:report["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
        report["native_CT"]={"charged_body_bytes":budget.used,"requests":budget.requests,"cap_bytes":budget.cap}
        if rel:report["released_CT"]={"body_bytes":rel.bytes,"chunks":rel.receipts}
    report["wall_seconds"]=time.monotonic()-t;report["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--self-test",action="store_true")
    for k in ("atlas","shared","gemini","cache","output","detector-result"):ap.add_argument("--"+k,type=pathlib.Path)
    a=ap.parse_args()
    if a.self_test:print("Tests passed:",self_test())
    else:
        if any(getattr(a,k) is None for k in ("atlas","shared","gemini","cache","output","detector_result")) or a.output.exists():ap.error("Required inputs and new output")
        run(a)
