#!/usr/bin/env python3
"""Bridge the two-chart atlas to the shared OriginalSurface/footprint renderer.
No duplicate numerical renderer; explicit original-face mapping per chart.
"""
from __future__ import annotations
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="2"
import argparse,hashlib,importlib.util,json,pathlib,resource,sys,time
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
PIN="43c12969b32c42b978e9d1cbb3bd70fb645a998100062d889d66228a130fd8d0"
PITCH=.05
DEPTHS=np.arange(-15,16,dtype=float)
CT_URL="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr"

def sha(path):
    with pathlib.Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(name)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def load_atlas(path,expected_sha256):
    """Read a verified archive without pickle or silent face reassignment."""
    if sha(path)!=expected_sha256:raise ValueError("Atlas SHA256 mismatch")
    with np.load(path,allow_pickle=False) as z:
        q0,q,x,f,c=[z[k].copy() for k in ("uv_initial","uv_corrected","source_xyz","source_triangles","chart_id")]
    if q0.shape!=q.shape or q.ndim!=2 or q.shape[1]!=2 or x.shape!=(len(q),3):raise ValueError("Invalid source arrays")
    if not all(np.isfinite(v).all() for v in (q0,q,x)):raise ValueError("Nonfinite geometry")
    if f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in "iu" or not len(f) or f.min()<0 or f.max()>=len(x):raise ValueError("Invalid faces")
    if c.shape!=(len(f),) or c.dtype.kind not in "iu" or not np.array_equal(np.unique(c),[0,1]):raise ValueError("Require two explicit face charts")
    for a in (q0,q,x,f,c):a.setflags(write=False)
    return q0,q,x,f,c

class ChartBridge:
    """One chart only. Existing renderer's protocol, global face ID extra field."""
    def __init__(self,atlas,chart,Surface):
        q0,q,x,f,c=atlas
        if chart not in (0,1):raise ValueError("Chart must be 0 or 1")
        self.chart=chart;self.uv=q if chart==0 else q0
        self.xyz=x;self.original_ids=np.flatnonzero(c==chart)
        if not len(self.original_ids):raise ValueError("Empty chart")
        self.faces=f[self.original_ids]
        self.surface=Surface(self.uv,self.xyz,self.faces)
    def footprint_support(self,*args,**kwargs):
        return self.surface.footprint_support(*args,**kwargs)
    def sample_grid(self,*args,**kwargs):
        r=self.surface.sample_grid(*args,**kwargs)
        ids=np.full(r["valid"].shape,-1,np.int64);v=r["valid"]
        ids[v]=self.original_ids[r["source_triangle_id"][v]]
        r["original_face_id"]=ids
        return r
    def dense_crop_bounds(self,origin,step,shape,depths):
        """Bound all mapped footprint points by intersecting original face vertices."""
        origin=np.asarray(origin,float);step=np.asarray(step,float)
        h,w=shape;low=origin-step/2;high=origin+step*np.array([w-.5,h-.5])
        pu=self.uv[self.faces]
        guard=16*np.finfo(float).eps*np.maximum(1,np.maximum(np.abs(low),np.abs(high)))
        ids=np.flatnonzero((pu.max(1)>=low-guard).all(1)&(pu.min(1)<=high+guard).all(1))
        if not len(ids):raise ValueError("No faces in ROI")
        xyz=self.xyz[np.unique(self.faces[ids])]
        # Unit normals imply every depth ray is within max abs depth in each axis.
        margin=float(np.max(np.abs(depths)))+2
        lo=np.floor(xyz.min(0)-margin).astype(np.int64)
        hi=np.ceil(xyz.max(0)+margin+1).astype(np.int64)
        if np.prod(hi-lo,dtype=np.int64)>4_000_000:raise ValueError("Dense crop exceeds 4M voxels")
        return lo,hi
    def export_npz(self,path):
        """Standard input geometry plus sorted local->original face index map."""
        p=pathlib.Path(path).resolve()
        if p.exists() or any((a/".git").exists() for a in (p.parent,*p.parent.parents)):
            raise ValueError("Use new output outside Git")
        np.savez_compressed(p,uv=self.uv,xyz=self.xyz,triangles=self.faces,original_face_ids=self.original_ids,chart_id=self.chart)

def self_test(Surface):
    # Both charts use separate disjoint original triangles, not one UV surface.
    q=np.array([[0,0],[1,0],[0,1],[2,0],[3,0],[2,1]],float)
    xyz=np.c_[q,np.ones(len(q))]
    f=np.array([[0,1,2],[3,4,5]])
    a=(q,q,xyz,f,np.array([1,0],dtype=np.uint8))
    b=ChartBridge(a,0,Surface);r=b.sample_grid([2.25,.25],[.1,.1],(1,1))
    assert r["valid"].item() and r["original_face_id"].item()==1
    c=ChartBridge(a,1,Surface)
    r=c.sample_grid([.45,.45],[.3,.3],(1,1))
    assert r["valid"].item() and r["original_face_id"].item()==0
    # Center supported; whole pixel crosses unsupported triangle boundary.
    assert not c.footprint_support([.45,.45],[.3,.3],(1,1))["valid"].item()
    assert not b.sample_grid([.2,.2],[.1,.1],(1,1))["valid"].item()
    return 4

def run(atlas_file,shared,gemini,out):
    start=time.monotonic()
    surface=module("shared_original_surface",shared/"surface.py")
    renderer=module("shared_footprint_renderer",shared/"render.py")
    tests=self_test(surface.OriginalSurface)
    atlas=load_atlas(atlas_file,PIN)
    indexed=module("adaptive_indexed_roi",HERE.parent/"indexed_native_renderer"/"run.py")
    sys.path.insert(0,str(gemini))
    import grow_m7_patch as gm
    cap=80*1024**2;budget=gm.ByteBudget(cap)
    data=[]
    with gm.BudgetedRemoteZarrV2(CT_URL,level="0",budget=budget,workers=1,cache_mb=128,max_retries=1,timeout=15) as reader:
        if reader.dtype!=np.uint8:raise ValueError("Expected native uint8")
        for chart in (0,1):
            row={"chart":chart}
            try:
                t=time.monotonic()
                b=ChartBridge(atlas,chart,surface.OriginalSurface)
                center=np.array([320.,118.]) if chart==0 else np.floor(b.uv[b.faces[len(b.faces)//2]].mean(0))
                origin=center-.8+.0125;step=np.array([PITCH,PITCH]);shape=(32,32)
                row["published_roi"]={"origin_uv":origin.tolist(),"shape_yx":list(shape),"pitch":PITCH}
                ref=indexed.FaceROIIndex(b.uv,b.xyz,b.faces,b.original_ids,pitch=PITCH).render(origin,shape)
                point=b.sample_grid(origin,step,shape)
                v=point["valid"];common=v&ref["valid"]
                err=float(np.max(np.abs(point["xyz"][common]-ref["xyz"][common]))) if common.any() else None
                werr=float(np.max(np.abs(point["barycentric_weights"][common]-ref["barycentric_weights"][common]))) if common.any() else None
                exact={k:bool(np.array_equal(point[k],ref[k])) for k in ("valid","original_face_id","coverage_count","reason_bits","ambiguity_mask")}
                row["point_parity"]={"exact":exact,"max_xyz_error_vox":err,"max_weight_error":werr,"valid_pixels":int(v.sum())}
                footprint=b.footprint_support(origin,step,shape)
                fv=footprint["valid"]
                row["footprint"]={"valid_pixels":int(fv.sum()),"point_only_pixels":int((v&~fv).sum()),"footprint_without_point":int((fv&~v).sum()),
                    "reason_counts":{str(k):int((footprint["reason_bits"]&k!=0).sum()) for k in (1,2,4,8,16,32,64)}}
                lo,hi=b.dense_crop_bounds(origin,step,shape,DEPTHS)
                block,org,support=reader.read_box(lo[::-1],hi[::-1],return_support=True)
                row["dense_crop"]={"shape_zyx":list(block.shape),"all_source_available":bool(support.all()),"source_voxels":int(block.size)}
                if tuple(org)!=tuple(lo[::-1]) or tuple(block.shape)!=tuple((hi-lo)[::-1]) or not support.all():
                    raise ValueError("Dense crop clipped or contains unavailable source voxels")
                volume=block.astype(np.float32)
                t1=time.monotonic()
                s1,m1,d1=renderer.render(volume,np.asarray(org[::-1],float),b,origin,step,shape,DEPTHS,samples_per_axis=1)
                t2=time.monotonic()
                s4,m4,d4=renderer.render(volume,np.asarray(org[::-1],float),b,origin,step,shape,DEPTHS,samples_per_axis=4)
                t3=time.monotonic()
                coords=point["xyz"][None]+DEPTHS[:,None,None,None]*point["normals"][None]
                # Only compare the whole-footprint arm's accepted pixels.
                q=coords[:,m1,:]
                values,available=gm.precise_sample_trilinear(reader,q,return_support=True)
                diff=s1[:,m1].astype(float)-values.astype(float)
                both=m1&m4
                delta=s4[:,both].astype(float)-s1[:,both].astype(float)
                row["CT"]={"n1_valid_pixels":int(m1.sum()),"n4_valid_pixels":int(m4.sum()),"common_pixels":int(both.sum()),
                           "masks_equal":bool(np.array_equal(m1,m4)),
                           "n1_equals_footprint_and_point":bool(np.array_equal(m1,fv&v)),
                           "all_precise_source_supported":bool(available.all() and available.size>0),
                           "point_CT_max_abs_diff":float(np.max(np.abs(diff))) if diff.size else None,
                           "point_CT_compared_samples":int(diff.size),
                           "quadrature_rms_change":float(np.sqrt(np.mean(delta**2))) if delta.size else None,
                           "quadrature_max_abs_change":float(np.max(np.abs(delta))) if delta.size else None,
                           "n1_sha256":hashlib.sha256(s1.tobytes()+m1.tobytes()).hexdigest(),
                           "n4_sha256":hashlib.sha256(s4.tobytes()+m4.tobytes()).hexdigest(),
                           "n1_seconds":t2-t1,"n4_seconds":t3-t2,
                           "n1_renderer_diagnostics":d1,"n4_renderer_diagnostics":d4}
                row["gates"]={"point_parity":bool(all(exact.values()) and err is not None and err<=1e-6 and werr is not None and werr<=1e-9),
                              "whole_footprint_subset_and_support":bool(not (fv&~v).any() and fv.sum()>=(.95*1024 if chart==0 else 50)),
                              "full_depth_real_CT_parity":bool(available.size and available.all() and np.max(np.abs(diff))<=.00003),
                              "quadrature_common_support":bool(np.array_equal(m1,m4) and np.array_equal(m1,fv&v) and m1.any())}
                row["wall_seconds"]=time.monotonic()-t
            except Exception as e:
                row["failure"]=f"{type(e).__name__}: {str(e)[:400]}";row["gates"]={"completed":False}
            data.append(row)
    report={"schema":"adaptive.shared_renderer_bridge.v1","source_atlas_sha256":PIN,"script_sha256":sha(__file__),
            "shared_source_hashes":{n:sha(shared/n) for n in ("surface.py","footprint.py","render.py")},
            "gemini_source_hashes":{n:sha(gemini/n) for n in ("grow_m7_patch.py","stream_snap_render.py")},
            "unit_checks":tests,"depths_native_voxels":DEPTHS.tolist(),"native_pitch_um":8.64,"cases":data,
            "body_bytes":budget.used,"charged_requests":budget.requests,"cap_bytes":cap,
            "pass":bool(len(data)==2 and all(all(c["gates"].values()) for c in data) and budget.used<=cap),
            "CNN_forwards":0,"wall_seconds":time.monotonic()-start,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    pathlib.Path(out).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--shared",type=pathlib.Path,required=True);ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--atlas",type=pathlib.Path);ap.add_argument("--gemini",type=pathlib.Path);ap.add_argument("--output",type=pathlib.Path)
    a=ap.parse_args()
    if a.self_test:
        m=module("shared_original_surface",a.shared/"surface.py")
        print("Tests passed:",self_test(m.OriginalSurface))
    else:
        if not a.atlas or not a.gemini or not a.output or a.output.exists():ap.error("atlas, gemini, new output required")
        run(a.atlas,a.shared,a.gemini,a.output)
