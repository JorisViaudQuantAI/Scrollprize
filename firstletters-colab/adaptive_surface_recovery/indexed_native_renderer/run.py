#!/usr/bin/env python3
"""Exact face preselection before original-triangle native ROI rasterization."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="4"
import argparse,hashlib,json,pathlib,sys,time,resource,statistics
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"metric_flatten"))
from topology import rasterize_triangles
FIELDS=("xyz","valid","barycentric_weights","coverage_count","reason_bits","ambiguity_mask","original_face_id","full_face_query_count")
PITCH=.05
SHA="43c12969b32c42b978e9d1cbb3bd70fb645a998100062d889d66228a130fd8d0"

def sha(p):
    with pathlib.Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

class FaceROIIndex:
    """The original rasterizer is unchanged. Only face enumeration is filtered.
    Index per pitch. All source IDs are retained, sorted in original input order.
    Component-level counts returned by the rasterizer are ROI-local.
    """
    def __init__(self,uv,xyz,faces,original_ids=None,*,pitch=.05):
        self.uv=np.asarray(uv,np.float64)/pitch
        self.xyz=np.asarray(xyz,np.float64)
        self.faces=np.asarray(faces,np.int64)
        self.ids=np.arange(len(faces)) if original_ids is None else np.asarray(original_ids,np.int64)
        self.pitch=pitch
        if self.ids.shape!=(len(self.faces),) or len(np.unique(self.ids))!=len(self.ids):raise ValueError("face IDs")
        if not np.isfinite(pitch) or pitch<=0 or not np.isfinite(self.uv).all():raise ValueError("bad pitch/UV")
        t=self.uv[self.faces]
        self.lo=t.min(1);self.hi=t.max(1)
        a=t[:,1]-t[:,0];b=t[:,2]-t[:,0]
        det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        self.pad=np.where(np.abs(det)<=1e-12,1e-9,1e-9*(np.abs(a)+np.abs(b)).max(1)*2)
    def select(self,origin_xy,shape):
        origin=np.asarray(origin_xy,np.float64)/self.pitch
        # Match operation order of original rasterizer; do not use a lo+pad
        # precomputation, which can differ at an integer boundary.
        lo=np.ceil(self.lo-origin-self.pad[:,None])
        hi=np.floor(self.hi-origin+self.pad[:,None])
        return np.flatnonzero((np.maximum(lo,0)<=np.minimum(hi,[shape[1]-1,shape[0]-1])).all(1))
    def render(self,origin_xy,shape,*,baseline=False,query_mask=None):
        ii=np.arange(len(self.faces)) if baseline else self.select(origin_xy,shape)
        r=rasterize_triangles(self.uv,self.xyz,self.faces[ii],shape,origin_xy=np.asarray(origin_xy,float)/self.pitch,query_mask=query_mask)
        valid=r["valid"];owners=np.full(valid.shape,-1,np.int64)
        owners[valid]=self.ids[ii[r["source_triangle_id"][valid]]]
        counts=np.zeros(len(self.faces),np.int64);counts[ii]=r["source_triangle_covered_query_count"]
        r["original_face_id"]=owners;r["full_face_query_count"]=counts
        r["faces_considered"]=len(ii);r["source_component_counts_scope"]="selected_faces_for_this_ROI"
        return r

def exact(a,b):
    return {k:bool(np.array_equal(a[k],b[k],equal_nan=True)) for k in FIELDS}

def self_test():
    # Ordinary adjacent triangles, an inverted face, an epsilon-degenerate face,
    # and two nonadjacent coincident/overlapping positive triangles.
    q=np.array([[0,0],[1,0],[0,1],[1,1],[.1,.2],[.9,.2],[.2,.9],[2,0],[3,0],[4,0]],float)
    x=np.c_[q,np.ones(len(q))]
    f=np.array([[0,1,2],[1,3,2],[0,2,1],[7,8,9],[4,5,6]])
    idx=FaceROIIndex(q,x,f,np.arange(len(f))+100,pitch=.1)
    cases=[((0,0),(11,11),None),((.025,.025),(11,11),None),((1.9,0),(3,23),None),((50,50),(4,4),None)]
    mask=np.ones((11,11),bool);mask[::2,::2]=False;cases.append(((0,0),(11,11),mask))
    for o,s,m in cases:
        a=idx.render(o,s,baseline=True,query_mask=m);b=idx.render(o,s,query_mask=m)
        assert all(exact(a,b).values()),(o,exact(a,b))
    return len(cases)

def ct_check(reader,gm,xyz,f,a,b):
    def rays(r):
        # Central native 64x64, ray convention is the original source face normal.
        pts=r["xyz"][48:112,48:112].copy()
        valid=r["valid"][48:112,48:112];ids=r["original_face_id"][48:112,48:112]
        normals=np.zeros_like(pts)
        tri=xyz[f[ids[valid]]]
        n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        normals[valid]=n/np.linalg.norm(n,axis=1)[:,None]
        return pts[None]+np.arange(-2,3)[:,None,None,None]*normals[None],valid
    p,va=rays(a);q,vb=rays(b)
    start=time.monotonic()
    v1,s1=gm.precise_sample_trilinear(reader,p,return_support=True)
    v2,s2=gm.precise_sample_trilinear(reader,q,return_support=True)
    same=bool(np.array_equal(p,q,equal_nan=True) and np.array_equal(v1,v2,equal_nan=True) and np.array_equal(s1,s2) and np.array_equal(va,vb))
    support=va&s1.all(0)
    h=hashlib.sha256()
    h.update(np.ascontiguousarray(v1).tobytes());h.update(np.ascontiguousarray(s1).tobytes())
    return {"exact_values_rays_support":same,"mesh_valid_pixels":int(va.sum()),"full_source_supported_pixels":int(support.sum()),
        "full_source_supported_fraction_of_mesh":float(support.sum()/max(1,va.sum())),"stack_and_support_sha256":h.hexdigest(),
        "supported_value_range":[float(v1[s1].min()),float(v1[s1].max())] if s1.any() else None,
        "seconds_both_CT_queries":time.monotonic()-start}

def run(src,gemini,out):
    started=time.monotonic();tests=self_test()
    if sha(src)!=SHA:raise ValueError("Pinned atlas hash differs")
    with np.load(src,allow_pickle=False) as p:
        q0,q,x,f,ch=[p[k].copy() for k in ("uv_initial","uv_corrected","source_xyz","source_triangles","chart_id")]
    pf=np.flatnonzero(ch==0);ff=np.flatnonzero(ch==1)
    before=time.monotonic()
    indexes={"primary":FaceROIIndex(q,x,f[pf],pf,pitch=PITCH),
             "fallback":FaceROIIndex(q0,x,f[ff],ff,pitch=PITCH),
             "unsafe_B":FaceROIIndex(q,x,f,pitch=PITCH)}
    setup=time.monotonic()-before
    cases=[]
    for c in [(140,90),(230,110),(320,118),(340,140)]:cases.append(("primary",c))
    for quant in [.25,.5,.75]:
        face=ff[int(len(ff)*quant)];cases.append(("fallback",tuple(np.floor(q0[f[face]].mean(0)).astype(int))))
    for quant in [.25,.75]:
        face=ff[int(len(ff)*quant)];cases.append(("unsafe_B",tuple(np.floor(q[f[face]].mean(0)).astype(int))))
    cases.append(("primary",(-20,-20)))
    data=[];ct_pairs={}
    for case_i,(chart,center) in enumerate(cases):
        origin=tuple(float(v)-4+.0125 for v in center);idx=indexes[chart]
        times={"baseline":[],"indexed":[]}
        for rep in range(2):
            results={}
            for name in (("baseline","indexed") if rep==0 else ("indexed","baseline")):
                t=time.perf_counter();results[name]=idx.render(origin,(160,160),baseline=name=="baseline")
                times[name].append(time.perf_counter()-t)
            same=exact(results["baseline"],results["indexed"])
            if not all(same.values()):raise AssertionError((case_i,same))
        a,b=results["baseline"],results["indexed"]
        data.append({"case":case_i,"chart":chart,"public_center_xy":list(map(int,center)),"origin_xy":list(origin),"shape":[160,160],
            "baseline_face_count":a["faces_considered"],"selected_face_count":b["faces_considered"],
            "valid_pixels":int(a["valid"].sum()),"ambiguous_pixels":int(a["ambiguity_mask"].sum()),
            "source_faces_with_query":int((a["full_face_query_count"]>0).sum()),
            "exact":same,"seconds":times,"median_speedup":statistics.median(times["baseline"])/statistics.median(times["indexed"])})
        if case_i in (2,5):ct_pairs[chart]=(a,b)
    total_b=sum(sum(z["seconds"]["baseline"]) for z in data)
    total_i=setup+sum(sum(z["seconds"]["indexed"]) for z in data)
    primary_speed=statistics.median(z["median_speedup"] for z in data[:4])
    sys.path.insert(0,str(gemini))
    import grow_m7_patch as gm
    import stream_snap_render as ss
    cap=80*1024**2;budget=gm.ByteBudget(cap)
    ct={};ct_error=None
    url="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr"
    try:
        with gm.BudgetedRemoteZarrV2(url,level="0",budget=budget,workers=1,cache_mb=128,max_retries=1,timeout=15) as reader:
            if reader.dtype!=np.uint8:raise ValueError("Expected native uint8")
            for chart,(a,b) in ct_pairs.items():ct[chart]=ct_check(reader,gm,x,f,a,b)
    except Exception as e:
        ct_error=f"{type(e).__name__}: {str(e)[:400]}"
    gates={"all_exact":True,"primary_median_speedup_ge5":primary_speed>=5,
        "total_speedup_including_setup_ge3":total_b/total_i>=3,
        "real_CT_exact_supported":ct_error is None and len(ct)==2 and all(z["exact_values_rays_support"] and z["full_source_supported_pixels"]>=50 for z in ct.values()) and budget.used<=cap}
    report={"schema":"q10.indexed_native_renderer.v1","source_atlas_sha256":SHA,"synthetic_checks_passed":tests,
      "script_sha256":sha(__file__),"pitch_um":8.64,"index_setup_seconds":setup,"cases":data,
      "primary_median_speedup":primary_speed,"total_baseline_seconds":total_b,"total_indexed_seconds_including_setup":total_i,
      "total_speedup":total_b/total_i,"CT":{"source":url,"body_bytes":budget.used,"requests_charged":budget.requests,"cap_bytes":cap,"cases":ct,"error":ct_error},
      "source_module_hashes":{"grow_m7_patch.py":sha(gemini/"grow_m7_patch.py"),"stream_snap_render.py":sha(gemini/"stream_snap_render.py")},
      "gates":gates,"pass":all(gates.values()),"CNN_forwards":0,"wall_seconds":time.monotonic()-started,
      "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    pathlib.Path(out).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--self-test",action="store_true");ap.add_argument("--atlas",type=pathlib.Path)
    ap.add_argument("--gemini",type=pathlib.Path);ap.add_argument("--output",type=pathlib.Path);a=ap.parse_args()
    if a.self_test:print("Tests passed:",self_test())
    else:
        if not a.atlas or not a.gemini or not a.output or a.output.exists():ap.error("atlas, gemini and new output required")
        run(a.atlas,a.gemini,a.output)
