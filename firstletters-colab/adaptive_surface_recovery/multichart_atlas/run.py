#!/usr/bin/env python3
"""Two-chart source-face atlas for public geometry; never blend across charts."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="4"
import argparse, hashlib, json, pathlib, sys, time, resource
import numpy as np
import shapely
from shapely import STRtree
from scipy.interpolate import LinearNDInterpolator
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"metric_flatten"))
from flatten import solve_mesh
from topology import rasterize_triangles,triangle_metrics
import reference
from experiment import grid_metrics,grid_summary,four_corner,box

def det(uv,f):
    t=uv[f];a=t[:,1]-t[:,0];b=t[:,2]-t[:,0]
    return a[:,0]*b[:,1]-a[:,1]*b[:,0]

def intersections(uv,faces):
    """Mark both incompatible participants. Batches bound intersection work."""
    faces=np.asarray(faces,dtype=np.int64)
    if (det(uv,faces)<=1e-12).any(): raise ValueError("Require positive faces")
    polys=shapely.polygons(uv[faces]); tree=STRtree(polys)
    bad=np.zeros(len(faces),bool)
    counts={"candidate_intersecting_pairs":0,"positive_area_pairs":0,
            "nonincident_touch_pairs":0,"single_vertex_long_contact_pairs":0,"incompatible_pairs":0}
    for start in range(0,len(faces),2048):
        i,j=tree.query(polys[start:start+2048],predicate="intersects")
        i=i+start; take=i<j; i,j=i[take],j[take]
        counts["candidate_intersecting_pairs"]+=len(i)
        for lo in range(0,len(i),8192):
            a,b=i[lo:lo+8192],j[lo:lo+8192]
            shared=(faces[a,:,None]==faces[b,None,:]).any(2).sum(1)
            overlap=shapely.intersection(polys[a],polys[b])
            area=np.asarray(shapely.area(overlap))
            length=np.asarray(shapely.length(overlap))
            if not np.isfinite(area).all() or not np.isfinite(length).all(): raise ValueError("Nonfinite intersection")
            big=area>1e-10
            foreign=(shared==0)&(~big)
            line=(shared==1)&(~big)&(length>1e-9)
            fail=big|foreign|line
            bad[a[fail]]=True; bad[b[fail]]=True
            for k,x in [("positive_area_pairs",big),("nonincident_touch_pairs",foreign),("single_vertex_long_contact_pairs",line),("incompatible_pairs",fail)]:
                counts[k]+=int(x.sum())
    return bad,counts

def partition(initial,corrected,faces):
    bad=det(corrected,faces)<=1e-12
    flips=int(bad.sum()); eligible=np.flatnonzero(~bad)
    conflicts,counts=intersections(corrected,faces[eligible]);bad[eligible[conflicts]]=True
    return ~bad,{"nonpositive_or_degenerate_faces":flips,"additional_conflicting_positive_faces":int(conflicts.sum()),"initial_positive_pair_counts":counts}

def centroids_unique(uv,faces):
    p=shapely.polygons(uv[faces]);tree=STRtree(p);bad=0
    for start in range(0,len(p),4096):
        stop=min(start+4096,len(p));n=stop-start
        i,j=tree.query(shapely.centroid(p[start:stop]),predicate="within")
        counts=np.bincount(i,minlength=n);own=np.bincount(i,weights=(j==i+start).astype(int),minlength=n)
        bad+=int(((counts!=1)|(own!=1)).sum())
    return bad

def render_roi(uv,xyz,faces,*,origin_xy,shape,pitch=1):
    """Evaluate an individual chart directly at requested spacing in UV units.
    Never interpolate a raster mask or mix different charts. Bounded integer lattice."""
    if not np.isfinite(pitch) or pitch<=0: raise ValueError("Invalid pitch")
    if int(np.prod(shape))>4_000_000: raise ValueError("Partition ROI into <=4M queries")
    origin=np.asarray(origin_xy,float)/pitch
    return rasterize_triangles(uv/pitch,xyz,faces,shape,origin_xy=origin)

def self_test():
    q=np.array([[0,0],[1,0],[0,1],[1,1]],float);f=np.array([[0,1,2],[1,3,2]])
    assert not intersections(q,f)[0].any()
    a=np.vstack([q,[[.2,.2],[.8,.2],[.2,.8]]]);g=np.array([[0,1,2],[4,5,6]])
    assert intersections(a,g)[0].all()
    a=np.vstack([q,[[1,0],[2,0],[1,-1]]]);g=np.array([[0,1,2],[4,6,5]])
    assert intersections(a,g)[0].all()
    x=q.copy();x[3]=[.1,.1]
    keep,_=partition(q,x,f);assert keep.tolist()==[True,False]
    assert centroids_unique(q,f)==0
    xyz=np.c_[q,np.ones(len(q))]
    r=render_roi(q,xyz,f,origin_xy=(0,0),shape=(3,3),pitch=.5)
    assert r["valid"].all() and np.allclose(r["xyz"][1,1],[.5,.5,1])
    return 5

def raster(uv,xyz,faces,ids,shape0):
    face=faces[ids];pts=uv[np.unique(face)]
    lo=np.floor(pts.min(0)).astype(int);hi=np.ceil(pts.max(0)).astype(int)
    # Include full source carrier frame to measure exact B query coverage.
    lo=np.minimum(lo,[0,0]);hi=np.maximum(hi,[shape0[1]-1,shape0[0]-1])
    shape=(int(hi[1]-lo[1]+1),int(hi[0]-lo[0]+1))
    r=render_roi(uv,xyz,face,origin_xy=lo,shape=shape)
    valid=r["valid"];idx=r["source_triangle_id"][valid];weights=r["barycentric_weights"][valid]
    reconstructed=np.einsum("ij,ijk->ik",weights,xyz[face[idx]])
    err=float(np.max(np.abs(reconstructed-r["xyz"][valid]))) if valid.any() else None
    provenance=bool(valid.any() and (r["reason_bits"][valid]==0).all() and (weights>=-1e-9).all() and np.allclose(weights.sum(1),1,atol=1e-9,rtol=0) and err<=1e-6)
    counts={**r["counts"],"original_face_centroids_nonunique":centroids_unique(uv,face),
            "max_barycentric_xyz_error_vox":err,"provenance_pass":provenance,"shape_yx":list(shape),"origin_xy":lo.tolist()}
    fullids=np.full(valid.shape,-1,dtype=np.int64);fullids[valid]=ids[idx]
    y,x=-int(lo[1]),-int(lo[0]);h,w=shape0
    return r,fullids,counts,(slice(y,y+h),slice(x,x+w))

def run(cache,out,atlas):
    started=time.monotonic();tests=self_test();acq=reference.prepare(cache)
    p,_=reference.reconstruct_touchup(cache);target=reference.load_published_xyz(cache)
    config=json.loads((HERE.parents[1]/"metric_flatten"/"protocol.json").read_text())["geometry"]
    spacing=float(config["spacing_voxels"]);voxel=float(config["source_voxel_um"]);pitch=spacing*voxel/1000
    jj,ii=np.meshgrid(np.arange(p.shape[1]),np.arange(p.shape[0]))
    u=config["carrier_origin_mm"][0]+jj*pitch;v=config["carrier_origin_mm"][1]+ii*pitch
    reg=np.isfinite(p).all(-1)&box(u,v,config["region_mm"],upper_inclusive=True)
    anchor=box(u,v,config["anchor_mm"],upper_inclusive=True)
    fix=int(np.argmin(np.abs(u[reg])+np.abs(v[reg])))
    s=solve_mesh(p,spacing,reference.load_arap_functions(cache),region_mask=reg,anchor_mask=anchor,iterations=60,fold_weight=.02,fold_threshold=1.25,fix_index=fix)
    q0,q,xyz,f=[s[k] for k in ("uv_initial","uv","xyz","triangles")]
    b=np.full(p.shape,np.nan);b[reg]=LinearNDInterpolator(q,xyz)(q0);b=b.astype(np.float32).astype(float)
    vb=np.isfinite(b).all(-1);vt=np.isfinite(target).all(-1)
    if not np.array_equal(vb,vt) or not np.array_equal(b[vb],target[vt]): raise ValueError("Exact B replay failed")
    keep,partition_stats=partition(q0,q,f)
    primary=np.flatnonzero(keep);fallback=np.flatnonzero(~keep)
    if not len(primary) or not len(fallback): raise ValueError("Expected two nonempty face sets")
    badp,cp=intersections(q,f[primary]);badf,cf=intersections(q0,f[fallback])
    m0=triangle_metrics(q0,xyz/spacing,f);m1=triangle_metrics(q,xyz/spacing,f)
    a=m0["source_3d_area"];pfrac=float(a[keep].sum()/a.sum())
    def bad(m):
        sing=m["singular_values"];return ~np.isfinite(sing).all(1)|np.any(np.abs(sing-1)>.05,axis=1)
    b0,b1=bad(m0),bad(m1);combined=np.where(keep,b1,b0)
    def frac(mask,domain):
        return float(a[mask&domain].sum()/a[domain].sum())
    allfaces=np.ones(len(f),bool)
    metric={"A_bad_fraction":float(b0.mean()),"B_bad_fraction":float(b1.mean()),"atlas_bad_fraction":float(combined.mean()),
            "A_area_weighted_bad_fraction":frac(b0,allfaces),"B_area_weighted_bad_fraction":frac(b1,allfaces),
            "atlas_area_weighted_bad_fraction":frac(combined,allfaces),"primary_area_weighted_bad_fraction":frac(b1,keep),
            "fallback_area_weighted_bad_fraction":frac(b0,~keep)}
    rp,ip,sp,slp=raster(q,xyz,f,primary,p.shape[:2]);rf,iff,sf,slf=raster(q0,xyz,f,fallback,p.shape[:2])
    vp=rp["valid"][slp];pr=rp["xyz"][slp].astype(np.float32).astype(float)
    crop=np.zeros_like(reg);r0,r1=config["claim_crop_rows_half_open"];c0,c1=config["claim_crop_columns_half_open"];crop[r0:r1,c0:c1]=True
    retain=float((vp&vb&crop).sum()/(vb&crop).sum())
    common=four_corner(np.isfinite(p).all(-1))&four_corner(vb)&four_corner(vp)
    right=box(u[:-1,:-1]+pitch/2,v[:-1,:-1]+pitch/2,[15.5,23.5,-2.7,1.6])
    g={k:grid_summary(grid_metrics(z,spacing),common&right) for k,z in [("A",p),("B",b),("primary",pr)]}
    edge=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    assignment=np.tile(keep.astype(np.int8),3);_,inv=np.unique(edge,axis=0,return_inverse=True)
    n=np.bincount(inv);pcount=np.bincount(inv,weights=assignment,minlength=len(n))
    seams=int(((n==2)&(pcount==1)).sum())
    area_sum=float(a[keep].sum()+a[~keep].sum())
    checks={"exact_B":True,"face_partition_exact":bool(np.array_equal(np.sort(np.r_[primary,fallback]),np.arange(len(f))) and abs(area_sum-a.sum())/a.sum()<=1e-10),
      "zero_incompatible_intersections":bool(not badp.any() and not badf.any()),
      "zero_nonpositive_faces":bool((det(q,f[keep])>1e-12).all() and (det(q0,f[~keep])>1e-12).all()),
      "primary_area_ge_90pct":pfrac>=.9,"weighted_metric_improvement_ge_10pp":metric["A_area_weighted_bad_fraction"]-metric["atlas_area_weighted_bad_fraction"]>=.1,
      "primary_claim_support_ge_995":retain>=.995,
      "right_group":bool(g["primary"]["cells"]>=20 and g["primary"]["median_abs_v_stretch_error"]<=.5*g["A"]["median_abs_v_stretch_error"] and g["primary"]["either_axis_outside_5pct_fraction"]<=.05 and g["primary"]["p95_abs_shear_degrees"]<=5),
      "unique_centroids_and_raster_provenance":bool(all(z["provenance_pass"] and z["original_face_centroids_nonunique"]==0 for z in (sp,sf)))}
    atlas.mkdir(exist_ok=False)
    np.savez_compressed(atlas/"source_atlas.npz",uv_initial=q0,uv_corrected=q,source_xyz=xyz,source_triangles=f,chart_id=np.where(keep,0,1).astype(np.uint8),source_voxel_um=voxel,source_spacing_voxels=spacing)
    for name,rr,fi,st in [("primary",rp,ip,sp),("fallback",rf,iff,sf)]:
        np.savez_compressed(atlas/(name+"_coarse_samples.npz"),xyz=rr["xyz"].astype(np.float32),valid=rr["valid"],original_face_id=fi,barycentric_weights=rr["barycentric_weights"],reason_bits=rr["reason_bits"],origin_xy=np.asarray(st["origin_xy"]))
    report={"schema":"q10.multichart_atlas.v1","public_control":"PHerc0306B","synthetic_checks_passed":tests,
      "acquisition":{k:acq[k] for k in ("requests","body_bytes") if k in acq},
      "source_manifest_sha256":hashlib.sha256((HERE.parents[1]/"metric_flatten"/"source_manifest.json").read_bytes()).hexdigest(),
      "script_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
      "source_vertices":len(xyz),"source_faces":len(f),"primary_faces":len(primary),"fallback_faces":len(fallback),
      "primary_source_area_fraction":pfrac,"fallback_source_area_fraction":1-pfrac,"total_3d_area_mm2":float(a.sum()*pitch*pitch),
      "partition":partition_stats,"primary_recheck":cp,"fallback_recheck":cf,"seam_edges":seams,
      "metric":metric,"primary_same_grid_B_retention":float((vp&vb).sum()/vb.sum()),"primary_claim_B_retention":retain,
      "right_group":g,"primary_raster":sp,"fallback_raster":sf,
      "artifacts":{x.name:{"bytes":x.stat().st_size,"sha256":hashlib.file_digest(x.open("rb"),"sha256").hexdigest()} for x in atlas.iterdir()},
      "gates":checks,"pass":all(checks.values()),"CT_requests":0,"CNN_forwards":0,
      "wall_seconds":time.monotonic()-started,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    pathlib.Path(out).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--cache",type=pathlib.Path);ap.add_argument("--output",type=pathlib.Path);ap.add_argument("--atlas",type=pathlib.Path)
    args=ap.parse_args()
    if args.self_test:print("Tests passed:",self_test())
    else:
        if not args.cache or not args.output or not args.atlas or args.output.exists() or args.atlas.exists():ap.error("new output and atlas paths required")
        run(args.cache,args.output,args.atlas)
