#!/usr/bin/env python3
"""Fixed-boundary, orientation-constrained UV correction; public 306B replay."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="4"
import argparse, hashlib, json, pathlib, sys, time, resource
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import LinearNDInterpolator
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"metric_flatten"))
from flatten import solve_mesh
from topology import rasterize_triangles,triangle_metrics
import reference
from experiment import grid_metrics,grid_summary,four_corner,box

def determinant(q,f):
    t=q[f]; a=t[:,1]-t[:,0]; b=t[:,2]-t[:,0]
    return a[:,0]*b[:,1]-a[:,1]*b[:,0]

def constrain(initial,target,faces,ij):
    """Use original grid color classes, fixed source boundary and exact face limits."""
    initial=np.asarray(initial,dtype=float); target=np.asarray(target,dtype=float)
    f=np.asarray(faces,dtype=np.int64); n=len(initial)
    if initial.shape!=target.shape or not np.isfinite(target).all(): raise ValueError("target shape or finite")
    d0=determinant(initial,f)
    if (d0<=0).any(): raise ValueError("initial UV is not positively oriented")
    edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    edges,multiplicity=np.unique(edges,axis=0,return_counts=True)
    if (multiplicity>2).any(): raise ValueError("nonmanifold source edge")
    boundary=np.zeros(n,bool); boundary[edges[multiplicity==1].ravel()]=True
    u,v=edges.T
    G=sparse.coo_matrix((np.ones(2*len(edges)),(np.r_[u,v],np.r_[v,u])),shape=(n,n)).tocsr()
    L=sparse.diags(np.asarray(G.sum(1)).ravel())-G
    displacement=target-initial
    H=np.zeros_like(initial); H[boundary]=displacement[boundary]
    free=np.flatnonzero(~boundary); fixed=np.flatnonzero(boundary)
    if not len(free) or not len(fixed): raise ValueError("no interior/boundary")
    H[free]=spsolve(L[free][:,free].tocsc(),-(L[free][:,fixed]@H[fixed]))
    desired=target-H
    desired[boundary]=initial[boundary]
    colors=(ij[:,0]%2)*2+ij[:,1]%2
    if np.any(np.diff(np.sort(colors[f],axis=1),axis=1)==0): raise ValueError("color conflict")
    q=initial.copy(); floor=.05*d0
    stats=[]
    for sweep in range(24):
        for color in range(4):
            move=(colors==color)&(~boundary)
            relevant=move[f].any(1)
            ff=f[relevant]
            if not len(ff): continue
            which=ff[move[ff]]
            assert len(which)==len(ff)
            old=determinant(q,ff)
            trial=q.copy(); trial[move]=desired[move]
            new=determinant(trial,ff)
            limited=new<floor[relevant]
            alpha=np.ones(n)
            if limited.any():
                bound=.95*(old[limited]-floor[relevant][limited])/(old[limited]-new[limited])
                np.minimum.at(alpha,which[limited],np.clip(bound,0,1))
            q[move]+=alpha[move,None]*(desired[move]-q[move])
            if (determinant(q,ff)<floor[relevant]-1e-9).any(): raise ArithmeticError("face floor violated")
        stats.append(float(np.sqrt(np.mean(np.sum((q-desired)**2,axis=1)))))
    return q,boundary,{"boundary_vertices":int(boundary.sum()),"interior_vertices":int((~boundary).sum()),
        "minimum_det_ratio":float(np.min(determinant(q,f)/d0)),
        "target_rmse_by_sweep":stats,"harmonic_target_max_displacement":float(np.linalg.norm(desired-initial,axis=1).max()),
        "maximum_actual_displacement":float(np.linalg.norm(q-initial,axis=1).max())}

def self_test():
    row,col=np.indices((7,7)); ij=np.stack([row.ravel(),col.ravel()],1); q=ij[:,::-1].astype(float)
    ids=np.arange(49).reshape(7,7); a,b,c,d=ids[:-1,:-1].ravel(),ids[:-1,1:].ravel(),ids[1:,:-1].ravel(),ids[1:,1:].ravel()
    f=np.concatenate([np.stack([a,b,d],1),np.stack([a,d,c],1)])
    x=q.copy(); x[24]=[-3,8]
    z,bound,meta=constrain(q,x,f,ij)
    assert np.array_equal(z[bound],q[bound]) and meta["minimum_det_ratio"]>=.05-1e-9
    assert (determinant(x,f)<0).any() and (determinant(z,f)>0).all()
    assert np.array_equal(constrain(q,q,f,ij)[0],q)
    assert np.allclose(constrain(q,q+[3.,-2.],f,ij)[0],q,atol=1e-12)
    t=q.copy(); t[24]+=[.2,.1]; assert np.allclose(constrain(q,t,f,ij)[0],t,atol=1e-12)
    return 4

def run(cache,out,mesh_out=None):
    started=time.monotonic()
    tests=self_test()
    acq=reference.prepare(cache)
    p,_=reference.reconstruct_touchup(cache); target=reference.load_published_xyz(cache)
    config=json.loads((HERE.parents[1]/"metric_flatten"/"protocol.json").read_text())["geometry"]
    spacing=float(config["spacing_voxels"]); voxel=float(config["source_voxel_um"]); pitch=spacing*voxel/1000
    jj,ii=np.meshgrid(np.arange(p.shape[1]),np.arange(p.shape[0]))
    u=config["carrier_origin_mm"][0]+jj*pitch; v=config["carrier_origin_mm"][1]+ii*pitch
    reg=np.isfinite(p).all(-1)&box(u,v,config["region_mm"],upper_inclusive=True)
    anchor=box(u,v,config["anchor_mm"],upper_inclusive=True)
    fix=int(np.argmin(np.abs(u[reg])+np.abs(v[reg])))
    sol=solve_mesh(p,spacing,reference.load_arap_functions(cache),region_mask=reg,anchor_mask=anchor,
                   iterations=60,fold_weight=.02,fold_threshold=1.25,fix_index=fix)
    initial,uv,xyz,f,ij=[sol[k] for k in ("uv_initial","uv","xyz","triangles","source_ij")]
    b=np.full(p.shape,np.nan); b[reg]=LinearNDInterpolator(uv,xyz)(initial)
    b=b.astype(np.float32).astype(float); vb=np.isfinite(b).all(-1); vt=np.isfinite(target).all(-1)
    if not np.array_equal(vb,vt) or not np.array_equal(b[vb],target[vt]):
        raise ValueError("Published B replay differs; stop")
    q,bound,solver=constrain(initial,uv,f,ij)
    # All queries restricted to original supported vertex positions.
    result=rasterize_triangles(q,xyz,f,p.shape[:2],query_mask=reg)
    vd=result["valid"]; raw=result["xyz"]; d=raw.astype(np.float32).astype(float)
    crop=np.zeros_like(reg); r0,r1=config["claim_crop_rows_half_open"]; c0,c1=config["claim_crop_columns_half_open"]; crop[r0:r1,c0:c1]=True
    retain=float((vb&vd).sum()/vb.sum()); claim=float((vb&vd&crop).sum()/(vb&crop).sum())
    common=four_corner(np.isfinite(p).all(-1))&four_corner(vb)&four_corner(vd)
    right=box(u[:-1,:-1]+pitch/2,v[:-1,:-1]+pitch/2,[15.5,23.5,-2.7,1.6])
    g={k:grid_summary(grid_metrics(z,spacing),common&right) for k,z in [("A",p),("B",b),("D",d)]}
    m={k:triangle_metrics(z,xyz/spacing,f) for k,z in [("A",initial),("B",uv),("D",q)]}
    tm={}
    for k,x in m.items():
        s=x["singular_values"]
        bad=~np.isfinite(s).all(1)|np.any(np.abs(s-1)>.05,axis=1)
        tm[k]={"counts":x["counts"],"bad_fraction":float(bad.mean()),
               "area_weighted_bad_fraction":float(x["source_3d_area"][bad].sum()/x["source_3d_area"].sum())}
    ids=result["source_triangle_id"][vd]; weights=result["barycentric_weights"][vd]
    reconstruction=np.einsum("ij,ijk->ik",weights,xyz[f[ids]])
    error=float(np.max(np.abs(reconstruction-raw[vd]))) if len(ids) else None
    gates={"exact_B_replay":True,"whole_B_retained":retain>=.99,"claim_B_retained":claim>=.995,
           "original_support_exact":bool(np.array_equal(vd,reg)),
           "boundary_exact":bool(np.array_equal(q[bound],initial[bound])),
           "all_face_area_floor":bool(solver["minimum_det_ratio"]>=.05-1e-9),
           "no_invalid_provenance":bool(len(ids) and (result["reason_bits"][vd]==0).all() and m["D"]["positive_uv"][ids].all() and (weights>=-1e-9).all() and np.allclose(weights.sum(1),1,rtol=0,atol=1e-9) and error<=1e-6),
           "right_group":bool(g["D"]["cells"]>=20 and g["D"]["median_abs_v_stretch_error"]<=.5*g["A"]["median_abs_v_stretch_error"] and g["D"]["either_axis_outside_5pct_fraction"]<=.05 and g["D"]["p95_abs_shear_degrees"]<=5),
           "global_bad_fraction_drop_5pp":tm["A"]["bad_fraction"]-tm["D"]["bad_fraction"]>=.05}
    report={"schema":"q10.fold_free_flatten.v1","public_control":"PHerc0306B, same source region as ChatGPT030",
         "synthetic_checks_passed":tests,"acquisition":{k:acq[k] for k in ("requests","body_bytes") if k in acq},
         "source_manifest_sha256":hashlib.sha256((HERE.parents[1]/"metric_flatten"/"source_manifest.json").read_bytes()).hexdigest(),
         "script_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
         "solver":solver,"whole_B_retained":retain,"claim_B_retained":claim,"original_nodes":int(reg.sum()),"B_nodes":int(vb.sum()),"D_nodes":int(vd.sum()),
         "right_group_common_cells":g,"same_source_triangles":tm,"max_barycentric_xyz_error_vox":error,
         "gates":gates,"pass":all(gates.values()),"CT_requests":0,"CNN_forwards":0,
         "wall_seconds":time.monotonic()-started,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    if mesh_out:
        from flatten import write_mesh
        write_mesh(mesh_out,raw,vd,spacing,voxel,provenance={"source_uv":q,"source_xyz":xyz,"source_triangles":f,"source_triangle_id":result["source_triangle_id"],"barycentric_weights":result["barycentric_weights"],"valid":vd,"reason_bits":result["reason_bits"]},summary={"method":"fixed_boundary_fold_free_v1","gates":gates})
    pathlib.Path(out).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true"); parser.add_argument("--cache",type=pathlib.Path)
    parser.add_argument("--output",type=pathlib.Path);parser.add_argument("--mesh-out",type=pathlib.Path)
    a=parser.parse_args()
    if a.self_test: print("Self-tests passed:",self_test())
    else:
        if not a.cache or not a.output or a.output.exists(): parser.error("new output and cache required")
        run(a.cache,a.output,a.mesh_out)
