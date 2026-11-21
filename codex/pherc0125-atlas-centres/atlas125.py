"""Physical geometry and fixed 384-square charts for two published PHerc. 125 windings."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import tifffile

NAMES = ("z14144_w045", "z11136_w035")
PITCH_UM = 9.362
SIDE = 384
STEP = 20
DEPTH = np.arange(-32,33,dtype="f4")


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def xyz_from(folder):
    mesh=np.stack([tifffile.imread(folder/f"{axis}.tif") for axis in "xyz"],axis=-1).astype("f8")
    mesh[(~np.isfinite(mesh).all(-1)) | (mesh<0).any(-1)] = np.nan
    return mesh


def node_normals(p):
    def derivative(axis):
        prev, nxt = np.roll(p,1,axis),np.roll(p,-1,axis)
        if axis==0:
            prev[0]=np.nan
            nxt[-1]=np.nan
        else:
            prev[:,0]=np.nan
            nxt[:,-1]=np.nan
        left,right=p-prev,nxt-p
        return np.where(np.isfinite(left+right),(left+right)/2,np.where(np.isfinite(right),right,left))
    norm=np.cross(derivative(1),derivative(0))
    mag=np.linalg.norm(norm,axis=-1,keepdims=True)
    return norm/np.where(mag>1e-8,mag,np.nan)


def surface_area(p):
    a,b,c,d=p[:-1,:-1],p[:-1,1:],p[1:,:-1],p[1:,1:]
    t1=.5*np.linalg.norm(np.cross(b-a,d-a),axis=-1)
    t2=.5*np.linalg.norm(np.cross(d-a,c-a),axis=-1)
    complete=np.isfinite(t1+t2)
    cm2=(PITCH_UM/10000)**2
    # Count triangles with three existing vertices under a fixed oriented diagonal.
    return dict(triangles_3_vertices=int(np.isfinite(t1).sum()+np.isfinite(t2).sum()),
                full_quads=int(complete.sum()),total_quads=int(complete.size),
                measured_cm2=float((np.nansum(t1)+np.nansum(t2))*cm2),
                full_quad_cm2=float(np.sum((t1+t2)[complete])*cm2))


def choose_chart(p):
    valid=np.isfinite(p).all(-1)
    cc,n=ndimage.label(valid,structure=ndimage.generate_binary_structure(2,1))
    if not n:
        raise ValueError("No valid mesh components")
    counts=np.bincount(cc.ravel());counts[0]=0
    dominant=cc==counts.argmax()
    filled=ndimage.minimum_filter(dominant.astype("u1"),size=2*int(np.ceil(SIDE/(2*STEP)))+1,mode="constant")
    locations=np.argwhere(filled)
    if not len(locations):
        locations=np.argwhere(dominant)
    mid=(np.array(p.shape[:2])-1)/2
    centre=locations[np.argmin(np.sum((locations-mid)**2,axis=1))]
    origin=centre*STEP-SIDE//2
    return origin.astype(int), dict(grid_centre_rc=centre.tolist(),uv_origin_rc=origin.tolist(),
        n_components=int(n),n_valid=int(valid.sum()),n_grid=int(valid.size),largest_component=int(counts.max()))


def chart(p,origin):
    rr,cc=np.mgrid[:SIDE,:SIDE]
    y=(rr+origin[0])/STEP;x=(cc+origin[1])/STEP
    row,col=np.floor(y).astype(int),np.floor(x).astype(int)
    if row.min()<0 or col.min()<0 or row.max()+1>=p.shape[0] or col.max()+1>=p.shape[1]:
        raise ValueError("Fixed chart crosses the grid border")
    y=(y-row)[...,None];x=(x-col)[...,None]
    a,b,c,d=p[row,col],p[row,col+1],p[row+1,col],p[row+1,col+1]
    pos=(1-y)*((1-x)*a+x*b)+y*((1-x)*c+x*d)
    du=(1-y)*(b-a)+y*(d-c)
    dv=(1-x)*(c-a)+x*(d-b)
    norm=np.cross(du,dv); mag=np.linalg.norm(norm,axis=-1,keepdims=True)
    return pos,norm/np.where(mag>1e-7,mag,np.nan)


def fixed_survey(p):
    n=node_normals(p)
    good=np.isfinite(n).all(-1)
    good=ndimage.binary_erosion(good,iterations=1)
    rows,cols=np.nonzero(good)
    lo=np.array([rows.min(),cols.min()]);hi=np.array([rows.max(),cols.max()])
    rc=np.stack((rows,cols),axis=1);tree=cKDTree(rc)
    q=np.arange(.1,1,.2)
    targets=np.array([lo+(hi-lo)*[v,min(.97,u+(.035 if i%2 else 0.))]
                      for i,v in enumerate(q) for u in q])
    selected=rc[tree.query(targets)[1]]
    if len(np.unique(selected,axis=0))!=25:raise ValueError("Duplicate fixed survey points")
    return selected


def summarize(folder):
    p=xyz_from(folder)
    origin,summary=choose_chart(p)
    flat,norm=chart(p,origin);survey=fixed_survey(p)
    meta=json.loads((folder/"meta.json").read_text())
    if meta["scale"]!=[.05,.05]:raise ValueError("Unsupported mesh scale")
    val=p[np.isfinite(p).all(-1)]
    summary.update(grid_shape=list(p.shape[:2]),physical_area=surface_area(p),
        bbox_xyz=[val.min(0).tolist(),val.max(0).tolist()],publisher_bbox=meta["bbox"],
        publisher_area_vx2=meta["area_vx2"],chart_area=surface_area(flat),
        chart_finite_normals=int(np.isfinite(norm).all(-1).sum()),
        chart_center_xyz=flat[SIDE//2,SIDE//2].tolist(),chart_bbox=[np.nanmin(flat,axis=(0,1)).tolist(),np.nanmax(flat,axis=(0,1)).tolist()],
        survey_indices_rc=survey.tolist())
    return summary


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("atlas",type=Path)
    ap.add_argument("result",type=Path)
    args=ap.parse_args()
    dump(args.result,{name:summarize(args.atlas/name) for name in NAMES})
