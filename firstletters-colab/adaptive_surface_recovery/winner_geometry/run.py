#!/usr/bin/env python3
"""Audit a published winner's final grid; never confuse UV with 3D injectivity."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,hashlib,importlib.util,json,pathlib,sys,time,urllib.parse,xml.etree.ElementTree as ET
import numpy as np
import requests,tifffile
HERE=pathlib.Path(__file__).resolve().parent
BUCKET="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
PREFIX="PHerc0211/segments/20261028201525-crop1711-lines123-v9/"
OFFSETS=(-12,-8,-4,0,4,8,12)
CAP=16*1024**2
NAMES=("x.tif","y.tif","z.tif","meta.json")

def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def fetch(session,url,cap,params=None,expected=None):
    with session.get(url,params=params,headers={"Accept-Encoding":"identity"},timeout=(10,30),stream=True) as r:
        r.raise_for_status()
        if r.status_code!=200:raise ValueError("Expected complete object")
        if expected is not None:
            if expected.get("etag") and r.headers.get("ETag")!=expected["etag"]:raise ValueError("Object ETag changed")
        b=bytearray()
        for q in r.iter_content(65536):
            b.extend(q)
            if len(b)>cap:raise ValueError("Object exceeds byte cap")
        if expected is not None and len(b)!=expected["bytes"]:raise ValueError("Object size changed")
        return bytes(b)

def listing(session,prefix):
    b=fetch(session,BUCKET,1024**2,{"list-type":"2","prefix":prefix,"delimiter":"/"})
    root=ET.fromstring(b);ns={"s":"http://s3.amazonaws.com/doc/2006-03-01/"}
    if root.findtext("s:IsTruncated",namespaces=ns)=="true":raise ValueError("Listing exceeds declared page bound")
    files=[{"key":x.findtext("s:Key",namespaces=ns),"bytes":int(x.findtext("s:Size",namespaces=ns)),"etag":x.findtext("s:ETag",namespaces=ns)} for x in root.findall("s:Contents",ns)]
    dirs=[x.findtext("s:Prefix",namespaces=ns) for x in root.findall("s:CommonPrefixes",ns)]
    return files,dirs,{"prefix":prefix,"sha256":hashlib.sha256(b).hexdigest(),"body_bytes":len(b)}

def acquire(cache):
    cache.mkdir(parents=True,exist_ok=False)
    with requests.Session() as s:
        files,dirs,rec=listing(s,PREFIX);allfiles=list(files);alldirs=list(dirs);lists=[rec]
        mesh={x["key"].split("/")[-1]:x for x in files}
        if not all(n in mesh for n in NAMES):
            options=[d for d in dirs if "tifxyz" in d.lower() or d.lower().rstrip("/").endswith("/mesh")]
            if len(options)!=1:raise ValueError("No unique immediate mesh folder")
            ff,dd,r=listing(s,options[0]);allfiles+=ff;alldirs+=dd;lists.append(r);mesh={x["key"].split("/")[-1]:x for x in ff}
        if not all(n in mesh for n in NAMES):raise ValueError("Missing winner mesh objects")
        if sum(mesh[n]["bytes"] for n in NAMES)>CAP:raise ValueError("Mesh exceeds 16MiB")
        objs={}
        for n in NAMES:
            r=mesh[n];b=fetch(s,BUCKET+"/"+urllib.parse.quote(r["key"],safe="/"),CAP,expected=r)
            (cache/n).write_bytes(b);objs[n]={**r,"sha256":hashlib.sha256(b).hexdigest()}
    return {"listing":lists,"objects":objs,"public_file_inventory":allfiles,"public_directory_inventory":alldirs,
            "mesh_bytes":sum(x["bytes"] for x in objs.values())}

def grid_faces(P):
    h,w,_=P.shape
    valid=np.isfinite(P).all(2)&(P>=0).all(2)&(P!=0).any(2)
    rows,cols=np.indices((h,w));uv=np.stack([cols,rows],axis=-1).reshape(-1,2).astype(float)
    ids=np.arange(h*w).reshape(h,w)
    cell=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
    a=ids[:-1,:-1][cell];b=ids[:-1,1:][cell];c=ids[1:,:-1][cell];d=ids[1:,1:][cell]
    faces=np.concatenate((np.stack([a,b,d],1),np.stack([a,d,c],1)),axis=0)
    if not len(faces) or len(faces)>250000:raise ValueError("Original face count outside cap")
    used=np.unique(faces);remap=np.full(h*w,-1,int);remap[used]=np.arange(len(used))
    return uv[used],P.reshape(-1,3)[used],remap[faces],{"shape_yx":[h,w],"valid_nodes":int(valid.sum()),"used_nodes":len(used),"complete_cells":int(cell.sum()),"faces":len(faces)}

def face_normals(xyz,f):
    tri=xyz[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area2=np.linalg.norm(cross,axis=1);ok=area2>1e-12
    u=np.zeros_like(cross);u[ok]=cross[ok]/area2[ok,None]
    return cross,area2,u,ok

def tube(xyz,f,vertex_normals=None):
    cross,area2,n,ok=face_normals(xyz,f)
    if vertex_normals is None:
        v=np.zeros_like(xyz)
        for j in range(3):np.add.at(v,f[:,j],cross)
        vn=np.linalg.norm(v,axis=1)
        valid=vn>1e-12;v[valid]/=vn[valid,None]
    else:
        v=np.asarray(vertex_normals,float);valid=np.isfinite(v).all(1)&(np.abs(np.linalg.norm(v,axis=1)-1)<1e-8)
    supported=ok&valid[f].all(1)
    if not supported.any():return {"faces":len(f),"supported_faces":0,"depths":[],"union_risky_faces":0,"invalid_normal_vertices":int((~valid).sum())}
    total_area=float(area2[supported].sum())
    rows=[];union=np.zeros(len(f),bool)
    for off in OFFSETS:
        a=(xyz+off*v)[f]
        cr=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])
        proj=np.einsum("ij,ij->i",cr,n)
        ratio=np.linalg.norm(cr,axis=1)[supported]/area2[supported]
        risky=supported&(proj<=1e-12)
        union|=risky
        rows.append({"offset_vox":off,"risky_faces":int(risky.sum()),"risky_source_area_fraction":float(area2[risky].sum()/total_area),"area_ratio_q05_q50_q95":np.quantile(ratio,[.05,.5,.95]).tolist(),"near_zero_projected":int((supported&(np.abs(proj)<=1e-12)).sum())})
    return {"faces":len(f),"supported_faces":int(supported.sum()),"physical_degenerate_faces":int((~ok).sum()),"invalid_normal_vertices":int((~valid).sum()),"depths":rows,"union_risky_faces":int(union.sum()),"union_risky_source_area_fraction":float(area2[union].sum()/total_area)}

def adjacent(xyz,f):
    _,_,n,ok=face_normals(xyz,f)
    edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    fi=np.tile(np.arange(len(f)),3)
    order=np.lexsort((edges[:,1],edges[:,0]));e=edges[order];fi=fi[order]
    bounds=np.r_[0,np.flatnonzero(np.any(e[1:]!=e[:-1],axis=1))+1,len(e)];cnt=np.diff(bounds)
    i=bounds[:-1][cnt==2];a,b=fi[i],fi[i+1];take=ok[a]&ok[b]
    a,b=a[take],b[take]
    ang=np.degrees(np.arccos(np.clip((n[a]*n[b]).sum(1),-1,1)))
    return {"shared_edges":int((cnt==2).sum()),"boundary_edges":int((cnt==1).sum()),"nonmanifold_edges":int((cnt>2).sum()),"valid_shared_angles":len(ang),"dihedral_deg_q50_q95_q99":np.quantile(ang,[.5,.95,.99]).tolist() if len(ang) else None,"angles_gt90":int((ang>90).sum()),"angles_gt30":int((ang>30).sum())}

def synthetic():
    v,u=np.indices((4,9));P=np.stack([u+30,v+30,np.full_like(u,30)],-1).astype(float)
    uv,x,f,_=grid_faces(P)
    planar=tube(x,f)
    # Analytic cylinder radius 8 with exact outward vertex normals.
    th=uv[:,0]*.1;rad=np.stack([np.sin(th),np.zeros_like(th),np.cos(th)],1)
    c=8*rad+np.stack([np.full(len(th),30),uv[:,1]+30,np.full(len(th),30)],1)
    cyl=tube(c,f,rad);d={r["offset_vox"]:r for r in cyl["depths"]}
    zero=x.copy();zero[f[0,1]]=zero[f[0,0]]
    deg=tube(zero,f)
    g={"planar":planar["union_risky_faces"]==0 and all(np.allclose(r["area_ratio_q05_q50_q95"],1) for r in planar["depths"]),
       "cylinder":d[-8]["risky_faces"]==len(f) and d[-12]["risky_faces"]==len(f) and d[0]["risky_faces"]==0,
       "physical_degenerate":deg["physical_degenerate_faces"]>=1}
    return {"gates":g,"planar":planar,"cylinder":cyl,"degenerate":deg,"pass":all(g.values())}

def run(a):
    t=time.monotonic();r={"schema":"adaptive.winner_geometry.v1","script_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),"synthetic":synthetic(),"pass":False}
    if r["synthetic"]["pass"]:
        try:
            r["source"]=acquire(a.cache)
            meta=json.loads((a.cache/"meta.json").read_text());scale=np.asarray(meta["scale"],float).reshape(-1)
            if len(scale) not in (1,2) or not np.isfinite(scale).all() or (scale<=0).any() or not np.allclose(scale,scale[0],rtol=1e-6):raise ValueError("Missing isotropic declared scale")
            spacing=1/float(scale[0])
            xyz=[tifffile.imread(a.cache/(n+".tif")).astype(np.float64) for n in "xyz"]
            if any(q.ndim!=2 or q.shape!=xyz[0].shape for q in xyz):raise ValueError("Invalid XYZ grid")
            P=np.stack(xyz,axis=-1);uv,x,f,grid=grid_faces(P)
            r["grid"]=grid;r["native_voxels_per_grid_cell"]=spacing;r["native_pitch_um"]=9.362
            m=mod("original_atlas",HERE.parent/"multichart_atlas"/"run.py")
            tm=m.triangle_metrics(uv,x/spacing,f)
            positive=tm["positive_uv"];bad,counts=m.intersections(uv,f[positive])
            area=tm["source_3d_area"];sv=tm["singular_values"]
            if not np.isfinite(sv).all() or area.sum()<=0:raise ValueError("Nonfinite winner triangle metric")
            rel=(np.abs(sv-1)>.05).any(1);rel20=(np.abs(sv-1)>.2).any(1)
            r["metric"]={"orientation_counts":tm["counts"],"uv_intersection_counts":counts,
                "physical_area_cm2":float(area.sum()*(spacing*9.362e-4)**2),
                "singular_q05_q50_q95":np.quantile(sv,[.05,.5,.95],axis=0).tolist(),
                "area_weighted_fraction_outside_5pct":float(area[rel].sum()/area.sum()),
                "area_weighted_fraction_outside_20pct":float(area[rel20].sum()/area.sum())}
            r["adjacent_physical_dihedral"]=adjacent(x,f);r["depth_tube"]=tube(x,f)
            r["gates"]={"regular_uv":bool(positive.all() and not bad.any()),"finite_metric":True,"complete_tube":r["depth_tube"]["supported_faces"]>0}
            r["pass"]=all(r["gates"].values())
        except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    else:r["stop"]="synthetic gate failed"
    r["wall_seconds"]=time.monotonic()-t
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--cache",type=pathlib.Path,required=True);p.add_argument("--output",type=pathlib.Path,required=True)
    run(p.parse_args())
