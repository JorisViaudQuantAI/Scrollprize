#!/usr/bin/env python3
"""Exact reciprocal normal-ray labels on original public Claude/winner triangles."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,base64,hashlib,importlib.util,io,json,pathlib,resource,subprocess,sys,time,zlib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import requests,tifffile
HERE=pathlib.Path(__file__).resolve().parent
PIN="ab5b297de8cf696e098852ccb9b3be766317acb2"
PREFIX="firstletters-colab/pherc0211_fit_windings"
STATES=("fitted","snap","flat")
WINDS=("w033","w034","w035","w036")
LIM=24.
COS=np.cos(np.deg2rad(30.))
CODES={"invalid":0,"no_hit":1,"ambiguous":2,"oblique_or_invalid_normal":3,"near_le3":4,"intermediate":5,"far_ge6":6}
CAP=16*1024**2

def mod(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
winner=mod("winner052",HERE.parent/"winner_geometry"/"run.py")
def sha(b):return hashlib.sha256(b).hexdigest()
def arrsha(a):return sha(np.ascontiguousarray(a).tobytes())
def pack(a):
    a=np.ascontiguousarray(a,dtype=np.uint8)
    return {"shape":list(a.shape),"dtype":"uint8","sha256":arrsha(a),"encoding":"zlib+base64; C-order","data":base64.b64encode(zlib.compress(a.tobytes(),9)).decode()}
class Mesh:
    def __init__(self,P):
        P=np.asarray(P,np.float64)
        if P.ndim!=3 or P.shape[-1]!=3:raise ValueError("Bad grid")
        self.shape=P.shape[:2];self.P=P
        self.valid=np.isfinite(P).all(2)&(P[...,2]>0)&(P[...,:2]>=0).all(2)
        ids=np.arange(np.prod(self.shape)).reshape(self.shape)
        cell=self.valid[:-1,:-1]&self.valid[:-1,1:]&self.valid[1:,:-1]&self.valid[1:,1:]
        a=ids[:-1,:-1][cell];b=ids[:-1,1:][cell];c=ids[1:,:-1][cell];d=ids[1:,1:][cell]
        self.faces=np.concatenate((np.stack([a,b,d],1),np.stack([a,d,c],1)),axis=0)
        if not len(self.faces) or len(self.faces)>250000:raise ValueError("Face cap or empty mesh")
        self.xyz=P.reshape(-1,3)
        tri=self.xyz[self.faces];cr=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cr,axis=1)
        self.degenerate=int((area<=1e-12).sum());ok=area>1e-12
        self.faces=self.faces[ok];tri=tri[ok];cr=cr[ok];area=area[ok]
        self.used=np.unique(self.faces)
        self.normals=np.zeros_like(self.xyz);self.weight=np.zeros(len(self.xyz),np.float64)
        for j in range(3):
            np.add.at(self.normals,self.faces[:,j],cr)
            np.add.at(self.weight,self.faces[:,j],area/6.)
        n=np.linalg.norm(self.normals,axis=1);good=n>1e-12
        self.normals[good]/=n[good,None];self.normals[~good]=np.nan
        self.index=RayIndex(self.xyz,self.faces)
        self.receipt={"shape_yx":list(self.shape),"valid_nodes":int(self.valid.sum()),"used_nodes":len(self.used),"faces":len(self.faces),"degenerate_faces":self.degenerate,"area_vox2":float(self.weight.sum()),"normal_invalid_used":int((~good[self.used]).sum()),"coordinate_sha256":arrsha(P)}
    def cast(self,target):
        codes=np.zeros(len(self.xyz),np.uint8);depth=np.full(len(self.xyz),np.nan,np.float64)
        for i in self.used:
            q,d=target.index.hit(self.xyz[i],self.normals[i]);codes[i]=q;depth[i]=d
        return codes.reshape(self.shape),depth.reshape(self.shape)

class RayIndex:
    def __init__(self,xyz,f):
        self.f=np.asarray(f,np.int64);self.tri=np.asarray(xyz,np.float64)[self.f]
        self.a=self.tri[:,0];self.e1=self.tri[:,1]-self.a;self.e2=self.tri[:,2]-self.a
        cr=np.cross(self.e1,self.e2);self.area=np.linalg.norm(cr,axis=1)
        if (self.area<=1e-12).any():raise ValueError("Degenerate face passed into index")
        self.n=cr/self.area[:,None]
        self.c=self.tri.mean(1);self.rad=np.linalg.norm(self.tri-self.c[:,None,:],axis=-1).max(1)
        self.tree=cKDTree(self.c);self.bound=float(self.rad.max())
    def hit(self,p,n,exhaustive=False):
        if not np.isfinite(n).all() or not np.isfinite(p).all() or abs(np.linalg.norm(n)-1)>1e-8:return 3,np.nan
        ids=np.arange(len(self.f)) if exhaustive else np.asarray(self.tree.query_ball_point(p,LIM+self.bound+1e-7),np.int64)
        if len(ids)>25000:raise ValueError("Candidate cap")
        if not len(ids):return 1,np.nan
        # Conservative bounding sphere to finite segment distance, including endpoints.
        r=self.c[ids]-p;along=np.clip(r@n,-LIM,LIM)
        take=np.linalg.norm(r-along[:,None]*n,axis=1)<=self.rad[ids]+1e-7
        ids=ids[take]
        if not len(ids):return 1,np.nan
        angle=self.n[ids]@n
        # Conservatively reject possible coplanar contact rather than treating it as absent.
        parallel=np.abs(angle)<=1e-10
        cop=parallel&(np.abs(np.einsum("ij,ij->i",self.a[ids]-p,self.n[ids]))<=1e-8)
        if cop.any():return 3,np.nan
        ids=ids[~parallel];angle=angle[~parallel]
        if not len(ids):return 1,np.nan
        pv=np.cross(n,self.e2[ids]);det=np.einsum("ij,ij->i",self.e1[ids],pv)
        tvec=p-self.a[ids];u=np.einsum("ij,ij->i",tvec,pv)/det
        qv=np.cross(tvec,self.e1[ids]);v=(qv@n)/det
        depth=np.einsum("ij,ij->i",self.e2[ids],qv)/det
        weights=np.column_stack((1-u-v,u,v))
        keep=(weights>=-1e-9).all(1)&(depth>=-LIM-1e-7)&(depth<=LIM+1e-7)
        ids=ids[keep];angle=angle[keep];depth=depth[keep];weights=weights[keep]
        if not len(ids):return 1,np.nan
        carriers=[tuple(sorted(self.f[i][w>1e-8].tolist())) for i,w in zip(ids,weights)]
        if any(c!=carriers[0] for c in carriers) or np.ptp(depth)>1e-6:return 2,np.nan
        if np.min(np.abs(angle))<COS:return 3,np.nan
        d=float(depth.mean())
        return (4 if abs(d)<=3 else (6 if abs(d)>=6 else 5)),d

def self_test():
    p=np.array([[0,0,5],[10,0,5],[0,10,5],[10,10,5]],float);f=np.array([[0,1,3],[0,3,2]])
    a=RayIndex(p,f);n=np.array([0.,0.,1.]);q=np.array([5.,5.,0.])
    def check(idx,p,n,code,depth=None):
        c,d=idx.hit(np.asarray(p,float),np.asarray(n,float))
        return c==code and (depth is None or abs(d-depth)<1e-10)
    coincident=RayIndex(np.vstack((p,p)),np.vstack((f,f+4)))
    multi=RayIndex(np.vstack((p,p+[0,0,5])),np.vstack((f,f+4)))
    # plane with normal 60deg to query direction, intersecting at q+[0,0,5].
    p2=np.array([[0,0,5-np.sqrt(3)*5],[10,0,5+np.sqrt(3)*5],[0,10,5-np.sqrt(3)*5],[10,10,5+np.sqrt(3)*5]])
    oblique=RayIndex(p2,f)
    gates={"shared_diagonal":check(a,q,n,5,5),"normal_reversal":check(a,q,-n,5,-5),"shared_vertex":check(a,[0,0,0],n,5,5),
      "distinct_coincident_ambiguous":check(coincident,q,n,2),"two_layers_ambiguous":check(multi,q,n,2),
      "no_lateral_support":check(a,[20,20,0],n,1),"oblique_rejected":check(oblique,q,n,3)}
    rng=np.random.default_rng(6701);queries=rng.uniform([-2,-2,-15],[12,12,15],(200,3))
    parity=True
    for pt in queries:
        u,v=multi.hit(pt,n);w,z=multi.hit(pt,n,True)
        parity &= u==w and ((np.isnan(v) and np.isnan(z)) or abs(v-z)<1e-10)
    gates["broadphase_exhaustive"]=bool(parity)
    return {"gates":{k:bool(v) for k,v in gates.items()},"pass":all(gates.values())}

def git_blob(repo,rel):
    expected=subprocess.check_output(["git","-C",str(repo),"rev-parse",PIN+":"+rel],text=True).strip()
    p=repo/rel;b=p.read_bytes()
    got=hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest()
    if got!=expected:raise ValueError("Pinned Git blob mismatch "+rel)
    return b,{"git_blob":got,"sha256":sha(b),"bytes":len(b)}
def load_fit(repo,state,w):
    root=f"{PREFIX}/{state}/{w}";names=("x.tif","y.tif","z.tif","meta.json","nearest_dist_to_winner_vox.npy")
    bodies={};rec={}
    for n in names:bodies[n],rec[n]=git_blob(repo,root+"/"+n)
    P=np.stack([tifffile.imread(io.BytesIO(bodies[n])).astype(np.float64) for n in names[:3]],axis=-1)
    meta=json.loads(bodies["meta.json"]);scale=np.asarray(meta["scale"],float)
    if not np.allclose(scale,.05,atol=1e-8,rtol=0):raise ValueError("Unexpected scale")
    mesh=Mesh(P);legacy=np.load(io.BytesIO(bodies[names[-1]]),allow_pickle=False)
    if legacy.shape!=mesh.shape:raise ValueError("Legacy distance shape mismatch")
    return mesh,legacy,{"files":rec,"mesh":mesh.receipt,"source_bytes":sum(q["bytes"] for q in rec.values())}

def load_winner():
    r=json.loads((HERE.parent/"winner_geometry"/"layout_result.json").read_text())
    bodies={};receipt={}
    with requests.Session() as s:
        for name,q in r["source"]["objects"].items():
            b=winner.fetch(s,winner.BUCKET+"/"+q["key"],2*1024**2,expected=q)
            if sha(b)!=q["sha256"]:raise ValueError("Winner source hash mismatch")
            bodies[name]=b;receipt[name]={"sha256":sha(b),"bytes":len(b),"source":q["key"]}
    P=np.stack([tifffile.imread(io.BytesIO(bodies[n])).astype(np.float64) for n in ("x.tif","y.tif","z.tif")],axis=-1)
    return Mesh(P),receipt
def stats(mesh,codes,depth):
    weight=mesh.weight.reshape(mesh.shape);total=float(weight.sum());out={}
    for name,k in CODES.items():
        m=codes==k;out[name]={"vertices":int(m.sum()),"area_fraction":float(weight[m].sum()/total)}
    v=np.isfinite(depth)
    return {"codes":out,"valid_hit_vertices":int(v.sum()),"valid_hit_area_fraction":float(weight[v].sum()/total),
            "signed_depth_q05_q50_q95":np.quantile(depth[v],[.05,.5,.95]).tolist() if v.any() else None,
            "abs_depth_median":float(np.median(np.abs(depth[v]))) if v.any() else None}
def legacy_check(mesh,codes,legacy):
    used=mesh.weight.reshape(mesh.shape)>0;m=used&np.isfinite(legacy)
    near=m&(legacy<=3);far=m&(legacy>3)
    return {"used_legacy_finite":int(m.sum()),"legacy_le3":int(near.sum()),"legacy_le3_exact_near":int((near&(codes==4)).sum()),
      "legacy_le3_exact_intermediate_or_far":int((near&np.isin(codes,[5,6])).sum()),"legacy_le3_unknown":int((near&np.isin(codes,[1,2,3])).sum()),
      "legacy_gt3_exact_near":int((far&(codes==4)).sum())}
def common(w,cs,ds):
    cs=np.stack(cs);ds=np.stack(ds);near=cs==4
    weight=w.weight.reshape(w.shape);total=float(weight.sum());used=weight>0
    anynear=near.any(0);multiple=near.sum(0)>1
    finite=np.isin(cs,[4,5,6]);ab=np.where(finite,np.abs(ds),np.inf)
    ix=np.argsort(ab,axis=0,kind="stable");best=np.take_along_axis(ab,ix[:1],axis=0)[0];second=np.take_along_axis(ab,ix[1:2],axis=0)[0]
    # Unknown on another winding is not an observed competitor; keep this caveat explicit.
    unique=anynear&((second>=best+2)|~np.isfinite(second))
    chosen=np.where(unique,ix[0]+1,0).astype(np.uint8)
    nearest=np.where(unique,np.take_along_axis(ds,ix[:1],axis=0)[0],np.nan)
    f=lambda m:float(weight[m].sum()/total)
    per=[]
    for i,name in enumerate(WINDS):
        mm=chosen==i+1;lab,n=ndi.label(mm);counts=np.bincount(lab.ravel())[1:]
        per.append({"winding":name,"near_area_fraction":f(near[i]),"chosen_area_fraction":f(mm),"chosen_vertices":int(mm.sum()),
          "components":int(n),"largest_component_vertices":int(counts.max(initial=0))})
    jumps=[]
    for ax in (0,1):
        l=[slice(None)]*2;r=l.copy();l[ax]=slice(None,-1);r[ax]=slice(1,None);l=tuple(l);r=tuple(r)
        m=(chosen[l]>0)&(chosen[r]>0)&(chosen[l]!=chosen[r])
        jumps.extend(np.abs(nearest[l][m]-nearest[r][m]).tolist())
    single=max(q["near_area_fraction"] for q in per)
    return {"per_winding":per,"winner_used_vertices":int(used.sum()),"near_union_area_fraction":f(anynear),"largest_single_area_fraction":single,
      "absolute_union_gain":f(anynear)-single,"multiple_near_area_fraction":f(multiple),"unique_margin_area_fraction":f(unique),
      "near_but_margin_unknown_area_fraction":f(anynear&~unique),"no_near_area_fraction":f(used&~anynear),
      "all_four_have_valid_unique_hit_area_fraction":f(finite.all(0)),"seam_edges":len(jumps),
      "seam_signed_depth_difference_q50_q95_max":np.quantile(jumps,[.5,.95,1]).tolist() if jumps else None,
      "chosen_map":pack(chosen)},chosen,nearest

def run(a):
    t=time.monotonic();r={"schema":"adaptive.claude_carrier_atlas.v1","protocol_sha256":sha((HERE/"protocol.md").read_bytes()),"script_sha256":sha(pathlib.Path(__file__).read_bytes()),"pinned_claude_commit":PIN,
        "states":{},"pass":False,"codebook":CODES,"geometry_bytes":0}
    arrays={};reciprocal={}
    try:
        r["synthetic"]=self_test()
        if not r["synthetic"]["pass"]:raise ValueError("Synthetic ray test failed")
        head=subprocess.check_output(["git","-C",str(a.claude),"rev-parse","HEAD"],text=True).strip()
        if head!=PIN:raise ValueError("Wrong Claude checkout")
        w,src=load_winner();r["winner"]={"mesh":w.receipt,"files":src};r["geometry_bytes"]=sum(q["bytes"] for q in src.values())
        for state in STATES:
            row={"windings":{}};cs=[];ds=[]
            for name in WINDS:
                m,legacy,rec=load_fit(a.claude,state,name);r["geometry_bytes"]+=rec["source_bytes"]
                if r["geometry_bytes"]>CAP:raise ValueError("Geometry cap")
                fw,fd=w.cast(m);rv,rd=m.cast(w)
                rec.update(winner_to_fit=stats(w,fw,fd),fit_to_winner=stats(m,rv,rd),legacy_comparison=legacy_check(m,rv,legacy),
                           reciprocal_code_map=pack(rv))
                row["windings"][name]=rec;cs.append(fw);ds.append(fd);reciprocal[(state,name)]=rv
                for k,v in (("winner_code",fw),("winner_depth",fd),("fit_code",rv),("fit_depth",rd)):
                    arrays[f"{state}_{name}_{k}"]=v
            row["common"],chosen,dep=common(w,cs,ds)
            arrays[state+"_chosen"]=chosen;arrays[state+"_chosen_depth"]=dep
            r["states"][state]=row
            print("STATE_JSON="+json.dumps({"state":state,"common":{k:v for k,v in row["common"].items() if k!="chosen_map"}},separators=(",",":"),allow_nan=False),flush=True)
        r["fitted_snap_index_transitions"]={}
        for name in WINDS:
            p=reciprocal[("fitted",name)];q=reciprocal[("snap",name)]
            if p.shape!=q.shape:raise ValueError("Fitted/snap grid mismatch")
            use=(p>0)&(q>0);c=np.zeros((7,7),np.int64);np.add.at(c,(p[use],q[use]),1)
            r["fitted_snap_index_transitions"][name]={"common_used":int(use.sum()),"counts_0to6":c.tolist(),"near_to_far":int(c[4,6]),"far_to_near":int(c[6,4]),"near_to_unknown":int(c[4,1:4].sum())}
        flat=r["states"]["flat"]["common"]
        r["gates"]={"complete":len(r["states"])==3,"flat_union_gain":flat["absolute_union_gain"]>=.10,"flat_union_area":flat["near_union_area_fraction"]>=.50}
        r["pass"]=all(r["gates"].values())
        np.savez_compressed(a.maps,**arrays)
        r["maps_artifact_sha256"]=sha(a.maps.read_bytes())
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["wall_seconds"]=time.monotonic()-t;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("claude","output","maps"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
