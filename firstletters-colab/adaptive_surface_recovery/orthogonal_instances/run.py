#!/usr/bin/env python3
"""Measured comparison of external tensor watershed and Mutex Watershed.
Only the union of manual labels is supplied to test algorithms; GT IDs score them.
"""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse, hashlib, importlib.metadata, json, math, pathlib, resource, subprocess, sys, time
import numpy as np
import requests,nrrd,tifffile
from scipy import ndimage as ndi, sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import bioimage_cpp as bic

HERE=pathlib.Path(__file__).resolve().parent
ROOT="https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels/instance-labels-harmonized/"
CUBES=("01744_02000_04048","08400_04304_03024","10192_03024_03280")
OFFSETS=[(a,d) for d in (1,3,6) for a in range(3)]
CAP=160*1024**2
SEED=5301

def sha(b):return hashlib.sha256(b).hexdigest()
def arrsha(a):return sha(np.ascontiguousarray(a).tobytes())
def sha_file(p):
    with open(p,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def get(session,url,path,remaining):
    if remaining<=0:raise ValueError("Byte cap")
    t=time.monotonic();n=0;h=hashlib.sha256()
    with session.get(url,headers={"Accept-Encoding":"identity"},stream=True,timeout=(15,40)) as r:
        if r.status_code!=200:raise ValueError(f"Public NRRD HTTP {r.status_code}")
        if "Content-Length" in r.headers and int(r.headers["Content-Length"])>remaining:raise ValueError("Declared byte cap")
        with path.open("wb") as f:
            for b in r.iter_content(65536):
                n+=len(b)
                if n>remaining:raise ValueError("Received byte cap")
                f.write(b);h.update(b)
        return {"url":url,"bytes":n,"sha256":h.hexdigest(),"etag":r.headers.get("ETag"),"seconds":time.monotonic()-t}

def read_cube(session,root,cube,report):
    out=[];rec={}
    for typ in ("volume","mask"):
        name=cube+"_"+typ+".nrrd";p=root/name
        r=get(session,ROOT+cube+"/"+name,p,CAP-report["source_bytes"]);report["source_bytes"]+=r["bytes"]
        rec[typ]=r;report["sources"][cube]=rec
        a,h=nrrd.read(str(p),index_order="C")
        if a.shape!=(256,256,256):raise ValueError("Unexpected cube shape")
        if typ=="mask" and (a.dtype.kind not in "iu" or a.min()<0):raise ValueError("Bad instance labels")
        rec[typ].update(dtype=str(a.dtype),header={k:str(h[k]) for k in ("type","dimension","sizes","encoding","endian","space directions","space origin") if k in h})
        a=np.ascontiguousarray(a[80:176,80:176,80:176])
        rec[typ]["crop_sha256"]=arrsha(a)
        out.append(a)
        p.unlink()
    ct,gt=out
    if not np.isfinite(ct).all():raise ValueError("Invalid CT")
    lo,hi=np.percentile(ct,[1,99])
    if hi<=lo:raise ValueError("No contrast")
    ct=np.rint(np.clip((ct.astype(np.float32)-lo)*255/(hi-lo),0,255)).astype(np.uint8)
    ids,cts=np.unique(gt[gt>0],return_counts=True)
    rec["crop"]={"shape":list(gt.shape),"foreground":int((gt>0).sum()),"instances":len(ids),"instances_ge64":int((cts>=64).sum()),"ct_percentile_1_99":[float(lo),float(hi)],"normalized_CT_sha256":arrsha(ct)}
    if (gt>0).sum()<1000 or (cts>=64).sum()<2:raise ValueError("Fixed crop insufficient labeled instance population")
    return ct,gt

def partition(gt,pred):
    g=np.asarray(gt);p=np.asarray(pred)
    if g.shape!=p.shape or g.size==0 or (g<=0).any() or (p<=0).any():raise ValueError("Metrics require all labeled foreground assigned")
    _,gi=np.unique(g,return_inverse=True);_,pi=np.unique(p,return_inverse=True)
    G=int(gi.max())+1;P=int(pi.max())+1
    keys=gi.astype(np.int64)*P+pi;uk,c=np.unique(keys,return_counts=True)
    gc=np.bincount(gi);pc=np.bincount(pi);n=len(g)
    pair=lambda x:int(np.sum(x.astype(np.int64)*(x.astype(np.int64)-1)//2))
    both=pair(c);gp=pair(gc);pp=pair(pc)
    precision=both/pp if pp else (1. if gp==0 else 0.)
    recall=both/gp if gp else (1. if pp==0 else 0.)
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    w=c/n
    split=-float(np.sum(w*np.log2(c/gc[uk//P])));merge=-float(np.sum(w*np.log2(c/pc[uk%P])))
    return {"voxels":n,"true_instances":G,"predicted_instances_on_GT":P,"rand_precision":precision,"rand_recall":recall,"rand_F1":f1,
        "VI_split_bits":split,"VI_merge_bits":merge,"ARI":float(adjusted_rand_score(g,p)),"same_pair_counts":{"joint":both,"true":gp,"pred":pp}}

def edge_graph(n,uv,w,muv,mw):
    uv=np.asarray(uv,np.uint64).reshape(-1,2);w=np.asarray(w,np.float32);muv=np.asarray(muv,np.uint64).reshape(-1,2);mw=np.asarray(mw,np.float32)
    graph=bic.graph.UndirectedGraph.from_edges(n,uv)
    # Retrieve exact graph edge order; map by packed keys if not insertion order.
    actual=np.asarray(graph.uv_ids(),dtype=np.uint64)
    keys=uv[:,0]*np.uint64(n)+uv[:,1];target=actual[:,0]*np.uint64(n)+actual[:,1]
    if len(keys)!=len(target):raise ValueError("Graph deduplicated edges")
    if not np.array_equal(keys,target):
        idx=np.argsort(keys);q=np.searchsorted(keys[idx],target)
        if (q>=len(keys)).any() or not np.array_equal(keys[idx[q]],target):raise ValueError("Graph order mismatch")
        w=w[idx[q]]
    return bic.graph.mutex_watershed.mutex_watershed_clustering(graph,w,muv,mw)

def self_test():
    # Two strips, strongest edges along strips, weak connecting bridge.
    uv=np.array([[0,1],[1,2],[3,4],[4,5],[2,3]],np.uint64)
    w=np.array([.99,.98,.97,.96,.55],np.float32)
    muv=np.array([[1,4]],np.uint64);mw=np.array([.9],np.float32)
    gt=np.array([1,1,1,2,2,2])
    M=edge_graph(6,uv,w,muv,mw)+1
    A=edge_graph(6,uv,w,np.empty((0,2),np.uint64),np.empty(0,np.float32))+1
    q=partition(gt,M);a=partition(gt,A)
    ident=partition(gt,gt+27);split=partition(gt,np.arange(6)+1)
    tests={"mutex_blocks_bridge":q["ARI"]==1 and a["predicted_instances_on_GT"]==1,
           "identity_and_permutation":ident["ARI"]==1 and ident["rand_F1"]==1 and ident["VI_split_bits"]==0 and ident["VI_merge_bits"]==0,
           "merge_split_separated":a["rand_recall"]==1 and a["VI_split_bits"]==0 and a["VI_merge_bits"]>0 and split["VI_merge_bits"]==0 and split["VI_split_bits"]>0}
    if not all(tests.values()):raise ValueError("Synthetic contract failed")
    return {"gates":tests,"mutex":q,"attraction_only":a,"all_split":split}

def features_base(ct,up):
    a=ct.astype(np.float32)/255.
    n,p=up.structure_normals(a)
    g=np.sqrt(sum(ndi.gaussian_filter(a,1.2,order=tuple(int(i==j) for i in range(3)))**2 for j in range(3)))
    return a,n,p,g

def endpoints(mask,ax,d):
    sl=[slice(None)]*3;sr=sl.copy();sl[ax]=slice(None,-d);sr[ax]=slice(d,None);sl=tuple(sl);sr=tuple(sr)
    a=np.array(np.nonzero(mask[sl]&mask[sr])).T
    b=a.copy();b[:,ax]+=d
    return a,b

def feat(base,a,b,ax,d):
    ct,n,p,g=base
    ai=tuple(a.T);bi=tuple(b.T)
    na=n[ai];nb=n[bi];pa=p[ai];pb=p[bi];c0=ct[ai];c1=ct[bi]
    line=[]
    for k in range(d+1):
        q=a.copy();q[:,ax]+=k;line.append(ct[tuple(q.T)])
    line=np.stack(line,axis=1)
    return np.column_stack([np.abs(c0-c1),line.mean(1),line.min(1),
        (g[ai]+g[bi])*.5,np.abs((na*nb).sum(1))*np.sqrt(pa*pb),
        (np.abs(na[:,ax])*pa+np.abs(nb[:,ax])*pb)*.5,
        np.full(len(a),d,np.float32)]).astype(np.float32)

def train(ct,gt,up):
    rng=np.random.default_rng(SEED);base=features_base(ct,up);mask=gt>0;X=[];Y=[];rows=[]
    for ax,d in OFFSETS:
        a,b=endpoints(mask,ax,d);y=(gt[tuple(a.T)]==gt[tuple(b.T)]);i0=np.flatnonzero(~y);i1=np.flatnonzero(y);n=min(len(i0),len(i1),20000)
        if n:
            sel=np.r_[rng.choice(i0,n,replace=False),rng.choice(i1,n,replace=False)]
            X.append(feat(base,a[sel],b[sel],ax,d));Y.append(y[sel])
        rows.append({"axis":ax,"distance":d,"available_same":len(i1),"available_different":len(i0),"sampled_per_class":n})
    if not X:raise ValueError("No calibration examples")
    x=np.concatenate(X);y=np.concatenate(Y)
    if len(x)<100:raise ValueError("Too few affinity labels")
    scaler=StandardScaler().fit(x);model=LogisticRegression(C=1,random_state=SEED,max_iter=500).fit(scaler.transform(x),y)
    return scaler,model,{"training_rows":len(x),"pairs_by_offset":rows,"coef":model.coef_.ravel().tolist(),"intercept":model.intercept_.tolist(),"feature_mean":scaler.mean_.tolist(),"feature_scale":scaler.scale_.tolist()}

def external(ct,mask,external_scripts,cache):
    cache.mkdir()
    cp=cache/"ct.tif";mp=cache/"mask.tif";tifffile.imwrite(cp,ct);tifffile.imwrite(mp,mask.astype(np.uint8))
    t=time.monotonic()
    proc=subprocess.run([sys.executable,str(external_scripts/"split_patch.py"),str(cp),str(mp),str(cache)],capture_output=True,text=True,timeout=160)
    if proc.returncode:raise ValueError("Pinned external splitter failed: "+proc.stderr[-400:].replace(str(cache),"[cache]"))
    with np.load(cache/"mask_off.npz",allow_pickle=False) as z:off=z["labels"]
    with np.load(cache/"mask_on.npz",allow_pickle=False) as z:on=z["labels"]
    pars=json.loads((cache/"mask_params.json").read_text());pars["seconds"]=time.monotonic()-t
    return off,on,pars

def learned(ct,gt,mask,base,scaler,model):
    n=int(mask.sum());node=np.full(mask.shape,-1,np.int64);node[mask]=np.arange(n)
    Au=[];Aw=[];Mu=[];Mw=[];scores=[];targets=[];edge_stats=[]
    for ax,d in OFFSETS:
        a,b=endpoints(mask,ax,d)
        if not len(a):continue
        probs=[]
        for lo in range(0,len(a),50000):
            hi=min(len(a),lo+50000)
            probs.append(model.predict_proba(scaler.transform(feat(base,a[lo:hi],b[lo:hi],ax,d)))[:,1])
        prob=np.concatenate(probs);ai=tuple(a.T);bi=tuple(b.T)
        u=node[ai];v=node[bi];uv=np.column_stack([np.minimum(u,v),np.maximum(u,v)]).astype(np.uint64)
        g0=gt[ai];g1=gt[bi];known=(g0>0)&(g1>0)
        y=(g0==g1)[known]
        auc=float(roc_auc_score(y,prob[known])) if y.any() and (~y).any() else None
        edge_stats.append({"axis":ax,"distance":d,"edges":len(prob),"known":int(known.sum()),"AUC":auc})
        if d==1:
            take=prob>=.5;Au.append(uv[take]);Aw.append(prob[take])
        else:
            take=prob<=.5;Mu.append(uv[take]);Mw.append(1-prob[take])
    uv=np.concatenate(Au);aw=np.concatenate(Aw).astype(np.float32)
    mu=np.concatenate(Mu);mw=np.concatenate(Mw).astype(np.float32)
    graph=sparse.coo_matrix((np.ones(len(uv),np.uint8),(uv[:,0],uv[:,1])),shape=(n,n)).tocsr()
    _,D=connected_components(graph,directed=False)
    E=edge_graph(n,uv,aw,mu,mw)
    F=edge_graph(n,uv,aw,np.empty((0,2),np.uint64),np.empty(0,np.float32))
    parity=adjusted_rand_score(D,F)==1.
    out=[]
    for p in (D,E,F):
        q=np.zeros(mask.shape,np.int64);q[mask]=p.astype(np.int64)+1;out.append(q)
    return (*out,{"edges_attractive":len(uv),"edges_repulsive":len(mu),"no_mutex_partition_equal_to_threshold_CC":bool(parity),"affinity_by_offset":edge_stats})

def run(a):
    t=time.monotonic()
    report={"schema":"adaptive.orthogonal_instances.v1","script_sha256":sha_file(__file__),"protocol_sha256":sha_file(HERE/"protocol.md"),
      "sources":{},"source_bytes":0,"cases":[],"pass":False,"packages":{k:importlib.metadata.version(k) for k in ("numpy","scipy","scikit-learn","scikit-image","bioimage-cpp","pynrrd","tifffile")}}
    try:
        report["synthetic"]=self_test()
        report["external_code"]={p:sha_file(a.external/p) for p in ("split_patch.py","split_stacked.py","separate_sheets.py")}
        sys.path.insert(0,str(a.external));import split_stacked as up
        a.cache.mkdir(parents=True,exist_ok=False)
        with requests.Session() as session:
            ct,gt=read_cube(session,a.cache,CUBES[0],report)
            scaler,model,train_receipt=train(ct,gt,up);report["training"]=train_receipt
            del ct,gt
            for cube in CUBES[1:]:
                ct,gt=read_cube(session,a.cache,cube,report)
                base=features_base(ct,up);orig=gt>0
                for name,mask in (("original",orig),("bridged",orig|ndi.binary_closing(orig,structure=ndi.generate_binary_structure(3,1),iterations=1))):
                    ts=time.monotonic();row={"cube":cube,"mask":name,"foreground":int(orig.sum()),"added_bridge_voxels":int((mask&~orig).sum()),"metrics":{}}
                    A=ndi.label(mask)[0];B,C,ext=external(ct,mask,a.external,a.cache/(cube+"_"+name))
                    D,E,F,learn=learned(ct,gt,mask,base,scaler,model)
                    for key,pred in (("A_CC",A),("B_EDT",B),("C_tensor",C),("D_affinity_CC",D),("E_mutex",E),("F_no_mutex",F)):
                        if not np.array_equal(pred>0,mask):raise ValueError("Predicted foreground changed: "+key)
                        row["metrics"][key]={**partition(gt[orig],pred[orig]),"partition_sha256":arrsha(pred),"instances_full_mask":int(np.unique(pred[mask]).size)}
                    row["external"]=ext;row["affinity"]=learn;row["wall_seconds"]=time.monotonic()-ts
                    report["cases"].append(row);print("CASE_JSON="+json.dumps(row,separators=(",",":"),allow_nan=False),flush=True)
        checks={}
        for cube in CUBES[1:]:
            a0=next(x for x in report["cases"] if x["cube"]==cube and x["mask"]=="original")
            b0=next(x for x in report["cases"] if x["cube"]==cube and x["mask"]=="bridged")
            r=b0["metrics"];s=a0["metrics"]
            checks[cube]={"mutex_gain_vs_tensor_bridged":r["E_mutex"]["rand_F1"]>=r["C_tensor"]["rand_F1"]+.03,
              "mutex_merge_gain_vs_attraction":r["E_mutex"]["VI_merge_bits"]<=r["D_affinity_CC"]["VI_merge_bits"]-.05,
              "mutex_split_nonharm":r["E_mutex"]["VI_split_bits"]<=r["D_affinity_CC"]["VI_split_bits"]+.10,
              "mutex_original_nonharm":s["E_mutex"]["rand_F1"]>=s["C_tensor"]["rand_F1"]-.03,
              "ablation_parity":a0["affinity"]["no_mutex_partition_equal_to_threshold_CC"] and b0["affinity"]["no_mutex_partition_equal_to_threshold_CC"],
              "tensor_gain_vs_EDT":r["C_tensor"]["rand_F1"]>=r["B_EDT"]["rand_F1"]+.03,
              "tensor_original_nonharm":s["C_tensor"]["rand_F1"]>=s["B_EDT"]["rand_F1"]-.03}
        report["gates"]=checks;report["complete"]=len(report["cases"])==4
        report["tensor_pass"]=report["complete"] and all(g["tensor_gain_vs_EDT"] and g["tensor_original_nonharm"] for g in checks.values())
        report["pass"]=report["complete"] and all(all(v for k,v in g.items() if not k.startswith("tensor")) for g in checks.values())
    except Exception as e:
        report["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    report["wall_seconds"]=time.monotonic()-t;report["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("cache","output","external"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
