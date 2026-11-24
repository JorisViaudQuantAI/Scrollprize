#!/usr/bin/env python3
"""Frozen A-hn scale transfer using optical-labelled validation fragments."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="4"
import argparse,hashlib,importlib.util,io,json,math,pathlib,re,resource,sys,time
import numpy as np
import requests
from PIL import Image
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parents[1]
REV="d67b1c1548ef93df9e00e804870f89127817f4c5"
ROOT="https://huggingface.co/datasets/YoussefMoNader/ink-8um-v8-patchpack/resolve/"+REV+"/"
HASHES={
 "segments.json":"4675eb308650d682b33de0319c1af8c4790a0eacd9dfdf973818ec23db479943",
 "val_seg_idx.npy":"2fa069d4fa5b7a22023f7a8234ee3f56ffc7e09629b3320967e3548322314537",
 "val_xyxys.npy":"6ca76194f9ab587ed4e1f437b9c38247c66fac60d5c4b5a0c6266e860c506357",
 "val_gt/Frag3_inklabels.png":"601da07f6aa07271c97456fbae5e91a173140ecfb3554e2527e3a802b3046e2e",
 "val_gt/Frag3_mask.png":"d2a5604cce3efaf7f1d34344279ff4359814197105be0ef742604fd22bdf9b46",
 "val_gt/Frag4_inklabels.png":"430b43859f372afb76200130ab6aac30fa7093d39df93beffc5f720ff574d5d7",
 "val_gt/Frag4_mask.png":"f57f7a5045031dccd2a11b34d1a932dada9c2ded561927038f48fc09a6549ca2"}
IMAGE_SHA="fb8f9dfe314ec488911d91eacff9295dff273cab5019c3f37bc2e45e4a6c7d4b"
ROW_BYTES=64*64*24
SHAPE=(43249,64,64,24)

def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sha(b):return hashlib.sha256(b).hexdigest()
def arrsha(a):return sha(np.ascontiguousarray(a).tobytes())
class Pack:
    def __init__(self):
        self.s=requests.Session();self.small_bytes=0;self.range_bytes=0;self.receipts=[];self.total=None;self.etag=None;self.header=None
    def get(self,name,interval=None):
        h={"Accept-Encoding":"identity"}
        if interval is not None:h["Range"]="bytes=%d-%d"%interval
        with self.s.get(ROOT+name,headers=h,timeout=(15,60),stream=True) as r:
            expected=206 if interval else 200
            if r.status_code!=expected:raise ValueError(f"{name}: HTTP {r.status_code}, expected {expected}")
            if interval:
                a,b=interval;cr=r.headers.get("Content-Range","")
                m=re.fullmatch(fr"bytes {a}-{b}/([0-9]+)",cr)
                if not m:raise ValueError("Exact Content-Range not returned")
                total=int(m.group(1))
                if self.total is not None and total!=self.total:raise ValueError("Object size changed")
                self.total=total
                et=r.headers.get("ETag")
                if self.etag and et and self.etag!=et:raise ValueError("Object ETag changed")
                if et:self.etag=et
                cap=b-a+1
                if self.range_bytes+cap>8*1024**2:raise ValueError("Range cap")
            else:cap=16*1024**2-self.small_bytes
            data=bytearray()
            for block in r.iter_content(65536):
                data.extend(block)
                if interval:self.range_bytes+=len(block)
                else:self.small_bytes+=len(block)
                if len(data)>cap:raise ValueError("Body cap")
            data=bytes(data)
            if interval and len(data)!=cap:raise ValueError("Short range")
            if not interval and sha(data)!=HASHES[name]:raise ValueError("Pinned source hash mismatch "+name)
            self.receipts.append({"name":name,"range":list(interval) if interval else None,"bytes":len(data),"sha256":sha(data),"etag":r.headers.get("ETag")})
            return data
    def open_image_array(self):
        b=self.get("val_images.npy",(0,4095));bio=io.BytesIO(b)
        version=np.lib.format.read_magic(bio)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(bio)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(bio)
        else:raise ValueError("Unsupported NPY version")
        if tuple(shape)!=SHAPE or fortran or dtype!=np.dtype("uint8"):raise ValueError("Unexpected patchpack array layout")
        self.header=bio.tell()
        if self.total!=self.header+SHAPE[0]*ROW_BYTES:raise ValueError("NPY total length mismatch")
    def row(self,i):
        if self.header is None or not 0<=i<SHAPE[0]:raise ValueError("Invalid patch row")
        a=self.header+i*ROW_BYTES;b=a+ROW_BYTES-1
        return np.frombuffer(self.get("val_images.npy",(a,b)),np.uint8).reshape(64,64,24)

def binary(b):
    a=np.asarray(Image.open(io.BytesIO(b)).convert("L"))
    if not np.isin(np.unique(a),[0,1,255]).all():raise ValueError("Expected binary optical labels/mask")
    return a>0

def select(segments,idx,boxes,truth,mask,name):
    seg=[(i,v) for i,v in enumerate(segments["val"]) if v["id"]==name]
    if len(seg)!=1:raise ValueError("Missing unique specimen")
    k,rec=seg[0]
    if tuple(rec["canvas_shape"])!=truth.shape or mask.shape!=truth.shape:raise ValueError("Optical canvas mismatch")
    ids=np.flatnonzero(idx==k);lookup={}
    for i in ids:
        x,y,x2,y2=map(int,boxes[i])
        if x2-x!=64 or y2-y!=64:raise ValueError("Invalid published tile size")
        if (y,x) in lookup:raise ValueError("Duplicate patch")
        lookup[y,x]=int(i)
    cands=[]
    for (y,x),i in sorted(lookup.items()):
        if (y,x-32) not in lookup or (y,x+32) not in lookup:continue
        if y<0 or x<32 or y+64>truth.shape[0] or x+96>truth.shape[1]:continue
        if not mask[y:y+64,x-32:x+96].all():continue
        core=truth[y+16:y+48,x+16:x+48];n=int(core.sum())
        kind="mixed" if 128<=n<=896 else ("blank" if not truth[y:y+64,x-32:x+96].any() else None)
        if kind:cands.append({"specimen":name,"kind":kind,"yx":[y,x],"rows":[lookup[y,x-32],i,lookup[y,x+32]],"ink_pixels":n,"background_pixels":1024-n})
    out=[]
    for kind,count in (("mixed",4),("blank",2)):
        for _ in range(count):
            rows=[]
            for q in cands:
                if q["kind"]!=kind:continue
                dist=min((sum((a-b)**2 for a,b in zip(q["yx"],r["yx"])) for r in out),default=0)
                if out and dist<128**2:continue
                key=(-dist,*q["yx"]) if out else (abs(q["ink_pixels"]-512),*q["yx"])
                rows.append((key,q))
            if not rows:raise ValueError(name+" insufficient frozen "+kind+" contexts")
            out.append(min(rows,key=lambda x:x[0])[1])
    return out,rec

def mosaic(patches):
    l,b,r=patches
    if not np.array_equal(l[:,32:],b[:,:32]) or not np.array_equal(b[:,32:],r[:,:32]):raise ValueError("Native overlap bytes differ")
    full=np.concatenate([l,b[:,32:],r[:,32:]],axis=1)
    if full.shape!=(64,128,24) or not np.array_equal(full[:,32:96],b):raise ValueError("Identity crop mismatch")
    return full

def inputs(full,stretch,d):
    p=full.transpose(2,0,1)
    co=63.5+(np.arange(64)-31.5)/.6
    c=d.clip_tile(stretch.interp_x(p,co));b=d.clip_tile(p[...,32:96])
    if not np.array_equal(b,full[:,32:96]) or full.max()>200:raise ValueError("Patchpack clipping/identity mismatch")
    return b,c

def roc_point(s,y):
    bg=np.sort(s[~y])[::-1];n=int(math.floor(.05*len(bg)))
    if not len(bg) or not y.any():raise ValueError("Need both classes")
    t=float(bg[n]);sel=s>t
    return {"threshold_descriptive":t,"recall":float(sel[y].mean()),"background_flag":float(sel[~y].mean())}
def pooled(scores,truth,d):
    s=np.concatenate([x.ravel() for x in scores]);y=np.concatenate([x.ravel() for x in truth])
    m=d.metrics(s,np.where(y,2,1).astype(np.uint8));m["matched_5pct"]=roc_point(s,y)
    return m

def self_test(stretch,d):
    x=np.broadcast_to(np.arange(128,dtype=np.uint8)[None,:,None],(64,128,24)).copy()
    q=mosaic([x[:,:64],x[:,32:96],x[:,64:128]])
    assert np.array_equal(q,x)
    b,c=inputs(q,stretch,d)
    assert np.array_equal(b,x[:,32:96]) and c.shape==(64,64,24)
    assert np.allclose(stretch.physical_core(np.broadcast_to(np.arange(64),(64,64)),.6),np.broadcast_to(31.5+.6*(np.arange(16,48)-31.5),(32,32)))
    q=np.arange(200,dtype=float);y=q>=100
    assert roc_point(q,y)["background_flag"]==.05
    return 4

def run(a):
    t=time.monotonic()
    report={"schema":"adaptive.optical_scale_transfer.v1","revision":REV,"declared_full_images_sha256":IMAGE_SHA,"full_images_sha256_verified":False,"forwards":0,"rows":[],"pass":False}
    pack=Pack();d=mod("det040",HERE.parent/"detector_bridge"/"run.py");stretch=mod("stretch046",HERE.parent/"stretch_tolerance"/"run.py")
    sys.path.insert(0,str(BASE/"tile_phase"))
    common=mod("common",BASE/"tile_phase"/"common.py")
    prep=mod("tile_prepare",BASE/"tile_phase"/"prepare.py");api=mod("tile_detector",BASE/"tile_phase"/"run.py")
    old=json.loads(a.prior.read_text())
    report["unit_checks"]=self_test(stretch,d)
    report["script_sha256"]=sha(pathlib.Path(__file__).read_bytes());report["protocol_sha256"]=sha((HERE/"protocol.md").read_bytes())
    try:
        a.cache.mkdir(parents=True,exist_ok=False)
        segments=json.loads(pack.get("segments.json"))
        idx=np.load(io.BytesIO(pack.get("val_seg_idx.npy")),allow_pickle=False).reshape(-1)
        boxes=np.load(io.BytesIO(pack.get("val_xyxys.npy")),allow_pickle=False)
        if idx.shape!=(43249,) or boxes.shape!=(43249,4) or idx.dtype.kind not in "iu" or boxes.dtype.kind not in "iu":raise ValueError("Unexpected metadata arrays")
        selected=[];labels={};sources={}
        for name in ("Frag3","Frag4"):
            truth=binary(pack.get("val_gt/"+name+"_inklabels.png"));mask=binary(pack.get("val_gt/"+name+"_mask.png"))
            sel,rec=select(segments,idx,boxes,truth,mask,name);selected+=sel;labels[name]=truth;sources[name]=rec
        report["selection"]=selected;report["segments"]=sources
        (a.cache/"selection.json").write_text(json.dumps(selected,indent=2)+"\n")
        print("SELECTION_JSON="+json.dumps(selected,separators=(",",":")),flush=True)
        pack.open_image_array()
        allinputs=[]
        for row in selected:
            y,x=row["yx"];truth=labels[row["specimen"]][y+16:y+48,x+16:x+48]
            if int(truth.sum())!=row["ink_pixels"]:raise ValueError("Selection changed")
            patches=[pack.row(i) for i in row["rows"]]
            b,c=inputs(mosaic(patches),stretch,d)
            allinputs.append((row,truth,b,c))
        if len(allinputs)!=12:raise ValueError("Expected twelve fixed tiles")
        modeldir=a.cache/"model";modeldir.mkdir()
        for name,(url,digest) in common.FILES.items():prep.fetch(url,modeldir/name,digest)
        report["verified_model_inputs"]=api.verified_inputs(modeldir)
        if report["verified_model_inputs"]!=old["verified_model_inputs"]:raise ValueError("Model provenance changed")
        model,torch=api.load_model(modeldir,"cpu");report["runtime"]=api.runtime(torch,"cpu")
        arrays={n:{"B":[],"C":[],"truth":[],"kind":[]} for n in ("Frag3","Frag4")}
        for row,y,b,c in allinputs:
            arow={**row,"arms":{}}
            for arm,v in (("B",b),("C",c)):
                ts=time.monotonic();p=api.predict(model,torch,v,"cpu");report["forwards"]+=1
                s=p[16:48,16:48] if arm=="B" else stretch.physical_core(p,.6)
                arrays[row["specimen"]][arm].append(s)
                m=d.metrics(s,np.where(y,2,1).astype(np.uint8)) if row["kind"]=="mixed" else {"blank_flagged_at_05":float((s>=.5).mean())}
                arow["arms"][arm]={**m,"input_sha256":arrsha(v),"score_sha256":arrsha(s),"forward_seconds":time.monotonic()-ts}
            arrays[row["specimen"]]["truth"].append(y);arrays[row["specimen"]]["kind"].append(row["kind"])
            report["rows"].append(arow);print("TILE_JSON="+json.dumps(arow,separators=(",",":")),flush=True)
        report["metrics"]={};report["gates"]={}
        for name,v in arrays.items():
            m={a:pooled(v[a],v["truth"],d) for a in ("B","C")}
            for a in ("B","C"):
                m[a]["blank_flagged_at_05"]=float(np.mean([(s>=.5).mean() for s,k in zip(v[a],v["kind"]) if k=="blank"]))
            delta=[r["arms"]["C"]["trace_AP"]-r["arms"]["B"]["trace_AP"] for r in report["rows"] if r["specimen"]==name and r["kind"]=="mixed"]
            B,C=m["B"],m["C"];m["mixed_tile_AP_deltas"]=delta
            report["metrics"][name]=m
            report["gates"][name]={"baseline_AP":B["trace_AP"]>=.65,"AP_gain":C["trace_AP"]>=B["trace_AP"]+.02,
              "matched_recall_gain":C["matched_5pct"]["recall"]>=B["matched_5pct"]["recall"]+.03,
              "known_background_bounded":C["distant_untraced_flagged_at_05"]<=B["distant_untraced_flagged_at_05"]+.03,
              "blank_bounded":C["blank_flagged_at_05"]<=B["blank_flagged_at_05"]+.03,
              "mixed_tile_nonharm":len(delta)==4 and sum(x>=-.1 for x in delta)>=3}
        report["complete"]=report["forwards"]==24 and len(report["rows"])==12
        report["pass"]=report["complete"] and all(all(g.values()) for g in report["gates"].values())
    except Exception as e:report["failure"]=f"{type(e).__name__}: "+re.sub(r"https?://[^\s]+","[URL omitted]",str(e))[:450]
    report["source"]={"small_bytes":pack.small_bytes,"range_bytes":pack.range_bytes,"object_total_bytes":pack.total,"header_bytes":pack.header,"receipts":pack.receipts}
    report["wall_seconds"]=time.monotonic()-t;report["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");print("RESULT_JSON="+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("cache","output","prior"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
