#!/usr/bin/env python3
"""Paired A-hn forward comparison of shared atlas rendering on published text.
The trace is human markup of a model display, not independent pixel truth.
Reuse existing geometry, renderer, model and acquisition modules.
"""
from __future__ import annotations
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="4"
import argparse, hashlib, importlib.util, json, pathlib, resource, sys, time, urllib.request
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parents[1]
TRACE_PATH="submission/supporting/reading/papyrologist_label_2026-10-16_conservative.png"
TRACE_URL="https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/"+TRACE_PATH
TRACE_BLOB="6a60f572ed7e53d458e5715ee6f9b99df2c59208"
PACK_SHAPE=(2437,5278)
RAW_ORIGIN=(863,2187)
GRID=(115,694)
DEPTHS=np.arange(-12,12,dtype=float)
STRIDE=21
CLAIM=(int(np.ceil((21-6.84)/.00864)),int(np.ceil((36-15)/.00864)),int(np.floor((21+2.72)/.00864)),int(np.floor((36+23)/.00864)))
BLANK=(116,7060,2778,7639)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def sha(p):
    with pathlib.Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def trace_labels(path):
    body=path.read_bytes()
    got=hashlib.sha1(f"blob {len(body)}\0".encode()+body).hexdigest()
    if len(body)!=74483 or got!=TRACE_BLOB:raise ValueError("Trace source blob differs")
    with Image.open(path) as im:
        a=np.asarray(im)
    g=a[...,:3].max(-1) if a.ndim==3 else a
    # OpenCV INTER_NEAREST uses floor(source size / destination size * index).
    rr=np.floor(np.arange(PACK_SHAPE[0])*g.shape[0]/PACK_SHAPE[0]).astype(int)
    cc=np.floor(np.arange(PACK_SHAPE[1])*g.shape[1]/PACK_SHAPE[1]).astype(int)
    trace=(g[np.ix_(rr,cc)]>127)[::-1].copy()
    pos=ndi.distance_transform_edt(trace)>2
    neg=ndi.distance_transform_edt(~trace)>4
    labels=np.zeros(trace.shape,np.uint8);labels[neg]=1;labels[pos]=2
    return labels,{"blob_sha1":got,"sha256":sha(path),"original_shape":list(a.shape),
                   "native_pack_shape":list(trace.shape),"trace_pixels":int(trace.sum()),
                   "eroded_trace_pixels":int(pos.sum()),"distant_untraced_pixels":int(neg.sum())}

def axis(lo,hi,origin,tile=64):
    a=int(np.ceil((lo-origin)/STRIDE)); b=int(np.floor((hi-tile-origin)/STRIDE))
    return np.arange(a,b+1,dtype=int)*STRIDE+origin

def count_boxes(mask,ys,xs):
    # core [16,48) in raw native coordinates, vectorized integral image.
    p=np.pad(mask.astype(np.int32),((1,0),(1,0))).cumsum(0,dtype=np.int32).cumsum(1,dtype=np.int32)
    y=ys-RAW_ORIGIN[0]+16;x=xs-RAW_ORIGIN[1]+16
    return p[np.ix_(y+32,x+32)]-p[np.ix_(y,x+32)]-p[np.ix_(y+32,x)]+p[np.ix_(y,x)]

def nearest_origin(center):
    return tuple(int(g+STRIDE*round((c-32-g)/STRIDE)) for c,g in zip(center,GRID))

def uv(yx):return (yx[1]/20.,yx[0]/20.)

def footprint(chart,yx):
    return bool(chart.footprint_support(uv(yx),(.05,.05),(64,64))["valid"].all())

def select(labels,chart):
    cal=nearest_origin((2360,6400))
    if not footprint(chart,cal):raise ValueError("Fixed calibration tile lacks full geometry")
    y0,x0,y1,x1=CLAIM
    ys,xs=axis(y0,y1,GRID[0]),axis(x0,x1,GRID[1])
    if not len(ys) or not len(xs):raise ValueError("Claim grid empty")
    if ys.min()-RAW_ORIGIN[0]<0 or xs.min()-RAW_ORIGIN[1]<0 or ys.max()+64-RAW_ORIGIN[0]>labels.shape[0] or xs.max()+64-RAW_ORIGIN[1]>labels.shape[1]:raise ValueError("Claim outside annotation frame")
    p=count_boxes(labels==2,ys,xs);n=count_boxes(labels==1,ys,xs)
    selected=[];attempts=[]
    def separate(yx):
        return all(np.linalg.norm(np.asarray(yx)-np.asarray(a))>=128 for a in [cal]+[tuple(t["yx"]) for t in selected])
    for third in range(3):
        lo=x0+(x1-x0)*third/3;hi=x0+(x1-x0)*(third+1)/3
        eligible=(p>=64)&(n>=64)&(xs[None,:]+32>=lo)&(xs[None,:]+32<hi)
        r,c=np.nonzero(eligible);share=p[r,c]/(p[r,c]+n[r,c])
        order=np.lexsort((xs[c],ys[r],np.abs(share-.5)))
        for slot in range(2):
            tried=0;found=False
            for k in order:
                yy,xx=int(ys[r[k]]),int(xs[c[k]])
                if not separate((yy,xx)):continue
                tried+=1;ok=footprint(chart,(yy,xx))
                attempts.append({"kind":"trace","third":third,"slot":slot,"yx":[yy,xx],"full_geometry":ok})
                if ok:
                    selected.append({"kind":"trace","third":third,"yx":[yy,xx],"positive_core":int(p[r[k],c[k]]),"untraced_core":int(n[r[k],c[k]])})
                    found=True;break
                if tried>=20:break
            if not found:raise ValueError(f"Trace stratum {third} slot {slot} lacks full geometry")
    ys,xs=axis(BLANK[0],BLANK[2],GRID[0]),axis(BLANK[1],BLANK[3],GRID[1])
    for slot,center in enumerate([(1900,7270),(2400,7270)]):
        yy,xx=np.meshgrid(ys,xs,indexing="ij");yy=yy.ravel();xx=xx.ravel()
        d=(yy+32-center[0])**2+(xx+32-center[1])**2
        order=np.lexsort((xx,yy,d));tried=0;found=False
        for k in order:
            yx=(int(yy[k]),int(xx[k]))
            if not separate(yx):continue
            tried+=1;ok=footprint(chart,yx)
            attempts.append({"kind":"author_blank","slot":slot,"yx":list(yx),"full_geometry":ok})
            if ok:
                selected.append({"kind":"author_blank","yx":list(yx)});found=True;break
            if tried>=20:break
        if not found:raise ValueError(f"Blank slot {slot} lacks full geometry")
    return cal,selected,attempts

class ReleasedRender:
    """Small tile extraction using the existing verified-file fetch helper."""
    def __init__(self,root,fetch,HF_DATA):
        import numcodecs
        self.root=root;self.fetch=fetch;self.base=HF_DATA
        self.zpath="results/claim_surface_render_41layers.zarr/0/"
        meta=root/"render-zarray.json";sums=root/"dataset-SHA256SUMS"
        fetch(HF_DATA+self.zpath+".zarray",meta,"6e9dfd265acd7a74f56d3ec82a82fd7de1715ec473cd966c5d3d141c3cd94d51")
        fetch(HF_DATA+"SHA256SUMS",sums,"8c114c6359efd95c30508c7b9731ebfab3ee799bf2ec6dcf84cf92e841eb07cf")
        m=json.loads(meta.read_text())
        if m["shape"]!=[41,4900,7780] or m["chunks"]!=[41,128,128] or m["dtype"]!="|u1":raise ValueError("Published render shape changed")
        self.checks={}
        for line in sums.read_text().splitlines():
            f=line.split(maxsplit=1)
            if len(f)==2:self.checks[f[1].lstrip(" *").removeprefix("./")]=f[0]
        self.decoder=numcodecs.get_codec(m["compressor"]);self.cache={};self.receipts=[];self.bytes=0
    def chunk(self,cy,cx):
        k=(cy,cx)
        if k in self.cache:return self.cache[k]
        if len(self.cache)>=64:raise ValueError("Released render chunk cap")
        key=self.zpath+f"0/{cy}/{cx}";path=self.root/"chunks"/f"0-{cy}-{cx}"
        if key not in self.checks:raise ValueError("Missing checksum "+key)
        self.fetch(self.base+key,path,self.checks[key])
        self.bytes+=path.stat().st_size
        if self.bytes>64*1024**2:raise ValueError("Released render byte cap")
        raw=np.frombuffer(self.decoder.decode(path.read_bytes()),np.uint8).reshape(41,128,128)
        self.cache[k]=raw
        self.receipts.append({"chunk":key,"sha256":self.checks[key],"bytes":path.stat().st_size})
        return raw
    def tile(self,yx):
        y,x=yx
        out=np.zeros((24,64,64),np.uint8);seen=np.zeros((64,64),bool)
        for cy in range(y//128,(y+63)//128+1):
            for cx in range(x//128,(x+63)//128+1):
                a,b=max(y,128*cy),max(x,128*cx);c,d=min(y+64,128*(cy+1)),min(x+64,128*(cx+1))
                out[:,a-y:c-y,b-x:d-x]=self.chunk(cy,cx)[8:32,a-128*cy:c-128*cy,b-128*cx:d-128*cx]
                seen[a-y:c-y,b-x:d-x]=True
        if not seen.all():raise ValueError("Incomplete released render")
        return out

def stats(a,b):
    a,b=np.asarray(a,float).ravel(),np.asarray(b,float).ravel()
    d=a-b
    return {"mae":float(np.mean(np.abs(d))),"rmse":float(np.sqrt(np.mean(d*d))),
            "r":float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None}

def source_crop(chart,reader,yx):
    lo,hi=chart.dense_crop_bounds(uv(yx),(.05,.05),(64,64),DEPTHS)
    block,org,support=reader.read_box(lo[::-1],hi[::-1],return_support=True)
    if tuple(org)!=tuple(lo[::-1]) or tuple(block.shape)!=tuple((hi-lo)[::-1]) or not support.all():raise ValueError("Dense crop lacks full stored support")
    return block.astype(np.float32),np.asarray(org[::-1],float)

def render_native(renderer,chart,volume,org,yx,depths,n):
    s,m,d=renderer.render(volume,org,chart,uv(yx),(.05,.05),(64,64),depths,samples_per_axis=n)
    if not m.all() or s.shape!=(24,64,64):raise ValueError("Full 64x64 context not supported")
    return s

def clip_tile(s):
    return np.ascontiguousarray(np.minimum(np.clip(np.rint(s),0,255).astype(np.uint8),200).transpose(1,2,0))

def metrics(score,labels):
    q=score[labels>0].astype(float);truth=labels[labels>0]==2
    p=int(truth.sum());n=int((~truth).sum())
    if not p or not n:raise ValueError("Two proxy classes required")
    order=np.argsort(-q,kind="stable");z=q[order];t=truth[order]
    ends=np.r_[np.flatnonzero(np.diff(z)),len(z)-1];tp=np.cumsum(t)[ends];fp=1+ends-tp
    rec=tp/p;prec=tp/(tp+fp)
    auc=float(np.trapz(np.r_[0,rec],np.r_[0,fp/n]))
    return {"trace_AP":float(np.sum(np.diff(np.r_[0,rec])*prec)),"trace_AUC":auc,
            "trace_recall_at_05":float((q[truth]>=.5).mean()),
            "distant_untraced_flagged_at_05":float((q[~truth]>=.5).mean()),
            "trace_pixels":p,"distant_untraced_pixels":n}

def self_test():
    yx=nearest_origin((2360,6400))
    assert all((v-g)%STRIDE==0 for v,g in zip(yx,GRID))
    t=np.arange(24*64*64,dtype=np.float32).reshape(24,64,64)%250
    z=clip_tile(t);assert z.shape==(64,64,24) and z.dtype==np.uint8 and z.max()==200
    m=metrics(np.array([[.1,.9],[.2,.8]]),np.array([[1,2],[1,2]]))
    assert m["trace_AP"]==1 and m["trace_AUC"]==1 and m["trace_recall_at_05"]==1
    return 3

def run(args):
    start=time.monotonic()
    report={"schema":"adaptive.detector_bridge.v1","unit_checks":self_test(),"protocol_sha256":sha(HERE/"protocol.md"),"script_sha256":sha(__file__),
            "forwards":0,"rows":[],"pass":False}
    root=args.cache;root.mkdir(parents=True,exist_ok=False)
    bridge=module("adaptive_bridge",HERE.parent/"shared_renderer_bridge"/"run.py")
    surface=module("shared_original_surface",args.shared/"surface.py")
    renderer=module("shared_footprint_renderer",args.shared/"render.py")
    sys.path.insert(0,str(args.gemini))
    import grow_m7_patch as gm
    sys.path.insert(0,str(BASE/"tile_phase"))
    common=module("common",BASE/"tile_phase"/"common.py")
    prep=module("existing_tile_prepare",BASE/"tile_phase"/"prepare.py")
    det=module("existing_tile_detector",BASE/"tile_phase"/"run.py")
    budget=gm.ByteBudget(160*1024**2)
    release=None
    try:
        chart=bridge.ChartBridge(bridge.load_atlas(args.atlas,bridge.PIN),0,surface.OriginalSurface)
        trace=root/"trace.png"
        with urllib.request.urlopen(TRACE_URL,timeout=45) as s:trace.write_bytes(s.read(75000))
        labels,meta=trace_labels(trace);report["annotation"]=meta
        cal,tiles,attempts=select(labels,chart)
        report["selection"]={"calibration_yx":list(cal),"tiles":tiles,"geometry_attempts":attempts}
        (root/"selection.json").write_text(json.dumps(report["selection"],indent=2)+"\n")
        print("SELECTION_JSON="+json.dumps(report["selection"],separators=(",",":")),flush=True)
        release=ReleasedRender(root,prep.fetch,common.HF_DATA)
        bcal=release.tile(cal)
        with gm.BudgetedRemoteZarrV2(bridge.CT_URL,level="0",budget=budget,workers=1,cache_mb=128,max_retries=1,timeout=20) as reader:
            if reader.dtype!=np.uint8:raise ValueError("Expected uint8 native CT")
            vol,org=source_crop(chart,reader,cal)
            calplus=render_native(renderer,chart,vol,org,cal,DEPTHS,1)
            calminus=render_native(renderer,chart,vol,org,cal,-DEPTHS,1)
            pair=[stats(bcal,calplus),stats(bcal,calminus)]
            choice=int(pair[1]["mae"]<pair[0]["mae"]);sign=1 if choice==0 else -1
            best,other=pair[choice],pair[1-choice]
            advantage=(other["mae"]-best["mae"])/max(other["mae"],1e-12)
            calpass=bool(advantage>=.1 and best["mae"]<=25 and best["r"] is not None and best["r"]>=.5)
            report["calibration"]={"plus":pair[0],"minus":pair[1],"chosen_sign":sign,"relative_mae_advantage":advantage,"pass":calpass}
            print("CALIBRATION_JSON="+json.dumps(report["calibration"],separators=(",",":")),flush=True)
            if not calpass:raise ValueError("Frozen CT-frame calibration gate failed before model acquisition")
            # All selected source data are prepared before first model prediction.
            inputs=[]
            for tile in tiles:
                yx=tuple(tile["yx"])
                raw=release.tile(yx);vol,org=source_crop(chart,reader,yx)
                p=render_native(renderer,chart,vol,org,yx,sign*DEPTHS,1)
                q=render_native(renderer,chart,vol,org,yx,sign*DEPTHS,4)
                if int((raw[12]>0).sum())<1024:raise ValueError("Released baseline tile admission failed")
                data={"B":clip_tile(raw),"P":clip_tile(p),"Q":clip_tile(q)}
                inputs.append((tile,data,{"B_P":stats(raw,p),"B_Q":stats(raw,q),"P_Q":stats(p,q)}))
        # Exact published checkpoint and loader from existing tile_phase lane.
        modeldir=root/"model";modeldir.mkdir()
        for name,(url,expected) in common.FILES.items():
            prep.fetch(url,modeldir/name,expected)
        report["verified_model_inputs"]=det.verified_inputs(modeldir)
        report["model_weight_bytes"]=(modeldir/"seed0.safetensors").stat().st_size
        model,torch=det.load_model(modeldir,"cpu")
        report["runtime"]=det.runtime(torch,"cpu")
        for tile,data,ct in inputs:
            y,x=tile["yx"];row={**tile,"CT":ct,"arms":{}}
            if tile["kind"]=="trace":
                lab=labels[y-RAW_ORIGIN[0]+16:y-RAW_ORIGIN[0]+48,x-RAW_ORIGIN[1]+16:x-RAW_ORIGIN[1]+48]
            for arm in ("B","P","Q"):
                t=time.monotonic();score=det.predict(model,torch,data[arm],"cpu");report["forwards"]+=1
                core=score[16:48,16:48]
                result=metrics(core,lab) if tile["kind"]=="trace" else {"author_blank_flagged_at_05":float((core>=.5).mean()),"mean_score":float(core.mean())}
                row["arms"][arm]={**result,"forward_seconds":time.monotonic()-t,
                    "input_sha256":hashlib.sha256(data[arm].tobytes()).hexdigest(),"score_sha256":hashlib.sha256(score.tobytes()).hexdigest()}
            report["rows"].append(row)
            print("TILE_JSON="+json.dumps(row,separators=(",",":")),flush=True)
        pos=[r for r in report["rows"] if r["kind"]=="trace"];neg=[r for r in report["rows"] if r["kind"]=="author_blank"]
        macro={}
        for arm in ("B","P","Q"):
            macro[arm]={k:float(np.mean([r["arms"][arm][k] for r in pos])) for k in ("trace_AP","trace_AUC","trace_recall_at_05","distant_untraced_flagged_at_05")}
            macro[arm]["author_blank_flagged_at_05"]=float(np.mean([r["arms"][arm]["author_blank_flagged_at_05"] for r in neg]))
        B,P,Q=(macro[k] for k in ("B","P","Q"));report["macro"]=macro
        gates={"complete":len(pos)==6 and len(neg)==2 and report["forwards"]==24,
               "baseline_trace_signal":B["trace_AUC"]>=.65 and B["trace_recall_at_05"]>=.2,
               "quadrature_AP_improvement_vs_point":Q["trace_AP"]>=P["trace_AP"]+.03,
               "quadrature_no_harm_vs_published":Q["trace_AP"]>=B["trace_AP"]-.03 and Q["trace_recall_at_05"]>=B["trace_recall_at_05"]-.05,
               "blank_control_no_material_increase":Q["author_blank_flagged_at_05"]<=max(B["author_blank_flagged_at_05"],P["author_blank_flagged_at_05"])+.03}
        report["gates"]=gates;report["pass"]=all(gates.values())
    except Exception as e:
        report["failure"]=f"{type(e).__name__}: {str(e)[:500]}";report["pass"]=False
    report["native_CT"]={"charged_body_bytes":budget.used,"requests":budget.requests,"cap_bytes":budget.cap}
    if release:report["released_CT"]={"body_bytes":release.bytes,"chunks":release.receipts}
    report["wall_seconds"]=time.monotonic()-start;report["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--self-test",action="store_true")
    for k in ("atlas","shared","gemini","cache","output"):ap.add_argument("--"+k,type=pathlib.Path)
    a=ap.parse_args()
    if a.self_test:print("Tests passed:",self_test())
    else:
        if any(getattr(a,k) is None for k in ("atlas","shared","gemini","cache","output")) or a.output.exists():ap.error("Required pinned inputs and new output")
        run(a)
