#!/usr/bin/env python3
"""CPU Q10/Q49: paired horizontal-scale stress test, exact original physical cores."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="4"
import argparse,hashlib,importlib.util,json,pathlib,resource,sys,time,urllib.request
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parents[1]
SCALES=(.6,2.)

def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def sha(path):
    with pathlib.Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def read_patch(rel,y,x,w):
    if y<0 or x<0 or y+64>4900 or x+w>7780:raise ValueError("Published source field exceeded")
    out=np.zeros((24,64,w),np.uint8);seen=np.zeros((64,w),bool)
    for cy in range(y//128,(y+63)//128+1):
        for cx in range(x//128,(x+w-1)//128+1):
            a,b=max(y,cy*128),max(x,cx*128)
            c,d=min(y+64,(cy+1)*128),min(x+w,(cx+1)*128)
            out[:,a-y:c-y,b-x:d-x]=rel.chunk(cy,cx)[8:32,a-cy*128:c-cy*128,b-cx*128:d-cx*128]
            seen[a-y:c-y,b-x:d-x]=True
    if not seen.all():raise ValueError("Uncovered source patch")
    return out

def interp_x(a,x):
    """Linear interpolation with no padding, arbitrary leading dimensions."""
    lo=np.floor(x).astype(np.int64);hi=np.ceil(x).astype(np.int64)
    if lo.min()<0 or hi.max()>=a.shape[-1]:raise ValueError("Interpolation outside source")
    w=x-lo
    return a[...,lo]*(1-w)+a[...,hi]*w

def pull_input(rel,yx,s):
    y,x=yx
    coords=x+31.5+(np.arange(64,dtype=float)-31.5)/s
    a=int(np.floor(coords.min()));b=int(np.ceil(coords.max()))+1
    src=read_patch(rel,y,a,b-a)
    out=interp_x(src,coords-a)
    return out,(a,b)

def physical_core(score,s):
    coords=31.5+s*(np.arange(16,48,dtype=float)-31.5)
    return interp_x(score[16:48],coords).astype(np.float32)

def self_test():
    a=np.broadcast_to(np.arange(128,dtype=float),(24,64,128))
    q=np.arange(64,dtype=float)+17
    assert np.array_equal(interp_x(a,q),a[...,17:81])
    q=np.arange(64,dtype=float)*.6+17.25
    assert np.allclose(interp_x(a,q)[0,0],q)
    ramp=np.broadcast_to(np.arange(64,dtype=float),(64,64))
    for s in SCALES:
        expect=31.5+s*(np.arange(16,48)-31.5)
        assert np.allclose(physical_core(ramp,s),np.broadcast_to(expect,(32,32)))
    try:interp_x(a,np.array([-0.01]));raise AssertionError("padding accepted")
    except ValueError:pass
    return 4

def run(a):
    t=time.monotonic();report={"schema":"adaptive.stretch_tolerance.v1","protocol_sha256":sha(HERE/"protocol.md"),"script_sha256":sha(__file__),"unit_checks":self_test(),"rows":[],"forwards":0,"pass":False}
    old=json.loads(a.prior.read_text());report["prior_040_sha256"]=sha(a.prior)
    d=mod("detector040",HERE.parent/"detector_bridge"/"run.py")
    sys.path.insert(0,str(BASE/"tile_phase"))
    common=mod("common",BASE/"tile_phase"/"common.py")
    prep=mod("existing_tile_prepare",BASE/"tile_phase"/"prepare.py")
    modelapi=mod("existing_tile_detector",BASE/"tile_phase"/"run.py")
    rel=None
    try:
        if old["forwards"]!=24:raise ValueError("Expected completed 040 baseline")
        a.cache.mkdir(parents=True,exist_ok=False);trace=a.cache/"trace.png"
        with urllib.request.urlopen(d.TRACE_URL,timeout=45) as s:trace.write_bytes(s.read(75000))
        labels,meta=d.trace_labels(trace)
        if meta["sha256"]!=old["annotation"]["sha256"]:raise ValueError("Trace source changed")
        report["annotation"]=meta;report["selection"]=old["selection"]
        rel=d.ReleasedRender(a.cache,prep.fetch,common.HF_DATA)
        inputs=[]
        for tile,prior in zip(old["selection"]["tiles"],old["rows"],strict=True):
            if tile["yx"]!=prior["yx"] or tile["kind"]!=prior["kind"]:raise ValueError("Frozen tile mismatch")
            yx=tuple(tile["yx"]);y,x=yx
            ident,_=pull_input(rel,yx,1.)
            if arrsha(d.clip_tile(ident))!=prior["arms"]["B"]["input_sha256"]:raise ValueError("Identity did not reproduce B hash")
            lab=None
            if tile["kind"]=="trace":
                lab=labels[y-d.RAW_ORIGIN[0]+16:y-d.RAW_ORIGIN[0]+48,x-d.RAW_ORIGIN[1]+16:x-d.RAW_ORIGIN[1]+48]
                if int((lab==2).sum())!=tile["positive_core"] or int((lab==1).sum())!=tile["untraced_core"]:raise ValueError("Trace class counts changed")
            arms={}
            for s in SCALES:
                raw,bounds=pull_input(rel,yx,s)
                if raw.shape!=(24,64,64) or int((raw[12]>0).sum())<1024:raise ValueError("Stretched tile fails source/admission")
                arms[str(s)]=(d.clip_tile(raw),list(bounds))
            inputs.append((tile,prior,lab,arms))
        if len(inputs)!=8:raise ValueError("Expected 8 frozen tiles")
        modeldir=a.cache/"model";modeldir.mkdir()
        for name,(url,expected) in common.FILES.items():prep.fetch(url,modeldir/name,expected)
        report["verified_model_inputs"]=modelapi.verified_inputs(modeldir)
        if report["verified_model_inputs"]!=old["verified_model_inputs"]:raise ValueError("Model source differs")
        model,torch=modelapi.load_model(modeldir,"cpu");report["runtime"]=modelapi.runtime(torch,"cpu")
        for tile,prior,lab,arms in inputs:
            row={**tile,"identity_B_input_exact":True,"trace_counts_exact":True,"B":prior["arms"]["B"],"stretched":{}}
            for s in SCALES:
                v,bounds=arms[str(s)];tick=time.monotonic()
                score=modelapi.predict(model,torch,v,"cpu");report["forwards"]+=1
                core=physical_core(score,s)
                m=d.metrics(core,lab) if tile["kind"]=="trace" else {"author_blank_flagged_at_05":float((core>=.5).mean()),"mean_score":float(core.mean())}
                row["stretched"][str(s)]={**m,"input_sha256":arrsha(v),"score_sha256":arrsha(score),
                    "native_source_x_half_open":bounds,"forward_seconds":time.monotonic()-tick}
            report["rows"].append(row)
            print("TILE_JSON="+json.dumps(row,separators=(",",":")),flush=True)
        pos=[r for r in report["rows"] if r["kind"]=="trace"];neg=[r for r in report["rows"] if r["kind"]=="author_blank"]
        base=old["macro"]["B"];report["macro"]={"B":base};report["gates"]={}
        for s in SCALES:
            k=str(s);m={name:float(np.mean([r["stretched"][k][name] for r in pos])) for name in ("trace_AP","trace_AUC","trace_recall_at_05","distant_untraced_flagged_at_05")}
            m["author_blank_flagged_at_05"]=float(np.mean([r["stretched"][k]["author_blank_flagged_at_05"] for r in neg]))
            thirds=[]
            for third in range(3):
                rows=[r for r in pos if r["third"]==third]
                thirds.append(float(np.mean([r["stretched"][k]["trace_AP"]-r["B"]["trace_AP"] for r in rows])))
            m["third_AP_delta_vs_B"]=thirds;report["macro"][k]=m
            report["gates"][k]={"macro_AP_noninferior":m["trace_AP"]>=base["trace_AP"]-.05,
                "trace_recall_noninferior":m["trace_recall_at_05"]>=base["trace_recall_at_05"]-.10,
                "blank_bounded":m["author_blank_flagged_at_05"]<=base["author_blank_flagged_at_05"]+.03,
                "third_AP_noninferior":min(thirds)>=-.15}
        report["complete"]=len(pos)==6 and len(neg)==2 and report["forwards"]==16
        report["pass"]=bool(report["complete"] and all(all(g.values()) for g in report["gates"].values()))
    except Exception as e:report["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    if rel:report["released_CT"]={"body_bytes":rel.bytes,"chunks":rel.receipts}
    report.update(wall_seconds=time.monotonic()-t,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for k in ("cache","output","prior"):p.add_argument("--"+k,type=pathlib.Path)
    a=p.parse_args()
    if a.self_test:print("Tests passed:",self_test())
    else:
        if any(getattr(a,k) is None for k in ("cache","output","prior")) or a.output.exists():p.error("Required inputs and new output")
        run(a)
