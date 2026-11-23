#!/usr/bin/env python3
"""One corrected point-render A-hn arm; compare frozen prior model receipts."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="4"
import argparse,hashlib,importlib.util,json,pathlib,resource,sys,time,urllib.request
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parents[1]
DEPTHS=np.arange(-12,12,dtype=float)

def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def sha(p):
    with pathlib.Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def origin(yx):return ((yx[1]+.5)/20.,(yx[0]+.5)/20.)

def self_test():
    assert np.allclose(origin((100,200)),[10.025,5.025])
    return 1

def run(a):
    t=time.monotonic()
    report={"schema":"adaptive.corrected_detector.v1","unit_checks":self_test(),"script_sha256":sha(__file__),
            "protocol_sha256":sha(HERE/"protocol.md"),"forwards":0,"rows":[],"pass":False}
    old=json.loads(a.prior.read_text());ctprev=json.loads(a.ct_result.read_text())
    report["prior_040_sha256"]=sha(a.prior);report["prior_042_sha256"]=sha(a.ct_result)
    bridge=mod("adaptive_bridge",HERE.parent/"shared_renderer_bridge"/"run.py")
    d=mod("adaptive_detector",HERE.parent/"detector_bridge"/"run.py")
    surf=mod("shared_original_surface",a.shared/"surface.py")
    renderer=mod("shared_footprint_renderer",a.shared/"render.py")
    sys.path.insert(0,str(a.gemini));import grow_m7_patch as gm
    sys.path.insert(0,str(BASE/"tile_phase"))
    common=mod("common",BASE/"tile_phase"/"common.py")
    prep=mod("existing_tile_prepare",BASE/"tile_phase"/"prepare.py")
    modelapi=mod("existing_tile_detector",BASE/"tile_phase"/"run.py")
    budget=gm.ByteBudget(160*1024**2);rel=None
    try:
        if not ctprev["pass"] or not old["calibration"]["pass"] or old["calibration"]["chosen_sign"]!=1 or old["forwards"]!=24:
            raise ValueError("Prior calibrated 040 and passing 042 receipt required")
        atlas=bridge.load_atlas(a.atlas,bridge.PIN);chart=bridge.ChartBridge(atlas,0,surf.OriginalSurface)
        a.cache.mkdir(parents=True,exist_ok=False);trace=a.cache/"trace.png"
        with urllib.request.urlopen(d.TRACE_URL,timeout=45) as s:trace.write_bytes(s.read(75000))
        labels,meta=d.trace_labels(trace)
        if meta["sha256"]!=old["annotation"]["sha256"] or meta["eroded_trace_pixels"]!=old["annotation"]["eroded_trace_pixels"]:raise ValueError("Trace source changed")
        report["annotation"]=meta;report["selection"]=old["selection"]
        rel=d.ReleasedRender(a.cache,prep.fetch,common.HF_DATA)
        inputs=[]
        with gm.BudgetedRemoteZarrV2(bridge.CT_URL,level="0",budget=budget,workers=1,cache_mb=128,max_retries=1,timeout=20) as reader:
            for tile,prior in zip(old["selection"]["tiles"],old["rows"],strict=True):
                if tile["yx"]!=prior["yx"] or tile["kind"]!=prior["kind"]:raise ValueError("Frozen tile list mismatch")
                yx=tuple(tile["yx"]);y,x=yx
                B=rel.tile(yx)
                if arrsha(d.clip_tile(B))!=prior["arms"]["B"]["input_sha256"]:raise ValueError("Published B input differs")
                label=None
                if tile["kind"]=="trace":
                    label=labels[y-d.RAW_ORIGIN[0]+16:y-d.RAW_ORIGIN[0]+48,x-d.RAW_ORIGIN[1]+16:x-d.RAW_ORIGIN[1]+48]
                    if int((label==2).sum())!=tile["positive_core"] or int((label==1).sum())!=tile["untraced_core"]:raise ValueError("Frozen trace class counts mismatch")
                o=origin(yx)
                lo,hi=chart.dense_crop_bounds(o,(.05,.05),(64,64),DEPTHS)
                block,org,support=reader.read_box(lo[::-1],hi[::-1],return_support=True)
                if tuple(org)!=tuple(lo[::-1]) or tuple(block.shape)!=tuple((hi-lo)[::-1]) or not support.all():raise ValueError("Dense crop incomplete")
                H,m,_=renderer.render(block.astype(np.float32),np.asarray(org[::-1],float),chart,o,(.05,.05),(64,64),DEPTHS,samples_per_axis=1)
                if not m.all() or H.shape!=(24,64,64):raise ValueError("Corrected full context incomplete")
                inputs.append((tile,prior,label,d.clip_tile(H),d.stats(B,H)))
        if len(inputs)!=8:raise ValueError("Expected eight frozen tiles")
        md=a.cache/"model";md.mkdir()
        for name,(url,expected) in common.FILES.items():prep.fetch(url,md/name,expected)
        report["verified_model_inputs"]=modelapi.verified_inputs(md)
        if report["verified_model_inputs"]!=old["verified_model_inputs"]:raise ValueError("Model input provenance mismatch")
        model,torch=modelapi.load_model(md,"cpu");report["runtime"]=modelapi.runtime(torch,"cpu")
        for tile,prior,lab,H,ct in inputs:
            tick=time.monotonic()
            s=modelapi.predict(model,torch,H,"cpu");report["forwards"]+=1
            core=s[16:48,16:48]
            r=d.metrics(core,lab) if tile["kind"]=="trace" else {"author_blank_flagged_at_05":float((core>=.5).mean()),"mean_score":float(core.mean())}
            row={**tile,"CT_B_H":ct,"CT_B_P":prior["CT"]["B_P"],
                 "H":{**r,"input_sha256":arrsha(H),"score_sha256":arrsha(s),"forward_seconds":time.monotonic()-tick},
                 "prior_B_input_exact":True,"prior_trace_counts_exact":True}
            report["rows"].append(row);print("TILE_JSON="+json.dumps(row,separators=(",",":")),flush=True)
        pos=[r for r in report["rows"] if r["kind"]=="trace"];neg=[r for r in report["rows"] if r["kind"]=="author_blank"]
        H={k:float(np.mean([r["H"][k] for r in pos])) for k in ("trace_AP","trace_AUC","trace_recall_at_05","distant_untraced_flagged_at_05")}
        H["author_blank_flagged_at_05"]=float(np.mean([r["H"]["author_blank_flagged_at_05"] for r in neg]))
        report["macro"]={**old["macro"],"H":H};B=old["macro"]["B"];P=old["macro"]["P"]
        p=float(np.mean([r["CT_B_P"]["mae"] for r in report["rows"]]));h=float(np.mean([r["CT_B_H"]["mae"] for r in report["rows"]]))
        report["mean_CT_MAE"]={"P":p,"H":h,"relative_reduction":1-h/p}
        gates={"complete":len(pos)==6 and len(neg)==2 and report["forwards"]==8,
               "old_inputs_labels_exact":all(r["prior_B_input_exact"] and r["prior_trace_counts_exact"] for r in report["rows"]),
               "AP_gain_vs_P_ge02":H["trace_AP"]>=P["trace_AP"]+.02,
               "AP_noninferior_to_B":H["trace_AP"]>=B["trace_AP"]-.03,
               "trace_recall_noninferior_to_B":H["trace_recall_at_05"]>=B["trace_recall_at_05"]-.05,
               "blank_flag_bounded":H["author_blank_flagged_at_05"]<=B["author_blank_flagged_at_05"]+.03 and H["author_blank_flagged_at_05"]<=P["author_blank_flagged_at_05"],
               "CT_MAE_reduction_ge20pct":h<=.8*p}
        report["gates"]=gates;report["pass"]=all(gates.values())
    except Exception as e:report["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    report["native_CT"]={"charged_body_bytes":budget.used,"requests":budget.requests,"cap_bytes":budget.cap}
    if rel:report["released_CT"]={"body_bytes":rel.bytes,"chunks":rel.receipts}
    report["wall_seconds"]=time.monotonic()-t;report["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--self-test",action="store_true")
    for k in ("atlas","shared","gemini","cache","output","prior","ct-result"):ap.add_argument("--"+k,type=pathlib.Path)
    a=ap.parse_args()
    if a.self_test:print("Tests passed:",self_test())
    else:
        if any(getattr(a,k) is None for k in ("atlas","shared","gemini","cache","output","prior","ct_result")) or a.output.exists():ap.error("Required inputs and new output")
        run(a)
