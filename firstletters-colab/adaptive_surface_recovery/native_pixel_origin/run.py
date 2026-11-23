#!/usr/bin/env python3
"""Check existing renderer's half-pixel convention on the shared material atlas."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="4"
import argparse,hashlib,importlib.util,json,pathlib,resource,sys,time
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parents[1]
DEPTHS=np.arange(-12,12,dtype=float)

def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def sha(p):
    with pathlib.Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def origin(yx,offset):
    return ((yx[1]+offset)/20.,(yx[0]+offset)/20.)

def self_test():
    assert np.allclose(np.array(origin((100,200),.5))-origin((100,200),0),[.025,.025])
    return 1

def run(a):
    start=time.monotonic()
    report={"schema":"adaptive.native_pixel_origin.v1","unit_checks":self_test(),"protocol_sha256":sha(HERE/"protocol.md"),
            "script_sha256":sha(__file__),"cases":[],"pass":False,"CNN_forwards":0}
    bridge=mod("adaptive_bridge",HERE.parent/"shared_renderer_bridge"/"run.py")
    det=mod("adaptive_detector",HERE.parent/"detector_bridge"/"run.py")
    surface=mod("shared_original_surface",a.shared/"surface.py")
    renderer=mod("shared_footprint_renderer",a.shared/"render.py")
    sys.path.insert(0,str(a.gemini));import grow_m7_patch as gm
    sys.path.insert(0,str(BASE/"tile_phase"))
    common=mod("common",BASE/"tile_phase"/"common.py")
    prep=mod("existing_tile_prepare",BASE/"tile_phase"/"prepare.py")
    old=json.loads(a.previous.read_text())
    report["previous_041_sha256"]=sha(a.previous)
    budget=gm.ByteBudget(80*1024**2);rel=None
    try:
        if not old["detector_040_calibration"]["pass"] or old["detector_040_calibration"]["chosen_sign"]!=1:raise ValueError("Missing original positive sign calibration")
        chart=bridge.ChartBridge(bridge.load_atlas(a.atlas,bridge.PIN),0,surface.OriginalSurface)
        a.cache.mkdir(parents=True,exist_ok=False)
        rel=det.ReleasedRender(a.cache,prep.fetch,common.HF_DATA)
        with gm.BudgetedRemoteZarrV2(bridge.CT_URL,level="0",budget=budget,workers=1,cache_mb=128,max_retries=1,timeout=15) as reader:
            for i,center in enumerate([(1800,2800),(2200,4600),(2360,6400),(2800,6800)]):
                yx=det.nearest_origin(center)
                if old["cases"][i]["yx"]!=list(yx):raise ValueError("Prior tile origin mismatch")
                B=rel.tile(yx)
                origins=[origin(yx,0),origin(yx,.5)]
                bounds=[chart.dense_crop_bounds(o,(.05,.05),(64,64),DEPTHS) for o in origins]
                lo=np.minimum(bounds[0][0],bounds[1][0]);hi=np.maximum(bounds[0][1],bounds[1][1])
                if int(np.prod(hi-lo))>4_000_000:raise ValueError("Crop cap")
                block,org,support=reader.read_box(lo[::-1],hi[::-1],return_support=True)
                if tuple(org)!=tuple(lo[::-1]) or tuple(block.shape)!=tuple((hi-lo)[::-1]) or not support.all():raise ValueError("Incomplete dense CT source")
                v=block.astype(np.float32);org=np.asarray(org[::-1],float)
                values=[];masks=[];owners=[]
                for o in origins:
                    st,m,_=renderer.render(v,org,chart,o,(.05,.05),(64,64),DEPTHS,samples_per_axis=1)
                    r=chart.sample_grid(o,(.05,.05),(64,64))
                    values.append(st);masks.append(m);owners.append(r["original_face_id"])
                if not all(m.all() for m in masks):raise ValueError("A fixed tile lacks full footprint support")
                z,h=[det.stats(B,s) for s in values]
                baseline=old["cases"][i]["B_P"]["mae"]
                parity=abs(z["mae"]-baseline)<=1e-6
                row={"index":i,"calibration_excluded":i==2,"yx":list(yx),"B_zero":z,"B_half":h,
                     "zero_half":det.stats(*values),"relative_MAE_reduction":1-h["mae"]/z["mae"],
                     "old_041_B_P_MAE":baseline,"old_MAE_reproduced":bool(parity),
                     "supported_pixels":[int(m.sum()) for m in masks],
                     "different_original_face_owners":int((owners[0]!=owners[1]).sum())}
                report["cases"].append(row)
                if not parity:raise ValueError("Old baseline numerical replay failed")
        test=[x for x in report["cases"] if not x["calibration_excluded"]]
        z=float(np.mean([r["B_zero"]["mae"] for r in test]));h=float(np.mean([r["B_half"]["mae"] for r in test]))
        report["mean_CT_MAE"]={"zero":z,"half":h,"relative_reduction":1-h/z}
        report["gates"]={"complete":len(test)==3 and len(report["cases"])==4,
                         "old_baseline_exact_MAE":all(r["old_MAE_reproduced"] for r in report["cases"]),
                         "macro_reduction_ge20pct":h<=.8*z,
                         "every_test_nonworse":all(r["B_half"]["mae"]<=r["B_zero"]["mae"] for r in test),
                         "full_support":all(r["supported_pixels"]==[4096,4096] for r in report["cases"])}
        report["pass"]=all(report["gates"].values())
    except Exception as e:report["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    report["native_CT"]={"charged_body_bytes":budget.used,"requests":budget.requests,"cap_bytes":budget.cap}
    if rel:report["released_CT"]={"body_bytes":rel.bytes,"chunks":rel.receipts}
    report["wall_seconds"]=time.monotonic()-start;report["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--self-test",action="store_true")
    for k in ("atlas","shared","gemini","cache","output","previous"):ap.add_argument("--"+k,type=pathlib.Path)
    a=ap.parse_args()
    if a.self_test:print("Tests passed:",self_test())
    else:
        if any(getattr(a,k) is None for k in ("atlas","shared","gemini","cache","output","previous")) or a.output.exists():ap.error("Required inputs and new output")
        run(a)
