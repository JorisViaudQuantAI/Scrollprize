#!/usr/bin/env python3
"""Actual public map replay of the frozen 071 funnel; no new GPU inference."""
from __future__ import annotations
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,importlib.util,json,math,pathlib,resource,signal,sys,time,urllib.request
import numpy as np
import cv2
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import router as f
CASES=("pherc306b_context_raw","pherc306b_blank_raw","pherc306b_shuffle_raw","pherc306b_context_matched_raw","pherc343_context_display")
CAP=240*1024**2
def mod(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def acquire(src,cache,receipt):
    cache.mkdir(parents=True,exist_ok=False)
    prep=mod("public_line_prepare",src/"firstletters-colab/line_controls/prepare.py")
    used=0
    for record in prep.SOURCES:
        name,url,_,_=record
        p=cache/name
        with urllib.request.urlopen(url,timeout=60) as response,p.open("wb") as output:
            while True:
                block=response.read(min(1024*1024,CAP-used+1))
                if not block:break
                used+=len(block)
                if used>CAP:raise ValueError("Source cap exceeded")
                output.write(block)
        receipt["source_bytes"]=used
        prep.fetch_verified(cache,record)
    # The original immutable extractor uses the verified local source files.
    argv=sys.argv
    try:
        sys.argv=["prepare.py","--output-dir",str(cache)];prep.main()
    finally:sys.argv=argv
    return json.loads((cache/"inventory.json").read_text())
def random_spread(ts,valid,targets,spacing):
    n=len(ts);h,w=valid.shape
    # Disjoint tiles permit exact additive coverage, without drawing 128 full masks.
    areas=np.array([t["supported"] for t in ts],np.int64)
    box=np.zeros((n,len(targets)),np.int64);line=box.copy()
    for j,t in enumerate(targets):
        x,y,X,Y=t["box_xyxy"];x0=max(0,int(math.floor(x)));y0=max(0,int(math.floor(y)));x1=min(w,int(math.ceil(X)));y1=min(h,int(math.ceil(Y)))
        yc=min(h-1,max(0,int(round((y+Y)/2))))
        for i,q in enumerate(ts):
            a,b,A,B=q["box"];u=max(y0,a);v=max(x0,b);U=min(y1,A);V=min(x1,B)
            if U>u and V>v:box[i,j]=valid[u:U,v:V].sum()
            if a<=yc<A and V>v:line[i,j]=valid[yc,v:V].sum()
    bd=box.sum(0);ld=line.sum(0)
    allmetrics={}
    for budget in f.BUDGETS:
        k=math.ceil(budget*n);rows=[]
        for seed in range(7102,7134):
            sel=f.permutation(n,seed)[:k]
            bc=np.divide(box[sel].sum(0),bd,out=np.zeros_like(bd,dtype=float),where=bd>0)
            lc=np.divide(line[sel].sum(0),ld,out=np.zeros_like(ld,dtype=float),where=ld>0)
            rows.append((areas[sel].sum()/areas.sum(),float(lc[ld>0].min()) if (ld>0).any() else None,int(((bc>=.5)&(bd>0)).sum())))
        d={}
        for j,key in enumerate(("supported_area_fraction","worst_centerline","boxes_half")):
            vals=[q[j] for q in rows if q[j] is not None]
            d[key]={"min":float(min(vals)),"mean":float(np.mean(vals)),"max":float(max(vals))} if vals else None
        allmetrics[str(budget)]=d
    return allmetrics
def alarm(sig,frame):raise TimeoutError("Line ranker exceeded frozen 180 s/case")
def evaluate_case(cache,name,expected,ranker,stroke):
    p=cache/"cases"/name;m=json.loads((p/"manifest.json").read_text())
    for k in ("score_sha256","mask_sha256","shape","valid_pixels","targets","spacing_um_yx","map_kind"):
        if m[k]!=expected[k]:raise ValueError("Prepared manifest mismatch: "+k)
    for k,hashkey in (("map","score_sha256"),("mask","mask_sha256")):
        if f.sha(p/m[k])!=m[hashkey]:raise ValueError("Prepared array hash mismatch: "+k)
    score=np.load(p/m["map"],allow_pickle=False);valid=np.load(p/m["mask"],allow_pickle=False)
    signal.signal(signal.SIGALRM,alarm);signal.alarm(180);t=time.monotonic()
    try:rep=ranker.rank_map(score,valid,m["spacing_um_yx"],m["map_kind"],ranker.Config())
    finally:signal.alarm(0)
    line_seconds=time.monotonic()-t;t=time.monotonic()
    ts,step,scores,orders=f.rank_tiles(score,valid,m["spacing_um_yx"],rep["proposals"],stroke.component_strokeness)
    tile_seconds=time.monotonic()-t;t=time.monotonic();metrics={}
    for k,order in orders.items():
        metrics[k]={str(q):f.metric(ts,order,math.ceil(q*len(ts)),valid,m["targets"],m["spacing_um_yx"]) for q in f.BUDGETS}
    rs=random_spread(ts,valid,m["targets"],m["spacing_um_yx"])
    for k,hashkey in (("map","score_sha256"),("mask","mask_sha256")):
        if f.sha(p/m[k])!=m[hashkey]:raise ValueError("Input changed during replay")
    full=all(v["1.0"]["supported_area_fraction"]==1 and all(q["box_fraction"] in (None,1) and q["centerline_fraction"] in (None,1) for q in v["1.0"]["targets"]) for v in metrics.values())
    if not full:raise ValueError("Full budget failed exact support coverage")
    return {"id":name,"complete":True,"kind":m["map_kind"],"role":m["role"],"map_sha256":m["score_sha256"],"mask_sha256":m["mask_sha256"],
      "supported_pixels":int(valid.sum()),"tile_count":len(ts),"tile_shape_nominal":list(step),"tiles":ts,"scores":scores,"orders":orders,"metrics":metrics,
      "random32":rs,"line_proposals":len(rep["proposals"]),"line_truncated":rep["search_truncated"],"rank_seconds":{"line":line_seconds,"tile":tile_seconds,"metrics":time.monotonic()-t},
      "whole_map_available":True,"blank_verdict":None}
def run(a):
    t=time.monotonic();r={"schema":"adaptive.search_funnel.v1","protocol_sha256":f.sha(HERE/"protocol.md"),"router_sha256":f.sha(HERE/"router.py"),"script_sha256":f.sha(__file__),"pass":False,"cases":[]}
    try:
        r["synthetic"]=f.self_test()
        if not r["synthetic"]["pass"]:raise ValueError("Router synthetic contracts failed")
        matrix=a.sources/"gemini_lane/evidence/coverage_matrix.json"
        payload=matrix.read_bytes()
        import hashlib
        blob=hashlib.sha1(f"blob {len(payload)}\0".encode()+payload).hexdigest()
        if blob!="a4b94bb278d3b414be794b6e76ba51edb83fb984":raise ValueError("Inventory blob mismatch")
        r["inventory"]=f.inventory(json.loads(payload));r["inventory_sha256"]=f.sha(matrix)
        r["inventory_summary"]={"scrolls":len(r["inventory"]),"register_and_preflight":sum(q["route"]=="REGISTER_AND_PREFLIGHT" for q in r["inventory"]),"generate_geometry":sum(q["route"]=="GENERATE_GEOMETRY" for q in r["inventory"])}
        ranker=mod("line_review071",a.sources/"firstletters-colab/line_review/review.py");stroke=mod("gemini_stroke071",a.sources/"gemini_lane/screening_fp_filter.py")
        r["upstream_hashes"]={str(p.relative_to(a.sources)):f.sha(p) for p in (a.sources/"firstletters-colab/line_review/review.py",a.sources/"gemini_lane/screening_fp_filter.py",a.sources/"firstletters-colab/line_controls/prepare.py",a.sources/"firstletters-colab/line_evaluation/protocol.v2.json")}
        cv2.setNumThreads(4)
        inv=acquire(a.sources,a.cache,r);r["public_sources"]=inv["source_files"]
        prev=json.loads((a.sources/"firstletters-colab/line_evaluation/protocol.v2.json").read_text())
        for name in CASES:
            try:q=evaluate_case(a.cache,name,prev["cases"][name],ranker,stroke)
            except Exception as e:q={"id":name,"complete":False,"failure":f"{type(e).__name__}: {str(e)[:300]}"}
            r["cases"].append(q);print("CASE_JSON="+json.dumps({k:v for k,v in q.items() if k not in ("tiles","scores","orders","metrics","random32")},separators=(",",":"),allow_nan=False),flush=True)
        m={q["id"]:q for q in r["cases"] if q["complete"]};g={"all_cases":len(m)==5,"all_inventory":len(r["inventory"])==19}
        if len(m)==5:
            a0=m[CASES[0]]["metrics"];a1=m[CASES[4]]["metrics"]
            F=a0["F"]["0.5"];H=a0["H"]["0.5"];G=a1["F"]["0.5"];J=a1["H"]["0.5"]
            g.update(full306b_both_rows=F["worst_centerline"]>=.90,display343_boxes=G["boxes_half"]>=16,
                     nonharm_H_306b=F["worst_centerline"]>=H["worst_centerline"]-.05,nonharm_H_343=G["boxes_half"]>=J["boxes_half"]-1,
                     no_truncation=all(not q["line_truncated"] for q in m.values()))
        r["gates"]=g;r["pass"]=all(g.values())
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["wall_seconds"]=time.monotonic()-t;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("sources","cache","output"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
