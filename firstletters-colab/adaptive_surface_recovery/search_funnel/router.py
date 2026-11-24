#!/usr/bin/env python3
"""Non-destructive funnel routing and budgeted ranking.

No automatic blank/candidate verdict. Operational chart IDs and maps stay private.
Public experiment uses only previously awarded controls.
"""
from __future__ import annotations
import hashlib,json,math,random,re
from pathlib import Path
import numpy as np
import cv2

BUDGETS=(.25,.5,.75,1.)
SEED=7101
PUBLIC_SCROLLS=tuple("PHerc"+x for x in ("0125","0175A","0175B","0191","0257","0268","0358","0483A","0483B","0490A","0490B","0800","0813","0826","0846A","0846B","1203","1218","1545"))
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def outside_git(path):
    p=Path(path).resolve()
    if any((q/".git").exists() for q in (p,*p.parents)):raise ValueError("Private outputs must be outside Git worktrees")
    return p
def inventory(matrix):
    rows=matrix["scrolls"]
    if sorted(q["scroll"] for q in rows)!=sorted(PUBLIC_SCROLLS):raise ValueError("Eligible inventory mismatch")
    out=[]
    for q in rows:
        atlas=int(q["pscamillo_meshes"]);fits=int(q["hf_spiral_fit_runs"]);seg=int(q["s3_segments_count"])
        if min(atlas,fits,seg)<0:raise ValueError("Negative source count")
        out.append({"scroll":q["scroll"],"public_mesh_count":atlas,"public_fit_runs":fits,"official_segment_count":seg,
          "fine_scan":bool(q["has_2_4um_rescan"]),"m7_available":bool(q["s3_has_m7_and_normal_grids"]),
          "route":"REGISTER_AND_PREFLIGHT" if atlas+fits+seg else "GENERATE_GEOMETRY",
          "exploration_required":True,"whole_scroll_coverage":None})
    return out

def route(row):
    """Missing is not false, and false is not blank. Return the next action only."""
    if not row.get("geometry_sha256") or not row.get("frame_id"):return "REGISTER_GEOMETRY"
    if row.get("source_support")!="verified":return "REPAIR_SUPPORT"
    if row.get("geometry_state")!="verified_local":return "FLATTEN_OR_REPAIR"
    if row.get("positive_gate")!="same_configuration_pass":return "RUN_MATCHED_CONTROL"
    if row.get("coarse_orders")!=["asc","desc"] and row.get("coarse_orders")!=["desc","asc"]:return "COARSE_BOTH_ORDERS"
    if not row.get("render_sha256") or not row.get("recipe_sha256"):return "REGISTER_COARSE_PROVENANCE"
    if not isinstance(row.get("cost_seconds"),(int,float)) or not math.isfinite(row["cost_seconds"]) or row["cost_seconds"]<=0:return "MEASURE_COST"
    return "PRIORITIZE_AND_EXPLORE"

def plan(rows):
    seen=set();out=[]
    for row in rows:
        if row.get("scroll") not in PUBLIC_SCROLLS or not row.get("chart_id"):raise ValueError("Invalid private manifest row")
        state=route(row)
        key=(row.get("scroll"),row.get("frame_id"),row.get("geometry_sha256"),row.get("render_sha256"),row.get("recipe_sha256"))
        # Distinct geometry is never deduplicated by mere spatial proximity.
        if state=="PRIORITIZE_AND_EXPLORE":
            if key in seen:state="EXACT_DUPLICATE"
            else:seen.add(key)
        out.append({"chart_id":row["chart_id"],"scroll":row["scroll"],"next_action":state,"blank_verdict":None})
    return out

def tiles(valid,spacing):
    h,w=valid.shape;dy,dx=[max(1,int(round(10000/s))) for s in spacing]
    out=[]
    for y in range(0,h,dy):
        for x in range(0,w,dx):
            Y=min(h,y+dy);X=min(w,x+dx);n=int(valid[y:Y,x:X].sum())
            if n:out.append({"id":len(out),"box":[y,x,Y,X],"supported":n,"support_fraction":n/((Y-y)*(X-x))})
    return out,(dy,dx)
def stable(scores):
    return sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i))
def permutation(n,seed):
    x=list(range(n));random.Random(seed).shuffle(x);return x
def portfolio(orders):
    n=len(orders[0]);out=[];used=set();ptr=[0]*len(orders)
    if any(sorted(q)!=list(range(n)) for q in orders):raise ValueError("Not a permutation")
    while len(out)<n:
        j=len(out)%len(orders);q=orders[j]
        while q[ptr[j]] in used:ptr[j]+=1
        x=q[ptr[j]];out.append(x);used.add(x);ptr[j]+=1
    return out
def polygon_scores(ts,props):
    s=np.zeros(len(ts),float)
    for i,p in enumerate(props):
        poly=np.asarray(p["band_polygon_xy"],np.float32).reshape(-1,2)
        if len(poly)<3 or not np.isfinite(poly).all():continue
        poly=cv2.convexHull(poly)
        for t in ts:
            y,x,Y,X=t["box"]
            box=np.array([[x,y],[X,y],[X,Y],[x,Y]],np.float32)
            area,_=cv2.intersectConvexConvex(poly,box,handleNested=True)
            q=math.sqrt(max(0.,min(1.,float(area)/((Y-y)*(X-x))))/math.sqrt(1+i))
            s[t["id"]]=max(s[t["id"]],q)
    return s
def stroke_score(score,valid,spacing,stroke):
    h,w=score.shape;H=max(1,int(round(h*spacing[0]/35)));W=max(1,int(round(w*spacing[1]/35)))
    den=cv2.resize(valid.astype(np.float32),(W,H),interpolation=cv2.INTER_AREA)
    num=cv2.resize(np.where(valid,score,0).astype(np.float32),(W,H),interpolation=cv2.INTER_AREA)
    ok=den>=1-1e-6
    hit=(num/np.maximum(den,1e-6)>=.7)&ok
    d=stroke(hit,voxel_um=math.sqrt(spacing[0]*h/H*spacing[1]*w/W))
    return float(d["weighted_mean"] or 0.)
def rank_tiles(score,valid,spacing,props,stroke):
    if score.shape!=valid.shape or valid.dtype!=bool or not np.isfinite(score[valid]).all():raise ValueError("Bad map/mask")
    ts,step=tiles(valid,spacing)
    if len(ts)>120 or not ts:raise ValueError("Tile count outside frozen experiment")
    mass=[];st=[]
    for t in ts:
        y,x,Y,X=t["box"];m=valid[y:Y,x:X];a=score[y:Y,x:X]
        mass.append(float(np.quantile(a[m],.95)));st.append(stroke_score(a,m,spacing,stroke))
    line=polygon_scores(ts,props)
    orders={"H":stable(mass),"L":stable(line),"S":stable(st),"R":permutation(len(ts),SEED)}
    orders["F"]=portfolio([orders[k] for k in ("H","L","S","R")])
    return ts,step,{"H":mass,"L":line.tolist(),"S":st},orders

def covered(ts,order,k,valid):
    out=np.zeros(valid.shape,bool)
    for i in order[:k]:
        y,x,Y,X=ts[i]["box"];out[y:Y,x:X]=True
    return out&valid
def coverage(mask,valid,targets):
    h,w=valid.shape;out=[]
    for t in targets:
        x,y,X,Y=t["box_xyxy"]
        x0,y0=max(0,int(math.floor(x))),max(0,int(math.floor(y)))
        x1,y1=min(w,int(math.ceil(X))),min(h,int(math.ceil(Y)))
        if x1<=x0 or y1<=y0:raise ValueError("Target outside map")
        n=int(valid[y0:y1,x0:x1].sum());a=int(mask[y0:y1,x0:x1].sum())
        yy=min(h-1,max(0,int(round((y+Y)/2))));nn=int(valid[yy,x0:x1].sum());aa=int(mask[yy,x0:x1].sum())
        out.append({"id":t["id"],"support_pixels":n,"covered_pixels":a,"box_fraction":a/n if n else None,
            "centerline_support":nn,"centerline_covered":aa,"centerline_fraction":aa/nn if nn else None})
    return out
def metric(ts,order,k,valid,targets,spacing):
    m=covered(ts,order,k,valid);t=coverage(m,valid,targets);c=[x["centerline_fraction"] for x in t if x["centerline_fraction"] is not None]
    return {"tiles":k,"padded_cost_fraction":k/len(ts),"supported_area_cm2":int(m.sum())*math.prod(spacing)/1e8,
        "supported_area_fraction":int(m.sum())/int(valid.sum()),"repair_tiles":sum(ts[i]["support_fraction"]<.75 for i in order[:k]),
        "targets":t,"worst_centerline":min(c) if c else None,
        "boxes_half":sum(x["box_fraction"] is not None and x["box_fraction"]>=.5 for x in t)}
def self_test():
    v=np.ones((5,7),bool);v[0,0]=False
    # 2px tiles, truncated edges.
    ts,_=tiles(v,[5000,5000]);n=len(ts);ids=list(range(n))
    tar=[{"id":"tiny","box_xyxy":[0,0,2,2]}]
    m=covered(ts,ids,1,v);c=coverage(m,v,tar)[0]
    q=portfolio([ids,ids[::-1],ids,permutation(n,SEED)])
    validrow={"scroll":"PHerc0125","chart_id":"private-a","geometry_sha256":"a","frame_id":"frame","source_support":"verified","geometry_state":"verified_local","positive_gate":"same_configuration_pass","coarse_orders":["asc","desc"],"render_sha256":"r","recipe_sha256":"c","cost_seconds":1.}
    routes=plan([validrow,{**validrow,"chart_id":"private-b"},{**validrow,"chart_id":"private-c","geometry_sha256":"b"}])
    gates={"tile_partition":int(covered(ts,ids,n,v).sum())==int(v.sum())==sum(x["supported"] for x in ts),
        "target_oracle":c["support_pixels"]==3 and c["covered_pixels"]==3 and c["box_fraction"]==1,
        "stable_ties":stable([0,1,1])==[1,2,0],"portfolio":len(q)==n and sorted(q)==ids and permutation(n,SEED)==permutation(n,SEED),
        "missing_support":route({**validrow,"source_support":None})=="REPAIR_SUPPORT","bad_geometry":route({**validrow,"geometry_state":"unknown"})=="FLATTEN_OR_REPAIR",
        "missing_gate":route({**validrow,"positive_gate":None})=="RUN_MATCHED_CONTROL",
        "one_order":route({**validrow,"coarse_orders":["asc"]})=="COARSE_BOTH_ORDERS",
        "exact_only_dedup":[r["next_action"] for r in routes]==["PRIORITIZE_AND_EXPLORE","EXACT_DUPLICATE","PRIORITIZE_AND_EXPLORE"],
        "no_blank_verdict":all(r["blank_verdict"] is None for r in routes)}
    return {"gates":gates,"pass":all(gates.values())}
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--out-private",type=Path,required=True)
    a=p.parse_args();out=outside_git(a.out_private)
    if out.exists():raise ValueError("Use a new private output path")
    out.write_text(json.dumps({"jobs":plan(json.loads(a.manifest.read_text()))},indent=2)+"\n")
