#!/usr/bin/env python3
"""Review amendment 073: actual S0 table and geometry accounting stress."""
import argparse,csv,hashlib,importlib.util,io,json,math,pathlib,resource,sys,time
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parent/"adaptive_surface_recovery"
LEVELS=(.08,.20,.32,.44,.56,.68,.80,.92)
COLUMNS=("scroll","status","z_fraction","z_slice","slabs","clean_slabs","edge_rise_um_q75","merged_frac","coherence","air_frac")
CELL=64
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def blob(p):
    b=p.read_bytes();return hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest()
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def normalize_scroll(s):return s.replace("PHerc. ","PHerc").replace(" ","")
def parse_s0(path,eligible):
    reader=csv.DictReader(io.StringIO(path.read_text()))
    if tuple(reader.fieldnames)!=COLUMNS:raise ValueError("S0 column mismatch")
    groups={}
    for row in reader:
        scroll=normalize_scroll(row["scroll"]);f=round(float(row["z_fraction"]),2)
        z=int(row["z_slice"]);n=int(row["slabs"]);c=int(row["clean_slabs"])
        vals=[float(row[k]) for k in COLUMNS[6:]]
        if f not in LEVELS or z<0 or not 1<=n<=4 or not 0<=c<=n or not np.isfinite(vals).all() or vals[0]<0 or any(not 0<=q<=1 for q in vals[1:]):raise ValueError("Invalid S0 row")
        is_eligible=row["status"]=="eligible, unread"
        if is_eligible!=(scroll in eligible):raise ValueError("S0 eligibility mismatch")
        g=groups.setdefault(scroll,{"status":row["status"],"bands":{}})
        if f in g["bands"] or g["status"]!=row["status"]:raise ValueError("Duplicate band or status mismatch")
        g["bands"][f]={"fraction":f,"z_slice":z,"slabs":n,"clean":c,"rise":vals[0],"merged":vals[1],"coherence":vals[2],"air":vals[3]}
    if not eligible<=set(groups):raise ValueError("Missing eligible scroll")
    return groups
def summarize_s0(groups,eligible):
    out=[];private={}
    for s,g in sorted(groups.items()):
        rows=list(g["bands"].values());n=sum(q["slabs"] for q in rows);c=sum(q["clean"] for q in rows)
        rank=sorted(rows,key=lambda q:(-q["clean"]/q["slabs"],q["merged"],q["rise"],-q["coherence"],q["fraction"]))
        missing=[f for f in LEVELS if f not in g["bands"]]
        private[s]={"rank":rank,"missing_levels":missing,"proposed_first_two":rank[:2]}
        out.append({"scroll":s,"status":g["status"],"sampled_levels":len(rows),"missing_levels":len(missing),
            "slabs":n,"clean":c,"clean_fraction":c/n,"zero_clean_levels":sum(q["clean"]==0 for q in rows),
            "route":"GEOMETRY_FIRST" if c==0 else "PRIORITIZE_WITHIN_SCROLL",
            "unsampled_band_survey_required":bool(missing),"low_rank_survey_required":True})
    e=[q for q in out if q["scroll"] in eligible]
    controls=[q for q in out if q["scroll"] not in eligible]
    awarded=[q for q in controls if q["scroll"] in ("PHerc0211","PHerc0306B","PHerc0343")]
    return {"rows":sum(q["sampled_levels"] for q in out),"eligible_scrolls":len(e),"read_controls":len(controls),
        "eligible_rows":sum(q["sampled_levels"] for q in e),"eligible_missing_levels":sum(q["missing_levels"] for q in e),
        "eligible_slabs":sum(q["slabs"] for q in e),"eligible_clean":sum(q["clean"] for q in e),
        "scrolls":out,"awarded_rank_among_19_eligible":[{"scroll":q["scroll"],"clean_fraction":q["clean_fraction"],
            "eligible_strictly_higher":sum(a["clean_fraction"]>q["clean_fraction"] for a in e),
            "eligible_tied":sum(a["clean_fraction"]==q["clean_fraction"] for a in e)} for q in awarded],
        "rank_scope":"Within scroll only. Sparse exposed-control anatomy proxy, not ink probability."},private

def costs():
    # Claude's estimates, not measured by this run or owner-authorized grants.
    n=19*10;fit=(5.6,8.);coarse=3.6;deep=3.5;budget=140.
    band=[fit[0]+coarse,fit[1]+coarse];both=[fit[0]+2*coarse,fit[1]+2*coarse]
    return {"estimated_bands":n,"reviewer_reported_L4_hours_remaining_unverified":budget,
       "all_fit_plus_one_order_hours":[n*x for x in band],"all_fit_plus_both_orders_hours":[n*x for x in both],
       "bands_affordable_one_order_before_other_costs":[math.floor(budget/band[1]),math.floor(budget/band[0])],
       "bands_affordable_both_orders_before_other_costs":[math.floor(budget/both[1]),math.floor(budget/both[0])],
       "half_deep_fraction_saved_if_fit6_coarse3p6_deep3p5":(deep/2)/(6+coarse+deep),
       "IID_illustrations":[{"p_assumed":p,"n_for_hit_chance_95pct":math.ceil(math.log(.05)/math.log1p(-p))} for p in (.01,.03,.05,.10,1/60)],
       "warning":"Approximate review estimates only. IID p not measured; no automatic scroll stopping or allocation."}

def intersects(tri,center,side=CELL):
    """Triangle vs closed axis-aligned cube, 13 separating axes, vectorized."""
    v=np.asarray(tri,float)-np.asarray(center,float)[:,None,:];h=side/2
    eps=1e-8
    ok=(v.min(1)<=h+eps).all(1)&(v.max(1)>=-h-eps).all(1)
    e=np.roll(v,-1,axis=1)-v
    normal=np.cross(e[:,0],e[:,1])
    axes=[normal]+[np.cross(e[:,i],a) for i in range(3) for a in np.eye(3)]
    for a in axes:
        p=np.einsum("nij,nj->ni",v,a);radius=h*np.abs(a).sum(1)
        tol=eps*(1+np.abs(a).sum(1))
        ok&=(p.min(1)<=radius+tol)&(p.max(1)>=-radius-tol)
    return ok

def occupied(tri,side=CELL):
    tri=np.asarray(tri,float)
    if tri.ndim!=3 or tri.shape[1:]!=(3,3) or not np.isfinite(tri).all():raise ValueError("Invalid triangles")
    # Closed boxes: grid-aligned boundaries touch both neighboring cells.
    lo=np.floor((tri.min(1)-1e-8)/side).astype(np.int64)
    hi=np.floor((tri.max(1)+1e-8)/side).astype(np.int64)
    counts=hi-lo+1
    if int(np.prod(counts,axis=1).sum())>2_000_000:raise ValueError("Triangle/cell broadphase cap")
    if not len(tri):return set()
    cells=set()
    for delta in np.ndindex(tuple(counts.max(0))):
        d=np.array(delta);ids=np.flatnonzero((d[None]<counts).all(1))
        if not len(ids):continue
        c=lo[ids]+d;yes=intersects(tri[ids],(c+.5)*side,side)
        cells.update(map(tuple,c[yes].tolist()))
    return cells

def self_test():
    tri=np.array([[[10,10,16],[50,10,16],[10,50,16]]],float)
    tri2=tri+np.array([0,0,20])
    boundary=tri.copy();boundary[...,2]=64
    outside=np.array([[[65,65,32],[80,50,32],[50,80,32]]],float)
    # outside triangle satisfies x+y>=130; target [0,64]^2 has x+y<=128.
    gates={"plane_one_cell":occupied(tri)=={(0,0,0)},"separated_planes_share_cell":occupied(tri)==occupied(tri2),
        "boundary_both_cells":occupied(boundary)=={(0,0,0),(0,0,1)},
        "overlapping_aabb_but_disjoint":not intersects(outside,np.array([[32,32,32]]))[0],
        "negative_grid":occupied(tri-np.array([64,64,64]))=={(-1,-1,-1)}}
    return {"gates":{k:bool(v) for k,v in gates.items()},"pass":all(gates.values())}

def geometry(claude):
    atlas=module("atlas067_for_footprint",BASE/"claude_carrier_atlas"/"run.py")
    prior=json.loads((BASE/"claude_carrier_atlas"/"result.json").read_text())
    if prior["pinned_claude_commit"]!=atlas.PIN or not prior["pass"]:raise ValueError("067 provenance mismatch")
    w,source=atlas.load_winner()
    if w.receipt["coordinate_sha256"]!=prior["winner"]["mesh"]["coordinate_sha256"]:raise ValueError("Winner mesh mismatch")
    W=occupied(w.xyz[w.faces]);rows=[];union=set(W);total=sum(q["bytes"] for q in source.values())
    for name in atlas.WINDS:
        m,_,rec=atlas.load_fit(claude,"flat",name)
        old=prior["states"]["flat"]["windings"][name]
        if rec["mesh"]["coordinate_sha256"]!=old["mesh"]["coordinate_sha256"] or rec["files"]!=old["files"]:raise ValueError("067 fit source mismatch")
        total+=rec["source_bytes"]
        if total>16*1024**2:raise ValueError("Geometry cap")
        import base64,zlib
        mp=old["reciprocal_code_map"];c=np.frombuffer(zlib.decompress(base64.b64decode(mp["data"])),dtype=np.uint8).reshape(mp["shape"])
        if atlas.arrsha(c)!=mp["sha256"] or c.shape!=m.shape:raise ValueError("067 map mismatch")
        F=occupied(m.xyz[m.faces]);union.update(F)
        same=np.zeros(len(m.xyz),bool)
        cells=np.floor(m.xyz[m.used]/CELL).astype(np.int64)
        same[m.used]=[tuple(q) in W for q in cells]
        code=c.ravel();weight=m.weight
        a={}
        for key,mask in (("near_le3",code==4),("intermediate",code==5),("far_ge6",code==6),("unknown",np.isin(code,[1,2,3]))):
            num=int(mask.sum());hits=int((same&mask).sum());ww=float(weight[mask].sum())
            a[key]={"vertices":num,"same_cell":hits,"same_cell_fraction":hits/num if num else None,
                "area_fraction_same_cell":float(weight[mask&same].sum()/ww) if ww else None}
        rows.append({"winding":name,"fit_cells":len(F),"shared_cells":len(F&W),"shared_fraction_fit_cells":len(F&W)/len(F),
            "shared_fraction_winner_cells":len(F&W)/len(W),"ray_code_groups":a})
    far=sum(q["ray_code_groups"]["far_ge6"]["vertices"] for q in rows)
    hits=sum(q["ray_code_groups"]["far_ge6"]["same_cell"] for q in rows)
    near=sum(q["ray_code_groups"]["near_le3"]["vertices"] for q in rows)
    nh=sum(q["ray_code_groups"]["near_le3"]["same_cell"] for q in rows)
    return {"source_bytes":total,"cell_side_native_vox":CELL,"winner_cells":len(W),"union_cells":len(union),"windings":rows,
        "far_vertices":far,"far_same_cell":hits,"near_vertices":near,"near_same_cell":nh,
        "cell_only_dedup_rejected":hits>0,"m7_denominator":None,"physical_sheet_area_fraction":None,
        "scope":"Exact closed triangle-cell occupancy plus existing conservative 067 geometry labels; not anatomical identity."}

def run(a):
    t=time.monotonic();r={"schema":"funnel.amendment073.v1","amendment_sha256":sha(HERE/"AMENDMENT-001.md"),"script_sha256":sha(pathlib.Path(__file__)),
      "synthetic":self_test(),"complete":False}
    try:
        if not r["synthetic"]["pass"]:raise ValueError("Occupancy synthetic checks failed")
        prior=json.loads((BASE/"search_funnel"/"result.json").read_text())
        eligible={q["scroll"] for q in prior["inventory"]}
        if len(eligible)!=19 or blob(a.s0)!="333735fa42e032171d0fc8ebd3df4f4a0da9f1b2":raise ValueError("Pinned S0 or eligibility mismatch")
        g=parse_s0(a.s0,eligible);r["S0"],private=summarize_s0(g,eligible)
        r["S0_source_sha256"]=sha(a.s0);r["costs"]=costs()
        # The ranked band plan is never in the public receipt.
        from control_plane import private_root
        root=private_root(a.private_dir)
        p=root/"s0-private-ranking.json"
        with p.open("x") as f:json.dump(private,f,indent=2)
        r["geometry"]=geometry(a.claude)
        r["complete"]=True
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:350]}"
    r["wall_seconds"]=time.monotonic()-t;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
    if not r["complete"]:raise SystemExit(1)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("s0","claude","output","private-dir"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
