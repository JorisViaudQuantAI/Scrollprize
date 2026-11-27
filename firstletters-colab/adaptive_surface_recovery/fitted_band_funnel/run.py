#!/usr/bin/env python3
"""Rank actual published fit-quality maps. No new inference or location export."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,hashlib,importlib.util,json,math,pathlib,re,resource,subprocess,sys,time
import numpy as np
from PIL import Image
HERE=pathlib.Path(__file__).resolve().parent
PIN="1770a7b9f223a016167fda440f1867848373119d"
GEMINI="91bff462143d059793b98c88a40d9bf513fd7eda"
ROOT="firstletters-colab/results/pherc0211_calibration/maps_export211"
BUDGET=(7,15,25,40,60,100,149)
RANDOM_RANKS={27,29,37,43,52,64,76,88,91,108,122,127,129,138,144}
KNOWN={2,7};FLAG={6}
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def verify_source(root):
    git=lambda *a:subprocess.check_output(["git","-C",str(root),*a])
    if git("rev-parse","HEAD").decode().strip()!=PIN:raise ValueError("Claude revision mismatch")
    rows=git("ls-tree","-r","-z",PIN,"--",ROOT).split(b"\0");out={};total=0
    for row in rows:
        if not row:continue
        head,path=row.split(b"\t",1);kind,h=head.decode().split()[1:]
        if kind!="blob":raise ValueError("Non-blob source")
        p=root/path.decode();raw=p.read_bytes()
        total+=len(raw)
        if total>40*1024**2 or blob(raw)!=h:raise ValueError("Source hash/cap")
        out[str(p.relative_to(root/ROOT))]={"blob":h,"sha256":sha(raw),"bytes":len(raw)}
    return out,total
def image(p):
    with Image.open(p) as a:
        if a.width*a.height>5_000_000:raise ValueError("Image cap")
        v=np.asarray(a)
    if v.ndim==3:
        if v.shape[-1] not in (3,4) or not np.array_equal(v[...,0],v[...,1]) or not np.array_equal(v[...,0],v[...,2]):raise ValueError("Not grayscale probability/mask")
        v=v[...,0]
    if v.ndim!=2 or v.dtype!=np.uint8:raise ValueError("Expected 8bit grayscale")
    return v
def aligned_mask(a,shape):
    if a.shape==shape:return a>0,False
    # nearest only, preserving binary mask semantics; never imply native support.
    b=np.array(Image.fromarray(a).resize((shape[1],shape[0]),Image.Resampling.NEAREST))
    return b>0,True
def crop_bounds(x0,x1,W,w):
    if not 0<=x0<x1<=W:raise ValueError("Native crop outside canvas")
    return max(0,(x0*w)//W),min(w,(x1*w+W-1)//W)
def order(scores,ids):
    return sorted(range(len(ids)),key=lambda i:(-float(scores[i]),ids[i]))
def uniform(n,seed):return np.random.default_rng(seed).permutation(n).tolist()
def interleave(orders,n,initial=()):
    chosen=list(initial);seen=set(chosen);pos=[0]*len(orders);r=0
    while len(chosen)<n:
        j=r%len(orders);r+=1
        while pos[j]<len(orders[j]) and orders[j][pos[j]] in seen:pos[j]+=1
        if pos[j]==len(orders[j]):
            if all(all(i in seen for i in a) for a in orders):raise ValueError("Incomplete rank")
            continue
        i=orders[j][pos[j]];pos[j]+=1;seen.add(i);chosen.append(i)
    return chosen
def p95score(a,m):return float(np.quantile(a[m].astype(np.float64)/255,.95)) if m.any() else -1.
def strokes(a,m,stroke):
    if not m.any():return -1.
    v=stroke((a.astype(np.float64)/255>.7)&m,voxel_um=50.)
    return float(v["weighted_mean"] or 0.)
def share(a,m):return float((a[m].astype(float)/255>.7).mean()) if m.any() else None
def miss_prob(N,n,M):
    return math.comb(N-M,n)/math.comb(N,n) if N-M>=n else 0.
def upper(N,n,alpha=.05):
    return max(M for M in range(N+1) if miss_prob(N,n,M)>=alpha)
def recall_bounds(selected,positive,unknown):
    sel=set(selected);a=len(sel&positive);s=len(sel&unknown);o=len(unknown-sel);h=len(positive)
    return [a/(h+o) if h+o else None,(a+s)/(h+s) if h+s else None]
def metrics(selected,rows,reviewed,known,flag):
    sel=set(selected);n=len(sel);hits=known|flag
    unknown=set(range(len(rows)))-reviewed
    confirmed_unknown=unknown|flag
    return {"slots":n,"reported_chart_cm2":sum(float(rows[i]["valid_cm2"]) for i in sel),
      "reviewed":len(sel&reviewed),"unreviewed":len(sel&unknown),
      "confirmed_known":len(sel&known),"unconfirmed_flag_retained":bool(sel&flag),"reported_hits":len(sel&hits),
      "reviewed_nonhit":len(sel&(reviewed-hits)),
      "reviewed_hit_retention":len(sel&hits)/len(hits),
      "reviewed_hit_share_conditional":len(sel&hits)/len(sel&reviewed) if sel&reviewed else None,
      "all_population_reported_hit_recall_bounds":recall_bounds(sel,hits,unknown),
      "all_population_confirmed_recall_bounds":recall_bounds(sel,known,confirmed_unknown),
      "counterfactual_deep_GPU_seconds":110*n,"counterfactual_coarse_plus_deep_GPU_seconds":5801+110*n}
def synthetic():
    import itertools
    s={0,1};pos={0};unknown={1,2,3}
    vals=[]
    for k in range(4):
        for q in itertools.combinations(unknown,k):
            h=pos|set(q);vals.append(len(s&h)/len(h))
    g={"crop_exact":crop_bounds(1,9,10,7)==(0,7) and crop_bounds(0,10,10,7)==(0,7),
       "stable_ties":order([1,1,0],["b","a","c"])==[1,0,2],
       "portfolio_unique":len(set(interleave([[0,1,2],[1,0,2],[2,0,1]],3)))==3,
       "empty_mask":p95score(np.zeros((2,2),np.uint8),np.zeros((2,2),bool))==-1,
       "partial_identification":np.allclose(recall_bounds(s,pos,unknown),[min(vals),max(vals)]),
       "hypergeom":abs(miss_prob(10,3,1)-.7)<1e-12 and upper(10,10)==0,
       "local_order_reversal_fixture":share(np.array([[255,0]],np.uint8),np.ones((1,2),bool))==.5}
    return {"gates":{k:bool(v) for k,v in g.items()},"pass":all(g.values())}
def analyze(root,gemini):
    t=time.monotonic()
    receipt,source_bytes=verify_source(root);p=root/ROOT
    if receipt["windows.json"]["blob"]!="d6a337f299fa5b893dbc84f6de23542adbbd83a6" or receipt["README.json"]["blob"]!="011130abd6d701cea1d06b94c50ce276ffe9a499":raise ValueError("Frozen manifest mismatch")
    readme=json.loads((p/"README.json").read_text())
    rows=sorted(json.loads((p/"windows.json").read_text()),key=lambda r:r["rank"])
    if len(rows)!=149 or [r["rank"] for r in rows]!=list(range(1,150)):raise ValueError("149 exact ranked windows required")
    ids=[r["window"] for r in rows]
    if len(set(ids))!=149:raise ValueError("Duplicate window")
    if any(not math.isfinite(float(r["valid_cm2"])) or r["valid_cm2"]<=0 for r in rows):raise ValueError("Bad area")
    winds=sorted({r["winding"] for r in rows})
    if len(winds)!=60:raise ValueError("Winding count")
    deep=[]
    for x in (p/"maps_s96").glob("red__PHerc0211_pw_*_stdesc_z20.png"):
        m=re.fullmatch(r"red__PHerc0211_pw_(fw\d+_x\d+)_stdesc_z20\.png",x.name)
        if not m:raise ValueError("Deep filename")
        deep.append(m.group(1))
    reviewed={i for i,k in enumerate(ids) if k in deep}
    if len(deep)!=40 or len(reviewed)!=40:raise ValueError("Deep population mismatch")
    rank_review={rows[i]["rank"] for i in reviewed}
    if rank_review!=set(range(1,26))|RANDOM_RANKS:raise ValueError("Declared deep reviewed rank mismatch")
    rng_replay=set(int(x) for x in np.random.default_rng(2110).choice(np.arange(26,150),15,replace=False))
    if blob(gemini.read_bytes())!=GEMINI:raise ValueError("Gemini blob mismatch")
    sp=importlib.util.spec_from_file_location("gemini_stroke076",gemini);mod=importlib.util.module_from_spec(sp);sys.modules[sp.name]=mod;sp.loader.exec_module(mod)
    x95=np.full(len(rows),-1.,float);stroke=x95.copy();max95=x95.copy()
    win_valid=np.zeros(len(rows),int);area=np.zeros(len(rows),float);mask_resizes=0;paired=[];paired_windows=0;local_opp=0;local_strong=0;global_strong=0;global_direction={"asc":0,"desc":0,"tie":0};both_strong=0
    expected_asc={f"fw{n:03d}" for n in (14,18,21,30,33,41,47,49,52)}
    asc_set={x.name.split("__")[0] for x in (p/"raw20").glob("fw*__s96asc.png")}
    desc_set={x.name.split("__")[0] for x in (p/"raw20").glob("fw*__s96desc.png")}
    if desc_set!=set(winds):raise ValueError("Desc inventory mismatch")
    if not asc_set<=desc_set:raise ValueError("Unknown opposite winding")
    for w in winds:
        m=json.loads((p/"raw20"/f"{w}_render_meta.json").read_text())
        W,H=m["canvas_wh"]
        if m["voxel_size_um"]!=9.362 or m["pixel_size_um"]!=9.362:raise ValueError("Native scale mismatch")
        a=image(p/"raw20"/f"{w}__s96desc.png")
        mask,r1=aligned_mask(image(p/"maps_s96"/f"{w}_valid_mask20.png"),a.shape)
        par,r2=aligned_mask(image(p/"maps_s96"/f"{w}_parallel_mask20.png"),a.shape)
        mask_resizes+=int(r1)+int(r2);support=mask&par
        other=image(p/"raw20"/f"{w}__s96asc.png") if w in asc_set else None
        if other is not None and other.shape!=a.shape:raise ValueError("Paired shape mismatch")
        if other is not None:
            shd,sha0=share(a,support),share(other,support)
            if shd is None:global_order="tie"
            else:global_order="asc" if sha0>shd else ("desc" if shd>sha0 else "tie")
            global_direction[global_order]+=1
            if shd is not None and abs(shd-sha0)>=.005 and max(shd,sha0)>=2*min(shd,sha0):global_strong+=1
            paired.append(w)
        local_directions=set()
        for i,row in enumerate(rows):
            if row["winding"]!=w:continue
            lo,hi=crop_bounds(int(row["x0"]),int(row["x1"]),int(W),a.shape[1])
            s=support[:,lo:hi];d=a[:,lo:hi]
            win_valid[i]=int(s.sum());area[i]=s.sum()*(.05**2)/100.
            x95[i]=p95score(d,s);stroke[i]=strokes(d,s,mod.component_strokeness);max95[i]=x95[i]
            if other is not None:
                o=other[:,lo:hi];max95[i]=max(x95[i],p95score(o,s));paired_windows+=1
                ds,as_=share(d,s),share(o,s)
                if ds is None:continue
                if abs(ds-as_)>=.005 and max(ds,as_)>=2*min(ds,as_):
                    local_strong+=1
                    direction="asc" if as_>ds else "desc";local_directions.add(direction)
                    if global_order!="tie" and direction!=global_order:local_opp+=1
        if local_directions=={"asc","desc"}:both_strong+=1
    N=len(rows)
    def declared(key):
        return [float(r[key]) if isinstance(r.get(key),(int,float)) and math.isfinite(r[key]) else -1e10 for r in rows]
    orders={"L":list(range(N)),"H":order(declared("hot"),ids),"G":order(declared("glyph"),ids),"R":order(declared("row"),ids),"P95":order(x95,ids),"S":order(stroke,ids),"U":uniform(N,7601)}
    orders["F"]=interleave([orders[k] for k in ("L","P95","S","U")],N)
    if any(sorted(x)!=list(range(N)) for x in orders.values()):raise ValueError("Incomplete order")
    # No target labels passed to scoring/ranking functions. Decode published ranks only here.
    known={r-1 for r in KNOWN};flag={r-1 for r in FLAG}
    if not known|flag<=reviewed:raise ValueError("Reported hits outside reviewed")
    result={}
    for K in BUDGET:
        result[str(K)]={k:metrics(a[:K],rows,reviewed,known,flag) for k,a in orders.items()}
        D=interleave([orders[k] for k in ("S","P95","U")],K,orders["L"][:math.ceil(.75*K)])
        result[str(K)]["D"]=metrics(D,rows,reviewed,known,flag)
        u=[metrics(uniform(N,seed)[:K],rows,reviewed,known,flag) for seed in range(7602,7730)]
        result[str(K)]["random128"]={"mean_confirmed":float(np.mean([r["confirmed_known"] for r in u])),
          "mean_reported_hits":float(np.mean([r["reported_hits"] for r in u])),
          "all_known_probability_empirical":float(np.mean([r["confirmed_known"]==2 for r in u])),
          "all_reported_hits_probability_empirical":float(np.mean([r["reported_hits"]==3 for r in u]))}
    cap=upper(124,15);ks=set(orders["P95"][:25]);kt=set(order(max95,ids)[:25])
    observed=result["25"];f=observed["F"];l=observed["L"]
    rel=np.abs(area-np.array([r["valid_cm2"] for r in rows]))/np.array([r["valid_cm2"] for r in rows])
    # Include only aggregate data; original raw file list is independently hashed.
    return {"source_bytes":source_bytes,"source_files":len(receipt),"source_manifest_sha256":sha(json.dumps(receipt,sort_keys=True,separators=(",",":")).encode()),
        "gemini_blob":GEMINI,"population":{"windings":len(winds),"windows":N,"reviewed":len(reviewed),"unreviewed":N-len(reviewed),"reported_hit_windows":3,"known_positive_windows":2,"unconfirmed_flag_windows":1,
          "reported_chart_area_cm2":sum(r["valid_cm2"] for r in rows),"recomputed_downscaled_parallel_area_cm2":float(area.sum()),
          "area_relative_error_q50_q95":np.quantile(rel,[.5,.95]).tolist(),"empty_support_windows":int((win_valid==0).sum()),"mask_resize_count":mask_resizes,
          "all_desc_maps":len(desc_set),"actual_asc_maps":len(asc_set),"declared_random_asc":len(expected_asc),"declared_random_asc_present":len(expected_asc&asc_set),
          "extra_asc_maps":len(asc_set-expected_asc),"missing_declared_asc_maps":len(expected_asc-asc_set),"reported_deep_selection_matches_files":True,
          "rng2110_rank_order_replay_matches":rng_replay==RANDOM_RANKS},
        "budget_curves":result,"primary_gate":{"both_known_retained":f["confirmed_known"]==2,"no_worse_than_line":f["confirmed_known"]>=l["confirmed_known"],"unconfirmed_flag_retained_descriptive":f["unconfirmed_flag_retained"],"scientific_promotion":False},
        "survey":{"stratum_N":124,"sample_n":15,"reviewed_hits":0,"unreviewed":109,"exact_upper95_reported_lower_stratum_hits":cap,
          "upper95_fraction_lower_stratum":cap/124,"whole_band_upper95_reported_hit_windows":3+cap,
          "whole_band_deterministic_reported_hit_range":[3,112],"probability_zero_if_true_misses":{str(m):miss_prob(124,15,m) for m in (1,3,5,10)},
          "min_n_to_exclude_misses_zero_events":{str(m):next(n for n in range(1,125) if miss_prob(124,n,m)<.05) for m in (1,5,10)},
          "scope":"Conditional on reported SRS without replacement and fixed reviewer-hit definition; no independent randomization proof, no latent ink inference."},
        "paired_order":{"paired_charts":len(paired),"paired_windows":paired_windows,"global_preference":global_direction,"global_strong":global_strong,
          "strong_local_windows":local_strong,"strong_local_opposite_global":local_opp,"charts_both_strong_local_orders":both_strong,
          "P95_top25_replaced_by_max_order_partial":len(ks-kt),"scope":"Score preference only, different signed slabs; missing opposite maps unknown, not validated orientation."},
        "cost":{"reported_GPU_coarse_seconds":5801,"reported_GPU_deep40_seconds":4400,"counterfactual_all149_deep_only_seconds":149*110,
          "counterfactual_all149_coarse_plus_deep_seconds":5801+149*110,"actual_GPU_coarse_plus40_deep_seconds":10201,
          "estimated_deep_reduction_fraction":1-40/149,"estimated_vs_deep_only_GPU_reduction_fraction":1-10201/(149*110),
          "reported_windowing_CPU_seconds":1112,"scope":"Equal110s/window approximation; historical coarse and deep sum not total end-to-end cost. No measured exhaustive run."},
        "analysis_seconds":time.monotonic()-t}
def run(a):
    start=time.monotonic();r={"schema":"adaptive.fitted_band_funnel.v1","source_commit":PIN,
       "protocol_sha256":sha((HERE/"protocol.md").read_bytes()),"script_sha256":sha(pathlib.Path(__file__).read_bytes()),"complete":False}
    try:
        r["synthetic"]=synthetic()
        if not r["synthetic"]["pass"]:raise ValueError("Synthetic gate failure")
        r.update(analyze(a.claude,a.gemini));r["complete"]=True
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:350]}"
    r["wall_seconds"]=time.monotonic()-start;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
    if not r["complete"]:raise SystemExit(1)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("claude","gemini","output"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
