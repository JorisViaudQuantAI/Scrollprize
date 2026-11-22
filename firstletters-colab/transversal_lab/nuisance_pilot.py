#!/usr/bin/env python3
"""Fixed local-standardization pilot on released 306B scores, never unread CT.

The source-defined rows are localization targets, not ink masks. All outputs
stay outside Git. A changed score representation receives no old calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

import cv2
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter, minimum_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "line_review"))
from review import Config, outside_git, rank_map, validate_inputs, work_grid

PROTOCOL_PATH = HERE / "nuisance_protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
INVENTORY_PATH = HERE.parent / "line_controls/inventory.json"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def grid_context(score, valid, source_spacing):
    """Downsample once; every view and rank then shares the same grid/mask."""
    grid, observed, spacing_mm = work_grid(score, valid, source_spacing, Config())
    sigma = np.array([PROTOCOL["local_sigma_mm"] / d for d in spacing_mm])
    radii = np.ceil(PROTOCOL["gaussian_truncate"] * sigma).astype(int)
    common = minimum_filter(observed.astype(np.uint8), size=tuple(2*radii+1),
                            mode="constant", cval=0).astype(bool)
    # Work-grid numerical values at missing cells must not leak into valid
    # kernels. The erosion above excludes every scored kernel touching one.
    filled = np.where(observed, grid, 0).astype(np.float32)
    return filled, observed, common, spacing_mm, sigma, radii


def standardized_views(grid, common, sigma):
    if not common.any():
        return None, {"status": "unassessable_no_complete_kernel_support"}
    floor = PROTOCOL["standard_deviation_floor"]
    values = grid[common].astype(np.float64)
    global_mean, global_sd = float(values.mean()), float(values.std())
    local_mean = gaussian_filter(grid, sigma=sigma, truncate=PROTOCOL["gaussian_truncate"],
                                 mode="constant", cval=0)
    second = gaussian_filter(grid*grid, sigma=sigma, truncate=PROTOCOL["gaussian_truncate"],
                             mode="constant", cval=0)
    local_sd = np.sqrt(np.maximum(second-local_mean*local_mean, 0))
    global_q = np.clip((grid-global_mean)/max(global_sd, floor)/4, 0, 1).astype(np.float32)
    local_q = np.clip((grid-local_mean)/np.maximum(local_sd, floor)/4, 0, 1).astype(np.float32)
    return dict(raw=grid, global_standardized=global_q, local_standardized=local_q), {
        "global_mean": global_mean, "global_population_sd": global_sd,
        "global_sd_after_floor": max(global_sd, floor),
        "local_sd_quantiles_on_common": np.quantile(local_sd[common], [0,.25,.5,.75,1]).tolist(),
        "local_sd_floor_fraction_on_common": float((local_sd[common]<floor).mean())}


def target_descriptor(target, source_shape, work_shape, common):
    """Keep every original target pixel in its denominator after coarsening."""
    h, w = work_shape
    sy, sx = h/source_shape[0], w/source_shape[1]
    x0,y0,x1,y1 = target["box_xyxy"]
    xa,xb = int(np.floor(x0)),int(np.ceil(x1))
    ya,yb = int(np.floor(y0)),int(np.ceil(y1))
    if not (0<=xa<xb<=source_shape[1] and 0<=ya<yb<=source_shape[0]):
        raise ValueError("Frozen target must lie inside its original source")
    xs = np.floor((np.arange(xa,xb)+.5)*sx).astype(int)
    ys = np.floor((np.arange(ya,yb)+.5)*sy).astype(int)
    wx = np.bincount(xs, minlength=w); wy = np.bincount(ys, minlength=h)
    lx,rx,ty,by = int(xs.min()),int(xs.max()+1),int(ys.min()),int(ys.max()+1)
    weights = wy[ty:by,None]*wx[None,lx:rx]
    cy_native = int(round((y0+y1)/2))
    cy = int(np.floor((cy_native+.5)*sy))
    line_weights = wx[lx:rx]
    line_valid = common[cy,lx:rx]
    box_valid = common[ty:by,lx:rx]
    line_n = int(line_weights[line_valid].sum())
    box_n = int(weights[box_valid].sum())
    record = dict(id=target["id"], source_box_xyxy=target["box_xyxy"],
        work_box_xyxy=[x0*sx,y0*sy,x1*sx,y1*sy],
        full_source_centreline_pixels=xb-xa, full_source_box_pixels=(xb-xa)*(yb-ya),
        observed_source_centreline_pixels=line_n, observed_source_box_pixels=box_n,
        source_centreline_support_fraction=line_n/(xb-xa),
        source_box_support_fraction=box_n/((xb-xa)*(yb-ya)),
        status="observed" if line_n else "unassessable_centreline",
        policy="Original pixel-centre samples mapped to work cells; proposal polygons rasterized on work grid, not an exact replay of prior native-grid boundary rounding")
    return record,(lx,rx,ty,by,cy,weights,line_weights,box_valid,line_valid)


def row_metrics(proposals, targets, source_shape, common):
    described=[target_descriptor(t, source_shape, common.shape, common) for t in targets]
    rows=[dict(r, top_k={}, individual=[]) for r,_ in described]
    one=np.zeros(common.shape,np.uint8);union=np.zeros_like(one)

    def coverage(mask, desc):
        lx,rx,ty,by,cy,weights,line_weights,box_valid,line_valid=desc
        line_n=int(line_weights[line_valid].sum());box_n=int(weights[box_valid].sum())
        lc=int(line_weights[line_valid & (mask[cy,lx:rx]>0)].sum())
        bc=int(weights[box_valid & (mask[ty:by,lx:rx]>0)].sum())
        return dict(centreline_coverage=lc/line_n if line_n else None,
                    box_coverage=bc/box_n if box_n else None,
                    covered_source_centreline_pixels=lc,covered_source_box_pixels=bc)

    for rank,p in enumerate(proposals,1):
        poly=np.rint(p["band_polygon_xy"]).astype(np.int32)
        one.fill(0);cv2.fillPoly(one,[poly],1);cv2.fillPoly(union,[poly],1)
        for r,(_,d) in zip(rows,described):
            r["individual"].append(dict(rank=rank,**coverage(one,d)))
            if rank in (1,3,5,10):r["top_k"][str(rank)]=coverage(union,d)
    for r,(_,d) in zip(rows,described):
        for k in (1,3,5,10):r["top_k"].setdefault(str(k),coverage(union,d))
        measured=[x for x in r["individual"] if x["centreline_coverage"] is not None]
        r["first_individual_half_rank"]=next((x["rank"] for x in measured if x["centreline_coverage"]>=.5),None)
        r["best_individual_centreline_coverage"]=max((x["centreline_coverage"] for x in measured),default=None)
    return rows


def localization_gate(views, role):
    if role=="matched-depth-shuffle-process-control":
        return {"status":"not_applicable_process_intervention","passed":None}
    if not views:return {"status":"unassessable_no_complete_kernel_support","passed":False}
    byname={v["name"]:v for v in views}
    targets={name:{r["id"]:r for r in v["rows"]} for name,v in byname.items()}
    weak,strong=PROTOCOL["weak_row_id"],PROTOCOL["strong_row_id"]
    gate=PROTOCOL["localization_continuation_gate"]
    local=byname["local_standardized"];lr=targets["local_standardized"][weak]["first_individual_half_rank"]
    tests={"adequate_original_centreline_support":all(r["source_centreline_support_fraction"]>=PROTOCOL["minimum_original_centreline_support_fraction"] for t in targets.values() for r in t.values()),
           "weak_row_rank_at_most_five":lr is not None and lr<=gate["weak_first_individual_half_coverage_rank_at_most"],
           "no_search_truncation":not any(v["search_truncated"] for v in views)}
    for baseline in ("raw","global_standardized"):
        br=targets[baseline][weak]["first_individual_half_rank"]
        tests["weak_rank_improves_vs_"+baseline]=lr is not None and (br is None or lr<br)
        lc=targets["local_standardized"][strong]["top_k"]["5"]["centreline_coverage"]
        bc=targets[baseline][strong]["top_k"]["5"]["centreline_coverage"]
        tests["strong_preserved_vs_"+baseline]=lc is not None and bc is not None and lc>=bc-gate["strong_top5_centreline_coverage_allowed_loss_vs_each_baseline"]
        tests["bounded_proposals_vs_"+baseline]=local["proposal_count"]<=gate["maximum_proposals_relative_to_each_baseline"]*max(1,byname[baseline]["proposal_count"])
    return dict(status="continuation_gate_only",passed=all(tests.values()),checks=tests,
                interpretation="A pass would justify a held-out-control follow-up, not deployment, text accuracy, or a finding")


def plot_views(views, arrays, common, case, spacing_mm, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    width=common.shape[1]*spacing_mm[1];height=common.shape[0]*spacing_mm[0]
    fig,axes=plt.subplots(3,3,figsize=(18,12),gridspec_kw={"height_ratios":[height,8,8]},constrained_layout=True)
    cmap=plt.get_cmap("gray").copy();cmap.set_bad("#4c2355")
    for col,v in enumerate(views):
        image=np.ma.array(arrays[v["name"]],mask=~common)
        for row,ax in enumerate(axes[:,col]):
            ax.set_facecolor("#4c2355");ax.imshow(image,cmap=cmap,vmin=0,vmax=1,
                extent=(0,width,height,0),interpolation="nearest")
            ax.set_xlim(0,width);ax.set_aspect("equal",adjustable="box")
            ax.set_xlabel("Published chart x, mm");ax.set_ylabel("Published chart y, mm")
            if row==0:
                ax.set_ylim(height,0);ax.set_title(v["name"].replace("_"," ")+f"; {v['proposal_count']} proposals")
                for rank,p in enumerate(v["proposals"][:5],1):
                    xy=np.asarray(p["band_polygon_xy"])*np.array(spacing_mm)[::-1]
                    ax.add_patch(Polygon(xy,fill=False,color="#ff8b35",linewidth=.7))
                    ax.text(*xy.mean(0),str(rank),color="#ff8b35",fontsize=8)
            else:
                target=case["targets"][row-1];cy=(target["box_xyxy"][1]+target["box_xyxy"][3])/2*case["spacing_um_yx"][0]/1000
                ax.set_ylim(cy+4,cy-4);ax.set_title(f"Published row {row}, fixed ±4 mm; no overlays",fontsize=9)
    fig.suptitle(case["id"]+" — raw scores versus fixed standardized display signals\nPurple = excluded/unknown; distinct numerical scales, not three calibrated probabilities",fontsize=11)
    fig.savefig(output,dpi=110);plt.close(fig)


def run_case(case, source_root, output):
    start=time.perf_counter();folder=source_root/"cases"/case["id"]
    paths={"score":folder/"score.npy","mask":folder/"validity.npy"}
    expected={"score":case["score_sha256"],"mask":case["mask_sha256"]}
    for k,p in paths.items():
        if sha(p)!=expected[k]:raise ValueError("Pinned source hash mismatch: "+case["id"]+"/"+k)
    score=np.load(paths["score"],allow_pickle=False);valid=np.load(paths["mask"],allow_pickle=False)
    validate_inputs(score,valid,case["spacing_um_yx"],"model_score")
    if list(score.shape)!=case["shape"]:raise ValueError("Unexpected source grid")
    grid,observed,common,spacing_mm,sigma,radii=grid_context(score,valid,case["spacing_um_yx"])
    del score,valid
    arrays,stats=standardized_views(grid,common,sigma)
    out=output/case["id"];out.mkdir()
    np.save(out/"common_valid.npy",common,allow_pickle=False)
    descriptor=[target_descriptor(t,case["shape"],common.shape,common)[0] for t in case["targets"]]
    result=dict(id=case["id"],role=case["role"],source_hashes=expected,source_shape=case["shape"],
        work_shape_yx=list(grid.shape),effective_work_spacing_mm_yx=list(spacing_mm),
        original_source_valid_pixels=case["valid_pixels"],work_observed_pixels=int(observed.sum()),
        common_complete_kernel_pixels=int(common.sum()),common_fraction_of_observed_work=float(common.sum()/max(1,observed.sum())),
        sigma_work_pixels=sigma.tolist(),kernel_radii_work_pixels=radii.tolist(),
        source_target_support=descriptor,statistics=stats,views=[])
    if arrays is not None:
        for name in PROTOCOL["views"]:
            p=arrays[name]
            report=rank_map(p,common,[1000*d for d in spacing_mm],"model_score" if name=="raw" else "display_signal",Config())
            if report["work_shape_yx"]!=list(grid.shape):raise ValueError("Unexpected second spatial resampling")
            rows=row_metrics(report["proposals"],case["targets"],case["shape"],common)
            np.save(out/(name+".npy"),np.where(common,p,np.nan).astype(np.float32),allow_pickle=False)
            result["views"].append(dict(name=name,proposal_count=len(report["proposals"]),search_truncated=report["search_truncated"],
                seconds=report["seconds"],score_quantiles=np.quantile(p[common],[0,.25,.5,.75,.95,.99,1]).tolist(),
                rows=rows,proposals=report["proposals"],ranker_trials=report["trials"],
                ranker_work_shape_yx=report["work_shape_yx"],ranker_work_spacing_mm_yx=report["work_spacing_mm_yx"]))
        plot_views(result["views"],arrays,common,case,spacing_mm,out/"comparison.png")
    result["localization_gate"]=localization_gate(result["views"],case["role"])
    for k,p in paths.items():
        if sha(p)!=expected[k]:raise ValueError("Source changed during pilot")
    result["status"]="completed";result["seconds_total"]=time.perf_counter()-start
    result["output_sha256"]={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    save_json(out/"report.json",result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--public-controls-root",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args();outside_git(args.output_dir)
    if args.output_dir.exists():raise FileExistsError("Use a new external directory")
    inventory=json.loads(INVENTORY_PATH.read_text());cases={c["id"]:c for c in inventory["cases"]}
    cv2.setNumThreads(4);args.output_dir.mkdir(parents=True)
    signature=dict(protocol_sha256=sha(PROTOCOL_PATH),script_sha256=sha(__file__),
        ranker_sha256=sha(HERE.parent/"line_review/review.py"),inventory_sha256=sha(INVENTORY_PATH),
        runtime={"python":sys.version.split()[0],"numpy":np.__version__,"scipy":scipy.__version__,"opencv":cv2.__version__},
        config=asdict(Config()),started_utc=datetime.now(timezone.utc).isoformat(),network_forward_calls=0)
    save_json(args.output_dir/"run_started.json",signature)
    def expire(_a,_b):raise TimeoutError("Frozen 180-second input-case budget exhausted")
    old=signal.signal(signal.SIGALRM,expire);results=[]
    try:
        for cid in PROTOCOL["cases"]:
            signal.setitimer(signal.ITIMER_REAL,PROTOCOL["time_limit_seconds_per_input_case"])
            try:r=run_case(cases[cid],args.public_controls_root,args.output_dir)
            except TimeoutError:
                results.append({"id":cid,"status":"incomplete_timeout"});break
            finally:signal.setitimer(signal.ITIMER_REAL,0)
            results.append(r)
            print(json.dumps({"case":cid,"common_work_pixels":r["common_complete_kernel_pixels"],
                "views":[{"name":v["name"],"proposals":v["proposal_count"]} for v in r["views"]],
                "localization_gate":r["localization_gate"]}),flush=True)
    finally:signal.signal(signal.SIGALRM,old)
    result=dict(signature,status="completed" if len(results)==len(PROTOCOL["cases"]) and all(r["status"]=="completed" for r in results) else "incomplete",cases=results,
        operational_gate="not_established_single_previously_inspected_published_source",
        limitations=["Coarse published rows are localization targets, not ink truth", "All unlabelled pixels remain unknown", "No new model inference or independent detector", "No population false-positive estimate", "Local standardization can enhance crack and blob rims", "Any continuation gate requires independent controls before operational use"])
    save_json(args.output_dir/"result.json",result)
    return 0 if result["status"]=="completed" else 3


if __name__=="__main__":raise SystemExit(main())
