#!/usr/bin/env python3
"""Q33 native-CT rigid-tilt stress, with prospectively fixed physical labels.

prepare never imports torch. Identity must reproduce the author's released
uint8 inputs exactly; infer checks the historical identity predictions before
submitting any stress forward. Nothing is trained or selected by model score.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "studio_control"))
from run_public_control import (BASE, CONTROL_ID, DATA_REVISION, DATASET, FILES,
    SOURCE_COMMIT, WEIGHTS_SHA256, WEIGHTS_URL, metrics, raw_crop, sha256,
    verified_file, write_json)
from assets import acquire_geometry
from geometry import rotate_patch, inverse_readback
from native import load_surface, SupportedNative

ARMS = [dict(name="identity", axis="x", angle_deg=0)] + [
    dict(name=f"{axis}_{'minus' if angle < 0 else 'plus'}{abs(angle)}", axis=axis, angle_deg=angle)
    for axis in ("x", "y") for angle in (-20, -10, -5, 5, 10, 20)]
HISTORICAL = HERE.parent / "studio_scale_stress/evidence/scored_arrays.npz"
HISTORICAL_SHA256 = "04b107ef27c8a17cec25fbc90939d4ccb1b36190240a74f238fa75cf5a860e08"


def code_hashes():
    paths = [HERE / n for n in ("run.py", "native.py", "assets.py", "geometry.py", "test_geometry.py")]
    paths += [HERE.parent/"studio_control/run_public_control.py", ROOT/"gemini_lane/stream_snap_render.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def verify_source(source_repo):
    actual = subprocess.check_output(["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != SOURCE_COMMIT:
        raise ValueError("Unexpected upstream revision")
    paths = ["studio-multiscroll-cpt-20261023/hf/model.py", "studio-multiscroll-cpt-20261023/hf/inference.py",
             "donor-labels-20260906/expansion/volume_sampling.py"]
    subprocess.run(["git", "-C", str(source_repo), "diff", "--exit-code", "HEAD", "--", *paths], check=True)
    return {p: sha256(source_repo/p) for p in paths}


def select_controls(labels, coordinates):
    groups = {"mixed": [], "ink": [], "background": []}
    for index, (yy, xx) in enumerate(coordinates):
        y, x = int(yy), int(xx)
        state = labels[y+48:y+80, x+48:x+80]
        ink, bg = int((state == 2).sum()), int((state == 1).sum())
        known = ink + bg
        if known < 512:
            continue
        category = ("mixed" if known <= 10*ink <= 9*known else
                    "ink" if ink == known else "background" if bg == known else None)
        if category:
            groups[category].append(dict(index=index, y=y, x=x, ink=ink, background=bg, category=category))
    selected, pure_order = list(groups["mixed"]), {}
    for category in ("ink", "background"):
        remaining = sorted(groups[category], key=lambda r: (r["y"], r["x"], r["index"]))
        if len(remaining) < 3:
            raise ValueError("Insufficient pure controls")
        chosen = [remaining.pop(0)]
        while len(chosen) < 3:
            distance = lambda r: min((r["y"]-s["y"])**2 + (r["x"]-s["x"])**2 for s in chosen)
            best = min(remaining, key=lambda r: (-distance(r), r["y"], r["x"], r["index"]))
            chosen.append(best)
            remaining.remove(best)
        pure_order[category] = [r["index"] for r in chosen]
        selected.extend(chosen)
    selected.sort(key=lambda r: r["index"])
    return selected, pure_order


def historical(selected, states):
    if sha256(HISTORICAL) != HISTORICAL_SHA256:
        raise ValueError("Historical archive changed")
    with np.load(HISTORICAL, allow_pickle=False) as old:
        ids = [int(np.flatnonzero(old["author_indices"] == r["index"])[0]) for r in selected]
        identity_index = int(np.flatnonzero(old["condition_names"] == "identity")[0])
        if not np.array_equal(old["states"][ids], states):
            raise ValueError("Historical labels differ")
        return old["probabilities"][identity_index, ids].copy()


def prepare(args):
    if args.prepared.exists():
        raise ValueError("Use a fresh prepared directory")
    started = time.monotonic()
    source = verify_source(args.source_repo)
    for name, (digest, size) in FILES.items():
        verified_file(BASE+name, args.cache/name, digest, size)
    verified_file(WEIGHTS_URL, args.checkpoint, WEIGHTS_SHA256, 893443724)
    labels = np.load(args.cache/"labels.npy", allow_pickle=False)
    coordinates = np.load(args.cache/"coordinates.npy", allow_pickle=False)
    ct = np.load(args.cache/"ct-tiles.npy", mmap_mode="r", allow_pickle=False)
    tiles = np.load(args.cache/"tiles.npy", allow_pickle=False)
    selected, pure_order = select_controls(labels, coordinates)
    states = np.stack([labels[r["y"]+48:r["y"]+80,r["x"]+48:r["x"]+80] for r in selected])
    if len(selected) != 17 or len({(r["y"],r["x"]) for r in selected}) != 17:
        raise ValueError("Frozen label-only selection changed")
    if any(max(abs(a["y"]-b["y"]),abs(a["x"]-b["x"])) < 32
           for i,a in enumerate(selected) for b in selected[i+1:]):
        raise ValueError("Overlapping evaluation cores")
    baseline = historical(selected, states)
    mesh, recipe, acquisition = acquire_geometry(args.geometry_cache)
    surface = load_surface(args.source_repo, mesh, recipe)
    queries, depth, valid, residual, rotations, pivots, diagnostics = [], [], [], [], [], [], []
    for arm in ARMS:
        arm_queries, arm_depth, arm_valid, arm_residual, arm_rotations, arm_pivots = [],[],[],[],[],[]
        for item in selected:
            patch = rotate_patch(surface, item["y"], item["x"], arm["angle_deg"], axis=arm["axis"],
                                 pitch_um=recipe["xy_pitch_um"], offsets_um=recipe["offsets_um"])
            if not patch["mesh_valid"].all():
                raise ValueError("Full model patch lacks geometry; no core replacement allowed")
            inverse = inverse_readback(surface, item["y"], item["x"], patch["rotation"], patch["pivot_xyz"],
                                       pitch_um=recipe["xy_pitch_um"])
            arm_queries.append(inverse["query_yx"]); arm_depth.append(inverse["depth_vox"])
            arm_valid.append(inverse["valid"]); arm_residual.append(inverse["residual_vox"])
            arm_rotations.append(patch["rotation"]); arm_pivots.append(patch["pivot_xyz"])
            diagnostics.append(dict(arm=arm["name"],author_index=item["index"],**inverse["diagnostics"]))
        queries.append(arm_queries); depth.append(arm_depth); valid.append(arm_valid)
        residual.append(arm_residual); rotations.append(arm_rotations); pivots.append(arm_pivots)
    queries, depth, valid, residual = map(np.asarray, (queries,depth,valid,residual))
    common = valid.all(axis=0)
    scored_states = np.where(common, states, 0).astype(np.uint8)
    retention = (scored_states > 0).sum(axis=(1,2))/(states > 0).sum(axis=(1,2))
    overall_retention = float((scored_states > 0).sum()/(states > 0).sum())
    minimum_core_retention = .95 if args.geometry_policy == "initial95" else .90
    if overall_retention < .95 or (retention < minimum_core_retention).any():
        args.prepared.mkdir(parents=True)
        write_json(args.prepared/"failed_geometry_gate.json",dict(
            geometry_policy=args.geometry_policy,stage="Geometry only, before native patch batch preparation or any model load",
            code_hashes=code_hashes(),controls=selected,per_core_known_retention=retention.tolist(),known_retention=overall_retention,
            per_arm_core=diagnostics,minimum_core_retention=minimum_core_retention,minimum_global_retention=.95,
            model_forwards=0,failed=True))
        raise ValueError(f"Geometry retention gate failed; no inference, no replacement: {retention.tolist()}")
    print("GEOMETRY_GATE",len(selected),"cores",overall_retention,"min_core",float(retention.min()),flush=True)
    args.prepared.mkdir(parents=True)
    np.save(args.prepared/"original_states.npy",states,allow_pickle=False)
    np.save(args.prepared/"states.npy",scored_states,allow_pickle=False)
    np.save(args.prepared/"historical_identity.npy",baseline,allow_pickle=False)
    np.savez_compressed(args.prepared/"mapping.npz",query_yx=queries,depth_vox=depth,valid=valid,
                        residual_vox=residual,common_mask=common,rotations=np.asarray(rotations),pivots=np.asarray(pivots))
    write_json(args.prepared/"geometry_diagnostics.json",dict(per_arm_core=diagnostics,per_core_known_retention=retention.tolist(),
        known_retention=overall_retention,excluded_known_pixels=int(((states>0)&~common).sum()),
        original_ink=int((states==2).sum()),original_background=int((states==1).sum()),
        common_ink=int((scored_states==2).sum()),common_background=int((scored_states==1).sum())))
    volume = SupportedNative(recipe["url"], args.native_cache)
    inputs = np.lib.format.open_memmap(args.prepared/"inputs.npy",mode="w+",dtype=np.uint8,
                                      shape=(len(ARMS),len(selected),16,128,128))
    hashes, identity_records = [], []
    for ai,arm in enumerate(ARMS):
        for pi,item in enumerate(selected):
            patch = rotate_patch(surface,item["y"],item["x"],arm["angle_deg"],axis=arm["axis"],
                                 pitch_um=recipe["xy_pitch_um"],offsets_um=recipe["offsets_um"])
            raw = np.rint(volume.sample_exact(patch["coords"])).clip(0,255).astype(np.uint8)
            if ai == 0:
                reference = raw_crop(ct,tiles,item["y"],item["x"])
                if not np.array_equal(raw,reference):
                    raise ValueError(f"Native reconstruction mismatch at author index {item['index']}")
                identity_records.append(dict(author_index=item["index"],bytes=int(raw.nbytes),exact_equal=True))
            inputs[ai,pi] = raw
            hashes.append(dict(arm=arm["name"],author_index=item["index"],sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
        inputs.flush()
        print("PREPARED_ARM",arm["name"],"seconds",round(time.monotonic()-started,2),flush=True)
    write_json(args.prepared/"native_receipt.json",volume.receipt())
    write_json(args.prepared/"geometry_acquisition.json",acquisition)
    write_json(args.prepared/"native_reconstruction.json",dict(all_exact=True,records=identity_records,total_compared_bytes=sum(r["bytes"] for r in identity_records)))
    array_names = ["inputs.npy","states.npy","original_states.npy","historical_identity.npy","mapping.npz"]
    protocol = dict(version="q33-native-oblique-v2",base_commit="2d0d77f673e3e35dce73e60be12249d73e99df53",
        purpose="Measure released detector sensitivity to rigid local 3D sampling-tube tilt on fixed original surface labels",
        scope="Repeatedly evaluated author PHerc0841 donor holdout; integration/sensitivity evidence, not independent legible-text certification",
        source_commit=SOURCE_COMMIT,source_files=source,code_hashes=code_hashes(),dataset=DATASET,data_revision=DATA_REVISION,
        public_control=CONTROL_ID,geometry_recipe=recipe,weights=dict(url=WEIGHTS_URL,sha256=WEIGHTS_SHA256),
        acquisition={n:dict(url=BASE+n,sha256=h,bytes=s) for n,(h,s) in FILES.items()},
        historical_archive_sha256=HISTORICAL_SHA256,arms=ARMS,controls=selected,selected_patch_count=len(selected),pure_maximin_order=pure_order,
        selection="All central32 cores with >=512 known and ink fraction in [.1,.9], plus3 ink-only and3 background-only with >=512 known. Each pure class starts at lexicographically smallest(y,x,index); then maximizes minimum squared XY distance to already selected members of that same class, ties lexicographic. Final order author_index. No post-geometry substitution.",
        geometry_gate=dict(maximum_physical_residual_native_voxels=.01,minimum_known_retention_global=.95,
                           minimum_known_retention_each_core=minimum_core_retention,policy=args.geometry_policy,
                           max_iterations=15,depth_bounds_native_voxels=[-8,7],model_margin_px=16,action="Abort preparation/inference on any failed gate"),
        mask="Intersection of inverse validity of every arm, fixed before model load; original unknown labels remain unknown; identical scored_states for every condition",
        preflight_adaptation="Initial >=95% retention in every core failed in geometry-only preflight. The retained90 policy lowers only per-core coverage to90%, keeps global>=95%, same17 cores/13 arms/residual .01 and fixed common mask. This coverage choice is data-informed before any detector inference, not fully prospective before geometry inspection. No score or model output motivated the change.",
        denominator=dict(ink=int((scored_states==2).sum()),background=int((scored_states==1).sum()),
                         excluded_known=int(((states>0)&~common).sum()),original_unknown=int((states==0).sum())),
        rotation="Rigid world-XYZ right-handed rotation of the complete curved sampling tube about centre reference-surface point and positive projected local chart tangent; both anchors and signed cross(drow,dcol) normals rotate",
        inverse="For original reference surface X=S(u,v), solve S(u',v')+d'n(u',v')=C+R.T(X-C); read detector probability at (u',v'), bilinear. Labels define original surface columns, not known 3D ink voxel locations.",
        interpolation="Native uint8 CT, author's FP32 source coordinates/local-origin subtraction, scipy order1 outputFP32 nearest/prefilterFalse, rint/clip0..255 once. Every positive-weight CT corner must be available. No prepared-stack warp.",
        normalization="FP32 clip[0,200]/255 once; unchanged stored depth order -8..+7, surface depth at index8",
        inference="Pinned released model, CPU FP32 eval batch1; no retraining, ensemble, tiled blending, or best-angle choice",
        identity_gate=dict(native_inputs="Exact byte equality at all17 cores",historical_max_probability_difference=1e-7,
                           historical_max_AP_difference=1e-6,action="Abort all stress forwards if failed"),
        primary=dict(metric="Pooled AP difference versus identity on identical common pixels",material_drop_below=-.05),
        secondary=dict(recall_loss_below=-.10,background_FPR_increase_above=.03,
                       metrics=["ROC AUC","recall@.5","FPR@.5","mixed-core AP changes","per-core support/depth/residual"]),
        interpretation="Thresholds are prospective engineering magnitudes, not statistical significance; no IID-pixel intervals;17 spatially related cores from one region",
        caveats=["Combined surface position, normal direction, depth/context support and resampling perturbation; not isolated detector rotation equivariance",
                 "All scored reference points remain in the frozen depth band; outer128 context may leave the original surface band",
                 "Local inverse convergence/nonfolding does not prove global tube injectivity",
                 "Two arbitrary chart tangent axes are not certified handwriting directions",
                 "No claim about rows of dots, letter readability, unseen-scroll performance or a correctable production angle"],
        runtime=dict(workers=args.workers,threads_per_worker=args.threads,interop_threads=1),
        arrays={n:dict(sha256=sha256(args.prepared/n),bytes=(args.prepared/n).stat().st_size) for n in array_names},
        receipts={n:sha256(args.prepared/n) for n in ("native_receipt.json","geometry_acquisition.json","geometry_diagnostics.json","native_reconstruction.json")},
        input_hashes=hashes,preparation_seconds=time.monotonic()-started)
    write_json(args.prepared/"protocol.json",protocol)
    print("PROTOCOL_FROZEN",sha256(args.prepared/"protocol.json"),flush=True)


_MODEL = _INPUTS = None


def init_worker(source_repo,checkpoint,prepared,threads):
    global _MODEL,_INPUTS
    import torch
    torch.set_num_threads(threads); torch.set_num_interop_threads(1)
    sys.path.insert(0,str(Path(source_repo)/"studio-multiscroll-cpt-20261023/hf"))
    from inference import load_model
    _MODEL = load_model(Path(checkpoint),"cpu")
    _INPUTS = np.load(Path(prepared)/"inputs.npy",mmap_mode="r",allow_pickle=False)


def forward(task):
    import torch
    ai,pi = task
    x = torch.from_numpy(np.array(_INPUTS[ai,pi],dtype=np.float32,copy=True)[None,None]).clamp_(0,200).div_(255)
    started = time.monotonic()
    with torch.inference_mode():
        probability = _MODEL(x)[0,0].float().sigmoid().numpy()
    if probability.shape != (128,128) or not np.isfinite(probability).all():
        raise ValueError("Invalid model output")
    return ai,pi,probability,time.monotonic()-started,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,os.getpid()


def checked_protocol(args):
    protocol = json.loads((args.prepared/"protocol.json").read_text())
    if protocol["code_hashes"] != code_hashes():
        raise ValueError("Code changed after protocol freeze")
    for name,meta in protocol["arrays"].items():
        if sha256(args.prepared/name) != meta["sha256"]:
            raise ValueError("Prepared array changed: "+name)
    return protocol


def infer(args):
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    protocol = checked_protocol(args)
    if verify_source(args.source_repo) != protocol["source_files"] or sha256(args.checkpoint) != WEIGHTS_SHA256:
        raise ValueError("Source/checkpoint mismatch")
    args.output.mkdir(parents=True)
    n = protocol["selected_patch_count"]
    full = np.lib.format.open_memmap(args.output/"full_probabilities.npy",mode="w+",dtype=np.float32,shape=(len(ARMS),n,128,128))
    full[:] = np.nan
    states = np.load(args.prepared/"states.npy",allow_pickle=False)
    baseline = np.load(args.prepared/"historical_identity.npy",allow_pickle=False)
    started,records = time.monotonic(),[]
    rt = protocol["runtime"]
    with concurrent.futures.ProcessPoolExecutor(max_workers=rt["workers"],mp_context=multiprocessing.get_context("spawn"),
            initializer=init_worker,initargs=(str(args.source_repo),str(args.checkpoint),str(args.prepared),rt["threads_per_worker"])) as pool:
        for phase,tasks in [("identity",[(0,pi) for pi in range(n)]),
                            ("stress",[(ai,pi) for pi in range(n) for ai in range(1,len(ARMS))])]:
            futures = [pool.submit(forward,t) for t in tasks]
            for future in concurrent.futures.as_completed(futures):
                ai,pi,p,seconds,rss,pid = future.result()
                full[ai,pi] = p
                records.append(dict(arm=ARMS[ai]["name"],patch=pi,author_index=protocol["controls"][pi]["index"],seconds=seconds,peak_rss_kib=rss,pid=pid))
                if len(records)%10 == 0 or len(records) in (n,n*len(ARMS)):
                    full.flush()
                    print("PROGRESS",len(records),"/",n*len(ARMS),"seconds",round(time.monotonic()-started,2),flush=True)
            if phase == "identity":
                predicted = np.array(full[0,:,48:80,48:80])
                difference = float(np.max(np.abs(predicted-baseline)))
                measured,reference = metrics(predicted,states),metrics(baseline,states)
                ap_difference = abs(measured["ap"]-reference["ap"])
                passed = difference <= protocol["identity_gate"]["historical_max_probability_difference"] and ap_difference <= protocol["identity_gate"]["historical_max_AP_difference"]
                check = dict(passed=passed,maximum_probability_difference=difference,AP_difference=ap_difference,metrics=measured,historical_same_population=reference)
                write_json(args.output/"identity_check.json",check)
                print("IDENTITY_CHECK",json.dumps(check),flush=True)
                if not passed:
                    raise ValueError("Identity prediction gate failed; stress forwards not submitted")
    full.flush()
    import torch,scipy
    receipt = dict(protocol_sha256=sha256(args.prepared/"protocol.json"),code_hashes=code_hashes(),source_commit=SOURCE_COMMIT,
                   checkpoint_sha256=WEIGHTS_SHA256,wall_seconds=time.monotonic()-started,summed_forward_seconds=sum(r["seconds"] for r in records),
                   forwards=records,full_probabilities_sha256=sha256(args.output/"full_probabilities.npy"),
                   python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,runtime=rt)
    write_json(args.output/"inference_receipt.json",receipt)
    print("INFERENCE_COMPLETE",receipt["wall_seconds"],flush=True)
    summarize(args)


def summarize(args):
    protocol = checked_protocol(args)
    receipt = json.loads((args.output/"inference_receipt.json").read_text())
    if receipt["protocol_sha256"] != sha256(args.prepared/"protocol.json"):
        raise ValueError("Receipt mismatch")
    path = args.output/"full_probabilities.npy"
    if sha256(path) != receipt["full_probabilities_sha256"]:
        raise ValueError("Predictions changed")
    full = np.load(path,mmap_mode="r",allow_pickle=False)
    states = np.load(args.prepared/"states.npy",allow_pickle=False)
    with np.load(args.prepared/"mapping.npz",allow_pickle=False) as mapping:
        queries = mapping["query_yx"]
        common = mapping["common_mask"]
    aligned = np.stack([np.stack([map_coordinates(p,q.transpose(2,0,1),order=1,mode="constant",cval=np.nan,prefilter=False)
                                for p,q in zip(full[ai],queries[ai])]) for ai in range(len(ARMS))])
    if not np.isfinite(aligned[:,common]).all():
        raise ValueError("Invalid probability on scored support")
    names = [a["name"] for a in ARMS]
    pooled = {name:metrics(p,states) for name,p in zip(names,aligned)}
    per_core = [dict(author_index=r["index"],category=r["category"],metrics={name:metrics(p[pi],states[pi]) for name,p in zip(names,aligned)})
                for pi,r in enumerate(protocol["controls"])]
    comparisons = {}
    for name in names[1:]:
        deltas = [row["metrics"][name]["ap"]-row["metrics"]["identity"]["ap"] for row in per_core if row["category"]=="mixed"]
        base,m = pooled["identity"],pooled[name]
        dap,dr,df = m["ap"]-base["ap"],m["recall_at_0_5"]-base["recall_at_0_5"],m["false_positive_rate_at_0_5"]-base["false_positive_rate_at_0_5"]
        comparisons[name] = dict(delta_AP=dap,delta_AUC=m["auc"]-base["auc"],delta_recall=dr,delta_FPR=df,
            median_mixed_core_AP_delta=float(np.median(deltas)),mixed_core_AP_deltas=deltas,
            mixed_cores_losing_more_than_0_10_AP=int((np.asarray(deltas)<-.10).sum()),
            material_AP_drop=bool(dap<-.05),material_recall_loss=bool(dr<-.10),material_FPR_increase=bool(df>.03))
    np.savez_compressed(args.output/"scored_arrays.npz",probabilities=aligned,states=states,
        original_states=np.load(args.prepared/"original_states.npy",allow_pickle=False),common_mask=common,
        author_indices=np.array([r["index"] for r in protocol["controls"]],dtype=np.int64),condition_names=np.array(names))
    report = dict(protocol_sha256=sha256(args.prepared/"protocol.json"),pooled=pooled,comparisons=comparisons,per_core=per_core,
                  scored_archive_sha256=sha256(args.output/"scored_arrays.npz"),scope=protocol["scope"],statistics=protocol["interpretation"])
    write_json(args.output/"results.json",report)
    print("RESULTS",json.dumps(pooled),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","infer","summarize"))
    for name in ("source-repo","cache","geometry-cache","native-cache","checkpoint","prepared","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--geometry-policy",choices=("initial95","retained90"),default="initial95")
    args = parser.parse_args()
    if args.workers<1 or args.threads<1:
        parser.error("Positive runtime worker/thread counts required")
    {"prepare":prepare,"infer":infer,"summarize":summarize}[args.command](args)


if __name__ == "__main__":
    main()
