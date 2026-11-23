#!/usr/bin/env python3
"""Frozen Q10 Studio sampling/scale stress on the author's public PHerc0841 holdout.

prepare never imports the model. infer consumes hashed prepared arrays and code;
summarize scores fixed physical labels, including a prospectively specified mean.
Large arrays stay outside git. No unread surface or candidate is used.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "studio_control"))
from run_public_control import (BASE, CONTROL_ID, DATA_REVISION, DATASET, FILES,
                                SOURCE_COMMIT, WEIGHTS_SHA256, WEIGHTS_URL,
                                metrics, raw_crop, sha256, verified_file, write_json)
from sampling import build_mosaic, context_support, sample_input, map_prediction

ARMS = [
    dict(name="identity", sx=1.0, sy=1.0),
    dict(name="x_half", sx=0.5, sy=1.0),
    dict(name="x_double", sx=2.0, sy=1.0),
    dict(name="y_half", sx=1.0, sy=0.5),
    dict(name="y_double", sx=1.0, sy=2.0),
]


def code_hashes():
    paths = [HERE / "run.py", HERE / "sampling.py", HERE / "test_sampling.py",
             HERE.parent / "studio_control" / "run_public_control.py"]
    return {str(p.relative_to(HERE.parent)): sha256(p) for p in paths}


def verify_source(source_repo):
    actual = subprocess.check_output(["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != SOURCE_COMMIT:
        raise ValueError("Unexpected Studio source revision")
    paths = ["studio-multiscroll-cpt-20261023/hf/model.py", "studio-multiscroll-cpt-20261023/hf/inference.py"]
    subprocess.run(["git", "-C", str(source_repo), "diff", "--exit-code", "HEAD", "--", *paths], check=True)
    return {p: sha256(source_repo / p) for p in paths}


def prepare(args):
    if args.prepared.exists():
        raise ValueError("Use a new prepared directory")
    upstream = verify_source(args.source_repo)
    start = time.monotonic()
    for name, (digest, size) in FILES.items():
        verified_file(BASE + name, args.cache / name, digest, size)
    verified_file(WEIGHTS_URL, args.checkpoint, WEIGHTS_SHA256, 893443724)
    labels = np.load(args.cache / "labels.npy", mmap_mode="r", allow_pickle=False)
    ct = np.load(args.cache / "ct-tiles.npy", mmap_mode="r", allow_pickle=False)
    coordinates = np.load(args.cache / "coordinates.npy", allow_pickle=False)
    tiles = np.load(args.cache / "tiles.npy", allow_pickle=False)
    if ct.shape != (28, 512, 512, 16) or ct.dtype != np.uint8 or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Unexpected fixture encoding")
    mosaic = build_mosaic(ct, tiles)
    selected, excluded, states = [], [], []
    for index, (yy, xx) in enumerate(coordinates):
        y, x = int(yy), int(xx)
        state = np.array(labels[y + 48:y + 80, x + 48:x + 80], copy=True)
        if not np.any(state > 0):
            continue
        if state.shape != (32, 32):
            raise ValueError("Physical core outside labels")
        item = dict(index=index, y=y, x=x, ink=int((state == 2).sum()), background=int((state == 1).sum()))
        support = {a["name"]: context_support(mosaic, y, x, sx=a["sx"], sy=a["sy"]) for a in ARMS}
        item["support"] = support
        if not all(s["complete"] for s in support.values()):
            excluded.append(item)
            continue
        selected.append(item)
        states.append(state)
    if len(selected) + len(excluded) != 53:
        raise ValueError("Author holdout selection drifted")
    if not selected:
        raise ValueError("No common supported cores")
    # Distinct physical core coordinates are a denominator requirement.
    occupied = set()
    for item in selected:
        core = {(item["y"] + 48 + j, item["x"] + 48 + i) for j in range(32) for i in range(32)}
        if occupied.intersection(core):
            raise ValueError("Overlapping physical evaluation cores")
        occupied.update(core)
    states = np.stack(states)
    if not excluded and (int((states == 2).sum()), int((states == 1).sum())) != (15791, 29558):
        raise ValueError("Published label denominator does not match")
    args.prepared.mkdir(parents=True)
    np.save(args.prepared / "states.npy", states, allow_pickle=False)
    inputs = np.lib.format.open_memmap(args.prepared / "inputs.npy", mode="w+", dtype=np.float32,
                                     shape=(len(ARMS), len(selected), 16, 128, 128))
    input_hashes = []
    for ai, arm in enumerate(ARMS):
        for pi, item in enumerate(selected):
            raw = sample_input(mosaic, item["y"], item["x"], sx=arm["sx"], sy=arm["sy"])
            if raw.shape != (16, 128, 128) or raw.dtype != np.float32 or not np.isfinite(raw).all():
                raise ValueError("Invalid prepared model input")
            if ai == 0 and not np.array_equal(raw, raw_crop(ct, tiles, item["y"], item["x"]).astype(np.float32)):
                raise ValueError("Identity does not exactly reproduce existing native crop")
            inputs[ai, pi] = raw
            input_hashes.append(dict(arm=arm["name"], index=item["index"], sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
    inputs.flush()
    protocol = dict(
        version="q10-studio-scale-v1", purpose="Controlled anisotropic sampling/scale stress of a released ink model",
        scope="Author's repeatedly evaluated PHerc0841 donor holdout; not independent scroll transfer, new text, or a mesh comparison",
        base_commit="cff6d9a03ca0a50f685c9a9e9e83cd6b52ad7f2b",
        source_commit=SOURCE_COMMIT, source_files=upstream,
        dataset=DATASET, data_revision=DATA_REVISION, public_control=CONTROL_ID,
        acquisition={n: dict(url=BASE+n, sha256=h, bytes=s) for n, (h, s) in FILES.items()},
        weights=dict(url=WEIGHTS_URL, sha256=WEIGHTS_SHA256, bytes=893443724),
        code_hashes=code_hashes(), arms=ARMS,
        label_selection="All published coordinates with any known label in original central32, intersected across measured-tile context support of every arm before model load",
        original_labeled_patch_count=53, selected_patch_count=len(selected), controls=selected, exclusions=excluded,
        denominator=dict(ink=int((states == 2).sum()), background=int((states == 1).sum()), unknown=int((states == 0).sum())),
        source_support="Presence in hash-verified prepared tile manifest; this does not independently re-audit raw scan coverage inside the author's tiles",
        input_mapping="source_xy = native_patch_origin + 63.5 + (model_xy - 63.5) / scale_xy",
        output_mapping="model_xy = 63.5 + scale_xy * (physical_core_index - 15.5); bilinear probability interpolation; labels untouched",
        interpolation="Raw CT: Gaussian/bilinear accumulation in float64, rounded once to FP32; separable Gaussian only on compressed XY axis, sigma=(1/s-1)/2=0.5 and truncate=4, then bilinear; no depth filtering; no padding; complete positive-weight stencil including Gaussian halo required",
        normalization="After resampling: FP32 clip[0,200]/255 exactly once; stored depth order unchanged",
        inference="Pinned released model, eval mode, CPU FP32, batch1, no retraining, no spatial tiled blending; output readback lies inside central96",
        runtime=dict(workers=args.workers, threads_per_worker=args.threads, interop_threads=1),
        identity_gate=dict(metric="AP", reference=0.7536, maximum_absolute_difference=0.02,
                           action="Stop interpretation/inference of stress arms if failed; gate applies only if all53 cores retained"),
        primary=dict(metric="Pooled AP delta versus identity on identical known physical pixels", materially_harmful_delta_below=-0.05,
                     interpretation="Prospective engineering magnitude, not a statistical significance threshold"),
        secondary=dict(metrics=["ROC AUC", "recall_at_0.5", "background_FPR_at_0.5", "mixed-core AP deltas"],
                       readback_only_control="Resample the identity probability field onto each scaled grid, then back to physical32; bilinear in both directions, no Gaussian, no new network forward. Quantifies score-grid interpolation alone; never subtract this from neural deltas",
                       fusion="Unweighted arithmetic mean of five aligned probabilities; no selection of best arm per pixel or label",
                       fusion_advance_gate=dict(minimum_pooled_AP_gain=0.02, minimum_median_mixed_core_AP_delta=0.0,
                                                maximum_FPR_increase=0.02, maximum_fraction_mixed_cores_losing_more_than_0_10_AP=0.10),
                       advance_meaning="Candidate for a separate independent-control experiment only; never immediate deployment"),
        statistics="Descriptive paired comparisons. Cores lie in one compact annotated region; no IID-pixel CI or claim of independent53 samples",
        caveats=["Scale also changes physical context/receptive field, interpolation and anti-aliasing; this does not isolate intrinsic scale equivariance",
                 "Image x/y are not asserted to align with handwriting lines",
                 "The checkpoint trained with XY scale augmentation around0.9-1.1, so0.5/2 tests extrapolation",
                 "No line readability, glyph recovery, tilt, layer error, or unread-scroll detection is measured"],
        arrays={n: dict(sha256=sha256(args.prepared/n), bytes=(args.prepared/n).stat().st_size) for n in ["inputs.npy", "states.npy"]},
        input_hashes=input_hashes, preparation_seconds=time.monotonic()-start,
    )
    write_json(args.prepared / "protocol.json", protocol)
    print("PROTOCOL_FROZEN", sha256(args.prepared/"protocol.json"), "cores", len(selected), "exclusions", len(excluded), flush=True)


_MODEL = _INPUTS = None


def init_worker(source_repo, checkpoint, prepared, threads):
    global _MODEL, _INPUTS
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(Path(source_repo) / "studio-multiscroll-cpt-20261023/hf"))
    from inference import load_model
    _MODEL = load_model(Path(checkpoint), "cpu")
    _INPUTS = np.load(Path(prepared) / "inputs.npy", mmap_mode="r", allow_pickle=False)


def forward(task):
    import torch
    ai, pi = task
    x = torch.from_numpy(np.array(_INPUTS[ai, pi], copy=True)[None, None]).clamp_(0, 200).div_(255)
    start = time.monotonic()
    with torch.inference_mode():
        p = _MODEL(x)[0, 0].float().sigmoid().numpy()
    if not np.isfinite(p).all() or p.shape != (128, 128):
        raise ValueError("Invalid model prediction")
    return ai, pi, p, time.monotonic()-start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, os.getpid()


def infer(args):
    if args.output.exists():
        raise ValueError("Use a new output directory")
    protocol = json.loads((args.prepared/"protocol.json").read_text())
    if protocol["code_hashes"] != code_hashes():
        raise ValueError("Code changed after protocol freeze")
    if protocol["source_files"] != verify_source(args.source_repo):
        raise ValueError("Upstream files differ")
    if sha256(args.checkpoint) != WEIGHTS_SHA256:
        raise ValueError("Checkpoint mismatch")
    for n, meta in protocol["arrays"].items():
        if sha256(args.prepared/n) != meta["sha256"]:
            raise ValueError("Prepared arrays changed")
    args.output.mkdir(parents=True)
    n = protocol["selected_patch_count"]
    predictions = np.lib.format.open_memmap(args.output/"full_probabilities.npy", mode="w+", dtype=np.float32,
                                           shape=(len(ARMS), n, 128, 128))
    predictions[:] = np.nan
    states = np.load(args.prepared/"states.npy", allow_pickle=False)
    runtime = protocol["runtime"]
    records, start = [], time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=runtime["workers"], mp_context=multiprocessing.get_context("spawn"),
            initializer=init_worker, initargs=(str(args.source_repo), str(args.checkpoint), str(args.prepared), runtime["threads_per_worker"])) as pool:
        # Identity executes first so a failed release reproduction cannot be hidden by perturbation results.
        for phase, tasks in [("identity", [(0, pi) for pi in range(n)]),
                             ("stress", [(ai, pi) for pi in range(n) for ai in range(1, len(ARMS))])]:
            futures = [pool.submit(forward, t) for t in tasks]
            for future in concurrent.futures.as_completed(futures):
                ai, pi, p, seconds, rss, pid = future.result()
                predictions[ai, pi] = p
                records.append(dict(arm=ARMS[ai]["name"], patch=pi, author_index=protocol["controls"][pi]["index"], seconds=seconds, worker_peak_rss_kib=rss, pid=pid))
                if len(records) % 10 == 0 or len(records) in (n, n*len(ARMS)):
                    predictions.flush()
                    print("PROGRESS", len(records), "/", n*len(ARMS), "wall_seconds", round(time.monotonic()-start,2), flush=True)
            if phase == "identity":
                value = metrics(np.array(predictions[0, :, 48:80, 48:80]), states)
                passed = None if protocol["exclusions"] else abs(value["ap"]-protocol["identity_gate"]["reference"]) <= protocol["identity_gate"]["maximum_absolute_difference"]
                write_json(args.output/"identity_check.json", dict(metrics=value, gate_passed=passed))
                print("IDENTITY_CHECK", json.dumps(dict(metrics=value, gate_passed=passed)), flush=True)
                if passed is False:
                    raise RuntimeError("Identity integration gate failed; stress inference aborted")
    predictions.flush()
    import torch
    import scipy
    receipt = dict(protocol_sha256=sha256(args.prepared/"protocol.json"), code_hashes=code_hashes(),
                   source_commit=SOURCE_COMMIT, checkpoint_sha256=WEIGHTS_SHA256,
                   wall_seconds=time.monotonic()-start, summed_forward_seconds=sum(r["seconds"] for r in records),
                   forwards=records, full_probabilities_sha256=sha256(args.output/"full_probabilities.npy"),
                   python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, scipy=scipy.__version__,
                   cpu_count=os.cpu_count(), runtime=runtime,
                   peak_worker_rss_kib={str(pid): max(r["worker_peak_rss_kib"] for r in records if r["pid"]==pid) for pid in set(r["pid"] for r in records)})
    write_json(args.output/"inference_receipt.json", receipt)
    print("INFERENCE_COMPLETE", "wall_seconds", round(receipt["wall_seconds"],2), flush=True)
    summarize(args)


def summarize(args):
    from scipy.ndimage import map_coordinates
    protocol = json.loads((args.prepared/"protocol.json").read_text())
    receipt = json.loads((args.output/"inference_receipt.json").read_text())
    if receipt["protocol_sha256"] != sha256(args.prepared/"protocol.json"):
        raise ValueError("Receipt belongs to another protocol")
    if protocol["code_hashes"] != code_hashes():
        raise ValueError("Scoring code changed after freeze")
    if sha256(args.prepared/"states.npy") != protocol["arrays"]["states.npy"]["sha256"]:
        raise ValueError("Physical labels changed after freeze")
    full = np.load(args.output/"full_probabilities.npy", mmap_mode="r", allow_pickle=False)
    states = np.load(args.prepared/"states.npy", allow_pickle=False)
    if sha256(args.output/"full_probabilities.npy") != receipt["full_probabilities_sha256"] or not np.isfinite(full).all():
        raise ValueError("Incomplete or changed model outputs")
    aligned = np.stack([np.stack([map_prediction(p, sx=a["sx"], sy=a["sy"]) for p in full[ai]]) for ai,a in enumerate(ARMS)])
    aligned = np.concatenate([aligned, aligned.mean(axis=0, keepdims=True, dtype=np.float32)])
    np.save(args.output/"aligned_probabilities.npy", aligned, allow_pickle=False)
    names = [a["name"] for a in ARMS] + ["uniform_mean5"]
    pooled = {name: metrics(p, states) for name,p in zip(names, aligned)}
    readback_only = {}
    for arm in ARMS[1:]:
        gy, gx = np.meshgrid(63.5+(np.arange(128)-63.5)/arm["sy"],
                             63.5+(np.arange(128)-63.5)/arm["sx"], indexing="ij")
        valid = (gy >= 0) & (gy <= 127) & (gx >= 0) & (gx <= 127)
        # Out-of-support fill is never read: map_prediction checks its full stencil.
        q = np.stack([map_prediction(map_coordinates(p, [gy, gx], order=1, mode="constant", cval=0, prefilter=False),
                                     sx=arm["sx"], sy=arm["sy"], available_hw=valid) for p in full[0]])
        m = metrics(q, states)
        readback_only[arm["name"]] = dict(metrics=m, delta_AP=m["ap"]-pooled["identity"]["ap"],
                                         mean_absolute_probability_change=float(np.abs(q-aligned[0]).mean()))
    per_core = [{"patch": pi, "author_index": protocol["controls"][pi]["index"],
                 "metrics": {name: metrics(p[pi], states[pi]) for name,p in zip(names, aligned)}} for pi in range(len(states))]
    comparisons = {}
    for name in names[1:]:
        deltas = np.array([r["metrics"][name]["ap"]-r["metrics"]["identity"]["ap"] for r in per_core if r["metrics"]["identity"]["ap"] is not None])
        row = dict(delta_AP=pooled[name]["ap"]-pooled["identity"]["ap"], delta_AUC=pooled[name]["auc"]-pooled["identity"]["auc"],
                   delta_recall_at_0_5=pooled[name]["recall_at_0_5"]-pooled["identity"]["recall_at_0_5"],
                   delta_FPR_at_0_5=pooled[name]["false_positive_rate_at_0_5"]-pooled["identity"]["false_positive_rate_at_0_5"],
                   mixed_core_count=len(deltas), median_mixed_core_AP_delta=float(np.median(deltas)),
                   mixed_cores_losing_more_than_0_10_AP=int((deltas < -0.10).sum()),
                   fraction_mixed_cores_losing_more_than_0_10_AP=float((deltas < -0.10).mean()),
                   mixed_core_AP_delta_range=[float(deltas.min()), float(deltas.max())],
                   materially_harmful_AP_loss=pooled[name]["ap"]-pooled["identity"]["ap"] < protocol["primary"]["materially_harmful_delta_below"])
        if name == "uniform_mean5":
            g = protocol["secondary"]["fusion_advance_gate"]
            row["advance_gate_components"] = dict(pooled_AP_gain=row["delta_AP"] >= g["minimum_pooled_AP_gain"],
                median_mixed_core_AP_delta=row["median_mixed_core_AP_delta"] >= g["minimum_median_mixed_core_AP_delta"],
                FPR_increase=row["delta_FPR_at_0_5"] <= g["maximum_FPR_increase"],
                mixed_core_harm=row["fraction_mixed_cores_losing_more_than_0_10_AP"] <= g["maximum_fraction_mixed_cores_losing_more_than_0_10_AP"])
            row["advance_to_independent_control"] = all(row["advance_gate_components"].values())
        comparisons[name] = row
    result = dict(protocol_sha256=receipt["protocol_sha256"], pooled=pooled, comparisons=comparisons,
                  readback_only_control=readback_only, per_core=per_core,
                  aligned_probabilities_sha256=sha256(args.output/"aligned_probabilities.npy"), states_sha256=sha256(args.prepared/"states.npy"),
                  inference_receipt_sha256=sha256(args.output/"inference_receipt.json"),
                  interpretation="Paired development sensitivity on an exposed author holdout; no untouched-scroll, text reading, or population inference")
    write_json(args.output/"results.json", result)
    print("RESULTS", json.dumps(dict(pooled=pooled, comparisons=comparisons)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "infer", "summarize"])
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or args.workers < 1:
        parser.error("Positive workers and threads required")
    globals()[args.phase](args)


if __name__ == "__main__":
    main()
