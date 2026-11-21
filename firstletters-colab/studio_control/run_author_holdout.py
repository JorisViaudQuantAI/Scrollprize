#!/usr/bin/env python3
"""Replay all labeled central32 pixels of the author's fixed PHerc0841 holdout.

This follows the three-cell diagnostic; it is a release-reproduction check, not
a newly independent benchmark. Patches without known labels in their central32
are omitted because they contribute no pixels to the published AP statistic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

from run_public_control import (BASE, CONTROL_ID, DATA_REVISION, DATASET, FILES,
                                SOURCE_COMMIT, WEIGHTS_SHA256, WEIGHTS_URL,
                                metrics, raw_crop, sha256, verified_file, write_json)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-repo", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.threads < 1 or args.output_dir.exists():
        ap.error("Use positive --threads and a new --output-dir")
    head = subprocess.check_output(["git", "-C", str(args.source_repo), "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE_COMMIT:
        ap.error("Unexpected upstream source revision")
    subprocess.run(["git", "-C", str(args.source_repo), "diff", "--exit-code", "HEAD", "--",
                    "studio-multiscroll-cpt-20261023/hf/model.py",
                    "studio-multiscroll-cpt-20261023/hf/inference.py"], check=True)
    for name, (digest, size) in FILES.items():
        verified_file(BASE + name, args.cache / name, digest, size)
    verified_file(WEIGHTS_URL, args.checkpoint, WEIGHTS_SHA256, 893443724)
    labels = np.load(args.cache / "labels.npy", mmap_mode="r", allow_pickle=False)
    ct = np.load(args.cache / "ct-tiles.npy", mmap_mode="r", allow_pickle=False)
    coordinates = np.load(args.cache / "coordinates.npy", allow_pickle=False)
    tiles = np.load(args.cache / "tiles.npy", allow_pickle=False)
    selected = []
    states = []
    for index, (yy, xx) in enumerate(coordinates):
        y, x = int(yy), int(xx)
        state = labels[y + 48:y + 80, x + 48:x + 80]
        if not np.any(state > 0):
            continue
        if y < 0 or x < 0 or y + 128 > labels.shape[0] or x + 128 > labels.shape[1]:
            raise ValueError("A selected public control needs padding; this runner does not invent it")
        selected.append(dict(index=index, y=y, x=x, ink=int((state == 2).sum()), background=int((state == 1).sum())))
        states.append(np.array(state, copy=True))
    state_array = np.stack(states)
    known_ink, known_background = int((state_array == 2).sum()), int((state_array == 1).sum())
    if (known_ink, known_background) != (15791, 29558):
        raise ValueError("Selected pixels do not match the published holdout denominator")
    args.output_dir.mkdir(parents=True)
    protocol = dict(
        purpose="Reproduce released soft-loss checkpoint on every labeled central32 pixel of the author's fixed holdout",
        source_commit=SOURCE_COMMIT, data_revision=DATA_REVISION, dataset=DATASET, public_control=CONTROL_ID,
        checkpoint_sha256=WEIGHTS_SHA256,
        script_sha256=sha256(__file__), shared_helper_sha256=sha256(Path(__file__).with_name("run_public_control.py")),
        original_patch_count=len(coordinates), selected_patch_count=len(selected),
        known_ink=known_ink, known_background=known_background, controls=selected,
        selection="All published patches with any known central32 label; other patches cannot affect AP/AUC",
        sampling="Native prepared16, stored order unchanged, no resampling or padding",
        inference="Released model; uint8 to FP32 clip[0,200]/255; central32 of one128 forward; batch1; CPU",
        metrics="Pooled AP/AUC over all known pixels; unknown excluded; diagnostic threshold0.5",
        comparison=dict(published_softloss_ap=0.7536, prospective_absolute_ap_tolerance=0.02,
                        interpretation="Engineering reproduction tolerance, not an ink-accuracy or prize gate"),
        sequence="Chosen after inspecting three-cell diagnostic; no threshold, model or data changes; not an untouched research test",
        differences_from_author="CPU FP32 eager batch1 rather than CUDA BF16 batches; skip zero-known-label central32 cores; no loss computation",
        threads=args.threads,
    )
    write_json(args.output_dir / "protocol.json", protocol)
    protocol_hash = sha256(args.output_dir / "protocol.json")
    print("PROTOCOL_FROZEN", protocol_hash, "patches", len(selected), flush=True)
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.source_repo / "studio-multiscroll-cpt-20261023/hf"))
    from inference import load_model
    start = time.monotonic()
    model = load_model(args.checkpoint, "cpu")
    load_seconds = time.monotonic() - start
    predictions = []
    seconds = []
    for n, item in enumerate(selected, 1):
        raw = raw_crop(ct, tiles, item["y"], item["x"])
        x = torch.from_numpy(raw[None, None]).float().clamp_(0, 200).div_(255)
        start = time.monotonic()
        with torch.inference_mode():
            p = model(x)[0, 0, 48:80, 48:80].float().sigmoid().numpy()
        seconds.append(time.monotonic() - start)
        if not np.isfinite(p).all():
            raise ValueError("Nonfinite model probabilities")
        predictions.append(p)
        if n % 10 == 0 or n == len(selected):
            print("PROGRESS", n, "/", len(selected), "inference_seconds", round(sum(seconds), 2), flush=True)
    predictions = np.stack(predictions)
    score = metrics(predictions, state_array)
    np.save(args.output_dir / "probabilities.npy", predictions, allow_pickle=False)
    np.save(args.output_dir / "states.npy", state_array, allow_pickle=False)
    difference = abs(score["ap"] - protocol["comparison"]["published_softloss_ap"])
    result = dict(
        protocol_sha256=protocol_hash, checkpoint_sha256=WEIGHTS_SHA256,
        metrics=score, absolute_ap_difference_from_published=difference,
        within_prospective_reproduction_tolerance=difference <= protocol["comparison"]["prospective_absolute_ap_tolerance"],
        selected_patches=len(selected), skipped_zero_known_core_patches=len(coordinates)-len(selected),
        load_seconds=load_seconds, inference_seconds=sum(seconds), seconds_per_forward=seconds,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        numpy_version=np.__version__, torch_version=torch.__version__, python_version=platform.python_version(),
        probability_sha256=sha256(args.output_dir / "probabilities.npy"), states_sha256=sha256(args.output_dir / "states.npy"),
        limitation="Author's repeatedly evaluated donor holdout; correlated pixels; reproduces integration and does not prove unseen-scroll transfer or T4 performance",
    )
    write_json(args.output_dir / "results.json", result)
    print("FINISHED", json.dumps({k: v for k, v in result.items() if k != "seconds_per_forward"}), flush=True)


if __name__ == "__main__":
    main()
