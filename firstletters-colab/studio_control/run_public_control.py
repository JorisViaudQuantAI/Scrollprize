#!/usr/bin/env python3
"""Small, fixed Studio release replay on an author's public PHerc0841 holdout.

Only public, already released data are fetched. This is a CPU integration check,
not an independent ink benchmark or a search of an unread scroll.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import urllib.request

import numpy as np


SOURCE_COMMIT = "bf66f753176792a966bca17822629fb8821f20c8"
DATA_REVISION = "ad919d288ede1c4daf92725288bb6fe29c558591"
DATASET = "paranke/vesuvius-814-841-9b-studio-training-data"
CONTROL_ID = "841-expanded-layer-03-approved-r000001"
PREFIX = f"donor-studio-resunet-scratch-20260926/prepared/{CONTROL_ID}/"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REVISION}/{PREFIX}"
WEIGHTS_SHA256 = "96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90"
WEIGHTS_URL = (
    "https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/"
    "download/studio-softloss-20261023/studio-softloss-20261023-ema.safetensors"
)
FILES = {
    "ct-tiles.npy": ("0c2950497da0dc362ba572c2fb1c08f88e3accb51c1e741d0a85633fb46dde33", 117440640),
    "labels.npy": ("ff445929d549e081133a27141e96e30ff22136c441c882055215bace58bcf548", 7234885),
    "coordinates.npy": ("4a93ec41845b451ce0652f64a9e89b50521f1484b5b5440cf7d27dda471ab1dd", 10800),
    "tiles.npy": ("fb8d65ff2a4c41e063a26c5cbc2312c34bd906f3927c5721653211286dfb16be", 352),
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_file(url, path, expected, size):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name + ".part")
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if temporary.stat().st_size != size or sha256(temporary) != expected:
            raise ValueError(f"Downloaded content failed verification: {path.name}")
        temporary.replace(path)
    if path.stat().st_size != size or sha256(path) != expected:
        raise ValueError(f"Cached content failed verification: {path.name}")


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def select_controls(labels, coordinates):
    """Labels select three nonoverlapping cells before any model is loaded."""
    selected = []
    for category in ("mixed", "sparse", "background"):
        eligible = []
        for y, x in sorted(map(tuple, coordinates.tolist())):
            if y < 0 or x < 0 or y + 128 > labels.shape[0] or x + 128 > labels.shape[1]:
                continue
            if any(abs(y - s["y"]) < 128 and abs(x - s["x"]) < 128 for s in selected):
                continue
            state = labels[y + 16:y + 112, x + 16:x + 112]
            ink, background = int((state == 2).sum()), int((state == 1).sum())
            matches = {
                "mixed": ink >= 256 and background >= 256 and 0.2 <= ink / max(ink + background, 1) <= 0.8,
                "sparse": 16 <= ink <= 128 and background >= 512,
                "background": ink == 0 and background >= 2048,
            }
            if matches[category]:
                eligible.append(dict(category=category, y=y, x=x, ink=ink, background=background))
        if not eligible:
            raise ValueError(f"No control meets the frozen {category} definition")
        selected.append(dict(eligible[0], eligible_count=len(eligible)))
    return selected


def raw_crop(ct, tiles, y, x):
    """Copy the same native prepared samples, including at tile boundaries."""
    lookup = {(int(yy) // 512, int(xx) // 512): k for k, (yy, xx) in enumerate(tiles)}
    yy, xx = y + np.arange(128), x + np.arange(128)
    raw = np.empty((128, 128, 16), np.uint8)
    for ty in np.unique(yy // 512):
        for tx in np.unique(xx // 512):
            my, mx = yy // 512 == ty, xx // 512 == tx
            raw[np.ix_(my, mx)] = ct[lookup[(int(ty), int(tx))]][np.ix_(yy[my] % 512, xx[mx] % 512)]
    return np.ascontiguousarray(raw.transpose(2, 0, 1))


def metrics(probability, state):
    known = state > 0  # Unknown is never treated as background.
    truth = state[known] == 2
    score = probability[known]
    positive, negative = int(truth.sum()), int((~truth).sum())
    tp, fp = int(((score >= 0.5) & truth).sum()), int(((score >= 0.5) & ~truth).sum())
    result = dict(ink_pixels=positive, background_pixels=negative, unknown_pixels=int((~known).sum()),
                  true_positive_at_0_5=tp, false_positive_at_0_5=fp,
                  recall_at_0_5=tp / positive if positive else None,
                  false_positive_rate_at_0_5=fp / negative if negative else None,
                  ap=None, auc=None)
    if positive and negative:
        order = np.argsort(-score, kind="stable")
        truth, score = truth[order], score[order]
        ends = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]
        tps = np.cumsum(truth)[ends]
        fps = 1 + ends - tps
        recall = tps / positive
        precision = tps / (tps + fps)
        result["ap"] = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
        result["auc"] = float(np.trapezoid(np.r_[0.0, recall], np.r_[0.0, fps / negative]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output_dir.exists():
        parser.error("Output directory already exists; use a new directory")
    actual_commit = subprocess.check_output(["git", "-C", str(args.source_repo), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != SOURCE_COMMIT:
        parser.error(f"Expected upstream commit {SOURCE_COMMIT}")
    model_dir = args.source_repo / "studio-multiscroll-cpt-20261023/hf"
    subprocess.run(["git", "-C", str(args.source_repo), "diff", "--exit-code", "HEAD", "--",
                    "studio-multiscroll-cpt-20261023/hf/model.py",
                    "studio-multiscroll-cpt-20261023/hf/inference.py"], check=True)
    for name, (digest, size) in FILES.items():
        verified_file(BASE + name, args.cache / name, digest, size)
    verified_file(WEIGHTS_URL, args.checkpoint, WEIGHTS_SHA256, 893443724)
    labels = np.load(args.cache / "labels.npy", mmap_mode="r", allow_pickle=False)
    ct = np.load(args.cache / "ct-tiles.npy", mmap_mode="r", allow_pickle=False)
    coords = np.load(args.cache / "coordinates.npy", allow_pickle=False)
    tiles = np.load(args.cache / "tiles.npy", allow_pickle=False)
    if ct.dtype != np.uint8 or ct.shape != (28, 512, 512, 16) or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Unexpected public fixture encoding")
    selected = select_controls(labels, coords)
    args.output_dir.mkdir(parents=True)
    protocol = dict(
        kind="public author-holdout integration check; not independent generalization",
        source_commit=SOURCE_COMMIT, dataset=DATASET, data_revision=DATA_REVISION, public_control=CONTROL_ID,
        files={name: dict(sha256=digest, bytes=size) for name, (digest, size) in FILES.items()},
        checkpoint_sha256=WEIGHTS_SHA256, script_sha256=sha256(__file__),
        selection="Lexicographically first distinct 128x128 cells from published coordinates satisfying label-only strata; no prediction used",
        definitions={"mixed": "central96 ink>=256, background>=256, 0.2<=ink/(ink+background)<=0.8",
                     "sparse": "central96 16<=ink<=128, background>=512", "background": "central96 ink=0, background>=2048"},
        controls=selected, order_trials=["stored", "same-band-reversed"],
        physical_sampling="Unchanged author's native prepared16; no mesh-centre claim; XY9.36um, depth9.366um per source recipe",
        acquisition_coverage="Author-prepared fixture verified by published content hashes; source chunk coverage not independently re-audited",
        input="raw uint8 [16,128,128]; float32 clip[0,200]/255 exactly once",
        inference="Unchanged released network; one direct128 forward per trial; no tiled canvas, TTA or calibration",
        metrics="Known labels only (1 background, 2 ink); central96 and central32; fixed threshold0.5; tie-grouped AP and ROC AUC",
        runtime=dict(device="cpu", precision="fp32", batch_size=1, threads=args.threads),
        interpretation="Check execution, input provenance and direction sensitivity. No accuracy gate for atlas and no reversal-based candidate veto.",
    )
    for item in selected:
        raw = raw_crop(ct, tiles, item["y"], item["x"])
        np.save(args.output_dir / (item["category"] + "_input.npy"), raw, allow_pickle=False)
        item["input_sha256"] = sha256(args.output_dir / (item["category"] + "_input.npy"))
    write_json(args.output_dir / "protocol.json", protocol)
    protocol_hash = sha256(args.output_dir / "protocol.json")
    print("PROTOCOL_FROZEN", protocol_hash, flush=True)

    # The model is imported and loaded only after the protocol is written.
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(model_dir))
    from inference import load_model
    start = time.monotonic()
    model = load_model(args.checkpoint, "cpu")
    load_seconds = time.monotonic() - start
    print("MODEL_LOADED", round(load_seconds, 3), flush=True)
    records = []
    for item in selected:
        raw = np.load(args.output_dir / (item["category"] + "_input.npy"), allow_pickle=False)
        state = labels[item["y"]:item["y"] + 128, item["x"]:item["x"] + 128]
        for order in protocol["order_trials"]:
            samples = raw if order == "stored" else raw[::-1]
            x = torch.from_numpy(np.ascontiguousarray(samples)[None, None]).float().clamp_(0, 200).div_(255)
            start = time.monotonic()
            with torch.inference_mode():
                probability = model(x)[0, 0].float().sigmoid().numpy()
            seconds = time.monotonic() - start
            if not np.isfinite(probability).all():
                raise ValueError("Nonfinite model probabilities")
            name = item["category"] + "_" + order + "_probability.npy"
            np.save(args.output_dir / name, probability, allow_pickle=False)
            record = dict(category=item["category"], order=order, seconds=seconds,
                          central96=metrics(probability[16:112, 16:112], state[16:112, 16:112]),
                          central32=metrics(probability[48:80, 48:80], state[48:80, 48:80]),
                          output_sha256=sha256(args.output_dir / name))
            records.append(record)
            print("RESULT", json.dumps(record), flush=True)
    report = dict(protocol_sha256=protocol_hash, checkpoint_sha256=WEIGHTS_SHA256,
                  numpy_version=np.__version__, torch_version=torch.__version__, python_version=platform.python_version(),
                  load_seconds=load_seconds, inference_seconds=sum(r["seconds"] for r in records),
                  process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  results=records, scope="Three neighboring, label-selected public donor-holdout cells; correlated pixels; no T4 measurement")
    write_json(args.output_dir / "results.json", report)
    print("FINISHED", json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
