#!/usr/bin/env python3
"""CPU-only, source-pinned gA plumbing check on already published controls.

This program does not discover text, calibrate a gate, or supply independent
validation. All public control locations are selected in protocol.json first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import time

import cv2
import numpy as np
import torch

from vendor import GlyphNet, _tta, a_from_z, zscore

WEIGHT_SHA = "36ae0d6d9fdaac9d8681af75ec9be17e9317f7f77aa9422ea8ff8d09dd6f19ad"
PROTOCOL_SHA = "dfe06c991a6e9d35802fc743a1d8c1d559adeda94696e4b44bb70e81d65c8bdf"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def released_model(path: Path):
    """Verify bytes before safe tensor loading; expose only allowed metadata."""
    if sha256(path) != WEIGHT_SHA:
        raise ValueError("Checkpoint does not match the pinned released gA SHA256")
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Unexpected checkpoint container")
    arch = checkpoint.get("arch", {})
    if not isinstance(arch, dict) or set(arch) - {"cin", "w"}:
        raise ValueError("Unexpected model architecture fields")
    if arch.get("cin", 2) != 2 or arch.get("w", 24) != 24:
        raise ValueError("Unexpected released gA architecture")
    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("Unexpected checkpoint metadata type")
    allowed = {key: args.get(key) for key in ("realpos", "n", "seed", "legib")}
    if allowed != {"realpos": "e343,h841,w1447", "n": 300000, "seed": 0, "legib": False}:
        raise ValueError("Allowed checkpoint metadata differs from the pinned audit")
    model = GlyphNet(**arch).cpu().eval()
    model.load_state_dict(checkpoint["state"], strict=True)
    return model, {
        "sha256": WEIGHT_SHA,
        "architecture": {"cin": 2, "w": 24},
        "allowed_training_metadata": allowed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "load_method": "torch.load(weights_only=True, map_location='cpu'); strict state dict",
        "overlap": {
            "pherc343": "Explicitly named in actual checkpoint realpos; not held out",
            "pherc306b": "Negative training corpus not released; exclusion cannot be certified",
        },
    }


def prepare_windows(controls_root: Path, protocol: dict):
    """Preserve unknown support and reject unsupported annotation-centred crops."""
    records, images, masks, cases = [], [], [], []
    for item in protocol["cases"]:
        folder = controls_root / "cases" / item["case"]
        for filename, expected in (
            ("manifest.json", item["manifest_sha256"]),
            ("score.npy", item["score_sha256"]),
            ("validity.npy", item["mask_sha256"]),
        ):
            if sha256(folder / filename) != expected:
                raise ValueError(f"Public control integrity mismatch: {item['case']} {filename}")
        score = np.load(folder / "score.npy", allow_pickle=False)
        mask = np.load(folder / "validity.npy", allow_pickle=False)
        if score.dtype != np.float32 or mask.dtype != bool or score.shape != mask.shape or score.ndim != 2:
            raise ValueError("Expected float32 score and same-shape bool mask")
        if not np.isfinite(score).all() or score.min() < 0 or score.max() > 1:
            raise ValueError("Expected finite declared 0..1 input values")
        um_y, um_x = item["spacing_um_yx"]
        if um_y != um_x or um_x != 8.64:
            raise ValueError("This pinned control replay requires the declared 8.64 um isotropic inputs")
        values = score[mask]
        if values.size < 100 or values.std() == 0:
            raise ValueError("Cannot standardize an empty or constant map")
        prepared_a, prepared_m = a_from_z(zscore(score, mask), um_x, mask)
        cases.append({
            "case": item["case"],
            "map_kind": item["map_kind"],
            "native_shape": list(score.shape),
            "prepared_shape": list(prepared_a.shape),
            "declared_output_um": 50.0,
            "valid_native_pixels": int(values.size),
            "native_valid_mean": float(values.mean()),
            "native_valid_std": float(values.std()),
            "transformation": "Valid-map zscore then exact upstream a_from_z(down=3)",
            "limitation": ("Already contrast-adjusted display opacity; raw model signal cannot be recovered"
                           if item["map_kind"] == "display_signal" else
                           "Released final product map, not the original discovery-lane fused z map"),
        })
        for chosen in item["windows"]:
            cx, cy = chosen["center_native_xy"]
            x0, y0 = round(cx * um_x / 50 - 64), round(cy * um_y / 50 - 64)
            record = {
                "case": item["case"], "annotation_id": chosen["id"],
                "center_native_xy": [cx, cy], "crop_50um_xyxy": [x0, y0, x0 + 128, y0 + 128],
            }
            records.append(record)
            if min(x0, y0) < 0 or y0 + 128 > prepared_a.shape[0] or x0 + 128 > prepared_a.shape[1]:
                record["status"] = "rejected_out_of_bounds"
                continue
            a = prepared_a[y0:y0 + 128, x0:x0 + 128]
            m = prepared_m[y0:y0 + 128, x0:x0 + 128]
            coverage = float(m.mean())
            record["valid_fraction"] = coverage
            if coverage < protocol["min_output_mask_fraction"]:
                record["status"] = "rejected_low_coverage"
                continue
            record["status"] = "prepared"
            record["input_sha256"] = hashlib.sha256(a.tobytes() + m.tobytes()).hexdigest()
            record["mean_a_on_valid"] = float(a[m].mean())
            images.append(np.ascontiguousarray(a))
            masks.append(np.ascontiguousarray(m.astype(np.float32)))
    return records, images, masks, cases


class CpuBudgetExceeded(RuntimeError):
    pass


def _budget_handler(_signum, _frame):
    raise CpuBudgetExceeded("CPU inference budget exhausted")


def score_cpu(model, records, images, masks, budget):
    selected = [r for r in records if r["status"] == "prepared"]
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    # The 118-second soft ceiling leaves room for rounding and report generation.
    # This is Linux process CPU time, including CPU work from both torch threads.
    old_limits = resource.getrlimit(resource.RLIMIT_CPU)
    soft = math.ceil(start_cpu + budget - 2)
    if old_limits[1] != resource.RLIM_INFINITY:
        soft = min(soft, old_limits[1] - 1)
    resource.setrlimit(resource.RLIMIT_CPU, (soft, old_limits[1]))
    previous_handler = signal.signal(signal.SIGXCPU, _budget_handler)
    largest_batch_cpu = 0.0
    incomplete = False
    try:
        with torch.no_grad():
            for i in range(0, len(selected), 4):
                used = time.process_time() - start_cpu
                if used + max(5.0, largest_batch_cpu * 2) >= budget - 2:
                    incomplete = True
                    break
                a = torch.from_numpy(np.stack(images[i:i + 4])[:, None])
                m = torch.from_numpy(np.stack(masks[i:i + 4])[:, None])
                before = time.process_time()
                scores = _tta(model, a, m, True).float().cpu().numpy()
                largest_batch_cpu = max(largest_batch_cpu, time.process_time() - before)
                for record, value in zip(selected[i:i + 4], scores):
                    record["mean_tta_logit"] = float(value)
                    record["status"] = "scored"
    except CpuBudgetExceeded:
        incomplete = True
    finally:
        resource.setrlimit(resource.RLIMIT_CPU, old_limits)
        signal.signal(signal.SIGXCPU, previous_handler)
    for record in selected:
        if record["status"] == "prepared":
            record["status"] = "not_scored_cpu_budget"
    return {
        "cpu_seconds": time.process_time() - start_cpu,
        "wall_seconds": time.perf_counter() - start_wall,
        "cpu_budget_seconds": budget,
        "budget_incomplete": incomplete,
        "interpretation": "Raw mean logit only; no calibrated text probability or gate",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--controls-root", required=True, type=Path)
    p.add_argument("--protocol", default=Path(__file__).with_name("protocol.json"), type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing replay output")
    if sha256(args.protocol) != PROTOCOL_SHA:
        raise ValueError("Protocol differs from the frozen public-control protocol")
    protocol = json.loads(args.protocol.read_text())
    if sum(len(c["windows"]) for c in protocol["cases"]) != 20:
        raise ValueError("Expected exactly the 20 frozen window requests")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    model, audit = released_model(args.weights)
    records, images, masks, cases = prepare_windows(args.controls_root, protocol)
    timing = score_cpu(model, records, images, masks, protocol["inference"]["max_cpu_seconds"])
    result = {
        "schema": "public-glyph-plumbing-result-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "source_commit": protocol["source_commit"],
        "checkpoint_audit": audit,
        "implementation_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ("replay.py", "vendor.py")},
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__, "device": "cpu", "threads": 2},
        "cases": cases,
        "windows": records,
        "timing": timing,
        "limitations": protocol["prohibited_interpretations"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    counts = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    print(json.dumps({"window_status_counts": counts, "timing": timing}, indent=2))


if __name__ == "__main__":
    main()
