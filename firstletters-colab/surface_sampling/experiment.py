#!/usr/bin/env python3
"""Frozen CPU rendering experiment: public geometry + new physical CT phantom.

The old PHerc. 306B geometry is an already-examined engineering replay. The
NIST pixels must not be accessed before this implementation and protocol are
committed. Generated charts are numerical sampling fixtures, not recovered
papyrus sheets. No labels, ink inference, model training or reading is claimed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import PIL
from surface import OriginalSurface
from render import render
import nist_source

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def outside_git(path):
    path = Path(path).resolve()
    if any((p / ".git").exists() for p in (path, *path.parents)):
        raise ValueError("Raw data and maps must stay outside every Git checkout")
    return path


def frozen(commit, protocol):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("A full public freeze commit is required")
    result = {}
    for name in protocol["frozen_files"]:
        path = (HERE / name).resolve()
        relative = path.relative_to(ROOT).as_posix()
        body = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
        if body != path.read_bytes():
            raise ValueError(f"Source differs from the public freeze: {name}")
        result[name] = sha(path)
    if result["nist_manifest.json"] != protocol["source_manifest_sha256"]:
        raise ValueError("NIST manifest is not the one selected in the protocol")
    return result


def fixture(shape_zyx, kind):
    """Two fixed charts selected by dimensions alone, before CT intensities."""
    zsize, ysize, xsize = shape_zyx
    if min(shape_zyx) < 64:
        raise ValueError("Physical phantom is too small for the frozen fixtures")
    u, v = np.meshgrid(np.arange(-4., 69., 4.), np.arange(-4., 69., 4.))
    tu, tv = (u - 32.) / 32., (v - 32.) / 32.
    x = (xsize - 1) * (.5 + .20 * tu + .06 * tv)
    y = (ysize - 1) * (.5 + .02 * tu + .08 * tv)
    if kind == "affine_oblique":
        z = (zsize - 1) / 2 + 10 * tu + 4 * tv
    elif kind == "curved":
        z = (zsize - 1) / 2 + 12 * np.sin(np.pi * tu / 2) + 8 * tv ** 2
    else:
        raise ValueError("Unknown frozen fixture")
    width = u.shape[1]
    triangles = []
    for yy in range(u.shape[0] - 1):
        for xx in range(width - 1):
            a = yy * width + xx
            triangles.extend(((a, a + 1, a + width + 1),
                              (a, a + width + 1, a + width)))
    return OriginalSurface(np.column_stack((u.ravel(), v.ravel())),
                           np.column_stack((x.ravel(), y.ravel(), z.ravel())),
                           np.asarray(triangles, np.int64))


def error(candidate, reference, mask):
    difference = candidate.astype(np.float64)[:, mask] - reference.astype(np.float64)[:, mask]
    if not difference.size:
        return {"values": 0, "rmse": None, "mae": None, "max_abs": None}
    return {"values": int(difference.size), "rmse": float(np.sqrt(np.mean(difference ** 2))),
            "mae": float(np.mean(np.abs(difference))), "max_abs": float(np.max(np.abs(difference)))}


def metric_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def geometry_replay(cache, protocol, out):
    geometry = protocol["geometry_replay"]
    for name, expected in geometry["sha256"].items():
        if sha(cache / name) != expected:
            raise ValueError(f"Previously published geometry changed: {name}")
    with np.load(cache / "replay_geometry.npz", allow_pickle=False) as data:
        surface = OriginalSurface(data["uv"], data["source_xyz"], data["source_triangles"])
        region = data["region"].copy()
    with np.load(cache / "C_rasterization.npz", allow_pickle=False) as data:
        old_valid = data["valid"].copy()
        old_xyz = data["raw_xyz"].copy()
        old_face = data["source_triangle_id"].copy()
        old_reason = data["reason_bits"].copy()
    direct = surface.sample_grid((0., 0.), (1., 1.), old_valid.shape, query_mask=region)
    common = direct["valid"] & old_valid
    delta = float(np.max(np.abs(direct["xyz"][common] - old_xyz[common]))) if common.any() else None
    result = {"source": "already examined public PHerc. 306B geometry; no new CT", "retrospective": True,
              "source_vertices": int(len(surface.uv)), "source_faces": int(len(surface.triangles)),
              "old_valid_nodes": int(old_valid.sum()), "new_valid_nodes": int(direct["valid"].sum()),
              "support_mismatches": int(np.count_nonzero(old_valid != direct["valid"])),
              "reason_mismatches": int(np.count_nonzero(old_reason != direct["reason_bits"])),
              "face_mismatches_on_common": int(np.count_nonzero(old_face[common] != direct["source_triangle_id"][common])),
              "max_absolute_xyz_error_voxels": delta,
              "previous_99pct_retention_gate": "FAIL, unchanged; this is a different engineering contract",
              "patches": []}
    result["pass"] = (result["support_mismatches"] == result["reason_mismatches"] == result["face_mismatches_on_common"] == 0
                      and delta is not None and delta <= 1e-6)
    for i, patch in enumerate(geometry["patches"]):
        origin, shape = patch["origin_uv"], patch["shape_yx"]
        points = surface.sample_grid(origin, (1., 1.), shape)
        whole = surface.footprint_support(origin, (1., 1.), shape)
        dense = surface.sample_grid(np.asarray(origin) - .375, (.25, .25),
                                    tuple(4 * v for v in shape))
        dense_support = dense["valid"].reshape(shape[0], 4, shape[1], 4).all(axis=(1, 3))
        np.savez_compressed(out / f"geometry_patch_{i}.npz", center_valid=points["valid"],
                            footprint_valid=whole["valid"], footprint_reason=whole["reason_bits"],
                            four_by_four_all_valid=dense_support)
        result["patches"].append({**patch, "center_valid": int(points["valid"].sum()),
                                   "all_16_points_valid": int(dense_support.sum()),
                                   "whole_footprint_valid": int(whole["valid"].sum()),
                                   "point_only_not_footprint": int((points["valid"] & ~whole["valid"]).sum()),
                                   "all_16_but_not_footprint": int((dense_support & ~whole["valid"]).sum()),
                                   "whole_support": whole["diagnostics"]})
    return result


def execute(args):
    protocol = json.loads((HERE / "protocol.json").read_text())
    hashes = frozen(args.freeze_commit, protocol)
    out = outside_git(args.out)
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {"schema": "scrollprize.surface-sampling.result.v1", "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(), "freeze_commit": args.freeze_commit,
              "frozen_sha256": hashes, "new_reading_claim": False,
              "unread_scroll_regions": 0, "CNN_forwards": 0, "training_updates": 0,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "pillow": PIL.__version__, "requested_numeric_threads": 1},
              "cases": [], "stages": [], "gates": {}}

    def checkpoint(stage):
        elapsed = time.monotonic() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        status = {"stage": stage, "wall_seconds": elapsed, "peak_rss_bytes": rss}
        report["stages"].append(status)
        dump(out / "progress.json", report)
        print(json.dumps(status), flush=True)
        if elapsed > protocol["budgets"]["wall_seconds"]:
            raise TimeoutError("Frozen wall-time budget exceeded")
        if rss > protocol["budgets"]["peak_rss_bytes"]:
            raise MemoryError("Frozen stage-boundary RSS budget exceeded")

    def timeout(_signal, _frame):
        raise TimeoutError("Frozen wall-time budget exceeded")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(protocol["budgets"]["wall_seconds"])
    try:
        dump(out / "protocol.json", protocol)
        report["geometry_replay"] = geometry_replay(Path(args.geometry_cache), protocol, out)
        report["gates"]["geometry_identity"] = report["geometry_replay"]["pass"]
        checkpoint("public_geometry_replay_complete")
        if not report["gates"]["geometry_identity"]:
            raise RuntimeError("Public geometry replay failed; do not acquire new CT")
        volume, receipt = nist_source.acquire(args.source_cache, freeze_path=HERE / "protocol.json",
                                             freeze_sha256=sha(HERE / "protocol.json"))
        report["acquisition"] = receipt
        np.save(Path(args.source_cache) / "stack.npy", volume, allow_pickle=False)
        dump(out / "source_receipt.json", receipt)
        report["data"] = {"shape_zyx": list(volume.shape), "dtype": str(volume.dtype),
                          "minimum": float(volume.min()), "maximum": float(volume.max()),
                          "source_stack_sha256": sha(Path(args.source_cache) / "stack.npy"),
                          "nominal_voxel_um": 130, "local_volume_origin_xyz": [0, 0, 0],
                          "local_z0_released_slice_index": 246,
                          "scope": "modern physical lead-ink phantom, not an ancient ink control"}
        checkpoint("nist_ct_acquired")
        settings = protocol["numerical_rendering"]
        for kind in settings["fixtures"]:
            surface = fixture(volume.shape, kind)
            np.savez_compressed(out / f"fixture_{kind}.npz", uv=surface.uv, xyz=surface.xyz,
                                triangles=surface.triangles)
            for phase in settings["phase_offsets"]:
                label = f"{kind}_phase_{phase:.2f}"
                origin = np.array([.5 + phase, .5 + phase])
                arms, masks, diagnostics, times = {}, {}, {}, {}
                for n in (1, 4, 8):
                    t = time.monotonic()
                    arms[n], masks[n], diagnostics[n] = render(volume, (0, 0, 0), surface, origin,
                                                               (1, 1), (64, 64), settings["depths_voxels"], n)
                    times[n] = time.monotonic() - t
                common = masks[1] & masks[4] & masks[8]
                e1, e4 = error(arms[1], arms[8], common), error(arms[4], arms[8], common)
                # A grid refinement using the SAME renderer: each pixel is split
                # into four half-step cells, each with its own 8x8 quadrature.
                # Their equal-area average samples the 16x16 midpoint locations.
                # Each half-cell was already rounded to float32; therefore this
                # is a composite reference, not a once-rounded exact 16x16 sum.
                fine_origin = origin + 28 - .25
                fine, fine_mask, fine_diagnostics = render(volume, (0, 0, 0), surface, fine_origin,
                                                          (.5, .5), (16, 16), settings["depths_voxels"], 8)
                reference16 = fine.astype(np.float64).reshape(len(settings["depths_voxels"]), 8, 2, 8, 2).mean(axis=(2, 4))
                reference16 = reference16.astype(np.float32)
                refined_mask = fine_mask.reshape(8, 2, 8, 2).all(axis=(1, 3)) & common[28:36, 28:36]
                e16 = {n: error(arms[n][:, 28:36, 28:36], reference16, refined_mask) for n in (1, 4, 8)}
                ratio = metric_ratio(e4["rmse"], e1["rmse"]) if e1["rmse"] is not None else None
                case = {"fixture": kind, "phase_offset_uv": phase, "common_pixels": int(common.sum()),
                        "common_fraction": float(common.mean()), "rmse_point_vs_8": e1,
                        "rmse_4_vs_8": e4, "rmse_ratio_4_over_point": ratio,
                        "fixed_center_8x8_reference16": {"common_pixels": int(refined_mask.sum()),
                                                          "errors": e16, "diagnostics": fine_diagnostics},
                        "render_seconds": times, "render_diagnostics": diagnostics}
                case["gates"] = {
                    "support": case["common_fraction"] >= settings["min_common_fraction"] and int(refined_mask.sum()) == 64,
                    "four_better_than_point": ratio is not None and ratio < 1,
                    "four_rmse_ratio_at_most_half": ratio is not None and ratio <= .5,
                    "dense_reference_convergence": (e16[1]["rmse"] is not None and e16[1]["rmse"] > 0
                                                     and e16[8]["rmse"] < e16[4]["rmse"] < e16[1]["rmse"]),
                }
                np.savez_compressed(out / f"{label}.npz", point=arms[1], four=arms[4], eight=arms[8],
                                    point_valid=masks[1], four_valid=masks[4], eight_valid=masks[8],
                                    common=common, reference16=reference16, refined_mask=refined_mask)
                report["cases"].append(case)
                checkpoint(label)
        report["gates"]["fixture_support"] = all(c["gates"]["support"] for c in report["cases"])
        report["gates"]["four_point_convergence"] = all(c["gates"]["four_rmse_ratio_at_most_half"] for c in report["cases"])
        report["gates"]["reference_convergence"] = all(c["gates"]["dense_reference_convergence"] for c in report["cases"])
        report["numerical_engineering_gate_pass"] = all(report["gates"].values())
        report["status"] = "complete"
        report["interpretation_limit"] = "Numerical integration convergence only; generated charts are not traced papyrus sheets, the 16x16 rule is not continuous truth, and no ink efficacy is measured."
    except Exception as failure:
        report["status"] = "failed"
        report["failure"] = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        signal.alarm(0)
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        report["wall_seconds"] = time.monotonic() - started
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["artifacts"] = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                               for p in sorted(out.glob("*.npz"))}
        dump(out / "result.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--geometry-cache", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
