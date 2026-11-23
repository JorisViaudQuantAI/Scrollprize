#!/usr/bin/env python3
"""Portable, geometry-only ARAP adapter with original-triangle provenance.

The numerical optimizer is the separately hash-verified published 306B kernel.
This module adds explicit mesh contracts and standard tifxyz output. It does
not acquire CT, run an ink detector, choose a reading region or identify recto.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_mesh(path):
    path = Path(path)
    arrays = []
    for channel in "xyz":
        with Image.open(path / f"{channel}.tif") as image:
            arr = np.asarray(image)
        if arr.ndim != 2 or arr.dtype.kind != "f":
            raise ValueError("XYZ must be two-dimensional floating-point TIFFs")
        arrays.append(arr.astype(np.float64))
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("XYZ shapes differ")
    points = np.stack(arrays, axis=-1)
    valid = np.isfinite(points).all(-1) & (points[..., 2] > 0) & (points != -1).all(-1)
    if (path / "mask.tif").exists():
        with Image.open(path / "mask.tif") as image:
            mask = np.asarray(image)
        if mask.shape != valid.shape:
            raise ValueError("Explicit source mask shape differs")
        valid &= mask == 255
    points[~valid] = np.nan
    metadata = json.loads((path / "meta.json").read_text())
    return points, metadata


def solve_mesh(points, spacing_voxels, functions, *, region_mask=None,
               anchor_mask=None, iterations=60, fold_weight=0.02,
               fold_threshold=1.25, fix_index=None):
    """Use unchanged upstream numerical functions; never drop components."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 3 or p.shape[2] != 3 or min(p.shape[:2]) < 2:
        raise ValueError("Expected an HxWx3 grid")
    if not np.isfinite(spacing_voxels) or spacing_voxels <= 0:
        raise ValueError("A positive isotropic nominal carrier pitch is required")
    if not (1 <= iterations <= 60 and 0 < fold_weight <= 1 and
            np.isfinite(fold_threshold) and fold_threshold > 1):
        raise ValueError("Unsupported optimizer parameters")
    valid = np.isfinite(p).all(-1) & (p != -1).all(-1) & (p[..., 2] > 0)
    if region_mask is not None:
        if np.shape(region_mask) != valid.shape:
            raise ValueError("Region mask shape differs")
        valid &= np.asarray(region_mask, bool)
    row, col = np.nonzero(valid)
    if len(row) < 4 or len(row) > 100_000:
        raise ValueError("Require 4..100000 selected source vertices")
    _, triangles = functions["tris_of"](valid)
    triangles = np.asarray(triangles, np.int64)
    if not len(triangles) or len(triangles) > 200_000:
        raise ValueError("Require 1..200000 source triangles")
    incidence = np.bincount(triangles.ravel(), minlength=len(row))
    if (incidence == 0).any():
        raise ValueError("Isolated source vertices: split or repair explicitly; none dropped")
    first = triangles[:, [0, 1, 2]].ravel()
    second = triangles[:, [1, 2, 0]].ravel()
    graph = sparse.coo_matrix((np.ones(len(first)), (first, second)), shape=(len(row), len(row))).tocsr()
    count, _ = connected_components(graph, directed=False)
    if count != 1:
        raise ValueError(f"Selected source has {count} components; no implicit largest-component selection")
    xyz = p[valid]
    normalized = xyz / spacing_voxels
    a, b, c = (normalized[triangles[:, k]] for k in range(3))
    area3 = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)
    if not np.isfinite(area3).all() or (area3 <= 1e-12).any():
        raise ValueError("Nonfinite or degenerate source triangle")
    uv0 = np.column_stack((col, row)).astype(np.float64)
    ratio = area3 / 0.5
    folds = (ratio > fold_threshold) | (ratio < 1 / fold_threshold)
    weights = np.where(folds, fold_weight, 1.0)
    if anchor_mask is None:
        h, w = valid.shape
        anchor = (row >= .4 * (h - 1)) & (row <= .6 * (h - 1)) & (col >= .4 * (w - 1)) & (col <= .6 * (w - 1))
    else:
        if np.shape(anchor_mask) != valid.shape:
            raise ValueError("Anchor mask shape differs")
        anchor = np.asarray(anchor_mask, bool)[valid]
    if anchor.sum() < 3 or np.linalg.matrix_rank(uv0[anchor] - uv0[anchor].mean(0)) < 2:
        raise ValueError("Need at least 3 noncollinear anchor vertices")
    if fix_index is None:
        candidate = np.flatnonzero(anchor)
        fix_index = int(candidate[np.argmin(np.sum((uv0[anchor] - uv0[anchor].mean(0)) ** 2, axis=1))])
    if not 0 <= fix_index < len(xyz):
        raise ValueError("Fixed vertex index outside source")
    started = time.monotonic()
    uv = functions["arap"](normalized, triangles, uv0, iters=iterations, fix=int(fix_index), tw=weights)
    aligned = functions["procrustes"](uv[anchor], uv0[anchor])(uv)
    if not np.isfinite(aligned).all():
        raise ValueError("Optimizer produced nonfinite parameter coordinates")
    return {"xyz": xyz, "triangles": triangles, "uv_initial": uv0,
            "uv": aligned, "region_mask": valid,
            "source_ij": np.column_stack((row, col)), "anchor_mask_vertices": anchor,
            "solver_summary": {"source_vertices": len(xyz), "source_triangles": len(triangles),
                "source_components": int(count), "isolated_vertices_dropped": 0,
                "source_degenerate_triangles": 0, "fold_downweighted_triangles": int(folds.sum()),
                "iterations": iterations, "fold_weight": fold_weight, "fold_threshold": fold_threshold,
                "fix_index": int(fix_index), "anchor_vertices": int(anchor.sum()),
                "proper_rigid_alignment_without_scale": True,
                "wall_seconds": time.monotonic() - started}}


def write_mesh(path, points, valid, spacing_voxels, voxel_um, *, provenance, summary):
    """Create a standard tifxyz; metadata contains no source filesystem paths."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    valid = np.asarray(valid, bool)
    q = np.asarray(points, np.float64)
    if q.shape != valid.shape + (3,) or not np.isfinite(q[valid]).all():
        raise ValueError("Invalid output geometry/support")
    stored = np.where(valid[..., None], q, -1).astype(np.float32)
    if not np.isfinite(stored).all():
        raise ValueError("Output overflows finite float32 geometry")
    for k, name in enumerate("xyz"):
        Image.fromarray(stored[..., k]).save(path / f"{name}.tif", compression="raw")
    Image.fromarray((valid.astype(np.uint8) * 255)).save(path / "mask.tif", compression="raw")
    meta = {"format": "tifxyz", "name": "metric_reparameterized_patch",
            "scale": [1 / spacing_voxels, 1 / spacing_voxels],
            "source_voxel_um": voxel_um, "nominal_grid_spacing_voxels": spacing_voxels,
            "invalid_xyz": [-1, -1, -1], "physical_face_established": False,
            "bbox": [stored[valid].min(0).tolist(), stored[valid].max(0).tolist()] if valid.any() else None,
            "geometry_summary": summary,
            "source_provenance_file": "provenance.npz",
            "limitations": ["Geometry only; no acquired CT or ink validation",
                            "Renderer must honor support and avoid crossing invalid seams",
                            "Integer target-sample ambiguity checks are not a global injectivity proof"]}
    (path / "meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(path / "provenance.npz", **provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream-cache", type=Path, required=True)
    parser.add_argument("--spacing-voxels", type=float, required=True)
    parser.add_argument("--voxel-um", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--fold-weight", type=float, default=.02)
    parser.add_argument("--fold-threshold", type=float, default=1.25)
    args = parser.parse_args()
    from reference import load_arap_functions
    from topology import rasterize_triangles
    root = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    if out.exists() or out == root or root in out.parents:
        parser.error("Use a new output directory outside the repository")
    if not np.isfinite(args.voxel_um) or args.voxel_um <= 0:
        parser.error("voxel-um must be positive")
    def timeout(_sig, _frame):
        raise TimeoutError("600-second geometry limit")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(600)
    p, meta = read_mesh(args.mesh)
    if "source_voxel_um" in meta and not np.isclose(float(meta["source_voxel_um"]), args.voxel_um,
                                                   rtol=2 * np.finfo(np.float32).eps, atol=0):
        parser.error("voxel-um must agree with source metadata")
    old_scale = np.asarray(meta.get("scale", []), float)
    if old_scale.shape != (2,) or not np.allclose(old_scale, 1 / args.spacing_voxels,
                                               rtol=2 * np.finfo(np.float32).eps, atol=0):
        parser.error("Explicit source scale must agree with the nominal input carrier pitch")
    solution = solve_mesh(p, args.spacing_voxels, load_arap_functions(args.upstream_cache),
                          iterations=args.iterations, fold_weight=args.fold_weight,
                          fold_threshold=args.fold_threshold)
    result = rasterize_triangles(solution["uv"], solution["xyz"], solution["triangles"],
                                 shape=p.shape[:2], query_mask=solution["region_mask"])
    # Result interpretation and acceptance remain with the consuming experiment.
    write_mesh(out, result["xyz"], result["valid"], args.spacing_voxels, args.voxel_um,
               provenance={"source_ij": solution["source_ij"], "source_xyz": solution["xyz"],
                           "source_triangles": solution["triangles"],
                           "source_uv": solution["uv"], "source_triangle_id": result["source_triangle_id"],
                           "barycentric_weights": result["barycentric_weights"],
                           "valid": result["valid"], "reason_bits": result["reason_bits"]},
               summary={**solution["solver_summary"], "rasterizer": result["counts"],
                        "reason_names": result["reason_names"],
                        "code_sha256": {name: sha(Path(__file__).with_name(name))
                                        for name in ("flatten.py", "reference.py", "topology.py")},
                        "input_sha256": {name: sha(args.mesh / name) for name in
                            ("x.tif", "y.tif", "z.tif", "meta.json", "mask.tif")
                            if (args.mesh / name).exists()}})
    signal.alarm(0)
    print(json.dumps({"status": "geometry_written_for_controlled_review", "valid_nodes": int(result["valid"].sum()),
                      "new_CT_requests": 0, "CNN_forwards": 0, "reading_claim": False}))


if __name__ == "__main__":
    main()
