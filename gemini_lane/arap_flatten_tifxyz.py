# Copyright (c) 2026 Scrollprize Contributors (MIT License)
# Implements the Liu et al. (2008) local/global ARAP formulation from
# SirProdigle/pherc0306b-first-letters (submission/code/surface/reflatten.py
# @ feabff6c54f95bd8e52a90e0a7a76881ee67d4cc, MIT License).
"""Cross-platform CPU-only 2D ARAP flattening and grid-QA gating for tifxyz grids.

Unlike 1D axial arc-length resampling (``spiral_winding_resample.py``), 2D
As-Rigid-As-Possible (ARAP) parameterization removes pre-existing 2D
parallelogram shear and anisotropic stretch together before barycentric
re-sampling onto a uniform ``spacing_vx`` carrier grid.

CPU-only (zero torch/CUDA imports). Windows and POSIX compatible.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

try:
    from . import grid_quality_audit as GQA
    from . import stream_snap_render as SSR
except ImportError:  # pragma: no cover - direct script execution fallback
    import grid_quality_audit as GQA  # type: ignore[no-redef]
    import stream_snap_render as SSR  # type: ignore[no-redef]


DEFAULT_SPACING_VX = 20.0
DEFAULT_ARAP_ITERS = 100
SHEAR_P95_GATE_DEG = 30.0
SHEAR_AREA_GT_20DEG_GATE_PCT = 20.0
AREA_INFLATION_BOUNDS = (0.85, 1.15)
MAX_FLIPPED_TRIANGLE_PCT = 1.0


def compute_grid_qa(
    points: np.ndarray,
    *,
    spacing_vx: float = DEFAULT_SPACING_VX,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute ``grid_quality_audit.py`` distortion summary on ``(H, W, 3)`` XYZ."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) grid, got {pts.shape}")
    qa = GQA.grid_quality(
        pts[..., 0],
        pts[..., 1],
        pts[..., 2],
        meta=meta,
        nominal=(float(spacing_vx), float(spacing_vx)),
    )
    sh8 = qa.get("shear_8cell") or {}
    sh1 = qa.get("shear_1cell") or {}
    su = qa.get("stretch_along_u_columns") or {}
    sv = qa.get("stretch_along_v_rows") or {}
    return {
        "shape": [int(pts.shape[0]), int(pts.shape[1])],
        "cells": int(qa.get("cells", 0)),
        "shear_8cell_abs_p95_deg": sh8.get("abs_p95_deg"),
        "shear_8cell_pct_area_gt_20deg": sh8.get("pct_area_gt_20deg"),
        "shear_1cell_abs_p95_deg": sh1.get("abs_p95_deg"),
        "stretch_u_p05_p50_p95": [su.get("p05"), su.get("p50"), su.get("p95")],
        "stretch_v_p05_p50_p95": [sv.get("p05"), sv.get("p50"), sv.get("p95")],
        "area_inflation_true_over_nominal": qa.get("area_inflation_true_over_nominal"),
        "roughness": qa.get("roughness"),
    }


def passes_shear_gate(
    qa_summary: Mapping[str, Any],
    *,
    flipped_pct: float = 0.0,
) -> tuple[bool, list[str]]:
    """Check whether a grid satisfies the 2D shear, inflation, and fold gates."""
    reasons: list[str] = []
    cells = int(qa_summary.get("cells", 0))
    sh8_p95 = qa_summary.get("shear_8cell_abs_p95_deg")
    sh8_gt20 = qa_summary.get("shear_8cell_pct_area_gt_20deg")
    sh1_p95 = qa_summary.get("shear_1cell_abs_p95_deg")
    infl = qa_summary.get("area_inflation_true_over_nominal")

    if cells < 4:
        reasons.append("insufficient_valid_cells")
    shear_ref = sh8_p95 if sh8_p95 is not None else sh1_p95
    if shear_ref is None or float(shear_ref) > SHEAR_P95_GATE_DEG:
        reasons.append(f"shear_p95_gt_{SHEAR_P95_GATE_DEG:.0f}deg")
    if sh8_gt20 is not None and float(sh8_gt20) > SHEAR_AREA_GT_20DEG_GATE_PCT:
        reasons.append(f"pct_area_gt_20deg_gt_{SHEAR_AREA_GT_20DEG_GATE_PCT:.0f}pct")
    if (
        infl is None
        or not (AREA_INFLATION_BOUNDS[0] <= float(infl) <= AREA_INFLATION_BOUNDS[1])
    ):
        reasons.append("area_inflation_out_of_bounds")
    if float(flipped_pct) >= MAX_FLIPPED_TRIANGLE_PCT:
        reasons.append("flipped_triangles_gte_1pct")
    return (len(reasons) == 0), reasons


def _extract_largest_triangle_component(
    pts: np.ndarray, valid: np.ndarray, spacing_vx: float
) -> dict[str, Any]:
    """Triangulate valid quads and isolate the largest 3D-nondegenerate component."""
    H, W = valid.shape
    idx = -np.ones((H, W), dtype=np.int64)
    ii, jj = np.nonzero(valid)
    idx[ii, jj] = np.arange(len(ii), dtype=np.int64)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    q = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    if not q.any():
        raise ValueError("Surface has zero valid 4-corner quads for 2D triangulation")
    tris = np.concatenate(
        [np.stack([a[q], b[q], d[q]], axis=1), np.stack([a[q], d[q], c[q]], axis=1)],
        axis=0,
    )
    X_all = pts[valid] / float(spacing_vx)
    uv_all = np.column_stack((jj, ii)).astype(np.float64)

    p0, p1, p2 = X_all[tris[:, 0]], X_all[tris[:, 1]], X_all[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    degen_count = int((area <= 1e-8).sum())
    tris = tris[area > 1e-8]
    if len(tris) == 0:
        raise ValueError("All triangulated faces are degenerate (zero 3D area)")

    u_e = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    v_e = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    graph = sparse.coo_matrix(
        (np.ones(len(u_e)), (u_e, v_e)), shape=(len(X_all), len(X_all))
    ).tocsr()
    n_comp, labels = connected_components(graph, directed=False)
    t_lab = labels[tris[:, 0]]
    counts = np.bincount(t_lab)
    best_lab = int(counts.argmax())
    tris = tris[t_lab == best_lab]

    used_v = np.unique(tris)
    remap = -np.ones(len(X_all), dtype=np.int64)
    remap[used_v] = np.arange(len(used_v), dtype=np.int64)
    return {
        "X": X_all[used_v],
        "uv0": uv_all[used_v],
        "tris": remap[tris],
        "source_valid_vertices": int(valid.sum()),
        "kept_vertices": int(len(used_v)),
        "dropped_vertices": int(valid.sum()) - int(len(used_v)),
        "triangle_components": int((counts > 0).sum()),
        "total_graph_components": int(n_comp),
        "degenerate_triangles_dropped": degen_count,
    }


def _arap_solve_uv(
    X: np.ndarray,
    tris: np.ndarray,
    uv0: np.ndarray,
    *,
    iters: int = DEFAULT_ARAP_ITERS,
    fold_weight: float | None = None,
    fold_threshold: float = 1.25,
) -> np.ndarray:
    """Solve Liu et al. (2008) local/global cotangent ARAP and rigidly align to uv0."""
    n = len(X)
    p0, p1, p2 = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    e1 = p1 - p0
    e2 = p2 - p0
    l1 = np.linalg.norm(e1, axis=1)
    ex = e1 / l1[:, None]
    nrm = np.cross(e1, e2)
    ey = np.cross(nrm, e1)
    ey /= np.linalg.norm(ey, axis=1)[:, None]

    loc = np.zeros((len(tris), 3, 2), dtype=np.float64)
    loc[:, 1, 0] = l1
    loc[:, 2, 0] = (e2 * ex).sum(axis=1)
    loc[:, 2, 1] = (e2 * ey).sum(axis=1)

    cot = np.zeros((len(tris), 3), dtype=np.float64)
    for k in range(3):
        a_ = loc[:, (k + 1) % 3] - loc[:, k]
        b_ = loc[:, (k + 2) % 3] - loc[:, k]
        cross2 = np.abs(a_[:, 0] * b_[:, 1] - a_[:, 1] * b_[:, 0])
        cot[:, k] = (a_ * b_).sum(axis=1) / np.maximum(cross2, 1e-9)
    cot = np.clip(cot, 1e-3, 1e3) * 0.5

    if fold_weight is not None:
        area3 = 0.5 * np.linalg.norm(nrm, axis=1)
        ratio = area3 / 0.5
        folds = (ratio > fold_threshold) | (ratio < 1.0 / fold_threshold)
        cot = cot * np.where(folds, float(fold_weight), 1.0)[:, None]

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for k in range(3):
        i, j = tris[:, (k + 1) % 3], tris[:, (k + 2) % 3]
        w = cot[:, k]
        rows.extend([i, j, i, j])
        cols.extend([i, j, j, i])
        vals.extend([w, w, -w, -w])
    lap = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tolil()

    # Anchor vertex nearest the mean initial UV coordinate
    fix = int(np.argmin(np.sum((uv0 - uv0.mean(axis=0)) ** 2, axis=1)))
    lap[fix, :] = 0.0
    lap[fix, fix] = 1.0
    lu = splu(lap.tocsc())

    uv = uv0.copy()
    for _ in range(int(iters)):
        cov = np.zeros((len(tris), 2, 2), dtype=np.float64)
        for k in range(3):
            i, j = tris[:, (k + 1) % 3], tris[:, (k + 2) % 3]
            du = uv[i] - uv[j]
            dx = loc[:, (k + 1) % 3] - loc[:, (k + 2) % 3]
            cov += cot[:, k, None, None] * du[:, :, None] * dx[:, None, :]
        uq, _, vt = np.linalg.svd(cov)
        rot = uq @ vt
        neg = np.linalg.det(rot) < 0.0
        if neg.any():
            uq[neg, :, -1] *= -1.0
            rot[neg] = uq[neg] @ vt[neg]

        rhs = np.zeros((n, 2), dtype=np.float64)
        for k in range(3):
            i, j = tris[:, (k + 1) % 3], tris[:, (k + 2) % 3]
            dx = loc[:, (k + 1) % 3] - loc[:, (k + 2) % 3]
            r = np.einsum("tab,tb->ta", rot, dx) * cot[:, k, None]
            np.add.at(rhs, i, r)
            np.add.at(rhs, j, -r)
        rhs[fix] = uv0[fix]
        uv = np.column_stack([lu.solve(rhs[:, 0]), lu.solve(rhs[:, 1])])
        if not np.isfinite(uv).all():
            raise ValueError("ARAP solver produced nonfinite UV coordinates")

    # Rigid Procrustes alignment (no scale) back to initial UV frame
    m0 = uv0.mean(axis=0)
    m1 = uv.mean(axis=0)
    up, _, vtp = np.linalg.svd((uv - m1).T @ (uv0 - m0))
    rp = up @ vtp
    if np.linalg.det(rp) < 0.0:
        up[:, -1] *= -1.0
        rp = up @ vtp
    return (uv - m1) @ rp + m0


def _rasterize_barycentric_grid(
    X_vx: np.ndarray,
    tris: np.ndarray,
    uv: np.ndarray,
) -> dict[str, Any]:
    """Rasterize positively oriented UV triangles onto unit integer UV coordinates."""
    u0, u1, u2 = uv[tris[:, 0]], uv[tris[:, 1]], uv[tris[:, 2]]
    det2 = (u1[:, 0] - u0[:, 0]) * (u2[:, 1] - u0[:, 1]) - (u1[:, 1] - u0[:, 1]) * (
        u2[:, 0] - u0[:, 0]
    )
    sgn = 1.0 if int((det2 > 0).sum()) >= int((det2 < 0).sum()) else -1.0
    flipped = int((det2 * sgn <= 0.0).sum())

    umin, vmin = np.floor(uv.min(axis=0)).astype(np.int64)
    umax, vmax = np.ceil(uv.max(axis=0)).astype(np.int64)
    out_w = int(umax - umin + 1)
    out_h = int(vmax - vmin + 1)
    if out_w < 2 or out_h < 2 or out_w * out_h > 40_000_000:
        raise ValueError(f"Unreasonable rasterized UV grid dimensions: {(out_h, out_w)}")

    out_xyz = np.full((out_h, out_w, 3), -1.0, dtype=np.float64)
    out_valid = np.zeros((out_h, out_w), dtype=bool)
    for ti in range(len(tris)):
        d2 = det2[ti] * sgn
        if d2 <= 1e-9:
            continue
        p = uv[tris[ti]]
        c0 = max(0, int(np.floor(p[:, 0].min())) - int(umin))
        c1 = min(out_w - 1, int(np.ceil(p[:, 0].max())) - int(umin))
        r0 = max(0, int(np.floor(p[:, 1].min())) - int(vmin))
        r1 = min(out_h - 1, int(np.ceil(p[:, 1].max())) - int(vmin))
        if c1 < c0 or r1 < r0:
            continue
        cc = np.arange(c0, c1 + 1, dtype=np.float64) + float(umin)
        rr = np.arange(r0, r1 + 1, dtype=np.float64) + float(vmin)
        grid_c, grid_r = np.meshgrid(cc, rr)
        ax, ay = p[0]
        bx, by = p[1]
        cx, cy = p[2]
        w0 = ((by - cy) * (grid_c - cx) + (cx - bx) * (grid_r - cy)) * sgn / d2
        w1 = ((cy - ay) * (grid_c - cx) + (ax - cx) * (grid_r - cy)) * sgn / d2
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if inside.any():
            x0, x1, x2 = X_vx[tris[ti, 0]], X_vx[tris[ti, 1]], X_vx[tris[ti, 2]]
            interp = w0[inside, None] * x0 + w1[inside, None] * x1 + w2[inside, None] * x2
            rr_i, cc_i = np.nonzero(inside)
            out_xyz[r0 + rr_i, c0 + cc_i] = interp
            out_valid[r0 + rr_i, c0 + cc_i] = True

    return {
        "points": out_xyz,
        "valid": out_valid,
        "flipped_triangles": flipped,
        "total_triangles": int(len(tris)),
        "flipped_pct": round(100.0 * flipped / max(1, len(tris)), 4),
    }


def arap_flatten_grid(
    points: np.ndarray,
    valid: np.ndarray,
    *,
    spacing_vx: float = DEFAULT_SPACING_VX,
    iters: int = DEFAULT_ARAP_ITERS,
    fold_weight: float | None = None,
    fold_threshold: float = 1.25,
    source_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """2D-ARAP flatten an ``(H, W, 3)`` tifxyz grid and report before/after QA."""
    if not (spacing_vx > 0 and np.isfinite(spacing_vx)):
        raise ValueError("spacing_vx must be positive and finite")
    if not (1 <= int(iters) <= 300):
        raise ValueError("iters must be in [1, 300]")
    pts = np.asarray(points, dtype=np.float64)
    val = np.asarray(valid, dtype=bool)
    if pts.ndim != 3 or pts.shape[2] != 3 or pts.shape[:2] != val.shape:
        raise ValueError(f"Mismatched points {pts.shape} and valid {val.shape}")

    qa_before = compute_grid_qa(pts, spacing_vx=spacing_vx, meta=source_meta)
    t0 = time.perf_counter()
    comp = _extract_largest_triangle_component(pts, val, spacing_vx)
    uv = _arap_solve_uv(
        comp["X"],
        comp["tris"],
        comp["uv0"],
        iters=iters,
        fold_weight=fold_weight,
        fold_threshold=fold_threshold,
    )
    rast = _rasterize_barycentric_grid(comp["X"] * float(spacing_vx), comp["tris"], uv)
    qa_after = compute_grid_qa(rast["points"], spacing_vx=spacing_vx)
    passed, gate_reasons = passes_shear_gate(qa_after, flipped_pct=rast["flipped_pct"])

    return {
        "points": rast["points"],
        "valid": rast["valid"],
        "qa_before": qa_before,
        "qa_after": qa_after,
        "passes_shear_gate": passed,
        "gate_failure_reasons": gate_reasons,
        "solver": {
            "method": "mit_arap_2d",
            "spacing_vx": float(spacing_vx),
            "iters": int(iters),
            "fold_weight": fold_weight,
            "wall_seconds": round(time.perf_counter() - t0, 3),
            "source_valid_vertices": comp["source_valid_vertices"],
            "kept_vertices": comp["kept_vertices"],
            "dropped_vertices": comp["dropped_vertices"],
            "triangle_components": comp["triangle_components"],
            "total_triangles": rast["total_triangles"],
            "flipped_triangles": rast["flipped_triangles"],
            "flipped_pct": rast["flipped_pct"],
            "output_valid_vertices": int(rast["valid"].sum()),
        },
    }


def _fetch_remote_tifxyz(url: str, dst: Path) -> None:
    """Download x/y/z.tif (+ optional meta.json/mask.tif) from an HTTP(S) prefix."""
    base = url.rstrip("/") + "/"
    dst.mkdir(parents=True, exist_ok=True)
    for fname in ("x.tif", "y.tif", "z.tif", "meta.json", "mask.tif"):
        req = urllib.request.Request(base + fname, headers={"User-Agent": "scrollprize-arap/1"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                (dst / fname).write_bytes(resp.read())
        except Exception:
            if fname in ("x.tif", "y.tif", "z.tif"):
                raise


def flatten_tifxyz_to_dir(
    mesh_src: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    voxel_um: float,
    spacing_vx: float = DEFAULT_SPACING_VX,
    iters: int = DEFAULT_ARAP_ITERS,
    fold_weight: float | None = None,
) -> dict[str, Any]:
    """2D-ARAP flatten a local dir or remote HTTPS tifxyz into a private directory.

    Refuses output paths inside any Git repository worktree. Writes standard
    ``x.tif``, ``y.tif``, ``z.tif``, ``mask.tif``, and sanitized ``meta.json``.
    """
    out_path = SSR._private_output_path(out_dir)
    parsed = urllib.parse.urlparse(str(mesh_src))
    if parsed.scheme in ("http", "https"):
        with tempfile.TemporaryDirectory(prefix="arap-fetch-") as td:
            src_dir = Path(td) / "src"
            _fetch_remote_tifxyz(str(mesh_src), src_dir)
            x, y, z, valid, src_meta = SSR.read_tifxyz(src_dir)
    else:
        x, y, z, valid, src_meta = SSR.read_tifxyz(mesh_src)

    pts = np.stack([x, y, z], axis=-1).astype(np.float64)
    res = arap_flatten_grid(
        pts,
        valid,
        spacing_vx=spacing_vx,
        iters=iters,
        fold_weight=fold_weight,
        source_meta=src_meta,
    )

    out_pts = res["points"]
    out_val = res["valid"]
    out_meta = {
        "format": "tifxyz",
        "type": "seg",
        "scale": [1.0 / float(spacing_vx), 1.0 / float(spacing_vx)],
        "voxel_um": float(voxel_um),
        "parameterization_method": "mit_arap_2d",
        "source_scale": src_meta.get("scale"),
        "passes_shear_gate": bool(res["passes_shear_gate"]),
        "qa_before": {k: v for k, v in res["qa_before"].items() if k != "roughness"},
        "qa_after": {k: v for k, v in res["qa_after"].items() if k != "roughness"},
        "solver": res["solver"],
    }
    SSR.write_tifxyz(
        out_path,
        out_pts[..., 0],
        out_pts[..., 1],
        out_pts[..., 2],
        meta=out_meta,
    )
    Image.fromarray((out_val.astype(np.uint8) * 255)).save(out_path / "mask.tif")
    (out_path / "arap_qa.json").write_text(
        json.dumps(
            {
                "passes_shear_gate": res["passes_shear_gate"],
                "gate_failure_reasons": res["gate_failure_reasons"],
                "qa_before": res["qa_before"],
                "qa_after": res["qa_after"],
                "solver": res["solver"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "voxel_um": float(voxel_um),
        "passes_shear_gate": res["passes_shear_gate"],
        "gate_failure_reasons": res["gate_failure_reasons"],
        "qa_before": res["qa_before"],
        "qa_after": res["qa_after"],
        "solver": res["solver"],
    }
