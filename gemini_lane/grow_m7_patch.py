# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""CPU-only m7 patch grower for zero-mesh scrolls (PHerc0846A, PHerc1545).

Builds on ``stream_snap_render.py`` (source-support version, commit 4f347c1):

1. Stream one interior box of a binary m7 surface prediction (``RemoteZarrV2``).
2. Extract one smooth, connected, single-sheet height field from that box:
   26-connected m7 components, columns along the dominant normal axis that cross
   the component exactly once, a local-median smoothness test, and the largest
   4-connected region of good columns.
3. Sample it on a regular grid into a sentinel-clean ``tifxyz`` patch
   (``-1.0`` on all three channels for invalid vertices, ``scale=[0.05, 0.05]``).
4. Apply ``snap_tifxyz_to_m7`` (default two passes).
5. Render a CT surface stack and report ``onsheet_score`` (``f_parallel``) and
   ``valid_ct_cm2``, then gate on ``f_parallel >= 0.70`` and ``valid_ct_cm2 >= 1.0``.

All coordinates live in one *target frame* (for PHerc0846A: m7-L2 level 0, i.e.
the 2.403 um scan's pyramid level 2 at 9.612 um). Coarser pyramid levels enter
through :class:`ScaledReader`, which maps target-frame ``xyz`` to a level with an
integer down-sampling factor. A shared :class:`ByteBudget` hard-caps every byte
streamed (metadata included). No chunk ever touches disk; the patch is written
only to a private directory outside every Git worktree, and the public receipt
carries no coordinates, images, ink predictions or local paths.

The gate is a geometric/coherence preflight. It does not certify that the patch
is papyrus rather than a mask boundary, nor that ink is present.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import stream_snap_render as SSR  # noqa: E402

REPO_ROOT = HERE.parent
ONSHEET_PATH = REPO_ROOT / "firstletters-colab" / "onsheet_score" / "onsheet_score.py"

S3_BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
F_PARALLEL_MIN = 0.70
VALID_CT_CM2_MIN = 1.0
DEFAULT_SCALE = (0.05, 0.05)

# Public dataset names (no local paths). Frame level = CT pyramid level that m7 level 0 matches.
SCROLL_SPECS: dict[str, dict[str, Any]] = {
    "PHerc0846A": {
        "m7": "PHerc0846A/representations/predictions/surfaces/"
              "20260319102732-surface-20260413222639-surface-m7-L2-th0.2.zarr",
        "normal_grids": "PHerc0846A/representations/predictions/surfaces/"
                        "20260319102732-surface-20260413222639-surface-m7-L2-th0.2.normal-grids",
        "ct": "PHerc0846A/volumes/20260319102732-2.403um-0.2m-77keV-masked.zarr",
        "ct_level0_um": 2.403,
        "m7_frame_ct_level": 2,
        "lasagna": None,
    },
    "PHerc1545": {
        "m7": "PHerc1545/representations/predictions/surfaces/"
              "20250821151648-surface-20260413222639-surface-m7-L0-th0.2.zarr",
        "normal_grids": "PHerc1545/representations/predictions/surfaces/"
                        "20250821151648-surface-20260413222639-surface-m7-L0-th0.2.normal-grids",
        "ct": "PHerc1545/volumes/20250821151648-9.362um-1.2m-113keV-masked.zarr",
        "ct_level0_um": 9.362,
        "m7_frame_ct_level": 0,
        "lasagna": "PHerc1545/representations/predictions/lasagna/20250821151648-lasagna-20260419180421",
    },
}

# Keys allowed in the public receipt. Anything else (bbox, origin, seeds, paths) is dropped.
PUBLIC_METRIC_KEYS = frozenset({
    "grid_shape", "valid_vertex_count", "valid_vertex_fraction", "seeds_tried", "tracked_columns",
    "patch_column_fraction",
    "height_axis", "single_run_column_fraction", "smooth_column_fraction",
    "patch_columns", "max_abs_height_step_vx", "median_abs_slope", "p95_abs_slope",
    "mesh_area_cm2", "connected_components_in_grid",
    "on_sheet_raw", "on_sheet_after_snap", "median_disp_vx", "p90_disp_vx",
    "center_vs_outer_disp_diff", "passes", "search_radius", "step", "threshold",
    "m7_source_available_fraction_raw", "m7_source_available_fraction_after_snap",
    "valid_ct_px", "valid_ct_cm2", "render_px", "num_slices", "slice_step",
    "f_parallel", "f_drift", "f_still", "tiles", "response_median",
    "shift_median_confident", "backend", "dk", "zero_shift_dominant_fraction",
    "extracted_valid_vertex_count", "extracted_mesh_area_cm2", "ct_chunks_estimated",
    "extract_seconds", "render_seconds", "onsheet_step", "onsheet_tile", "r_t", "s_t", "min_valid",
})


# ---------------------------------------------------------------------------
# Byte budget and level-scaled readers
# ---------------------------------------------------------------------------

class BudgetExceeded(RuntimeError):
    """Raised before or after a fetch that would take streamed bytes past the cap."""


class ByteBudget:
    """Shared hard cap on raw bytes streamed (compressed bytes as transferred)."""

    def __init__(self, cap_bytes: int) -> None:
        if cap_bytes <= 0:
            raise ValueError("cap_bytes must be > 0")
        self.cap = int(cap_bytes)
        self.used = 0
        self.requests = 0

    @property
    def remaining(self) -> int:
        return self.cap - self.used

    def precheck(self, expected: int = 0) -> None:
        if self.used + max(0, int(expected)) > self.cap:
            raise BudgetExceeded(
                f"byte budget: {self.used} used + {int(expected)} expected > cap {self.cap}"
            )

    def charge(self, n: int) -> None:
        self.used += int(n)
        self.requests += 1
        if self.used > self.cap:
            raise BudgetExceeded(f"byte budget exceeded: {self.used} > {self.cap}")


class BudgetedRemoteZarrV2(SSR.RemoteZarrV2):
    """``RemoteZarrV2`` whose every fetched key (metadata and chunks) is charged to a budget.

    Uncompressed arrays are pre-checked against the known chunk size, so the cap is
    never crossed by them; compressed chunks are charged after transfer.
    """

    def __init__(self, root, level="0", *, budget: ByteBudget, **kw) -> None:
        self.budget = budget
        self._precheck_bytes = 0
        super().__init__(root, level, **kw)
        if self._decode is None:
            self._precheck_bytes = self._expected_chunk_bytes

    def _get_bytes(self, key: str, *, allow_404: bool = True) -> bytes | None:
        self.budget.precheck(self._precheck_bytes)
        raw = super()._get_bytes(key, allow_404=allow_404)
        if raw is not None:
            self.budget.charge(len(raw))
        return raw


def precise_sample_trilinear(reader, xyz: np.ndarray, *, return_support: bool = False):
    """Trilinear sample that fetches ONLY chunks holding positive-weight corners.

    Same value/support semantics as ``RemoteZarrV2.sample_trilinear(return_support=True)``
    (out-of-bounds -> fill and unsupported; missing chunk -> fill value, unsupported;
    non-finite stored value -> 0, unsupported), but without its dense bounding-box /
    margin reads, which can pull neighbouring chunks that carry zero weight. Needed to
    keep uncompressed 2 MiB CT chunks inside a strict byte budget.
    """
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.shape[-1] != 3:
        raise ValueError(f"Expected last dimension 3 for xyz, got shape {pts.shape}")
    orig = pts.shape[:-1]
    flat = pts.reshape(-1, 3)
    fill = float(reader.fill_value)
    out = np.full(flat.shape[0], fill, dtype=np.float32)
    avail = np.zeros(flat.shape[0], dtype=bool)

    def result():
        v = out.reshape(orig)
        return (v, avail.reshape(orig)) if return_support else v

    sz, sy, sx = reader.shape
    with np.errstate(invalid="ignore"):
        inb = (np.isfinite(flat).all(axis=1)
               & (flat[:, 0] >= 0) & (flat[:, 0] <= sx - 1)
               & (flat[:, 1] >= 0) & (flat[:, 1] <= sy - 1)
               & (flat[:, 2] >= 0) & (flat[:, 2] <= sz - 1))
    idx = np.flatnonzero(inb)
    if idx.size == 0:
        return result()
    q = flat[idx][:, ::-1]
    f0 = np.floor(q).astype(np.int64)
    fr = q - f0
    C = np.asarray(reader.chunks, dtype=np.int64)
    nchunk = -(-np.asarray(reader.shape, dtype=np.int64) // C)
    corners = []
    for dz in (0, 1):
        wz = fr[:, 0] if dz else 1.0 - fr[:, 0]
        for dy in (0, 1):
            wy = fr[:, 1] if dy else 1.0 - fr[:, 1]
            for dx in (0, 1):
                wx = fr[:, 2] if dx else 1.0 - fr[:, 2]
                w = wz * wy * wx
                pos = np.flatnonzero(w > 0.0)
                c = f0[pos] + np.array([dz, dy, dx], dtype=np.int64)
                cid = c // C
                key = (cid[:, 0] * nchunk[1] + cid[:, 1]) * nchunk[2] + cid[:, 2]
                corners.append((pos, w[pos], c, cid, key))
    keys = np.unique(np.concatenate([k for *_, k in corners]))
    def unkey(k):
        return (int(k // (nchunk[1] * nchunk[2])), int((k // nchunk[2]) % nchunk[1]), int(k % nchunk[2]))
    chunks = reader.get_chunks([unkey(k) for k in keys])
    acc = np.zeros(idx.size, dtype=np.float64)
    ok = np.ones(idx.size, dtype=bool)
    for pos, w, c, cid, key in corners:
        if pos.size == 0:
            continue
        vals = np.full(pos.size, fill, dtype=np.float64)
        have = np.zeros(pos.size, dtype=bool)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        bounds = np.flatnonzero(np.concatenate(([True], ks[1:] != ks[:-1], [True])))
        for s, e in zip(bounds[:-1], bounds[1:]):
            sel = order[s:e]
            arr = chunks[unkey(ks[s])]
            if arr is None:
                continue
            loc = c[sel] - cid[sel] * C
            v = arr[loc[:, 0], loc[:, 1], loc[:, 2]].astype(np.float64)
            fin = np.isfinite(v)
            vals[sel] = np.where(fin, v, 0.0)
            have[sel] = fin
        acc[pos] += w * vals
        ok[pos] &= have
    out[idx] = acc.astype(np.float32)
    avail[idx] = ok
    return result()


class ScaledReader:
    """Present a coarser pyramid level in a finer target frame.

    ``factor`` = target voxels per reader voxel (e.g. 2 for CT level 3 under a level-2
    frame). Target index ``i`` maps to reader index ``i / factor`` (OME level ``k``
    voxel ``j`` is centred on level-0 voxel ``j * 2**k`` for stride/nearest pyramids).
    Use ``factor=1`` for exact-chunk sampling of a reader in its own frame.
    """

    def __init__(self, reader, factor: float) -> None:
        if not (factor > 0 and math.isfinite(factor)):
            raise ValueError("factor must be finite and > 0")
        self.reader = reader
        self.factor = float(factor)
        self.dtype = reader.dtype
        self.fill_value = reader.fill_value
        self.shape = tuple(int(math.floor((s - 1) * self.factor)) + 1 for s in reader.shape)

    @property
    def bytes_fetched(self) -> int:
        return int(self.reader.bytes_fetched)

    @property
    def chunks_fetched(self) -> int:
        return int(self.reader.chunks_fetched)

    def sample_trilinear(self, xyz: np.ndarray, *, return_support: bool = False, **_ignored):
        """Exact-chunk trilinear sample (see :func:`precise_sample_trilinear`)."""
        return precise_sample_trilinear(
            self.reader, np.asarray(xyz, dtype=np.float64) / self.factor, return_support=return_support
        )

    def close(self) -> None:
        self.reader.close()


def chunk_indices_for_points(
    xyz: np.ndarray, shape_zyx: Sequence[int], chunks_zyx: Sequence[int]
) -> set[tuple[int, int, int]]:
    """Chunk indices touched by trilinear sampling at reader-frame ``xyz`` points."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        return set()
    zyx = pts[:, ::-1]
    shp = np.asarray(shape_zyx, dtype=np.int64)
    ch = np.asarray(chunks_zyx, dtype=np.int64)
    out: set[tuple[int, int, int]] = set()
    base = np.floor(zyx).astype(np.int64)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                c = base + np.array([dz, dy, dx])
                ok = ((c >= 0) & (c < shp)).all(axis=1)
                idx = np.unique(c[ok] // ch, axis=0)
                out.update(tuple(int(v) for v in row) for row in idx)
    return out


# ---------------------------------------------------------------------------
# Single-sheet height-field extraction
# ---------------------------------------------------------------------------

def dominant_normal_axis(binary_zyx: np.ndarray, sigma: float = 2.0) -> int:
    """Return the zyx axis (0/1/2) most aligned with the mean structure-tensor normal."""
    f = ndi.gaussian_filter(np.asarray(binary_zyx, dtype=np.float32), sigma)
    g = np.gradient(f)
    t = np.array([[float(np.mean(g[i] * g[j])) for j in range(3)] for i in range(3)])
    w, v = np.linalg.eigh(t)
    return int(np.argmax(np.abs(v[:, int(np.argmax(w))])))


def _largest_component_2d(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def column_runs(m: np.ndarray, max_runs: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Per-column run centres and lengths along axis 0 of a boolean ``(D, A, B)`` array.

    Returns ``(centres, lengths)`` of shape ``(A, B, K)``, NaN / 0 padded, K <= max_runs.
    """
    d = m.shape[0]
    pad = np.zeros((1,) + m.shape[1:], dtype=bool)
    mm = np.concatenate([pad, m, pad], axis=0)
    starts = mm[1:] & ~mm[:-1]       # (D+1, A, B): run starts at index i
    ends = ~mm[1:] & mm[:-1]         # run ends before index i
    nrun = starts.sum(axis=0)
    k = int(min(max_runs, max(1, nrun.max())))
    a_dim, b_dim = m.shape[1:]
    centres = np.full((a_dim, b_dim, k), np.nan)
    lengths = np.zeros((a_dim, b_dim, k), dtype=np.int32)
    si = np.nonzero(starts)
    ei = np.nonzero(ends)
    # np.nonzero is C-ordered over (i, a, b); sort by column then by depth.
    so = np.lexsort((si[0], si[2], si[1]))
    eo = np.lexsort((ei[0], ei[2], ei[1]))
    s_d, s_a, s_b = si[0][so], si[1][so], si[2][so]
    e_d = ei[0][eo]
    col = s_a * b_dim + s_b
    first = np.r_[0, np.flatnonzero(np.diff(col)) + 1]
    rank = np.arange(col.size) - np.repeat(first, np.diff(np.r_[first, col.size]))
    keep = rank < k
    centres[s_a[keep], s_b[keep], rank[keep]] = 0.5 * (s_d[keep] + e_d[keep] - 1)
    lengths[s_a[keep], s_b[keep], rank[keep]] = (e_d[keep] - s_d[keep]).astype(np.int32)
    del d
    return centres, lengths


def track_sheet(
    centres: np.ndarray,
    lengths: np.ndarray,
    seed_ab: tuple[int, int],
    seed_k: int,
    *,
    max_run: int = 6,
    max_step: float = 1.5,
    min_separation: float = 3.0,
    depth_limits: tuple[float, float] = (-np.inf, np.inf),
    max_iterations: int = 100_000,
    return_order: bool = False,
):
    """Grow one sheet over columns from a seed run; return heights ``(A, B)`` (NaN = not on sheet).

    With ``return_order=True`` also returns the BFS ring index at which each column
    joined (-1 if never), i.e. a 4-connected geodesic distance from the seed.

    A neighbouring column joins when its run nearest to the predicted height (first-order
    extrapolation from assigned neighbours) lies within ``max_step``, is at most ``max_run``
    long, stays inside ``depth_limits``, and every other run in that column is at least
    ``min_separation`` farther from the prediction (no ambiguous jump between wraps).
    """
    a_dim, b_dim, _ = centres.shape
    h = np.full((a_dim, b_dim), np.nan)
    order_map = np.full((a_dim, b_dim), -1, dtype=np.int64)
    sa, sb = seed_ab
    c0 = centres[sa, sb, seed_k]
    if not np.isfinite(c0) or lengths[sa, sb, seed_k] > max_run:
        return (h, order_map) if return_order else h
    h[sa, sb] = c0
    order_map[sa, sb] = 0
    frontier = np.array([[sa, sb]], dtype=np.int64)
    nbrs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int64)
    lo_d, hi_d = depth_limits
    for it in range(1, max_iterations + 1):
        cand = (frontier[:, None, :] + nbrs[None]).reshape(-1, 2)
        ok = (cand[:, 0] >= 0) & (cand[:, 0] < a_dim) & (cand[:, 1] >= 0) & (cand[:, 1] < b_dim)
        cand = np.unique(cand[ok], axis=0)
        if cand.size == 0:
            break
        cand = cand[np.isnan(h[cand[:, 0], cand[:, 1]])]
        if cand.size == 0:
            break
        pred_sum = np.zeros(len(cand))
        pred_n = np.zeros(len(cand))
        for dv in nbrs:
            n1 = cand + dv
            n2 = cand + 2 * dv
            in1 = (n1[:, 0] >= 0) & (n1[:, 0] < a_dim) & (n1[:, 1] >= 0) & (n1[:, 1] < b_dim)
            h1 = np.full(len(cand), np.nan)
            h1[in1] = h[n1[in1, 0], n1[in1, 1]]
            in2 = (n2[:, 0] >= 0) & (n2[:, 0] < a_dim) & (n2[:, 1] >= 0) & (n2[:, 1] < b_dim)
            h2 = np.full(len(cand), np.nan)
            h2[in2] = h[n2[in2, 0], n2[in2, 1]]
            p = np.where(np.isfinite(h2), 2.0 * h1 - h2, h1)
            good = np.isfinite(p)
            pred_sum[good] += p[good]
            pred_n[good] += 1
        has = pred_n > 0
        cand = cand[has]
        pred = pred_sum[has] / pred_n[has]
        c = centres[cand[:, 0], cand[:, 1]]          # (M, K)
        ln = lengths[cand[:, 0], cand[:, 1]]
        dist = np.abs(c - pred[:, None])
        dist = np.where(np.isfinite(dist), dist, np.inf)
        order = np.argsort(dist, axis=1)
        best = order[:, 0]
        rows = np.arange(len(cand))
        d1 = dist[rows, best]
        d2 = dist[rows, order[:, 1]] if dist.shape[1] > 1 else np.full(len(cand), np.inf)
        with np.errstate(invalid="ignore"):
            sep = np.where(np.isinf(d2), np.inf, d2 - d1)
        cb = c[rows, best]
        accept = (
            (d1 <= max_step) & (sep >= min_separation) & (ln[rows, best] <= max_run)
            & (cb >= lo_d) & (cb <= hi_d)
        )
        if not accept.any():
            break
        acc = cand[accept]
        h[acc[:, 0], acc[:, 1]] = cb[accept]
        order_map[acc[:, 0], acc[:, 1]] = it
        frontier = acc
    return (h, order_map) if return_order else h


def cut_seams(h: np.ndarray, seed_ab: tuple[int, int], *, seam_step: float = 3.0, dilate: int = 1) -> np.ndarray:
    """Mask of the seed's 4-connected region after removing height seams.

    A tracked sheet that reaches an adjacent wrap and meets itself shows a seam
    (neighbouring columns more than ``seam_step`` voxels apart). Seam columns, dilated
    by ``dilate``, are removed; the component holding the seed is kept.
    """
    fin = np.isfinite(h)
    seam = np.zeros_like(fin)
    with np.errstate(invalid="ignore"):
        d0 = np.abs(np.diff(h, axis=0)) > seam_step
        d1 = np.abs(np.diff(h, axis=1)) > seam_step
    seam[:-1] |= d0
    seam[1:] |= d0
    seam[:, :-1] |= d1
    seam[:, 1:] |= d1
    if dilate > 0 and seam.any():
        seam = ndi.binary_dilation(seam, iterations=int(dilate))
    lab, _ = ndi.label(fin & ~seam)
    sid = lab[seed_ab]
    return (lab == sid) if sid else np.zeros_like(fin)


def extract_height_field(
    binary_zyx: np.ndarray,
    *,
    height_axis: int | None = None,
    max_run: int = 4,
    max_step: float = 1.0,
    min_separation: float = 3.0,
    seam_step: float = 3.0,
    median_size: int = 5,
    depth_margin: float = 0.0,
    lateral_margin: int = 0,
    max_seeds: int = 24,
    max_hole: int = 16,
    close_lateral: int = 0,
    seed_window: int = 2,
    smooth_sigma: float = 2.0,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Track the largest smooth single sheet through a binary m7 block.

    Returns ``(height, axis, metrics)``. ``height`` has the block shape with ``axis``
    removed, in voxels along ``axis``, NaN outside the patch. Seeds are every run in
    the columns around the block centre; each seed grows with :func:`track_sheet`
    (unambiguous nearest run, at most ``max_run`` long, at least ``depth_margin``
    voxels from the depth faces), is cut along height seams (:func:`cut_seams`) and
    keeps the seed's component; columns within ``lateral_margin`` of the lateral faces
    are dropped. Only holes of at most ``max_hole`` columns with a level rim are filled, never seam gaps.
    ``close_lateral`` > 0 first closes small in-sheet gaps with a cross-shaped
    structure perpendicular to the height axis (it cannot merge wraps along it). The largest result is cleaned by a local-median test
    (``|h - median| <= max_step``), reduced to its largest 4-connected region, and
    interior holes are filled by masked Gaussian smoothing.
    """
    b = np.asarray(binary_zyx, dtype=bool)
    if b.ndim != 3:
        raise ValueError("binary_zyx must be 3D")
    axis = dominant_normal_axis(b) if height_axis is None else int(height_axis)
    if axis not in (0, 1, 2):
        raise ValueError("height_axis must be 0, 1 or 2")
    m = np.moveaxis(b, axis, 0)
    depth = m.shape[0]
    a_dim, b_dim = m.shape[1:]
    empty = np.full((a_dim, b_dim), np.nan)
    base: dict[str, Any] = {"height_axis": "zyx"[axis]}
    if not m.any():
        return empty, axis, {**base, "patch_columns": 0, "seeds_tried": 0}
    if close_lateral > 0:
        st = np.zeros((1, 3, 3), dtype=bool)
        st[0, 1, :] = st[0, :, 1] = True
        m = ndi.binary_closing(m, structure=st, iterations=int(close_lateral))
    centres, lengths = column_runs(m)
    occupied = np.isfinite(centres[..., 0])
    single = (np.isfinite(centres).sum(axis=-1) == 1) & (lengths[..., 0] <= max_run)
    limits = (float(depth_margin), float(depth - 1 - depth_margin))

    ca, cb = a_dim // 2, b_dim // 2
    seeds = []
    for da in range(-seed_window, seed_window + 1):
        for db in range(-seed_window, seed_window + 1):
            a, bb = ca + da * max(1, a_dim // 8), cb + db * max(1, b_dim // 8)
            if 0 <= a < a_dim and 0 <= bb < b_dim:
                for k in np.flatnonzero(np.isfinite(centres[a, bb])):
                    c = centres[a, bb, k]
                    if limits[0] <= c <= limits[1] and lengths[a, bb, k] <= max_run:
                        seeds.append((a, bb, int(k)))
    best_h, best_area, tried = None, 0, 0
    covered = np.zeros((a_dim, b_dim, centres.shape[-1]), dtype=bool)
    for a, bb, k in seeds[: max_seeds * 8]:
        if tried >= max_seeds:
            break
        if covered[a, bb, k]:
            continue
        tried += 1
        h = track_sheet(centres, lengths, (a, bb), k, max_run=max_run, max_step=max_step,
                        min_separation=min_separation, depth_limits=limits)
        fin = np.isfinite(h)
        disc = cut_seams(h, (a, bb), seam_step=seam_step)
        if lateral_margin > 0:
            lm = int(lateral_margin)
            disc[:lm] = disc[-lm:] = False
            disc[:, :lm] = disc[:, -lm:] = False
        # Mark seed runs this sheet already passes through so they are not re-tracked.
        kk = np.nanargmin(np.abs(np.where(fin[..., None], centres - h[..., None], np.inf)), axis=-1) if fin.any() else None
        if kk is not None:
            ia, ib = np.nonzero(fin)
            covered[ia, ib, kk[ia, ib]] = True
        area = int(disc.sum())
        if area > best_area:
            best_h, best_area = np.where(disc, h, np.nan), area
    if best_h is None:
        return empty, axis, {**base, "patch_columns": 0, "seeds_tried": tried}

    h = best_h
    fin = np.isfinite(h)
    idx = ndi.distance_transform_edt(~fin, return_distances=False, return_indices=True)
    h_med = ndi.median_filter(h[idx[0], idx[1]], size=median_size, mode="nearest")
    smooth = fin & (np.abs(h - h_med) <= max_step)
    region = _largest_component_2d(smooth)
    # Patch columns keep their tracked height. Small interior holes are filled by
    # normalized convolution; a filled hole that creates a step > seam_step with any
    # neighbour (a short seam gap, not a pore) is emptied again.
    holes = ndi.binary_fill_holes(region) & ~region
    hl, nh = ndi.label(holes)
    w = ndi.gaussian_filter(region.astype(np.float64), smooth_sigma)
    hsum = ndi.gaussian_filter(np.where(region, h, 0.0), smooth_sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        hs = np.where(region, h, hsum / w)
    cand = np.zeros_like(region)
    if nh:
        hsz = np.bincount(hl.ravel(), minlength=nh + 1)
        cand = holes & (hsz[hl] <= max_hole) & (w > 1e-6) & np.isfinite(hs)
        k0 = region | cand
        bad = np.zeros_like(region)
        for ax in (0, 1):
            d = np.abs(np.diff(np.where(k0, hs, 0.0), axis=ax)) > seam_step
            if ax == 0:
                pair = k0[1:] & k0[:-1] & d
                bad[1:] |= pair
                bad[:-1] |= pair
            else:
                pair = k0[:, 1:] & k0[:, :-1] & d
                bad[:, 1:] |= pair
                bad[:, :-1] |= pair
        bad_labels = np.unique(hl[bad & cand])
        cand &= ~np.isin(hl, bad_labels[bad_labels > 0])
    keep = region | cand
    if lateral_margin > 0:
        lm = int(lateral_margin)
        keep[:lm] = keep[-lm:] = False
        keep[:, :lm] = keep[:, -lm:] = False
    height = np.where(keep, hs, np.nan)
    hv = np.where(keep, height, 0.0)
    step_r = np.abs(np.diff(hv, axis=0))[keep[1:] & keep[:-1]]
    step_c = np.abs(np.diff(hv, axis=1))[keep[:, 1:] & keep[:, :-1]]
    steps = np.concatenate([step_r, step_c, [0.0]])
    gy, gx = np.gradient(np.where(keep, height, h_med))
    slope = np.hypot(gy, gx)[keep]
    return height, axis, {
        **base,
        "seeds_tried": int(tried),
        "single_run_column_fraction": float(single.sum() / max(1, occupied.sum())),
        "tracked_columns": int(best_area),
        "smooth_column_fraction": float(smooth.sum() / max(1, best_area)),
        "patch_columns": int(keep.sum()),
        "patch_column_fraction": float(keep.mean()),
        "max_abs_height_step_vx": float(steps.max()),
        "median_abs_slope": float(np.median(slope)) if slope.size else 0.0,
        "p95_abs_slope": float(np.percentile(slope, 95)) if slope.size else 0.0,
    }


def height_field_to_tifxyz(
    height: np.ndarray,
    axis: int,
    origin_zyx: Sequence[float],
    *,
    grid_step: float,
    factor: float = 1.0,
    max_gap_mismatch: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a block height field on a regular grid and return target-frame tifxyz.

    ``grid_step`` is in block (reader) voxels; ``factor`` scales block voxels into the
    target frame. An edge between finite grid neighbours whose straight column path
    leaves the height field is accepted only if its height change matches the mean
    local gradient of its two ends within ``max_gap_mismatch`` voxels; otherwise both
    nodes are dropped, so no quad spans a slit between wraps. Then only
    the largest 4-connected group of nodes is kept, so the patch is one connected sheet.
    Invalid vertices are ``-1.0`` on all three channels.
    """
    if grid_step <= 0:
        raise ValueError("grid_step must be > 0")
    ha, hb = height.shape
    ra = np.arange(0.0, ha - 1 + 1e-9, grid_step)
    rb = np.arange(0.0, hb - 1 + 1e-9, grid_step)
    ga, gb = np.meshgrid(ra, rb, indexing="ij")
    ia = np.rint(ga).astype(int)
    ib = np.rint(gb).astype(int)
    hv = height[ia, ib]
    node_ok = np.isfinite(hv)
    bad = ~np.isfinite(height)
    fill_idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True) if bad.any() else None
    hf = height[fill_idx[0], fill_idx[1]] if fill_idx is not None else height
    g_a, g_b = np.gradient(ndi.median_filter(hf, size=3, mode="nearest"))
    cb = np.concatenate([np.zeros((ha, 1), np.int64), np.cumsum(bad, axis=1)], axis=1)
    ca = np.concatenate([np.zeros((1, hb), np.int64), np.cumsum(bad, axis=0)], axis=0)
    drop = np.zeros_like(node_ok)
    if hv.shape[1] > 1:
        r, c0, c1 = ia[:, :-1], ib[:, :-1], ib[:, 1:]
        cross = (cb[r, c1 + 1] - cb[r, c0]) > 0
        pred = 0.5 * (g_b[r, c0] + g_b[r, c1]) * (c1 - c0)
        mism = np.abs((hv[:, 1:] - hv[:, :-1]) - pred) > max_gap_mismatch
        e = node_ok[:, :-1] & node_ok[:, 1:] & cross & mism
        drop[:, :-1] |= e
        drop[:, 1:] |= e
    if hv.shape[0] > 1:
        c, r0, r1 = ib[:-1, :], ia[:-1, :], ia[1:, :]
        cross = (ca[r1 + 1, c] - ca[r0, c]) > 0
        pred = 0.5 * (g_a[r0, c] + g_a[r1, c]) * (r1 - r0)
        mism = np.abs((hv[1:, :] - hv[:-1, :]) - pred) > max_gap_mismatch
        e = node_ok[:-1, :] & node_ok[1:, :] & cross & mism
        drop[:-1, :] |= e
        drop[1:, :] |= e
    valid = _largest_component_2d(node_ok & ~drop)
    coords = np.zeros(ga.shape + (3,), dtype=np.float64)
    others = [a for a in range(3) if a != axis]
    coords[..., axis] = np.where(valid, hv, 0.0)
    coords[..., others[0]] = ga
    coords[..., others[1]] = gb
    coords += np.asarray(origin_zyx, dtype=np.float64)
    coords *= float(factor)
    x = np.where(valid, coords[..., 2], -1.0)
    y = np.where(valid, coords[..., 1], -1.0)
    z = np.where(valid, coords[..., 0], -1.0)
    cx, cy, cz, v = SSR.validate_tifxyz_channels(x, y, z)
    return cx, cy, cz, v


def mesh_geometry_metrics(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, voxel_um: float
) -> dict[str, Any]:
    """Triangle area (cm2), slope proxies and connectivity of a tifxyz grid."""
    cx, cy, cz, valid = SSR.validate_tifxyz_channels(x, y, z)
    p = np.stack([cx, cy, cz], axis=-1).astype(np.float64)
    quad = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    a = p[:-1, :-1]
    b = p[1:, :-1]
    c = p[:-1, 1:]
    d = p[1:, 1:]
    t1 = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)
    t2 = 0.5 * np.linalg.norm(np.cross(b - d, c - d), axis=-1)
    area_vx2 = float(((t1 + t2) * quad).sum())
    _, n = ndi.label(valid)
    return {
        "grid_shape": [int(s) for s in valid.shape],
        "valid_vertex_count": int(valid.sum()),
        "valid_vertex_fraction": float(valid.mean()) if valid.size else 0.0,
        "connected_components_in_grid": int(n),
        "mesh_area_cm2": area_vx2 * (float(voxel_um) * 1e-4) ** 2,
    }


# ---------------------------------------------------------------------------
# Coherence scoring and gates
# ---------------------------------------------------------------------------

def _load_onsheet_module():
    if not ONSHEET_PATH.is_file():
        return None
    try:
        import cv2  # noqa: F401
    except ImportError:
        return None
    spec = importlib.util.spec_from_file_location("onsheet_score", ONSHEET_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def score_stack_onsheet(stack_lhw: np.ndarray, valid_yx: np.ndarray, mesh_index: int, dk: int = 2) -> dict[str, Any]:
    """``onsheet_score.score_stack`` on a renderer-masked stack (fallback: Pearson diagnostic)."""
    masked = np.array(stack_lhw, copy=True)
    masked[:, ~np.asarray(valid_yx, dtype=bool)] = 0
    mod = _load_onsheet_module()
    if mod is not None:
        res = dict(mod.score_stack(masked, mid=mesh_index, dk=dk))
        res["backend"] = "onsheet_score"
        res.setdefault("f_parallel", 0.0)
        return res
    fb = SSR.layer_coherence_score(masked, mesh_index=mesh_index, delta=dk, valid_mask=valid_yx)
    return {
        "f_parallel": fb["zero_shift_dominant_fraction"],
        "zero_shift_dominant_fraction": fb["zero_shift_dominant_fraction"],
        "tiles": fb["patch_count"],
        "dk": int(dk),
        "backend": "stream_snap_render_fallback",
    }


def evaluate_gates(
    metrics: dict[str, Any],
    *,
    f_parallel_min: float = F_PARALLEL_MIN,
    valid_ct_cm2_min: float = VALID_CT_CM2_MIN,
) -> dict[str, Any]:
    f_par = float(metrics.get("f_parallel") or 0.0)
    area = float(metrics.get("valid_ct_cm2") or 0.0)
    tiles = int(metrics.get("tiles") or 0)
    g = {
        "f_parallel_min": f_parallel_min,
        "valid_ct_cm2_min": valid_ct_cm2_min,
        "f_parallel_ok": bool(tiles > 0 and f_par >= f_parallel_min),
        "valid_ct_cm2_ok": bool(area >= valid_ct_cm2_min),
    }
    g["passed"] = bool(g["f_parallel_ok"] and g["valid_ct_cm2_ok"])
    return g


def sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep only whitelisted scalar geometry/coherence keys (no coordinates, paths, maps)."""
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if k not in PUBLIC_METRIC_KEYS:
            continue
        if isinstance(v, (np.floating, float)):
            out[k] = round(float(v), 6) if math.isfinite(float(v)) else None
        elif isinstance(v, (np.integer, int, bool, str)) or v is None:
            out[k] = v.item() if isinstance(v, np.generic) else v
        elif isinstance(v, (list, tuple)) and all(isinstance(e, (int, np.integer)) for e in v):
            out[k] = [int(e) for e in v]
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def extract_patch(
    extract_reader,
    *,
    box_lo_zyx: Sequence[int],
    box_hi_zyx: Sequence[int],
    extract_factor: float,
    voxel_um: float,
    scale: Sequence[float] = DEFAULT_SCALE,
    threshold: float = 128.0,
    height_axis: int | None = None,
    extract_kw: dict[str, Any] | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    """Read one m7 box (``extract_reader`` voxels) and return a target-frame tifxyz patch."""
    sx, sy = SSR._normalize_scale(list(scale))
    if not math.isclose(sx, sy):
        raise ValueError("anisotropic scale is not supported")
    t0 = time.perf_counter()
    block, origin = extract_reader.read_box(box_lo_zyx, box_hi_zyx)
    binary = np.asarray(block) >= threshold
    del block
    height, axis, ext = extract_height_field(binary, height_axis=height_axis, **(extract_kw or {}))
    grid_step = (1.0 / sx) / float(extract_factor)
    x, y, z, _ = height_field_to_tifxyz(height, axis, origin, grid_step=grid_step, factor=extract_factor)
    metrics: dict[str, Any] = dict(ext)
    metrics["extract_seconds"] = round(time.perf_counter() - t0, 2)
    metrics.update(mesh_geometry_metrics(x, y, z, voxel_um))
    return (x, y, z), metrics


def snap_render_score(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    snap_reader,
    ct_reader,
    voxel_um: float,
    scale: Sequence[float] = DEFAULT_SCALE,
    threshold: float = 128.0,
    snap_passes: int = 2,
    snap_radius: float = 12.0,
    num_slices: int = 15,
    slice_step: float = 1.0,
    ct_budget: ByteBudget | None = None,
    ct_chunk_bytes: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    """Snap a target-frame patch to m7, render CT, score ``onsheet_score`` and gate it.

    ``snap_reader`` and ``ct_reader`` take target-frame ``xyz`` (wrap coarser levels in
    :class:`ScaledReader`). With ``ct_budget`` and ``ct_chunk_bytes`` the CT chunk set
    is estimated first and rendering is refused if it cannot fit the remaining budget.
    One rendered pixel = one target-frame voxel when ``scale`` = 1 / grid spacing.
    """
    sx, sy = SSR._normalize_scale(list(scale))
    metrics: dict[str, Any] = {}
    cx, cy, cz, valid = SSR.validate_tifxyz_channels(x, y, z)
    if not valid.any():
        metrics.update(valid_ct_px=0, valid_ct_cm2=0.0, f_parallel=0.0, tiles=0)
        metrics["gates"] = evaluate_gates(metrics)
        return (cx, cy, cz), metrics
    if snap_passes > 0:
        cx, cy, cz, snap_m = SSR.snap_tifxyz_to_m7(
            cx, cy, cz, snap_reader, passes=snap_passes, search_radius=snap_radius, threshold=threshold
        )
        metrics.update(snap_m)
    metrics.update(mesh_geometry_metrics(cx, cy, cz, voxel_um))
    cx, cy, cz, valid = SSR.validate_tifxyz_channels(cx, cy, cz)

    if ct_budget is not None and ct_chunk_bytes:
        est = estimate_render_chunks(cx, cy, cz, ct_reader, num_slices=num_slices, slice_step=slice_step)
        metrics["ct_chunks_estimated"] = len(est)
        ct_budget.precheck(len(est) * int(ct_chunk_bytes))

    t1 = time.perf_counter()
    stack, valid_ct = SSR.render_surface_stack(
        cx, cy, cz, ct_reader, num_slices=num_slices, slice_step=slice_step, scale=[sx, sy]
    )
    metrics["render_seconds"] = round(time.perf_counter() - t1, 2)
    metrics["render_px"] = [int(stack.shape[1]), int(stack.shape[2])]
    metrics["num_slices"] = int(num_slices)
    metrics["slice_step"] = float(slice_step)
    metrics["valid_ct_px"] = int(valid_ct.sum())
    metrics["valid_ct_cm2"] = float(valid_ct.sum()) * (float(voxel_um) * 1e-4) ** 2
    coh = score_stack_onsheet(stack, valid_ct, (num_slices - 1) // 2, dk=2)
    # onsheet_score's tile geometry must not overwrite the snap's ``step`` / radius keys.
    for key in ("step", "tile"):
        if key in coh:
            coh[f"onsheet_{key}"] = coh.pop(key)
    metrics.update(coh)
    metrics["gates"] = evaluate_gates(metrics)
    del stack
    return (cx, cy, cz), metrics


def grow_m7_patch(
    extract_reader,
    *,
    box_lo_zyx: Sequence[int],
    box_hi_zyx: Sequence[int],
    extract_factor: float,
    snap_reader,
    ct_reader,
    voxel_um: float,
    scale: Sequence[float] = DEFAULT_SCALE,
    threshold: float = 128.0,
    height_axis: int | None = None,
    extract_kw: dict[str, Any] | None = None,
    **snap_render_kw,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    """:func:`extract_patch` followed by :func:`snap_render_score`."""
    (x, y, z), m = extract_patch(
        extract_reader, box_lo_zyx=box_lo_zyx, box_hi_zyx=box_hi_zyx, extract_factor=extract_factor,
        voxel_um=voxel_um, scale=scale, threshold=threshold, height_axis=height_axis, extract_kw=extract_kw,
    )
    m = {f"extracted_{k}" if k in ("valid_vertex_count", "mesh_area_cm2") else k: v for k, v in m.items()}
    (x, y, z), m2 = snap_render_score(
        x, y, z, snap_reader=snap_reader, ct_reader=ct_reader, voxel_um=voxel_um, scale=scale,
        threshold=threshold, **snap_render_kw,
    )
    m.update(m2)
    return (x, y, z), m


def estimate_snap_chunks(x, y, z, snap_reader, *, search_radius: float = 12.0) -> set:
    """m7 chunks touched by snapping: vertices +/- ``search_radius`` along normals."""
    cx, cy, cz, valid = SSR.validate_tifxyz_channels(x, y, z)
    n, ok = SSR.compute_grid_normals(cx, cy, cz, valid=valid)
    p = np.stack([cx, cy, cz], axis=-1).astype(np.float64)[ok]
    nn = n[ok].astype(np.float64)
    offs = np.linspace(-search_radius, search_radius, 5)
    rays = (p[None] + offs[:, None, None] * nn[None]).reshape(-1, 3)
    base = snap_reader.reader if isinstance(snap_reader, ScaledReader) else snap_reader
    if isinstance(snap_reader, ScaledReader):
        rays = rays / snap_reader.factor
    return chunk_indices_for_points(rays, base.shape, base.chunks)


def estimate_render_chunks(x, y, z, ct_reader, *, num_slices: int, slice_step: float) -> set:
    """Estimate CT chunks touched by rendering: vertices, edge midpoints, all slice offsets."""
    cx, cy, cz, valid = SSR.validate_tifxyz_channels(x, y, z)
    n, ok = SSR.compute_grid_normals(cx, cy, cz, valid=valid)
    p = np.stack([cx, cy, cz], axis=-1).astype(np.float64)
    pts = [p[ok]]
    nn = [n[ok].astype(np.float64)]
    for sl_a, sl_b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        both = ok[sl_a] & ok[sl_b]
        pts.append(0.5 * (p[sl_a][both] + p[sl_b][both]))
        nn.append(0.5 * (n[sl_a][both] + n[sl_b][both]).astype(np.float64))
    quad = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    pts.append(0.25 * (p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:])[quad])
    nn.append(0.25 * (n[:-1, :-1] + n[1:, :-1] + n[:-1, 1:] + n[1:, 1:])[quad].astype(np.float64))
    P = np.concatenate(pts)
    N = np.concatenate(nn)
    half = (num_slices - 1) // 2
    offs = (np.arange(num_slices) - half) * float(slice_step)
    rays = (P[None] + offs[:, None, None] * N[None]).reshape(-1, 3)
    base = ct_reader.reader if isinstance(ct_reader, ScaledReader) else ct_reader
    if isinstance(ct_reader, ScaledReader):
        rays = rays / ct_reader.factor
    return chunk_indices_for_points(rays, base.shape, base.chunks)


def select_interior_box(
    overview_zyx: np.ndarray,
    overview_factor: float,
    *,
    box_depth: int,
    box_lateral: int,
    radius_fraction: float = 0.5,
    z_fraction: float = 0.5,
    depth_align: int = 1,
    directions: Iterable[str] = ("+x", "-x", "+y", "-y"),
) -> tuple[tuple[int, int, int], tuple[int, int, int], dict[str, Any]]:
    """Choose an interior box (in fine voxels) whose dominant sheet normal is ~x or ~y.

    Uses a coarse binary m7 overview: centroid of the z-slab as the scroll axis, the
    95th-percentile radius, then the densest of the four radial directions at
    ``radius_fraction`` of that radius. ``box_depth`` is along the radial axis, aligned
    to ``depth_align``; ``box_lateral`` along the tangent and along z.
    """
    o = np.asarray(overview_zyx, dtype=bool)
    zs = np.flatnonzero(o.any(axis=(1, 2)))
    if zs.size == 0:
        raise ValueError("empty overview")
    zc = int(zs[0] + z_fraction * (zs[-1] - zs[0]))
    slab = o[max(0, zc - 2) : zc + 3].any(axis=0)
    yy, xx = np.nonzero(slab)
    cy, cx = float(yy.mean()), float(xx.mean())
    r95 = float(np.percentile(np.hypot(yy - cy, xx - cx), 95))
    f = float(overview_factor)
    best = None
    for d in directions:
        sgn = 1.0 if d[0] == "+" else -1.0
        py = cy + (sgn * radius_fraction * r95 if d[1] == "y" else 0.0)
        px = cx + (sgn * radius_fraction * r95 if d[1] == "x" else 0.0)
        hw = max(1, int(box_lateral / f / 2))
        hd = max(1, int(box_depth / f / 2))
        if d[1] == "x":
            win = o[max(0, zc - hw) : zc + hw, max(0, int(py) - hw) : int(py) + hw, max(0, int(px) - hd) : int(px) + hd]
        else:
            win = o[max(0, zc - hw) : zc + hw, max(0, int(py) - hd) : int(py) + hd, max(0, int(px) - hw) : int(px) + hw]
        dens = float(win.mean()) if win.size else 0.0
        if best is None or dens > best[0]:
            best = (dens, d, py * f, px * f)
    dens, d, fy, fx = best
    fz = zc * f
    if d[1] == "x":
        x0 = int(round((fx - box_depth / 2) / depth_align)) * depth_align
        lo = (int(fz - box_lateral / 2), int(fy - box_lateral / 2), x0)
        hi = (lo[0] + box_lateral, lo[1] + box_lateral, x0 + box_depth)
    else:
        y0 = int(round((fy - box_depth / 2) / depth_align)) * depth_align
        lo = (int(fz - box_lateral / 2), y0, int(fx - box_lateral / 2))
        hi = (lo[0] + box_lateral, y0 + box_depth, lo[2] + box_lateral)
    size = [h - l for l, h in zip(lo, hi)]
    extent = [int(s * f) for s in o.shape]
    lo = tuple(max(0, min(l, e - sz)) for l, e, sz in zip(lo, extent, size))
    hi = tuple(l + sz for l, sz in zip(lo, size))
    return lo, hi, {"direction": d, "overview_density": dens, "radius_fraction": radius_fraction}


# ---------------------------------------------------------------------------
# Private output and public metadata inspection
# ---------------------------------------------------------------------------

def write_private_patch(out_dir: str | os.PathLike[str], x, y, z, meta: dict[str, Any]) -> dict[str, str]:
    """Write tifxyz outside every Git worktree; return sha256 of each written file."""
    root = SSR._private_output_path(out_dir)
    SSR.write_tifxyz(root, x, y, z, meta={"scale": list(DEFAULT_SCALE), **meta})
    return {name: _sha256(root / name) for name in ("x.tif", "y.tif", "z.tif", "meta.json")}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_public_json(key: str, budget: ByteBudget, *, base: str = S3_BUCKET, timeout: float = 60.0) -> Any:
    """GET a small public JSON object; 404 -> None. Bytes are charged to ``budget``."""
    import urllib.error
    import urllib.request

    budget.precheck()
    if not str(base).lower().startswith(("http://", "https://")):  # local mirror (tests)
        path = Path(base).joinpath(*key.split("/"))
        if not path.is_file():
            return None
        raw = path.read_bytes()
        budget.charge(len(raw))
        return json.loads(raw.decode("utf-8"))
    try:
        with urllib.request.urlopen(f"{base}/{key}", timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    budget.charge(len(raw))
    return json.loads(raw.decode("utf-8"))


def inspect_public_metadata(
    scroll: str, budget: ByteBudget, *, levels: Sequence[int] = (0, 1, 2, 3), fetch=None
) -> dict[str, Any]:
    """Summarize small public metadata (m7, CT, normal-grids, lasagna) for one scroll."""
    spec = SCROLL_SPECS[scroll]
    if fetch is None:
        def fetch(key, b):
            return fetch_public_json(key, b, base=S3_BUCKET)
    out: dict[str, Any] = {
        "scroll": scroll,
        "ct_level0_um": spec["ct_level0_um"],
        "m7_frame_ct_level": spec["m7_frame_ct_level"],
        "m7_dataset": spec["m7"].rsplit("/", 1)[-1],
        "ct_dataset": spec["ct"].rsplit("/", 1)[-1],
    }
    for name in ("m7", "ct"):
        lv = {}
        for level in levels:
            za = fetch(f"{spec[name]}/{level}/.zarray", budget)
            if za:
                comp = za.get("compressor") or {}
                lv[str(level)] = {
                    "shape": za["shape"], "chunks": za["chunks"], "dtype": za["dtype"],
                    "compressor": comp.get("cname") or comp.get("id"),
                }
        out[f"{name}_levels"] = lv
    ng = fetch(f"{spec['normal_grids']}/metadata.json", budget)
    if ng:
        out["normal_grids"] = {k: ng[k] for k in ("grid-step", "input-level", "spiral-step", "sparse-volume") if k in ng}
    if spec["lasagna"]:
        lj = fetch(f"{spec['lasagna']}/{scroll}.lasagna.json", budget)
        if lj:
            out["lasagna"] = {
                "dataset": spec["lasagna"].rsplit("/", 1)[-1],
                "version": lj.get("version"),
                "base_shape_zyx": lj.get("base_shape_zyx"),
                "groups": {g: {"scaledown": v.get("scaledown"), "channels": v.get("channels")}
                           for g, v in (lj.get("groups") or {}).items()},
            }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scroll", choices=sorted(SCROLL_SPECS), required=True)
    p.add_argument("--budget-mb", type=float, default=115.0, help="hard cap on streamed MB (1e6 bytes)")
    p.add_argument("--overview-level", type=int, default=4)
    p.add_argument("--extract-level", type=int, default=1, help="m7 pyramid level for extraction")
    p.add_argument("--snap-level", type=int, default=0, help="preferred m7 level for snapping")
    p.add_argument("--ct-level", type=int, default=None, help="CT pyramid level (default: frame level + extract level)")
    p.add_argument("--box-target", type=int, nargs=6, default=None, metavar=("Z0", "Y0", "X0", "Z1", "Y1", "X1"),
                   help="explicit target-frame box (skips the overview); never written to the receipt")
    p.add_argument("--height-axis", choices=("z", "y", "x", "auto"), default="auto")
    p.add_argument("--box-depth", type=int, default=256, help="target-frame voxels along the radial axis")
    p.add_argument("--box-lateral", type=int, default=1152, help="target-frame voxels along tangent and z")
    p.add_argument("--radius-fraction", type=float, default=0.5)
    p.add_argument("--z-fraction", type=float, default=0.5)
    p.add_argument("--snap-passes", type=int, default=2)
    p.add_argument("--num-slices", type=int, default=15)
    p.add_argument("--private-out", required=True, help="private tifxyz directory, outside every Git worktree")
    p.add_argument("--evidence-out", required=True, help="sanitized public JSON receipt")
    return p


def run(args: argparse.Namespace) -> dict[str, Any]:
    spec = SCROLL_SPECS[args.scroll]
    budget = ByteBudget(int(args.budget_mb * 1e6))
    private_dir = SSR._private_output_path(args.private_out)
    t0 = time.perf_counter()
    meta_inspect = {s: inspect_public_metadata(s, budget) for s in SCROLL_SPECS}
    frame_level = int(spec["m7_frame_ct_level"])
    voxel_um = float(spec["ct_level0_um"]) * 2 ** frame_level
    ct_level = frame_level + args.extract_level if args.ct_level is None else args.ct_level
    ct_factor = 2 ** (ct_level - frame_level)
    m7_url = f"{S3_BUCKET}/{spec['m7']}"
    ct_url = f"{S3_BUCKET}/{spec['ct']}"
    kw = dict(budget=budget, workers=8, cache_mb=256)
    ef = 2 ** args.extract_level
    seed: dict[str, Any] = {"mode": "explicit" if args.box_target else "overview"}

    if args.box_target:
        lo_t, hi_t = tuple(args.box_target[:3]), tuple(args.box_target[3:])
        height_axis = None if args.height_axis == "auto" else "zyx".index(args.height_axis)
    else:
        ov = BudgetedRemoteZarrV2(m7_url, str(args.overview_level), **kw)
        try:
            overview, _ = ov.read_box((0, 0, 0), ov.shape)
        finally:
            ov.close()
        lo_t, hi_t, sel = select_interior_box(
            overview >= 128, 2 ** args.overview_level, box_depth=args.box_depth, box_lateral=args.box_lateral,
            radius_fraction=args.radius_fraction, z_fraction=args.z_fraction, depth_align=128 * ct_factor,
        )
        del overview
        height_axis = 2 if sel["direction"][1] == "x" else 1
        seed.update(direction=sel["direction"], radius_fraction=args.radius_fraction, z_fraction=args.z_fraction)
    seed["box_size_target_vx"] = [int(h - l) for l, h in zip(lo_t, hi_t)]

    margin = int(math.ceil(((args.num_slices - 1) // 2) / ef)) + 2
    extract_kw = {"depth_margin": margin, "lateral_margin": margin, "close_lateral": 1}
    seed["extract_kw"] = extract_kw
    ext = BudgetedRemoteZarrV2(m7_url, str(args.extract_level), **kw)
    readers = [ext]
    try:
        (x, y, z), m = extract_patch(
            ext, box_lo_zyx=[v // ef for v in lo_t], box_hi_zyx=[-(-v // ef) for v in hi_t],
            extract_factor=ef, voxel_um=voxel_um, height_axis=height_axis, extract_kw=extract_kw,
        )
        m = {f"extracted_{k}" if k in ("valid_vertex_count", "mesh_area_cm2") else k: v for k, v in m.items()}
        ct_raw = BudgetedRemoteZarrV2(ct_url, str(ct_level), **kw)
        readers.append(ct_raw)
        ct = ScaledReader(ct_raw, ct_factor)
        ct_est = len(estimate_render_chunks(x, y, z, ct, num_slices=args.num_slices, slice_step=1.0))
        ct_est_bytes = int(1.15 * ct_est * ct_raw._expected_chunk_bytes)

        snap_level, snap_note = args.snap_level, "preferred level"
        snap_r = ext
        if args.snap_level != args.extract_level:
            fine = BudgetedRemoteZarrV2(m7_url, str(args.snap_level), **kw)
            readers.append(fine)
            n_fine = len(estimate_snap_chunks(x, y, z, ScaledReader(fine, 2 ** args.snap_level)))
            per_chunk = ext.bytes_fetched / max(1, ext.chunks_fetched)
            snap_est_bytes = int(1.5 * n_fine * per_chunk)
            if budget.used + snap_est_bytes + ct_est_bytes <= budget.cap:
                snap_r = fine
            else:
                snap_level, snap_note = args.extract_level, "fine snap would not fit budget; used extract level"
        snap = ScaledReader(snap_r, 2 ** snap_level)
        (x, y, z), m2 = snap_render_score(
            x, y, z, snap_reader=snap, ct_reader=ct, voxel_um=voxel_um, snap_passes=args.snap_passes,
            num_slices=args.num_slices, ct_budget=budget, ct_chunk_bytes=ct_raw._expected_chunk_bytes,
        )
        m.update(m2)
        streams = {
            "m7_extract_bytes": int(ext.bytes_fetched), "m7_extract_chunks": int(ext.chunks_fetched),
            "m7_snap_bytes": int(snap_r.bytes_fetched) if snap_r is not ext else 0,
            "m7_snap_chunks": int(snap_r.chunks_fetched) if snap_r is not ext else 0,
            "ct_bytes": int(ct_raw.bytes_fetched), "ct_chunks": int(ct_raw.chunks_fetched),
            "ct_chunks_estimated": int(ct_est),
        }
    finally:
        for r in readers:
            r.close()
    hashes = write_private_patch(private_dir, x, y, z, {
        "uuid": f"{args.scroll}_m7grow_preflight", "source": "grow_m7_patch.py",
        "frame": f"{spec['m7'].rsplit('/', 1)[-1]} level 0", "voxel_size_um": voxel_um,
    })
    receipt = {
        "schema": "virgin_scroll_patch_preflight/1.0.0",
        "method": "CPU-only m7 single-sheet height-field extraction + m7 snap + onsheet_score; no ink detector run",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scroll": args.scroll,
        "m7_dataset": spec["m7"].rsplit("/", 1)[-1],
        "ct_dataset": spec["ct"].rsplit("/", 1)[-1],
        "frame": {"name": "m7 level 0", "voxel_um": voxel_um, "ct_pyramid_level": frame_level},
        "levels": {
            "extract_m7": args.extract_level, "extract_voxel_um": voxel_um * ef,
            "snap_m7": snap_level, "snap_voxel_um": voxel_um * 2 ** snap_level, "snap_level_note": snap_note,
            "render_ct": ct_level, "render_ct_voxel_um": float(spec["ct_level0_um"]) * 2 ** ct_level,
            "render_pixel_um": voxel_um,
        },
        "seed": seed,
        "metrics": sanitize_metrics(m),
        "gates": m["gates"],
        "stream": {**streams, "total_bytes": budget.used, "budget_bytes": budget.cap,
                   "requests": budget.requests, "persistent_ct_on_disk": False},
        "private_patch": {"written": True, "location": "private, outside git", "sha256": hashes},
        "s3_metadata_inspection": meta_inspect,
        "seconds": round(time.perf_counter() - t0, 1),
        "caveats": [
            "CT is rendered from a coarser pyramid level than the frame and resampled at frame pitch.",
            "valid_ct_cm2 counts frame-pitch pixels with full source support on every rendered plane.",
            "f_parallel is an upper bound on the on-sheet share; tiles without CT structure count as parallel.",
            "No ink detector run; nothing here claims ink or text.",
        ],
    }
    Path(args.evidence_out).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    r = run(args)
    print(json.dumps({"gates": r["gates"], "stream_total_bytes": r["stream"]["total_bytes"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
