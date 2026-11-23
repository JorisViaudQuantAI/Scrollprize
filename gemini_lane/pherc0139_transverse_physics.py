# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Exp T3: PHerc. 0139 2.399 um MSII + Directional Fiber Subtraction + 9.362 um Paganin Ablation.

Strictly CPU-only, zero persistent CT/volumes on disk, hard-capped HTTP byte
budget (<= 60 MiB Part A + <= 60 MiB Part B <= 120 MiB global).

Context:
- ``PHerc. 0139`` is a published read control (NOT an eligible First Letters scroll).
- Part A operates on the native ``2.399 um`` pre-rendered surface-volume Zarr and
  published ``2.399 um`` ink reference map on segment
  ``20260422000000-title_2026042222_zmid_flatboi``:
  1. Sub-voxel papyrus peak height-field ``h(u, v) = n_peak(u, v) - G_sigma(n_peak)``
     plus Mara GigaMesh Multi-Scale Integral Invariants (MSII spherical-volume
     occupancy at ``r = 20, 60, 140 um``), local mean/Gaussian curvature, and
     relief roughness.
  2. Unsupervised Directional Fiber-Weave Subtraction: 2D FFT orientation
     spectrum peak estimation of quasi-orthogonal fiber axes, smooth directional
     FFT notch filtering, and ``StructN2V``-style directional blind-mask ridge
     linear prediction along the fiber axes, comparing local crackle energy on
     raw CT vs fiber-subtracted residuals.
- Part B streams the spatially co-registered ROI across the three aligned
  ``9.362 um`` Paganin-ablation reconstructions (``pag0``, ``pag50``, ``pag1000``)
  reusing ``stream_snap_render.py`` source-support semantics, and compares raw CT
  contrast, local high-frequency phase-edge / scattering variance, and
  surface-height relief.
- Every feature is evaluated with exact Mann-Whitney ``ROC-AUC``, ``Cohen's d``,
  and a 16-shift spatially rolled-label null against the predeclared gate:
  ``ROC-AUC >= 0.70 AND >= +0.10 above rolled-label null mean``.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import io
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Sequence
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata
import tifffile

try:
    import requests as _requests
except ImportError:  # pragma: no cover - optional fallback
    _requests = None

requests = _requests

try:
    from .stream_snap_render import (
        RemoteZarrV2,
        _positive_weight_stencil_support,
        _sanitize_source_label,
        compute_grid_normals,
        validate_tifxyz_channels,
    )
except ImportError:  # pragma: no cover - direct script execution
    from stream_snap_render import (
        RemoteZarrV2,
        _positive_weight_stencil_support,
        _sanitize_source_label,
        compute_grid_normals,
        validate_tifxyz_channels,
    )


# ---------------------------------------------------------------------------
# Public PHerc. 0139 S3 endpoints & predeclared experiment constants
# ---------------------------------------------------------------------------

S3_BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
SEG_ROOT = f"{S3_BASE}/PHerc0139/segments/20260422000000-title_2026042222_zmid_flatboi"

DEFAULT_URLS: dict[str, str] = {
    "surface_volume_2p4um": (
        f"{SEG_ROOT}/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr"
    ),
    "ink_reference_2p4um": (
        f"{SEG_ROOT}/ink-detection/"
        "PHerc0139-20260422000000-2.399um-0.22m-78keV-volume-20260102150214-"
        "20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"
    ),
    "pag1000_mesh": f"{SEG_ROOT}/mesh/20260422000000-on-20250728140407-9.362um.tifxyz",
    "pag1000_vol": f"{S3_BASE}/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr",
    "pag50_mesh": f"{SEG_ROOT}/mesh/20260422000000-on-20251107135911-9.362um.tifxyz",
    "pag50_vol": f"{S3_BASE}/PHerc0139/volumes/20251107135911-9.362um-1.2m-113keV-pag50-masked.zarr",
    "pag0_mesh": f"{SEG_ROOT}/mesh/20260422000000-on-20251107132835-9.362um.tifxyz",
    "pag0_vol": f"{S3_BASE}/PHerc0139/volumes/20251107132835-9.362um-1.2m-113keV-pag0-masked.zarr",
}

VOXEL_UM_PART_A = 2.399
VOXEL_UM_PART_B = 9.362
MSII_RADII_UM: tuple[float, float, float] = (20.0, 60.0, 140.0)

# Predeclared byte caps (strictly enforced in code)
MIB = 1024 * 1024
DEFAULT_PART_A_BUDGET_BYTES = 60 * MIB
DEFAULT_PART_B_BUDGET_BYTES = 60 * MIB
DEFAULT_GLOBAL_BUDGET_BYTES = 120 * MIB

# Predeclared evaluation gate (NEVER tuned post-hoc)
GATE_MIN_ROC_AUC = 0.70
GATE_MIN_ABOVE_NULL = 0.10
INK_HIGH_U8_THRESHOLD = 179   # >= 0.70 probability on [0, 255] ink map
BLANK_LOW_U8_THRESHOLD = 64   # (0, 0.25] intact blank papyrus (published map blank mode ~32; 0 is masked background)
ROLLED_SHIFT_FRACTIONS: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80)

FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z])[A-Za-z]:\\"),
    re.compile(r"(?<![A-Za-z])[A-Za-z]:/"),
    re.compile(r"/Users/[^/]+/", re.I),
    re.compile(r"\\Users\\[^\\]+\\", re.I),
    re.compile(r"/home/[^/]+/"),
    re.compile(r"/gpfs/", re.I),
    re.compile(r"/data/visitor/", re.I),
    re.compile(r"file://", re.I),
)


# ---------------------------------------------------------------------------
# 1. Privacy / sanitization guardrails & strict HTTP byte budget
# ---------------------------------------------------------------------------

class BudgetExceededError(RuntimeError):
    """Raised before or during a remote fetch that would exceed the HTTP byte cap."""


def assert_no_local_paths(payload: Any) -> None:
    """Raise ValueError if serialized JSON payload contains any local filesystem path."""
    text = json.dumps(payload, sort_keys=True)
    for pat in FORBIDDEN_PATH_PATTERNS:
        m = pat.search(text)
        if m is not None:
            raise ValueError(
                f"Sanitized JSON receipt contains forbidden local path pattern: {m.group(0)!r}"
            )


class HttpByteBudget:
    """Thread-safe strict byte-cap tracker across Part A, Part B, and global HTTP fetches."""

    def __init__(
        self,
        *,
        part_a_limit_bytes: int = DEFAULT_PART_A_BUDGET_BYTES,
        part_b_limit_bytes: int = DEFAULT_PART_B_BUDGET_BYTES,
        global_limit_bytes: int = DEFAULT_GLOBAL_BUDGET_BYTES,
    ) -> None:
        if part_a_limit_bytes <= 0 or part_b_limit_bytes <= 0 or global_limit_bytes <= 0:
            raise ValueError("All byte budget limits must be > 0")
        self._lock = threading.Lock()
        self.part_limits: dict[str, int] = {
            "part_a": int(part_a_limit_bytes),
            "part_b": int(part_b_limit_bytes),
        }
        self.global_limit: int = int(global_limit_bytes)
        self.part_bytes: dict[str, int] = {"part_a": 0, "part_b": 0}
        self.part_requests: dict[str, int] = {"part_a": 0, "part_b": 0}
        self.breakdown: dict[str, int] = {}

    def _validate_part(self, part: str) -> None:
        if part not in self.part_limits:
            raise ValueError(f"Unknown budget part {part!r}; expected 'part_a' or 'part_b'")

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return int(sum(self.part_bytes.values()))

    def remaining(self, part: str) -> int:
        self._validate_part(part)
        with self._lock:
            rem_part = self.part_limits[part] - self.part_bytes[part]
            rem_global = self.global_limit - sum(self.part_bytes.values())
            return max(0, int(min(rem_part, rem_global)))

    def check_planned(self, part: str, planned_bytes: int, *, label: str = "") -> None:
        """Abort cleanly BEFORE fetching if ``planned_bytes`` exceeds part or global cap."""
        self._validate_part(part)
        pb = int(planned_bytes)
        if pb < 0:
            raise ValueError("planned_bytes must be >= 0")
        with self._lock:
            cur_part = self.part_bytes[part]
            cur_global = sum(self.part_bytes.values())
            if cur_part + pb > self.part_limits[part]:
                raise BudgetExceededError(
                    f"Planned fetch {label!r} ({pb} B) would exceed {part} limit "
                    f"({cur_part + pb} B > {self.part_limits[part]} B)"
                )
            if cur_global + pb > self.global_limit:
                raise BudgetExceededError(
                    f"Planned fetch {label!r} ({pb} B) would exceed global limit "
                    f"({cur_global + pb} B > {self.global_limit} B)"
                )

    def charge(self, part: str, nbytes: int, *, label: str = "http") -> None:
        """Charge ``nbytes`` to ``part`` and raise if any cap would be exceeded."""
        self._validate_part(part)
        nb = int(nbytes)
        if nb < 0:
            raise ValueError("nbytes must be >= 0")
        with self._lock:
            cur_part = self.part_bytes[part]
            cur_global = sum(self.part_bytes.values())
            if cur_part + nb > self.part_limits[part]:
                raise BudgetExceededError(
                    f"Fetch {label!r} ({nb} B) exceeded {part} cap "
                    f"({cur_part + nb} B > {self.part_limits[part]} B)"
                )
            if cur_global + nb > self.global_limit:
                raise BudgetExceededError(
                    f"Fetch {label!r} ({nb} B) exceeded global cap "
                    f"({cur_global + nb} B > {self.global_limit} B)"
                )
            self.part_bytes[part] = cur_part + nb
            self.part_requests[part] += 1
            if label:
                self.breakdown[label] = self.breakdown.get(label, 0) + nb

    def summary(self) -> dict[str, Any]:
        with self._lock:
            total = int(sum(self.part_bytes.values()))
            return {
                "part_a_bytes": int(self.part_bytes["part_a"]),
                "part_a_mib": round(self.part_bytes["part_a"] / MIB, 4),
                "part_a_limit_mib": round(self.part_limits["part_a"] / MIB, 4),
                "part_b_bytes": int(self.part_bytes["part_b"]),
                "part_b_mib": round(self.part_bytes["part_b"] / MIB, 4),
                "part_b_limit_mib": round(self.part_limits["part_b"] / MIB, 4),
                "global_bytes": total,
                "global_mib": round(total / MIB, 4),
                "global_limit_mib": round(self.global_limit / MIB, 4),
                "within_limits": bool(
                    self.part_bytes["part_a"] <= self.part_limits["part_a"]
                    and self.part_bytes["part_b"] <= self.part_limits["part_b"]
                    and total <= self.global_limit
                ),
                "requests": dict(self.part_requests),
                "breakdown_bytes": dict(sorted(self.breakdown.items())),
            }


def http_head_content_length(url: str, *, timeout: float = 20.0) -> int | None:
    """Return remote Content-Length via HTTP HEAD without downloading body."""
    if requests is not None:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl is not None else None
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl is not None else None


def budgeted_http_get(
    url: str,
    budget: HttpByteBudget,
    part: str,
    *,
    label: str = "http",
    timeout: float = 45.0,
    allow_404: bool = False,
    session: Any | None = None,
) -> bytes | None:
    """Fetch HTTP(S) resource strictly in RAM, enforcing ``budget`` incrementally."""
    rem = budget.remaining(part)
    if rem <= 0:
        raise BudgetExceededError(f"Zero remaining budget in {part} before fetching {label!r}")

    if requests is not None:
        sess = session if session is not None else requests
        with sess.get(url, timeout=timeout, stream=True) as resp:
            status = int(resp.status_code)
            if status == 404 and allow_404:
                return None
            resp.raise_for_status()
            cl = resp.headers.get("Content-Length")
            if cl is not None:
                budget.check_planned(part, int(cl), label=label)
            buf = bytearray()
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                if len(buf) + len(chunk) > rem:
                    raise BudgetExceededError(
                        f"Streamed response for {label!r} exceeded remaining {part} budget ({rem} B)"
                    )
                buf.extend(chunk)
            data = bytes(buf)
            budget.charge(part, len(data), label=label)
            return data

    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cl = resp.headers.get("Content-Length")
            if cl is not None:
                budget.check_planned(part, int(cl), label=label)
            buf = bytearray()
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                if len(buf) + len(chunk) > rem:
                    raise BudgetExceededError(
                        f"Streamed response for {label!r} exceeded remaining {part} budget ({rem} B)"
                    )
                buf.extend(chunk)
            data = bytes(buf)
            budget.charge(part, len(data), label=label)
            return data
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and allow_404:
            return None
        raise


class BudgetedRemoteZarrV2(RemoteZarrV2):
    """Subclass of ``stream_snap_render.RemoteZarrV2`` enforcing ``HttpByteBudget``.

    Reuses ``RemoteZarrV2`` decoding, 404-only empty-chunk handling, LRU caching,
    and positive-weight trilinear source support without modifying
    ``stream_snap_render.py``.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        level: str | int | None = "0",
        *,
        budget: HttpByteBudget,
        part: str,
        label: str = "zarr",
        cache_mb: float = 96.0,
        workers: int = 4,
        timeout: float = 60.0,
    ) -> None:
        self._budget = budget
        self._part = part
        self._label = label
        super().__init__(
            root,
            level=level,
            cache_mb=cache_mb,
            workers=workers,
            timeout=timeout,
        )

    def _get_bytes(self, key: str, *, allow_404: bool = True) -> bytes | None:
        if self.local:
            return super()._get_bytes(key, allow_404=allow_404)
        url = f"{self.root}/{key}"
        return budgeted_http_get(
            url,
            self._budget,
            self._part,
            label=f"{self._label}:{'meta' if key.endswith('.zarray') else 'chunk'}",
            timeout=self.timeout,
            allow_404=allow_404,
            session=self._session(),
        )

    def _uncached_chunk_upper_bytes(self, idxs: Sequence[tuple[int, int, int]]) -> tuple[int, int]:
        unique_idxs = list(dict.fromkeys(tuple(int(v) for v in i) for i in idxs))
        with self._lock:
            missing = [idx for idx in unique_idxs if idx not in self.cache]
        # Exact for uncompressed Zarr (compressor=None) and conservative upper bound otherwise.
        return len(missing), len(missing) * int(self._expected_chunk_bytes)

    def plan_box_chunks(
        self, lo_zyx: Sequence[int], hi_zyx: Sequence[int]
    ) -> tuple[int, int]:
        """Return ``(uncached_chunk_count, upper_bound_bytes)`` for ``read_box``."""
        lo = [max(0, int(v)) for v in lo_zyx]
        hi = [min(s, int(v)) for s, v in zip(self.shape, hi_zyx)]
        if any(h <= l for l, h in zip(lo, hi)):
            return 0, 0
        c0 = [l // c for l, c in zip(lo, self.chunks)]
        c1 = [(h - 1) // c for h, c in zip(hi, self.chunks)]
        idxs = [
            (cz, cy, cx)
            for cz in range(c0[0], c1[0] + 1)
            for cy in range(c0[1], c1[1] + 1)
            for cx in range(c0[2], c1[2] + 1)
        ]
        return self._uncached_chunk_upper_bytes(idxs)

    def plan_trilinear_chunks(
        self, xyz: np.ndarray, *, max_direct_box_voxels: int = 1
    ) -> tuple[int, int]:
        """Exact chunk planner matching ``RemoteZarrV2.sample_trilinear`` chunk cells."""
        pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            return 0, 0
        sz, sy, sx = self.shape
        in_bounds = (
            np.isfinite(pts).all(axis=-1)
            & (pts[:, 0] >= 0.0) & (pts[:, 0] <= sx - 1.0)
            & (pts[:, 1] >= 0.0) & (pts[:, 1] <= sy - 1.0)
            & (pts[:, 2] >= 0.0) & (pts[:, 2] <= sz - 1.0)
        )
        if not in_bounds.any():
            return 0, 0
        q_zyx = pts[in_bounds][:, ::-1]
        lo = np.maximum(0, np.floor(q_zyx.min(axis=0)).astype(int) - 1)
        hi = np.minimum(self.shape, np.ceil(q_zyx.max(axis=0)).astype(int) + 2)
        box_voxels = int(np.prod(np.maximum(0, hi - lo)))
        if box_voxels <= max_direct_box_voxels:
            return self.plan_box_chunks(lo.tolist(), hi.tolist())

        cshape = np.asarray(self.chunks, dtype=np.int64)
        # Every trilinear sample at q_zyx touches corners floor(q) and ceil(q) (+/- 1 cell
        # for read_box(slo, shi) where slo = floor - 1, shi = ceil + 2).
        # Compute exact cell-grouped bounds without slow void-view mergesort.
        cell_ids = np.floor(q_zyx).astype(np.int64) // cshape[None, :]
        n_cy = (int(sy) + int(self.chunks[1]) - 1) // int(self.chunks[1])
        n_cx = (int(sx) + int(self.chunks[2]) - 1) // int(self.chunks[2])
        packed = (cell_ids[:, 0] * n_cy + cell_ids[:, 1]) * n_cx + cell_ids[:, 2]
        uniq_packed, inv = np.unique(packed, return_inverse=True)
        # For each unique chunk cell, find min/max q_zyx inside that cell using ufunc.at.
        n_u = int(uniq_packed.size)
        min_z = np.full(n_u, np.inf, dtype=np.float64)
        min_y = np.full(n_u, np.inf, dtype=np.float64)
        min_x = np.full(n_u, np.inf, dtype=np.float64)
        max_z = np.full(n_u, -np.inf, dtype=np.float64)
        max_y = np.full(n_u, -np.inf, dtype=np.float64)
        max_x = np.full(n_u, -np.inf, dtype=np.float64)
        np.minimum.at(min_z, inv, q_zyx[:, 0])
        np.minimum.at(min_y, inv, q_zyx[:, 1])
        np.minimum.at(min_x, inv, q_zyx[:, 2])
        np.maximum.at(max_z, inv, q_zyx[:, 0])
        np.maximum.at(max_y, inv, q_zyx[:, 1])
        np.maximum.at(max_x, inv, q_zyx[:, 2])

        all_idxs: set[tuple[int, int, int]] = set()
        for k in range(n_u):
            slo = (
                max(0, int(math.floor(min_z[k])) - 1),
                max(0, int(math.floor(min_y[k])) - 1),
                max(0, int(math.floor(min_x[k])) - 1),
            )
            shi = (
                min(int(sz), int(math.ceil(max_z[k])) + 2),
                min(int(sy), int(math.ceil(max_y[k])) + 2),
                min(int(sx), int(math.ceil(max_x[k])) + 2),
            )
            c0 = (slo[0] // int(self.chunks[0]), slo[1] // int(self.chunks[1]), slo[2] // int(self.chunks[2]))
            c1 = ((shi[0] - 1) // int(self.chunks[0]), (shi[1] - 1) // int(self.chunks[1]), (shi[2] - 1) // int(self.chunks[2]))
            for cz in range(c0[0], c1[0] + 1):
                for cy in range(c0[1], c1[1] + 1):
                    for cx in range(c0[2], c1[2] + 1):
                        all_idxs.add((cz, cy, cx))
        return self._uncached_chunk_upper_bytes(sorted(all_idxs))


# ---------------------------------------------------------------------------
# 2. In-RAM loaders for tifxyz meshes, ink map, and volume metadata
# ---------------------------------------------------------------------------

def load_tifxyz_in_ram(
    root: str | os.PathLike[str],
    budget: HttpByteBudget | None = None,
    part: str = "part_b",
    *,
    label: str = "mesh",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a small tifxyz mesh strictly in RAM (HTTP or local), validated with 3-channel sentinels."""
    root_str = str(root).rstrip("/\\")
    is_http = root_str.lower().startswith(("http://", "https://"))
    if not is_http:
        p = Path(root_str)
        x = tifffile.imread(str(p / "x.tif")).astype(np.float32)
        y = tifffile.imread(str(p / "y.tif")).astype(np.float32)
        z = tifffile.imread(str(p / "z.tif")).astype(np.float32)
        meta_path = p / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    else:
        assert budget is not None
        meta_raw = budgeted_http_get(f"{root_str}/meta.json", budget, part, label=f"{label}:meta")
        assert meta_raw is not None
        meta = json.loads(meta_raw.decode("utf-8"))
        arrs: dict[str, np.ndarray] = {}
        for ch in ("x", "y", "z"):
            raw = budgeted_http_get(f"{root_str}/{ch}.tif", budget, part, label=f"{label}:tif")
            assert raw is not None
            arrs[ch] = tifffile.imread(io.BytesIO(raw)).astype(np.float32)
        x, y, z = arrs["x"], arrs["y"], arrs["z"]

    cx, cy, cz, valid = validate_tifxyz_channels(x, y, z)
    return cx, cy, cz, valid, meta


def load_ink_map_in_ram(
    source: str | os.PathLike[str],
    budget: HttpByteBudget | None = None,
    part: str = "part_a",
) -> np.ndarray:
    """Load the 2D uint8 ink reference TIFF strictly in RAM (zero disk write)."""
    src = str(source)
    if not src.lower().startswith(("http://", "https://")):
        arr = tifffile.imread(src)
    else:
        assert budget is not None
        cl = http_head_content_length(src)
        if cl is not None:
            budget.check_planned(part, cl, label="ink_reference_tif")
        raw = budgeted_http_get(src, budget, part, label="ink_reference_tif")
        assert raw is not None
        arr = tifffile.imread(io.BytesIO(raw))
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D ink reference map, got shape {arr.shape}")
    return arr.astype(np.uint8, copy=False)


def load_paganin_volume_metadata(
    zarr_root: str | os.PathLike[str],
    budget: HttpByteBudget | None = None,
    part: str = "part_b",
    *,
    label: str = "vol_meta",
) -> dict[str, Any]:
    """Fetch and sanitize numeric Paganin / window_f32 metadata (never keeping raw cluster paths)."""
    root_str = str(zarr_root).rstrip("/\\")
    raw_meta: dict[str, Any] = {}
    if root_str.lower().startswith(("http://", "https://")):
        assert budget is not None
        raw = budgeted_http_get(
            f"{root_str}/metadata.json", budget, part, label=label, allow_404=True
        )
        if raw is not None:
            raw_meta = json.loads(raw.decode("utf-8"))
    else:
        mp = Path(root_str) / "metadata.json"
        if mp.is_file():
            raw_meta = json.loads(mp.read_text(encoding="utf-8"))

    ze = raw_meta.get("zarr_export", {}) if isinstance(raw_meta, dict) else {}
    proc = raw_meta.get("scan", {}).get("tomo", {}).get("processing", {}) if isinstance(raw_meta, dict) else {}
    phase = proc.get("preprocessing", {}).get("phase", {}) if isinstance(proc, dict) else {}
    conv = proc.get("postprocessing", {}).get("32BitsConversion", {}) if isinstance(proc, dict) else {}
    acq = raw_meta.get("scan", {}).get("tomo", {}).get("acquisition", {}) if isinstance(raw_meta, dict) else {}

    out: dict[str, Any] = {
        "energy_keV": float(acq["energy"]) if "energy" in acq else None,
        "propagation_distance_mm": (
            float(acq["sampleDetectorDistance"]) if "sampleDetectorDistance" in acq else None
        ),
        "phase_method": str(phase["method"]) if "method" in phase else None,
        "delta_beta": float(phase["delta_beta"]) if "delta_beta" in phase else None,
        "unsharp_coeff": float(phase["unsharp_coeff"]) if "unsharp_coeff" in phase else None,
        "unsharp_sigma": float(phase["unsharp_sigma"]) if "unsharp_sigma" in phase else None,
        "target_window_f32": (
            [float(ze["target_window_f32_min"]), float(ze["target_window_f32_max"])]
            if "target_window_f32_min" in ze and "target_window_f32_max" in ze
            else None
        ),
        "conversion_32bit_window": (
            [float(conv["dataset_used_min"]), float(conv["dataset_used_max"])]
            if "dataset_used_min" in conv and "dataset_used_max" in conv
            else None
        ),
    }
    return out


# ---------------------------------------------------------------------------
# 3. Coordinate mapping (2.399 um canvas <-> 9.362 um tifxyz meshes)
# ---------------------------------------------------------------------------

def map_canvas_to_mesh_uv(
    rows_sv: np.ndarray,
    cols_sv: np.ndarray,
    sv_hw: tuple[int, int],
    mesh_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Center-aligned mapping from surface-volume ``(row, col)`` to tifxyz ``(gv, gu)``.

    Matches the flattened segment canvas alignment verified to ``< 0.01 vx``
    median bias between ``pag1000`` ``(89, 268)`` and ``pag50``/``pag0`` ``(87, 261)``.
    """
    h_sv, w_sv = int(sv_hw[0]), int(sv_hw[1])
    h_m, w_m = int(mesh_hw[0]), int(mesh_hw[1])
    gv = (np.asarray(rows_sv, dtype=np.float64) + 0.5) * (h_m / float(h_sv)) - 0.5
    gu = (np.asarray(cols_sv, dtype=np.float64) + 0.5) * (w_m / float(w_sv)) - 0.5
    return gv, gu


def build_part_b_rays_for_roi(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    *,
    sv_hw: tuple[int, int],
    roi_bounds: tuple[int, int, int, int],
    stride_sv: int = 4,
    num_slices: int = 21,
    slice_step: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate 3D normal rays ``(L, H_b, W_b, 3)`` and mesh usability mask for Part B ROI.

    Uses ``stream_snap_render`` ``compute_grid_normals`` and
    ``_positive_weight_stencil_support`` so sentinel vertices never corrupt rays.
    """
    r0, r1, c0, c1 = [int(v) for v in roi_bounds]
    h_b = (r1 - r0) // stride_sv
    w_b = (c1 - c0) // stride_sv
    if h_b < 2 or w_b < 2:
        raise ValueError(f"ROI too small for stride {stride_sv}: {roi_bounds}")

    cx, cy, cz, val = validate_tifxyz_channels(x, y, z)
    val = val & np.asarray(valid, dtype=bool)
    normals, normal_ok = compute_grid_normals(cx, cy, cz, valid=val)
    usable = val & normal_ok

    # Pixel centers of each stride_sv x stride_sv block in 2.399 um canvas coordinates.
    rows_1d = r0 + np.arange(h_b, dtype=np.float64) * stride_sv + 0.5 * (stride_sv - 1.0)
    cols_1d = c0 + np.arange(w_b, dtype=np.float64) * stride_sv + 0.5 * (stride_sv - 1.0)
    rr, cc = np.meshgrid(rows_1d, cols_1d, indexing="ij")
    gv, gu = map_canvas_to_mesh_uv(rr, cc, sv_hw, usable.shape)

    mesh_coords = np.stack([np.zeros(gv.size), gv.ravel(), gu.ravel()])
    mesh_usable = _positive_weight_stencil_support(
        usable[None, ...], mesh_coords
    ).reshape(h_b, w_b)

    pts = np.stack([cx, cy, cz], axis=-1).astype(np.float64)
    if val.any() and not val.all():
        idx_v = ndi.distance_transform_edt(~val, return_distances=False, return_indices=True)
        ptsf = pts[idx_v[0], idx_v[1]]
    else:
        ptsf = pts
    if usable.any() and not usable.all():
        idx_u = ndi.distance_transform_edt(~usable, return_distances=False, return_indices=True)
        nrmf = normals.astype(np.float64)[idx_u[0], idx_u[1]]
    else:
        nrmf = normals.astype(np.float64)

    p_out = np.zeros((h_b, w_b, 3), dtype=np.float64)
    n_out = np.zeros((h_b, w_b, 3), dtype=np.float64)
    if mesh_usable.any():
        gvm = gv[mesh_usable]
        gum = gu[mesh_usable]
        for ch in range(3):
            p_out[mesh_usable, ch] = ndi.map_coordinates(
                ptsf[..., ch], [gvm, gum], order=1, mode="nearest"
            )
            n_out[mesh_usable, ch] = ndi.map_coordinates(
                nrmf[..., ch], [gvm, gum], order=1, mode="nearest"
            )
        n_len = np.linalg.norm(n_out, axis=-1, keepdims=True)
        n_good = mesh_usable & (n_len[..., 0] > 1e-6)
        mesh_usable = n_good
        n_out[mesh_usable] /= n_len[mesh_usable]
        n_out[~mesh_usable] = 0.0

    center_idx = (num_slices - 1) // 2
    offsets = (np.arange(num_slices, dtype=np.float64) - float(center_idx)) * float(slice_step)
    rays = p_out[None, :, :, :] + offsets[:, None, None, None] * n_out[None, :, :, :]
    return rays, mesh_usable


def select_coregistered_roi(
    ink_map: np.ndarray,
    sv_chunks: tuple[int, int, int],
    part_b_meshes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]],
    part_b_readers: dict[str, BudgetedRemoteZarrV2],
    *,
    roi_h: int = 512,
    roi_w: int = 640,
    part_b_max_planned_bytes: int = 54 * MIB,
    part_b_num_slices: int = 21,
    stride_sv: int = 4,
    ink_thresh: int = INK_HIGH_U8_THRESHOLD,
    blank_thresh: int = BLANK_LOW_U8_THRESHOLD,
) -> dict[str, Any]:
    """Deterministically select a chunk-aligned co-registered ROI with ink + blank papyrus.

    Selection uses ONLY the ink reference map, mesh validity masks, and planned
    Zarr chunk counts (zero peeking at CT pixel values).
    """
    h_sv, w_sv = ink_map.shape
    cy_sz, cx_sz = int(sv_chunks[1]), int(sv_chunks[2])
    if roi_h % cy_sz != 0 or roi_w % cx_sz != 0:
        raise ValueError(f"ROI {(roi_h, roi_w)} must align to surface-volume chunks {(cy_sz, cx_sz)}")

    # Precompute mesh usability grids once per variant.
    mesh_usables: dict[str, np.ndarray] = {}
    for name, (mx, my, mz, mval, _) in part_b_meshes.items():
        _, n_ok = compute_grid_normals(mx, my, mz, valid=mval)
        mesh_usables[name] = mval & n_ok

    # First-stage fast, tiny-RAM scan on the chunk-tile grid (H//cy_sz x W//cx_sz).
    n_cy = h_sv // cy_sz
    n_cx = w_sv // cx_sz
    ny_win = roi_h // cy_sz
    nx_win = roi_w // cx_sz
    tile_ink = np.zeros((n_cy, n_cx), dtype=np.int64)
    tile_blank = np.zeros((n_cy, n_cx), dtype=np.int64)
    tile_val = np.zeros((n_cy, n_cx), dtype=np.int64)
    for ty in range(n_cy):
        strip = ink_map[ty * cy_sz : (ty + 1) * cy_sz, : n_cx * cx_sz]
        strip_t = strip.reshape(cy_sz, n_cx, cx_sz)
        im = strip_t >= ink_thresh
        bm = (strip_t > 0) & (strip_t <= blank_thresh)
        tile_ink[ty] = im.sum(axis=(0, 2))
        tile_blank[ty] = bm.sum(axis=(0, 2))
        tile_val[ty] = np.where(im, strip_t, 0).sum(axis=(0, 2), dtype=np.int64)

    def _integral(a: np.ndarray) -> np.ndarray:
        out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.int64)
        out[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
        return out

    ii_ink = _integral(tile_ink)
    ii_blank = _integral(tile_blank)
    ii_val = _integral(tile_val)
    area = float(roi_h * roi_w)

    candidates: list[tuple[float, int, int, float, float, float]] = []
    for ty0 in range(0, n_cy - ny_win + 1):
        ty1 = ty0 + ny_win
        r0, r1 = ty0 * cy_sz, ty1 * cy_sz
        for tx0 in range(0, n_cx - nx_win + 1):
            tx1 = tx0 + nx_win
            c0, c1 = tx0 * cx_sz, tx1 * cx_sz
            n_ink = int(ii_ink[ty1, tx1] - ii_ink[ty0, tx1] - ii_ink[ty1, tx0] + ii_ink[ty0, tx0])
            n_blank = int(ii_blank[ty1, tx1] - ii_blank[ty0, tx1] - ii_blank[ty1, tx0] + ii_blank[ty0, tx0])
            ink_frac = n_ink / area
            blank_frac = n_blank / area
            if ink_frac < 0.08 or ink_frac > 0.50 or blank_frac < 0.30:
                continue
            # Quick mesh interior validity check across all 3 variants.
            ok_meshes = True
            for name, usable in mesh_usables.items():
                h_m, w_m = usable.shape
                gv0 = (r0 + 0.5) * (h_m / float(h_sv)) - 0.5
                gv1 = (r1 - 0.5) * (h_m / float(h_sv)) - 0.5
                gu0 = (c0 + 0.5) * (w_m / float(w_sv)) - 0.5
                gu1 = (c1 - 0.5) * (w_m / float(w_sv)) - 0.5
                iv0, iv1 = int(math.floor(gv0)), int(math.ceil(gv1))
                iu0, iu1 = int(math.floor(gu0)), int(math.ceil(gu1))
                if iv0 < 0 or iu0 < 0 or iv1 >= h_m or iu1 >= w_m:
                    ok_meshes = False
                    break
                if float(usable[iv0 : iv1 + 1, iu0 : iu1 + 1].mean()) < 0.99:
                    ok_meshes = False
                    break
            if not ok_meshes:
                continue
            sum_val = int(ii_val[ty1, tx1] - ii_val[ty0, tx1] - ii_val[ty1, tx0] + ii_val[ty0, tx0])
            mean_ink_conf = (sum_val / max(n_ink, 1)) / 255.0
            balance = min(ink_frac, blank_frac)
            score = balance * mean_ink_conf
            candidates.append((score, r0, c0, ink_frac, blank_frac, mean_ink_conf))

    if not candidates:
        raise RuntimeError(
            "No chunk-aligned ROI satisfies ink/blank balance and 3-mesh coverage requirements."
        )

    # Evaluate exact Part B chunk footprint on top candidates (highest ink/blank balance first).
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    best_choice: dict[str, Any] | None = None
    best_rank: tuple[int, float] | None = None

    for score, r0, c0, ink_frac, blank_frac, mean_ink_conf in candidates[:250]:
        bounds = (r0, r0 + roi_h, c0, c0 + roi_w)
        total_b_bytes = 0
        per_variant_chunks: dict[str, int] = {}
        min_mesh_frac = 1.0
        for name, (mx, my, mz, mval, _) in part_b_meshes.items():
            rays, m_use = build_part_b_rays_for_roi(
                mx, my, mz, mval,
                sv_hw=(h_sv, w_sv),
                roi_bounds=bounds,
                stride_sv=stride_sv,
                num_slices=part_b_num_slices,
                slice_step=1.0,
            )
            min_mesh_frac = min(min_mesh_frac, float(m_use.mean()))
            if min_mesh_frac < 0.98:
                total_b_bytes = part_b_max_planned_bytes + 1
                break
            n_ch, b_bytes = part_b_readers[name].plan_trilinear_chunks(
                rays[:, m_use, :], max_direct_box_voxels=1
            )
            per_variant_chunks[name] = int(n_ch)
            total_b_bytes += int(b_bytes)

        if total_b_bytes <= part_b_max_planned_bytes and min_mesh_frac >= 0.98:
            # Prefer high balance/confidence while comfortably under Part B byte cap.
            # Since candidates are sorted by descending score, the first rank[0]==0 hit is optimal.
            rank = (0 if total_b_bytes <= int(0.85 * part_b_max_planned_bytes) else 1, -score)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_choice = {
                    "roi_bounds_2p4um": [int(r0), int(r0 + roi_h), int(c0), int(c0 + roi_w)],
                    "roi_shape_2p4um": [int(roi_h), int(roi_w)],
                    "roi_physical_mm": [
                        round(roi_h * VOXEL_UM_PART_A / 1000.0, 4),
                        round(roi_w * VOXEL_UM_PART_A / 1000.0, 4),
                    ],
                    "ink_fraction_high": round(ink_frac, 4),
                    "blank_fraction_low": round(blank_frac, 4),
                    "mean_high_ink_prob": round(mean_ink_conf, 4),
                    "min_part_b_mesh_valid_fraction": round(min_mesh_frac, 4),
                    "planned_part_b_chunks": per_variant_chunks,
                    "planned_part_b_chunk_bytes": int(total_b_bytes),
                    "planned_part_b_chunk_mib": round(total_b_bytes / MIB, 4),
                    "candidates_screened": len(candidates),
                }
                if rank[0] == 0:
                    break

    if best_choice is None:
        raise BudgetExceededError(
            f"None of the top {min(len(candidates), 80)} balanced ROIs fit within "
            f"Part B planned chunk cap ({part_b_max_planned_bytes / MIB:.2f} MiB)."
        )
    return best_choice


# ---------------------------------------------------------------------------
# 4. Part A Physics: Sub-voxel height field, MSII, curvature, FFT + StructN2V
# ---------------------------------------------------------------------------

def _fill_nearest_2d(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.all() or not valid.any():
        return arr.copy()
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return arr[idx[0], idx[1]]


def _normalized_gaussian(arr: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur normalized by valid pixel support so invalid borders never bleed."""
    if sigma <= 0.0:
        return arr.astype(np.float64, copy=True)
    w = valid.astype(np.float64)
    num = ndi.gaussian_filter(np.where(valid, arr.astype(np.float64), 0.0), sigma=sigma, mode="nearest")
    den = ndi.gaussian_filter(w, sigma=sigma, mode="nearest")
    return np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), 0.0)


def extract_subvoxel_height_field(
    slab: np.ndarray,
    *,
    voxel_um: float,
    valid_mask: np.ndarray | None = None,
    search_half_slices: int | None = None,
    z_smooth_sigma: float = 1.0,
    detrend_sigma_um: float = 200.0,
    band_half_width: int = 1,
) -> dict[str, Any]:
    """Extract sub-voxel papyrus peak height-field ``h(u, v) = n_peak - G(n_peak)`` and surface CT.

    Finds the central papyrus sheet peak near the middle of ``slab`` (shape ``[L, H, W]``),
    refines ``n_peak`` with 3-point parabolic sub-voxel interpolation, detrends slow
    macroscopic waviness via normalized Gaussian filtering, and samples the
    sheet-aligned surface CT band around ``n_peak``.
    """
    vol = np.asarray(slab, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D slab [L, H, W], got {vol.shape}")
    depth, h, w = vol.shape
    if depth < 3:
        raise ValueError(f"Slab depth must be >= 3, got {depth}")

    vm = np.ones((h, w), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).copy()
    if vm.shape != (h, w):
        raise ValueError(f"valid_mask shape {vm.shape} != {(h, w)}")

    # Smooth lightly along depth only (zero lateral blur before peak detection).
    vol_s = ndi.gaussian_filter1d(vol, sigma=float(z_smooth_sigma), axis=0, mode="nearest") if z_smooth_sigma > 0 else vol

    # Locate the central sheet layer from the median intensity profile within central half.
    c_lo = max(1, depth // 4)
    c_hi = min(depth - 1, depth - depth // 4)
    if c_hi <= c_lo:
        c_lo, c_hi = 0, depth
    if vm.any():
        med_profile = np.median(vol_s[:, vm], axis=1)
    else:
        med_profile = np.median(vol_s.reshape(depth, -1), axis=1)
    z_center = int(c_lo + int(np.argmax(med_profile[c_lo:c_hi])))

    half = int(search_half_slices) if search_half_slices is not None else min(24, max(2, depth // 3))
    z_lo = max(0, z_center - half)
    z_hi = min(depth, z_center + half + 1)
    sub = vol_s[z_lo:z_hi]  # (W_z, H, W)
    w_z = sub.shape[0]

    k_max = np.argmax(sub, axis=0)  # (H, W) in [0, w_z - 1]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    peak_val = sub[k_max, yy, xx].astype(np.float64)

    # 3-point parabolic sub-voxel interpolation on interior maxima.
    interior = (k_max > 0) & (k_max < w_z - 1)
    km_safe = np.clip(k_max, 1, max(1, w_z - 2))
    y_l = sub[km_safe - 1, yy, xx].astype(np.float64)
    y_c = sub[km_safe, yy, xx].astype(np.float64)
    y_r = sub[km_safe + 1, yy, xx].astype(np.float64)
    denom = y_l - 2.0 * y_c + y_r
    good_parab = interior & (denom < -1e-6) & (y_c >= y_l) & (y_c >= y_r)
    frac = np.where(good_parab, np.clip(0.5 * (y_l - y_r) / np.where(good_parab, denom, -1.0), -0.5, 0.5), 0.0)
    n_peak = (z_lo + k_max.astype(np.float64) + frac)

    # Exclude empty/zero-peak columns from height analysis.
    sheet_valid = vm & np.isfinite(n_peak) & (peak_val > 0.0)

    # Detrend slow sheet waviness via normalized Gaussian filter.
    sigma_detrend_px = float(detrend_sigma_um) / float(voxel_um)
    n_bg = _normalized_gaussian(n_peak, sheet_valid, sigma=sigma_detrend_px)
    h_vox = np.where(sheet_valid, n_peak - n_bg, 0.0)
    h_um = h_vox * float(voxel_um)

    # Sample sheet-aligned surface CT band [n_peak - band_half_width .. n_peak + band_half_width]
    # via linear depth interpolation on the unsmoothed slab.
    offsets = np.arange(-int(band_half_width), int(band_half_width) + 1, dtype=np.float64)
    surf_acc = np.zeros((h, w), dtype=np.float64)
    for off in offsets:
        z_q = np.clip(n_peak + off, 0.0, float(depth - 1))
        z0 = np.floor(z_q).astype(np.int64)
        z1 = np.minimum(z0 + 1, depth - 1)
        wz = z_q - z0
        v0 = vol[z0, yy, xx].astype(np.float64)
        v1 = vol[z1, yy, xx].astype(np.float64)
        surf_acc += (1.0 - wz) * v0 + wz * v1
    surface_ct = (surf_acc / float(len(offsets))).astype(np.float32)

    return {
        "z_center_slice": z_center,
        "z_search_window": [int(z_lo), int(z_hi)],
        "n_peak": n_peak.astype(np.float32),
        "h_um": h_um.astype(np.float32),
        "surface_ct": surface_ct,
        "peak_intensity": peak_val.astype(np.float32),
        "sheet_valid": sheet_valid,
    }


def compute_msii_spherical_occupancy(
    h_um: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    radius_um: float,
    max_samples_axis: int = 11,
    row_block: int = 16,
) -> np.ndarray:
    """Compute exact 3D spherical-volume occupancy MSII ``V_r(u, v) in [0, 1]`` on ``z = h(u, v)``.

    For a flat plane, ``V_r == 0.5`` identically (strictly invariant to ``h -> h + c``).
    Convex bumps (raised ink relief) give ``V_r < 0.5`` (``convexity = 0.5 - V_r > 0``);
    concave pits give ``V_r > 0.5``.
    """
    if radius_um <= 0.0 or voxel_um <= 0.0:
        raise ValueError("radius_um and voxel_um must be positive")
    h_arr = np.asarray(h_um, dtype=np.float32)
    vm = np.asarray(valid_mask, dtype=bool)
    h_filled = _fill_nearest_2d(h_arr, vm).astype(np.float32, copy=False)
    rows, cols = h_filled.shape

    r_px = float(radius_um) / float(voxel_um)
    stride = max(1, int(round(r_px / float(max_samples_axis))))
    max_step = int(math.floor(r_px / stride)) * stride
    offs = np.arange(-max_step, max_step + 1, stride, dtype=np.int64)
    dy_g, dx_g = np.meshgrid(offs, offs, indexing="ij")
    rho2_um = (dy_g * float(voxel_um)) ** 2 + (dx_g * float(voxel_um)) ** 2
    in_disk = rho2_um < (float(radius_um) ** 2)
    dy = dy_g[in_disk]
    dx = dx_g[in_disk]
    cap_half = np.sqrt(np.maximum(0.0, float(radius_um) ** 2 - rho2_um[in_disk])).astype(np.float32)
    cap_full = (2.0 * cap_half).astype(np.float32)
    w_tot = float(cap_full.sum())
    if w_tot <= 0.0:
        return np.full((rows, cols), 0.5, dtype=np.float32)

    pad = int(max(np.max(np.abs(dy)), np.max(np.abs(dx))))
    h_pad = np.pad(h_filled, pad, mode="reflect")
    out = np.empty((rows, cols), dtype=np.float32)

    rb = max(1, int(row_block))
    for r0 in range(0, rows, rb):
        r1 = min(r0 + rb, rows)
        center = h_filled[r0:r1, :]  # (B, W)
        acc = np.zeros((r1 - r0, cols), dtype=np.float32)
        for d_y, d_x, ch, cf in zip(dy.tolist(), dx.tolist(), cap_half.tolist(), cap_full.tolist()):
            neigh = h_pad[r0 + pad + d_y : r1 + pad + d_y, pad + d_x : pad + d_x + cols]
            acc += np.clip((neigh - center) + ch, 0.0, cf)
        out[r0:r1, :] = acc / np.float32(w_tot)
    return out


def compute_msii_and_curvature(
    h_um: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    radii_um: Sequence[float] = MSII_RADII_UM,
    curvature_sigma_um: float | None = None,
    roughness_sigma_um: float = 20.0,
) -> dict[str, np.ndarray]:
    """Compute multi-scale MSII convexity, mean/Gaussian curvature, and relief roughness."""
    vm = np.asarray(valid_mask, dtype=bool)
    h_arr = np.asarray(h_um, dtype=np.float32)
    h_filled = _fill_nearest_2d(h_arr, vm).astype(np.float64)

    feats: dict[str, np.ndarray] = {
        "height_detrended_um": h_arr.astype(np.float32),
    }

    occupancies: dict[float, np.ndarray] = {}
    for r_um in radii_um:
        vr = compute_msii_spherical_occupancy(h_arr, vm, voxel_um=voxel_um, radius_um=float(r_um))
        occupancies[float(r_um)] = vr
        tag = f"{int(round(r_um))}um"
        feats[f"msii_convexity_{tag}"] = (0.5 - vr).astype(np.float32)

    if len(radii_um) >= 2:
        r_lo, r_hi = float(min(radii_um)), float(max(radii_um))
        feats[f"msii_diff_{int(round(r_lo))}_{int(round(r_hi))}um"] = (
            occupancies[r_hi] - occupancies[r_lo]
        ).astype(np.float32)

    # Local mean curvature H and Gaussian curvature K of Monge patch z = h(u, v).
    sig_curv_um = float(curvature_sigma_um) if curvature_sigma_um is not None else max(8.0, 1.5 * voxel_um)
    h_sm = ndi.gaussian_filter(h_filled, sigma=sig_curv_um / float(voxel_um), mode="reflect")
    hv, hu = np.gradient(h_sm, float(voxel_um))
    hvv, hvu = np.gradient(hv, float(voxel_um))
    huv, huu = np.gradient(hu, float(voxel_um))
    huv = 0.5 * (huv + hvu)
    denom = 1.0 + hu * hu + hv * hv
    mean_curv = ((1.0 + hv * hv) * huu - 2.0 * hu * hv * huv + (1.0 + hu * hu) * hvv) / (
        2.0 * np.power(denom, 1.5)
    )
    gauss_curv = (huu * hvv - huv * huv) / (denom * denom)
    feats["mean_curvature_convex"] = (-mean_curv).astype(np.float32)
    feats["gaussian_curvature"] = gauss_curv.astype(np.float32)

    # Local relief roughness (local std of h_um over roughness_sigma_um).
    sig_r_px = float(roughness_sigma_um) / float(voxel_um)
    m1 = _normalized_gaussian(h_arr, vm, sigma=sig_r_px)
    m2 = _normalized_gaussian(h_arr.astype(np.float64) ** 2, vm, sigma=sig_r_px)
    rough = np.sqrt(np.maximum(0.0, m2 - m1 * m1))
    feats[f"relief_roughness_{int(round(roughness_sigma_um))}um"] = rough.astype(np.float32)
    return feats


def estimate_fiber_axes_fft(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    min_wavelength_um: float = 25.0,
    max_wavelength_um: float = 160.0,
    ortho_tolerance_deg: float = 20.0,
) -> dict[str, Any]:
    """Estimate dominant quasi-orthogonal papyrus fiber axes from 2D FFT power spectrum.

    Convention:
    - Frequency angle ``theta_deg = degrees(atan2(fy, fx)) % 180`` points along the
      wavevector perpendicular to the fiber ridges.
    - Spatial fiber axis ``phi_deg = (theta_deg + 90) % 180`` has unit tangent
      vector ``(dy, dx) = (sin(phi), cos(phi))`` running ALONG the fibers.
    """
    img = np.asarray(image, dtype=np.float64)
    vm = np.asarray(valid_mask, dtype=bool)
    h, w = img.shape
    mean_v = float(img[vm].mean()) if vm.any() else 0.0
    centered = np.where(vm, img - mean_v, 0.0)

    win = np.outer(np.hanning(h), np.hanning(w))
    spec = np.fft.fftshift(np.fft.fft2(centered * win))
    power = (spec.real ** 2 + spec.imag ** 2)

    fy = np.fft.fftshift(np.fft.fftfreq(h, d=float(voxel_um)))
    fx = np.fft.fftshift(np.fft.fftfreq(w, d=float(voxel_um)))
    fyy, fxx = np.meshgrid(fy, fx, indexing="ij")
    f_mag = np.hypot(fyy, fxx)

    f_lo = 1.0 / float(max_wavelength_um)
    f_hi = 1.0 / float(min_wavelength_um)
    annulus = (f_mag >= f_lo) & (f_mag <= f_hi)

    theta_deg = np.degrees(np.arctan2(fyy, fxx)) % 180.0
    bins = np.floor(theta_deg[annulus]).astype(np.int64) % 180
    hist = np.bincount(bins, weights=power[annulus], minlength=180).astype(np.float64)
    hist_smooth = ndi.gaussian_filter1d(hist, sigma=2.0, mode="wrap")

    theta1 = int(np.argmax(hist_smooth))
    target2 = (theta1 + 90) % 180
    tol = int(round(ortho_tolerance_deg))
    cand_angles = [(target2 + d) % 180 for d in range(-tol, tol + 1)]
    theta2 = max(cand_angles, key=lambda a: float(hist_smooth[a]))

    phi1 = float((theta1 + 90) % 180)
    phi2 = float((theta2 + 90) % 180)
    med_e = float(np.median(hist_smooth))
    aniso_ratio = float(0.5 * (hist_smooth[theta1] + hist_smooth[theta2]) / max(med_e, 1e-12))

    return {
        "freq_angle_1_deg": float(theta1),
        "freq_angle_2_deg": float(theta2),
        "fiber_axis_1_deg": phi1,
        "fiber_axis_2_deg": phi2,
        "anisotropy_ratio": round(aniso_ratio, 4),
        "wavelength_band_um": [float(min_wavelength_um), float(max_wavelength_um)],
    }


def _periodic_smooth_decompose(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Moisan (2011) periodic + smooth (p + s) decomposition to eliminate FFT border ringing."""
    u = np.asarray(img, dtype=np.float64)
    h, w = u.shape
    v = np.zeros_like(u)
    v[0, :] += u[-1, :] - u[0, :]
    v[-1, :] += u[0, :] - u[-1, :]
    v[:, 0] += u[:, -1] - u[:, 0]
    v[:, -1] += u[:, 0] - u[:, -1]

    v_hat = np.fft.fft2(v)
    ky = np.arange(h, dtype=np.float64)[:, None]
    kx = np.arange(w, dtype=np.float64)[None, :]
    denom = 2.0 * np.cos(2.0 * np.pi * ky / h) + 2.0 * np.cos(2.0 * np.pi * kx / w) - 4.0
    denom[0, 0] = 1.0
    s_hat = v_hat / denom
    s_hat[0, 0] = 0.0
    smooth = np.fft.ifft2(s_hat).real
    periodic = u - smooth
    return periodic, smooth


def directional_fft_notch(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    freq_angle_1_deg: float,
    freq_angle_2_deg: float,
    angular_sigma_deg: float = 12.0,
    min_wavelength_um: float = 20.0,
    max_wavelength_um: float = 250.0,
) -> tuple[np.ndarray, float]:
    """Attenuate oriented fiber-weave wedges in 2D FFT (using Moisan p+s decomposition)."""
    img = np.asarray(image, dtype=np.float64)
    vm = np.asarray(valid_mask, dtype=bool)
    h, w = img.shape
    mean_v = float(img[vm].mean()) if vm.any() else 0.0
    filled = _fill_nearest_2d(img, vm) - mean_v
    periodic, smooth = _periodic_smooth_decompose(filled)

    fft_img = np.fft.fftshift(np.fft.fft2(periodic))
    fy = np.fft.fftshift(np.fft.fftfreq(h, d=float(voxel_um)))
    fx = np.fft.fftshift(np.fft.fftfreq(w, d=float(voxel_um)))
    fyy, fxx = np.meshgrid(fy, fx, indexing="ij")
    f_mag = np.hypot(fyy, fxx)
    theta = np.degrees(np.arctan2(fyy, fxx)) % 180.0

    # Smooth radial bandpass gate (never touches DC).
    f_lo = 1.0 / float(max_wavelength_um)
    f_hi = 1.0 / float(min_wavelength_um)
    radial = np.exp(-0.5 * (np.maximum(0.0, f_lo - f_mag) / max(0.35 * f_lo, 1e-6)) ** 2) * np.exp(
        -0.5 * (np.maximum(0.0, f_mag - f_hi) / max(0.35 * f_hi, 1e-6)) ** 2
    )
    radial[f_mag < 1e-9] = 0.0

    def _ang_gauss(target_deg: float) -> np.ndarray:
        diff = np.abs(theta - float(target_deg))
        diff = np.minimum(diff, 180.0 - diff)
        return np.exp(-0.5 * (diff / float(angular_sigma_deg)) ** 2)

    notch = radial * np.maximum(_ang_gauss(freq_angle_1_deg), _ang_gauss(freq_angle_2_deg))
    keep = np.clip(1.0 - notch, 0.0, 1.0)

    p_before = float(np.sum((fft_img.real ** 2 + fft_img.imag ** 2) * (radial > 0.5)))
    fft_filt = fft_img * keep
    p_after = float(np.sum((fft_filt.real ** 2 + fft_filt.imag ** 2) * (radial > 0.5)))
    removed_frac = float(max(0.0, (p_before - p_after) / max(p_before, 1e-12)))

    resid = np.fft.ifft2(np.fft.ifftshift(fft_filt)).real + smooth + mean_v
    return resid.astype(np.float32), removed_frac


def structn2v_fiber_subtract(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    fiber_axis_1_deg: float,
    fiber_axis_2_deg: float,
    tap_distances_um: Sequence[float] = (12.0, 20.0, 32.0, 48.0),
    ridge_alpha: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Unsupervised StructN2V-style directional blind-mask linear predictor along fiber axes.

    Blinds the center neighborhood (``< min(tap_distances_um)``) and predicts each
    pixel solely from symmetric context taps along the two estimated fiber axes,
    removing correlated vascular-bundle structure while preserving localized
    ink micro-crackle in the residual. Out-of-bounds taps fall back to their
    in-bounds symmetric along-axis partner so borders do not flip fiber angles.
    """
    img = np.asarray(image, dtype=np.float64)
    vm = np.asarray(valid_mask, dtype=bool)
    h, w = img.shape
    filled = _fill_nearest_2d(img, vm)

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    taps: list[np.ndarray] = []
    max_d_px = 0.0
    for phi_deg in (float(fiber_axis_1_deg), float(fiber_axis_2_deg)):
        rad = math.radians(phi_deg)
        dy = math.sin(rad)
        dx = math.cos(rad)
        for d_um in tap_distances_um:
            d_px = float(d_um) / float(voxel_um)
            max_d_px = max(max_d_px, d_px)
            y_pos = yy + d_px * dy
            x_pos = xx + d_px * dx
            y_neg = yy - d_px * dy
            x_neg = xx - d_px * dx
            in_pos = (y_pos >= 0.0) & (y_pos <= h - 1.0) & (x_pos >= 0.0) & (x_pos <= w - 1.0)
            in_neg = (y_neg >= 0.0) & (y_neg <= h - 1.0) & (x_neg >= 0.0) & (x_neg <= w - 1.0)
            s_pos = ndi.map_coordinates(filled, [np.clip(y_pos, 0.0, h - 1.0), np.clip(x_pos, 0.0, w - 1.0)], order=1, mode="nearest")
            s_neg = ndi.map_coordinates(filled, [np.clip(y_neg, 0.0, h - 1.0), np.clip(x_neg, 0.0, w - 1.0)], order=1, mode="nearest")
            taps.append(np.where(in_pos, s_pos, np.where(in_neg, s_neg, s_pos)))
            taps.append(np.where(in_neg, s_neg, np.where(in_pos, s_pos, s_neg)))

    n_feats = len(taps)
    X_all = np.empty((h * w, n_feats + 1), dtype=np.float64)
    X_all[:, 0] = 1.0
    for j, t in enumerate(taps, start=1):
        X_all[:, j] = t.ravel()

    # Fit ridge weights strictly on interior valid pixels (away from border fallback).
    margin = int(math.ceil(max_d_px))
    interior = vm.copy()
    if h > 2 * margin + 4 and w > 2 * margin + 4:
        interior[:margin, :] = False
        interior[-margin:, :] = False
        interior[:, :margin] = False
        interior[:, -margin:] = False

    y_all = filled.ravel()
    fit_mask = interior.ravel() if int(interior.sum()) >= n_feats + 16 else vm.ravel()
    if int(fit_mask.sum()) < n_feats + 4:
        return np.zeros((h, w), dtype=np.float32), 0.0

    X_fit = X_all[fit_mask]
    y_fit = y_all[fit_mask]
    mu = X_fit[:, 1:].mean(axis=0)
    sd = np.maximum(X_fit[:, 1:].std(axis=0), 1e-6)
    Xn = np.empty_like(X_fit)
    Xn[:, 0] = 1.0
    Xn[:, 1:] = (X_fit[:, 1:] - mu) / sd

    gram = Xn.T @ Xn
    reg = np.eye(n_feats + 1, dtype=np.float64) * (float(ridge_alpha) * float(Xn.shape[0]))
    reg[0, 0] = 0.0
    beta = np.linalg.solve(gram + reg, Xn.T @ y_fit)

    X_all_n = np.empty_like(X_all)
    X_all_n[:, 0] = 1.0
    X_all_n[:, 1:] = (X_all[:, 1:] - mu) / sd
    y_hat = (X_all_n @ beta).reshape(h, w)
    resid = (filled - y_hat).astype(np.float32)

    y_var = float(np.var(y_fit))
    r_var = float(np.var(resid.ravel()[fit_mask]))
    r2 = float(max(0.0, 1.0 - r_var / max(y_var, 1e-12)))
    return resid, r2


def compute_crackle_energy(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    sigma_lo_um: float = 3.5,
    sigma_hi_um: float = 12.0,
    env_sigma_um: float = 25.0,
) -> np.ndarray:
    """Compute local crackle bandpass RMS energy on ``image``."""
    vm = np.asarray(valid_mask, dtype=bool)
    filled = _fill_nearest_2d(np.asarray(image, dtype=np.float64), vm)
    g_lo = ndi.gaussian_filter(filled, sigma=float(sigma_lo_um) / float(voxel_um), mode="reflect")
    g_hi = ndi.gaussian_filter(filled, sigma=float(sigma_hi_um) / float(voxel_um), mode="reflect")
    hp = g_lo - g_hi
    energy = _normalized_gaussian(hp * hp, vm, sigma=float(env_sigma_um) / float(voxel_um))
    return np.sqrt(np.maximum(0.0, energy)).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Exact Mann-Whitney ROC-AUC, Cohen's d & 16-shift rolled-label null
# ---------------------------------------------------------------------------

def exact_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Exact tie-aware Mann-Whitney U ROC-AUC in pure NumPy/SciPy."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(s, method="average")
    u_stat = float(ranks[y].sum()) - 0.5 * n_pos * (n_pos + 1)
    return float(u_stat / (float(n_pos) * float(n_neg)))


def cohens_d(pos_vals: np.ndarray, neg_vals: np.ndarray) -> float:
    """Signed Cohen's d = (mean_pos - mean_neg) / pooled_std."""
    p = np.asarray(pos_vals, dtype=np.float64)
    n = np.asarray(neg_vals, dtype=np.float64)
    if p.size < 2 or n.size < 2:
        return 0.0
    vp = float(np.var(p, ddof=1))
    vn = float(np.var(n, ddof=1))
    pooled = math.sqrt(max(0.0, ((p.size - 1) * vp + (n.size - 1) * vn) / float(p.size + n.size - 2)))
    if pooled < 1e-12:
        return 0.0
    return float((float(p.mean()) - float(n.mean())) / pooled)


def build_16_rolled_shifts(shape_hw: tuple[int, int]) -> list[tuple[int, int]]:
    """Deterministic 4x4 grid of 16 strictly non-zero 2D toroidal shifts."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if h < 5 or w < 5:
        raise ValueError(f"Grid too small for 16 distinct shifts: {shape_hw}")
    shifts: list[tuple[int, int]] = []
    for fy in ROLLED_SHIFT_FRACTIONS:
        dy = max(1, min(h - 1, int(round(fy * h))))
        for fx in ROLLED_SHIFT_FRACTIONS:
            dx = max(1, min(w - 1, int(round(fx * w))))
            shifts.append((dy, dx))
    return shifts


def evaluate_feature_with_rolled_null(
    feature_map: np.ndarray,
    ink_mask: np.ndarray,
    blank_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    gate_min_auc: float = GATE_MIN_ROC_AUC,
    gate_min_above_null: float = GATE_MIN_ABOVE_NULL,
) -> dict[str, Any]:
    """Evaluate feature ROC-AUC, Cohen's d, and conservative two-sided 16-shift rolled-label null."""
    feat = np.asarray(feature_map, dtype=np.float64)
    vm = np.asarray(valid_mask, dtype=bool) & np.isfinite(feat)
    ink = np.asarray(ink_mask, dtype=bool) & vm
    blank = np.asarray(blank_mask, dtype=bool) & vm & (~ink)

    n_ink = int(ink.sum())
    n_blank = int(blank.sum())
    if n_ink < 10 or n_blank < 10:
        return {
            "n_ink": n_ink,
            "n_blank": n_blank,
            "raw_auc": 0.5,
            "polarity": "+",
            "roc_auc": 0.5,
            "cohens_d": 0.0,
            "rolled_null_mean": 0.5,
            "rolled_null_std": 0.0,
            "rolled_null_max": 0.5,
            "rolled_null_signed_mean": 0.5,
            "rolled_null_signed_std": 0.0,
            "auc_above_null": 0.0,
            "passes_gate": False,
        }

    eval_m = ink | blank
    raw_auc = exact_roc_auc(feat[eval_m], ink[eval_m])
    polarity = "+" if raw_auc >= 0.5 else "-"
    oriented_auc = max(raw_auc, 1.0 - raw_auc)
    d_val = cohens_d(feat[ink], feat[blank])

    # 16-shift toroidal rolled-label null on the unmasked ink/blank labels intersected with valid_mask.
    ink_orig = np.asarray(ink_mask, dtype=bool)
    blank_orig = np.asarray(blank_mask, dtype=bool) & (~ink_orig)
    shifts = build_16_rolled_shifts(feat.shape)
    null_raw: list[float] = []
    null_oriented: list[float] = []
    for dy, dx in shifts:
        ri = np.roll(ink_orig, shift=(dy, dx), axis=(0, 1)) & vm
        rb = np.roll(blank_orig, shift=(dy, dx), axis=(0, 1)) & vm & (~ri)
        if int(ri.sum()) < 10 or int(rb.sum()) < 10:
            null_raw.append(0.5)
            null_oriented.append(0.5)
            continue
        em = ri | rb
        auc_s = exact_roc_auc(feat[em], ri[em])
        null_raw.append(auc_s)
        null_oriented.append(max(auc_s, 1.0 - auc_s))

    null_o_arr = np.asarray(null_oriented, dtype=np.float64)
    null_r_arr = np.asarray(null_raw, dtype=np.float64)
    null_mean = float(null_o_arr.mean())
    auc_above = float(oriented_auc - null_mean)
    passes = bool((oriented_auc >= float(gate_min_auc)) and (auc_above >= float(gate_min_above_null)))

    return {
        "n_ink": n_ink,
        "n_blank": n_blank,
        "raw_auc": round(raw_auc, 5),
        "polarity": polarity,
        "roc_auc": round(oriented_auc, 5),
        "cohens_d": round(d_val, 5),
        "rolled_null_mean": round(null_mean, 5),
        "rolled_null_std": round(float(null_o_arr.std()), 5),
        "rolled_null_max": round(float(null_o_arr.max()), 5),
        "rolled_null_signed_mean": round(float(null_r_arr.mean()), 5),
        "rolled_null_signed_std": round(float(null_r_arr.std()), 5),
        "auc_above_null": round(auc_above, 5),
        "passes_gate": passes,
    }


# ---------------------------------------------------------------------------
# 6. Part B Physics: 9.362 um Paganin (pag0 / pag50 / pag1000) features
# ---------------------------------------------------------------------------

def compute_part_b_features(
    slab_u8: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float = VOXEL_UM_PART_B,
    window_f32: tuple[float, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute raw CT, high-frequency phase-edge/scattering variance, and surface-height relief."""
    vol = np.asarray(slab_u8, dtype=np.float32)
    if window_f32 is not None and window_f32[1] > window_f32[0]:
        lo, hi = float(window_f32[0]), float(window_f32[1])
        vol_phys = lo + (vol / 255.0) * (hi - lo)
    else:
        vol_phys = vol / 255.0

    # Extract sub-voxel sheet peak on the raw uint8 slab (strictly positive on papyrus).
    hf = extract_subvoxel_height_field(
        vol,
        voxel_um=voxel_um,
        valid_mask=valid_mask,
        search_half_slices=min(6, (vol.shape[0] - 1) // 2),
        z_smooth_sigma=0.8,
        detrend_sigma_um=200.0,
        band_half_width=0,
    )
    sv = hf["sheet_valid"]
    h_um = hf["h_um"]
    n_peak = hf["n_peak"]

    # Raw CT contrast at sheet peak (in physical attenuation units when window_f32 available).
    h, w = sv.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z_q = np.clip(n_peak, 0.0, float(vol_phys.shape[0] - 1))
    z0 = np.floor(z_q).astype(np.int64)
    z1 = np.minimum(z0 + 1, vol_phys.shape[0] - 1)
    wz = (z_q - z0).astype(np.float32)
    raw_ct_peak = (1.0 - wz) * vol_phys[z0, yy, xx] + wz * vol_phys[z1, yy, xx]

    # Also center-slice CT (raw mesh center without peak re-centering) as diagnostic baseline.
    c_idx = (vol_phys.shape[0] - 1) // 2
    raw_ct_center = vol_phys[c_idx].astype(np.float32)

    # Local high-frequency phase-edge / scattering variance on surface CT (in-plane DoG + Laplacian energy).
    filled_ct = _fill_nearest_2d(raw_ct_peak.astype(np.float64), sv)
    hp_dog = filled_ct - ndi.gaussian_filter(filled_ct, sigma=1.5, mode="reflect")
    hf_phase_var = _normalized_gaussian(hp_dog * hp_dog, sv, sigma=2.5)
    lap = ndi.laplace(filled_ct, mode="reflect")
    hf_lap_energy = np.sqrt(np.maximum(0.0, _normalized_gaussian(lap * lap, sv, sigma=2.5)))

    # Through-thickness normal-gradient scattering variance around n_peak (+/- 2 slices).
    grad_z = np.gradient(vol_phys.astype(np.float64), axis=0)
    z_stack_grads = []
    for off in (-2, -1, 0, 1, 2):
        zq = np.clip(np.rint(n_peak + off).astype(np.int64), 0, vol_phys.shape[0] - 1)
        z_stack_grads.append(grad_z[zq, yy, xx])
    normal_scatter_var = np.var(np.stack(z_stack_grads, axis=0), axis=0)

    # Surface-height relief features: detrended height h_um, local relief roughness, and 60 um MSII.
    m1 = _normalized_gaussian(h_um, sv, sigma=2.5)
    m2 = _normalized_gaussian(h_um.astype(np.float64) ** 2, sv, sigma=2.5)
    relief_rough = np.sqrt(np.maximum(0.0, m2 - m1 * m1))
    msii_60 = 0.5 - compute_msii_spherical_occupancy(h_um, sv, voxel_um=voxel_um, radius_um=60.0)

    features = {
        "raw_ct_peak": raw_ct_peak.astype(np.float32),
        "raw_ct_center": raw_ct_center.astype(np.float32),
        "hf_phase_edge_variance": hf_phase_var.astype(np.float32),
        "hf_laplacian_energy": hf_lap_energy.astype(np.float32),
        "normal_scattering_variance": normal_scatter_var.astype(np.float32),
        "surface_height_relief_um": h_um.astype(np.float32),
        "surface_relief_roughness": relief_rough.astype(np.float32),
        "surface_msii_convexity_60um": msii_60.astype(np.float32),
    }
    diagnostics = {
        "z_center_slice": int(hf["z_center_slice"]),
        "mesh_center_slice": int(c_idx),
        "sheet_valid_fraction": round(float(sv.mean()), 4),
        "raw_ct_peak_std": round(float(np.std(raw_ct_peak[sv])) if sv.any() else 0.0, 6),
        "hf_phase_edge_var_mean": round(float(np.mean(hf_phase_var[sv])) if sv.any() else 0.0, 8),
        "height_relief_std_um": round(float(np.std(h_um[sv])) if sv.any() else 0.0, 4),
    }
    return features, diagnostics


# ---------------------------------------------------------------------------
# 7. Full live / offline end-to-end pipeline
# ---------------------------------------------------------------------------

def run_pherc0139_ablation(
    *,
    urls: dict[str, str] | None = None,
    part_a_budget_mib: float = 60.0,
    part_b_budget_mib: float = 60.0,
    global_budget_mib: float = 120.0,
    roi_h: int = 512,
    roi_w: int = 640,
    roi_bounds_override: tuple[int, int, int, int] | None = None,
    part_b_num_slices: int = 21,
    stride_sv: int = 4,
) -> dict[str, Any]:
    """Run Exp T3 Part A (2.399 um) and Part B (9.362 um pag0/pag50/pag1000) cleanly in RAM."""
    t0 = time.monotonic()
    cfg = dict(DEFAULT_URLS)
    if urls:
        cfg.update(urls)

    budget = HttpByteBudget(
        part_a_limit_bytes=int(round(part_a_budget_mib * MIB)),
        part_b_limit_bytes=int(round(part_b_budget_mib * MIB)),
        global_limit_bytes=int(round(global_budget_mib * MIB)),
    )

    with ExitStack() as stack:
        # Step 1: Inspect .zarray metadata FIRST (tiny metadata fetches).
        sv_reader = stack.enter_context(
            BudgetedRemoteZarrV2(
                cfg["surface_volume_2p4um"],
                level="0",
                budget=budget,
                part="part_a",
                label="sv_2p4um",
                cache_mb=64.0,
                workers=8,
            )
        )
        part_b_readers: dict[str, BudgetedRemoteZarrV2] = {}
        part_b_meta: dict[str, dict[str, Any]] = {}
        part_b_meshes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}

        for variant in ("pag0", "pag50", "pag1000"):
            part_b_readers[variant] = stack.enter_context(
                BudgetedRemoteZarrV2(
                    cfg[f"{variant}_vol"],
                    level="0",
                    budget=budget,
                    part="part_b",
                    label=f"vol_{variant}",
                    cache_mb=32.0,
                    workers=8,
                )
            )
            part_b_meta[variant] = load_paganin_volume_metadata(
                cfg[f"{variant}_vol"], budget=budget, part="part_b", label=f"meta_{variant}"
            )
            part_b_meshes[variant] = load_tifxyz_in_ram(
                cfg[f"{variant}_mesh"], budget=budget, part="part_b", label=f"mesh_{variant}"
            )

        # Load ink reference map strictly in RAM (HEAD-checked against Part A cap first).
        ink_map = load_ink_map_in_ram(cfg["ink_reference_2p4um"], budget=budget, part="part_a")
        sv_hw = (int(sv_reader.shape[1]), int(sv_reader.shape[2]))
        if ink_map.shape != sv_hw:
            raise ValueError(
                f"Ink reference shape {ink_map.shape} does not match surface volume HW {sv_hw}"
            )

        # Step 2: Select or validate spatially co-registered ROI within BOTH Part A and Part B caps.
        if roi_bounds_override is not None:
            r0, r1, c0, c1 = [int(v) for v in roi_bounds_override]
            patch_ink = ink_map[r0:r1, c0:c1]
            roi_info: dict[str, Any] = {
                "roi_bounds_2p4um": [r0, r1, c0, c1],
                "roi_shape_2p4um": [r1 - r0, c1 - c0],
                "roi_physical_mm": [
                    round((r1 - r0) * VOXEL_UM_PART_A / 1000.0, 4),
                    round((c1 - c0) * VOXEL_UM_PART_A / 1000.0, 4),
                ],
                "ink_fraction_high": round(float((patch_ink >= INK_HIGH_U8_THRESHOLD).mean()), 4),
                "blank_fraction_low": round(float(((patch_ink > 0) & (patch_ink <= BLANK_LOW_U8_THRESHOLD)).mean()), 4),
                "selection_mode": "override",
            }
        else:
            rem_b_for_chunks = max(0, budget.remaining("part_b") - 2 * MIB)
            roi_info = select_coregistered_roi(
                ink_map,
                sv_reader.chunks,
                part_b_meshes,
                part_b_readers,
                roi_h=roi_h,
                roi_w=roi_w,
                part_b_max_planned_bytes=rem_b_for_chunks,
                part_b_num_slices=part_b_num_slices,
                stride_sv=stride_sv,
            )
            roi_info["selection_mode"] = "deterministic_ink_blank_and_budget"

        r0, r1, c0, c1 = [int(v) for v in roi_info["roi_bounds_2p4um"]]

        # Pre-flight Part A chunk budget check BEFORE reading surface volume ROI.
        n_sv_chunks, sv_planned_bytes = sv_reader.plan_box_chunks(
            (0, r0, c0), (sv_reader.shape[0], r1, c1)
        )
        budget.check_planned("part_a", sv_planned_bytes, label="sv_2p4um_roi")
        roi_info["planned_part_a_chunks"] = int(n_sv_chunks)
        roi_info["planned_part_a_chunk_mib"] = round(sv_planned_bytes / MIB, 4)

        # ------------------------------------------------------------------
        # PART A: 2.399 um Surface-Topography MSII + Directional Fiber Sub
        # ------------------------------------------------------------------
        sv_slab, _, sv_support = sv_reader.read_box(
            (0, r0, c0), (sv_reader.shape[0], r1, c1), return_support=True
        )
        sv_valid_yx = sv_support.all(axis=0) & (sv_slab.max(axis=0) > 0)

        ink_roi = ink_map[r0:r1, c0:c1]
        ink_mask_a = ink_roi >= INK_HIGH_U8_THRESHOLD
        blank_mask_a = (ink_roi > 0) & (ink_roi <= BLANK_LOW_U8_THRESHOLD)

        hf_a = extract_subvoxel_height_field(
            sv_slab,
            voxel_um=VOXEL_UM_PART_A,
            valid_mask=sv_valid_yx,
            search_half_slices=min(24, (sv_slab.shape[0] - 1) // 2),
            z_smooth_sigma=1.0,
            detrend_sigma_um=200.0,
            band_half_width=1,
        )
        sheet_valid_a = hf_a["sheet_valid"]
        surface_ct_a = hf_a["surface_ct"]

        # MSII (20, 60, 140 um) + curvature + relief roughness
        topo_feats = compute_msii_and_curvature(
            hf_a["h_um"],
            sheet_valid_a,
            voxel_um=VOXEL_UM_PART_A,
            radii_um=MSII_RADII_UM,
        )

        # Directional Fiber-Weave Subtraction (2D FFT + directional notch + StructN2V)
        fiber_axes = estimate_fiber_axes_fft(
            surface_ct_a, sheet_valid_a, voxel_um=VOXEL_UM_PART_A
        )
        fft_resid, fft_power_removed = directional_fft_notch(
            surface_ct_a,
            sheet_valid_a,
            voxel_um=VOXEL_UM_PART_A,
            freq_angle_1_deg=fiber_axes["freq_angle_1_deg"],
            freq_angle_2_deg=fiber_axes["freq_angle_2_deg"],
        )
        s2v_resid, s2v_r2 = structn2v_fiber_subtract(
            surface_ct_a,
            sheet_valid_a,
            voxel_um=VOXEL_UM_PART_A,
            fiber_axis_1_deg=fiber_axes["fiber_axis_1_deg"],
            fiber_axis_2_deg=fiber_axes["fiber_axis_2_deg"],
        )
        combined_resid, combined_r2 = structn2v_fiber_subtract(
            fft_resid,
            sheet_valid_a,
            voxel_um=VOXEL_UM_PART_A,
            fiber_axis_1_deg=fiber_axes["fiber_axis_1_deg"],
            fiber_axis_2_deg=fiber_axes["fiber_axis_2_deg"],
        )

        raw_crackle = compute_crackle_energy(surface_ct_a, sheet_valid_a, voxel_um=VOXEL_UM_PART_A)
        fft_crackle = compute_crackle_energy(fft_resid, sheet_valid_a, voxel_um=VOXEL_UM_PART_A)
        s2v_crackle = compute_crackle_energy(s2v_resid, sheet_valid_a, voxel_um=VOXEL_UM_PART_A)
        comb_crackle = compute_crackle_energy(combined_resid, sheet_valid_a, voxel_um=VOXEL_UM_PART_A)
        s2v_signed_smooth = _normalized_gaussian(s2v_resid, sheet_valid_a, sigma=15.0 / VOXEL_UM_PART_A).astype(np.float32)

        part_a_feature_maps: dict[str, np.ndarray] = {
            "raw_ct_surface": surface_ct_a,
            "raw_crackle_energy": raw_crackle,
            "fft_notch_crackle_energy": fft_crackle,
            "structn2v_crackle_energy": s2v_crackle,
            "combined_fiber_sub_crackle_energy": comb_crackle,
            "structn2v_residual_smoothed": s2v_signed_smooth,
            **topo_feats,
        }

        # Exclude outer border guard band (<= 32 px when ROI is large enough) so
        # FFT/StructN2V/MSII boundary padding never contaminates Part A metrics.
        eval_valid_a = sheet_valid_a.copy()
        guard_a = min(32, min(eval_valid_a.shape) // 8)
        if guard_a > 0:
            eval_valid_a[:guard_a, :] = False
            eval_valid_a[-guard_a:, :] = False
            eval_valid_a[:, :guard_a] = False
            eval_valid_a[:, -guard_a:] = False

        part_a_results: dict[str, dict[str, Any]] = {}
        for fname, fmap in part_a_feature_maps.items():
            part_a_results[fname] = evaluate_feature_with_rolled_null(
                fmap, ink_mask_a, blank_mask_a, eval_valid_a
            )

        # Free Part A 3D slab immediately to keep RAM low.
        del sv_slab, sv_support

        # ------------------------------------------------------------------
        # PART B: 9.362 um Paganin Ablation (pag0 vs pag50 vs pag1000)
        # ------------------------------------------------------------------
        # Co-registered block-averaged ink reference on the exact Part B grid.
        h_b = (r1 - r0) // stride_sv
        w_b = (c1 - c0) // stride_sv
        ink_roi_trim = ink_roi[: h_b * stride_sv, : w_b * stride_sv]
        ink_b_mean = ink_roi_trim.reshape(h_b, stride_sv, w_b, stride_sv).mean(axis=(1, 3))
        ink_mask_b = ink_b_mean >= float(INK_HIGH_U8_THRESHOLD)
        blank_mask_b = (ink_b_mean > 0.0) & (ink_b_mean <= float(BLANK_LOW_U8_THRESHOLD))

        # Pre-flight Part B rays & combined byte budget check across ALL 3 volumes BEFORE fetching.
        part_b_rays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        total_planned_b = 0
        planned_b_breakdown: dict[str, dict[str, Any]] = {}
        for variant in ("pag0", "pag50", "pag1000"):
            mx, my, mz, mval, _ = part_b_meshes[variant]
            rays, m_use = build_part_b_rays_for_roi(
                mx, my, mz, mval,
                sv_hw=sv_hw,
                roi_bounds=(r0, r1, c0, c1),
                stride_sv=stride_sv,
                num_slices=part_b_num_slices,
                slice_step=1.0,
            )
            n_ch, b_bytes = part_b_readers[variant].plan_trilinear_chunks(
                rays[:, m_use, :], max_direct_box_voxels=1
            )
            part_b_rays[variant] = (rays, m_use)
            total_planned_b += int(b_bytes)
            planned_b_breakdown[variant] = {
                "planned_chunks": int(n_ch),
                "planned_mib": round(b_bytes / MIB, 4),
                "mesh_usable_fraction": round(float(m_use.mean()), 4),
            }

        budget.check_planned("part_b", total_planned_b, label="part_b_all_variants")

        part_b_results: dict[str, Any] = {}
        for variant in ("pag0", "pag50", "pag1000"):
            rays, m_use = part_b_rays[variant]
            reader = part_b_readers[variant]
            slab_b = np.zeros((part_b_num_slices, h_b, w_b), dtype=np.float32)
            src_avail_b = np.zeros((part_b_num_slices, h_b, w_b), dtype=bool)
            if m_use.any():
                vals, supp = reader.sample_trilinear(
                    rays[:, m_use, :], max_direct_box_voxels=1, return_support=True
                )
                slab_b[:, m_use] = vals
                src_avail_b[:, m_use] = supp

            full_source_yx = m_use & src_avail_b.all(axis=0)
            nonzero_yx = slab_b.max(axis=0) > 0.0
            valid_b = full_source_yx & nonzero_yx

            win = part_b_meta[variant].get("conversion_32bit_window") or part_b_meta[variant].get(
                "target_window_f32"
            )
            win_tuple = (float(win[0]), float(win[1])) if win is not None else None
            b_feats, b_diag = compute_part_b_features(
                slab_b, valid_b, voxel_um=VOXEL_UM_PART_B, window_f32=win_tuple
            )
            eval_table: dict[str, dict[str, Any]] = {}
            for fname, fmap in b_feats.items():
                eval_table[fname] = evaluate_feature_with_rolled_null(
                    fmap, ink_mask_b, blank_mask_b, valid_b
                )

            part_b_results[variant] = {
                "volume_shape": list(reader.shape),
                "volume_chunks": list(reader.chunks),
                "mesh_shape": list(part_b_meshes[variant][3].shape),
                "mesh_scale": part_b_meshes[variant][4].get("scale"),
                "paganin_metadata": part_b_meta[variant],
                "source_support": {
                    "mesh_usable_fraction": round(float(m_use.mean()), 4),
                    "full_source_available_fraction": round(float(full_source_yx.mean()), 4),
                    "nonzero_intensity_fraction": round(float(nonzero_yx.mean()), 4),
                    "valid_mask_fraction": round(float(valid_b.mean()), 4),
                    "chunks_fetched": int(reader.chunks_fetched),
                    "bytes_fetched": int(reader.bytes_fetched),
                },
                "diagnostics": b_diag,
                "features": eval_table,
            }

    # Summarize Part A & Part B verdicts honestly against the predeclared gate.
    best_a_name = max(part_a_results, key=lambda k: (part_a_results[k]["roc_auc"], part_a_results[k]["auc_above_null"]))
    best_a = part_a_results[best_a_name]

    part_b_comparison: dict[str, dict[str, Any]] = {}
    key_b_features = (
        "raw_ct_peak",
        "raw_ct_center",
        "hf_phase_edge_variance",
        "hf_laplacian_energy",
        "normal_scattering_variance",
        "surface_height_relief_um",
        "surface_relief_roughness",
        "surface_msii_convexity_60um",
    )
    best_b_tuple: tuple[float, float, str, str] = (0.0, -1.0, "", "")
    for fname in key_b_features:
        row: dict[str, Any] = {}
        for variant in ("pag0", "pag50", "pag1000"):
            m = part_b_results[variant]["features"][fname]
            row[variant] = {
                "roc_auc": m["roc_auc"],
                "raw_auc": m["raw_auc"],
                "cohens_d": m["cohens_d"],
                "rolled_null_mean": m["rolled_null_mean"],
                "auc_above_null": m["auc_above_null"],
                "passes_gate": m["passes_gate"],
            }
            cand = (m["roc_auc"], m["auc_above_null"], variant, fname)
            if cand[:2] > best_b_tuple[:2]:
                best_b_tuple = cand
        row["delta_pag50_minus_pag1000_auc"] = round(
            row["pag50"]["roc_auc"] - row["pag1000"]["roc_auc"], 5
        )
        row["delta_pag0_minus_pag1000_auc"] = round(
            row["pag0"]["roc_auc"] - row["pag1000"]["roc_auc"], 5
        )
        part_b_comparison[fname] = row

    part_a_baseline_names = {"raw_ct_surface", "raw_crackle_energy"}
    part_a_transverse_passing = [
        k for k, v in part_a_results.items() if v["passes_gate"] and k not in part_a_baseline_names
    ]
    part_a_baseline_passing = [
        k for k, v in part_a_results.items() if v["passes_gate"] and k in part_a_baseline_names
    ]
    part_b_low_paganin_passing = [
        f"{var}:{fn}"
        for var in ("pag0", "pag50")
        for fn in key_b_features
        if part_b_results[var]["features"][fn]["passes_gate"]
    ]
    part_b_pag1000_passing = [
        f"pag1000:{fn}"
        for fn in key_b_features
        if part_b_results["pag1000"]["features"][fn]["passes_gate"]
    ]

    part_a_transverse_pass = len(part_a_transverse_passing) > 0
    part_b_low_paganin_pass = len(part_b_low_paganin_passing) > 0
    overall_verdict = "PASS" if (part_a_transverse_pass or part_b_low_paganin_pass) else "CLOSED NO-GO"

    receipt: dict[str, Any] = {
        "experiment_id": "exp_t3_pherc0139_transverse_physics_ablation",
        "scroll": "PHerc0139",
        "scroll_status": "published_read_control_not_eligible_first_letters",
        "segment": "20260422000000-title_2026042222_zmid_flatboi",
        "compute_mode": "CPU_ONLY",
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "sources": {k: _sanitize_source_label(v) for k, v in cfg.items()},
        "predeclared_gate": {
            "min_roc_auc": GATE_MIN_ROC_AUC,
            "min_auc_above_rolled_null": GATE_MIN_ABOVE_NULL,
            "rolled_null_shifts": 16,
            "ink_high_u8_threshold": INK_HIGH_U8_THRESHOLD,
            "blank_low_u8_threshold": BLANK_LOW_U8_THRESHOLD,
            "null_statistic": "two_sided_max(auc, 1-auc)_across_16_toroidal_shifts",
        },
        "http_byte_budget": budget.summary(),
        "shapes": {
            "surface_volume_2p4um": list(sv_reader.shape),
            "surface_volume_chunks": list(sv_reader.chunks),
            "ink_reference_2p4um": list(ink_map.shape),
            "part_b_grid_hw": [int(h_b), int(w_b)],
            "part_b_stride_sv": int(stride_sv),
            "part_b_num_slices": int(part_b_num_slices),
        },
        "roi": roi_info,
        "part_a_2p4um": {
            "voxel_um": VOXEL_UM_PART_A,
            "z_center_slice": int(hf_a["z_center_slice"]),
            "z_search_window": hf_a["z_search_window"],
            "sheet_valid_fraction": round(float(sheet_valid_a.mean()), 4),
            "height_field_std_um": round(
                float(np.std(hf_a["h_um"][sheet_valid_a])) if sheet_valid_a.any() else 0.0, 4
            ),
            "msii_radii_um": list(MSII_RADII_UM),
            "fiber_axes_fft": fiber_axes,
            "directional_subtraction_diagnostics": {
                "fft_notch_bandpass_power_removed": round(fft_power_removed, 4),
                "structn2v_fiber_r2_raw": round(s2v_r2, 4),
                "structn2v_fiber_r2_after_fft_notch": round(combined_r2, 4),
                "crackle_auc_delta_fft_minus_raw": round(
                    part_a_results["fft_notch_crackle_energy"]["roc_auc"]
                    - part_a_results["raw_crackle_energy"]["roc_auc"],
                    5,
                ),
                "crackle_auc_delta_structn2v_minus_raw": round(
                    part_a_results["structn2v_crackle_energy"]["roc_auc"]
                    - part_a_results["raw_crackle_energy"]["roc_auc"],
                    5,
                ),
                "crackle_auc_delta_combined_minus_raw": round(
                    part_a_results["combined_fiber_sub_crackle_energy"]["roc_auc"]
                    - part_a_results["raw_crackle_energy"]["roc_auc"],
                    5,
                ),
            },
            "features": part_a_results,
            "best_feature": {"name": best_a_name, **best_a},
            "part_a_transverse_verdict": "PASS" if part_a_transverse_pass else "CLOSED NO-GO",
        },
        "part_b_9p362um_paganin_ablation": {
            "voxel_um": VOXEL_UM_PART_B,
            "coordinate_mapping": "center_aligned_canvas_to_mesh_uv",
            "planned_chunk_preflight": planned_b_breakdown,
            "variants": part_b_results,
            "comparison_table": part_b_comparison,
            "best_feature": {
                "variant": best_b_tuple[2],
                "feature": best_b_tuple[3],
                "roc_auc": round(best_b_tuple[0], 5),
                "auc_above_null": round(best_b_tuple[1], 5),
            },
            "part_b_low_paganin_verdict": "PASS" if part_b_low_paganin_pass else "CLOSED NO-GO",
        },
        "verdict": {
            "overall": overall_verdict,
            "part_a_transverse": "PASS" if part_a_transverse_pass else "CLOSED NO-GO",
            "part_b_low_paganin": "PASS" if part_b_low_paganin_pass else "CLOSED NO-GO",
            "part_a_transverse_passing_features": part_a_transverse_passing,
            "part_a_baseline_passing_features": part_a_baseline_passing,
            "part_b_low_paganin_passing_features": part_b_low_paganin_passing,
            "part_b_standard_pag1000_passing_features": part_b_pag1000_passing,
        },
    }

    assert_no_local_paths(receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exp T3: PHerc. 0139 2.399 um MSII + Fiber Subtraction + 9.362 um Paganin Ablation (CPU-only)."
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(__file__).resolve().parent / "evidence" / "pherc0139_transverse_physics_ablation.json",
        help="Destination path for sanitized JSON receipt.",
    )
    parser.add_argument("--part-a-budget-mib", type=float, default=60.0)
    parser.add_argument("--part-b-budget-mib", type=float, default=60.0)
    parser.add_argument("--global-budget-mib", type=float, default=120.0)
    parser.add_argument("--roi-h", type=int, default=512)
    parser.add_argument("--roi-w", type=int, default=640)
    args = parser.parse_args(argv)

    receipt = run_pherc0139_ablation(
        part_a_budget_mib=args.part_a_budget_mib,
        part_b_budget_mib=args.part_b_budget_mib,
        global_budget_mib=args.global_budget_mib,
        roi_h=args.roi_h,
        roi_w=args.roi_w,
    )
    assert_no_local_paths(receipt)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    b = receipt["http_byte_budget"]
    print(
        f"[Exp T3] verdict={receipt['verdict']['overall']} "
        f"(Part A transverse={receipt['verdict']['part_a_transverse']}, "
        f"Part B low-Paganin={receipt['verdict']['part_b_low_paganin']}) | "
        f"HTTP bytes: PartA={b['part_a_mib']:.2f} MiB, PartB={b['part_b_mib']:.2f} MiB, "
        f"Total={b['global_mib']:.2f} / {b['global_limit_mib']:.2f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
