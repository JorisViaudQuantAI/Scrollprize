# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""CPU-only, low-memory (< 180 MB) streaming dual-energy CT preflight and strip scanner for PHerc. Paris 4.

Implements the different-ink (high-Z mineral/metallic) dual-energy CT protocol
documented in ``gemini_lane/evidence/paris4_title_recon.md``:
- Streaming OME-Zarr v2 reads via ``RemoteZarrV2`` (strict 404-only empty-chunk
  handling, strictly bounded in-memory LRU cache, zero persistent CT downloads).
- HTTP Range-based ``tifxyz`` mesh patch reading (zero full-mesh downloads).
- Exact ``window_f32`` linear attenuation conversion from volume ``metadata.json``
  for 78 keV (20260411134726), 137 keV (20260323153942), and 54 keV DLS (20230205180739).
- S3 affine transform composition (78 <-> DLS <-> 137 keV) followed by mandatory
  local 3D band-pass NCC translation lock (du, dv, dn) per sub-tile, explicitly
  gated against 6 displaced null offsets (+/-20 um) before computing residuals.
- Linear papyrus background fit strictly on Control B (confirmed blank papyrus),
  dual-energy residuals R_{78/137} and R_{54/78}, surface-vs-bulk depth top-hat
  (|n| <= 8 um minus |n| in [16, 36] um), and wrap-behind ghosting exclusion
  (90-260 um).
- Strict privacy boundary: public JSON receipts contain ZERO local filesystem
  paths; private agraphon maps/plots are refused inside any git repository.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any, Sequence
import urllib.parse
import urllib.request

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None

requests = _requests

try:
    from .stream_snap_render import (
        RemoteZarrV2,
        _read_tiff_2d,
        _sanitize_source_label,
        compute_grid_normals,
        layer_coherence_score,
        validate_tifxyz_channels,
    )
except ImportError:  # pragma: no cover - script execution
    from stream_snap_render import (
        RemoteZarrV2,
        _read_tiff_2d,
        _sanitize_source_label,
        compute_grid_normals,
        layer_coherence_score,
        validate_tifxyz_channels,
    )


# ---------------------------------------------------------------------------
# Public PHerc. Paris 4 S3 endpoints & radiometric defaults
# ---------------------------------------------------------------------------

S3_BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
DLS_BASE = "https://data.aws.ash2txt.org/samples/PHercParis4"

DEFAULT_URLS: dict[str, str] = {
    "vol_78": f"{S3_BASE}/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr",
    "vol_137": f"{S3_BASE}/PHercParis4/volumes/20260323153942-2.400um-0.2m-137keV-masked.zarr",
    "vol_54": f"{DLS_BASE}/volumes/20230205180739-7.910um-54keV-masked.zarr",
    "vol_1129": f"{S3_BASE}/PHercParis4/volumes/20260608103018-1.129um-0.2m-78keV-masked.zarr",
    "transform_78_to_dls": (
        f"{S3_BASE}/PHercParis4/transforms/20260411134726-to-20230205180739-r1/fit/transform.json"
    ),
    "transform_137_to_dls": (
        f"{S3_BASE}/PHercParis4/transforms/20260323153942-to-20230205180739-r1/fit/transform.json"
    ),
    "transform_1129_to_78_r2": (
        f"{S3_BASE}/PHercParis4/transforms/20260608103018-to-20260411134726-r2/fit/transform.json"
    ),
    "mesh_78": (
        f"{S3_BASE}/PHercParis4/segments/20260701183124-w010-027/mesh/"
        "20260701183124-on-20260411134726-2.4um.tifxyz"
    ),
    "mesh_outer_dls": (
        f"{S3_BASE}/PHercParis4/segments/20260926044226-spiral-output-generation-19/mesh/"
        "20260926044226-on-20230205180739-7.91um.tifxyz"
    ),
}

DEFAULT_VOXEL_UM: dict[str, float] = {
    "78keV": 2.400,
    "137keV": 2.400,
    "54keV": 7.910,
    "1129_78keV": 1.129,
}

# Verified from S3 metadata.json (zarr_export.target_window_f32_{min,max}).
DEFAULT_WINDOW_F32: dict[str, tuple[float, float]] = {
    "78keV": (-0.04, 0.22),
    "137keV": (0.01, 0.16),
    "54keV": (-0.03, 0.145),
    "1129_78keV": (-0.03, 0.145),
}

FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z])[A-Za-z]:\\"),
    re.compile(r"(?<![A-Za-z])[A-Za-z]:/"),
    re.compile(r"/Users/[^/]+/", re.I),
    re.compile(r"\\Users\\[^\\]+\\", re.I),
    re.compile(r"/home/[^/]+/"),
    re.compile(r"file://", re.I),
)


# ---------------------------------------------------------------------------
# 1. Privacy / sanitization helpers & radiometry (window_f32)
# ---------------------------------------------------------------------------

def assert_no_local_paths(payload: Any) -> None:
    """Raise ValueError if serialized JSON payload contains any local filesystem path."""
    text = json.dumps(payload, sort_keys=True)
    for pat in FORBIDDEN_PATH_PATTERNS:
        m = pat.search(text)
        if m is not None:
            raise ValueError(f"Sanitized JSON contains forbidden local path pattern: {m.group(0)!r}")


def _assert_outside_git_repo(out_dir: Path) -> None:
    """Refuse writing private candidate maps/plots inside a git repository worktree."""
    resolved = out_dir.resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / ".git").exists():
            raise ValueError(
                "Privacy boundary violation: private output directory must lie outside any git repository."
            )


def _http_get_bytes(
    url: str,
    *,
    session: Any | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> bytes:
    """Fetch raw bytes from HTTP(S) URL using requests when available, else urllib."""
    if requests is not None:
        sess = session if session is not None else requests.Session()
        resp = sess.get(url, headers=headers or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_volume_window_f32(
    zarr_root: str | os.PathLike[str],
    *,
    fallback: tuple[float, float] | None = None,
    session: Any | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """Read exact ``window_f32`` linear attenuation bounds from a volume's metadata JSON.

    Checks ``metadata.json`` and ``meta.json`` under ``zarr_root`` (HTTP(S) or local).
    Extracts ``zarr_export.target_window_f32_{min,max}`` or top-level ``window_f32``.
    """
    root_str = str(zarr_root).rstrip("/\\")
    is_http = root_str.lower().startswith(("http://", "https://"))
    meta: dict[str, Any] = {}

    for fname in ("metadata.json", "meta.json"):
        try:
            if is_http:
                raw = _http_get_bytes(f"{root_str}/{fname}", session=session, timeout=20.0)
                meta = json.loads(raw.decode("utf-8"))
                break
            path = Path(root_str) / fname
            if path.is_file():
                meta = json.loads(path.read_text(encoding="utf-8"))
                break
        except Exception:
            continue

    ze = meta.get("zarr_export") if isinstance(meta, dict) else None
    if isinstance(ze, dict) and "target_window_f32_min" in ze and "target_window_f32_max" in ze:
        lo = float(ze["target_window_f32_min"])
        hi = float(ze["target_window_f32_max"])
        if hi <= lo:
            raise ValueError(f"Invalid target_window_f32 bounds: {(lo, hi)}")
        return lo, hi, meta

    if isinstance(meta, dict) and "window_f32" in meta:
        w = meta["window_f32"]
        if isinstance(w, (list, tuple)) and len(w) == 2:
            lo, hi = float(w[0]), float(w[1])
            if hi <= lo:
                raise ValueError(f"Invalid window_f32 bounds: {(lo, hi)}")
            return lo, hi, meta

    if fallback is not None:
        return float(fallback[0]), float(fallback[1]), meta
    raise FileNotFoundError(f"Could not locate window_f32 metadata under {_sanitize_source_label(root_str)}")


def uint8_to_linear_attenuation(
    u8: np.ndarray,
    f32_min: float,
    f32_max: float,
) -> np.ndarray:
    """Convert quantized uint8 (or float in [0, 255]) CT values to linear attenuation mu (f32).

    Formula: ``mu = f32_min + (u8 / 255.0) * (f32_max - f32_min)``.
    """
    if f32_max <= f32_min:
        raise ValueError(f"f32_max ({f32_max}) must exceed f32_min ({f32_min})")
    arr = np.asarray(u8, dtype=np.float32)
    scale = np.float32((float(f32_max) - float(f32_min)) / 255.0)
    return np.float32(f32_min) + arr * scale


# ---------------------------------------------------------------------------
# 2. S3 affine transform loading & composition (78 <-> DLS <-> 137 keV)
# ---------------------------------------------------------------------------

def load_affine_transform(
    source: str | os.PathLike[str],
    *,
    session: Any | None = None,
) -> np.ndarray:
    """Load a 3x4 or 4x4 affine ``transformation_matrix`` JSON into a 4x4 float64 matrix."""
    src_str = str(source)
    if src_str.lower().startswith(("http://", "https://")):
        data = json.loads(_http_get_bytes(src_str, session=session, timeout=20.0).decode("utf-8"))
    else:
        data = json.loads(Path(src_str).read_text(encoding="utf-8"))
    mat = np.asarray(data["transformation_matrix"], dtype=np.float64)
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = mat
        return out
    if mat.shape == (4, 4):
        return mat
    raise ValueError(f"Expected 3x4 or 4x4 transformation_matrix, got shape {mat.shape}")


def compose_affine_via_dls(
    m_src_to_dls: np.ndarray,
    m_dst_to_dls: np.ndarray,
) -> np.ndarray:
    """Compose ``M_{src->dst} = inv(M_{dst->DLS}) @ M_{src->DLS}`` in 4x4 homogeneous form."""
    m_s = np.asarray(m_src_to_dls, dtype=np.float64)
    m_d = np.asarray(m_dst_to_dls, dtype=np.float64)
    if m_s.shape != (4, 4) or m_d.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrices, got {m_s.shape} and {m_d.shape}")
    return np.linalg.inv(m_d) @ m_s


def apply_affine_points(pts_xyz: np.ndarray, matrix_4x4: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine matrix to ``(..., 3)`` xyz coordinates."""
    pts = np.asarray(pts_xyz, dtype=np.float64)
    if pts.shape[-1] != 3:
        raise ValueError(f"Expected last dim 3, got shape {pts.shape}")
    mat = np.asarray(matrix_4x4, dtype=np.float64)
    return pts @ mat[:3, :3].T + mat[:3, 3]


def apply_affine_vectors(vecs_xyz: np.ndarray, matrix_4x4: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Map 3D direction vectors through the linear part ``M[:3, :3]`` of an affine transform."""
    v = np.asarray(vecs_xyz, dtype=np.float64)
    mat = np.asarray(matrix_4x4, dtype=np.float64)
    mapped = v @ mat[:3, :3].T
    if normalize:
        nrm = np.linalg.norm(mapped, axis=-1, keepdims=True)
        good = nrm[..., 0] > 1e-12
        mapped = np.where(good[..., None], mapped / np.where(good[..., None], nrm, 1.0), 0.0)
    return mapped


def load_transform_bundle(
    source: str | os.PathLike[str],
    *,
    session: Any | None = None,
) -> dict[str, Any]:
    """Load an S3/local transform.json including its 4x4 affine matrix and optional TPS deformation."""
    src_str = str(source)
    if src_str.lower().startswith(("http://", "https://")):
        data = json.loads(_http_get_bytes(src_str, session=session, timeout=20.0).decode("utf-8"))
    else:
        data = json.loads(Path(src_str).read_text(encoding="utf-8"))
    mat = np.asarray(data["transformation_matrix"], dtype=np.float64)
    if mat.shape == (3, 4):
        aff = np.eye(4, dtype=np.float64)
        aff[:3, :4] = mat
    elif mat.shape == (4, 4):
        aff = mat
    else:
        raise ValueError(f"Expected 3x4 or 4x4 transformation_matrix, got shape {mat.shape}")

    deform = data.get("deformation")
    inv_deform = deform.get("inverse") if isinstance(deform, dict) else None
    if isinstance(inv_deform, dict) and "kernel" not in inv_deform and isinstance(deform, dict):
        inv_deform = dict(inv_deform)
        inv_deform["kernel"] = deform.get("kernel", "-r")
    return {
        "matrix": aff,
        "deformation": deform if isinstance(deform, dict) else None,
        "inverse_deformation": inv_deform if isinstance(inv_deform, dict) else None,
        "schema_version": data.get("schema_version"),
    }


def eval_tps_displacement(
    pts_xyz: np.ndarray,
    tps_dict: dict[str, Any],
    *,
    batch_size: int = 2048,
) -> np.ndarray:
    """Evaluate a 3D thin-plate-spline displacement field ``K(x_n, c_n) @ W + [1, x_n] @ A``."""
    pts = np.asarray(pts_xyz, dtype=np.float64)
    if pts.shape[-1] != 3:
        raise ValueError(f"Expected last dim 3, got shape {pts.shape}")
    orig_shape = pts.shape
    flat = pts.reshape(-1, 3)
    if flat.shape[0] == 0:
        return np.zeros(orig_shape, dtype=np.float64)

    kernel = str(tps_dict.get("kernel", "-r"))
    if kernel != "-r":
        raise NotImplementedError(f"Unsupported TPS kernel: {kernel!r}")

    centre = np.asarray(tps_dict["centre"], dtype=np.float64)
    scale = float(tps_dict["scale"])
    if scale <= 0.0:
        raise ValueError(f"TPS scale must be positive, got {scale}")
    cp = np.asarray(tps_dict["control_points"], dtype=np.float64)
    weights = np.asarray(tps_dict["weights"], dtype=np.float64)
    affine = np.asarray(tps_dict["affine"], dtype=np.float64)

    cp_norm = (cp - centre) / scale
    out = np.empty_like(flat)
    step = max(1, int(batch_size))
    for i0 in range(0, flat.shape[0], step):
        i1 = min(i0 + step, flat.shape[0])
        xn = (flat[i0:i1] - centre) / scale
        diff = xn[:, None, :] - cp_norm[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        aug = np.concatenate([np.ones((i1 - i0, 1), dtype=np.float64), xn], axis=1)
        out[i0:i1] = (-r) @ weights + aug @ affine
    return out.reshape(orig_shape)


def map_points_affine_and_tps(
    pts_xyz: np.ndarray,
    matrix_4x4: np.ndarray,
    *,
    tps_on_input: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply 4x4 affine transform plus optional TPS displacement evaluated on input points."""
    mapped = apply_affine_points(pts_xyz, matrix_4x4)
    if tps_on_input is not None:
        mapped = mapped + eval_tps_displacement(pts_xyz, tps_on_input)
    return mapped


# ---------------------------------------------------------------------------
# 3. Streaming Range-based tifxyz mesh patch reader (local or remote HTTP)
# ---------------------------------------------------------------------------

class TifxyzPatchReader:
    """Read sub-windows ``[v0:v1, u0:u1]`` from local or remote uncompressed float32 tifxyz.

    For remote HTTP(S) uncompressed single-strip IEEE float32 TIFFs (standard
    Vesuvius ``tifxyz`` format on S3), reads ONLY the requested byte ranges via
    HTTP ``Range:`` requests, keeping mesh I/O to a few kilobytes per tile.
    """

    def __init__(self, root: str | os.PathLike[str], *, timeout: float = 30.0) -> None:
        self.root = str(root).rstrip("/\\")
        self.local = not self.root.lower().startswith(("http://", "https://"))
        self.timeout = float(timeout)
        self._session = requests.Session() if (not self.local and requests is not None) else None
        self.bytes_fetched: int = 0
        self.requests_made: int = 0

        self.meta: dict[str, Any] = {}
        self.height: int = 0
        self.width: int = 0
        self._strip_offsets: dict[str, int] = {}
        self._dtype_str: str = "<f4"
        self._local_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

        if self.local:
            root_p = Path(self.root)
            mp = root_p / "meta.json"
            if mp.is_file():
                self.meta = json.loads(mp.read_text(encoding="utf-8"))
            x = _read_tiff_2d(root_p / "x.tif").astype(np.float32)
            y = _read_tiff_2d(root_p / "y.tif").astype(np.float32)
            z = _read_tiff_2d(root_p / "z.tif").astype(np.float32)
            cx, cy, cz, valid = validate_tifxyz_channels(x, y, z)
            self._local_cache = (cx, cy, cz, valid)
            self.height, self.width = int(cx.shape[0]), int(cx.shape[1])
        else:
            raw_meta = self._fetch_range(f"{self.root}/meta.json", None)
            self.meta = json.loads(raw_meta.decode("utf-8"))
            for ch in ("x", "y", "z"):
                w, h, strip_off, dtype_str = self._inspect_remote_tiff(f"{self.root}/{ch}.tif")
                if self.width == 0:
                    self.width, self.height, self._dtype_str = w, h, dtype_str
                elif (w, h) != (self.width, self.height):
                    raise ValueError(f"Channel {ch}.tif shape {(h, w)} != {(self.height, self.width)}")
                self._strip_offsets[ch] = strip_off

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> TifxyzPatchReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _fetch_range(self, url: str, byte_range: tuple[int, int] | None) -> bytes:
        headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range is not None else {}
        data = _http_get_bytes(url, session=self._session, headers=headers, timeout=self.timeout)
        self.bytes_fetched += len(data)
        self.requests_made += 1
        return data

    def _inspect_remote_tiff(self, url: str) -> tuple[int, int, int, str]:
        hdr = self._fetch_range(url, (0, 255))
        if len(hdr) < 8:
            raise OSError(f"Short TIFF header for {url}")
        endian = "<" if hdr[:2] == b"II" else ">"
        magic = struct.unpack(endian + "H", hdr[2:4])[0]
        if magic != 42:
            raise NotImplementedError(f"Only classic TIFF (magic=42) is supported for Range reads, got {magic}")
        ifd_off = struct.unpack(endian + "I", hdr[4:8])[0]
        if ifd_off + 256 <= len(hdr):
            ifd = hdr[ifd_off:]
        else:
            ifd = self._fetch_range(url, (ifd_off, ifd_off + 2047))
        n_ent = struct.unpack(endian + "H", ifd[:2])[0]
        tags: dict[int, tuple[int, int, int]] = {}
        for i in range(n_ent):
            o = 2 + i * 12
            tag, typ, cnt, val = struct.unpack(endian + "HHII", ifd[o : o + 12])
            if typ == 3 and cnt == 1:
                val = struct.unpack(endian + "H", ifd[o + 8 : o + 10])[0]
            tags[tag] = (typ, cnt, val)
        width = tags[256][2]
        height = tags[257][2]
        bps = tags[258][2]
        comp = tags.get(259, (3, 1, 1))[2]
        strip_cnt = tags[273][1]
        strip_off = tags[273][2]
        sample_fmt = tags.get(339, (3, 1, 1))[2]
        if comp != 1 or bps != 32 or sample_fmt != 3 or strip_cnt != 1:
            raise NotImplementedError(
                f"Remote Range TIFF requires uncompressed single-strip float32; "
                f"got comp={comp}, bps={bps}, fmt={sample_fmt}, strips={strip_cnt}"
            )
        return int(width), int(height), int(strip_off), f"{endian}f4"

    def read_patch(
        self,
        v0: int,
        v1: int,
        u0: int,
        u1: int,
        *,
        v_stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read ``(x, y, z, valid)`` over ``v in range(v0, v1, v_stride)`` and ``u in [u0, u1)``."""
        v0_c = max(0, int(v0))
        v1_c = min(self.height, int(v1))
        u0_c = max(0, int(u0))
        u1_c = min(self.width, int(u1))
        step = max(1, int(v_stride))
        if v1_c <= v0_c or u1_c <= u0_c:
            empty = np.empty((0, 0), dtype=np.float32)
            return empty, empty, empty, np.empty((0, 0), dtype=bool)

        if self.local:
            assert self._local_cache is not None
            cx, cy, cz, valid = self._local_cache
            return (
                cx[v0_c:v1_c:step, u0_c:u1_c].copy(),
                cy[v0_c:v1_c:step, u0_c:u1_c].copy(),
                cz[v0_c:v1_c:step, u0_c:u1_c].copy(),
                valid[v0_c:v1_c:step, u0_c:u1_c].copy(),
            )

        rows = list(range(v0_c, v1_c, step))
        cols = u1_c - u0_c
        row_bytes = cols * 4
        out: dict[str, np.ndarray] = {}
        for ch in ("x", "y", "z"):
            base_off = self._strip_offsets[ch]
            arr = np.empty((len(rows), cols), dtype=np.float32)
            # If reading nearly the full width contiguously with step==1, one range read is faster.
            if step == 1 and cols >= int(0.85 * self.width):
                start = base_off + v0_c * self.width * 4
                end = base_off + v1_c * self.width * 4 - 1
                raw = self._fetch_range(f"{self.root}/{ch}.tif", (start, end))
                block = np.frombuffer(raw, dtype=self._dtype_str).reshape(v1_c - v0_c, self.width)
                arr[:] = block[:, u0_c:u1_c]
            else:
                for i, v in enumerate(rows):
                    start = base_off + (v * self.width + u0_c) * 4
                    raw = self._fetch_range(f"{self.root}/{ch}.tif", (start, start + row_bytes - 1))
                    arr[i] = np.frombuffer(raw, dtype=self._dtype_str)
            out[ch] = arr
        return validate_tifxyz_channels(out["x"], out["y"], out["z"])


# ---------------------------------------------------------------------------
# 4. Surface tangent/normal frame & aligned 3D slab sampling
# ---------------------------------------------------------------------------

def _fill_nearest_2d(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.all() or not valid.any():
        return arr
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return arr[idx[0], idx[1]]


def build_surface_frame(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    *,
    grid_spacing_u_um: float,
    grid_spacing_v_um: float,
    target_spacing_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a mesh patch to isotropic physical UV spacing and compute orthonormal (e_u, e_v, e_n).

    Returns ``(pts_l0, e_u, e_v, e_n, uv_valid)`` with spatial shape ``(H_out, W_out)``.
    """
    cx, cy, cz, val = validate_tifxyz_channels(x, y, z)
    val = val & np.asarray(valid, dtype=bool)
    h_g, w_g = val.shape
    if h_g < 2 or w_g < 2 or not val.any():
        empty3 = np.zeros((0, 0, 3), dtype=np.float64)
        return empty3, empty3, empty3, empty3, np.zeros((0, 0), dtype=bool)

    step_v = float(target_spacing_um) / float(grid_spacing_v_um)
    step_u = float(target_spacing_um) / float(grid_spacing_u_um)
    out_h = max(2, int(round((h_g - 1) / step_v)) + 1)
    out_w = max(2, int(round((w_g - 1) / step_u)) + 1)

    pts_grid = np.stack([cx, cy, cz], axis=-1).astype(np.float64)
    pts_filled = _fill_nearest_2d(pts_grid, val)
    vm_float = val.astype(np.float64)

    gv_1d = np.linspace(0.0, h_g - 1.0, out_h, dtype=np.float64)
    gu_1d = np.linspace(0.0, w_g - 1.0, out_w, dtype=np.float64)
    gv, gu = np.meshgrid(gv_1d, gu_1d, indexing="ij")

    uv_valid = ndi.map_coordinates(vm_float, [gv, gu], order=1, mode="constant", cval=0.0) > 0.999
    pts = np.empty((out_h, out_w, 3), dtype=np.float64)
    for c in range(3):
        pts[..., c] = ndi.map_coordinates(pts_filled[..., c], [gv, gu], order=1, mode="nearest")

    # Compute local tangent vectors dP/du, dP/dv on the resampled surface, then orthonormalize.
    dp_du = np.gradient(pts, axis=1)
    dp_dv = np.gradient(pts, axis=0)
    n_vec = np.cross(dp_du, dp_dv)
    n_len = np.linalg.norm(n_vec, axis=-1, keepdims=True)
    u_len = np.linalg.norm(dp_du, axis=-1, keepdims=True)

    good = uv_valid & (n_len[..., 0] > 1e-9) & (u_len[..., 0] > 1e-9)
    e_n = np.where(good[..., None], n_vec / np.where(good[..., None], n_len, 1.0), 0.0)
    e_u = np.where(good[..., None], dp_du / np.where(good[..., None], u_len, 1.0), 0.0)
    # Gram-Schmidt orthogonalize e_v = normalize(cross(e_n, e_u)) to preserve right-handedness.
    e_v_raw = np.cross(e_n, e_u)
    v_len = np.linalg.norm(e_v_raw, axis=-1, keepdims=True)
    good &= v_len[..., 0] > 1e-9
    e_v = np.where(good[..., None], e_v_raw / np.where(good[..., None], v_len, 1.0), 0.0)
    # Re-align sign of e_v with dp_dv if flipped.
    dot_v = np.sum(e_v * dp_dv, axis=-1)
    flip = dot_v < 0.0
    e_v = np.where(flip[..., None], -e_v, e_v)
    return pts, e_u, e_v, e_n, good


def sample_trilinear_exact_minimal(
    reader: RemoteZarrV2,
    xyz: np.ndarray,
) -> np.ndarray:
    """Exact trilinear sampling fetching ONLY the minimal set of chunks touched by the 8 interpolation corners.

    Unlike box-padded sampling (which can pull up to 27 surrounding chunks per
    chunk cell for curved surfaces), this gathers the exact 8 corners
    ``(floor(q) + {0, 1})`` directly from ``reader.get_chunks(...)``, cutting
    network I/O by 3-4x on curved papyrus sheets while matching ``map_coordinates(order=1)``.
    """
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.shape[-1] != 3:
        raise ValueError(f"Expected last dim 3 for xyz, got shape {pts.shape}")
    orig_shape = pts.shape[:-1]
    flat = pts.reshape(-1, 3)
    out = np.full(flat.shape[0], float(reader.fill_value), dtype=np.float32)
    if flat.shape[0] == 0:
        return out.reshape(orig_shape)

    sz, sy, sx = reader.shape
    cz_sz, cy_sz, cx_sz = reader.chunks
    in_bounds = (
        np.isfinite(flat).all(axis=-1)
        & (flat[:, 0] >= 0.0)
        & (flat[:, 0] <= sx - 1.0)
        & (flat[:, 1] >= 0.0)
        & (flat[:, 1] <= sy - 1.0)
        & (flat[:, 2] >= 0.0)
        & (flat[:, 2] <= sz - 1.0)
    )
    if not in_bounds.any():
        return out.reshape(orig_shape)

    valid_idx = np.flatnonzero(in_bounds)
    q_zyx = flat[valid_idx][:, ::-1]  # (M, 3) in z, y, x
    i0 = np.floor(q_zyx).astype(np.int64)
    i1 = np.minimum(i0 + 1, np.array([sz - 1, sy - 1, sx - 1], dtype=np.int64))
    f = (q_zyx - i0).astype(np.float32)  # (M, 3)

    # Identify exact unique chunk indices needed across all 8 corners.
    needed: set[tuple[int, int, int]] = set()
    z_pair = (i0[:, 0], i1[:, 0])
    y_pair = (i0[:, 1], i1[:, 1])
    x_pair = (i0[:, 2], i1[:, 2])
    c_strides = (
        (sy + cy_sz - 1) // cy_sz * ((sx + cx_sz - 1) // cx_sz),
        (sx + cx_sz - 1) // cx_sz,
    )
    ny_c = (sy + cy_sz - 1) // cy_sz
    nx_c = (sx + cx_sz - 1) // cx_sz
    for za in z_pair:
        cza = za // cz_sz
        for ya in y_pair:
            cya = ya // cy_sz
            for xa in x_pair:
                cxa = xa // cx_sz
                packed = cza * c_strides[0] + cya * c_strides[1] + cxa
                for code in np.unique(packed):
                    ci = int(code)
                    cz_i = ci // (ny_c * nx_c)
                    rem = ci - cz_i * (ny_c * nx_c)
                    cy_i = rem // nx_c
                    cx_i = rem - cy_i * nx_c
                    needed.add((cz_i, cy_i, cx_i))

    chunks = reader.get_chunks(needed)
    fill_f32 = np.float32(reader.fill_value)

    def _gather_corner(za: np.ndarray, ya: np.ndarray, xa: np.ndarray) -> np.ndarray:
        vals = np.full(za.shape[0], fill_f32, dtype=np.float32)
        cza = za // cz_sz
        cya = ya // cy_sz
        cxa = xa // cx_sz
        lza = za - cza * cz_sz
        lya = ya - cya * cy_sz
        lxa = xa - cxa * cx_sz
        packed = cza * c_strides[0] + cya * c_strides[1] + cxa
        order = np.argsort(packed, kind="mergesort")
        sp = packed[order]
        change = np.flatnonzero(np.concatenate(([True], sp[1:] != sp[:-1], [True])))
        for s_i, e_i in zip(change[:-1], change[1:]):
            grp = order[s_i:e_i]
            ci = int(sp[s_i])
            cz_i = ci // (ny_c * nx_c)
            rem = ci - cz_i * (ny_c * nx_c)
            cy_i = rem // nx_c
            cx_i = rem - cy_i * nx_c
            blk = chunks.get((cz_i, cy_i, cx_i))
            if blk is not None:
                vals[grp] = blk[lza[grp], lya[grp], lxa[grp]].astype(np.float32, copy=False)
        return vals

    fz, fy, fx = f[:, 0], f[:, 1], f[:, 2]
    wz = (1.0 - fz, fz)
    wy = (1.0 - fy, fy)
    wx = (1.0 - fx, fx)
    acc = np.zeros(valid_idx.shape[0], dtype=np.float32)
    for bz in (0, 1):
        for by in (0, 1):
            wzy = wz[bz] * wy[by]
            for bx in (0, 1):
                cval = _gather_corner(z_pair[bz], y_pair[by], x_pair[bx])
                acc += (wzy * wx[bx]) * cval

    out[valid_idx] = acc
    return out.reshape(orig_shape)


def sample_surface_slab(
    reader: RemoteZarrV2,
    pts_l0: np.ndarray,
    e_n: np.ndarray,
    uv_valid: np.ndarray,
    n_offsets_um: Sequence[float],
    *,
    native_voxel_um: float,
    level: int,
    f32_min: float,
    f32_max: float,
    shift_xyz_l0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a 3D slab ``[N_n, H, W]`` along ``e_n`` from a multiscale OME-Zarr v2 volume.

    Returns ``(mu_f32, u8_raw, valid_3d)`` where ``valid_3d`` requires both mesh
    validity and non-zero (non-fill) scanner support inside volume bounds.
    """
    n_arr = np.asarray(n_offsets_um, dtype=np.float64)
    h, w = uv_valid.shape
    num_n = len(n_arr)
    mu_out = np.zeros((num_n, h, w), dtype=np.float32)
    u8_out = np.zeros((num_n, h, w), dtype=np.float32)
    valid_3d = np.zeros((num_n, h, w), dtype=bool)
    if not uv_valid.any() or num_n == 0:
        return mu_out, u8_out, valid_3d

    scale = float(2 ** int(level))
    offs_l0 = n_arr / float(native_voxel_um)  # (N_n,)
    p_base = pts_l0[uv_valid]  # (M, 3)
    if shift_xyz_l0 is not None:
        p_base = p_base + np.asarray(shift_xyz_l0, dtype=np.float64)
    n_base = e_n[uv_valid]  # (M, 3)

    rays_l0 = p_base[None, :, :] + offs_l0[:, None, None] * n_base[None, :, :]  # (N_n, M, 3)
    # Center-aligned OME-Zarr multiscale coordinate mapping.
    rays_lvl = (rays_l0 + 0.5) / scale - 0.5

    sz, sy, sx = reader.shape
    in_bounds = (
        np.isfinite(rays_lvl).all(axis=-1)
        & (rays_lvl[..., 0] >= 0.0)
        & (rays_lvl[..., 0] <= sx - 1.0)
        & (rays_lvl[..., 1] >= 0.0)
        & (rays_lvl[..., 1] <= sy - 1.0)
        & (rays_lvl[..., 2] >= 0.0)
        & (rays_lvl[..., 2] <= sz - 1.0)
    )
    sampled = sample_trilinear_exact_minimal(reader, rays_lvl)  # (N_n, M) float32
    nonfill = in_bounds & (sampled > 0.5)
    mu_vals = np.where(nonfill, uint8_to_linear_attenuation(sampled, f32_min, f32_max), np.float32(0.0))

    u8_out[:, uv_valid] = np.where(nonfill, sampled, np.float32(0.0))
    mu_out[:, uv_valid] = mu_vals
    valid_3d[:, uv_valid] = nonfill
    return mu_out, u8_out, valid_3d


# ---------------------------------------------------------------------------
# 5. Mandatory 3D band-pass NCC translation lock + 6 displaced null gate
# ---------------------------------------------------------------------------

def bandpass_3d(
    vol: np.ndarray,
    valid_mask: np.ndarray,
    *,
    sigma_low_vx: float = 0.75,
    sigma_high_vx: float = 3.5,
) -> np.ndarray:
    """Mask-normalized 3D Difference-of-Gaussians band-pass filter.

    Uses normalized convolution ``G(vol*mask) / G(mask)`` so zero-fill edges
    never introduce artificial step-edge responses.
    """
    v = np.asarray(vol, dtype=np.float64)
    m = np.asarray(valid_mask, dtype=np.float64)
    if not np.any(m > 0.5):
        return np.zeros_like(v, dtype=np.float64)

    vm = np.where(m > 0.5, v, 0.0)

    def _norm_gauss(sig: float) -> np.ndarray:
        if sig <= 0.0:
            return vm.copy()
        num = ndi.gaussian_filter(vm, sigma=sig, mode="constant", cval=0.0)
        den = ndi.gaussian_filter(m, sigma=sig, mode="constant", cval=0.0)
        return np.where(den > 1e-6, num / np.where(den > 1e-6, den, 1.0), 0.0)

    lo = _norm_gauss(float(sigma_low_vx))
    hi = _norm_gauss(float(sigma_high_vx))
    bp = np.where(m > 0.5, lo - hi, 0.0)
    return bp


def _masked_pearson_r(a: np.ndarray, b: np.ndarray, mask: np.ndarray, *, min_voxels: int = 32) -> float:
    n = int(mask.sum())
    if n < min_voxels:
        return -1.0
    av = a[mask]
    bv = b[mask]
    ac = av - av.mean()
    bc = bv - bv.mean()
    da = float(np.dot(ac, ac))
    db = float(np.dot(bc, bc))
    if da <= 1e-18 or db <= 1e-18:
        return 0.0
    return float(np.dot(ac, bc) / math.sqrt(da * db))


def _eval_shifted_ncc_exact(
    ref_bp: np.ndarray,
    ref_mask: np.ndarray,
    mov_bp: np.ndarray,
    mov_mask: np.ndarray,
    shift_nvu: tuple[float, float, float],
    *,
    min_voxels: int = 32,
) -> float:
    """Evaluate exact masked Pearson NCC between ``ref_bp`` and ``mov_bp`` shifted by ``(dn, dv, du)``.

    Convention: ``mov_shifted[n, v, u] = mov[n + dn, v + dv, u + du]``.
    """
    dn, dv, du = shift_nvu
    # Build coordinate grid in mov_bp corresponding to ref_bp + (dn, dv, du).
    zn, zv, zu = np.meshgrid(
        np.arange(ref_bp.shape[0], dtype=np.float64) + float(dn),
        np.arange(ref_bp.shape[1], dtype=np.float64) + float(dv),
        np.arange(ref_bp.shape[2], dtype=np.float64) + float(du),
        indexing="ij",
    )
    in_b = (
        (zn >= 0.0)
        & (zn <= mov_bp.shape[0] - 1.0)
        & (zv >= 0.0)
        & (zv <= mov_bp.shape[1] - 1.0)
        & (zu >= 0.0)
        & (zu <= mov_bp.shape[2] - 1.0)
    )
    if not in_b.any():
        return -1.0
    mov_s = ndi.map_coordinates(mov_bp, [zn, zv, zu], order=1, mode="constant", cval=0.0)
    mov_m = ndi.map_coordinates(mov_mask.astype(np.float64), [zn, zv, zu], order=1, mode="constant", cval=0.0) > 0.99
    joint = ref_mask & in_b & mov_m
    return _masked_pearson_r(ref_bp, mov_s, joint, min_voxels=min_voxels)


@dataclass
class NccLockResult:
    shift_nvu_vx: tuple[float, float, float]
    shift_nvu_um: tuple[float, float, float]
    ncc_peak: float
    ncc_zero_shift: float
    ncc_6_nulls: dict[str, float]
    max_null_ncc: float
    null_margin: float
    passed: bool
    null_offset_um: float
    search_radius_vx: tuple[int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "shift_nvu_vx": [round(float(v), 4) for v in self.shift_nvu_vx],
            "shift_nvu_um": [round(float(v), 3) for v in self.shift_nvu_um],
            "shift_du_dv_dn_um": {
                "du_um": round(float(self.shift_nvu_um[2]), 3),
                "dv_um": round(float(self.shift_nvu_um[1]), 3),
                "dn_um": round(float(self.shift_nvu_um[0]), 3),
            },
            "ncc_peak": round(float(self.ncc_peak), 6),
            "ncc_zero_shift": round(float(self.ncc_zero_shift), 6),
            "ncc_6_nulls": {k: round(float(v), 6) for k, v in self.ncc_6_nulls.items()},
            "max_null_ncc": round(float(self.max_null_ncc), 6),
            "null_margin": round(float(self.null_margin), 6),
            "null_offset_um": float(self.null_offset_um),
            "search_radius_vx": [int(v) for v in self.search_radius_vx],
            "passed": bool(self.passed),
        }


def local_ncc_3d_lock(
    ref_vol: np.ndarray,
    ref_mask: np.ndarray,
    mov_vol: np.ndarray,
    mov_mask: np.ndarray,
    *,
    spacing_nvu_um: tuple[float, float, float],
    max_shift_um: float = 55.0,
    null_offset_um: float = 20.0,
    sigma_low_vx: float = 0.75,
    sigma_high_vx: float = 3.5,
    min_ncc: float = 0.10,
    min_overlap_fraction: float = 0.25,
) -> NccLockResult:
    """Estimate local 3D band-pass NCC translation ``(dn, dv, du)`` and gate against 6 displaced nulls.

    Both ``ref_vol`` and ``mov_vol`` share shape ``(Dn, Hv, Wu)`` sampled on the
    nominal (global-affine-mapped) surface slab. Returns exact ``NccLockResult``
    where ``shift_nvu_vx = (dn, dv, du)`` satisfies ``mov[n+dn, v+dv, u+du] ≈ ref[n, v, u]``.
    """
    ref = np.asarray(ref_vol, dtype=np.float64)
    mov = np.asarray(mov_vol, dtype=np.float64)
    rm = np.asarray(ref_mask, dtype=bool)
    mm = np.asarray(mov_mask, dtype=bool)
    if ref.shape != mov.shape or ref.ndim != 3:
        raise ValueError(f"Expected matching 3D volumes, got {ref.shape} and {mov.shape}")

    dn_um, dv_um, du_um = [float(s) for s in spacing_nvu_um]
    # Limit search radius to leave at least half of each dimension for overlap.
    rn = max(1, min(int(math.ceil(max_shift_um / dn_um)), (ref.shape[0] - 2) // 3))
    rv = max(1, min(int(math.ceil(max_shift_um / dv_um)), (ref.shape[1] - 2) // 3))
    ru = max(1, min(int(math.ceil(max_shift_um / du_um)), (ref.shape[2] - 2) // 3))

    ref_bp = bandpass_3d(ref, rm, sigma_low_vx=sigma_low_vx, sigma_high_vx=sigma_high_vx)
    mov_bp = bandpass_3d(mov, mm, sigma_low_vx=sigma_low_vx, sigma_high_vx=sigma_high_vx)

    # Crop inner template from ref so valid-mode FFT correlation tests all lags in [-r, +r].
    tpl = ref_bp[rn : ref.shape[0] - rn, rv : ref.shape[1] - rv, ru : ref.shape[2] - ru]
    tpl_m = rm[rn : ref.shape[0] - rn, rv : ref.shape[1] - rv, ru : ref.shape[2] - ru]
    a_m = tpl_m.astype(np.float64)
    a = np.where(tpl_m, tpl, 0.0)
    b_m = mm.astype(np.float64)
    b = np.where(mm, mov_bp, 0.0)

    def _xcorr(y_arr: np.ndarray, x_arr: np.ndarray) -> np.ndarray:
        return fftconvolve(y_arr, x_arr[::-1, ::-1, ::-1], mode="valid")

    n_ov = _xcorr(b_m, a_m)
    s_a = _xcorr(b_m, a)
    s_b = _xcorr(b, a_m)
    s_aa = _xcorr(b_m, a * a)
    s_bb = _xcorr(b * b, a_m)
    s_ab = _xcorr(b, a)

    min_vox = max(32, int(round(min_overlap_fraction * float(tpl_m.sum()))))
    safe_n = np.where(n_ov >= min_vox, n_ov, 1.0)
    cov = s_ab - (s_a * s_b) / safe_n
    var_a = np.maximum(0.0, s_aa - (s_a * s_a) / safe_n)
    var_b = np.maximum(0.0, s_bb - (s_b * s_b) / safe_n)
    denom = np.sqrt(var_a * var_b)
    ok = (n_ov >= min_vox) & (denom > 1e-12)
    ncc_vol = np.where(ok, cov / np.where(ok, denom, 1.0), -1.0)

    if not np.any(ok):
        null_names = ("+u", "-u", "+v", "-v", "+n", "-n")
        return NccLockResult(
            shift_nvu_vx=(0.0, 0.0, 0.0),
            shift_nvu_um=(0.0, 0.0, 0.0),
            ncc_peak=-1.0,
            ncc_zero_shift=-1.0,
            ncc_6_nulls={k: -1.0 for k in null_names},
            max_null_ncc=-1.0,
            null_margin=0.0,
            passed=False,
            null_offset_um=float(null_offset_um),
            search_radius_vx=(rn, rv, ru),
        )

    best_flat = int(np.argmax(ncc_vol))
    pn, pv, pu = np.unravel_index(best_flat, ncc_vol.shape)
    dn_int = int(pn - rn)
    dv_int = int(pv - rv)
    du_int = int(pu - ru)

    # Sub-voxel parabolic refinement on each axis around the discrete peak.
    def _refine_1d(vals: np.ndarray, idx: int) -> float:
        if 0 < idx < len(vals) - 1:
            yl, yc, yr = float(vals[idx - 1]), float(vals[idx]), float(vals[idx + 1])
            d2 = yl - 2.0 * yc + yr
            if d2 < -1e-9 and yc >= yl and yc >= yr:
                return float(np.clip(0.5 * (yl - yr) / d2, -0.5, 0.5))
        return 0.0

    dn_sub = float(dn_int) + _refine_1d(ncc_vol[:, pv, pu], int(pn))
    dv_sub = float(dv_int) + _refine_1d(ncc_vol[pn, :, pu], int(pv))
    du_sub = float(du_int) + _refine_1d(ncc_vol[pn, pv, :], int(pu))

    # Re-evaluate peak, zero-shift, and all 6 displaced nulls (+/-20 um) consistently on the full band-pass slabs.
    peak_ncc = _eval_shifted_ncc_exact(ref_bp, rm, mov_bp, mm, (dn_sub, dv_sub, du_sub))
    zero_ncc = _eval_shifted_ncc_exact(ref_bp, rm, mov_bp, mm, (0.0, 0.0, 0.0))

    off_n = float(null_offset_um) / dn_um
    off_v = float(null_offset_um) / dv_um
    off_u = float(null_offset_um) / du_um
    null_shifts = {
        "+u": (dn_sub, dv_sub, du_sub + off_u),
        "-u": (dn_sub, dv_sub, du_sub - off_u),
        "+v": (dn_sub, dv_sub + off_v, du_sub),
        "-v": (dn_sub, dv_sub - off_v, du_sub),
        "+n": (dn_sub + off_n, dv_sub, du_sub),
        "-n": (dn_sub - off_n, dv_sub, du_sub),
    }
    ncc_6_nulls = {
        label: _eval_shifted_ncc_exact(ref_bp, rm, mov_bp, mm, sh)
        for label, sh in null_shifts.items()
    }
    max_null = float(max(ncc_6_nulls.values()))
    margin = float(peak_ncc - max_null)
    passed = bool((peak_ncc >= float(min_ncc)) and (peak_ncc > max_null))

    return NccLockResult(
        shift_nvu_vx=(dn_sub, dv_sub, du_sub),
        shift_nvu_um=(dn_sub * dn_um, dv_sub * dv_um, du_sub * du_um),
        ncc_peak=peak_ncc,
        ncc_zero_shift=zero_ncc,
        ncc_6_nulls=ncc_6_nulls,
        max_null_ncc=max_null,
        null_margin=margin,
        passed=passed,
        null_offset_um=float(null_offset_um),
        search_radius_vx=(rn, rv, ru),
    )


def shift_moving_slab(
    mov_vol: np.ndarray,
    mov_mask: np.ndarray,
    shift_nvu_vx: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp ``mov_vol`` into ``ref_vol`` frame via ``mov_aligned[n,v,u] = mov[n+dn, v+dv, u+du]``."""
    dn, dv, du = [float(s) for s in shift_nvu_vx]
    zn, zv, zu = np.meshgrid(
        np.arange(mov_vol.shape[0], dtype=np.float64) + dn,
        np.arange(mov_vol.shape[1], dtype=np.float64) + dv,
        np.arange(mov_vol.shape[2], dtype=np.float64) + du,
        indexing="ij",
    )
    in_b = (
        (zn >= 0.0)
        & (zn <= mov_vol.shape[0] - 1.0)
        & (zv >= 0.0)
        & (zv <= mov_vol.shape[1] - 1.0)
        & (zu >= 0.0)
        & (zu <= mov_vol.shape[2] - 1.0)
    )
    warped = ndi.map_coordinates(mov_vol.astype(np.float64), [zn, zv, zu], order=1, mode="constant", cval=0.0)
    w_mask = (
        in_b
        & (ndi.map_coordinates(mov_mask.astype(np.float64), [zn, zv, zu], order=1, mode="constant", cval=0.0) > 0.99)
    )
    return warped.astype(np.float32), w_mask


# ---------------------------------------------------------------------------
# 6. Dual-energy background fit, residuals, depth top-hat, and ghosting check
# ---------------------------------------------------------------------------

def psf_match_volume(
    vol: np.ndarray,
    valid_mask: np.ndarray,
    *,
    src_fwhm_um: float,
    dst_fwhm_um: float,
    spacing_nvu_um: tuple[float, float, float],
) -> np.ndarray:
    """PSF-match a higher-resolution volume (e.g. 78 keV) to a coarser scan (e.g. 54 keV DLS 7.91 um)."""
    if dst_fwhm_um <= src_fwhm_um:
        return np.asarray(vol, dtype=np.float32)
    sigma_um = math.sqrt(max(0.0, dst_fwhm_um ** 2 - src_fwhm_um ** 2)) / 2.354820045
    sigmas_vx = tuple(sigma_um / float(s) for s in spacing_nvu_um)
    v = np.asarray(vol, dtype=np.float64)
    m = np.asarray(valid_mask, dtype=np.float64)
    num = ndi.gaussian_filter(np.where(m > 0.5, v, 0.0), sigma=sigmas_vx, mode="constant", cval=0.0)
    den = ndi.gaussian_filter(m, sigma=sigmas_vx, mode="constant", cval=0.0)
    out = np.where(den > 1e-6, num / np.where(den > 1e-6, den, 1.0), 0.0)
    return out.astype(np.float32)


def fit_papyrus_background(
    mu_ref: np.ndarray,
    mu_tgt: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float | int]:
    """Fit linear papyrus background ``mu_tgt ≈ alpha * mu_ref + beta`` strictly on blank papyrus."""
    m = np.asarray(valid_mask, dtype=bool)
    x = np.asarray(mu_ref, dtype=np.float64)[m]
    y = np.asarray(mu_tgt, dtype=np.float64)[m]
    n = int(x.size)
    if n < 16:
        raise ValueError(f"Insufficient valid blank-papyrus pixels for linear fit: {n}")
    xm = float(x.mean())
    ym = float(y.mean())
    xc = x - xm
    yc = y - ym
    sxx = float(np.dot(xc, xc))
    if sxx <= 1e-18:
        raise ValueError("Zero variance in reference attenuation on Control B; cannot fit background.")
    alpha = float(np.dot(xc, yc) / sxx)
    beta = float(ym - alpha * xm)
    pred = alpha * x + beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum(yc ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else 0.0
    return {
        "alpha": alpha,
        "beta": beta,
        "r2": r2,
        "n_samples": n,
        "ref_mean": xm,
        "tgt_mean": ym,
        "residual_std_raw": float(np.std(y - pred)),
    }


def compute_dual_energy_residual(
    mu_ref: np.ndarray,
    mu_tgt: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Compute dual-energy residual ``R = mu_tgt - (alpha * mu_ref + beta)``."""
    return (
        np.asarray(mu_tgt, dtype=np.float32)
        - (np.float32(alpha) * np.asarray(mu_ref, dtype=np.float32) + np.float32(beta))
    )


def compute_depth_tophat(
    slab_3d: np.ndarray,
    valid_3d: np.ndarray,
    n_offsets_um: Sequence[float],
    *,
    surface_max_um: float = 8.0,
    bulk_min_um: float = 16.0,
    bulk_max_um: float = 36.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute surface-vs-bulk depth top-hat: ``mean(|n| <= 8 um) - mean(|n| in [16, 36] um)``.

    Returns ``(tophat_2d, surface_2d, bulk_2d, valid_2d)``.
    """
    arr = np.asarray(slab_3d, dtype=np.float64)
    val = np.asarray(valid_3d, dtype=bool)
    offs = np.abs(np.asarray(n_offsets_um, dtype=np.float64))
    if arr.shape[0] != len(offs):
        raise ValueError(f"Slab depth {arr.shape[0]} != len(n_offsets_um) {len(offs)}")

    surf_sel = offs <= float(surface_max_um) + 1e-6
    bulk_sel = (offs >= float(bulk_min_um) - 1e-6) & (offs <= float(bulk_max_um) + 1e-6)
    if not surf_sel.any() or not bulk_sel.any():
        raise ValueError(
            f"n_offsets_um must include both |n|<={surface_max_um} and [{bulk_min_um},{bulk_max_um}] um."
        )

    def _masked_mean(sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sub_v = val[sel]
        cnt = sub_v.sum(axis=0)
        tot = np.where(sub_v, arr[sel], 0.0).sum(axis=0)
        ok = cnt > 0
        mean = np.where(ok, tot / np.where(ok, cnt, 1.0), 0.0)
        return mean, ok

    surf_mean, surf_ok = _masked_mean(surf_sel)
    bulk_mean, bulk_ok = _masked_mean(bulk_sel)
    valid_2d = surf_ok & bulk_ok
    tophat = np.where(valid_2d, surf_mean - bulk_mean, 0.0)
    return (
        tophat.astype(np.float32),
        surf_mean.astype(np.float32),
        bulk_mean.astype(np.float32),
        valid_2d,
    )


def evaluate_ghosting_exclusion(
    slab_residual: np.ndarray,
    valid_3d: np.ndarray,
    n_offsets_um: Sequence[float],
    candidate_mask_2d: np.ndarray | None = None,
    *,
    surface_max_um: float = 12.0,
    ghost_min_um: float = 90.0,
    ghost_max_um: float = 260.0,
) -> dict[str, Any]:
    """Check whether signal peaks in the 90-260 um wrap-behind zone instead of |n| <= 12 um.

    Per ``collab/claude-011.md`` §4, features peaking 90-260 um outward are expected
    ghosts of the neighbouring wrap and MUST be rejected.
    """
    arr = np.asarray(slab_residual, dtype=np.float64)
    val = np.asarray(valid_3d, dtype=bool)
    offs = np.asarray(n_offsets_um, dtype=np.float64)
    abs_offs = np.abs(offs)

    surf_idx = np.flatnonzero(abs_offs <= float(surface_max_um) + 1e-6)
    ghost_idx = np.flatnonzero((abs_offs >= float(ghost_min_um) - 1e-6) & (abs_offs <= float(ghost_max_um) + 1e-6))
    if len(surf_idx) == 0 or len(ghost_idx) == 0:
        return {
            "checked": False,
            "reason": "missing_surface_or_ghost_depths",
            "surface_peak_abs": None,
            "ghost_peak_abs": None,
            "ghost_to_surface_ratio": None,
            "is_ghost_dominated": False,
        }

    if candidate_mask_2d is not None and np.any(candidate_mask_2d):
        roi = np.asarray(candidate_mask_2d, dtype=bool)
    else:
        roi = val[surf_idx].any(axis=0) & val[ghost_idx].any(axis=0)

    if not np.any(roi):
        return {
            "checked": False,
            "reason": "empty_roi",
            "surface_peak_abs": None,
            "ghost_peak_abs": None,
            "ghost_to_surface_ratio": None,
            "is_ghost_dominated": False,
        }

    # Evaluate RMS / 99th-percentile magnitude at every sampled depth inside the ROI.
    depth_p99 = []
    for k in range(len(offs)):
        mk = val[k] & roi
        if mk.sum() >= 4:
            depth_p99.append(float(np.percentile(np.abs(arr[k][mk]), 99)))
        else:
            depth_p99.append(0.0)
    depth_p99_arr = np.asarray(depth_p99, dtype=np.float64)

    surf_peak = float(depth_p99_arr[surf_idx].max())
    ghost_peak = float(depth_p99_arr[ghost_idx].max())
    peak_k = int(np.argmax(depth_p99_arr))
    peak_depth_um = float(offs[peak_k])
    ratio = float(ghost_peak / max(surf_peak, 1e-9))

    # Check spatial correlation between surface mean and ghost-band max-abs plane.
    surf_map = np.mean(arr[surf_idx], axis=0)
    ghost_best_k = int(ghost_idx[int(np.argmax(depth_p99_arr[ghost_idx]))])
    ghost_map = arr[ghost_best_k]
    joint = roi & val[surf_idx].all(axis=0) & val[ghost_best_k]
    corr = _masked_pearson_r(surf_map, ghost_map, joint, min_voxels=8) if joint.sum() >= 8 else 0.0

    is_ghost = bool((abs(peak_depth_um) >= float(ghost_min_um)) or (ratio > 1.15 and abs(corr) > 0.35))
    return {
        "checked": True,
        "roi_pixels": int(roi.sum()),
        "surface_peak_abs_p99": round(surf_peak, 6),
        "ghost_peak_abs_p99": round(ghost_peak, 6),
        "ghost_to_surface_ratio": round(ratio, 4),
        "peak_depth_um": round(peak_depth_um, 2),
        "surface_vs_ghost_r": round(float(corr), 4),
        "is_ghost_dominated": is_ghost,
    }


def compute_control_separation(vals_a: np.ndarray, vals_b: np.ndarray) -> dict[str, float]:
    """Compute Cohen's d and exact Mann-Whitney ROC AUC between Control A and Control B."""
    a = np.asarray(vals_a, dtype=np.float64).ravel()
    b = np.asarray(vals_b, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {"cohens_d": 0.0, "roc_auc": 0.5, "mean_a": 0.0, "mean_b": 0.0}

    mean_a, mean_b = float(a.mean()), float(b.mean())
    var_a, var_b = float(a.var(ddof=1)), float(b.var(ddof=1))
    pooled = math.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b) / max(1, a.size + b.size - 2))
    cohens_d = float((mean_a - mean_b) / pooled) if pooled > 1e-12 else 0.0

    # Exact ROC AUC via rank sum (handles ties with midranks).
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="mergesort")
    sorted_vals = combined[order]
    n_tot = len(combined)
    ranks = np.empty(n_tot, dtype=np.float64)
    i = 0
    while i < n_tot:
        j = i + 1
        while j < n_tot and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_a = float(ranks[: a.size].sum())
    u_stat = rank_sum_a - a.size * (a.size + 1) / 2.0
    auc = float(u_stat / (a.size * b.size))
    return {
        "cohens_d": round(cohens_d, 6),
        "roc_auc": round(auc, 6),
        "mean_a": round(mean_a, 6),
        "mean_b": round(mean_b, 6),
    }


def compute_blank_fpr_stats(
    tophat_blank: np.ndarray,
    valid_blank: np.ndarray,
) -> dict[str, float | int]:
    """Measure blank-papyrus noise floor sigma_blank and empirical FPR at 3 sigma / 4 sigma."""
    vals = np.asarray(tophat_blank, dtype=np.float64)[np.asarray(valid_blank, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size < 16:
        raise ValueError(f"Insufficient valid blank pixels: {vals.size}")
    mu = float(vals.mean())
    sigma = float(vals.std(ddof=1))
    mad_sigma = float(1.4826 * np.median(np.abs(vals - np.median(vals))))
    if sigma <= 1e-12:
        return {
            "n_pixels": int(vals.size),
            "mean_blank": mu,
            "sigma_blank": 0.0,
            "mad_sigma_blank": mad_sigma,
            "fpr_3sigma": 0.0,
            "fpr_4sigma": 0.0,
            "fpr_pos_3sigma": 0.0,
            "fpr_pos_4sigma": 0.0,
            "p99_9_abs_z": 0.0,
        }
    z = (vals - mu) / sigma
    abs_z = np.abs(z)
    return {
        "n_pixels": int(vals.size),
        "mean_blank": round(mu, 6),
        "sigma_blank": round(sigma, 6),
        "mad_sigma_blank": round(mad_sigma, 6),
        "fpr_3sigma": round(float(np.mean(abs_z >= 3.0)), 6),
        "fpr_4sigma": round(float(np.mean(abs_z >= 4.0)), 6),
        "fpr_pos_3sigma": round(float(np.mean(z >= 3.0)), 6),
        "fpr_pos_4sigma": round(float(np.mean(z >= 4.0)), 6),
        "p99_9_abs_z": round(float(np.percentile(abs_z, 99.9)), 4),
        "p99_9_pos_z": round(float(np.percentile(z, 99.9)), 4),
    }


# ---------------------------------------------------------------------------
# 7. End-to-end tile evaluation & Step 0 / Step 1 live runners
# ---------------------------------------------------------------------------

def evaluate_paired_tile(
    mesh_reader: TifxyzPatchReader,
    reader_ref: RemoteZarrV2,
    reader_tgt: RemoteZarrV2,
    m_ref_to_tgt: np.ndarray,
    *,
    v0: int,
    v1: int,
    u0: int,
    u1: int,
    ref_voxel_um: float,
    tgt_voxel_um: float,
    ref_level: int,
    tgt_level: int,
    ref_window: tuple[float, float],
    tgt_window: tuple[float, float],
    uv_spacing_um: float,
    n_offsets_um: Sequence[float],
    ghost_offsets_um: Sequence[float] | None = None,
    max_shift_um: float = 55.0,
    null_offset_um: float = 20.0,
    min_ncc: float = 0.10,
    psf_match_to_um: float | None = None,
    v_stride: int = 1,
    tps_on_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stream one paired surface tile, run mandatory 3D NCC lock, and return aligned slabs + diagnostics."""
    scale_mesh = mesh_reader.meta.get("scale", [0.05, 0.05])
    mesh_du_um = (1.0 / float(scale_mesh[0])) * ref_voxel_um
    mesh_dv_um = (1.0 / float(scale_mesh[1])) * ref_voxel_um * float(max(1, v_stride))

    x, y, z, valid = mesh_reader.read_patch(v0, v1, u0, u1, v_stride=v_stride)
    pts_ref, e_u_ref, e_v_ref, e_n_ref, uv_valid = build_surface_frame(
        x,
        y,
        z,
        valid,
        grid_spacing_u_um=mesh_du_um,
        grid_spacing_v_um=mesh_dv_um,
        target_spacing_um=uv_spacing_um,
    )
    if not uv_valid.any():
        raise ValueError(f"Empty valid mesh patch for v=[{v0},{v1}), u=[{u0},{u1})")

    pts_tgt = map_points_affine_and_tps(pts_ref, m_ref_to_tgt, tps_on_input=tps_on_ref)
    e_n_tgt = apply_affine_vectors(e_n_ref, m_ref_to_tgt, normalize=True)

    n_arr = np.asarray(n_offsets_um, dtype=np.float64)
    dn_um = float(np.median(np.diff(n_arr))) if len(n_arr) >= 2 else float(uv_spacing_um)

    mu_ref, u8_ref, val_ref = sample_surface_slab(
        reader_ref,
        pts_ref,
        e_n_ref,
        uv_valid,
        n_arr,
        native_voxel_um=ref_voxel_um,
        level=ref_level,
        f32_min=ref_window[0],
        f32_max=ref_window[1],
    )
    mu_tgt_raw, u8_tgt_raw, val_tgt_raw = sample_surface_slab(
        reader_tgt,
        pts_tgt,
        e_n_tgt,
        uv_valid,
        n_arr,
        native_voxel_um=tgt_voxel_um,
        level=tgt_level,
        f32_min=tgt_window[0],
        f32_max=tgt_window[1],
    )

    ref_fwhm_um = float(ref_voxel_um) * float(2 ** int(ref_level))
    tgt_fwhm_um = float(tgt_voxel_um) * float(2 ** int(tgt_level))
    target_fwhm: float | None = None
    if psf_match_to_um is not None:
        target_fwhm = max(float(psf_match_to_um), ref_fwhm_um, tgt_fwhm_um)
        mu_ref_cmp = (
            psf_match_volume(
                mu_ref,
                val_ref,
                src_fwhm_um=ref_fwhm_um,
                dst_fwhm_um=target_fwhm,
                spacing_nvu_um=(dn_um, uv_spacing_um, uv_spacing_um),
            )
            if target_fwhm > ref_fwhm_um + 1e-6
            else mu_ref
        )
        mu_tgt_cmp = (
            psf_match_volume(
                mu_tgt_raw,
                val_tgt_raw,
                src_fwhm_um=tgt_fwhm_um,
                dst_fwhm_um=target_fwhm,
                spacing_nvu_um=(dn_um, uv_spacing_um, uv_spacing_um),
            )
            if target_fwhm > tgt_fwhm_um + 1e-6
            else mu_tgt_raw
        )
    else:
        mu_ref_cmp = mu_ref
        mu_tgt_cmp = mu_tgt_raw

    lock = local_ncc_3d_lock(
        mu_ref_cmp,
        val_ref,
        mu_tgt_cmp,
        val_tgt_raw,
        spacing_nvu_um=(dn_um, uv_spacing_um, uv_spacing_um),
        max_shift_um=max_shift_um,
        null_offset_um=null_offset_um,
        min_ncc=min_ncc,
    )

    mu_tgt_aligned, val_tgt_aligned = shift_moving_slab(mu_tgt_cmp, val_tgt_raw, lock.shift_nvu_vx)
    mu_tgt_native_aligned, _ = (
        shift_moving_slab(mu_tgt_raw, val_tgt_raw, lock.shift_nvu_vx)
        if mu_tgt_cmp is not mu_tgt_raw
        else (mu_tgt_aligned, val_tgt_aligned)
    )
    joint_valid = val_ref & val_tgt_aligned

    # Optional wrap-behind ghosting slab sampling using the locked translation in target space.
    ghost_data: dict[str, Any] | None = None
    if ghost_offsets_um is not None and len(ghost_offsets_um) > 0:
        g_offs = np.asarray(ghost_offsets_um, dtype=np.float64)
        e_u_tgt = apply_affine_vectors(e_u_ref, m_ref_to_tgt, normalize=True)
        e_v_tgt = apply_affine_vectors(e_v_ref, m_ref_to_tgt, normalize=True)
        dn_s_um, dv_s_um, du_s_um = lock.shift_nvu_um
        # Build per-point physical shift in target L0 voxel coordinates.
        shift_tgt_l0 = (
            (du_s_um / tgt_voxel_um) * e_u_tgt[uv_valid]
            + (dv_s_um / tgt_voxel_um) * e_v_tgt[uv_valid]
            + (dn_s_um / tgt_voxel_um) * e_n_tgt[uv_valid]
        )
        mu_ref_g, _, val_ref_g = sample_surface_slab(
            reader_ref,
            pts_ref,
            e_n_ref,
            uv_valid,
            g_offs,
            native_voxel_um=ref_voxel_um,
            level=ref_level,
            f32_min=ref_window[0],
            f32_max=ref_window[1],
        )
        mu_tgt_g, _, val_tgt_g = sample_surface_slab(
            reader_tgt,
            pts_tgt,
            e_n_tgt,
            uv_valid,
            g_offs,
            native_voxel_um=tgt_voxel_um,
            level=tgt_level,
            f32_min=tgt_window[0],
            f32_max=tgt_window[1],
            shift_xyz_l0=shift_tgt_l0,
        )
        if target_fwhm is not None:
            # In-plane PSF matching on sparse ghost depths (do not blur across 50+ um ghost gaps).
            if target_fwhm > ref_fwhm_um + 1e-6:
                mu_ref_g = psf_match_volume(
                    mu_ref_g,
                    val_ref_g,
                    src_fwhm_um=ref_fwhm_um,
                    dst_fwhm_um=target_fwhm,
                    spacing_nvu_um=(1e9, uv_spacing_um, uv_spacing_um),
                )
            if target_fwhm > tgt_fwhm_um + 1e-6:
                mu_tgt_g = psf_match_volume(
                    mu_tgt_g,
                    val_tgt_g,
                    src_fwhm_um=tgt_fwhm_um,
                    dst_fwhm_um=target_fwhm,
                    spacing_nvu_um=(1e9, uv_spacing_um, uv_spacing_um),
                )
        ghost_data = {
            "offsets_um": g_offs,
            "mu_ref": mu_ref_g,
            "mu_tgt": mu_tgt_g,
            "valid": val_ref_g & val_tgt_g,
        }

    z_ref_vals = pts_ref[uv_valid, 2]
    return {
        "tile_bounds_vu": [int(v0), int(v1), int(u0), int(u1)],
        "z78_range": [round(float(z_ref_vals.min()), 1), round(float(z_ref_vals.max()), 1)],
        "slab_shape_nvu": [int(s) for s in mu_ref.shape],
        "uv_valid_fraction": round(float(uv_valid.mean()), 4),
        "joint_valid_fraction": round(float(joint_valid.mean()), 4),
        "psf_matched_fwhm_um": round(float(target_fwhm), 3) if target_fwhm is not None else None,
        "n_offsets_um": n_arr,
        "mu_ref": mu_ref_cmp,
        "u8_ref": u8_ref,
        "mu_tgt": mu_tgt_aligned,
        "mu_tgt_native": mu_tgt_native_aligned,
        "joint_valid": joint_valid,
        "lock": lock,
        "ghost_data": ghost_data,
    }


def evaluate_sheet_integrity_tile(
    mesh_reader: TifxyzPatchReader,
    reader_vol: RemoteZarrV2,
    *,
    v0: int,
    v1: int,
    u0: int,
    u1: int,
    level: int,
    window_78: tuple[float, float],
    uv_spacing_um: float,
    v_stride: int = 1,
    voxel_um: float = DEFAULT_VOXEL_UM["78keV"],
) -> dict[str, Any]:
    """Evaluate surface sheet integrity (continuity, SNR, layer coherence)."""
    scale_mesh = mesh_reader.meta.get("scale", [0.05, 0.05])
    mesh_du_um = (1.0 / float(scale_mesh[0])) * float(voxel_um)
    mesh_dv_um = (1.0 / float(scale_mesh[1])) * float(voxel_um) * float(max(1, v_stride))

    x, y, z, valid = mesh_reader.read_patch(v0, v1, u0, u1, v_stride=v_stride)
    pts, _, _, e_n, uv_valid = build_surface_frame(
        x,
        y,
        z,
        valid,
        grid_spacing_u_um=mesh_du_um,
        grid_spacing_v_um=mesh_dv_um,
        target_spacing_um=uv_spacing_um,
    )
    step_um = max(8.0, float(uv_spacing_um))
    n_offs = np.arange(-4, 5, dtype=np.float64) * step_um
    mu_vol, u8_vol, val_3d = sample_surface_slab(
        reader_vol,
        pts,
        e_n,
        uv_valid,
        n_offs,
        native_voxel_um=float(voxel_um),
        level=level,
        f32_min=window_78[0],
        f32_max=window_78[1],
    )
    mid_idx = len(n_offs) // 2
    valid_2d = val_3d[mid_idx]
    coherence = layer_coherence_score(
        np.clip(np.rint(u8_vol), 0, 255).astype(np.uint8),
        mesh_index=mid_idx,
        delta=1,
        patch_size=min(64, int(u8_vol.shape[1]), int(u8_vol.shape[2])),
        max_shift=3,
        valid_mask=valid_2d,
    )
    # Surface-vs-surround contrast along normal.
    surf_vals = mu_vol[mid_idx][valid_2d]
    outer_vals = np.concatenate([mu_vol[0][val_3d[0]], mu_vol[-1][val_3d[-1]]])
    z_vals = pts[uv_valid, 2]
    return {
        "tile_bounds_vu": [int(v0), int(v1), int(u0), int(u1)],
        "z78_range": [round(float(z_vals.min()), 1), round(float(z_vals.max()), 1)],
        "slab_shape_nvu": [int(s) for s in mu_vol.shape],
        "valid_surface_fraction": round(float(valid_2d.mean()), 4),
        "surface_mu_mean": round(float(surf_vals.mean()), 6) if surf_vals.size else 0.0,
        "surface_mu_std": round(float(surf_vals.std()), 6) if surf_vals.size else 0.0,
        "outer_mu_mean": round(float(outer_vals.mean()), 6) if outer_vals.size else 0.0,
        "sheet_parallel_fraction": round(float(coherence["sheet_parallel_fraction"]), 4),
        "mean_zero_shift_r": round(float(coherence["mean_zero_shift_r"]), 4),
        "mean_best_shifted_r": round(float(coherence["mean_best_shifted_r"]), 4),
        "patch_count": int(coherence["patch_count"]),
    }


def _save_private_heatmap_png(arr: np.ndarray, valid: np.ndarray, path: Path, vmin: float, vmax: float) -> None:
    """Write a diverging grayscale/RGB visualization PNG strictly to the private directory."""
    from PIL import Image

    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    # Diverging blue (low) -> neutral gray (0) -> red/yellow (high); invalid = dark slate.
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    r = np.clip(2.0 * norm, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(2.0 * norm - 1.0), 0.0, 1.0)
    b = np.clip(2.0 * (1.0 - norm), 0.0, 1.0)
    rgb[..., 0] = np.where(valid, np.rint(r * 255.0), 16).astype(np.uint8)
    rgb[..., 1] = np.where(valid, np.rint(g * 255.0), 16).astype(np.uint8)
    rgb[..., 2] = np.where(valid, np.rint(b * 255.0), 24).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(str(path), format="PNG")


def run_step0_preflight(
    *,
    vol_78_url: str = DEFAULT_URLS["vol_78"],
    vol_137_url: str = DEFAULT_URLS["vol_137"],
    transform_78_url: str = DEFAULT_URLS["transform_78_to_dls"],
    transform_137_url: str = DEFAULT_URLS["transform_137_to_dls"],
    mesh_78_url: str = DEFAULT_URLS["mesh_78"],
    level: int = 2,
    cache_mb: float = 256.0,
    workers: int = 8,
    out_json: str | os.PathLike[str] | None = None,
    private_out_dir: str | os.PathLike[str] | None = None,
    tile_a_vu: tuple[int, int, int, int] = (1428, 1444, 4082, 4114),
    tile_b_vu: tuple[int, int, int, int] = (1428, 1444, 5109, 5141),
    tile_c_vu: tuple[int, int, int, int] = (2934, 2950, 5129, 5161),
    uv_spacing_um: float | None = None,
    run_step1: bool = False,
    step1_level: int = 3,
    max_total_mb: float = 176.0,
) -> dict[str, Any]:
    """Execute live Step 0 Preflight (< 45 MB) and optional Step 1 dual-energy strip (< 180 MB total)."""
    if private_out_dir is not None:
        priv_path = Path(private_out_dir)
        _assert_outside_git_repo(priv_path)
        priv_path.mkdir(parents=True, exist_ok=True)
    else:
        priv_path = None

    sess = requests.Session() if requests is not None else None
    w78_lo, w78_hi, _ = load_volume_window_f32(vol_78_url, fallback=DEFAULT_WINDOW_F32["78keV"], session=sess)
    w137_lo, w137_hi, _ = load_volume_window_f32(vol_137_url, fallback=DEFAULT_WINDOW_F32["137keV"], session=sess)
    m78 = load_affine_transform(transform_78_url, session=sess)
    m137 = load_affine_transform(transform_137_url, session=sess)
    m_78_to_137 = compose_affine_via_dls(m78, m137)

    vox_step_um = DEFAULT_VOXEL_UM["78keV"] * (2 ** int(level))
    uv_um = float(uv_spacing_um) if uv_spacing_um is not None else vox_step_um
    # Dense depth grid covering |n| <= 8 um, |n| in [16, 36] um, plus margin for NCC +/-20 um nulls.
    n_step_um = min(8.0, vox_step_um)
    n_offsets_um = np.arange(-8, 9, dtype=np.float64) * n_step_um
    ghost_offsets_um = np.array([-160.0, -110.0, 0.0, 110.0, 160.0], dtype=np.float64)

    with (
        TifxyzPatchReader(mesh_78_url) as mesh_reader,
        RemoteZarrV2(vol_78_url, level=str(level), cache_mb=cache_mb, workers=workers) as r78,
        RemoteZarrV2(vol_137_url, level=str(level), cache_mb=cache_mb, workers=workers) as r137,
    ):
        tile_b = evaluate_paired_tile(
            mesh_reader,
            r78,
            r137,
            m_78_to_137,
            v0=tile_b_vu[0],
            v1=tile_b_vu[1],
            u0=tile_b_vu[2],
            u1=tile_b_vu[3],
            ref_voxel_um=DEFAULT_VOXEL_UM["78keV"],
            tgt_voxel_um=DEFAULT_VOXEL_UM["137keV"],
            ref_level=level,
            tgt_level=level,
            ref_window=(w78_lo, w78_hi),
            tgt_window=(w137_lo, w137_hi),
            uv_spacing_um=uv_um,
            n_offsets_um=n_offsets_um,
            ghost_offsets_um=ghost_offsets_um,
            max_shift_um=55.0,
            null_offset_um=20.0,
            min_ncc=0.10,
        )

        tile_a = evaluate_paired_tile(
            mesh_reader,
            r78,
            r137,
            m_78_to_137,
            v0=tile_a_vu[0],
            v1=tile_a_vu[1],
            u0=tile_a_vu[2],
            u1=tile_a_vu[3],
            ref_voxel_um=DEFAULT_VOXEL_UM["78keV"],
            tgt_voxel_um=DEFAULT_VOXEL_UM["137keV"],
            ref_level=level,
            tgt_level=level,
            ref_window=(w78_lo, w78_hi),
            tgt_window=(w137_lo, w137_hi),
            uv_spacing_um=uv_um,
            n_offsets_um=n_offsets_um,
            ghost_offsets_um=ghost_offsets_um,
            max_shift_um=55.0,
            null_offset_um=20.0,
            min_ncc=0.10,
        )

        tile_c = evaluate_sheet_integrity_tile(
            mesh_reader,
            r78,
            v0=tile_c_vu[0],
            v1=tile_c_vu[1],
            u0=tile_c_vu[2],
            u1=tile_c_vu[3],
            level=level,
            window_78=(w78_lo, w78_hi),
            uv_spacing_um=uv_um,
        )

        step0_bytes_78 = int(r78.bytes_fetched)
        step0_bytes_137 = int(r137.bytes_fetched)
        step0_chunks_78 = int(r78.chunks_fetched)
        step0_chunks_137 = int(r137.chunks_fetched)
        step0_mesh_bytes = int(mesh_reader.bytes_fetched)
        step0_total_bytes = step0_bytes_78 + step0_bytes_137 + step0_mesh_bytes

        # Fit linear papyrus background strictly on Control B (confirmed blank papyrus).
        bg_fit = fit_papyrus_background(tile_b["mu_ref"], tile_b["mu_tgt"], tile_b["joint_valid"])
        alpha = float(bg_fit["alpha"])
        beta = float(bg_fit["beta"])

        res_b = compute_dual_energy_residual(tile_b["mu_ref"], tile_b["mu_tgt"], alpha, beta)
        res_a = compute_dual_energy_residual(tile_a["mu_ref"], tile_a["mu_tgt"], alpha, beta)

        th_b, surf_b, bulk_b, v2d_b = compute_depth_tophat(res_b, tile_b["joint_valid"], n_offsets_um)
        th_a, surf_a, bulk_a, v2d_a = compute_depth_tophat(res_a, tile_a["joint_valid"], n_offsets_um)
        th78_b, _, _, v78_b = compute_depth_tophat(tile_b["mu_ref"], tile_b["joint_valid"], n_offsets_um)
        th78_a, _, _, v78_a = compute_depth_tophat(tile_a["mu_ref"], tile_a["joint_valid"], n_offsets_um)

        blank_stats = compute_blank_fpr_stats(th_b, v2d_b)
        sep_tophat_r = compute_control_separation(th_a[v2d_a], th_b[v2d_b])
        sep_surf_r = compute_control_separation(surf_a[v2d_a], surf_b[v2d_b])
        sep_tophat_78 = compute_control_separation(th78_a[v78_a], th78_b[v78_b])

        # Wrap-behind ghosting check on Tile A & Tile B.
        ghost_a = evaluate_ghosting_exclusion(
            compute_dual_energy_residual(tile_a["ghost_data"]["mu_ref"], tile_a["ghost_data"]["mu_tgt"], alpha, beta),
            tile_a["ghost_data"]["valid"],
            tile_a["ghost_data"]["offsets_um"],
        )
        ghost_b = evaluate_ghosting_exclusion(
            compute_dual_energy_residual(tile_b["ghost_data"]["mu_ref"], tile_b["ghost_data"]["mu_tgt"], alpha, beta),
            tile_b["ghost_data"]["valid"],
            tile_b["ghost_data"]["offsets_um"],
        )

        lock_a: NccLockResult = tile_a["lock"]
        lock_b: NccLockResult = tile_b["lock"]
        step0_ncc_passed = bool(lock_a.passed and lock_b.passed)

        # Save optional private control plots outside the git repo.
        if priv_path is not None:
            sig = max(float(blank_stats["sigma_blank"]), 1e-6)
            mu_b = float(blank_stats["mean_blank"])
            _save_private_heatmap_png((th_a - mu_b) / sig, v2d_a, priv_path / "step0_control_a_zmap.png", -4.0, 4.0)
            _save_private_heatmap_png((th_b - mu_b) / sig, v2d_b, priv_path / "step0_control_b_zmap.png", -4.0, 4.0)

        step1_summary: dict[str, Any] | None = None
        step1_bytes = 0
        step1_chunks = 0

        if run_step1 and step0_ncc_passed and priv_path is not None:
            remaining_mb = float(max_total_mb) - (step0_total_bytes / (1024.0 * 1024.0))
            if remaining_mb > 20.0:
                step1_summary = _run_step1_agraphon_strip(
                    mesh_reader=mesh_reader,
                    vol_78_url=vol_78_url,
                    vol_137_url=vol_137_url,
                    m_78_to_137=m_78_to_137,
                    w78=(w78_lo, w78_hi),
                    w137=(w137_lo, w137_hi),
                    level=step1_level,
                    cache_mb=cache_mb,
                    workers=workers,
                    private_out_dir=priv_path,
                    budget_mb=remaining_mb,
                )
                step1_bytes = int(step1_summary["bytes_fetched"])
                step1_chunks = int(step1_summary["chunks_fetched"])

    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "pipeline": "gemini_lane/paris4_dual_energy.py",
        "mode": "cpu_only_streaming",
        "sources": {
            "vol_78": _sanitize_source_label(vol_78_url),
            "vol_137": _sanitize_source_label(vol_137_url),
            "transform_78_to_dls": _sanitize_source_label(transform_78_url),
            "transform_137_to_dls": _sanitize_source_label(transform_137_url),
            "mesh_78": _sanitize_source_label(mesh_78_url),
        },
        "radiometry": {
            "window_f32_78keV": [w78_lo, w78_hi],
            "window_f32_137keV": [w137_lo, w137_hi],
            "preflight_zarr_level": int(level),
            "preflight_voxel_um": round(vox_step_um, 3),
            "uv_spacing_um": round(uv_um, 3),
            "n_offsets_um": [round(float(v), 2) for v in n_offsets_um],
        },
        "step0_preflight_passed": step0_ncc_passed,
        # Top-level summary keys explicitly requested for quick inspection.
        "ncc_peak": round(float(min(lock_a.ncc_peak, lock_b.ncc_peak)), 6),
        "ncc_6_nulls": {
            "tile_a": {k: round(float(v), 6) for k, v in lock_a.ncc_6_nulls.items()},
            "tile_b": {k: round(float(v), 6) for k, v in lock_b.ncc_6_nulls.items()},
        },
        "sigma_blank": blank_stats["sigma_blank"],
        "fpr_3sigma": blank_stats["fpr_3sigma"],
        "fpr_4sigma": blank_stats["fpr_4sigma"],
        "control_a_vs_b_auc_or_d": {
            "tophat_R_78_137": sep_tophat_r,
            "surface_R_78_137": sep_surf_r,
            "tophat_mu_78": sep_tophat_78,
        },
        "bytes_fetched": int(step0_total_bytes),
        "chunks_fetched": int(step0_chunks_78 + step0_chunks_137),
        "streaming_breakdown": {
            "step0_bytes_78": step0_bytes_78,
            "step0_bytes_137": step0_bytes_137,
            "step0_mesh_range_bytes": step0_mesh_bytes,
            "step0_total_mb": round(step0_total_bytes / (1024.0 * 1024.0), 3),
            "step0_chunks_78": step0_chunks_78,
            "step0_chunks_137": step0_chunks_137,
            "step1_bytes": step1_bytes,
            "step1_chunks": step1_chunks,
            "combined_total_bytes": int(step0_total_bytes + step1_bytes),
            "combined_total_mb": round((step0_total_bytes + step1_bytes) / (1024.0 * 1024.0), 3),
            "cache_mb": float(cache_mb),
        },
        "papyrus_background_fit_control_b": {
            k: (round(float(v), 6) if isinstance(v, float) else int(v)) for k, v in bg_fit.items()
        },
        "blank_noise_stats": blank_stats,
        "controls": {
            "tile_a_last_legible_column": {
                "tile_bounds_vu": tile_a["tile_bounds_vu"],
                "z78_range": tile_a["z78_range"],
                "slab_shape_nvu": tile_a["slab_shape_nvu"],
                "uv_valid_fraction": tile_a["uv_valid_fraction"],
                "joint_valid_fraction": tile_a["joint_valid_fraction"],
                "ncc_lock": lock_a.to_dict(),
                "ghosting_check": ghost_a,
            },
            "tile_b_blank_agraphon": {
                "tile_bounds_vu": tile_b["tile_bounds_vu"],
                "z78_range": tile_b["z78_range"],
                "slab_shape_nvu": tile_b["slab_shape_nvu"],
                "uv_valid_fraction": tile_b["uv_valid_fraction"],
                "joint_valid_fraction": tile_b["joint_valid_fraction"],
                "ncc_lock": lock_b.to_dict(),
                "ghosting_check": ghost_b,
            },
            "tile_c_lower_extension_integrity": tile_c,
        },
    }

    assert_no_local_paths(result)
    if out_json is not None:
        out_p = Path(out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if sess is not None:
        sess.close()
    if step1_summary is not None:
        result["_step1_private_summary"] = step1_summary
    return result


# Verified 137 keV non-fill wrap islands on 20260701183124-w010-027 (last legible col + 11 agraphon wraps).
STEP1_WRAP_ISLANDS: tuple[tuple[str, int, int, str], ...] = (
    ("wrap_06_last_col", 4055, 4230, "control_text"),
    ("wrap_07_blank_margin", 4582, 4744, "control_blank"),
    ("wrap_08_blank_agraphon", 5080, 5230, "control_blank"),
    ("wrap_09_agraphon", 5550, 5688, "test_agraphon"),
    ("wrap_10_agraphon", 5992, 6120, "test_agraphon"),
    ("wrap_11_agraphon", 6406, 6518, "test_agraphon"),
    ("wrap_12_agraphon", 6790, 6888, "test_agraphon"),
    ("wrap_13_agraphon", 7142, 7228, "test_agraphon"),
    ("wrap_14_agraphon", 7468, 7545, "test_agraphon"),
    ("wrap_15_agraphon", 7772, 7842, "test_agraphon"),
    ("wrap_16_agraphon", 8052, 8114, "test_agraphon"),
    ("wrap_17_agraphon", 8318, 8366, "test_agraphon"),
)


def _run_step1_agraphon_strip(
    *,
    mesh_reader: TifxyzPatchReader,
    vol_78_url: str,
    vol_137_url: str,
    m_78_to_137: np.ndarray,
    w78: tuple[float, float],
    w137: tuple[float, float],
    level: int,
    cache_mb: float,
    workers: int,
    private_out_dir: Path,
    budget_mb: float,
    v_range: tuple[int, int] = (1368, 1502),
) -> dict[str, Any]:
    """Run Step 1 (78 vs 137 keV) across all 12 wrap islands in the 137 keV non-fill overlap.

    Saves all private maps, arrays, and wrap reports strictly outside the git repo.
    """
    _assert_outside_git_repo(private_out_dir)
    vox_um = DEFAULT_VOXEL_UM["78keV"] * (2 ** int(level))
    uv_um = vox_um
    n_offsets_um = np.arange(-4, 5, dtype=np.float64) * 10.0  # -40..+40 um in 10 um steps
    ghost_offsets_um = np.array([-160.0, -110.0, 0.0, 110.0, 160.0], dtype=np.float64)
    mesh_bytes_start = int(mesh_reader.bytes_fetched)

    wrap_records: list[dict[str, Any]] = []
    blank_fit_data: dict[str, Any] | None = None
    alpha, beta = 0.0, 0.0
    mu_blank, sigma_blank, p999_blank = 0.0, 1.0, 3.0

    with (
        RemoteZarrV2(vol_78_url, level=str(level), cache_mb=cache_mb, workers=workers) as r78,
        RemoteZarrV2(vol_137_url, level=str(level), cache_mb=cache_mb, workers=workers) as r137,
    ):
        def _budget_used_mb() -> float:
            b = r78.bytes_fetched + r137.bytes_fetched + (mesh_reader.bytes_fetched - mesh_bytes_start)
            return float(b) / (1024.0 * 1024.0)

        def _estimate_extra_mb(u0: int, u1: int) -> float:
            # Sparse 5-row probe to count exact uncached chunks for this wrap before fetching CT data.
            x_p, y_p, z_p, v_p = mesh_reader.read_patch(v_range[0], v_range[1], u0, u1, v_stride=24)
            if not v_p.any():
                return 0.0
            pts78 = np.stack([x_p[v_p], y_p[v_p], z_p[v_p]], axis=-1).astype(np.float64)
            pts137 = apply_affine_points(pts78, m_78_to_137)
            scale = float(2 ** int(level))
            extra_chunks = 0
            for reader, pts in ((r78, pts78), (r137, pts137)):
                sz, sy, sx = reader.shape
                cz_s, cy_s, cx_s = reader.chunks
                q = (pts + 0.5) / scale - 0.5
                ok = (q[:, 0] >= 0) & (q[:, 0] < sx) & (q[:, 1] >= 0) & (q[:, 1] < sy) & (q[:, 2] >= 0) & (q[:, 2] < sz)
                if not ok.any():
                    continue
                c = np.floor(q[ok][:, ::-1] / np.array([cz_s, cy_s, cx_s])).astype(int)
                for row in np.unique(c, axis=0):
                    cid = (int(row[0]), int(row[1]), int(row[2]))
                    if cid not in reader.cache:
                        extra_chunks += 1
            return float(extra_chunks) * 2.0

        ordered_islands = [w for w in STEP1_WRAP_ISLANDS if w[0] == "wrap_08_blank_agraphon"] + [
            w for w in STEP1_WRAP_ISLANDS if w[0] != "wrap_08_blank_agraphon"
        ]

        for name, u0, u1, role in ordered_islands:
            used_mb = _budget_used_mb()
            est_extra = _estimate_extra_mb(u0, u1)
            if used_mb + est_extra > float(budget_mb) - 2.0 and name != "wrap_08_blank_agraphon":
                wrap_records.append(
                    {
                        "wrap": name,
                        "role": role,
                        "skipped_budget": True,
                        "est_extra_mb": round(est_extra, 2),
                    }
                )
                continue

            tile = evaluate_paired_tile(
                mesh_reader,
                r78,
                r137,
                m_78_to_137,
                v0=v_range[0],
                v1=v_range[1],
                u0=u0,
                u1=u1,
                ref_voxel_um=DEFAULT_VOXEL_UM["78keV"],
                tgt_voxel_um=DEFAULT_VOXEL_UM["137keV"],
                ref_level=level,
                tgt_level=level,
                ref_window=w78,
                tgt_window=w137,
                uv_spacing_um=uv_um,
                n_offsets_um=n_offsets_um,
                ghost_offsets_um=ghost_offsets_um,
                max_shift_um=55.0,
                null_offset_um=20.0,
                min_ncc=0.10,
                v_stride=2,
            )

            lock: NccLockResult = tile["lock"]
            if name == "wrap_08_blank_agraphon":
                blank_fit_data = fit_papyrus_background(tile["mu_ref"], tile["mu_tgt"], tile["joint_valid"])
                alpha = float(blank_fit_data["alpha"])
                beta = float(blank_fit_data["beta"])

            if blank_fit_data is None or not lock.passed:
                wrap_records.append(
                    {
                        "wrap": name,
                        "role": role,
                        "u_range": [u0, u1],
                        "z78_range": tile["z78_range"],
                        "ncc_lock": lock.to_dict(),
                        "residual_computed": False,
                    }
                )
                continue

            res = compute_dual_energy_residual(tile["mu_ref"], tile["mu_tgt"], alpha, beta)
            th, surf, bulk, v2d = compute_depth_tophat(res, tile["joint_valid"], n_offsets_um)

            if name == "wrap_08_blank_agraphon":
                b_stats = compute_blank_fpr_stats(th, v2d)
                mu_blank = float(b_stats["mean_blank"])
                sigma_blank = max(float(b_stats["sigma_blank"]), 1e-6)
                p999_blank = float(b_stats["p99_9_abs_z"])

            z_map = np.where(v2d, (th - mu_blank) / sigma_blank, 0.0).astype(np.float32)
            vals_z = z_map[v2d]
            abs_z = np.abs(vals_z)
            # Check stroke-like connected components at |z| >= max(4.0, p99.9_blank) (min 15 pixels ~ 0.005 cm^2).
            hi_mask = v2d & (np.abs(z_map) >= max(4.0, p999_blank))
            lbl, n_comp = ndi.label(hi_mask)
            max_comp_px = 0
            max_comp_span_mm = 0.0
            if n_comp > 0:
                sizes = np.bincount(lbl.ravel())[1:]
                top_cid = int(np.argmax(sizes)) + 1
                max_comp_px = int(sizes[top_cid - 1])
                ys, xs = np.nonzero(lbl == top_cid)
                max_comp_span_mm = float(
                    math.hypot((ys.max() - ys.min() + 1) * uv_um, (xs.max() - xs.min() + 1) * uv_um) / 1000.0
                )

            ghost = evaluate_ghosting_exclusion(
                compute_dual_energy_residual(tile["ghost_data"]["mu_ref"], tile["ghost_data"]["mu_tgt"], alpha, beta),
                tile["ghost_data"]["valid"],
                tile["ghost_data"]["offsets_um"],
                candidate_mask_2d=hi_mask if hi_mask.sum() >= 8 else None,
            )

            # Save private arrays & PNGs strictly inside private_out_dir.
            np.savez_compressed(
                private_out_dir / f"step1_{name}.npz",
                z_map=z_map,
                tophat_R=th,
                surface_R=surf,
                valid_2d=v2d,
            )
            _save_private_heatmap_png(z_map, v2d, private_out_dir / f"step1_{name}_zmap.png", -4.5, 4.5)

            rec = {
                "wrap": name,
                "role": role,
                "u_range": [u0, u1],
                "z78_range": tile["z78_range"],
                "slab_shape_nvu": tile["slab_shape_nvu"],
                "valid_pixels": int(v2d.sum()),
                "ncc_lock": lock.to_dict(),
                "residual_computed": True,
                "z_mean": round(float(vals_z.mean()), 4) if vals_z.size else 0.0,
                "z_std": round(float(vals_z.std()), 4) if vals_z.size else 0.0,
                "z_p99_9_abs": round(float(np.percentile(abs_z, 99.9)), 4) if vals_z.size else 0.0,
                "z_max_abs": round(float(abs_z.max()), 4) if vals_z.size else 0.0,
                "frac_ge_3sigma": round(float(np.mean(abs_z >= 3.0)), 6) if vals_z.size else 0.0,
                "frac_ge_4sigma": round(float(np.mean(abs_z >= 4.0)), 6) if vals_z.size else 0.0,
                "max_4sigma_component_px": max_comp_px,
                "max_4sigma_component_span_mm": round(max_comp_span_mm, 3),
                "ghosting_check": ghost,
                "exceeds_stopping_rule": bool(
                    (max_comp_span_mm >= 15.0) and (not ghost.get("is_ghost_dominated", False))
                ),
            }
            wrap_records.append(rec)

        step1_bytes = int(r78.bytes_fetched + r137.bytes_fetched + (mesh_reader.bytes_fetched - mesh_bytes_start))
        step1_chunks = int(r78.chunks_fetched + r137.chunks_fetched)

    summary = {
        "step1_level": int(level),
        "voxel_um": round(vox_um, 3),
        "v_range": list(v_range),
        "background_fit_wrap08": blank_fit_data,
        "blank_sigma": round(sigma_blank, 6),
        "blank_p99_9_abs_z": round(p999_blank, 4),
        "bytes_fetched": step1_bytes,
        "bytes_fetched_mb": round(step1_bytes / (1024.0 * 1024.0), 3),
        "chunks_fetched": step1_chunks,
        "wraps_evaluated": sum(1 for r in wrap_records if r.get("residual_computed")),
        "wraps_locked": sum(1 for r in wrap_records if r.get("ncc_lock", {}).get("passed")),
        "any_stopping_rule_trigger": any(bool(r.get("exceeds_stopping_rule")) for r in wrap_records),
        "wraps": wrap_records,
    }
    (private_out_dir / "step1_strip_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# 8. Phase 2: 54 vs 78 keV (R_{54/78}), outer wraps w130-w135, extended height,
#    and 1.129 um native crust preflight (strictly < 250 MB, zero overlap)
# ---------------------------------------------------------------------------

FORBIDDEN_UNREAD_COORD_KEYS = frozenset(
    {
        "u_range",
        "v_range",
        "tile_bounds_vu",
        "z78_range",
        "z_dls_range",
        "bbox_vu",
        "candidate_xy",
        "candidate_uv",
        "wraps",
    }
)

# Non-overlapping Phase 2 outer-wrap tiles on 20260926044226-on-20230205180739-7.91um.tifxyz
# Scaled from meta.json winding_bounds (grid 956x56913 -> TIFF 1173x69925, factor ~1.2286).
PHASE2_OUTER_WRAPS: tuple[tuple[str, int, int], ...] = (
    ("w135", 540, 554),
    ("w134", 1590, 1604),
    ("w133", 2600, 2614),
    ("w132", 3610, 3624),
    ("w131", 4608, 4622),
    ("w130", 5600, 5614),
)


def assert_no_unread_coordinates(payload: Any, *, unread_section_names: Sequence[str] = ()) -> None:
    """Raise ValueError if any unread-scan section in public JSON exposes candidate coordinates."""
    sections = set(
        unread_section_names
        or (
            "agraphon_strip_54_78",
            "outer_wraps_w130_w135",
            "extended_height_20260701",
            "unread_scans",
        )
    )

    def _walk_forbidden(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k in FORBIDDEN_UNREAD_COORD_KEYS:
                    raise ValueError(f"Unread candidate coordinate key {k!r} exposed in public JSON at {path}.{k}")
                _walk_forbidden(v, f"{path}.{k}")
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                _walk_forbidden(item, f"{path}[{idx}]")

    def _scan_root(node: Any, path: str = "$") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                child_path = f"{path}.{k}"
                if k in sections or "unread" in str(k).lower():
                    _walk_forbidden(v, child_path)
                else:
                    _scan_root(v, child_path)
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                _scan_root(item, f"{path}[{idx}]")

    _scan_root(payload)


def compute_crust_metrics(
    mu_native: np.ndarray,
    valid_3d: np.ndarray,
    n_offsets_um: Sequence[float],
    *,
    surface_max_um: float = 4.0,
    bulk_min_um: float = 10.0,
    bulk_max_um: float = 22.0,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray]:
    """Compute native 1.129 um crust depth top-hat and surface high-frequency micro-roughness."""
    th, surf, bulk, v2d = compute_depth_tophat(
        mu_native,
        valid_3d,
        n_offsets_um,
        surface_max_um=surface_max_um,
        bulk_min_um=bulk_min_um,
        bulk_max_um=bulk_max_um,
    )
    vals_th = th[v2d]
    vals_surf = surf[v2d]
    vals_bulk = bulk[v2d]

    # Measure high-frequency micro-roughness via 2D discrete Laplacian strictly inside eroded interior.
    interior = ndi.binary_erosion(v2d, iterations=1) if v2d.any() else v2d
    lap = ndi.laplace(np.where(v2d, surf.astype(np.float64), 0.0))
    lap_vals = lap[interior]
    hf_rms = float(np.sqrt(np.mean(lap_vals ** 2))) if lap_vals.size else 0.0

    stats: dict[str, float | int] = {
        "valid_pixels": int(v2d.sum()),
        "surface_mu_mean": round(float(vals_surf.mean()), 6) if vals_surf.size else 0.0,
        "surface_mu_std": round(float(vals_surf.std()), 6) if vals_surf.size else 0.0,
        "bulk_mu_mean": round(float(vals_bulk.mean()), 6) if vals_bulk.size else 0.0,
        "tophat_mean": round(float(vals_th.mean()), 6) if vals_th.size else 0.0,
        "tophat_std": round(float(vals_th.std()), 6) if vals_th.size else 0.0,
        "surface_hf_roughness_rms": round(hf_rms, 6),
    }
    return stats, th, surf, v2d


def _estimate_extra_chunks_mb(
    mesh_reader: TifxyzPatchReader,
    readers_and_mappers: Sequence[tuple[RemoteZarrV2, int, Any]],
    v0: int,
    v1: int,
    u0: int,
    u1: int,
    *,
    v_stride: int = 12,
    pad_vx_lvl: int = 1,
) -> float:
    """Estimate uncached chunk traffic (MiB) for a mesh patch including normal/slab padding."""
    x_p, y_p, z_p, v_p = mesh_reader.read_patch(v0, v1, u0, u1, v_stride=max(1, int(v_stride)))
    if not v_p.any():
        return 0.0
    pts_base = np.stack([x_p[v_p], y_p[v_p], z_p[v_p]], axis=-1).astype(np.float64)
    extra_bytes = 0
    pad = max(0, int(pad_vx_lvl))
    for reader, level, mapper in readers_and_mappers:
        pts = mapper(pts_base) if mapper is not None else pts_base
        scale = float(2 ** int(level))
        sz, sy, sx = reader.shape
        cz_s, cy_s, cx_s = reader.chunks
        q = (pts + 0.5) / scale - 0.5
        ok = (q[:, 0] >= 0) & (q[:, 0] < sx) & (q[:, 1] >= 0) & (q[:, 1] < sy) & (q[:, 2] >= 0) & (q[:, 2] < sz)
        if not ok.any():
            continue
        q_ok = q[ok]
        c_set: set[tuple[int, int, int]] = set()
        offsets = (-pad, 0, pad) if pad > 0 else (0,)
        for dz in offsets:
            for dy in offsets:
                for dx in offsets:
                    qp = np.clip(
                        q_ok + np.array([dx, dy, dz], dtype=np.float64),
                        [0.0, 0.0, 0.0],
                        [sx - 1.0, sy - 1.0, sz - 1.0],
                    )
                    c = np.floor(qp[:, ::-1] / np.array([cz_s, cy_s, cx_s])).astype(int)
                    for row in np.unique(c, axis=0):
                        c_set.add((int(row[0]), int(row[1]), int(row[2])))
        chunk_bytes = int(cz_s * cy_s * cx_s * np.dtype(reader.dtype).itemsize)
        for cid in c_set:
            if cid not in reader.cache:
                extra_bytes += chunk_bytes
    return float(extra_bytes) / (1024.0 * 1024.0)


def _summarize_unread_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Produce strictly non-identifying aggregate public metrics from unread wrap/tile records."""
    evaluated = [r for r in records if r.get("residual_computed")]
    locked = [r for r in records if r.get("ncc_lock", {}).get("passed")]
    skipped = [r for r in records if r.get("skipped_budget")]
    ncc_peaks = [float(r["ncc_lock"]["ncc_peak"]) for r in records if "ncc_lock" in r]
    null_margins = [float(r["ncc_lock"]["null_margin"]) for r in records if "ncc_lock" in r]
    ghost_dom = sum(1 for r in evaluated if r.get("ghosting_check", {}).get("is_ghost_dominated"))
    return {
        "tiles_planned": len(records),
        "tiles_evaluated": len(evaluated),
        "tiles_skipped_budget": len(skipped),
        "tiles_ncc_locked": len(locked),
        "ncc_peak_min": round(min(ncc_peaks), 6) if ncc_peaks else None,
        "ncc_peak_median": round(float(np.median(ncc_peaks)), 6) if ncc_peaks else None,
        "ncc_peak_max": round(max(ncc_peaks), 6) if ncc_peaks else None,
        "null_margin_min": round(min(null_margins), 6) if null_margins else None,
        "max_z_p99_9_abs": round(max((float(r["z_p99_9_abs"]) for r in evaluated), default=0.0), 4),
        "max_z_abs": round(max((float(r["z_max_abs"]) for r in evaluated), default=0.0), 4),
        "max_frac_ge_3sigma": round(max((float(r["frac_ge_3sigma"]) for r in evaluated), default=0.0), 6),
        "max_frac_ge_4sigma": round(max((float(r["frac_ge_4sigma"]) for r in evaluated), default=0.0), 6),
        "max_4sigma_component_span_mm": round(
            max((float(r["max_4sigma_component_span_mm"]) for r in evaluated), default=0.0), 3
        ),
        "ghost_dominated_count": int(ghost_dom),
        "any_stopping_rule_trigger": any(bool(r.get("exceeds_stopping_rule")) for r in evaluated),
    }


def _compute_component_stats(
    z_map: np.ndarray,
    valid_2d: np.ndarray,
    *,
    threshold_z: float,
    uv_spacing_um: float,
) -> tuple[np.ndarray, int, float]:
    """Find largest connected component exceeding ``|z| >= threshold_z`` and its physical span in mm."""
    hi_mask = valid_2d & (np.abs(z_map) >= float(threshold_z))
    lbl, n_comp = ndi.label(hi_mask)
    max_comp_px = 0
    max_comp_span_mm = 0.0
    if n_comp > 0:
        sizes = np.bincount(lbl.ravel())[1:]
        top_cid = int(np.argmax(sizes)) + 1
        max_comp_px = int(sizes[top_cid - 1])
        ys, xs = np.nonzero(lbl == top_cid)
        max_comp_span_mm = float(
            math.hypot((ys.max() - ys.min() + 1) * uv_spacing_um, (xs.max() - xs.min() + 1) * uv_spacing_um) / 1000.0
        )
    return hi_mask, max_comp_px, max_comp_span_mm


def run_phase2_scans(
    *,
    vol_78_url: str = DEFAULT_URLS["vol_78"],
    vol_54_url: str = DEFAULT_URLS["vol_54"],
    vol_1129_url: str = DEFAULT_URLS["vol_1129"],
    transform_78_url: str = DEFAULT_URLS["transform_78_to_dls"],
    transform_1129_url: str = DEFAULT_URLS["transform_1129_to_78_r2"],
    mesh_78_url: str = DEFAULT_URLS["mesh_78"],
    mesh_outer_url: str = DEFAULT_URLS["mesh_outer_dls"],
    preflight_level_78: int = 2,
    preflight_level_54: int = 0,
    strip_level_78: int = 3,
    strip_level_54: int = 1,
    crust_level_1129: int = 0,
    crust_level_78: int = 1,
    cache_mb: float = 256.0,
    workers: int = 8,
    max_total_mb: float = 225.0,
    stage_2a1_budget_mb: float = 88.0,
    stage_2b1_budget_mb: float = 40.0,
    stage_2b2_budget_mb: float = 24.0,
    stage_2b3_budget_mb: float = 34.0,
    out_json: str | os.PathLike[str] | None = None,
    private_out_dir: str | os.PathLike[str] | None = None,
    preflight_tile_a_vu: tuple[int, int, int, int] = (640, 656, 4082, 4114),
    preflight_tile_b_vu: tuple[int, int, int, int] = (640, 656, 5109, 5141),
    strip_v_range: tuple[int, int] = (1956, 2000),
    strip_wraps: Sequence[tuple[str, int, int, str]] = STEP1_WRAP_ISLANDS,
    outer_v_range: tuple[int, int] = (500, 510),
    outer_wraps: Sequence[tuple[str, int, int]] = PHASE2_OUTER_WRAPS,
    upper_ext_vu: tuple[int, int, int, int] = (112, 126, 5112, 5132),
    lower_ext_vu: tuple[int, int, int, int] = (3182, 3196, 5112, 5132),
    crust_tile_a_vu: tuple[int, int, int, int] = (1100, 1103, 4092, 4096),
    crust_tile_b_vu: tuple[int, int, int, int] = (1100, 1103, 5122, 5126),
    run_strip: bool = True,
    run_outer_wraps: bool = True,
    run_extended_height: bool = True,
    run_crust_1129: bool = True,
) -> dict[str, Any]:
    """Execute Phase 2 non-overlapping Paris 4 dual-energy CPU scans strictly within < 250 MB.

    Covers:
      (2A-0) 54 vs 78 keV PSF-matched high-res control preflight outside the 137 keV band;
      (2A-1) 54 vs 78 keV PSF-matched multi-wrap agraphon strip on Band L (non-overlapping);
      (2B-1) Outermost wraps w130-w135 (54 vs 78 keV + sheet integrity);
      (2B-2) 20260701 extended axial height (upper z_78~71k & lower z_78~16k);
      (2B-3) 1.129 um native L0 crust preflight via r2 affine+TPS + local 3D NCC lock.
    """
    if float(cache_mb) > 256.0:
        raise ValueError(f"cache_mb ({cache_mb}) exceeds strict 256.0 MiB RAM cap.")
    if float(max_total_mb) > 238.0:
        raise ValueError(f"max_total_mb ({max_total_mb} MiB) exceeds strict < 250 MB ceiling.")

    if private_out_dir is not None:
        priv_path = Path(private_out_dir)
        _assert_outside_git_repo(priv_path)
        priv_path.mkdir(parents=True, exist_ok=True)
    else:
        priv_path = None

    sess = requests.Session() if requests is not None else None
    w78_lo, w78_hi, _ = load_volume_window_f32(vol_78_url, fallback=DEFAULT_WINDOW_F32["78keV"], session=sess)
    w54_lo, w54_hi, _ = load_volume_window_f32(vol_54_url, fallback=DEFAULT_WINDOW_F32["54keV"], session=sess)
    m78_to_dls = load_affine_transform(transform_78_url, session=sess)
    m_dls_to_78 = np.linalg.inv(m78_to_dls)

    # Track exact bytes & chunks across every stage.
    stage_io: dict[str, dict[str, int | float]] = {}
    total_bytes = 0
    total_chunks = 0

    def _remaining_mb() -> float:
        return float(max_total_mb) - (total_bytes / (1024.0 * 1024.0))

    # ------------------------------------------------------------------
    # Stage 2A-0: 54 vs 78 keV PSF-matched high-res control preflight
    # ------------------------------------------------------------------
    vox_78_pf = DEFAULT_VOXEL_UM["78keV"] * (2 ** int(preflight_level_78))
    vox_54_pf = DEFAULT_VOXEL_UM["54keV"] * (2 ** int(preflight_level_54))
    psf_pf_um = max(vox_78_pf, vox_54_pf)
    uv_pf_um = psf_pf_um
    n_step_pf = min(8.0, uv_pf_um)
    n_offs_pf = np.arange(-12, 13, dtype=np.float64) * n_step_pf  # -96..+96 um for 78->DLS drift margin
    ghost_offs = np.array([-160.0, -110.0, 0.0, 110.0, 160.0], dtype=np.float64)

    with (
        TifxyzPatchReader(mesh_78_url) as mesh_78,
        RemoteZarrV2(vol_78_url, level=str(preflight_level_78), cache_mb=cache_mb, workers=workers) as r78_pf,
        RemoteZarrV2(vol_54_url, level=str(preflight_level_54), cache_mb=cache_mb, workers=workers) as r54_pf,
    ):
        mesh_b0 = int(mesh_78.bytes_fetched)
        tile_b = evaluate_paired_tile(
            mesh_78,
            r78_pf,
            r54_pf,
            m78_to_dls,
            v0=preflight_tile_b_vu[0],
            v1=preflight_tile_b_vu[1],
            u0=preflight_tile_b_vu[2],
            u1=preflight_tile_b_vu[3],
            ref_voxel_um=DEFAULT_VOXEL_UM["78keV"],
            tgt_voxel_um=DEFAULT_VOXEL_UM["54keV"],
            ref_level=preflight_level_78,
            tgt_level=preflight_level_54,
            ref_window=(w78_lo, w78_hi),
            tgt_window=(w54_lo, w54_hi),
            uv_spacing_um=uv_pf_um,
            n_offsets_um=n_offs_pf,
            ghost_offsets_um=ghost_offs,
            max_shift_um=70.0,
            null_offset_um=20.0,
            min_ncc=0.10,
            psf_match_to_um=psf_pf_um,
        )
        tile_a = evaluate_paired_tile(
            mesh_78,
            r78_pf,
            r54_pf,
            m78_to_dls,
            v0=preflight_tile_a_vu[0],
            v1=preflight_tile_a_vu[1],
            u0=preflight_tile_a_vu[2],
            u1=preflight_tile_a_vu[3],
            ref_voxel_um=DEFAULT_VOXEL_UM["78keV"],
            tgt_voxel_um=DEFAULT_VOXEL_UM["54keV"],
            ref_level=preflight_level_78,
            tgt_level=preflight_level_54,
            ref_window=(w78_lo, w78_hi),
            tgt_window=(w54_lo, w54_hi),
            uv_spacing_um=uv_pf_um,
            n_offsets_um=n_offs_pf,
            ghost_offsets_um=ghost_offs,
            max_shift_um=70.0,
            null_offset_um=20.0,
            min_ncc=0.10,
            psf_match_to_um=psf_pf_um,
        )
        b_2a0 = int(r78_pf.bytes_fetched + r54_pf.bytes_fetched + mesh_78.bytes_fetched)
        c_2a0 = int(r78_pf.chunks_fetched + r54_pf.chunks_fetched)
        stage_io["stage_2a0_preflight_54_78"] = {
            "bytes": b_2a0,
            "mb": round(b_2a0 / (1024.0 * 1024.0), 3),
            "chunks": c_2a0,
            "chunks_78": int(r78_pf.chunks_fetched),
            "chunks_54": int(r54_pf.chunks_fetched),
        }
        total_bytes += b_2a0
        total_chunks += c_2a0

        bg_pf = fit_papyrus_background(tile_b["mu_ref"], tile_b["mu_tgt"], tile_b["joint_valid"])
        alpha_pf, beta_pf = float(bg_pf["alpha"]), float(bg_pf["beta"])
        res_b = compute_dual_energy_residual(tile_b["mu_ref"], tile_b["mu_tgt"], alpha_pf, beta_pf)
        res_a = compute_dual_energy_residual(tile_a["mu_ref"], tile_a["mu_tgt"], alpha_pf, beta_pf)

        th_b, surf_b, _, v2d_b = compute_depth_tophat(res_b, tile_b["joint_valid"], n_offs_pf)
        th_a, surf_a, _, v2d_a = compute_depth_tophat(res_a, tile_a["joint_valid"], n_offs_pf)
        th54_b, _, _, v54_b = compute_depth_tophat(tile_b["mu_tgt"], tile_b["joint_valid"], n_offs_pf)
        th54_a, _, _, v54_a = compute_depth_tophat(tile_a["mu_tgt"], tile_a["joint_valid"], n_offs_pf)

        blank_pf = compute_blank_fpr_stats(th_b, v2d_b)
        sep_tophat_r54 = compute_control_separation(th_a[v2d_a], th_b[v2d_b])
        sep_surf_r54 = compute_control_separation(surf_a[v2d_a], surf_b[v2d_b])
        sep_tophat_54 = compute_control_separation(th54_a[v54_a], th54_b[v54_b])

        ghost_a = evaluate_ghosting_exclusion(
            compute_dual_energy_residual(tile_a["ghost_data"]["mu_ref"], tile_a["ghost_data"]["mu_tgt"], alpha_pf, beta_pf),
            tile_a["ghost_data"]["valid"],
            tile_a["ghost_data"]["offsets_um"],
        )
        ghost_b = evaluate_ghosting_exclusion(
            compute_dual_energy_residual(tile_b["ghost_data"]["mu_ref"], tile_b["ghost_data"]["mu_tgt"], alpha_pf, beta_pf),
            tile_b["ghost_data"]["valid"],
            tile_b["ghost_data"]["offsets_um"],
        )

        lock_a: NccLockResult = tile_a["lock"]
        lock_b: NccLockResult = tile_b["lock"]
        pf_passed = bool(lock_a.passed and lock_b.passed)

        if priv_path is not None:
            sig_pf = max(float(blank_pf["sigma_blank"]), 1e-6)
            mu_pf_b = float(blank_pf["mean_blank"])
            _save_private_heatmap_png((th_a - mu_pf_b) / sig_pf, v2d_a, priv_path / "phase2_2a0_ctrl_a_r54_78_zmap.png", -4.0, 4.0)
            _save_private_heatmap_png((th_b - mu_pf_b) / sig_pf, v2d_b, priv_path / "phase2_2a0_ctrl_b_r54_78_zmap.png", -4.0, 4.0)

        # ------------------------------------------------------------------
        # Stage 2B-2 lower/upper extended-height 78 keV L2 integrity (reuses r78_pf!)
        # ------------------------------------------------------------------
        ext_height_private: dict[str, Any] = {"executed": False}
        ext_height_public: dict[str, Any] = {"executed": False}
        b_ext_l2 = 0
        c_ext_l2 = 0
        if run_extended_height and _remaining_mb() > 10.0:
            b78_before = int(r78_pf.bytes_fetched)
            c78_before = int(r78_pf.chunks_fetched)
            m_before = int(mesh_78.bytes_fetched)
            upper_int = evaluate_sheet_integrity_tile(
                mesh_78,
                r78_pf,
                v0=upper_ext_vu[0],
                v1=upper_ext_vu[1],
                u0=upper_ext_vu[2],
                u1=upper_ext_vu[3],
                level=preflight_level_78,
                window_78=(w78_lo, w78_hi),
                uv_spacing_um=vox_78_pf,
            )
            lower_int = evaluate_sheet_integrity_tile(
                mesh_78,
                r78_pf,
                v0=lower_ext_vu[0],
                v1=lower_ext_vu[1],
                u0=lower_ext_vu[2],
                u1=lower_ext_vu[3],
                level=preflight_level_78,
                window_78=(w78_lo, w78_hi),
                uv_spacing_um=vox_78_pf,
            )
            b_ext_l2 = int((r78_pf.bytes_fetched - b78_before) + (mesh_78.bytes_fetched - m_before))
            c_ext_l2 = int(r78_pf.chunks_fetched - c78_before)
            total_bytes += b_ext_l2
            total_chunks += c_ext_l2
            ext_height_private = {
                "executed": True,
                "upper_extension_integrity": upper_int,
                "lower_extension_integrity": lower_int,
            }
            ext_height_public = {
                "executed": True,
                "upper_extension_integrity": {
                    k: v for k, v in upper_int.items() if k not in FORBIDDEN_UNREAD_COORD_KEYS
                },
                "lower_extension_integrity": {
                    k: v for k, v in lower_int.items() if k not in FORBIDDEN_UNREAD_COORD_KEYS
                },
            }

        # ------------------------------------------------------------------
        # Stages 2A-1 (Band L agraphon strip) + 2B-1 (outer wraps w130-w135)
        # + Upper-extension 54 vs 78 keV paired check (shared L3/L1 readers!)
        # ------------------------------------------------------------------
        vox_78_st = DEFAULT_VOXEL_UM["78keV"] * (2 ** int(strip_level_78))
        vox_54_st = DEFAULT_VOXEL_UM["54keV"] * (2 ** int(strip_level_54))
        psf_st_um = max(vox_78_st, vox_54_st)
        uv_st_um = psf_st_um
        n_offs_st = np.arange(-10, 11, dtype=np.float64) * 10.0  # -100..+100 um

        strip_records: list[dict[str, Any]] = []
        outer_records: list[dict[str, Any]] = []
        strip_bg_fit: dict[str, Any] | None = None
        alpha_st, beta_st = alpha_pf, beta_pf
        mu_st_b, sigma_st_b, p999_st_b = float(blank_pf["mean_blank"]), max(float(blank_pf["sigma_blank"]), 1e-6), float(blank_pf["p99_9_abs_z"])

        with ExitStack() as st_stack:
            r78_st = (
                st_stack.enter_context(
                    RemoteZarrV2(vol_78_url, level=str(strip_level_78), cache_mb=cache_mb, workers=workers)
                )
                if (run_strip or run_outer_wraps or run_extended_height)
                else None
            )
            r54_st = (
                st_stack.enter_context(
                    RemoteZarrV2(vol_54_url, level=str(strip_level_54), cache_mb=cache_mb, workers=workers)
                )
                if (run_strip or run_outer_wraps or run_extended_height)
                else None
            )
            # --- Stage 2A-1: 54 vs 78 keV non-overlapping Band L agraphon strip ---
            if run_strip and pf_passed and _remaining_mb() > 16.0:
                b78_s0, b54_s0, m_s0 = int(r78_st.bytes_fetched), int(r54_st.bytes_fetched), int(mesh_78.bytes_fetched)
                c78_s0, c54_s0 = int(r78_st.chunks_fetched), int(r54_st.chunks_fetched)
                eff_2a1_cap = min(float(stage_2a1_budget_mb), _remaining_mb() - 4.0)

                blank_wrap_name = next(
                    (w[0] for w in strip_wraps if w[0] == "wrap_08_blank_agraphon"),
                    next((w[0] for w in strip_wraps if w[3] == "control_blank"), strip_wraps[0][0] if strip_wraps else ""),
                )
                ordered_wraps = [w for w in strip_wraps if w[0] == blank_wrap_name] + [
                    w for w in strip_wraps if w[0] != blank_wrap_name
                ]
                for name, u0, u1, role in ordered_wraps:
                    used_2a1 = (
                        (r78_st.bytes_fetched - b78_s0)
                        + (r54_st.bytes_fetched - b54_s0)
                        + (mesh_78.bytes_fetched - m_s0)
                    ) / (1024.0 * 1024.0)
                    est_extra = _estimate_extra_chunks_mb(
                        mesh_78,
                        [
                            (r78_st, strip_level_78, None),
                            (r54_st, strip_level_54, lambda p: apply_affine_points(p, m78_to_dls)),
                        ],
                        strip_v_range[0],
                        strip_v_range[1],
                        u0,
                        u1,
                        v_stride=16,
                    )
                    if used_2a1 + est_extra > eff_2a1_cap and name != blank_wrap_name:
                        strip_records.append(
                            {
                                "wrap": name,
                                "role": role,
                                "skipped_budget": True,
                                "est_extra_mb": round(est_extra, 2),
                            }
                        )
                        continue

                    tile = evaluate_paired_tile(
                        mesh_78,
                        r78_st,
                        r54_st,
                        m78_to_dls,
                        v0=strip_v_range[0],
                        v1=strip_v_range[1],
                        u0=u0,
                        u1=u1,
                        ref_voxel_um=DEFAULT_VOXEL_UM["78keV"],
                        tgt_voxel_um=DEFAULT_VOXEL_UM["54keV"],
                        ref_level=strip_level_78,
                        tgt_level=strip_level_54,
                        ref_window=(w78_lo, w78_hi),
                        tgt_window=(w54_lo, w54_hi),
                        uv_spacing_um=uv_st_um,
                        n_offsets_um=n_offs_st,
                        ghost_offsets_um=ghost_offs,
                        max_shift_um=70.0,
                        null_offset_um=20.0,
                        min_ncc=0.10,
                        psf_match_to_um=psf_st_um,
                        v_stride=2,
                    )
                    lock: NccLockResult = tile["lock"]
                    if name == blank_wrap_name and lock.passed:
                        strip_bg_fit = fit_papyrus_background(tile["mu_ref"], tile["mu_tgt"], tile["joint_valid"])
                        alpha_st = float(strip_bg_fit["alpha"])
                        beta_st = float(strip_bg_fit["beta"])

                    if not lock.passed:
                        strip_records.append(
                            {
                                "wrap": name,
                                "role": role,
                                "u_range": [u0, u1],
                                "z78_range": tile["z78_range"],
                                "ncc_lock": lock.to_dict(),
                                "residual_computed": False,
                            }
                        )
                        continue

                    res = compute_dual_energy_residual(tile["mu_ref"], tile["mu_tgt"], alpha_st, beta_st)
                    th, surf, _, v2d = compute_depth_tophat(res, tile["joint_valid"], n_offs_st)
                    if name == blank_wrap_name:
                        bs = compute_blank_fpr_stats(th, v2d)
                        mu_st_b = float(bs["mean_blank"])
                        sigma_st_b = max(float(bs["sigma_blank"]), 1e-6)
                        p999_st_b = float(bs["p99_9_abs_z"])

                    z_map = np.where(v2d, (th - mu_st_b) / sigma_st_b, 0.0).astype(np.float32)
                    vals_z = z_map[v2d]
                    abs_z = np.abs(vals_z)
                    hi_mask, max_comp_px, max_comp_span_mm = _compute_component_stats(
                        z_map, v2d, threshold_z=max(4.0, p999_st_b), uv_spacing_um=uv_st_um
                    )
                    ghost = evaluate_ghosting_exclusion(
                        compute_dual_energy_residual(tile["ghost_data"]["mu_ref"], tile["ghost_data"]["mu_tgt"], alpha_st, beta_st),
                        tile["ghost_data"]["valid"],
                        tile["ghost_data"]["offsets_um"],
                        candidate_mask_2d=hi_mask if hi_mask.sum() >= 8 else None,
                    )
                    if priv_path is not None:
                        np.savez_compressed(
                            priv_path / f"phase2_2a1_{name}.npz",
                            z_map=z_map,
                            tophat_R=th,
                            surface_R=surf,
                            valid_2d=v2d,
                        )
                        _save_private_heatmap_png(z_map, v2d, priv_path / f"phase2_2a1_{name}_zmap.png", -4.5, 4.5)

                    strip_records.append(
                        {
                            "wrap": name,
                            "role": role,
                            "u_range": [u0, u1],
                            "z78_range": tile["z78_range"],
                            "slab_shape_nvu": tile["slab_shape_nvu"],
                            "valid_pixels": int(v2d.sum()),
                            "ncc_lock": lock.to_dict(),
                            "residual_computed": True,
                            "z_mean": round(float(vals_z.mean()), 4) if vals_z.size else 0.0,
                            "z_std": round(float(vals_z.std()), 4) if vals_z.size else 0.0,
                            "z_p99_9_abs": round(float(np.percentile(abs_z, 99.9)), 4) if vals_z.size else 0.0,
                            "z_max_abs": round(float(abs_z.max()), 4) if vals_z.size else 0.0,
                            "frac_ge_3sigma": round(float(np.mean(abs_z >= 3.0)), 6) if vals_z.size else 0.0,
                            "frac_ge_4sigma": round(float(np.mean(abs_z >= 4.0)), 6) if vals_z.size else 0.0,
                            "max_4sigma_component_px": max_comp_px,
                            "max_4sigma_component_span_mm": round(max_comp_span_mm, 3),
                            "ghosting_check": ghost,
                            "exceeds_stopping_rule": bool(
                                (max_comp_span_mm >= 15.0) and (not ghost.get("is_ghost_dominated", False))
                            ),
                        }
                    )

                b_2a1 = int(
                    (r78_st.bytes_fetched - b78_s0)
                    + (r54_st.bytes_fetched - b54_s0)
                    + (mesh_78.bytes_fetched - m_s0)
                )
                c_2a1 = int((r78_st.chunks_fetched - c78_s0) + (r54_st.chunks_fetched - c54_s0))
                stage_io["stage_2a1_agraphon_strip_54_78"] = {
                    "bytes": b_2a1,
                    "mb": round(b_2a1 / (1024.0 * 1024.0), 3),
                    "chunks": c_2a1,
                }
                total_bytes += b_2a1
                total_chunks += c_2a1

            # --- Stage 2B-2 upper-extension 54 vs 78 keV paired check ---
            if run_extended_height and _remaining_mb() > 10.0:
                b78_u0, b54_u0, m_u0 = int(r78_st.bytes_fetched), int(r54_st.bytes_fetched), int(mesh_78.bytes_fetched)
                c78_u0, c54_u0 = int(r78_st.chunks_fetched), int(r54_st.chunks_fetched)
                est_u = _estimate_extra_chunks_mb(
                    mesh_78,
                    [
                        (r78_st, strip_level_78, None),
                        (r54_st, strip_level_54, lambda p: apply_affine_points(p, m78_to_dls)),
                    ],
                    upper_ext_vu[0],
                    upper_ext_vu[1],
                    upper_ext_vu[2],
                    upper_ext_vu[3],
                    v_stride=4,
                )
                if est_u <= min(float(stage_2b2_budget_mb), _remaining_mb()):
                    tile_u = evaluate_paired_tile(
                        mesh_78,
                        r78_st,
                        r54_st,
                        m78_to_dls,
                        v0=upper_ext_vu[0],
                        v1=upper_ext_vu[1],
                        u0=upper_ext_vu[2],
                        u1=upper_ext_vu[3],
                        ref_voxel_um=DEFAULT_VOXEL_UM["78keV"],
                        tgt_voxel_um=DEFAULT_VOXEL_UM["54keV"],
                        ref_level=strip_level_78,
                        tgt_level=strip_level_54,
                        ref_window=(w78_lo, w78_hi),
                        tgt_window=(w54_lo, w54_hi),
                        uv_spacing_um=uv_st_um,
                        n_offsets_um=n_offs_st,
                        ghost_offsets_um=ghost_offs,
                        max_shift_um=70.0,
                        null_offset_um=20.0,
                        min_ncc=0.10,
                        psf_match_to_um=psf_st_um,
                    )
                    lock_u: NccLockResult = tile_u["lock"]
                    if lock_u.passed:
                        res_u = compute_dual_energy_residual(tile_u["mu_ref"], tile_u["mu_tgt"], alpha_st, beta_st)
                        th_u, surf_u, _, v2d_u = compute_depth_tophat(res_u, tile_u["joint_valid"], n_offs_st)
                        z_u = np.where(v2d_u, (th_u - mu_st_b) / sigma_st_b, 0.0).astype(np.float32)
                        vz_u = z_u[v2d_u]
                        abs_u = np.abs(vz_u)
                        hi_u, px_u, span_u = _compute_component_stats(
                            z_u, v2d_u, threshold_z=max(4.0, p999_st_b), uv_spacing_um=uv_st_um
                        )
                        ghost_u = evaluate_ghosting_exclusion(
                            compute_dual_energy_residual(
                                tile_u["ghost_data"]["mu_ref"], tile_u["ghost_data"]["mu_tgt"], alpha_st, beta_st
                            ),
                            tile_u["ghost_data"]["valid"],
                            tile_u["ghost_data"]["offsets_um"],
                            candidate_mask_2d=hi_u if hi_u.sum() >= 8 else None,
                        )
                        if priv_path is not None:
                            _save_private_heatmap_png(z_u, v2d_u, priv_path / "phase2_2b2_upper_ext_r54_78_zmap.png", -4.5, 4.5)
                        upper_dual = {
                            "ncc_lock": lock_u.to_dict(),
                            "valid_pixels": int(v2d_u.sum()),
                            "z_p99_9_abs": round(float(np.percentile(abs_u, 99.9)), 4) if abs_u.size else 0.0,
                            "z_max_abs": round(float(abs_u.max()), 4) if abs_u.size else 0.0,
                            "frac_ge_3sigma": round(float(np.mean(abs_u >= 3.0)), 6) if abs_u.size else 0.0,
                            "frac_ge_4sigma": round(float(np.mean(abs_u >= 4.0)), 6) if abs_u.size else 0.0,
                            "max_4sigma_component_span_mm": round(span_u, 3),
                            "ghosting_check": ghost_u,
                            "exceeds_stopping_rule": bool(
                                (span_u >= 15.0) and (not ghost_u.get("is_ghost_dominated", False))
                            ),
                        }
                    else:
                        upper_dual = {"ncc_lock": lock_u.to_dict(), "residual_computed": False}
                    ext_height_private["upper_extension_54_78"] = {
                        **upper_dual,
                        "tile_bounds_vu": tile_u["tile_bounds_vu"],
                        "z78_range": tile_u["z78_range"],
                    }
                    ext_height_public["upper_extension_54_78"] = upper_dual

                b_ext_u = int(
                    (r78_st.bytes_fetched - b78_u0)
                    + (r54_st.bytes_fetched - b54_u0)
                    + (mesh_78.bytes_fetched - m_u0)
                )
                c_ext_u = int((r78_st.chunks_fetched - c78_u0) + (r54_st.chunks_fetched - c54_u0))
                total_bytes += b_ext_u
                total_chunks += c_ext_u
                stage_io["stage_2b2_extended_height"] = {
                    "bytes": b_ext_l2 + b_ext_u,
                    "mb": round((b_ext_l2 + b_ext_u) / (1024.0 * 1024.0), 3),
                    "chunks": c_ext_l2 + c_ext_u,
                }

            # --- Stage 2B-1: Outer wraps w130-w135 (54 vs 78 keV + sheet integrity) ---
            if run_outer_wraps and _remaining_mb() > 12.0:
                b78_o0, b54_o0 = int(r78_st.bytes_fetched), int(r54_st.bytes_fetched)
                c78_o0, c54_o0 = int(r78_st.chunks_fetched), int(r54_st.chunks_fetched)
                eff_2b1_cap = min(float(stage_2b1_budget_mb), _remaining_mb() - 4.0)

                with TifxyzPatchReader(mesh_outer_url) as mesh_outer:
                    for wname, u0, u1 in outer_wraps:
                        used_2b1 = (
                            (r78_st.bytes_fetched - b78_o0)
                            + (r54_st.bytes_fetched - b54_o0)
                            + mesh_outer.bytes_fetched
                        ) / (1024.0 * 1024.0)
                        est_o = _estimate_extra_chunks_mb(
                            mesh_outer,
                            [
                                (r54_st, strip_level_54, None),
                                (r78_st, strip_level_78, lambda p: apply_affine_points(p, m_dls_to_78)),
                            ],
                            outer_v_range[0],
                            outer_v_range[1],
                            u0,
                            u1,
                            v_stride=4,
                        )
                        if used_2b1 + est_o > eff_2b1_cap:
                            outer_records.append(
                                {
                                    "wrap": wname,
                                    "skipped_budget": True,
                                    "est_extra_mb": round(est_o, 2),
                                }
                            )
                            continue

                        # Mesh is in 54 keV DLS coordinates -> ref=54 keV, tgt=78 keV.
                        tile_o = evaluate_paired_tile(
                            mesh_outer,
                            r54_st,
                            r78_st,
                            m_dls_to_78,
                            v0=outer_v_range[0],
                            v1=outer_v_range[1],
                            u0=u0,
                            u1=u1,
                            ref_voxel_um=DEFAULT_VOXEL_UM["54keV"],
                            tgt_voxel_um=DEFAULT_VOXEL_UM["78keV"],
                            ref_level=strip_level_54,
                            tgt_level=strip_level_78,
                            ref_window=(w54_lo, w54_hi),
                            tgt_window=(w78_lo, w78_hi),
                            uv_spacing_um=uv_st_um,
                            n_offsets_um=n_offs_st,
                            ghost_offsets_um=ghost_offs,
                            max_shift_um=70.0,
                            null_offset_um=20.0,
                            min_ncc=0.10,
                            psf_match_to_um=psf_st_um,
                        )
                        # Compute sheet integrity on the already-fetched 54 keV L1 chunks (0 extra CT bytes!).
                        integ_o = evaluate_sheet_integrity_tile(
                            mesh_outer,
                            r54_st,
                            v0=outer_v_range[0],
                            v1=outer_v_range[1],
                            u0=u0,
                            u1=u1,
                            level=strip_level_54,
                            window_78=(w54_lo, w54_hi),
                            uv_spacing_um=vox_54_st,
                            voxel_um=DEFAULT_VOXEL_UM["54keV"],
                        )
                        lock_o: NccLockResult = tile_o["lock"]
                        if not lock_o.passed:
                            outer_records.append(
                                {
                                    "wrap": wname,
                                    "u_range": [u0, u1],
                                    "z_dls_range": tile_o["z78_range"],
                                    "ncc_lock": lock_o.to_dict(),
                                    "sheet_integrity": integ_o,
                                    "residual_computed": False,
                                }
                            )
                            continue

                        # Keep exact R_{54/78} = mu_54 - (alpha * mu_78 + beta) sign convention!
                        res_o = compute_dual_energy_residual(tile_o["mu_tgt"], tile_o["mu_ref"], alpha_st, beta_st)
                        th_o, surf_o, _, v2d_o = compute_depth_tophat(res_o, tile_o["joint_valid"], n_offs_st)
                        z_o = np.where(v2d_o, (th_o - mu_st_b) / sigma_st_b, 0.0).astype(np.float32)
                        vz_o = z_o[v2d_o]
                        abs_o = np.abs(vz_o)
                        hi_o, px_o, span_o = _compute_component_stats(
                            z_o, v2d_o, threshold_z=max(4.0, p999_st_b), uv_spacing_um=uv_st_um
                        )
                        ghost_o = evaluate_ghosting_exclusion(
                            compute_dual_energy_residual(
                                tile_o["ghost_data"]["mu_tgt"], tile_o["ghost_data"]["mu_ref"], alpha_st, beta_st
                            ),
                            tile_o["ghost_data"]["valid"],
                            tile_o["ghost_data"]["offsets_um"],
                            candidate_mask_2d=hi_o if hi_o.sum() >= 8 else None,
                        )
                        if priv_path is not None:
                            np.savez_compressed(
                                priv_path / f"phase2_2b1_outer_{wname}.npz",
                                z_map=z_o,
                                tophat_R=th_o,
                                surface_R=surf_o,
                                valid_2d=v2d_o,
                            )
                            _save_private_heatmap_png(z_o, v2d_o, priv_path / f"phase2_2b1_outer_{wname}_zmap.png", -4.5, 4.5)

                        outer_records.append(
                            {
                                "wrap": wname,
                                "u_range": [u0, u1],
                                "z_dls_range": tile_o["z78_range"],
                                "slab_shape_nvu": tile_o["slab_shape_nvu"],
                                "valid_pixels": int(v2d_o.sum()),
                                "ncc_lock": lock_o.to_dict(),
                                "sheet_integrity": integ_o,
                                "residual_computed": True,
                                "z_mean": round(float(vz_o.mean()), 4) if vz_o.size else 0.0,
                                "z_std": round(float(vz_o.std()), 4) if vz_o.size else 0.0,
                                "z_p99_9_abs": round(float(np.percentile(abs_o, 99.9)), 4) if abs_o.size else 0.0,
                                "z_max_abs": round(float(abs_o.max()), 4) if abs_o.size else 0.0,
                                "frac_ge_3sigma": round(float(np.mean(abs_o >= 3.0)), 6) if abs_o.size else 0.0,
                                "frac_ge_4sigma": round(float(np.mean(abs_o >= 4.0)), 6) if abs_o.size else 0.0,
                                "max_4sigma_component_px": px_o,
                                "max_4sigma_component_span_mm": round(span_o, 3),
                                "ghosting_check": ghost_o,
                                "exceeds_stopping_rule": bool(
                                    (span_o >= 15.0) and (not ghost_o.get("is_ghost_dominated", False))
                                ),
                            }
                        )

                    b_2b1 = int(
                        (r78_st.bytes_fetched - b78_o0)
                        + (r54_st.bytes_fetched - b54_o0)
                        + mesh_outer.bytes_fetched
                    )
                    c_2b1 = int((r78_st.chunks_fetched - c78_o0) + (r54_st.chunks_fetched - c54_o0))
                    stage_io["stage_2b1_outer_wraps_w130_w135"] = {
                        "bytes": b_2b1,
                        "mb": round(b_2b1 / (1024.0 * 1024.0), 3),
                        "chunks": c_2b1,
                    }
                    total_bytes += b_2b1
                    total_chunks += c_2b1

        # ------------------------------------------------------------------
        # Stage 2B-3: 1.129 um native L0 crust preflight (Control A & B)
        # ------------------------------------------------------------------
        crust_section: dict[str, Any] = {"executed": False}
        w1129_lo, w1129_hi = DEFAULT_WINDOW_F32["1129_78keV"]
        if run_crust_1129 and _remaining_mb() > 14.0:
            w1129_lo, w1129_hi, _ = load_volume_window_f32(
                vol_1129_url, fallback=DEFAULT_WINDOW_F32["1129_78keV"], session=sess
            )
            t_bundle = load_transform_bundle(transform_1129_url, session=sess)
            m_78_to_1129 = np.linalg.inv(t_bundle["matrix"])
            tps_inv = t_bundle.get("inverse_deformation")

            vox_78_cr = DEFAULT_VOXEL_UM["78keV"] * (2 ** int(crust_level_78))
            vox_1129_cr = DEFAULT_VOXEL_UM["1129_78keV"] * (2 ** int(crust_level_1129))
            psf_cr_um = max(vox_78_cr, vox_1129_cr)
            uv_cr_um = max(2.4, vox_1129_cr * 2.0)
            n_offs_cr = np.arange(-8, 9, dtype=np.float64) * 3.0  # -24..+24 um (3 um step)

            with (
                RemoteZarrV2(vol_78_url, level=str(crust_level_78), cache_mb=cache_mb, workers=workers) as r78_cr,
                RemoteZarrV2(vol_1129_url, level=str(crust_level_1129), cache_mb=cache_mb, workers=workers) as r1129_cr,
            ):
                m_c0 = int(mesh_78.bytes_fetched)
                est_cr = _estimate_extra_chunks_mb(
                    mesh_78,
                    [
                        (r78_cr, crust_level_78, None),
                        (
                            r1129_cr,
                            crust_level_1129,
                            lambda p: map_points_affine_and_tps(p, m_78_to_1129, tps_on_input=tps_inv),
                        ),
                    ],
                    crust_tile_a_vu[0],
                    crust_tile_a_vu[1],
                    crust_tile_a_vu[2],
                    crust_tile_a_vu[3],
                    v_stride=1,
                    pad_vx_lvl=6,
                ) + _estimate_extra_chunks_mb(
                    mesh_78,
                    [
                        (r78_cr, crust_level_78, None),
                        (
                            r1129_cr,
                            crust_level_1129,
                            lambda p: map_points_affine_and_tps(p, m_78_to_1129, tps_on_input=tps_inv),
                        ),
                    ],
                    crust_tile_b_vu[0],
                    crust_tile_b_vu[1],
                    crust_tile_b_vu[2],
                    crust_tile_b_vu[3],
                    v_stride=1,
                    pad_vx_lvl=6,
                )
                eff_2b3_cap = min(float(stage_2b3_budget_mb), _remaining_mb())
                if est_cr <= eff_2b3_cap + 14.0 and _remaining_mb() >= 18.0:
                    cr_b = evaluate_paired_tile(
                        mesh_78,
                        r78_cr,
                        r1129_cr,
                        m_78_to_1129,
                        v0=crust_tile_b_vu[0],
                        v1=crust_tile_b_vu[1],
                        u0=crust_tile_b_vu[2],
                        u1=crust_tile_b_vu[3],
                        ref_voxel_um=DEFAULT_VOXEL_UM["78keV"],
                        tgt_voxel_um=DEFAULT_VOXEL_UM["1129_78keV"],
                        ref_level=crust_level_78,
                        tgt_level=crust_level_1129,
                        ref_window=(w78_lo, w78_hi),
                        tgt_window=(w1129_lo, w1129_hi),
                        uv_spacing_um=uv_cr_um,
                        n_offsets_um=n_offs_cr,
                        max_shift_um=25.0,
                        null_offset_um=20.0,
                        min_ncc=0.10,
                        psf_match_to_um=psf_cr_um,
                        tps_on_ref=tps_inv,
                    )
                    cr_a = evaluate_paired_tile(
                        mesh_78,
                        r78_cr,
                        r1129_cr,
                        m_78_to_1129,
                        v0=crust_tile_a_vu[0],
                        v1=crust_tile_a_vu[1],
                        u0=crust_tile_a_vu[2],
                        u1=crust_tile_a_vu[3],
                        ref_voxel_um=DEFAULT_VOXEL_UM["78keV"],
                        tgt_voxel_um=DEFAULT_VOXEL_UM["1129_78keV"],
                        ref_level=crust_level_78,
                        tgt_level=crust_level_1129,
                        ref_window=(w78_lo, w78_hi),
                        tgt_window=(w1129_lo, w1129_hi),
                        uv_spacing_um=uv_cr_um,
                        n_offsets_um=n_offs_cr,
                        max_shift_um=25.0,
                        null_offset_um=20.0,
                        min_ncc=0.10,
                        psf_match_to_um=psf_cr_um,
                        tps_on_ref=tps_inv,
                    )
                    cm_b, th_cb, surf_cb, v_cb = compute_crust_metrics(
                        cr_b["mu_tgt_native"], cr_b["joint_valid"], n_offs_cr
                    )
                    cm_a, th_ca, surf_ca, v_ca = compute_crust_metrics(
                        cr_a["mu_tgt_native"], cr_a["joint_valid"], n_offs_cr
                    )
                    sep_crust_th = compute_control_separation(th_ca[v_ca], th_cb[v_cb])
                    sep_crust_surf = compute_control_separation(surf_ca[v_ca], surf_cb[v_cb])
                    blank_cr = compute_blank_fpr_stats(th_cb, v_cb)

                    if priv_path is not None:
                        sig_c = max(float(blank_cr["sigma_blank"]), 1e-6)
                        mu_c = float(blank_cr["mean_blank"])
                        _save_private_heatmap_png((th_ca - mu_c) / sig_c, v_ca, priv_path / "phase2_2b3_1129_ctrl_a_crust_zmap.png", -4.0, 4.0)
                        _save_private_heatmap_png((th_cb - mu_c) / sig_c, v_cb, priv_path / "phase2_2b3_1129_ctrl_b_crust_zmap.png", -4.0, 4.0)

                    b_2b3 = int(r78_cr.bytes_fetched + r1129_cr.bytes_fetched + (mesh_78.bytes_fetched - m_c0))
                    c_2b3 = int(r78_cr.chunks_fetched + r1129_cr.chunks_fetched)
                    stage_io["stage_2b3_crust_1129"] = {
                        "bytes": b_2b3,
                        "mb": round(b_2b3 / (1024.0 * 1024.0), 3),
                        "chunks": c_2b3,
                        "chunks_1129": int(r1129_cr.chunks_fetched),
                        "chunks_78": int(r78_cr.chunks_fetched),
                    }
                    total_bytes += b_2b3
                    total_chunks += c_2b3

                    crust_section = {
                        "executed": True,
                        "native_voxel_um": round(vox_1129_cr, 3),
                        "ref_78_voxel_um": round(vox_78_cr, 3),
                        "psf_matched_fwhm_um": round(psf_cr_um, 3),
                        "uv_spacing_um": round(uv_cr_um, 3),
                        "tps_deformation_applied": bool(tps_inv is not None),
                        "ncc_passed_both_controls": bool(cr_a["lock"].passed and cr_b["lock"].passed),
                        "control_a_vs_b_crust": {
                            "tophat_1129_native": sep_crust_th,
                            "surface_1129_native": sep_crust_surf,
                        },
                        "blank_crust_noise_stats": blank_cr,
                        "control_a_last_legible_column": {
                            "tile_bounds_vu": cr_a["tile_bounds_vu"],
                            "z78_range": cr_a["z78_range"],
                            "ncc_lock": cr_a["lock"].to_dict(),
                            "crust_metrics": cm_a,
                        },
                        "control_b_blank_agraphon": {
                            "tile_bounds_vu": cr_b["tile_bounds_vu"],
                            "z78_range": cr_b["z78_range"],
                            "ncc_lock": cr_b["lock"].to_dict(),
                            "crust_metrics": cm_b,
                        },
                    }
                else:
                    crust_section = {
                        "executed": False,
                        "skipped_reason": "budget_exceeded",
                        "est_mb": round(est_cr, 2),
                        "remaining_mb": round(_remaining_mb(), 2),
                    }

    outer_integ_vals = [r["sheet_integrity"] for r in outer_records if "sheet_integrity" in r]
    outer_public_summary: dict[str, Any] = {
        **_summarize_unread_records(outer_records),
        "sheet_integrity_aggregate": {
            "tiles_checked": len(outer_integ_vals),
            "valid_surface_fraction_min": round(
                min((float(x["valid_surface_fraction"]) for x in outer_integ_vals), default=0.0), 4
            ),
            "sheet_parallel_fraction_mean": round(
                float(np.mean([float(x["sheet_parallel_fraction"]) for x in outer_integ_vals]))
                if outer_integ_vals
                else 0.0,
                4,
            ),
            "mean_zero_shift_r_median": round(
                float(np.median([float(x["mean_zero_shift_r"]) for x in outer_integ_vals]))
                if outer_integ_vals
                else 0.0,
                4,
            ),
        },
    }

    strip_public_summary: dict[str, Any] = {
        "zarr_levels": {"vol_78": int(strip_level_78), "vol_54": int(strip_level_54)},
        "voxel_um": {"vol_78": round(vox_78_st, 3), "vol_54": round(vox_54_st, 3), "psf_matched": round(psf_st_um, 3)},
        "non_overlapping_with_137keV_band": True,
        "background_fit_control_b": (
            {k: (round(float(v), 6) if isinstance(v, float) else int(v)) for k, v in strip_bg_fit.items()}
            if strip_bg_fit is not None
            else None
        ),
        "blank_sigma": round(sigma_st_b, 6),
        "blank_p99_9_abs_z": round(p999_st_b, 4),
        **_summarize_unread_records(strip_records),
    }

    public_result: dict[str, Any] = {
        "schema_version": "2.0.0",
        "pipeline": "gemini_lane/paris4_dual_energy.py",
        "phase": "phase2_54_78_outer_wraps_extended_1129",
        "mode": "cpu_only_streaming",
        "sources": {
            "vol_78": _sanitize_source_label(vol_78_url),
            "vol_54": _sanitize_source_label(vol_54_url),
            "vol_1129": _sanitize_source_label(vol_1129_url),
            "transform_78_to_dls": _sanitize_source_label(transform_78_url),
            "transform_1129_to_78_r2": _sanitize_source_label(transform_1129_url),
            "mesh_78": _sanitize_source_label(mesh_78_url),
            "mesh_outer_dls": _sanitize_source_label(mesh_outer_url),
        },
        "radiometry": {
            "window_f32_78keV": [w78_lo, w78_hi],
            "window_f32_54keV": [w54_lo, w54_hi],
            "window_f32_1129_78keV": [w1129_lo, w1129_hi],
            "preflight_levels": {"vol_78": int(preflight_level_78), "vol_54": int(preflight_level_54)},
            "preflight_psf_matched_um": round(psf_pf_um, 3),
        },
        "preflight_54_78_passed": pf_passed,
        "ncc_peak_54_78": round(float(min(lock_a.ncc_peak, lock_b.ncc_peak)), 6),
        "ncc_6_nulls_54_78": {
            "tile_a": {k: round(float(v), 6) for k, v in lock_a.ncc_6_nulls.items()},
            "tile_b": {k: round(float(v), 6) for k, v in lock_b.ncc_6_nulls.items()},
        },
        "sigma_blank_54_78": blank_pf["sigma_blank"],
        "fpr_3sigma_54_78": blank_pf["fpr_3sigma"],
        "fpr_4sigma_54_78": blank_pf["fpr_4sigma"],
        "control_a_vs_b_54_78": {
            "tophat_R_54_78": sep_tophat_r54,
            "surface_R_54_78": sep_surf_r54,
            "tophat_mu_54": sep_tophat_54,
        },
        "papyrus_background_fit_control_b_54_78": {
            k: (round(float(v), 6) if isinstance(v, float) else int(v)) for k, v in bg_pf.items()
        },
        "blank_noise_stats_54_78": blank_pf,
        "controls_54_78": {
            "tile_a_last_legible_column": {
                "tile_bounds_vu": tile_a["tile_bounds_vu"],
                "z78_range": tile_a["z78_range"],
                "slab_shape_nvu": tile_a["slab_shape_nvu"],
                "joint_valid_fraction": tile_a["joint_valid_fraction"],
                "psf_matched_fwhm_um": tile_a["psf_matched_fwhm_um"],
                "ncc_lock": lock_a.to_dict(),
                "ghosting_check": ghost_a,
            },
            "tile_b_blank_agraphon": {
                "tile_bounds_vu": tile_b["tile_bounds_vu"],
                "z78_range": tile_b["z78_range"],
                "slab_shape_nvu": tile_b["slab_shape_nvu"],
                "joint_valid_fraction": tile_b["joint_valid_fraction"],
                "psf_matched_fwhm_um": tile_b["psf_matched_fwhm_um"],
                "ncc_lock": lock_b.to_dict(),
                "ghosting_check": ghost_b,
            },
        },
        "agraphon_strip_54_78": strip_public_summary,
        "outer_wraps_w130_w135": outer_public_summary,
        "extended_height_20260701": ext_height_public,
        "native_crust_1129_preflight": crust_section,
        "bytes_fetched": int(total_bytes),
        "chunks_fetched": int(total_chunks),
        "total_mib": round(total_bytes / (1024.0 * 1024.0), 3),
        "total_mb_decimal": round(total_bytes / 1_000_000.0, 3),
        "within_250mb_budget": bool(total_bytes < 250_000_000),
        "cache_mb": float(cache_mb),
        "streaming_breakdown": stage_io,
        "any_stopping_rule_trigger": bool(
            strip_public_summary["any_stopping_rule_trigger"]
            or outer_public_summary["any_stopping_rule_trigger"]
            or bool(ext_height_public.get("upper_extension_54_78", {}).get("exceeds_stopping_rule", False))
        ),
    }

    assert_no_local_paths(public_result)
    assert_no_unread_coordinates(public_result)
    if out_json is not None:
        out_p = Path(out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(public_result, indent=2) + "\n", encoding="utf-8")

    if priv_path is not None:
        private_summary = {
            "schema_version": "2.0.0-private",
            "strip_v_range": list(strip_v_range),
            "outer_v_range": list(outer_v_range),
            "strip_54_78_wraps": strip_records,
            "outer_wraps_w130_w135": outer_records,
            "extended_height_20260701": ext_height_private,
            "native_crust_1129_preflight": crust_section,
            "streaming_breakdown": stage_io,
            "total_mib": public_result["total_mib"],
            "total_mb_decimal": public_result["total_mb_decimal"],
        }
        (priv_path / "phase2_private_summary.json").write_text(
            json.dumps(private_summary, indent=2) + "\n", encoding="utf-8"
        )

    if sess is not None:
        sess.close()
    return public_result


# ---------------------------------------------------------------------------
# 9. CLI entrypoint
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="CPU-only (<250 MB) PHerc. Paris 4 dual-energy CT preflight, agraphon strip, outer-wrap, and 1.129 um crust scanner."
    )
    ap.add_argument("--vol-78", default=DEFAULT_URLS["vol_78"], help="78 keV OME-Zarr v2 URL or local dir")
    ap.add_argument("--vol-137", default=DEFAULT_URLS["vol_137"], help="137 keV OME-Zarr v2 URL or local dir")
    ap.add_argument("--vol-54", default=DEFAULT_URLS["vol_54"], help="54 keV DLS OME-Zarr v2 URL or local dir")
    ap.add_argument("--vol-1129", default=DEFAULT_URLS["vol_1129"], help="1.129 um 78 keV OME-Zarr v2 URL or local dir")
    ap.add_argument("--transform-78", default=DEFAULT_URLS["transform_78_to_dls"], help="78->DLS transform.json")
    ap.add_argument("--transform-137", default=DEFAULT_URLS["transform_137_to_dls"], help="137->DLS transform.json")
    ap.add_argument("--transform-1129", default=DEFAULT_URLS["transform_1129_to_78_r2"], help="1.129->78 r2 transform.json")
    ap.add_argument("--mesh-78", default=DEFAULT_URLS["mesh_78"], help="w010-027 tifxyz URL or local dir")
    ap.add_argument("--mesh-outer", default=DEFAULT_URLS["mesh_outer_dls"], help="20260926 outer-wrap DLS tifxyz URL or local dir")
    ap.add_argument("--level", type=int, default=2, help="Zarr pyramid level for Step 0 preflight (default: 2 = 9.6 um)")
    ap.add_argument("--step1-level", type=int, default=3, help="Zarr pyramid level for Step 1 strip (default: 3 = 19.2 um)")
    ap.add_argument("--cache-mb", type=float, default=256.0, help="In-memory LRU cache cap in MiB (default: 256)")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent chunk-fetch workers (default: 8)")
    ap.add_argument("--max-total-mb", type=float, default=225.0, help="Hard network streaming cap in MiB (default: 225)")
    ap.add_argument("--out-json", required=True, help="Output path for public sanitized JSON receipt")
    ap.add_argument(
        "--private-out-dir",
        default=None,
        help="Private output directory OUTSIDE the git repo for maps/PNGs/private summaries",
    )
    ap.add_argument("--run-step1", action="store_true", help="Execute Phase 1 Step 1 strip if Step 0 NCC lock passes")
    ap.add_argument("--phase2", action="store_true", help="Execute Phase 2 (54 vs 78 keV, outer wraps w130-w135, extended height, 1.129 um crust)")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.phase2:
        res = run_phase2_scans(
            vol_78_url=args.vol_78,
            vol_54_url=args.vol_54,
            vol_1129_url=args.vol_1129,
            transform_78_url=args.transform_78,
            transform_1129_url=args.transform_1129,
            mesh_78_url=args.mesh_78,
            mesh_outer_url=args.mesh_outer,
            cache_mb=args.cache_mb,
            workers=args.workers,
            max_total_mb=args.max_total_mb,
            out_json=args.out_json,
            private_out_dir=args.private_out_dir,
        )
        print(
            json.dumps(
                {
                    "preflight_54_78_passed": res["preflight_54_78_passed"],
                    "ncc_peak_54_78": res["ncc_peak_54_78"],
                    "sigma_blank_54_78": res["sigma_blank_54_78"],
                    "agraphon_wraps_locked": res["agraphon_strip_54_78"]["tiles_ncc_locked"],
                    "outer_wraps_locked": res["outer_wraps_w130_w135"]["tiles_ncc_locked"],
                    "crust_1129_executed": res["native_crust_1129_preflight"]["executed"],
                    "any_stopping_rule_trigger": res["any_stopping_rule_trigger"],
                    "total_mib": res["total_mib"],
                    "total_mb_decimal": res["total_mb_decimal"],
                },
                indent=2,
            )
        )
        return 0

    res = run_step0_preflight(
        vol_78_url=args.vol_78,
        vol_137_url=args.vol_137,
        transform_78_url=args.transform_78,
        transform_137_url=args.transform_137,
        mesh_78_url=args.mesh_78,
        level=args.level,
        cache_mb=args.cache_mb,
        workers=args.workers,
        out_json=args.out_json,
        private_out_dir=args.private_out_dir,
        run_step1=bool(args.run_step1),
        step1_level=args.step1_level,
        max_total_mb=min(args.max_total_mb, 178.0),
    )
    print(
        json.dumps(
            {
                "step0_preflight_passed": res["step0_preflight_passed"],
                "ncc_peak": res["ncc_peak"],
                "sigma_blank": res["sigma_blank"],
                "fpr_3sigma": res["fpr_3sigma"],
                "fpr_4sigma": res["fpr_4sigma"],
                "step0_total_mb": res["streaming_breakdown"]["step0_total_mb"],
                "combined_total_mb": res["streaming_breakdown"]["combined_total_mb"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
