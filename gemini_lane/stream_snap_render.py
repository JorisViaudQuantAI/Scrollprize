# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Zero-disk-bloat streaming m7-snapper, surface-volume renderer, and layer-coherence filter.

Designed for CPU-bounded streaming against remote HTTP(S) or local OME-Zarr v2
volumes (supporting the RTX 4060 ink queue: PHerc0125, PHerc1203, PHerc0191,
PHerc0826) with:
- Strict 404-only empty-chunk handling (HTTP 403/5xx/other errors raise cleanly).
- Strictly bounded in-memory LRU chunk cache (zero disk cache by default).
- Three-channel sentinel-safe tifxyz validation and two-pass m7 normal snapping.
- Trilinear surface-volume rendering with mesh + CT acquisition support masking.
- Patch-wise layer-coherence on-sheet filtering (per collab/claude-010.md).
"""
from __future__ import annotations

import argparse
import bz2
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Sequence
import urllib.error
import urllib.parse
import urllib.request
import zlib

import numpy as np
from scipy import ndimage as ndi

try:
    import requests as _requests
except ImportError:  # pragma: no cover - requests is optional fallback
    _requests = None

# Expose module-level `requests` attribute so callers/tests can patch requests.Session.
requests = _requests


# ---------------------------------------------------------------------------
# Internal helpers: TIFF I/O, Zarr codecs, and path sanitization
# ---------------------------------------------------------------------------

def _read_tiff_2d(path: Path) -> np.ndarray:
    """Read a 2D TIFF image using tifffile when available, else Pillow."""
    try:
        import tifffile  # type: ignore[import-not-found]

        return np.asarray(tifffile.imread(str(path)))
    except ImportError:
        from PIL import Image

        with Image.open(path) as img:
            return np.asarray(img)


def _write_tiff_float32(path: Path, arr: np.ndarray) -> None:
    """Write a 2D float32 TIFF using tifffile when available, else Pillow."""
    data = np.ascontiguousarray(arr, dtype=np.float32)
    try:
        import tifffile  # type: ignore[import-not-found]

        tifffile.imwrite(str(path), data)
    except ImportError:
        from PIL import Image

        Image.fromarray(data, mode="F").save(str(path), format="TIFF")


def _build_zarr_decoder(compressor: dict[str, Any] | None):
    """Return a bytes -> bytes decoder callable for a Zarr v2 compressor spec."""
    if compressor is None:
        return None
    try:
        import numcodecs  # type: ignore[import-not-found]

        codec = numcodecs.get_codec(compressor)
        return codec.decode
    except ImportError:
        cid = str(compressor.get("id", "")).lower()
        if cid == "zlib":
            return zlib.decompress
        if cid == "gzip":
            return gzip.decompress
        if cid == "bz2":
            return bz2.decompress
        if cid == "lzma":
            return lzma.decompress
        raise RuntimeError(
            f"Zarr compressor {compressor!r} requires 'numcodecs' to be installed."
        ) from None


def _sanitize_source_label(source: str | os.PathLike[str] | None) -> str | None:
    """Return a public URL (without credentials/query) or bare basename for local paths."""
    if source is None:
        return None
    raw = str(source).strip()
    if raw.lower().startswith(("http://", "https://")):
        parsed = urllib.parse.urlsplit(raw)
        host = parsed.hostname or ""
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    # Strip any local filesystem directory components (Windows or POSIX).
    normalized = raw.replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] if normalized else "local"


# ---------------------------------------------------------------------------
# 1. RemoteZarrV2: streaming read-only Zarr v2 client with strict LRU cache
# ---------------------------------------------------------------------------

class RemoteZarrV2:
    """Streaming read-only Zarr v2 reader over HTTP(S) or a local directory.

    Guarantees:
    - Zero disk cache: chunks live only in an in-memory LRU OrderedDict capped
      at ``cache_mb`` MiB (``cache_bytes <= cache_max`` strictly enforced).
    - Strict 404-only empty-chunk handling: only HTTP 404 (or a missing local
      chunk file) yields the fill value. HTTP 403, 5xx, short reads, and
      transport errors raise ``requests.HTTPError`` / ``OSError``.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        level: str | int | None = "0",
        *,
        cache_mb: float = 512.0,
        workers: int = 8,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
    ) -> None:
        if cache_mb < 0:
            raise ValueError("cache_mb must be >= 0")
        self.root = str(root).rstrip("/\\")
        self.local = not self.root.lower().startswith(("http://", "https://"))
        self.workers = max(1, int(workers))
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))

        self._tls = threading.local()
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None
        )

        # Resolve multiscale level prefix vs direct array directory.
        self.prefix = "" if level is None or str(level) == "" else str(level).strip("/")
        if self.local:
            root_path = Path(self.root)
            if self.prefix and (root_path / self.prefix / ".zarray").is_file():
                meta_key = f"{self.prefix}/.zarray"
            elif (root_path / ".zarray").is_file():
                self.prefix = ""
                meta_key = ".zarray"
            else:
                expected = (root_path / self.prefix / ".zarray") if self.prefix else (root_path / ".zarray")
                raise FileNotFoundError(f"Missing Zarr v2 metadata file: {expected.name}")
        else:
            meta_key = f"{self.prefix}/.zarray" if self.prefix else ".zarray"

        raw_meta = self._get_bytes(meta_key, allow_404=False)
        if raw_meta is None:  # pragma: no cover - defensive
            raise FileNotFoundError(f"Missing Zarr v2 metadata: {meta_key}")
        meta = json.loads(raw_meta.decode("utf-8"))
        if meta.get("zarr_format") != 2:
            raise ValueError(f"Only zarr_format == 2 is supported, got {meta.get('zarr_format')!r}")
        if meta.get("filters"):
            raise NotImplementedError("Zarr filters are not supported")

        self.shape: tuple[int, ...] = tuple(int(s) for s in meta["shape"])
        self.chunks: tuple[int, ...] = tuple(int(c) for c in meta["chunks"])
        if len(self.shape) != 3 or len(self.chunks) != 3:
            raise ValueError(f"Expected 3D Zarr v2 array, got shape={self.shape}, chunks={self.chunks}")
        self.dtype = np.dtype(meta["dtype"])
        fill = meta.get("fill_value", 0)
        self.fill_value = 0 if fill is None else fill
        self.dimension_separator: str = str(meta.get("dimension_separator", "."))
        self.order: str = str(meta.get("order", "C"))
        if self.order not in ("C", "F"):
            raise ValueError(f"Unsupported Zarr array order: {self.order!r}")
        self._decode = _build_zarr_decoder(meta.get("compressor"))
        self._expected_chunk_bytes = int(np.prod(self.chunks)) * int(self.dtype.itemsize)

        # In-memory LRU cache state (zero disk cache).
        self.cache: OrderedDict[tuple[int, int, int], np.ndarray | None] = OrderedDict()
        self.cache_bytes: int = 0
        self.cache_max: int = int(round(float(cache_mb) * 1024 * 1024))
        self.chunks_fetched: int = 0
        self.bytes_fetched: int = 0
        self.cache_hits: int = 0
        self.cache_evictions: int = 0

    def __enter__(self) -> RemoteZarrV2:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release worker pool resources and clear in-memory chunk cache."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        with self._lock:
            self.cache.clear()
            self.cache_bytes = 0

    def _session(self):
        if requests is None:
            return None
        sess = getattr(self._tls, "session", None)
        if sess is None:
            sess = requests.Session()
            self._tls.session = sess
        return sess

    def _get_bytes(self, key: str, *, allow_404: bool = True) -> bytes | None:
        if self.local:
            parts = [p for p in key.split("/") if p]
            path = Path(self.root).joinpath(*parts)
            if not path.is_file():
                if allow_404:
                    return None
                raise FileNotFoundError(f"Missing local Zarr key: {key}")
            return path.read_bytes()

        url = f"{self.root}/{key}"
        if requests is not None:
            return self._get_http_requests(url, allow_404=allow_404)
        return self._get_http_urllib(url, allow_404=allow_404)

    def _get_http_requests(self, url: str, *, allow_404: bool) -> bytes | None:
        assert requests is not None
        for attempt in range(self.max_retries):
            try:
                resp = self._session().get(url, timeout=self.timeout)
                status = int(resp.status_code)
                if status == 404 and allow_404:
                    return None
                if status == 200:
                    return resp.content
                # Fail fast on non-404 4xx errors (e.g., 401/403): never treat as empty chunk.
                if 400 <= status < 500:
                    resp.raise_for_status()
                    raise requests.HTTPError(
                        f"HTTP {status} for {url}; refusing to treat as empty chunk",
                        response=resp,
                    )
                # 5xx or unexpected status: retry transiently, then raise HTTPError.
                if attempt + 1 >= self.max_retries:
                    resp.raise_for_status()
                    raise requests.HTTPError(
                        f"Unexpected HTTP {status} for {url}; refusing to treat as empty chunk",
                        response=resp,
                    )
            except requests.HTTPError:
                raise
            except requests.RequestException:
                if attempt + 1 >= self.max_retries:
                    raise
            if self.retry_backoff > 0.0:
                time.sleep(self.retry_backoff * (attempt + 1))
        raise OSError(f"Exhausted retries fetching {url}")  # pragma: no cover

    def _get_http_urllib(self, url: str, *, allow_404: bool) -> bytes | None:
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = int(getattr(resp, "status", 200))
                    if status == 200:
                        return resp.read()
                    raise urllib.error.HTTPError(
                        url, status, f"Unexpected HTTP {status}", resp.headers, None
                    )
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and allow_404:
                    return None
                if 400 <= exc.code < 500 or attempt + 1 >= self.max_retries:
                    raise
            except OSError:
                if attempt + 1 >= self.max_retries:
                    raise
            if self.retry_backoff > 0.0:
                time.sleep(self.retry_backoff * (attempt + 1))
        raise OSError(f"Exhausted retries fetching {url}")  # pragma: no cover

    def _chunk_key(self, idx: tuple[int, int, int]) -> str:
        joined = self.dimension_separator.join(str(i) for i in idx)
        return f"{self.prefix}/{joined}" if self.prefix else joined

    def _load_chunk(self, idx: tuple[int, int, int]) -> np.ndarray | None:
        key = self._chunk_key(idx)
        raw = self._get_bytes(key, allow_404=True)
        if raw is None:
            return None
        raw_len = len(raw)
        decoded = self._decode(raw) if self._decode is not None else raw
        if len(decoded) != self._expected_chunk_bytes:
            raise OSError(
                f"Short or corrupt Zarr chunk {key}: expected {self._expected_chunk_bytes} bytes, "
                f"got {len(decoded)}"
            )
        arr = np.frombuffer(decoded, dtype=self.dtype).reshape(self.chunks, order=self.order).copy()
        with self._lock:
            self.chunks_fetched += 1
            self.bytes_fetched += raw_len
        return arr

    def _cache_put_locked(self, idx: tuple[int, int, int], arr: np.ndarray | None) -> None:
        nbytes = 0 if arr is None else int(arr.nbytes)
        if idx in self.cache:
            old = self.cache.pop(idx)
            if old is not None:
                self.cache_bytes -= int(old.nbytes)
        # If a single chunk exceeds cache_max, do not store it so cache_bytes <= cache_max holds.
        if nbytes > self.cache_max:
            return
        self.cache[idx] = arr
        self.cache_bytes += nbytes
        while self.cache_bytes > self.cache_max and self.cache:
            _, evicted = self.cache.popitem(last=False)
            if evicted is not None:
                self.cache_bytes -= int(evicted.nbytes)
            self.cache_evictions += 1

    def get_chunk(self, idx: tuple[int, int, int]) -> np.ndarray | None:
        """Fetch a single chunk index ``(cz, cy, cx)`` with LRU caching."""
        return self.get_chunks([idx])[idx]

    def get_chunks(
        self, idxs: Iterable[tuple[int, int, int]]
    ) -> dict[tuple[int, int, int], np.ndarray | None]:
        """Fetch multiple chunk indices ``(cz, cy, cx)`` -> ndarray or None (404/missing)."""
        unique_idxs = list(dict.fromkeys(tuple(int(v) for v in i) for i in idxs))
        out: dict[tuple[int, int, int], np.ndarray | None] = {}
        need: list[tuple[int, int, int]] = []
        with self._lock:
            for idx in unique_idxs:
                if idx in self.cache:
                    self.cache.move_to_end(idx)
                    out[idx] = self.cache[idx]
                    self.cache_hits += 1
                else:
                    need.append(idx)

        if need:
            if self._pool is not None and len(need) > 1:
                fetched = list(self._pool.map(self._load_chunk, need))
            else:
                fetched = [self._load_chunk(idx) for idx in need]
            with self._lock:
                for idx, arr in zip(need, fetched):
                    out[idx] = arr
                    self._cache_put_locked(idx, arr)
        return out

    def read_box(
        self, lo_zyx: Sequence[int], hi_zyx: Sequence[int]
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Read dense block for ``[lo_zyx, hi_zyx)`` clipped to array bounds.

        Returns ``(block, origin_zyx)``.
        """
        lo = [max(0, int(v)) for v in lo_zyx]
        hi = [min(s, int(v)) for s, v in zip(self.shape, hi_zyx)]
        if any(h <= l for l, h in zip(lo, hi)):
            return np.full((1, 1, 1), self.fill_value, dtype=self.dtype), (0, 0, 0)

        c0 = [l // c for l, c in zip(lo, self.chunks)]
        c1 = [(h - 1) // c for h, c in zip(hi, self.chunks)]
        idxs = [
            (cz, cy, cx)
            for cz in range(c0[0], c1[0] + 1)
            for cy in range(c0[1], c1[1] + 1)
            for cx in range(c0[2], c1[2] + 1)
        ]
        chunks = self.get_chunks(idxs)
        block = np.full([h - l for l, h in zip(lo, hi)], self.fill_value, dtype=self.dtype)
        for idx, arr in chunks.items():
            if arr is None:
                continue
            s = [i * c for i, c in zip(idx, self.chunks)]
            a0 = [max(l, x) for l, x in zip(lo, s)]
            a1 = [min(h, x + c, sh) for h, x, c, sh in zip(hi, s, self.chunks, self.shape)]
            if any(y <= x for x, y in zip(a0, a1)):
                continue
            block[
                a0[0] - lo[0] : a1[0] - lo[0],
                a0[1] - lo[1] : a1[1] - lo[1],
                a0[2] - lo[2] : a1[2] - lo[2],
            ] = arr[
                a0[0] - s[0] : a1[0] - s[0],
                a0[1] - s[1] : a1[1] - s[1],
                a0[2] - s[2] : a1[2] - s[2],
            ]
        return block, (lo[0], lo[1], lo[2])

    def sample_trilinear(
        self,
        xyz: np.ndarray,
        *,
        max_direct_box_voxels: int = 16_000_000,
    ) -> np.ndarray:
        """Trilinearly sample volume at ``xyz`` coordinates (shape ``(..., 3)``).

        Coordinates are in ``(x, y, z)`` voxel order (tifxyz convention) while the
        underlying Zarr volume is indexed ``(z, y, x)``. Out-of-bounds or non-finite
        coordinates receive ``fill_value``. For sparse or wide point sets, queries
        are partitioned by chunk cell to avoid allocating a huge bounding box.
        """
        pts = np.asarray(xyz, dtype=np.float64)
        if pts.shape[-1] != 3:
            raise ValueError(f"Expected last dimension 3 for xyz, got shape {pts.shape}")
        orig_shape = pts.shape[:-1]
        flat = pts.reshape(-1, 3)
        out = np.full(flat.shape[0], float(self.fill_value), dtype=np.float32)
        if flat.shape[0] == 0:
            return out.reshape(orig_shape)

        sz, sy, sx = self.shape
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

        valid_indices = np.flatnonzero(in_bounds)
        q = flat[valid_indices]  # (N, 3) in xyz
        q_zyx = q[:, ::-1]

        lo = np.maximum(0, np.floor(q_zyx.min(axis=0)).astype(int) - 1)
        hi = np.minimum(self.shape, np.ceil(q_zyx.max(axis=0)).astype(int) + 2)
        box_voxels = int(np.prod(np.maximum(0, hi - lo)))

        if box_voxels <= max_direct_box_voxels:
            block, org = self.read_box(lo.tolist(), hi.tolist())
            blockf = block.astype(np.float32, copy=False)
            coords = np.stack(
                [q_zyx[:, 0] - org[0], q_zyx[:, 1] - org[1], q_zyx[:, 2] - org[2]],
                axis=0,
            )
            out[valid_indices] = ndi.map_coordinates(
                blockf,
                coords,
                order=1,
                mode="constant",
                cval=float(self.fill_value),
                output=np.float32,
            )
            return out.reshape(orig_shape)

        # Partition by Zarr chunk cell so memory stays bounded for curved/wide meshes.
        cshape = np.asarray(self.chunks, dtype=np.int64)
        cell_ids = np.floor(q_zyx).astype(np.int64) // cshape[None, :]
        # Encode 3D chunk indices into unique groups via structured view.
        keys_view = np.ascontiguousarray(cell_ids).view(
            np.dtype((np.void, cell_ids.dtype.itemsize * 3))
        ).ravel()
        order = np.argsort(keys_view, kind="mergesort")
        sorted_keys = keys_view[order]
        change = np.flatnonzero(np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1], [True])))

        for start, end in zip(change[:-1], change[1:]):
            grp = order[start:end]
            sub_zyx = q_zyx[grp]
            slo = np.maximum(0, np.floor(sub_zyx.min(axis=0)).astype(int) - 1)
            shi = np.minimum(self.shape, np.ceil(sub_zyx.max(axis=0)).astype(int) + 2)
            block, org = self.read_box(slo.tolist(), shi.tolist())
            blockf = block.astype(np.float32, copy=False)
            coords = np.stack(
                [sub_zyx[:, 0] - org[0], sub_zyx[:, 1] - org[1], sub_zyx[:, 2] - org[2]],
                axis=0,
            )
            out[valid_indices[grp]] = ndi.map_coordinates(
                blockf,
                coords,
                order=1,
                mode="constant",
                cval=float(self.fill_value),
                output=np.float32,
            )
        return out.reshape(orig_shape)


# ---------------------------------------------------------------------------
# 2. Sentinel-safe tifxyz validation, grid normals, and two-pass m7 snapper
# ---------------------------------------------------------------------------

def validate_tifxyz_channels(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate all 3 vertex channels (matching collab/claude-003.md).

    A vertex is valid ONLY if x, y, and z are all finite and >= 0.
    Invalid vertices are coerced to -1.0 across all 3 channels.
    """
    xf = np.asarray(x, dtype=np.float32)
    yf = np.asarray(y, dtype=np.float32)
    zf = np.asarray(z, dtype=np.float32)
    if xf.ndim != 2 or xf.shape != yf.shape or xf.shape != zf.shape:
        raise ValueError(
            f"Expected matching 2D arrays for x, y, z; got {xf.shape}, {yf.shape}, {zf.shape}"
        )
    valid = np.isfinite(xf) & np.isfinite(yf) & np.isfinite(zf)
    with np.errstate(invalid="ignore"):
        valid &= (xf >= 0.0) & (yf >= 0.0) & (zf >= 0.0)

    clean_x = np.where(valid, xf, np.float32(-1.0))
    clean_y = np.where(valid, yf, np.float32(-1.0))
    clean_z = np.where(valid, zf, np.float32(-1.0))
    return clean_x, clean_y, clean_z, valid


def read_tifxyz(
    mesh_dir: str | os.PathLike[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a tifxyz directory (x.tif, y.tif, z.tif, optional meta.json & mask.tif)."""
    root = Path(mesh_dir)
    x = _read_tiff_2d(root / "x.tif").astype(np.float32)
    y = _read_tiff_2d(root / "y.tif").astype(np.float32)
    z = _read_tiff_2d(root / "z.tif").astype(np.float32)
    meta_path = root / "meta.json"
    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    cx, cy, cz, valid = validate_tifxyz_channels(x, y, z)

    mask_path = root / "mask.tif"
    if mask_path.is_file():
        m = _read_tiff_2d(mask_path)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape == valid.shape:
            valid &= m >= 255
            cx = np.where(valid, cx, np.float32(-1.0))
            cy = np.where(valid, cy, np.float32(-1.0))
            cz = np.where(valid, cz, np.float32(-1.0))
    return cx, cy, cz, valid, meta


def write_tifxyz(
    mesh_dir: str | os.PathLike[str],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    meta: dict[str, Any] | None = None,
) -> None:
    """Write a sentinel-clean tifxyz directory."""
    root = Path(mesh_dir)
    root.mkdir(parents=True, exist_ok=True)
    cx, cy, cz, valid = validate_tifxyz_channels(x, y, z)
    _write_tiff_float32(root / "x.tif", cx)
    _write_tiff_float32(root / "y.tif", cy)
    _write_tiff_float32(root / "z.tif", cz)

    out_meta = dict(meta or {})
    out_meta.setdefault("format", "tifxyz")
    out_meta.setdefault("type", "seg")
    out_meta.setdefault("scale", [0.05, 0.05])
    if valid.any():
        out_meta["bbox"] = [
            [float(cx[valid].min()), float(cy[valid].min()), float(cz[valid].min())],
            [float(cx[valid].max()), float(cy[valid].max()), float(cz[valid].max())],
        ]
    else:
        out_meta["bbox"] = None
    (root / "meta.json").write_text(json.dumps(out_meta, indent=2) + "\n", encoding="utf-8")


def compute_grid_normals(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, valid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unit normals ``N = normalize(cross(dP/du, dP/dv))`` over valid neighborhoods.

    Uses central differences where both neighbors are valid and one-sided differences
    adjacent to boundaries or invalid vertices so sentinels never corrupt valid normals.
    Convention matches VC3D and ``firstletters-colab/ink/render_tifxyz.py``:
    ``u`` = column axis (axis 1), ``v`` = row axis (axis 0).
    """
    cx, cy, cz, val = validate_tifxyz_channels(x, y, z)
    if valid is not None:
        val = val & np.asarray(valid, dtype=bool)
    h, w = val.shape
    pts = np.zeros((h, w, 3), dtype=np.float64)
    pts[val, 0] = cx[val]
    pts[val, 1] = cy[val]
    pts[val, 2] = cz[val]

    def _diff(axis: int) -> tuple[np.ndarray, np.ndarray]:
        d = np.zeros((h, w, 3), dtype=np.float64)
        vf = np.zeros((h, w), dtype=bool)
        vb = np.zeros((h, w), dtype=bool)
        fwd = np.zeros((h, w, 3), dtype=np.float64)
        bwd = np.zeros((h, w, 3), dtype=np.float64)

        if axis == 1 and w >= 2:
            vf[:, :-1] = val[:, :-1] & val[:, 1:]
            vb[:, 1:] = val[:, 1:] & val[:, :-1]
            fwd[:, :-1] = pts[:, 1:] - pts[:, :-1]
            bwd[:, 1:] = pts[:, 1:] - pts[:, :-1]
        elif axis == 0 and h >= 2:
            vf[:-1, :] = val[:-1, :] & val[1:, :]
            vb[1:, :] = val[1:, :] & val[:-1, :]
            fwd[:-1, :] = pts[1:, :] - pts[:-1, :]
            bwd[1:, :] = pts[1:, :] - pts[:-1, :]

        both = vf & vb
        only_f = vf & ~vb
        only_b = vb & ~vf
        d[both] = 0.5 * (fwd[both] + bwd[both])
        d[only_f] = fwd[only_f]
        d[only_b] = bwd[only_b]
        return d, (vf | vb)

    du, ok_u = _diff(1)
    dv, ok_v = _diff(0)
    n = np.cross(du, dv)
    ln = np.linalg.norm(n, axis=-1)
    normal_ok = val & ok_u & ok_v & (ln > 1e-9)
    n[normal_ok] /= ln[normal_ok, None]
    n[~normal_ok] = 0.0
    return n.astype(np.float32), normal_ok


def _pick_nearest_ridge_offset(
    profile: np.ndarray, offsets: np.ndarray, threshold: float, step: float
) -> float:
    """Choose the nearest supra-threshold peak offset along a 1D m7 profile."""
    above = profile >= threshold
    if not above.any():
        return 0.0

    # Segment contiguous supra-threshold runs [start, end] (inclusive).
    padded = np.concatenate(([False], above, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts = edges[0::2]
    ends = edges[1::2] - 1

    best_offset = 0.0
    best_abs_dist = float("inf")
    k_len = len(profile)

    for s, e in zip(starts, ends):
        seg = profile[s : e + 1]
        seg_off = offsets[s : e + 1]
        max_val = float(seg.max())
        # Locate local peak(s) inside this supra-threshold component.
        max_rel = np.flatnonzero(seg >= max_val - 1e-6)
        if len(max_rel) == 1:
            k_peak = int(s + max_rel[0])
            cand = float(offsets[k_peak])
            if 0 < k_peak < k_len - 1:
                y_l = float(profile[k_peak - 1])
                y_c = float(profile[k_peak])
                y_r = float(profile[k_peak + 1])
                denom = y_l - 2.0 * y_c + y_r
                if denom < -1e-9 and y_c > y_l and y_c > y_r:
                    frac = 0.5 * (y_l - y_r) / denom
                    cand += float(np.clip(frac, -0.5, 0.5)) * step
        else:
            # Flat/clamped ridge plateau: use intensity-weighted centroid of the component.
            wsum = float(seg.sum())
            cand = float((seg * seg_off).sum() / wsum) if wsum > 0.0 else float(seg_off.mean())

        if abs(cand) < best_abs_dist:
            best_abs_dist = abs(cand)
            best_offset = cand

    return best_offset


def _compute_center_vs_outer_disp_diff(disp_grid: np.ndarray, valid: np.ndarray) -> float:
    """Compute median displacement difference (center half minus outer half) over valid grid."""
    ys, xs = np.nonzero(valid)
    n = len(ys)
    if n < 2:
        return 0.0
    cy = float(ys.mean())
    cx = float(xs.mean())
    sy = max(float(ys.std()), 1.0)
    sx = max(float(xs.std()), 1.0)
    r2 = ((ys - cy) / sy) ** 2 + ((xs - cx) / sx) ** 2
    med_r2 = float(np.median(r2))
    center_mask = r2 <= med_r2
    outer_mask = r2 > med_r2
    if not outer_mask.any():
        return 0.0
    d_vals = disp_grid[ys, xs]
    return float(np.median(d_vals[center_mask]) - np.median(d_vals[outer_mask]))


def snap_tifxyz_to_m7(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    m7_reader: RemoteZarrV2,
    passes: int = 2,
    search_radius: float = 12.0,
    step: float = 1.0,
    threshold: float = 51.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Snap valid tifxyz vertices onto the m7 surface prediction along local unit normals.

    Returns ``(snapped_x, snapped_y, snapped_z, geometry_metrics)``.
    """
    if passes < 0:
        raise ValueError("passes must be >= 0")
    if search_radius <= 0 or step <= 0:
        raise ValueError("search_radius and step must be > 0")

    cx, cy, cz, valid = validate_tifxyz_channels(x, y, z)
    p_raw = np.stack([cx, cy, cz], axis=-1).astype(np.float64)
    p_cur = p_raw.copy()

    n_valid = int(valid.sum())
    n_total = int(valid.size)
    valid_fraction = float(n_valid / n_total) if n_total > 0 else 0.0

    if n_valid == 0:
        metrics = {
            "valid_vertex_fraction": 0.0,
            "valid_vertex_count": 0,
            "total_vertex_count": n_total,
            "on_sheet_raw": 0.0,
            "on_sheet_after_snap": 0.0,
            "median_disp_vx": 0.0,
            "p90_disp_vx": 0.0,
            "center_vs_outer_disp_diff": 0.0,
            "passes": int(passes),
            "search_radius": float(search_radius),
            "step": float(step),
            "threshold": float(threshold),
        }
        return cx, cy, cz, metrics

    raw_vals = m7_reader.sample_trilinear(p_raw[valid])
    on_sheet_raw = float(np.mean(raw_vals >= threshold))

    n_steps = int(round(float(search_radius) / float(step)))
    offsets = np.arange(-n_steps, n_steps + 1, dtype=np.float64) * float(step)
    sz, sy, sx = m7_reader.shape
    max_xyz = np.array([sx - 1.0, sy - 1.0, sz - 1.0], dtype=np.float64)

    for _ in range(int(passes)):
        normals, normal_ok = compute_grid_normals(
            p_cur[..., 0], p_cur[..., 1], p_cur[..., 2], valid=valid
        )
        active = valid & normal_ok
        if not active.any():
            break

        pts_act = p_cur[active]  # (M, 3)
        nrm_act = normals[active].astype(np.float64)  # (M, 3)
        # Sample in bounded batches to keep transient array RAM small.
        batch_size = 4096
        disps = np.zeros(pts_act.shape[0], dtype=np.float64)
        for i0 in range(0, pts_act.shape[0], batch_size):
            i1 = min(i0 + batch_size, pts_act.shape[0])
            rays = pts_act[i0:i1, None, :] + offsets[None, :, None] * nrm_act[i0:i1, None, :]
            profiles = m7_reader.sample_trilinear(rays)  # (B, K)
            for j in range(i1 - i0):
                disps[i0 + j] = _pick_nearest_ridge_offset(
                    profiles[j], offsets, float(threshold), float(step)
                )

        shifted = pts_act + disps[:, None] * nrm_act
        shifted = np.clip(shifted, 0.0, max_xyz[None, :])
        p_cur[active] = shifted

    # Enforce 3-channel sentinel consistency for invalid vertices.
    snapped_x = np.where(valid, p_cur[..., 0], -1.0).astype(np.float32)
    snapped_y = np.where(valid, p_cur[..., 1], -1.0).astype(np.float32)
    snapped_z = np.where(valid, p_cur[..., 2], -1.0).astype(np.float32)

    p_final = np.stack([snapped_x, snapped_y, snapped_z], axis=-1).astype(np.float64)
    after_vals = m7_reader.sample_trilinear(p_final[valid])
    on_sheet_after_snap = float(np.mean(after_vals >= threshold))

    disp_grid = np.zeros(valid.shape, dtype=np.float64)
    disp_grid[valid] = np.linalg.norm(p_final[valid] - p_raw[valid], axis=-1)
    valid_disps = disp_grid[valid]

    metrics = {
        "valid_vertex_fraction": valid_fraction,
        "valid_vertex_count": n_valid,
        "total_vertex_count": n_total,
        "on_sheet_raw": on_sheet_raw,
        "on_sheet_after_snap": on_sheet_after_snap,
        "median_disp_vx": float(np.median(valid_disps)),
        "p90_disp_vx": float(np.percentile(valid_disps, 90)),
        "center_vs_outer_disp_diff": _compute_center_vs_outer_disp_diff(disp_grid, valid),
        "passes": int(passes),
        "search_radius": float(search_radius),
        "step": float(step),
        "threshold": float(threshold),
    }
    return snapped_x, snapped_y, snapped_z, metrics


# ---------------------------------------------------------------------------
# 3. Surface-volume renderer (pure NumPy/SciPy, zero cv2 dependency)
# ---------------------------------------------------------------------------

def _fill_invalid_nearest(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid 2D/3D-last grid cells from nearest valid neighbor to prevent edge ringing."""
    if valid.all() or not valid.any():
        return arr
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return arr[idx[0], idx[1]]


def _normalize_scale(scale: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(scale, (tuple, list, np.ndarray)):
        if len(scale) != 2:
            raise ValueError(f"Expected 2 elements for scale, got {scale!r}")
        sx, sy = float(scale[0]), float(scale[1])
    else:
        sx = sy = float(scale)
    if sx <= 0.0 or sy <= 0.0:
        raise ValueError(f"scale must be positive, got {(sx, sy)!r}")
    return sx, sy


def render_surface_stack(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ct_reader: RemoteZarrV2,
    num_slices: int = 31,
    slice_step: float = 1.0,
    flip_normals: bool = False,
    scale: float | Sequence[float] = 0.05,
    *,
    uv_stride: float | None = None,
    tile_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a ``[num_slices, H_out, W_out]`` uint8 CT surface volume stack.

    - Upsamples valid mesh grid coordinates by ``1 / scale`` (or uses ``uv_stride``
      directly when provided, e.g. ``uv_stride=1.0`` for 1:1 grid rendering).
    - Computes unit normals ``N = normalize(cross(dP/du, dP/dv))`` (negated when
      ``flip_normals=True``).
    - Samples ``num_slices`` planes along ``N`` with center mesh at
      ``(num_slices - 1) // 2``.
    - Returns ``(stack_uint8_lhw, valid_mask)`` where ``valid_mask`` is True only
      where BOTH the mesh is valid AND the CT volume has non-zero acquisition support.
    """
    if num_slices < 1:
        raise ValueError("num_slices must be >= 1")
    if slice_step <= 0:
        raise ValueError("slice_step must be > 0")
    sx, sy = _normalize_scale(uv_stride if uv_stride is not None else scale)

    cx, cy, cz, valid = validate_tifxyz_channels(x, y, z)
    h_grid, w_grid = valid.shape
    out_h = 1 if h_grid == 1 else max(1, int(round((h_grid - 1) / sy)) + 1)
    out_w = 1 if w_grid == 1 else max(1, int(round((w_grid - 1) / sx)) + 1)

    stack = np.zeros((num_slices, out_h, out_w), dtype=np.uint8)
    mesh_valid_out = np.zeros((out_h, out_w), dtype=bool)
    if not valid.any():
        return stack, mesh_valid_out

    normals, normal_ok = compute_grid_normals(cx, cy, cz, valid=valid)
    if flip_normals:
        normals = -normals
    usable = valid & normal_ok
    if not usable.any():
        return stack, mesh_valid_out

    pts = np.stack([cx, cy, cz], axis=-1).astype(np.float32)
    ptsf = _fill_invalid_nearest(pts, valid)
    nrmf = _fill_invalid_nearest(normals, usable)
    vmask = usable.astype(np.float32)

    mesh_idx = (num_slices - 1) // 2
    offsets = (np.arange(num_slices, dtype=np.float64) - float(mesh_idx)) * float(slice_step)
    dtype_max = (
        float(np.iinfo(ct_reader.dtype).max)
        if np.issubdtype(ct_reader.dtype, np.integer) and ct_reader.dtype != np.uint8
        else None
    )

    t_sz = max(16, int(tile_size))
    for ty in range(0, out_h, t_sz):
        th = min(t_sz, out_h - ty)
        gv_1d = (np.arange(ty, ty + th, dtype=np.float64)) * sy
        for tx in range(0, out_w, t_sz):
            tw = min(t_sz, out_w - tx)
            gu_1d = (np.arange(tx, tx + tw, dtype=np.float64)) * sx
            gv, gu = np.meshgrid(gv_1d, gu_1d, indexing="ij")

            m = ndi.map_coordinates(vmask, [gv, gu], order=1, mode="constant", cval=0.0) > 0.999
            if not m.any():
                continue

            gv_m = gv[m]
            gu_m = gu[m]
            p_x = ndi.map_coordinates(ptsf[..., 0], [gv_m, gu_m], order=1, mode="nearest")
            p_y = ndi.map_coordinates(ptsf[..., 1], [gv_m, gu_m], order=1, mode="nearest")
            p_z = ndi.map_coordinates(ptsf[..., 2], [gv_m, gu_m], order=1, mode="nearest")
            p_tile = np.stack([p_x, p_y, p_z], axis=-1).astype(np.float64)

            n_x = ndi.map_coordinates(nrmf[..., 0], [gv_m, gu_m], order=1, mode="nearest")
            n_y = ndi.map_coordinates(nrmf[..., 1], [gv_m, gu_m], order=1, mode="nearest")
            n_z = ndi.map_coordinates(nrmf[..., 2], [gv_m, gu_m], order=1, mode="nearest")
            n_tile = np.stack([n_x, n_y, n_z], axis=-1).astype(np.float64)
            n_norm = np.linalg.norm(n_tile, axis=-1, keepdims=True)
            n_good = n_norm[..., 0] > 1e-6
            if not n_good.any():
                continue
            n_tile[n_good] /= n_norm[n_good]

            # Update tile validity if interpolated normal degenerated.
            if not n_good.all():
                m_idx = np.flatnonzero(m)
                m_flat = m.ravel()
                m_flat[m_idx[~n_good]] = False
                m = m_flat.reshape(m.shape)
                p_tile = p_tile[n_good]
                n_tile = n_tile[n_good]

            # Sample all num_slices along N for this tile.
            # Shape: (num_slices, M, 3) -> sampled (num_slices, M)
            rays = p_tile[None, :, :] + offsets[:, None, None] * n_tile[None, :, :]
            vals = ct_reader.sample_trilinear(rays)  # (L, M) float32
            if dtype_max is not None and dtype_max > 0.0:
                vals = vals * (255.0 / dtype_max)
            vals_u8 = np.clip(np.rint(vals), 0.0, 255.0).astype(np.uint8)

            tile_out = np.zeros((num_slices, th, tw), dtype=np.uint8)
            tile_out[:, m] = vals_u8
            stack[:, ty : ty + th, tx : tx + tw] = tile_out
            mesh_valid_out[ty : ty + th, tx : tx + tw] = m

    ct_support = stack.max(axis=0) > 0
    valid_mask = mesh_valid_out & ct_support
    return stack, valid_mask


# ---------------------------------------------------------------------------
# 4. Layer-coherence on-sheet filter (per collab/claude-010.md)
# ---------------------------------------------------------------------------

def layer_coherence_score(
    stack: np.ndarray,
    mesh_index: int,
    delta: int = 2,
    patch_size: int = 64,
    max_shift: int = 4,
    *,
    valid_mask: np.ndarray | None = None,
    min_valid_fraction: float = 0.5,
) -> dict[str, Any]:
    """Score whether a rendered stack is sheet-parallel vs an oblique cut.

    Compares plane ``mesh_index - delta`` against plane ``mesh_index + delta``
    over non-overlapping valid patches: on a sheet-parallel render, features align
    at zero shift ``(0, 0)`` so zero-shift Pearson correlation beats the maximum
    shifted-patch correlation over ``[-max_shift, max_shift]^2 \\ {(0, 0)}``;
    on an oblique cut, the sheet pattern translates laterally so a non-zero shift
    correlates more strongly.
    """
    arr = np.asarray(stack)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D stack [L, H, W], got shape {arr.shape}")
    depth, height, width = arr.shape
    if delta < 1:
        raise ValueError("delta must be >= 1")
    if max_shift < 1:
        raise ValueError("max_shift must be >= 1")
    idx_a = int(mesh_index) - int(delta)
    idx_b = int(mesh_index) + int(delta)
    if idx_a < 0 or idx_b >= depth:
        raise ValueError(
            f"Planes [{idx_a}, {idx_b}] out of bounds for stack depth {depth}"
        )

    plane_a = arr[idx_a].astype(np.float64)
    plane_b = arr[idx_b].astype(np.float64)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape != (height, width):
            raise ValueError(f"valid_mask shape {vm.shape} does not match stack HW {(height, width)}")
    else:
        vm = (plane_a > 0.0) | (plane_b > 0.0)

    s = int(max_shift)
    # Use requested patch_size when the canvas fits; fall back to full frame on tiny test stacks.
    p_h = int(patch_size) if height >= patch_size else height
    p_w = int(patch_size) if width >= patch_size else width
    if p_h <= 2 * s or p_w <= 2 * s:
        return {
            "sheet_parallel_fraction": 0.0,
            "patch_count": 0,
            "mean_zero_shift_r": 0.0,
            "mean_best_shifted_r": 0.0,
            "patch_scores": [],
        }

    shifts = [
        (dy, dx)
        for dy in range(-s, s + 1)
        for dx in range(-s, s + 1)
        if not (dy == 0 and dx == 0)
    ]

    patch_scores: list[dict[str, Any]] = []
    inner_h = p_h - 2 * s
    inner_w = p_w - 2 * s

    for r0 in range(0, height - p_h + 1, p_h):
        for c0 in range(0, width - p_w + 1, p_w):
            vm_in = vm[r0 + s : r0 + s + inner_h, c0 + s : c0 + s + inner_w]
            if float(vm_in.mean()) < min_valid_fraction:
                continue

            a = plane_a[r0 + s : r0 + s + inner_h, c0 + s : c0 + s + inner_w]
            a_vals = a[vm_in]
            a_centered = a_vals - a_vals.mean()
            a_norm = float(np.linalg.norm(a_centered))
            if a_norm < 1e-6:
                continue

            b0 = plane_b[r0 + s : r0 + s + inner_h, c0 + s : c0 + s + inner_w]
            b0_vals = b0[vm_in]
            b0_centered = b0_vals - b0_vals.mean()
            b0_norm = float(np.linalg.norm(b0_centered))
            if b0_norm < 1e-6:
                continue

            zero_r = float(np.dot(a_centered, b0_centered) / (a_norm * b0_norm))

            best_r = -1.0
            best_shift = (0, 0)
            for dy, dx in shifts:
                bs = plane_b[
                    r0 + s + dy : r0 + s + dy + inner_h,
                    c0 + s + dx : c0 + s + dx + inner_w,
                ]
                bs_vals = bs[vm_in]
                bs_centered = bs_vals - bs_vals.mean()
                bs_norm = float(np.linalg.norm(bs_centered))
                if bs_norm < 1e-6:
                    r_val = 0.0
                else:
                    r_val = float(np.dot(a_centered, bs_centered) / (a_norm * bs_norm))
                if r_val > best_r:
                    best_r = r_val
                    best_shift = (dy, dx)

            is_parallel = bool(zero_r > best_r and zero_r > 0.0)
            patch_scores.append(
                {
                    "row": int(r0),
                    "col": int(c0),
                    "zero_shift_r": round(zero_r, 6),
                    "best_shifted_r": round(best_r, 6),
                    "best_shift": [int(best_shift[0]), int(best_shift[1])],
                    "sheet_parallel": is_parallel,
                }
            )

    count = len(patch_scores)
    if count == 0:
        return {
            "sheet_parallel_fraction": 0.0,
            "patch_count": 0,
            "mean_zero_shift_r": 0.0,
            "mean_best_shifted_r": 0.0,
            "patch_scores": [],
        }

    parallel_frac = float(sum(1 for p in patch_scores if p["sheet_parallel"]) / count)
    mean_zero = float(np.mean([p["zero_shift_r"] for p in patch_scores]))
    mean_best = float(np.mean([p["best_shifted_r"] for p in patch_scores]))
    return {
        "sheet_parallel_fraction": parallel_frac,
        "patch_count": count,
        "mean_zero_shift_r": mean_zero,
        "mean_best_shifted_r": mean_best,
        "patch_scores": patch_scores,
    }


# ---------------------------------------------------------------------------
# 5. CLI pipeline with zero-local-path JSON receipt
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Zero-disk-bloat streaming m7-snapper, surface-volume renderer, and layer-coherence filter."
    )
    ap.add_argument("--mesh-dir", required=True, help="Input tifxyz directory (x.tif, y.tif, z.tif, meta.json)")
    ap.add_argument("--ct-zarr", required=True, help="CT OME-Zarr v2 URL or local directory")
    ap.add_argument("--m7-zarr", default=None, help="Optional m7 surface-prediction Zarr v2 URL or local directory")
    ap.add_argument("--ct-level", default="0", help="Multiscale level for CT Zarr (default: 0)")
    ap.add_argument("--m7-level", default="0", help="Multiscale level for m7 Zarr (default: 0)")
    ap.add_argument("--snap-passes", type=int, default=2, help="Number of m7 normal-snapping passes (default: 2)")
    ap.add_argument("--search-radius", type=float, default=12.0, help="Snapping search radius in voxels (default: 12)")
    ap.add_argument("--snap-step", type=float, default=1.0, help="Snapping sampling step along normal (default: 1.0)")
    ap.add_argument("--m7-threshold", type=float, default=51.0, help="m7 surface probability threshold (default: 51)")
    ap.add_argument("--num-slices", type=int, default=31, help="Number of rendered depth slices (default: 31)")
    ap.add_argument("--slice-step", type=float, default=1.0, help="Voxel spacing between slices (default: 1.0)")
    ap.add_argument("--flip-normals", action="store_true", help="Negate surface normals during rendering")
    ap.add_argument("--scale", type=float, default=None, help="Override mesh grid scale (default: meta.json or 0.05)")
    ap.add_argument("--uv-stride", type=float, default=None, help="Optional direct UV sampling stride (e.g. 1.0)")
    ap.add_argument("--coherence-delta", type=int, default=2, help="Plane half-offset for layer coherence (default: 2)")
    ap.add_argument("--coherence-patch-size", type=int, default=64, help="Patch size for layer coherence (default: 64)")
    ap.add_argument("--coherence-max-shift", type=int, default=4, help="Max lateral shift for coherence (default: 4)")
    ap.add_argument("--cache-mb", type=float, default=512.0, help="In-memory LRU chunk cache cap in MiB (default: 512)")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent chunk-fetch workers (default: 8)")
    ap.add_argument("--out-npy", required=True, help="Output .npy path for rendered [L, H, W] uint8 stack")
    ap.add_argument("--out-receipt", required=True, help="Output .json path for zero-local-path receipt")
    ap.add_argument("--out-snapped-mesh", default=None, help="Optional output directory for snapped tifxyz")
    return ap


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Execute snap -> render -> layer-coherence and write stack .npy + sanitized receipt."""
    x, y, z, valid, meta = read_tifxyz(args.mesh_dir)
    mesh_shape = [int(x.shape[0]), int(x.shape[1])]

    if args.scale is not None:
        scale_val: float | Sequence[float] = float(args.scale)
    elif "scale" in meta:
        scale_val = meta["scale"]
    else:
        scale_val = 0.05
    sx, sy = _normalize_scale(scale_val)

    m7_chunks = 0
    m7_bytes = 0
    if args.m7_zarr and args.snap_passes > 0:
        with RemoteZarrV2(
            args.m7_zarr, level=args.m7_level, cache_mb=args.cache_mb, workers=args.workers
        ) as m7_reader:
            x, y, z, geom_metrics = snap_tifxyz_to_m7(
                x,
                y,
                z,
                m7_reader,
                passes=args.snap_passes,
                search_radius=args.search_radius,
                step=args.snap_step,
                threshold=args.m7_threshold,
            )
            m7_chunks = m7_reader.chunks_fetched
            m7_bytes = m7_reader.bytes_fetched
    else:
        n_valid = int(valid.sum())
        n_total = int(valid.size)
        geom_metrics = {
            "valid_vertex_fraction": float(n_valid / n_total) if n_total > 0 else 0.0,
            "valid_vertex_count": n_valid,
            "total_vertex_count": n_total,
            "on_sheet_raw": None,
            "on_sheet_after_snap": None,
            "median_disp_vx": 0.0,
            "p90_disp_vx": 0.0,
            "center_vs_outer_disp_diff": 0.0,
            "passes": 0,
            "search_radius": float(args.search_radius),
            "step": float(args.snap_step),
            "threshold": float(args.m7_threshold),
        }

    if args.out_snapped_mesh:
        write_tifxyz(args.out_snapped_mesh, x, y, z, meta=meta)

    with RemoteZarrV2(
        args.ct_zarr, level=args.ct_level, cache_mb=args.cache_mb, workers=args.workers
    ) as ct_reader:
        stack, valid_mask = render_surface_stack(
            x,
            y,
            z,
            ct_reader,
            num_slices=args.num_slices,
            slice_step=args.slice_step,
            flip_normals=bool(args.flip_normals),
            scale=(sx, sy),
            uv_stride=args.uv_stride,
        )
        ct_chunks = ct_reader.chunks_fetched
        ct_bytes = ct_reader.bytes_fetched

    mesh_idx = (int(args.num_slices) - 1) // 2
    if int(args.num_slices) > 2 * int(args.coherence_delta):
        coherence = layer_coherence_score(
            stack,
            mesh_index=mesh_idx,
            delta=args.coherence_delta,
            patch_size=args.coherence_patch_size,
            max_shift=args.coherence_max_shift,
            valid_mask=valid_mask,
        )
    else:
        coherence = {
            "sheet_parallel_fraction": 0.0,
            "patch_count": 0,
            "mean_zero_shift_r": 0.0,
            "mean_best_shifted_r": 0.0,
            "patch_scores": [],
        }

    out_npy = Path(args.out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, stack)

    stack_sha256 = hashlib.sha256(np.ascontiguousarray(stack).tobytes()).hexdigest()
    valid_pixels = int(valid_mask.sum())
    total_pixels = int(valid_mask.size)

    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "renderer": "gemini_lane/stream_snap_render.py",
        "inputs": {
            "mesh_name": _sanitize_source_label(args.mesh_dir),
            "ct_source": _sanitize_source_label(args.ct_zarr),
            "m7_source": _sanitize_source_label(args.m7_zarr),
            "out_npy_name": _sanitize_source_label(args.out_npy),
        },
        "params": {
            "snap_passes": int(args.snap_passes),
            "search_radius": float(args.search_radius),
            "snap_step": float(args.snap_step),
            "m7_threshold": float(args.m7_threshold),
            "num_slices": int(args.num_slices),
            "slice_step": float(args.slice_step),
            "flip_normals": bool(args.flip_normals),
            "scale": [sx, sy],
            "uv_stride": None if args.uv_stride is None else float(args.uv_stride),
            "cache_mb": float(args.cache_mb),
        },
        "mesh_shape": mesh_shape,
        "stack_shape": [int(s) for s in stack.shape],
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "valid_pixel_fraction": float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0,
        "stack_sha256": stack_sha256,
        "chunks_fetched": int(ct_chunks + m7_chunks),
        "bytes_fetched": int(ct_bytes + m7_bytes),
        "ct_chunks_fetched": int(ct_chunks),
        "ct_bytes_fetched": int(ct_bytes),
        "m7_chunks_fetched": int(m7_chunks),
        "m7_bytes_fetched": int(m7_bytes),
        "geometry_metrics": geom_metrics,
        "coherence": coherence,
    }

    out_receipt = Path(args.out_receipt)
    out_receipt.parent.mkdir(parents=True, exist_ok=True)
    out_receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
