# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Exp T4 (Q22): Multi-Scroll CT Radiometric Shift Audit & Nyul-Udupa Standardizer.

Strictly CPU-only, zero persistent CT on disk (streamed purely in RAM) with a
hard global HTTP byte cap <= 100 MiB across all scrolls combined and <= 6 MiB
per scroll. Reuses ``RemoteZarrV2`` from ``gemini_lane/stream_snap_render.py``
read-only via a byte-budgeted subclass without modifying ``stream_snap_render.py``.

Context (``collab/claude-018`` Question ``Q22``):
- ``Studio`` (and ``v8-in`` / ``A-hn``) clips ``uint8`` CT inputs at ``200``
  and divides by ``255``, trained with affine augmentation gain in ``[0.8, 1.2]``
  and bias in ``[-15, +15]``.
- Public S3 OME-Zarr v2 level-0 volumes span ``8.64 um / 116 keV`` and
  ``9.36 um / 113 keV`` campaigns with potentially different export windows.
- This module audits 4 read controls (``PHerc0841``, ``PHerc0343``,
  ``PHerc0306B``, ``PHerc0211``) and 9 eligible unread scrolls by streaming
  2-3 deterministic interior non-empty CT chunks per scroll, computing full
  non-zero CT statistics and upper-half/upper-tercile papyrus-wall core stats,
  measuring affine drift against the pooled reference control median
  (``PHerc0841``/``0343``/``0306B``), flagging scrolls outside the ``Studio``
  envelope, and providing a reusable piecewise-linear Nyul & Udupa (1999)
  standardizer (``nyul_udupa_standardize_stack``).

Privacy invariant:
- Public JSON receipts contain ZERO local filesystem paths, ZERO tokens, and
  ZERO specific chunk indices or spatial coordinates for eligible unread scrolls.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request

import numpy as np

try:
    import requests as _requests
except ImportError:  # pragma: no cover - optional fallback
    _requests = None

requests = _requests

try:
    from .stream_snap_render import RemoteZarrV2, _sanitize_source_label
except ImportError:  # pragma: no cover - direct script execution
    from stream_snap_render import RemoteZarrV2, _sanitize_source_label


# ---------------------------------------------------------------------------
# Constants, scroll manifest, and privacy/budget rules
# ---------------------------------------------------------------------------

S3_BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
MIB = 1024 * 1024

DEFAULT_PER_SCROLL_BUDGET_BYTES = 6 * MIB
DEFAULT_GLOBAL_BUDGET_BYTES = 100 * MIB
DEFAULT_TARGET_CHUNKS = 3
DEFAULT_MIN_CHUNKS = 2
DEFAULT_MIN_NONZERO_FRAC = 0.10

REFERENCE_CONTROL_SCROLLS: tuple[str, ...] = ("PHerc0841", "PHerc0343", "PHerc0306B")
REPORTED_PERCENTILES: tuple[int, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)
AFFINE_LANDMARK_PERCENTILES: tuple[int, ...] = (10, 25, 50, 75, 90)
DEFAULT_NYUL_PERCENTILES: tuple[int, ...] = (1, 10, 25, 50, 75, 90, 99)

STUDIO_GAIN_MIN = 0.8
STUDIO_GAIN_MAX = 1.2
STUDIO_BIAS_ABS_MAX = 15.0
STUDIO_CLIP_THRESHOLD = 200
STUDIO_CLIP_HIGH_THRESHOLD = 220
STUDIO_CLIP_FRAC_MAX = 0.05

PROXY_METHODOLOGY_NOTE = (
    "Interior level-0 Zarr chunk non-zero CT (> 0) is a bounded radiometric proxy "
    "for on-sheet 16-plane surface-volume stacks, because full-volume interior "
    "chunks mix inter-wrap air pores and papyrus walls. Upper-half (> p50) and "
    "upper-tercile (> p66.7) papyrus-wall core distributions are reported "
    "alongside full non-zero CT so air-pore vs papyrus-wall modes remain distinct."
)

FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z])[A-Za-z]:\\"),
    re.compile(r"(?<![A-Za-z])[A-Za-z]:/"),
    re.compile(r"/Users/[^/]+/", re.I),
    re.compile(r"\\Users\\[^\\]+\\", re.I),
    re.compile(r"/home/[^/]+/"),
    re.compile(r"/gpfs/", re.I),
    re.compile(r"/data/visitor/", re.I),
    re.compile(r"file://", re.I),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)

FORBIDDEN_UNREAD_COORD_KEYS = frozenset(
    {
        "chunk_indices",
        "chunk_index",
        "probed_indices",
        "probed_chunks",
        "anchor_indices",
        "coords",
        "coordinates",
        "voxel_coords",
        "voxel_bounds",
        "bbox",
        "origin",
        "origin_zyx",
        "cz",
        "cy",
        "cx",
        "z_idx",
        "y_idx",
        "x_idx",
        "z_range",
        "y_range",
        "x_range",
        "chunk_keys",
    }
)

CHUNK_KEY_LEAK_RE = re.compile(r"\b0/\d+[./]\d+[./]\d+\b")


@dataclass(frozen=True)
class ScrollSpec:
    """Public S3 level-0 volume specification for one control or eligible scroll."""

    scroll_id: str
    role: str  # "control" or "eligible_unread"
    is_reference_control: bool
    voxel_um: float
    energy_kev: int
    relative_path: str

    @property
    def url(self) -> str:
        return f"{S3_BASE}/{self.relative_path.lstrip('/')}"


DEFAULT_SCROLL_SPECS: tuple[ScrollSpec, ...] = (
    # Controls (4)
    ScrollSpec(
        scroll_id="PHerc0841",
        role="control",
        is_reference_control=True,
        voxel_um=9.366,
        energy_kev=113,
        relative_path="PHerc0841/volumes/20250821151531-9.366um-1.2m-113keV-masked.zarr",
    ),
    ScrollSpec(
        scroll_id="PHerc0343",
        role="control",
        is_reference_control=True,
        voxel_um=8.640,
        energy_kev=116,
        relative_path="PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr",
    ),
    ScrollSpec(
        scroll_id="PHerc0306B",
        role="control",
        is_reference_control=True,
        voxel_um=8.640,
        energy_kev=116,
        relative_path="PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr",
    ),
    ScrollSpec(
        scroll_id="PHerc0211",
        role="control",
        is_reference_control=False,
        voxel_um=9.362,
        energy_kev=113,
        relative_path="PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr",
    ),
    # Eligible Unread (9)
    ScrollSpec(
        scroll_id="PHerc0125",
        role="eligible_unread",
        is_reference_control=False,
        voxel_um=9.362,
        energy_kev=113,
        relative_path="PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr",
    ),
    ScrollSpec(
        scroll_id="PHerc1203",
        role="eligible_unread",
        is_reference_control=False,
        voxel_um=9.362,
        energy_kev=113,
        relative_path="PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr",
    ),
    ScrollSpec(
        scroll_id="PHerc0191",
        role="eligible_unread",
        is_reference_control=False,
        voxel_um=9.362,
        energy_kev=113,
        relative_path="PHerc0191/volumes/20250821151635-9.362um-1.2m-113keV-masked.zarr",
    ),
    ScrollSpec(
        scroll_id="PHerc0826",
        role="eligible_unread",
        is_reference_control=False,
        voxel_um=9.362,
        energy_kev=113,
        relative_path="PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr",
    ),
    ScrollSpec(
        scroll_id="PHerc0846A",
        role="eligible_unread",
        is_reference_control=False,
        voxel_um=9.362,
        energy_kev=113,
        relative_path="PHerc0846A/volumes/20250728152254-9.362um-1.2m-113keV-masked.zarr",
    ),
    ScrollSpec(
        scroll_id="PHerc0175A",
        role="eligible_unread",
        is_reference_control=False,
        voxel_um=8.640,
        energy_kev=116,
        relative_path="PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr",
    ),
    ScrollSpec(
        scroll_id="PHerc0358",
        role="eligible_unread",
        is_reference_control=False,
        voxel_um=9.362,
        energy_kev=113,
        relative_path="PHerc0358/volumes/20250821151737-9.362um-1.2m-113keV-masked.zarr",
    ),
    ScrollSpec(
        scroll_id="PHerc1218",
        role="eligible_unread",
        is_reference_control=False,
        voxel_um=8.640,
        energy_kev=116,
        relative_path="PHerc1218/volumes/20250521120456-8.640um-1.2m-116keV-masked.zarr",
    ),
    ScrollSpec(
        scroll_id="PHerc1545",
        role="eligible_unread",
        is_reference_control=False,
        voxel_um=9.362,
        energy_kev=113,
        relative_path="PHerc1545/volumes/20250821151648-9.362um-1.2m-113keV-masked.zarr",
    ),
)


# ---------------------------------------------------------------------------
# 1. Privacy guardrails & strict multi-scroll HTTP byte budget
# ---------------------------------------------------------------------------

class BudgetExceededError(RuntimeError):
    """Raised before or during a fetch that would exceed per-scroll or global caps."""


def assert_no_local_paths(payload: Any) -> None:
    """Raise ValueError if serialized JSON payload contains local paths or secrets."""
    text = json.dumps(payload, sort_keys=True)
    for pat in FORBIDDEN_PATH_PATTERNS:
        m = pat.search(text)
        if m is not None:
            raise ValueError(
                f"Sanitized JSON receipt contains forbidden pattern: {m.group(0)!r}"
            )


def _walk_forbidden_keys(obj: Any, path: str = "") -> None:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            ks = str(k)
            if ks.lower() in FORBIDDEN_UNREAD_COORD_KEYS:
                raise ValueError(
                    f"Unread scroll receipt contains forbidden coordinate/chunk key {path}.{ks}"
                )
            _walk_forbidden_keys(v, f"{path}.{ks}" if path else ks)
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _walk_forbidden_keys(item, f"{path}[{i}]")


def assert_receipt_privacy(receipt: Mapping[str, Any]) -> None:
    """Enforce zero local paths, zero secrets, and zero unread chunk coordinates."""
    assert_no_local_paths(receipt)

    # Check budget summary never leaks chunk keys.
    budget_text = json.dumps(receipt.get("budget", {}), sort_keys=True)
    if CHUNK_KEY_LEAK_RE.search(budget_text):
        raise ValueError("Budget summary leaked a raw Zarr chunk key")

    scrolls = receipt.get("scrolls", [])
    if isinstance(scrolls, list):
        for entry in scrolls:
            if not isinstance(entry, Mapping):
                continue
            if entry.get("role") == "eligible_unread":
                _walk_forbidden_keys(entry, str(entry.get("scroll_id", "unread")))
                entry_text = json.dumps(entry, sort_keys=True)
                if CHUNK_KEY_LEAK_RE.search(entry_text):
                    raise ValueError(
                        f"Unread scroll {entry.get('scroll_id')!r} leaked a Zarr chunk key"
                    )


class MultiScrollByteBudget:
    """Thread-safe strict byte-cap tracker across per-scroll and global fetches."""

    def __init__(
        self,
        *,
        per_scroll_limit_bytes: int = DEFAULT_PER_SCROLL_BUDGET_BYTES,
        global_limit_bytes: int = DEFAULT_GLOBAL_BUDGET_BYTES,
    ) -> None:
        if per_scroll_limit_bytes <= 0 or global_limit_bytes <= 0:
            raise ValueError("Byte budget limits must be > 0")
        self._lock = threading.Lock()
        self.per_scroll_limit: int = int(per_scroll_limit_bytes)
        self.global_limit: int = int(global_limit_bytes)
        self._scroll_bytes: dict[str, int] = {}
        self._scroll_requests: dict[str, int] = {}

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return int(sum(self._scroll_bytes.values()))

    def scroll_bytes(self, scroll_id: str) -> int:
        with self._lock:
            return int(self._scroll_bytes.get(scroll_id, 0))

    def scroll_requests(self, scroll_id: str) -> int:
        with self._lock:
            return int(self._scroll_requests.get(scroll_id, 0))

    def remaining(self, scroll_id: str) -> int:
        with self._lock:
            cur_scroll = self._scroll_bytes.get(scroll_id, 0)
            cur_global = sum(self._scroll_bytes.values())
            rem_scroll = self.per_scroll_limit - cur_scroll
            rem_global = self.global_limit - cur_global
            return max(0, int(min(rem_scroll, rem_global)))

    def check_planned(self, scroll_id: str, planned_bytes: int, *, label: str = "") -> None:
        """Raise BudgetExceededError before downloading if planned_bytes exceeds caps."""
        pb = int(planned_bytes)
        if pb < 0:
            raise ValueError("planned_bytes must be >= 0")
        with self._lock:
            cur_scroll = self._scroll_bytes.get(scroll_id, 0)
            cur_global = sum(self._scroll_bytes.values())
            if cur_scroll + pb > self.per_scroll_limit:
                raise BudgetExceededError(
                    f"Planned fetch {label or scroll_id!r} ({pb} B) would exceed per-scroll "
                    f"limit ({cur_scroll + pb} B > {self.per_scroll_limit} B)"
                )
            if cur_global + pb > self.global_limit:
                raise BudgetExceededError(
                    f"Planned fetch {label or scroll_id!r} ({pb} B) would exceed global "
                    f"limit ({cur_global + pb} B > {self.global_limit} B)"
                )

    def charge(self, scroll_id: str, nbytes: int, *, label: str = "") -> None:
        """Charge downloaded bytes to scroll_id and global budget."""
        nb = int(nbytes)
        if nb < 0:
            raise ValueError("nbytes must be >= 0")
        with self._lock:
            cur_scroll = self._scroll_bytes.get(scroll_id, 0)
            cur_global = sum(self._scroll_bytes.values())
            if cur_scroll + nb > self.per_scroll_limit:
                raise BudgetExceededError(
                    f"Fetch {label or scroll_id!r} ({nb} B) exceeded per-scroll cap "
                    f"({cur_scroll + nb} B > {self.per_scroll_limit} B)"
                )
            if cur_global + nb > self.global_limit:
                raise BudgetExceededError(
                    f"Fetch {label or scroll_id!r} ({nb} B) exceeded global cap "
                    f"({cur_global + nb} B > {self.global_limit} B)"
                )
            self._scroll_bytes[scroll_id] = cur_scroll + nb
            self._scroll_requests[scroll_id] = self._scroll_requests.get(scroll_id, 0) + 1

    def summary(self) -> dict[str, Any]:
        with self._lock:
            total = int(sum(self._scroll_bytes.values()))
            per_scroll = {
                sid: {
                    "bytes": int(b),
                    "mib": round(b / MIB, 4),
                    "requests": int(self._scroll_requests.get(sid, 0)),
                    "within_per_scroll_limit": bool(b <= self.per_scroll_limit),
                }
                for sid, b in sorted(self._scroll_bytes.items())
            }
            all_scrolls_ok = all(b <= self.per_scroll_limit for b in self._scroll_bytes.values())
            return {
                "global_bytes": total,
                "global_mib": round(total / MIB, 4),
                "global_limit_bytes": int(self.global_limit),
                "global_limit_mib": round(self.global_limit / MIB, 4),
                "per_scroll_limit_bytes": int(self.per_scroll_limit),
                "per_scroll_limit_mib": round(self.per_scroll_limit / MIB, 4),
                "within_limits": bool(all_scrolls_ok and total <= self.global_limit),
                "per_scroll": per_scroll,
            }


class BudgetedRemoteZarrV2(RemoteZarrV2):
    """Read-only subclass of ``stream_snap_render.RemoteZarrV2`` enforcing ``MultiScrollByteBudget``.

    Reuses ``RemoteZarrV2`` metadata parsing, codec decoding, strict 404-only
    empty-chunk semantics, and in-memory LRU cache without modifying
    ``stream_snap_render.py``.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        scroll_id: str,
        budget: MultiScrollByteBudget,
        level: str | int | None = "0",
        cache_mb: float = 16.0,
        workers: int = 1,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
    ) -> None:
        self._scroll_id = str(scroll_id)
        self._budget = budget
        self.meta_raw: dict[str, Any] = {}
        super().__init__(
            root,
            level=level,
            cache_mb=cache_mb,
            workers=workers,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

    def _get_bytes(self, key: str, *, allow_404: bool = True) -> bytes | None:
        rem = self._budget.remaining(self._scroll_id)
        if rem <= 0:
            raise BudgetExceededError(
                f"Zero remaining budget for {self._scroll_id!r} before fetching key"
            )

        if self.local:
            parts = [p for p in key.split("/") if p]
            path = Path(self.root).joinpath(*parts)
            if not path.is_file():
                if allow_404:
                    return None
                raise FileNotFoundError(f"Missing local Zarr key: {key}")
            size = int(path.stat().st_size)
            self._budget.check_planned(self._scroll_id, size, label=self._scroll_id)
            data = path.read_bytes()
            self._budget.charge(self._scroll_id, len(data), label=self._scroll_id)
            if key.endswith(".zarray"):
                self.meta_raw = json.loads(data.decode("utf-8"))
            return data

        url = f"{self.root}/{key}"
        if requests is not None:
            data = self._get_http_requests_budgeted(url, allow_404=allow_404)
        else:
            data = self._get_http_urllib_budgeted(url, allow_404=allow_404)

        if data is not None and key.endswith(".zarray"):
            self.meta_raw = json.loads(data.decode("utf-8"))
        return data

    def _get_http_requests_budgeted(self, url: str, *, allow_404: bool) -> bytes | None:
        assert requests is not None
        import time

        for attempt in range(self.max_retries):
            try:
                rem = self._budget.remaining(self._scroll_id)
                if rem <= 0:
                    raise BudgetExceededError(
                        f"Zero remaining budget for {self._scroll_id!r}"
                    )
                with self._session().get(url, timeout=self.timeout, stream=True) as resp:
                    status = int(resp.status_code)
                    if status == 404 and allow_404:
                        return None
                    if status == 200:
                        cl = resp.headers.get("Content-Length")
                        if cl is not None:
                            self._budget.check_planned(
                                self._scroll_id, int(cl), label=self._scroll_id
                            )
                        buf = bytearray()
                        for piece in resp.iter_content(chunk_size=65536):
                            if not piece:
                                continue
                            if len(buf) + len(piece) > rem:
                                raise BudgetExceededError(
                                    f"Streamed response for {self._scroll_id!r} exceeded remaining "
                                    f"budget ({rem} B)"
                                )
                            buf.extend(piece)
                        data = bytes(buf)
                        self._budget.charge(self._scroll_id, len(data), label=self._scroll_id)
                        return data
                    if 400 <= status < 500:
                        resp.raise_for_status()
                        raise requests.HTTPError(
                            f"HTTP {status} for {url}; refusing to treat as empty chunk",
                            response=resp,
                        )
                    if attempt + 1 >= self.max_retries:
                        resp.raise_for_status()
                        raise requests.HTTPError(
                            f"Unexpected HTTP {status} for {url}; refusing to treat as empty chunk",
                            response=resp,
                        )
            except (BudgetExceededError, requests.HTTPError):
                raise
            except requests.RequestException:
                if attempt + 1 >= self.max_retries:
                    raise
            if self.retry_backoff > 0.0:
                time.sleep(self.retry_backoff * (attempt + 1))
        raise OSError(f"Exhausted retries fetching {url}")  # pragma: no cover

    def _get_http_urllib_budgeted(self, url: str, *, allow_404: bool) -> bytes | None:
        import time

        for attempt in range(self.max_retries):
            rem = self._budget.remaining(self._scroll_id)
            if rem <= 0:
                raise BudgetExceededError(f"Zero remaining budget for {self._scroll_id!r}")
            req = urllib.request.Request(url, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = int(getattr(resp, "status", 200))
                    if status == 200:
                        cl = resp.headers.get("Content-Length")
                        if cl is not None:
                            self._budget.check_planned(
                                self._scroll_id, int(cl), label=self._scroll_id
                            )
                        buf = bytearray()
                        while True:
                            piece = resp.read(65536)
                            if not piece:
                                break
                            if len(buf) + len(piece) > rem:
                                raise BudgetExceededError(
                                    f"Streamed response for {self._scroll_id!r} exceeded remaining "
                                    f"budget ({rem} B)"
                                )
                            buf.extend(piece)
                        data = bytes(buf)
                        self._budget.charge(self._scroll_id, len(data), label=self._scroll_id)
                        return data
                    raise urllib.error.HTTPError(
                        url, status, f"Unexpected HTTP {status}", resp.headers, None
                    )
            except BudgetExceededError:
                raise
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

    def head_chunk_bytes(self, idx: tuple[int, int, int]) -> int | None:
        """Inspect compressed chunk size (bytes) with 0 body download when available."""
        key = self._chunk_key(idx)
        if self.local:
            parts = [p for p in key.split("/") if p]
            path = Path(self.root).joinpath(*parts)
            return int(path.stat().st_size) if path.is_file() else 0
        url = f"{self.root}/{key}"
        try:
            if requests is not None:
                resp = self._session().head(url, timeout=min(self.timeout, 15.0), allow_redirects=True)
                if int(resp.status_code) == 404:
                    return 0
                if int(resp.status_code) == 200:
                    cl = resp.headers.get("Content-Length")
                    return int(cl) if cl is not None else None
                return None
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=min(self.timeout, 15.0)) as resp:
                cl = resp.headers.get("Content-Length")
                return int(cl) if cl is not None else None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return 0
            return None
        except Exception:
            return None

    def try_read_export_window_f32(self) -> tuple[float, float] | None:
        """Optionally inspect volume root metadata.json/meta.json for numeric window_f32 only."""
        for fname in ("metadata.json", "meta.json"):
            try:
                raw = self._get_bytes(fname, allow_404=True)
                if raw is None:
                    continue
                meta = json.loads(raw.decode("utf-8"))
                if not isinstance(meta, dict):
                    continue
                ze = meta.get("zarr_export")
                if isinstance(ze, dict) and "target_window_f32_min" in ze and "target_window_f32_max" in ze:
                    lo = float(ze["target_window_f32_min"])
                    hi = float(ze["target_window_f32_max"])
                    if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                        return (lo, hi)
                w = meta.get("window_f32")
                if isinstance(w, (list, tuple)) and len(w) == 2:
                    lo, hi = float(w[0]), float(w[1])
                    if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                        return (lo, hi)
            except BudgetExceededError:
                raise
            except Exception:
                continue
        return None


# ---------------------------------------------------------------------------
# 2. Exact CT distribution statistics (uint8 histogram & general float array)
# ---------------------------------------------------------------------------

def _weighted_quantile_from_sorted(
    values: np.ndarray, counts: np.ndarray, q: float
) -> float:
    """Exact linear-interpolated quantile matching ``np.percentile(..., method='linear')``."""
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("Cannot compute quantile on empty distribution")
    if total == 1:
        idx = int(np.flatnonzero(counts > 0)[0])
        return float(values[idx])
    csum = np.cumsum(counts, dtype=np.int64)
    pos = (float(q) / 100.0) * float(total - 1)
    i0 = int(math.floor(pos))
    i1 = int(math.ceil(pos))
    frac = pos - i0
    v0 = float(values[int(np.searchsorted(csum, i0, side="right"))])
    if i1 == i0:
        return v0
    v1 = float(values[int(np.searchsorted(csum, i1, side="right"))])
    return v0 + frac * (v1 - v0)


def _compute_stats_from_weighted_values(
    values: np.ndarray,
    counts: np.ndarray,
    *,
    percentiles: Sequence[float] = REPORTED_PERCENTILES,
) -> dict[str, Any]:
    """Compute exact distribution stats on positive weighted values."""
    vals = np.asarray(values, dtype=np.float64)
    cnts = np.asarray(counts, dtype=np.int64)
    pos_mask = cnts > 0
    vals = vals[pos_mask]
    cnts = cnts[pos_mask]
    total = int(cnts.sum())
    if total <= 0:
        out: dict[str, Any] = {
            "count": 0,
            "mean": None,
            "std": None,
            "mad": None,
            "robust_sigma": None,
            "skewness": None,
            "frac_ge_200": None,
            "frac_ge_220": None,
        }
        for q in percentiles:
            out[f"p{int(q):02d}"] = None
        return out

    order = np.argsort(vals, kind="mergesort")
    vals = vals[order]
    cnts = cnts[order]

    out = {"count": total}
    for q in percentiles:
        key = f"p{int(q):02d}" if float(q).is_integer() else f"p{q:g}"
        out[key] = round(_weighted_quantile_from_sorted(vals, cnts, float(q)), 4)

    w = cnts.astype(np.float64)
    mean = float(np.sum(w * vals) / total)
    diff = vals - mean
    var = float(np.sum(w * (diff ** 2)) / total)
    std = math.sqrt(max(0.0, var))
    if std > 0.0:
        m3 = float(np.sum(w * (diff ** 3)) / total)
        skewness = float(m3 / (std ** 3))
    else:
        skewness = 0.0

    med = _weighted_quantile_from_sorted(vals, cnts, 50.0)
    abs_dev = np.abs(vals - med)
    dev_order = np.argsort(abs_dev, kind="mergesort")
    mad = _weighted_quantile_from_sorted(abs_dev[dev_order], cnts[dev_order], 50.0)
    robust_sigma = 1.4826 * mad

    frac_ge_200 = float(cnts[vals >= STUDIO_CLIP_THRESHOLD].sum() / total)
    frac_ge_220 = float(cnts[vals >= STUDIO_CLIP_HIGH_THRESHOLD].sum() / total)

    out.update(
        {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "mad": round(mad, 4),
            "robust_sigma": round(robust_sigma, 4),
            "skewness": round(skewness, 4),
            "frac_ge_200": round(frac_ge_200, 6),
            "frac_ge_220": round(frac_ge_220, 6),
        }
    )
    return out


def compute_ct_stats_from_uint8_hist(
    hist_256: np.ndarray,
) -> dict[str, Any]:
    """Compute exact full non-zero CT stats + papyrus-wall core (>p50 and >p66.7) stats."""
    h = np.asarray(hist_256, dtype=np.int64)
    if h.shape != (256,):
        raise ValueError(f"Expected 256-bin uint8 histogram, got shape {h.shape}")
    total_voxels = int(h.sum())
    pos_counts = h[1:]
    pos_vals = np.arange(1, 256, dtype=np.float64)
    nonzero_count = int(pos_counts.sum())
    nonzero_fraction = float(nonzero_count / total_voxels) if total_voxels > 0 else 0.0

    nonzero_stats = _compute_stats_from_weighted_values(pos_vals, pos_counts)

    # Upper-half (> p50) papyrus-wall core stats (excludes inter-wrap air-pore mode).
    p50_raw = _weighted_quantile_from_sorted(pos_vals, pos_counts, 50.0) if nonzero_count > 0 else 0.0
    p66_7_raw = (
        _weighted_quantile_from_sorted(pos_vals, pos_counts, 200.0 / 3.0) if nonzero_count > 0 else 0.0
    )

    core_gt_p50_mask = pos_vals > p50_raw
    core_gt_p50 = _compute_stats_from_weighted_values(
        pos_vals[core_gt_p50_mask], pos_counts[core_gt_p50_mask]
    )
    core_gt_p50["threshold_p50"] = round(float(p50_raw), 4)

    upper_tercile_mask = pos_vals > p66_7_raw
    upper_tercile = _compute_stats_from_weighted_values(
        pos_vals[upper_tercile_mask], pos_counts[upper_tercile_mask]
    )
    upper_tercile["threshold_p66_7"] = round(float(p66_7_raw), 4)

    return {
        "total_voxels_sampled": total_voxels,
        "nonzero_voxel_count": nonzero_count,
        "nonzero_fraction": round(nonzero_fraction, 6),
        "nonzero_ct": nonzero_stats,
        "papyrus_wall_core_gt_p50": core_gt_p50,
        "papyrus_wall_upper_tercile_gt_p66_7": upper_tercile,
    }


def compute_ct_stats_from_values(arr: np.ndarray) -> dict[str, Any]:
    """Compute non-zero CT + papyrus-wall core stats on any numeric CT array."""
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        hist = np.bincount(a.ravel(), minlength=256).astype(np.int64)
        return compute_ct_stats_from_uint8_hist(hist)

    flat = a.ravel().astype(np.float64, copy=False)
    finite = np.isfinite(flat)
    total_voxels = int(flat.size)
    pos = flat[finite & (flat > 0.0)]
    nonzero_count = int(pos.size)
    nonzero_fraction = float(nonzero_count / total_voxels) if total_voxels > 0 else 0.0

    def _stats_1d(v: np.ndarray) -> dict[str, Any]:
        n = int(v.size)
        if n == 0:
            d: dict[str, Any] = {
                "count": 0,
                "mean": None,
                "std": None,
                "mad": None,
                "robust_sigma": None,
                "skewness": None,
                "frac_ge_200": None,
                "frac_ge_220": None,
            }
            for q in REPORTED_PERCENTILES:
                d[f"p{int(q):02d}"] = None
            return d
        pcts = np.percentile(v, REPORTED_PERCENTILES)
        mean = float(np.mean(v))
        std = float(np.std(v, ddof=0))
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        robust_sigma = 1.4826 * mad
        skewness = float(np.mean(((v - mean) / std) ** 3)) if std > 0.0 else 0.0
        d = {"count": n}
        for q, val in zip(REPORTED_PERCENTILES, pcts):
            d[f"p{int(q):02d}"] = round(float(val), 4)
        d.update(
            {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "mad": round(mad, 4),
                "robust_sigma": round(robust_sigma, 4),
                "skewness": round(skewness, 4),
                "frac_ge_200": round(float(np.mean(v >= STUDIO_CLIP_THRESHOLD)), 6),
                "frac_ge_220": round(float(np.mean(v >= STUDIO_CLIP_HIGH_THRESHOLD)), 6),
            }
        )
        return d

    nonzero_stats = _stats_1d(pos)
    p50_raw = float(np.percentile(pos, 50.0)) if nonzero_count > 0 else 0.0
    p66_7_raw = float(np.percentile(pos, 200.0 / 3.0)) if nonzero_count > 0 else 0.0
    core_gt_p50 = _stats_1d(pos[pos > p50_raw])
    core_gt_p50["threshold_p50"] = round(p50_raw, 4)
    upper_tercile = _stats_1d(pos[pos > p66_7_raw])
    upper_tercile["threshold_p66_7"] = round(p66_7_raw, 4)

    return {
        "total_voxels_sampled": total_voxels,
        "nonzero_voxel_count": nonzero_count,
        "nonzero_fraction": round(nonzero_fraction, 6),
        "nonzero_ct": nonzero_stats,
        "papyrus_wall_core_gt_p50": core_gt_p50,
        "papyrus_wall_upper_tercile_gt_p66_7": upper_tercile,
    }


# ---------------------------------------------------------------------------
# 3. Affine drift fit & Studio training envelope flagging
# ---------------------------------------------------------------------------

def fit_affine_to_reference(
    src_landmarks: Sequence[float],
    ref_landmarks: Sequence[float],
) -> dict[str, float]:
    """Least-squares affine ``y_ref = gain * x_src + bias`` on quantile landmarks."""
    x = np.asarray(src_landmarks, dtype=np.float64)
    y = np.asarray(ref_landmarks, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("src_landmarks and ref_landmarks must be matching 1D sequences of len >= 2")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Landmarks must be finite")

    xm = float(np.mean(x))
    ym = float(np.mean(y))
    dx = x - xm
    dy = y - ym
    denom = float(np.dot(dx, dx))
    if denom <= 1e-12:
        gain = 0.0
        bias = ym
    else:
        gain = float(np.dot(dx, dy) / denom)
        bias = float(ym - gain * xm)

    pred = gain * x + bias
    resid = y - pred
    rmse = float(math.sqrt(float(np.mean(resid ** 2))))
    max_abs_err = float(np.max(np.abs(resid)))
    ss_tot = float(np.dot(dy, dy))
    ss_res = float(np.dot(resid, resid))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

    return {
        "gain": round(gain, 6),
        "bias": round(bias, 4),
        "r2": round(r2, 6),
        "rmse": round(rmse, 4),
        "max_abs_err": round(max_abs_err, 4),
    }


def evaluate_studio_envelope_flags(
    *,
    gain: float,
    bias: float,
    frac_ge_200: float,
    gain_bounds: tuple[float, float] = (STUDIO_GAIN_MIN, STUDIO_GAIN_MAX),
    max_abs_bias: float = STUDIO_BIAS_ABS_MAX,
    max_frac_ge_200: float = STUDIO_CLIP_FRAC_MAX,
) -> dict[str, Any]:
    """Flag scrolls outside Studio's training envelope or clipping > 5% at >= 200."""
    flag_gain = (not math.isfinite(gain)) or (gain < gain_bounds[0]) or (gain > gain_bounds[1])
    flag_bias = (not math.isfinite(bias)) or (abs(bias) > max_abs_bias)
    flag_clip_200 = (not math.isfinite(frac_ge_200)) or (frac_ge_200 > max_frac_ge_200)

    reasons: list[str] = []
    if flag_gain:
        reasons.append(f"gain_outside_{gain_bounds[0]}_{gain_bounds[1]}")
    if flag_bias:
        reasons.append(f"abs_bias_gt_{max_abs_bias:g}")
    if flag_clip_200:
        reasons.append(f"frac_ge_200_gt_{max_frac_ge_200:g}")

    return {
        "flagged": bool(flag_gain or flag_bias or flag_clip_200),
        "flag_gain": bool(flag_gain),
        "flag_bias": bool(flag_bias),
        "flag_clip_200": bool(flag_clip_200),
        "flag_reasons": reasons,
    }


# ---------------------------------------------------------------------------
# 4. Reusable piecewise-linear Nyul & Udupa (1999) stack standardizer
# ---------------------------------------------------------------------------

def _resolve_ref_landmarks(
    ref_landmarks: Sequence[float] | Mapping[Any, float],
    percentiles: np.ndarray,
) -> np.ndarray:
    if isinstance(ref_landmarks, Mapping):
        vals: list[float] = []
        for q in percentiles:
            qf = float(q)
            qi = int(round(qf))
            candidates = (qf, qi, f"p{qi:02d}", f"p{qf:g}", str(qi), str(qf))
            found = False
            for k in candidates:
                if k in ref_landmarks:
                    vals.append(float(ref_landmarks[k]))
                    found = True
                    break
            if not found:
                raise KeyError(f"Missing reference landmark for percentile {q!r}")
        arr = np.asarray(vals, dtype=np.float64)
    else:
        arr = np.asarray(ref_landmarks, dtype=np.float64)
        if arr.ndim != 1 or arr.size != percentiles.size:
            raise ValueError(
                f"ref_landmarks length ({arr.size}) must match percentiles length ({percentiles.size})"
            )

    if not np.isfinite(arr).all():
        raise ValueError("ref_landmarks must be finite")
    if np.any(np.diff(arr) < -1e-9):
        raise ValueError("ref_landmarks must be monotonically non-decreasing")
    return arr


def _piecewise_nyul_map(
    vals: np.ndarray,
    src_lm: np.ndarray,
    ref_lm: np.ndarray,
    pcts: np.ndarray,
    out_lo: float,
    out_hi: float,
) -> np.ndarray:
    """Apply piecewise-linear Nyul-Udupa mapping with linear tail extrapolation."""
    if float(src_lm[-1] - src_lm[0]) <= 1e-12:
        ref_med = float(np.interp(50.0, pcts, ref_lm))
        ref_med = float(np.clip(ref_med, out_lo, out_hi))
        return np.full(vals.shape, ref_med, dtype=np.float64)

    # Collapse repeated source landmarks (e.g., saturated uint8 quantiles) cleanly.
    ux: list[float] = []
    uy: list[float] = []
    i = 0
    n = int(src_lm.size)
    while i < n:
        j = i + 1
        while j < n and abs(float(src_lm[j] - src_lm[i])) <= 1e-12:
            j += 1
        ux.append(float(np.mean(src_lm[i:j])))
        uy.append(float(np.mean(ref_lm[i:j])))
        i = j

    x_u = np.asarray(ux, dtype=np.float64)
    y_u = np.asarray(uy, dtype=np.float64)
    if x_u.size < 2:
        ref_med = float(np.interp(50.0, pcts, ref_lm))
        ref_med = float(np.clip(ref_med, out_lo, out_hi))
        return np.full(vals.shape, ref_med, dtype=np.float64)

    mapped = np.interp(vals, x_u, y_u)

    # Linear extrapolation below p_min and above p_max using end-segment slopes.
    below = vals < x_u[0]
    if below.any():
        slope_lo = float((y_u[1] - y_u[0]) / (x_u[1] - x_u[0]))
        if slope_lo > 0.0:
            mapped[below] = y_u[0] + slope_lo * (vals[below] - x_u[0])

    above = vals > x_u[-1]
    if above.any():
        slope_hi = float((y_u[-1] - y_u[-2]) / (x_u[-1] - x_u[-2]))
        if slope_hi > 0.0:
            mapped[above] = y_u[-1] + slope_hi * (vals[above] - x_u[-1])

    return np.clip(mapped, out_lo, out_hi)


def nyul_udupa_standardize_stack(
    stack_lhw: np.ndarray,
    valid_mask: np.ndarray | None,
    ref_landmarks: Sequence[float] | Mapping[Any, float],
    percentiles: Sequence[float] = (1, 10, 25, 50, 75, 90, 99),
    out_range: tuple[float, float] = (1.0, 255.0),
) -> np.ndarray:
    """Piecewise-linear Nyul & Udupa (1999) CT intensity standardization.

    Parameters
    ----------
    stack_lhw:
        Input CT stack (e.g. shape ``(L, H, W)`` or arbitrary ND numeric array).
    valid_mask:
        Optional boolean mask matching ``stack_lhw.shape`` (or ``(H, W)`` for a
        3D ``(L, H, W)`` stack, broadcast across ``L``). Pixels outside
        ``valid_mask``, non-finite pixels, and zero/non-positive pixels are
        strictly preserved as ``0.0``.
    ref_landmarks:
        Reference intensity landmarks corresponding to ``percentiles`` (either
        a sequence of equal length or a dict keyed by percentile values or
        ``"pXX"`` strings).
    percentiles:
        Strictly increasing percentile knots in ``[0, 100]``.
    out_range:
        Inclusive ``(lo, hi)`` clip range for valid positive standardized voxels
        (default ``(1.0, 255.0)`` so valid voxels never collide with ``0.0``).

    Returns
    -------
    np.ndarray
        ``float32`` standardized array of identical shape to ``stack_lhw``.
    """
    arr = np.asarray(stack_lhw, dtype=np.float64)
    pcts = np.asarray(percentiles, dtype=np.float64)
    if pcts.ndim != 1 or pcts.size < 2:
        raise ValueError("percentiles must be a 1D sequence of at least 2 values")
    if not np.isfinite(pcts).all() or np.any(pcts < 0.0) or np.any(pcts > 100.0):
        raise ValueError("percentiles must be finite and within [0, 100]")
    if np.any(np.diff(pcts) <= 0.0):
        raise ValueError("percentiles must be strictly increasing")

    out_lo, out_hi = float(out_range[0]), float(out_range[1])
    if not (math.isfinite(out_lo) and math.isfinite(out_hi)) or out_lo > out_hi:
        raise ValueError(f"Invalid out_range: {out_range!r}")

    ref_lm = _resolve_ref_landmarks(ref_landmarks, pcts)

    out = np.zeros(arr.shape, dtype=np.float32)
    if arr.size == 0:
        return out

    fg = np.isfinite(arr) & (arr > 0.0)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape == arr.shape:
            mask_b = vm
        elif arr.ndim == 3 and vm.ndim == 2 and vm.shape == arr.shape[1:]:
            mask_b = np.broadcast_to(vm[None, :, :], arr.shape)
        else:
            try:
                mask_b = np.broadcast_to(vm, arr.shape)
            except ValueError as exc:
                raise ValueError(
                    f"valid_mask shape {vm.shape} is not compatible with stack shape {arr.shape}"
                ) from exc
        fg &= mask_b

    if not fg.any():
        return out

    vals = arr[fg]
    src_lm = np.percentile(vals, pcts)
    mapped = _piecewise_nyul_map(vals, src_lm, ref_lm, pcts, out_lo, out_hi)
    out[fg] = mapped.astype(np.float32)
    out[~fg] = np.float32(0.0)
    return out


def simulate_nyul_on_uint8_hist(
    hist_256: np.ndarray,
    ref_landmarks: Sequence[float] | Mapping[Any, float],
    percentiles: Sequence[float] = DEFAULT_NYUL_PERCENTILES,
    out_range: tuple[float, float] = (1.0, 255.0),
) -> dict[str, Any]:
    """Evaluate exact post-Nyul-Udupa distribution stats on a 256-bin uint8 histogram."""
    h = np.asarray(hist_256, dtype=np.int64)
    pos_counts = h[1:]
    pos_vals = np.arange(1, 256, dtype=np.float64)
    if int(pos_counts.sum()) == 0:
        return _compute_stats_from_weighted_values(pos_vals, pos_counts)
    pcts = np.asarray(percentiles, dtype=np.float64)
    ref_lm = _resolve_ref_landmarks(ref_landmarks, pcts)
    src_lm = np.asarray(
        [_weighted_quantile_from_sorted(pos_vals, pos_counts, float(q)) for q in pcts],
        dtype=np.float64,
    )
    mapped_vals = _piecewise_nyul_map(
        pos_vals, src_lm, ref_lm, pcts, float(out_range[0]), float(out_range[1])
    )
    return _compute_stats_from_weighted_values(mapped_vals, pos_counts)


# ---------------------------------------------------------------------------
# 5. Deterministic interior chunk selection & scroll audit
# ---------------------------------------------------------------------------

def _chunk_grid_dims(shape: tuple[int, int, int], chunks: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return full-chunk grid counts along (z, y, x), falling back to ceil if small."""
    dims: list[int] = []
    for s, c in zip(shape, chunks):
        full = int(s) // int(c)
        dims.append(full if full >= 1 else max(1, math.ceil(int(s) / int(c))))
    return (dims[0], dims[1], dims[2])


def build_interior_anchor_probe_plan(
    shape: tuple[int, int, int],
    chunks: tuple[int, int, int],
    *,
    target_chunks: int = DEFAULT_TARGET_CHUNKS,
) -> list[list[tuple[int, int, int]]]:
    """Build deterministic interior anchor bands with center +/- 1 fallback probes.

    Each band returns an ordered list of candidate ``(cz, cy, cx)`` indices:
    the anchor itself first, followed by deterministic ``+/- 1`` (then ``+/- 2``
    along y/x) interior neighbors.
    """
    nz, ny, nx = _chunk_grid_dims(shape, chunks)
    cz_mid = nz // 2
    cy_mid = ny // 2
    cx_mid = nx // 2

    if nz >= 4:
        cz_lo = max(1, min(nz - 2, int(round(0.35 * nz))))
        cz_hi = max(1, min(nz - 2, int(round(0.65 * nz))))
    else:
        cz_lo = max(0, cz_mid - 1)
        cz_hi = min(nz - 1, cz_mid + 1)

    base_anchors: list[tuple[int, int, int]] = []
    for cz in (cz_mid, cz_lo, cz_hi):
        cand = (cz, cy_mid, cx_mid)
        if cand not in base_anchors:
            base_anchors.append(cand)

    # Extra fallback anchors if a tiny test volume has < target_chunks unique z planes.
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        if len(base_anchors) >= target_chunks:
            break
        ny_i = cy_mid + dy
        nx_i = cx_mid + dx
        if 0 <= ny_i < ny and 0 <= nx_i < nx:
            cand = (cz_mid, ny_i, nx_i)
            if cand not in base_anchors:
                base_anchors.append(cand)

    offsets = (
        (0, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (0, 1, 1),
        (0, -1, -1),
        (0, 1, -1),
        (0, -1, 1),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 2, 0),
        (0, -2, 0),
        (0, 0, 2),
        (0, 0, -2),
    )

    bands: list[list[tuple[int, int, int]]] = []
    for az, ay, ax in base_anchors[:target_chunks]:
        cands: list[tuple[int, int, int]] = []
        for dz, dy, dx in offsets:
            iz, iy, ix = az + dz, ay + dy, ax + dx
            if 0 <= iz < nz and 0 <= iy < ny and 0 <= ix < nx:
                cands.append((iz, iy, ix))
        bands.append(cands)
    return bands


def _valid_chunk_slice(
    arr: np.ndarray,
    idx: tuple[int, int, int],
    shape: tuple[int, int, int],
    chunks: tuple[int, int, int],
) -> np.ndarray:
    """Trim edge-chunk padding beyond the Zarr array bounds."""
    s0 = [i * c for i, c in zip(idx, chunks)]
    e0 = [min(sh - st, c) for st, sh, c in zip(s0, shape, chunks)]
    return arr[: e0[0], : e0[1], : e0[2]]


def audit_single_scroll(
    spec: ScrollSpec,
    budget: MultiScrollByteBudget,
    *,
    root_override: str | os.PathLike[str] | None = None,
    level: str | int | None = "0",
    target_chunks: int = DEFAULT_TARGET_CHUNKS,
    min_chunks: int = DEFAULT_MIN_CHUNKS,
    min_nonzero_frac: float = DEFAULT_MIN_NONZERO_FRAC,
    timeout: float = 30.0,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Stream 2-3 interior non-empty CT chunks for one scroll under strict byte caps.

    Returns ``(scroll_record, uint8_hist_or_none)``. Specific chunk indices are
    NEVER stored for ``role == "eligible_unread"`` scrolls.
    """
    root = str(root_override) if root_override is not None else spec.url
    with BudgetedRemoteZarrV2(
        root,
        scroll_id=spec.scroll_id,
        budget=budget,
        level=level,
        cache_mb=8.0,
        workers=1,
        timeout=timeout,
    ) as reader:
        window_f32 = reader.try_read_export_window_f32()
        zarray_meta = {
            "shape": list(reader.shape),
            "chunks": list(reader.chunks),
            "dtype": str(reader.dtype),
            "order": reader.order,
            "dimension_separator": reader.dimension_separator,
            "compressor_id": (reader.meta_raw.get("compressor") or {}).get("id"),
            "uncompressed_chunk_mib": round(reader._expected_chunk_bytes / MIB, 4),
            "window_f32": [round(window_f32[0], 6), round(window_f32[1], 6)]
            if window_f32 is not None
            else None,
        }

        bands = build_interior_anchor_probe_plan(
            reader.shape, reader.chunks, target_chunks=target_chunks
        )
        visited: set[tuple[int, int, int]] = set()
        accepted_indices: list[tuple[int, int, int]] = []
        probed_count = 0
        skipped_empty_or_sparse = 0
        stopped_by_budget = False

        is_u8 = reader.dtype == np.uint8
        hist_256 = np.zeros(256, dtype=np.int64) if is_u8 else None
        non_u8_arrays: list[np.ndarray] = []

        for band in bands:
            if len(accepted_indices) >= target_chunks:
                break
            for idx in band:
                if idx in visited:
                    continue
                visited.add(idx)

                # Pre-check remaining budget with 0-body HEAD when available.
                rem = budget.remaining(spec.scroll_id)
                head_sz = reader.head_chunk_bytes(idx)
                if head_sz == 0:
                    # 404 missing chunk; skip without wasting a GET.
                    probed_count += 1
                    skipped_empty_or_sparse += 1
                    continue
                if head_sz is not None and head_sz > rem:
                    if len(accepted_indices) >= min_chunks:
                        stopped_by_budget = True
                        break
                    # Try next candidate or raise if no chunks can fit.
                    if len(accepted_indices) == 0:
                        budget.check_planned(spec.scroll_id, head_sz, label=spec.scroll_id)
                    stopped_by_budget = True
                    break

                probed_count += 1
                try:
                    chunk = reader.get_chunk(idx)
                except BudgetExceededError:
                    if len(accepted_indices) >= min_chunks:
                        stopped_by_budget = True
                        break
                    raise

                if chunk is None:
                    skipped_empty_or_sparse += 1
                    continue

                valid_sub = _valid_chunk_slice(chunk, idx, reader.shape, reader.chunks)
                nz_frac = float(np.count_nonzero(valid_sub > 0) / valid_sub.size) if valid_sub.size > 0 else 0.0
                if nz_frac < min_nonzero_frac:
                    skipped_empty_or_sparse += 1
                    continue

                accepted_indices.append(idx)
                if is_u8 and hist_256 is not None:
                    hist_256 += np.bincount(valid_sub.ravel(), minlength=256).astype(np.int64)
                else:
                    non_u8_arrays.append(valid_sub.ravel())
                break  # 1 accepted non-empty chunk per interior z-band

            if stopped_by_budget:
                break

        # If fewer than min_chunks were accepted (e.g., sparse scroll), sweep remaining candidates.
        if len(accepted_indices) < min_chunks and not stopped_by_budget:
            for band in bands:
                for idx in band:
                    if len(accepted_indices) >= min_chunks:
                        break
                    if idx in visited:
                        continue
                    visited.add(idx)
                    rem = budget.remaining(spec.scroll_id)
                    head_sz = reader.head_chunk_bytes(idx)
                    if head_sz == 0:
                        probed_count += 1
                        skipped_empty_or_sparse += 1
                        continue
                    if head_sz is not None and head_sz > rem:
                        stopped_by_budget = True
                        break
                    probed_count += 1
                    try:
                        chunk = reader.get_chunk(idx)
                    except BudgetExceededError:
                        if len(accepted_indices) >= 1:
                            stopped_by_budget = True
                            break
                        raise
                    if chunk is None:
                        skipped_empty_or_sparse += 1
                        continue
                    valid_sub = _valid_chunk_slice(chunk, idx, reader.shape, reader.chunks)
                    nz_frac = float(np.count_nonzero(valid_sub > 0) / valid_sub.size) if valid_sub.size > 0 else 0.0
                    if nz_frac < min_nonzero_frac:
                        skipped_empty_or_sparse += 1
                        continue
                    accepted_indices.append(idx)
                    if is_u8 and hist_256 is not None:
                        hist_256 += np.bincount(valid_sub.ravel(), minlength=256).astype(np.int64)
                    else:
                        non_u8_arrays.append(valid_sub.ravel())

        if len(accepted_indices) == 0:
            raise RuntimeError(
                f"No interior non-empty CT chunks (>= {min_nonzero_frac:.0%} non-zero) "
                f"found for {spec.scroll_id}"
            )

        if is_u8 and hist_256 is not None:
            stats = compute_ct_stats_from_uint8_hist(hist_256)
        else:
            combined = np.concatenate(non_u8_arrays)
            stats = compute_ct_stats_from_values(combined)

        bytes_used = budget.scroll_bytes(spec.scroll_id)
        record: dict[str, Any] = {
            "scroll_id": spec.scroll_id,
            "role": spec.role,
            "is_reference_control": bool(spec.is_reference_control),
            "voxel_um": float(spec.voxel_um),
            "energy_kev": int(spec.energy_kev),
            "volume_source": _sanitize_source_label(root),
            "zarray": zarray_meta,
            "sampling": {
                "chunks_accepted": int(len(accepted_indices)),
                "chunks_probed": int(probed_count),
                "chunks_skipped_sparse_or_missing": int(skipped_empty_or_sparse),
                "min_nonzero_frac_threshold": float(min_nonzero_frac),
                "stopped_by_per_scroll_budget": bool(stopped_by_budget),
                "bytes_streamed": int(bytes_used),
                "mib_streamed": round(bytes_used / MIB, 4),
            },
            **stats,
        }
        # Strictly omit chunk indices for eligible_unread scrolls; only record on controls.
        if spec.role == "control":
            record["sampling"]["control_chunk_indices"] = [list(i) for i in accepted_indices]

        return record, hist_256


# ---------------------------------------------------------------------------
# 6. Full 13-scroll audit orchestrator & report formatter
# ---------------------------------------------------------------------------

def run_radiometric_shift_audit(
    specs: Sequence[ScrollSpec] = DEFAULT_SCROLL_SPECS,
    *,
    root_overrides: Mapping[str, str | os.PathLike[str]] | None = None,
    per_scroll_limit_bytes: int = DEFAULT_PER_SCROLL_BUDGET_BYTES,
    global_limit_bytes: int = DEFAULT_GLOBAL_BUDGET_BYTES,
    target_chunks: int = DEFAULT_TARGET_CHUNKS,
    min_chunks: int = DEFAULT_MIN_CHUNKS,
    min_nonzero_frac: float = DEFAULT_MIN_NONZERO_FRAC,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Execute the multi-scroll CT radiometric shift audit purely in RAM on CPU."""
    budget = MultiScrollByteBudget(
        per_scroll_limit_bytes=per_scroll_limit_bytes,
        global_limit_bytes=global_limit_bytes,
    )
    overrides = dict(root_overrides or {})

    scroll_records: list[dict[str, Any]] = []
    hist_by_scroll: dict[str, np.ndarray | None] = {}

    for spec in specs:
        rec, hist = audit_single_scroll(
            spec,
            budget,
            root_override=overrides.get(spec.scroll_id),
            target_chunks=target_chunks,
            min_chunks=min_chunks,
            min_nonzero_frac=min_nonzero_frac,
            timeout=timeout,
        )
        scroll_records.append(rec)
        hist_by_scroll[spec.scroll_id] = hist

    # Pooled reference control median across PHerc0841 / PHerc0343 / PHerc0306B.
    ref_records = [r for r in scroll_records if r["is_reference_control"]]
    if not ref_records:
        raise ValueError("At least one reference control scroll is required")

    ref_landmarks_all: dict[str, float] = {}
    for q in REPORTED_PERCENTILES:
        key = f"p{q:02d}"
        vals = [float(r["nonzero_ct"][key]) for r in ref_records if r["nonzero_ct"][key] is not None]
        ref_landmarks_all[key] = round(float(np.median(vals)), 4)

    ref_affine_vec = [ref_landmarks_all[f"p{q:02d}"] for q in AFFINE_LANDMARK_PERCENTILES]
    ref_nyul_vec = [ref_landmarks_all[f"p{q:02d}"] for q in DEFAULT_NYUL_PERCENTILES]

    # Also compute pooled reference median on papyrus-wall core (> p50).
    ref_core_landmarks: dict[str, float] = {}
    for q in REPORTED_PERCENTILES:
        key = f"p{q:02d}"
        vals = [
            float(r["papyrus_wall_core_gt_p50"][key])
            for r in ref_records
            if r["papyrus_wall_core_gt_p50"][key] is not None
        ]
        if vals:
            ref_core_landmarks[key] = round(float(np.median(vals)), 4)

    flagged_scrolls: list[str] = []
    flagged_unread: list[str] = []

    for rec in scroll_records:
        nz = rec["nonzero_ct"]
        src_affine_vec = [float(nz[f"p{q:02d}"]) for q in AFFINE_LANDMARK_PERCENTILES]
        affine = fit_affine_to_reference(src_affine_vec, ref_affine_vec)
        flags = evaluate_studio_envelope_flags(
            gain=affine["gain"],
            bias=affine["bias"],
            frac_ge_200=float(nz["frac_ge_200"]),
        )
        rec["affine_to_ref_control"] = affine
        rec["studio_envelope_check"] = flags

        # Evaluate post-Nyul-Udupa standardization alignment on the scroll's exact uint8 histogram.
        hist = hist_by_scroll.get(rec["scroll_id"])
        if hist is not None:
            post_nyul = simulate_nyul_on_uint8_hist(
                hist, ref_nyul_vec, percentiles=DEFAULT_NYUL_PERCENTILES, out_range=(1.0, 255.0)
            )
            post_affine = fit_affine_to_reference(
                [float(post_nyul[f"p{q:02d}"]) for q in AFFINE_LANDMARK_PERCENTILES],
                ref_affine_vec,
            )
            post_flags = evaluate_studio_envelope_flags(
                gain=post_affine["gain"],
                bias=post_affine["bias"],
                frac_ge_200=float(post_nyul["frac_ge_200"]),
            )
            rec["post_nyul_udupa_validation"] = {
                "p10": post_nyul["p10"],
                "p25": post_nyul["p25"],
                "p50": post_nyul["p50"],
                "p75": post_nyul["p75"],
                "p90": post_nyul["p90"],
                "frac_ge_200": post_nyul["frac_ge_200"],
                "affine_to_ref_after": post_affine,
                "flagged_after_nyul": post_flags["flagged"],
            }

        if flags["flagged"]:
            flagged_scrolls.append(rec["scroll_id"])
            if rec["role"] == "eligible_unread":
                flagged_unread.append(rec["scroll_id"])

    receipt: dict[str, Any] = {
        "schema_version": "q22_radiometric_shift_audit_v1",
        "experiment": "Exp T4 (Q22): Multi-Scroll CT Radiometric Shift Audit & Nyul-Udupa Standardizer",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "execution_mode": "cpu_only_in_ram_streaming",
        "methodology_caveat": PROXY_METHODOLOGY_NOTE,
        "studio_training_envelope": {
            "clip_u8_threshold": STUDIO_CLIP_THRESHOLD,
            "divide_by": 255.0,
            "gain_bounds": [STUDIO_GAIN_MIN, STUDIO_GAIN_MAX],
            "bias_abs_max": STUDIO_BIAS_ABS_MAX,
            "clip_200_frac_flag_threshold": STUDIO_CLIP_FRAC_MAX,
            "affine_fit_percentiles": list(AFFINE_LANDMARK_PERCENTILES),
        },
        "reference_control_baseline": {
            "reference_scrolls": [r["scroll_id"] for r in ref_records],
            "pooled_median_nonzero_ct_landmarks": ref_landmarks_all,
            "nyul_udupa_default_percentiles": list(DEFAULT_NYUL_PERCENTILES),
            "nyul_udupa_ref_landmarks": ref_nyul_vec,
            "pooled_median_papyrus_wall_core_gt_p50_landmarks": ref_core_landmarks,
        },
        "summary": {
            "total_scrolls_audited": len(scroll_records),
            "control_count": sum(1 for r in scroll_records if r["role"] == "control"),
            "eligible_unread_count": sum(
                1 for r in scroll_records if r["role"] == "eligible_unread"
            ),
            "flagged_scroll_count": len(flagged_scrolls),
            "flagged_scrolls": flagged_scrolls,
            "flagged_eligible_unread_count": len(flagged_unread),
            "flagged_eligible_unread": flagged_unread,
            "all_scrolls_unflagged_after_nyul": all(
                not r.get("post_nyul_udupa_validation", {}).get("flagged_after_nyul", False)
                for r in scroll_records
            ),
        },
        "budget": budget.summary(),
        "scrolls": scroll_records,
    }

    assert_receipt_privacy(receipt)
    return receipt


def format_comparison_table(receipt: Mapping[str, Any]) -> str:
    """Format a compact Markdown comparison table across all audited scrolls."""
    header = (
        "| Scroll | Role | Voxel/keV | p10 | p25 | p50 | p75 | p90 | p99 | "
        "Mean | RobSig | >=200 | Core>p50 Med | Gain | Bias | Flagged | MiB |"
    )
    sep = (
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"
    )
    lines = [header, sep]
    for r in receipt.get("scrolls", []):
        nz = r["nonzero_ct"]
        core = r["papyrus_wall_core_gt_p50"]
        aff = r["affine_to_ref_control"]
        flg = r["studio_envelope_check"]
        role_short = "ref-ctrl" if r["is_reference_control"] else ("ctrl" if r["role"] == "control" else "unread")
        flag_str = "YES (" + ",".join(flg["flag_reasons"]) + ")" if flg["flagged"] else "no"
        lines.append(
            f"| {r['scroll_id']} | {role_short} | {r['voxel_um']:.3f}um/{r['energy_kev']}keV | "
            f"{nz['p10']:.1f} | {nz['p25']:.1f} | {nz['p50']:.1f} | {nz['p75']:.1f} | "
            f"{nz['p90']:.1f} | {nz['p99']:.1f} | {nz['mean']:.1f} | {nz['robust_sigma']:.1f} | "
            f"{100.0 * nz['frac_ge_200']:.2f}% | {core['p50']:.1f} | {aff['gain']:.3f} | "
            f"{aff['bias']:+.1f} | {flag_str} | {r['sampling']['mib_streamed']:.2f} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exp T4 (Q22): Multi-scroll CT radiometric shift audit (CPU-only, <=100 MiB)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "evidence" / "q22_radiometric_shift_audit.json",
        help="Output JSON receipt path (default: gemini_lane/evidence/q22_radiometric_shift_audit.json)",
    )
    parser.add_argument(
        "--per-scroll-mib",
        type=float,
        default=6.0,
        help="Per-scroll HTTP byte cap in MiB (default: 6.0)",
    )
    parser.add_argument(
        "--global-mib",
        type=float,
        default=100.0,
        help="Global HTTP byte cap in MiB across all scrolls (default: 100.0)",
    )
    parser.add_argument(
        "--target-chunks",
        type=int,
        default=DEFAULT_TARGET_CHUNKS,
        help="Target non-empty interior chunks per scroll (default: 3)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    receipt = run_radiometric_shift_audit(
        per_scroll_limit_bytes=int(round(args.per_scroll_mib * MIB)),
        global_limit_bytes=int(round(args.global_mib * MIB)),
        target_chunks=int(args.target_chunks),
    )
    assert_receipt_privacy(receipt)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print(format_comparison_table(receipt))
    b = receipt["budget"]
    print(
        f"\nStreamed total: {b['global_mib']:.4f} MiB / {b['global_limit_mib']:.1f} MiB "
        f"(within_limits={b['within_limits']})"
    )
    print(
        f"Flagged scrolls ({receipt['summary']['flagged_scroll_count']}): "
        f"{receipt['summary']['flagged_scrolls']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
