# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Exp T15-GridQuality-Audit: shear / stretch / roughness of tifxyz grids (CPU-only, in-RAM, byte-budgeted).

Reproduces the formulas of ``firstletters-colab/methods_ledger/code/grid_quality.py`` (claude-023, ledger S8) with
NumPy / SciPy / tifffile only, and adds the stretch and area-inflation figures that claude-023 reports in prose:

* **shear** - angle between the two grid axes minus 90 degrees, per quad ``a = P[r, c+1] - P[r, c]`` (columns),
  ``b = P[r+1, c] - P[r, c]`` (rows); signed (5th / 95th), absolute (median / 95th), share of cells and of area
  above 10 and 20 degrees. Reported over 1 cell (Claude's script on disk) and over 8 cells (about 1.5 mm at 20 vx,
  the span quoted in claude-023).
* **stretch** - ``|a| / nominal_u`` and ``|b| / nominal_v`` with ``nominal = 1 / scale`` from ``meta.json``
  (20 vx for ``scale = 0.05``): 5th / 50th / 95th, share above 1.5x and share off by more than 1.5x either way,
  and the area inflation ``sum |a x b| / (cells * nominal_u * nominal_v)``.
* **roughness** - angle between a cell normal and the normalised mean normal of its ``k x k`` (default 9 x 9)
  neighbourhood, over cells whose neighbourhood is more than 60 % valid: median / 95th, share with a negative
  dot product (flip), above 20 and above 45 degrees.

Validity follows Claude's script (``z > 0`` and finite) and additionally requires finite ``x`` and ``y``; a quad is
used when its three corners ``(r, c)``, ``(r, c+1)``, ``(r+1, c)`` are valid.

Network: every public tifxyz (``x.tif``, ``y.tif``, ``z.tif``, ``meta.json``) is read into RAM and parsed from
bytes; nothing but the final JSON receipt is written. All HTTP goes through :class:`BudgetedHttp`, a hard cap of
60 MiB (counting retries, redirects' bodies and error bodies). A non-200 answer (for example a Hugging Face
rate limit) is recorded with its exact status and is never silently dropped.

Privacy: the receipt holds aggregates only. No coordinates, no ``bbox``, no ``uv_crop``, no hotspots, no local paths.
Strictly CPU-only: no torch / cupy / jax / tensorflow / pycuda import.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tifffile
from scipy.ndimage import uniform_filter

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SCHEMA = "grid_quality_audit/v1"
EXPERIMENT = "Exp T15-GridQuality-Audit"
DEFAULT_OUT = HERE / "evidence" / "grid_quality_audit.json"
BUDGET_BYTES = 60 * 1024 * 1024
ROUGHNESS_WINDOW = 9
SHEAR_SPAN_CELLS = 8
MIN_NEIGHBOURHOOD_VALID = 0.6
CHANNELS = ("x.tif", "y.tif", "z.tif", "meta.json")

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
PHERC0211_WINNER = (
    f"{S3}/PHerc0211/segments/20261028201525-crop1711-lines123-v9/mesh/"
    "20261028201525-on-20250821151803-9.362um.tifxyz/"
)
HF_FITS = (
    "https://huggingface.co/datasets/rodriguescarson/eligible-scroll-spiral-fits/resolve/"
    "63c641ba1dc794073208c24f7f88fa59c4f77621/"
)
HF_0826_BANDS = (
    "2026-09-15_PHerc0826_slice-6528-7328_0-patch",
    "2026-09-15_PHerc0826_slice-8928-9728_0-patch",
    "2026-09-15_PHerc0826_slice-9328-10128_0-patch",
)
HF_0191_RUN = "2026-09-15_PHerc0191_slice-11600-12400_0-patch_w90"
PSCAMILLO_MESHES = (
    "https://raw.githubusercontent.com/pscamillo/vesuvius-eligible-meshes/"
    "66000516c97e043b696ce874f145e29638109c91/meshes/"
)
# Windings the Gemini G1 / triage lanes actually read (evidence/pherc0826_1203t2_onsheet_triage.json,
# evidence/pherc0125_0191_onsheet_triage.json, evidence/firstletters_rtx4060_screen_ledger.json).
G1_WINDINGS = ("w020", "w030", "w040", "w050")
G1_0826_BAND = HF_0826_BANDS[1]
N_WINDINGS = 90
GENERIC_PRIVACY_TOKENS = ("uv_crop", "top_hotspots", "bbox", "hotspot", "C:\\", "AppData", ".t3/worktrees")


# ---------------------------------------------------------------------------
# 1. Byte-budgeted HTTP (in RAM, no persistent downloads)
# ---------------------------------------------------------------------------

class ByteBudgetExceeded(RuntimeError):
    """Raised before a request (declared length) or mid-stream when the 60 MiB cap would be crossed."""


class BudgetedHttp:
    """GET helper with a hard total byte cap, exact status accounting and bounded retries."""

    def __init__(
        self,
        budget_bytes: int = BUDGET_BYTES,
        *,
        session: Any = None,
        retries: int = 2,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        min_interval_s: float = 0.0,
    ) -> None:
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        self.budget_bytes = int(budget_bytes)
        self.bytes_used = 0
        self.requests = 0
        self.retries = int(retries)
        self.timeout = float(timeout)
        self.status_counts: dict[str, int] = {}
        self._sleep = sleep
        self._min_interval = float(min_interval_s)
        self._last = 0.0
        self._session = session

    @property
    def session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _count(self, key: str) -> None:
        self.status_counts[key] = self.status_counts.get(key, 0) + 1

    def get(self, url: str) -> tuple[int, bytes | None]:
        """Return ``(status, body)``; ``body`` is ``None`` unless the status is 200. Status 0 = transport error."""
        status = 0
        for attempt in range(self.retries + 1):
            if self._min_interval:
                wait = self._min_interval - (time.monotonic() - self._last)
                if wait > 0:
                    self._sleep(wait)
            self._last = time.monotonic()
            if self.bytes_used >= self.budget_bytes:
                raise ByteBudgetExceeded(f"{self.bytes_used} >= {self.budget_bytes} bytes before request")
            self.requests += 1
            retry_after = 0.0
            try:
                resp = self.session.get(
                    url, stream=True, timeout=self.timeout, headers={"Accept-Encoding": "identity"}
                )
            except Exception:  # transport error: counted, retried, then reported as status 0
                status = 0
                self._count("transport_error")
            else:
                status = int(resp.status_code)
                self._count(str(status))
                try:
                    declared = resp.headers.get("Content-Length")
                    if declared is not None and str(declared).isdigit():
                        if self.bytes_used + int(declared) > self.budget_bytes:
                            raise ByteBudgetExceeded(
                                f"declared {declared} bytes would cross the {self.budget_bytes}-byte cap"
                            )
                    chunks: list[bytes] = []
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if not chunk:
                            continue
                        self.bytes_used += len(chunk)
                        if self.bytes_used > self.budget_bytes:
                            raise ByteBudgetExceeded(f"stream crossed the {self.budget_bytes}-byte cap")
                        chunks.append(chunk)
                    if status == 200:
                        return 200, b"".join(chunks)
                    ra = resp.headers.get("Retry-After")
                    if ra is not None and re.fullmatch(r"\d+(\.\d+)?", str(ra)):
                        retry_after = min(float(ra), 20.0)
                finally:
                    close = getattr(resp, "close", None)
                    if close:
                        close()
                if status not in (429, 500, 502, 503, 504):
                    return status, None
            if attempt < self.retries:
                self._sleep(retry_after or min(2.0 ** attempt, 8.0))
        return status, None

    def summary(self) -> dict[str, Any]:
        return {
            "budget_bytes": self.budget_bytes,
            "http_bytes": self.bytes_used,
            "http_mib": round(self.bytes_used / 2 ** 20, 3),
            "requests": self.requests,
            "status_counts": dict(sorted(self.status_counts.items())),
            "persistent_downloads_on_disk": False,
        }


def fetch_tifxyz(http: BudgetedHttp, base_url: str) -> dict[str, Any]:
    """Fetch one tifxyz directory into RAM. Returns ``{"x","y","z","meta"}`` or ``{"error": {...}}``."""
    base = base_url if base_url.endswith("/") else base_url + "/"
    got: dict[str, bytes] = {}
    for name in CHANNELS:
        status, body = http.get(base + name)
        if status != 200 or body is None:
            return {"error": {"file": name, "http_status": status}}
        got[name] = body
    try:
        arrs = [np.asarray(tifffile.imread(io.BytesIO(got[f"{c}.tif"]))) for c in "xyz"]
        meta = json.loads(got["meta.json"].decode("utf-8"))
    except Exception as exc:  # corrupt TIFF / JSON: report the exception class only
        return {"error": {"file": "decode", "http_status": 200, "exception": type(exc).__name__}}
    if not (arrs[0].shape == arrs[1].shape == arrs[2].shape and arrs[0].ndim == 2):
        return {"error": {"file": "decode", "http_status": 200, "exception": "ShapeMismatch"}}
    return {"x": arrs[0], "y": arrs[1], "z": arrs[2], "meta": meta}


# ---------------------------------------------------------------------------
# 2. Grid quality (Claude's formulas) per cell
# ---------------------------------------------------------------------------

def nominal_step_vx(meta: Mapping[str, Any] | None, default: float = 20.0) -> tuple[float, float]:
    """``(nominal_u, nominal_v)`` = ``1 / scale`` along (columns, rows); ``scale: [0.05, 0.05]`` -> 20 vx."""
    scale = (meta or {}).get("scale") if isinstance(meta, Mapping) else None
    try:
        if isinstance(scale, (list, tuple)) and len(scale) == 2:
            su, sv = float(scale[0]), float(scale[1])
        elif scale is not None:
            su = sv = float(scale)
        else:
            return float(default), float(default)
    except (TypeError, ValueError):
        return float(default), float(default)
    if not (su > 0 and sv > 0 and np.isfinite(su) and np.isfinite(sv)):
        return float(default), float(default)
    return 1.0 / su, 1.0 / sv


def _valid_vertices(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 0)


def _signed_shear_deg(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed ``angle(a, b) - 90`` (degrees) and the cross-product magnitude, over cells selected by ``m``."""
    ca = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-9)
    ang = np.degrees(np.arccos(np.clip(ca[m], -1.0, 1.0)))
    area = np.linalg.norm(np.cross(a, b), axis=-1)[m]
    return ang - 90.0, area


def grid_quality_cells(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    nominal: tuple[float, float] = (20.0, 20.0),
    window: int = ROUGHNESS_WINDOW,
    span: int = SHEAR_SPAN_CELLS,
) -> dict[str, np.ndarray]:
    """Per-cell arrays (valid cells only) for shear, stretch, area and roughness of one tifxyz grid."""
    xf, yf, zf = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    P = np.stack([xf, yf, zf], axis=-1)
    ok = _valid_vertices(xf, yf, zf)
    out: dict[str, np.ndarray] = {}
    empty = np.zeros(0)
    if P.shape[0] < 2 or P.shape[1] < 2:
        return {k: empty for k in ("shear1", "shear1_area", "shear8", "shear8_area", "stretch_u", "stretch_v",
                                   "area", "rough_deg", "rough_dot")}

    a = P[:-1, 1:] - P[:-1, :-1]          # along columns (u)
    b = P[1:, :-1] - P[:-1, :-1]          # along rows (v)
    n = np.cross(a, b)
    m = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1]
    area_all = np.linalg.norm(n, axis=-1)
    nn = n / np.maximum(area_all, 1e-9)[..., None]
    nn[~m] = 0.0

    k = int(window)
    sm = np.stack([uniform_filter(nn[..., i], k, mode="nearest") for i in range(3)], axis=-1)
    w = uniform_filter(m.astype(float), k, mode="nearest")
    sm = sm / np.maximum(np.linalg.norm(sm, axis=-1), 1e-9)[..., None]
    dot = (nn * sm).sum(-1)[m & (w > MIN_NEIGHBOURHOOD_VALID)]
    out["rough_dot"] = dot
    out["rough_deg"] = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    out["shear1"], out["shear1_area"] = _signed_shear_deg(a, b, m)
    out["stretch_u"] = np.linalg.norm(a, axis=-1)[m] / float(nominal[0])
    out["stretch_v"] = np.linalg.norm(b, axis=-1)[m] / float(nominal[1])
    out["area"] = area_all[m]

    s = int(span)
    if P.shape[0] > s and P.shape[1] > s:
        a8 = P[:-s, s:] - P[:-s, :-s]
        b8 = P[s:, :-s] - P[:-s, :-s]
        m8 = ok[:-s, :-s] & ok[:-s, s:] & ok[s:, :-s]
        out["shear8"], out["shear8_area"] = _signed_shear_deg(a8, b8, m8)
    else:
        out["shear8"], out["shear8_area"] = empty, empty
    out["nominal_cell_area"] = np.array([float(nominal[0]) * float(nominal[1])])
    return out


def _r(v: float | None, nd: int = 4) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def _shear_summary(shear: np.ndarray, area: np.ndarray) -> dict[str, Any] | None:
    if shear.size == 0:
        return None
    ab = np.abs(shear)
    tot = float(area.sum())

    def _share(th: float) -> float:
        return float((ab > th).mean())

    def _area_share(th: float) -> float | None:
        return float(area[ab > th].sum() / tot) if tot > 0 else None

    return {
        "cells": int(shear.size),
        "signed_p05_deg": _r(np.percentile(shear, 5), 3),
        "signed_p95_deg": _r(np.percentile(shear, 95), 3),
        "abs_median_deg": _r(np.median(ab), 3),
        "abs_p95_deg": _r(np.percentile(ab, 95), 3),
        "pct_cells_gt_10deg": _r(100 * _share(10.0), 3),
        "pct_cells_gt_20deg": _r(100 * _share(20.0), 3),
        "pct_area_gt_10deg": _r(None if _area_share(10.0) is None else 100 * _area_share(10.0), 3),
        "pct_area_gt_20deg": _r(None if _area_share(20.0) is None else 100 * _area_share(20.0), 3),
    }


def _stretch_summary(s: np.ndarray) -> dict[str, Any] | None:
    if s.size == 0:
        return None
    return {
        "p05": _r(np.percentile(s, 5)),
        "p50": _r(np.percentile(s, 50)),
        "p95": _r(np.percentile(s, 95)),
        "pct_gt_1p5x": _r(100 * float((s > 1.5).mean()), 3),
        "pct_off_by_gt_1p5x_either_way": _r(100 * float(((s > 1.5) | (s < 1 / 1.5)).mean()), 3),
    }


def summarize_cells(cells: Mapping[str, np.ndarray], meshes: int = 1) -> dict[str, Any]:
    """Aggregate summary of (pooled) per-cell arrays."""
    n_cells = int(cells["stretch_u"].size)
    if n_cells == 0:
        return {"meshes": meshes, "cells": 0}
    nominal_area = float(cells["nominal_cell_area"][0])
    rough = cells["rough_deg"]
    out: dict[str, Any] = {
        "meshes": int(meshes),
        "cells": n_cells,
        "shear_1cell": _shear_summary(cells["shear1"], cells["shear1_area"]),
        "shear_8cell": _shear_summary(cells["shear8"], cells["shear8_area"]),
        "stretch_along_u_columns": _stretch_summary(cells["stretch_u"]),
        "stretch_along_v_rows": _stretch_summary(cells["stretch_v"]),
        "area_inflation_true_over_nominal": _r(float(cells["area"].sum()) / (n_cells * nominal_area)),
    }
    if rough.size:
        out["roughness"] = {
            "cells": int(rough.size),
            "window_cells": ROUGHNESS_WINDOW,
            "median_deg": _r(np.median(rough), 3),
            "p95_deg": _r(np.percentile(rough, 95), 3),
            "pct_flip_dot_lt_0": _r(100 * float((cells["rough_dot"] < 0).mean()), 4),
            "pct_gt_20deg": _r(100 * float((rough > 20).mean()), 3),
            "pct_gt_45deg": _r(100 * float((rough > 45).mean()), 3),
        }
    else:
        out["roughness"] = None
    return out


def grid_quality(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, meta: Mapping[str, Any] | None = None, *,
    nominal: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Summary for one tifxyz grid; nominal spacing from ``meta['scale']`` unless ``nominal`` is given."""
    nom = nominal if nominal is not None else nominal_step_vx(meta)
    return summarize_cells(grid_quality_cells(x, y, z, nominal=nom))


class Pool:
    """Concatenates per-cell arrays across meshes (all meshes of a set share the nominal cell area)."""

    KEYS = ("shear1", "shear1_area", "shear8", "shear8_area", "stretch_u", "stretch_v", "area",
            "rough_deg", "rough_dot")

    def __init__(self) -> None:
        self._parts: dict[str, list[np.ndarray]] = {k: [] for k in self.KEYS}
        self._nominal_area: float | None = None
        self.meshes = 0

    def add(self, cells: Mapping[str, np.ndarray]) -> None:
        if cells["stretch_u"].size == 0:
            return
        for k in self.KEYS:
            self._parts[k].append(np.asarray(cells[k], dtype=np.float64))
        na = float(cells["nominal_cell_area"][0])
        if self._nominal_area is not None and abs(na - self._nominal_area) > 1e-9:
            raise ValueError("cannot pool meshes with different nominal cell areas")
        self._nominal_area = na
        self.meshes += 1

    def summary(self) -> dict[str, Any]:
        if not self.meshes:
            return {"meshes": 0, "cells": 0}
        merged = {k: np.concatenate(v) for k, v in self._parts.items()}
        merged["nominal_cell_area"] = np.array([self._nominal_area])
        return summarize_cells(merged, self.meshes)


def _compact_row(mesh_id: str, summary: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    s1 = summary.get("shear_1cell") or {}
    s8 = summary.get("shear_8cell") or {}
    su = summary.get("stretch_along_u_columns") or {}
    rg = summary.get("roughness") or {}
    row = {
        "mesh_id": mesh_id,
        "cells": summary.get("cells"),
        "shear_1cell_signed_p05_p95_deg": [s1.get("signed_p05_deg"), s1.get("signed_p95_deg")],
        "shear_8cell_signed_p05_p95_deg": [s8.get("signed_p05_deg"), s8.get("signed_p95_deg")],
        "stretch_u_p05_p50_p95": [su.get("p05"), su.get("p50"), su.get("p95")],
        "area_inflation_true_over_nominal": summary.get("area_inflation_true_over_nominal"),
        "roughness_median_p95_deg": [rg.get("median_deg"), rg.get("p95_deg")],
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# 3. Live public sets
# ---------------------------------------------------------------------------

def _failure(mesh_id: str, err: Mapping[str, Any]) -> dict[str, Any]:
    return {"mesh_id": mesh_id, **err}


def run_mesh_set(
    http: BudgetedHttp,
    items: Sequence[tuple[str, str]],
    *,
    set_id: str,
    source: str,
    row_ids: Iterable[str] = (),
    extra_pools: Mapping[str, Callable[[str], bool]] | None = None,
    g1_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Fetch ``(mesh_id, url)`` items under the shared budget and pool their grid quality."""
    pool = Pool()
    sub = {name: Pool() for name in (extra_pools or {})}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    row_ids, g1_ids = set(row_ids), set(g1_ids)
    skipped_budget: list[str] = []
    attempted = 0
    b0 = http.bytes_used
    for idx, (mesh_id, url) in enumerate(items):
        try:
            got = fetch_tifxyz(http, url)
        except ByteBudgetExceeded:
            skipped_budget = [m for m, _ in items[idx:]]
            break
        attempted += 1
        if "error" in got:
            failures.append(_failure(mesh_id, got["error"]))
            continue
        nom = nominal_step_vx(got["meta"])
        cells = grid_quality_cells(got["x"], got["y"], got["z"], nominal=nom)
        if cells["stretch_u"].size == 0:
            failures.append(_failure(mesh_id, {"file": "decode", "http_status": 200, "exception": "NoValidCells"}))
            continue
        pool.add(cells)
        for name, pred in (extra_pools or {}).items():
            if pred(mesh_id):
                sub[name].add(cells)
        if mesh_id in row_ids:
            rows.append(_compact_row(mesh_id, summarize_cells(cells), nominal_step_vx=[round(nom[0], 3), round(nom[1], 3)],
                                     read_by_gemini_g1=mesh_id in g1_ids))
    result: dict[str, Any] = {
        "set_id": set_id,
        "source": source,
        "meshes_requested": len(items),
        "meshes_attempted": attempted,
        "meshes_ok": pool.meshes,
        "meshes_failed": len(failures),
        "meshes_skipped_budget": len(skipped_budget),
        "bytes_this_set": http.bytes_used - b0,
        "pooled": pool.summary(),
    }
    if sub:
        result["pooled_subsets"] = {k: p.summary() for k, p in sub.items()}
    if rows:
        result["per_mesh"] = rows
    if failures:
        result["failures"] = failures
    if skipped_budget:
        result["skipped_budget_mesh_ids"] = skipped_budget
    return result


def winding_ids(stride: int, n: int = N_WINDINGS) -> list[str]:
    """G1 windings first, then every ``stride``-th winding (``w000`` ...), de-duplicated, in a stable order."""
    ids = list(G1_WINDINGS)
    for i in range(0, n, max(1, int(stride))):
        wid = f"w{i:03d}"
        if wid not in ids:
            ids.append(wid)
    return ids


def build_live_sets(http: BudgetedHttp, *, winding_stride: int = 3) -> list[dict[str, Any]]:
    sets: list[dict[str, Any]] = []
    # (a) PHerc 0211 winner mesh.
    sets.append(run_mesh_set(
        http, [("PHerc0211_winner_v9", PHERC0211_WINNER)], set_id="pherc0211_winner_mesh",
        source="vesuvius-challenge-open-data S3: PHerc0211 segments 20261028201525-crop1711-lines123-v9 (public)",
        row_ids=["PHerc0211_winner_v9"],
    ))
    # (c) PHerc 0125 eligible meshes already screened NEGATIVE (ledger tags only).
    ledger = json.loads((HERE / "evidence" / "firstletters_rtx4060_screen_ledger.json").read_text(encoding="utf-8"))
    tags = sorted({m["mesh_id"] for m in ledger["meshes"] if m["scroll"] == "PHerc0125" and m.get("verdict") == "NEGATIVE"})
    sets.append(run_mesh_set(
        http, [(t, f"{PSCAMILLO_MESHES}PHerc0125/{t}/") for t in tags],
        set_id="pherc0125_eligible_meshes_screened_negative",
        source="pscamillo/vesuvius-eligible-meshes @66000516 (public); tags = ledger NEGATIVE verdicts",
        row_ids=tags,
    ))
    ids = winding_ids(winding_stride)
    g1 = set(G1_WINDINGS)
    # (b) PHerc 0191 and PHerc 0826 spiral fits (Hugging Face, pinned revision).
    sets.append(run_mesh_set(
        http, [(w, f"{HF_FITS}{HF_0191_RUN}/meshes/fitted/{w}/") for w in ids],
        set_id="pherc0191_hf_spiral_fits",
        source="rodriguescarson/eligible-scroll-spiral-fits @63c641ba, run slice-11600-12400 (w90), raw fit grids",
        row_ids=G1_WINDINGS, g1_ids=g1,
        extra_pools={"inner_w000_w019": lambda m: int(m[1:]) < 20, "outer_w020_w089": lambda m: int(m[1:]) >= 20},
    ))
    for band in HF_0826_BANDS:
        sets.append(run_mesh_set(
            http, [(w, f"{HF_FITS}{band}/meshes/fitted/{w}/") for w in ids],
            set_id=f"pherc0826_hf_spiral_fits_{band.split('_')[2]}",
            source=f"rodriguescarson/eligible-scroll-spiral-fits @63c641ba, band {band.split('_')[2]}, raw fit grids",
            row_ids=G1_WINDINGS if band == G1_0826_BAND else (),
            g1_ids=g1 if band == G1_0826_BAND else (),
            extra_pools={"inner_w000_w019": lambda m: int(m[1:]) < 20, "outer_w020_w089": lambda m: int(m[1:]) >= 20},
        ))
    return sets


# ---------------------------------------------------------------------------
# 4. Exp T11 RGT-stitched sheets (synthetic slab, in RAM)
# ---------------------------------------------------------------------------

def make_synthetic_rgt_slab(shape=(48, 160, 160), n_wraps=5, spacing=6.0, sigma=1.0, noise=3.5, seed=42,
                            fold=2.8, pinch=1.4):
    """Folded / pinched 5-wrap CT + m7 slab (same recipe as the Exp T11 unit-test fixture)."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    d, a, b = shape
    aa, bb = np.meshgrid(np.arange(a, dtype=float), np.arange(b, dtype=float), indexing="ij")
    fold_f = fold * np.sin(2.0 * np.pi * aa / a) * np.cos(np.pi * bb / b)
    local_sp = spacing + 0.8 * np.sin(2.0 * np.pi * bb / b)
    pinch_f = pinch * np.exp(-((aa - 0.62 * a) ** 2 + (bb - 0.38 * b) ** 2) / (2.0 * 12.0 ** 2))
    c = (n_wraps - 1) / 2.0
    heights = []
    for m in range(n_wraps):
        h = d / 2.0 + fold_f + (m - c) * local_sp
        if m == 2:
            h = h + pinch_f
        heights.append(h)
    gap_mask = ((aa - 0.28 * a) ** 2 + (bb - 0.72 * b) ** 2) < (11.0 ** 2)
    z = np.arange(d, dtype=float)[:, None, None]
    vol = np.zeros(shape)
    clean = np.zeros(shape)
    for m, h in enumerate(heights):
        texture = 1.0 + 0.35 * ndi.gaussian_filter(rng.normal(0.0, 1.0, (a, b)), 1.3) / 0.25
        mask_factor = np.where(gap_mask, 0.0, 1.0) if m == 1 else 1.0
        g = np.exp(-((z - h[None]) ** 2) / (2.0 * sigma ** 2)) * mask_factor
        vol += np.clip(texture, 0.4, 1.6) * g
        clean += g
    ct = np.clip(40.0 + 150.0 * vol + rng.normal(0.0, noise, shape), 1, 255).astype(np.uint8)
    m7 = np.clip(255.0 * np.exp(-0.5 * ((clean - 1.0) / 0.45) ** 2) * (clean > 0.22), 0, 255).astype(np.uint8)
    bridge = (np.abs(aa - 0.5 * a) < 18) & (np.abs(bb - 0.5 * b) < 18)
    z_bridge = (z >= heights[1][None] - 0.5) & (z <= heights[2][None] + 0.5) & bridge[None]
    m7 = np.where(z_bridge, np.maximum(m7, np.uint8(200)), m7)
    return ct, m7, np.stack(heights)


def _shift_global(x, y, z, valid, offset: float = 1000.0):
    """Translate valid vertices into a global-like frame (as the private export does) and keep ``-1`` sentinels."""
    f = lambda a: np.where(valid, np.asarray(a, dtype=np.float64) + offset, -1.0)  # noqa: E731
    return f(x), f(y), f(z)


def run_rgt_synthetic() -> dict[str, Any]:
    """Quantify Cartesian RGT-stitched sheets (1 vertex per lateral voxel) with the same grid-quality metrics."""
    import seismic_rgt_stitched_grower as SRG
    import seismic_rgt_unwrapper as RGT

    t0 = time.perf_counter()
    ct, m7, true_h = make_synthetic_rgt_slab()
    stitched = SRG.stitch_overlapping_rgt_tiles_3d(
        ct, m7, tile_lateral=96, overlap_lateral=32, min_spacing_vx=4.0, max_spacing_vx=18.0
    )
    extracted = SRG.extract_and_snap_stitched_rgt_sheets(
        stitched["phi"], ct, m7, normal_axis=stitched["normal_axis"], voxel_um=9.362, snap_passes=2, num_slices=9
    )
    nominal = (1.0, 1.0)           # export_private_rgt_tifxyz_meshes writes scale (1, 1): one vertex per voxel
    snapped, pre_snap, truth = Pool(), Pool(), Pool()
    rows = []
    for i, sheet in enumerate(extracted["sheets"]):
        x, y, z = _shift_global(sheet["x"], sheet["y"], sheet["z"], np.asarray(sheet["valid"], dtype=bool))
        cells = grid_quality_cells(x, y, z, nominal=nominal)
        snapped.add(cells)
        rows.append(_compact_row(f"rgt_sheet_{i:02d}_after_snap", summarize_cells(cells)))
        hx, hy, hz, hv = RGT.height_to_tifxyz(sheet["height_pre_snap"], sheet["valid"], int(sheet["normal_axis"]))
        pre_snap.add(grid_quality_cells(*_shift_global(hx, hy, hz, hv), nominal=nominal))
    for h in true_h:
        tx, ty, tz, tv = RGT.height_to_tifxyz(h, np.ones_like(h, dtype=bool), 0)
        truth.add(grid_quality_cells(*_shift_global(tx, ty, tz, tv), nominal=nominal))
    return {
        "set_id": "exp_t11_rgt_stitched_synthetic",
        "source": "seismic_rgt_stitched_grower.py on a synthetic 48x160x160 five-wrap folded slab, in RAM; "
                  "no real scroll data, no coordinates",
        "grid_note": "Cartesian lateral lattice, 1 vertex per voxel (nominal step 1 vx), height along the normal axis; "
                     "roughness window is 9 voxels here against 9 cells = 180 vx on a 20 vx spiral-fit grid",
        "sheets": len(extracted["sheets"]),
        "pooled_after_snap": snapped.summary(),
        "pooled_before_snap": pre_snap.summary(),
        "synthetic_truth_surfaces_reference": truth.summary(),
        "per_sheet": rows,
        "seam_phase_rmse_rad": _r(stitched.get("seam_phase_rmse_rad"), 5),
        "crossing_fraction_rgt": extracted.get("crossing_fraction_rgt"),
        "runtime_seconds": round(time.perf_counter() - t0, 2),
    }


# ---------------------------------------------------------------------------
# 5. Code-verified answers to claude-022 / claude-023 (read from source on disk)
# ---------------------------------------------------------------------------

def _src_files(root: Path) -> list[Path]:
    """Non-test gemini_lane modules, excluding this audit (it names the tokens it searches for)."""
    return sorted(p for p in root.glob("*.py") if not p.name.startswith("test_") and p.stem != "grid_quality_audit")


def _imports_module(path: Path, module: str) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == module for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == module:
            return True
    return False


def _function_source(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node) or ""
    return ""


def render_sample_convention_probe() -> dict[str, Any]:
    """Numerical proof of ``render_surface_stack``'s sample point, using a stub CT reader that returns ray x."""
    import stream_snap_render as SSR

    class _Reader:
        dtype = np.uint8

        def sample_trilinear(self, rays, return_support=False):
            vals = np.asarray(rays)[..., 0]
            return (vals, np.ones_like(vals, dtype=bool)) if return_support else vals

    step = 40.0                                        # 40 vx per grid cell, x = 40 * column -> value = 2 * column_position
    cols = np.arange(8, dtype=np.float32)
    x = np.tile((step * cols + 3.0)[None, :], (6, 1))   # +3 keeps z/x > 0 and away from sentinel
    y = np.tile((step * np.arange(6, dtype=np.float32) + 3.0)[:, None], (1, 8))
    z = np.full_like(x, 50.0)
    stack, valid = SSR.render_surface_stack(x, y, z, _Reader(), num_slices=1, scale=0.05, require_nonzero_intensity=False)
    row = stack[0, 0, :]
    pix = np.arange(row.size)
    # Under p/20 (grid index = pixel * scale): x = 40 * p/20 + 3 = 2p + 3.   Under (p + 0.5)/20: x = 2p + 4.
    ok = valid[0, :]
    index_conv = np.clip(np.rint(2.0 * pix + 3.0), 0, 255).astype(np.uint8)
    centre_conv = np.clip(np.rint(2.0 * pix + 4.0), 0, 255).astype(np.uint8)
    return {
        "output_pixels_along_columns": int(row.size),
        "matches_grid_index_equals_pixel_times_scale": bool(np.array_equal(row[ok], index_conv[ok])),
        "matches_grid_index_equals_pixel_plus_half_times_scale": bool(np.array_equal(row[ok], centre_conv[ok])),
    }


def code_verified_answers(repo_root: Path | None = None) -> dict[str, Any]:
    """Facts about conventions and flattening, derived from the source and evidence on disk."""
    gl = Path(repo_root) / "gemini_lane" if repo_root else HERE
    ssr_path, swr_path = gl / "stream_snap_render.py", gl / "spiral_winding_resample.py"
    ssr_fn = _function_source(ssr_path, "render_surface_stack")
    uses_half = bool(re.search(r"\+\s*0\.5", ssr_fn))
    grid_index_lines = [ln.strip() for ln in ssr_fn.splitlines() if re.search(r"gv_1d|gu_1d", ln) and "arange" in ln]
    screen_py = (gl / "screen_scroll_rtx4060.py").read_text(encoding="utf-8")
    snap_then_render = bool(re.search(r"snap_tifxyz_to_m7\(", screen_py)) and bool(re.search(r"render_surface_stack\(", screen_py))
    reads_meta_scale = 'meta.get("scale", [0.05, 0.05])' in screen_py

    sources = _src_files(gl)
    swr_importers = sorted(p.stem for p in sources if p.stem != "spiral_winding_resample" and _imports_module(p, "spiral_winding_resample"))
    render_users = sorted(p.stem for p in sources if p.stem != "stream_snap_render" and "render_surface_stack(" in p.read_text(encoding="utf-8"))
    flatten_tokens = {}
    for tok in ("flatten_spiral_checkpoint", "flatboi"):
        flatten_tokens[tok] = sorted(p.stem for p in sources if tok in p.read_text(encoding="utf-8"))

    ev = gl / "evidence"
    t0826 = json.loads((ev / "pherc0826_1203t2_onsheet_triage.json").read_text(encoding="utf-8"))
    t0191 = json.loads((ev / "pherc0125_0191_onsheet_triage.json").read_text(encoding="utf-8"))
    ledger = json.loads((ev / "firstletters_rtx4060_screen_ledger.json").read_text(encoding="utf-8"))
    ids_0826 = sorted(t["mesh_id"] for t in t0826["targets"] if t["scroll"] == "PHerc0826")
    ids_0191 = sorted(s["mesh_id"] for sc in t0191["scrolls"] if sc["scroll"] == "PHerc0191" for s in sc["sheets"])
    det_0191 = sorted((m["mesh_id"], m.get("verdict")) for m in ledger["meshes"] if m["scroll"] == "PHerc0191")
    det_0826 = sorted((m["mesh_id"], m.get("verdict")) for m in ledger["meshes"] if m["scroll"] == "PHerc0826")

    return {
        "q_which_tifxyz_grid_coordinate_convention_does_stream_snap_render_use": {
            "answer": "Vertex-aligned grid-index convention: output pixel p samples the grid at index p * scale "
                      "(= p / 20 for scale 0.05), with zero-based vertex indices and no half-pixel offset; the "
                      "pixel-centre convention (p + 0.5) / 20 is NOT used.",
            "offset_between_conventions": "0.5 output px = 0.025 grid cells = 0.5 voxel at the nominal 20 vx step "
                                          "(0.5 x local stretch in true voxels on a stretched fit grid)",
            "render_surface_stack_contains_plus_0p5": uses_half,
            "sampling_lines": grid_index_lines,
            "output_size_rule": "round((n_grid - 1) / scale) + 1 pixels per axis (vertex 0 and vertex n-1 land on pixels)",
            "numerical_probe_stub_reader": render_sample_convention_probe(),
            "source": "gemini_lane/stream_snap_render.py::render_surface_stack",
        },
        "q_per_surface_flattening_vs_raw_fit_grid_vs_spiral_winding_resample": {
            "answer": "No gemini_lane screen used per-surface flattening or spiral_winding_resample. Every screen "
                      "(screen_scroll_rtx4060.py) read the raw fit grid at meta scale (default 0.05), optionally "
                      "m7-snapped, then render_surface_stack; spiral_winding_resample.py is a diagnostic imported "
                      "only by the Exp T14 dossier verifier.",
            "screen_scroll_rtx4060_reads_meta_scale_default_0p05": reads_meta_scale,
            "screen_scroll_rtx4060_snaps_then_renders": snap_then_render,
            "snap_keeps_grid_topology": "snap_tifxyz_to_m7 moves vertices along local normals only; no re-gridding "
                                        "(spacing along u/v changes only through the displacement)",
            "modules_importing_spiral_winding_resample": swr_importers,
            "modules_calling_render_surface_stack": render_users,
            "modules_mentioning_flatten_spiral_checkpoint": flatten_tokens["flatten_spiral_checkpoint"],
            "modules_mentioning_flatboi": flatten_tokens["flatboi"],
            "flatboi_note": "the two 'flatboi' mentions are the name of a public PHerc0139 segment "
                            "(..._zmid_flatboi) read by the transverse-physics controls, not a flattening applied to a spiral fit",
        },
        "q_which_pherc0826_pherc0191_fits_did_g1_read_and_on_which_grid": {
            "answer": "CPU on-sheet triage only (no ink detector) on raw fit grids, m7 2x snap, 5 slices: PHerc0826 "
                      "band slice-8928-9728 windings w020/w030/w040/w050 (public HF @63c641ba); PHerc0191 run "
                      "slice-11600-12400 windings w020/w030/w040/w050. Ink detectors (Studio + A-hn, RTX 4060) ran "
                      "only on PHerc0191 w030 and w050 (both NEGATIVE). The other two PHerc0826 bands and all other "
                      "windings were not read by Gemini. Grid: raw fit grid at meta scale 0.05 (nominal 20 vx), "
                      "unflattened and un-resampled, so the claude-023 shear / stretch applies to what G1 read.",
            "pherc0826_cpu_triage_mesh_ids": ids_0826,
            "pherc0191_cpu_triage_mesh_ids": ids_0191,
            "pherc0191_ink_detector_reads": [{"mesh_id": m, "verdict": v} for m, v in det_0191],
            "pherc0826_ink_detector_reads": [{"mesh_id": m, "verdict": v} for m, v in det_0826],
            "pherc0826_triage_no_ink_detector_run": bool(t0826.get("renderer_semantics", {}).get("no_ink_detector_run")),
            "source": "evidence/pherc0826_1203t2_onsheet_triage.json, evidence/pherc0125_0191_onsheet_triage.json, "
                      "evidence/firstletters_rtx4060_screen_ledger.json",
        },
    }


# ---------------------------------------------------------------------------
# 6. Receipt, privacy and CLI
# ---------------------------------------------------------------------------

def assert_public_receipt(receipt: Mapping[str, Any]) -> None:
    """Raise if the receipt would leak local paths, user names or unread-scroll fields."""
    import leak_guard

    text = json.dumps(receipt, sort_keys=True)
    if leak_guard.find_local_paths(text):
        raise ValueError("receipt contains a local path")
    if leak_guard.find_user_leaks(text, leak_guard.account_name()):
        raise ValueError("receipt contains the account name")
    for tok in GENERIC_PRIVACY_TOKENS:
        if tok.lower() in text.lower():
            raise ValueError(f"receipt contains forbidden token {tok!r}")


def build_receipt(sets: list[dict[str, Any]], http: BudgetedHttp, rgt: dict[str, Any] | None,
                  *, started: float) -> dict[str, Any]:
    receipt = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "generated_utc_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "method": "grid_quality formulas of methods_ledger/code/grid_quality.py (claude-023 ledger S8) re-implemented "
                  "with NumPy/SciPy/tifffile; tifxyz read into RAM from public URLs; aggregates only",
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": any(m in sys.modules for m in ("torch", "cupy", "jax", "tensorflow", "pycuda")),
        "parameters": {
            "roughness_window_cells": ROUGHNESS_WINDOW,
            "roughness_min_neighbourhood_valid": MIN_NEIGHBOURHOOD_VALID,
            "shear_spans_cells": [1, SHEAR_SPAN_CELLS],
            "stretch_nominal": "1 / meta.scale per axis (20 vx for scale 0.05); stretch_u along columns, v along rows",
            "roughness_note": "claude-023 'roughness' = angle to the 9x9 mean normal",
            "validity": "finite and z > 0 (Claude), plus finite x, y; quad needs corners (r,c), (r,c+1), (r+1,c)",
        },
        "http": http.summary(),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "sets": sets,
        "claude_reference_numbers_claude_023": {
            "pherc0211_winner_shear_p05_p95_deg": [-1.7, 1.8],
            "pherc0175a_snapped_shear_p05_p95_deg": [-31, 29],
            "pherc1203_shear_p05_p95_deg": [-28, 28],
            "note": "Claude's figures use an 8-cell span; compare with shear_8cell",
        },
        "code_verified_answers": code_verified_answers(),
        "caveats": [
            "Grid quality is a parameterisation diagnostic, not an on-sheet or ink certificate.",
            "HF windings are single fitted windings (w000-w089) sampled at a stride; pooled figures are cell-pooled.",
            "The PHerc0826 / PHerc0191 fits are author-flagged unvalidated geometry (CC BY-NC 4.0).",
            "RGT sheets come from a synthetic slab; no unread scroll geometry was touched.",
        ],
    }
    if rgt is not None:
        receipt["sets"].append(rgt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--budget-mib", type=float, default=60.0, help="hard HTTP cap, <= 60 MiB")
    ap.add_argument("--winding-stride", type=int, default=3)
    ap.add_argument("--min-interval-s", type=float, default=0.05, help="polite delay between requests")
    ap.add_argument("--skip-live", action="store_true", help="do not touch the network")
    ap.add_argument("--skip-rgt", action="store_true")
    args = ap.parse_args(argv)
    if not (0 < args.budget_mib <= 60.0):
        ap.error("--budget-mib must be in (0, 60]")
    started = time.perf_counter()
    http = BudgetedHttp(int(args.budget_mib * 1024 * 1024), min_interval_s=args.min_interval_s)
    sets = [] if args.skip_live else build_live_sets(http, winding_stride=args.winding_stride)
    rgt = None if args.skip_rgt else run_rgt_synthetic()
    receipt = build_receipt(sets, http, rgt, started=started)
    assert_public_receipt(receipt)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": out.name, "http": receipt["http"], "sets": [s["set_id"] for s in receipt["sets"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
