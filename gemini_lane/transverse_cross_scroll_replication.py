# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Exp T7: Cross-Scroll Zero-Shot Replication, Off-Sheet Kill-Test (+/-8 slices), & Cryo-ET Slice Coherence.

Strictly CPU-only (zero torch/CUDA imports), zero persistent CT on disk (RAM only),
and hard-capped HTTP byte budgets (<= 40 MiB per volume, <= 120 MiB global).

Tests zero-shot cross-scroll replication of the Exp T5 transverse physics suite on TWO
held-out public controls (3 surface volumes total):
1. PHerc. 0841 w00 — High-res 2.403 um (`2.403um-0.22m-77keV-volume-20260319124803.zarr`):
   Prefers human ground-truth labels discovered via S3 XML API under `ink-labels/`
   (`inklabels.zarr` + `supervision.zarr`, Zarr v3 sharded, ~0.12 MiB total); falls back
   to model-derived ink-detection TIFF only if human labels are absent or incompatible.
2. PHerc. 0841 w00 — Faint 9.366 um (`9.366um-1.2m-113keV-volume-20250821151531.zarr`):
   Native 1:1 `argus-v3` ink-detection reference (explicitly disclosed; avoids ~16 px
   cross-scan registration offset vs the 2.403 um scan canvas).
3. PHerc. 0343 w047-w048 — Faint 8.64 um (`8.64um-1.2m-116keV-volume-20250521140437.zarr`):
   Native 1:1 `argus-v3` ink-detection reference (explicitly disclosed in JSON).

Three Core Scientific Additions over Exp T5:
1. Cryo-ET / CryoSamba-inspired adjacent-slice coherence (`cryo_slice_coherence`):
   Local zero-mean normalized patch cross-correlation / Frobenius cosine similarity
   between adjacent slice pairs `(k-1, k)` and `(k, k+1)` around the papyrus peak.
2. Unsupervised equal-weight rank-average fusion (`rank_fusion_t5_triplet`):
   Combines `oct_hilbert_env_peak`, `wst_s1_isotropic`, and `pc_min_moment_corners_pooled`
   under FIXED signed polarities observed in Exp T5 (`fixed_t5_polarity`), with
   `unsigned_best_polarity` logged strictly as a diagnostic (never used to claim PASS).
3. Adversarial Off-Sheet Kill-Test (`claude-016` geometry-leakage safeguard):
   Evaluates every feature both on-sheet and shifted off-sheet by +/-8 slices
   (`shift_minus8`, `shift_plus8`) AND at the farthest stack boundary sub-slabs
   (`slice 0..4` and `slice L-5..L-1`), paired with a 16-shift spatially rolled-label null.

Predeclared Gate (fixed_t5_polarity ONLY; NO post-hoc tuning):
- Per gated feature & volume:
  `on_sheet_AUC >= 0.70 AND delta_null >= +0.10 AND (on_sheet_AUC - off_sheet_AUC >= +0.08)`
- Verdict per volume (`PASS` / `PARTIAL` / `CLOSED NO-GO`).
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import io
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Sequence
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np
from scipy import ndimage as ndi
from scipy import signal as ssig
from scipy.stats import rankdata

try:
    from .pherc0139_transverse_physics import (
        BLANK_LOW_U8_THRESHOLD,
        INK_HIGH_U8_THRESHOLD,
        MIB,
        BudgetExceededError,
        BudgetedRemoteZarrV2,
        HttpByteBudget,
        _fill_nearest_2d,
        _normalized_gaussian,
        assert_no_local_paths,
        budgeted_http_get,
        build_16_rolled_shifts,
        cohens_d,
        exact_roc_auc,
        extract_subvoxel_height_field,
        load_ink_map_in_ram,
    )
    from .stream_snap_render import (
        _build_zarr_decoder,
        _sanitize_source_label,
    )
    from .transverse_wavelet_oct_physics import (
        OCT_HALF_SPAN_VX_0139,
        OCT_HALF_SPAN_VX_0306B,
        OCT_WAVEFORM_DIM,
        compute_kovesi_phase_congruency_features,
        compute_wavelet_scattering_features,
        extract_peak_aligned_depth_profiles,
    )
except ImportError:  # pragma: no cover - direct script execution
    from pherc0139_transverse_physics import (
        BLANK_LOW_U8_THRESHOLD,
        INK_HIGH_U8_THRESHOLD,
        MIB,
        BudgetExceededError,
        BudgetedRemoteZarrV2,
        HttpByteBudget,
        _fill_nearest_2d,
        _normalized_gaussian,
        assert_no_local_paths,
        budgeted_http_get,
        build_16_rolled_shifts,
        cohens_d,
        exact_roc_auc,
        extract_subvoxel_height_field,
        load_ink_map_in_ram,
    )
    from stream_snap_render import (
        _build_zarr_decoder,
        _sanitize_source_label,
    )
    from transverse_wavelet_oct_physics import (
        OCT_HALF_SPAN_VX_0139,
        OCT_HALF_SPAN_VX_0306B,
        OCT_WAVEFORM_DIM,
        compute_kovesi_phase_congruency_features,
        compute_wavelet_scattering_features,
        extract_peak_aligned_depth_profiles,
    )


# ---------------------------------------------------------------------------
# Public Control Endpoints & Predeclared Experiment Constants
# ---------------------------------------------------------------------------

S3_BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"

VOLUME_ORDER: tuple[str, ...] = (
    "pherc0841_2p403um",
    "pherc0841_9p366um",
    "pherc0343_8p64um",
)

DEFAULT_VOLUME_SPECS: dict[str, dict[str, Any]] = {
    "pherc0841_2p403um": {
        "scroll": "PHerc0841",
        "segment": "20260220213127-w00",
        "voxel_um": 2.403,
        "regime": "high_res_2p4um",
        "sv_zarr_url": (
            f"{S3_BASE}/PHerc0841/segments/20260220213127-w00/surface-volumes/"
            "2.403um-0.22m-77keV-volume-20260319124803.zarr"
        ),
        "ink_labels_s3_prefix": "PHerc0841/segments/20260220213127-w00/ink-labels/",
        "ink_labels_match_token": "2.403um",
        "fallback_ink_tif_url": (
            f"{S3_BASE}/PHerc0841/segments/20260220213127-w00/ink-detection/"
            "PHerc0841-20260220213127-2.403um-0.22m-77keV-volume-20260319124803-"
            "20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"
        ),
        "fallback_provenance": "model_derived_new_canon_autoresearch_recipe",
        "border_guard_px": 32,
        "z_smooth_sigma": 1.0,
    },
    "pherc0841_9p366um": {
        "scroll": "PHerc0841",
        "segment": "20260220213127-w00",
        "voxel_um": 9.366,
        "regime": "faint_8_9um",
        "sv_zarr_url": (
            f"{S3_BASE}/PHerc0841/segments/20260220213127-w00/surface-volumes/"
            "9.366um-1.2m-113keV-volume-20250821151531.zarr"
        ),
        "ink_labels_s3_prefix": "PHerc0841/segments/20260220213127-w00/ink-labels/",
        "ink_labels_match_token": "9.366um",
        "fallback_ink_tif_url": (
            f"{S3_BASE}/PHerc0841/segments/20260220213127-w00/ink-detection/"
            "PHerc0841-20260220213127-9.366um-1.2m-113keV-volume-20250821151531-"
            "20261117205746-argus-v3-tile384-stride192.tif"
        ),
        "fallback_provenance": "model_derived_argus_v3_tile384_stride192",
        "border_guard_px": 16,
        "z_smooth_sigma": 0.8,
    },
    "pherc0343_8p64um": {
        "scroll": "PHerc0343",
        "segment": "20261007213123-concat_w047-w048_R5B2_z9500-11000",
        "voxel_um": 8.64,
        "regime": "faint_8_9um",
        "sv_zarr_url": (
            f"{S3_BASE}/PHerc0343/segments/20261007213123-concat_w047-w048_R5B2_z9500-11000/"
            "surface-volumes/8.64um-1.2m-116keV-volume-20250521140437.zarr"
        ),
        "ink_labels_s3_prefix": (
            "PHerc0343/segments/20261007213123-concat_w047-w048_R5B2_z9500-11000/ink-labels/"
        ),
        "ink_labels_match_token": "8.64um",
        "fallback_ink_tif_url": (
            f"{S3_BASE}/PHerc0343/segments/20261007213123-concat_w047-w048_R5B2_z9500-11000/"
            "ink-detection/PHerc0343-20261007213123-8.64um-1.2m-116keV-volume-20250521140437-"
            "20261117205746-argus-v3-tile384-stride192.tif"
        ),
        "fallback_provenance": "model_derived_argus_v3_tile384_stride192",
        "border_guard_px": 16,
        "z_smooth_sigma": 0.8,
    },
}

# Predeclared FIXED signed polarities observed in Exp T5 (commit 62b2c02,
# gemini_lane/evidence/transverse_wavelet_oct_ablation.json).
# NEVER flipped or tuned on held-out Exp T7 scrolls!
FIXED_T5_POLARITIES: dict[str, dict[str, int]] = {
    # Observed on PHerc0139 2.399 um in Exp T5
    "high_res_2p4um": {
        "raw_ct_surface": +1,               # T5 raw_auc = 0.71484 (+)
        "oct_hilbert_env_peak": +1,         # T5 raw_auc = 0.77962 (+)
        "wst_s1_isotropic": -1,             # T5 raw_auc = 0.44693 (-)
        "wst_s1_scale_slope": -1,           # T5 raw_auc = 0.20326 (-)
        "pc_min_moment_corners_pooled": -1, # T5 raw_auc = 0.39973 (-)
        "rank_fusion_t5_triplet": +1,       # constituents signed prior to rank-averaging
    },
    # Observed on PHerc0306B 8.64 um in Exp T5
    "faint_8_9um": {
        "raw_ct_surface": +1,               # T5 raw_auc = 0.51208 (+)
        "oct_hilbert_env_peak": -1,         # T5 raw_auc = 0.36904 (-)
        "wst_s1_isotropic": -1,             # T5 raw_auc = 0.30938 (-)
        "wst_s1_scale_slope": +1,           # T5 raw_auc = 0.60766 (+)
        "pc_min_moment_corners_pooled": -1, # T5 raw_auc = 0.32757 (-)
        "rank_fusion_t5_triplet": +1,       # constituents signed prior to rank-averaging
    },
}

FUSION_TRIPLET: tuple[str, str, str] = (
    "oct_hilbert_env_peak",
    "wst_s1_isotropic",
    "pc_min_moment_corners_pooled",
)

GATED_TRANSVERSE_FEATURES: tuple[str, ...] = (
    "oct_hilbert_env_peak",
    "wst_s1_isotropic",
    "wst_s1_scale_slope",
    "pc_min_moment_corners_pooled",
    "rank_fusion_t5_triplet",
)

BASELINE_FEATURES: frozenset[str] = frozenset({"raw_ct_surface"})
EXPLORATORY_FEATURES: frozenset[str] = frozenset({"cryo_slice_coherence"})

ALL_FEATURE_NAMES: tuple[str, ...] = (
    "raw_ct_surface",
    "oct_hilbert_env_peak",
    "wst_s1_isotropic",
    "wst_s1_scale_slope",
    "pc_min_moment_corners_pooled",
    "rank_fusion_t5_triplet",
    "cryo_slice_coherence",
)

# Predeclared HTTP byte caps (strictly enforced)
DEFAULT_PER_VOLUME_BUDGET_MIB = 40.0
DEFAULT_GLOBAL_BUDGET_MIB = 120.0
HUMAN_LABEL_MAX_SHARD_BYTES = 5 * MIB

# Predeclared ROI geometry (chunk-aligned multiples of 128 x 128)
DEFAULT_ROI_H = 512
DEFAULT_ROI_W = 640

# Predeclared Off-Sheet Kill-Test & Gate Thresholds (NEVER tuned post-hoc)
OFF_SHEET_SHIFT_SLICES = 8
OFF_SHEET_SUBSLAB_DEPTH = 5
GATE_MIN_ON_SHEET_AUC = 0.70
GATE_MIN_DELTA_NULL = 0.10
GATE_MIN_OFF_SHEET_DROP = 0.08


# ---------------------------------------------------------------------------
# 1. Multi-Volume Strict HTTP Byte Budget (<= 40 MiB/vol, <= 120 MiB global)
# ---------------------------------------------------------------------------

class MultiVolumeHttpByteBudget(HttpByteBudget):
    """Thread-safe strict byte-cap tracker across arbitrary named volumes + global cap."""

    def __init__(
        self,
        part_limits_bytes: dict[str, int],
        *,
        global_limit_bytes: int = int(round(DEFAULT_GLOBAL_BUDGET_MIB * MIB)),
    ) -> None:
        if not part_limits_bytes:
            raise ValueError("part_limits_bytes must be non-empty")
        if global_limit_bytes <= 0 or any(int(v) <= 0 for v in part_limits_bytes.values()):
            raise ValueError("All byte budget limits must be > 0")
        self._lock = threading.Lock()
        self.part_limits: dict[str, int] = {str(k): int(v) for k, v in part_limits_bytes.items()}
        self.global_limit: int = int(global_limit_bytes)
        self.part_bytes: dict[str, int] = {k: 0 for k in self.part_limits}
        self.part_requests: dict[str, int] = {k: 0 for k in self.part_limits}
        self.breakdown: dict[str, int] = {}

    def summary(self) -> dict[str, Any]:
        with self._lock:
            total = int(sum(self.part_bytes.values()))
            per_volume: dict[str, dict[str, Any]] = {}
            all_parts_ok = True
            for k, lim in self.part_limits.items():
                used = int(self.part_bytes[k])
                ok = used <= lim
                all_parts_ok = all_parts_ok and ok
                per_volume[k] = {
                    "bytes": used,
                    "mib": round(used / MIB, 4),
                    "limit_bytes": int(lim),
                    "limit_mib": round(lim / MIB, 4),
                    "requests": int(self.part_requests[k]),
                    "within_limit": bool(ok),
                }
            return {
                "per_volume": per_volume,
                "global_bytes": total,
                "global_mib": round(total / MIB, 4),
                "global_limit_bytes": int(self.global_limit),
                "global_limit_mib": round(self.global_limit / MIB, 4),
                "within_limits": bool(all_parts_ok and total <= self.global_limit),
                "breakdown_bytes": dict(sorted(self.breakdown.items())),
            }


# ---------------------------------------------------------------------------
# 2. S3 XML Listing, Zarr v3 Sharded Human Labels, and Alignment Helpers
# ---------------------------------------------------------------------------

def list_s3_prefix_budgeted(
    s3_base: str,
    prefix: str,
    budget: HttpByteBudget,
    part: str,
    *,
    label: str = "s3_list",
) -> list[tuple[str, int]]:
    """List keys & sizes under an S3 prefix via ListObjectsV2 XML API strictly within budget."""
    base = s3_base.rstrip("/")
    qs = urllib.parse.urlencode({"list-type": "2", "prefix": prefix})
    url = f"{base}/?{qs}"
    raw = budgeted_http_get(url, budget, part, label=label, allow_404=True)
    if raw is None:
        return []
    root = ET.fromstring(raw)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    out: list[tuple[str, int]] = []
    for elem in root.findall("s3:Contents", ns):
        key_el = elem.find("s3:Key", ns)
        sz_el = elem.find("s3:Size", ns)
        if key_el is not None and key_el.text:
            size = int(sz_el.text) if (sz_el is not None and sz_el.text) else 0
            out.append((key_el.text, size))
    return out


def _map_zarr3_blosc_codec(cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate Zarr v3 blosc codec configuration to Zarr v2/numcodecs dict."""
    shuffle_raw = cfg.get("shuffle", "noshuffle")
    shuffle_map = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
    shuffle_val = shuffle_map.get(str(shuffle_raw).lower(), 0) if isinstance(shuffle_raw, str) else int(shuffle_raw)
    return {
        "id": "blosc",
        "cname": str(cfg.get("cname", "zstd")),
        "clevel": int(cfg.get("clevel", 3)),
        "shuffle": shuffle_val,
        "blocksize": int(cfg.get("blocksize", 0)),
    }


def load_zarr3_sharded_2d_u8(
    root: str | os.PathLike[str],
    budget: HttpByteBudget | None = None,
    part: str = "",
    *,
    level: str = "0",
    label: str = "zarr3_label",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a 2D uint8 Zarr v3 sharded array (`0/zarr.json` + `0/c/0/0`) strictly in RAM.

    Supports both HTTP(S) URLs (budget-tracked via `budgeted_http_get`) and local
    directories (for offline synthetic unit tests). Decodes inner chunks via
    `stream_snap_render._build_zarr_decoder` (or raw bytes when uncompressed).
    """
    root_str = str(root).rstrip("/\\")
    is_http = root_str.lower().startswith(("http://", "https://"))

    def _fetch(rel_key: str, *, allow_404: bool = False) -> bytes | None:
        if not is_http:
            p = Path(root_str).joinpath(*[seg for seg in rel_key.split("/") if seg])
            if not p.is_file():
                if allow_404:
                    return None
                raise FileNotFoundError(f"Missing local Zarr v3 key: {rel_key}")
            return p.read_bytes()
        assert budget is not None
        return budgeted_http_get(
            f"{root_str}/{rel_key}",
            budget,
            part,
            label=f"{label}:{'meta' if rel_key.endswith('.json') else 'shard'}",
            allow_404=allow_404,
        )

    meta_raw = _fetch(f"{level}/zarr.json", allow_404=False)
    assert meta_raw is not None
    meta = json.loads(meta_raw.decode("utf-8"))
    if int(meta.get("zarr_format", 0)) != 3:
        raise ValueError(f"Expected zarr_format == 3, got {meta.get('zarr_format')!r}")
    shape = tuple(int(s) for s in meta["shape"])
    if len(shape) != 2:
        raise ValueError(f"Expected 2D Zarr v3 label array, got shape {shape}")
    if str(meta.get("data_type", "")) != "uint8":
        raise ValueError(f"Expected uint8 Zarr v3 label array, got {meta.get('data_type')!r}")

    shard_shape = tuple(int(c) for c in meta["chunk_grid"]["configuration"]["chunk_shape"])
    n_sy = (shape[0] + shard_shape[0] - 1) // shard_shape[0]
    n_sx = (shape[1] + shard_shape[1] - 1) // shard_shape[1]

    codecs = meta.get("codecs", [])
    if not codecs or codecs[0].get("name") != "sharding_indexed":
        raise ValueError(f"Expected sharding_indexed codec in Zarr v3 array, got {codecs!r}")
    shard_cfg = codecs[0]["configuration"]
    inner_shape = tuple(int(c) for c in shard_cfg["chunk_shape"])
    if shard_shape[0] % inner_shape[0] != 0 or shard_shape[1] % inner_shape[1] != 0:
        raise ValueError(f"Shard shape {shard_shape} not divisible by inner chunk {inner_shape}")
    ny_in = shard_shape[0] // inner_shape[0]
    nx_in = shard_shape[1] // inner_shape[1]

    # Build inner chunk decoder (supports raw bytes, blosc, gzip, zlib).
    inner_decoder = None
    for ic in shard_cfg.get("codecs", []):
        cname = str(ic.get("name", "")).lower()
        if cname == "bytes":
            continue
        if cname == "blosc":
            inner_decoder = _build_zarr_decoder(_map_zarr3_blosc_codec(ic.get("configuration", {})))
        elif cname in ("gzip", "zlib"):
            inner_decoder = _build_zarr_decoder({"id": cname})
        else:
            raise NotImplementedError(f"Unsupported Zarr v3 inner codec: {ic!r}")

    index_loc = str(shard_cfg.get("index_location", "end")).lower()
    index_codecs = shard_cfg.get("index_codecs", [])
    has_crc32c = any(str(c.get("name", "")).lower() == "crc32c" for c in index_codecs)
    idx_bytes = ny_in * nx_in * 16
    crc_bytes = 4 if has_crc32c else 0
    empty_sentinel = (1 << 64) - 1
    expected_inner_bytes = inner_shape[0] * inner_shape[1]

    fill_val = int(meta.get("fill_value", 0) or 0)
    out = np.full(shape, fill_val, dtype=np.uint8)
    total_shard_bytes = 0
    nonempty_inner = 0

    for sy in range(n_sy):
        for sx in range(n_sx):
            shard = _fetch(f"{level}/c/{sy}/{sx}", allow_404=True)
            if shard is None:
                continue
            total_shard_bytes += len(shard)
            if index_loc == "start":
                idx_raw = shard[:idx_bytes]
            else:
                idx_raw = shard[len(shard) - idx_bytes - crc_bytes : len(shard) - crc_bytes]
            offsets_lens = np.frombuffer(idx_raw, dtype="<u8").reshape(ny_in, nx_in, 2)
            base_y = sy * shard_shape[0]
            base_x = sx * shard_shape[1]
            for iy in range(ny_in):
                for ix in range(nx_in):
                    off = int(offsets_lens[iy, ix, 0])
                    ln = int(offsets_lens[iy, ix, 1])
                    if off == empty_sentinel or ln == empty_sentinel:
                        continue
                    payload = shard[off : off + ln]
                    decoded = inner_decoder(payload) if inner_decoder is not None else payload
                    if len(decoded) != expected_inner_bytes:
                        raise OSError(
                            f"Corrupt Zarr v3 inner chunk ({iy},{ix}): expected {expected_inner_bytes} B, got {len(decoded)} B"
                        )
                    tile = np.frombuffer(decoded, dtype=np.uint8).reshape(inner_shape)
                    y0 = base_y + iy * inner_shape[0]
                    x0 = base_x + ix * inner_shape[1]
                    y1 = min(shape[0], y0 + inner_shape[0])
                    x1 = min(shape[1], x0 + inner_shape[1])
                    if y1 > y0 and x1 > x0:
                        out[y0:y1, x0:x1] = tile[: y1 - y0, : x1 - x0]
                        nonempty_inner += 1

    info = {
        "shape": list(shape),
        "shard_shape": list(shard_shape),
        "inner_chunk_shape": list(inner_shape),
        "shard_bytes": int(total_shard_bytes),
        "nonempty_inner_chunks": int(nonempty_inner),
    }
    return out, info


def align_label_map_to_sv(
    label_arr: np.ndarray,
    sv_hw: tuple[int, int],
    *,
    is_binary_mask: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministically align a 2D label/probability map to surface-volume ``(H, W)``.

    Accepts exact 1:1 shape match OR exact integer 2x/4x scale factor in both axes.
    Raises ``ValueError`` on ambiguous non-integer mismatches to prevent silent misregistration.
    """
    arr = np.asarray(label_arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label map, got shape {arr.shape}")
    lh, lw = int(arr.shape[0]), int(arr.shape[1])
    sh, sw = int(sv_hw[0]), int(sv_hw[1])

    if (lh, lw) == (sh, sw):
        return arr, {
            "mode": "exact_1to1",
            "source_shape": [lh, lw],
            "target_shape": [sh, sw],
            "scale_factor": 1,
        }

    # Exact integer upsampling (label is 2x or 4x smaller than SV)
    if sh % lh == 0 and sw % lw == 0 and (sh // lh) == (sw // lw) and (sh // lh) in (2, 4):
        fac = sh // lh
        up = np.repeat(np.repeat(arr, fac, axis=0), fac, axis=1)
        return up, {
            "mode": f"exact_integer_upsample_{fac}x",
            "source_shape": [lh, lw],
            "target_shape": [sh, sw],
            "scale_factor": int(fac),
        }

    # Exact integer downsampling (label is 2x or 4x larger than SV)
    if lh % sh == 0 and lw % sw == 0 and (lh // sh) == (lw // sw) and (lh // sh) in (2, 4):
        fac = lh // sh
        reshaped = arr.reshape(sh, fac, sw, fac)
        if is_binary_mask:
            if arr.dtype == np.bool_ or int(arr.max()) <= 1:
                down = (reshaped.mean(axis=(1, 3)) >= 0.5).astype(arr.dtype)
            else:
                hi = int(arr.max())
                down = np.where(reshaped.mean(axis=(1, 3)) >= 0.5 * hi, hi, 0).astype(arr.dtype)
        else:
            down = np.rint(reshaped.mean(axis=(1, 3))).astype(arr.dtype)
        return down, {
            "mode": f"exact_integer_downsample_{fac}x",
            "source_shape": [lh, lw],
            "target_shape": [sh, sw],
            "scale_factor": round(1.0 / fac, 4),
        }

    raise ValueError(
        f"Label shape {(lh, lw)} does not match surface volume {(sh, sw)} (exact 1:1 or exact 2x/4x required)"
    )


def discover_and_load_reference_labels(
    spec: dict[str, Any],
    sv_hw: tuple[int, int],
    *,
    budget: HttpByteBudget,
    part: str,
    s3_base: str = S3_BASE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Discover human labels via S3 XML API (or local spec override) and fall back to ink TIFF.

    Returns ``(ink_mask, blank_mask, confidence_u8, provenance_info)``.
    """
    # Optional offline/explicit human Zarr v3 override (for unit tests or direct runs)
    explicit_ink_z3 = spec.get("human_inklabels_zarr")
    explicit_sup_z3 = spec.get("human_supervision_zarr")
    if explicit_ink_z3:
        ink_raw, ink_meta = load_zarr3_sharded_2d_u8(
            explicit_ink_z3, budget=budget, part=part, label=f"{part}:human_ink"
        )
        ink_aligned, align_info = align_label_map_to_sv(ink_raw, sv_hw, is_binary_mask=True)
        if explicit_sup_z3:
            sup_raw, sup_meta = load_zarr3_sharded_2d_u8(
                explicit_sup_z3, budget=budget, part=part, label=f"{part}:human_sup"
            )
            sup_aligned, _ = align_label_map_to_sv(sup_raw, sv_hw, is_binary_mask=True)
            sup_mask = sup_aligned >= 128
        else:
            sup_meta = None
            sup_mask = np.ones(sv_hw, dtype=bool)
        ink_mask = (ink_aligned >= 128) & sup_mask
        blank_mask = sup_mask & (ink_aligned == 0)
        return ink_mask, blank_mask, ink_aligned.astype(np.uint8), {
            "label_type": "human_ink_labels_zarr_v3",
            "is_model_derived": False,
            "inklabels_source": _sanitize_source_label(explicit_ink_z3),
            "supervision_source": _sanitize_source_label(explicit_sup_z3),
            "alignment": align_info,
            "inklabels_meta": ink_meta,
            "supervision_meta": sup_meta,
        }

    # Check S3 XML API for compact human labels matching this volume's resolution token.
    s3_prefix = spec.get("ink_labels_s3_prefix")
    match_token = str(spec.get("ink_labels_match_token", ""))
    s3_listing_count = 0
    matched_zarr_dirs: list[tuple[str, int]] = []
    if s3_prefix and str(spec["sv_zarr_url"]).lower().startswith(("http://", "https://")):
        entries = list_s3_prefix_budgeted(
            s3_base, str(s3_prefix), budget, part, label=f"{part}:s3_ink_labels_list"
        )
        s3_listing_count = len(entries)
        # Find inklabels.zarr/0/c/0/0 matching the resolution token
        by_key = dict(entries)
        for key, size in entries:
            if key.endswith("/inklabels.zarr/0/c/0/0") and (not match_token or match_token in key):
                matched_zarr_dirs.append((key[: -len("/inklabels.zarr/0/c/0/0")], int(size)))
        matched_zarr_dirs.sort(key=lambda item: item[0], reverse=True)  # newest dated folder first

        for zdir, ink_shard_sz in matched_zarr_dirs:
            sup_key = f"{zdir}/supervision.zarr/0/c/0/0"
            sup_shard_sz = int(by_key.get(sup_key, 0))
            if ink_shard_sz + sup_shard_sz > HUMAN_LABEL_MAX_SHARD_BYTES:
                continue
            ink_url = f"{s3_base.rstrip('/')}/{zdir}/inklabels.zarr"
            sup_url = f"{s3_base.rstrip('/')}/{zdir}/supervision.zarr" if sup_key in by_key else None
            try:
                ink_raw, ink_meta = load_zarr3_sharded_2d_u8(
                    ink_url, budget=budget, part=part, label=f"{part}:human_ink"
                )
                ink_aligned, align_info = align_label_map_to_sv(ink_raw, sv_hw, is_binary_mask=True)
                sup_meta = None
                if sup_url is not None:
                    sup_raw, sup_meta = load_zarr3_sharded_2d_u8(
                        sup_url, budget=budget, part=part, label=f"{part}:human_sup"
                    )
                    sup_aligned, _ = align_label_map_to_sv(sup_raw, sv_hw, is_binary_mask=True)
                    sup_mask = sup_aligned >= 128
                else:
                    sup_mask = np.ones(sv_hw, dtype=bool)
                ink_mask = (ink_aligned >= 128) & sup_mask
                blank_mask = sup_mask & (ink_aligned == 0)
                return ink_mask, blank_mask, ink_aligned.astype(np.uint8), {
                    "label_type": "human_ink_labels_zarr_v3",
                    "is_model_derived": False,
                    "s3_ink_labels_entries_found": int(s3_listing_count),
                    "inklabels_source": _sanitize_source_label(ink_url),
                    "supervision_source": _sanitize_source_label(sup_url),
                    "alignment": align_info,
                    "inklabels_meta": ink_meta,
                    "supervision_meta": sup_meta,
                }
            except (ValueError, NotImplementedError, OSError):
                continue

    # Fallback: native model-derived ink-detection TIFF (explicitly disclosed!)
    tif_url = str(spec["fallback_ink_tif_url"])
    from PIL import Image as _PILImage

    if tif_url.lower().startswith(("http://", "https://")):
        raw_tif = budgeted_http_get(tif_url, budget, part, label="ink_reference_tif")
        with _PILImage.open(io.BytesIO(raw_tif)) as _im:
            arr_tif = np.asarray(_im)
        del raw_tif
    else:
        with _PILImage.open(tif_url) as _im:
            arr_tif = np.asarray(_im)
    if arr_tif.ndim == 3:
        arr_tif = arr_tif[..., 0]
    ink_u8_raw = np.ascontiguousarray(arr_tif.astype(np.uint8, copy=False))
    ink_u8, align_info = align_label_map_to_sv(ink_u8_raw, sv_hw, is_binary_mask=False)
    ink_mask = ink_u8 >= INK_HIGH_U8_THRESHOLD
    blank_mask = (ink_u8 > 0) & (ink_u8 <= BLANK_LOW_U8_THRESHOLD)
    return ink_mask, blank_mask, ink_u8, {
        "label_type": str(spec.get("fallback_provenance", "model_derived_ink_detection_tif")),
        "is_model_derived": True,
        "s3_ink_labels_entries_found": int(s3_listing_count),
        "s3_ink_labels_matched_for_resolution": len(matched_zarr_dirs),
        "ink_detection_source": _sanitize_source_label(tif_url),
        "ink_high_u8_threshold": INK_HIGH_U8_THRESHOLD,
        "blank_low_u8_threshold": BLANK_LOW_U8_THRESHOLD,
        "alignment": align_info,
    }


def select_chunk_aligned_roi(
    ink_mask: np.ndarray,
    blank_mask: np.ndarray,
    confidence_u8: np.ndarray,
    sv_chunks_hw: tuple[int, int],
    *,
    roi_h: int = DEFAULT_ROI_H,
    roi_w: int = DEFAULT_ROI_W,
    min_ink_frac: float = 0.12,
    max_ink_frac: float = 0.50,
    min_blank_frac: float = 0.30,
    min_labeled_frac: float = 0.70,
) -> dict[str, Any]:
    """Deterministically select a chunk-aligned ROI with strong ink + blank papyrus.

    Uses ONLY the reference label masks and chunk geometry (ZERO peeking at CT intensities).
    """
    h_sv, w_sv = int(ink_mask.shape[0]), int(ink_mask.shape[1])
    cy_sz, cx_sz = int(sv_chunks_hw[0]), int(sv_chunks_hw[1])
    if roi_h > h_sv or roi_w > w_sv:
        raise ValueError(f"ROI {(roi_h, roi_w)} larger than surface volume {(h_sv, w_sv)}")
    if roi_h % cy_sz != 0 or roi_w % cx_sz != 0:
        raise ValueError(f"ROI {(roi_h, roi_w)} must align to chunks {(cy_sz, cx_sz)}")

    n_cy = h_sv // cy_sz
    n_cx = w_sv // cx_sz
    ny_win = roi_h // cy_sz
    nx_win = roi_w // cx_sz

    tile_ink = np.zeros((n_cy, n_cx), dtype=np.int64)
    tile_blank = np.zeros((n_cy, n_cx), dtype=np.int64)
    tile_conf = np.zeros((n_cy, n_cx), dtype=np.int64)

    for ty in range(n_cy):
        y0, y1 = ty * cy_sz, (ty + 1) * cy_sz
        im_s = ink_mask[y0:y1, : n_cx * cx_sz].reshape(cy_sz, n_cx, cx_sz)
        bm_s = blank_mask[y0:y1, : n_cx * cx_sz].reshape(cy_sz, n_cx, cx_sz)
        cf_s = confidence_u8[y0:y1, : n_cx * cx_sz].reshape(cy_sz, n_cx, cx_sz)
        tile_ink[ty] = im_s.sum(axis=(0, 2), dtype=np.int64)
        tile_blank[ty] = bm_s.sum(axis=(0, 2), dtype=np.int64)
        tile_conf[ty] = np.where(im_s, cf_s, 0).sum(axis=(0, 2), dtype=np.int64)

    def _integral(a: np.ndarray) -> np.ndarray:
        out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.int64)
        out[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
        return out

    ii_ink = _integral(tile_ink)
    ii_blank = _integral(tile_blank)
    ii_conf = _integral(tile_conf)
    area = float(roi_h * roi_w)

    def _scan(ink_lo: float, blank_lo: float, lab_lo: float) -> list[tuple[float, int, int, float, float, float]]:
        cands: list[tuple[float, int, int, float, float, float]] = []
        for ty0 in range(0, n_cy - ny_win + 1):
            ty1 = ty0 + ny_win
            r0 = ty0 * cy_sz
            for tx0 in range(0, n_cx - nx_win + 1):
                tx1 = tx0 + nx_win
                c0 = tx0 * cx_sz
                n_i = int(ii_ink[ty1, tx1] - ii_ink[ty0, tx1] - ii_ink[ty1, tx0] + ii_ink[ty0, tx0])
                n_b = int(ii_blank[ty1, tx1] - ii_blank[ty0, tx1] - ii_blank[ty1, tx0] + ii_blank[ty0, tx0])
                if_r = n_i / area
                bf_r = n_b / area
                if if_r < ink_lo or if_r > max_ink_frac or bf_r < blank_lo or (if_r + bf_r) < lab_lo:
                    continue
                sum_c = int(ii_conf[ty1, tx1] - ii_conf[ty0, tx1] - ii_conf[ty1, tx0] + ii_conf[ty0, tx0])
                mean_c = (sum_c / max(n_i, 1)) / 255.0
                # Strongly favor well-covered, balanced ink+blank ROIs
                score = min(if_r, bf_r) * (if_r + bf_r) * mean_c
                cands.append((score, r0, c0, if_r, bf_r, mean_c))
        return cands

    candidates = _scan(min_ink_frac, min_blank_frac, min_labeled_frac)
    relaxed = False
    if not candidates:
        relaxed = True
        candidates = _scan(0.05, 0.20, 0.40)
    if not candidates:
        raise RuntimeError("No chunk-aligned ROI satisfies ink and blank coverage requirements.")

    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    best_score, r0, c0, if_r, bf_r, mean_c = candidates[0]
    return {
        "roi_bounds_yx": [int(r0), int(r0 + roi_h), int(c0), int(c0 + roi_w)],
        "roi_shape_yx": [int(roi_h), int(roi_w)],
        "ink_fraction": round(float(if_r), 4),
        "blank_fraction": round(float(bf_r), 4),
        "labeled_fraction": round(float(if_r + bf_r), 4),
        "mean_ink_confidence": round(float(mean_c), 4),
        "candidates_screened": int(len(candidates)),
        "selection_score": round(float(best_score), 5),
        "selection_relaxed": bool(relaxed),
        "selection_mode": "deterministic_chunk_aligned_label_balance_zero_ct_peek",
    }


# ---------------------------------------------------------------------------
# 3. Cryo-ET / CryoSamba-Inspired Adjacent-Slice Coherence & Rank Fusion
# ---------------------------------------------------------------------------

def _sample_slab_at_depth(slab: np.ndarray, z_map: np.ndarray) -> np.ndarray:
    """Linearly interpolate a 3D slab ``(L, H, W)`` at per-pixel float depth ``z_map (H, W)``."""
    depth, h, w = slab.shape
    z_q = np.clip(np.asarray(z_map, dtype=np.float64), 0.0, float(depth - 1))
    z0 = np.floor(z_q).astype(np.int64)
    z1 = np.minimum(z0 + 1, depth - 1)
    wz = (z_q - z0).astype(np.float64)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    v0 = slab[z0, yy, xx].astype(np.float64)
    v1 = slab[z1, yy, xx].astype(np.float64)
    return (1.0 - wz) * v0 + wz * v1


def _local_patch_cosine_similarity(
    slice_a: np.ndarray,
    slice_b: np.ndarray,
    valid_mask: np.ndarray,
    *,
    patch_sigma_px: float = 2.5,
) -> np.ndarray:
    """Local zero-mean normalized patch cross-correlation (Frobenius cosine similarity) in [-1, 1].

    Strictly invariant to positive affine transforms ``A -> alpha * A + beta`` (``alpha > 0``)
    independently on each slice.
    """
    a = np.asarray(slice_a, dtype=np.float64)
    b = np.asarray(slice_b, dtype=np.float64)
    vm = np.asarray(valid_mask, dtype=bool)
    if not vm.any():
        return np.zeros(a.shape, dtype=np.float64)

    sig = max(0.75, float(patch_sigma_px))
    mu_a = _normalized_gaussian(a, vm, sigma=sig)
    mu_b = _normalized_gaussian(b, vm, sigma=sig)
    da = np.where(vm, a - mu_a, 0.0)
    db = np.where(vm, b - mu_b, 0.0)

    cov_ab = _normalized_gaussian(da * db, vm, sigma=sig)
    var_a = np.maximum(0.0, _normalized_gaussian(da * da, vm, sigma=sig))
    var_b = np.maximum(0.0, _normalized_gaussian(db * db, vm, sigma=sig))
    denom = np.sqrt(var_a * var_b)

    # Scale-adaptive floor preserves exact affine invariance while preventing 0/0 on flat patches.
    scale_a = max(float(np.std(a[vm])), 1e-12)
    scale_b = max(float(np.std(b[vm])), 1e-12)
    floor = 1e-9 * scale_a * scale_b
    good = vm & (denom > floor)
    out = np.zeros(a.shape, dtype=np.float64)
    out[good] = np.clip(cov_ab[good] / denom[good], -1.0, 1.0)
    return out


def cryo_slice_coherence(
    stack_lhw: np.ndarray,
    valid_mask: np.ndarray,
    center_idx: int | float | np.ndarray,
    *,
    patch_sigma_px: float = 2.5,
) -> np.ndarray:
    """Cryo-ET / CryoSamba-inspired local adjacent-slice coherence around papyrus peak.

    Computes local normalized patch cross-correlation / Frobenius cosine similarity
    averaged across adjacent slice pairs ``(k-1, k)`` and ``(k, k+1)`` where ``k`` is
    ``center_idx`` (either a scalar slice index or a 2D per-pixel ``n_peak`` map).
    """
    vol = np.asarray(stack_lhw, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D stack [L, H, W], got {vol.shape}")
    depth, h, w = vol.shape
    if depth < 3:
        raise ValueError(f"stack_lhw depth must be >= 3, got {depth}")

    vm = np.asarray(valid_mask, dtype=bool)
    if vm.shape != (h, w):
        raise ValueError(f"valid_mask shape {vm.shape} != {(h, w)}")
    if not vm.any():
        return np.zeros((h, w), dtype=np.float32)

    if np.ndim(center_idx) == 0:
        c_f = float(np.clip(float(center_idx), 1.0, float(depth - 2)))
        if abs(c_f - round(c_f)) < 1e-9:
            k = int(round(c_f))
            s_prev = vol[k - 1]
            s_curr = vol[k]
            s_next = vol[k + 1]
        else:
            z_map = np.full((h, w), c_f, dtype=np.float64)
            s_prev = _sample_slab_at_depth(vol, z_map - 1.0)
            s_curr = _sample_slab_at_depth(vol, z_map)
            s_next = _sample_slab_at_depth(vol, z_map + 1.0)
    else:
        z_map = np.clip(np.asarray(center_idx, dtype=np.float64), 1.0, float(depth - 2))
        if z_map.shape != (h, w):
            raise ValueError(f"center_idx shape {z_map.shape} != {(h, w)}")
        s_prev = _sample_slab_at_depth(vol, z_map - 1.0)
        s_curr = _sample_slab_at_depth(vol, z_map)
        s_next = _sample_slab_at_depth(vol, z_map + 1.0)

    c_prev = _local_patch_cosine_similarity(s_prev, s_curr, vm, patch_sigma_px=patch_sigma_px)
    c_next = _local_patch_cosine_similarity(s_curr, s_next, vm, patch_sigma_px=patch_sigma_px)
    coh = 0.5 * (c_prev + c_next)
    return np.where(vm, coh, 0.0).astype(np.float32)


def compute_rank_average_fusion(
    feature_maps: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    fixed_polarities: dict[str, int],
    *,
    feature_names: Sequence[str] = FUSION_TRIPLET,
) -> np.ndarray:
    """Unsupervised equal-weight fractional rank-average fusion under fixed signed polarities.

    Uses ZERO ink/blank labels: ranks each constituent over ``valid_mask`` after applying
    its predeclared Exp T5 sign, rescales ranks to ``(0, 1)``, and averages equally.
    """
    vm = np.asarray(valid_mask, dtype=bool)
    h, w = vm.shape
    if not vm.any() or not feature_names:
        return np.zeros((h, w), dtype=np.float32)

    valid_all = vm.copy()
    for name in feature_names:
        valid_all &= np.isfinite(feature_maps[name])
    n_valid = int(valid_all.sum())
    if n_valid == 0:
        return np.zeros((h, w), dtype=np.float32)

    acc = np.zeros(n_valid, dtype=np.float64)
    for name in feature_names:
        sign = 1.0 if int(fixed_polarities[name]) >= 0 else -1.0
        vals = sign * np.asarray(feature_maps[name][valid_all], dtype=np.float64)
        ranks = (rankdata(vals, method="average") - 0.5) / float(n_valid)
        acc += ranks
    acc /= float(len(feature_names))

    out = np.zeros((h, w), dtype=np.float32)
    out[valid_all] = acc.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 4. Feature Computation (On-Sheet & Adversarial Off-Sheet Sub-Slabs)
# ---------------------------------------------------------------------------

def _compute_oct_hilbert_env_peak(
    slab: np.ndarray,
    n_peak: np.ndarray,
    valid_mask: np.ndarray,
    *,
    half_span_vx: float,
    waveform_dim: int = OCT_WAVEFORM_DIM,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact Exp T5 `oct_hilbert_env_peak` (Hilbert analytic envelope at peak center)."""
    profiles, offsets, in_bounds = extract_peak_aligned_depth_profiles(
        slab, n_peak, half_span_vx=half_span_vx, waveform_dim=waveform_dim
    )
    c = profiles.shape[0] // 2
    vm = np.asarray(valid_mask, dtype=bool)
    p64 = profiles.astype(np.float64)
    p_zero = p64 - p64.mean(axis=0, keepdims=True)
    z_shifted = ssig.hilbert(np.fft.ifftshift(p_zero, axes=0), axis=0)
    z_analytic = np.fft.fftshift(z_shifted, axes=0)
    env_peak = np.abs(z_analytic)[c]
    out = np.where(vm, env_peak, 0.0).astype(np.float32)
    diag = {
        "waveform_dim": int(waveform_dim),
        "half_span_vx": float(half_span_vx),
        "step_vx": round(float(offsets[1] - offsets[0]), 4),
        "fully_in_bounds_fraction": round(float(in_bounds[vm].mean()) if vm.any() else 0.0, 4),
    }
    return out, diag


def extract_shifted_subslab(
    slab: np.ndarray,
    n_center: np.ndarray | float,
    shift_slices: float,
    *,
    subslab_depth: int = OFF_SHEET_SUBSLAB_DEPTH,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Extract a ``subslab_depth``-slice off-sheet slab centered at ``n_center + shift_slices``.

    Clamps center strictly into ``[half, depth - 1 - half]`` so the extracted sub-slab is
    always complete and strictly disjoint from the on-sheet peak when ``|shift_slices| >= half + 1``.
    """
    vol = np.asarray(slab, dtype=np.float32)
    depth, h, w = vol.shape
    d_sub = int(subslab_depth)
    if d_sub < 3 or d_sub % 2 == 0 or d_sub > depth:
        raise ValueError(f"subslab_depth must be odd in [3, {depth}], got {d_sub}")
    half = d_sub // 2
    lo_c = float(half)
    hi_c = float(depth - 1 - half)

    if np.ndim(n_center) == 0:
        c_raw = np.full((h, w), float(n_center) + float(shift_slices), dtype=np.float64)
    else:
        c_raw = np.asarray(n_center, dtype=np.float64) + float(shift_slices)
    c_clamped = np.clip(c_raw, lo_c, hi_c)

    sub = np.empty((d_sub, h, w), dtype=np.float32)
    for k, off in enumerate(range(-half, half + 1)):
        sub[k] = _sample_slab_at_depth(vol, c_clamped + float(off)).astype(np.float32)

    mean_eff_shift = float(np.mean(c_clamped - (c_raw - float(shift_slices))))
    info = {
        "requested_shift_slices": float(shift_slices),
        "mean_effective_shift_slices": round(mean_eff_shift, 3),
        "mean_center_slice": round(float(c_clamped.mean()), 3),
        "subslab_depth": int(d_sub),
    }
    return sub, float(half), info


def compute_t7_features_for_slab(
    slab: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    regime: str,
    oct_half_span_vx: float,
    search_half_slices: int,
    z_smooth_sigma: float = 1.0,
    fixed_center_idx: float | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute all Exp T7 features on either an on-sheet slab or a fixed-depth off-sheet sub-slab."""
    vol = np.asarray(slab, dtype=np.float32)
    depth, h, w = vol.shape
    vm = np.asarray(valid_mask, dtype=bool)
    pols = FIXED_T5_POLARITIES[regime]

    if fixed_center_idx is None:
        hf = extract_subvoxel_height_field(
            vol,
            voxel_um=voxel_um,
            valid_mask=vm,
            search_half_slices=search_half_slices,
            z_smooth_sigma=z_smooth_sigma,
            detrend_sigma_um=200.0,
            band_half_width=1,
        )
        sheet_valid = hf["sheet_valid"]
        surface_ct = hf["surface_ct"]
        n_peak = hf["n_peak"]
        hf_diag: dict[str, Any] = {
            "mode": "on_sheet_peak_aligned",
            "z_center_slice": int(hf["z_center_slice"]),
            "z_search_window": hf["z_search_window"],
            "n_peak_median": round(float(np.median(n_peak[sheet_valid])) if sheet_valid.any() else 0.0, 3),
            "sheet_valid_fraction": round(float(sheet_valid.mean()), 4),
        }
    else:
        c_idx = float(np.clip(float(fixed_center_idx), 1.0, float(depth - 2)))
        n_peak = np.full((h, w), c_idx, dtype=np.float32)
        sheet_valid = vm.copy()
        s_m1 = _sample_slab_at_depth(vol, n_peak - 1.0)
        s_0 = _sample_slab_at_depth(vol, n_peak)
        s_p1 = _sample_slab_at_depth(vol, n_peak + 1.0)
        surface_ct = ((s_m1 + s_0 + s_p1) / 3.0).astype(np.float32)
        hf_diag = {
            "mode": "off_sheet_fixed_depth",
            "fixed_center_idx": round(c_idx, 3),
            "subslab_depth": int(depth),
            "sheet_valid_fraction": round(float(sheet_valid.mean()), 4),
        }

    oct_env, oct_diag = _compute_oct_hilbert_env_peak(
        vol, n_peak, sheet_valid, half_span_vx=oct_half_span_vx, waveform_dim=OCT_WAVEFORM_DIM
    )
    wst_all, wst_diag = compute_wavelet_scattering_features(
        surface_ct, sheet_valid, voxel_um=voxel_um
    )
    pc_all, pc_diag = compute_kovesi_phase_congruency_features(
        surface_ct, sheet_valid, voxel_um=voxel_um
    )
    cryo_sigma_px = max(2.0, 20.0 / float(voxel_um))
    cryo_coh = cryo_slice_coherence(
        vol, sheet_valid, n_peak if fixed_center_idx is None else float(fixed_center_idx),
        patch_sigma_px=cryo_sigma_px,
    )

    feats: dict[str, np.ndarray] = {
        "raw_ct_surface": np.where(sheet_valid, surface_ct, 0.0).astype(np.float32),
        "oct_hilbert_env_peak": oct_env,
        "wst_s1_isotropic": wst_all["wst_s1_isotropic"],
        "wst_s1_scale_slope": wst_all["wst_s1_scale_slope"],
        "pc_min_moment_corners_pooled": pc_all["pc_min_moment_corners_pooled"],
        "cryo_slice_coherence": cryo_coh,
    }
    feats["rank_fusion_t5_triplet"] = compute_rank_average_fusion(
        feats, sheet_valid, pols, feature_names=FUSION_TRIPLET
    )

    diagnostics = {
        "height_field": hf_diag,
        "oct_hilbert_env_peak": oct_diag,
        "wavelet_scattering_2d": wst_diag,
        "kovesi_phase_congruency": pc_diag,
        "cryo_slice_coherence": {
            "patch_sigma_px": round(cryo_sigma_px, 3),
            "mean": round(float(cryo_coh[sheet_valid].mean()) if sheet_valid.any() else 0.0, 5),
            "std": round(float(cryo_coh[sheet_valid].std()) if sheet_valid.any() else 0.0, 5),
        },
    }
    return feats, sheet_valid, n_peak, diagnostics


# ---------------------------------------------------------------------------
# 5. Fixed-Polarity + Rolled-Null + Off-Sheet Kill-Test Evaluator
# ---------------------------------------------------------------------------

def evaluate_feature_fixed_and_unsigned(
    feature_map: np.ndarray,
    ink_mask: np.ndarray,
    blank_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    fixed_polarity: int | None,
    compute_rolled_null: bool = True,
) -> dict[str, Any]:
    """Evaluate a feature with predeclared fixed T5 polarity AND diagnostic unsigned polarity."""
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
            "observed_polarity": "+",
            "fixed_t5_polarity": ("+" if fixed_polarity >= 0 else "-") if fixed_polarity is not None else None,
            "fixed_t5_polarity_auc": 0.5 if fixed_polarity is not None else None,
            "unsigned_best_polarity_auc": 0.5,
            "polarity_matches_t5": None,
            "cohens_d_raw": 0.0,
            "cohens_d_fixed": 0.0 if fixed_polarity is not None else None,
            "rolled_null_two_sided_mean": 0.5,
            "rolled_null_two_sided_std": 0.0,
            "rolled_null_two_sided_max": 0.5,
            "rolled_null_signed_mean": 0.5,
            "delta_null_fixed": 0.0 if fixed_polarity is not None else None,
            "delta_null_unsigned": 0.0,
        }

    eval_m = ink | blank
    raw_auc = exact_roc_auc(feat[eval_m], ink[eval_m])
    obs_pol = "+" if raw_auc >= 0.5 else "-"
    unsigned_auc = max(raw_auc, 1.0 - raw_auc)
    d_raw = cohens_d(feat[ink], feat[blank])

    if fixed_polarity is not None:
        sgn = 1 if int(fixed_polarity) >= 0 else -1
        pol_str: str | None = "+" if sgn > 0 else "-"
        fixed_auc: float | None = raw_auc if sgn > 0 else (1.0 - raw_auc)
        d_fixed: float | None = float(sgn) * d_raw
        pol_match: bool | None = (obs_pol == pol_str)
    else:
        sgn = 1
        pol_str = None
        fixed_auc = None
        d_fixed = None
        pol_match = None

    null_two_mean = 0.5
    null_two_std = 0.0
    null_two_max = 0.5
    null_signed_mean = 0.5
    if compute_rolled_null:
        ink_orig = np.asarray(ink_mask, dtype=bool)
        blank_orig = np.asarray(blank_mask, dtype=bool) & (~ink_orig)
        shifts = build_16_rolled_shifts(feat.shape)
        null_raw: list[float] = []
        null_two: list[float] = []
        for dy, dx in shifts:
            ri = np.roll(ink_orig, shift=(dy, dx), axis=(0, 1)) & vm
            rb = np.roll(blank_orig, shift=(dy, dx), axis=(0, 1)) & vm & (~ri)
            if int(ri.sum()) < 10 or int(rb.sum()) < 10:
                null_raw.append(0.5)
                null_two.append(0.5)
                continue
            em = ri | rb
            a_s = exact_roc_auc(feat[em], ri[em])
            a_fixed_s = a_s if sgn > 0 else (1.0 - a_s)
            null_raw.append(a_fixed_s)
            null_two.append(max(a_s, 1.0 - a_s))
        nt_arr = np.asarray(null_two, dtype=np.float64)
        nr_arr = np.asarray(null_raw, dtype=np.float64)
        null_two_mean = float(nt_arr.mean())
        null_two_std = float(nt_arr.std())
        null_two_max = float(nt_arr.max())
        null_signed_mean = float(nr_arr.mean())

    delta_null_fixed = (
        round(float(fixed_auc - null_two_mean), 5) if fixed_auc is not None else None
    )
    delta_null_unsigned = round(float(unsigned_auc - null_two_mean), 5)

    return {
        "n_ink": n_ink,
        "n_blank": n_blank,
        "raw_auc": round(float(raw_auc), 5),
        "observed_polarity": obs_pol,
        "fixed_t5_polarity": pol_str,
        "fixed_t5_polarity_auc": round(float(fixed_auc), 5) if fixed_auc is not None else None,
        "unsigned_best_polarity_auc": round(float(unsigned_auc), 5),
        "polarity_matches_t5": pol_match,
        "cohens_d_raw": round(float(d_raw), 5),
        "cohens_d_fixed": round(float(d_fixed), 5) if d_fixed is not None else None,
        "rolled_null_two_sided_mean": round(null_two_mean, 5),
        "rolled_null_two_sided_std": round(null_two_std, 5),
        "rolled_null_two_sided_max": round(null_two_max, 5),
        "rolled_null_signed_mean": round(null_signed_mean, 5),
        "delta_null_fixed": delta_null_fixed,
        "delta_null_unsigned": delta_null_unsigned,
    }


def evaluate_volume_on_and_off_sheet(
    slab: np.ndarray,
    valid_mask: np.ndarray,
    ink_mask: np.ndarray,
    blank_mask: np.ndarray,
    *,
    voxel_um: float,
    regime: str,
    border_guard_px: int,
    z_smooth_sigma: float = 1.0,
    off_sheet_shift_slices: int = OFF_SHEET_SHIFT_SLICES,
    off_sheet_subslab_depth: int = OFF_SHEET_SUBSLAB_DEPTH,
    gate_min_on_sheet_auc: float = GATE_MIN_ON_SHEET_AUC,
    gate_min_delta_null: float = GATE_MIN_DELTA_NULL,
    gate_min_off_sheet_drop: float = GATE_MIN_OFF_SHEET_DROP,
) -> dict[str, Any]:
    """Run on-sheet evaluation + adversarial +/-8 slice & boundary off-sheet kill-tests."""
    depth = int(slab.shape[0])
    pols = FIXED_T5_POLARITIES[regime]

    if regime == "high_res_2p4um":
        oct_hs = min(OCT_HALF_SPAN_VX_0139, 0.30 * float(depth - 1))
        search_half = min(24, (depth - 1) // 2)
    else:
        oct_hs = min(OCT_HALF_SPAN_VX_0306B, 0.30 * float(depth - 1))
        search_half = min(6, (depth - 1) // 2)

    on_feats, sheet_valid, n_peak, on_diag = compute_t7_features_for_slab(
        slab,
        valid_mask,
        voxel_um=voxel_um,
        regime=regime,
        oct_half_span_vx=oct_hs,
        search_half_slices=search_half,
        z_smooth_sigma=z_smooth_sigma,
        fixed_center_idx=None,
    )

    # Consistent interior evaluation mask across on-sheet and all off-sheet probes.
    eval_valid = sheet_valid.copy()
    guard = min(int(border_guard_px), min(eval_valid.shape) // 8)
    if guard > 0:
        eval_valid[:guard, :] = False
        eval_valid[-guard:, :] = False
        eval_valid[:, :guard] = False
        eval_valid[:, -guard:] = False

    # Build 4 adversarial off-sheet sub-slabs:
    # 1) minus8: shifted -8 slices from n_peak
    # 2) plus8:  shifted +8 slices from n_peak
    # 3) boundary_low:  farthest bottom boundary slice 0..4
    # 4) boundary_high: farthest top boundary slice L-5..L-1
    d_sub = min(int(off_sheet_subslab_depth), depth)
    if d_sub % 2 == 0:
        d_sub = max(3, d_sub - 1)
    sub_half = float(d_sub // 2)

    sub_m8, c_m8, info_m8 = extract_shifted_subslab(
        slab, n_peak, -float(off_sheet_shift_slices), subslab_depth=d_sub
    )
    sub_p8, c_p8, info_p8 = extract_shifted_subslab(
        slab, n_peak, +float(off_sheet_shift_slices), subslab_depth=d_sub
    )
    sub_blo = slab[:d_sub]
    sub_bhi = slab[depth - d_sub :]
    info_blo = {
        "slice_range": [0, int(d_sub)],
        "center_slice": float(sub_half),
        "subslab_depth": int(d_sub),
    }
    info_bhi = {
        "slice_range": [int(depth - d_sub), int(depth)],
        "center_slice": float(depth - d_sub) + float(sub_half),
        "subslab_depth": int(d_sub),
    }

    off_probes = {
        "shift_minus8": (sub_m8, c_m8, info_m8),
        "shift_plus8": (sub_p8, c_p8, info_p8),
        "boundary_low": (sub_blo, sub_half, info_blo),
        "boundary_high": (sub_bhi, sub_half, info_bhi),
    }

    off_feature_maps: dict[str, dict[str, np.ndarray]] = {}
    off_probe_info: dict[str, dict[str, Any]] = {}
    for pname, (pslab, pcenter, pinfo) in off_probes.items():
        pfeats, _, _, _ = compute_t7_features_for_slab(
            pslab,
            sheet_valid,
            voxel_um=voxel_um,
            regime=regime,
            oct_half_span_vx=sub_half,
            search_half_slices=1,
            z_smooth_sigma=z_smooth_sigma,
            fixed_center_idx=pcenter,
        )
        off_feature_maps[pname] = pfeats
        off_probe_info[pname] = pinfo

    feature_results: dict[str, dict[str, Any]] = {}
    for fname in ALL_FEATURE_NAMES:
        is_expl = fname in EXPLORATORY_FEATURES
        is_base = fname in BASELINE_FEATURES
        fpol = None if is_expl else pols[fname]

        on_res = evaluate_feature_fixed_and_unsigned(
            on_feats[fname],
            ink_mask,
            blank_mask,
            eval_valid,
            fixed_polarity=fpol,
            compute_rolled_null=True,
        )
        off_evals: dict[str, dict[str, Any]] = {}
        for pname in off_probes:
            off_evals[pname] = evaluate_feature_fixed_and_unsigned(
                off_feature_maps[pname][fname],
                ink_mask,
                blank_mask,
                eval_valid,
                fixed_polarity=fpol,
                compute_rolled_null=False,
            )

        if not is_expl:
            on_auc = float(on_res["fixed_t5_polarity_auc"])
            d_null = float(on_res["delta_null_fixed"])
            m8_auc = float(off_evals["shift_minus8"]["fixed_t5_polarity_auc"])
            p8_auc = float(off_evals["shift_plus8"]["fixed_t5_polarity_auc"])
            blo_auc = float(off_evals["boundary_low"]["fixed_t5_polarity_auc"])
            bhi_auc = float(off_evals["boundary_high"]["fixed_t5_polarity_auc"])

            pm8_mean = 0.5 * (m8_auc + p8_auc)
            pm8_max = max(m8_auc, p8_auc)
            bnd_mean = 0.5 * (blo_auc + bhi_auc)
            bnd_max = max(blo_auc, bhi_auc)

            drop_pm8_mean = on_auc - pm8_mean
            drop_pm8_max = on_auc - pm8_max
            drop_bnd_mean = on_auc - bnd_mean
            drop_bnd_max = on_auc - bnd_max

            pass_auc = on_auc >= float(gate_min_on_sheet_auc)
            pass_null = d_null >= float(gate_min_delta_null)
            pass_pm8 = drop_pm8_mean >= float(gate_min_off_sheet_drop)
            pass_bnd = drop_bnd_mean >= float(gate_min_off_sheet_drop)

            # Unsigned diagnostic (NEVER used to claim PASS!)
            u_on = float(on_res["unsigned_best_polarity_auc"])
            u_pm8 = 0.5 * (
                float(off_evals["shift_minus8"]["unsigned_best_polarity_auc"])
                + float(off_evals["shift_plus8"]["unsigned_best_polarity_auc"])
            )
            u_bnd = 0.5 * (
                float(off_evals["boundary_low"]["unsigned_best_polarity_auc"])
                + float(off_evals["boundary_high"]["unsigned_best_polarity_auc"])
            )

            feature_results[fname] = {
                "role": "baseline" if is_base else "gated_transverse",
                "exploratory": False,
                "n_ink": on_res["n_ink"],
                "n_blank": on_res["n_blank"],
                "fixed_t5_polarity": on_res["fixed_t5_polarity"],
                "observed_polarity": on_res["observed_polarity"],
                "polarity_matches_t5": on_res["polarity_matches_t5"],
                "on_sheet_auc_fixed": round(on_auc, 5),
                "cohens_d_fixed": on_res["cohens_d_fixed"],
                "rolled_null_two_sided_mean": on_res["rolled_null_two_sided_mean"],
                "rolled_null_two_sided_std": on_res["rolled_null_two_sided_std"],
                "rolled_null_two_sided_max": on_res["rolled_null_two_sided_max"],
                "rolled_null_signed_mean": on_res["rolled_null_signed_mean"],
                "delta_null_fixed": round(d_null, 5),
                "off_sheet_minus8_auc_fixed": round(m8_auc, 5),
                "off_sheet_plus8_auc_fixed": round(p8_auc, 5),
                "off_sheet_pm8_mean_auc_fixed": round(pm8_mean, 5),
                "off_sheet_pm8_max_auc_fixed": round(pm8_max, 5),
                "drop_vs_pm8_fixed": round(drop_pm8_mean, 5),
                "drop_vs_pm8_max_fixed": round(drop_pm8_max, 5),
                "off_sheet_boundary_low_auc_fixed": round(blo_auc, 5),
                "off_sheet_boundary_high_auc_fixed": round(bhi_auc, 5),
                "off_sheet_boundary_mean_auc_fixed": round(bnd_mean, 5),
                "off_sheet_boundary_max_auc_fixed": round(bnd_max, 5),
                "drop_vs_boundary_fixed": round(drop_bnd_mean, 5),
                "drop_vs_boundary_max_fixed": round(drop_bnd_max, 5),
                "passes_on_sheet_auc": bool(pass_auc),
                "passes_delta_null": bool(pass_null),
                "passes_off_sheet_pm8_drop": bool(pass_pm8),
                "passes_off_sheet_boundary_drop": bool(pass_bnd),
                "passes_gate_pm8": bool(pass_auc and pass_null and pass_pm8),
                "passes_gate_boundary": bool(pass_auc and pass_null and pass_bnd),
                "passes_gate_strict_both": bool(pass_auc and pass_null and pass_pm8 and pass_bnd),
                "unsigned_best_polarity_diagnostic": {
                    "note": "Diagnostic only; NEVER used to claim PASS.",
                    "on_sheet_raw_auc": on_res["raw_auc"],
                    "on_sheet_unsigned_auc": round(u_on, 5),
                    "delta_null_unsigned": on_res["delta_null_unsigned"],
                    "off_sheet_pm8_mean_unsigned_auc": round(u_pm8, 5),
                    "off_sheet_boundary_mean_unsigned_auc": round(u_bnd, 5),
                    "drop_vs_pm8_unsigned": round(u_on - u_pm8, 5),
                    "drop_vs_boundary_unsigned": round(u_on - u_bnd, 5),
                },
            }
        else:
            # Exploratory feature (cryo_slice_coherence): report BOTH polarities honestly!
            raw_on = float(on_res["raw_auc"])
            plus_on = raw_on
            minus_on = 1.0 - raw_on
            best_pol = "+" if plus_on >= 0.5 else "-"
            best_sgn = +1 if best_pol == "+" else -1

            def _orient(raw_v: float, sgn: int) -> float:
                return raw_v if sgn > 0 else (1.0 - raw_v)

            pol_reports: dict[str, dict[str, Any]] = {}
            for p_lbl, sgn in (("plus", +1), ("minus", -1)):
                on_p = _orient(raw_on, sgn)
                m8_p = _orient(float(off_evals["shift_minus8"]["raw_auc"]), sgn)
                p8_p = _orient(float(off_evals["shift_plus8"]["raw_auc"]), sgn)
                blo_p = _orient(float(off_evals["boundary_low"]["raw_auc"]), sgn)
                bhi_p = _orient(float(off_evals["boundary_high"]["raw_auc"]), sgn)
                pm8_m = 0.5 * (m8_p + p8_p)
                bnd_m = 0.5 * (blo_p + bhi_p)
                dn_p = on_p - float(on_res["rolled_null_two_sided_mean"])
                pol_reports[p_lbl] = {
                    "on_sheet_auc": round(on_p, 5),
                    "delta_null": round(dn_p, 5),
                    "off_sheet_minus8_auc": round(m8_p, 5),
                    "off_sheet_plus8_auc": round(p8_p, 5),
                    "off_sheet_pm8_mean_auc": round(pm8_m, 5),
                    "drop_vs_pm8": round(on_p - pm8_m, 5),
                    "off_sheet_boundary_low_auc": round(blo_p, 5),
                    "off_sheet_boundary_high_auc": round(bhi_p, 5),
                    "off_sheet_boundary_mean_auc": round(bnd_m, 5),
                    "drop_vs_boundary": round(on_p - bnd_m, 5),
                }

            best_rep = pol_reports["plus" if best_sgn > 0 else "minus"]
            feature_results[fname] = {
                "role": "exploratory_cryo_et",
                "exploratory": True,
                "n_ink": on_res["n_ink"],
                "n_blank": on_res["n_blank"],
                "observed_polarity": best_pol,
                "cohens_d_raw": on_res["cohens_d_raw"],
                "rolled_null_two_sided_mean": on_res["rolled_null_two_sided_mean"],
                "rolled_null_two_sided_std": on_res["rolled_null_two_sided_std"],
                "rolled_null_two_sided_max": on_res["rolled_null_two_sided_max"],
                "polarity_plus": pol_reports["plus"],
                "polarity_minus": pol_reports["minus"],
                "best_polarity_summary": best_rep,
            }

    # Honest per-volume verdict strictly from GATED_TRANSVERSE_FEATURES under fixed_t5_polarity!
    passing_strict_both = [
        k for k in GATED_TRANSVERSE_FEATURES if feature_results[k]["passes_gate_strict_both"]
    ]
    passing_pm8 = [
        k for k in GATED_TRANSVERSE_FEATURES if feature_results[k]["passes_gate_pm8"]
    ]
    passing_boundary = [
        k for k in GATED_TRANSVERSE_FEATURES if feature_results[k]["passes_gate_boundary"]
    ]

    if passing_strict_both:
        verdict = "PASS"
    elif passing_pm8 or passing_boundary:
        verdict = "PARTIAL"
    else:
        verdict = "CLOSED NO-GO"

    best_gated_name = max(
        GATED_TRANSVERSE_FEATURES,
        key=lambda k: (
            feature_results[k]["on_sheet_auc_fixed"],
            feature_results[k]["delta_null_fixed"],
        ),
    )

    summary = {
        "regime": regime,
        "border_guard_px": int(guard),
        "eval_valid_fraction": round(float(eval_valid.mean()), 4),
        "off_sheet_shift_slices": int(off_sheet_shift_slices),
        "off_sheet_shift_um": round(float(off_sheet_shift_slices * voxel_um), 3),
        "off_sheet_probes": off_probe_info,
        "passing_strict_both": passing_strict_both,
        "passing_pm8_gate": passing_pm8,
        "passing_boundary_gate": passing_boundary,
        "best_gated_transverse_fixed_polarity": {
            "name": best_gated_name,
            "on_sheet_auc_fixed": feature_results[best_gated_name]["on_sheet_auc_fixed"],
            "delta_null_fixed": feature_results[best_gated_name]["delta_null_fixed"],
            "drop_vs_pm8_fixed": feature_results[best_gated_name]["drop_vs_pm8_fixed"],
            "drop_vs_boundary_fixed": feature_results[best_gated_name]["drop_vs_boundary_fixed"],
            "passes_gate_strict_both": feature_results[best_gated_name]["passes_gate_strict_both"],
        },
        "verdict": verdict,
    }
    return {
        "diagnostics": on_diag,
        "summary": summary,
        "features": feature_results,
    }


# ---------------------------------------------------------------------------
# 6. Full Live / Offline End-to-End Pipeline
# ---------------------------------------------------------------------------

def run_cross_scroll_replication(
    *,
    volume_specs: dict[str, dict[str, Any]] | None = None,
    per_volume_budget_mib: float = DEFAULT_PER_VOLUME_BUDGET_MIB,
    global_budget_mib: float = DEFAULT_GLOBAL_BUDGET_MIB,
    roi_h: int = DEFAULT_ROI_H,
    roi_w: int = DEFAULT_ROI_W,
    roi_overrides: dict[str, tuple[int, int, int, int]] | None = None,
    off_sheet_shift_slices: int = OFF_SHEET_SHIFT_SLICES,
) -> dict[str, Any]:
    """Run Exp T7 across PHerc0841 (2.403 & 9.366 um) and PHerc0343 (8.64 um) strictly on CPU in RAM."""
    t0 = time.monotonic()
    specs = {k: dict(v) for k, v in (volume_specs or DEFAULT_VOLUME_SPECS).items()}
    vol_keys = [k for k in VOLUME_ORDER if k in specs] + [k for k in specs if k not in VOLUME_ORDER]

    budget = MultiVolumeHttpByteBudget(
        {k: int(round(per_volume_budget_mib * MIB)) for k in vol_keys},
        global_limit_bytes=int(round(global_budget_mib * MIB)),
    )

    volume_results: dict[str, Any] = {}
    with ExitStack() as stack:
        for vkey in vol_keys:
            spec = specs[vkey]
            voxel_um = float(spec["voxel_um"])
            regime = str(spec["regime"])

            # Step 1: Inspect .zarray shape/chunks/axes FIRST!
            reader = stack.enter_context(
                BudgetedRemoteZarrV2(
                    spec["sv_zarr_url"],
                    level="0",
                    budget=budget,
                    part=vkey,
                    label=f"{vkey}:sv_zarr",
                    cache_mb=64.0,
                    workers=8,
                )
            )
            sv_depth, sv_h, sv_w = int(reader.shape[0]), int(reader.shape[1]), int(reader.shape[2])
            sv_hw = (sv_h, sv_w)

            # Step 2: Discover human labels via S3 XML API first (prefer compact human labels),
            # else fall back to native ink-detection TIFF with explicit provenance.
            ink_mask_full, blank_mask_full, conf_u8_full, label_prov = discover_and_load_reference_labels(
                spec, sv_hw, budget=budget, part=vkey
            )

            # Step 3: Select chunk-aligned ROI (ZERO peeking at CT pixels!) & pre-flight byte check.
            if roi_overrides and vkey in roi_overrides:
                r0, r1, c0, c1 = [int(v) for v in roi_overrides[vkey]]
                im_r = ink_mask_full[r0:r1, c0:c1]
                bm_r = blank_mask_full[r0:r1, c0:c1]
                roi_info: dict[str, Any] = {
                    "roi_bounds_yx": [r0, r1, c0, c1],
                    "roi_shape_yx": [r1 - r0, c1 - c0],
                    "ink_fraction": round(float(im_r.mean()), 4),
                    "blank_fraction": round(float(bm_r.mean()), 4),
                    "labeled_fraction": round(float((im_r | bm_r).mean()), 4),
                    "selection_mode": "override",
                }
            else:
                eff_h = min(int(roi_h), (sv_h // int(reader.chunks[1])) * int(reader.chunks[1]))
                eff_w = min(int(roi_w), (sv_w // int(reader.chunks[2])) * int(reader.chunks[2]))
                roi_info = select_chunk_aligned_roi(
                    ink_mask_full,
                    blank_mask_full,
                    conf_u8_full,
                    (int(reader.chunks[1]), int(reader.chunks[2])),
                    roi_h=eff_h,
                    roi_w=eff_w,
                )

            r0, r1, c0, c1 = [int(v) for v in roi_info["roi_bounds_yx"]]
            roi_info["roi_physical_mm"] = [
                round((r1 - r0) * voxel_um / 1000.0, 4),
                round((c1 - c0) * voxel_um / 1000.0, 4),
            ]

            n_ch, planned_bytes = reader.plan_box_chunks((0, r0, c0), (sv_depth, r1, c1))
            budget.check_planned(vkey, planned_bytes, label=f"{vkey}:sv_roi")
            roi_info["planned_sv_chunks"] = int(n_ch)
            roi_info["planned_sv_bytes"] = int(planned_bytes)
            roi_info["planned_sv_mib"] = round(planned_bytes / MIB, 4)

            # Crop reference masks & free full-canvas arrays immediately to keep RAM low.
            ink_roi = ink_mask_full[r0:r1, c0:c1].copy()
            blank_roi = blank_mask_full[r0:r1, c0:c1].copy()
            del ink_mask_full, blank_mask_full, conf_u8_full

            # Step 4: Stream ROI slab strictly in RAM and run on-sheet + off-sheet kill-tests.
            sv_slab, _, sv_support = reader.read_box(
                (0, r0, c0), (sv_depth, r1, c1), return_support=True
            )
            valid_yx = sv_support.all(axis=0) & (sv_slab.max(axis=0) > 0)

            eval_out = evaluate_volume_on_and_off_sheet(
                sv_slab,
                valid_yx,
                ink_roi,
                blank_roi,
                voxel_um=voxel_um,
                regime=regime,
                border_guard_px=int(spec.get("border_guard_px", 16)),
                z_smooth_sigma=float(spec.get("z_smooth_sigma", 1.0)),
                off_sheet_shift_slices=int(off_sheet_shift_slices),
            )

            source_support = {
                "full_source_available_fraction": round(float(sv_support.all(axis=0).mean()), 4),
                "nonzero_intensity_fraction": round(float((sv_slab.max(axis=0) > 0).mean()), 4),
                "valid_mask_fraction": round(float(valid_yx.mean()), 4),
                "chunks_fetched": int(reader.chunks_fetched),
                "bytes_fetched": int(reader.bytes_fetched),
            }
            vol_shape = list(reader.shape)
            vol_chunks = list(reader.chunks)

            # Free 3D slab immediately before moving to the next volume.
            del sv_slab, sv_support
            reader.close()

            volume_results[vkey] = {
                "scroll": str(spec["scroll"]),
                "segment": str(spec["segment"]),
                "voxel_um": voxel_um,
                "regime": regime,
                "surface_volume_source": _sanitize_source_label(spec["sv_zarr_url"]),
                "surface_volume_shape": vol_shape,
                "surface_volume_chunks": vol_chunks,
                "reference_label_provenance": label_prov,
                "roi": roi_info,
                "source_support": source_support,
                "diagnostics": eval_out["diagnostics"],
                "summary": eval_out["summary"],
                "features": eval_out["features"],
            }

    per_vol_verdicts = {k: volume_results[k]["summary"]["verdict"] for k in vol_keys}
    n_pass = sum(1 for v in per_vol_verdicts.values() if v == "PASS")
    n_partial = sum(1 for v in per_vol_verdicts.values() if v == "PARTIAL")
    if n_pass == len(vol_keys):
        overall_verdict = "PASS"
    elif n_pass > 0 or n_partial > 0:
        overall_verdict = "PARTIAL"
    else:
        overall_verdict = "CLOSED NO-GO"

    receipt: dict[str, Any] = {
        "experiment_id": "exp_t7_cross_scroll_replication_offsheet_killtest_cryo_coherence",
        "compute_mode": "CPU_ONLY",
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "controls_status": "held_out_public_controls_zero_shot_from_exp_t5",
        "fixed_t5_polarities": {
            reg: {k: ("+" if s > 0 else "-") for k, s in mp.items()}
            for reg, mp in FIXED_T5_POLARITIES.items()
        },
        "fusion_definition": {
            "name": "rank_fusion_t5_triplet",
            "method": "unsupervised_equal_weight_fractional_rank_average",
            "constituents": list(FUSION_TRIPLET),
            "polarity_source": "fixed_t5_polarity_locked_before_held_out_evaluation",
        },
        "predeclared_gate": {
            "min_on_sheet_auc_fixed": GATE_MIN_ON_SHEET_AUC,
            "min_delta_null_fixed": GATE_MIN_DELTA_NULL,
            "min_off_sheet_drop_fixed": GATE_MIN_OFF_SHEET_DROP,
            "off_sheet_shift_slices": int(off_sheet_shift_slices),
            "off_sheet_boundary_depth": int(OFF_SHEET_SUBSLAB_DEPTH),
            "rolled_null_shifts": 16,
            "null_statistic": "conservative_two_sided_max(auc, 1-auc)_across_16_toroidal_shifts",
            "gated_features": list(GATED_TRANSVERSE_FEATURES),
            "exploratory_features": sorted(EXPLORATORY_FEATURES),
            "per_volume_verdict_rule": (
                "PASS iff >=1 gated transverse feature passes on_sheet>=0.70, delta_null>=+0.10, "
                "drop_vs_pm8>=+0.08 AND drop_vs_boundary>=+0.08 under fixed_t5_polarity; "
                "PARTIAL iff >=1 passes on_sheet>=0.70, delta_null>=+0.10 and at least one of "
                "drop_vs_pm8>=+0.08 or drop_vs_boundary>=+0.08; else CLOSED NO-GO."
            ),
        },
        "http_byte_budget": budget.summary(),
        "volumes": volume_results,
        "verdict": {
            "overall": overall_verdict,
            "per_volume": per_vol_verdicts,
            "passing_strict_both_by_volume": {
                k: volume_results[k]["summary"]["passing_strict_both"] for k in vol_keys
            },
            "passing_pm8_by_volume": {
                k: volume_results[k]["summary"]["passing_pm8_gate"] for k in vol_keys
            },
            "passing_boundary_by_volume": {
                k: volume_results[k]["summary"]["passing_boundary_gate"] for k in vol_keys
            },
        },
    }
    assert_no_local_paths(receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exp T7: Cross-Scroll Zero-Shot Replication, Off-Sheet Kill-Test (+/-8 slices), "
            "& Cryo-ET Slice Coherence (CPU-only, strictly <= 40 MiB/vol, <= 120 MiB total)."
        )
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(__file__).resolve().parent / "evidence" / "transverse_cross_scroll_replication.json",
        help="Destination path for sanitized JSON receipt.",
    )
    parser.add_argument("--per-volume-budget-mib", type=float, default=DEFAULT_PER_VOLUME_BUDGET_MIB)
    parser.add_argument("--global-budget-mib", type=float, default=DEFAULT_GLOBAL_BUDGET_MIB)
    parser.add_argument("--roi-h", type=int, default=DEFAULT_ROI_H)
    parser.add_argument("--roi-w", type=int, default=DEFAULT_ROI_W)
    args = parser.parse_args(argv)

    receipt = run_cross_scroll_replication(
        per_volume_budget_mib=args.per_volume_budget_mib,
        global_budget_mib=args.global_budget_mib,
        roi_h=args.roi_h,
        roi_w=args.roi_w,
    )
    assert_no_local_paths(receipt)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    b = receipt["http_byte_budget"]
    v = receipt["verdict"]
    vol_str = ", ".join(f"{k}={v['per_volume'][k]}" for k in v["per_volume"])
    mib_str = ", ".join(f"{k}={b['per_volume'][k]['mib']:.2f}MiB" for k in b["per_volume"])
    print(
        f"[Exp T7] overall={v['overall']} ({vol_str}) | "
        f"HTTP: {mib_str}, Total={b['global_mib']:.2f}/{b['global_limit_mib']:.2f} MiB | "
        f"elapsed={receipt['elapsed_seconds']:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
