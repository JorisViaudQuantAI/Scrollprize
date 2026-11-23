# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""CPU-only Exp T8 (Q31 + Q34): Cloud-Masked Opposite-Face Stroke-Bounded TFCE Rescue.

Tests the ``collab/claude-018`` Question ``Q31`` failure mode on three real public
Vesuvius Challenge S3 ink probability maps (``PHerc0343`` ``argus-v3``, ``PHerc0343``
``reader-v3``, and ``PHerc0841`` ``argus-v3``):

1. Downloads only the 3 public S3 ink TIFFs under a hard 35 MiB global HTTP byte cap
   (preflight ``HEAD`` check + streaming ``HttpByteBudget``), converts ``uint8 [0..255]``
   to ``float32 [0..1]``, and infers exact interior valid masks excluding border
   zero-padding.
2. Evaluates unmodified real ``512 x 2048`` text-rich strip crops (and independent
   blank margin controls) with Exp T6's :func:`tfce_stroke_calibration.stroke_bounded_tfce_2d`
   and :func:`tfce_stroke_calibration.tfce_opposite_face_fwer` (``q34_score``,
   ``max_tfce_ratio``, ``p995_tfce_ratio``, ``diff_tfce_p99``).
3. Stress-tests ``Q31`` by injecting smooth isotropic ``2-4 mm`` Gaussian-blob clouds
   (``p ~ 0.85-0.92``) at identical ``(y, x)`` coordinates onto BOTH ``prob_dom`` and
   ``prob_opp`` at ``0%, 10%, 20%, 30%`` valid surface area fraction (outside protected
   true-text tiles) and comparing three per-``512x512``-tile ranking/rescue pipelines:
     * **Baseline (A)**: Standard global per-surface ``z > 2`` hot-share;
     * **Rescue (B)**: Bilateral-cloud-masked robust MAD z-score
       (``mask = min(prob_dom, prob_opp) < 0.45``, score ``(prob_dom - prob_opp).clip(0)``);
     * **Rescue (C)**: Cloud-Masked Opposite-Face Differential + ``stroke_bounded_tfce_2d``
       (ranked by SB-TFCE ``p995`` / ``max`` with ``q34_score`` verification).
4. Also runs a supplementary ``20%``-cloud inter-face parallax ablation (``0.15 mm``
   shift between faces) demonstrating why Rescue (C)'s stroke-width bounding provides
   additional protection beyond pure bilateral masking alone.

Strictly CPU-only: pure NumPy + SciPy + PIL/tifffile; never imports torch or CUDA.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import tfce_stroke_calibration as tfce_cal  # noqa: E402
import transverse_stroke_topology as topo_mod  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

GLOBAL_HTTP_BYTE_CAP = 35 * 1024 * 1024  # 35 MiB hard ceiling
S3_BASE_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"

DEFAULT_TILE_SIZE = 512
DEFAULT_STRIP_SHAPE_HW = (512, 2048)
DEFAULT_MIN_TILE_VALID_FRAC = 0.75
DEFAULT_BILATERAL_CLOUD_THRESHOLD = 0.45
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_MAD_SIGMA_FLOOR = 0.02
DEFAULT_CONTAMINATION_FRACTIONS = (0.0, 0.10, 0.20, 0.30)
DEFAULT_CLOUD_DIAMETER_MM_RANGE = (2.0, 4.0)
DEFAULT_CLOUD_PEAK_RANGE = (0.85, 0.92)
DEFAULT_CLOUD_TILE_MIN_FRAC = 0.05
DEFAULT_PROTECTED_TEXT_FRAC = 0.20  # Top 20% valid tiles protected as clean true-text set
DEFAULT_SEEDS = (31, 131)

_CONN4 = ndi.generate_binary_structure(2, 1)
_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


@dataclass(frozen=True)
class PublicInkMapSpec:
    """Metadata for a public S3 ink probability TIFF."""

    map_id: str
    scroll: str
    model: str
    voxel_um: float
    s3_key: str

    @property
    def url(self) -> str:
        return f"{S3_BASE_URL}{self.s3_key}"


PUBLIC_INK_MAPS: tuple[PublicInkMapSpec, ...] = (
    PublicInkMapSpec(
        map_id="pherc0343_argus_v3",
        scroll="PHerc0343",
        model="argus-v3",
        voxel_um=8.64,
        s3_key=(
            "PHerc0343/segments/20261007213123-concat_w047-w048_R5B2_z9500-11000/"
            "ink-detection/PHerc0343-20261007213123-8.64um-1.2m-116keV-"
            "volume-20250521140437-20261117205746-argus-v3-tile384-stride192.tif"
        ),
    ),
    PublicInkMapSpec(
        map_id="pherc0343_reader_v3",
        scroll="PHerc0343",
        model="reader-v3",
        voxel_um=8.64,
        s3_key=(
            "PHerc0343/segments/20261007213123-concat_w047-w048_R5B2_z9500-11000/"
            "ink-detection/PHerc0343-20261007213123-8.64um-1.2m-116keV-"
            "volume-20250521140437-20261019111505-reader-v3-tile384-stride192.tif"
        ),
    ),
    PublicInkMapSpec(
        map_id="pherc0841_argus_v3",
        scroll="PHerc0841",
        model="argus-v3",
        voxel_um=9.366,
        s3_key=(
            "PHerc0841/segments/20260220213127-w00/ink-detection/"
            "PHerc0841-20260220213127-9.366um-1.2m-113keV-"
            "volume-20250821151531-20261117205746-argus-v3-tile384-stride192.tif"
        ),
    ),
)


def preflight_head_sizes(
    specs: tuple[PublicInkMapSpec, ...] = PUBLIC_INK_MAPS,
    *,
    cap_bytes: int = GLOBAL_HTTP_BYTE_CAP,
    timeout: float = 30.0,
) -> dict:
    """Issue lightweight HTTP HEAD requests to verify cumulative file size <= ``cap_bytes``."""
    items = []
    total_bytes = 0
    for spec in specs:
        req = urllib.request.Request(
            spec.url,
            method="HEAD",
            headers={"User-Agent": "scrollprize-q31-rescue/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cl_hdr = resp.headers.get("Content-Length")
            if cl_hdr is None:
                raise RuntimeError(f"Missing Content-Length header for {spec.url}")
            size_b = int(cl_hdr)
        total_bytes += size_b
        items.append(
            {
                "map_id": spec.map_id,
                "url": spec.url,
                "content_length_bytes": size_b,
                "content_length_mib": round(size_b / (1024.0 * 1024.0), 4),
            }
        )
    if total_bytes > int(cap_bytes):
        raise RuntimeError(
            f"Preflight total {total_bytes} B exceeds HTTP byte cap {cap_bytes} B"
        )
    return {
        "cap_bytes": int(cap_bytes),
        "cap_mib": round(cap_bytes / (1024.0 * 1024.0), 4),
        "total_bytes": int(total_bytes),
        "total_mib": round(total_bytes / (1024.0 * 1024.0), 4),
        "within_cap": True,
        "files": items,
    }


def decode_ink_tiff_bytes(raw_bytes: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    """Decode an ink TIFF payload into ``(prob_f32, raw_2d, metadata)``.

    Uses Pillow first (built-in libtiff LZW support without requiring ``imagecodecs``)
    and falls back to ``tifffile`` if Pillow cannot decode the file. Converts
    ``uint8 [0..255]`` cleanly to ``float32 [0..1]``.
    """
    decoder = "PIL"
    arr: np.ndarray | None = None
    pil_mode: str | None = None
    try:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            pil_mode = str(im.mode)
            arr = np.asarray(im)
    except Exception:
        import tifffile

        decoder = "tifffile"
        arr = tifffile.imread(io.BytesIO(raw_bytes))

    if arr.ndim == 3:
        orig_shape = list(arr.shape)
        arr = arr[..., 0]
    elif arr.ndim == 2:
        orig_shape = list(arr.shape)
    else:
        raise ValueError(f"Expected 2D or 3D TIFF image, got shape {arr.shape!r}")

    dtype_str = str(arr.dtype)
    if arr.dtype == np.uint8:
        prob = (arr.astype(np.float32) / 255.0).astype(np.float32)
    elif arr.dtype == np.uint16:
        prob = (arr.astype(np.float32) / 65535.0).astype(np.float32)
    else:
        prob = np.clip(arr.astype(np.float32), 0.0, 1.0)

    meta = {
        "decoder": decoder,
        "pil_mode": pil_mode,
        "orig_shape": orig_shape,
        "shape_hw": [int(prob.shape[0]), int(prob.shape[1])],
        "dtype": dtype_str,
        "raw_min": float(np.nanmin(arr)),
        "raw_max": float(np.nanmax(arr)),
        "prob_min": round(float(np.nanmin(prob)), 6),
        "prob_max": round(float(np.nanmax(prob)), 6),
    }
    return prob, arr, meta


def infer_valid_mask(raw_array: np.ndarray) -> tuple[np.ndarray, dict]:
    """Infer active interior surface mask by excluding border-connected zero padding."""
    arr = np.asarray(raw_array)
    if arr.ndim != 2:
        raise ValueError(f"raw_array must be 2D, got shape {arr.shape!r}")
    finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.floating) else np.ones(arr.shape, dtype=bool)
    zero_mask = (arr == 0) & finite

    if zero_mask.any():
        lbl, n_comp = ndi.label(zero_mask, structure=_CONN4)
        border_ids = set()
        border_ids.update(int(x) for x in np.unique(lbl[0, :]))
        border_ids.update(int(x) for x in np.unique(lbl[-1, :]))
        border_ids.update(int(x) for x in np.unique(lbl[:, 0]))
        border_ids.update(int(x) for x in np.unique(lbl[:, -1]))
        border_ids.discard(0)
        if border_ids:
            is_border = np.zeros(n_comp + 1, dtype=bool)
            for bid in border_ids:
                is_border[bid] = True
            border_zero = is_border[lbl]
        else:
            border_zero = np.zeros(arr.shape, dtype=bool)
    else:
        border_zero = np.zeros(arr.shape, dtype=bool)

    valid_mask = finite & (~border_zero)
    total_px = int(arr.size)
    valid_px = int(valid_mask.sum())
    border_zero_px = int(border_zero.sum())
    interior_zero_px = int((zero_mask & valid_mask).sum())

    diag = {
        "total_px": total_px,
        "valid_px": valid_px,
        "valid_frac": round(valid_px / max(total_px, 1), 6),
        "border_zero_px": border_zero_px,
        "border_zero_frac": round(border_zero_px / max(total_px, 1), 6),
        "interior_zero_px": interior_zero_px,
        "interior_zero_frac": round(interior_zero_px / max(total_px, 1), 6),
    }
    return valid_mask, diag


def extract_valid_tiles(
    valid_mask: np.ndarray,
    *,
    tile_size: int = DEFAULT_TILE_SIZE,
    min_valid_frac: float = DEFAULT_MIN_TILE_VALID_FRAC,
) -> list[dict]:
    """Partition ``valid_mask`` into non-overlapping ``tile_size x tile_size`` active tiles."""
    h, w = valid_mask.shape
    ts = int(tile_size)
    nr, nc = h // ts, w // ts
    tiles: list[dict] = []
    for r in range(nr):
        y0, y1 = r * ts, (r + 1) * ts
        for c in range(nc):
            x0, x1 = c * ts, (c + 1) * ts
            tv = valid_mask[y0:y1, x0:x1]
            vpx = int(tv.sum())
            vf = float(vpx / float(ts * ts))
            if vf >= float(min_valid_frac):
                tiles.append(
                    {
                        "tile_id": f"r{r:02d}_c{c:02d}",
                        "row": r,
                        "col": c,
                        "y0": y0,
                        "y1": y1,
                        "x0": x0,
                        "x1": x1,
                        "valid_px": vpx,
                        "valid_frac": round(vf, 6),
                    }
                )
    return tiles


def _dihedral_transform(tile: np.ndarray, idx: int) -> np.ndarray:
    """Deterministic dihedral group (D4) transform indexed by ``idx % 8``."""
    k = idx % 8
    out = np.rot90(tile, k % 4)
    if k >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def build_margin_opposite_face_control(
    prob_dom: np.ndarray,
    valid_mask: np.ndarray,
    *,
    tile_size: int = DEFAULT_TILE_SIZE,
    n_donor_tiles: int = 4,
) -> tuple[np.ndarray, dict]:
    """Synthesize a realistic opposite-face blank floor control from real off-text tiles.

    Public S3 ``argus-v3`` and ``reader-v3`` TIFFs are single-face ink probability maps
    without published paired opposite-face (``desc``) volumes. To preserve exact real
    papyrus baseline floor, spatial weave texture, and ``min(prob_dom, prob_opp)``
    behavior, this function selects the lowest-activity high-validity ``tile_size``
    off-text tiles on the *same* map and tiles them with deterministic D4 dihedral
    transforms across ``valid_mask``.
    """
    h, w = prob_dom.shape
    ts = int(tile_size)
    candidates = extract_valid_tiles(valid_mask, tile_size=ts, min_valid_frac=0.90)
    if len(candidates) < max(1, n_donor_tiles):
        candidates = extract_valid_tiles(valid_mask, tile_size=ts, min_valid_frac=0.50)

    scored: list[tuple[float, float, float, int, int, dict]] = []
    for t in candidates:
        sub_p = prob_dom[t["y0"] : t["y1"], t["x0"] : t["x1"]]
        sub_v = valid_mask[t["y0"] : t["y1"], t["x0"] : t["x1"]]
        vals = sub_p[sub_v]
        f50 = float((vals >= 0.50).mean())
        p995 = float(np.percentile(vals, 99.5))
        mean_v = float(vals.mean())
        scored.append((f50, p995, mean_v, t["row"], t["col"], t))
    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

    donors_meta: list[dict] = []
    donor_patches: list[np.ndarray] = []
    if scored:
        chosen = scored[: max(1, min(int(n_donor_tiles), len(scored)))]
        for f50, p995, mean_v, r, c, t in chosen:
            sub_p = prob_dom[t["y0"] : t["y1"], t["x0"] : t["x1"]].astype(np.float32, copy=True)
            sub_v = valid_mask[t["y0"] : t["y1"], t["x0"] : t["x1"]]
            if not sub_v.all():
                fill = float(np.median(sub_p[sub_v]))
                sub_p = np.where(sub_v, sub_p, fill).astype(np.float32)
            donor_patches.append(sub_p)
            donors_meta.append(
                {
                    "tile_id": t["tile_id"],
                    "row": r,
                    "col": c,
                    "valid_frac": t["valid_frac"],
                    "frac_ge_050": round(f50, 6),
                    "p995": round(p995, 6),
                    "mean": round(mean_v, 6),
                }
            )
    else:
        # Fallback for small synthetic test arrays smaller than tile_size
        vals = prob_dom[valid_mask]
        med = float(np.median(vals)) if vals.size else 0.0
        patch = np.full((ts, ts), med, dtype=np.float32)
        donor_patches.append(patch)

    prob_opp = np.zeros((h, w), dtype=np.float32)
    n_don = len(donor_patches)
    nr = (h + ts - 1) // ts
    nc = (w + ts - 1) // ts
    for r in range(nr):
        y0 = r * ts
        y1 = min(y0 + ts, h)
        for c in range(nc):
            x0 = c * ts
            x1 = min(x0 + ts, w)
            d_idx = (r * 7 + c * 3) % n_don
            t_idx = (r * 5 + c * 11) % 8
            patch = _dihedral_transform(donor_patches[d_idx], t_idx)
            prob_opp[y0:y1, x0:x1] = patch[: y1 - y0, : x1 - x0]

    prob_opp = np.where(valid_mask, np.clip(prob_opp, 0.0, 1.0), 0.0).astype(np.float32)
    ov = prob_opp[valid_mask]
    provenance = {
        "method": "real_off_text_margin_tile_dihedral_control",
        "assumption_disclosure": (
            "Public S3 argus-v3 and reader-v3 TIFFs are single-face ink probability maps "
            "with no paired opposite-face (desc) TIFF published on S3. prob_opp is "
            "constructed deterministically by tiling the lowest-activity real off-text "
            "512x512 margin tiles from the same map with D4 dihedral transforms, preserving "
            "true papyrus baseline floor, spatial weave autocorrelation, and realistic "
            "min(prob_dom, prob_opp) null behavior."
        ),
        "n_donor_tiles": len(donors_meta),
        "donor_tiles": donors_meta,
        "opp_valid_stats": {
            "mean": round(float(ov.mean()), 6) if ov.size else 0.0,
            "median": round(float(np.median(ov)), 6) if ov.size else 0.0,
            "std": round(float(ov.std()), 6) if ov.size else 0.0,
            "p995": round(float(np.percentile(ov, 99.5)), 6) if ov.size else 0.0,
            "frac_ge_045": round(float((ov >= 0.45).mean()), 6) if ov.size else 0.0,
            "frac_ge_050": round(float((ov >= 0.50).mean()), 6) if ov.size else 0.0,
        },
    }
    return prob_opp, provenance


def select_text_rich_strip(
    prob_dom: np.ndarray,
    valid_mask: np.ndarray,
    *,
    strip_hw: tuple[int, int] = DEFAULT_STRIP_SHAPE_HW,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> dict:
    """Deterministically select the strongest ``strip_hw`` text-rich window on ``prob_dom``."""
    h, w = prob_dom.shape
    sh, sw = min(int(strip_hw[0]), h), min(int(strip_hw[1]), w)
    step_y = min(int(tile_size), sh)
    step_x = min(int(tile_size), sw)

    best: tuple[float, float, float, int, int, float] | None = None
    for min_vf in (0.98, 0.90, 0.50, 0.0):
        for y0 in range(0, h - sh + 1, step_y):
            y1 = y0 + sh
            for x0 in range(0, w - sw + 1, step_x):
                x1 = x0 + sw
                tv = valid_mask[y0:y1, x0:x1]
                vf = float(tv.mean())
                if vf < min_vf or not tv.any():
                    continue
                vals = prob_dom[y0:y1, x0:x1][tv]
                f60 = float((vals >= 0.60).mean())
                f50 = float((vals >= 0.50).mean())
                p995 = float(np.percentile(vals, 99.5))
                cand = (f60, f50, p995, -y0, -x0, vf)
                if best is None or cand > best:
                    best = cand
        if best is not None:
            break

    assert best is not None, "Could not find a valid strip window"
    f60, f50, p995, neg_y0, neg_x0, vf = best
    y0, x0 = -neg_y0, -neg_x0
    return {
        "y0": int(y0),
        "y1": int(y0 + sh),
        "x0": int(x0),
        "x1": int(x0 + sw),
        "shape_hw": [int(sh), int(sw)],
        "valid_frac": round(vf, 6),
        "frac_ge_060": round(f60, 6),
        "frac_ge_050": round(f50, 6),
        "p995": round(p995, 6),
    }


def select_blank_margin_strip(
    prob_dom: np.ndarray,
    valid_mask: np.ndarray,
    *,
    strip_hw: tuple[int, int] = (512, 1024),
    tile_size: int = DEFAULT_TILE_SIZE,
) -> dict:
    """Deterministically select a low-activity real blank margin strip window."""
    h, w = prob_dom.shape
    sh, sw = min(int(strip_hw[0]), h), min(int(strip_hw[1]), w)
    step_y = min(int(tile_size), sh)
    step_x = min(int(tile_size), sw)

    best: tuple[float, float, float, int, int, float] | None = None
    for min_vf in (0.95, 0.80, 0.50, 0.0):
        for y0 in range(0, h - sh + 1, step_y):
            y1 = y0 + sh
            for x0 in range(0, w - sw + 1, step_x):
                x1 = x0 + sw
                tv = valid_mask[y0:y1, x0:x1]
                vf = float(tv.mean())
                if vf < min_vf or not tv.any():
                    continue
                vals = prob_dom[y0:y1, x0:x1][tv]
                f50 = float((vals >= 0.50).mean())
                p995 = float(np.percentile(vals, 99.5))
                mean_v = float(vals.mean())
                cand = (f50, p995, mean_v, y0, x0, vf)
                if best is None or cand < best:
                    best = cand
        if best is not None:
            break

    assert best is not None
    f50, p995, mean_v, y0, x0, vf = best
    return {
        "y0": int(y0),
        "y1": int(y0 + sh),
        "x0": int(x0),
        "x1": int(x0 + sw),
        "shape_hw": [int(sh), int(sw)],
        "valid_frac": round(vf, 6),
        "frac_ge_050": round(f50, 6),
        "p995": round(p995, 6),
        "mean": round(mean_v, 6),
    }


def identify_protected_true_text_tiles(
    prob_dom: np.ndarray,
    valid_mask: np.ndarray,
    tiles: list[dict],
    *,
    protected_frac: float = DEFAULT_PROTECTED_TEXT_FRAC,
    min_true_tiles: int = 3,
    max_true_tiles: int = 12,
) -> tuple[set[str], np.ndarray, list[dict]]:
    """Select clean ground-truth true-text tiles deterministically from unmodified ``prob_dom``.

    Ranks all valid ``512x512`` tiles by unmodified active ink share ``frac(p >= 0.50)``
    (tie-broken by ``frac(p >= 0.65)`` and ``p995``) and protects the top ``N``
    strongest real Greek text tiles from cloud injection.
    """
    scored: list[dict] = []
    for t in tiles:
        sub_p = prob_dom[t["y0"] : t["y1"], t["x0"] : t["x1"]]
        sub_v = valid_mask[t["y0"] : t["y1"], t["x0"] : t["x1"]]
        vals = sub_p[sub_v]
        f50 = float((vals >= 0.50).mean()) if vals.size else 0.0
        f65 = float((vals >= 0.65).mean()) if vals.size else 0.0
        p995 = float(np.percentile(vals, 99.5)) if vals.size else 0.0
        scored.append(
            {
                **t,
                "clean_frac_ge_050": round(f50, 6),
                "clean_frac_ge_065": round(f65, 6),
                "clean_p995": round(p995, 6),
            }
        )
    scored.sort(
        key=lambda d: (d["clean_frac_ge_050"], d["clean_frac_ge_065"], d["clean_p995"], d["tile_id"]),
        reverse=True,
    )
    n_target = int(round(len(tiles) * float(protected_frac)))
    n_true = max(int(min_true_tiles), min(int(max_true_tiles), n_target, len(tiles)))
    chosen = scored[:n_true]
    true_ids = {d["tile_id"] for d in chosen}
    protected_mask = np.zeros(prob_dom.shape, dtype=bool)
    for d in chosen:
        protected_mask[d["y0"] : d["y1"], d["x0"] : d["x1"]] = True
    return true_ids, protected_mask, chosen


def inject_bilateral_gaussian_clouds(
    prob_dom: np.ndarray,
    prob_opp: np.ndarray,
    valid_mask: np.ndarray,
    protected_mask: np.ndarray,
    voxel_um: float,
    target_area_frac: float,
    *,
    seed: int = 31,
    diameter_mm_range: tuple[float, float] = DEFAULT_CLOUD_DIAMETER_MM_RANGE,
    peak_range: tuple[float, float] = DEFAULT_CLOUD_PEAK_RANGE,
    bilateral_threshold: float = DEFAULT_BILATERAL_CLOUD_THRESHOLD,
    opp_shift_px: tuple[int, int] = (0, 0),
    max_blobs: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Inject smooth isotropic ``2-4 mm`` Gaussian clouds onto BOTH ``prob_dom`` and ``prob_opp``.

    Clouds are placed at identical ``(y, x)`` coordinates (unless ``opp_shift_px != (0, 0)``
    for the supplementary parallax ablation) strictly outside ``protected_mask`` until
    the covered valid area fraction ``mean((cloud_dom >= bilateral_threshold)[valid_mask])``
    reaches ``target_area_frac``.
    """
    h, w = prob_dom.shape
    valid_px = int(valid_mask.sum())
    if target_area_frac <= 0.0 or valid_px == 0:
        return (
            prob_dom.astype(np.float32, copy=True),
            prob_opp.astype(np.float32, copy=True),
            np.zeros((h, w), dtype=bool),
            {
                "target_area_frac": float(target_area_frac),
                "achieved_area_frac": 0.0,
                "n_blobs": 0,
                "seed": int(seed),
                "opp_shift_px": [int(opp_shift_px[0]), int(opp_shift_px[1])],
                "diameter_mm_range": [float(diameter_mm_range[0]), float(diameter_mm_range[1])],
                "peak_range": [float(peak_range[0]), float(peak_range[1])],
            },
        )

    rng = np.random.default_rng(int(seed))
    px_per_mm = 1000.0 / float(voxel_um)
    d_lo, d_hi = float(diameter_mm_range[0]), float(diameter_mm_range[1])
    p_lo, p_hi = float(peak_range[0]), float(peak_range[1])
    dy_opp, dx_opp = int(opp_shift_px[0]), int(opp_shift_px[1])

    allowed = valid_mask & (~protected_mask)
    if not allowed.any():
        raise ValueError("No non-protected valid pixels available for cloud injection")

    # Prefer centers separated from protected_mask by at least half the min cloud diameter
    min_r_px = max(1.0, 0.5 * d_lo * px_per_mm)
    if protected_mask.any():
        dist_from_prot = ndi.distance_transform_edt(~protected_mask)
        safe_centers = allowed & (dist_from_prot >= 0.65 * min_r_px)
        if not safe_centers.any():
            safe_centers = allowed
    else:
        safe_centers = allowed

    # Subsample candidate pool on an 8px stride for fast O(1) sampling
    stride = 8 if min(h, w) >= 256 else 2
    sub_mask = safe_centers[::stride, ::stride]
    sub_ys, sub_xs = np.nonzero(sub_mask)
    if sub_ys.size == 0:
        sub_ys, sub_xs = np.nonzero(safe_centers)
        cand_ys, cand_xs = sub_ys, sub_xs
    else:
        cand_ys = sub_ys * stride
        cand_xs = sub_xs * stride

    cloud_dom = np.zeros((h, w), dtype=np.float32)
    cloud_opp = np.zeros((h, w), dtype=np.float32)
    target_px = int(round(float(target_area_frac) * valid_px))
    n_blobs = 0
    diameters_used: list[float] = []
    peaks_used: list[float] = []

    for _ in range(int(max_blobs)):
        covered_px = int(((cloud_dom >= bilateral_threshold) & valid_mask).sum())
        if covered_px >= target_px:
            break

        # Sample from not-yet-covered candidates so clouds spread efficiently across the surface
        uncovered_idx = np.flatnonzero(cloud_dom[cand_ys, cand_xs] < 0.30)
        if uncovered_idx.size > 0:
            pick = int(uncovered_idx[int(rng.integers(0, uncovered_idx.size))])
        else:
            pick = int(rng.integers(0, cand_ys.size))

        cy = float(cand_ys[pick]) + float(rng.uniform(-stride * 0.5, stride * 0.5))
        cx = float(cand_xs[pick]) + float(rng.uniform(-stride * 0.5, stride * 0.5))
        cy = float(np.clip(cy, 0.0, h - 1.0))
        cx = float(np.clip(cx, 0.0, w - 1.0))

        diam_mm = float(rng.uniform(d_lo, d_hi))
        sigma_px = diam_mm * px_per_mm * _FWHM_TO_SIGMA
        peak_dom = float(rng.uniform(p_lo, p_hi))
        # Slight realistic amplitude jitter (<= 0.015) while staying in [p_lo, p_hi]
        peak_opp = float(np.clip(peak_dom + rng.uniform(-0.015, 0.015), p_lo, p_hi))

        rad = int(math.ceil(3.2 * sigma_px))
        inv_2s2 = 1.0 / (2.0 * sigma_px * sigma_px)

        # Dominant face blob at (cy, cx)
        y0 = max(0, int(math.floor(cy)) - rad)
        y1 = min(h, int(math.ceil(cy)) + rad + 1)
        x0 = max(0, int(math.floor(cx)) - rad)
        x1 = min(w, int(math.ceil(cx)) + rad + 1)
        yy = (np.arange(y0, y1, dtype=np.float32) - cy)[:, None]
        xx = (np.arange(x0, x1, dtype=np.float32) - cx)[None, :]
        g_dom = (peak_dom * np.exp(-(yy * yy + xx * xx) * inv_2s2)).astype(np.float32)
        np.maximum(cloud_dom[y0:y1, x0:x1], g_dom, out=cloud_dom[y0:y1, x0:x1])

        # Opposite face blob at (cy + dy_opp, cx + dx_opp)
        if dy_opp == 0 and dx_opp == 0:
            g_opp = (g_dom * (peak_opp / max(peak_dom, 1e-6))).astype(np.float32)
            np.maximum(cloud_opp[y0:y1, x0:x1], g_opp, out=cloud_opp[y0:y1, x0:x1])
        else:
            cy_o = cy + dy_opp
            cx_o = cx + dx_opp
            oy0 = max(0, int(math.floor(cy_o)) - rad)
            oy1 = min(h, int(math.ceil(cy_o)) + rad + 1)
            ox0 = max(0, int(math.floor(cx_o)) - rad)
            ox1 = min(w, int(math.ceil(cx_o)) + rad + 1)
            if oy1 > oy0 and ox1 > ox0:
                oyy = (np.arange(oy0, oy1, dtype=np.float32) - cy_o)[:, None]
                oxx = (np.arange(ox0, ox1, dtype=np.float32) - cx_o)[None, :]
                g_opp = (peak_opp * np.exp(-(oyy * oyy + oxx * oxx) * inv_2s2)).astype(np.float32)
                np.maximum(cloud_opp[oy0:oy1, ox0:ox1], g_opp, out=cloud_opp[oy0:oy1, ox0:ox1])

        # Enforce zero contamination inside protected core stroke tiles
        if protected_mask.any():
            cloud_dom[y0:y1, x0:x1] = np.where(
                protected_mask[y0:y1, x0:x1], 0.0, cloud_dom[y0:y1, x0:x1]
            )
            if dy_opp == 0 and dx_opp == 0:
                cloud_opp[y0:y1, x0:x1] = np.where(
                    protected_mask[y0:y1, x0:x1], 0.0, cloud_opp[y0:y1, x0:x1]
                )
            elif oy1 > oy0 and ox1 > ox0:
                cloud_opp[oy0:oy1, ox0:ox1] = np.where(
                    protected_mask[oy0:oy1, ox0:ox1], 0.0, cloud_opp[oy0:oy1, ox0:ox1]
                )

        n_blobs += 1
        diameters_used.append(diam_mm)
        peaks_used.append(peak_dom)

    cloud_dom = np.where(allowed, cloud_dom, 0.0).astype(np.float32)
    cloud_opp = np.where(allowed, cloud_opp, 0.0).astype(np.float32)

    contam_dom = np.where(valid_mask, np.maximum(prob_dom, cloud_dom), 0.0).astype(np.float32)
    contam_opp = np.where(valid_mask, np.maximum(prob_opp, cloud_opp), 0.0).astype(np.float32)
    cloud_mask = (cloud_dom >= bilateral_threshold) & valid_mask
    achieved_frac = float(cloud_mask.sum() / max(valid_px, 1))

    diag = {
        "target_area_frac": round(float(target_area_frac), 6),
        "achieved_area_frac": round(achieved_frac, 6),
        "n_blobs": int(n_blobs),
        "seed": int(seed),
        "opp_shift_px": [dy_opp, dx_opp],
        "diameter_mm_range": [d_lo, d_hi],
        "peak_range": [p_lo, p_hi],
        "mean_blob_diameter_mm": round(float(np.mean(diameters_used)), 4) if diameters_used else 0.0,
        "mean_blob_peak": round(float(np.mean(peaks_used)), 4) if peaks_used else 0.0,
    }
    return contam_dom, contam_opp, cloud_mask, diag


def score_pipeline_a_global_z2(
    prob_dom: np.ndarray,
    valid_mask: np.ndarray,
    tiles: list[dict],
    *,
    z_thresh: float = DEFAULT_Z_THRESHOLD,
) -> dict:
    """Baseline (A): Standard global per-surface ``z > 2`` hot-share ranked per ``512x512`` tile."""
    vals = prob_dom[valid_mask].astype(np.float64)
    mean_v = float(vals.mean()) if vals.size else 0.0
    std_v = float(vals.std()) if vals.size else 0.0
    denom = max(std_v, 1e-6)
    z2_prob_thresh = mean_v + float(z_thresh) * std_v

    rows: list[dict] = []
    for t in tiles:
        sub_p = prob_dom[t["y0"] : t["y1"], t["x0"] : t["x1"]].astype(np.float64)
        sub_v = valid_mask[t["y0"] : t["y1"], t["x0"] : t["x1"]]
        tv = sub_p[sub_v]
        z_t = (tv - mean_v) / denom
        hot_share = float((z_t > float(z_thresh)).mean()) if z_t.size else 0.0
        p99_z = float(np.percentile(z_t, 99.0)) if z_t.size else 0.0
        mean_z = float(z_t.mean()) if z_t.size else 0.0
        rows.append(
            {
                "tile_id": t["tile_id"],
                "score": round(hot_share, 6),
                "hot_share": round(hot_share, 6),
                "p99_z": round(p99_z, 6),
                "mean_z": round(mean_z, 6),
            }
        )

    rows.sort(key=lambda r: (r["hot_share"], r["p99_z"], r["mean_z"], r["tile_id"]), reverse=True)
    return {
        "pipeline": "A_global_z2_hot_share",
        "global_mean": round(mean_v, 6),
        "global_std": round(std_v, 6),
        "z2_prob_threshold": round(z2_prob_thresh, 6),
        "zero_hot_share_collapse": bool(rows[0]["hot_share"] <= 0.0) if rows else True,
        "ranked_tiles": rows,
    }


def score_pipeline_b_cloud_masked_mad(
    prob_dom: np.ndarray,
    prob_opp: np.ndarray,
    valid_mask: np.ndarray,
    tiles: list[dict],
    *,
    bilateral_thresh: float = DEFAULT_BILATERAL_CLOUD_THRESHOLD,
    z_thresh: float = DEFAULT_Z_THRESHOLD,
    sigma_floor: float = DEFAULT_MAD_SIGMA_FLOOR,
) -> dict:
    """Rescue (B): Bilateral-cloud-masked robust MAD z-score on ``(prob_dom - prob_opp).clip(0)``.

    Masks out bilateral clouds where ``min(prob_dom, prob_opp) >= bilateral_thresh``,
    recomputes robust median/MAD on the unmasked valid signed differential so zero-clipping
    cannot collapse MAD to 0, then scores ``(prob_dom - prob_opp).clip(0)`` normalized per
    tile by ``tile_valid`` pixels (masked cloud pixels contribute 0).
    """
    bilat_cloud = valid_mask & (np.minimum(prob_dom, prob_opp) >= float(bilateral_thresh))
    unmasked_valid = valid_mask & (~bilat_cloud)

    signed_diff = prob_dom.astype(np.float64) - prob_opp.astype(np.float64)
    if unmasked_valid.any():
        s_vals = signed_diff[unmasked_valid]
        med_s = float(np.median(s_vals))
        mad_s = 1.4826 * float(np.median(np.abs(s_vals - med_s)))
        dom_u = prob_dom[unmasked_valid].astype(np.float64)
        med_dom = float(np.median(dom_u))
        mad_dom = 1.4826 * float(np.median(np.abs(dom_u - med_dom)))
    else:
        med_s, mad_s, med_dom, mad_dom = 0.0, 0.0, 0.0, 0.0

    eff_sigma = max(mad_s, float(sigma_floor))
    center_pos = max(med_s, 0.0)
    diff_pos = np.where(unmasked_valid, np.clip(signed_diff, 0.0, 1.0), 0.0)
    z_pos = np.where(unmasked_valid, (diff_pos - center_pos) / eff_sigma, 0.0)

    rows: list[dict] = []
    for t in tiles:
        y0, y1, x0, x1 = t["y0"], t["y1"], t["x0"], t["x1"]
        sub_v = valid_mask[y0:y1, x0:x1]
        vpx = max(int(sub_v.sum()), 1)
        sub_u = unmasked_valid[y0:y1, x0:x1]
        sub_z = z_pos[y0:y1, x0:x1]
        sub_d = diff_pos[y0:y1, x0:x1]
        hot_mask = sub_u & (sub_z > float(z_thresh))
        hot_share = float(hot_mask.sum() / vpx)
        mean_excess = float(np.where(hot_mask, sub_d, 0.0).sum() / vpx)
        p995_diff = float(np.percentile(sub_d[sub_v], 99.5)) if sub_v.any() else 0.0
        masked_frac = float((bilat_cloud[y0:y1, x0:x1] & sub_v).sum() / vpx)
        rows.append(
            {
                "tile_id": t["tile_id"],
                "score": round(mean_excess, 6),
                "hot_share": round(hot_share, 6),
                "mean_excess_diff": round(mean_excess, 6),
                "p995_diff": round(p995_diff, 6),
                "bilat_masked_frac": round(masked_frac, 6),
            }
        )

    rows.sort(
        key=lambda r: (r["mean_excess_diff"], r["hot_share"], r["p995_diff"], r["tile_id"]),
        reverse=True,
    )
    valid_px = max(int(valid_mask.sum()), 1)
    return {
        "pipeline": "B_bilateral_masked_robust_mad",
        "bilateral_threshold": float(bilateral_thresh),
        "masked_cloud_frac": round(float(bilat_cloud.sum() / valid_px), 6),
        "unmasked_diff_median": round(med_s, 6),
        "unmasked_diff_mad_sigma": round(mad_s, 6),
        "effective_mad_sigma": round(eff_sigma, 6),
        "unmasked_dom_median": round(med_dom, 6),
        "unmasked_dom_mad_sigma": round(mad_dom, 6),
        "ranked_tiles": rows,
    }


def score_pipeline_c_cloud_masked_sb_tfce(
    prob_dom: np.ndarray,
    prob_opp: np.ndarray,
    valid_mask: np.ndarray,
    tiles: list[dict],
    voxel_um: float,
    *,
    bilateral_thresh: float = DEFAULT_BILATERAL_CLOUD_THRESHOLD,
    n_steps: int = tfce_cal.DEFAULT_TFCE_STEPS,
    max_stroke_width_mm: float = tfce_cal.DEFAULT_MAX_STROKE_WIDTH_MM,
    sigma_floor: float = DEFAULT_MAD_SIGMA_FLOOR,
    eval_top_q34_k: int = 3,
) -> dict:
    """Rescue (C): Cloud-Masked Opposite-Face Differential + ``stroke_bounded_tfce_2d``.

    Zeros out bilateral cloud pixels (``min(prob_dom, prob_opp) >= bilateral_thresh``),
    computes positive opposite-face differential ``clip(prob_dom - prob_opp, 0, 1)``,
    applies Exp T6's :func:`stroke_bounded_tfce_2d` per ``512x512`` tile (suppressing
    any residual wide ``> 0.90 mm`` blobs), ranks tiles by SB-TFCE ``p995`` (tie-broken
    by ``max``), and computes full Exp T6 ``q34_score`` on the top-ranked tiles.
    """
    bilat_cloud = valid_mask & (np.minimum(prob_dom, prob_opp) >= float(bilateral_thresh))
    unmasked_valid = valid_mask & (~bilat_cloud)

    signed_diff = prob_dom.astype(np.float64) - prob_opp.astype(np.float64)
    if unmasked_valid.any():
        s_vals = signed_diff[unmasked_valid]
        med_s = float(np.median(s_vals))
        mad_s = 1.4826 * float(np.median(np.abs(s_vals - med_s)))
    else:
        med_s, mad_s = 0.0, 0.0
    eff_sigma = max(mad_s, float(sigma_floor))

    diff_clean = np.where(unmasked_valid, np.clip(signed_diff, 0.0, 1.0), 0.0).astype(np.float32)

    rows: list[dict] = []
    tile_by_id = {t["tile_id"]: t for t in tiles}
    for t in tiles:
        y0, y1, x0, x1 = t["y0"], t["y1"], t["x0"], t["x1"]
        sub_v = valid_mask[y0:y1, x0:x1]
        sub_d = diff_clean[y0:y1, x0:x1]
        vpx = max(int(sub_v.sum()), 1)
        sb = tfce_cal.stroke_bounded_tfce_2d(
            sub_d,
            sub_v,
            voxel_um=voxel_um,
            max_stroke_width_mm=max_stroke_width_mm,
            n_steps=n_steps,
        )
        sb_vals = sb[sub_v]
        p995 = float(np.percentile(sb_vals, 99.5)) if sb_vals.size else 0.0
        p99 = float(np.percentile(sb_vals, 99.0)) if sb_vals.size else 0.0
        sb_max = float(sb_vals.max()) if sb_vals.size else 0.0
        masked_frac = float((bilat_cloud[y0:y1, x0:x1] & sub_v).sum() / vpx)
        rows.append(
            {
                "tile_id": t["tile_id"],
                "score": round(p995, 6),
                "sb_tfce_p995": round(p995, 6),
                "sb_tfce_p99": round(p99, 6),
                "sb_tfce_max": round(sb_max, 6),
                "bilat_masked_frac": round(masked_frac, 6),
            }
        )

    rows.sort(
        key=lambda r: (r["sb_tfce_p995"], r["sb_tfce_max"], r["sb_tfce_p99"], r["tile_id"]),
        reverse=True,
    )

    # Evaluate full Exp T6 Q34 score on the Top-K ranked tiles (cloud-masked)
    top_q34_summaries: list[dict] = []
    for row in rows[: max(0, int(eval_top_q34_k))]:
        t = tile_by_id[row["tile_id"]]
        y0, y1, x0, x1 = t["y0"], t["y1"], t["x0"], t["x1"]
        sub_u = unmasked_valid[y0:y1, x0:x1]
        dom_t = np.where(sub_u, prob_dom[y0:y1, x0:x1], 0.0)
        opp_t = np.where(sub_u, prob_opp[y0:y1, x0:x1], 0.0)
        fwer = tfce_cal.tfce_opposite_face_fwer(
            dom_t,
            opp_t,
            sub_u,
            voxel_um=voxel_um,
            n_steps=n_steps,
            max_stroke_width_mm=max_stroke_width_mm,
        )
        q34_s = float(fwer["q34_calibrated"]["calibrated_score"])
        row["q34_score"] = round(q34_s, 6)
        row["max_tfce_ratio"] = fwer["max_tfce_ratio"]
        row["passes_q34_bar"] = bool(fwer["q34_calibrated"]["passes_q34_bar"])
        row["q34_regime_label"] = str(fwer["q34_calibrated"]["regime_label"])
        top_q34_summaries.append(
            {
                "tile_id": row["tile_id"],
                "sb_tfce_p995": row["sb_tfce_p995"],
                "sb_tfce_max": row["sb_tfce_max"],
                "q34_score": row["q34_score"],
                "max_tfce_ratio": row["max_tfce_ratio"],
                "passes_q34_bar": row["passes_q34_bar"],
                "q34_regime_label": row["q34_regime_label"],
            }
        )

    valid_px = max(int(valid_mask.sum()), 1)
    return {
        "pipeline": "C_cloud_masked_diff_sb_tfce",
        "bilateral_threshold": float(bilateral_thresh),
        "max_stroke_width_mm": float(max_stroke_width_mm),
        "n_steps": int(n_steps),
        "masked_cloud_frac": round(float(bilat_cloud.sum() / valid_px), 6),
        "unmasked_diff_mad_sigma": round(mad_s, 6),
        "effective_mad_sigma": round(eff_sigma, 6),
        "top_q34_verified": top_q34_summaries,
        "ranked_tiles": rows,
    }


def evaluate_tile_ranking(
    ranked_tiles: list[dict],
    true_text_ids: set[str],
    cloud_tile_ids: set[str],
    clean_ref_top3_ids: set[str],
) -> dict:
    """Compute Top-1 hit rate, Top-3 recall, and cloud-tile false-alarm rates."""
    if not ranked_tiles:
        return {
            "top1_hit": 0.0,
            "top3_true_text_recall": 0.0,
            "top3_clean_ref_recall": 0.0,
            "cloud_top3_false_alarm_rate": 0.0,
            "cloud_outranks_text_fpr": 0.0,
            "top3_tile_ids": [],
        }
    top3 = ranked_tiles[:3]
    top3_ids = [r["tile_id"] for r in top3]
    k_denom = float(min(3, len(top3)))

    top1_hit = 1.0 if ranked_tiles[0]["tile_id"] in true_text_ids else 0.0
    top3_true_hit = sum(1 for tid in top3_ids if tid in true_text_ids) / k_denom
    ref_denom = float(max(1, min(3, len(clean_ref_top3_ids))))
    top3_clean_recall = sum(1 for tid in top3_ids if tid in clean_ref_top3_ids) / ref_denom
    cloud_top3_far = sum(1 for tid in top3_ids if tid in cloud_tile_ids) / k_denom

    # Fraction of cloud-contaminated tiles that outrank the 3rd-best true-text tile
    if cloud_tile_ids:
        true_scores = sorted(
            (r["score"] for r in ranked_tiles if r["tile_id"] in true_text_ids),
            reverse=True,
        )
        bar = true_scores[min(2, len(true_scores) - 1)] if true_scores else 0.0
        n_outrank = sum(
            1 for r in ranked_tiles if r["tile_id"] in cloud_tile_ids and r["score"] > bar
        )
        cloud_outranks_fpr = float(n_outrank / len(cloud_tile_ids))
    else:
        cloud_outranks_fpr = 0.0

    return {
        "top1_hit": round(top1_hit, 6),
        "top3_true_text_recall": round(top3_true_hit, 6),
        "top3_clean_ref_recall": round(top3_clean_recall, 6),
        "cloud_top3_false_alarm_rate": round(cloud_top3_far, 6),
        "cloud_outranks_text_fpr": round(cloud_outranks_fpr, 6),
        "top3_tile_ids": top3_ids,
    }


def _compact_strip_eval(label: str, eval_res: dict) -> dict:
    """Extract key Exp T6 / Q34 metrics from :func:`tfce_opposite_face_fwer`."""
    topo = eval_res["transverse_stroke_topology"]
    sb = eval_res["stroke_bounded_tfce"]
    raw = eval_res["raw_tfce"]
    q34 = eval_res["q34_calibrated"]
    return {
        "label": label,
        "valid_cm2": topo["valid_cm2"],
        "raw_tfce_dom_max": raw["dom_max"],
        "sb_tfce_dom_max": sb["dom_max"],
        "sb_tfce_dom_p995": sb["dom_p995"],
        "sb_tfce_opp_max": sb["opp_max"],
        "max_tfce_ratio": eval_res["max_tfce_ratio"],
        "p995_tfce_ratio": eval_res["p995_tfce_ratio"],
        "diff_tfce_p99": eval_res["diff_tfce_p99"],
        "sb_retention_max": sb["retention_max"],
        "width_median_hi_mm": topo["stroke_width"]["hi"]["width_median_mm"],
        "narrow_stroke_frac_hi": topo["stroke_width"]["hi"]["narrow_stroke_frac"],
        "wide_blob_frac_hi": topo["stroke_width"]["hi"]["wide_blob_frac"],
        "max_b1": topo["betti"]["dominant"]["max_b1"],
        "stroke_consistency_score": topo["composite"]["stroke_consistency_score"],
        "q34_score": q34["calibrated_score"],
        "passes_q34_bar": q34["passes_q34_bar"],
        "q34_regime_label": q34["regime_label"],
    }


def evaluate_single_map_q31(
    map_id: str,
    prob_dom: np.ndarray,
    valid_mask: np.ndarray,
    voxel_um: float,
    *,
    tile_size: int = DEFAULT_TILE_SIZE,
    contamination_fractions: tuple[float, ...] = DEFAULT_CONTAMINATION_FRACTIONS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_steps: int = tfce_cal.DEFAULT_TFCE_STEPS,
    run_parallax_ablation: bool = True,
) -> dict:
    """Run full unmodified Exp T6 evaluation + Q31 cloud-contamination stress test on one map."""
    t_map = time.perf_counter()
    prob_opp, opp_prov = build_margin_opposite_face_control(
        prob_dom, valid_mask, tile_size=tile_size
    )
    tiles = extract_valid_tiles(valid_mask, tile_size=tile_size)
    true_ids, protected_mask, true_tile_rows = identify_protected_true_text_tiles(
        prob_dom, valid_mask, tiles
    )

    # 1) Unmodified real text-rich strip + blank margin control (Exp T6 / Q34)
    text_strip = select_text_rich_strip(
        prob_dom, valid_mask, strip_hw=DEFAULT_STRIP_SHAPE_HW, tile_size=tile_size
    )
    blank_strip = select_blank_margin_strip(
        prob_dom, valid_mask, strip_hw=(tile_size, tile_size * 2), tile_size=tile_size
    )

    ty0, ty1, tx0, tx1 = text_strip["y0"], text_strip["y1"], text_strip["x0"], text_strip["x1"]
    t_strip_eval = tfce_cal.tfce_opposite_face_fwer(
        prob_dom[ty0:ty1, tx0:tx1],
        prob_opp[ty0:ty1, tx0:tx1],
        valid_mask[ty0:ty1, tx0:tx1],
        voxel_um=voxel_um,
        n_steps=n_steps,
    )
    by0, by1, bx0, bx1 = blank_strip["y0"], blank_strip["y1"], blank_strip["x0"], blank_strip["x1"]
    b_strip_eval = tfce_cal.tfce_opposite_face_fwer(
        prob_dom[by0:by1, bx0:bx1],
        prob_opp[by0:by1, bx0:bx1],
        valid_mask[by0:by1, bx0:bx1],
        voxel_um=voxel_um,
        n_steps=n_steps,
    )

    # 2) Clean (0%) baseline scores and reference Top-3 for each pipeline
    clean_a = score_pipeline_a_global_z2(prob_dom, valid_mask, tiles)
    clean_b = score_pipeline_b_cloud_masked_mad(prob_dom, prob_opp, valid_mask, tiles)
    clean_c = score_pipeline_c_cloud_masked_sb_tfce(
        prob_dom, prob_opp, valid_mask, tiles, voxel_um, n_steps=n_steps
    )
    base_std_a = max(clean_a["global_std"], 1e-6)
    base_mad_b = max(clean_b["effective_mad_sigma"], 1e-6)
    base_mad_c = max(clean_c["effective_mad_sigma"], 1e-6)
    ref_top3_a = {r["tile_id"] for r in clean_a["ranked_tiles"][:3]}
    ref_top3_b = {r["tile_id"] for r in clean_b["ranked_tiles"][:3]}
    ref_top3_c = {r["tile_id"] for r in clean_c["ranked_tiles"][:3]}

    # 3) Q31 Contamination Sweep (0%, 10%, 20%, 30%)
    sweep_trials: list[dict] = []
    for frac in contamination_fractions:
        frac_f = float(frac)
        trial_seeds = (int(seeds[0]),) if frac_f <= 0.0 else tuple(int(s) for s in seeds)
        for s in trial_seeds:
            if frac_f <= 0.0:
                res_a, res_b, res_c = clean_a, clean_b, clean_c
                cloud_ids: set[str] = set()
                cloud_diag = {
                    "target_area_frac": 0.0,
                    "achieved_area_frac": 0.0,
                    "n_blobs": 0,
                    "seed": s,
                    "opp_shift_px": [0, 0],
                }
            else:
                cdom, copp, cmask, cloud_diag = inject_bilateral_gaussian_clouds(
                    prob_dom,
                    prob_opp,
                    valid_mask,
                    protected_mask,
                    voxel_um,
                    frac_f,
                    seed=s,
                )
                cloud_ids = set()
                for t in tiles:
                    sub_v = valid_mask[t["y0"] : t["y1"], t["x0"] : t["x1"]]
                    sub_c = cmask[t["y0"] : t["y1"], t["x0"] : t["x1"]]
                    if sub_v.any() and float((sub_c & sub_v).sum() / sub_v.sum()) >= DEFAULT_CLOUD_TILE_MIN_FRAC:
                        cloud_ids.add(t["tile_id"])
                res_a = score_pipeline_a_global_z2(cdom, valid_mask, tiles)
                res_b = score_pipeline_b_cloud_masked_mad(cdom, copp, valid_mask, tiles)
                res_c = score_pipeline_c_cloud_masked_sb_tfce(
                    cdom, copp, valid_mask, tiles, voxel_um, n_steps=n_steps
                )

            eval_a = evaluate_tile_ranking(res_a["ranked_tiles"], true_ids, cloud_ids, ref_top3_a)
            eval_b = evaluate_tile_ranking(res_b["ranked_tiles"], true_ids, cloud_ids, ref_top3_b)
            eval_c = evaluate_tile_ranking(res_c["ranked_tiles"], true_ids, cloud_ids, ref_top3_c)

            sweep_trials.append(
                {
                    "target_cloud_frac": frac_f,
                    "seed": s,
                    "n_cloud_tiles": len(cloud_ids),
                    "cloud_injection": cloud_diag,
                    "pipelines": {
                        "A_global_z2": {
                            **eval_a,
                            "global_sigma": res_a["global_std"],
                            "global_sigma_inflation": round(res_a["global_std"] / base_std_a, 6),
                            "z2_prob_threshold": res_a["z2_prob_threshold"],
                            "zero_hot_share_collapse": res_a["zero_hot_share_collapse"],
                            "top1_score": res_a["ranked_tiles"][0]["score"] if res_a["ranked_tiles"] else 0.0,
                        },
                        "B_masked_mad": {
                            **eval_b,
                            "effective_sigma": res_b["effective_mad_sigma"],
                            "effective_sigma_inflation": round(res_b["effective_mad_sigma"] / base_mad_b, 6),
                            "masked_cloud_frac": res_b["masked_cloud_frac"],
                            "top1_score": res_b["ranked_tiles"][0]["score"] if res_b["ranked_tiles"] else 0.0,
                        },
                        "C_masked_sb_tfce": {
                            **eval_c,
                            "effective_sigma": res_c["effective_mad_sigma"],
                            "effective_sigma_inflation": round(res_c["effective_mad_sigma"] / base_mad_c, 6),
                            "masked_cloud_frac": res_c["masked_cloud_frac"],
                            "top1_sb_tfce_p995": res_c["ranked_tiles"][0]["sb_tfce_p995"] if res_c["ranked_tiles"] else 0.0,
                            "top1_sb_tfce_max": res_c["ranked_tiles"][0]["sb_tfce_max"] if res_c["ranked_tiles"] else 0.0,
                            "top_q34_verified": res_c["top_q34_verified"],
                        },
                    },
                }
            )

    # 4) Supplementary parallax ablation (20% cloud with ~0.15 mm inter-face shift)
    parallax_ablation = None
    if run_parallax_ablation:
        shift_px = max(2, int(round(0.15 * 1000.0 / float(voxel_um))))
        pdom, popp, pmask, pdiag = inject_bilateral_gaussian_clouds(
            prob_dom,
            prob_opp,
            valid_mask,
            protected_mask,
            voxel_um,
            0.20,
            seed=int(seeds[0]) + 900,
            opp_shift_px=(shift_px, -shift_px),
        )
        p_cloud_ids = set()
        for t in tiles:
            sub_v = valid_mask[t["y0"] : t["y1"], t["x0"] : t["x1"]]
            sub_c = pmask[t["y0"] : t["y1"], t["x0"] : t["x1"]]
            if sub_v.any() and float((sub_c & sub_v).sum() / sub_v.sum()) >= DEFAULT_CLOUD_TILE_MIN_FRAC:
                p_cloud_ids.add(t["tile_id"])
        pa = score_pipeline_a_global_z2(pdom, valid_mask, tiles)
        pb = score_pipeline_b_cloud_masked_mad(pdom, popp, valid_mask, tiles)
        pc = score_pipeline_c_cloud_masked_sb_tfce(
            pdom, popp, valid_mask, tiles, voxel_um, n_steps=n_steps
        )
        parallax_ablation = {
            "shift_mm": round(shift_px * float(voxel_um) / 1000.0, 4),
            "shift_px": [shift_px, -shift_px],
            "n_cloud_tiles": len(p_cloud_ids),
            "cloud_injection": pdiag,
            "A_global_z2": {
                **evaluate_tile_ranking(pa["ranked_tiles"], true_ids, p_cloud_ids, ref_top3_a),
                "global_sigma_inflation": round(pa["global_std"] / base_std_a, 6),
            },
            "B_masked_mad": {
                **evaluate_tile_ranking(pb["ranked_tiles"], true_ids, p_cloud_ids, ref_top3_b),
                "effective_sigma_inflation": round(pb["effective_mad_sigma"] / base_mad_b, 6),
            },
            "C_masked_sb_tfce": {
                **evaluate_tile_ranking(pc["ranked_tiles"], true_ids, p_cloud_ids, ref_top3_c),
                "effective_sigma_inflation": round(pc["effective_mad_sigma"] / base_mad_c, 6),
                "top_q34_verified": pc["top_q34_verified"],
            },
        }

    return {
        "map_id": map_id,
        "voxel_um": float(voxel_um),
        "n_valid_tiles": len(tiles),
        "n_protected_true_text_tiles": len(true_ids),
        "protected_true_text_tile_ids": sorted(true_ids),
        "protected_true_text_top3": [r["tile_id"] for r in true_tile_rows[:3]],
        "prob_opp_provenance": opp_prov,
        "unmodified_strip_crops": {
            "text_rich_strip": {
                "window": text_strip,
                "exp_t6_q34": _compact_strip_eval(f"{map_id}_text_strip", t_strip_eval),
            },
            "blank_margin_control": {
                "window": blank_strip,
                "exp_t6_q34": _compact_strip_eval(f"{map_id}_blank_margin", b_strip_eval),
            },
        },
        "q31_stress_trials": sweep_trials,
        "supplementary_parallax_ablation_20pct": parallax_ablation,
        "elapsed_seconds": round(time.perf_counter() - t_map, 3),
    }


def aggregate_rescue_table(
    map_results: list[dict],
    contamination_fractions: tuple[float, ...] = DEFAULT_CONTAMINATION_FRACTIONS,
) -> list[dict]:
    """Aggregate per-pipeline metrics across all maps and seeds for each cloud fraction."""
    pipelines = ("A_global_z2", "B_masked_mad", "C_masked_sb_tfce")
    rows: list[dict] = []
    for frac in contamination_fractions:
        frac_f = float(frac)
        for p_key in pipelines:
            top1s, top3_true, top3_ref, far_top3, fpr_out, sig_inf = [], [], [], [], [], []
            collapse_count = 0
            for mres in map_results:
                for tr in mres["q31_stress_trials"]:
                    if abs(tr["target_cloud_frac"] - frac_f) > 1e-9:
                        continue
                    pm = tr["pipelines"][p_key]
                    top1s.append(float(pm["top1_hit"]))
                    top3_true.append(float(pm["top3_true_text_recall"]))
                    top3_ref.append(float(pm["top3_clean_ref_recall"]))
                    far_top3.append(float(pm["cloud_top3_false_alarm_rate"]))
                    fpr_out.append(float(pm["cloud_outranks_text_fpr"]))
                    if p_key == "A_global_z2":
                        sig_inf.append(float(pm["global_sigma_inflation"]))
                        if pm.get("zero_hot_share_collapse"):
                            collapse_count += 1
                    else:
                        sig_inf.append(float(pm["effective_sigma_inflation"]))

            n = max(len(top1s), 1)
            row = {
                "cloud_area_frac": frac_f,
                "pipeline": p_key,
                "n_trials": len(top1s),
                "top1_hit_rate": round(sum(top1s) / n, 6),
                "top3_true_text_recall": round(sum(top3_true) / n, 6),
                "top3_clean_ref_recall": round(sum(top3_ref) / n, 6),
                "cloud_top3_false_alarm_rate": round(sum(far_top3) / n, 6),
                "cloud_outranks_text_fpr": round(sum(fpr_out) / n, 6),
                "mean_sigma_inflation_factor": round(sum(sig_inf) / n, 6),
            }
            if p_key == "A_global_z2":
                row["zero_hot_share_collapse_trials"] = collapse_count
            rows.append(row)
    return rows


def run_q31_benchmark(
    output_json: Path | str | None = None,
    *,
    byte_cap: int = GLOBAL_HTTP_BYTE_CAP,
    specs: tuple[PublicInkMapSpec, ...] = PUBLIC_INK_MAPS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_steps: int = tfce_cal.DEFAULT_TFCE_STEPS,
) -> dict:
    """Download the 3 public S3 TIFFs under ``byte_cap`` and run the full Exp T8 benchmark."""
    t0 = time.perf_counter()
    preflight = preflight_head_sizes(specs, cap_bytes=byte_cap)
    budget = tfce_cal.HttpByteBudget(cap_bytes=int(byte_cap))

    source_records: list[dict] = []
    map_results: list[dict] = []

    for spec in specs:
        raw_bytes = budget.fetch(spec.url, timeout=120.0)
        sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
        prob_dom, raw_arr, dec_meta = decode_ink_tiff_bytes(raw_bytes)
        valid_mask, valid_diag = infer_valid_mask(raw_arr)
        pv = prob_dom[valid_mask]

        source_records.append(
            {
                "map_id": spec.map_id,
                "scroll": spec.scroll,
                "model": spec.model,
                "voxel_um": spec.voxel_um,
                "url": spec.url,
                "bytes": len(raw_bytes),
                "mib": round(len(raw_bytes) / (1024.0 * 1024.0), 4),
                "sha256": sha256_hex,
                "tiff_meta": dec_meta,
                "valid_mask": valid_diag,
                "valid_prob_stats": {
                    "mean": round(float(pv.mean()), 6) if pv.size else 0.0,
                    "std": round(float(pv.std()), 6) if pv.size else 0.0,
                    "median": round(float(np.median(pv)), 6) if pv.size else 0.0,
                    "p95": round(float(np.percentile(pv, 95.0)), 6) if pv.size else 0.0,
                    "p99": round(float(np.percentile(pv, 99.0)), 6) if pv.size else 0.0,
                    "frac_ge_050": round(float((pv >= 0.50).mean()), 6) if pv.size else 0.0,
                    "frac_ge_070": round(float((pv >= 0.70).mean()), 6) if pv.size else 0.0,
                },
            }
        )

        mres = evaluate_single_map_q31(
            spec.map_id,
            prob_dom,
            valid_mask,
            spec.voxel_um,
            seeds=seeds,
            n_steps=n_steps,
        )
        map_results.append(mres)

    summary_table = aggregate_rescue_table(map_results)
    unmodified_table = [
        {
            "map_id": m["map_id"],
            "voxel_um": m["voxel_um"],
            "text_strip": m["unmodified_strip_crops"]["text_rich_strip"]["exp_t6_q34"],
            "blank_margin": m["unmodified_strip_crops"]["blank_margin_control"]["exp_t6_q34"],
        }
        for m in map_results
    ]

    # Summarize parallax ablation across maps
    parallax_summary = {}
    for p_key in ("A_global_z2", "B_masked_mad", "C_masked_sb_tfce"):
        items = [
            m["supplementary_parallax_ablation_20pct"][p_key]
            for m in map_results
            if m.get("supplementary_parallax_ablation_20pct")
        ]
        if items:
            n = len(items)
            parallax_summary[p_key] = {
                "top1_hit_rate": round(sum(x["top1_hit"] for x in items) / n, 6),
                "top3_true_text_recall": round(sum(x["top3_true_text_recall"] for x in items) / n, 6),
                "cloud_top3_false_alarm_rate": round(sum(x["cloud_top3_false_alarm_rate"] for x in items) / n, 6),
                "cloud_outranks_text_fpr": round(sum(x["cloud_outranks_text_fpr"] for x in items) / n, 6),
            }

    payload = {
        "experiment": (
            "Exp T8 (Q31 + Q34): Cloud-Masked Opposite-Face Stroke-Bounded TFCE Rescue "
            "on Real argus-v3 & reader-v3 Ink Maps"
        ),
        "device": "cpu_only_numpy_scipy_pil",
        "http_budget": {
            "cap_bytes": int(budget.cap_bytes),
            "cap_mib": round(budget.cap_bytes / (1024.0 * 1024.0), 4),
            "used_bytes": int(budget.used_bytes),
            "used_mib": round(budget.used_bytes / (1024.0 * 1024.0), 4),
            "within_cap": bool(budget.used_bytes <= budget.cap_bytes),
            "preflight_head": preflight,
        },
        "parameters": {
            "tile_size_px": DEFAULT_TILE_SIZE,
            "strip_shape_hw": list(DEFAULT_STRIP_SHAPE_HW),
            "min_tile_valid_frac": DEFAULT_MIN_TILE_VALID_FRAC,
            "protected_true_text_frac": DEFAULT_PROTECTED_TEXT_FRAC,
            "bilateral_cloud_threshold": DEFAULT_BILATERAL_CLOUD_THRESHOLD,
            "z_threshold": DEFAULT_Z_THRESHOLD,
            "mad_sigma_floor": DEFAULT_MAD_SIGMA_FLOOR,
            "contamination_fractions": list(DEFAULT_CONTAMINATION_FRACTIONS),
            "cloud_diameter_mm_range": list(DEFAULT_CLOUD_DIAMETER_MM_RANGE),
            "cloud_peak_range": list(DEFAULT_CLOUD_PEAK_RANGE),
            "tfce_n_steps": int(n_steps),
            "tfce_max_stroke_width_mm": tfce_cal.DEFAULT_MAX_STROKE_WIDTH_MM,
            "seeds": list(seeds),
        },
        "sources": source_records,
        "unmodified_exp_t6_summary": unmodified_table,
        "q31_rescue_summary_table": summary_table,
        "supplementary_parallax_ablation_20pct_summary": parallax_summary,
        "per_map_results": map_results,
        "total_elapsed_seconds": round(time.perf_counter() - t0, 3),
    }

    if output_json is not None:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, indent=2)
        if "C:\\Users\\" in serialized or "C:/Users/" in serialized:
            raise RuntimeError("Refusing to write local user path into evidence JSON")
        out_path.write_text(serialized + "\n", encoding="utf-8")

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Exp T8 (Q31 + Q34) CPU-only cloud-masked TFCE rescue benchmark."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_THIS_DIR / "evidence" / "q31_cloud_masked_tfce_rescue.json",
        help="Destination path for JSON evidence output.",
    )
    args = parser.parse_args(argv)
    payload = run_q31_benchmark(output_json=args.output)
    hb = payload["http_budget"]
    print(
        f"Wrote {args.output.name} | HTTP {hb['used_mib']:.3f} / {hb['cap_mib']:.1f} MiB | "
        f"elapsed {payload['total_elapsed_seconds']:.1f}s"
    )
    print("\n--- Unmodified Real Map Exp T6 (Q34) Strip Evaluation ---")
    for row in payload["unmodified_exp_t6_summary"]:
        ts = row["text_strip"]
        bm = row["blank_margin"]
        print(
            f"  {row['map_id']:<20} TEXT : sb_max={ts['sb_tfce_dom_max']:>6.2f} "
            f"maxR={ts['max_tfce_ratio']:>6.2f} diff99={ts['diff_tfce_p99']:>6.2f} "
            f"q34={ts['q34_score']:.4f} pass={ts['passes_q34_bar']} ({ts['q34_regime_label']})"
        )
        print(
            f"  {row['map_id']:<20} BLANK: sb_max={bm['sb_tfce_dom_max']:>6.2f} "
            f"maxR={bm['max_tfce_ratio']:>6.2f} diff99={bm['diff_tfce_p99']:>6.2f} "
            f"q34={bm['q34_score']:.4f} pass={bm['passes_q34_bar']} ({bm['q34_regime_label']})"
        )
    print("\n--- Q31 Bilateral Cloud Contamination Rescue Summary ---")
    for row in payload["q31_rescue_summary_table"]:
        print(
            f"  cloud={int(round(row['cloud_area_frac']*100)):>2}% | {row['pipeline']:<18} | "
            f"top1={row['top1_hit_rate']:.3f} | top3_true={row['top3_true_text_recall']:.3f} | "
            f"top3_ref={row['top3_clean_ref_recall']:.3f} | cloud_top3_FAR={row['cloud_top3_false_alarm_rate']:.3f} | "
            f"cloud_FPR={row['cloud_outranks_text_fpr']:.3f} | sigma_inf={row['mean_sigma_inflation_factor']:.3f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
