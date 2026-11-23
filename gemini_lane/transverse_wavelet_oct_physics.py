# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Exp T5: Wavelet Scattering + Kovesi Phase Congruency + OCT 31D Depth-Waveform CPU Physics Test.

Strictly CPU-only (zero torch/CUDA imports), zero persistent CT on disk (RAM only),
and hard-capped HTTP byte budgets (<= 60 MiB PHerc0306B + <= 60 MiB PHerc0139 <= 120 MiB global).

Tests three genuinely non-redundant transverse physics families on TWO public read
controls (BOTH NOT eligible First Letters scrolls):
1. Control 1 — Faint 8.64 um PHerc. 0306B w074 (`PHerc0306B_20250521133212_w074own_touchup_V5Rc`):
   Bounded Line 1 ROI + surrounding blank papyrus margin (<= 25 slices via
   ``stream_snap_render.py`` source-support path, strictly <= 60 MiB total HTTP bytes).
2. Control 2 — High-res 2.399 um PHerc. 0139 (`20260422000000-title_2026042222_zmid_flatboi`):
   Exact co-registered ROI ``[4224:4736, 13312:13952]`` (``ink >= 179``, ``blank in (0, 64]``,
   <= 60 MiB) matching Exp T3 ``raw_ct_surface = 0.7148``.

Three Pure NumPy/SciPy Feature Families:
1. OCT / Seismic 31D Peak-Aligned Depth-Waveform Analysis:
   - Sub-voxel peak-aligned 31D depth profile ``I(n)`` around papyrus peak
   - Hilbert analytic envelope peak, circular instantaneous phase asymmetry, envelope asymmetry,
     and normalized ``|I(+n) - I(-n)|`` profile asymmetry
   - 3rd-order normal depth skewness + front-vs-back slope log-ratio
   - Unsupervised Local Reed-Xiaoli (RX) Mahalanobis distance (hollow-annulus background mean
     + locally blended shrinkage-regularized precision across full 31D and 15D antisymmetric vectors)
   - Odd antisymmetric vs symmetric energy ratio + Antisymmetric MNF & PCA depth unmixing
2. Astrophysics 2D Wavelet Scattering (S_1 & S_2; Mallat 2012 / Cheng & Menard 2020):
   - Pure NumPy FFT 2D analytic Log-Gabor bank across 3 scales x 6 orientations
   - S_1 isotropic energy, orientation anisotropy, scale spectral slope
   - S_2 cross-orientation (Delta_theta = 90 deg) vs parallel (Delta_theta = 0 deg) modulation
     ratio and normalized S_2 / S_1 coupling
3. Kovesi Log-Gabor Phase Congruency (1999):
   - Strictly contrast- and offset-invariant multi-scale local phase alignment ``PC``
   - Maximum covariance moment ``M`` (edges), minimum covariance moment ``m`` (crackle corners),
     corner-to-edge ratio, and locally pooled corner density

Predeclared Evaluation & Gate (NO post-hoc tuning):
- Exact Mann-Whitney ``ROC-AUC``, ``Cohen's d``, and conservative 16-shift spatially
  rolled-label null (``AUC - NullMean``) on BOTH scrolls.
- Predeclared PASS gate:
  (a) Faint PHerc0306B transverse feature achieves ``ROC-AUC >= 0.70 AND DeltaNull >= +0.10``, OR
  (b) High-res PHerc0139 transverse feature beats ``raw_ct_surface`` (``0.7148``) by
      ``>= +0.05`` (``ROC-AUC >= 0.765 AND DeltaNull >= +0.10``).
  Otherwise honest ``CLOSED NO-GO``.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import io
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy import linalg as sla
from scipy import ndimage as ndi
from scipy import signal as ssig

try:
    from .pherc0139_transverse_physics import (
        BLANK_LOW_U8_THRESHOLD,
        DEFAULT_URLS as PHERC0139_DEFAULT_URLS,
        INK_HIGH_U8_THRESHOLD,
        MIB,
        BudgetExceededError,
        BudgetedRemoteZarrV2,
        HttpByteBudget,
        _fill_nearest_2d,
        _normalized_gaussian,
        assert_no_local_paths,
        budgeted_http_get,
        cohens_d,
        evaluate_feature_with_rolled_null,
        exact_roc_auc,
        extract_subvoxel_height_field,
        http_head_content_length,
        load_ink_map_in_ram,
        load_tifxyz_in_ram,
    )
    from .stream_snap_render import (
        _positive_weight_stencil_support,
        _sanitize_source_label,
        compute_grid_normals,
        validate_tifxyz_channels,
    )
except ImportError:  # pragma: no cover - direct script execution
    from pherc0139_transverse_physics import (
        BLANK_LOW_U8_THRESHOLD,
        DEFAULT_URLS as PHERC0139_DEFAULT_URLS,
        INK_HIGH_U8_THRESHOLD,
        MIB,
        BudgetExceededError,
        BudgetedRemoteZarrV2,
        HttpByteBudget,
        _fill_nearest_2d,
        _normalized_gaussian,
        assert_no_local_paths,
        budgeted_http_get,
        cohens_d,
        evaluate_feature_with_rolled_null,
        exact_roc_auc,
        extract_subvoxel_height_field,
        http_head_content_length,
        load_ink_map_in_ram,
        load_tifxyz_in_ram,
    )
    from stream_snap_render import (
        _positive_weight_stencil_support,
        _sanitize_source_label,
        compute_grid_normals,
        validate_tifxyz_channels,
    )


# ---------------------------------------------------------------------------
# Public Control Endpoints & Predeclared Experiment Constants
# ---------------------------------------------------------------------------

PHERC0306B_COMMIT = "feabff6c54f95bd8e52a90e0a7a76881ee67d4cc"
PHERC0306B_GH_BASE = (
    f"https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/{PHERC0306B_COMMIT}/submission"
)

DEFAULT_URLS: dict[str, str] = {
    "pherc0306b_ct_zarr": (
        "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
        "PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr"
    ),
    "pherc0306b_mesh": (
        f"{PHERC0306B_GH_BASE}/mesh/PHerc0306B_20250521133212_w074own_touchup_V5Rc"
    ),
    "pherc0306b_claim_png": (
        f"{PHERC0306B_GH_BASE}/PHerc0306B_20250521133212_w074own_touchup_V5Rc.png"
    ),
    "pherc0306b_claim_json": (
        f"{PHERC0306B_GH_BASE}/PHerc0306B_20250521133212_w074own_touchup_V5Rc.json"
    ),
    "pherc0139_sv_zarr": PHERC0139_DEFAULT_URLS["surface_volume_2p4um"],
    "pherc0139_ink_tif": PHERC0139_DEFAULT_URLS["ink_reference_2p4um"],
}

PHERC0306B_VOXEL_UM = 8.640
PHERC0139_VOXEL_UM = 2.399
PHERC0139_FIXED_ROI: tuple[int, int, int, int] = (4224, 4736, 13312, 13952)
PHERC0139_RAW_CT_REFERENCE_AUC = 0.7148

# Predeclared budgets (strictly enforced via HttpByteBudget)
DEFAULT_0306B_BUDGET_MIB = 60.0
DEFAULT_0139_BUDGET_MIB = 60.0
DEFAULT_GLOBAL_BUDGET_MIB = 120.0

# Predeclared evaluation gates (NEVER tuned post-hoc)
GATE_0306B_MIN_AUC = 0.70
GATE_0139_MIN_AUC = 0.765  # beats PHerc0139 raw_ct_surface (0.7148) by >= +0.05
GATE_MIN_ABOVE_NULL = 0.10

# Predeclared PHerc0306B claim-PNG label thresholds (matching rows.method d >= 0.5)
PHERC0306B_INK_D_THRESHOLD = 0.50
PHERC0306B_BLANK_D_THRESHOLD = 0.05
PHERC0306B_NUM_SLICES = 25
PHERC0306B_ROI_H = 288
PHERC0306B_ROI_W = 320

# Predeclared OCT 31D waveform geometry
OCT_WAVEFORM_DIM = 31
OCT_HALF_SPAN_VX_0306B = 7.5   # +/- 7.5 vx = +/- 64.8 um inside 25-slice slab
OCT_HALF_SPAN_VX_0139 = 15.0   # +/- 15.0 vx = +/- 36.0 um inside 109-slice slab

BASELINE_FEATURE_NAMES = frozenset({"raw_ct_surface"})

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "baseline": ("raw_ct_surface",),
    "oct_31d_depth_waveform": (
        "oct_hilbert_env_peak",
        "oct_hilbert_phase_asym",
        "oct_hilbert_env_asym",
        "oct_profile_abs_asym",
        "oct_depth_skewness",
        "oct_front_back_slope_ratio",
        "oct_rx_mahalanobis_local",
        "oct_rx_mahalanobis_antisym",
        "oct_antisym_energy_ratio",
        "oct_antisym_mnf_pc1",
        "oct_antisym_pca_pc1",
    ),
    "wavelet_scattering_2d": (
        "wst_s1_isotropic",
        "wst_s1_anisotropy",
        "wst_s1_scale_slope",
        "wst_s2_cross_par_ratio",
        "wst_s2_normalized_coupling",
    ),
    "kovesi_phase_congruency": (
        "pc_total",
        "pc_max_moment_edges",
        "pc_min_moment_corners",
        "pc_corner_edge_ratio",
        "pc_min_moment_corners_pooled",
    ),
}


# ---------------------------------------------------------------------------
# 1. In-RAM loaders & PHerc. 0306B claim-frame geometry / alignment
# ---------------------------------------------------------------------------

def load_json_in_ram(
    source: str | os.PathLike[str],
    budget: HttpByteBudget | None = None,
    part: str = "part_a",
    *,
    label: str = "json",
) -> dict[str, Any]:
    """Load JSON strictly in RAM from HTTP(S) or local path."""
    src = str(source)
    if not src.lower().startswith(("http://", "https://")):
        return json.loads(Path(src).read_text(encoding="utf-8"))
    assert budget is not None
    raw = budgeted_http_get(src, budget, part, label=label)
    assert raw is not None
    return json.loads(raw.decode("utf-8"))


def load_png_rgb_in_ram(
    source: str | os.PathLike[str],
    budget: HttpByteBudget | None = None,
    part: str = "part_a",
    *,
    label: str = "claim_png",
) -> np.ndarray:
    """Load RGB PNG strictly in RAM (HEAD-checked against budget before download)."""
    src = str(source)
    if not src.lower().startswith(("http://", "https://")):
        with Image.open(src) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    assert budget is not None
    cl = http_head_content_length(src)
    if cl is not None:
        budget.check_planned(part, cl, label=label)
    raw = budgeted_http_get(src, budget, part, label=label)
    assert raw is not None
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def decode_claim_png_tone(
    rgb: np.ndarray,
    claim_meta: dict[str, Any],
    *,
    max_perp_err: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert ``pixel = page_rgb * (1 - d) + ink_rgb * d`` to recover displayed ink tone ``d``.

    Returns ``(d_tone, on_sheet_mask, perp_rgb_err)``.
    Off-sheet blue-grey pixels and colored overlays have large ``perp_rgb_err`` and are
    strictly excluded from ``on_sheet_mask``.
    """
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got {arr.shape}")
    arr = arr[..., :3]
    disp = claim_meta.get("display", {})
    page = np.asarray(disp.get("page_rgb", [243.0, 236.0, 222.0]), dtype=np.float32)
    ink = np.asarray(disp.get("ink_rgb", [22.0, 20.0, 18.0]), dtype=np.float32)
    vec = ink - page
    denom = float(np.dot(vec, vec))
    if denom <= 1e-6:
        raise ValueError("Degenerate page_rgb / ink_rgb display mapping")

    diff = arr - page[None, None, :]
    d_raw = np.einsum("ijk,k->ij", diff, vec) / denom
    recon = page[None, None, :] + d_raw[..., None] * vec[None, None, :]
    perp_err = np.linalg.norm(arr - recon, axis=-1).astype(np.float32)
    on_sheet = (perp_err < float(max_perp_err)) & (d_raw >= -0.05) & (d_raw <= 1.05)
    d_tone = np.clip(d_raw, 0.0, 1.0).astype(np.float32)
    return d_tone, on_sheet, perp_err


def _extract_0306b_frame_params(
    claim_meta: dict[str, Any], mesh_meta: dict[str, Any]
) -> dict[str, float]:
    """Parse exact claim-frame physical mapping constants from claim & mesh metadata."""
    img_px = claim_meta["image_px"]
    ctx = claim_meta["context_mm"]
    px_per_mm = float(claim_meta["px_per_mm"])
    voxel_mm = float(claim_meta.get("voxel_size_um", PHERC0306B_VOXEL_UM)) / 1000.0
    scale = mesh_meta.get("scale", [0.05, 0.05])
    scale_x = float(scale[0]) if isinstance(scale, (list, tuple)) and len(scale) >= 1 else 0.05
    scale_y = float(scale[1]) if isinstance(scale, (list, tuple)) and len(scale) >= 2 else scale_x
    vx_per_node_u = 1.0 / scale_x
    vx_per_node_v = 1.0 / scale_y
    cell_u_mm = vx_per_node_u * voxel_mm
    cell_v_mm = vx_per_node_v * voxel_mm

    # Carrier grid origin from mesh meta ("node (i, j) <-> (u, v) = (-36 + 0.1728 j, -21 + 0.1728 i) mm")
    # Also allow explicit override in synthetic tests via mesh_meta["carrier_origin_mm"].
    origin = mesh_meta.get("carrier_origin_mm", [-36.0, -21.0])
    u_origin_mm = float(origin[0])
    v_origin_mm = float(origin[1])

    return {
        "map_left": float(img_px["map_left"]),
        "map_top": float(img_px["map_top"]),
        "map_w": float(img_px["map_w"]),
        "map_h": float(img_px["map_h"]),
        "context_u0_mm": float(ctx["u0"]),
        "context_u1_mm": float(ctx["u1"]),
        "context_v0_mm": float(ctx["v0"]),
        "context_v1_mm": float(ctx["v1"]),
        "px_per_mm": px_per_mm,
        "u_origin_mm": u_origin_mm,
        "v_origin_mm": v_origin_mm,
        "cell_u_mm": cell_u_mm,
        "cell_v_mm": cell_v_mm,
    }


def map_0306b_image_px_to_mesh_uv(
    y_img: np.ndarray,
    x_img: np.ndarray,
    claim_meta: dict[str, Any],
    mesh_meta: dict[str, Any],
    *,
    flip_v: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map PHerc. 0306B claim PNG ``(y_img, x_img)`` to tifxyz mesh ``(gv, gu)`` and ``(u_mm, v_mm)``.

    Reading orientation flips ``v`` top-to-bottom relative to the raw render:
    ``u_mm = context_u0 + (x_img - map_left) / K``,
    ``v_mm = context_v1 - (y_img - map_top) / K`` (when ``flip_v=True``).
    Mesh node ``(gv, gu) = ((v_mm - v_origin) / cell_v_mm, (u_mm - u_origin) / cell_u_mm)``.
    """
    p = _extract_0306b_frame_params(claim_meta, mesh_meta)
    xf = np.asarray(x_img, dtype=np.float64)
    yf = np.asarray(y_img, dtype=np.float64)
    u_mm = p["context_u0_mm"] + (xf - p["map_left"]) / p["px_per_mm"]
    if flip_v:
        v_mm = p["context_v1_mm"] - (yf - p["map_top"]) / p["px_per_mm"]
    else:
        v_mm = p["context_v0_mm"] + (yf - p["map_top"]) / p["px_per_mm"]
    gu = (u_mm - p["u_origin_mm"]) / p["cell_u_mm"]
    gv = (v_mm - p["v_origin_mm"]) / p["cell_v_mm"]
    return gv, gu, u_mm, v_mm


def verify_0306b_alignment(
    png_rgb: np.ndarray,
    claim_meta: dict[str, Any],
    valid_mesh: np.ndarray,
    mesh_meta: dict[str, Any],
    *,
    sample_stride: int = 20,
) -> dict[str, Any]:
    """Verify PHerc. 0306B PNG-to-mesh alignment via claim box round-trip and on-sheet mask IoU."""
    p = _extract_0306b_frame_params(claim_meta, mesh_meta)
    box_mm = claim_meta["claim_box_mm"]
    box_px = claim_meta["claim_box_image_px"]

    pred_x0 = p["map_left"] + (float(box_mm["u0"]) - p["context_u0_mm"]) * p["px_per_mm"]
    pred_x1 = p["map_left"] + (float(box_mm["u1"]) - p["context_u0_mm"]) * p["px_per_mm"]
    pred_y0 = p["map_top"] + (p["context_v1_mm"] - float(box_mm["v1"])) * p["px_per_mm"]
    pred_y1 = p["map_top"] + (p["context_v1_mm"] - float(box_mm["v0"])) * p["px_per_mm"]
    box_err = max(
        abs(pred_x0 - float(box_px["x0"])),
        abs(pred_x1 - float(box_px["x1"])),
        abs(pred_y0 - float(box_px["y0"])),
        abs(pred_y1 - float(box_px["y1"])),
    )

    h_img, w_img = png_rgb.shape[:2]
    y_lo = max(0, int(round(p["map_top"])) + 8)
    y_hi = min(h_img, int(round(p["map_top"] + p["map_h"])) - 8)
    x_lo = max(0, int(round(p["map_left"])) + 8)
    x_hi = min(w_img, int(round(p["map_left"] + p["map_w"])) - 8)
    step = max(1, int(sample_stride))
    ys = np.arange(y_lo, y_hi, step, dtype=np.int64)
    xs = np.arange(x_lo, x_hi, step, dtype=np.int64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    _, png_on_sheet, _ = decode_claim_png_tone(png_rgb[yy, xx], claim_meta)

    def _iou(flip_v: bool) -> tuple[float, float]:
        gv, gu, _, _ = map_0306b_image_px_to_mesh_uv(yy, xx, claim_meta, mesh_meta, flip_v=flip_v)
        iv = np.rint(gv).astype(np.int64)
        iu = np.rint(gu).astype(np.int64)
        inb = (iv >= 0) & (iv < valid_mesh.shape[0]) & (iu >= 0) & (iu < valid_mesh.shape[1])
        m_on = np.zeros_like(inb)
        m_on[inb] = valid_mesh[iv[inb], iu[inb]]
        inter = int((png_on_sheet & m_on).sum())
        union = int((png_on_sheet | m_on).sum())
        agree = float((png_on_sheet == m_on).mean())
        return (float(inter / max(union, 1)), agree)

    iou_flipped, agree_flipped = _iou(flip_v=True)
    iou_unflipped, agree_unflipped = _iou(flip_v=False)

    return {
        "claim_box_roundtrip_max_px_error": round(float(box_err), 4),
        "on_sheet_iou_reading_flip": round(iou_flipped, 5),
        "on_sheet_agreement_reading_flip": round(agree_flipped, 5),
        "on_sheet_iou_unflipped": round(iou_unflipped, 5),
        "on_sheet_agreement_unflipped": round(agree_unflipped, 5),
        "orientation_verified_flipped": bool(iou_flipped >= iou_unflipped and box_err <= 2.0),
    }


def build_0306b_rays_for_roi(
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    valid: np.ndarray,
    *,
    claim_meta: dict[str, Any],
    mesh_meta: dict[str, Any],
    roi_bounds_img_px: tuple[int, int, int, int],
    num_slices: int = PHERC0306B_NUM_SLICES,
    slice_step: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Interpolate 3D normal rays ``(L, H, W, 3)`` and mesh usability mask for PHerc. 0306B ROI.

    Uses ``stream_snap_render.compute_grid_normals`` and
    ``_positive_weight_stencil_support`` so sentinel vertices never corrupt rays.
    """
    y0, y1, x0, x1 = [int(v) for v in roi_bounds_img_px]
    h = y1 - y0
    w = x1 - x0
    if h < 4 or w < 4:
        raise ValueError(f"ROI too small: {roi_bounds_img_px}")

    vx, vy, vz, val = validate_tifxyz_channels(cx, cy, cz)
    val = val & np.asarray(valid, dtype=bool)
    normals, normal_ok = compute_grid_normals(vx, vy, vz, valid=val)
    usable = val & normal_ok

    yy, xx = np.meshgrid(
        np.arange(y0, y1, dtype=np.float64),
        np.arange(x0, x1, dtype=np.float64),
        indexing="ij",
    )
    gv, gu, u_mm, v_mm = map_0306b_image_px_to_mesh_uv(yy, xx, claim_meta, mesh_meta, flip_v=True)

    mesh_coords = np.stack([np.zeros(gv.size), gv.ravel(), gu.ravel()])
    mesh_usable = _positive_weight_stencil_support(
        usable[None, ...], mesh_coords
    ).reshape(h, w)

    pts = np.stack([vx, vy, vz], axis=-1).astype(np.float64)
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

    p_out = np.zeros((h, w, 3), dtype=np.float64)
    n_out = np.zeros((h, w, 3), dtype=np.float64)
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
        mesh_usable = mesh_usable & (n_len[..., 0] > 1e-6)
        n_out[mesh_usable] /= n_len[mesh_usable]
        n_out[~mesh_usable] = 0.0

    center_idx = (num_slices - 1) // 2
    offsets = (np.arange(num_slices, dtype=np.float64) - float(center_idx)) * float(slice_step)
    rays = p_out[None, :, :, :] + offsets[:, None, None, None] * n_out[None, :, :, :]
    coord_summary = {
        "u_mm_range": [round(float(u_mm.min()), 4), round(float(u_mm.max()), 4)],
        "v_mm_range": [round(float(v_mm.min()), 4), round(float(v_mm.max()), 4)],
        "mesh_gu_range": [round(float(gu.min()), 3), round(float(gu.max()), 3)],
        "mesh_gv_range": [round(float(gv.min()), 3), round(float(gv.max()), 3)],
    }
    return rays, mesh_usable, coord_summary


def select_0306b_line1_roi(
    png_rgb: np.ndarray,
    claim_meta: dict[str, Any],
    mesh_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]],
    reader: BudgetedRemoteZarrV2,
    *,
    roi_h: int = PHERC0306B_ROI_H,
    roi_w: int = PHERC0306B_ROI_W,
    num_slices: int = PHERC0306B_NUM_SLICES,
    max_planned_bytes: int = 44 * MIB,
    ink_d_thresh: float = PHERC0306B_INK_D_THRESHOLD,
    blank_d_thresh: float = PHERC0306B_BLANK_D_THRESHOLD,
    step_x: int = 32,
) -> dict[str, Any]:
    """Deterministically select a bounded Line 1 ROI + blank margin on PHerc. 0306B w074.

    Selection uses ONLY the published claim PNG tone, mesh usability mask, and exact
    Zarr chunk planner (ZERO peeking at CT intensities).
    """
    cx, cy, cz, valid, mesh_meta = mesh_tuple
    box = claim_meta["claim_box_image_px"]
    y_line1 = int(claim_meta["rows"]["centre_y_image_px"][0])
    # Keep strictly inside the claim box border and above the Line 1 / Line 2 midpoint.
    y_min = int(box["y0"]) + 16
    y_max = int(box["y1"]) - 16
    if len(claim_meta["rows"]["centre_y_image_px"]) >= 2:
        y_mid = (y_line1 + int(claim_meta["rows"]["centre_y_image_px"][1])) // 2
        y_max = min(y_max, y_mid)

    y0 = max(y_min, min(y_max - roi_h, y_line1 - roi_h // 2))
    y1 = y0 + roi_h
    if y1 > y_max:
        raise ValueError(f"Requested roi_h={roi_h} does not fit in Line 1 band [{y_min}, {y_max}]")

    x_lo = int(box["x0"]) + 32
    x_hi = int(box["x1"]) - 32
    if x_hi - x_lo < roi_w:
        x_lo = int(box["x0"])
        x_hi = int(box["x1"])

    strip_rgb = png_rgb[y0:y1, x_lo:x_hi]
    d_strip, on_strip, err_strip = decode_claim_png_tone(strip_rgb, claim_meta)

    _, normal_ok = compute_grid_normals(cx, cy, cz, valid=valid)
    usable = valid & normal_ok
    pts = np.stack([cx, cy, cz], axis=-1).astype(np.float64)
    idx_v = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    ptsf = pts[idx_v[0], idx_v[1]]
    idx_u = ndi.distance_transform_edt(~usable, return_distances=False, return_indices=True)
    nrmf = compute_grid_normals(cx, cy, cz, valid=valid)[0].astype(np.float64)[idx_u[0], idx_u[1]]

    coarse_cands: list[tuple[int, float, int, float, float, float]] = []
    offs = np.arange(-(num_slices // 2), num_slices // 2 + 1, dtype=np.float64)

    for rel_x0 in range(0, (x_hi - x_lo) - roi_w + 1, max(1, int(step_x))):
        x0 = x_lo + rel_x0
        x1 = x0 + roi_w
        dw = d_strip[:, rel_x0 : rel_x0 + roi_w]
        ow = on_strip[:, rel_x0 : rel_x0 + roi_w]
        if float(ow.mean()) < 0.99:
            continue
        im = ow & (dw >= ink_d_thresh)
        bm = ow & (dw <= blank_d_thresh)
        ink_f = float(im.mean())
        blank_f = float(bm.mean())
        if ink_f < 0.12 or ink_f > 0.50 or blank_f < 0.35:
            continue

        # Fast 4x4 coarse ray chunk estimate.
        yy_c, xx_c = np.meshgrid(
            np.arange(y0, y1, 4, dtype=np.float64),
            np.arange(x0, x1, 4, dtype=np.float64),
            indexing="ij",
        )
        gv_c, gu_c, _, _ = map_0306b_image_px_to_mesh_uv(yy_c, xx_c, claim_meta, mesh_meta, flip_v=True)
        iv0, iv1 = int(math.floor(gv_c.min())), int(math.ceil(gv_c.max()))
        iu0, iu1 = int(math.floor(gu_c.min())), int(math.ceil(gu_c.max()))
        if iv0 < 0 or iu0 < 0 or iv1 >= usable.shape[0] or iu1 >= usable.shape[1]:
            continue
        if float(usable[iv0 : iv1 + 1, iu0 : iu1 + 1].mean()) < 0.99:
            continue
        p_c = np.stack(
            [ndi.map_coordinates(ptsf[..., ch], [gv_c, gu_c], order=1, mode="nearest") for ch in range(3)],
            axis=-1,
        )
        n_c = np.stack(
            [ndi.map_coordinates(nrmf[..., ch], [gv_c, gu_c], order=1, mode="nearest") for ch in range(3)],
            axis=-1,
        )
        n_c /= np.maximum(np.linalg.norm(n_c, axis=-1, keepdims=True), 1e-9)
        rays_c = p_c[None] + offs[:, None, None, None] * n_c[None]
        n_ch_c, _ = reader.plan_trilinear_chunks(rays_c, max_direct_box_voxels=1)
        mean_ink_d = float(dw[im].mean()) if im.any() else 0.0
        balance_score = min(ink_f, blank_f) * mean_ink_d
        coarse_cands.append((int(n_ch_c), -balance_score, x0, ink_f, blank_f, mean_ink_d))

    if not coarse_cands:
        raise RuntimeError("No PHerc. 0306B Line 1 ROI candidate met ink/blank balance and mesh coverage.")

    coarse_cands.sort(key=lambda t: (t[0], t[1], t[2]))

    # Exact full-resolution ray chunk check on top candidates (smallest chunk footprint first).
    best_choice: dict[str, Any] | None = None
    best_key: tuple[int, float] | None = None
    for _, neg_score, x0, ink_f, blank_f, mean_ink_d in coarse_cands[:12]:
        bounds = (y0, y1, x0, x0 + roi_w)
        rays, m_use, coord_summary = build_0306b_rays_for_roi(
            cx, cy, cz, valid,
            claim_meta=claim_meta,
            mesh_meta=mesh_meta,
            roi_bounds_img_px=bounds,
            num_slices=num_slices,
            slice_step=1.0,
        )
        if float(m_use.mean()) < 0.99:
            continue
        n_ch, n_bytes = reader.plan_trilinear_chunks(rays[:, m_use, :], max_direct_box_voxels=1)
        if n_bytes > max_planned_bytes:
            continue
        # Strongly prefer comfortably under budget while maximizing Line 1 ink/blank balance.
        tier = 0 if n_bytes <= int(0.80 * max_planned_bytes) else 1
        key = (tier, neg_score)
        if best_key is None or key < best_key:
            best_key = key
            rel_x0 = x0 - x_lo
            max_err = float(err_strip[:, rel_x0 : rel_x0 + roi_w].max())
            best_choice = {
                "roi_bounds_image_px": [int(y0), int(y1), int(x0), int(x0 + roi_w)],
                "roi_shape_px": [int(roi_h), int(roi_w)],
                "roi_physical_mm": [
                    round(roi_h * PHERC0306B_VOXEL_UM / 1000.0, 4),
                    round(roi_w * PHERC0306B_VOXEL_UM / 1000.0, 4),
                ],
                "line1_centre_y_image_px": int(y_line1),
                "ink_d_threshold": float(ink_d_thresh),
                "blank_d_threshold": float(blank_d_thresh),
                "ink_fraction": round(ink_f, 4),
                "blank_fraction": round(blank_f, 4),
                "mean_ink_d": round(mean_ink_d, 4),
                "max_rgb_projection_err": round(max_err, 4),
                "mesh_usable_fraction": round(float(m_use.mean()), 4),
                "planned_zarr_chunks": int(n_ch),
                "planned_zarr_bytes": int(n_bytes),
                "planned_zarr_mib": round(n_bytes / MIB, 4),
                "candidates_screened": len(coarse_cands),
                "selection_mode": "deterministic_line1_tone_and_budget_zero_ct_peek",
                **coord_summary,
            }

    if best_choice is None:
        raise BudgetExceededError(
            f"No PHerc. 0306B Line 1 ROI fit within planned Zarr budget ({max_planned_bytes / MIB:.2f} MiB)."
        )
    return best_choice


# ---------------------------------------------------------------------------
# 2. Family 1: OCT / Seismic 31D Peak-Aligned Depth-Waveform Physics
# ---------------------------------------------------------------------------

def extract_peak_aligned_depth_profiles(
    slab: np.ndarray,
    n_peak: np.ndarray,
    *,
    half_span_vx: float,
    waveform_dim: int = OCT_WAVEFORM_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-voxel align each pixel's depth profile ``I(n)`` to ``n_peak`` on a symmetric ``D``-point grid.

    Returns ``(profiles [D, H, W], offsets_vx [D], fully_in_bounds [H, W])``.
    ``D`` must be odd so ``offsets_vx[D // 2] == 0.0`` and ``offsets_vx[c + m] == -offsets_vx[c - m]``.
    """
    vol = np.asarray(slab, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D slab [L, H, W], got {vol.shape}")
    depth, h, w = vol.shape
    d_dim = int(waveform_dim)
    if d_dim < 5 or d_dim % 2 == 0:
        raise ValueError(f"waveform_dim must be odd >= 5, got {d_dim}")
    hs = float(half_span_vx)
    if hs <= 0.0:
        raise ValueError(f"half_span_vx must be positive, got {hs}")

    npk = np.asarray(n_peak, dtype=np.float64)
    offsets = np.linspace(-hs, hs, d_dim, dtype=np.float64)
    in_bounds = (npk - hs >= 0.0) & (npk + hs <= float(depth - 1))

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    profiles = np.empty((d_dim, h, w), dtype=np.float32)
    max_z = float(depth - 1)
    for k, off in enumerate(offsets):
        z_q = np.clip(npk + off, 0.0, max_z)
        z0 = np.floor(z_q).astype(np.int64)
        z1 = np.minimum(z0 + 1, depth - 1)
        wz = (z_q - z0).astype(np.float32)
        profiles[k] = (1.0 - wz) * vol[z0, yy, xx] + wz * vol[z1, yy, xx]
    return profiles, offsets, in_bounds


def _local_blended_rx_mahalanobis(
    vectors: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    inner_sigma_um: float = 30.0,
    outer_sigma_um: float = 180.0,
    n_grid: int = 3,
    shrinkage: float = 0.05,
    trim_top_frac: float = 0.25,
) -> np.ndarray:
    """Unsupervised Local Reed-Xiaoli (RX) Mahalanobis distance with hollow-annulus background.

    Estimates background mean via a normalized dual-Gaussian hollow annulus (suppressing
    self-leakage of localized ink strokes) and blends shrinkage-regularized local precision
    matrices smoothly across a ``n_grid x n_grid`` grid of overlapping spatial blocks.
    Uses iterative top-norm trimming inside each block to approximate non-ink background
    covariance strictly WITHOUT ground-truth labels.
    """
    vecs = np.asarray(vectors, dtype=np.float64)
    d_dim, h, w = vecs.shape
    vm = np.asarray(valid_mask, dtype=bool)
    if not vm.any():
        return np.zeros((h, w), dtype=np.float32)

    sig_in = max(1.0, float(inner_sigma_um) / float(voxel_um))
    sig_out = max(sig_in * 2.5, float(outer_sigma_um) / float(voxel_um))
    # Approximate Gaussian hollow annulus: subtract central peak contribution from outer blur.
    w_in = min(0.35, (sig_in / sig_out) ** 2)

    resid = np.empty_like(vecs)
    for k in range(d_dim):
        ch = vecs[k]
        g_out = _normalized_gaussian(ch, vm, sigma=sig_out)
        g_in = _normalized_gaussian(ch, vm, sigma=sig_in)
        mu_bg = (g_out - w_in * g_in) / (1.0 - w_in)
        resid[k] = np.where(vm, ch - mu_bg, 0.0)

    def _fit_precision(pts_dn: np.ndarray) -> np.ndarray:
        n_pts = pts_dn.shape[1]
        if n_pts < max(d_dim + 4, 16):
            return np.eye(d_dim, dtype=np.float64)
        # Unsupervised robust background trimming: drop top trim_top_frac by L2 residual norm.
        if 0.0 < trim_top_frac < 0.45 and n_pts > 40:
            norms = np.einsum("ij,ij->j", pts_dn, pts_dn)
            cut = float(np.quantile(norms, 1.0 - trim_top_frac))
            keep = norms <= cut
            if int(keep.sum()) >= max(d_dim + 4, 16):
                pts_dn = pts_dn[:, keep]
                n_pts = pts_dn.shape[1]
        mu = pts_dn.mean(axis=1, keepdims=True)
        xc = pts_dn - mu
        cov = (xc @ xc.T) / float(max(n_pts - 1, 1))
        tr_mean = float(np.trace(cov)) / float(d_dim)
        if not math.isfinite(tr_mean) or tr_mean <= 1e-12:
            return np.eye(d_dim, dtype=np.float64)
        lam = float(np.clip(shrinkage, 1e-4, 0.99))
        cov_reg = (1.0 - lam) * cov + lam * tr_mean * np.eye(d_dim, dtype=np.float64)
        try:
            cho, lower = sla.cho_factor(cov_reg, lower=True, check_finite=False)
            inv_cov = sla.cho_solve((cho, lower), np.eye(d_dim, dtype=np.float64), check_finite=False)
            return 0.5 * (inv_cov + inv_cov.T)
        except sla.LinAlgError:
            return np.linalg.pinv(cov_reg)

    # Global fallback precision for sparse blocks.
    global_pts = resid[:, vm]
    prec_global = _fit_precision(global_pts)

    ng = max(1, int(n_grid))
    if ng == 1 or min(h, w) < 32:
        # Direct global RX on tiny grids.
        flat_r = resid.reshape(d_dim, -1)
        m2 = np.einsum("dn,dk,kn->n", flat_r, prec_global, flat_r).reshape(h, w)
        return np.where(vm, np.sqrt(np.maximum(0.0, m2)), 0.0).astype(np.float32)

    # Fit precision matrices on overlapping n_grid x n_grid local windows.
    cy_centers = np.linspace(0.5 * h / ng, h - 0.5 * h / ng, ng)
    cx_centers = np.linspace(0.5 * w / ng, w - 0.5 * w / ng, ng)
    half_h = 0.75 * h / ng
    half_w = 0.75 * w / ng
    precs = np.empty((ng, ng, d_dim, d_dim), dtype=np.float64)
    for iy, cy in enumerate(cy_centers):
        r0 = max(0, int(math.floor(cy - half_h)))
        r1 = min(h, int(math.ceil(cy + half_h)))
        for ix, cx in enumerate(cx_centers):
            c0 = max(0, int(math.floor(cx - half_w)))
            c1 = min(w, int(math.ceil(cx + half_w)))
            sub_vm = vm[r0:r1, c0:c1]
            if int(sub_vm.sum()) >= max(2 * d_dim, 32):
                precs[iy, ix] = _fit_precision(resid[:, r0:r1, c0:c1][:, sub_vm])
            else:
                precs[iy, ix] = prec_global

    # Smooth bilinear interpolation weights across tile centers (convex combination of SPD matrices).
    yy = np.arange(h, dtype=np.float64)
    xx = np.arange(w, dtype=np.float64)

    def _interp_1d_weights(coords: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(centers) == 1:
            z = np.zeros(len(coords), dtype=np.int64)
            return z, z, np.zeros(len(coords), dtype=np.float64)
        idx_hi = np.searchsorted(centers, coords)
        idx_hi = np.clip(idx_hi, 1, len(centers) - 1)
        idx_lo = idx_hi - 1
        t = (coords - centers[idx_lo]) / np.maximum(centers[idx_hi] - centers[idx_lo], 1e-9)
        return idx_lo, idx_hi, np.clip(t, 0.0, 1.0)

    iy0, iy1, wy = _interp_1d_weights(yy, cy_centers)
    ix0, ix1, wx = _interp_1d_weights(xx, cx_centers)

    # Evaluate quadratic form r^T P_tile r for each (iy, ix) tile and blend bilinearly!
    # Because r^T (sum_t w_t P_t) r == sum_t w_t (r^T P_t r), this is mathematically EXACT
    # and runs in O(ng^2 * D^2 * H * W) vectorized BLAS with tiny RAM!
    flat_r = resid.reshape(d_dim, h * w)
    q_tiles = np.empty((ng, ng, h, w), dtype=np.float64)
    for iy in range(ng):
        for ix in range(ng):
            pr = precs[iy, ix] @ flat_r
            q_tiles[iy, ix] = np.einsum("dn,dn->n", flat_r, pr).reshape(h, w)

    w00 = (1.0 - wy)[:, None] * (1.0 - wx)[None, :]
    w01 = (1.0 - wy)[:, None] * wx[None, :]
    w10 = wy[:, None] * (1.0 - wx)[None, :]
    w11 = wy[:, None] * wx[None, :]

    row_idx = np.arange(h)[:, None]
    col_idx = np.arange(w)[None, :]
    m2 = (
        w00 * q_tiles[iy0[:, None], ix0[None, :], row_idx, col_idx]
        + w01 * q_tiles[iy0[:, None], ix1[None, :], row_idx, col_idx]
        + w10 * q_tiles[iy1[:, None], ix0[None, :], row_idx, col_idx]
        + w11 * q_tiles[iy1[:, None], ix1[None, :], row_idx, col_idx]
    )
    return np.where(vm, np.sqrt(np.maximum(0.0, m2)), 0.0).astype(np.float32)


def _antisymmetric_mnf_and_pca(
    antisym: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute unsupervised Minimum Noise Fraction (MNF, Green 1988) and PCA on odd depth vectors."""
    a_arr = np.asarray(antisym, dtype=np.float64)
    m_dim, h, w = a_arr.shape
    vm = np.asarray(valid_mask, dtype=bool)
    n_valid = int(vm.sum())
    if n_valid < max(m_dim + 4, 24):
        zeros = np.zeros((h, w), dtype=np.float32)
        return zeros, zeros, {"mnf_top_snr": 1.0, "pca_pc1_var_frac": 0.0}

    pts = a_arr[:, vm]
    mu = pts.mean(axis=1, keepdims=True)
    xc = pts - mu
    cov_total = (xc @ xc.T) / float(max(n_valid - 1, 1))
    cov_total = 0.5 * (cov_total + cov_total.T)

    # PCA leading mode
    evals_p, evecs_p = sla.eigh(cov_total)
    pc1_vec = evecs_p[:, -1]
    # Deterministic sign convention: positive sum of weights (or largest-magnitude entry > 0).
    if abs(float(pc1_vec.sum())) > 1e-9:
        if float(pc1_vec.sum()) < 0.0:
            pc1_vec = -pc1_vec
    elif float(pc1_vec[int(np.argmax(np.abs(pc1_vec)))]) < 0.0:
        pc1_vec = -pc1_vec
    pca_var_frac = float(max(0.0, evals_p[-1]) / max(float(np.sum(np.maximum(evals_p, 0.0))), 1e-12))

    # MNF shift-difference noise covariance (Green et al. 1988)
    vm_y = vm[1:, :] & vm[:-1, :]
    vm_x = vm[:, 1:] & vm[:, :-1]
    dy = (a_arr[:, 1:, :] - a_arr[:, :-1, :])[:, vm_y]
    dx = (a_arr[:, :, 1:] - a_arr[:, :, :-1])[:, vm_x]
    n_diff = dy.shape[1] + dx.shape[1]
    if n_diff >= max(m_dim + 4, 24):
        diffs = np.concatenate([dy, dx], axis=1)
        # Var(X_1 - X_2) = 2 * Var(noise) -> divide by 2
        cov_noise = 0.5 * (diffs @ diffs.T) / float(n_diff)
    else:
        cov_noise = np.eye(m_dim, dtype=np.float64) * max(float(np.trace(cov_total)) / m_dim, 1e-6)

    cov_noise = 0.5 * (cov_noise + cov_noise.T)
    tr_n = float(np.trace(cov_noise)) / float(m_dim)
    cov_noise_reg = 0.95 * cov_noise + 0.05 * max(tr_n, 1e-9) * np.eye(m_dim, dtype=np.float64)

    try:
        evals_m, evecs_m = sla.eigh(cov_total, cov_noise_reg)
        mnf1_vec = evecs_m[:, -1]
        top_snr = float(evals_m[-1])
    except sla.LinAlgError:
        mnf1_vec = pc1_vec.copy()
        top_snr = 1.0

    # Normalize and enforce deterministic sign
    mnf1_vec = mnf1_vec / max(float(np.linalg.norm(mnf1_vec)), 1e-12)
    if abs(float(mnf1_vec.sum())) > 1e-9:
        if float(mnf1_vec.sum()) < 0.0:
            mnf1_vec = -mnf1_vec
    elif float(mnf1_vec[int(np.argmax(np.abs(mnf1_vec)))]) < 0.0:
        mnf1_vec = -mnf1_vec

    centered_all = a_arr - mu[:, :, None]
    mnf_map = np.tensordot(mnf1_vec, centered_all, axes=(0, 0))
    pca_map = np.tensordot(pc1_vec, centered_all, axes=(0, 0))
    mnf_map = np.where(vm, mnf_map, 0.0).astype(np.float32)
    pca_map = np.where(vm, pca_map, 0.0).astype(np.float32)
    return mnf_map, pca_map, {
        "mnf_top_snr": round(top_snr, 4),
        "pca_pc1_var_frac": round(pca_var_frac, 4),
    }


def compute_oct_depth_waveform_features(
    slab: np.ndarray,
    n_peak: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    half_span_vx: float,
    waveform_dim: int = OCT_WAVEFORM_DIM,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute Family 1: OCT / Seismic 31D Peak-Aligned Depth-Waveform features."""
    profiles, offsets, in_bounds = extract_peak_aligned_depth_profiles(
        slab, n_peak, half_span_vx=half_span_vx, waveform_dim=waveform_dim
    )
    d_dim, h, w = profiles.shape
    c = d_dim // 2  # 15 for D=31
    vm = np.asarray(valid_mask, dtype=bool)
    p64 = profiles.astype(np.float64)
    eps = 1e-9

    # (a) Hilbert analytic signal along centered depth waveform.
    # Circularly shift so center index c maps to index 0 before FFT Hilbert so that
    # any exact even profile P[c-m] == P[c+m] produces exact conjugate symmetry Z[c-m] == conj(Z[c+m]).
    p_mean = p64.mean(axis=0, keepdims=True)
    p_zero = p64 - p_mean
    z_shifted = ssig.hilbert(np.fft.ifftshift(p_zero, axes=0), axis=0)
    z_analytic = np.fft.fftshift(z_shifted, axes=0)
    env = np.abs(z_analytic)
    phi = np.angle(z_analytic)

    hilbert_env_peak = env[c].astype(np.float32)

    # Use inner-to-mid offsets m = 1..k_asym (avoiding extreme endpoints) for Hilbert phase/env asymmetry.
    k_asym = min(c, max(4, (2 * c) // 3))  # 10 for c=15
    pos_idx = np.arange(c + 1, c + 1 + k_asym)
    neg_idx = np.arange(c - 1, c - 1 - k_asym, -1)

    env_pos = env[pos_idx]
    env_neg = env[neg_idx]
    w_pair = env_pos * env_neg
    phase_asym_sin = np.abs(np.sin(phi[pos_idx] + phi[neg_idx]))
    hilbert_phase_asym = (
        np.sum(w_pair * phase_asym_sin, axis=0) / (np.sum(w_pair, axis=0) + eps)
    ).astype(np.float32)

    hilbert_env_asym = (
        np.sum(env_pos - env_neg, axis=0) / (np.sum(env_pos + env_neg, axis=0) + eps)
    ).astype(np.float32)

    p_std = np.std(p64, axis=0)
    profile_abs_asym = (
        np.mean(np.abs(p64[pos_idx] - p64[neg_idx]), axis=0) / (p_std + eps)
    ).astype(np.float32)

    # (b) 3rd-order normal depth skewness (anchored at papyrus peak delta=0) + front-vs-back slope ratio.
    # Subtract lower-quartile air/background floor and apply a symmetric Gaussian depth
    # taper so far-tail noise at +/-half_span cannot overwhelm the cubic moment (delta^3).
    p_base = np.percentile(p64, 25.0, axis=0, keepdims=True)
    depth_taper = np.exp(-0.5 * (offsets / max(0.45 * float(half_span_vx), 1e-6)) ** 2)[:, None, None]
    w_mass = np.maximum(0.0, p64 - p_base) * depth_taper
    w_sum = np.sum(w_mass, axis=0) + eps
    offs_col = offsets[:, None, None]
    var_d = np.sum(w_mass * (offs_col ** 2), axis=0) / w_sum
    skew_d = np.sum(w_mass * (offs_col ** 3), axis=0) / (w_sum * np.power(np.maximum(var_d, 1e-9), 1.5))
    depth_skewness = np.clip(skew_d, -10.0, 10.0).astype(np.float32)

    k_slope = min(c, 6)
    slope_pos = np.mean(np.abs(np.diff(p64[c : c + k_slope + 1], axis=0)), axis=0)
    slope_neg = np.mean(np.abs(np.diff(p64[c - k_slope : c + 1], axis=0)), axis=0)
    front_back_slope_ratio = np.log((slope_pos + 1e-4) / (slope_neg + 1e-4)).astype(np.float32)

    # (d) Exact even/odd decomposition around papyrus peak c.
    p_pos_all = p64[c + 1 :]            # (c, H, W)
    p_neg_all = p64[c - 1 :: -1]        # (c, H, W)
    p_even = 0.5 * (p_pos_all + p_neg_all)
    p_odd = 0.5 * (p_pos_all - p_neg_all)

    e_odd = np.sum(p_odd ** 2, axis=0)
    p_even_c = p_even - np.mean(p_even, axis=0, keepdims=True)
    e_even = np.sum(p_even_c ** 2, axis=0)
    antisym_energy_ratio = (e_odd / (e_odd + e_even + eps)).astype(np.float32)

    mnf_pc1, pca_pc1, mnf_diag = _antisymmetric_mnf_and_pca(p_odd, vm)

    # (c) Unsupervised Local Reed-Xiaoli (RX) Mahalanobis distance (31D full & 15D antisymmetric).
    rx_local_full = _local_blended_rx_mahalanobis(
        p64, vm, voxel_um=voxel_um, inner_sigma_um=30.0, outer_sigma_um=180.0, n_grid=3, trim_top_frac=0.25
    )
    rx_local_antisym = _local_blended_rx_mahalanobis(
        p_odd, vm, voxel_um=voxel_um, inner_sigma_um=30.0, outer_sigma_um=180.0, n_grid=3, trim_top_frac=0.25
    )

    feats = {
        "oct_hilbert_env_peak": np.where(vm, hilbert_env_peak, 0.0).astype(np.float32),
        "oct_hilbert_phase_asym": np.where(vm, hilbert_phase_asym, 0.0).astype(np.float32),
        "oct_hilbert_env_asym": np.where(vm, hilbert_env_asym, 0.0).astype(np.float32),
        "oct_profile_abs_asym": np.where(vm, profile_abs_asym, 0.0).astype(np.float32),
        "oct_depth_skewness": np.where(vm, depth_skewness, 0.0).astype(np.float32),
        "oct_front_back_slope_ratio": np.where(vm, front_back_slope_ratio, 0.0).astype(np.float32),
        "oct_rx_mahalanobis_local": rx_local_full,
        "oct_rx_mahalanobis_antisym": rx_local_antisym,
        "oct_antisym_energy_ratio": np.where(vm, antisym_energy_ratio, 0.0).astype(np.float32),
        "oct_antisym_mnf_pc1": mnf_pc1,
        "oct_antisym_pca_pc1": pca_pc1,
    }
    diagnostics = {
        "waveform_dim": int(d_dim),
        "half_span_vx": float(half_span_vx),
        "half_span_um": round(float(half_span_vx * voxel_um), 3),
        "step_vx": round(float(offsets[1] - offsets[0]), 4),
        "fully_in_bounds_fraction": round(float(in_bounds[vm].mean()) if vm.any() else 0.0, 4),
        "antisym_energy_ratio_mean": round(float(antisym_energy_ratio[vm].mean()) if vm.any() else 0.0, 5),
        **mnf_diag,
    }
    return feats, diagnostics


# ---------------------------------------------------------------------------
# 3. Family 2: Astrophysics 2D Wavelet Scattering (S_1 & S_2, Mallat / Cheng & Menard 2020)
# ---------------------------------------------------------------------------

def _build_log_gabor_bank_fft(
    shape_hw: tuple[int, int],
    *,
    wavelengths_px: Sequence[float],
    n_orientations: int = 6,
    sigma_onf: float = 0.65,
    dtheta_on_sigma: float = 1.2,
    angular_window: str = "gaussian",
    l2_normalize: bool = True,
) -> np.ndarray:
    """Build analytic 2D Log-Gabor wavelet bank ``(J, L, H, W)`` in unshifted FFT coordinates.

    Every filter is strictly zero at DC ``(0, 0)`` and one-sided in frequency so its
    spatial impulse response is complex analytic (even real part, odd imaginary part).
    ``angular_window="raised_cosine"`` uses Kovesi's (1999/2003) compact-support raised
    cosine ``0.5 * (cos(min(dtheta * L / 2, pi)) + 1)`` which is identically zero for
    orthogonal orientations ``|dtheta| >= 2*pi/L``; ``"gaussian"`` uses a smooth Morlet/
    Log-Gabor angular envelope for wavelet scattering.
    """
    h, w = int(shape_hw[0]), int(shape_hw[1])
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    fyy, fxx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.hypot(fyy, fxx)
    radius_safe = np.where(radius > 0.0, radius, 1.0)
    theta = np.arctan2(-fyy, fxx)  # standard math angle in [-pi, pi]

    j_cnt = len(wavelengths_px)
    l_cnt = int(n_orientations)
    sigma_theta = (math.pi / float(l_cnt)) / float(dtheta_on_sigma)
    log_sig2 = 2.0 * (math.log(float(sigma_onf)) ** 2)

    # Precompute radial filters.
    radials = np.empty((j_cnt, h, w), dtype=np.float64)
    for j, wl in enumerate(wavelengths_px):
        f0 = 1.0 / float(wl)
        rad = np.exp(-((np.log(radius_safe / f0)) ** 2) / log_sig2)
        rad[0, 0] = 0.0
        # Smooth high-frequency corner cutoff near Nyquist (0.5 cyc/px) for isotropy.
        lp = 1.0 / (1.0 + np.power(radius / 0.45, 16.0))
        radials[j] = rad * lp

    # Precompute directional spreads (one-sided analytic filters over [0, pi)).
    spreads = np.empty((l_cnt, h, w), dtype=np.float64)
    for l_idx in range(l_cnt):
        ang = l_idx * math.pi / float(l_cnt)
        # Signed angular distance wrapped to [-pi, pi]
        ds = np.sin(theta) * math.cos(ang) - np.cos(theta) * math.sin(ang)
        dc = np.cos(theta) * math.cos(ang) + np.sin(theta) * math.sin(ang)
        dtheta = np.abs(np.arctan2(ds, dc))
        if angular_window == "raised_cosine":
            # Exact Kovesi (2003) compact-support angular spread: 0 for |dtheta| >= pi / (L/2)
            # Tightened to half-width pi / L so a 1D edge never leaks into +/-60 deg or 90 deg channels.
            dtheta_scaled = np.minimum(dtheta * float(l_cnt), math.pi)
            spreads[l_idx] = 0.5 * (np.cos(dtheta_scaled) + 1.0)
        else:
            spreads[l_idx] = np.exp(-(dtheta ** 2) / (2.0 * sigma_theta * sigma_theta))

    bank = np.empty((j_cnt, l_cnt, h, w), dtype=np.float64)
    for j in range(j_cnt):
        for l_idx in range(l_cnt):
            filt = radials[j] * spreads[l_idx]
            filt[0, 0] = 0.0
            if l2_normalize:
                norm = math.sqrt(float(np.sum(filt * filt)) / float(h * w))
                if norm > 1e-12:
                    filt = filt / norm
            bank[j, l_idx] = filt
    return bank


def compute_wavelet_scattering_features(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    wavelengths_px: Sequence[float] = (4.0, 8.0, 16.0),
    n_orientations: int = 6,
    avg_sigma_um: float = 35.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute Family 2: Astrophysics 2D Wavelet Scattering (S_1 & S_2, Cheng & Menard 2020)."""
    img = np.asarray(image, dtype=np.float64)
    vm = np.asarray(valid_mask, dtype=bool)
    h, w = img.shape
    if not vm.any():
        zeros = np.zeros((h, w), dtype=np.float32)
        return {k: zeros.copy() for k in FEATURE_FAMILIES["wavelet_scattering_2d"]}, {}

    filled = _fill_nearest_2d(img, vm)
    mu = float(filled[vm].mean())
    sd = max(float(filled[vm].std()), 1e-6)
    normed = (filled - mu) / sd

    pad = min(32, max(4, min(h, w) - 1))
    padded = np.pad(normed, pad, mode="reflect")
    hp, wp = padded.shape

    bank = _build_log_gabor_bank_fft(
        (hp, wp),
        wavelengths_px=wavelengths_px,
        n_orientations=n_orientations,
        sigma_onf=0.65,
        dtheta_on_sigma=1.2,
        l2_normalize=True,
    )
    j_cnt, l_cnt = bank.shape[:2]
    fft_img = np.fft.fft2(padded)

    sigma_avg_px = max(1.5, float(avg_sigma_um) / float(voxel_um))
    eps = 1e-9

    # 1st-order scattering fields U_1(j, l) and locally averaged S_1(j, l).
    u1_padded = np.empty((j_cnt, l_cnt, hp, wp), dtype=np.float64)
    s1 = np.empty((j_cnt, l_cnt, h, w), dtype=np.float64)
    for j in range(j_cnt):
        for l_idx in range(l_cnt):
            resp = np.fft.ifft2(fft_img * bank[j, l_idx])
            mod = np.abs(resp)
            u1_padded[j, l_idx] = mod
            crop = mod[pad : pad + h, pad : pad + w]
            s1[j, l_idx] = _normalized_gaussian(crop, vm, sigma=sigma_avg_px)

    s1_isotropic = np.mean(s1, axis=(0, 1))
    s1_scale_mean = np.mean(s1, axis=1)  # (J, H, W)
    s1_scale_aniso = (np.max(s1, axis=1) - np.min(s1, axis=1)) / (s1_scale_mean + eps)
    s1_anisotropy = np.mean(s1_scale_aniso, axis=0)
    s1_scale_slope = np.log((s1_scale_mean[0] + 1e-5) / (s1_scale_mean[-1] + 1e-5))

    # 2nd-order scattering S_2(j1, l1, j2, l2) for j2 > j1:
    # parallel (Delta_theta = 0 deg -> l2 == l1) vs perpendicular (Delta_theta = 90 deg -> l2 == (l1 + L//2) % L).
    ortho_shift = l_cnt // 2
    s2_par_acc = np.zeros((h, w), dtype=np.float64)
    s2_perp_acc = np.zeros((h, w), dtype=np.float64)
    s2_norm_perp_acc = np.zeros((h, w), dtype=np.float64)
    pair_count = 0

    for j1 in range(j_cnt - 1):
        for l1 in range(l_cnt):
            fft_u1 = np.fft.fft2(u1_padded[j1, l1])
            l_perp = (l1 + ortho_shift) % l_cnt
            for j2 in range(j1 + 1, j_cnt):
                u2_par = np.abs(np.fft.ifft2(fft_u1 * bank[j2, l1]))[pad : pad + h, pad : pad + w]
                u2_perp = np.abs(np.fft.ifft2(fft_u1 * bank[j2, l_perp]))[pad : pad + h, pad : pad + w]
                s2_par = _normalized_gaussian(u2_par, vm, sigma=sigma_avg_px)
                s2_perp = _normalized_gaussian(u2_perp, vm, sigma=sigma_avg_px)
                s2_par_acc += s2_par
                s2_perp_acc += s2_perp
                s2_norm_perp_acc += s2_perp / (s1[j1, l1] + 1e-5)
                pair_count += 1

    denom_pairs = float(max(pair_count, 1))
    s2_par_mean = s2_par_acc / denom_pairs
    s2_perp_mean = s2_perp_acc / denom_pairs
    s2_cross_par_ratio = np.log((s2_perp_mean + 1e-5) / (s2_par_mean + 1e-5))
    s2_normalized_coupling = s2_norm_perp_acc / denom_pairs

    feats = {
        "wst_s1_isotropic": np.where(vm, s1_isotropic, 0.0).astype(np.float32),
        "wst_s1_anisotropy": np.where(vm, s1_anisotropy, 0.0).astype(np.float32),
        "wst_s1_scale_slope": np.where(vm, s1_scale_slope, 0.0).astype(np.float32),
        "wst_s2_cross_par_ratio": np.where(vm, s2_cross_par_ratio, 0.0).astype(np.float32),
        "wst_s2_normalized_coupling": np.where(vm, s2_normalized_coupling, 0.0).astype(np.float32),
    }
    diagnostics = {
        "wavelengths_px": [float(w_) for w_ in wavelengths_px],
        "wavelengths_um": [round(float(w_ * voxel_um), 3) for w_ in wavelengths_px],
        "n_orientations": int(l_cnt),
        "avg_sigma_px": round(sigma_avg_px, 3),
        "s2_pairs_evaluated": int(pair_count),
        "s1_isotropic_mean": round(float(s1_isotropic[vm].mean()), 5),
        "s2_cross_par_ratio_mean": round(float(s2_cross_par_ratio[vm].mean()), 5),
    }
    return feats, diagnostics


# ---------------------------------------------------------------------------
# 4. Family 3: Kovesi Log-Gabor Phase Congruency (1999)
# ---------------------------------------------------------------------------

def compute_kovesi_phase_congruency_features(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    n_scales: int = 4,
    n_orientations: int = 6,
    min_wavelength_px: float = 3.0,
    mult: float = 2.1,
    sigma_onf: float = 0.55,
    k_noise: float = 2.0,
    cutoff: float = 0.4,
    gain: float = 10.0,
    pool_sigma_um: float = 30.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute Family 3: Kovesi (1999) Log-Gabor Phase Congruency + Covariance Moments (M, m).

    Strictly invariant to affine contrast changes ``I -> alpha * I + beta`` (``alpha > 0``):
    DC removal eliminates ``beta``, while ``alpha`` cancels identically between phase energy,
    Rayleigh noise threshold ``T``, and amplitude sum ``sum_A``.
    """
    img = np.asarray(image, dtype=np.float64)
    vm = np.asarray(valid_mask, dtype=bool)
    h, w = img.shape
    if not vm.any():
        zeros = np.zeros((h, w), dtype=np.float32)
        return {k: zeros.copy() for k in FEATURE_FAMILIES["kovesi_phase_congruency"]}, {}

    filled = _fill_nearest_2d(img, vm)
    # Remove global mean (exact zero-DC safety for float64 precision at large offsets).
    filled = filled - float(filled[vm].mean())

    pad = min(32, max(4, min(h, w) - 1))
    padded = np.pad(filled, pad, mode="reflect")
    hp, wp = padded.shape

    wavelengths = [float(min_wavelength_px) * (float(mult) ** s) for s in range(int(n_scales))]
    bank = _build_log_gabor_bank_fft(
        (hp, wp),
        wavelengths_px=wavelengths,
        n_orientations=n_orientations,
        sigma_onf=sigma_onf,
        dtheta_on_sigma=1.2,
        angular_window="raised_cosine",
        l2_normalize=False,
    )
    fft_img = np.fft.fft2(padded)

    eps = 1e-12
    # Scale-adaptive floor proportional to image std so zero-energy orthogonal channels
    # never blow up on noiseless edges while preserving exact invariance under I -> alpha * I.
    img_scale = max(float(np.std(filled[vm])), 1e-12)
    amp_eps = 1e-3 * img_scale

    cov_x2 = np.zeros((h, w), dtype=np.float64)
    cov_y2 = np.zeros((h, w), dtype=np.float64)
    cov_xy = np.zeros((h, w), dtype=np.float64)
    pc_sum = np.zeros((h, w), dtype=np.float64)
    noise_thresholds: list[float] = []

    for o_idx in range(int(n_orientations)):
        ang = o_idx * math.pi / float(n_orientations)
        eo_list = []
        an_list = []
        sum_e = np.zeros((h, w), dtype=np.float64)
        sum_o = np.zeros((h, w), dtype=np.float64)
        sum_an = np.zeros((h, w), dtype=np.float64)
        max_an = np.zeros((h, w), dtype=np.float64)

        for s_idx in range(int(n_scales)):
            resp = np.fft.ifft2(fft_img * bank[s_idx, o_idx])[pad : pad + h, pad : pad + w]
            eo_list.append(resp)
            an = np.abs(resp)
            an_list.append(an)
            sum_e += resp.real
            sum_o += resp.imag
            sum_an += an
            max_an = np.maximum(max_an, an)

        # Kovesi (1999) Rayleigh noise energy estimation from finest scale (s = 0).
        a0 = an_list[0][vm]
        median_e2n = float(np.median(a0 * a0))
        mean_e2n = -median_e2n / math.log(0.5)
        tau = math.sqrt(max(mean_e2n, 0.0)) * math.sqrt(math.pi / 2.0)
        total_tau = tau * (1.0 - (1.0 / float(mult)) ** int(n_scales)) / (1.0 - 1.0 / float(mult))
        est_noise_mean = total_tau * math.sqrt(math.pi / 2.0)
        est_noise_sigma = total_tau * math.sqrt((4.0 - math.pi) / 2.0)
        t_thresh = est_noise_mean + float(k_noise) * est_noise_sigma
        noise_thresholds.append(t_thresh)

        # Mean phase vector & phase deviation energy.
        x_energy = np.hypot(sum_e, sum_o) + amp_eps
        mean_e = sum_e / x_energy
        mean_o = sum_o / x_energy

        energy = np.zeros((h, w), dtype=np.float64)
        for resp in eo_list:
            e_r = resp.real
            e_i = resp.imag
            energy += (e_r * mean_e + e_i * mean_o) - np.abs(e_r * mean_o - e_i * mean_e)

        # Frequency spread weighting W(x).
        width = (sum_an / (max_an + amp_eps) - 1.0) / float(max(int(n_scales) - 1, 1))
        weight = 1.0 / (1.0 + np.exp(float(gain) * (float(cutoff) - width)))

        pc_o = weight * np.maximum(energy - t_thresh, 0.0) / (sum_an + amp_eps)
        pc_o = np.clip(pc_o, 0.0, 1.0)

        pc_sum += pc_o
        pc_x = pc_o * math.cos(ang)
        pc_y = pc_o * math.sin(ang)
        cov_x2 += pc_x * pc_x
        cov_y2 += pc_y * pc_y
        cov_xy += 2.0 * pc_x * pc_y

    pc_total = pc_sum / float(n_orientations)
    denom = np.sqrt(cov_xy * cov_xy + (cov_x2 - cov_y2) ** 2 + eps)
    m_max = 0.5 * (cov_x2 + cov_y2 + denom)
    m_min = np.maximum(0.0, 0.5 * (cov_x2 + cov_y2 - denom))
    corner_edge_ratio = m_min / (m_max + 1e-6)

    pool_sigma_px = max(1.5, float(pool_sigma_um) / float(voxel_um))
    m_min_pooled = _normalized_gaussian(m_min, vm, sigma=pool_sigma_px)

    feats = {
        "pc_total": np.where(vm, pc_total, 0.0).astype(np.float32),
        "pc_max_moment_edges": np.where(vm, m_max, 0.0).astype(np.float32),
        "pc_min_moment_corners": np.where(vm, m_min, 0.0).astype(np.float32),
        "pc_corner_edge_ratio": np.where(vm, corner_edge_ratio, 0.0).astype(np.float32),
        "pc_min_moment_corners_pooled": np.where(vm, m_min_pooled, 0.0).astype(np.float32),
    }
    diagnostics = {
        "wavelengths_px": [round(wl, 3) for wl in wavelengths],
        "wavelengths_um": [round(wl * voxel_um, 3) for wl in wavelengths],
        "n_scales": int(n_scales),
        "n_orientations": int(n_orientations),
        "pool_sigma_px": round(pool_sigma_px, 3),
        "mean_rayleigh_noise_threshold": round(float(np.mean(noise_thresholds)), 6),
        "pc_total_mean": round(float(pc_total[vm].mean()), 5),
        "pc_max_moment_mean": round(float(m_max[vm].mean()), 5),
        "pc_min_moment_mean": round(float(m_min[vm].mean()), 5),
    }
    return feats, diagnostics


# ---------------------------------------------------------------------------
# 5. Unified Feature Suite + Per-Scroll Evaluation
# ---------------------------------------------------------------------------

def compute_all_t5_features(
    slab: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    oct_half_span_vx: float,
    search_half_slices: int,
    z_smooth_sigma: float = 1.0,
    detrend_sigma_um: float = 200.0,
    band_half_width: int = 1,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Run all three Exp T5 feature families + raw_ct_surface baseline on a 3D slab."""
    hf = extract_subvoxel_height_field(
        slab,
        voxel_um=voxel_um,
        valid_mask=valid_mask,
        search_half_slices=search_half_slices,
        z_smooth_sigma=z_smooth_sigma,
        detrend_sigma_um=detrend_sigma_um,
        band_half_width=band_half_width,
    )
    sheet_valid = hf["sheet_valid"]
    surface_ct = hf["surface_ct"]
    n_peak = hf["n_peak"]

    oct_feats, oct_diag = compute_oct_depth_waveform_features(
        slab,
        n_peak,
        sheet_valid,
        voxel_um=voxel_um,
        half_span_vx=oct_half_span_vx,
        waveform_dim=OCT_WAVEFORM_DIM,
    )
    wst_feats, wst_diag = compute_wavelet_scattering_features(
        surface_ct,
        sheet_valid,
        voxel_um=voxel_um,
    )
    pc_feats, pc_diag = compute_kovesi_phase_congruency_features(
        surface_ct,
        sheet_valid,
        voxel_um=voxel_um,
    )

    all_feats: dict[str, np.ndarray] = {
        "raw_ct_surface": surface_ct.astype(np.float32),
        **oct_feats,
        **wst_feats,
        **pc_feats,
    }
    diagnostics = {
        "height_field": {
            "z_center_slice": int(hf["z_center_slice"]),
            "z_search_window": hf["z_search_window"],
            "sheet_valid_fraction": round(float(sheet_valid.mean()), 4),
            "height_field_std_um": round(
                float(np.std(hf["h_um"][sheet_valid])) if sheet_valid.any() else 0.0, 4
            ),
        },
        "oct_31d_depth_waveform": oct_diag,
        "wavelet_scattering_2d": wst_diag,
        "kovesi_phase_congruency": pc_diag,
    }
    return all_feats, sheet_valid, diagnostics


def _evaluate_feature_suite(
    feature_maps: dict[str, np.ndarray],
    ink_mask: np.ndarray,
    blank_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    gate_min_auc: float,
    gate_min_above_null: float,
    border_guard_px: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Evaluate all features against the predeclared ROC-AUC + 16-shift rolled-label null gate."""
    eval_valid = np.asarray(valid_mask, dtype=bool).copy()
    guard = min(int(border_guard_px), min(eval_valid.shape) // 8)
    if guard > 0:
        eval_valid[:guard, :] = False
        eval_valid[-guard:, :] = False
        eval_valid[:, :guard] = False
        eval_valid[:, -guard:] = False

    results: dict[str, dict[str, Any]] = {}
    for fname, fmap in feature_maps.items():
        res = evaluate_feature_with_rolled_null(
            fmap,
            ink_mask,
            blank_mask,
            eval_valid,
            gate_min_auc=gate_min_auc,
            gate_min_above_null=gate_min_above_null,
        )
        # Tag family explicitly for reporting clarity.
        fam = "unknown"
        for fk, fnames in FEATURE_FAMILIES.items():
            if fname in fnames:
                fam = fk
                break
        res["family"] = fam
        results[fname] = res

    transverse_names = [k for k in results if k not in BASELINE_FEATURE_NAMES]
    best_trans_name = max(
        transverse_names,
        key=lambda k: (results[k]["roc_auc"], results[k]["auc_above_null"]),
    )
    passing_trans = [k for k in transverse_names if results[k]["passes_gate"]]
    per_family_best: dict[str, dict[str, Any]] = {}
    for fam, fnames in FEATURE_FAMILIES.items():
        present = [k for k in fnames if k in results]
        if present:
            bk = max(present, key=lambda k: (results[k]["roc_auc"], results[k]["auc_above_null"]))
            per_family_best[fam] = {"feature": bk, **results[bk]}

    summary = {
        "border_guard_px": int(guard),
        "eval_valid_fraction": round(float(eval_valid.mean()), 4),
        "gate_min_roc_auc": float(gate_min_auc),
        "gate_min_auc_above_null": float(gate_min_above_null),
        "passing_transverse_features": passing_trans,
        "best_transverse_feature": {"name": best_trans_name, **results[best_trans_name]},
        "per_family_best": per_family_best,
        "verdict": "PASS" if len(passing_trans) > 0 else "CLOSED NO-GO",
    }
    return results, summary


# ---------------------------------------------------------------------------
# 6. Full Live / Offline End-to-End Pipeline
# ---------------------------------------------------------------------------

def run_wavelet_oct_ablation(
    *,
    urls: dict[str, str] | None = None,
    budget_0306b_mib: float = DEFAULT_0306B_BUDGET_MIB,
    budget_0139_mib: float = DEFAULT_0139_BUDGET_MIB,
    global_budget_mib: float = DEFAULT_GLOBAL_BUDGET_MIB,
    pherc0306b_roi_override: tuple[int, int, int, int] | None = None,
    pherc0306b_roi_h: int = PHERC0306B_ROI_H,
    pherc0306b_roi_w: int = PHERC0306B_ROI_W,
    pherc0306b_num_slices: int = PHERC0306B_NUM_SLICES,
    pherc0139_roi_bounds: tuple[int, int, int, int] = PHERC0139_FIXED_ROI,
) -> dict[str, Any]:
    """Run Exp T5 on PHerc. 0306B (8.64 um) and PHerc. 0139 (2.399 um) strictly in RAM on CPU."""
    t0 = time.monotonic()
    cfg = dict(DEFAULT_URLS)
    if urls:
        cfg.update(urls)

    budget = HttpByteBudget(
        part_a_limit_bytes=int(round(budget_0306b_mib * MIB)),
        part_b_limit_bytes=int(round(budget_0139_mib * MIB)),
        global_limit_bytes=int(round(global_budget_mib * MIB)),
    )

    with ExitStack() as stack:
        # ==================================================================
        # CONTROL 1: Faint 8.64 um PHerc. 0306B w074 (budget part_a <= 60 MiB)
        # ==================================================================
        claim_meta = load_json_in_ram(
            cfg["pherc0306b_claim_json"], budget=budget, part="part_a", label="0306b:claim_json"
        )
        mesh_0306b = load_tifxyz_in_ram(
            cfg["pherc0306b_mesh"], budget=budget, part="part_a", label="0306b:mesh"
        )
        cx306, cy306, cz306, val306, mesh_meta_306 = mesh_0306b
        png_0306b = load_png_rgb_in_ram(
            cfg["pherc0306b_claim_png"], budget=budget, part="part_a", label="0306b:claim_png"
        )

        align_306 = verify_0306b_alignment(png_0306b, claim_meta, val306, mesh_meta_306)
        reader_306 = stack.enter_context(
            BudgetedRemoteZarrV2(
                cfg["pherc0306b_ct_zarr"],
                level="0",
                budget=budget,
                part="part_a",
                label="0306b:ct_zarr",
                cache_mb=64.0,
                workers=8,
            )
        )

        rem_306_for_zarr = max(0, budget.remaining("part_a") - 2 * MIB)
        if pherc0306b_roi_override is not None:
            y0_306, y1_306, x0_306, x1_306 = [int(v) for v in pherc0306b_roi_override]
            rays_306, m_use_306, coord_306 = build_0306b_rays_for_roi(
                cx306, cy306, cz306, val306,
                claim_meta=claim_meta,
                mesh_meta=mesh_meta_306,
                roi_bounds_img_px=(y0_306, y1_306, x0_306, x1_306),
                num_slices=pherc0306b_num_slices,
                slice_step=1.0,
            )
            n_ch_306, planned_b_306 = reader_306.plan_trilinear_chunks(
                rays_306[:, m_use_306, :], max_direct_box_voxels=1
            )
            d_patch, on_patch, err_patch = decode_claim_png_tone(
                png_0306b[y0_306:y1_306, x0_306:x1_306], claim_meta
            )
            roi_info_306: dict[str, Any] = {
                "roi_bounds_image_px": [y0_306, y1_306, x0_306, x1_306],
                "roi_shape_px": [y1_306 - y0_306, x1_306 - x0_306],
                "roi_physical_mm": [
                    round((y1_306 - y0_306) * PHERC0306B_VOXEL_UM / 1000.0, 4),
                    round((x1_306 - x0_306) * PHERC0306B_VOXEL_UM / 1000.0, 4),
                ],
                "line1_centre_y_image_px": int(claim_meta["rows"]["centre_y_image_px"][0]),
                "ink_d_threshold": PHERC0306B_INK_D_THRESHOLD,
                "blank_d_threshold": PHERC0306B_BLANK_D_THRESHOLD,
                "ink_fraction": round(float((on_patch & (d_patch >= PHERC0306B_INK_D_THRESHOLD)).mean()), 4),
                "blank_fraction": round(float((on_patch & (d_patch <= PHERC0306B_BLANK_D_THRESHOLD)).mean()), 4),
                "max_rgb_projection_err": round(float(err_patch.max()), 4),
                "mesh_usable_fraction": round(float(m_use_306.mean()), 4),
                "planned_zarr_chunks": int(n_ch_306),
                "planned_zarr_bytes": int(planned_b_306),
                "planned_zarr_mib": round(planned_b_306 / MIB, 4),
                "selection_mode": "override",
                **coord_306,
            }
        else:
            roi_info_306 = select_0306b_line1_roi(
                png_0306b,
                claim_meta,
                mesh_0306b,
                reader_306,
                roi_h=pherc0306b_roi_h,
                roi_w=pherc0306b_roi_w,
                num_slices=pherc0306b_num_slices,
                max_planned_bytes=rem_306_for_zarr,
            )
            y0_306, y1_306, x0_306, x1_306 = [int(v) for v in roi_info_306["roi_bounds_image_px"]]
            rays_306, m_use_306, _ = build_0306b_rays_for_roi(
                cx306, cy306, cz306, val306,
                claim_meta=claim_meta,
                mesh_meta=mesh_meta_306,
                roi_bounds_img_px=(y0_306, y1_306, x0_306, x1_306),
                num_slices=pherc0306b_num_slices,
                slice_step=1.0,
            )
            _, planned_b_306 = reader_306.plan_trilinear_chunks(
                rays_306[:, m_use_306, :], max_direct_box_voxels=1
            )

        budget.check_planned("part_a", planned_b_306, label="0306b:ct_zarr_roi")

        h306, w306 = y1_306 - y0_306, x1_306 - x0_306
        slab_306 = np.zeros((pherc0306b_num_slices, h306, w306), dtype=np.float32)
        src_avail_306 = np.zeros((pherc0306b_num_slices, h306, w306), dtype=bool)
        if m_use_306.any():
            vals_306, supp_306 = reader_306.sample_trilinear(
                rays_306[:, m_use_306, :], max_direct_box_voxels=1, return_support=True
            )
            slab_306[:, m_use_306] = vals_306
            src_avail_306[:, m_use_306] = supp_306

        d_roi_306, on_roi_306, _ = decode_claim_png_tone(
            png_0306b[y0_306:y1_306, x0_306:x1_306], claim_meta
        )
        # Free large claim PNG immediately.
        del png_0306b

        full_src_306 = m_use_306 & src_avail_306.all(axis=0)
        nonzero_306 = slab_306.max(axis=0) > 0.0
        valid_306 = full_src_306 & nonzero_306 & on_roi_306
        ink_mask_306 = on_roi_306 & (d_roi_306 >= PHERC0306B_INK_D_THRESHOLD)
        blank_mask_306 = on_roi_306 & (d_roi_306 <= PHERC0306B_BLANK_D_THRESHOLD)

        half_span_306 = min(OCT_HALF_SPAN_VX_0306B, 0.30 * float(pherc0306b_num_slices - 1))
        feats_306, sheet_valid_306, diag_306 = compute_all_t5_features(
            slab_306,
            valid_306,
            voxel_um=PHERC0306B_VOXEL_UM,
            oct_half_span_vx=half_span_306,
            search_half_slices=min(6, (pherc0306b_num_slices - 1) // 2),
            z_smooth_sigma=0.8,
            detrend_sigma_um=200.0,
            band_half_width=1,
        )
        res_306, sum_306 = _evaluate_feature_suite(
            feats_306,
            ink_mask_306,
            blank_mask_306,
            sheet_valid_306,
            gate_min_auc=GATE_0306B_MIN_AUC,
            gate_min_above_null=GATE_MIN_ABOVE_NULL,
            border_guard_px=16,
        )

        # Free PHerc. 0306B slab and arrays before streaming PHerc. 0139.
        source_support_306 = {
            "mesh_usable_fraction": round(float(m_use_306.mean()), 4),
            "full_source_available_fraction": round(float(full_src_306.mean()), 4),
            "nonzero_intensity_fraction": round(float(nonzero_306.mean()), 4),
            "valid_mask_fraction": round(float(valid_306.mean()), 4),
            "chunks_fetched": int(reader_306.chunks_fetched),
            "bytes_fetched": int(reader_306.bytes_fetched),
        }
        vol_shape_306 = list(reader_306.shape)
        vol_chunks_306 = list(reader_306.chunks)
        mesh_shape_306 = list(val306.shape)
        del slab_306, src_avail_306, rays_306, feats_306

        # ==================================================================
        # CONTROL 2: High-res 2.399 um PHerc. 0139 (budget part_b <= 60 MiB)
        # ==================================================================
        sv_reader_139 = stack.enter_context(
            BudgetedRemoteZarrV2(
                cfg["pherc0139_sv_zarr"],
                level="0",
                budget=budget,
                part="part_b",
                label="0139:sv_2p4um",
                cache_mb=64.0,
                workers=8,
            )
        )
        ink_map_139 = load_ink_map_in_ram(
            cfg["pherc0139_ink_tif"], budget=budget, part="part_b"
        )
        sv_hw_139 = (int(sv_reader_139.shape[1]), int(sv_reader_139.shape[2]))
        if ink_map_139.shape != sv_hw_139:
            raise ValueError(
                f"PHerc0139 ink map shape {ink_map_139.shape} != surface volume HW {sv_hw_139}"
            )

        r0, r1, c0, c1 = [int(v) for v in pherc0139_roi_bounds]
        n_sv_ch_139, planned_b_139 = sv_reader_139.plan_box_chunks(
            (0, r0, c0), (sv_reader_139.shape[0], r1, c1)
        )
        budget.check_planned("part_b", planned_b_139, label="0139:sv_2p4um_roi")

        sv_slab_139, _, sv_supp_139 = sv_reader_139.read_box(
            (0, r0, c0), (sv_reader_139.shape[0], r1, c1), return_support=True
        )
        sv_valid_139 = sv_supp_139.all(axis=0) & (sv_slab_139.max(axis=0) > 0)

        ink_roi_139 = ink_map_139[r0:r1, c0:c1]
        del ink_map_139
        ink_mask_139 = ink_roi_139 >= INK_HIGH_U8_THRESHOLD
        blank_mask_139 = (ink_roi_139 > 0) & (ink_roi_139 <= BLANK_LOW_U8_THRESHOLD)

        half_span_139 = min(OCT_HALF_SPAN_VX_0139, 0.30 * float(sv_slab_139.shape[0] - 1))
        feats_139, sheet_valid_139, diag_139 = compute_all_t5_features(
            sv_slab_139,
            sv_valid_139,
            voxel_um=PHERC0139_VOXEL_UM,
            oct_half_span_vx=half_span_139,
            search_half_slices=min(24, (sv_slab_139.shape[0] - 1) // 2),
            z_smooth_sigma=1.0,
            detrend_sigma_um=200.0,
            band_half_width=1,
        )
        res_139, sum_139 = _evaluate_feature_suite(
            feats_139,
            ink_mask_139,
            blank_mask_139,
            sheet_valid_139,
            gate_min_auc=GATE_0139_MIN_AUC,
            gate_min_above_null=GATE_MIN_ABOVE_NULL,
            border_guard_px=32,
        )
        sv_shape_139 = list(sv_reader_139.shape)
        sv_chunks_139 = list(sv_reader_139.chunks)
        sv_fetched_139 = int(sv_reader_139.chunks_fetched)
        sv_bytes_139 = int(sv_reader_139.bytes_fetched)
        del sv_slab_139, sv_supp_139, feats_139

    # Honest predeclared verdict synthesis
    pass_306 = len(sum_306["passing_transverse_features"]) > 0
    pass_139 = len(sum_139["passing_transverse_features"]) > 0
    overall_verdict = "PASS" if (pass_306 or pass_139) else "CLOSED NO-GO"

    raw_budget = budget.summary()
    budget_receipt = {
        "pherc0306b_bytes": raw_budget["part_a_bytes"],
        "pherc0306b_mib": raw_budget["part_a_mib"],
        "pherc0306b_limit_mib": raw_budget["part_a_limit_mib"],
        "pherc0139_bytes": raw_budget["part_b_bytes"],
        "pherc0139_mib": raw_budget["part_b_mib"],
        "pherc0139_limit_mib": raw_budget["part_b_limit_mib"],
        "global_bytes": raw_budget["global_bytes"],
        "global_mib": raw_budget["global_mib"],
        "global_limit_mib": raw_budget["global_limit_mib"],
        "within_limits": raw_budget["within_limits"],
        "requests": {
            "pherc0306b": raw_budget["requests"]["part_a"],
            "pherc0139": raw_budget["requests"]["part_b"],
        },
        "breakdown_bytes": raw_budget["breakdown_bytes"],
    }

    receipt: dict[str, Any] = {
        "experiment_id": "exp_t5_wavelet_scattering_phase_congruency_oct_31d_ablation",
        "compute_mode": "CPU_ONLY",
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "controls_status": "both_published_read_controls_not_eligible_first_letters",
        "sources": {k: _sanitize_source_label(v) for k, v in cfg.items()},
        "predeclared_gate": {
            "pherc0306b_min_roc_auc": GATE_0306B_MIN_AUC,
            "pherc0306b_min_auc_above_null": GATE_MIN_ABOVE_NULL,
            "pherc0139_reference_raw_ct_auc": PHERC0139_RAW_CT_REFERENCE_AUC,
            "pherc0139_min_roc_auc": GATE_0139_MIN_AUC,
            "pherc0139_min_auc_above_null": GATE_MIN_ABOVE_NULL,
            "rolled_null_shifts": 16,
            "null_statistic": "two_sided_max(auc, 1-auc)_across_16_toroidal_shifts",
            "rule": "PASS iff any transverse feature passes PHerc0306B gate OR PHerc0139 gate",
        },
        "http_byte_budget": budget_receipt,
        "pherc0306b_8p64um": {
            "scroll": "PHerc0306B",
            "mesh_name": "PHerc0306B_20250521133212_w074own_touchup_V5Rc",
            "voxel_um": PHERC0306B_VOXEL_UM,
            "num_slices": int(pherc0306b_num_slices),
            "volume_shape": vol_shape_306,
            "volume_chunks": vol_chunks_306,
            "mesh_shape": mesh_shape_306,
            "alignment_verification": align_306,
            "roi": roi_info_306,
            "source_support": source_support_306,
            "diagnostics": diag_306,
            "summary": sum_306,
            "features": res_306,
        },
        "pherc0139_2p399um": {
            "scroll": "PHerc0139",
            "segment": "20260422000000-title_2026042222_zmid_flatboi",
            "voxel_um": PHERC0139_VOXEL_UM,
            "surface_volume_shape": sv_shape_139,
            "surface_volume_chunks": sv_chunks_139,
            "roi": {
                "roi_bounds_2p4um": [r0, r1, c0, c1],
                "roi_shape_2p4um": [r1 - r0, c1 - c0],
                "roi_physical_mm": [
                    round((r1 - r0) * PHERC0139_VOXEL_UM / 1000.0, 4),
                    round((c1 - c0) * PHERC0139_VOXEL_UM / 1000.0, 4),
                ],
                "ink_high_u8_threshold": INK_HIGH_U8_THRESHOLD,
                "blank_low_u8_threshold": BLANK_LOW_U8_THRESHOLD,
                "ink_fraction_high": round(float(ink_mask_139.mean()), 4),
                "blank_fraction_low": round(float(blank_mask_139.mean()), 4),
                "planned_sv_chunks": int(n_sv_ch_139),
                "planned_sv_mib": round(planned_b_139 / MIB, 4),
                "chunks_fetched": sv_fetched_139,
                "bytes_fetched": sv_bytes_139,
            },
            "raw_ct_surface_delta_vs_exp_t3": round(
                res_139["raw_ct_surface"]["roc_auc"] - PHERC0139_RAW_CT_REFERENCE_AUC, 5
            ),
            "diagnostics": diag_139,
            "summary": sum_139,
            "features": res_139,
        },
        "verdict": {
            "overall": overall_verdict,
            "pherc0306b_transverse": sum_306["verdict"],
            "pherc0139_transverse": sum_139["verdict"],
            "pherc0306b_passing_features": sum_306["passing_transverse_features"],
            "pherc0139_passing_features": sum_139["passing_transverse_features"],
            "pherc0306b_best_transverse": sum_306["best_transverse_feature"],
            "pherc0139_best_transverse": sum_139["best_transverse_feature"],
        },
    }
    assert_no_local_paths(receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exp T5: 2D Wavelet Scattering + Kovesi Phase Congruency + OCT 31D Depth-Waveform "
            "CPU Physics Benchmark on PHerc. 0306B (8.64 um) and PHerc. 0139 (2.399 um)."
        )
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(__file__).resolve().parent / "evidence" / "transverse_wavelet_oct_ablation.json",
        help="Destination path for sanitized JSON receipt.",
    )
    parser.add_argument("--budget-0306b-mib", type=float, default=DEFAULT_0306B_BUDGET_MIB)
    parser.add_argument("--budget-0139-mib", type=float, default=DEFAULT_0139_BUDGET_MIB)
    parser.add_argument("--global-budget-mib", type=float, default=DEFAULT_GLOBAL_BUDGET_MIB)
    args = parser.parse_args(argv)

    receipt = run_wavelet_oct_ablation(
        budget_0306b_mib=args.budget_0306b_mib,
        budget_0139_mib=args.budget_0139_mib,
        global_budget_mib=args.global_budget_mib,
    )
    assert_no_local_paths(receipt)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    b = receipt["http_byte_budget"]
    v = receipt["verdict"]
    print(
        f"[Exp T5] verdict={v['overall']} "
        f"(0306B={v['pherc0306b_transverse']}, 0139={v['pherc0139_transverse']}) | "
        f"HTTP: 0306B={b['pherc0306b_mib']:.2f} MiB, 0139={b['pherc0139_mib']:.2f} MiB, "
        f"Total={b['global_mib']:.2f} / {b['global_limit_mib']:.2f} MiB | "
        f"elapsed={receipt['elapsed_seconds']:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
