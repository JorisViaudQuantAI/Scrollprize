# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Frozen numerical adapter for two PHerc0139 T5 features and their baseline.

Endpoint: agreement with the published ink-detector map, not independently
established ink truth. This module accepts no labels and performs no I/O.

Arithmetic is ported from commit
``62b2c02c6b43675131afdf642fcf5ba6221f0970``:

* ``gemini_lane/pherc0139_transverse_physics.py``: nearest filling, normalized
  Gaussian blur, and ``extract_subvoxel_height_field``;
* ``gemini_lane/transverse_wavelet_oct_physics.py``: peak-aligned sampling,
  Log-Gabor bank, the Hilbert-envelope-peak statements from
  ``compute_oct_depth_waveform_features``, and first-order scattering statements
  from ``compute_wavelet_scattering_features``.

The native PHerc0139 path uses a complete 109-plane, 2.399 micrometre surface
volume: depth Gaussian sigma 1, median-profile sheet centre, +/-24-plane peak
search, parabolic peak refinement, and an unsmoothed three-plane surface mean.
The returned peak is float32, as upstream, before the 31-point +/-15-voxel OCT
profile is sampled. Wavelets receive the surface mean and the legacy sheet mask,
with wavelengths (4, 8, 16) pixels, six orientations, and 35 micrometre averaging.

Feature values retain upstream units and signs. Apply ``POLARITIES[name]``
exactly once when calculating fixed-direction AUC; do not choose a sign from
test labels. The sole support change is a separate common evaluation mask:
legacy sheet validity AND full OCT depth bounds AND finite selected features.
It does not alter feature values or the masks used to compute those values.
The experiment's lateral border guard belongs to the evaluator, not this module.

Input contract: every supplied depth voxel has been acquired. A caller carrying
source-support metadata must pass its all-depth per-column support as
``valid_mask``; finite zero-filled missing chunks cannot be inferred here.
"""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy import signal as ssig


SOURCE_COMMIT = "62b2c02c6b43675131afdf642fcf5ba6221f0970"
SOURCE_SHA256: Mapping[str, str] = MappingProxyType({
    "gemini_lane/pherc0139_transverse_physics.py":
        "0dde2684c51e7be6076244c587be2224e0c379a00d0e8598be8b89113c8f32ef",
    "gemini_lane/transverse_wavelet_oct_physics.py":
        "88d07f2a2d3362b250ff6e5a385022f5df92ac70a0ca9cbedc1e316904a01e46",
})
POLARITIES: Mapping[str, int] = MappingProxyType({
    "raw_ct_surface": 1,
    "oct_hilbert_env_peak": 1,
    "wst_s1_scale_slope": -1,
})
EXPECTED_DEPTH = 109
NATIVE_VOXEL_UM = 2.399
OCT_WAVEFORM_DIM = 31
OCT_HALF_SPAN_VX = 15.0
WST_WAVELENGTHS_PX = (4.0, 8.0, 16.0)
WST_ORIENTATIONS = 6
WST_AVERAGING_UM = 35.0


# These three helpers preserve the original statements and evaluation order.
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
    """Exact upstream peak, height diagnostic, and surface-band preprocessing."""
    vol = np.asarray(slab, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D slab [L, H, W], got {vol.shape}")
    depth, h, w = vol.shape
    if depth < 3:
        raise ValueError(f"Slab depth must be >= 3, got {depth}")

    vm = np.ones((h, w), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).copy()
    if vm.shape != (h, w):
        raise ValueError(f"valid_mask shape {vm.shape} != {(h, w)}")

    vol_s = ndi.gaussian_filter1d(vol, sigma=float(z_smooth_sigma), axis=0, mode="nearest") if z_smooth_sigma > 0 else vol

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
    sub = vol_s[z_lo:z_hi]
    w_z = sub.shape[0]

    k_max = np.argmax(sub, axis=0)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    peak_val = sub[k_max, yy, xx].astype(np.float64)

    interior = (k_max > 0) & (k_max < w_z - 1)
    km_safe = np.clip(k_max, 1, max(1, w_z - 2))
    y_l = sub[km_safe - 1, yy, xx].astype(np.float64)
    y_c = sub[km_safe, yy, xx].astype(np.float64)
    y_r = sub[km_safe + 1, yy, xx].astype(np.float64)
    denom = y_l - 2.0 * y_c + y_r
    good_parab = interior & (denom < -1e-6) & (y_c >= y_l) & (y_c >= y_r)
    frac = np.where(good_parab, np.clip(0.5 * (y_l - y_r) / np.where(good_parab, denom, -1.0), -0.5, 0.5), 0.0)
    n_peak = (z_lo + k_max.astype(np.float64) + frac)

    sheet_valid = vm & np.isfinite(n_peak) & (peak_val > 0.0)

    sigma_detrend_px = float(detrend_sigma_um) / float(voxel_um)
    n_bg = _normalized_gaussian(n_peak, sheet_valid, sigma=sigma_detrend_px)
    h_vox = np.where(sheet_valid, n_peak - n_bg, 0.0)
    h_um = h_vox * float(voxel_um)

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


def extract_peak_aligned_depth_profiles(
    slab: np.ndarray,
    n_peak: np.ndarray,
    *,
    half_span_vx: float,
    waveform_dim: int = OCT_WAVEFORM_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact upstream linear sampling; the separate bounds mask detects clamping."""
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
    """Exact upstream Log-Gabor bank, including its frequency normalization."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    fyy, fxx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.hypot(fyy, fxx)
    radius_safe = np.where(radius > 0.0, radius, 1.0)
    theta = np.arctan2(-fyy, fxx)

    j_cnt = len(wavelengths_px)
    l_cnt = int(n_orientations)
    sigma_theta = (math.pi / float(l_cnt)) / float(dtheta_on_sigma)
    log_sig2 = 2.0 * (math.log(float(sigma_onf)) ** 2)

    radials = np.empty((j_cnt, h, w), dtype=np.float64)
    for j, wl in enumerate(wavelengths_px):
        f0 = 1.0 / float(wl)
        rad = np.exp(-((np.log(radius_safe / f0)) ** 2) / log_sig2)
        rad[0, 0] = 0.0
        lp = 1.0 / (1.0 + np.power(radius / 0.45, 16.0))
        radials[j] = rad * lp

    spreads = np.empty((l_cnt, h, w), dtype=np.float64)
    for l_idx in range(l_cnt):
        ang = l_idx * math.pi / float(l_cnt)
        ds = np.sin(theta) * math.cos(ang) - np.cos(theta) * math.sin(ang)
        dc = np.cos(theta) * math.cos(ang) + np.sin(theta) * math.sin(ang)
        dtheta = np.abs(np.arctan2(ds, dc))
        if angular_window == "raised_cosine":
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


def _wavelet_scale_slope(image: np.ndarray, valid: np.ndarray, voxel_um: float) -> np.ndarray:
    """Upstream first-order scale slope, omitting unselected S2 features."""
    img = np.asarray(image, dtype=np.float64)
    vm = np.asarray(valid, dtype=bool)
    h, w = img.shape
    if not vm.any():
        return np.zeros((h, w), dtype=np.float32)

    filled = _fill_nearest_2d(img, vm)
    mu = float(filled[vm].mean())
    sd = max(float(filled[vm].std()), 1e-6)
    normed = (filled - mu) / sd

    pad = min(32, max(4, min(h, w) - 1))
    padded = np.pad(normed, pad, mode="reflect")
    hp, wp = padded.shape
    bank = _build_log_gabor_bank_fft(
        (hp, wp), wavelengths_px=WST_WAVELENGTHS_PX,
        n_orientations=WST_ORIENTATIONS, sigma_onf=0.65,
        dtheta_on_sigma=1.2, l2_normalize=True,
    )
    j_cnt, l_cnt = bank.shape[:2]
    fft_img = np.fft.fft2(padded)
    sigma_avg_px = max(1.5, float(WST_AVERAGING_UM) / float(voxel_um))

    s1 = np.empty((j_cnt, l_cnt, h, w), dtype=np.float64)
    for j in range(j_cnt):
        for l_idx in range(l_cnt):
            resp = np.fft.ifft2(fft_img * bank[j, l_idx])
            mod = np.abs(resp)
            crop = mod[pad : pad + h, pad : pad + w]
            s1[j, l_idx] = _normalized_gaussian(crop, vm, sigma=sigma_avg_px)

    s1_scale_mean = np.mean(s1, axis=1)
    slope = np.log((s1_scale_mean[0] + 1e-5) / (s1_scale_mean[-1] + 1e-5))
    return np.where(vm, slope, 0.0).astype(np.float32)


def compute(
    stack: np.ndarray,
    voxel_um: float = NATIVE_VOXEL_UM,
    *,
    valid_mask: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Return three original feature maps, a common support mask, and diagnostics.

    ``stack`` must be a finite numeric ``[109, H, W]`` complete native surface
    volume in original CT units. ``valid_mask`` optionally excludes columns
    without all-depth source support. Feature maps are float32, while validity
    is boolean. No polarity is inferred or applied to the returned values.

    Diagnostics are JSON-serializable and include exact source hashes,
    preprocessing constants, and support counts. The returned mask has no
    lateral border erosion; apply the frozen evaluator's guard identically
    to every feature and baseline after this call.
    """
    arr = np.asarray(stack)
    if arr.ndim != 3 or arr.shape[0] != EXPECTED_DEPTH or min(arr.shape[1:]) < 2:
        raise ValueError("stack must have shape [109, H, W] with H and W >= 2")
    if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
        raise ValueError("stack must contain real numeric CT values")
    if not np.isfinite(arr).all():
        raise ValueError("stack must be finite; missing source support must be explicit")
    if not math.isclose(float(voxel_um), NATIVE_VOXEL_UM, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("this frozen PHerc0139 adapter requires voxel_um=2.399")

    vol = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(vol).all():
        raise ValueError("CT values must remain finite when converted to float32")
    h, w = vol.shape[1:]
    source_valid = np.ones((h, w), dtype=bool)
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.shape != (h, w) or mask.dtype != np.bool_:
            raise ValueError("valid_mask must be a boolean [H, W] all-depth source-support mask")
        source_valid &= mask
    # Exact native T5 precondition: sv_supp.all(axis=0) & (sv_slab.max(axis=0) > 0).
    input_valid = source_valid & (vol.max(axis=0) > 0.0)
    hf = extract_subvoxel_height_field(
        vol, voxel_um=voxel_um, valid_mask=input_valid,
        search_half_slices=24, z_smooth_sigma=1.0,
        detrend_sigma_um=200.0, band_half_width=1,
    )
    sheet_valid = hf["sheet_valid"]
    profiles, offsets, in_bounds = extract_peak_aligned_depth_profiles(
        vol, hf["n_peak"], half_span_vx=OCT_HALF_SPAN_VX,
        waveform_dim=OCT_WAVEFORM_DIM,
    )
    # Exact selected statements from upstream compute_oct_depth_waveform_features.
    p64 = profiles.astype(np.float64)
    del profiles
    p_mean = p64.mean(axis=0, keepdims=True)
    p_zero = p64 - p_mean
    del p64, p_mean
    z_shifted = ssig.hilbert(np.fft.ifftshift(p_zero, axes=0), axis=0)
    del p_zero
    z_analytic = np.fft.fftshift(z_shifted, axes=0)
    del z_shifted
    env = np.abs(z_analytic)
    del z_analytic
    hilbert_env_peak = env[OCT_WAVEFORM_DIM // 2].astype(np.float32)
    del env

    features = {
        "raw_ct_surface": hf["surface_ct"].astype(np.float32),
        "oct_hilbert_env_peak": np.where(sheet_valid, hilbert_env_peak, 0.0).astype(np.float32),
        "wst_s1_scale_slope": _wavelet_scale_slope(hf["surface_ct"], sheet_valid, voxel_um),
    }
    feature_finite = np.logical_and.reduce([np.isfinite(v) for v in features.values()])
    # This new mask is deliberately applied AFTER computing every legacy value.
    common_valid = sheet_valid & in_bounds & feature_finite
    n_sheet = int(sheet_valid.sum())
    diagnostics = {
        "source_commit": SOURCE_COMMIT,
        "source_sha256": dict(SOURCE_SHA256),
        "endpoint": "agreement_with_published_detector_map_not_independent_ink_truth",
        "feature_values": "original_unoriented_values",
        "fixed_polarities": dict(POLARITIES),
        "input_shape": list(vol.shape),
        "voxel_um": float(voxel_um),
        "height_field": {
            "z_center_slice": int(hf["z_center_slice"]),
            "z_search_window": hf["z_search_window"],
            "search_half_slices": 24,
            "z_smooth_sigma": 1.0,
            "detrend_sigma_um": 200.0,
            "band_half_width": 1,
            "height_field_std_um": float(np.std(hf["h_um"][sheet_valid])) if n_sheet else 0.0,
        },
        "oct": {
            "waveform_dim": OCT_WAVEFORM_DIM,
            "half_span_vx": OCT_HALF_SPAN_VX,
            "half_span_um": float(OCT_HALF_SPAN_VX * voxel_um),
            "step_vx": float(offsets[1] - offsets[0]),
        },
        "wavelet": {
            "wavelengths_px": list(WST_WAVELENGTHS_PX),
            "wavelengths_um": [float(v * voxel_um) for v in WST_WAVELENGTHS_PX],
            "n_orientations": WST_ORIENTATIONS,
            "avg_sigma_um": WST_AVERAGING_UM,
            "avg_sigma_px": max(1.5, WST_AVERAGING_UM / float(voxel_um)),
        },
        "support": {
            "total_columns": h * w,
            "source_valid_columns": int(source_valid.sum()),
            "input_nonzero_valid_columns": int(input_valid.sum()),
            "legacy_sheet_valid_columns": n_sheet,
            "sheet_valid_but_profile_out_of_bounds": int((sheet_valid & ~in_bounds).sum()),
            "sheet_valid_but_nonfinite_feature": int((sheet_valid & ~feature_finite).sum()),
            "common_valid_columns": int(common_valid.sum()),
            "fully_in_bounds_fraction_of_sheet": float(in_bounds[sheet_valid].mean()) if n_sheet else 0.0,
            "common_valid_fraction_of_sheet": float(common_valid.sum() / n_sheet) if n_sheet else 0.0,
            "lateral_border_guard_px": 0,
        },
    }
    return features, common_valid, diagnostics
