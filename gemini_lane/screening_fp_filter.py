# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""CPU-only false-positive diagnostics for First Letters screening probability maps.

Targets the screening false-positive modes seen on the RTX 4060 / Colab lanes:

* crack-edge ``Studio > 0.7`` specks on delamination crack walls / air voids and
  mesh-border erosion halos (``PHerc0191``) -> :func:`build_crack_void_halo_mask`;
* amorphous ``2-4 mm`` bilateral mottle clouds (both depth orders fire) and
  single-face ``1 mm`` dot rows -> :func:`score_probability_map_diagnostics`
  (bilateral overlap, opposite-face asymmetry, component strokeness and
  Studio / A-hn two-family agreement).

Pure NumPy + SciPy; never imports torch or CUDA.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi

DEFAULT_VOXEL_UM = 9.362
FACES = ("asc", "desc")
VOID_MODES = ("robust_sigma", "physical", "hybrid")
DEFAULT_DARK_FRAC_OF_MEDIAN = 0.5


@dataclass(frozen=True)
class FPThresholds:
    """Decision thresholds turning diagnostics into boolean false-positive flags."""

    hi: float = 0.7
    lo: float = 0.5
    max_halo_share: float = 0.5
    max_bilateral_overlap: float = 0.3
    min_strokeness: float = 1.5
    min_asymmetry: float = 3.0
    min_two_family_iou: float = 0.1


def _disk(radius: int) -> np.ndarray:
    """Return a boolean disk structuring element of the given pixel radius."""
    r = int(max(radius, 0))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (yy * yy + xx * xx) <= r * r


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a disk; identity for ``radius <= 0``."""
    if radius <= 0:
        return mask.copy()
    return ndi.binary_dilation(mask, structure=_disk(radius))


def _masked_smooth(img: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized Gaussian smoothing so invalid pixels never bleed into valid ones."""
    if sigma <= 0:
        return img
    w = valid.astype(np.float32)
    num = ndi.gaussian_filter(img * w, sigma)
    den = ndi.gaussian_filter(w, sigma)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)


def void_threshold(
    vals: np.ndarray,
    *,
    void_mode: str = "robust_sigma",
    dark_sigma: float = 4.0,
    abs_dark_floor: float | None = None,
    dark_frac_of_median: float = DEFAULT_DARK_FRAC_OF_MEDIAN,
) -> float:
    """Automatic dark (void) threshold for valid CT values under ``void_mode`` (see the halo mask)."""
    if void_mode not in VOID_MODES:
        raise ValueError(f"void_mode must be one of {VOID_MODES}, got {void_mode!r}")
    vals = np.asarray(vals, dtype=np.float32)
    med = float(np.median(vals))
    robust = med - dark_sigma * 1.4826 * float(np.median(np.abs(vals - med)))
    if void_mode == "robust_sigma":
        return robust
    physical = float(abs_dark_floor) if abs_dark_floor is not None else float(dark_frac_of_median) * med
    return physical if void_mode == "physical" else max(robust, physical)


def build_crack_void_halo_mask(
    ct_mid: np.ndarray,
    valid_mask: np.ndarray,
    *,
    dark_threshold: float | None = None,
    dark_percentile: float | None = None,
    dark_sigma: float = 4.0,
    smooth_sigma: float = 1.0,
    close_radius: int = 2,
    min_void_px: int = 40,
    halo_radius: int = 8,
    border_halo_radius: int = 6,
    void_mode: str = "robust_sigma",
    abs_dark_floor: float | None = None,
    dark_frac_of_median: float = DEFAULT_DARK_FRAC_OF_MEDIAN,
) -> np.ndarray:
    """Flag delamination cracks / air voids plus their wall halo and mesh-border erosion halo.

    Dark pixels are those of the (masked-smoothed) CT mid-slice below ``dark_threshold``;
    if unset, below ``percentile(dark_percentile)`` when given, else below the robust
    ``median - dark_sigma * 1.4826 * MAD`` of valid pixels (so intact woven papyrus,
    whose weave contrast stays within a few sigma, is left clean). Dark pixels are
    closed by ``close_radius``, components smaller than ``min_void_px`` are dropped,
    and the survivors are dilated by ``halo_radius``. Invalid (unrendered) pixels are
    never treated as voids, but valid pixels within ``border_halo_radius`` of them are
    flagged as erosion halo. Returns a boolean mask restricted to ``valid_mask``.

    ``void_mode`` selects the automatic rule used when neither ``dark_threshold`` nor
    ``dark_percentile`` is given (both explicit options still take precedence):

    * ``"robust_sigma"`` (default, backward compatible): ``median - dark_sigma * robust_sigma``.
      Contrast-adaptive, but on real CT papyrus the weave / fibre texture inflates the robust
      sigma (``~27-37`` uint8 levels on PHerc0306B / PHerc0191 for a ``~100`` median), which
      pushes the threshold to or below zero so true air cracks are never flagged.
    * ``"physical"``: an absolute air/crack floor in CT intensity units, ``abs_dark_floor`` when
      given, else ``dark_frac_of_median * median``. Calibration rationale: inside a masked CT
      volume empty space reads near ``0`` and compact papyrus near the valid-area median, so
      under linear partial-volume mixing a smoothed pixel below ``f * median`` is at least
      ``(1 - f)`` air; the default ``f = 0.5`` flags pixels that are majority air. It ignores
      weave contrast entirely, so it fires on real cracks but also on any low-density sheet
      region (thin / worn papyrus, inter-sheet gaps the mesh grazes), trading clean-papyrus
      retention for void recall. Fixed a priori; not tuned on benchmark strips. Real-CT receipt
      (``evidence/fp_filter_real_ct_benchmark.json``): at ``f = 0.5`` it fires on every PHerc0191
      w050/w030 window but also removes ~37% of intact PHerc0306B papyrus (whose dark inter-fibre
      gaps put ~9-14% of pixels below half the median), so it is NOT a weave-vs-void discriminator
      on its own; lower ``f`` (e.g. ``0.35``) or pair it with geometry/m7 evidence before use.
    * ``"hybrid"``: ``max(robust_sigma, physical)``, i.e. the physical floor rescues the robust
      rule when sigma collapses the threshold, and the robust rule takes over on unusually
      low-contrast scans where it is the stricter (higher) bound.
    """
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != np.shape(ct_mid):
        raise ValueError("ct_mid and valid_mask shapes differ")
    out = np.zeros_like(valid)
    if not valid.any():
        return out
    ct = _masked_smooth(np.asarray(ct_mid, dtype=np.float32), valid, smooth_sigma)
    vals = ct[valid]
    if dark_threshold is not None:
        thr = float(dark_threshold)
    elif dark_percentile is not None:
        thr = float(np.percentile(vals, dark_percentile))
    else:
        thr = void_threshold(
            vals,
            void_mode=void_mode,
            dark_sigma=dark_sigma,
            abs_dark_floor=abs_dark_floor,
            dark_frac_of_median=dark_frac_of_median,
        )
    dark = (ct < thr) & valid
    if close_radius > 0:
        # OR keeps closing extensive at the image edge (scipy erodes with border_value=0).
        dark = (dark | ndi.binary_closing(dark, structure=_disk(close_radius))) & valid
    labels, n = ndi.label(dark)
    if n:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_void_px
        keep[0] = False
        out |= _dilate(keep[labels], halo_radius)
    if border_halo_radius > 0 and not valid.all():
        out |= _dilate(~valid, border_halo_radius)
    return out & valid


def _zhang_suen_thin(mask: np.ndarray) -> np.ndarray:
    """Vectorized Zhang-Suen thinning of a boolean mask to a 1-px skeleton."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            c = img[1:-1, 1:-1]
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            ring = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            b = sum(p.astype(np.int16) for p in ring[:8])
            a = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                cond = ((p2 & p4 & p6) == 0) & ((p4 & p6 & p8) == 0)
            else:
                cond = ((p2 & p4 & p8) == 0) & ((p2 & p6 & p8) == 0)
            rm = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if rm.any():
                c[rm] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def component_strokeness(
    hits: np.ndarray,
    *,
    voxel_um: float = DEFAULT_VOXEL_UM,
    top_k: int = 5,
    min_component_px: int = 30,
    open_radius: int = 1,
) -> dict:
    """Score the ``top_k`` largest hit components by ``skeleton_length / equiv_diameter``.

    Crisp thin strokes score well above 1 (long skeleton, small area); compact clouds
    and dots score near 0. ``open_radius`` removes 1-px boundary jitter before thinning.
    """
    labels, n = ndi.label(hits)
    comps = []
    if n:
        sizes = np.bincount(labels.ravel())[1:]
        order = [i for i in np.argsort(sizes)[::-1] if sizes[i] >= min_component_px][:top_k]
        slices = ndi.find_objects(labels)
        mm_per_px = voxel_um / 1000.0
        for i in order:
            sl = slices[i]
            comp = labels[sl] == (i + 1)
            if open_radius > 0:
                opened = ndi.binary_opening(comp, structure=_disk(open_radius))
                comp = opened if opened.any() else comp
            area = int(comp.sum())
            eqd = float(np.sqrt(4.0 * area / np.pi))
            skel = int(_zhang_suen_thin(comp).sum())
            comps.append(
                {
                    "area_px": area,
                    "skeleton_px": skel,
                    "equiv_diameter_mm": round(eqd * mm_per_px, 4),
                    "extent_mm": round(max(comp.shape) * mm_per_px, 4),
                    "strokeness": round(skel / max(eqd, 1e-6), 4),
                }
            )
    if not comps:
        return {"n_components": 0, "weighted_mean": None, "median": None, "top_components": []}
    s = np.array([c["strokeness"] for c in comps])
    w = np.array([c["area_px"] for c in comps], dtype=np.float64)
    return {
        "n_components": int(n),
        "weighted_mean": round(float((s * w).sum() / w.sum()), 4),
        "median": round(float(np.median(s)), 4),
        "top_components": comps,
    }


def _face_hit_shares(prob: np.ndarray, clean: np.ndarray, halo: np.ndarray, th: FPThresholds) -> dict:
    """Clean-papyrus vs crack-halo hit counts / shares at the ``hi`` and ``lo`` thresholds."""
    out = {}
    for tag, t in (("gt07", th.hi), ("gt05", th.lo)):
        hit = prob > t
        n_clean, n_halo = int((hit & clean).sum()), int((hit & halo).sum())
        out[f"clean_{tag}"] = n_clean
        out[f"halo_{tag}"] = n_halo
        out[f"halo_share_{tag}"] = round(n_halo / max(n_clean + n_halo, 1), 6)
    return out


def _dominant_face(maps: dict[str, np.ndarray], active: np.ndarray, hi: float) -> tuple[str, str, dict[str, int]]:
    """Return (dominant, opposite, hit counts) for an ``{asc, desc}`` map pair on ``active``."""
    counts = {f: int(((maps[f] > hi) & active).sum()) for f in FACES}
    dom = max(FACES, key=lambda f: counts[f])
    return dom, FACES[1 - FACES.index(dom)], counts


def _two_family_agreement(
    studio_hits: np.ndarray,
    studio_dom: str,
    ahn: dict[str, np.ndarray],
    active: np.ndarray,
    hi: float,
    tolerance_px: int,
) -> dict:
    """IoU / overlap between Studio and A-hn dominant-face hit masks."""
    ahn_dom, _, counts = _dominant_face(ahn, active, hi)
    ahn_hits = (ahn[ahn_dom] > hi) & active
    inter = int((studio_hits & ahn_hits).sum())
    union = int((studio_hits | ahn_hits).sum())
    covered = int((studio_hits & _dilate(ahn_hits, tolerance_px)).sum())
    return {
        "ahn_dominant_face": ahn_dom,
        "ahn_hits": counts,
        "face_agreement": ahn_dom == studio_dom,
        "iou": round(inter / max(union, 1), 6),
        "studio_covered_by_ahn": round(covered / max(int(studio_hits.sum()), 1), 6),
    }


def _regional_face_dominance(
    hits_asc: np.ndarray,
    hits_desc: np.ndarray,
    active: np.ndarray,
    *,
    tile_px: int = 512,
) -> dict:
    """Measure local depth-order dominance across half-maps and ``tile_px`` regions.

    Prevents global asymmetry heuristics from suppressing multi-wrap windings where
    separate halves carry crisp single-sided text on opposite depth orders.
    """
    h, w = active.shape
    regions: list[tuple[slice, slice]] = []
    if w >= 2:
        mid_w = w // 2
        regions.extend([(slice(0, h), slice(0, mid_w)), (slice(0, h), slice(mid_w, w))])
    if h >= 2:
        mid_h = h // 2
        regions.extend([(slice(0, mid_h), slice(0, w)), (slice(mid_h, h), slice(0, w))])
    step = max(int(tile_px), 64)
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            regions.append((slice(y0, min(h, y0 + step)), slice(x0, min(w, x0 + step))))

    max_asym = 1.0
    max_dom_hits = 0
    has_asc_region = False
    has_desc_region = False
    for sy, sx in regions:
        if int(active[sy, sx].sum()) < 256:
            continue
        c_asc = int(hits_asc[sy, sx].sum())
        c_desc = int(hits_desc[sy, sx].sum())
        dom_c = max(c_asc, c_desc)
        opp_c = min(c_asc, c_desc)
        ratio = dom_c / max(opp_c, 1)
        if dom_c > 0 and (ratio > max_asym or (ratio == max_asym and dom_c > max_dom_hits)):
            max_asym = ratio
            max_dom_hits = dom_c
        if c_asc >= 100 and c_asc >= 3 * max(c_desc, 1):
            has_asc_region = True
        if c_desc >= 100 and c_desc >= 3 * max(c_asc, 1):
            has_desc_region = True

    return {
        "tile_px": int(tile_px),
        "max_regional_asymmetry": round(float(max_asym), 4),
        "max_regional_dom_hits": int(max_dom_hits),
        "has_opposite_order_regions": bool(has_asc_region and has_desc_region),
    }


def score_probability_map_diagnostics(
    prob_asc: np.ndarray,
    prob_desc: np.ndarray,
    valid_mask: np.ndarray,
    crack_halo_mask: np.ndarray | None = None,
    ahn_asc: np.ndarray | None = None,
    ahn_desc: np.ndarray | None = None,
    voxel_um: float = DEFAULT_VOXEL_UM,
    *,
    thresholds: FPThresholds = FPThresholds(),
    top_k: int = 5,
    min_component_px: int = 30,
    agreement_tolerance_px: int = 3,
) -> dict:
    """Compute false-positive diagnostics for a Studio ``asc``/``desc`` depth-order map pair.

    Returns a JSON-safe dict with per-face clean-papyrus vs crack-halo hit shares,
    the dominant face and ``dominant / opposite`` asymmetry of ``> hi`` hits, bilateral
    overlap (``asc > hi & desc > hi`` over their union and over active area), regional
    alternating-order dominance, dominant face component strokeness, optional Studio / A-hn
    two-family agreement, and boolean ``flags`` plus an ``ink_like_candidate`` verdict.
    ``active`` = valid minus crack halo.
    """
    valid = np.asarray(valid_mask, dtype=bool)
    maps = {"asc": np.asarray(prob_asc, dtype=np.float32), "desc": np.asarray(prob_desc, dtype=np.float32)}
    if any(m.shape != valid.shape for m in maps.values()):
        raise ValueError("probability maps and valid_mask shapes differ")
    halo = valid & (np.zeros_like(valid) if crack_halo_mask is None else np.asarray(crack_halo_mask, dtype=bool))
    active = valid & ~halo
    th = thresholds
    n_valid = int(valid.sum())
    px_cm2 = (voxel_um / 10000.0) ** 2

    dom, opp, counts = _dominant_face(maps, active, th.hi)
    hits = {f: (maps[f] > th.hi) & active for f in FACES}
    both, either = hits["asc"] & hits["desc"], hits["asc"] | hits["desc"]
    stroke = component_strokeness(hits[dom], voxel_um=voxel_um, top_k=top_k, min_component_px=min_component_px)

    two_family = None
    if (ahn_asc is None) != (ahn_desc is None):
        raise ValueError("supply both ahn_asc and ahn_desc, or neither")
    if ahn_asc is not None:
        ahn = {"asc": np.asarray(ahn_asc, dtype=np.float32), "desc": np.asarray(ahn_desc, dtype=np.float32)}
        two_family = _two_family_agreement(hits[dom], dom, ahn, active, th.hi, agreement_tolerance_px)

    faces = {f: _face_hit_shares(maps[f], active, halo, th) for f in FACES}
    asymmetry = counts[dom] / max(counts[opp], 1)
    bilateral = int(both.sum()) / max(int(either.sum()), 1)
    regional = _regional_face_dominance(hits["asc"], hits["desc"], active)
    no_hits = counts[dom] == 0
    regional_rescue = (
        bilateral < 0.05
        and regional["max_regional_asymmetry"] >= th.min_asymmetry
        and regional["max_regional_dom_hits"] >= 500
    )
    flags = {
        "no_hits": no_hits,
        "crack_halo_dominated": faces[dom]["halo_share_gt07"] > th.max_halo_share,
        "bilateral_mottle": bilateral >= th.max_bilateral_overlap,
        "low_asymmetry": (asymmetry < th.min_asymmetry) and not regional_rescue,
        "low_strokeness": stroke["weighted_mean"] is not None and stroke["weighted_mean"] < th.min_strokeness,
        "two_family_disagree": two_family is not None
        and (two_family["iou"] < th.min_two_family_iou or not two_family["face_agreement"]),
    }
    return {
        "voxel_um": voxel_um,
        "valid_cm2": round(n_valid * px_cm2, 6),
        "clean_papyrus_area_frac": round(int(active.sum()) / max(n_valid, 1), 6),
        "crack_halo_area_frac": round(int(halo.sum()) / max(n_valid, 1), 6),
        "faces": faces,
        "dominant_face": dom,
        "asymmetry_ratio": round(float(asymmetry), 4),
        "bilateral_overlap_frac": round(float(bilateral), 6),
        "bilateral_overlap_area_frac": round(int(both.sum()) / max(int(active.sum()), 1), 6),
        "regional_asymmetry": regional,
        "strokeness": stroke,
        "two_family": two_family,
        "thresholds": asdict(th),
        "flags": flags,
        "ink_like_candidate": not any(flags.values()),
    }
