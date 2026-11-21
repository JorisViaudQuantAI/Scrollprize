#!/usr/bin/env python3
"""CPU single-line proposals for mandatory human review, never a text verdict.

Physical mark/chain construction is adapted from Liam Warner and Nicola Bodill's
MIT-licensed PHerc0306B line_find.py. See LICENSE.source and README.md. The input
scale, masks, incomplete-flank handling, bounded search and packet I/O are new.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import html
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

SOURCE_REVISION = "feabff6c54f95bd8e52a90e0a7a76881ee67d4cc"


@dataclass(frozen=True)
class Config:
    thresholds: tuple[float, ...] = (0.25, 0.5, 0.75)
    angles_deg: tuple[float, ...] = tuple(float(x) for x in range(0, 180, 5))
    work_um: float = 35.0
    link_mm: float = 0.15
    minimum_area_mm2: float = 0.06
    minimum_extent_mm: float = 0.4
    band_half_mm: float = 1.0
    band_step_mm: float = 0.25
    maximum_mark_height_mm: float = 4.0
    maximum_mark_length_mm: float = 8.0
    slot_gap_mm: float = 1.2
    maximum_pitch_mm: float = 6.5
    minimum_slots: int = 4
    minimum_length_mm: float = 8.0
    maximum_length_mm: float = 60.0
    flank_inner_mm: float = 1.75
    flank_outer_mm: float = 3.5
    minimum_flank_coverage: float = 0.5
    maximum_chains_per_threshold: int = 4000
    maximum_work_pixels: int = 8_000_000
    crop_count: int = 12

    def validate(self):
        if not self.thresholds or len(self.thresholds) > 16 or any(not math.isfinite(t) or not 0 < t < 1 for t in self.thresholds):
            raise ValueError("Use 1..16 fixed thresholds strictly inside (0,1)")
        if not self.angles_deg or len(self.angles_deg) > 180 or any(not math.isfinite(a) or not 0 <= a < 180 for a in self.angles_deg):
            raise ValueError("Use 1..180 fixed angles in [0,180)")
        if len(set(self.angles_deg)) != len(self.angles_deg) or len(set(self.thresholds)) != len(self.thresholds):
            raise ValueError("Repeated search parameters are not allowed")
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_slots < 2 or self.maximum_length_mm < self.minimum_length_mm:
            raise ValueError("Invalid line limits")
        if not self.band_half_mm < self.flank_inner_mm < self.flank_outer_mm:
            raise ValueError("Band and flanks must be separated")
        if self.minimum_flank_coverage > 1:
            raise ValueError("Flank coverage must be <=1")


def validate_inputs(score, valid, spacing_um_yx, map_kind):
    if not isinstance(score, np.ndarray) or score.ndim != 2 or min(score.shape) < 1:
        raise ValueError("Map must be a nonempty 2D NumPy array")
    if not np.issubdtype(score.dtype, np.floating):
        raise ValueError("Map must have explicit floating values in [0,1]; no implicit image scaling")
    if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_ or valid.shape != score.shape:
        raise ValueError("Validity must be a boolean array with exactly the map shape")
    if not np.isfinite(score[valid]).all() or np.any((score[valid] < 0) | (score[valid] > 1)):
        raise ValueError("Valid samples must be finite and in [0,1]")
    if len(spacing_um_yx) != 2 or any(not math.isfinite(float(s)) or float(s) <= 0 for s in spacing_um_yx):
        raise ValueError("Provide finite positive physical [dy,dx] spacing in micrometres")
    if map_kind not in ("probability", "model_score", "display_signal"):
        raise ValueError("map_kind must be probability, model_score or display_signal")


def work_grid(score, valid, spacing_um_yx, config):
    """Area mean only over valid samples; do not compact or bridge missing space."""
    dy, dx = map(float, spacing_um_yx)
    h, w = score.shape
    hh = max(1, min(h, round(h * dy / config.work_um)))
    ww = max(1, min(w, round(w * dx / config.work_um)))
    if hh * ww > config.maximum_work_pixels:
        raise ValueError("Work-grid resource limit exceeded; partition in physical coordinates before review")
    numerator = cv2.resize(np.where(valid, score, 0).astype(np.float32), (ww, hh), interpolation=cv2.INTER_AREA)
    fraction = cv2.resize(valid.astype(np.float32), (ww, hh), interpolation=cv2.INTER_AREA)
    result = np.divide(numerator, fraction, out=np.zeros_like(numerator), where=fraction > 0)
    # A mixed valid/invalid work cell cannot hide a narrow tear or mask boundary.
    observed = fraction >= 1 - 1e-6
    return result, observed, (h * dy / hh / 1000, w * dx / ww / 1000)


def extract_marks(score, valid, spacing_mm_yx, threshold, config):
    dy, dx = spacing_mm_yx
    hot = (score >= threshold) & valid
    ry, rx = max(1, round(config.link_mm / (2 * dy))), max(1, round(config.link_mm / (2 * dx)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1))
    linked = (cv2.dilate(hot.astype(np.uint8), kernel) > 0) & valid
    _, labels = cv2.connectedComponents(linked.astype(np.uint8), connectivity=8)
    yy, xx = np.nonzero(hot)
    ids = labels[yy, xx]
    order = np.argsort(ids, kind="stable")
    yy, xx, ids = yy[order], xx[order], ids[order]
    cuts = np.flatnonzero(np.diff(ids)) + 1
    marks = []
    for ys, xs in zip(np.split(yy, cuts), np.split(xx, cuts)):
        if not len(xs):
            continue
        area = len(xs) * dy * dx
        extent = max((xs.max() - xs.min() + 1) * dx, (ys.max() - ys.min() + 1) * dy)
        if area < config.minimum_area_mm2 or extent < config.minimum_extent_mm:
            continue
        weights = np.maximum(score[ys, xs] - threshold, 0.01)
        x, y = (xs + 0.5) * dx, (ys + 0.5) * dy
        marks.append(dict(x=x, y=y, cx=float(np.average(x, weights=weights)),
                          cy=float(np.average(y, weights=weights)), area=float(area)))
    return marks, hot


def extents(marks, angle):
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    bounds = []
    for m in marks:
        u, v = m["x"] * c + m["y"] * s, -m["x"] * s + m["y"] * c
        bounds.append((np.percentile(u, 2), np.percentile(u, 98), np.percentile(v, 3), np.percentile(v, 97)))
    return np.asarray(bounds, dtype=np.float64).reshape(-1, 4)


def slots(indices, along, bounds, config):
    out = [[indices[0]]]
    for index in indices[1:]:
        previous = out[-1]
        if along[index] - along[previous[-1]] < config.slot_gap_mm or bounds[index, 0] <= max(bounds[k, 1] for k in previous):
            previous.append(index)
        else:
            out.append([index])
    return out


def find_chains(marks, config):
    found = {}
    scans = 0
    if len(marks) < config.minimum_slots:
        return [], scans, False
    centers = np.array([(m["cx"], m["cy"]) for m in marks])
    areas = np.array([m["area"] for m in marks])
    truncated = False
    for angle in config.angles_deg:
        c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        bounds = extents(marks, angle)
        along = centers[:, 0] * c + centers[:, 1] * s
        across = -centers[:, 0] * s + centers[:, 1] * c
        height, length = bounds[:, 3] - bounds[:, 2], bounds[:, 1] - bounds[:, 0]
        candidates = np.flatnonzero((height <= config.maximum_mark_height_mm) & (length <= config.maximum_mark_length_mm))
        if len(candidates) < config.minimum_slots:
            continue
        start = math.floor(float(across[candidates].min()) / config.band_step_mm) * config.band_step_mm
        stop = float(across[candidates].max()) + config.band_step_mm
        for offset in np.arange(start, stop, config.band_step_mm):
            scans += 1
            chosen = candidates[np.abs(across[candidates] - offset) <= config.band_half_mm]
            if len(chosen) < config.minimum_slots:
                continue
            chosen = chosen[np.argsort(along[chosen], kind="stable")]
            grouped = slots(list(chosen), along, bounds, config)
            slot_centers = np.array([np.average(along[group], weights=areas[group]) for group in grouped])
            split = np.flatnonzero(np.diff(slot_centers) > config.maximum_pitch_mm) + 1
            for run in np.split(np.arange(len(grouped)), split):
                if len(run) < config.minimum_slots:
                    continue
                members = frozenset(i for j in run for i in grouped[j])
                idx = list(members)
                line_length = float(bounds[idx, 1].max() - bounds[idx, 0].min())
                if line_length < config.minimum_length_mm:
                    continue
                item = dict(marks=sorted(members), angle=float(angle), slot_count=len(run))
                if members not in found or found[members]["slot_count"] < len(run):
                    found[members] = item
                if len(found) >= config.maximum_chains_per_threshold:
                    truncated = True
                    return list(found.values()), scans, truncated
    return list(found.values()), scans, truncated


def chain_features(chain, marks, hot, valid, spacing_mm_yx, original_shape, original_spacing_um_yx, config):
    angle = chain["angle"]
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    bounds = extents([marks[i] for i in chain["marks"]], angle)
    area = np.array([marks[i]["area"] for i in chain["marks"]])
    cx = np.array([marks[i]["cx"] for i in chain["marks"]])
    cy = np.array([marks[i]["cy"] for i in chain["marks"]])
    across = -cx * s + cy * c
    center = float(np.average(across, weights=area))
    along = cx * c + cy * s
    u0, u1 = float(bounds[:, 0].min()), float(bounds[:, 1].max())
    length = u1 - u0
    dy, dx = spacing_mm_yx
    # Work only on a bounding rectangle around the band's full geometric flanks.
    corners = np.array([(u * c - v * s, u * s + v * c)
                        for u in (u0, u1) for v in (center - config.flank_outer_mm, center + config.flank_outer_mm)])
    h, w = valid.shape
    x0, x1 = max(0, int(np.floor(corners[:, 0].min() / dx))), min(w, int(np.ceil(corners[:, 0].max() / dx)))
    y0, y1 = max(0, int(np.floor(corners[:, 1].min() / dy))), min(h, int(np.ceil(corners[:, 1].max() / dy)))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    x, y = (xx + 0.5) * dx, (yy + 0.5) * dy
    u, v = x * c + y * s, -x * s + y * c - center
    in_length = (u >= u0) & (u <= u1)
    vv, hh = valid[y0:y1, x0:x1], hot[y0:y1, x0:x1]
    regions = {
        "band": in_length & (np.abs(v) <= config.band_half_mm),
        "upper": in_length & (v >= -config.flank_outer_mm) & (v <= -config.flank_inner_mm),
        "lower": in_length & (v >= config.flank_inner_mm) & (v <= config.flank_outer_mm),
    }
    fractions, shares = {}, {}
    for name, region in regions.items():
        observed = region & vv
        width = 2 * config.band_half_mm if name == "band" else config.flank_outer_mm - config.flank_inner_mm
        expected_area = length * width
        fractions[name] = min(1.0, float(observed.sum() * dx * dy / max(expected_area, 1e-12)))
        shares[name] = float(hh[observed].mean()) if observed.any() else None
    flank_ok = all(fractions[k] >= config.minimum_flank_coverage and shares[k] is not None for k in ("upper", "lower"))
    flank_hot = float(np.mean([shares["upper"], shares["lower"]])) if flank_ok else None
    contrast = (shares["band"] + 0.01) / (flank_hot + 0.01) if flank_ok and shares["band"] is not None else None
    residual = float(np.sqrt(np.average((across - center) ** 2, weights=area)))
    crowd = sum(u0 <= m["cx"] * c + m["cy"] * s <= u1 and
                config.flank_inner_mm <= abs(-m["cx"] * s + m["cy"] * c - center) <= config.flank_outer_mm
                for m in marks) / chain["slot_count"]
    contrast_term = math.log2(max(contrast, 0.25)) if contrast is not None else 0.0
    rank_score = contrast_term + math.log2(chain["slot_count"] / 4) - 0.5 * min(crowd, 2)
    rank_score -= 2 * max(0, 0.85 - fractions["band"]) / 0.85
    if length > config.maximum_length_mm:
        rank_score -= 1
    # A crop is axis-aligned in the ORIGINAL map and contains unmodified samples.
    crop_corners = np.array([(u * c - v * s, u * s + v * c)
                             for u in (u0 - 1, u1 + 1) for v in (center - config.flank_outer_mm, center + config.flank_outer_mm)])
    ody, odx = np.array(original_spacing_um_yx) / 1000
    oh, ow = original_shape
    bbox = [max(0, int(math.floor(crop_corners[:, 1].min() / ody))),
            max(0, int(math.floor(crop_corners[:, 0].min() / odx))),
            min(oh, int(math.ceil(crop_corners[:, 1].max() / ody))),
            min(ow, int(math.ceil(crop_corners[:, 0].max() / odx)))]
    polygon = [[float((u * c - v * s) / odx), float((u * s + v * c) / ody)]
               for u, v in [(u0, center - config.band_half_mm), (u1, center - config.band_half_mm),
                            (u1, center + config.band_half_mm), (u0, center + config.band_half_mm)]]
    warnings = []
    if not flank_ok:
        warnings.append("incomplete_flank_support_no_contrast_bonus")
    if fractions["band"] < 0.85:
        warnings.append("incomplete_band_support")
    if length > config.maximum_length_mm:
        warnings.append("long_chain")
    return dict(score=float(rank_score), angle_deg=angle, length_mm=length,
                slot_count=chain["slot_count"], mark_count=len(chain["marks"]),
                residual_mm=residual, band_coverage=fractions["band"],
                upper_flank_coverage=fractions["upper"], lower_flank_coverage=fractions["lower"],
                band_hot_fraction=shares["band"], flank_hot_fraction=flank_hot, contrast=contrast,
                crowd_per_slot=float(crowd), bbox_yxyx=bbox, band_polygon_xy=polygon, warnings=warnings)


def _box_overlap(a, b):
    y0, x0, y1, x1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, y1 - y0) * max(0, x1 - x0)
    return intersection / max(1, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def rank_map(score, valid, spacing_um_yx, map_kind, config=None):
    config = config or Config()
    config.validate()
    validate_inputs(score, valid, spacing_um_yx, map_kind)
    start = time.monotonic()
    work, observed, spacing_mm = work_grid(score, valid, spacing_um_yx, config)
    proposals, trials = [], []
    for threshold in config.thresholds:
        marks, hot = extract_marks(work, observed, spacing_mm, threshold, config)
        chains, scans, truncated = find_chains(marks, config)
        trials.append(dict(threshold=threshold, marks=len(marks), angle_band_trials=scans,
                           unique_mark_sets=len(chains), truncated=truncated))
        for chain in chains:
            result = chain_features(chain, marks, hot, observed, spacing_mm, score.shape, spacing_um_yx, config)
            result["threshold"] = threshold
            proposals.append(result)
    proposals.sort(key=lambda r: (-r["score"], -r["slot_count"], r["threshold"], r["angle_deg"], r["bbox_yxyx"]))
    kept = []
    for item in proposals:
        # De-duplication only affects packet ordering; no map disappears.
        if any(_box_overlap(item["bbox_yxyx"], old["bbox_yxyx"]) >= 0.8 and
               abs((item["angle_deg"] - old["angle_deg"] + 90) % 180 - 90) <= 15 for old in kept):
            continue
        kept.append(item)
    return dict(schema="single-line-review-v1", map_kind=map_kind, source_revision=SOURCE_REVISION,
                shape_yx=list(score.shape), spacing_um_yx=list(map(float, spacing_um_yx)),
                config=asdict(config), work_shape_yx=list(work.shape), work_spacing_mm_yx=list(spacing_mm),
                valid_area_cm2=float(valid.sum() * np.prod(spacing_um_yx) / 1e8),
                trials=trials, proposals=kept, proposal_count_before_dedup=len(proposals),
                search_truncated=any(t["truncated"] for t in trials), seconds=time.monotonic() - start,
                review_required=True, automated_text_or_blank_verdict=None,
                calibration="none; ranking score is not a probability, p-value, letter count, or prize criterion")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def outside_git(path):
    path = Path(path).resolve()
    for ancestor in [path, *path.parents]:
        if (ancestor / ".git").exists():
            raise ValueError("Generated review packets must be outside Git worktrees")


def _display(score, valid):
    pixels = np.rint(np.clip(np.where(valid, score, 0), 0, 1) * 255).astype(np.uint8)
    rgb = np.repeat(pixels[..., None], 3, axis=2)
    rgb[~valid] = (100, 30, 100)
    return Image.fromarray(rgb)


def write_packet(score, valid, report, output_dir):
    output = Path(output_dir)
    outside_git(output)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "original_map.npy", score, allow_pickle=False)
    np.save(output / "validity.npy", valid, allow_pickle=False)
    overview = _display(score, valid)
    scale = min(1.0, 1800 / max(overview.size))
    if scale < 1:
        overview = overview.resize(tuple(max(1, round(s * scale)) for s in overview.size), Image.Resampling.BOX)
    draw = ImageDraw.Draw(overview)
    cards = []
    for i, p in enumerate(report["proposals"][:report["config"]["crop_count"]], 1):
        y0, x0, y1, x1 = p["bbox_yxyx"]
        name = f"proposal_{i:03d}"
        np.save(output / (name + ".npy"), score[y0:y1, x0:x1], allow_pickle=False)
        np.save(output / (name + "_validity.npy"), valid[y0:y1, x0:x1], allow_pickle=False)
        preview = _display(score[y0:y1, x0:x1], valid[y0:y1, x0:x1])
        preview.thumbnail((1500, 600), Image.Resampling.BOX)
        preview.save(output / (name + ".png"))
        polygon = [(x * scale, y * scale) for x, y in p["band_polygon_xy"]]
        draw.line(polygon + polygon[:1], fill=(255, 60, 20), width=2)
        draw.text((x0 * scale, y0 * scale), str(i), fill=(255, 60, 20))
        cards.append(f'<section><h2>Proposal {i}: score {p["score"]:.3f}</h2><p>{p["slot_count"]} geometric slots; '
                     f'{p["length_mm"]:.2f} mm; threshold {p["threshold"]}; '
                     f'{html.escape(", ".join(p["warnings"]) or "No support warning")}</p>'
                     f'<img src="{name}.png" alt="Original-map proposal crop"></section>')
    overview.save(output / "overview.png")
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    page = ('<!doctype html><meta charset="utf-8"><title>Private single-line review</title>'
            '<style>body{max-width:1500px;margin:24px auto;font:17px system-ui}img{max-width:100%}section{margin:32px 0}</style>'
            '<h1>Every map requires review</h1><p>This packet is private. Proposals are geometric rankings, not letters, '
            'text verdicts or calibrated probabilities. A map with zero proposals is not a negative result. '
            'Purple pixels are unobserved. Previews may be reduced; inspect original_map.npy and original crop arrays '
            'at full resolution before recording review completion.</p>'
            f'<p>Declared input: {html.escape(report["map_kind"])}. No contrast stretch was added.</p>'
            '<h2>Complete map overview</h2><img src="overview.png" alt="Complete input map with proposal bands">'
            + ''.join(cards) + '<p>Review status: pending. No automated text or blank verdict.</p>')
    (output / "review.html").write_text(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--mask", type=Path, required=True)
    ap.add_argument("--spacing-um", type=float, nargs=2, required=True, metavar=("DY", "DX"))
    ap.add_argument("--map-kind", choices=["probability", "model_score", "display_signal"], required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--angles", type=float, nargs="+", default=list(range(0, 180, 5)))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    outside_git(args.output_dir)
    if args.output_dir.exists():
        ap.error("Choose a new output directory")
    if args.threads < 1:
        ap.error("Use a positive thread count")
    cv2.setNumThreads(args.threads)
    score = np.load(args.map, allow_pickle=False)
    valid = np.load(args.mask, allow_pickle=False)
    config = Config(thresholds=tuple(args.thresholds), angles_deg=tuple(args.angles))
    report = rank_map(score, valid, args.spacing_um, args.map_kind, config)
    report["input_sha256"] = sha256(args.map)
    report["mask_sha256"] = sha256(args.mask)
    report["script_sha256"] = sha256(__file__)
    write_packet(score, valid, report, args.output_dir)
    # No source paths, candidate locations, names, scores or readings in stdout.
    print(json.dumps({"review_required": True, "automated_verdict": None,
                      "proposal_count": len(report["proposals"]), "seconds": round(report["seconds"], 3),
                      "search_truncated": report["search_truncated"]}))


if __name__ == "__main__":
    main()
