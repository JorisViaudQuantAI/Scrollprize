#!/usr/bin/env python3
"""Retain both published PHerc. 306B row contexts at one physical display scale.

This diagnostic is restricted to the published reference frame. It does not
run an ink model or establish that two declared tile origins were really used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "line_review"))
from review import Config, outside_git, rank_map, validate_inputs

PROTOCOL = json.loads((HERE / "protocol.json").read_text())
SHA = re.compile(r"[0-9a-f]{64}\Z")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference_case():
    path = HERE.parent / "line_controls/inventory.json"
    if digest(path) != PROTOCOL["source_inventory_sha256"]:
        raise ValueError("Reference inventory differs from the source-pinned protocol")
    return next(c for c in json.loads(path.read_text())["cases"] if c["id"] == PROTOCOL["source_case"])


def local_targets(reference, field_box_yxyx):
    """Translate published boxes, without clipping their original extent away."""
    y0, x0, _, _ = field_box_yxyx
    return [dict(t, box_xyxy=[t["box_xyxy"][0]-x0, t["box_xyxy"][1]-y0,
                             t["box_xyxy"][2]-x0, t["box_xyxy"][3]-y0])
            for t in reference["targets"]]


def validate_manifest(manifest, reference):
    if manifest.get("reference_case") != reference["id"]:
        raise ValueError("This tool only supports the pinned published 306B frame")
    box = manifest.get("field_box_yxyx")
    if not isinstance(box, list) or len(box) != 4 or not all(type(v) is int for v in box):
        raise ValueError("Provide integer field_box_yxyx in the prepared reference grid")
    y0, x0, y1, x1 = box
    if not (0 <= y0 < y1 <= reference["shape"][0] and 0 <= x0 < x1 <= reference["shape"][1]):
        raise ValueError("Field lies outside the pinned reference")
    rows = manifest.get("maps")
    if not isinstance(rows, list) or len(rows) not in (1, 2):
        raise ValueError("Provide one published reference map or two genuine grid outputs")
    for row in rows:
        if not isinstance(row, dict) or not all(k in row for k in ("map", "mask", "map_sha256", "mask_sha256")):
            raise ValueError("Every map needs explicit score/mask paths and hashes")
        if not all(isinstance(row[k], str) and SHA.fullmatch(row[k]) for k in ("map_sha256", "mask_sha256")):
            raise ValueError("Map/mask SHA-256 values must be explicit")
    if len(rows) == 1:
        if manifest.get("experiment_kind") != "single_published_reference":
            raise ValueError("One map is a descriptive reference, never a phase comparison")
        if box != [0, 0, *reference["shape"]] or rows[0]["map_sha256"] != reference["score_sha256"] or rows[0]["mask_sha256"] != reference["mask_sha256"]:
            raise ValueError("Single-map mode requires the exact complete released control")
        return
    if manifest.get("experiment_kind") != "tile_grid_pair":
        raise ValueError("Two maps require a declared tile_grid_pair experiment")
    required = ("model", "source_frame", "depth_order", "preprocess", "fusion", "precision", "tta", "tile_shape_yx_px", "stride_yx_px", "coverage_policy")
    signatures = [row.get("run_signature") for row in rows]
    if any(not isinstance(s, dict) or not all(k in s and s[k] not in (None, "", {}, []) for k in required) for s in signatures):
        raise ValueError("Both runs need complete inference signatures")
    if signatures[0] != signatures[1]:
        raise ValueError("A phase-only comparison must match all shared inference settings")
    for key in ("tile_shape_yx_px", "stride_yx_px"):
        value = signatures[0][key]
        if not isinstance(value, list) or len(value) != 2 or not all(type(v) is int and v > 0 for v in value):
            raise ValueError("Tile shape and stride must be positive integer YX pairs")
    phases = [row.get("tile_origin_yx_px") for row in rows]
    if any(not isinstance(p, list) or len(p) != 2 or not all(type(v) is int for v in p) for p in phases):
        raise ValueError("Declare the two actual tile origins in the same native raster")
    strides = signatures[0]["stride_yx_px"]
    if all((phases[0][i]-phases[1][i]) % strides[i] == 0 for i in range(2)):
        raise ValueError("Tile origins have identical residues at the declared stride")


def row_metrics(proposals, targets, valid):
    """Rasterize on the original grid; missing source extent stays explicit."""
    h, w = valid.shape
    result, descriptors = [], []
    for target in targets:
        x0, y0, x1, y1 = target["box_xyxy"]
        source_area = (int(np.ceil(x1))-int(np.floor(x0))) * (int(np.ceil(y1))-int(np.floor(y0)))
        source_width = int(np.ceil(x1))-int(np.floor(x0))
        ax0, ay0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        ax1, ay1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
        intersects = ax0 < ax1 and ay0 < ay1
        cy = int(round((y0+y1)/2))
        centre_inside = intersects and 0 <= cy < h
        box_support = valid[ay0:ay1, ax0:ax1] if intersects else None
        n = int(box_support.sum()) if intersects else 0
        line_n = int(valid[cy, ax0:ax1].sum()) if centre_inside else 0
        record = dict(id=target["id"], source_box_pixels=source_area,
                      source_centreline_pixels=source_width,
                      source_box_support_fraction=n/source_area,
                      source_centreline_support_fraction=line_n/source_width,
                      status="source_target_outside_field" if not intersects else
                             "observed" if n else "source_target_unobserved",
                      top_k={}, individual=[])
        result.append(record)
        descriptors.append((ax0,ay0,ax1,ay1,cy,intersects,n,line_n))

    def coverage(union, descriptor, identifier):
        ax0,ay0,ax1,ay1,cy,intersects,n,line_n = descriptor
        n_covered = int(((union[ay0:ay1,ax0:ax1] > 0) & valid[ay0:ay1,ax0:ax1]).sum()) if n else 0
        line_covered = int(((union[cy,ax0:ax1] > 0) & valid[cy,ax0:ax1]).sum()) if line_n else 0
        return dict(id=identifier, supported_box_pixels=n,
                    box_support_fraction=n/((ax1-ax0)*(ay1-ay0)) if intersects else None,
                    box_coverage=n_covered/n if n else None,
                    centerline_support_pixels=line_n,
                    centerline_coverage=line_covered/line_n if line_n else None,
                    omitted_observed_centreline=(line_covered == 0) if line_n else None)

    # Cropping before polygon rasterization subtly changes OpenCV's clipping
    # and boundary pixels. Use the complete original field, just as the frozen
    # earlier evaluation does. Retain only two raster buffers, not one per band.
    individual = np.zeros(valid.shape, np.uint8)
    union = np.zeros(valid.shape, np.uint8)
    for rank, proposal in enumerate(proposals, 1):
        polygon = np.rint(proposal["band_polygon_xy"]).astype(np.int32)
        individual.fill(0)
        cv2.fillPoly(individual, [polygon], 1)
        cv2.fillPoly(union, [polygon], 1)
        for record, descriptor in zip(result, descriptors):
            value = coverage(individual, descriptor, record["id"])
            record["individual"].append(dict(rank=rank, box_coverage=value["box_coverage"],
                                              centerline_coverage=value["centerline_coverage"]))
            if rank in PROTOCOL["top_k"]:
                record["top_k"][str(rank)] = coverage(union, descriptor, record["id"])
    for record, descriptor in zip(result, descriptors):
        for k in PROTOCOL["top_k"]:
            if str(k) not in record["top_k"]:
                record["top_k"][str(k)] = coverage(union, descriptor, record["id"])
        record["top_k"] = {str(k):record["top_k"][str(k)] for k in PROTOCOL["top_k"]}
        measured = [(v["rank"], v["centerline_coverage"]) for v in record["individual"] if v["centerline_coverage"] is not None]
        record["first_centreline_overlap_rank"] = next((r for r, v in measured if v > 0), None)
        record["first_half_centreline_rank"] = next((r for r, v in measured if v >= .5), None)
        best = max((v for _, v in measured), default=0)
        record["greatest_centreline_overlap_rank"] = next((r for r, v in measured if v == best and v > 0), None)
        record["greatest_individual_centreline_coverage"] = best if measured else None
    return result


def plot_contexts(scores, valid, reports, reference, box, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    dy, dx = np.asarray(reference["spacing_um_yx"]) / 1000
    full_h, full_w = reference["shape"]
    y0, x0, y1, x1 = box
    height_mm, width_mm = full_h*dy, full_w*dx
    half = PROTOCOL["fixed_context_half_height_mm"]
    fig, axes = plt.subplots(3, len(scores), squeeze=False, dpi=120,
                             figsize=(12*len(scores), 11.5),
                             gridspec_kw={"height_ratios": [height_mm, 2*half, 2*half]},
                             constrained_layout=True)
    for col, score in enumerate(scores):
        sample_w = min(1600, score.shape[1])
        size = (sample_w, max(1, round(score.shape[0] * sample_w / score.shape[1])))
        preview = cv2.resize(np.where(valid, score, 0).astype(np.float32), size, interpolation=cv2.INTER_AREA)
        support = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        rgb = np.repeat(np.rint(np.clip(preview, 0, 1)*255).astype(np.uint8)[:, :, None], 3, axis=2)
        rgb[~support] = (76, 35, 85)
        for row, ax in enumerate(axes[:, col]):
            ax.set_facecolor("#4c2355")
            ax.imshow(rgb, extent=(x0*dx, x1*dx, y1*dy, y0*dy), interpolation="nearest")
            ax.set_xlim(0, width_mm)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Reference chart x (mm)")
            ax.set_ylabel("Reference chart y (mm)")
            if row == 0:
                ax.set_ylim(height_mm, 0)
                ax.set_title(f"Map {chr(65+col)}: full reference field; first three proposal bands", loc="left")
                for target in reference["targets"]:
                    xa, ya, xb, yb = target["box_xyxy"]
                    ax.add_patch(Rectangle((xa*dx, ya*dy), (xb-xa)*dx, (yb-ya)*dy,
                                           fill=False, edgecolor="#28d7df", linewidth=.8, linestyle="--"))
                for rank, proposal in enumerate(reports[col]["proposals"][:PROTOCOL["figure_top_k"]], 1):
                    polygon = (np.asarray(proposal["band_polygon_xy"])+[x0, y0])*[dx, dy]
                    ax.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor="#ff8b35", linewidth=1.1))
                    x, y = polygon.mean(axis=0)
                    ax.text(x, y, str(rank), color="white", fontsize=9,
                            bbox={"facecolor":"#333333", "alpha":.8, "edgecolor":"none", "pad":1})
                scale_y = height_mm-1.3
                ax.plot([2,7], [scale_y,scale_y], color="white", linewidth=3)
                ax.text(4.5, scale_y-.5, "5 mm", color="white", ha="center", fontsize=9,
                        bbox={"facecolor":"#333333", "alpha":.8, "edgecolor":"none", "pad":1})
            else:
                target = reference["targets"][row-1]
                centre = (target["box_xyxy"][1]+target["box_xyxy"][3])*dy/2
                ax.set_ylim(centre+half, centre-half)
                ax.set_title(f"Map {chr(65+col)}: published row {row}, fixed ±{half:g} mm context; no proposal overlays", loc="left", fontsize=10)
        # Identical physical axes and width retain the same millimetre scale in
        # the row strips; a low-ranked or absent row is never zoomed away.
    fig.suptitle("Published PHerc. 306B — both row contexts retained regardless of rank\n"
                 "One [0,1] display scale; purple = unobserved; axes describe the published chart. Localization only, not letter recognition.", fontsize=11)
    fig.canvas.draw()
    scales = [np.abs(ax.transData.transform((1,1))-ax.transData.transform((0,0))).tolist() for ax in axes.flat]
    if not np.allclose(scales, scales[0], rtol=1e-6, atol=1e-6):
        raise RuntimeError("Figure panels do not share the required physical scale")
    fig.savefig(output, dpi="figure", metadata={"Title":"Published 306B fixed row contexts"})
    plt.close(fig)
    return dict(pixels_per_mm_xy=scales, all_panels_same_physical_scale=True,
                preview_only=True, supplied_score_limits=[0,1])


def run(manifest, base_dir, output_dir):
    reference = reference_case()
    validate_manifest(manifest, reference)
    output = Path(output_dir).resolve()
    outside_git(output)
    if output.exists():
        raise FileExistsError("Use a new private output directory")
    box = manifest["field_box_yxyx"]
    expected_shape = (box[2]-box[0], box[3]-box[1])
    scores, masks, paths = [], [], []
    for record in manifest["maps"]:
        pair = [Path(base_dir) / record[key] for key in ("map", "mask")]
        if any(digest(path) != record[key] for path, key in zip(pair, ("map_sha256", "mask_sha256"))):
            raise ValueError("An input does not match its declared hash")
        score, mask = [np.load(path, allow_pickle=False) for path in pair]
        validate_inputs(score, mask, reference["spacing_um_yx"], "model_score")
        if score.shape != expected_shape:
            raise ValueError("Input shape differs from its declared reference field")
        scores.append(score); masks.append(mask); paths.append(pair)
    common = np.logical_and.reduce(masks)
    if not common.any():
        raise ValueError("No common observed pixels")
    targets = local_targets(reference, box)
    config = Config()
    cv2.setNumThreads(4)
    reports = [rank_map(s, common, reference["spacing_um_yx"], "model_score", config) for s in scores]
    rows = [row_metrics(r["proposals"], targets, common) for r in reports]
    for record, pair in zip(manifest["maps"], paths):
        if any(digest(path) != record[key] for path, key in zip(pair, ("map_sha256", "mask_sha256"))):
            raise ValueError("An input changed during processing; no packet was written")
    output.mkdir(parents=True)
    np.save(output / "common_valid.npy", common, allow_pickle=False)
    for col, score in enumerate(scores):
        np.save(output / f"map-{chr(65+col)}.npy", score, allow_pickle=False)
    figure_receipt = plot_contexts(scores, common, reports, reference, box, output / "row_contexts.png")
    contexts = []
    dy = reference["spacing_um_yx"][0]/1000
    half_px = PROTOCOL["fixed_context_half_height_mm"]/dy
    for row_index, target in enumerate(targets, 1):
        centre = (target["box_xyxy"][1]+target["box_xyxy"][3])/2
        lo, hi = max(0, int(np.floor(centre-half_px))), min(common.shape[0], int(np.ceil(centre+half_px)))
        context = dict(row=row_index, local_slice_yxyx=None, native_arrays=[])
        if lo < hi:
            context["local_slice_yxyx"] = [lo, 0, hi, common.shape[1]]
            for col, score in enumerate(scores):
                name = f"row-{row_index}-map-{chr(65+col)}.npy"
                np.save(output / name, score[lo:hi], allow_pickle=False)
                context["native_arrays"].append(name)
            np.save(output / f"row-{row_index}-common-valid.npy", common[lo:hi], allow_pickle=False)
        contexts.append(context)
    result = dict(schema="public-306b-row-context-result-v1", experiment_kind=manifest["experiment_kind"],
                  protocol_sha256=digest(HERE/"protocol.json"), script_sha256=digest(__file__),
                  ranker_sha256=digest(HERE.parent/"line_review/review.py"),
                  source_inventory_sha256=PROTOCOL["source_inventory_sha256"],
                  reference_case=reference["id"], reference_shape=reference["shape"], field_box_yxyx=box,
                  spacing_um_yx=reference["spacing_um_yx"], common_valid_pixels=int(common.sum()),
                  common_mask_sha256=digest(output/"common_valid.npy"), own_valid_pixels=[int(m.sum()) for m in masks],
                  input_receipts=[{k:v for k,v in m.items() if k not in ("map", "mask")} for m in manifest["maps"]],
                  config=asdict(config), contexts=contexts, figure=figure_receipt,
                  maps=[dict(label=chr(65+i), proposals=len(r["proposals"]), seconds=r["seconds"],
                             search_truncated=r["search_truncated"], row_metrics=rows[i],
                             full_proposals=r["proposals"]) for i,r in enumerate(reports)],
                  comparison_status="Single published map only; no tile-placement experiment" if len(scores)==1 else "Declared paired inference settings checked; actual physical alignment and genuine model execution are not certified by this tool",
                  limitations=["Rows are published coarse annotations, not ink or glyph truth", "All maps require full inspection irrespective of rank", "No false-positive rate or candidate verdict", "Score maps keep their supplied scale; no new contrast stretch"])
    if len(scores) == 2:
        comparison = []
        for left, right in zip(rows[0], rows[1]):
            deltas = {}
            for k in PROTOCOL["top_k"]:
                l, r = left["top_k"][str(k)], right["top_k"][str(k)]
                deltas[str(k)] = {key+"_B_minus_A":r[key]-l[key] if l[key] is not None and r[key] is not None else None
                                  for key in ("box_coverage", "centerline_coverage")}
            comparison.append(dict(id=left["id"], top_k_coverage_differences=deltas,
                                   rank_pairs_A_B={key:[left[key],right[key]] for key in
                                       ("first_centreline_overlap_rank", "first_half_centreline_rank", "greatest_centreline_overlap_rank")}))
        result["paired_descriptive_changes"] = comparison
    (output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    body = "<h1>Published PHerc. 306B: fixed row contexts</h1><p>Every map remains review-required. The figure retains both published rows regardless of proposal rank. Missing support is purple; all panels use the same physical scale. A row overlap is not a letter reading.</p>"
    body += '<p><a href="report.json">Complete metrics and provenance</a></p><img src="row_contexts.png" style="max-width:100%" alt="Full field and both fixed row contexts">'
    (output/"review.html").write_text("<!doctype html><meta charset='utf-8'><title>Public 306B row contexts</title>"+body)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--public-controls-root", type=Path)
    group.add_argument("--pair-manifest", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.public_controls_root:
        reference = reference_case()
        folder = args.public_controls_root / "cases" / reference["id"]
        manifest = dict(reference_case=reference["id"], experiment_kind="single_published_reference",
                        field_box_yxyx=[0,0,*reference["shape"]],
                        maps=[dict(map="score.npy",mask="validity.npy",map_sha256=reference["score_sha256"],mask_sha256=reference["mask_sha256"])])
    else:
        manifest = json.loads(args.pair_manifest.read_text())
        folder = args.pair_manifest.parent
    result = run(manifest, folder, args.output_dir)
    print(json.dumps(dict(status="complete", map_count=len(result["maps"]),
                         proposals=[m["proposals"] for m in result["maps"]],
                         full_map_review_required=True, tile_placement_experiment=len(result["maps"])==2)))


if __name__ == "__main__":
    main()
