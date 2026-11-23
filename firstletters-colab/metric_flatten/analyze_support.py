#!/usr/bin/env python3
"""Post-run diagnosis of the frozen public 306B geometry control.

Read saved artifacts only. No source acquisition, optimization, CT, ink model,
threshold selection or gate revision occurs. The optional-looking Delaunay
diagnostic is the SAME baseline construction from the saved UV solution; its
decoded float32 output must reproduce B exactly before attribution is reported.
The diagnostic does not establish that the original mesh is anatomical truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay


THRESHOLDS_VOXELS = (1, 20, 100)
REQUIRED_ARTIFACTS = ("protocol.json", "replay_geometry.npz", "C_rasterization.npz")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return [float(x) for x in np.percentile(values, [5, 50, 95])] if len(values) else None


def distance_summary(values, voxel_um):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("A common-support coordinate difference is nonfinite")
    return {
        "nodes": int(len(values)),
        "exact_zero_nodes": int((values == 0).sum()),
        "euclidean_distance_voxels_p5_p50_p95": quantiles(values),
        "euclidean_distance_voxels_max": float(values.max()) if len(values) else None,
        "euclidean_distance_um_p5_p50_p95": quantiles(values * voxel_um),
        "euclidean_distance_um_max": float(values.max() * voxel_um) if len(values) else None,
        "strictly_greater_than_voxels": {str(t): int((values > t).sum()) for t in THRESHOLDS_VOXELS},
    }


def in_box(u, v, bounds):
    x0, x1, y0, y1 = bounds
    return (u >= x0) & (u < x1) & (v >= y0) & (v < y1)


def four_corner(mask):
    return mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]


def grid_metrics(points, spacing):
    """Frozen runner's tangent convention, on explicitly matched cells."""
    du = (points[:, 1:] - points[:, :-1])[:-1]
    dv = (points[1:] - points[:-1])[:, :-1]
    un, vn = np.linalg.norm(du, axis=-1), np.linalg.norm(dv, axis=-1)
    cosine = np.sum(du * dv, axis=-1) / np.maximum(un * vn, 1e-9)
    return {"stretch_u": un / spacing, "stretch_v": vn / spacing,
            "shear_deg": np.degrees(np.arcsin(np.clip(cosine, -1, 1)))}


def row_keys(rows):
    rows = np.ascontiguousarray(np.sort(np.asarray(rows, dtype=np.int64), axis=1))
    return rows.view(np.dtype([(f"v{i}", np.int64) for i in range(rows.shape[1])])).reshape(-1)


def original_face_ids(original, reconstructed):
    """Orientation-independent membership, preserving original face identity."""
    keys, first = np.unique(row_keys(original), return_index=True)
    query = row_keys(reconstructed)
    positions = np.searchsorted(keys, query)
    found = positions < len(keys)
    found[found] &= keys[positions[found]] == query[found]
    result = np.full(len(query), -1, np.int64)
    result[found] = first[positions[found]]
    return result


def baseline_connections(saved, c_saved, common, lost, distances, voxel_um):
    """Attribute the unchanged B interpolation, after exact saved-B validation."""
    uv, xyz = saved["uv"], saved["source_xyz"]
    faces, ij = saved["source_triangles"], saved["source_ij"]
    region, initial = saved["region"], saved["uv_initial"]
    b = saved["B"]
    triangulation = Delaunay(uv)
    reconstructed = np.full_like(b, np.nan)
    reconstructed[region] = LinearNDInterpolator(triangulation, xyz)(initial)
    reconstructed = reconstructed.astype(np.float32).astype(np.float64)
    vb, vr = np.isfinite(b).all(-1), np.isfinite(reconstructed).all(-1)
    exact = bool(np.array_equal(vb, vr) and np.array_equal(b[vb], reconstructed[vb]))
    validation = {"equal_support": bool(np.array_equal(vb, vr)),
                  "exact_float32_coordinates": exact,
                  "max_absolute_coordinate_error_voxels": float(np.max(np.abs(b[vb & vr] - reconstructed[vb & vr]))) if (vb & vr).any() else None,
                  "optimizer_runs": 0, "saved_UV_only": True}
    if not exact:
        return {"status": "attribution_not_reported_baseline_reconstruction_not_exact", "validation": validation}
    simplex_local = triangulation.find_simplex(initial)
    simplex = np.full(region.shape, -1, np.int64)
    simplex[region] = simplex_local
    if (simplex[vb] < 0).any():
        raise ValueError("Delaunay point location disagrees with saved valid baseline")
    original_id = original_face_ids(faces, triangulation.simplices)
    span = np.ptp(ij[triangulation.simplices], axis=1)
    one_cell = (span <= 1).all(1)
    kind = np.where(original_id >= 0, 0, np.where(one_cell, 1, 2))
    kind_grid = np.full(region.shape, -1, np.int8)
    kind_grid[vb] = kind[simplex[vb]]
    names = ("original_source_triangle", "alternative_within_one_original_grid_cell",
             "nonlocal_original_grid_connection")
    strata = {}
    for code, name in enumerate(names):
        selected = kind_grid == code
        values = distances[selected & common]
        sp = span[simplex[selected & common]]
        strata[name] = {
            "B_valid_nodes": int((selected & vb).sum()),
            "B_only_nodes": int((selected & lost).sum()),
            "B_C_common_difference": distance_summary(values, voxel_um),
            "source_grid_row_span_max_on_common": int(sp[:, 0].max()) if len(sp) else None,
            "source_grid_column_span_max_on_common": int(sp[:, 1].max()) if len(sp) else None,
        }
    common_simplex = simplex[common]
    c_owner = c_saved["source_triangle_id"][common]
    same_face = original_id[common_simplex] == c_owner
    point_uv = np.column_stack(np.nonzero(common)[::-1]).astype(float)
    transform = triangulation.transform[common_simplex]
    first_two = np.einsum("ijk,ik->ij", transform[:, :2], point_uv - transform[:, 2])
    weights = np.column_stack((first_two, 1 - first_two.sum(1)))
    boundary = (np.abs(weights) <= 1e-9).any(1)
    loss_by_reason_and_connection = []
    for bits in np.unique(c_saved["reason_bits"][lost]):
        selected = lost & (c_saved["reason_bits"] == bits)
        loss_by_reason_and_connection.append({"reason_bits": int(bits),
            **{name: int((selected & (kind_grid == code)).sum()) for code, name in enumerate(names)}})
    return {
        "status": "exact_baseline_reconstruction_verified",
        "validation": validation,
        "classification": "Original membership compares sorted original vertex triplets. A local alternative fits within one original row/column cell; a nonlocal connection spans more than one original row or column step. These categories describe mesh provenance, not anatomical correctness.",
        "strata": strata,
        "same_original_face_as_C": distance_summary(distances[common][same_face], voxel_um),
        "different_original_face_from_C": distance_summary(distances[common][(original_id[common_simplex] >= 0) & ~same_face], voxel_um),
        "common_nodes_at_Delaunay_barycentric_boundary_1e_minus_9": int(boundary.sum()),
        "boundary_qualification": "Equivalent point-location ownership can differ on a shared boundary; original/nonlocal attribution of large interior displacements is stronger than boundary ownership.",
        "loss_reason_by_baseline_connection": loss_by_reason_and_connection,
        "new_CNN_forwards": 0, "new_CT_requests": 0,
    }


def load_completed(run):
    receipt_path = run / "result.json"
    receipt = json.loads(receipt_path.read_text())
    if not receipt.get("finished_utc") or receipt.get("status") not in {"geometry_gate_failure", "all_geometry_gates_pass"}:
        raise ValueError("Analysis requires a completed scientific-gate receipt")
    verified = {"result.json": digest(receipt_path)}
    for name in REQUIRED_ARTIFACTS:
        observed = digest(run / name)
        expected = receipt["artifacts"][name]["sha256"]
        if observed != expected:
            raise ValueError(f"Saved artifact differs from receipt: {name}")
        verified[name] = observed
    protocol = json.loads((run / "protocol.json").read_text())
    if protocol.get("source_manifest_sha256") != "087464bd4c3163ce7314779aa05d917dcacac5791a00280b6bc57073473f1ba2":
        raise ValueError("This diagnostic is limited to the pinned public PHerc. 306B control")
    if receipt["gates"]["retention"]["pass"]:
        raise ValueError("This report diagnoses the frozen failed-retention control")
    with np.load(run / "replay_geometry.npz", allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    with np.load(run / "C_rasterization.npz", allow_pickle=False) as archive:
        c_saved = {key: archive[key] for key in archive.files}
    return receipt, protocol, saved, c_saved, verified


def analyze(run):
    receipt, protocol, saved, c_saved, verified = load_completed(run)
    a, b = saved["A"], saved["B"]
    c = c_saved["float32_xyz"].astype(np.float64)
    vb, vc = np.isfinite(b).all(-1), c_saved["valid"]
    reasons = c_saved["reason_bits"]
    if vc.shape != vb.shape or not np.array_equal(np.isfinite(c).all(-1), vc):
        raise ValueError("Saved C support/coordinates are inconsistent")
    if (reasons[vc] != 0).any():
        raise ValueError("Saved C accepts rejection bits")
    common, lost, extra = vb & vc, vb & ~vc, vc & ~vb
    expected = receipt["gates"]["retention"]
    for key, count in (("B_valid_nodes", vb.sum()), ("C_valid_nodes", vc.sum()),
                       ("B_only_nodes", lost.sum()), ("C_only_nodes", extra.sum())):
        if int(count) != expected[key]:
            raise ValueError(f"Support accounting differs from receipt: {key}")
    config = protocol["geometry"]
    voxel_um = float(config["source_voxel_um"])
    pitch_mm = float(config["spacing_voxels"]) * voxel_um / 1000
    per_query_area = pitch_mm ** 2
    yy, xx = np.indices(vb.shape)
    u = config["carrier_origin_mm"][0] + xx * pitch_mm
    v = config["carrier_origin_mm"][1] + yy * pitch_mm
    right_box = protocol["gates"]["right_group"]["bounds_mm"]
    right_nodes = in_box(u, v, right_box)
    crop = np.zeros_like(vb)
    r0, r1 = config["claim_crop_rows_half_open"]
    c0, c1 = config["claim_crop_columns_half_open"]
    crop[r0:r1, c0:c1] = True
    reason_names = {int(k): value for k, value in receipt["C_rasterizer"]["reason_names"].items()}

    invalid_labels, _ = ndi.label(~vc, structure=np.ones((3, 3), bool))
    border_ids = np.unique(np.concatenate((invalid_labels[0], invalid_labels[-1], invalid_labels[:, 0], invalid_labels[:, -1])))
    border_ids = border_ids[border_ids != 0]
    exterior = np.isin(invalid_labels, border_ids)
    reason_rows = []
    for bits in np.unique(reasons[lost]):
        selected = lost & (reasons == bits)
        count = int(selected.sum())
        labels = [name for bit, name in sorted(reason_names.items()) if int(bits) & bit]
        reason_rows.append({"reason_bits": int(bits), "reasons": labels,
            "nodes": count, "fraction_of_all_B_only_nodes": count / int(lost.sum()),
            "fraction_of_B_valid_nodes": count / int(vb.sum()),
            "nominal_chart_sample_area_mm2": count * per_query_area,
            "in_published_padded_crop": int((selected & crop).sum()),
            "in_fixed_right_group": int((selected & right_nodes).sum()),
            "C_invalid_exterior_connected_8_neighbours": int((selected & exterior).sum()),
            "C_invalid_enclosed_8_neighbours": int((selected & ~exterior).sum())})
    if sum(row["nodes"] for row in reason_rows) != int(lost.sum()):
        raise AssertionError("The reason partition must account for every B-only node")

    distances = np.full(vb.shape, np.nan)
    distances[common] = np.linalg.norm(b[common] - c[common], axis=1)
    contexts = {"all_common": common, "published_padded_crop": common & crop,
                "fixed_right_group_nodes": common & right_nodes,
                "outside_published_padded_crop": common & ~crop}
    for row in reason_rows:
        near = ndi.binary_dilation(lost & (reasons == row["reason_bits"]), structure=np.ones((3, 3), bool))
        contexts[f"8_neighbour_adjacent_to_loss_reason_{row['reason_bits']}"] = common & near
    for name, selector in (("exterior_connected", exterior), ("enclosed", ~exterior)):
        contexts[f"8_neighbour_adjacent_to_{name}_loss"] = common & ndi.binary_dilation(lost & selector, structure=np.ones((3, 3), bool))
    contexts_report = {name: distance_summary(distances[mask], voxel_um) for name, mask in contexts.items()}
    labelled_loss, component_count = ndi.label(lost, structure=np.ones((3, 3), bool))
    component_sizes = np.bincount(labelled_loss.ravel())[1:]
    source_tri_xyz = saved["source_xyz"][saved["source_triangles"]]
    source_area_vox2 = .5 * np.linalg.norm(np.cross(source_tri_xyz[:, 1] - source_tri_xyz[:, 0],
                                                  source_tri_xyz[:, 2] - source_tri_xyz[:, 0]), axis=1)
    cells = np.logical_and.reduce([four_corner(np.isfinite(points).all(-1)) for points in (a, b, c)])
    cells &= in_box(u[:-1, :-1] + pitch_mm / 2, v[:-1, :-1] + pitch_mm / 2, right_box)
    metrics = {name: {key: values[cells] for key, values in grid_metrics(points, config["spacing_voxels"]).items()}
               for name, points in (("A", a), ("B", b), ("C", c))}
    for name in metrics:
        frozen = receipt["gates"]["right_group"]["metrics"][name]
        if int(cells.sum()) != frozen["cells"]:
            raise ValueError("Fixed right-group common population differs from receipt")
        for key, values in metrics[name].items():
            if not np.allclose(quantiles(values), frozen[f"{key}_p5_p50_p95"], rtol=0, atol=1e-10):
                raise ValueError("Recomputed right-group metric differs from receipt")

    report = {
        "schema": "scrollprize.metric-flatten.support-analysis.v1",
        "design": "Post-run descriptive diagnosis of the failed frozen gate; no threshold changes, optimization, new source data or ink evaluation",
        "source_protocol_commit": receipt["protocol_commit"],
        "source_finished_utc": receipt["finished_utc"],
        "source_status": receipt["status"],
        "source_artifact_sha256": verified,
        "failed_gates_unchanged": [name for name, gate in receipt["gates"].items() if not gate["pass"]],
        "decision_unchanged": receipt["decision"],
        "support": {"B_valid_nodes": int(vb.sum()), "C_valid_nodes": int(vc.sum()),
            "common_nodes": int(common.sum()), "B_only_nodes": int(lost.sum()), "C_only_nodes": int(extra.sum()),
            "whole_retention": float(common.sum() / vb.sum()),
            "frozen_whole_retention_minimum": protocol["gates"]["retention"]["whole_baseline_support_fraction_min"],
            "published_padded_crop_B_nodes": int((vb & crop).sum()),
            "published_padded_crop_retained_nodes": int((common & crop).sum()),
            "exhaustive_disjoint_B_only_reason_partition": reason_rows,
            "B_only_connected_components_8_neighbours": int(component_count),
            "B_only_component_sizes_descending": sorted(map(int, component_sizes), reverse=True),
            "raster_connectivity_qualification": "Exterior/enclosed describes 8-neighbour invalid SAMPLE connectivity to the image boundary; it does not identify physical cracks or prove continuous mesh topology."},
        "area_accounting": {"nominal_chart_pitch_mm": pitch_mm,
            "nominal_chart_area_per_query_mm2": per_query_area,
            "B_only_assigned_chart_sample_area_mm2": float(lost.sum() * per_query_area),
            "B_only_assigned_chart_sample_area_cm2": float(lost.sum() * per_query_area / 100),
            "full_original_piecewise_linear_source_area_mm2": float(source_area_vox2.sum() * (voxel_um / 1000) ** 2),
            "true_original_source_surface_area_lost_mm2": None,
            "qualification": "Node count times nominal pitch squared is an assigned chart-sample area, not integrated 3D surface area, lost papyrus area or lost ink area. Rejected nodes can have no source coverage or multiple conflicting source faces; no unique true lost source area is identified. The separately reported full source-triangle area is an exact area of the supplied piecewise-linear source mesh, not evidence that its sheet assignment is correct."},
        "B_C_common_coordinate_differences": contexts_report,
        "difference_context_definition": "Every common node has C reason_bits=0. Reason-specific contexts are its immediate 8-neighbour adjacency to B-only reason masks. Contexts can overlap and are descriptive, not new tests or revised gates.",
        "baseline_connection_attribution": baseline_connections(saved, c_saved, common, lost, distances, voxel_um),
        "fixed_right_group": {"bounds_mm": right_box, "common_four_corner_cells": int(cells.sum()),
            "receipt_metrics_reproduced": True, "frozen_receipt_metrics": receipt["gates"]["right_group"]["metrics"],
            "same_target_grid_positions_not_material_correspondence": True},
        "scope": {"new_CT_requests": 0, "new_CNN_forwards": 0, "new_optimizer_runs": 0,
            "anatomical_truth_established": False, "ink_gain_established": False,
            "new_reading_claim": False, "gate_relaxed": False},
    }
    plotting = {"B_valid": vb, "C_valid": vc, "common": common, "lost": lost,
                "extra": extra, "reasons": reasons, "u": u, "v": v,
                "distances": distances, "metrics": metrics, "protocol": protocol,
                "reason_rows": reason_rows, "voxel_um": voxel_um, "pitch_mm": pitch_mm}
    return report, plotting


def render_svg(path, report, data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "306b-support-analysis-v1"})
    figure = plt.figure(figsize=(15, 10), constrained_layout=False)
    grid = figure.add_gridspec(2, 3, left=.06, right=.975, bottom=.18, top=.86,
                              hspace=.57, wspace=.32, height_ratios=[1.15, 1])
    support_ax, distance_ax, reason_ax = [figure.add_subplot(grid[0, i]) for i in range(3)]
    u, v, pitch = data["u"], data["v"], data["pitch_mm"]
    extent = [float(u.min() - pitch / 2), float(u.max() + pitch / 2),
              float(v.min() - pitch / 2), float(v.max() + pitch / 2)]
    colors = ["#ffffff", "#bac6ce", "#dc5b29", "#9852a1", "#cc304a", "#d6a629", "#7155a4", "#007fa8"]
    image = np.zeros(data["B_valid"].shape, np.uint8)
    image[data["common"]] = 1
    rows = sorted(data["reason_rows"], key=lambda item: (-item["nodes"], item["reason_bits"]))
    if len(rows) > 5:
        raise ValueError("Figure needs explicit design for more than five loss reason combinations")
    row_colors = {}
    for code, row in enumerate(rows, start=2):
        image[data["lost"] & (data["reasons"] == row["reason_bits"])] = code
        row_colors[row["reason_bits"]] = colors[code]
    image[data["extra"]] = 7
    support_ax.imshow(image, origin="lower", extent=extent, interpolation="nearest",
                      cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(9) - .5, 8), aspect="equal")
    config = data["protocol"]["geometry"]
    r0, r1 = config["claim_crop_rows_half_open"]
    c0, c1 = config["claim_crop_columns_half_open"]
    x0 = config["carrier_origin_mm"][0] + (c0 - .5) * pitch
    y0 = config["carrier_origin_mm"][1] + (r0 - .5) * pitch
    for ax in (support_ax, distance_ax):
        ax.add_patch(Rectangle((x0, y0), (c1 - c0) * pitch, (r1 - r0) * pitch,
                               fill=False, edgecolor="#1b364c", linewidth=1, linestyle="--"))
        left, right, lower, upper = report["fixed_right_group"]["bounds_mm"]
        ax.add_patch(Rectangle((left, lower), right - left, upper - lower,
                               fill=False, edgecolor="#0a0a0a", linewidth=1.3))
        ax.set(xlabel="Carrier chart U (mm)", ylabel="Carrier chart V (mm)", xlim=extent[:2], ylim=extent[2:])
    support_ax.set_title("A. Full-field sample support", loc="left", fontweight="bold")
    support_ax.legend(handles=[Patch(color=colors[1], label="B and C common support"),
                              Patch(facecolor="none", edgecolor="#1b364c", linestyle="--", label="Published padded crop"),
                              Patch(facecolor="none", edgecolor="black", label="Fixed right-group box")],
                      loc="upper center", bbox_to_anchor=(.5, -.2), frameon=False, fontsize=8)
    distances_um = data["distances"] * data["voxel_um"]
    positive = data["common"] & (distances_um > 0)
    display = np.ma.masked_where(~positive, distances_um)
    vmax = max(float(np.nanmax(distances_um)), .1)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#edf0f2")
    shown = distance_ax.imshow(display, origin="lower", extent=extent, interpolation="nearest", aspect="equal",
                               cmap=cmap, norm=LogNorm(vmin=.01, vmax=vmax))
    distance_ax.set_title("B. Common-support B–C difference", loc="left", fontweight="bold")
    colorbar = figure.colorbar(shown, ax=distance_ax, fraction=.055, pad=.025, shrink=.82)
    colorbar.set_label("Euclidean XYZ difference (µm; log scale)", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    distance_ax.text(.5, -.23, "Grey = exact agreement or no common support\nDifference is not an anatomical error estimate",
                     transform=distance_ax.transAxes, ha="center", va="top", fontsize=8)
    labels = [textwrap.fill(" + ".join(row["reasons"]).replace("_", " "), 40) for row in rows]
    positions = np.arange(len(rows)) * 1.65 + .45
    reason_ax.barh(positions, [row["nodes"] for row in rows], color=[row_colors[row["reason_bits"]] for row in rows], height=.57)
    reason_ax.set_yticks([])
    reason_ax.set_ylim(float(positions[-1] + .7), -.65)
    maximum = max(row["nodes"] for row in rows) if rows else 1
    reason_ax.set_xlim(0, maximum * 1.22)
    for position, row, label in zip(positions, rows, labels):
        reason_ax.text(0, position - .36, label, va="bottom", ha="left", fontsize=9)
        reason_ax.text(row["nodes"] + maximum * .025, position, f"{row['nodes']:,}", va="center", fontsize=10)
    reason_ax.set(xlabel="B-only nodes (disjoint reasons)")
    reason_ax.set_title("C. All rejected B samples accounted for", loc="left", fontweight="bold")
    reason_ax.spines[["top", "right"]].set_visible(False)
    reason_ax.text(.5, -.23, f"{int(data['lost'].sum()):,} nodes × nominal pitch² = "
                   f"{report['area_accounting']['B_only_assigned_chart_sample_area_mm2']:.2f} mm²\n"
                   "Assigned chart area; true lost surface area is unknown",
                   transform=reason_ax.transAxes, ha="center", va="top", fontsize=8)
    arm_colors = ["#929aa1", "#008f9b", "#e28f38"]
    for column, (metric, title, target) in enumerate((("stretch_u", "D. Right-group U stretch", 1),
                                                     ("stretch_v", "E. Right-group V stretch", 1),
                                                     ("shear_deg", "F. Right-group signed shear", 0))):
        ax = figure.add_subplot(grid[1, column])
        stats = []
        for arm in ("A", "B", "C"):
            q = np.percentile(data["metrics"][arm][metric], [5, 25, 50, 75, 95])
            stats.append({"label": arm, "whislo": q[0], "q1": q[1], "med": q[2], "q3": q[3], "whishi": q[4]})
        artists = ax.bxp(stats, showfliers=False, patch_artist=True, widths=.5)
        for patch, color in zip(artists["boxes"], arm_colors):
            patch.set(facecolor=color, alpha=.85)
        for line in artists["medians"]:
            line.set(color="#1f2830", linewidth=1.5)
        ax.axhline(target, color="#334450", linewidth=.8)
        tolerance = .05 if target == 1 else 5
        for bound in (target - tolerance, target + tolerance):
            ax.axhline(bound, color="#bf4b45", linestyle="--", linewidth=.8, alpha=.7)
        ax.set_xticks([1, 2, 3], ["A\nBefore ARAP", "B\nDelaunay", "C\nOriginal triangles"], fontsize=9)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("Spacing / nominal spacing" if target else "Shear (degrees)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4e7e9", linewidth=.6)
        ax.set_axisbelow(True)
    retention = report["support"]["whole_retention"]
    minimum = report["support"]["frozen_whole_retention_minimum"]
    figure.suptitle("Published PHerc. 306B geometry control: the whole-support gate remains failed",
                     x=.06, y=.965, ha="left", fontsize=17, fontweight="bold")
    figure.text(.06, .918,
        f"Retained {retention:.3%} of B samples; frozen minimum {minimum:.0%}.  "
        f"Published padded crop retained {report['support']['published_padded_crop_retained_nodes']:,} / "
        f"{report['support']['published_padded_crop_B_nodes']:,} samples.  No ink evaluation.", fontsize=11)
    all_dist = report["B_C_common_coordinate_differences"]["all_common"]
    counts = all_dist["strictly_greater_than_voxels"]
    attribution = report["baseline_connection_attribution"]
    attribution_line = "Baseline connection attribution was not verified; no connection-specific claim is shown."
    if attribution["status"] == "exact_baseline_reconstruction_verified":
        strata = attribution["strata"]
        nonlocal_count = strata["nonlocal_original_grid_connection"]["B_C_common_difference"]["strictly_greater_than_voxels"]["100"]
        original = strata["original_source_triangle"]["B_C_common_difference"]
        attribution_line = (f"{nonlocal_count:,}/{counts['100']:,} differences above 100 voxels use nonlocal source-grid connections; "
                            f"{original['exact_zero_nodes']:,}/{original['nodes']:,} original-triangle samples agree exactly.")
    figure.text(.06, .04,
        f"Bottom panels: {report['fixed_right_group']['common_four_corner_cells']:,} matched four-corner cells at fixed chart positions; "
        "boxes show 25th–75th percentiles, whiskers 5th–95th.\n"
        f"Common B–C differences exceed 1 / 20 / 100 voxels at {counts['1']:,} / {counts['20']:,} / {counts['100']:,} nodes; "
        f"maximum {all_dist['euclidean_distance_um_max'] / 1000:.3f} mm.\n"
        f"{attribution_line}\n"
        "Source-triangle provenance does not establish anatomical correctness, better ink detection, or a reading. Gates were not revised.",
        fontsize=9, linespacing=1.6)
    figure.savefig(path, format="svg", metadata={"Date": None, "Creator": "Scrollprize public-control support analysis"})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    started = time.monotonic()
    report, plotting = analyze(args.run_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render_svg(args.out_dir / "geometry_summary.svg", report, plotting)
    report["analysis_code_sha256"] = digest(__file__)
    report["geometry_summary_svg_sha256"] = digest(args.out_dir / "geometry_summary.svg")
    report["analysis_wall_seconds"] = time.monotonic() - started
    (args.out_dir / "support_analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "descriptive_analysis_complete", "failed_gates_unchanged": report["failed_gates_unchanged"],
                      "B_only_nodes": report["support"]["B_only_nodes"],
                      "baseline_attribution": report["baseline_connection_attribution"]["status"],
                      "analysis_wall_seconds": report["analysis_wall_seconds"]}))


if __name__ == "__main__":
    main()
