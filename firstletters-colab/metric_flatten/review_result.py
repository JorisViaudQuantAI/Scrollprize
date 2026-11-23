#!/usr/bin/env python3
"""Independent saved-array review; NumPy/Pillow only, no experiment imports.

Usage: python review_result.py ARTIFACT_DIR --output results/independent_review.json
Does not rerun the optimizer, either interpolator, or the identity rasterization.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_dir
    result = json.loads((root / "result.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    checks, details = {}, {}

    def check(name, condition):
        checks[name] = bool(condition)

    def close(a, b):
        return np.allclose(a, b, rtol=1e-12, atol=1e-12)

    hashes = {}
    for name, expected in result["artifacts"].items():
        path = root / name
        hashes[name] = sha(path)
        check("artifact:" + name, hashes[name] == expected["sha256"] and path.stat().st_size == expected["bytes"])
    with np.load(root / "replay_geometry.npz") as z:
        replay = dict(z)
    with np.load(root / "C_rasterization.npz") as z:
        raster = dict(z)
    a, b, published = (replay[k] for k in ("A", "B", "published"))
    raw, c, vc = raster["raw_xyz"], raster["float32_xyz"].astype(float), raster["valid"]
    vb, vt = np.isfinite(b).all(-1), np.isfinite(published).all(-1)
    check("B_published_support_exact", np.array_equal(vb, vt))
    check("B_published_values_exact", np.array_equal(b[vb & vt], published[vb & vt]))
    check("B_is_float32_roundtrip", np.array_equal(b[vb], b[vb].astype(np.float32).astype(float)))
    check("C_raw_finite_support", np.array_equal(np.isfinite(raw).all(-1), vc))
    check("C_float32_roundtrip", np.array_equal(raw.astype(np.float32), raster["float32_xyz"], equal_nan=True))
    faces, uv, xyz, ij = (replay[k] for k in ("source_triangles", "uv", "source_xyz", "source_ij"))
    check("source_XYZ_matches_A", np.array_equal(xyz, a[ij[:, 0], ij[:, 1]]))
    ids, weights = raster["source_triangle_id"][vc], raster["barycentric_weights"][vc]
    check("C_original_face_ids_valid", np.all((ids >= 0) & (ids < len(faces))))
    reconstructed = (xyz[faces[ids]] * weights[..., None]).sum(axis=1)
    error = float(np.abs(reconstructed - raw[vc]).max())
    pu = uv[faces]
    edge1, edge2 = pu[:, 1] - pu[:, 0], pu[:, 2] - pu[:, 0]
    twice_area = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    pg = protocol["gates"]["provenance"]
    check("C_barycentric_XYZ_tolerance", error <= pg["accepted_query_xyz_error_voxels_max"])
    check("C_positive_UV_faces", np.all(twice_area[ids] > 1e-12))
    check("C_accepted_reason_bits_zero", np.all(raster["reason_bits"][vc] == 0))
    check("C_barycentric_simplex", np.all(weights >= -1e-9) and np.all(np.abs(weights.sum(1) - 1) <= pg["barycentric_sum_absolute_tolerance"]))
    uv_error = float(np.abs((pu[ids] * weights[..., None]).sum(1) - np.argwhere(vc)[:, ::-1]).max())
    check("C_barycentric_target_grid", uv_error <= 1e-9)
    details["C_provenance"] = {"accepted_queries": int(vc.sum()), "max_absolute_XYZ_error_voxels": error, "max_absolute_UV_error_grid_units": uv_error, "minimum_accepted_signed_UV_area": float(twice_area[ids].min() / 2)}
    cfg = protocol["geometry"]
    crop = np.zeros_like(vb)
    r0, r1 = cfg["claim_crop_rows_half_open"]
    c0, c1 = cfg["claim_crop_columns_half_open"]
    crop[r0:r1, c0:c1] = True
    ret = {"whole": float((vb & vc).sum() / vb.sum()), "published_claim": float((vb & vc & crop).sum() / (vb & crop).sum())}
    rg = result["gates"]["retention"]
    counts = {"B_valid_nodes": int(vb.sum()), "C_valid_nodes": int(vc.sum()), "B_only_nodes": int((vb & ~vc).sum()), "C_only_nodes": int((vc & ~vb).sum()), "claim_B_valid_nodes": int((vb & crop).sum()), "claim_C_valid_nodes": int((vc & crop).sum())}
    check("retention_receipt", all(rg[k] == v for k, v in counts.items()) and all(close(v, rg["baseline_support_retained"][k]) for k, v in ret.items()))
    check("whole_retention_gate_failure_preserved", ret["whole"] < protocol["gates"]["retention"]["whole_baseline_support_fraction_min"] and not rg["pass"])
    details["retention"] = {**counts, "baseline_support_retained": ret}
    arms = {"A": a, "B": b, "C": c}
    common = np.logical_and.reduce([np.isfinite(p).all(-1) for p in arms.values()])
    cells = common[:-1, :-1] & common[1:, :-1] & common[:-1, 1:] & common[1:, 1:]
    row, col = np.indices(cells.shape)
    pitch = cfg["spacing_voxels"] / (1000 / cfg["source_voxel_um"])
    u, v = cfg["carrier_origin_mm"][0] + (col + .5) * pitch, cfg["carrier_origin_mm"][1] + (row + .5) * pitch
    lo_u, hi_u, lo_v, hi_v = protocol["gates"]["right_group"]["bounds_mm"]
    cells &= (u >= lo_u) & (u < hi_u) & (v >= lo_v) & (v < hi_v)
    right = {}
    for arm, points in arms.items():
        du = (points[:-1, 1:] - points[:-1, :-1])[cells]
        dv = (points[1:, :-1] - points[:-1, :-1])[cells]
        un, vn = np.linalg.norm(du, axis=1), np.linalg.norm(dv, axis=1)
        su, sv = un / cfg["spacing_voxels"], vn / cfg["spacing_voxels"]
        shear = np.degrees(np.arcsin(np.clip((du * dv).sum(1) / (un * vn), -1, 1)))
        metrics = {"cells": int(cells.sum()), "median_abs_v_stretch_error": float(np.median(np.abs(sv - 1))), "p95_abs_shear_degrees": float(np.percentile(np.abs(shear), 95)), "either_axis_outside_5pct_fraction": float(np.mean((np.abs(su - 1) > .05) | (np.abs(sv - 1) > .05)))}
        right[arm] = metrics
        check("right_group:" + arm, all(close(value, result["gates"]["right_group"]["metrics"][arm][key]) for key, value in metrics.items()))
    details["right_group_common_four_corner_metrics"] = right
    delta = np.linalg.norm(b[vb & vc] - c[vb & vc], axis=1)
    bc = {"common_nodes": int(len(delta)), "distance_voxels_max": float(delta.max()), "strictly_greater_than_20_voxels": int((delta > 20).sum()), "strictly_greater_than_100_voxels": int((delta > 100).sum())}
    check("B_C_saved_coordinate_difference", close(delta.max(), result["B_C_coordinate_difference"]["distance_voxels_max"]) and close(np.percentile(delta, [5, 50, 95]), result["B_C_coordinate_difference"]["distance_voxels_p5_p50_p95"]))
    check("B_C_threshold_counts", bc["strictly_greater_than_20_voxels"] == 206 and bc["strictly_greater_than_100_voxels"] == 58)
    details["B_C_coordinate_difference"] = bc
    rounding = np.linalg.norm(c[vc] - raw[vc], axis=1)
    check("C_float32_rounding_receipt", close(rounding.max(), result["C_float32_serialization"]["distance_voxels_max"]))
    details["C_float32_serialization_max_distance_voxels"] = float(rounding.max())
    for arm, points in arms.items():
        valid = replay["region"] if arm == "A" else (vb if arm == "B" else vc)
        path = root / (arm + "_tifxyz")
        stored = np.stack([np.asarray(Image.open(path / (axis + ".tif"))) for axis in "xyz"], axis=-1)
        mask = np.asarray(Image.open(path / "mask.tif"))
        check("serialized_XYZ:" + arm, stored.dtype == np.float32 and np.array_equal(stored, np.where(valid[..., None], points, -1).astype(np.float32)))
        check("serialized_mask:" + arm, np.array_equal(mask, valid.astype(np.uint8) * 255))
        expected = {"source_ij": ij, "source_xyz": xyz, "original_source_triangles": faces, "parameter_uv": replay["uv_initial"] if arm == "A" else uv}
        if arm == "C":
            expected.update({k: raster[k] for k in ("source_triangle_id", "barycentric_weights", "valid", "reason_bits", "coverage_count")})
        with np.load(path / "provenance.npz") as z:
            check("serialized_provenance:" + arm, all(np.array_equal(z[k], value, equal_nan=True) for k, value in expected.items()))
    # Reconstruct triangle incidence on the published support, not rasterizer output.
    tl, tr, bl, br = vt[:-1, :-1], vt[:-1, 1:], vt[1:, :-1], vt[1:, 1:]
    quad = tl & tr & bl & br
    incident = np.zeros_like(vt)
    incident[:-1, :-1] |= quad
    incident[:-1, 1:] |= quad
    incident[1:, :-1] |= quad
    incident[1:, 1:] |= quad
    identity = result["gates"]["identity"]
    check("identity_expected_incidence_receipt", int(incident.sum()) == identity["expected_valid_nodes"] and int((vt & ~incident).sum()) == identity["valid_nodes_without_original_triangle_incidence"])
    check("identity_receipt_internal_consistency", identity["pass"] and identity["actual_valid_nodes"] == identity["expected_valid_nodes"] and identity["max_absolute_xyz_error_voxels"] <= protocol["gates"]["identity"]["max_absolute_xyz_error_voxels"])
    details["identity_scope"] = "Expected triangle-incidence counts recomputed; actual identity output and its zero error are receipt-only because arrays were not saved. No independent rasterizer rerun."
    with np.load(root / "support_perturbation.npz") as z:
        perturb = dict(z)
    carrier = np.column_stack((cfg["carrier_origin_mm"][0] + ij[:, 1] * pitch, cfg["carrier_origin_mm"][1] + ij[:, 0] * pitch))
    fix = int(np.argmin(np.abs(carrier).sum(1)))
    removed = np.all(np.abs(ij - ij[fix]) <= protocol["gates"]["support_perturbation"]["remove_source_square_side_vertices"] // 2, axis=1)
    kept = ~removed[faces].any(1)
    hole = np.zeros_like(vc)
    hole[vc] = removed[faces[ids]].all(1) & (weights > 1e-6).all(1)
    check("support_removal_saved_masks", np.array_equal(removed, perturb["removed_source_vertices"]) and np.array_equal(kept, perturb["kept_source_triangles"]) and np.array_equal(hole, perturb["hole_queries"]))
    hole_counts = {"removed_source_vertices": int(removed.sum()), "removed_source_triangles": int((~kept).sum()), "interior_test_queries": int(hole.sum()), "Delaunay_filled_interior_queries": int(np.isfinite(perturb["B_xyz"][hole]).all(-1).sum()), "original_triangle_filled_interior_queries": int(perturb["C_valid"][hole].sum())}
    check("support_removal_receipt", all(result["gates"]["support_perturbation"][k] == value for k, value in hole_counts.items()))
    check("support_removal_C_finite_support", np.array_equal(np.isfinite(perturb["C_xyz"]).all(-1), perturb["C_valid"]))
    details["support_removal_saved_array_counts"] = hole_counts
    details["support_removal_scope"] = "Saved removal/query masks and output support verified; neither interpolator independently rerun. Designed removal is not observed damage."
    check("only_retention_gate_reported_failed", [k for k, v in result["gates"].items() if not v["pass"]] == ["retention"] and result["status"] == "geometry_gate_failure")
    receipt = {"schema": "scrollprize.metric-flatten.independent-review.v1", "status": "saved_array_checks_pass" if all(checks.values()) else "discrepancy", "scope": "Independent NumPy/Pillow arithmetic on saved public-control arrays and TIFFs; no experiment or topology function imports, optimization, rasterizer rerun, CT, CNN, or network.", "review_script_sha256": sha(Path(__file__)), "result_sha256": sha(root / "result.json"), "protocol_sha256": sha(root / "protocol.json"), "verified_artifact_sha256": hashes, "checks": checks, "details": details, "discrepancies": [k for k, v in checks.items() if not v], "geometry_adoption_gate_still_failed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "checks": len(checks), "discrepancies": receipt["discrepancies"]}))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
