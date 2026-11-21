#!/usr/bin/env python3
"""Audit saved crest metrics or a paired-render manifest; never alter geometry.

The metrics input is Codex's local point-metrics.json list. Its public aggregate
result.json is deliberately rejected as insufficient evidence. Output contains
only aggregate decisions and no input paths, coordinates, source IDs or readings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
SHA = re.compile(r"[0-9a-f]{64}\Z")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def audit_point(row):
    """Pass only the fixed detected-crest rule; pass never means same sheet."""
    reasons = []
    needed = ("material_fraction", "n_peaks", "raw_nearest", "snapped_nearest",
              "after_same_peak", "unique_raw", "unique_snapped", "nearest_peak_changed")
    if not isinstance(row, dict) or any(key not in row for key in needed):
        return ["missing_per_site_evidence"]
    for key in ("material_fraction", "raw_nearest", "snapped_nearest", "after_same_peak"):
        if not finite(row[key]):
            return ["invalid_numeric_evidence"]
    if not 0 <= row["material_fraction"] <= 1 or any(row[k] < 0 for k in ("raw_nearest", "snapped_nearest", "after_same_peak")):
        return ["invalid_numeric_evidence"]
    if isinstance(row["n_peaks"], bool) or not isinstance(row["n_peaks"], int) or row["n_peaks"] < 0:
        return ["invalid_numeric_evidence"]
    for key in ("unique_raw", "unique_snapped", "nearest_peak_changed"):
        if not isinstance(row[key], bool):
            return ["invalid_boolean_evidence"]
    if row["material_fraction"] < PROTOCOL["point_audit"]["minimum_material_fraction"]:
        reasons.append("insufficient_material_support")
    if row["n_peaks"] < 1:
        reasons.append("no_detected_crest")
    if not (row["unique_raw"] and row["unique_snapped"]):
        reasons.append("ambiguous_nearest_crest")
    if row["nearest_peak_changed"]:
        reasons.append("detected_nearest_crest_changed")
    tolerance = PROTOCOL["point_audit"]["maximum_same_crest_distance_increase_vox"]
    if row["after_same_peak"] - row["raw_nearest"] > tolerance:
        reasons.append("same_crest_distance_increased")
    # If the same crest is nearest after the move, the two stored distances
    # must describe the same scalar. This small tolerance checks serialization,
    # not physical uncertainty or anatomical accuracy.
    if not row["nearest_peak_changed"] and not math.isclose(row["after_same_peak"], row["snapped_nearest"], rel_tol=1e-9, abs_tol=1e-8):
        reasons.append("inconsistent_stored_distances")
    return reasons


def audit_points(records):
    if not isinstance(records, list):
        return dict(status="insufficient_evidence", sites=None, audit_pass=None,
                    reason="Expected per-site point-metrics.json list; aggregate statistics cannot identify which proposals satisfy the conjunction")
    counts = Counter()
    passed = 0
    for row in records:
        reasons = audit_point(row)
        passed += not reasons
        counts.update(reasons)
    return dict(status="audited" if records else "insufficient_evidence",
                sites=len(records), audit_pass=passed, hold_for_review=len(records)-passed,
                nonexclusive_reason_counts=dict(sorted(counts.items())),
                meaning="Audit pass preserves a detected-crest association within the fixed scoring tolerance; it does not identify the physical sheet or authorize a mesh edit",
                evaluation_status="Exploratory on previously summarized Codex0813 sites; requires fresh independently reviewed control")


def audit_interval(doc):
    """Conditional 1-D nearest-crest invariance; error bounds are never guessed."""
    absent = dict(status="insufficient_evidence", reason="Complete sorted crest locations, explicit error bounds, their basis and a finite proposed displacement are required")
    if not isinstance(doc, dict):
        return absent
    peaks, errors = doc.get("peaks_vox"), doc.get("location_error_bounds_vox")
    proposal = doc.get("proposed_displacement_vox")
    if (not isinstance(peaks, list) or not isinstance(errors, list) or len(peaks) < 3
            or len(peaks) != len(errors) or not all(finite(v) for v in peaks)
            or not all(finite(v) and v >= 0 for v in errors) or not finite(proposal)
            or not isinstance(doc.get("error_bound_basis"), str) or not doc["error_bound_basis"]):
        return absent
    if any(a >= b for a, b in zip(peaks, peaks[1:])):
        return dict(status="insufficient_evidence", reason="Crest offsets must be strictly increasing along the unchanged original normal")
    nearest = min(range(len(peaks)), key=lambda i: abs(peaks[i]))
    if any(math.isclose(abs(p), abs(peaks[nearest]), rel_tol=0, abs_tol=1e-9) for i, p in enumerate(peaks) if i != nearest):
        return dict(status="hold_for_review", reason="Original nearest detected crest is tied")
    if nearest in (0, len(peaks)-1):
        return dict(status="hold_for_review", reason="A neighboring crest is unobserved on one side; no bounded cell can be asserted")
    i = nearest
    if peaks[i-1]+errors[i-1] >= peaks[i]-errors[i] or peaks[i]+errors[i] >= peaks[i+1]-errors[i+1]:
        return dict(status="hold_for_review", reason="Crest ordering is ambiguous under supplied error bounds")
    left = (peaks[i-1]+peaks[i]+errors[i-1]+errors[i])/2
    right = (peaks[i]+peaks[i+1]-errors[i]-errors[i+1])/2
    if not left < 0 < right:
        return dict(status="hold_for_review", reason="Original nearest-crest association is not robust under supplied error bounds")
    preserved = left < proposal < right
    return dict(status="conditional_interval_pass" if preserved else "hold_for_review",
                reason="Proposal is strictly inside the supplied-bound nearest-crest cell" if preserved else "Proposal reaches or crosses a possible nearest-crest boundary",
                admissible_interval_vox=[left, right],
                empirically_validated_error_bounds=False, anatomical_sheet_identity_certified=False,
                limitation="Conditional on complete neighboring crests and honest location bounds; does not establish sheet identity, ink depth or lack of tangential displacement")


def validate_pair(doc):
    """Validate declared metadata, without claiming its files are authentic."""
    errors = []
    if not isinstance(doc, dict):
        return dict(status="invalid_manifest", errors=["manifest_must_be_object"])
    raw, adjusted = doc.get("raw"), doc.get("adjusted")
    if not isinstance(raw, dict) or not isinstance(adjusted, dict):
        return dict(status="invalid_manifest", errors=["two_arm_objects_required"])
    shared_keys = ("frame", "chart", "model_sha256", "preprocess", "inference", "depth_offsets_um", "normal_convention")
    for key in shared_keys:
        if key not in raw or key not in adjusted:
            errors.append("missing_shared_" + key)
        elif raw[key] != adjusted[key]:
            errors.append("mismatched_" + key)
    for name, arm in (("raw", raw), ("adjusted", adjusted)):
        frame = arm.get("frame", {})
        if not isinstance(frame, dict) or not all(k in frame for k in ("acquisition", "volume_metadata_sha256", "axes", "units", "level", "voxel_um_xyz", "origin_native_xyz")):
            errors.append(name + "_incomplete_native_frame")
        else:
            if frame["axes"] != "XYZ" or frame["units"] != "native_voxels" or type(frame["level"]) is not int or frame["level"] != 0:
                errors.append(name + "_requires_explicit_native_xyz_frame")
            for field in ("voxel_um_xyz", "origin_native_xyz"):
                vals = frame[field]
                if not isinstance(vals, list) or len(vals) != 3 or not all(finite(v) for v in vals):
                    errors.append(name + "_invalid_" + field)
                elif field == "voxel_um_xyz" and not all(v > 0 for v in vals):
                    errors.append(name + "_nonpositive_voxel_size")
            if not isinstance(frame["acquisition"], str) or not frame["acquisition"]:
                errors.append(name + "_missing_acquisition")
            if not isinstance(frame["volume_metadata_sha256"], str) or not SHA.fullmatch(frame["volume_metadata_sha256"]):
                errors.append(name + "_invalid_volume_metadata_hash")
        chart = arm.get("chart", {})
        if not isinstance(chart, dict) or not all(k in chart for k in ("reference_sha256", "shape_yx", "pitch_um_yx", "pixel_center_convention")):
            errors.append(name + "_incomplete_chart")
        else:
            if not isinstance(chart["reference_sha256"], str) or not SHA.fullmatch(chart["reference_sha256"]):
                errors.append(name + "_invalid_chart_reference_hash")
            if not isinstance(chart["shape_yx"], list) or len(chart["shape_yx"]) != 2 or not all(type(v) is int and v > 0 for v in chart["shape_yx"]):
                errors.append(name + "_invalid_chart_shape")
            if not isinstance(chart["pitch_um_yx"], list) or len(chart["pitch_um_yx"]) != 2 or not all(finite(v) and v > 0 for v in chart["pitch_um_yx"]):
                errors.append(name + "_invalid_chart_pitch")
            if not isinstance(chart["pixel_center_convention"], str) or not chart["pixel_center_convention"]:
                errors.append(name + "_missing_pixel_center_convention")
        for field in ("mesh_sha256", "model_sha256", "normal_field_sha256", "input_sha256", "validity_mask_sha256"):
            if not isinstance(arm.get(field), str) or not SHA.fullmatch(arm[field]):
                errors.append(name + "_invalid_" + field)
        offsets = arm.get("depth_offsets_um")
        if not isinstance(offsets, list) or not offsets or not all(finite(v) for v in offsets) or len(set(offsets)) != len(offsets):
            errors.append(name + "_invalid_depth_offsets")
        for field in ("preprocess", "inference"):
            if not isinstance(arm.get(field), dict) or not arm[field]:
                errors.append(name + "_missing_" + field + "_settings")
        if not isinstance(arm.get("normal_convention"), str) or not arm["normal_convention"]:
            errors.append(name + "_missing_normal_convention")
    common_mask = doc.get("common_valid_mask_sha256")
    if not isinstance(common_mask, str) or not SHA.fullmatch(common_mask):
        errors.append("common_valid_mask_hash_required")
    policy = doc.get("normal_policy")
    if policy == "fixed_original_field":
        if raw.get("normal_field_sha256") != adjusted.get("normal_field_sha256"):
            errors.append("fixed_original_normal_fields_differ")
        scope = "geometry_change_with_declared_fixed_normal_field"
    elif policy == "recomputed_per_mesh":
        scope = "geometry_and_normal_change; not a pure depth-placement comparison"
    else:
        errors.append("explicit_normal_policy_required")
        scope = None
    basis = doc.get("comparison_basis")
    if basis == "common_chart":
        correspondence = "same chart coordinates only; moved meshes generally sample different native CT locations"
        if doc.get("claims_same_physical_positions") is not False:
            errors.append("common_chart_cannot_claim_same_physical_positions")
    elif basis == "fixed_native_anchors":
        anchor = doc.get("native_anchors")
        if not isinstance(anchor, dict) or not isinstance(anchor.get("sha256"), str) or not SHA.fullmatch(anchor["sha256"]) or anchor.get("frame") != raw.get("frame"):
            errors.append("common_native_anchor_hash_and_frame_required")
        for name, arm in (("raw", raw), ("adjusted", adjusted)):
            if not isinstance(arm.get("anchor_to_chart_map_sha256"), str) or not SHA.fullmatch(arm["anchor_to_chart_map_sha256"]):
                errors.append(name + "_native_anchor_to_chart_mapping_required")
        correspondence = "declared shared native anchors and separate chart mappings; mappings still require physical validation"
    else:
        errors.append("explicit_comparison_basis_required")
        correspondence = None
    return dict(status="invalid_manifest" if errors else "declared_metadata_consistent", errors=sorted(set(errors)),
                comparison_scope=scope, correspondence=correspondence,
                verified_file_contents=False, anatomical_alignment_certified=False,
                meaning="Metadata self-consistency only; hashes are declared, masks are not inspected, and UV area is not measured")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("points", "pair", "interval"))
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    doc = json.loads(payload)
    result = {"points": audit_points, "pair": validate_pair, "interval": audit_interval}[args.mode](doc)
    result["input_sha256"] = hashlib.sha256(payload).hexdigest()
    result["protocol_sha256"] = hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] in ("audited", "declared_metadata_consistent", "conditional_interval_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
