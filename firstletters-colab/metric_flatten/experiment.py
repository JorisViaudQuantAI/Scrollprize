#!/usr/bin/env python3
"""One frozen, CPU-only geometry comparison on the published PHerc. 306B control.

The first run acquires only the 17 small pinned public geometry/code assets.
An explicit --reuse-cache replay verifies every cached asset without network.
No CT, image of unread text, model weights, training or CNN execution is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np
import PIL
import scipy
from scipy.interpolate import LinearNDInterpolator

from flatten import solve_mesh, write_mesh
import reference
from topology import rasterize_triangles, triangle_metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def percentiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return [float(v) for v in np.percentile(values, [5, 50, 95])] if len(values) else None


def box(u, v, bounds, *, upper_inclusive=False):
    a, b, c, d = bounds
    return (u >= a) & ((u <= b) if upper_inclusive else (u < b)) & (v >= c) & ((v <= d) if upper_inclusive else (v < d))


def four_corner(valid):
    return valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]


def grid_metrics(p, spacing):
    # Same three-corner tangents as the publisher. Caller selects either the
    # original support or a stricter common four-corner population explicitly.
    du = (p[:, 1:] - p[:, :-1])[:-1]
    dv = (p[1:] - p[:-1])[:, :-1]
    un, vn = np.linalg.norm(du, axis=-1), np.linalg.norm(dv, axis=-1)
    cosine = np.sum(du * dv, axis=-1) / np.maximum(un * vn, 1e-9)
    return {"stretch_u": un / spacing, "stretch_v": vn / spacing,
            "shear_deg": np.degrees(np.arcsin(np.clip(cosine, -1, 1))),
            "area_ratio": np.linalg.norm(np.cross(du, dv), axis=-1) / spacing ** 2}


def grid_summary(metrics, mask):
    if not mask.any():
        return {"cells": 0}
    su, sv, sh = (metrics[k][mask] for k in ("stretch_u", "stretch_v", "shear_deg"))
    return {"cells": int(mask.sum()),
            **{f"{k}_p5_p50_p95": percentiles(value[mask]) for k, value in metrics.items()},
            "median_abs_v_stretch_error": float(np.median(np.abs(sv - 1))),
            "p95_abs_shear_degrees": float(np.percentile(np.abs(sh), 95)),
            "either_axis_outside_5pct_fraction": float(np.mean((np.abs(su - 1) > .05) | (np.abs(sv - 1) > .05))),
            "either_axis_outside_10pct_or_shear_10deg_fraction": float(np.mean((np.abs(su - 1) > .1) | (np.abs(sv - 1) > .1) | (np.abs(sh) > 10)))}


def triangle_summary(metrics, pitch_mm):
    singular = metrics["singular_values"]
    finite = np.isfinite(singular).all(1)
    bad = ~finite | np.any(np.abs(singular - 1) > .05, axis=1)
    area = metrics["source_3d_area"]
    return {**metrics["counts"], "metric_nonfinite_triangles": int((~finite).sum()),
            "singular_max_p5_p50_p95": percentiles(singular[:, 0]),
            "singular_min_p5_p50_p95": percentiles(singular[:, 1]),
            "either_singular_value_outside_5pct_or_nonfinite_fraction": float(bad.mean()),
            "source_area_weighted_outside_5pct_or_nonfinite_fraction": float(area[bad].sum() / area.sum()),
            "total_original_source_triangle_area_mm2": float(area.sum() * pitch_mm ** 2),
            "positive_uv_area_grid_units": float(metrics["signed_uv_area"][metrics["positive_uv"]].sum()),
            "negative_uv_absolute_area_grid_units": float(-metrics["signed_uv_area"][metrics["negative_uv"]].sum())}


def assert_frozen(commit, protocol):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use a complete immutable protocol commit SHA")
    result = {}
    for name in protocol["frozen_files"]:
        local = HERE / name
        relative = local.relative_to(ROOT).as_posix()
        frozen = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT,
                                check=True, capture_output=True).stdout
        if frozen != local.read_bytes():
            raise ValueError(f"Working source differs from freeze: {name}")
        result[name] = digest(local)
    if result["source_manifest.json"] != protocol["source_manifest_sha256"]:
        raise ValueError("Protocol/source manifest mismatch")
    return result


def execute(args):
    protocol = json.loads((HERE / "protocol.json").read_text())
    code_hashes = assert_frozen(args.freeze_commit, protocol)
    out = reference._outside_git(args.out)
    if not out.parent.is_dir():
        raise ValueError("Output parent must already exist")
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {"schema": "scrollprize.metric-flatten.result.v1", "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_commit": args.freeze_commit, "code_sha256": code_hashes,
              "design": protocol["design"], "gates": {},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "scipy": scipy.__version__, "pillow": PIL.__version__, "requested_cpu_threads": 4},
              "CT_requests": 0, "CNN_forwards": 0, "new_reading_claim": False,
              "network_this_run": {"requests": 0, "body_bytes": 0}, "stages": []}
    dump(out / "protocol.json", protocol)

    def checkpoint(stage):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        elapsed = time.monotonic() - started
        entry = {"stage": stage, "wall_seconds": elapsed, "peak_rss_bytes": peak}
        report["stages"].append(entry)
        dump(out / "progress.json", report)
        print(json.dumps(entry), flush=True)
        if peak > protocol["budgets"]["peak_rss_bytes_soft_at_stage_boundaries"]:
            raise MemoryError("Declared stage-boundary RSS budget exceeded")
        if elapsed > protocol["budgets"]["wall_seconds_soft"]:
            raise TimeoutError("Declared wall budget exceeded")

    def alarm(_signal, _frame):
        raise TimeoutError("Declared wall budget exceeded")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(protocol["budgets"]["wall_seconds_soft"])
    try:
        if args.reuse_cache:
            manifest = reference._manifest()
            for entry in manifest["files"]:
                reference._verified_bytes(args.cache, entry["path"])
            report["acquisition"] = {"mode": "verified cache reuse", "verified_files": len(manifest["files"]),
                                     "verified_asset_bytes": sum(x["bytes"] for x in manifest["files"])}
        else:
            acquisition = reference.prepare(args.cache)
            report["acquisition"] = acquisition
            report["network_this_run"] = {k: acquisition[k] for k in ("requests", "body_bytes")}
        checkpoint("source_verified")
        a, metadata = reference.reconstruct_touchup(args.cache)
        target = reference.load_published_xyz(args.cache)
        if a.shape != target.shape:
            raise ValueError("Reconstructed and published target shapes differ")
        functions = reference.load_arap_functions(args.cache)
        config = protocol["geometry"]
        spacing = float(config["spacing_voxels"])
        voxel_um = float(config["source_voxel_um"])
        pitch = spacing / (1000.0 / voxel_um)
        jj, ii = np.meshgrid(np.arange(a.shape[1]), np.arange(a.shape[0]))
        u = config["carrier_origin_mm"][0] + jj * pitch
        v = config["carrier_origin_mm"][1] + ii * pitch
        reg = np.isfinite(a).all(-1) & box(u, v, config["region_mm"], upper_inclusive=True)
        anchor = box(u, v, config["anchor_mm"], upper_inclusive=True)
        fix = int(np.argmin(np.abs(u[reg]) + np.abs(v[reg])))
        report["input"] = {"shape": list(a.shape), "valid_before_nodes": int(np.isfinite(a).all(-1).sum()),
                           "selected_region_nodes": int(reg.sum()), "neutral_metadata_adapter": metadata["numeric_adapter"]}
        checkpoint("touchup_reconstructed")
        solution = solve_mesh(a, spacing, functions, region_mask=reg, anchor_mask=anchor,
                              iterations=config["iterations"], fold_weight=config["fold_weight"],
                              fold_threshold=config["fold_threshold"], fix_index=fix)
        uv, xyz, faces = (solution[k] for k in ("uv", "xyz", "triangles"))
        initial = solution["uv_initial"]
        report["solver"] = solution["solver_summary"]
        checkpoint("published_arap_solved")
        b = np.full(a.shape, np.nan)
        b[reg] = LinearNDInterpolator(uv, xyz)(initial)
        b = b.astype(np.float32).astype(np.float64)
        vb, vt = np.isfinite(b).all(-1), np.isfinite(target).all(-1)
        common = vb & vt
        maximum = float(np.max(np.abs(b[common] - target[common]))) if common.any() else None
        gate = protocol["gates"]["replay"]
        replay_ok = bool(np.array_equal(vb, vt) and maximum is not None and maximum <= gate["max_absolute_xyz_error_voxels"])
        report["gates"]["replay"] = {"pass": replay_ok, "support_mismatch_nodes": int((vb != vt).sum()),
            "max_absolute_xyz_error_voxels": maximum, "max_absolute_xyz_error_um": maximum * voxel_um if maximum is not None else None,
            "exact_float32_values_on_common_support": bool(np.array_equal(b[common], target[common]))}
        np.savez_compressed(out / "replay_geometry.npz", A=a, B=b, published=target, uv=uv, uv_initial=initial,
                            source_xyz=xyz, source_triangles=faces, source_ij=solution["source_ij"], region=reg)
        checkpoint("published_target_compared")
        if not replay_ok:
            report["status"] = "replay_failed_stop"
            report["decision"] = "No C evaluation or downstream adoption; frozen replay gate failed"
            return report

        c_result = rasterize_triangles(uv, xyz, faces, a.shape[:2], query_mask=reg)
        raw_c = c_result["xyz"]
        c = raw_c.astype(np.float32).astype(np.float64)
        vc = c_result["valid"]
        np.savez_compressed(out / "C_rasterization.npz", raw_xyz=raw_c, float32_xyz=c.astype(np.float32),
            **{k: c_result[k] for k in ("valid", "source_triangle_id", "barycentric_weights", "reason_bits", "coverage_count")})
        report["C_rasterizer"] = {**c_result["counts"], "reason_names": c_result["reason_names"],
            "source_component_query_incidence_count": c_result["source_component_query_incidence_count"].tolist(),
            "source_component_output_pixel_count": c_result["source_component_output_pixel_count"].tolist()}
        rounding = np.linalg.norm(c[vc] - raw_c[vc], axis=1)
        report["C_float32_serialization"] = {"distance_voxels_p5_p50_p95": percentiles(rounding),
            "distance_voxels_max": float(rounding.max()) if len(rounding) else None,
            "distance_um_max": float(rounding.max() * voxel_um) if len(rounding) else None,
            "not_part_of_float64_barycentric_tolerance": True}
        r0, r1 = config["claim_crop_rows_half_open"]
        c0, c1 = config["claim_crop_columns_half_open"]
        if r1 > a.shape[0] or c1 > a.shape[1]:
            raise ValueError("Published crop exceeds verified geometry shape")
        crop = np.zeros(a.shape[:2], bool)
        crop[r0:r1, c0:c1] = True
        claim_baseline_count = int((vb & crop).sum())
        retained = {"whole": int((vb & vc).sum()) / int(vb.sum()),
                    "published_claim": int((vb & vc & crop).sum()) / claim_baseline_count if claim_baseline_count else None}
        rg = protocol["gates"]["retention"]
        report["gates"]["retention"] = {"pass": retained["whole"] >= rg["whole_baseline_support_fraction_min"] and
            retained["published_claim"] is not None and retained["published_claim"] >= rg["published_claim_support_fraction_min"],
            "baseline_support_retained": retained,
            "B_valid_nodes": int(vb.sum()), "C_valid_nodes": int(vc.sum()), "B_only_nodes": int((vb & ~vc).sum()),
            "C_only_nodes": int((vc & ~vb).sum()), "claim_B_valid_nodes": int((vb & crop).sum()),
            "claim_C_valid_nodes": int((vc & crop).sum())}
        ids, weights = c_result["source_triangle_id"][vc], c_result["barycentric_weights"][vc]
        reconstructed = np.einsum("ij,ijk->ik", weights, xyz[faces[ids]])
        max_provenance_error = float(np.max(np.abs(reconstructed - raw_c[vc]))) if len(ids) else None
        pm = triangle_metrics(uv, xyz / spacing, faces)
        pg = protocol["gates"]["provenance"]
        provenance_ok = bool(len(ids) and (c_result["reason_bits"][vc] == 0).all() and pm["positive_uv"][ids].all() and
            np.all(weights >= -1e-9) and np.all(np.abs(weights.sum(1) - 1) <= pg["barycentric_sum_absolute_tolerance"]) and
            max_provenance_error <= pg["accepted_query_xyz_error_voxels_max"])
        report["gates"]["provenance"] = {"pass": provenance_ok, "accepted_queries": len(ids),
            "max_absolute_xyz_reconstruction_error_voxels": max_provenance_error,
            "scope": "Float64 accepted integer target samples before TIFF rounding; not a global continuous injectivity proof"}
        common_bc = vb & vc
        distances = np.linalg.norm(b[common_bc] - c[common_bc], axis=1)
        report["B_C_coordinate_difference"] = {"common_nodes": int(common_bc.sum()),
            "distance_voxels_p5_p50_p95": percentiles(distances),
            "distance_voxels_max": float(distances.max()) if len(distances) else None,
            "distance_um_p5_p50_p95": percentiles(distances * voxel_um),
            "distance_um_max": float(distances.max() * voxel_um) if len(distances) else None}
        checkpoint("original_triangle_rasterized")

        arms = {"A": a, "B": b, "C": c}
        gm = {key: grid_metrics(value, spacing) for key, value in arms.items()}
        all_four = np.logical_and.reduce([four_corner(np.isfinite(value).all(-1)) for value in arms.values()])
        uc, vv = u[:-1, :-1] + pitch / 2, v[:-1, :-1] + pitch / 2
        report["common_four_corner_metrics"] = {}
        report["publisher_formula_three_corner_diagnostics"] = {}
        for name, bounds in protocol["metrics"]["diagnostic_regions_mm"].items():
            region_cells = box(uc, vv, bounds)
            report["common_four_corner_metrics"][name] = {key: grid_summary(value, all_four & region_cells) for key, value in gm.items()}
            report["publisher_formula_three_corner_diagnostics"][name] = {key: grid_summary(value, region_cells &
                np.isfinite(value["stretch_u"]) & np.isfinite(value["stretch_v"])) for key, value in gm.items()}
        right = report["common_four_corner_metrics"]["right_group"]
        gg = protocol["gates"]["right_group"]
        right_ok = bool(right["C"]["cells"] >= gg["minimum_common_four_corner_cells"] and
            right["C"]["median_abs_v_stretch_error"] <= right["A"]["median_abs_v_stretch_error"] * gg["median_absolute_v_stretch_error_ratio_max"] and
            right["C"]["either_axis_outside_5pct_fraction"] <= gg["either_axis_outside_5pct_fraction_max"] and
            right["C"]["p95_abs_shear_degrees"] <= gg["p95_absolute_shear_degrees_max"])
        report["gates"]["right_group"] = {"pass": right_ok, "metrics": right,
            "median_abs_v_error_ratio_C_over_A": right["C"].get("median_abs_v_stretch_error", 0) / right["A"].get("median_abs_v_stretch_error", 1) if right["A"].get("median_abs_v_stretch_error", 0) else None}
        before_tri = triangle_metrics(initial, xyz / spacing, faces)
        report["full_original_triangle_population"] = {"before": triangle_summary(before_tri, pitch),
            "after": triangle_summary(pm, pitch), "same_triangles_before_after": True,
            "B_and_C_share_exact_UV_solution_and_optimization_mesh": True}
        np.savez_compressed(out / "triangle_metrics.npz", triangles=faces,
            before_singular=before_tri["singular_values"], after_singular=pm["singular_values"],
            before_signed_uv_area=before_tri["signed_uv_area"], after_signed_uv_area=pm["signed_uv_area"],
            original_3d_area=before_tri["source_3d_area"])

        # A support identity check at the published target's own grid vertices.
        _, identity_faces = functions["tris_of"](vt)
        identity_uv = np.column_stack((jj[vt], ii[vt])).astype(float)
        identity = rasterize_triangles(identity_uv, target[vt], identity_faces, a.shape[:2], query_mask=vt)
        incident = np.zeros(int(vt.sum()), bool)
        incident[np.unique(identity_faces)] = True
        identity_expected = np.zeros_like(vt)
        identity_expected[vt] = incident
        identity_common = identity["valid"] & identity_expected
        identity_error = float(np.max(np.abs(identity["xyz"][identity_common] - target[identity_common]))) if identity_common.any() else None
        identity_ok = bool(identity_error is not None and np.array_equal(identity["valid"], identity_expected) and
                           identity_error <= protocol["gates"]["identity"]["max_absolute_xyz_error_voxels"])
        report["gates"]["identity"] = {"pass": identity_ok, "max_absolute_xyz_error_voxels": identity_error,
            "valid_nodes_without_original_triangle_incidence": int((~incident).sum()),
            "expected_valid_nodes": int(identity_expected.sum()), "actual_valid_nodes": int(identity["valid"].sum())}
        checkpoint("geometry_and_identity_measured")

        # Deliberately missing support on real public geometry, not papyrus damage.
        half_width = protocol["gates"]["support_perturbation"]["remove_source_square_side_vertices"] // 2
        centre = solution["source_ij"][fix]
        removed = np.all(np.abs(solution["source_ij"] - centre) <= half_width, axis=1)
        keep_faces = ~removed[faces].any(1)
        hole_result = rasterize_triangles(uv, xyz, faces[keep_faces], a.shape[:2], query_mask=reg)
        hole_baseline = np.full(a.shape, np.nan)
        hole_baseline[reg] = LinearNDInterpolator(uv[~removed], xyz[~removed])(initial)
        hole_queries = np.zeros_like(reg)
        hole_queries[vc] = removed[faces[ids]].all(1) & (weights > 1e-6).all(1)
        hole_count = int(hole_queries.sum())
        baseline_fills = int(np.isfinite(hole_baseline[hole_queries]).all(-1).sum())
        c_fills = int(hole_result["valid"][hole_queries].sum())
        hg = protocol["gates"]["support_perturbation"]
        hole_ok = bool(hole_count >= hg["minimum_hole_queries"] and baseline_fills == hole_count and c_fills == 0)
        report["gates"]["support_perturbation"] = {"pass": hole_ok, "removed_source_vertices": int(removed.sum()),
            "removed_source_triangles": int((~keep_faces).sum()), "interior_test_queries": hole_count,
            "Delaunay_filled_interior_queries": baseline_fills, "original_triangle_filled_interior_queries": c_fills,
            "interpretation": hg["interpretation"]}
        np.savez_compressed(out / "support_perturbation.npz", removed_source_vertices=removed,
            kept_source_triangles=keep_faces, hole_queries=hole_queries, B_xyz=hole_baseline,
            C_xyz=hole_result["xyz"], C_valid=hole_result["valid"], C_reason_bits=hole_result["reason_bits"])
        checkpoint("designed_support_removal_measured")

        base_provenance = {"source_ij": solution["source_ij"], "source_xyz": xyz,
                           "original_source_triangles": faces, "parameter_uv": uv}
        c_provenance = {**base_provenance, **{k: c_result[k] for k in
            ("source_triangle_id", "barycentric_weights", "valid", "reason_bits", "coverage_count")}}
        for name, coordinates, valid, provenance in (
            ("A_tifxyz", a, reg, {**base_provenance, "parameter_uv": initial}),
            ("B_tifxyz", b, vb, {**base_provenance, "resampler": np.asarray("Delaunay, not original triangles")}),
            ("C_tifxyz", c, vc, c_provenance)):
            write_mesh(out / name, coordinates, valid, spacing, voxel_um,
                       provenance=provenance, summary={"arm": name, "protocol_commit": args.freeze_commit,
                       "source_revision": reference.REVISION, "code_sha256": code_hashes,
                       "all_gates": report["gates"], "reason_names": c_result["reason_names"]})
        checkpoint("standard_mesh_outputs_written")
        all_pass = all(g["pass"] for g in report["gates"].values())
        report["status"] = "all_geometry_gates_pass" if all_pass else "geometry_gate_failure"
        report["decision"] = ("Permit one bounded matched public-known-text CT/ink comparison; no reading or ink-gain claim"
                              if all_pass else "Retain failed gates; block unattended adoption")
        return report
    except Exception as error:
        report["status"] = "execution_failed_stop"
        report["error_type"] = type(error).__name__
        # Local tracebacks remain local; source/working paths never enter a
        # public receipt through third-party exception strings.
        report["decision"] = "Incomplete experiment; no downstream adoption"
        failed = Path(args.cache) / "prepare_failed.json"
        if failed.exists() and not args.reuse_cache:
            failure = json.loads(failed.read_text())
            report["network_this_run"] = {k: failure[k] for k in ("requests", "body_bytes")}
        raise
    finally:
        signal.alarm(0)
        report["wall_seconds"] = time.monotonic() - started
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["artifacts"] = {p.relative_to(out).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
                               for p in sorted(out.rglob("*")) if p.is_file() and p.name not in ("result.json", "progress.json")}
        dump(out / "result.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    report = execute(args)
    print(json.dumps({"status": report["status"], "decision": report["decision"],
                      "gates": {k: v["pass"] for k, v in report["gates"].items()},
                      "wall_seconds": report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
