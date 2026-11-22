#!/usr/bin/env python3
"""Adaptive follow-up: 32 native TIFF/Zarr integers, then fixed raw-grid anatomy.

The failed surface-TIFF experiment stays failed. This imports the pinned original
sampling/metric/acquisition helpers and uses its unchanged scanner-grid design.
No photographic labels, model, coordinate fitting or best-offset choice.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import struct
import sys
import time

CPU_START = time.process_time()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_original(directory, protocol):
    runner = directory / "frag5_native_pair_stage1.py"
    plan_file = directory / "frag5_native_pair_stage1_protocol.json"
    if sha(runner) != protocol["original_runner_sha256"] or sha(plan_file) != protocol["original_protocol_sha256"]:
        raise ValueError("Original frozen scientific code/protocol changed")
    spec = importlib.util.spec_from_file_location("fixed_frag5_stage1", runner)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    plan = json.loads(plan_file.read_bytes())
    original.check_plan(plan)
    return original, plan


def check_protocol(protocol, original, plan):
    np = original.np
    if protocol["schema"] != "frag5-adaptive-native-source-and-raw-grid-v1":
        raise ValueError("Wrong follow-up protocol")
    if protocol["gate"] != {"corners": 32, "native_intensity_bytes": 64,
                            "maximum_absolute_integer_difference": 0,
                            "anatomy_only_if_all_32_equal": True,
                            "legacy_surface_contract_status": "failed_and_not_retested"}:
        raise ValueError("Unexpected source gate")
    expected = []
    native_base = protocol["native_volume_base_url"]
    for item in plan["centers"]:
        p = np.asarray(item["fine_xyz"])
        lo, hi = np.floor(p).astype(np.int64), np.ceil(p).astype(np.int64)
        if np.any(hi - lo != 1):
            raise ValueError("Exactly eight distinct integer corners required per center")
        for z, y in itertools.product((int(lo[2]), int(hi[2])), (int(lo[1]), int(hi[1]))):
            offset = 368 + 2 * (y * 4420 + int(lo[0]))
            expected.append({"center_id": item["id"], "slice_z": z,
                             "url": native_base + f"{z:04}.tif",
                             "xyz": [[int(lo[0]), y, z], [int(hi[0]), y, z]],
                             "byte_range": [offset, offset + 3]})
    if expected != protocol["corner_rows"]:
        raise ValueError("32 fixed source voxels do not reproduce from original PPM centers")
    if len({tuple(p) for r in expected for p in r["xyz"]}) != 32:
        raise ValueError("Duplicate source corners")
    if len(protocol["cached_fine_chunks"]) != 6:
        raise ValueError("Exactly the six old fine chunks must be pinned and reused")
    if protocol["cumulative_caps"] != {"source_chunk_body_bytes": 268435456,
                                      "tiff_metadata_body_bytes": 2097152,
                                      "decoded_unique_chunks": 64,
                                      "cpu_seconds": 3600, "threads": 4}:
        raise ValueError("Cumulative budgets changed")
    if protocol["prior_charges"]["source_chunk_bytes"] != 24571323:
        raise ValueError("Old raw source budget was not retained")
    if sum(c["bytes"] for c in protocol["cached_fine_chunks"]) != 24571323:
        raise ValueError("Old raw byte ledger does not balance")
    expected_keys = {tuple(k) for k in plan["support_plan"]["identity_fine_chunk_keys_zyx"]}
    if {tuple(c["key_zyx"]) for c in protocol["cached_fine_chunks"]} != expected_keys:
        raise ValueError("Cached fine support differs from original experiment")
    all_source_urls = {plan["volumes"][name]["base_url"] + "/" + "/".join(map(str, k))
                       for name in ("fine", "coarse") for k in plan["support_plan"][name + "_chunk_keys_zyx"]}
    if set(protocol["all_source_expected_heads"]) != all_source_urls:
        raise ValueError("All fourteen raw objects must be re-HEADed before mixing sources")
    expected_slices = {r["slice_z"] for r in expected}
    if len(protocol["native_slices"]) != 8 or {r["slice_z"] for r in protocol["native_slices"]} != expected_slices:
        raise ValueError("Exactly the eight fixed native TIFF headers are required")
    for source in protocol["native_slices"]:
        if [r["range"] for r in source["header_ranges"]] != [[0, 7], [8, 9], [10, 217]]:
            raise ValueError("Unexpected native header ranges")


def verify_prior(prior, protocol):
    for name, expected in protocol["prior_receipt_sha256"].items():
        if sha(prior / name) != expected:
            raise ValueError("Prior failed-result provenance changed")
    recorded = {r["url"]: r for r in json.loads((prior / "acquisition.json").read_bytes())["responses"]
                if r["category"] == "source_chunk"}
    for item in protocol["cached_fine_chunks"]:
        path = (prior / item["cache_file"]).resolve()
        if not path.is_relative_to(prior.resolve()) or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("Pinned old fine chunk missing or changed")
        old = recorded[item["url"]]
        if (old["sha256"] != item["sha256"] or old["expected_bytes"] != item["bytes"]
                or old["cache_file"] != item["cache_file"] or old["status"] != "complete"):
            raise ValueError("Chunk hash/URL/size does not match canonical acquisition provenance")


def inject_cached_fine(volume, prior, protocol, original):
    for item in protocol["cached_fine_chunks"]:
        raw = (prior / item["cache_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"] or len(raw) < 16:
            raise ValueError("Cached source changed before decode")
        header = struct.unpack("<4B3I", raw[:16])
        if header[4] != original.CHUNK_BYTES or header[6] != len(raw):
            raise ValueError("Unexpected old chunk Blosc size")
        decoded = volume.codec.decode(raw)
        if len(decoded) != original.CHUNK_BYTES:
            raise ValueError("Decoded old chunk size changed")
        volume.chunks[tuple(item["key_zyx"])] = original.np.frombuffer(decoded, dtype="<u2").reshape(original.CHUNK_SHAPE)


def all_corners_equal(rows):
    return len(rows) == 32 and all(r["native_uint16"] == r["zarr_uint16"] for r in rows)


def native_center_diagnostic(original, plan, rows, prior_identity):
    """Adaptive secondary report only; this never changes the 32-integer gate."""
    np = original.np
    old_by_id = {r["id"]: r for r in prior_identity["profiles"]}
    output = []
    for item in plan["centers"]:
        p = np.asarray(item["fine_xyz"], dtype=np.float64)
        lo = np.floor(p).astype(np.int64)
        cube = np.empty((2, 2, 2), dtype=np.uint16)
        for row in rows:
            if row["center_id"] == item["id"]:
                x, y, z = np.asarray(row["xyz"]) - lo
                cube[z, y, x] = row["native_uint16"]
        value = float(original.Region(cube, lo).sample(p[None, :])[0])
        rounded = int(np.rint(value))
        old = old_by_id[item["id"]]
        surface = int(old["tiff_values"][32])
        previous = int(old["primary_absolute_errors"][32])
        output.append({"id": item["id"], "native_corner_trilinear": value,
                       "native_corner_trilinear_rounded": rounded,
                       "previous_surface_tiff_k32": surface,
                       "native_minus_previous_surface_unrounded": value - surface,
                       "native_vs_previous_surface_rounded_absolute_error": abs(rounded - surface),
                       "previous_zarr_trilinear_center": old["fine_trilinear_values"][32],
                       "previous_zarr_vs_surface_rounded_absolute_error": previous,
                       "reproduces_previous_rounded_center_error": abs(rounded - surface) == previous})
    return {"adaptive_secondary_diagnostic_only": True, "affects_gate": False, "centers": output}


def anatomy(original, plan, fine, coarse, acquisition, out):
    """Same 33³ grids, 125 samples, 27 shifts and metrics as frozen stage1."""
    np = original.np
    support = plan["support_plan"]
    fine.load(support["fine_chunk_keys_zyx"])
    coarse.load(support["coarse_chunk_keys_zyx"])
    A = np.asarray(plan["affine_xyz"], dtype=np.float64)
    inverse, translation = np.linalg.inv(A[:3, :3]), A[:3, 3]
    z, y, x = np.meshgrid(np.arange(-16, 17), np.arange(-16, 17), np.arange(-16, 17), indexing="ij")
    offsets = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    quadrature = (np.arange(5, dtype=np.float64) + .5) / 5 - .5
    result = []
    for item, geometry in zip(plan["centers"], support["centers"]):
        acquisition.check_cpu()
        center = np.asarray(geometry["coarse_center_xyz"])
        q = center + offsets
        fine_region = fine.region(geometry["anatomy_fine_box_xyz"])
        fine_point = fine_region.sample((q - translation) @ inverse.T).reshape(33, 33, 33)
        integrated = np.zeros(len(q), dtype=np.float64)
        for u in itertools.product(quadrature, repeat=3):
            acquisition.check_cpu()
            integrated += fine_region.sample((q + u - translation) @ inverse.T)
        integrated = (integrated / 125).reshape(33, 33, 33)
        coarse_region = coarse.region(geometry["anatomy_coarse_box_xyz"])
        measures = []
        for shift in itertools.product((-1, 0, 1), repeat=3):
            native = coarse_region.sample(q + shift).reshape(33, 33, 33)
            for arm, simulated in (("fine_point", fine_point), ("fine_nominal_coarse_footprint", integrated)):
                measures.append({"shift_xyz_coarse_voxels": list(shift), "arm": arm,
                                 "original_transform_primary": shift == (0, 0, 0),
                                 "intensity": original.correlation(simulated, native),
                                 "laplacian_interior": original.correlation(original.laplacian(simulated), original.laplacian(native))})
        arrays = out / f"anatomy_{item['id']}.npz"
        np.savez_compressed(arrays, fine_point=fine_point, fine_nominal_coarse_footprint=integrated,
                            coarse_neighborhood=coarse_region.values,
                            coarse_center_xyz=center, coarse_neighborhood_origin_xyz=coarse_region.origin)
        result.append({"id": item["id"], "descriptive_statistics": {
            "fine_point": original.describe(fine_point), "fine_nominal_coarse_footprint": original.describe(integrated),
            "native_coarse_zero_shift": original.describe(coarse_region.sample(q))},
            "within_fine_point_vs_footprint": original.correlation(fine_point, integrated),
            "all_shifts": measures, "arrays_file": arrays.name, "arrays_sha256": sha(arrays)})
        original.write_json(out / "anatomy_progress.json", result)
        print(json.dumps({"stage": "raw_grid_center_complete", "id": item["id"]}), flush=True)
    return result


def run(protocol, original, plan, prior, out, freeze):
    started_wall = time.monotonic()
    charges = protocol["prior_charges"]
    caps = {"source_chunk_body_bytes": protocol["cumulative_caps"]["source_chunk_body_bytes"] - charges["source_chunk_bytes"],
            "tiff_metadata_body_bytes": protocol["cumulative_caps"]["tiff_metadata_body_bytes"] - charges["tiff_metadata_bytes"],
            "cpu_seconds": protocol["cumulative_caps"]["cpu_seconds"] - charges["reserved_cpu_seconds"], "threads": 4}
    acquisition = original.Acquisition(out, caps, CPU_START)
    fine, coarse = (original.Volume(plan["volumes"][name], acquisition) for name in ("fine", "coarse"))
    try:
        verify_prior(prior, protocol)
        # Decode only the six hash-pinned old fine chunks; there is no new CT GET here.
        inject_cached_fine(fine, prior, protocol, original)
        support = plan["support_plan"]
        remaining_fine = [k for k in support["fine_chunk_keys_zyx"] if tuple(k) not in fine.chunks]
        new_source_urls = [fine.url(k) for k in remaining_fine] + [coarse.url(k) for k in support["coarse_chunk_keys_zyx"]]
        source_urls = [fine.url(k) for k in support["fine_chunk_keys_zyx"]] + [coarse.url(k) for k in support["coarse_chunk_keys_zyx"]]
        native_urls = [s["url"] for s in protocol["native_slices"]]
        metadata_urls = [plan["volumes"][n]["metadata_url"] for n in ("fine", "coarse")]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(acquisition.head, source_urls + native_urls + metadata_urls))
        expected_heads = {s["url"]: s["head"] for s in protocol["native_slices"]}
        expected_heads.update(protocol["all_source_expected_heads"])
        for url, expected in expected_heads.items():
            actual = acquisition.heads[url]
            if actual["length"] != expected["length"] or actual["etag"] != expected["etag"]:
                raise ValueError("Source object changed since pinned preflight")
        planned_new_bytes = sum(acquisition.heads[url]["length"] for url in new_source_urls)
        if planned_new_bytes > caps["source_chunk_body_bytes"]:
            raise ValueError("Cumulative source cap would be exceeded")
        original.write_json(out / "new_heads_freeze.json", {
            "freeze_reference": freeze, "heads": acquisition.heads,
            "planned_new_source_bytes": planned_new_bytes,
            "prior_source_bytes": charges["source_chunk_bytes"],
            "new_native_or_raw_pixel_bodies_before_this_receipt": 0})
        for name in ("fine", "coarse"):
            spec = plan["volumes"][name]
            data = acquisition.fetch(spec["metadata_url"], "tiff_metadata")
            if hashlib.sha256(data).hexdigest() != spec["metadata_sha256"]:
                raise ValueError("Raw Zarr metadata changed")
        # Exact header byte hashes also cover absent orientation/predictor/fill-order tags.
        for source in protocol["native_slices"]:
            for item in source["header_ranges"]:
                data = acquisition.fetch(source["url"], "tiff_metadata", tuple(item["range"]))
                if hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("Native TIFF header contract changed")
        original.write_json(out / "native_header_gate.json", {"all_eight_native_headers_match": True,
                            "new_intensity_bodies_before_this_receipt": 0})
        rows = []
        for item in protocol["corner_rows"]:
            values = struct.unpack("<HH", acquisition.fetch(item["url"], "tiff_metadata", tuple(item["byte_range"])))
            for xyz, native in zip(item["xyz"], values):
                x, y, z = xyz
                stored = int(fine.chunks[(z // 128, y // 128, x // 128)][z % 128, y % 128, x % 128])
                rows.append({"center_id": item["center_id"], "xyz": xyz,
                             "native_uint16": native, "zarr_uint16": stored,
                             "signed_native_minus_zarr": native - stored})
        gate = {"passed_exact_32_of_32": all_corners_equal(rows),
                "equal_integer_values": sum(r["native_uint16"] == r["zarr_uint16"] for r in rows),
                "maximum_absolute_error": max(abs(r["signed_native_minus_zarr"]) for r in rows), "values": rows,
                "scope": "Exact native TIFF/Zarr fidelity at 32 fixed integer corners only; no whole-volume or legacy surface-render certification"}
        original.write_json(out / "native_source_gate.json", gate)
        diagnostic = native_center_diagnostic(original, plan, rows,
                                              json.loads((prior / "identity_result.json").read_bytes()))
        original.write_json(out / "native_center_diagnostic.json", diagnostic)
        print(json.dumps({"stage": "native_source_gate", "passed": gate["passed_exact_32_of_32"],
                          "equal_values": gate["equal_integer_values"], "gate_sha256": sha(out / "native_source_gate.json")}), flush=True)
        result = {"schema": protocol["schema"], "freeze_reference": freeze,
                  "adaptive_after_original_failure": True, "original_surface_contract_remains": "neither",
                  "native_source_gate": gate, "native_center_trilinear_diagnostic": diagnostic, "anatomy": [],
                  "labels_or_photographs_accessed": False, "registration_fitted": False,
                  "best_shift_selected": False, "automatic_label_stage_go": False}
        if gate["passed_exact_32_of_32"]:
            result["anatomy"] = anatomy(original, plan, fine, coarse, acquisition, out)
            result["outcome"] = "completed_separate_adaptive_raw_scanner_grid_anatomy"
        else:
            result["outcome"] = "native_source_fidelity_failed_anatomy_not_run"
        verify_prior(prior, protocol)
        result["resources"] = {"followup_cpu_seconds": time.process_time() - CPU_START,
                               "cumulative_cpu_seconds_with_prior_reservation": charges["reserved_cpu_seconds"] + time.process_time() - CPU_START,
                               "followup_wall_seconds": time.monotonic() - started_wall,
                               "new_body_bytes_prospectively_charged": acquisition.bytes,
                               "cumulative_source_chunk_bytes": charges["source_chunk_bytes"] + acquisition.bytes["source_chunk"],
                               "cumulative_tiff_metadata_bytes_with_declared_preflights": charges["tiff_metadata_bytes"] + acquisition.bytes["tiff_metadata"],
                               "decoded_unique_chunks": len(fine.chunks) + len(coarse.chunks)}
        acquisition.check_cpu()
        original.write_json(out / "result.json", result)
        acquisition.save()
        return result
    except Exception as error:
        acquisition.save()
        original.write_json(out / "error.json", {"error_type": type(error).__name__,
                            "message": "Execution stopped; no automatic retries, substitution or protocol changes.",
                            "followup_cpu_seconds": time.process_time() - CPU_START})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze-reference")
    args = parser.parse_args()
    if args.check_only == args.run:
        parser.error("Choose exactly one of --check-only and --run")
    protocol = json.loads(args.protocol.read_bytes())
    if sha(Path(__file__)) != protocol["runner_sha256"]:
        raise ValueError("Follow-up runner differs from frozen protocol")
    original, plan = load_original(args.stage1_dir, protocol)
    check_protocol(protocol, original, plan)
    verify_prior(args.prior_run, protocol)
    if args.check_only:
        print(json.dumps({"protocol_sha256": sha(args.protocol), "hash_and_geometry_preflight": "passed", "pixel_arrays_decoded": False}))
        return
    if not args.out or not args.freeze_reference or not re.fullmatch("[0-9a-f]{40}", args.freeze_reference):
        parser.error("Execution requires a new output path and full published freeze reference")
    prior, out = args.prior_run.resolve(), args.out.resolve()
    if out.exists() or out == prior or prior in out.parents or out in prior.parents:
        raise ValueError("Preserve prior data: require a new output tree separate from the original run")
    out.mkdir(parents=True); (out / "objects").mkdir()
    import scipy
    import numcodecs
    original.write_json(out / "start.json", {"freeze_reference": args.freeze_reference,
                        "protocol_sha256": sha(args.protocol), "runner_sha256": protocol["runner_sha256"],
                        "original_runner_sha256": protocol["original_runner_sha256"],
                        "original_protocol_sha256": protocol["original_protocol_sha256"],
                        "python_version": sys.version,
                        "numpy_version": original.np.__version__, "scipy_version": scipy.__version__,
                        "numcodecs_version": numcodecs.__version__, "prior_charges": protocol["prior_charges"],
                        "adaptive_followup_not_a_reclassification_of_original_failure": True})
    result = run(protocol, original, plan, prior, out, args.freeze_reference)
    print(json.dumps({"outcome": result["outcome"], "resources": result["resources"],
                      "result_sha256": sha(out / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
