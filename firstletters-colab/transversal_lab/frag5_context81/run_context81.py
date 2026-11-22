#!/usr/bin/env python3
"""Fixed-center, adaptive 81³ anatomy contrast on public opened Frag5.

Uses the previously verified sampling and transport code unchanged. No labels,
models, local alignment fit, or production coordinate update.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import struct
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def load_module(path, expected, name):
    if sha(path) != expected:
        raise ValueError("Pinned scientific helper changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_plan(plan, base, original):
    if plan["schema"] != "frag5-adaptive-context81-v1" or [r["id"] for r in plan["centers"]] != [20, 25]:
        raise ValueError("Unexpected adaptive pair")
    expected = {"side": 81, "quadrature_per_axis": 5, "coarse_shift_values": [-1, 0, 1],
                "nested_sides": [17, 33, 65, 81], "smoothing_support_side": 65,
                "binomial_passes": [0, 1, 4], "shell_outer_side": 81, "shell_excluded_side": 33,
                "prior_fine_max_error_counts": 1e-8, "prior_native_max_error_counts": 0}
    if plan["design"] != expected:
        raise ValueError("Numerical design changed")
    if plan["caps"] != {"cumulative_source_body_bytes": 268435456, "prior_source_body_bytes": 57064081,
                        "new_object_body_bytes": 4259840, "new_metadata_bytes": 1048576,
                        "cumulative_unique_raw_chunks": 64, "prior_unique_raw_chunks": 14,
                        "planned_new_chunks": 43, "planned_target_chunks": 49, "cpu_seconds": 360, "threads": 4}:
        raise ValueError("Frozen resource caps changed")
    A = np.array(base["affine_xyz"])
    inverse, t = np.linalg.inv(A[:3, :3]), A[:3, 3]
    originals = {r["id"]: r for r in base["centers"]}
    all_keys = {"fine": set(), "coarse": set()}
    for item in plan["centers"]:
        center = np.rint(A[:3, :3] @ np.array(originals[item["id"]]["fine_xyz"]) + t).astype(int)
        if center.tolist() != item["center_xyz"]:
            raise ValueError("Center changed")
        for arm, radius in (("fine_point", 40), ("fine_footprint", 40.4)):
            corners = np.array(list(itertools.product(*zip(center - radius, center + radius))))
            xyz = (corners - t) @ inverse.T
            lo, hi = np.floor(xyz.min(0)).astype(int), np.ceil(xyz.max(0)).astype(int)
            keys = original.box_chunks(lo, hi, base["volumes"]["fine"]["metadata"]["shape"])
            if item[arm]["inclusive_box_xyz"] != [lo.tolist(), hi.tolist()] or {tuple(k) for k in keys} != {tuple(k) for k in item[arm]["chunk_keys_zyx"]}:
                raise ValueError("Fine support changed")
            if arm == "fine_footprint":
                all_keys["fine"].update(tuple(k) for k in keys)
        lo, hi = center - 41, center + 41
        keys = original.box_chunks(lo, hi, base["volumes"]["coarse"]["metadata"]["shape"])
        spec = item["coarse_all_27_shifts"]
        if spec["inclusive_box_xyz"] != [lo.tolist(), hi.tolist()] or {tuple(k) for k in keys} != {tuple(k) for k in spec["chunk_keys_zyx"]}:
            raise ValueError("Coarse support changed")
        all_keys["coarse"].update(tuple(k) for k in keys)
    expected_objects = {(name, key, base["volumes"][name]["base_url"] + "/" + "/".join(map(str, key)))
                        for name, keys in all_keys.items() for key in keys}
    actual_objects = {(r["volume"], tuple(r["key_zyx"]), r["url"]) for r in plan["source_objects"]}
    if expected_objects != actual_objects or len(actual_objects) != 49 or len(plan["source_objects"]) != 49:
        raise ValueError("Raw object union changed")
    if sum(r["cached"] is None for r in plan["source_objects"]) != 43:
        raise ValueError("Cached/new split changed")


def verify_cache(plan, roots, prior_context):
    if sha(prior_context) != plan["prior_context_result_sha256"]:
        raise ValueError("Adaptive parent result changed")
    acquired = {}
    for group, root in roots.items():
        ledger_path = root / "acquisition.json"
        if sha(ledger_path) != plan["canonical_acquisition_sha256"][group]:
            raise ValueError("Canonical source ledger changed")
        ledger = json.loads(ledger_path.read_bytes())
        acquired[group] = {r["url"]: r for r in ledger["responses"] if r["category"] == "source_chunk"}
    for item in plan["source_objects"]:
        old = item["cached"]
        if old is None:
            continue
        root = roots[old["cache_group"]]
        path = (root / old["cache_file"]).resolve()
        if not path.is_relative_to(root.resolve()) or path.stat().st_size != old["bytes"] or sha(path) != old["sha256"]:
            raise ValueError("Old raw cache missing or changed")
        row = acquired[old["cache_group"]][item["url"]]
        if row["status"] != "complete" or row["sha256"] != old["sha256"] or row["expected_bytes"] != old["bytes"]:
            raise ValueError("Raw cache does not match canonical source receipt")
    for old in plan["prior_arrays"]:
        if sha(roots["native_grid"] / old["file"]) != old["sha256"]:
            raise ValueError("Old central array changed")


def cached_decode(item, volume, roots, original):
    old = item["cached"]
    raw = (roots[old["cache_group"]] / old["cache_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != old["sha256"]:
        raise ValueError("Cache changed before decoding")
    header = struct.unpack("<4B3I", raw[:16])
    if header[4] != original.CHUNK_BYTES or header[6] != len(raw):
        raise ValueError("Invalid cached Blosc size")
    decoded = volume.codec.decode(raw)
    if len(decoded) != original.CHUNK_BYTES:
        raise ValueError("Invalid decoded size")
    volume.chunks[tuple(item["key_zyx"])] = np.frombuffer(decoded, dtype="<u2").reshape(original.CHUNK_SHAPE)


def sample_grid(region, center, affine, side=81, check_cpu=lambda: None):
    r = side // 2
    z, y, x = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    q = np.column_stack([x.ravel(), y.ravel(), z.ravel()]) + center
    inv, t = np.linalg.inv(affine[:3, :3]), affine[:3, 3]
    point = region.sample((q - t) @ inv.T).reshape((side,) * 3)
    footprint = np.zeros(len(q), dtype=np.float64)
    nodes = (np.arange(5, dtype=np.float64) + .5) / 5 - .5
    for offset in itertools.product(nodes, repeat=3):
        check_cpu()
        footprint += region.sample((q + offset - t) @ inv.T)
    return point, (footprint / 125).reshape((side,) * 3)


def overlap_gate(new, old, local, tolerance):
    errors = {}
    for name, side in (("fine_point", 33), ("fine_nominal_coarse_footprint", 33), ("coarse_neighborhood", 35)):
        error = float(np.max(np.abs(local.crop(new[name], side) - old[name])))
        errors[name] = error
    passed = errors["coarse_neighborhood"] == 0 and all(errors[k] <= tolerance for k in ("fine_point", "fine_nominal_coarse_footprint"))
    return {"passed": passed, "maximum_absolute_errors": errors,
            "tolerance_fine_counts": tolerance, "native_requires_exact": True}


def variance_retention(filtered, raw):
    if np.ptp(raw) == 0:
        return None
    if np.ptp(filtered) == 0:
        return 0.
    return float(np.var(filtered) / np.var(raw))


def measurements(arrays, original, local, check_cpu):
    nested, smoothed, shell, quality = [], [], [], []
    shell_mask = np.ones((81,) * 3, dtype=bool)
    shell_mask[24:57, 24:57, 24:57] = False
    arms = ("fine_point", "fine_nominal_coarse_footprint")
    for i, data in arrays.items():
        native = data["coarse_neighborhood"]
        for side in (17, 33, 65, 81):
            for shift in itertools.product((-1, 0, 1), repeat=3):
                b = local.crop(native, side, shift)
                for arm in arms:
                    check_cpu()
                    nested.append({"id": i, "side": side, "arm": arm, "shift_xyz": list(shift),
                                   "zero_shift_primary": shift == (0, 0, 0),
                                   **original.correlation(local.crop(data[arm], side), b)})
        zero = local.crop(native, 81)
        for arm in arms:
            shell.append({"id": i, "arm": arm, "support": "81cube_excluding_central33cube",
                          **original.correlation(data[arm][shell_mask], zero[shell_mask])})
        for passes in (0, 1, 4):
            native_filtered = local.crop(local.smooth_valid(zero, passes), 65)
            for name in (*arms, "native_coarse"):
                raw = data[name] if name in arms else zero
                a = local.crop(local.smooth_valid(raw, passes), 65)
                stats = original.describe(a)
                if np.ptp(a) == 0:
                    stats.update({"mean": float(a.flat[0]), "sd": 0.})
                quality.append({"id": i, "array": name, "binomial_passes": passes,
                                "side": 65, "variance_retention": variance_retention(a, local.crop(raw, 65)),
                                **stats})
                if name in arms:
                    smoothed.append({"id": i, "arm": name, "side": 65, "binomial_passes": passes,
                                     **original.correlation(a, native_filtered)})
            check_cpu()
    return {"nested_contexts_all_shifts": nested, "fixed65_smoothing_zero_shift": smoothed,
            "outer_shell_zero_shift": shell, "fixed65_structure": quality}


def run(plan, base, original, local, roots, out, freeze, cpu_start):
    wall = time.monotonic()
    caps = plan["caps"]
    acq = original.Acquisition(out, {"source_chunk_body_bytes": caps["cumulative_source_body_bytes"] - caps["prior_source_body_bytes"],
                                    "tiff_metadata_body_bytes": caps["new_metadata_bytes"], "cpu_seconds": caps["cpu_seconds"], "threads": 4}, cpu_start)
    try:
        volumes = {name: original.Volume(base["volumes"][name], acq) for name in ("fine", "coarse")}
        urls = [s["url"] for s in plan["source_objects"]] + [base["volumes"][name]["metadata_url"] for name in volumes]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(acq.head, urls))
        new_bytes = 0
        for obj in plan["source_objects"]:
            head = acq.heads[obj["url"]]
            if obj["cached"] is not None:
                expected = obj["cached"]["head"]
                if head["length"] != expected["length"] or head["etag"] != expected["etag"]:
                    raise ValueError("Retained source object changed")
            else:
                if head["length"] > caps["new_object_body_bytes"]:
                    raise ValueError("New object exceeds admission cap")
                new_bytes += head["length"]
        if new_bytes + caps["prior_source_body_bytes"] > caps["cumulative_source_body_bytes"]:
            raise ValueError("Cumulative raw body cap would be exceeded")
        if caps["prior_unique_raw_chunks"] + caps["planned_new_chunks"] > caps["cumulative_unique_raw_chunks"]:
            raise ValueError("Cumulative chunk cap would be exceeded")
        write(out / "heads_before_new_data.json", {"heads": acq.heads, "planned_new_raw_bytes": new_bytes,
                                                   "new_raw_objects": 43, "raw_objects_reused": 6, "admitted": True})
        for name in volumes:
            spec = base["volumes"][name]
            body = acq.fetch(spec["metadata_url"], "tiff_metadata")
            if hashlib.sha256(body).hexdigest() != spec["metadata_sha256"]:
                raise ValueError("Raw Zarr metadata changed")
        for obj in plan["source_objects"]:
            if obj["cached"] is not None:
                cached_decode(obj, volumes[obj["volume"]], roots, original)
        for name, volume in volumes.items():
            keys = [s["key_zyx"] for s in plan["source_objects"] if s["volume"] == name]
            volume.load(keys)
        arrays, gates, artifacts = {}, [], []
        for item in plan["centers"]:
            check_cpu = acq.check_cpu
            fine = volumes["fine"].region(item["fine_footprint"]["inclusive_box_xyz"])
            point, footprint = sample_grid(fine, np.array(item["center_xyz"]), np.array(base["affine_xyz"]), check_cpu=check_cpu)
            coarse = volumes["coarse"].region(item["coarse_all_27_shifts"]["inclusive_box_xyz"])
            data = {"fine_point": point, "fine_nominal_coarse_footprint": footprint, "coarse_neighborhood": coarse.values}
            with np.load(roots["native_grid"] / f"anatomy_{item['id']}.npz", allow_pickle=False) as old:
                if old["coarse_center_xyz"].tolist() != item["center_xyz"]:
                    raise ValueError("Original center metadata differs")
                gate = overlap_gate(data, old, local, plan["design"]["prior_fine_max_error_counts"])
            gate["id"] = item["id"]
            gates.append(gate)
            arrays[item["id"]] = data
            target = out / f"context81_{item['id']}.npz"
            np.savez_compressed(target, **data, center_xyz=np.array(item["center_xyz"]))
            artifacts.append({"id": item["id"], "file": target.name, "sha256": sha(target)})
            write(out / "overlap_gates.json", gates)
            print(json.dumps({"stage": "context_prepared", "id": item["id"], "overlap_passed": gate["passed"]}), flush=True)
        result = {"schema": plan["schema"], "freeze_reference": freeze,
                  "adaptive_pair": True, "centers_and_affine_unchanged": True,
                  "overlap_gates": gates, "arrays": artifacts,
                  "labels_or_models_accessed": False, "registration_fitted": False,
                  "material_or_same_sheet_identity_established": False}
        if all(g["passed"] for g in gates):
            result.update(measurements(arrays, original, local, acq.check_cpu))
            result["outcome"] = "completed_fixed_context_comparison"
        else:
            result["outcome"] = "old_overlap_failed_no_new_anatomical_metrics"
        acq.check_cpu()
        result["resources"] = {"new_raw_body_bytes": acq.bytes["source_chunk"],
                               "cumulative_raw_body_bytes": caps["prior_source_body_bytes"] + acq.bytes["source_chunk"],
                               "new_metadata_body_bytes": acq.bytes["tiff_metadata"],
                               "new_raw_chunks": 43, "target_decoded_chunks": 49,
                               "cumulative_unique_raw_chunks": 57,
                               "process_cpu_seconds": time.process_time() - cpu_start,
                               "wall_seconds": time.monotonic() - wall}
        write(out / "result.json", result)
        return result
    finally:
        acq.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "stage1-dir", "local-structure-dir", "stage1-cache", "native-grid-cache", "prior-context-result"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze-reference")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cpu_start = time.process_time()
    plan = json.loads(args.protocol.read_bytes())
    original = load_module(args.stage1_dir / "frag5_native_pair_stage1.py", plan["stage1_runner_sha256"], "context81_original")
    local = load_module(args.local_structure_dir / "analyze_local_structure.py", plan["local_structure_runner_sha256"], "context81_local")
    base_path = args.stage1_dir / "frag5_native_pair_stage1_protocol.json"
    if sha(base_path) != plan["stage1_protocol_sha256"]:
        raise ValueError("Original geometric protocol changed")
    base = json.loads(base_path.read_bytes())
    verify_plan(plan, base, original)
    roots = {"stage1": args.stage1_cache, "native_grid": args.native_grid_cache}
    verify_cache(plan, roots, args.prior_context_result)
    if args.check_only:
        print("Geometry, source ledger and cache hashes verified; no pixels decoded or acquired.")
        return
    if not args.freeze_reference or not re.fullmatch("[0-9a-f]{40}", args.freeze_reference):
        parser.error("Require the public freeze commit")
    if args.out is None or args.out.exists():
        parser.error("Require a fresh output directory")
    (args.out / "objects").mkdir(parents=True)
    write(args.out / "start.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
          "freeze_reference": args.freeze_reference, "runner_sha256": sha(__file__),
          "protocol_sha256": sha(args.protocol), "prior_context_result_sha256": sha(args.prior_context_result),
          "scope": "Adaptive fixed-center public Frag5 context enlargement; old overlap checked before new metrics"})
    try:
        result = run(plan, base, original, local, roots, args.out, args.freeze_reference, cpu_start)
        print(json.dumps({"outcome": result["outcome"], "resources": result["resources"]}), flush=True)
    except Exception as exc:
        write(args.out / "failure.json", {"error_type": type(exc).__name__, "message": str(exc),
                                         "process_cpu_seconds": time.process_time() - cpu_start})
        raise


if __name__ == "__main__":
    main()
