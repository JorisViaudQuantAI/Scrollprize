#!/usr/bin/env python3
"""Check all saved G4 masks against the hash-pinned upstream implementation.

Only the original helper's returned dictionary is instrumented to expose its
already-computed Boolean mask and nearest-neighbour indices. No predicate,
normal, point, threshold or nearest-neighbour calculation is replaced.

This checks a geometry experiment, not input-tensor equality or ink recall.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ARRAY_ROLES = (
    "classification", "ray_separation", "source_reference", "target_reference",
    "source_reference_depth", "target_reference_depth", "source_ids", "match",
    "preserved_match",
)


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def load_experiment():
    path = HERE / "run.py"
    spec = importlib.util.spec_from_file_location("geometry_followup_checked", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import the local geometry experiment")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expose_upstream_outputs(body: bytes, expected_hash: str, namespace: dict):
    """Append three return fields, leaving all upstream calculations intact."""
    if sha(body) != expected_hash:
        raise ValueError("Upstream implementation hash mismatch")
    nodes = [node for node in ast.parse(body).body
             if isinstance(node, ast.FunctionDef)
             and node.name == "evaluate_normal_correspondence_dedup"]
    if len(nodes) != 1:
        raise ValueError("Expected one pinned correspondence helper")
    node = copy.deepcopy(nodes[0])
    if not isinstance(node.body[-1], ast.Return) or not isinstance(node.body[-1].value, ast.Dict):
        raise ValueError("Pinned helper no longer ends in the reviewed dictionary return")
    returned = node.body[-1].value
    extra = {
        "_exact_normal_match": "normal_match",
        "_exact_dense_ids": "idx_dense",
        "_exact_normal_vertex_ids": "idx_vert",
    }
    if any(isinstance(key, ast.Constant) and key.value in extra for key in returned.keys):
        raise ValueError("Instrumentation output name collision")
    for key, variable in extra.items():
        returned.keys.append(ast.Constant(key))
        returned.values.append(ast.Name(id=variable, ctx=ast.Load()))
    instrumented = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = dict(namespace)
    exec(compile(instrumented, "<pinned-upstream-return-fields-only>", "exec"), scope)
    return scope[node.name]


def check(args):
    started = time.monotonic()
    experiment = load_experiment()
    result_body = args.result.read_bytes()
    result = json.loads(result_body)
    run_body = (HERE / "run.py").read_bytes()
    if sha(run_body) != result["script_sha256"]:
        raise ValueError("Result was not produced by the current geometry script")
    if sha((HERE / "PROTOCOL.md").read_bytes()) != result["protocol_sha256"]:
        raise ValueError("Experiment protocol changed after the recorded execution")
    map_body = args.maps.read_bytes()
    if sha(map_body) != result["maps_sha256"]:
        raise ValueError("Saved numeric maps do not match the experiment receipt")
    g, reference, labels = experiment.upstream(args.repo)
    pin = experiment.PINS["gemini"]
    upstream_body = experiment.read_git(args.repo, pin[0], pin[1])
    upstream = expose_upstream_outputs(upstream_body, pin[2], g)
    expected_pins = {key: {"commit": value[0], "path": value[1], "sha256": value[2]}
                     for key, value in experiment.PINS.items()}
    if result["pins"] != expected_pins:
        raise ValueError("Experiment and verifier do not use the same pinned inputs")

    expected_pairs = {(state, source, target) for state in experiment.STATES
                      for source in experiment.WINDS for target in experiment.WINDS}
    pair_rows = {(row["state"], row["source"], row["target"]): row for row in result["pairs"]}
    if len(result["pairs"]) != 48 or set(pair_rows) != expected_pairs:
        raise ValueError("Expected all 48 directed experiment pairs exactly once")
    expected_arrays = {"_".join(pair) + "_" + role
                       for pair in expected_pairs for role in ARRAY_ROLES}
    rows, cache_rows = [], []
    with np.load(io.BytesIO(map_body), allow_pickle=False) as maps:
        if set(maps.files) != expected_arrays or len(maps.files) != 432:
            raise ValueError("Expected the final 432-array artifact, including signed depths")
        for state in experiment.STATES:
            prepared = {}
            for wind in experiment.WINDS:
                archived = labels["states"][state]["windings"][wind]
                mesh = experiment.load_mesh(args.data / "meshes" / state / wind,
                                            archived["files"], reference["Mesh"])
                # Coordinate bytes, topology and counts are exact. Summed area
                # can differ by an ULP with SIMD/allocation layout; it is not an
                # identity hash and must not reject identical pinned inputs.
                structural = {k: v for k, v in mesh.receipt.items() if k != "area_vox2"}
                archived_structural = {k: v for k, v in archived["mesh"].items() if k != "area_vox2"}
                same_area = np.isclose(mesh.receipt["area_vox2"], archived["mesh"]["area_vox2"],
                                       rtol=1e-14, atol=1e-7)
                if structural != archived_structural or not same_area:
                    raise ValueError("Geometry coordinate or triangulation receipt changed")
                prepared[wind] = experiment.prepare(mesh, g)
            keys = {wind: {} for wind in experiment.WINDS}
            for source in experiment.WINDS:
                a = prepared[source]
                surf_a = tuple(a["mesh"].P[..., j] for j in range(3)) + (a["mesh"].valid,)
                for target in experiment.WINDS:
                    b = prepared[target]
                    surf_b = tuple(b["mesh"].P[..., j] for j in range(3)) + (b["mesh"].valid,)
                    original = upstream(surf_a, surf_b)
                    reconstructed = experiment.correspond(a, b)
                    stem = "_".join((state, source, target))
                    mask = original["_exact_normal_match"]
                    checks = {
                        "stored_source_ids_equal": np.array_equal(maps[stem + "_source_ids"], a["ids"]),
                        "stored_mask_pointwise_equal": np.array_equal(maps[stem + "_match"], mask),
                        "reconstructed_mask_pointwise_equal": np.array_equal(reconstructed["match"], mask),
                        "dense_indices_equal": np.array_equal(reconstructed["dense_ids"], original["_exact_dense_ids"]),
                        "normal_vertex_indices_equal": np.array_equal(reconstructed["vertex_ids"], original["_exact_normal_vertex_ids"]),
                        "receipt_source_count_equal": len(mask) == pair_rows[state, source, target]["n_source"],
                        "receipt_accepted_count_equal": int(mask.sum()) == pair_rows[state, source, target]["n_accepted"],
                    }
                    if not all(checks.values()):
                        raise AssertionError({"pair": stem, "checks": checks})
                    rows.append({"state": state, "source": source, "target": target,
                                 "source_vertices": len(mask), "accepted": int(mask.sum()),
                                 "mask_sha256": sha(np.ascontiguousarray(mask).tobytes()),
                                 "all_checks_passed": True})
                    for dense_id, normal_id in zip(original["_exact_dense_ids"][mask],
                                                   original["_exact_normal_vertex_ids"][mask]):
                        keys[target].setdefault(int(dense_id), set()).add(int(normal_id))
            for target, point_keys in keys.items():
                cache_rows.append({
                    "state": state, "target": target,
                    "accepted_dense_points_with_multiple_normal_vertex_ids":
                        sum(len(normal_ids) > 1 for normal_ids in point_keys.values()),
                })

    # Do not certify one artifact while an overlapping replay replaces another.
    if args.result.read_bytes() != result_body or sha(args.maps.read_bytes()) != result["maps_sha256"]:
        raise ValueError("Experiment outputs changed during the independent check")
    reused = sum(row["accepted_dense_points_with_multiple_normal_vertex_ids"] for row in cache_rows)
    receipt = {
        "schema": "native.geometry_upstream_mask_check.v1",
        "pass": True,
        "checker_sha256": sha(Path(__file__).read_bytes()),
        "experiment_script_sha256": sha(run_body),
        "experiment_result_sha256": sha(result_body),
        "protocol_sha256": result["protocol_sha256"],
        "maps_sha256": result["maps_sha256"],
        "map_array_count": 432,
        "derived_mesh_area_comparison": {"atol_vox2": 1e-7, "rtol": 1e-14,
                                           "coordinate_hashes_and_other_receipt_fields": "exact"},
        "pins": expected_pins,
        "instrumentation": "Only three fields appended to the hash-pinned helper's final dictionary return; original computations unchanged.",
        "stored_masks_pointwise_equal_to_upstream": True,
        "reconstructed_nearest_indices_equal_to_upstream": True,
        "checked_pairs": len(rows),
        "checked_source_vertex_occurrences": sum(row["source_vertices"] for row in rows),
        "pairs": rows,
        "cache_key_diagnostic": {
            "scope": "Accepted old-predicate correspondences; self and non-self sources together, counted separately for each fixed target mesh.",
            "accepted_dense_points_with_multiple_normal_vertex_ids": reused,
            "per_target": cache_rows,
            "conclusion": "A target dense point alone does not identify the normal used by G4. The experiment keys target rays by (dense_id, normal_vertex_id).",
        },
        "geometry_only": True,
        "new_ink_inferences": 0,
        "new_discovered_letters": 0,
        "scientific_pruning_admission": "NOT_ESTABLISHED",
        "wall_seconds": time.monotonic() - started,
    }
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pass": True, "checked_pairs": len(rows), "maps_sha256": result["maps_sha256"],
                      "dense_points_with_multiple_normal_ids": reused,
                      "wall_seconds": receipt["wall_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--result", type=Path, default=HERE / "result.json")
    parser.add_argument("--maps", type=Path, default=HERE / "numeric_maps.npz")
    parser.add_argument("--out", type=Path, default=HERE / "upstream_mask_check.json")
    check(parser.parse_args())
