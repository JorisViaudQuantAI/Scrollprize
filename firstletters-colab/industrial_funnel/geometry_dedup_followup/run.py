#!/usr/bin/env python3
"""CPU audit of Gemini-009 correspondence proposals; no inference admission.

Reuses only reviewed, hash-pinned upstream geometry definitions. It deliberately
does not import producer side effects or silently replace the upstream predicate.
Inputs are public geometry and adaptive-067 reference labels, never CT or ink.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
from typing import Any
import zlib

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
GEMINI = "da94312727c7e95f7f7fdc2d7481d502265434c5"
ADAPTIVE = "2cba2f7a8a316ac9e1d42ed939d226ec255e5d22"
CLAUDE = "ab5b297de8cf696e098852ccb9b3be766317acb2"
STATES = ("fitted", "snap", "flat")
WINDS = ("w033", "w034", "w035", "w036")
PINS = {
    "gemini": (GEMINI, "gemini_lane/industrial_funnel_adapter.py",
                "7a43443989eaaa08d0c9bf1f8f9b98fac95bf768fbaa48ccebd3772589c57bec"),
    "atlas": (ADAPTIVE, "firstletters-colab/adaptive_surface_recovery/claude_carrier_atlas/run.py",
              "1130ba0bb36a6b877dd6e4f410c12367b247c6633ecea014c0750b9ce2676962"),
    "labels": (ADAPTIVE, "firstletters-colab/adaptive_surface_recovery/claude_carrier_atlas/result.json",
               "ca9dd1691afe97014344301d4d2ef9f9243cd8730d5b7a10e96d64f6274389b4"),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def arrsha(a):
    return sha(np.ascontiguousarray(a).tobytes())


def read_git(repo, commit, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{path}"])


def definitions(body, expected_sha, names, namespace):
    """Load selected definitions from a fixed reviewed source, never arbitrary code."""
    if sha(body) != expected_sha:
        raise ValueError("Pinned source hash mismatch")
    nodes = [n for n in ast.parse(body).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError("Pinned definitions missing")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<hash-pinned-upstream>", "exec"), namespace)
    return namespace


def upstream(repo):
    source = {k: read_git(repo, pin[0], pin[1]) for k, pin in PINS.items()}
    for k, body in source.items():
        if sha(body) != PINS[k][2]:
            raise ValueError("Pinned source mismatch: " + k)
    g = definitions(source["gemini"], PINS["gemini"][2],
                    {"_compute_grid_normals_simple", "_upsample_valid_quads",
                     "evaluate_normal_correspondence_dedup"},
                    {"np": np, "cKDTree": cKDTree, "Any": Any})
    a = definitions(source["atlas"], PINS["atlas"][2], {"Mesh", "RayIndex", "self_test"},
                    {"np": np, "cKDTree": cKDTree, "arrsha": arrsha,
                     "LIM": 24., "COS": np.cos(np.deg2rad(30.))})
    return g, a, json.loads(source["labels"])


def decode_codes(obj):
    b = zlib.decompress(base64.b64decode(obj["data"]))
    if sha(b) != obj["sha256"] or obj["dtype"] != "uint8":
        raise ValueError("Reference code-grid integrity failure")
    return np.frombuffer(b, np.uint8).reshape(obj["shape"])


def load_mesh(root, records, mesh_class):
    bodies = {}
    for name in ("x.tif", "y.tif", "z.tif", "meta.json"):
        b = (root / name).read_bytes()
        if sha(b) != records[name]["sha256"] or len(b) != records[name]["bytes"]:
            raise ValueError("Geometry integrity failure: " + str(root / name))
        bodies[name] = b
    xyz = np.stack([np.asarray(Image.open(io.BytesIO(bodies[n])), np.float64)
                    for n in ("x.tif", "y.tif", "z.tif")], axis=-1)
    mesh = mesh_class(xyz)
    return mesh


def same_reference_mesh(observed, expected):
    """Exact coordinates/topology; tolerate only platform rounding in area sum."""
    return (set(observed) == set(expected)
            and all(observed[k] == expected[k] for k in observed if k != "area_vox2")
            and np.isclose(observed["area_vox2"], expected["area_vox2"], rtol=1e-12, atol=1e-8))


def prepare(mesh, g):
    x, y, z = (mesh.P[..., j] for j in range(3))
    normals, ok = g["_compute_grid_normals_simple"](x, y, z, mesh.valid)
    ids = np.flatnonzero(ok)
    points = mesh.xyz[ids]
    n = normals.reshape(-1, 3)[ids]
    dense = g["_upsample_valid_quads"](x, y, z, mesh.valid, factor=4)
    # The proposed support correction preserves every usable original vertex.
    preserved = np.concatenate((dense, points), axis=0)
    return {"mesh": mesh, "ids": ids, "points": points, "normals": n,
            "dense": dense, "tree": cKDTree(dense),
            "preserved": preserved, "preserved_tree": cKDTree(preserved),
            "vertex_tree": cKDTree(points), "ref_codes": np.zeros(len(ids), np.uint8),
            "ref_depths": np.full(len(ids), np.nan), "ray_cache": {}}


def correspond(a, b, preserve_vertices=False):
    key = "preserved" if preserve_vertices else "dense"
    tree = b["preserved_tree"] if preserve_vertices else b["tree"]
    dist, j = tree.query(a["points"], k=1)
    q = b[key][j]
    delta = q - a["points"]
    dn = np.einsum("ij,ij->i", delta, a["normals"])
    lateral = np.sqrt(np.maximum(0, dist * dist - dn * dn))
    _, v = b["vertex_tree"].query(a["points"], k=1)
    nb = b["normals"][v]
    signed_cos = np.einsum("ij,ij->i", a["normals"], nb)
    match = (np.abs(dn) <= 3.) & (lateral <= 4.) & (np.abs(signed_cos) >= .7)
    aligned = nb * np.where(signed_cos < 0, -1., 1.)[:, None]
    nd = aligned - a["normals"]
    # Maximum on the continuous symmetric interval, exactly at an endpoint.
    ray_separation = np.maximum(np.linalg.norm(delta - 8 * nd, axis=1),
                                np.linalg.norm(delta + 8 * nd, axis=1))
    return {"match": match, "q": q, "nb": nb, "dist": dist,
            "dn": dn, "lateral": lateral, "cos": signed_cos,
            "ray_separation": ray_separation, "dense_ids": j, "vertex_ids": v}


def finite_summary(values):
    values = np.asarray(values)
    if values.size == 0:
        return {"n": 0, "q50": None, "q95": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite measured diagnostic")
    q = np.quantile(values, [.5, .95, 1.])
    return dict(n=int(values.size), q50=float(q[0]), q95=float(q[1]), max=float(q[2]))


def source_labels(a, selected, index):
    for i in np.flatnonzero(selected & (a["ref_codes"] == 0)):
        code, depth = index.hit(a["points"][i], a["normals"][i])
        a["ref_codes"][i], a["ref_depths"][i] = code, depth
    return a["ref_codes"]


def target_labels(b, pair, selected, index):
    labels = np.zeros(len(selected), np.uint8)
    depths = np.full(len(selected), np.nan)
    for i in np.flatnonzero(selected):
        key = (int(pair["dense_ids"][i]), int(pair["vertex_ids"][i]))
        if key not in b["ray_cache"]:
            b["ray_cache"][key] = index.hit(pair["q"][i], pair["nb"][i])
        labels[i], depths[i] = b["ray_cache"][key]
    return labels, depths


def synthetic(g, a):
    yy, xx = np.mgrid[:8, :9]
    p = np.stack((100 + 20 * xx, 100 + 20 * yy, np.full(xx.shape, 100)), axis=-1).astype(float)
    p[3:5, 4:6] = -1
    one = prepare(a["Mesh"](p), g)
    old, fixed = correspond(one, one), correspond(one, one, True)
    translated = p.copy()
    translated[one["mesh"].valid, 2] += 3
    two = prepare(a["Mesh"](translated), g)
    off = correspond(one, two, True)
    reversed_normal = dict(one, normals=-one["normals"])
    rev = correspond(one, reversed_normal, True)
    # Analytic 40-degree turn, inside |cos| >= .7; same center differs in depth.
    theta = np.deg2rad(40)
    n1, n2 = np.array([0., 0., 1.]), np.array([np.sin(theta), 0., np.cos(theta)])
    angular_depth_separation = float(np.linalg.norm(8 * (n2 - n1)))
    checks = {
        "reference_ray_synthetic": a["self_test"]()["pass"],
        "upstream_hole_identity_has_missing_support": bool((~old["match"]).any()),
        "preserved_hole_identity_complete": bool(fixed["match"].all()),
        "preserved_hole_identity_zero_distance": bool((fixed["dist"] == 0).all()),
        "parallel_3vox_accepted_but_not_same_sampling": bool(off["match"].all() and np.allclose(off["ray_separation"], 3)),
        "reversed_normals_symmetric_span_identity": bool(rev["match"].all() and np.allclose(rev["ray_separation"], 0)),
        "nonparallel_same_center_exceeds_5vox": bool(np.dot(n1, n2) >= .7 and angular_depth_separation > 5),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"checks": checks, "hole_old_fraction": float(old["match"].mean()),
            "hole_fixed_fraction": float(fixed["match"].mean()),
            "same_center_40deg_depth_separation_vox": angular_depth_separation}


def run(args):
    started = time.monotonic()
    g, ref, labels = upstream(args.repo)
    checks = synthetic(g, ref)
    winner = load_mesh(args.data / "winner", labels["winner"]["files"], ref["Mesh"])
    if not same_reference_mesh(winner.receipt, labels["winner"]["mesh"]):
        raise ValueError("Reference mesh triangulation/coordinates changed")
    arrays, rows, inputs = {}, [], []
    for state in STATES:
        prepared = {}
        for wind in WINDS:
            r = labels["states"][state]["windings"][wind]
            mesh = load_mesh(args.data / "meshes" / state / wind, r["files"], ref["Mesh"])
            if not same_reference_mesh(mesh.receipt, r["mesh"]):
                raise ValueError("Mesh triangulation/coordinates changed")
            prepared[wind] = prepare(mesh, g)
            prepared[wind]["published_codes"] = decode_codes(r["reciprocal_code_map"])
            inputs.append({"state": state, "winding": wind, "mesh": mesh.receipt,
                           "files": {n: r["files"][n] for n in ("x.tif", "y.tif", "z.tif", "meta.json")}})
        for wa in WINDS:
            for wb in WINDS:
                a, b = prepared[wa], prepared[wb]
                old = correspond(a, b)
                fixed = correspond(a, b, True)
                m = old["match"]
                published = a["published_codes"].ravel()[a["ids"]]
                ca = source_labels(a, m, winner.index)
                cb, db = target_labels(b, old, m, winner.index)
                table = np.zeros((7, 7), np.int64)
                np.add.at(table, (ca[m], cb[m]), 1)
                parity = np.zeros((7, 7), np.int64)
                np.add.at(parity, (published[m], ca[m]), 1)
                close_a, close_b = (ca == 4) & m, (cb == 4) & m
                ray = old["ray_separation"][m]
                # The upstream helper exposes aggregate summaries only. Verify
                # agreement at its four-decimal precision; this alone is not a
                # pointwise proof of equivalence of the masks. The individual
                # expressions above are also reviewed against the pinned source.
                sa = tuple(a["mesh"].P[..., j] for j in range(3)) + (a["mesh"].valid,)
                sb = tuple(b["mesh"].P[..., j] for j in range(3)) + (b["mesh"].valid,)
                upstream_summary = g["evaluate_normal_correspondence_dedup"](sa, sb)
                if round(float(m.mean()), 4) != upstream_summary["normal_correspondence_frac"]:
                    raise AssertionError("Upstream predicate mismatch")
                row = {"state": state, "source": wa, "target": wb, "self_pair": wa == wb,
                       "n_source": len(m), "n_accepted": int(m.sum()),
                       "accepted_fraction": float(m.mean()),
                       "preserved_n_accepted": int(fixed["match"].sum()),
                       "preserved_fraction": float(fixed["match"].mean()),
                       "preserved_added": int((fixed["match"] & ~m).sum()),
                       "preserved_removed": int((m & ~fixed["match"]).sum()),
                       "max_symmetric_ray_separation_vox": finite_summary(ray),
                       "accepted_ray_separation_gt1": int((ray > 1).sum()),
                       "accepted_ray_separation_gt3": int((ray > 3).sum()),
                       "accepted_ray_separation_gt6": int((ray > 6).sum()),
                       "reference_confusion_7x7": table.tolist(),
                       "source_published067_vs_recomputed_confusion_7x7": parity.tolist(),
                       "source_near": int(close_a.sum()),
                       "source_near_target_far": int((close_a & (cb == 6)).sum()),
                       "source_near_target_unknown": int((close_a & np.isin(cb, [0, 1, 2, 3])).sum()),
                       "source_near_target_intermediate": int((close_a & (cb == 5)).sum()),
                       "source_near_target_near": int((close_a & close_b).sum()),
                       "upstream_summary": upstream_summary}
                row["all_near_to_far_examples"] = [
                    {"source_grid_yx": [int(v) for v in np.unravel_index(a["ids"][i], a["mesh"].shape)],
                     "source_native_xyz": a["points"][i].tolist(),
                     "target_native_xyz": old["q"][i].tolist(),
                     "source_normal_xyz": a["normals"][i].tolist(),
                     "target_normal_xyz": old["nb"][i].tolist(),
                     "source_signed_reference_depth": float(a["ref_depths"][i]),
                     "target_signed_reference_depth": float(db[i]),
                     "abs_normal_displacement": float(abs(old["dn"][i])),
                     "lateral_displacement": float(old["lateral"][i]),
                     "abs_normal_cosine": float(abs(old["cos"][i])),
                     "symmetric_ray_separation": float(old["ray_separation"][i])}
                    for i in np.flatnonzero(close_a & (cb == 6))]
                rows.append(row)
                stem = f"{state}_{wa}_{wb}"
                grid = np.zeros(a["mesh"].shape, np.uint8)
                grid.ravel()[a["ids"][m]] = 1
                grid.ravel()[a["ids"][close_a]] = 2
                grid.ravel()[a["ids"][close_a & (cb == 6)]] = 3
                grid.ravel()[a["ids"][close_a & np.isin(cb, [0, 1, 2, 3])]] = 4
                grid.ravel()[a["ids"][close_a & (cb == 5)]] = 5
                arrays[stem + "_classification"] = grid
                separation = np.full(a["mesh"].shape, np.nan, np.float32)
                separation.ravel()[a["ids"][m]] = old["ray_separation"][m]
                arrays[stem + "_ray_separation"] = separation
                arrays[stem + "_source_reference"] = np.where(m, ca, 0).astype(np.uint8)
                arrays[stem + "_target_reference"] = cb
                arrays[stem + "_source_reference_depth"] = np.where(m, a["ref_depths"], np.nan)
                arrays[stem + "_target_reference_depth"] = db
                arrays[stem + "_source_ids"] = a["ids"].astype(np.int32)
                arrays[stem + "_match"] = m
                arrays[stem + "_preserved_match"] = fixed["match"]
                if time.monotonic() - started > 600:
                    raise TimeoutError("Frozen ten minute CPU wall limit exceeded")
                print(f"{stem}: {m.sum()}/{len(m)}; near/far={row['source_near_target_far']}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    maps = args.out / "numeric_maps.npz"
    np.savez_compressed(maps, **arrays)
    result = {"schema": "native.geometry_correspondence_followup.v1",
              "protocol_sha256": sha((HERE / "PROTOCOL.md").read_bytes()),
              "script_sha256": sha(Path(__file__).read_bytes()),
              "pins": {k: {"commit": v[0], "path": v[1], "sha256": v[2]} for k, v in PINS.items()},
              "mesh_commit": CLAUDE, "reference_is_geometry_proxy_not_ink_truth": True,
              "new_ink_inferences": 0, "new_discovered_letters": 0,
              "scientific_pruning_admission": "NOT_ESTABLISHED",
              "synthetic": checks, "inputs": inputs, "winner": labels["winner"],
              "pairs": rows, "maps_sha256": sha(maps.read_bytes()),
              "numpy_version": np.__version__, "wall_seconds": time.monotonic() - started}
    (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pairs": len(rows), "wall_seconds": result["wall_seconds"],
                      "maps_bytes": maps.stat().st_size}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True, help="Git checkout containing all three pinned commits")
    p.add_argument("--data", type=Path, required=True, help="Hash-checked public winner/ and meshes/ data")
    p.add_argument("--out", type=Path, default=HERE)
    run(p.parse_args())
