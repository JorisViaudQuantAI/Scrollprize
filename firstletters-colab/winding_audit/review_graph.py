#!/usr/bin/env python3
"""Independently check stored-coordinate and crossing remapping on real source arrays.

Checks 32 evenly spaced transverse IDs and 32 evenly spaced acquired longitudinal
IDs, chosen without inspecting a chain outcome. Executes no winding traversal.
Longitudinal coordinates were not acquired: their index mapping is verified, but
this checker does not claim to validate longitudinal XYZ or physical sheet identity.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


CONDITIONS = ("original", "reverse_transverse", "reverse_all", "random_half", "reverse_crossing_rows")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(source, runner):
    started = time.monotonic()
    protocol_path = runner.with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    spec = importlib.util.spec_from_file_location("reviewed_q12_runner", runner)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    graph = producer.Graph(source, protocol)
    names = ("transverse_ids", "coordinates", "coordinate_offsets", "family", "lengths", "node_ids",
             "cross_offsets", "partners", "self_local", "partner_local")
    raw = {name: np.load(source / (name + ".npy"), mmap_mode="r", allow_pickle=False) for name in names}
    before = {name: sha(source / (name + ".npy")) for name in names}
    hids, nodes, families = raw["transverse_ids"], raw["node_ids"], raw["family"]
    assert np.all(np.diff(hids) > 0) and np.all(np.diff(nodes) > 0)
    vids = nodes[families[nodes] == 1]
    selected = np.r_[hids[np.linspace(0, len(hids)-1, 32, dtype=int)],
                     vids[np.linspace(0, len(vids)-1, 32, dtype=int)]]
    assert len(np.unique(selected)) == 64
    rng = np.random.Generator(np.random.PCG64(protocol["random_half_seed"]))
    random_flags = rng.integers(0, 2, len(families), dtype=np.uint8).astype(bool)

    def reversed_flag(ids, condition):
        if condition == "reverse_transverse": return families[ids] == 0
        if condition == "reverse_all": return np.full(np.shape(ids), True)
        if condition == "random_half": return random_flags[ids]
        return np.full(np.shape(ids), False)

    def original_cross(node):
        i = np.searchsorted(nodes, node)
        if i == len(nodes) or nodes[i] != node: return None
        lo, hi = raw["cross_offsets"][i:i+2]
        return tuple(raw[n][lo:hi] for n in ("partners", "self_local", "partner_local"))

    def original_coordinates(node):
        i = np.searchsorted(hids, node)
        if i == len(hids) or hids[i] != node: return None
        lo, hi = raw["coordinate_offsets"][i:i+2]
        return raw["coordinates"][lo:hi]

    counts = {}
    for condition in CONDITIONS:
        tally = dict(nodes=0, adjacency_rows=0, coordinate_rows=0,
                     available_reciprocal_edges=0, endpoints_with_xyz_checked=0,
                     counterpart_adjacency_not_retained=0)
        for node in selected:
            node = int(node)
            p, s, o = original_cross(node)
            gp, gs, go = graph.crossings(node, condition)
            flipped = bool(reversed_flag(node, condition))
            # Restore the declared row order before comparing any edge endpoints.
            if condition == "reverse_crossing_rows": gp, gs, go = gp[::-1], gs[::-1], go[::-1]
            assert np.array_equal(gp, p)
            restored_s = raw["lengths"][node] - 1 - gs if flipped else gs
            restored_o = np.where(reversed_flag(p, condition), raw["lengths"][p]-1-go, go)
            assert np.array_equal(restored_s, s) and np.array_equal(restored_o, o)
            xyz = original_coordinates(node)
            if xyz is not None:
                transformed = graph.coordinates(node, condition)
                assert np.array_equal(transformed[::-1] if flipped else transformed, xyz)
                assert np.array_equal(transformed[gs], xyz[s])
                tally["coordinate_rows"] += len(xyz)
                tally["endpoints_with_xyz_checked"] += len(s)
            for partner, own, other, mapped_own, mapped_other in zip(p, s, o, gs, go):
                partner = int(partner)
                opposite = original_cross(partner)
                if opposite is not None:
                    rp, rs, ro = opposite
                    assert np.any((rp == node) & (rs == other) & (ro == own))
                    tp, ts, to = graph.crossings(partner, condition)
                    assert np.any((tp == node) & (ts == mapped_other) & (to == mapped_own))
                    tally["available_reciprocal_edges"] += 1
                else:
                    tally["counterpart_adjacency_not_retained"] += 1
                partner_xyz = original_coordinates(partner)
                if partner_xyz is not None:
                    changed_xyz = graph.coordinates(partner, condition)
                    assert np.array_equal(changed_xyz[mapped_other], partner_xyz[other])
                    tally["endpoints_with_xyz_checked"] += 1
            tally["nodes"] += 1
            tally["adjacency_rows"] += len(p)
        counts[condition] = tally
    after = {name: sha(source / (name + ".npy")) for name in names}
    assert after == before
    assert protocol["conditions"][1:-1] == list(CONDITIONS[1:])
    return {"status": "pass", "schema": "q12-independent-graph-transform-review-v1",
            "selection": "32 evenly spaced sorted H IDs and 32 evenly spaced sorted acquired family-1 IDs; no traversal run",
            "selected_ids_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
            "runner_sha256": sha(runner), "protocol_sha256": sha(protocol_path),
            "checker_sha256": sha(Path(__file__)), "source_file_sha256": before,
            "source_files_unchanged": True, "by_condition": counts,
            "seconds": time.monotonic()-started,
            "limits": "Samples 64 nodes. Verifies physical XYZ identity only where transverse coordinates were acquired; longitudinal endpoints are checked through original index identities and reciprocal records. No claim of anatomical sheet identity, all-edge validation or physical winding sense."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Use a new output filename")
    result = check(args.source.resolve(), args.runner.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "seconds": result["seconds"], "by_condition": result["by_condition"]}))


if __name__ == "__main__":
    main()
