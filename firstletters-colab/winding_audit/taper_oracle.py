#!/usr/bin/env python3
"""Independent analytic oracle for winding-sense chain traversal.

This constructs synthetic integer-coordinate tracks from known spiral surfaces,
then drives one or more CLI-compatible sense_chain.py programs. The oracle's
expected radial increment is analytic and does not import traversal helpers.

The physical surface is r(theta,z) = 500 + pitch*theta/(2*pi) + taper*z.
All graph edges are reciprocal exact XYZ crossings. The three transverse arcs
cover a little more than a turn and are joined by two longitudinal tracks.
Each transverse arc has at least 201 points. Each bridge moves 16 slices or less.

A true +20-voxel CW pitch with taper=-1 and end_z-start_z=32 produces dr=-12.
Reflection reverses dr to +12. Both pass the historical 5<abs(dr)<60 band, yet
both imply the WRONG physical sense if taper is ignored. Returning to the start
slice removes this confound. Rasterization introduces <=2 voxel oracle tolerance.

Example:
  python taper_oracle.py --algorithm fixed=/path/to/sense_chain.py \
      --output /tmp/taper-oracle/result.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


CENTER_YX = (1000, 1000)
INTERVALS = ((0.0, 2.4), (2.0, 4.6), (4.2, 6.8))
CROSSING_ANGLES = (2.2, 4.4)
ORACLE_TOLERANCE_VOXELS = 2.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def surface_point(theta, z, pitch, taper):
    """Round an analytic physical point to the track store's integer XYZ grid."""
    theta, z = np.broadcast_arrays(np.asarray(theta), np.asarray(z))
    radius = 500.0 + pitch * theta / (2 * np.pi) + taper * z
    yy = CENTER_YX[0] + radius * np.sin(theta)
    xx = CENTER_YX[1] + radius * np.cos(theta)
    return np.rint(np.stack((z, yy, xx), axis=-1)).astype("<i4")


def construct_fixture(directory, pitch, taper, z_path):
    """Build an exact crossing graph independently of the algorithm under test."""
    directory.mkdir(parents=True)
    transverse, theta_arrays = [], []
    for (lo, hi), z in zip(INTERVALS, z_path):
        knots = [a for a in CROSSING_ANGLES if lo <= a <= hi]
        theta = np.unique(np.r_[np.linspace(lo, hi, 201), knots])
        transverse.append(surface_point(theta, z, pitch, taper))
        theta_arrays.append(theta)
    all_tracks = list(transverse)
    edges = []
    for j, theta in enumerate(CROSSING_ANGLES):
        left, right = j, j + 1
        ia = int(np.flatnonzero(theta_arrays[left] == theta)[0])
        ib = int(np.flatnonzero(theta_arrays[right] == theta)[0])
        z_start, z_stop = z_path[left], z_path[right]
        z_values = np.linspace(z_start, z_stop, abs(z_stop - z_start) + 1)
        bridge = surface_point(theta, z_values, pitch, taper)
        vertical_id = len(all_tracks)
        all_tracks.append(bridge)
        edges += [(left, ia, vertical_id, 0),
                  (right, ib, vertical_id, len(bridge) - 1)]

    crossing_rows = [[] for _ in all_tracks]
    for ta, ia, tb, ib in edges:
        assert np.array_equal(all_tracks[ta][ia], all_tracks[tb][ib])
        crossing_rows[ta].append((tb, ia, ib))
        crossing_rows[tb].append((ta, ib, ia))
    records = [item for rows in crossing_rows for item in rows]
    cross_offsets = np.r_[0, np.cumsum([len(v) for v in crossing_rows])].astype("<i8")
    track_offsets = np.r_[0, np.cumsum([len(v) for v in all_tracks])].astype("<i8")
    coordinates = np.concatenate(all_tracks)
    z_bounds = np.asarray([[v[:, 0].min(), v[:, 0].max()] for v in all_tracks], dtype="<i4")
    families = np.array([0, 0, 0, 1, 1], dtype="i1")
    track_offsets.tofile(directory / "offsets.i64")
    coordinates.tofile(directory / "coordinates.i32")
    z_bounds.tofile(directory / "z_bounds.i32")
    families.tofile(directory / "family_codes.i8")
    np.savez(directory / "crossings.npz", offsets=cross_offsets,
             partners=np.asarray([r[0] for r in records], dtype="<i8"),
             self_local=np.asarray([r[1] for r in records], dtype="<i8"),
             partner_local=np.asarray([r[2] for r in records], dtype="<i8"))
    umbilicus = {"control_points": [dict(z=z, y=CENTER_YX[0], x=CENTER_YX[1])
                                     for z in (-64, 64)]}
    (directory / "umbilicus.json").write_text(json.dumps(umbilicus))

    # Independent input checks. No algorithm-under-test code is loaded here.
    assert min(map(len, transverse)) >= 201
    assert max(abs(np.diff(z_path))) <= 16
    assert len(records) == 8
    for t, rows in enumerate(crossing_rows):
        for partner, local, other_local in rows:
            assert (t, other_local, local) in crossing_rows[partner]
            assert np.array_equal(all_tracks[t][local], all_tracks[partner][other_local])
    crossing_radial_changes = []
    for j, theta in enumerate(CROSSING_ANGLES):
        exact_change = taper * (z_path[j + 1] - z_path[j])
        assert abs(exact_change) <= 16
        crossing_radial_changes.append(exact_change)
    return {"surface_equation": "r(theta,z)=500+pitch*theta/(2*pi)+taper*z",
            "pitch_voxels_per_turn": pitch, "taper_voxels_per_slice": taper,
            "z_path": list(z_path), "transverse_lengths": list(map(len, transverse)),
            "transverse_theta_intervals": INTERVALS,
            "crossing_angles": CROSSING_ANGLES,
            "exact_reciprocal_crossings": True, "directed_crossing_count": len(records),
            "bridge_radial_changes_voxels": crossing_radial_changes,
            "bridge_slice_changes": np.diff(z_path).tolist(),
            "integer_coordinate_rounding": "nearest-even per XYZ coordinate",
            "coordinate_sha256": sha256(directory / "coordinates.i32")}


def run_algorithm(script, directory, mirror):
    output = directory / ("results-mirrored.npy" if mirror == -1 else "results-plain.npy")
    command = [sys.executable, str(script), str(directory), str(directory / "crossings.npz"),
               str(directory / "umbilicus.json"), str(output),
               "16", "40", "3", "20", "0.3", "200", "16", "1500", str(mirror)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if completed.returncode:
        raise RuntimeError(f"Algorithm failed ({script}): {completed.stderr}")
    result_lines = [s for s in completed.stdout.splitlines() if s.strip().startswith("{")]
    if not result_lines:
        raise RuntimeError("Algorithm emitted no JSON summary")
    summary = json.loads(result_lines[-1])
    chains = np.load(output, allow_pickle=False)
    if not len(chains):
        chains = chains.reshape(0, 3)
    if chains.ndim != 2 or chains.shape[1] != 3 or not np.isfinite(chains).all():
        raise ValueError("Unexpected chain result encoding")
    return summary, chains


def run_suite(algorithms):
    records = []
    with tempfile.TemporaryDirectory(prefix="winding-taper-oracle-") as temp:
        for name, script in algorithms:
            for pitch in (20, -20):
                definitions = (
                    ("untapered_drifting_z", 0, (0, 16, 32)),
                    ("tapered_drifting_z", -int(np.sign(pitch)), (0, 16, 32)),
                    ("tapered_same_z_closure", -int(np.sign(pitch)), (0, 16, 0)),
                )
                for case, taper, z_path in definitions:
                    folder = Path(temp) / name / str(pitch) / case
                    fixture = construct_fixture(folder, pitch, taper, z_path)
                    for mirror in (1, -1):
                        summary, chains = run_algorithm(script, folder, mirror)
                        delta_z = (z_path[-1] - z_path[0]) * mirror
                        physical_turn_pitch = pitch * mirror
                        expected_dr = physical_turn_pitch + taper * delta_z
                        matched = (len(chains) == 1 and
                                   abs(float(chains[0, 0]) - expected_dr) <= ORACLE_TOLERANCE_VOXELS and
                                   float(chains[0, 1]) == delta_z and chains[0, 2] == 2)
                        observed_dr = float(chains[0, 0]) if len(chains) == 1 else None
                        corrected_for_taper = (observed_dr - taper * float(chains[0, 1])
                                               if observed_dr is not None else None)
                        records.append({"algorithm": name, "case": case, "mirror": mirror,
                                        "fixture": fixture, "true_physical_pitch_in_test_frame": physical_turn_pitch,
                                        "analytic_expected_dr": expected_dr, "analytic_expected_delta_z": delta_z,
                                        "oracle_tolerance_voxels": ORACLE_TOLERANCE_VOXELS,
                                        "traversal_matches_analytic_surface": bool(matched),
                                        "raw_sign_is_wrong_physical_sense": (bool(np.sign(observed_dr) != np.sign(physical_turn_pitch))
                                                                            if observed_dr is not None else None),
                                        "observed_dr_in_historical_pitch_band": (bool(5 < abs(observed_dr) < 60)
                                                                                 if observed_dr is not None else None),
                                        "dr_after_known_taper_correction": corrected_for_taper,
                                        "chain_rows_dr_dz_hops": chains.tolist(), "algorithm_summary": summary})
    return {"schema": "winding-taper-oracle-v1", "synthetic_analytic_control": True,
            "purpose": "Separate arithmetic traversal correctness from winding-sense identifiability under z drift",
            "true_winding_convention": "Image x right/y down: positive fixed-z radial pitch is CW",
            "algorithm_pins": {name: sha256(script) for name, script in algorithms},
            "oracle_script_sha256": sha256(__file__),
            "records": records,
            "interpretation": "Even exact crossings and mirror sign reversal cannot determine fixed-z winding sense from dr alone when end_z differs from start_z and radial taper is unbounded. The taper correction here is oracle-known; it is not estimated from the observations.",
            "limits": "Synthetic geometry and integer rounding; demonstrates an identifiability counterexample, not taper or wrong sense on any real scroll."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", action="append", required=True, metavar="NAME=SCRIPT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    algorithms = []
    for specification in args.algorithm:
        name, path = specification.split("=", 1)
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            parser.error("Algorithm names must contain only letters, digits, hyphen or underscore")
        algorithms.append((name, Path(path).resolve(strict=True)))
    if len({name for name, _ in algorithms}) != len(algorithms):
        parser.error("Algorithm names must be unique")
    if args.output.exists():
        parser.error("Output already exists; retain prior result and choose a fresh path")
    result = run_suite(algorithms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    summary = {}
    for name, _ in algorithms:
        rows = [r for r in result["records"] if r["algorithm"] == name]
        summary[name] = {"runs": len(rows),
                         "analytic_traversal_matches": sum(r["traversal_matches_analytic_surface"] for r in rows),
                         "wrong_sense_despite_pitch_band": sum(r["raw_sign_is_wrong_physical_sense"] is True and
                                                              r["observed_dr_in_historical_pitch_band"] is True for r in rows)}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
