#!/usr/bin/env python3
"""Re-score the small public Q10 archive without raw CT, checkpoints or inference.

Requires NumPy, scikit-learn and sibling verify_results.py. The latter supplies
independent sklearn-based metrics, not producer code. Validates the exported
NPY member digests, fixed labels/core layout, pooled and per-core metrics, mean
fusion and the frozen engineering gates. Cannot verify CNN inference or geometry
readback from this archive alone; those need the full-output review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import sklearn

from verify_results import ARM_SCALES, PROTOCOL_SHA256, metrics, sha


def audit(evidence, protocol_path):
    started = time.monotonic()
    manifest = json.loads((evidence/"scored_arrays_manifest.json").read_text())
    result = json.loads((evidence/"results.json").read_text())
    protocol = json.loads(protocol_path.read_text())
    assert sha(protocol_path) == result["protocol_sha256"] == manifest["protocol_sha256"] == PROTOCOL_SHA256
    path = evidence/"scored_arrays.npz"
    assert path.stat().st_size == manifest["archive_bytes"] and sha(path) == manifest["archive_sha256"]
    # Check the actual uncompressed member bytes rather than reserializing with
    # the reviewer's NumPy version, which could change the NPY header encoding.
    with zipfile.ZipFile(path) as zipped:
        for name, result_key in (("probabilities", "aligned_probabilities_sha256"), ("states", "states_sha256")):
            digest = hashlib.sha256(zipped.read(name+".npy")).hexdigest()
            assert digest == manifest[name]["source_npy_sha256"] == result[result_key]
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"probabilities", "states", "author_indices", "core_origins_yx", "condition_names"}
        p, states, indices, origins, names = (archive[name] for name in
            ("probabilities", "states", "author_indices", "core_origins_yx", "condition_names"))
    names = names.tolist()
    assert names == [a[0] for a in ARM_SCALES]+["uniform_mean5"] == manifest["condition_names"]
    assert p.dtype == np.float32 and p.shape == (6, 53, 32, 32)
    assert list(p.shape) == manifest["probabilities"]["shape"] and str(p.dtype) == manifest["probabilities"]["dtype"]
    assert states.shape == (53, 32, 32) and np.isin(states, (0, 1, 2)).all()
    assert list(states.shape) == manifest["states"]["shape"] and str(states.dtype) == manifest["states"]["dtype"]
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    assert indices.shape == (53,) and np.issubdtype(indices.dtype, np.integer)
    assert origins.shape == (53, 2) and np.issubdtype(origins.dtype, np.integer)
    assert indices.tolist() == [c["index"] for c in protocol["controls"]]
    assert origins.tolist() == [[c["y"]+48, c["x"]+48] for c in protocol["controls"]]
    assert np.unique(indices).size == 53 and len(result["per_core"]) == 53
    occupied = set()
    for y, x in origins:
        cells = {(int(y+j), int(x+k)) for j in range(32) for k in range(32)}
        assert not occupied.intersection(cells)
        occupied.update(cells)
    assert len(occupied) == 54272
    assert tuple(int((states == i).sum()) for i in (2, 1, 0)) == (15791, 29558, 8923)
    expected_mean = np.add.reduce(p[:5], axis=0, dtype=np.float32)/np.float32(5)
    mean_error = float(np.max(np.abs(expected_mean.astype(np.float64)-p[5])))
    assert mean_error <= 1.2e-7
    mixed = np.flatnonzero(np.any(states == 1, axis=(1, 2)) & np.any(states == 2, axis=(1, 2)))
    assert len(mixed) == 23
    pooled = {name: metrics(a, states) for name, a in zip(names, p)}
    local = [{name: metrics(a[i], states[i]) for name, a in zip(names, p)} for i in range(53)]
    count, largest_error = 0, 0.0
    def compare(actual, expected):
        nonlocal count, largest_error
        if isinstance(expected, dict):
            for key, value in expected.items(): compare(actual[key], value)
        elif isinstance(expected, list):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected): compare(a, b)
        else:
            count += 1
            if isinstance(expected, float):
                difference = abs(actual-expected)
                largest_error = max(largest_error, difference)
                assert difference < 1e-12
            else: assert actual == expected
    compare(result["pooled"], pooled)
    for i, expected in enumerate(local):
        assert result["per_core"][i]["patch"] == i and result["per_core"][i]["author_index"] == int(indices[i])
        compare(result["per_core"][i]["metrics"], expected)
    comparisons = {}
    for name in names[1:]:
        changes = np.array([local[i][name]["ap"]-local[i]["identity"]["ap"] for i in mixed])
        row = dict(delta_AP=pooled[name]["ap"]-pooled["identity"]["ap"],
                   delta_AUC=pooled[name]["auc"]-pooled["identity"]["auc"],
                   delta_recall_at_0_5=pooled[name]["recall_at_0_5"]-pooled["identity"]["recall_at_0_5"],
                   delta_FPR_at_0_5=pooled[name]["false_positive_rate_at_0_5"]-pooled["identity"]["false_positive_rate_at_0_5"],
                   mixed_core_count=len(mixed), median_mixed_core_AP_delta=float(np.median(changes)),
                   mixed_cores_losing_more_than_0_10_AP=int((changes < -.10).sum()),
                   fraction_mixed_cores_losing_more_than_0_10_AP=float((changes < -.10).mean()),
                   mixed_core_AP_delta_range=[float(changes.min()), float(changes.max())],
                   materially_harmful_AP_loss=bool(pooled[name]["ap"]-pooled["identity"]["ap"] < protocol["primary"]["materially_harmful_delta_below"]))
        if name == "uniform_mean5":
            gate = protocol["secondary"]["fusion_advance_gate"]
            components = dict(pooled_AP_gain=row["delta_AP"] >= gate["minimum_pooled_AP_gain"],
                median_mixed_core_AP_delta=row["median_mixed_core_AP_delta"] >= gate["minimum_median_mixed_core_AP_delta"],
                FPR_increase=row["delta_FPR_at_0_5"] <= gate["maximum_FPR_increase"],
                mixed_core_harm=row["fraction_mixed_cores_losing_more_than_0_10_AP"] <= gate["maximum_fraction_mixed_cores_losing_more_than_0_10_AP"])
            row.update(advance_gate_components=components, advance_to_independent_control=all(components.values()))
        compare(result["comparisons"][name], row)
        comparisons[name] = row
    identity = json.loads((evidence/"identity_check.json").read_text())
    compare(identity["metrics"], pooled["identity"])
    gate = protocol["identity_gate"]
    assert identity["gate_passed"] is True and abs(pooled["identity"]["ap"]-gate["reference"]) <= gate["maximum_absolute_difference"]
    return dict(status="pass", archive_sha256=sha(path), archive_bytes=path.stat().st_size,
                protocol_sha256=PROTOCOL_SHA256, scalar_checks=count, maximum_metric_absolute_error=largest_error,
                mean5_maximum_absolute_error=mean_error, mean5_bitwise_equal=bool(np.array_equal(expected_mean, p[5])),
                known_pixels=45349, physical_core_pixels=54272, mixed_cores=23,
                pooled=pooled, comparisons=comparisons,
                checker_sha256=sha(Path(__file__)), independent_metric_helper_sha256=sha(Path(__file__).with_name("verify_results.py")),
                numpy=np.__version__, sklearn=sklearn.__version__, seconds=time.monotonic()-started,
                limitation="This small archive verifies scoring and frozen gates only; it cannot certify original CT, CNN inference, readback geometry or independent-scroll generalization.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence", "protocol", "report"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists(): parser.error("Use a fresh review filename")
    result = audit(args.evidence, args.protocol)
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "archive_bytes", "scalar_checks", "maximum_metric_absolute_error", "mean5_bitwise_equal", "seconds")}))


if __name__ == "__main__":
    main()
