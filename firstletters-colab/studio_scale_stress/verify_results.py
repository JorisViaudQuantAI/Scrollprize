#!/usr/bin/env python3
"""Independent, offline verifier of frozen Q10 Studio scale-stress outputs.

Imports no producer modules. Uses rational-coordinate four-corner interpolation
and sklearn's AP/AUC implementations. No model or source acquisition is run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

PROTOCOL_SHA256 = "47bb7e5b6556fb7d6aba6702011d7088957c6f3dbe5e9b2cc6c98f41e747f99d"
ARM_SCALES = (("identity", 1, 1), ("x_half", .5, 1), ("x_double", 2, 1),
              ("y_half", 1, .5), ("y_double", 1, 2))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def four_corners(full, y, fy, x, fx):
    """Literal four-corner average, ignoring zero-weight upper neighbours."""
    y1, x1 = y+(fy > 0), x+(fx > 0)
    wy, wx = fy[:, None], fx[None, :]
    field = full.astype(np.float64)
    result = ((1-wy)*(1-wx)*field[:, y[:, None], x[None, :]]
              + (1-wy)*wx*field[:, y[:, None], x1[None, :]]
              + wy*(1-wx)*field[:, y1[:, None], x[None, :]]
              + wy*wx*field[:, y1[:, None], x1[None, :]])
    return result.astype(np.float32)


def readback(full, sx, sy):
    """Four-corner average from exact integer numerator/denominator coordinates."""
    def axis(scale):
        j = np.arange(32)
        numerator, denominator = {1: (48+j, 1), .5: (223+2*j, 4), 2: (65+4*j, 2)}[scale]
        lo, rem = np.divmod(numerator, denominator)
        return lo, rem.astype(np.float64)/denominator
    y, fy = axis(sy)
    x, fx = axis(sx)
    return four_corners(full, y, fy, x, fx)


def readback_only(full_identity, sx, sy):
    """Identity scores -> scaled model grid -> original core, no model or Gaussian."""
    def source_axis(scale):
        j = np.arange(128)
        numerator, denominator = {1: (j, 1), .5: (4*j-127, 2), 2: (2*j+127, 4)}[scale]
        valid = np.flatnonzero((numerator >= 0) & (numerator <= 127*denominator))
        lo, rem = np.divmod(numerator[valid], denominator)
        return valid, lo, rem.astype(np.float64)/denominator
    valid_y, y, fy = source_axis(sy)
    valid_x, x, fx = source_axis(sx)
    # NaN makes any accidental consumption of unsupported virtual pixels visible.
    virtual = np.full(full_identity.shape, np.nan, dtype=np.float32)
    virtual[:, valid_y[:, None], valid_x[None, :]] = four_corners(full_identity, y, fy, x, fx)
    restored = readback(virtual, sx, sy)
    assert np.isfinite(restored).all()
    return restored


def metrics(p, state):
    known = state != 0
    truth, score = state[known] == 2, p[known]
    npos, nneg = int(truth.sum()), int((~truth).sum())
    chosen = score >= .5
    tp, fp = int(np.sum(chosen & truth)), int(np.sum(chosen & ~truth))
    return dict(ink_pixels=npos, background_pixels=nneg, unknown_pixels=int((~known).sum()),
                true_positive_at_0_5=tp, false_positive_at_0_5=fp,
                recall_at_0_5=tp/npos if npos else None,
                false_positive_rate_at_0_5=fp/nneg if nneg else None,
                ap=float(average_precision_score(truth, score)) if npos and nneg else None,
                auc=float(roc_auc_score(truth, score)) if npos and nneg else None)


def verify(prepared, output, cache, checkpoint, history):
    started = time.monotonic()
    assert sha(prepared/"protocol.json") == PROTOCOL_SHA256
    protocol = json.loads((prepared/"protocol.json").read_text())
    result = json.loads((output/"results.json").read_text())
    receipt = json.loads((output/"inference_receipt.json").read_text())
    assert result["protocol_sha256"] == receipt["protocol_sha256"] == PROTOCOL_SHA256
    checked = {}
    for name, item in protocol["acquisition"].items():
        path = cache/name
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
        checked[str(path.name)] = sha(path)
    for name, item in protocol["arrays"].items():
        path = prepared/name
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
        checked["prepared/"+name] = sha(path)
    if checkpoint is not None:
        assert checkpoint.stat().st_size == protocol["weights"]["bytes"]
        assert sha(checkpoint) == protocol["weights"]["sha256"]
        checked["checkpoint"] = sha(checkpoint)
    assert sha(output/"full_probabilities.npy") == receipt["full_probabilities_sha256"]
    assert sha(output/"aligned_probabilities.npy") == result["aligned_probabilities_sha256"]
    assert sha(output/"inference_receipt.json") == result["inference_receipt_sha256"]
    assert sha(prepared/"states.npy") == result["states_sha256"]
    full = np.load(output/"full_probabilities.npy", allow_pickle=False)
    aligned = np.load(output/"aligned_probabilities.npy", allow_pickle=False)
    states = np.load(prepared/"states.npy", allow_pickle=False)
    assert full.shape == (5, 53, 128, 128) and aligned.shape == (6, 53, 32, 32)
    assert full.dtype == aligned.dtype == np.float32 and np.isfinite(full).all() and np.isfinite(aligned).all()
    assert states.shape == (53, 32, 32) and np.isin(states, (0, 1, 2)).all()
    assert (int((states == 2).sum()), int((states == 1).sum()), int((states == 0).sum())) == (15791, 29558, 8923)
    labels = np.load(cache/"labels.npy", mmap_mode="r", allow_pickle=False)
    coordinates = np.load(cache/"coordinates.npy", allow_pickle=False)
    actual_selection, occupied = [], set()
    for index, (y, x) in enumerate(coordinates):
        state = labels[y+48:y+80, x+48:x+80]
        if np.any(state > 0): actual_selection.append(index)
    assert actual_selection == [c["index"] for c in protocol["controls"]]
    for i, item in enumerate(protocol["controls"]):
        y, x = item["y"], item["x"]
        assert np.array_equal(states[i], labels[y+48:y+80, x+48:x+80])
        cells = {(y+j, x+k) for j in range(48, 80) for k in range(48, 80)}
        assert not cells.intersection(occupied)
        occupied.update(cells)
    assert [(a["name"], a["sx"], a["sy"]) for a in protocol["arms"]] == list(ARM_SCALES)
    projected = np.stack([readback(full[i], sx, sy) for i, (_, sx, sy) in enumerate(ARM_SCALES)])
    mean = np.add.reduce(projected, axis=0, dtype=np.float32)/np.float32(5)
    reference = np.concatenate((projected, mean[None]))
    error = float(np.max(np.abs(reference.astype(np.float64)-aligned)))
    assert error <= 1.2e-7  # At most one FP32 ULP for differing arithmetic association.
    mixed = np.any(states == 1, axis=(1, 2)) & np.any(states == 2, axis=(1, 2))
    assert int(mixed.sum()) == 23
    names = [a[0] for a in ARM_SCALES] + ["uniform_mean5"]
    pooled = {name: metrics(p, states) for name, p in zip(names, aligned)}
    local = [{name: metrics(p[i], states[i]) for name, p in zip(names, aligned)} for i in range(53)]
    scalar_checks, maximum_metric_error = 0, 0.0
    def compare(actual, expected):
        nonlocal scalar_checks, maximum_metric_error
        if isinstance(expected, dict):
            for key, value in expected.items(): compare(actual[key], value)
        elif isinstance(expected, list):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected): compare(a, b)
        else:
            scalar_checks += 1
            if isinstance(expected, float):
                difference = abs(actual-expected)
                maximum_metric_error = max(maximum_metric_error, difference)
                assert difference < 1e-12
            else: assert actual == expected
    compare(result["pooled"], pooled)
    for i, expected in enumerate(local): compare(result["per_core"][i]["metrics"], expected)
    interpolation_controls = {}
    for name, sx, sy in ARM_SCALES[1:]:
        restored = readback_only(full[0], sx, sy)
        scored = metrics(restored, states)
        interpolation_controls[name] = dict(metrics=scored,
            delta_AP=scored["ap"]-pooled["identity"]["ap"],
            mean_absolute_probability_change=float(np.abs(restored-aligned[0]).mean()))
    compare(result["readback_only_control"], interpolation_controls)
    compared = {}
    for name in names[1:]:
        delta = np.array([local[i][name]["ap"]-local[i]["identity"]["ap"] for i in np.flatnonzero(mixed)])
        row = dict(delta_AP=pooled[name]["ap"]-pooled["identity"]["ap"],
                   delta_AUC=pooled[name]["auc"]-pooled["identity"]["auc"],
                   delta_recall_at_0_5=pooled[name]["recall_at_0_5"]-pooled["identity"]["recall_at_0_5"],
                   delta_FPR_at_0_5=pooled[name]["false_positive_rate_at_0_5"]-pooled["identity"]["false_positive_rate_at_0_5"],
                   mixed_core_count=23, median_mixed_core_AP_delta=float(np.median(delta)),
                   mixed_cores_losing_more_than_0_10_AP=int(np.count_nonzero(delta < -.1)),
                   fraction_mixed_cores_losing_more_than_0_10_AP=float(np.mean(delta < -.1)),
                   mixed_core_AP_delta_range=[float(delta.min()), float(delta.max())],
                   materially_harmful_AP_loss=bool(pooled[name]["ap"] < pooled["identity"]["ap"]-.05))
        if name == "uniform_mean5":
            gates = dict(pooled_AP_gain=row["delta_AP"] >= .02,
                         median_mixed_core_AP_delta=row["median_mixed_core_AP_delta"] >= 0,
                         FPR_increase=row["delta_FPR_at_0_5"] <= .02,
                         mixed_core_harm=row["mixed_cores_losing_more_than_0_10_AP"] <= 2)
            row.update(advance_gate_components=gates, advance_to_independent_control=all(gates.values()))
        compare(result["comparisons"][name], row)
        compared[name] = row
    identity_check = json.loads((output/"identity_check.json").read_text())
    compare(identity_check["metrics"], pooled["identity"])
    assert abs(pooled["identity"]["ap"]-.7536) <= .02 and identity_check["gate_passed"] is True
    historical = json.loads(history.read_text())
    return dict(status="pass", protocol_sha256=PROTOCOL_SHA256,
                scalar_checks=scalar_checks, maximum_metric_absolute_error=maximum_metric_error,
                manual_readback_bitwise_equal=bool(np.array_equal(reference, aligned)),
                manual_readback_maximum_absolute_error=error,
                label_values_exactly_preserved=True, known_pixels=45349, mixed_cores=23,
                distinct_physical_core_pixels=len(occupied), source_sha256=checked,
                historical_result_sha256=sha(history),
                identity_AP_delta_vs_historical=pooled["identity"]["ap"]-historical["metrics"]["ap"],
                identity_AUC_delta_vs_historical=pooled["identity"]["auc"]-historical["metrics"]["auc"],
                checked_comparisons=compared, verifier_sha256=sha(Path(__file__)),
                readback_only_controls_recomputed=True, checked_readback_only_controls=interpolation_controls,
                numpy=np.__version__, sklearn=sklearn.__version__, seconds=time.monotonic()-started,
                limits="Verifies arithmetic, labels and artifact identity; no model rerun, independent-donor validation or statistical significance claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "output", "source-cache", "history", "report"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.report.exists(): parser.error("Use a fresh review filename")
    result = verify(args.prepared, args.output, args.source_cache, args.checkpoint, args.history)
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "scalar_checks", "maximum_metric_absolute_error", "manual_readback_bitwise_equal", "manual_readback_maximum_absolute_error", "seconds")}))


if __name__ == "__main__":
    main()
