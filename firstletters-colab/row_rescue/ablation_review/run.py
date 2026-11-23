#!/usr/bin/env python3
"""Retrospective selector ablation on fixed published proposal lists; no maps."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import time


LAB = Path(__file__).resolve().parents[2]
RESULTS_BLOB = "a614144401fefec6ca49fd124de2a28b2a28dc07"
PROTOCOL_BLOB = "a36c130e8d2d00c660ff2e5d6fbd8449d970645a"
CASES = ("pherc306b_context_raw", "pherc306b_blank_raw",
         "pherc306b_shuffle_raw", "pherc306b_context_matched_raw",
         "pherc343_context_display")


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def git_blob(payload):
    return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()


def inspect(proposals, spacing_um_y):
    """Independently express the published selector and two simpler ablations."""
    ps = proposals[:10]
    n = len(ps)
    names = ("published_rule", "longest_eligible", "score_times_length_eligible",
             "highest_score_eligible")
    baseline = list(range(min(4, n)))
    record = {"pool_count": n, "baseline_indices_zero_based": baseline}
    if n <= 3:
        record.update(reason="At most three proposals; all are retained",
                      selected_indices_zero_based={name: baseline for name in names})
        return record
    if not math.isfinite(spacing_um_y) or spacing_um_y <= 0:
        raise ValueError("Invalid declared physical spacing")
    y = [sum(point[1] for point in p["band_polygon_xy"]) / 4 * spacing_um_y / 1000
         for p in ps]
    weight = [max(0.0, float(p["score"])) for p in ps]
    lengths = [float(p["length_mm"]) for p in ps]
    if not all(math.isfinite(x) for x in y + weight + lengths):
        raise ValueError("Nonfinite proposal value")
    span = max(y[:3]) - min(y[:3])
    record["top_three_y_span_mm"] = span
    if span >= 3.0:
        record.update(reason="Published diversity condition is not triggered",
                      selected_indices_zero_based={name: baseline for name in names})
        return record
    centre = (sum(y[i] * weight[i] for i in range(3)) / sum(weight[:3])
              if sum(weight[:3]) else sum(y[:3]) / 3)
    eligible = [i for i in range(3, n) if abs(y[i] - centre) >= 3.0]
    record["original_fourth_distance_from_top_three_centre_mm"] = abs(y[3] - centre)
    record["eligible_indices_zero_based"] = eligible
    if not eligible:
        record.update(reason="No eligible distant proposal",
                      selected_indices_zero_based={name: baseline for name in names})
        return record
    q = [weight[i] * lengths[i] for i in range(n)]
    vote = [sum(q[j] * math.exp(-0.5 * ((y[i] - y[j]) / 1.25) ** 2)
                for j in range(n)) for i in range(n)]
    objectives = {
        "published_rule": [q[i] * vote[i] for i in range(n)],
        "longest_eligible": lengths,
        "score_times_length_eligible": q,
        "highest_score_eligible": weight,
    }
    chosen = {name: max(eligible, key=lambda i: (objective[i], -i))
              for name, objective in objectives.items()}
    selected = chosen["published_rule"]
    record.update(
        reason="Published condition triggered; compare fixed objectives",
        selected_indices_zero_based={name: [0, 1, 2, i] for name, i in chosen.items()},
        original_fourth_length_mm=lengths[3],
        published_replacement_length_mm=lengths[selected],
        published_replacement_self_vote_fraction=(q[selected] / vote[selected]
                                                  if vote[selected] else None),
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing result")
    started = time.perf_counter()
    sources = {
        "line_evaluation/results.v2.json": (LAB / "line_evaluation/results.v2.json").read_bytes(),
        "line_evaluation/protocol.v2.json": (LAB / "line_evaluation/protocol.v2.json").read_bytes(),
        "row_rescue/freeze.json": (LAB / "row_rescue/freeze.json").read_bytes(),
        "row_rescue/compare.py": (LAB / "row_rescue/compare.py").read_bytes(),
    }
    rb = sources["line_evaluation/results.v2.json"]
    pb = sources["line_evaluation/protocol.v2.json"]
    freeze = json.loads(sources["row_rescue/freeze.json"])
    if git_blob(rb) != RESULTS_BLOB or git_blob(pb) != PROTOCOL_BLOB:
        raise ValueError("Archived proposal/geometry input differs from the original freeze")
    if (freeze["results_git_blob_sha1"] != RESULTS_BLOB
            or freeze["protocol_git_blob_sha1"] != PROTOCOL_BLOB):
        raise ValueError("Row-rescue freeze does not identify the expected inputs")
    results = {r["id"]: r for r in json.loads(rb)["results"]}
    protocol = json.loads(pb)["cases"]
    cases = []
    for name in CASES:
        row = inspect(results[name]["top10_proposals"], protocol[name]["spacing_um_yx"][0])
        row["id"] = name
        row["archived_total_proposal_count"] = results[name]["proposals"]
        selections = row["selected_indices_zero_based"]
        row["both_simpler_lists_identical_to_published_rule"] = all(
            selections[key] == selections["published_rule"]
            for key in ("longest_eligible", "score_times_length_eligible"))
        cases.append(row)
    for name, payload in sources.items():
        if (LAB / name).read_bytes() != payload:
            raise ValueError("Input changed during comparison")
    receipt = {
        "schema": "scrollprize.row-rescue.retrospective-ablation.v1",
        "status": "complete",
        "scope": "Retrospective arithmetic ablation of known published proposal lists; no new method or gate",
        "source_sha256": {name: sha(payload) for name, payload in sources.items()},
        "source_git_blob_sha1": {"results": RESULTS_BLOB, "protocol": PROTOCOL_BLOB},
        "script_sha256": sha(Path(__file__).read_bytes()),
        "cases": cases,
        "all_five_lists_identical_for_both_simpler_rules": all(
            row["both_simpler_lists_identical_to_published_rule"] for row in cases),
        "native_masks_read": 0, "score_maps_read": 0, "coverage_recomputed": False,
        "CNN_forwards": 0, "network_requests": 0, "new_thresholds_chosen": False,
        "old_gate_changed": False, "unread_data_used": False,
        "interpretation": "Identical selected lists imply identical coverage and area for any fixed mask; no such metrics were newly computed. These cases do not demonstrate an advantage from the Gaussian neighborhood vote.",
        "limitations": ["Two306B cases share one previously seen field",
                        "343 is a no-op and both negative controls have no proposals",
                        "No independent positive/ordinary-blank validation or review-time measurement",
                        "This does not invalidate the original reported recovery or its development gate"],
        "dependencies": {"python": platform.python_version(), "third_party_packages": []},
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"],
                      "all_five_lists_identical": receipt["all_five_lists_identical_for_both_simpler_rules"],
                      "wall_seconds": receipt["wall_seconds"]}))


if __name__ == "__main__":
    main()
