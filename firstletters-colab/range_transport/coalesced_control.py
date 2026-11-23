#!/usr/bin/env python3
"""Frozen public 0841 chunk: measure exact Hecate-sized ROI via gap-coalesced HTTP ranges.

Only the source object's pinned public control bytes are processed; no model inference.
Never writes raw CT or URLs/tokens to disk. No retries or fallback to whole chunks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from run_transport import RangeClient, TransportError, outside_git, digest

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "coalesced_protocol.json"


def spans_for_box(box, shape):
    z0, y0, x0, z1, y1, x1 = box
    dz, dy, dx = shape
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in box + shape):
        raise ValueError("integer coordinates only")
    if not (0 <= z0 < z1 <= dz and 0 <= y0 < y1 <= dy and 0 <= x0 < x1 <= dx):
        raise ValueError("invalid box")
    return [((z * dy + y) * dx + x0, (z * dy + y) * dx + x1 - 1)
            for z in range(z0, z1) for y in range(y0, y1)]


def coalesce(spans, max_gap):
    if not isinstance(max_gap, int) or isinstance(max_gap, bool) or max_gap < 0:
        raise ValueError("invalid gap")
    groups = []
    for start, end in spans:
        if groups and start - groups[-1][1] - 1 <= max_gap:
            if start <= groups[-1][1]:
                raise ValueError("overlapping spans not allowed")
            groups[-1] = (groups[-1][0], end)
        else:
            if groups and start <= groups[-1][1]:
                raise ValueError("unsorted spans")
            groups.append((start, end))
    return groups


def reconstruct(groups, payloads, originals):
    """Recover just requested bytes; group overfetch must not enter the ROI."""
    if len(groups) != len(payloads):
        raise ValueError("group count mismatch")
    if any(len(p) != end-start+1 for (start, end), p in zip(groups, payloads)):
        raise ValueError("payload length mismatch")
    output = bytearray()
    g = 0
    for start, end in originals:
        while g < len(groups) and groups[g][1] < start:
            g += 1
        if g == len(groups) or groups[g][0] > start or groups[g][1] < end:
            raise ValueError("uncovered ROI row")
        offset = start - groups[g][0]
        output.extend(payloads[g][offset:offset + end-start+1])
    return bytes(output)


def check_freeze(commit):
    if len(commit) != 40 or not all(c in "0123456789abcdef" for c in commit):
        raise ValueError("full freeze commit sha required")
    repo = HERE.parents[1]
    for file in (Path(__file__), PROTOCOL):
        rel = file.relative_to(repo)
        pinned = subprocess.check_output(["git", "show", f"{commit}:{rel.as_posix()}"], cwd=repo)
        if pinned != file.read_bytes():
            raise ValueError("published code/protocol changed after freeze")


def run(spec, *, client):
    shape, box, source = spec["shape_zyx"], spec["box_zyx"], spec["source"]
    original = spans_for_box(box, shape)
    if len(original) != spec["expected_roi_rows"]:
        raise ValueError("row count disagrees with preregistration")
    plans = {}
    for policy in spec["policies"]:
        spans = coalesce(original, policy["max_gap_bytes"])
        size = sum(b-a+1 for a, b in spans)
        if len(spans) != policy["expected_requests"] or size != policy["expected_body_bytes"]:
            raise ValueError("plan disagrees with preregistration")
        plans[policy["name"]] = spans
    if sum(len(q) for q in plans.values()) + 2 > spec["limits"]["max_requests_including_redirects"]:
        raise ValueError("request plan exceeds budget")
    started = time.monotonic()
    whole, tag, resolved = client.get(source["url"], object_size=source["bytes"])
    if len(whole) != source["bytes"] or digest(whole) != source["sha256"]:
        raise TransportError("whole_public_source_hash_mismatch")
    baseline_seconds = time.monotonic() - started
    expected = b"".join(whole[a:b+1] for a, b in original)
    if len(expected) != spec["expected_roi_bytes"]:
        raise ValueError("ROI byte length mismatch")
    results = []
    for policy in spec["policies"]:
        groups = plans[policy["name"]]
        before = client.body_bytes_read
        then = time.monotonic()
        chunks = []
        for a, b in groups:
            if time.monotonic() - started > spec["limits"]["soft_wall_seconds"]:
                raise TransportError("soft_wall_elapsed")
            chunk, _, _ = client.get(resolved, object_size=source["bytes"], byte_range=(a, b), etag=tag)
            chunks.append(chunk)
        actual = reconstruct(groups, chunks, original)
        if actual != expected:
            raise TransportError("range_value_mismatch")
        results.append({
            "name": policy["name"], "range_requests": len(groups),
            "application_body_bytes": client.body_bytes_read - before,
            "elapsed_seconds": round(time.monotonic() - then, 6),
            "roi_sha256": digest(actual), "mismatches": 0,
            "overfetch_ratio_to_roi": (client.body_bytes_read - before) / len(expected),
        })
    return {
        "schema": "scrollprize.coalesced-public-control.v1", "status": "verified",
        "source_identity": "PHerc0841 published Zarr control, pinned earlier", "source_sha256": source["sha256"],
        "roi_sha256": digest(expected), "roi_bytes": len(expected), "whole_get_seconds": round(baseline_seconds, 6),
        "whole_get_application_body_bytes": len(whole), "results": results,
        "request_count_including_redirects": len(client.requests), "total_application_body_bytes": client.body_bytes_read,
        "no_training_or_model_forwards": True, "no_unread_scan_access": True,
        "limitations": "One fixed whole-first run on a public HF object; speed not a controlled throughput comparison; no proof about native Paris4 S3, TLS or full curved-surface source support.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-commit", required=True)
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    check_freeze(ns.freeze_commit)
    path = outside_git(ns.output)
    if path.exists():
        raise ValueError("refuse to overwrite receipt")
    spec = json.loads(PROTOCOL.read_text())
    budget = spec["limits"]
    client = RangeClient(body_cap=budget["max_application_body_bytes"],
                         request_cap=budget["max_requests_including_redirects"])
    try:
        receipt = run(spec, client=client)
    except Exception as err:
        receipt = {"schema": "scrollprize.coalesced-public-control.v1", "status": "failed",
                   "failure": type(err).__name__ if not isinstance(err, TransportError) else str(err),
                   "request_count_including_redirects": len(client.requests),
                   "total_application_body_bytes": client.body_bytes_read,
                   "no_training_or_model_forwards": True, "no_unread_scan_access": True}
    receipt["freeze_commit"] = ns.freeze_commit
    receipt["protocol_sha256"] = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    receipt["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt["transport_statuses"] = [x.get("status") for x in client.requests]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print("COALESCED_RESULT_JSON=" + json.dumps(receipt, sort_keys=True), flush=True)
    if receipt["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
