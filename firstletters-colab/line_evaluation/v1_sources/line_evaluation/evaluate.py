#!/usr/bin/env python3
"""Frozen localization diagnostic on released 306B/343 controls, not a text gate."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "line_review"))
from review import Config, outside_git, rank_map, write_packet

CASES = ("pherc306b_context_raw", "pherc306b_blank_raw", "pherc306b_shuffle_raw",
         "pherc306b_context_matched_raw", "pherc343_context_display")
TOP_K = (1, 3, 5, 10)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def target_coverage(proposals, targets, valid):
    """Union of original-grid band polygons, excluding unobserved target pixels."""
    union = np.zeros(valid.shape, np.uint8)
    for proposal in proposals:
        polygon = np.rint(proposal["band_polygon_xy"]).astype(np.int32)
        cv2.fillPoly(union, [polygon], 1)
    covered = (union > 0) & valid
    results = []
    h, w = valid.shape
    for target in targets:
        x0, y0, x1, y1 = target["box_xyxy"]
        x0, y0, x1, y1 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0))), min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Source target lies outside the extracted control")
        support = valid[y0:y1, x0:x1]
        denominator = int(support.sum())
        yc = min(h - 1, max(0, int(round((target["box_xyxy"][1] + target["box_xyxy"][3]) / 2))))
        line_support = valid[yc, x0:x1]
        line_n = int(line_support.sum())
        results.append(dict(id=target["id"], row=target.get("row"),
                            supported_box_pixels=denominator,
                            box_support_fraction=denominator / ((x1-x0)*(y1-y0)),
                            box_coverage=float(covered[y0:y1, x0:x1].sum()) / denominator if denominator else None,
                            centerline_support_pixels=line_n,
                            centerline_coverage=float(covered[yc, x0:x1].sum()) / line_n if line_n else None))
    return results


def timeout_handler(signum, frame):
    raise TimeoutError("Frozen 180-second per-map ranking budget exceeded")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    outside_git(args.output)
    if args.output.exists():
        ap.error("Use a new output directory")
    config = Config()
    manifests = {}
    for name in CASES:
        folder = args.data / "cases" / name
        m = json.loads((folder / "manifest.json").read_text())
        if m["id"] != name or digest(folder / m["map"]) != m["score_sha256"] or digest(folder / m["mask"]) != m["mask_sha256"]:
            raise ValueError("Prepared public control mismatch")
        manifests[name] = m
    args.output.mkdir(parents=True)
    protocol = dict(
        schema="line-localization-diagnostic-v1", frozen_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=digest(__file__), method_sha256=digest(HERE.parent / "line_review/review.py"),
        corpus_inventory_sha256=digest(args.data / "inventory.json"),
        cases=manifests, config=asdict(config), top_k=list(TOP_K),
        mask_policy="Known target pixels intersect supplied support only; other pixels are not negatives",
        primary_measure="At each K, report per-target fraction covered by union of proposal band polygons, in original pixel coordinates",
        supplementary_measure="Coverage of each supplied box's horizontal centerline; a coarse spatial diagnostic, not text recall",
        letter_box_summary="For343, also count boxes with at least50% supported-area coverage; never call this recognized letters",
        baseline="No model/glyph score; compare complete claim vs same claim under shuffle-common support explicitly",
        resource_limit="180 seconds of wall time per map for rank_map; timeout is incomplete, not zero proposals",
        interpretation="Geometric localization and implementation evidence on published controls; no candidate threshold, population FPR or unseen-model validation",
        adjustments_after_scores="none; preserve failures and incomplete searches",
    )
    save_json(args.output / "protocol.json", protocol)
    print("PROTOCOL_FROZEN", digest(args.output / "protocol.json"), flush=True)
    cv2.setNumThreads(4)
    signal.signal(signal.SIGALRM, timeout_handler)
    results = []
    for name in CASES:
        m = manifests[name]
        folder = args.data / "cases" / name
        score = np.load(folder / m["map"], allow_pickle=False)
        valid = np.load(folder / m["mask"], allow_pickle=False)
        signal.alarm(180)
        start = time.monotonic()
        try:
            report = rank_map(score, valid, m["spacing_um_yx"], m["map_kind"], config)
        except TimeoutError:
            results.append(dict(id=name, status="incomplete_timeout", elapsed=time.monotonic()-start))
            print("INCOMPLETE", name, flush=True)
            continue
        finally:
            signal.alarm(0)
        report["input_sha256"], report["mask_sha256"] = m["score_sha256"], m["mask_sha256"]
        report["script_sha256"] = protocol["method_sha256"]
        write_packet(score, valid, report, args.output / name)
        coverage = {str(k): target_coverage(report["proposals"][:k], m["targets"], valid) for k in TOP_K}
        rec = dict(id=name, status="completed", map_kind=m["map_kind"], role=m["role"],
                   seconds=report["seconds"], valid_area_cm2=report["valid_area_cm2"],
                   proposals=len(report["proposals"]), search_truncated=report["search_truncated"],
                   maximum_score=report["proposals"][0]["score"] if report["proposals"] else None,
                   trials=report["trials"], target_coverage=coverage,
                   top10_proposals=report["proposals"][:10],
                   no_automated_text_or_blank_verdict=True)
        if name.startswith("pherc343"):
            rec["letter_boxes_at_least_half_covered"] = {
                k: sum(t["box_coverage"] is not None and t["box_coverage"] >= .5 for t in ts)
                for k, ts in coverage.items()}
        results.append(rec)
        print("RESULT", json.dumps({k: rec[k] for k in ["id","seconds","proposals","search_truncated","maximum_score"]}), flush=True)
    final = dict(protocol_sha256=digest(args.output / "protocol.json"), results=results,
                 limitation="Five related control cases, not five independent scrolls; geometric coverage is not legibility; shuffle/background evidence has source-stated limits")
    save_json(args.output / "results.json", final)
    print("FINISHED", len(results), "cases", flush=True)


if __name__ == "__main__":
    main()
