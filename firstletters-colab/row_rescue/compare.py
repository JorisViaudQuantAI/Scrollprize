#!/usr/bin/env python3
"""Label-free row-rescue fourth proposal from an archived public top-ten line list."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import cv2
import numpy as np

SEP_MM = 3.0
SIGMA_MM = 1.25
K = 4
POOL = 10
CASES = ("pherc306b_context_raw", "pherc306b_blank_raw", "pherc306b_shuffle_raw",
         "pherc306b_context_matched_raw", "pherc343_context_display")

def blob_sha1(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def centre_y(p):
    return sum(y for x, y in p["band_polygon_xy"]) / 4

def rescue_indices(proposals, spacing_um_y):
    """Keep existing top three. Insert a supported distant long row into slot four."""
    n = min(len(proposals), POOL)
    if n <= 3:
        return list(range(n))
    ps = proposals[:n]
    yy = [centre_y(p) * spacing_um_y / 1000 for p in ps]
    if max(yy[:3]) - min(yy[:3]) >= SEP_MM:
        return list(range(min(K, n)))
    weight = [max(0.0, float(p["score"])) for p in ps]
    centre = (sum(yy[i] * weight[i] for i in range(3)) / sum(weight[:3])
              if sum(weight[:3]) else sum(yy[:3]) / 3)
    eligible = [i for i in range(3, n) if abs(yy[i] - centre) >= SEP_MM]
    if not eligible:
        return list(range(min(K, n)))
    def objective(i):
        votes = sum(weight[j] * float(ps[j]["length_mm"]) *
                    math.exp(-0.5 * ((yy[i] - yy[j]) / SIGMA_MM) ** 2)
                    for j in range(n))
        return weight[i] * float(ps[i]["length_mm"]) * votes
    return [0, 1, 2, max(eligible, key=lambda i: (objective(i), -i))]

def exact_coverage(proposals, targets, valid):
    band = np.zeros(valid.shape, np.uint8)
    for p in proposals:
        cv2.fillPoly(band, [np.rint(p["band_polygon_xy"]).astype(np.int32)], 1)
    shown = (band > 0) & valid
    h, w = valid.shape
    ret = []
    for target in targets:
        x0, y0, x1, y1 = target["box_xyxy"]
        x0, y0, x1, y1 = (max(0, int(np.floor(x0))), max(0, int(np.floor(y0))),
                          min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1))))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Published target outside image")
        count = int(valid[y0:y1, x0:x1].sum())
        yc = min(h - 1, max(0, int(round((target["box_xyxy"][1] + target["box_xyxy"][3]) / 2))))
        row_count = int(valid[yc, x0:x1].sum())
        ret.append({"id": target["id"], "row": target.get("row"),
                    "supported_box_pixels": count, "supported_centerline_pixels": row_count,
                    "box_coverage": float(shown[y0:y1, x0:x1].sum()) / count if count else None,
                    "centerline_coverage": float(shown[yc, x0:x1].sum()) / row_count if row_count else None})
    return {"targets": ret,
            "fraction_valid_map_selected": float(shown.sum()) / float(valid.sum()) if valid.any() else None,
            "boxes_at_least_half_covered": sum(t["box_coverage"] is not None and t["box_coverage"] >= .5 for t in ret)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "line_evaluation/results.v2.json")
    p.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "line_evaluation/protocol.v2.json")
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.output.exists() or repo == args.output.resolve() or repo in args.output.resolve().parents:
        p.error("Output must be new and outside Git")
    rb, pb = args.results.read_bytes(), args.protocol.read_bytes()
    freeze = json.loads((Path(__file__).parent / "freeze.json").read_text())
    if blob_sha1(rb) != freeze["results_git_blob_sha1"] or blob_sha1(pb) != freeze["protocol_git_blob_sha1"]:
        raise ValueError("Archived public-line inputs differ from frozen blobs")
    result = {r["id"]: r for r in json.loads(rb)["results"]}
    protocol = json.loads(pb)["cases"]
    records = []
    for name in CASES:
        meta, item = protocol[name], result[name]
        root = args.data / "cases" / name
        manifest = json.loads((root / "manifest.json").read_text())
        if (manifest["id"] != name or manifest["mask_sha256"] != meta["mask_sha256"]
                or manifest["score_sha256"] != meta["score_sha256"]):
            raise ValueError("Frozen public control manifest mismatch")
        f = root / manifest["mask"]
        if digest(f) != meta["mask_sha256"]:
            raise ValueError("Changed validity mask")
        valid = np.load(f, allow_pickle=False)
        if valid.dtype != bool or tuple(valid.shape) != tuple(meta["shape"]):
            raise ValueError("Unexpected mask dtype or shape")
        candidates = item["top10_proposals"]
        base = list(range(min(K, len(candidates))))
        rescue = rescue_indices(candidates, meta["spacing_um_yx"][0])
        records.append({
            "id": name, "archived_proposal_count": item["proposals"],
            "baseline_ranks_1based": [i + 1 for i in base],
            "rescue_ranks_1based": [i + 1 for i in rescue],
            "baseline": exact_coverage([candidates[i] for i in base], meta["targets"], valid),
            "rescue": exact_coverage([candidates[i] for i in rescue], meta["targets"], valid)})
    receipt = {
        "schema": "public-line-row-rescue-v1", "results_git_blob_sha1": freeze["results_git_blob_sha1"],
        "protocol_git_blob_sha1": freeze["protocol_git_blob_sha1"],
        "parameters": {"separation_mm": SEP_MM, "sigma_mm": SIGMA_MM, "pool": POOL, "band_count": K},
        "cases": records,
        "limitations": ["Post-hoc development on previously seen published cases, not blind held-out validation",
                        "No labels enter selection; proposals are not independent models",
                        "Geometric review aid only; no reading, text recognition, calibrated false-positive rate or prize claim"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(receipt, out, indent=2, allow_nan=False)
        out.write("\n")
    for c in records:
        print("CASE", c["id"], "baseline", c["baseline_ranks_1based"],
              "rescue", c["rescue_ranks_1based"], flush=True)
        for idx, target in enumerate(c["rescue"]["targets"]):
            print("TARGET", target["id"],
                  "baseline", c["baseline"]["targets"][idx]["centerline_coverage"],
                  "rescue", target["centerline_coverage"], flush=True)

if __name__ == "__main__":
    main()
