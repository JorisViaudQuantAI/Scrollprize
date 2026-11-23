"""Evaluate the frozen R50 on a preregistered panel of public PHerc. 841 labels."""
import argparse
import base64
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import label
import torch

from r50_model import DEPTH, MODEL, REVISION, WEIGHT_SHA, WIDTH, load_model, predict_tile
from r50_stack import array_sha256, depth_order
from r50_tile import LABELS, VOLUME, array, as_mask, cell_counts


PROTOCOL_COMMIT = "125d1c45a20cf6e80bf1a362f6fe4d6478245a36"
PREVIOUS_X, PREVIOUS_Y, PREVIOUS_W, PREVIOUS_H = 12544, 11264, 278, 272
STRATA = (("near_blank", 0.0, 0.01, 0.0), ("sparse", 0.05, 0.20, 0.125),
          ("mixed", 0.35, 0.65, 0.5))
ORDERS = ("asc", "desc", "permutation")
THRESHOLDS = (("0.5", 0.5), ("0.8", 0.8))
NEIGHBORS = np.ones((3, 3), bool)


def ratio(a, b):
    return a / b if b else None


def public_selection():
    urls = [LABELS + name + ".zarr" for name in ("inklabels", "supervision", "validation")]
    full = [array(url, 0) for url in urls]
    shape = tuple(full[0].shape)
    if len(shape) != 2 or any(tuple(a.shape) != shape for a in full):
        raise ValueError("Public annotation grids differ")
    ink, sup, val = [as_mask(array(url, 4)[:]) for url in urls]
    if not (ink.shape == sup.shape == val.shape):
        raise ValueError("Overview annotation grids differ")
    trusted = sup & val
    counts = cell_counts(trusted, 16)
    positives = cell_counts(trusted & ink, 16)
    candidates = []
    for row, col in np.argwhere(counts >= 0.75 * 16 * 16):
        y, x = int(row) * WIDTH, int(col) * WIDTH
        if y + WIDTH > shape[0] or x + WIDTH > shape[1]:
            continue
        intersects_previous = (x < PREVIOUS_X+PREVIOUS_W and x+WIDTH > PREVIOUS_X
                               and y < PREVIOUS_Y+PREVIOUS_H and y+WIDTH > PREVIOUS_Y)
        if intersects_previous:
            continue
        key = hashlib.sha256(f"841-w00-panel-v1:{y}:{x}".encode("ascii")).hexdigest()
        candidates.append((x, y, float(positives[row, col] / counts[row, col]),
                           int(counts[row, col]), key))
    if not candidates:
        raise ValueError("No separated public validation areas at the overview coverage minimum")

    @lru_cache(maxsize=400)
    def labels_at(x, y):
        window = np.s_[y:y+WIDTH, x:x+WIDTH]
        i, s, v = (as_mask(m[window]) for m in full)
        trust = s & v
        trusted_count = int(trust.sum())
        pos = int((i & trust).sum())
        return i, trust, trusted_count, pos, trusted_count - pos

    picked = []
    stratum_counts = Counter()
    def available(x, y, gap):
        return all(max(abs(x-p["x"]), abs(y-p["y"])) >= gap for p in picked)

    def register(x, y, stratum):
        i, t, n, p, q = labels_at(x, y)
        picked.append(dict(x=x, y=y, stratum=stratum, trusted=n, positives=p, negatives=q,
                           trusted_fraction=n / (WIDTH*WIDTH), ink_fraction=ratio(p, n),
                           ink_mask_sha256=array_sha256(i), trusted_mask_sha256=array_sha256(t)))
        stratum_counts[stratum] += 1

    ranks = {name: sorted(candidates, key=lambda a: (abs(a[2]-target), a[4]))[:72]
             for name, _, _, target in STRATA}
    for gap in (2*WIDTH, WIDTH):
        for name, low, high, _ in STRATA:
            for x, y, _, _, _ in ranks[name]:
                if stratum_counts[name] >= 2:
                    break
                if not available(x, y, gap):
                    continue
                _, _, n, p, _ = labels_at(x, y)
                if n >= .9*WIDTH*WIDTH and low <= p/n <= high:
                    register(x, y, name)
    for x, y, _, _, _ in sorted(candidates, key=lambda a: (-a[3], a[4]))[:120]:
        if len(picked) >= 6:
            break
        if available(x, y, WIDTH) and labels_at(x, y)[2] >= .75*WIDTH*WIDTH:
            register(x, y, "additional")
    if not picked:
        raise ValueError("The full-resolution masks did not support any candidate tile")
    missing = {name: 2-stratum_counts[name] for name, _, _, _ in STRATA if stratum_counts[name] < 2}
    return picked, labels_at, shape, missing


def confusion(hit, ink, trusted, components):
    pos, neg = ink & trusted, ~ink & trusted
    tp, fp = int((hit & pos).sum()), int((hit & neg).sum())
    out = dict(tp=tp, fp=fp, fn=int(pos.sum())-tp, tn=int(neg.sum())-fp)
    if components:
        false_mask = hit & neg
        islands, ni = label(false_mask, structure=NEIGHBORS)
        regions, nr = label(neg, structure=NEIGHBORS)
        sizes = np.bincount(islands.ravel())[1:]
        out.update(fp_islands=int(ni), fp_islands_ge16=int((sizes >= 16).sum()),
                   largest_fp_island=int(sizes.max(initial=0)), background_regions=int(nr),
                   background_regions_hit=int(np.unique(regions[false_mask]).size))
    return rates(out)


def rates(out):
    out["fpr"] = ratio(out["fp"], out["fp"]+out["tn"])
    out["recall"] = ratio(out["tp"], out["tp"]+out["fn"])
    return out


def merge_tables(tables):
    out = {}
    for order in (*ORDERS, "all_three"):
        out[order] = {}
        for key, _ in THRESHOLDS:
            parts = [t[order][key] for t in tables]
            row = {k: sum(v[k] for v in parts) for k in parts[0]
                   if k not in ("fpr", "recall", "largest_fp_island")}
            if key == "0.5":
                row["largest_fp_island"] = max(p["largest_fp_island"] for p in parts)
            out[order][key] = rates(row)
    return out


def merge_retention(retention):
    out = {}
    for key, _ in THRESHOLDS:
        v = {k: sum(r[key][k] for r in retention) for k in
             ("positive_asc", "positive_retained", "negative_asc", "negative_retained")}
        v["positive_retained_fraction"] = ratio(v["positive_retained"], v["positive_asc"])
        v["negative_retained_fraction"] = ratio(v["negative_retained"], v["negative_asc"])
        out[key] = v
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.set_num_interop_threads(1)
    picked, labels_at, label_shape, missing = public_selection()
    compact = lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
    selection_hash = hashlib.sha256(compact(picked).encode()).hexdigest()
    print("FROZEN_PUBLIC_WINDOWS " + compact(picked) + " " + selection_hash, flush=True)
    surface = array(VOLUME, 0)
    if tuple(surface.shape) != (109, *label_shape) or surface.dtype != np.uint8:
        raise ValueError(f"Unexpected public surface volume geometry {surface.shape} {surface.dtype}")
    model = load_model(args.cache, download=True)
    order_indices = {order: depth_order(order) for order in ORDERS}
    rows = []
    for window in picked:
        x, y = window["x"], window["y"]
        ink, trusted, _, _, _ = labels_at(x, y)
        raw = np.asarray(surface[23:85, y:y+WIDTH, x:x+WIDTH])
        if raw.shape != (DEPTH, WIDTH, WIDTH) or raw.dtype != np.uint8:
            raise ValueError("Incomplete public CT window")
        maps = {order: predict_tile(model, raw if order == "asc" else raw[list(indices)])
                for order, indices in order_indices.items()}
        metrics = {order: {} for order in (*ORDERS, "all_three")}
        retention = {}
        for key, threshold in THRESHOLDS:
            masks = {order: prob >= threshold for order, prob in maps.items()}
            masks["all_three"] = masks["asc"] & masks["desc"] & masks["permutation"]
            for order, hit in masks.items():
                metrics[order][key] = confusion(hit, ink, trusted, components=(key == "0.5"))
            both = masks["asc"] & masks["permutation"] & trusted
            retention[key] = {
                "positive_asc": metrics["asc"][key]["tp"], "positive_retained": int((both & ink).sum()),
                "negative_asc": metrics["asc"][key]["fp"], "negative_retained": int((both & ~ink).sum()),
            }
        rows.append(dict(**window, raw_sha256=array_sha256(raw),
                         probability_sha256={order: array_sha256(p) for order, p in maps.items()},
                         metrics=metrics, permutation_retention=merge_retention([retention])))
        print(f"Scored public {window['stratum']} window at XY {x},{y}", flush=True)

    def summarize(group):
        return dict(tiles=len(group), positives=sum(g["positives"] for g in group),
                    negatives=sum(g["negatives"] for g in group),
                    metrics=merge_tables([g["metrics"] for g in group]),
                    permutation_retention=merge_retention([g["permutation_retention"] for g in group]))

    strata = sorted({r["stratum"] for r in rows})
    report = dict(protocol_commit=PROTOCOL_COMMIT, source_code_commit=os.environ.get("GITHUB_SHA"),
                  source_surface=VOLUME, source_labels=LABELS, model=MODEL, model_revision=REVISION,
                  weight_sha256=WEIGHT_SHA, scan_pitch_um=2.403, tile_width_mm=WIDTH*2.403/1000,
                  depth_range_half_open=[23,85], shape=[DEPTH,WIDTH,WIDTH],
                  depth_order_indices={k:list(v) for k,v in order_indices.items()},
                  thresholds=[.5,.8], connectivity=8, eligibility="published validation AND supervision",
                  selection_sha256=selection_hash, quotas_unfilled=missing,
                  forward_passes=len(rows)*len(ORDERS), pooled=summarize(rows),
                  strata={k:summarize([r for r in rows if r["stratum"] == k]) for k in strata},
                  tiles=rows, torch=str(torch.__version__), numpy=np.__version__)
    data = (compact(report) + "\n").encode()
    args.output.write_bytes(data)
    print("R50_PANEL_SHA256 " + hashlib.sha256(data).hexdigest(), flush=True)
    # Preserve the exact numerical JSON in bounded, text-safe log records.
    payload = base64.b64encode(data).decode("ascii")
    for start in range(0, len(payload), 3000):
        print(f"R50_PANEL_PART {start//3000:03d} {payload[start:start+3000]}", flush=True)


if __name__ == "__main__":
    main()
