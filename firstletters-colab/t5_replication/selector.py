"""Deterministic label-only selection; no CT or feature-score input.

Selecting balanced fields from a model reference changes the target population;
this is conditional spatial replication, not independent ink validation.
"""
from __future__ import annotations

import math

import numpy as np


def rectangle_gap_squared(a, b):
    """Squared Euclidean gap between half-open y0,y1,x0,x1 rectangles."""
    dy = max(a[0] - b[1], b[0] - a[1], 0)
    dx = max(a[2] - b[3], b[2] - a[3], 0)
    return int(dy * dy + dx * dx)


def _block_counts(mask, size):
    h, w = mask.shape
    h, w = h // size * size, w // size * size
    reduced = np.add.reduceat(mask[:h, :w], np.arange(0, h, size),
                              axis=0, dtype=np.int64)
    return np.add.reduceat(reduced, np.arange(0, w, size),
                           axis=1, dtype=np.int64)


def select(reference, development_bounds, *, count=3, alignment=128,
           shape=(512, 640), min_class_fraction=.20,
           min_nonzero_fraction=.90, min_gap_px=1024, border_guard_px=64):
    """Choose farthest eligible fields, ties by increasing y0 then x0.

    Full fields AND guarded cores must pass the same label-fraction thresholds.
    There is no relaxation/replacement when fewer than count fields qualify.
    """
    reference = np.asarray(reference)
    if reference.ndim != 2 or reference.dtype != np.uint8:
        raise ValueError("Reference must be a uint8 two-dimensional map")
    if alignment <= 0 or any(s % alignment or s <= 0 for s in shape):
        raise ValueError("Shape must be positive and divisible by alignment")
    if not 0 <= border_guard_px < min(shape) / 2 or count < 1:
        raise ValueError("Invalid border guard or requested count")
    block = math.gcd(alignment, border_guard_px)
    if min(reference.shape) < block:
        raise ValueError("Reference is too small")
    high = _block_counts(reference >= 179, block)
    low = _block_counts((reference > 0) & (reference <= 64), block)
    nonzero = _block_counts(reference > 0, block)
    bh, bw = (s // block for s in shape)
    guard, stride = border_guard_px // block, alignment // block
    totals = [int(np.prod(shape)), int(np.prod([s - 2 * border_guard_px for s in shape]))]
    minima = [[int(np.ceil(total * min_class_fraction)),
               int(np.ceil(total * min_nonzero_fraction))] for total in totals]
    eligible, inspected = [], 0
    for by in range(0, high.shape[0] - bh + 1, stride):
        for bx in range(0, high.shape[1] - bw + 1, stride):
            inspected += 1
            slices = [np.s_[by:by + bh, bx:bx + bw],
                      np.s_[by + guard:by + bh - guard, bx + guard:bx + bw - guard]]
            counts = [[int(a[sl].sum()) for a in (high, low, nonzero)] for sl in slices]
            if any(min(c[:2]) < m[0] or c[2] < m[1] for c, m in zip(counts, minima)):
                continue
            y0, x0 = by * block, bx * block
            eligible.append({"bounds_y0_y1_x0_x1": [y0, y0 + shape[0], x0, x0 + shape[1]],
                             "full_counts_high_low_nonzero": counts[0],
                             "core_counts_high_low_nonzero": counts[1]})
    chosen, anchors = [], [list(development_bounds)]
    while len(chosen) < count:
        ranked = []
        for item in eligible:
            b = item["bounds_y0_y1_x0_x1"]
            distance = min(rectangle_gap_squared(b, a) for a in anchors)
            if distance >= min_gap_px ** 2 and b not in anchors:
                ranked.append((-distance, b[0], b[2], item))
        if not ranked:
            break
        distance, _, _, item = min(ranked, key=lambda x: x[:3])
        chosen.append({**item, "min_gap_squared_px_at_selection": -distance})
        anchors.append(item["bounds_y0_y1_x0_x1"])
    return {
        "schema": "t5-spatial-replication.selection.v1",
        "status": "complete" if len(chosen) == count else "unassessable",
        "selection_inputs": "published model reference only; no CT or feature scores",
        "reference_shape_yx": list(reference.shape),
        "parameters": {"count": count, "alignment_px": alignment,
                       "region_shape_yx": list(shape), "border_guard_px": border_guard_px,
                       "min_class_fraction": min_class_fraction,
                       "min_nonzero_fraction": min_nonzero_fraction,
                       "full_and_core_minimum_class_nonzero_counts": minima,
                       "min_rectangle_gap_px": min_gap_px,
                       "selection_order": "greedy maximum minimum squared Euclidean rectangle gap from development and previously selected fields; ties y0 then x0 ascending"},
        "development_bounds_y0_y1_x0_x1": list(development_bounds),
        "candidate_count": inspected, "eligible_count": len(eligible),
        "selected": chosen, "eligible_candidates": eligible,
        "no_fallback": True,
    }
