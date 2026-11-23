"""Label-anchored metrics for a fixed Hecate reconstruction comparison.

This module performs no inference, source access, selection, or image repair.
The two-dimensional common scoring core is 256 x 256 at 9.6 um. A native fine
core is 1024 x 1024 at 2.4 um: each original 256 x 256 prediction contributed
its [64:192, 64:192] centre to that core. Consequently the old central-crop
boundaries lie at common-grid coordinates 32, 64, ..., 224.

All cutoffs are descriptive, label-informed operating points. Matching a
within-field background rank does not calibrate a threshold for a new scroll,
make overlapping pixel pairs independent, or establish letter legibility.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


CORE_SIDE = 256
FINE_SCALE = 4
SEAM_STEP = 32
BACKGROUND_BUDGET_DENOMINATOR = 20


def _validate_field(score, labels, support):
    score, labels, support = map(np.asarray, (score, labels, support))
    expected = (CORE_SIDE, CORE_SIDE)
    if score.shape != expected or labels.shape != expected or support.shape != expected:
        raise ValueError("Score, labels, and support must share the fixed 256x256 core")
    if score.dtype.kind not in "fiu" or labels.dtype.kind not in "iu":
        raise ValueError("Scores must be real numeric values and labels must be integers")
    if support.dtype != np.dtype(bool):
        raise ValueError("Support must be a boolean mask")
    if not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("Labels must be 0=unknown, 1=background, or 2=ink")
    supported_scores = score[support]
    if not np.isfinite(supported_scores).all():
        raise ValueError("Every supported score, including unknown-label cells, must be finite")
    if ((supported_scores < 0) | (supported_scores > 1)).any():
        raise ValueError("Supported probabilities must be between zero and one")
    known = support & (labels != 0)
    ink = known & (labels == 2)
    background = known & (labels == 1)
    if not ink.any() or not background.any():
        raise ValueError("Both supported known ink and background classes are required")
    return score, labels, support, known, ink, background


def _operating_point(score, ink, background):
    """Use the (floor(Nbg/20)+1)-th largest background score as a strict cutoff.

    Whole ties are excluded. The observed background count can therefore be
    below the integer budget; no random tie-breaking or interpolation is used.
    Computing the budget as Nbg//20 avoids floating-point floor ambiguity.
    """
    background_values = score[background]
    background_count = int(background_values.size)
    budget = background_count // BACKGROUND_BUDGET_DENOMINATOR
    threshold = float(np.sort(background_values)[::-1][budget])
    positive = np.zeros(score.shape, dtype=bool)
    known = ink | background
    positive[known] = score[known] > threshold
    background_positive = int(np.count_nonzero(positive & background))
    ink_positive = int(np.count_nonzero(positive & ink))
    return positive, {
        "threshold_strict_greater": threshold,
        "threshold_background_rank_zero_based": budget,
        "allowed_background_count": budget,
        "background_positive_count": background_positive,
        "background_tied_at_threshold_count": int(np.count_nonzero(background_values == threshold)),
        "unused_background_budget_due_to_ties": budget - background_positive,
        "ink_positive_count": ink_positive,
        "ink_tied_at_threshold_count": int(np.count_nonzero(score[ink] == threshold)),
        "recall_at_background_5pct": float(ink_positive / np.count_nonzero(ink)),
        "background_exceedance": float(background_positive / background_count),
    }


def _pair_summary(positive, class_mask, crossing_seam, seam_period):
    """Count undirected 4-neighbour pairs once, with both endpoints eligible.

    For ink, both-positive fraction measures recovery of an existing labelled
    adjacency. For background it measures false adjacent predictions. Neither
    count treats a predicted connection through unknown/background as a true
    ink connection, and no morphological operation modifies predictions.
    """
    result = {"eligible_pairs": 0, "both_positive_pairs": 0,
              "exactly_one_positive_pairs": 0, "both_negative_pairs": 0}
    axes = {}
    for axis, name in ((0, "vertical"), (1, "horizontal")):
        first = [slice(None), slice(None)]
        second = [slice(None), slice(None)]
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        pair_support = class_mask[first] & class_mask[second]
        # Pair index k crosses the boundary between cells k and k+1.
        at_seam = (np.arange(1, CORE_SIDE) % seam_period) == 0
        seam_mask = at_seam[:, None] if axis == 0 else at_seam[None, :]
        pair_support &= seam_mask if crossing_seam else ~seam_mask
        left, right = positive[first], positive[second]
        counts = {
            "eligible_pairs": int(np.count_nonzero(pair_support)),
            "both_positive_pairs": int(np.count_nonzero(pair_support & left & right)),
            "exactly_one_positive_pairs": int(np.count_nonzero(pair_support & (left ^ right))),
            "both_negative_pairs": int(np.count_nonzero(pair_support & ~left & ~right)),
        }
        axes[name] = counts
        for key in result:
            result[key] += counts[key]
    denominator = result["eligible_pairs"]
    result["both_positive_fraction"] = (
        float(result["both_positive_pairs"] / denominator) if denominator else None)
    result["exactly_one_positive_fraction"] = (
        float(result["exactly_one_positive_pairs"] / denominator) if denominator else None)
    result["by_axis"] = axes
    return result


def measure(score, labels, support, seam_period=SEAM_STEP):
    """Return whole-field rank metrics and fixed-grid labelled-pair recovery.

    The same single field/arm cutoff is used for every seam and within-cell
    pair. A missing seam-pair class is retained as zero denominator/None, not
    filled with a favourable rate. Callers must mark such endpoints unassessable.
    """
    if not isinstance(seam_period, (int, np.integer)) or isinstance(seam_period, (bool, np.bool_)):
        raise ValueError("Seam period must be an integer")
    if seam_period < 1 or seam_period >= CORE_SIDE or CORE_SIDE % seam_period:
        raise ValueError("Seam period must be a positive proper divisor of the 256-pixel core")
    score, labels, support, known, ink, background = _validate_field(score, labels, support)
    positive, operating_point = _operating_point(score, ink, background)
    truth, values = ink[known], score[known]
    result = {
        "shape": [CORE_SIDE, CORE_SIDE],
        "seam_period": int(seam_period),
        "known": int(np.count_nonzero(known)),
        "ink": int(np.count_nonzero(ink)),
        "background": int(np.count_nonzero(background)),
        "unknown_or_unsupported": int(known.size - np.count_nonzero(known)),
        "supported_unknown": int(np.count_nonzero(support & (labels == 0))),
        "unsupported": int(np.count_nonzero(~support)),
        "ap": float(average_precision_score(truth, values)),
        "auc": float(roc_auc_score(truth, values)),
        "operating_point": (
            "One label-informed whole-field cutoff per arm; strict score greater than "
            "descending background[floor(Nbackground/20)]. Ties excluded. "
            "Descriptive comparison, not a transported detector threshold or significance test."
        ),
        **operating_point,
        "pairs": {
            "seam_ink": _pair_summary(positive, ink, True, seam_period),
            "seam_background": _pair_summary(positive, background, True, seam_period),
            "within_cell_ink": _pair_summary(positive, ink, False, seam_period),
            "within_cell_background": _pair_summary(positive, background, False, seam_period),
        },
    }
    return result


def field_metrics(score, labels, support):
    """Compatibility name for the fixed 32-pixel seam experiment."""
    return measure(score, labels, support, seam_period=SEAM_STEP)


def compare(central_metrics, hann_metrics, *, minimum_pair_count=20,
            minimum_seam_ink_gain=.05, maximum_regression=.01,
            maximum_seam_background_increase=.01):
    """Apply a new reconstruction-method decision, separately to each export.

    This deliberately does not call or amend the old native-detail advancement
    gate. A pass warrants a separately frozen larger known-glyph replication;
    it authorizes no unread-region extension. Counted neighbour pairs overlap
    and are not independent observations.
    """
    for name in ("shape", "seam_period", "known", "ink", "background", "unsupported", "supported_unknown"):
        if central_metrics[name] != hann_metrics[name]:
            raise ValueError(f"Paired metrics must share labels/support/geometry: {name}")
    relevant_pairs = ("seam_ink", "seam_background", "within_cell_ink")
    pair_counts = {}
    for name in relevant_pairs:
        first = central_metrics["pairs"][name]["eligible_pairs"]
        second = hann_metrics["pairs"][name]["eligible_pairs"]
        if first != second:
            raise ValueError(f"Paired metrics have different eligible pair counts: {name}")
        pair_counts[name] = first
    limits = {
        "minimum_pair_count": minimum_pair_count,
        "minimum_seam_ink_gain": minimum_seam_ink_gain,
        "maximum_regression": maximum_regression,
        "maximum_seam_background_increase": maximum_seam_background_increase,
    }
    result = {
        "pass": False,
        "reason": "insufficient_eligible_labelled_pairs",
        "eligible_pair_counts": pair_counts,
        "limits": limits,
        "scope": "Method-selection only; no old-gate change, statistical significance, complete-glyph claim, or unread extension.",
    }
    if any(count < minimum_pair_count for count in pair_counts.values()):
        return result
    def value(item, name):
        if name in relevant_pairs:
            return item["pairs"][name]["both_positive_fraction"]
        return item[name]
    names = (*relevant_pairs, "ap", "recall_at_background_5pct")
    central = {name: value(central_metrics, name) for name in names}
    hann = {name: value(hann_metrics, name) for name in names}
    if any(v is None or not np.isfinite(v) for v in (*central.values(), *hann.values())):
        result["reason"] = "nonfinite_or_missing_endpoint"
        return result
    checks = {
        "seam_ink_gain": hann["seam_ink"] >= central["seam_ink"] + minimum_seam_ink_gain,
        "ap_noninferiority": hann["ap"] >= central["ap"] - maximum_regression,
        "field_recall_noninferiority": hann["recall_at_background_5pct"] >= central["recall_at_background_5pct"] - maximum_regression,
        "within_cell_ink_noninferiority": hann["within_cell_ink"] >= central["within_cell_ink"] - maximum_regression,
        "seam_background_pair_limit": hann["seam_background"] <= central["seam_background"] + maximum_seam_background_increase,
    }
    result.update({
        "pass": bool(all(checks.values())),
        "reason": "all_new_method_conditions_pass" if all(checks.values()) else "new_method_condition_failed",
        "checks": checks,
        "hann_minus_central": {name: float(hann[name] - central[name]) for name in names},
    })
    return result


def common_grid(fine_core, *, uint8_export=False):
    """Reduce a complete 1024x1024 fine core by exact 4x4 area means.

    If uint8_export is true, first apply the released Hecate export operation
    np.rint(255*p).astype(uint8) to every fine pixel. Quantizing the common-grid
    mean afterwards would be a different operation. Float reductions match the
    earlier native-resolution runner: float64 mean, then float32 result.
    """
    fine_core = np.asarray(fine_core)
    if fine_core.shape != (CORE_SIDE * FINE_SCALE,) * 2:
        raise ValueError("A complete 1024x1024 fine core is required")
    if fine_core.dtype.kind not in "fiu" or not np.isfinite(fine_core).all():
        raise ValueError("The entire fine core must contain finite real probabilities")
    if ((fine_core < 0) | (fine_core > 1)).any():
        raise ValueError("Fine probabilities must be between zero and one")
    if uint8_export:
        # The official inference accumulator/export is float32.
        values = np.rint(fine_core.astype(np.float32) * np.float32(255)).astype(np.uint8)
        means = values.reshape(CORE_SIDE, FINE_SCALE, CORE_SIDE, FINE_SCALE).mean((1, 3), dtype=np.float64)
        return (means / 255.0).astype(np.float32)
    return fine_core.reshape(CORE_SIDE, FINE_SCALE, CORE_SIDE, FINE_SCALE).mean((1, 3), dtype=np.float64).astype(np.float32)


def evaluate_fine_core(fine_core, labels, support):
    """Apply identical endpoints before and after the official uint8 export.

    Both variants recompute their own strict ordinal background cutoff. Their
    actual background exceedance and tie counts remain visible. An equal
    nominal 5% budget is not falsely described as equal realised FPR if ties
    cause the exported variant to use less of that budget.
    """
    return {
        "float_probability": field_metrics(common_grid(fine_core), labels, support),
        "uint8_export": field_metrics(common_grid(fine_core, uint8_export=True), labels, support),
        "export_order": "Round each fine probability to uint8 first; average exact4x4 export pixels; divide255.",
    }
