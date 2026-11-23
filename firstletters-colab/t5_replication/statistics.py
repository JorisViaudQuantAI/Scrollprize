"""Frozen feature comparison against a model reference; no fitting or acquisition.

Spatial ranks diagnose misalignment, not valid FWER control or significance:
conditional ROI selection, a model-derived reference and nonstationarity remain.
Block intervals describe these three fields, not independent donor uncertainty.
"""
from __future__ import annotations

import numpy as np


FEATURES = ("raw_ct_surface", "oct_hilbert_env_peak", "wst_s1_scale_slope")
POLARITIES = (1, 1, -1)
SHAPE = (512, 640)


def _labels(reference):
    return np.where(reference >= 179, 1,
                    np.where((reference > 0) & (reference <= 64), 0, -1))


class _RankedScore:
    """Weighted Mann–Whitney AUC, with half credit for exactly tied scores."""

    def __init__(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise ValueError("Scores must be a finite one-dimensional array")
        self.order = np.argsort(scores, kind="stable")
        ordered = scores[self.order]
        self.starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]]) if len(scores) else np.array([], int)

    def auc(self, labels, weights=None):
        labels = np.asarray(labels)
        if labels.shape != self.order.shape or not np.isin(labels, [-1, 0, 1]).all():
            raise ValueError("Labels must match scores and use -1, 0, 1")
        weights = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=np.float64)
        if weights.shape != labels.shape or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Weights must be finite, nonnegative and match labels")
        if not len(labels):
            return float("nan")
        lab, weight = labels[self.order], weights[self.order]
        positive = np.add.reduceat(weight * (lab == 1), self.starts)
        negative = np.add.reduceat(weight * (lab == 0), self.starts)
        n_positive, n_negative = positive.sum(), negative.sum()
        if n_positive == 0 or n_negative == 0:
            return float("nan")
        return float(np.sum(positive * (np.cumsum(negative) - .5 * negative)) /
                     (n_positive * n_negative))


def _translation_offsets(shape, count, rng):
    """Ordered uniform sampling without replacement, excluding only (0, 0)."""
    flat = rng.choice(int(np.prod(shape)) - 1, size=count, replace=False) + 1
    return np.column_stack(np.divmod(flat, shape[1]))


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _prepare(case, block_size):
    valid, reference = np.asarray(case["valid"]), np.asarray(case["reference"])
    if valid.shape != SHAPE or valid.dtype != np.bool_:
        raise ValueError("Every valid mask must be bool with shape (512, 640)")
    if reference.shape != SHAPE or reference.dtype != np.uint8:
        raise ValueError("Every reference must be uint8 with shape (512, 640)")
    if set(case["features"]) != set(FEATURES):
        raise ValueError("Exactly the three frozen feature names are required")
    common, scores = valid.copy(), []
    for name, polarity in zip(FEATURES, POLARITIES):
        array = np.asarray(case["features"][name])
        if array.shape != SHAPE or array.dtype.kind not in "iuf":
            raise ValueError("Feature arrays must be real numeric with shape (512, 640)")
        array = array.astype(np.float64) * polarity
        common &= np.isfinite(array)
        scores.append(array)
    indices = np.flatnonzero(common)
    row, column = np.divmod(indices, SHAPE[1])
    return {"reference": reference, "common": common,
            "labels": _labels(reference[common]),
            "scores": [_RankedScore(score[common]) for score in scores],
            "blocks": (row // block_size) * (SHAPE[1] // block_size) + column // block_size,
            "geometry_valid_count": int(valid.sum()), "common_valid_count": int(common.sum())}


def _evaluate_labels(prepared, labels, weights=None):
    counts = np.array([np.sum(labels == 1), np.sum(labels == 0)], dtype=np.int64) if weights is None else np.array(
        [np.sum(weights[labels == 1]), np.sum(weights[labels == 0])], dtype=np.int64)
    auc = np.full(3, np.nan) if (counts == 0).any() else np.array(
        [score.auc(labels, weights) for score in prepared["scores"]])
    return auc, counts


def evaluate(cases, seed=20261123, null_replicates=199, bootstrap_replicates=199, block_size=128):
    """Compare three frozen features on exactly three 512 x 640 cases.

    Finite support is intersected across all features with the supplied geometry
    mask. Zero reference pixels and values 65..178 are excluded. Missing classes
    invalidate the assessment; no null or bootstrap draw is dropped or redrawn.
    """
    if len(cases) != 3:
        raise ValueError("Exactly three cases are required")
    names = [case["name"] for case in cases]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 3:
        raise ValueError("Case names must be distinct nonempty strings")
    for name, value in (("seed", seed), ("null_replicates", null_replicates),
                        ("bootstrap_replicates", bootstrap_replicates), ("block_size", block_size)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be an integer in its declared range")
    if block_size != 128:
        raise ValueError("This fixed definition requires block_size=128")
    if null_replicates >= np.prod(SHAPE):
        raise ValueError("Too many unique nonzero toroidal translations requested")
    prepared = [_prepare(case, block_size) for case in cases]
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(6)]
    block_count = (SHAPE[0] // block_size) * (SHAPE[1] // block_size)
    observed = np.full((3, 3), np.nan)
    null = np.full((null_replicates, 3, 3), np.nan)
    boot = np.full((bootstrap_replicates, 3, 3), np.nan)
    null_counts = np.zeros((null_replicates, 3, 2), dtype=np.int64)
    boot_counts = np.zeros((bootstrap_replicates, 3, 2), dtype=np.int64)
    offsets, draws, case_reports, failures = [], [], [], []
    for index, item in enumerate(prepared):
        observed[index], counts = _evaluate_labels(item, item["labels"])
        case_reports.append({"name": names[index], "geometry_valid_count": item["geometry_valid_count"],
                             "common_valid_count": item["common_valid_count"],
                             "ink_count": int(counts[0]), "blank_count": int(counts[1]),
                             "excluded_reference_count": int((item["labels"] == -1).sum()),
                             "auc": dict(zip(FEATURES, observed[index]))})
        if (counts == 0).any():
            failures.append({"stage": "observed", "case": names[index], "class_counts_ink_blank": counts})
        shifts = _translation_offsets(SHAPE, null_replicates, streams[index])
        offsets.append(shifts)
        for replicate, shift in enumerate(shifts):
            moved = np.roll(item["reference"], tuple(shift), axis=(0, 1))
            null[replicate, index], null_counts[replicate, index] = _evaluate_labels(item, _labels(moved[item["common"]]))
        selection = streams[index + 3].integers(0, block_count, size=(bootstrap_replicates, block_count))
        draws.append(selection)
        for replicate, blocks in enumerate(selection):
            weights = np.bincount(blocks, minlength=block_count)[item["blocks"]]
            boot[replicate, index], boot_counts[replicate, index] = _evaluate_labels(item, item["labels"], weights)
    for stage, counts in (("null", null_counts), ("bootstrap", boot_counts)):
        for replicate, index in np.argwhere((counts == 0).any(axis=2)):
            failures.append({"stage": stage, "case": names[index], "replicate": int(replicate),
                             "class_counts_ink_blank": counts[replicate, index]})
    macro = observed.mean(axis=0)
    null_macro, boot_macro = null.mean(axis=1), boot.mean(axis=1)
    delta, boot_delta = macro[1:] - macro[0], boot_macro[:, 1:] - boot_macro[:, :1]
    null_max = null_macro[:, 1:].max(axis=1)
    ranks, intervals, gates = {}, {}, {}
    for index, feature in enumerate(FEATURES[1:]):
        rank = ((1 + np.count_nonzero(null_max >= macro[index + 1])) / (null_replicates + 1)
                if np.isfinite(null_max).all() and np.isfinite(macro[index + 1]) else np.nan)
        interval = (np.percentile(boot_delta[:, index], [2.5, 97.5], method="linear")
                    if np.isfinite(boot_delta[:, index]).all() else np.full(2, np.nan))
        ranks[feature], intervals[feature] = rank, interval
        checks = {"each_case_auc_ge_0_70": bool((observed[:, index + 1] >= .70).all()),
                  "macro_delta_ge_0_05": bool(delta[index] >= .05),
                  "bootstrap_lower_gt_0": bool(interval[0] > 0),
                  "spatial_rank_le_0_05": bool(rank <= .05), "all_assessable": not failures}
        gates[feature] = {**checks, "pass": all(checks.values())}
    return _json_safe({
        "schema": "t5-spatial-replication.statistics.v1", "status": "unassessable" if failures else "assessable",
        "case_names": names, "feature_names": FEATURES,
        "parameters": {"seed": seed, "null_replicates": null_replicates,
                       "bootstrap_replicates": bootstrap_replicates, "block_size": block_size,
                       "shape_hw": SHAPE, "blocks_yx": [4, 5], "blocks_per_case": block_count,
                       "polarities": dict(zip(FEATURES, POLARITIES)), "ink_reference_range": [179, 255],
                       "blank_reference_range": [1, 64], "excluded_reference_ranges": [[0, 0], [65, 178]],
                       "common_support": "geometry valid AND finite in all three features; unchanged under shifts",
                       "auc": "weighted Mann-Whitney U, exact-score ties receive half credit; no orientation selection",
                       "case_weighting": "equal macro average", "percentile_method": "linear",
                       "rng": "default_rng(SeedSequence(seed).spawn(6)); null cases 0..2, bootstrap cases 3..5",
                       "gates": {"each_case_auc_min": .70, "macro_delta_min": .05,
                                 "bootstrap_lower_exclusive_min": 0., "spatial_rank_max": .05}},
        "observed": {"cases": case_reports, "macro_auc": dict(zip(FEATURES, macro)),
                     "macro_delta_vs_raw": dict(zip(FEATURES[1:], delta))},
        "null": {"interpretation": "Spatial-misalignment diagnostic, not valid FWER control or a significance proof: conditional ROI selection, model reference and nonstationarity remain",
                 "translations": "uniform unique nonzero toroidal offsets independently per case; no near-identity exclusion or redraw",
                 "offsets_yx": offsets, "offsets_layout": "case, replicate, yx",
                 "case_auc": null, "case_auc_layout": "replicate, case, feature",
                 "class_counts_ink_blank": null_counts, "macro_auc": null_macro,
                 "family_max_auc": null_max, "spatial_rank": ranks,
                 "rank_definition": "(1 + count(null maximum over two feature macro AUCs >= observed feature macro AUC)) / (null_replicates + 1)"},
        "bootstrap": {"interpretation": "Paired descriptive block interval for these three fields; not independent donor uncertainty",
                      "resampled_block_indices": draws, "draws_layout": "case, replicate, draw; blocks indexed row-major",
                      "case_auc": boot, "case_auc_layout": "replicate, case, feature",
                      "class_counts_ink_blank": boot_counts, "macro_auc": boot_macro,
                      "macro_delta_vs_raw": boot_delta, "delta_feature_names": FEATURES[1:],
                      "percentile_interval": intervals, "percentiles": [2.5, 97.5]},
        "gates": gates, "failures": failures})
