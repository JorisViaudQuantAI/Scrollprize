#!/usr/bin/env python3
"""Render committed numerical probe results; no image access or model fitting.

All directions and all fixed depth offsets are retained. The label-shift
envelope is descriptive and is not a confidence interval or a null distribution.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


EVIDENCE = Path(__file__).resolve().parent / "evidence"
ARMS = ("baseline", "depth", "features")
ARM_NAMES = ("Released output", "Depth weights", "Feature readout")


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def write_rows(name, rows):
    with (EVIDENCE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_row(metrics):
    return {
        "pixels": metrics["pixels"], "ink": metrics["ink"],
        "background": metrics["background"], "prevalence": metrics["prevalence"],
        "auroc": metrics["auroc"], "average_precision": metrics["average_precision"],
        "precision_top_1_percent": metrics["top_1_percent"]["precision"],
        "recall_top_1_percent": metrics["top_1_percent"]["recall"],
        "precision_top_5_percent": metrics["top_5_percent"]["precision"],
        "recall_top_5_percent": metrics["top_5_percent"]["recall"],
        "threshold": metrics["threshold"],
        "fpr_at_calibration_threshold": metrics["fpr_at_calibration_threshold"],
        "recall_at_calibration_threshold": metrics["recall_at_calibration_threshold"],
    }


def hecate():
    amended = read("hecate_role_result.json")
    rows = []
    for order in ("source", "reverse"):
        for arm, metrics in amended["training_diagnostics"][order]["parent"].items():
            rows.append({"training_order": order, "calibration_order": order,
                         "input_order": order, "split": "train_in_sample",
                         "scroll": "PHerc0841", "arm": arm, **metric_row(metrics)})
    for comparison in ("primary", "all_reversed_sensitivity"):
        item = amended[comparison]
        for split, scroll in (("calibration", "PHerc0009B"), ("test", "PHerc0814")):
            for arm, metrics in item[split]["parent"].items():
                rows.append({"training_order": item["training_order"],
                             "calibration_order": item["calibration_order"],
                             "input_order": item[f"{split}_order"], "split": split,
                             "scroll": scroll, "arm": arm, **metric_row(metrics)})
    write_rows("hecate_parent_metrics.csv", rows)
    rows = []
    grid = amended["all_eight_combinations"]
    for item in grid:
        for split in ("calibration", "test"):
            for arm, metrics in item[split]["parent"].items():
                rows.append({"training_order": item["training_order"],
                             "calibration_order": item["calibration_order"],
                             "test_order": item["test_order"], "split": split, "arm": arm,
                             **metric_row(metrics)})
    write_rows("hecate_orientation_metrics.csv", rows)

    # AP does not depend on the operating threshold. Show its four unique
    # train/test combinations once; show every calibration/test threshold case.
    rank_cases = [item for item in grid if item["calibration_order"] == "source"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.8), gridspec_kw={"width_ratios": [1, 1.35]})
    panels = ((axes[0], rank_cases, "average_precision", "Average precision", False),
              (axes[1], grid, "fpr_at_calibration_threshold", "Test background selected", True))
    for axis, cases, key, title, percentages in panels:
        values = np.array([[item["test"]["parent"].get(arm, {}).get(key, np.nan)
                            for arm in ARMS] for item in cases], dtype=float)
        shown = axis.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1,
                            cmap="viridis", aspect="auto")
        axis.set_xticks(range(3), ARM_NAMES, rotation=15, ha="right")
        if percentages:
            labels = [" / ".join(item[k][0].upper() for k in
                      ("training_order", "calibration_order", "test_order")) for item in cases]
            axis.set_ylabel("Training head / calibration input / test input")
        else:
            labels = [" / ".join(item[k][0].upper() for k in
                      ("training_order", "test_order")) for item in cases]
            axis.set_ylabel("Training head / test input")
        axis.set_yticks(range(len(cases)), labels)
        axis.set_title(title, loc="left", fontweight="bold")
        for row in range(len(cases)):
            for col in range(3):
                value = values[row, col]
                label = "N/A" if not np.isfinite(value) else (f"{100*value:.2f}%" if percentages else f"{value:.3f}")
                axis.text(col, row, label, ha="center", va="center",
                          color="black" if not np.isfinite(value) or value > .55 else "white", fontsize=10)
        primary_index = next(i for i, item in enumerate(cases)
                             if all(item[key] == "source" for key in
                                    ("training_order", "calibration_order", "test_order")))
        axis.add_patch(Rectangle((-.5, primary_index-.5), 3, 1,
                                 fill=False, edgecolor="#ef6a32", linewidth=2.5))
        fig.colorbar(shown, ax=axis, fraction=.04, pad=.035)
    first = grid[0]["test"]["parent"]["baseline"]
    fig.suptitle("Frozen Hecate readouts: one held-out PHerc. 0814 surface", fontsize=16, fontweight="bold")
    fig.text(.04, .075,
             f"S = source order; R = reversed. Orange border marks the role-amended primary comparison. "
             f"Scored pixels: {first['pixels']:,}; labeled ink: {first['prevalence']:.1%}.", fontsize=10)
    fig.text(.04, .035,
             "Each threshold selects at most 1% of its calibration background. Test rates are conditional on released labels; "
             "they do not estimate an unread-scroll false-positive rate.", fontsize=9)
    fig.subplots_adjust(left=.11, right=.97, top=.86, bottom=.21, wspace=.40)
    for suffix in ("png", "svg"):
        fig.savefig(EVIDENCE / f"hecate_readout_comparison.{suffix}", dpi=170, facecolor="white")
    plt.close(fig)


def parker():
    record = read("parker_profile_results.json")
    depths = np.arange(-16, 17)
    names = [f"depth_{int(depth):+d}" for depth in depths]
    native = record["native"]
    rows = []
    for name, metrics in native.items():
        rows.append({"descriptor": name, "n_positive": metrics["n_positive"],
                     "n_negative": metrics["n_negative"], "signed_auc": metrics["signed_auc"],
                     "signed_mean_difference": metrics.get("signed_mean_difference"),
                     "pooled_sd_contrast": metrics.get("pooled_sd_contrast"),
                     "rows_with_both_classes": metrics["rows_with_both_classes"],
                     "row_macro_signed_auc": metrics["row_macro_signed_auc"]})
    write_rows("parker_native_metrics.csv", rows)
    rows = []
    for shift in record["fixed_label_shifts"]:
        for name, comparison in shift["comparisons"].items():
            a = comparison["native_same_intersection"]
            b = comparison["shifted_labels_same_intersection"]
            rows.append({"axis": shift["axis"], "offset_surface_pixels": shift["offset_surface_pixels"],
                         "common_points": shift["n_common"], "descriptor": name,
                         "native_positive": a["n_positive"], "shifted_positive": b["n_positive"],
                         "native_signed_auc": a["signed_auc"], "shifted_signed_auc": b["signed_auc"],
                         "native_minus_shifted_auc": None if a["signed_auc"] is None or b["signed_auc"] is None
                         else a["signed_auc"]-b["signed_auc"]})
    write_rows("parker_shift_metrics.csv", rows)

    offset_um = depths * record["approximately_micrometres_per_voxel"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5))
    axes[0].plot(offset_um, [native[name]["signed_auc"] for name in names],
                 color="#165b84", lw=2.2, label="All eligible points")
    axes[0].plot(offset_um, [native[name]["row_macro_signed_auc"] for name in names],
                 color="#d26a24", lw=1.7, label="Mean of mixed-class rows")
    axes[0].axhline(native["geometry_x_nuisance"]["signed_auc"], color="#6d4f88",
                    linestyle=":", label="PPM height comparator")
    axes[0].axhline(.5, color=".6", lw=1)
    axes[0].set(ylim=(0, 1), ylabel="Signed AUROC", title="All 33 declared depths")
    axes[0].legend(loc="best", frameon=False, fontsize=9)
    differences = []
    for shift in record["fixed_label_shifts"]:
        differences.append([shift["comparisons"][name]["native_same_intersection"]["signed_auc"]
                            - shift["comparisons"][name]["shifted_labels_same_intersection"]["signed_auc"]
                            for name in names])
    differences = np.asarray(differences)
    axes[1].fill_between(offset_um, differences.min(0), differences.max(0), alpha=.17,
                          color="#165b84", label="Range across 12 fixed shifts")
    for values in differences:
        axes[1].plot(offset_um, values, color="#165b84", alpha=.25, lw=.8)
    axes[1].axhline(0, color=".4", lw=1)
    axes[1].set(ylabel="Native minus shifted-label AUROC",
                title="Paired comparisons on identical support")
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.set_xlabel("Offset along supplied −X axis (approximately µm)")
        axis.axvline(0, color=".75", lw=.8, linestyle="--")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Photograph-labeled Parker fragment: depth and registration sensitivity", fontsize=15, fontweight="bold")
    fig.text(.055, .045,
             "One published letter; supplied registration and fixed extraction axis. The shift envelope is descriptive, "
             "not a confidence interval or a false-positive bound.", fontsize=9)
    fig.subplots_adjust(left=.075, right=.98, top=.85, bottom=.19, wspace=.30)
    for suffix in ("png", "svg"):
        fig.savefig(EVIDENCE / f"parker_depth_comparison.{suffix}", dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    hecate()
    parker()
