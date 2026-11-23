#!/usr/bin/env python3
"""Fixed-scale review plots from completed native-resolution artifacts only.

No model import, new inference, source acquisition, fitting, threshold search,
image contrast adjustment, or reading of incomplete field results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

ARMS = ("A_native_fine", "B_degraded_fine", "C_coarse")
NAMES = ("A: Native fine", "B: Degraded fine", "C: Coarse")
COLORS = ("#2166AC", "#D98C28", "#27856A")
QUADRANTS = ("q00", "q01", "q10", "q11")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def outside_git(path):
    path = Path(path).resolve()
    if any(os.path.lexists(p/".git") for p in (path, *path.parents)):
        raise ValueError("Pixel-level review images must remain outside Git")
    return path


def validate_result(path):
    result = json.loads(Path(path).read_text())
    if result.get("status") != "complete":
        raise ValueError("Wait for a complete experiment; partial maps are not reviewed by this script")
    if {f["scroll"] for f in result["fields"]} != {"PHerc0841", "PHerc0814"} or len(result["fields"]) != 2:
        raise ValueError("Expected exactly the two frozen public fields")
    if result["forwards_completed"] not in (388, 390):
        raise ValueError("Unexpected completed forward count")
    return result


def load_field(root, field):
    folder = root/field["scroll"]
    score_path = folder/"scores.npz"
    if sha(score_path) != field["scores_sha256"] or sha(folder/"labels.npy") != field["labels_sha256"]:
        raise ValueError("Field artifact differs from completed receipt")
    with np.load(score_path, allow_pickle=False) as source:
        labels, support = source["labels"].copy(), source["support"].copy()
        maps = {arm: source[arm].copy() for arm in ARMS}
    if labels.shape != (256, 256) or support.shape != labels.shape or support.dtype != bool:
        raise ValueError("Invalid common-grid categorical/support shape")
    if not set(np.unique(labels)).issubset({0, 1, 2}):
        raise ValueError("Unexpected label categories")
    if not np.array_equal(labels, np.load(folder/"labels.npy", allow_pickle=False)):
        raise ValueError("Retained categorical labels differ")
    for values in maps.values():
        if values.shape != labels.shape or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError("Incomplete or invalid common-grid probabilities")
    return labels, support, maps


def review_plot(field, labels, support, maps, output):
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.8))
    fig.subplots_adjust(left=.045, right=.92, top=.76, bottom=.19, wspace=.12)
    mm_y, mm_x = field["nominal_core_mm"]
    extent = (0, mm_x, mm_y, 0)
    categorical = ListedColormap(("#B9B9B9", "#FFFFFF", "#101010"))
    categorical.set_bad("#D996A5")
    axes[0].imshow(np.ma.array(labels, mask=~support), cmap=categorical,
                   norm=BoundaryNorm((-.5, .5, 1.5, 2.5), categorical.N),
                   interpolation="nearest", extent=extent)
    axes[0].set_title("Released categorical labels", fontsize=10)
    cmap = plt.get_cmap("gray_r").copy()
    cmap.set_bad("#D996A5")
    for axis, arm, name in zip(axes[1:], ARMS, NAMES):
        image = axis.imshow(np.ma.array(maps[arm], mask=~support), cmap=cmap,
                            vmin=0, vmax=1, interpolation="nearest", extent=extent)
        axis.set_title(name, fontsize=11)
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("mm")
        axis.set_xticks((0, 1, 2))
        axis.set_yticks((0, 1, 2))
    axes[0].set_ylabel("mm")
    cax = fig.add_axes((.94, .23, .014, .49))
    bar = fig.colorbar(image, cax=cax, ticks=(0, .25, .5, .75, 1))
    bar.set_label("Model score (fixed 0–1)")
    fig.suptitle(f"{field['scroll']} — complete common-grid comparison", x=.045, y=.97,
                 ha="left", fontsize=16, weight="bold")
    fig.text(.045, .88, "Same physical core, labels and scoring support · 256 × 256 cells at 9.6 µm",
             fontsize=10, color="#4A5866")
    fig.legend([Patch(facecolor=c, edgecolor="#777") for c in ("#101010", "#FFFFFF", "#B9B9B9", "#D996A5")],
               ("Ink label", "Background label", "Unknown label", "Unsupported"),
               loc="lower center", bbox_to_anchor=(.5, .065), ncol=4, frameon=False, fontsize=9)
    fig.text(.045, .025,
             "Model-assisted development labels; no new reading is asserted. Fine A/B scores are area-averaged 4×4; no image-specific stretch.",
             fontsize=8.5, color="#4A5866")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def macro_values(field):
    values = []
    for arm in ARMS:
        recalls = [field["metrics"][q][arm]["recall_at_background_5pct"] for q in QUADRANTS]
        values.append(float(np.mean(recalls)) if all(x is not None for x in recalls) else np.nan)
    declared = field["gate"].get("macro_recall")
    if declared is not None:
        for arm, measured in zip(ARMS, values):
            if not np.isclose(measured, declared[arm], atol=1e-12, rtol=0):
                raise ValueError("Macro recall differs from completed field decision")
    return values


def aggregate_plot(result, output):
    fields = result["fields"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 6.4),
                                      gridspec_kw={"width_ratios": (1.1, 1)})
    fig.subplots_adjust(left=.075, right=.97, bottom=.26, top=.79, wspace=.35)
    means = np.array([macro_values(f) for f in fields])
    x = np.arange(len(fields))
    for i, (name, color) in enumerate(zip(NAMES, COLORS)):
        bars = left.bar(x+(i-1)*.24, means[:, i], width=.22, color=color, label=name)
        for bar, value in zip(bars, means[:, i]):
            if np.isfinite(value):
                left.annotate(f"{value:.3f}", (bar.get_x()+bar.get_width()/2, value),
                              xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    left.set_ylim(0, 1.08)
    left.set_yticks((0, .25, .5, .75, 1))
    left.set_xticks(x, [f["scroll"] for f in fields])
    left.set_ylabel("Equally weighted mean of four quadrant recalls")
    left.set_title("Primary: macro recall at 5% background", loc="left", fontsize=11, weight="bold")
    left.set_axisbelow(True)
    left.grid(axis="y", color="#DDE3E9", linewidth=.6)
    for side in ("top", "right"):
        left.spines[side].set_visible(False)
    rows = [[field["metrics"][q][arm]["ap"] for arm in ARMS] for field in fields for q in QUADRANTS]
    ap = np.array([[np.nan if v is None else v for v in row] for row in rows])
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#D8D8D8")
    right.imshow(ap, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for row in range(len(ap)):
        for col in range(3):
            value = ap[row, col]
            right.text(col, row, f"{value:.4f}" if np.isfinite(value) else "N/A",
                       ha="center", va="center", fontsize=9,
                       color="white" if np.isfinite(value) and value > .6 else "#263548")
    right.set_yticks(range(8), [f"{field['scroll']} {q}" for field in fields for q in QUADRANTS], fontsize=9)
    right.set_xticks(range(3), ("A: Fine", "B: Degraded", "C: Coarse"), fontsize=9)
    right.tick_params(length=0, pad=7)
    right.set_title("Within-quadrant average precision", loc="left", fontsize=11, weight="bold")
    for spine in right.spines.values():
        spine.set_visible(False)
    fig.suptitle("Native CT detail — two published development fields", x=.075, y=.97,
                 ha="left", fontsize=16, weight="bold")
    pilot = result["pilot_decision"]
    fig.text(.075, .90,
             f"{result['forwards_completed']} CPU forwards · fine {pilot['fine_precision']} / coarse fp32 · "
             f"{result['wall_seconds']:.1f} s wall time", fontsize=10, color="#4A5866")
    fig.legend(*left.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(.5, .15), ncol=3, frameon=False, fontsize=10)
    outcomes = " | ".join(f"{f['scroll']}: {'PASS' if f['gate']['pass'] else 'NO ADVANCEMENT'}" for f in fields)
    fig.text(.075, .115, outcomes, fontsize=10, weight="bold", color="#263548")
    fig.text(.075, .073,
             "Each quadrant has its own descriptive ROC operating point; these are not transferred thresholds or significance tests.",
             fontsize=8.7, color="#4A5866")
    fig.text(.075, .035,
             "Fixed metric scales; model-assisted labels on reused donors. A numeric gate does not establish glyph legibility or a prize.",
             fontsize=8.7, color="#4A5866")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white",
                metadata={"Description": "Aggregate metrics only; no CT, label pixels or prediction images."})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path)
    args = parser.parse_args()
    result = validate_result(args.result)
    out = outside_git(args.review_dir)
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    records = []
    for field in result["fields"]:
        labels, support, maps = load_field(args.result.parent, field)
        output = out/(field["scroll"]+"_common_grid_review.png")
        review_plot(field, labels, support, maps, output)
        records.append({"scroll": field["scroll"], "review_file": output.name,
                        "review_sha256": sha(output), "scores_sha256": field["scores_sha256"]})
    aggregate = args.aggregate_output or out/"aggregate_metrics.png"
    if aggregate.exists():
        raise FileExistsError("Use a fresh aggregate output path")
    aggregate_plot(result, aggregate)
    receipt = {"schema": "scrollprize.native-resolution-visualization.v1",
               "result_sha256": sha(args.result), "plotter_sha256": sha(__file__),
               "aggregate_file": aggregate.name, "aggregate_sha256": sha(aggregate),
               "review_images": records, "model_forwards": 0, "source_acquisitions": 0,
               "fixed_probability_range": [0, 1], "individual_contrast_stretch": False,
               "letter_reading_claim": False}
    (out/"visualization_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"review_directory": str(out), "aggregate_file": str(aggregate),
                      "aggregate_sha256": receipt["aggregate_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
