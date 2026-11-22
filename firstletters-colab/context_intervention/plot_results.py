#!/usr/bin/env python3
"""Plot aggregate-only context-intervention results; no source/model images.

The independent-review receipt must authenticate the exact input result bytes.
Axes and color ranges are fixed rather than adapted to the observed metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np

HERE = Path(__file__).resolve().parent
CONDITIONS = ("original", "zero_halo", "reflect_halo")
LABELS = ("Original", "Zero halo", "Reflected halo")
COLORS = ("#355C83", "#C77A35", "#168476")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=HERE / "evidence/result.json")
    parser.add_argument("--review", type=Path, default=HERE / "evidence/independent_review.json")
    parser.add_argument("--output", type=Path, default=HERE / "evidence/context_intervention_metrics.png")
    args = parser.parse_args(argv)
    result, review = json.loads(args.result.read_text()), json.loads(args.review.read_text())
    if result["status"] != "complete" or review["status"] != "pass" or review["result_sha256"] != sha(args.result):
        raise ValueError("Completed result does not match the passing independent review")
    rows = [result["pooled_descriptive_metrics"]]
    contexts = {row["context_id"]: row["conditions"] for row in result["contexts"]}
    rows.extend({c: contexts[k][c]["metrics"] for c in CONDITIONS} for k in ("814-c03", "814-c09"))
    row_labels = ("Pooled: five cores", "c03: mixed labels", "c09: mixed labels")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    fig.subplots_adjust(left=.14, right=.98, bottom=.20, top=.85, hspace=.65, wspace=.42)
    for ax, metric, title in zip(axes[0], ("average_precision", "auroc"),
                                 ("A  Average precision", "B  ROC AUC")):
        values = np.array([[row[c][metric] for c in CONDITIONS] for row in rows])
        ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto", extent=(-.5, 2.5, 2.5, -.5))
        for y in range(3):
            for x in range(3):
                ax.text(x, y, f"{values[y, x]:.4f}", ha="center", va="center",
                        color="white" if values[y, x] > .6 else "#172331", fontsize=10)
            delta = values[y, 2] - values[y, 0]
            ax.add_patch(Rectangle((2.5, y - .5), 1, 1, facecolor="#F1F3F5", edgecolor="white", lw=2))
            ax.text(3, y, f"{delta:+.4f}", ha="center", va="center", weight="bold",
                    color="#176C47" if delta > 0 else "#A82E35", fontsize=10)
        ax.set_xlim(-.5, 3.5)
        ax.set_ylim(2.5, -.5)
        ax.set_yticks(range(3), row_labels)
        ax.set_xticks(range(4), ("Original", "Zero halo", "Reflected", "Reflection Δ"))
        ax.tick_params(length=0, pad=7)
        ax.set_title(title, loc="left", pad=11)
        for spine in ax.spines.values(): spine.set_visible(False)
    for ax, metric, title in zip(axes[1], ("recall_at_0_5", "background_exceedance_at_0_5"),
                                 ("C  Ink recall at p ≥ 0.5", "D  Background exceedance at p ≥ 0.5")):
        for index, condition in enumerate(CONDITIONS):
            values = np.array([row[condition][metric] for row in rows]) * 100
            positions = np.arange(3) + (index - 1) * .24
            ax.bar(positions, values, width=.22, color=COLORS[index])
            for x, value in zip(positions, values):
                ax.annotate(f"{value:.1f}", (x, value), xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.5, color="#263548")
        ax.set_ylim(0, 108)
        ax.set_yticks((0, 25, 50, 75, 100), ("0%", "25%", "50%", "75%", "100%"))
        ax.set_xticks(range(3), ("Pooled", "c03", "c09"))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DDE3E9", linewidth=.6)
        ax.set_title(title, loc="left", pad=11)
    fig.suptitle("Same central CT, different external context", x=.14, y=.975,
                 ha="left", fontsize=17, weight="bold", color="#172331")
    fig.text(.14, .925,
             f"Hecate · reused public PHerc. 0814 controls · {result['forwards_completed']} CPU forwards · "
             f"{result['wall_seconds']:.2f} s wall time", fontsize=11, color="#4E5F70")
    fig.legend([Patch(facecolor=c) for c in COLORS], LABELS, loc="lower center",
               bbox_to_anchor=(.56, .116), ncol=3, frameon=False)
    fig.text(.14, .092,
             "Reflection improves pooled AP/AUC, but lowers both metrics within each mixed-label core.",
             fontsize=10, weight="bold", color="#172331")
    fig.text(.14, .061,
             "75% of the input replaced; central 0.3072 mm CT and labels unchanged. AP/AUC colors fixed at 0–1; rates share a fixed scale.",
             fontsize=8.8, color="#4E5F70")
    fig.text(.14, .033,
             "One previously evaluated physical donor; correlated pixels, model-assisted labels. No letter-reading or generalisation claim.",
             fontsize=8.8, color="#4E5F70")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white", metadata={"Description": "Aggregate-only metrics; no CT or ink-score image pixels."})
    plt.close(fig)
    print(json.dumps({"plot": args.output.name, "result_sha256": sha(args.result),
                      "review_sha256": sha(args.review), "plot_sha256": sha(args.output)}, indent=2))


if __name__ == "__main__":
    main()
