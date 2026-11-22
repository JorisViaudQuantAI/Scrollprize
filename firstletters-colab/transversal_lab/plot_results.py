#!/usr/bin/env python3
"""Plot all registered optical metrics relevant to the two fixed contrasts.

Reads published numeric evidence only; no source images, fitting or inference.
The complete 22-row numerical table remains the primary result.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    evidence = Path(__file__).resolve().parent / "evidence"
    record = json.loads((evidence / "profilometer_results.json").read_text())
    rows = record["results"]
    if len(rows) != 22 or record["status"] != "completed_fixed_protocol":
        raise ValueError("Expected the complete fixed experiment.")
    lookup = {(r["case_id"], r["model"], r["test_degradation_factor"]): r for r in rows}
    factors = [1, 2, 4]
    series = [
        ("Full head trained at factor 1", "full_train_f1", "#215c9b", "o", "-"),
        ("Full head trained at matched factor", "matched", "#d47715", "s", "-"),
        ("Height-only head trained at factor 1", "height_train_f1", "#777777", "^", "--"),
        ("Roughness-only head trained at factor 1", "roughness_train_f1", "#26816c", "D", "--"),
    ]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "transversal20261122"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True, sharey=True)
    for row_idx, (case, short) in enumerate([
        ("500P2_230925_H", "H"), ("500P2_230925_N2", "N2")
    ]):
        prevalence = lookup[(case, "full_train_f1", 1)]["ink_prevalence"]
        for col_idx, (metric, title) in enumerate([
            ("average_precision", "Average precision across the scored region"),
            ("precision_at_top_1_percent", "Precision among the highest-ranked 1%"),
        ]):
            ax = axes[row_idx, col_idx]
            for label, model, color, marker, style in series:
                values = [lookup[(case, f"full_train_f{f}" if model == "matched" else model, f)][metric]
                          for f in factors]
                ax.plot(range(3), values, label=label, color=color, marker=marker,
                        linestyle=style, linewidth=1.8, markersize=6,
                        markerfacecolor="white" if model == "matched" else color)
            ax.axhline(prevalence, color="#b5b5b5", linewidth=1, linestyle=":")
            ax.set_title(f"Region {short} | {title}", fontsize=10.5, pad=10)
            ax.set_ylim(0, 1)
            ax.set_xticks(range(3), ["×1", "×2", "×4"])
            ax.grid(axis="y", alpha=0.18)
            ax.text(0.98, 0.94, f"Scored ink prevalence: {100*prevalence:.2f}%",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555555")
            if row_idx == 1:
                ax.set_xlabel("Additional degradation of the working height map")
    fig.suptitle("Optical morphology pilot: region and review budget change the conclusion",
                 fontsize=15, y=0.98)
    fig.text(0.5, 0.933,
             "Two held-out regions of one papyrus. Working factor 1 already averages 4×4 native pixels. This is not CT.",
             ha="center", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=2, frameon=False, fontsize=9.5)
    fig.text(0.5, 0.012,
             "Source: frozen 22-row pilot; photograph-derived labels, normalized optical heights. Dotted lines mark prevalence.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(top=0.875, bottom=0.18, left=0.065, right=0.985, hspace=0.28, wspace=0.11)
    fig.savefig(evidence / "profilometer_review_budget.png", dpi=160, facecolor="white")
    fig.savefig(evidence / "profilometer_review_budget.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)
    print("Rendered the complete fixed optical comparison; no fitting or inference.")


if __name__ == "__main__":
    main()
