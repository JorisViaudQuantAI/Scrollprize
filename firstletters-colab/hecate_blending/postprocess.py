#!/usr/bin/env python3
"""Fixed-scale local inspection panels and an aggregate-only public figure."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--public-figure", required=True, type=Path)
    args = parser.parse_args()
    result = json.loads((args.run_dir / "result.json").read_text())
    if result["status"] != "complete":
        raise ValueError("Only a completed result may be illustrated")
    for field in result["fields"]:
        folder = args.run_dir / field["scroll"]
        labels = np.load(folder / "labels.npy", allow_pickle=False)
        label_view = np.choose(labels, [.65, 1., 0.])
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.7), constrained_layout=True)
        axes[0].imshow(label_view, vmin=0, vmax=1, cmap="gray", interpolation="nearest")
        axes[0].set_title("Published model-assisted labels\nblack=ink; grey=unknown")
        for ax, key, title in zip(axes[1:], ("central", "hann", "uniform"),
                                  ("Central crop", "Released Hann", "Uniform overlap (diagnostic)")):
            values = np.load(folder / (key + "_fine.npy"), allow_pickle=False)
            ax.imshow(values, vmin=0, vmax=1, cmap="gray_r", interpolation="nearest")
            ax.set_title(title + "\nprobability 0–1, fixed scale")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(field["scroll"] + " — complete 2.4576 mm square public development core; no new reading", fontsize=13)
        fig.savefig(folder / "complete_fixed_scale.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    titles = ("Labelled seam-pair recovery", "Whole-field ink recall", "Whole-field average precision",
              "Background seam-pair detections ↓")
    palette = ("#64748b", "#126d82", "#bc805a")
    for row, field in enumerate(result["fields"]):
        for col, title in enumerate(titles):
            ax = axes[row, col]
            values = []
            for arm in ("central", "hann", "uniform"):
                metric = field["metrics"]["float"][arm]
                values.append(metric["pairs"]["seam_ink"]["both_positive_fraction"] if col == 0 else
                              metric["recall_at_background_5pct"] if col == 1 else metric["ap"] if col == 2 else
                              metric["pairs"]["seam_background"]["both_positive_fraction"])
            bars = ax.bar(("Central", "Hann", "Uniform"), values, color=palette, width=.65)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+(.001 if col == 3 else .02),
                        f"{value:.4f}" if col == 3 else f"{value:.3f}", ha="center", fontsize=10)
            ax.set_ylim(0, .065 if col == 3 else 1.08)
            ax.set_yticks(np.linspace(0, .06, 4) if col == 3 else np.linspace(0, 1, 6))
            ax.set_title(title)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=.2)
            if col == 0:
                ax.set_ylabel(field["scroll"] + "\nFraction / score")
            if col == 3:
                gain = 100 * (values[1] - values[0])
                ax.text(.5, .96, f"Hann − central: +{gain:.3f} pp; limit +1 pp",
                        transform=ax.transAxes, ha="center", va="top", fontsize=9,
                        color="#991b1b" if gain > 1 else "#334155")
    fig.suptitle("Same Hecate predictions, three reconstructions — full gate FAILED\nFloat scores shown; 128 CPU forwards on two previously observed public development fields", fontsize=15)
    fig.supxlabel("One label-informed ≤5% background operating point per field/arm; these are not transferable false-alarm bounds.\n"
                  "The 0814 uint8 operating point detects no ink in any arm (saturation); not shown in bars.\n"
                  "Correlated pairs; model-assisted labels; no independent reading or unread validation.", fontsize=10)
    args.public_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.public_figure, dpi=150)
    plt.close(fig)
    print("Created full fixed-scale local panels and aggregate-only figure")


if __name__ == "__main__":
    main()
