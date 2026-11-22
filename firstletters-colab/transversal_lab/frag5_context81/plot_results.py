#!/usr/bin/env python3
"""Numbers-only presentation; no CT access or scientific parameter selection."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def table(path, rows):
    rows = [{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows]
    if rows:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    r = json.loads(args.result.read_bytes())
    if r["outcome"] != "completed_fixed_context_comparison":
        raise SystemExit("No completed anatomical measurements to present")
    args.out.mkdir(parents=True, exist_ok=True)
    for key in ("nested_contexts_all_shifts", "fixed65_smoothing_zero_shift", "outer_shell_zero_shift", "fixed65_structure"):
        table(args.out / (key + ".csv"), r[key])
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.1), constrained_layout=True)
    sides = [17, 33, 65, 81]
    x = np.array(sides) * 7.91 / 1000
    for col, i in enumerate((20, 25)):
        for row, metric in enumerate(("pearson", "spearman")):
            ax = axs[row, col]
            for arm, color, label in (("fine_point", "#a06d33", "Point"), ("fine_nominal_coarse_footprint", "#246b95", "Footprint")):
                selected = [a for a in r["nested_contexts_all_shifts"] if a["id"] == i and a["arm"] == arm]
                zero = [next(a[metric] for a in selected if a["side"] == s and a["zero_shift_primary"]) for s in sides]
                low = [min(a[metric] for a in selected if a["side"] == s and a[metric] is not None) for s in sides]
                high = [max(a[metric] for a in selected if a["side"] == s and a[metric] is not None) for s in sides]
                ax.fill_between(x, low, high, color=color, alpha=.13)
                ax.plot(x, zero, "o-", color=color, label=label)
            ax.set(xticks=x, xticklabels=[f"{v:.3f}" for v in x], ylim=(-.1, 1.01),
                   xlabel="Nominal cube support (mm)", ylabel=metric.capitalize(),
                   title=f"Record {i} — unchanged center and affine")
            ax.axhline(0, color="#888888", linewidth=.7)
            ax.legend(loc="best")
    fig.suptitle("Larger anatomical context; line = zero shift, band = all 27 fixed shifts\nBands are sensitivity ranges, not confidence intervals", fontsize=12)
    fig.savefig(args.out / "context_dependence.svg")
    fig.savefig(args.out / "context_dependence.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, i in zip(axs, (20, 25)):
        rows = [a for a in r["fixed65_smoothing_zero_shift"] if a["id"] == i and a["arm"] == "fine_nominal_coarse_footprint"]
        for metric, color in (("pearson", "#246b95"), ("spearman", "#b64536")):
            y = [next(a[metric] for a in rows if a["binomial_passes"] == p) for p in (0, 1, 4)]
            ax.plot(range(3), y, "o-", label=metric.capitalize(), color=color)
            for x, value in enumerate(y):
                ax.annotate(f"{value:.3f}", (x, value), xytext=(0, 7 if metric == "pearson" else -15),
                            textcoords="offset points", ha="center", fontsize=9)
        ax.set(xticks=range(3), xticklabels=["None", "1 pass", "4 passes"], ylim=(-.08, 1.06),
               xlabel="Fixed binomial smoothing", ylabel="Cross-scan correlation",
               title=f"Record {i}; unchanged center, zero shift")
        ax.legend(loc="lower right" if i == 20 else "upper left")
    fig.suptitle("Same 65³ support (0.514 mm): fine footprint versus native coarse\nAggregate Pearson and rank agreement can behave differently", fontsize=12)
    fig.savefig(args.out / "smoothing_and_rank.svg")
    fig.savefig(args.out / "smoothing_and_rank.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
