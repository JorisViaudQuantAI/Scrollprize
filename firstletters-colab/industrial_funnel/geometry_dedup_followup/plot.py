#!/usr/bin/env python3
"""Render the actual 48 correspondence masks and geometric measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
WINDS = ("w033", "w034", "w035", "w036")


def main():
    result = json.loads((HERE / "result.json").read_bytes())
    cmap = ListedColormap(["#e9edf2", "#3677ad", "#269c78", "#c7383a", "#9a62b8", "#d49a33"])
    labels = ["No accepted match", "Other accepted match", "Near → near", "Near → far", "Near → unknown", "Near → intermediate"]
    norm = BoundaryNorm(np.arange(-.5, 6.5), cmap.N)
    with np.load(HERE / "numeric_maps.npz", allow_pickle=False) as maps:
        for state in ("fitted", "snap", "flat"):
            fig, axes = plt.subplots(4, 4, figsize=(13.5, 10))
            fig.subplots_adjust(top=.89, bottom=.115, hspace=.27, wspace=.07)
            for ia, a in enumerate(WINDS):
                for ib, b in enumerate(WINDS):
                    row = next(r for r in result["pairs"] if (r["state"], r["source"], r["target"]) == (state, a, b))
                    ax = axes[ia, ib]
                    im = ax.imshow(maps[f"{state}_{a}_{b}_classification"], cmap=cmap, norm=norm, interpolation="nearest")
                    ax.set_title(f"{a} → {b}  |  accepted {row['accepted_fraction']:.1%}", fontsize=9)
                    ax.set_xticks([])
                    ax.set_yticks([])
            fig.suptitle(f"PHerc. 0211 / {state}: geometry correspondences on actual source grids\n"
                         "Reference-near ≤3 vox; far ≥6 vox. Geometry labels only — no ink or letter claim.", fontsize=13)
            fig.legend(handles=[Patch(facecolor=cmap(i), label=label) for i, label in enumerate(labels)],
                       loc="lower center", bbox_to_anchor=(.5, .018), ncol=3, frameon=False, fontsize=10)
            fig.savefig(HERE / f"atlas_{state}.png", dpi=145)
            plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.5), constrained_layout=True)
        states, fractions, tails = [], [], []
        for state in ("fitted", "snap", "flat"):
            rows = [r for r in result["pairs"] if r["state"] == state and not r["self_pair"]]
            total = sum(r["n_accepted"] for r in rows)
            states.append(state)
            fractions.append(100 * sum(r["accepted_ray_separation_gt3"] for r in rows) / total)
            tails.append(100 * sum(r["accepted_ray_separation_gt6"] for r in rows) / total)
        x = np.arange(3)
        bars = axes[0].bar(x, fractions, color=["#3677ad", "#269c78", "#d49a33"])
        axes[0].bar_label(bars, fmt="%.1f%%", padding=3)
        axes[0].set(xticks=x, xticklabels=states, ylim=(0, 100), ylabel="% of accepted directed correspondences",
                    title="Sampling-ray separation exceeds 3 vox\nMaximum over the symmetric depth span ±8 vox")
        self_rows = [r for r in result["pairs"] if r["self_pair"]]
        old = [100 * r["accepted_fraction"] for r in self_rows]
        axes[1].plot(range(12), old, "o-", color="#c7383a", label="Upstream sampling")
        axes[1].plot(range(12), [100]*12, "o-", color="#269c78", label="Preserve usable original vertices")
        axes[1].set(xticks=range(12), xticklabels=[("fit" if r["state"] == "fitted" else r["state"]) + "\n" + r["source"][1:] for r in self_rows],
                    ylim=(97, 100.45), ylabel="% accepted on self-comparison", title="Boundary support correction\nTwelve real mesh identity controls")
        axes[1].tick_params(axis="x", labelsize=8)
        axes[1].legend(loc="lower center", fontsize=8)
        fig.suptitle("Geometric closeness does not establish identical detector inputs", fontsize=13)
        fig.savefig(HERE / "geometry_summary.png", dpi=165)
        plt.close(fig)


if __name__ == "__main__":
    main()
