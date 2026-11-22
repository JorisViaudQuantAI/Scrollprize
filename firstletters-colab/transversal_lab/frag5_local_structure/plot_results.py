#!/usr/bin/env python3
"""Presentation of frozen numerical results; no CT, fit or new metric."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    r = json.loads(args.result.read_bytes())
    args.out.mkdir(parents=True, exist_ok=True)
    ids, passes = [20, 25, 65, 70], [0, 1, 4]
    primary = "fine_nominal_coarse_footprint"
    matches = {(a["id"], a["arm"], a["binomial_passes"]): a for a in r["matched_site_comparisons"]}
    quality = {(a["id"], a["array"], a["binomial_passes"]): a for a in r["quality"]}
    rows = []
    for a in r["matched_site_comparisons"]:
        s = a["correlation_surfaces"]["full"]
        rows.append({"id": a["id"], "arm": a["arm"], "binomial_passes": a["binomial_passes"],
                     **{k: v for k, v in s.items() if k != "argmax_shift_xyz"},
                     "argmax_xyz": json.dumps(s["argmax_shift_xyz"])})
    with (args.out / "matched_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (args.out / "site_pair_zero_shift.csv").open("w", newline="") as f:
        rows = r["all_site_pair_zero_shift_comparisons"]
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rows = []
    for a in r["quality"]:
        rows.append({k: a[k] for k in ("id", "array", "binomial_passes", "mean", "sd", "variance_retention", "gradient_energy_counts_squared_per_grid_voxel_squared")}
                    | a["lag_pearson_positive_xyz"]
                    | {"tensor_fractions": json.dumps(a["tensor_eigenvalue_fractions_descending"])})
    with (args.out / "structure_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = ["#195c85", "#b64536", "#2e7f59", "#8a679b"]
    for i, color in zip(ids, colors):
        vals = [matches[i, primary, p]["correlation_surfaces"]["full"]["zero_shift"] for p in passes]
        axs[0].plot(range(3), vals, "o-", color=color, label=f"Record {i}")
    axs[0].set(xticks=range(3), xticklabels=["None", "1 pass", "4 passes"], ylim=(-.08, 1.02),
               ylabel="Fine footprint vs native coarse Pearson", xlabel="Fixed binomial smoothing",
               title="Zero shift on identical 17³ support")
    axs[0].axhline(0, color="#888888", linewidth=.6); axs[0].legend(loc="center right")
    xpos = np.arange(4)
    for offset, name, label, color in [(-.18, primary, "Fine footprint", "#276c92"), (.18, "native_coarse", "Native coarse", "#e39c45")]:
        values = [100 * quality[i, name, 4]["variance_retention"] for i in ids]
        bars = axs[1].bar(xpos + offset, values, .35, color=color, label=label)
        axs[1].bar_label(bars, labels=[f"{v:.1f}%" for v in values], fontsize=9, padding=3)
    axs[1].set(xticks=xpos, xticklabels=[str(i) for i in ids], ylim=(0, 75), ylabel="Variance retained (%)",
               xlabel="Public geometry record", title="Four-pass variance relative to unfiltered")
    axs[1].legend(loc="upper left")
    fig.suptitle("Frag5 cached-control diagnostic — small, already observed neighborhoods", fontsize=13)
    fig.savefig(args.out / "local_structure_summary.svg")
    fig.savefig(args.out / "local_structure_summary.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.8), constrained_layout=True)
    for ai, arm in enumerate((primary, "fine_point")):
        for pi, p in enumerate(passes):
            values = np.array([[next(a["zero_shift_pearson"] for a in r["all_site_pair_zero_shift_comparisons"]
                                    if a["fine_id"] == i and a["coarse_id"] == j and a["arm"] == arm and a["binomial_passes"] == p)
                                for j in ids] for i in ids])
            ax = axs[ai, pi]
            im = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1)
            for y, x in np.ndindex(values.shape):
                ax.text(x, y, f"{values[y,x]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(values[y,x]) > .65 else "black")
            ax.set(xticks=range(4), xticklabels=ids, yticks=range(4), yticklabels=ids,
                   xlabel="Native coarse record", ylabel="Fine record", title=f"{'Footprint' if ai == 0 else 'Point'}; {p} passes")
    fig.colorbar(im, ax=axs, shrink=.7, label="Zero-shift Pearson")
    fig.suptitle("All site pairs: mismatches are stress controls, not a calibrated null", fontsize=12)
    fig.savefig(args.out / "site_pair_matrices.svg")
    fig.savefig(args.out / "site_pair_matrices.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
