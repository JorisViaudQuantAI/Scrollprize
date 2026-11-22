#!/usr/bin/env python3
"""Plot every retained Frag5 stage-1 condition from numerical results only.

This script reads the published result JSON. No network, CT extraction, model
inference, fitting, metric recalculation or transform selection is performed.
"""
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "source_profiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point_id", "depth_index", "tiff_uint16", "fine_trilinear_primary",
                         "fine_trilinear_reverse", "primary_rounded_abs_error", "reverse_rounded_abs_error"])
        for item in result["identity"]["profiles"]:
            for k in range(65):
                writer.writerow([item["id"], k, item["tiff_values"][k], item["fine_trilinear_values"][k],
                                 item["fine_trilinear_values"][64-k], item["primary_absolute_errors"][k],
                                 item["reverse_absolute_errors"][k]])
    with (args.out / "native_pair_all_shifts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point_id", "dx", "dy", "dz", "arm", "zero_shift_primary",
                         "intensity_n", "intensity_pearson", "intensity_spearman", "intensity_status",
                         "laplacian_n", "laplacian_pearson", "laplacian_spearman", "laplacian_status"])
        for item in result["anatomy"]:
            for row in item["all_shifts"]:
                a, b = row["intensity"], row["laplacian_interior"]
                writer.writerow([item["id"], *row["shift_xyz_coarse_voxels"], row["arm"],
                                 row["original_transform_primary"], a["n"], a["pearson"], a["spearman"], a["status"],
                                 b["n"], b["pearson"], b["spearman"], b["status"]])
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(65)
    for ax, item in zip(axes.flat, result["identity"]["profiles"]):
        ax.plot(x, item["tiff_values"], color="black", lw=2, label="Released TIFF sequence")
        ax.plot(x, item["fine_trilinear_values"], color="#087e8b", lw=1.3, label="Declared raw-fine profile")
        ax.plot(x, item["fine_trilinear_values"][::-1], color="#c36c00", lw=1.1, ls="--", label="Declared reverse alternative")
        ax.set_title(f"Fixed point {item['id']} | primary max error {item['primary_max_abs_error_uint16']} count(s)")
        ax.set_xlabel("TIFF depth index (declared k = 0…64)")
        ax.set_ylabel("Stored uint16 units")
        ax.grid(alpha=.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Frag5: four fixed source profiles\n" + result["identity"]["classification"] + " — no ink labels or fitted phase", fontsize=14)
    for suffix in ("svg", "png"):
        fig.savefig(args.out / f"source_identity.{suffix}", dpi=160)
    plt.close(fig)
    if not result["anatomy"]:
        return
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), sharex=True, sharey=True, constrained_layout=True)
    for row, item in enumerate(result["anatomy"]):
        records = item["all_shifts"]
        shifts = [tuple(r["shift_xyz_coarse_voxels"]) for r in records if r["arm"] == "fine_point"]
        zero = shifts.index((0, 0, 0))
        for col, metric in enumerate(("intensity", "laplacian_interior")):
            ax = axes[row, col]
            for arm, color in (("fine_point", "#6f4e7c"), ("fine_nominal_coarse_footprint", "#087e8b")):
                rows = [r for r in records if r["arm"] == arm]
                for kind, style in (("pearson", "-"), ("spearman", "--")):
                    values = [r[metric][kind] if r[metric][kind] is not None else np.nan for r in rows]
                    ax.plot(range(len(shifts)), values, style, color=color, lw=1.4,
                            label=("Fine point" if arm == "fine_point" else "Fine 125-midpoint footprint") + " / " + kind)
            ax.axvline(zero, color="#c0362c", lw=1, alpha=.8)
            ax.axhline(0, color="gray", lw=.7)
            ax.set_ylim(-1.04, 1.04)
            ax.set_title(f"Fixed point {item['id']} — " + ("stored-value structure" if col == 0 else "interior Laplacian"))
            ax.set_ylabel("Descriptive correlation")
            ax.set_xticks(range(len(shifts)), [str(s).replace(" ", "") for s in shifts], rotation=90, fontsize=7)
            ax.grid(alpha=.16)
    for ax in axes[-1]:
        ax.set_xlabel("All declared native-coarse XYZ shifts (voxels); red line is zero")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Frag5: published affine and all 27 shift sensitivities\nNo fitted alignment; four neighborhoods within one physical fragment", fontsize=14)
    for suffix in ("svg", "png"):
        fig.savefig(args.out / f"native_pair_all_shifts.{suffix}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
