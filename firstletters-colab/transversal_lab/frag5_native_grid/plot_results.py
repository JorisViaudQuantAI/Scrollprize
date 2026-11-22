#!/usr/bin/env python3
"""Present all retained raw-grid outcomes; no acquisition, fitting or metrics."""
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
    result = json.loads(args.result.read_bytes())
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "native_source_values.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["center_id", "x", "y", "z", "native_uint16", "zarr_uint16", "signed_native_minus_zarr"])
        for row in result["native_source_gate"]["values"]:
            writer.writerow([row["center_id"], *row["xyz"], row["native_uint16"], row["zarr_uint16"], row["signed_native_minus_zarr"]])
    with (args.out / "native_center_diagnostic.csv").open("w", newline="") as stream:
        fields = list(result["native_center_trilinear_diagnostic"]["centers"][0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["native_center_trilinear_diagnostic"]["centers"])
    if not result["anatomy"]:
        return
    with (args.out / "native_grid_all_shifts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point_id", "dx", "dy", "dz", "arm", "zero_shift_primary",
                         "intensity_n", "intensity_pearson", "intensity_spearman", "intensity_status",
                         "laplacian_n", "laplacian_pearson", "laplacian_spearman", "laplacian_status"])
        for item in result["anatomy"]:
            for row in item["all_shifts"]:
                a, b = row["intensity"], row["laplacian_interior"]
                writer.writerow([item["id"], *row["shift_xyz_coarse_voxels"], row["arm"], row["original_transform_primary"],
                                 a["n"], a["pearson"], a["spearman"], a["status"], b["n"], b["pearson"], b["spearman"], b["status"]])
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), sharex=True, sharey=True, constrained_layout=True)
    for index, item in enumerate(result["anatomy"]):
        records = item["all_shifts"]
        shifts = [tuple(r["shift_xyz_coarse_voxels"]) for r in records if r["arm"] == "fine_point"]
        zero = shifts.index((0, 0, 0))
        for col, metric in enumerate(("intensity", "laplacian_interior")):
            ax = axes[index, col]
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
    fig.suptitle("Frag5: raw scanner grids at the original affine and all 27 shifts\nSeparate adaptive experiment; four neighborhoods in one physical fragment", fontsize=14)
    for suffix in ("svg", "png"):
        fig.savefig(args.out / f"native_grid_all_shifts.{suffix}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
