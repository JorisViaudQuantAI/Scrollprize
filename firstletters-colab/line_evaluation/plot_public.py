#!/usr/bin/env python3
"""Plot the two fixed, already-published controls; never an unread-scroll packet."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
import numpy as np

CASES = (
    ("pherc306b_context_raw", "PHerc. 306B — released model product", "Top 3: first row partly localized; second row missed"),
    ("pherc343_context_display", "PHerc. 343 — published display signal", "Top 3: at least half-area coverage in 15 of 18 source boxes"),
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new figure path")
    results = {r["id"]: r for r in json.loads(args.results.read_text())["results"]}
    inventory = {r["id"]: r for r in json.loads((Path(__file__).parents[1] / "line_controls/inventory.json").read_text())["cases"]}
    fig, axes = plt.subplots(2, 1, figsize=(12, 11.7), constrained_layout=True)
    for ax, (name, title, subtitle) in zip(axes, CASES):
        folder = args.data / "cases" / name
        manifest = inventory[name]
        if sha(folder / "score.npy") != manifest["score_sha256"] or sha(folder / "validity.npy") != manifest["mask_sha256"]:
            raise ValueError("Only the frozen released public control is allowed here")
        score = np.load(folder / "score.npy", allow_pickle=False, mmap_mode="r")
        valid = np.load(folder / "validity.npy", allow_pickle=False, mmap_mode="r")
        h, w = score.shape
        dy, dx = np.asarray(manifest["spacing_um_yx"]) / 1000
        # Preview only: raw arrays and numerical metrics remain on their original grid.
        size = (1600, round(h * 1600 / w))
        preview = cv2.resize(np.asarray(score), size, interpolation=cv2.INTER_AREA)
        observed = cv2.resize(np.asarray(valid, np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        gray = np.rint(np.clip(preview, 0, 1) * 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        rgb[~observed] = (76, 35, 85)
        ax.imshow(rgb, extent=(0, w * dx, h * dy, 0), interpolation="nearest")
        for target in manifest["targets"]:
            x0, y0, x1, y1 = target["box_xyxy"]
            ax.add_patch(Rectangle((x0*dx, y0*dy), (x1-x0)*dx, (y1-y0)*dy,
                                   fill=False, edgecolor="#28d7df", linewidth=.8, linestyle="--"))
        for rank, proposal in enumerate(results[name]["top10_proposals"][:3], 1):
            polygon = np.asarray(proposal["band_polygon_xy"]) * [dx, dy]
            ax.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor="#ff8b35", linewidth=1.5))
            x, y = polygon.mean(axis=0)
            ax.text(x, y, str(rank), color="white", fontsize=10, weight="bold",
                    bbox={"facecolor": "#333333", "alpha": .85, "edgecolor": "none", "pad": 1.5})
        ax.set_title(title + "\n" + subtitle, loc="left", fontsize=12)
        ax.set_xlabel("Chart x (mm)")
        ax.set_ylabel("Chart y (mm)")
        ax.set_xlim(0, w*dx)
        ax.set_ylim(h*dy, 0)
    fig.legend(handles=[Line2D([], [], color="#28d7df", linestyle="--", label="Published coarse target"),
                        Line2D([], [], color="#ff8b35", label="Frozen geometric proposal"),
                        Rectangle((0, 0), 1, 1, color="#4c2355", label="Unobserved / excluded")],
               loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Review localization on published writing — no letter or candidate verdict\n"
                 "Fixed [0,1] display; previews reduced; metrics use original arrays. Sources: released 306B / 343 submissions, CC BY-NC 4.0.",
                 fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, metadata={"Title": "Public sparse-line controls; localization only"})
    plt.close(fig)


if __name__ == "__main__":
    main()
