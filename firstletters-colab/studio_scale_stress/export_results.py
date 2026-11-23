#!/usr/bin/env python3
"""Package scored public-control arrays and plot the already frozen results.

This is presentation/export only. It does not run a model, change selection,
rescore inputs, choose a scale, or modify the prospective inference code.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.prepared/"protocol.json").read_text())
    result = json.loads((args.results/"results.json").read_text())
    assert sha(args.prepared/"protocol.json") == result["protocol_sha256"]
    assert sha(args.prepared/"states.npy") == result["states_sha256"]
    assert sha(args.results/"aligned_probabilities.npy") == result["aligned_probabilities_sha256"]
    args.destination.mkdir(parents=True, exist_ok=True)
    figures = args.destination.parent/"figures"
    figures.mkdir(exist_ok=True)
    for name in ["results.json", "identity_check.json", "inference_receipt.json"]:
        shutil.copyfile(args.results/name, args.destination/name)
    names = [a["name"] for a in protocol["arms"]] + ["uniform_mean5"]
    arrays = np.load(args.results/"aligned_probabilities.npy", allow_pickle=False)
    states = np.load(args.prepared/"states.npy", allow_pickle=False)
    indices = np.array([a["index"] for a in protocol["controls"]], dtype=np.int32)
    coordinates = np.array([[a["y"]+48, a["x"]+48] for a in protocol["controls"]], dtype=np.int32)
    archive = args.destination/"scored_arrays.npz"
    np.savez_compressed(archive, probabilities=arrays, states=states, author_indices=indices,
                        core_origins_yx=coordinates, condition_names=np.array(names))
    with (args.destination/"pooled_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["condition"]+list(result["pooled"]["identity"]))
        writer.writeheader()
        for name in names:
            writer.writerow(dict(condition=name, **result["pooled"][name]))
    manifest = dict(archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                    purpose="Small scored physical cores for independent AP/AUC recomputation without1GB download or265 model forwards",
                    protocol_sha256=result["protocol_sha256"], condition_names=names,
                    probabilities=dict(shape=list(arrays.shape), dtype=str(arrays.dtype),
                                       source_npy_sha256=result["aligned_probabilities_sha256"]),
                    states=dict(shape=list(states.shape), dtype=str(states.dtype),
                                source_npy_sha256=result["states_sha256"]),
                    coordinates="Native prepared-array Y,X origin of the32x32 core; public annotated control only",
                    limitation="Archive contains scored probabilities, not full128 model fields or raw CT; full-output identity is in inference_receipt")
    (args.destination/"scored_arrays_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    display = ["Identity", "x × 0.5", "x × 2", "y × 0.5", "y × 2", "Mean of five"]
    colors = ["#275D85"]+["#B25B47"]*4+["#81642C"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.2), gridspec_kw={"width_ratios": [1, 1.14]})
    fig.subplots_adjust(top=.80, bottom=.20, left=.11, right=.97, wspace=.40)
    fig.suptitle("Studio: response to anisotropic image scale", x=.06, ha="left", y=.965, fontsize=18, weight="bold")
    fig.text(.06, .90, "Same 45,349 annotated physical pixels · 53 disjoint cores · 265 CPU forwards", fontsize=11)
    ax = axes[0]
    ap = [result["pooled"][n]["ap"] for n in names]
    positions = np.arange(6)
    ax.barh(positions, ap, color=colors, height=.64)
    ax.set_yticks(positions, display)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Pooled average precision (AP)")
    ax.set_title("Matched-pixel detection", loc="left", pad=12, fontsize=12, weight="bold")
    ax.axvline(ap[0], color=colors[0], linewidth=1, linestyle=":")
    for i, value in enumerate(ap):
        ax.text(value+.014, i, f"{value:.3f}", va="center", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=.3))
    ax.xaxis.grid(True, color="#e5e7eb", zorder=0)
    ax.set_axisbelow(True)
    ax = axes[1]
    rng = np.random.default_rng(20261123)
    for i, name in enumerate(names[1:]):
        changes = np.array([r["metrics"][name]["ap"]-r["metrics"]["identity"]["ap"]
                            for r in result["per_core"] if r["metrics"]["identity"]["ap"] is not None])
        ax.scatter(changes, i+rng.uniform(-.16, .16, len(changes)), s=22, color=colors[i+1], alpha=.72, edgecolors="none")
        med = np.median(changes)
        ax.plot([med, med], [i-.28, i+.28], color="#17212b", linewidth=2.3)
    ax.set_yticks(np.arange(5), display[1:])
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="#17212b", linewidth=1)
    ax.set_xlabel("AP change from identity, per mixed core")
    ax.set_title("Local effects: 23 mixed cores", loc="left", pad=12, fontsize=12, weight="bold")
    ax.xaxis.grid(True, color="#e5e7eb")
    ax.set_axisbelow(True)
    fig.text(.06, .09, "Dots: individual mixed-label cores; vertical marks: medians. No independence-based confidence intervals.", fontsize=9, color="#4b5563")
    fig.text(.06, .052, "Exposed PHerc0841 author holdout. Scale changes resampling and physical context; this is not unseen-scroll accuracy.", fontsize=9, color="#4b5563")
    fig.savefig(figures/"scale_stress.png", dpi=190, facecolor="white")
    fig.savefig(figures/"scale_stress.svg", facecolor="white")
    print(json.dumps({"archive_bytes": manifest["archive_bytes"], "archive_sha256": manifest["archive_sha256"],
                      "figure_png_bytes": (figures/"scale_stress.png").stat().st_size}))


if __name__ == "__main__":
    main()
