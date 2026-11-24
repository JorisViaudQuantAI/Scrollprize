#!/usr/bin/env python3
"""Export complete factorial results using matched within-block contrasts.

No inference or mask choice occurs here. This consumer imports no producer
module and independently checks counts, AP/AUC and factorial contrasts. The
figure uses differences from each block's own identity, not absolute AP curves
on varying populations. Undefined single-class metrics remain undefined.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_PROTOCOL_SHA = "c502a6bad514c4bd8eb63656f93cdcd82d62b58b11250ea290c8cf7690b7ef5c"
FAMILIES = ("identity", "normal_only", "anchor_only", "rigid")
GROUPS = ("all", "mixed", "pure_ink", "pure_background")
RATES = ("ap", "auc", "recall_at_0_5", "false_positive_rate_at_0_5")
COUNTS = ("ink_pixels", "background_pixels", "unknown_pixels",
          "true_positive_at_0_5", "false_positive_at_0_5")
CONTRASTS = {
    "ray_at_fixed_anchors": (-1, 1, 0, 0),
    "ray_at_rotated_anchors": (0, 0, -1, 1),
    "anchor_at_fixed_rays": (-1, 0, 1, 0),
    "anchor_at_rotated_rays": (0, -1, 0, 1),
    "interaction": (1, -1, -1, 1),
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def calculate(probability, states):
    """Independent counts and sklearn ranking metrics on known labels only."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    ink, background = states == 2, states == 1
    positive, negative = int(ink.sum()), int(background.sum())
    detected = probability >= .5
    tp, fp = int((ink & detected).sum()), int((background & detected).sum())
    row = dict(ink_pixels=positive, background_pixels=negative,
               unknown_pixels=int((states == 0).sum()),
               true_positive_at_0_5=tp, false_positive_at_0_5=fp,
               false_negative_at_0_5=positive-tp, true_negative_at_0_5=negative-fp,
               recall_at_0_5=tp/positive if positive else None,
               false_positive_rate_at_0_5=fp/negative if negative else None,
               ap=None, auc=None)
    known = states > 0
    scored = probability[known]
    require(np.isfinite(scored).all() and ((scored >= 0) & (scored <= 1)).all(),
            "Invalid probability on scored support")
    if positive and negative:
        truth = states[known] == 2
        row["ap"] = float(average_precision_score(truth, scored))
        row["auc"] = float(roc_auc_score(truth, scored))
    return row


def equal_metric(measured, expected, description):
    if measured is None or expected is None:
        require(measured is None and expected is None, "Undefined metric mismatch: "+description)
    else:
        require(np.isfinite(measured) and abs(measured-expected) <= 1e-12,
                "Metric mismatch: "+description)


def load_verified(prepared, output):
    protocol_path = prepared/"protocol.json"
    protocol, result = read_json(protocol_path), read_json(output/"results.json")
    require(sha256(protocol_path) == EXPECTED_PROTOCOL_SHA == result["protocol_sha256"],
            "Frozen protocol hash mismatch")
    archive_path = output/"scored_arrays.npz"
    require(sha256(archive_path) == result["scored_archive_sha256"], "Score archive hash mismatch")
    for name in ("states.npy", "original_states.npy"):
        require(sha256(prepared/name) == protocol["arrays"][name]["sha256"],
                "Prepared label array hash mismatch: "+name)
    require(tuple(protocol["families"]) == FAMILIES, "Family order changed")
    blocks, controls = protocol["blocks"], protocol["controls"]
    expected_blocks = [(axis, angle) for axis in ("x", "y") for angle in (-20, -5, 5, 20)]
    require([(b["axis"], b["angle_deg"]) for b in blocks] == expected_blocks,
            "Expected all eight fixed axis-angle blocks")
    require(len(controls) == 17, "Expected the same 17 cores")
    categories = np.array([c["category"] for c in controls])
    selectors = dict(all=np.ones(17, bool), mixed=categories == "mixed",
                     pure_ink=categories == "ink", pure_background=categories == "background")
    require([int(selectors[g].sum()) for g in GROUPS] == [17, 11, 3, 3], "Core groups changed")
    with np.load(archive_path, allow_pickle=False) as archive:
        require(archive["families"].tolist() == list(FAMILIES), "Archive family order changed")
        require(archive["block_names"].tolist() == [b["name"] for b in blocks], "Archive block order changed")
        require(archive["author_indices"].tolist() == [c["index"] for c in controls], "Archive core order changed")
        probabilities = archive["probabilities"]
        states, original = archive["states"], archive["original_states"]
        masks, global_mask = archive["common_masks"], archive["global_common"]
    require(probabilities.shape == (8, 4, 17, 32, 32), "Unexpected probability shape")
    require(states.shape == masks.shape == (8, 17, 32, 32), "Unexpected block state/mask shape")
    require(original.shape == global_mask.shape == (17, 32, 32), "Unexpected original/global shape")
    require(masks.dtype == global_mask.dtype == np.dtype(bool), "Expected Boolean masks")
    require(np.isin(original, (0, 1, 2)).all(), "Unknown original state encoding")
    require(np.array_equal(states, np.where(masks, original[None], 0)), "Block mask/label mismatch")
    require(np.array_equal(global_mask, masks.all(axis=0)), "Global intersection mismatch")
    require(np.array_equal(states, np.load(prepared/"states.npy", allow_pickle=False)), "Prepared states differ")
    require(np.array_equal(original, np.load(prepared/"original_states.npy", allow_pickle=False)), "Original states differ")
    require([r["name"] for r in result["blocks"]] == [b["name"] for b in blocks], "Result block order mismatch")
    require([r["name"] for r in result["secondary_global"]["blocks"]] == [b["name"] for b in blocks],
            "Secondary result block order mismatch")
    global_states = np.where(global_mask, original, 0).astype(np.uint8)
    require(int((global_states == 2).sum()) == result["secondary_global"]["ink"] and
            int((global_states == 1).sum()) == result["secondary_global"]["background"],
            "Secondary global denominator mismatch")
    return protocol, result, probabilities, states, original, global_states, selectors


def checked_rows(protocol, block_results, probabilities, states, original, selectors, population):
    cell_rows, contrast_rows, checked = [], [], []
    for bi, (block, reported) in enumerate(zip(protocol["blocks"], block_results)):
        measured = {}
        for family_index, family in enumerate(FAMILIES):
            measured[family] = {}
            require(set(reported["metrics"][family]) == set(GROUPS), "Incomplete metric groups")
            for group in GROUPS:
                choose = selectors[group]
                values = calculate(probabilities[bi, family_index, choose], states[bi, choose])
                recorded = reported["metrics"][family][group]
                for key in COUNTS:
                    require(values[key] == recorded[key], f"Count mismatch: {block['name']}/{family}/{group}/{key}")
                for key in RATES:
                    equal_metric(values[key], recorded[key], f"{block['name']}/{family}/{group}/{key}")
                measured[family][group] = values
                base = reported["metrics"]["identity"][group]
                original_known = int((original[choose] > 0).sum())
                scored_known = values["ink_pixels"] + values["background_pixels"]
                row = dict(population=population, block=block["name"], axis=block["axis"],
                    angle_deg=block["angle_deg"], family=family,
                    anchor_rotated=int(family in ("anchor_only", "rigid")),
                    ray_rotated=int(family in ("normal_only", "rigid")),
                    group=group, cores=int(choose.sum()), original_known=original_known,
                    excluded_known=original_known-scored_known,
                    known_retention=scored_known/original_known, **values)
                for key in RATES:
                    row["delta_"+key+"_vs_matched_identity"] = None if values[key] is None else values[key]-base[key]
                cell_rows.append(row)
        for group in GROUPS:
            for metric in RATES:
                values = [measured[family][group][metric] for family in FAMILIES]
                if any(value is None for value in values):
                    require(metric not in reported["contrasts"][group], "Undefined contrast was reported")
                    continue
                for contrast, coefficients in CONTRASTS.items():
                    value = float(sum(c*v for c, v in zip(coefficients, values)))
                    equal_metric(value, reported["contrasts"][group][metric][contrast],
                                 f"{block['name']}/{group}/{metric}/{contrast}")
                    denominator = measured["identity"][group]
                    contrast_rows.append(dict(population=population, block=block["name"],
                        axis=block["axis"], angle_deg=block["angle_deg"], group=group,
                        ink_pixels=denominator["ink_pixels"], background_pixels=denominator["background_pixels"],
                        metric=metric, contrast=contrast, value=value))
        checked.append(measured)
    return cell_rows, contrast_rows, checked


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.9f}" if isinstance(value, float) else value
                             for key, value in row.items()})


def export_figure(destination, protocol, checked):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "q33-anchor-ray-factorial"})
    fig = plt.figure(figsize=(16.6, 8.9))
    grid = fig.add_gridspec(1, 4, width_ratios=(1.7, 2.2, 2.0, 2.0),
                           left=.045, right=.985, top=.73, bottom=.26, wspace=.22)
    label_axis = fig.add_subplot(grid[0, 0])
    axes = [fig.add_subplot(grid[0, i]) for i in (1, 2, 3)]
    fig.suptitle("Studio: separating anchor placement and ray direction",
                 x=.045, y=.965, ha="left", fontsize=18, weight="bold")
    fig.text(.045, .905, "Q33 mechanistic follow-up · 11 mixed cores · 8 fixed axis–angle blocks · all four factorial cells retained",
             fontsize=11)
    styles = {"normal_only": ("#0072B2", "o", -.20, "Ray direction only"),
              "anchor_only": ("#D55E00", "s", 0.0, "Anchor placement only"),
              "rigid": ("#7B3294", "D", .20, "Both factors (rigid)")}
    handles = [Line2D([], [], color=c, marker=m, linestyle="none", markersize=6, label=label)
               for c, m, _, label in styles.values()]
    handles.append(Line2D([], [], color="#555555", linewidth=1, label="Matched identity (zero effect)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.04, .868),
               ncol=4, frameon=False, columnspacing=2.0)
    positions = np.arange(8)
    for ax in [label_axis, *axes]:
        ax.set_ylim(7.65, -.65)
    label_axis.set_xlim(0, 1)
    label_axis.axis("off")
    for x, text, alignment in ((0, "Block", "left"), (.42, "Recall₀", "right"),
                               (.72, "Ink n", "right"), (1.0, "BG n", "right")):
        label_axis.text(x, 1.035, text, transform=label_axis.transAxes,
                        ha=alignment, va="bottom", weight="bold", fontsize=9)
    for bi, (block, values) in enumerate(zip(protocol["blocks"], checked)):
        base = values["identity"]["mixed"]
        label = f"{block['axis']} {block['angle_deg']:+d}°".replace("-", "−")
        for x, text, alignment in ((0, label, "left"), (.42, f"{100*base['recall_at_0_5']:.1f}%", "right"),
                                  (.72, f"{base['ink_pixels']:,}", "right"),
                                  (1.0, f"{base['background_pixels']:,}", "right")):
            label_axis.text(x, bi, text, ha=alignment, va="center", fontsize=9)
    specs = (("recall_at_0_5", "Mixed ink recall", -.10),
             ("ap", "Mixed pooled AP", -.05),
             ("false_positive_rate_at_0_5", "Mixed background FPR", .03))
    for ax, (metric, title, threshold) in zip(axes, specs):
        all_deltas = []
        ax.axvline(0, color="#555555", linewidth=1, zorder=1)
        ax.axvline(100*threshold, color="#999999", linewidth=.9, linestyle="--", zorder=0)
        for family, (color, marker, offset, label) in styles.items():
            delta = np.array([100*(r[family]["mixed"][metric]-r["identity"]["mixed"][metric]) for r in checked])
            all_deltas.extend(delta.tolist())
            y = positions+offset
            ax.hlines(y, 0, delta, color=color, linewidth=1.35, alpha=.85)
            ax.scatter(delta, y, color=color, marker=marker, s=30, zorder=3)
        magnitude = max(10., 10.*np.ceil(max(abs(np.asarray(all_deltas)).max(), abs(100*threshold))/10.))
        ax.set_xlim(-min(105., magnitude+5), min(105., magnitude+5))
        ax.set_yticks(positions)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_title(title, fontsize=12, pad=11)
        ax.set_xlabel("Difference from matched identity (points)", fontsize=9, labelpad=8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:+.0f}" if x else "0"))
        ax.grid(axis="y", color="#e5e7eb", linewidth=.8)
        ax.set_axisbelow(True)
        ax.axhline(3.5, color="#b4bbc3", linewidth=1)
    label_axis.axhline(3.5, color="#b4bbc3", linewidth=1)
    fig.text(.045, .165, "Every effect uses the same pixels as its block's identity. Block populations differ; no absolute AP curve connects them.",
             fontsize=10, color="#374151")
    fig.text(.045, .121, "Dashed lines: predeclared engineering magnitudes (recall −10 points; AP −5 points; FPR +3 points), not significance bounds.",
             fontsize=9, color="#4b5563")
    fig.text(.045, .077, "Hybrid rays need not be normals of their anchor surface. Same previously inspected donor region; no independent-text or transfer claim.",
             fontsize=9, color="#4b5563")
    fig.savefig(destination/"factorial_paired_effects.png", dpi=200, facecolor="white")
    fig.savefig(destination/"factorial_paired_effects.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "output", "destination"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    protocol, result, probability, states, original, global_states, selectors = load_verified(args.prepared, args.output)
    cells, contrasts, checked = checked_rows(protocol, result["blocks"], probability, states,
                                             original, selectors, "within_block_primary")
    global_cells, global_contrasts, _ = checked_rows(protocol, result["secondary_global"]["blocks"], probability,
        np.broadcast_to(global_states, states.shape), original, selectors, "global_intersection_secondary")
    args.destination.mkdir(parents=True, exist_ok=True)
    for name, rows in (("factorial_cells.csv", cells), ("factorial_contrasts.csv", contrasts),
                       ("secondary_global_cells.csv", global_cells), ("secondary_global_contrasts.csv", global_contrasts)):
        write_csv(args.destination/name, rows)
    export_figure(args.destination, protocol, checked)
    products = ("factorial_cells.csv", "factorial_contrasts.csv", "secondary_global_cells.csv",
                "secondary_global_contrasts.csv", "factorial_paired_effects.png", "factorial_paired_effects.svg")
    manifest = dict(protocol_sha256=EXPECTED_PROTOCOL_SHA, results_sha256=sha256(args.output/"results.json"),
        scored_archive_sha256=result["scored_archive_sha256"], exporter_sha256=sha256(__file__),
        primary_cell_rows=len(cells), primary_contrast_rows=len(contrasts),
        groups=list(GROUPS), family_order=list(FAMILIES), probability_threshold=.5,
        figure="Within-block mixed-population contrasts only; separate paired identity and denominators for each block",
        undefined_metrics="Blank cell-table fields; undefined single-class contrasts omitted",
        interpretation="Descriptive factorial differences, not fractions of causal attribution. AP is nonadditive; block populations differ.",
        secondary="Global intersection reported separately, without choosing a population after inspecting scores",
        products={name:dict(bytes=(args.destination/name).stat().st_size,sha256=sha256(args.destination/name)) for name in products})
    (args.destination/"export_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(exported=list(products)+["export_manifest.json"], cell_rows=len(cells), contrast_rows=len(contrasts))))


if __name__ == "__main__":
    main()
