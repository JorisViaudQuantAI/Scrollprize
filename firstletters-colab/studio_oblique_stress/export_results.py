#!/usr/bin/env python3
"""Export completed Q33 results without importing the frozen experiment code.

Every arm is retained in protocol order. The figure shows pooled AP, recall,
and false-positive rate on the same physical pixels; local x and y share the
single identity observation at zero degrees. No model, arm selection, masking
change, or significance test is performed here.

Example:
    python export_results.py --prepared PREPARED --output OUTPUT \
        --destination EXPORT
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ANGLES = (-20, -10, -5, 0, 5, 10, 20)
RATE_KEYS = ("ap", "auc", "recall_at_0_5", "false_positive_rate_at_0_5")
METRIC_KEYS = (
    "ink_pixels", "background_pixels", "unknown_pixels",
    "true_positive_at_0_5", "false_positive_at_0_5",
    "ap", "auc", "recall_at_0_5", "false_positive_rate_at_0_5",
)
GROUP_NAMES = {"mixed": "mixed", "ink": "pure_ink", "background": "pure_background"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def direct_counts(probabilities, states):
    """Unknown and geometry-excluded pixels contribute to neither class."""
    ink, background = states == 2, states == 1
    positive, negative = int(ink.sum()), int(background.sum())
    detected = probabilities >= 0.5
    tp, fp = int((detected & ink).sum()), int((detected & background).sum())
    return dict(
        ink_pixels=positive, background_pixels=negative,
        unknown_pixels=int((states == 0).sum()),
        true_positive_at_0_5=tp, false_positive_at_0_5=fp,
        false_negative_at_0_5=positive - tp, true_negative_at_0_5=negative - fp,
        recall_at_0_5=tp / positive if positive else None,
        false_positive_rate_at_0_5=fp / negative if negative else None,
    )


def average_precision(probabilities, states):
    """Noninterpolated AP with tied scores evaluated as a single threshold.

    Leave single-class groups undefined, matching the frozen report's contract;
    in particular, pure-ink AP is not presented as a perfect detection result.
    """
    known = states > 0
    truth, score = states[known] == 2, probabilities[known]
    positive = int(truth.sum())
    if not positive or positive == len(truth):
        return None
    order = np.argsort(-score, kind="stable")
    truth, score = truth[order], score[order]
    ends = np.r_[np.flatnonzero(score[1:] != score[:-1]), len(score) - 1]
    cumulative_tp = np.cumsum(truth)[ends]
    added_tp = np.diff(np.r_[0, cumulative_tp])
    return float(np.sum((added_tp / positive) * (cumulative_tp / (ends + 1))))


def load_verified(prepared, output):
    protocol_path, result_path = prepared / "protocol.json", output / "results.json"
    protocol, result = read_json(protocol_path), read_json(result_path)
    require(result["protocol_sha256"] == sha256(protocol_path), "Protocol hash mismatch")
    archive = output / "scored_arrays.npz"
    require(result["scored_archive_sha256"] == sha256(archive), "Scored archive hash mismatch")
    state_path = prepared / "states.npy"
    require(sha256(state_path) == protocol["arrays"]["states.npy"]["sha256"],
            "Prepared states hash mismatch")
    arms, controls = protocol["arms"], protocol["controls"]
    names = [arm["name"] for arm in arms]
    expected = {("identity", "x", 0)} | {
        (f"{axis}_{'minus' if angle < 0 else 'plus'}{abs(angle)}", axis, angle)
        for axis in ("x", "y") for angle in ANGLES if angle
    }
    require(len(arms) == 13 and
            {(a["name"], a["axis"], a["angle_deg"]) for a in arms} == expected,
            "Expected identity and all twelve frozen local x/y tilt arms")
    require(set(result["pooled"]) == set(names), "Incomplete or unexpected pooled conditions")
    require(len(controls) == protocol["selected_patch_count"] == 17, "Expected all 17 frozen cores")
    categories = np.array([row["category"] for row in controls])
    require({name: int((categories == name).sum()) for name in GROUP_NAMES} ==
            {"mixed": 11, "ink": 3, "background": 3}, "Unexpected core groups")
    with np.load(archive, allow_pickle=False) as data:
        require(data["condition_names"].tolist() == names, "Archive condition order mismatch")
        require(data["author_indices"].tolist() == [row["index"] for row in controls],
                "Archive control order mismatch")
        probability, states = data["probabilities"], data["states"]
        original, common = data["original_states"], data["common_mask"]
    require(states.shape == (17, 32, 32), "Unexpected core state shape")
    require(probability.shape == (13, *states.shape), "Unexpected scored probability shape")
    require(original.shape == common.shape == states.shape and common.dtype == bool,
            "Unexpected original state or common mask shape/type")
    require(np.isin(original, (0, 1, 2)).all() and np.isin(states, (0, 1, 2)).all(),
            "Unknown state encoding")
    require(np.array_equal(states, np.load(state_path, allow_pickle=False)),
            "Archive and prepared states differ")
    require(np.array_equal(states, np.where(common, original, 0)),
            "Archive states do not use the fixed common mask")
    scored = probability[:, states > 0]
    require(np.isfinite(scored).all() and ((scored >= 0) & (scored <= 1)).all(),
            "Nonfinite or out-of-range probability on scored pixels")
    require(int((states == 2).sum()) == protocol["denominator"]["ink"] and
            int((states == 1).sum()) == protocol["denominator"]["background"],
            "Frozen denominator mismatch")
    for arm, values in zip(arms, probability):
        measured = result["pooled"][arm["name"]]
        for key in RATE_KEYS:
            value = measured[key]
            require(value is not None and np.isfinite(value) and 0 <= value <= 1,
                    f"Invalid pooled {key}: {arm['name']}")
        counts = direct_counts(values, states)
        for key in METRIC_KEYS:
            if key in counts:
                require(measured[key] == counts[key], f"Pooled count/rate mismatch: {arm['name']}/{key}")
        require(abs(measured["ap"] - average_precision(values, states)) <= 1e-12,
                f"Pooled AP does not reproduce: {arm['name']}")
    return protocol, result, probability, states, categories


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            # Raw arrays remain available for full precision; the presentation
            # tables avoid implying precision beyond these six decimal places.
            writer.writerow({key: (f"{value:.6f}" if isinstance(value, float) else value)
                             for key, value in row.items()})


def export_tables(destination, protocol, result, probability, states, categories):
    identity = result["pooled"]["identity"]
    pooled_rows, group_rows = [], []
    for arm, values in zip(protocol["arms"], probability):
        header = dict(condition=arm["name"], axis="identity" if arm["name"] == "identity" else arm["axis"],
                      angle_deg=arm["angle_deg"])
        measured = result["pooled"][arm["name"]]
        row = dict(header, **{key: measured[key] for key in METRIC_KEYS})
        row.update({"delta_" + key: measured[key] - identity[key] for key in RATE_KEYS})
        pooled_rows.append(row)
        for category, display in GROUP_NAMES.items():
            select = categories == category
            counts = direct_counts(values[select], states[select])
            group_rows.append(dict(header, group=display, cores=int(select.sum()),
                                   **counts, ap=average_precision(values[select], states[select])))
    write_csv(destination / "pooled_metrics.csv", list(pooled_rows[0]), pooled_rows)
    write_csv(destination / "group_metrics.csv", list(group_rows[0]), group_rows)


def export_figure(destination, protocol, result, states):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "q33-oblique-stress"})
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.985, top=.73, bottom=.31, wspace=.23)
    fig.suptitle("Studio: sensitivity to local 3D sampling tilt", x=.065, ha="left",
                 y=.965, fontsize=18, weight="bold")
    fig.text(.065, .902,
             f"PHerc0841 author holdout · {len(protocol['controls'])} cores · 13 conditions · "
             f"{int((states > 0).sum()):,} identical scored physical pixels", fontsize=11)
    styles = {"x": ("#0072B2", "o"), "y": ("#D55E00", "s")}
    specs = (("ap", "Average precision (AP)"),
             ("recall_at_0_5", "Ink recall at p ≥ 0.5"),
             ("false_positive_rate_at_0_5", "Background FPR at p ≥ 0.5"))
    for ax, (key, title) in zip(axes, specs):
        identity = result["pooled"]["identity"][key]
        ax.axhline(identity, color="#595959", linestyle="--", linewidth=1, zorder=1)
        ax.axvline(0, color="#d5d9dd", linewidth=.8, zorder=0)
        for axis, (color, marker) in styles.items():
            names = {a["angle_deg"]: a["name"] for a in protocol["arms"]
                     if a["axis"] == axis and a["angle_deg"] != 0}
            names[0] = "identity"
            values = [result["pooled"][names[angle]][key] for angle in ANGLES]
            ax.plot(ANGLES, values, color=color, marker=marker, markersize=5,
                    markevery=[0, 1, 2, 4, 5, 6], linewidth=1.7, zorder=3)
        ax.scatter([0], [identity], color="#222222", marker="D", s=35, zorder=4)
        ax.set(title=title, xlabel="Signed local tilt (degrees)", xlim=(-22, 22), ylim=(0, 1))
        ax.set_xticks(ANGLES)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(axis="y", color="#e3e6e8", linewidth=.8)
        ax.set_axisbelow(True)
    handles = [Line2D([], [], color=color, marker=marker, label=f"Local {axis} axis", linewidth=1.7)
               for axis, (color, marker) in styles.items()]
    handles.extend([
        Line2D([], [], color="#222222", marker="D", linestyle="none", label="Shared identity (0°)"),
        Line2D([], [], color="#595959", linestyle="--", linewidth=1, label="Identity reference"),
    ])
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.06, .867),
               ncol=4, frameon=False, columnspacing=2.2)
    fig.text(.065, .205, "All prespecified angles are shown. Lines connect measured conditions; no angle is selected or averaged.",
             fontsize=10, color="#374151")
    fig.text(.065, .153, "Fixed common mask; unknown labels are excluded. Local chart axes are not certified handwriting directions.",
             fontsize=9, color="#4b5563")
    fig.text(.065, .113, "One exposed author region; spatially related cores. No IID-pixel intervals, unseen-scroll accuracy, or readability claim.",
             fontsize=9, color="#4b5563")
    gate = protocol["geometry_gate"]
    policy = (f"{gate['policy']}: retention ≥{gate['minimum_known_retention_global']:.0%} overall, "
              f"≥{gate['minimum_known_retention_each_core']:.0%} per core.")
    if gate["policy"] == "retained90":
        policy += " Per-core gate amended after geometry inspection, before predictions."
    fig.text(.065, .069, policy, fontsize=9, color="#4b5563")
    fig.savefig(destination / "oblique_stress.png", dpi=200, facecolor="white")
    fig.savefig(destination / "oblique_stress.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("prepared", "output", "destination"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    protocol, result, probability, states, categories = load_verified(args.prepared, args.output)
    args.destination.mkdir(parents=True, exist_ok=True)
    export_tables(args.destination, protocol, result, probability, states, categories)
    export_figure(args.destination, protocol, result, states)
    products = ("pooled_metrics.csv", "group_metrics.csv", "oblique_stress.png", "oblique_stress.svg")
    manifest = dict(
        purpose="Presentation of all frozen Q33 conditions; no selection, inference, or mask changes",
        protocol_sha256=result["protocol_sha256"],
        results_sha256=sha256(args.output / "results.json"),
        scored_archive_sha256=result["scored_archive_sha256"],
        export_script_sha256=sha256(Path(__file__)),
        threshold=0.5, undefined_single_class_rates="Blank CSV fields; no denominator is invented",
        single_class_ap="Undefined for pure-ink and pure-background groups",
        statistics=result["statistics"], scope=result["scope"],
        products={name: dict(sha256=sha256(args.destination / name),
                             bytes=(args.destination / name).stat().st_size) for name in products},
    )
    (args.destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exported": list(products) + ["export_manifest.json"],
                      "conditions": len(protocol["arms"]), "scored_pixels": int((states > 0).sum())}))


if __name__ == "__main__":
    main()
