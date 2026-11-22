#!/usr/bin/env python3
"""Frozen CPU Hecate context interventions on previously acquired public data.

No downloader, training, adaptive selection or unread-scroll access. Run only
after the protocol and this script have been committed at --freeze-reference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent / "transversal_lab"
CONDITIONS = ("original", "zero_halo", "reflect_halo")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def outside_git(path):
    resolved = Path(path).resolve()
    if any(os.path.lexists(parent / ".git") for parent in (resolved, *resolved.parents)):
        raise ValueError("Experiment outputs must stay outside every Git worktree")
    return resolved


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def intervention(raw, condition):
    raw = np.asarray(raw)
    if raw.dtype != np.uint8 or raw.shape != (16, 64, 64):
        raise ValueError("Requires the frozen uint8 16x64x64 input")
    if condition == "original":
        output = raw.copy()
    elif condition == "zero_halo":
        output = np.zeros_like(raw)
        output[:, 16:48, 16:48] = raw[:, 16:48, 16:48]
    elif condition == "reflect_halo":
        output = np.pad(raw[:, 16:48, 16:48], ((0, 0), (16, 16), (16, 16)), mode="reflect")
    else:
        raise ValueError("Unknown frozen condition")
    if not np.array_equal(output[:, 16:48, 16:48], raw[:, 16:48, 16:48]):
        raise ValueError("Intervention changed the central CT")
    return np.ascontiguousarray(output)


def sigmoid(logits):
    x = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite model logit")
    return np.exp(-np.logaddexp(0.0, -x))


def metrics(probability, labels, support):
    probability, labels, support = map(np.asarray, (probability, labels, support))
    if probability.shape != labels.shape or support.shape != labels.shape or support.dtype != bool:
        raise ValueError("Metric arrays need matching grids and explicit boolean support")
    if not set(np.unique(labels)).issubset({0, 1, 2}):
        raise ValueError("Unexpected categorical label")
    if not np.isfinite(probability[support]).all() or np.any((probability[support] < 0) | (probability[support] > 1)):
        raise ValueError("Invalid supported probabilities")
    known = support & (labels != 0)
    truth, score = labels[known] == 2, probability[known]
    pos, neg = int(truth.sum()), int((~truth).sum())
    called = score >= 0.5
    tp, fp = int((called & truth).sum()), int((called & ~truth).sum())
    result = {
        "known_pixels": int(known.sum()), "ink": pos, "background": neg,
        "unknown_or_unsupported": int(known.size - known.sum()),
        "true_positive_at_0_5": tp, "false_positive_at_0_5": fp,
        "recall_at_0_5": tp / pos if pos else None,
        "background_exceedance_at_0_5": fp / neg if neg else None,
        "auroc": None, "average_precision": None,
    }
    if pos and neg:
        order = np.argsort(-score, kind="stable")
        ordered, sorted_score = truth[order], score[order]
        ends = np.r_[np.flatnonzero(np.diff(sorted_score)), len(score) - 1]
        tps = np.cumsum(ordered)[ends]
        fps = ends + 1 - tps
        recall, precision = tps / pos, tps / (tps + fps)
        result["average_precision"] = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
        result["auroc"] = float(np.trapezoid(np.r_[0.0, recall], np.r_[0.0, fps / neg]))
    return result


def paired_drift(original, changed, support):
    a, b = np.asarray(original)[support], np.asarray(changed)[support]
    if not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Need finite paired supported values")
    delta = b - a
    return {
        "pixels": int(a.size), "mean_signed_change": float(delta.mean()),
        "mean_absolute_change": float(np.abs(delta).mean()),
        "absolute_change_quantiles_0_50_95_99_100": np.quantile(np.abs(delta), [0, .5, .95, .99, 1]).tolist(),
        "pearson_r": float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None,
        "probability_0_5_disagreements": int(((a >= .5) != (b >= .5)).sum()),
    }


class LocalVerifiedCache:
    """Read-only cache; only assets listed in a pinned prior receipt are allowed."""
    def __init__(self, root, acquisition):
        self.root = Path(root)
        self.allowed = {row["url"]: row for row in acquisition["completed_assets"]}
        self.used = {}
        self.lock = threading.Lock()

    def get(self, url, expected_sha256=None, expected_bytes=None):
        row = self.allowed.get(url)
        if row is None:
            raise ValueError("Asset was not acquired in the pinned public-control run")
        path = self.root / hashlib.sha256(url.encode()).hexdigest()
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError("Required cached asset absent or byte count differs; network forbidden")
        with self.lock:
            if url not in self.used:
                if sha(path) != row["sha256"]:
                    raise ValueError("Cached source asset changed")
                self.used[url] = dict(row)
        if expected_sha256 is not None and row["sha256"] != expected_sha256:
            raise ValueError("Cache and source-manifest hashes differ")
        if expected_bytes is not None and row["bytes"] != expected_bytes:
            raise ValueError("Cache and source-manifest byte counts differ")
        return path

    def verify_unchanged(self):
        for url, row in self.used.items():
            path = self.root / hashlib.sha256(url.encode()).hexdigest()
            if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
                raise ValueError("Previously read cached source bytes changed")
        return len(self.used)


def verify_freeze(reference, protocol_path):
    if not re.fullmatch(r"[0-9a-f]{40}", reference):
        raise ValueError("A full published freeze commit is required")
    repo = HERE.parents[1]
    for path in (Path(__file__).resolve(), protocol_path.resolve()):
        relative = path.relative_to(repo).as_posix()
        committed = subprocess.check_output(["git", "show", f"{reference}:{relative}"], cwd=repo)
        if committed != path.read_bytes():
            raise ValueError("Runner or protocol differs from the frozen commit")


def execute(args, protocol, out):
    start = time.monotonic()
    completed = 0
    records = []
    stage = "verify_local_sources"
    signature = {"freeze_reference": args.freeze_reference, "protocol_sha256": sha(args.protocol),
                 "runner_sha256": sha(__file__), "network_requests": 0,
                 "training_steps": 0, "prior_outputs_used_for_selection": True}
    write_json(out / "start.json", signature)
    try:
        checks = {
            LAB / "hecate_probe.py": protocol["pins"]["original_runner_sha256"],
            LAB / "hecate_probe_protocol.json": protocol["pins"]["original_protocol_sha256"],
            LAB / "hecate_donor_manifest.json": protocol["pins"]["donor_manifest_sha256"],
            LAB / "evidence/hecate_role_source_acquisition.json": protocol["pins"]["prior_acquisition_sha256"],
            LAB / "evidence/hecate_role_result.json": protocol["pins"]["prior_result_sha256"],
            args.hecate_source: protocol["pins"]["hecate_source_sha256"],
            args.checkpoint: protocol["pins"]["checkpoint_sha256"],
        }
        for path, expected in checks.items():
            if sha(path) != expected:
                raise ValueError("Frozen source file hash mismatch: " + path.name)
        if args.checkpoint.stat().st_size != protocol["pins"]["checkpoint_bytes"]:
            raise ValueError("Checkpoint length mismatch")
        old = load_module(LAB / "hecate_probe.py", "context_intervention_prior")
        base = json.loads((LAB / "hecate_probe_protocol.json").read_text())
        manifest = json.loads((LAB / "hecate_donor_manifest.json").read_text())
        donor = next(d for d in manifest["donors"] if d["scroll"] == "PHerc0814")
        acquisition = json.loads((LAB / "evidence/hecate_role_source_acquisition.json").read_text())
        prior = json.loads((LAB / "evidence/hecate_role_result.json").read_text())
        cache = LocalVerifiedCache(args.cache, acquisition)
        reader = old.SurfaceReader(donor, cache)
        labels = old.read_categorical(donor, cache)
        geometry = old.read_categorical(donor, cache, geometry=True)
        selected = []
        seen_label_ids = set()
        pooled_ink, pooled_background = 0, 0
        for number in protocol["context_indices"]:
            context = donor["contexts"][number]
            if not (geometry[old.geometry_footprint(donor, context)] > 0).all():
                raise ValueError("Previously admitted geometry is now incomplete")
            slices, weights = zip(*old.context_geometry(donor, context, base))
            raw = old.resample_box(reader.read(slices), weights)[::-1].copy()
            yy, xx = old.label_indices(donor, context)
            state = labels[np.ix_(yy, xx)].copy()
            support = raw.any(axis=0)[16:48, 16:48]
            counts = metrics(np.zeros((32, 32)), state, support)
            label_ids = (yy[:, None] * donor["label_shape_yx"][1] + xx[None]).ravel()
            known_ids = label_ids[(support & (state != 0)).ravel()]
            if len(set(map(int, known_ids))) != len(known_ids) or seen_label_ids.intersection(map(int, known_ids)):
                raise ValueError("Physical source label cells overlap; do not inflate pooled counts")
            seen_label_ids.update(map(int, known_ids))
            pooled_ink += counts["ink"]
            pooled_background += counts["background"]
            expected = protocol["expected_label_counts"][context["id"]]
            if [counts["ink"], counts["background"]] != expected:
                raise ValueError("Selected control labels differ from prior recorded counts")
            feature_name = f"features/test-{number:02d}-reverse.npz"
            feature = args.prior_run / feature_name
            if sha(feature) != prior["new_feature_sha256"][feature_name]:
                raise ValueError("Prior baseline artifact changed")
            with np.load(feature, allow_pickle=False) as values:
                if not np.array_equal(values["support"], support):
                    raise ValueError("Reconstructed original support differs from prior")
                prior_logits = values["baseline"].copy()
            selected.append((context["id"], raw, state, support, prior_logits))
        if (pooled_ink, pooled_background, len(seen_label_ids)) != (1554, 3566, 5120):
            raise ValueError("Pooled physical label counts differ from the fixed prior support")
        cache.verify_unchanged()
        write_json(out / "source_receipt.json", {
            "pinned_acquisition_sha256": protocol["pins"]["prior_acquisition_sha256"],
            "used_source_assets": list(cache.used.values()), "network_requests": 0,
            "new_source_acquisition": False, "previous_source_bytes_redecoded": True,
            "interpretation": "Re-read previously acquired public source bytes; no new acquisition. Prior input cells are reconstructed exactly.",
        })
        stage = "load_frozen_model"
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        hecate = load_module(args.hecate_source, "context_intervention_hecate")
        load_started = time.monotonic()
        model = hecate.load_model(args.checkpoint, "cpu")
        load_seconds = time.monotonic() - load_started
        if any(layer.training for layer in model.modules()) or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Released model is not fully frozen in eval mode")
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        pooled = {condition: {"probabilities": [], "labels": [], "support": []} for condition in CONDITIONS}
        packet = {"central_ct_mean": [], "labels": [], "support": [], "logits": [], "probabilities": [], "context_ids": []}
        def timed_out(_signum, _frame):
            raise TimeoutError("Frozen wall-time cap reached")
        previous_handler = signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, max(.001, protocol["maximum_wall_seconds"] - (time.monotonic() - start)))
        try:
            for context_id, raw, state, support, prior_logits in selected:
                one = {"context_id": context_id, "conditions": {}, "comparisons": {},
                       "central_ct_array_sha256": array_sha(raw[:, 16:48, 16:48]),
                       "labels_array_sha256": array_sha(state), "central_source_pixels": int(support.sum())}
                output, output_logits = {}, {}
                for condition in CONDITIONS:
                    stage = f"inference_{context_id}_{condition}"
                    if completed >= protocol["maximum_forwards"]:
                        raise RuntimeError("Frozen forward cap reached")
                    value = intervention(raw, condition)
                    np.save(out / f"{context_id}-{condition}-input.npy", value, allow_pickle=False)
                    input_tensor = torch.from_numpy((value.astype(np.float32) / np.float32(255))[None, None])
                    valid = torch.from_numpy(value.any(axis=0)[None, None, None]).expand(1, 1, 16, 64, 64)
                    t0 = time.monotonic()
                    with torch.inference_mode():
                        logits = model(input_tensor, valid)[0, 0, 16:48, 16:48].float().numpy().copy()
                    elapsed = time.monotonic() - t0
                    completed += 1
                    prob = sigmoid(logits)
                    output[condition] = prob
                    output_logits[condition] = logits
                    np.save(out / f"{context_id}-{condition}-logits.npy", logits, allow_pickle=False)
                    one["conditions"][condition] = {
                        "metrics": metrics(prob, state, support), "forward_seconds": elapsed,
                        "central_ct_unchanged": np.array_equal(value[:, 16:48, 16:48], raw[:, 16:48, 16:48]),
                        "central_ct_array_sha256": array_sha(value[:, 16:48, 16:48]),
                        "labels_array_sha256": array_sha(state),
                        "input_sha256": sha(out / f"{context_id}-{condition}-input.npy"),
                        "logits_sha256": sha(out / f"{context_id}-{condition}-logits.npy"),
                        "conservative_full_observed_context_pixels": int(support.sum()) if condition == "original" else 0,
                    }
                    if condition == "original":
                        drift = float(np.max(np.abs(logits - prior_logits)))
                        one["prior_baseline_max_abs_logit_difference"] = drift
                        if drift > protocol["prior_baseline_max_abs_logit_tolerance"]:
                            raise ValueError("Original forward failed fixed prior-baseline reproduction gate")
                    pooled[condition]["probabilities"].append(prob)
                    pooled[condition]["labels"].append(state)
                    pooled[condition]["support"].append(support)
                    write_json(out / "progress.json", {**signature, "status": "running", "stage": stage,
                               "forwards_completed": completed, "completed_contexts": len(records),
                               "wall_seconds": time.monotonic() - start, "last_context_partial": one})
                    print(json.dumps({"stage": stage, "forwards_completed": completed,
                                      "forward_seconds": round(elapsed, 3)}), flush=True)
                for condition in CONDITIONS[1:]:
                    one["comparisons"][condition + "_vs_original"] = paired_drift(output["original"], output[condition], support)
                records.append(one)
                packet["central_ct_mean"].append(raw[:, 16:48, 16:48].mean(axis=0).astype(np.float32))
                packet["labels"].append(state)
                packet["support"].append(support)
                packet["logits"].append(np.stack([output_logits[c] for c in CONDITIONS]))
                packet["probabilities"].append(np.stack([output[c] for c in CONDITIONS]))
                packet["context_ids"].append(context_id)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        unchanged = all(torch.equal(value, dict(model.named_buffers())[name]) for name, value in buffers.items())
        if not unchanged or completed != 15:
            raise ValueError("Incomplete forward count or mutated model buffers")
        cache.verify_unchanged()
        pooled_scores = {condition: metrics(np.stack(v["probabilities"]), np.stack(v["labels"]),
                                           np.stack(v["support"])) for condition, v in pooled.items()}
        np.savez_compressed(out / "review_arrays.npz", **{k: np.stack(v) for k, v in packet.items()},
                            conditions=np.asarray(CONDITIONS))
        result = {**signature, "status": "complete", "experiment": protocol["id"],
                  "contexts": records, "pooled_descriptive_metrics": pooled_scores,
                  "forwards_completed": completed, "model_load_seconds": load_seconds,
                  "wall_seconds": time.monotonic() - start,
                  "inference_seconds": sum(v["forward_seconds"] for r in records for v in r["conditions"].values()),
                  "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "model_buffers_unchanged": unchanged,
                  "review_arrays_sha256": sha(out / "review_arrays.npz"),
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                              "device": "cpu", "precision": "float32", "threads": 4, "batch_size": 1},
                  "interpretation": protocol["limitations"]}
        write_json(out / "result.json", result)
        print(json.dumps({"status": "complete", "forwards_completed": completed,
                          "wall_seconds": round(result["wall_seconds"], 3)}), flush=True)
        return result
    except BaseException as error:
        write_json(out / "failure.json", {**signature, "status": "incomplete", "stage": stage,
                   "error_type": type(error).__name__, "forwards_completed": completed,
                   "wall_seconds": time.monotonic() - start, "completed_contexts": records,
                   "interpretation": "Partial execution; no completed experiment or substituted contexts."})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--freeze-reference", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hecate-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    verify_freeze(args.freeze_reference, args.protocol)
    out = outside_git(args.output_dir)
    if out.exists():
        raise FileExistsError("Use a fresh output directory; no retries or overwrites")
    protocol = json.loads(args.protocol.read_text())
    if protocol["context_indices"] != [0, 3, 8, 9, 12] or protocol["conditions"] != list(CONDITIONS):
        raise ValueError("Unexpected frozen experiment design")
    out.mkdir(parents=True)
    execute(args, protocol, out)


if __name__ == "__main__":
    main()
