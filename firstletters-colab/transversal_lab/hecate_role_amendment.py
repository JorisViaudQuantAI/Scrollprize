#!/usr/bin/env python3
"""Prospective support-only role amendment after an unassessable Hecate run.

Fit on the existing PHerc0841 caches; calibrate FPR on existing PHerc0009B
background; keep PHerc0814 locked until all heads/thresholds are frozen.
The failed original experiment and all its files remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import sys
import time

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "4"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def import_pinned(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LocalLabels:
    def __init__(self, root):
        self.root = root

    def get(self, url, expected_sha256=None, expected_bytes=None):
        path = self.root/hashlib.sha256(url.encode()).hexdigest()
        if not path.is_file() or sha(path) != expected_sha256 or path.stat().st_size != expected_bytes:
            raise ValueError("Previously acquired label bytes do not match the frozen manifest")
        return path


def prior_extraction(donor, prior_run):
    # Keep original prefixes. PHerc0841 is calibration-XX; PHerc0009B is train-XX.
    retained = []
    for number, context in enumerate(donor["contexts"]):
        paths = {order: str(prior_run/"features"/f"{donor['split']}-{number:02d}-{order}.npz")
                 for order in ("source", "reverse")}
        retained.append({"context": context, "paths": paths})
    return {"donor": donor, "retained": retained, "excluded": []}


def verify_prior(args, protocol):
    checks = {
        args.original_runner: protocol["original_runner_sha256"],
        args.original_protocol: protocol["original_protocol_sha256"],
        args.manifest: protocol["original_manifest_sha256"],
    }
    for name, digest in protocol["prior_receipt_sha256"].items():
        checks[args.prior_run/name] = digest
    for name, digest in protocol["prior_feature_sha256"].items():
        checks[args.prior_run/name] = digest
    for path, expected in checks.items():
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Frozen prior artifact mismatch: {path.name}")
    if list((args.prior_run/"features").glob("test-*.npz")) or (args.prior_run/"readout_freeze.json").exists():
        raise ValueError("Prior run unexpectedly contains a fitted freeze or held-out features")
    prior_bytes = sum(path.stat().st_size for path in args.prior_run.rglob("*") if path.is_file())
    if prior_bytes != protocol["prior_run_directory_bytes"]:
        raise ValueError("Prior run directory bytes changed; preserve its complete failed attempt")
    acquisition = json.loads((args.prior_run/"source_acquisition.json").read_text())
    if acquisition["transferred_body_bytes"] != protocol["prior_source_bytes"] or acquisition["failed_assets"]:
        raise ValueError("Unexpected prior acquisition totals or failures")
    return prior_bytes


def execute(args, protocol, prior_bytes, start_wall, start_cpu):
    import numpy as np
    import torch
    original = import_pinned(args.original_runner, "frozen_original_hecate_probe")
    base = json.loads(args.original_protocol.read_text())
    manifest = json.loads(args.manifest.read_text())
    if sha(args.hecate_source) != base["model"]["source_sha256"]:
        raise ValueError("Hecate source hash mismatch")
    if (sha(args.checkpoint) != base["model"]["checkpoint_sha256"]
            or args.checkpoint.stat().st_size != base["model"]["checkpoint_bytes"]):
        raise ValueError("Hecate checkpoint bytes mismatch")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/"features").mkdir()
    # This in-memory budget copy is the separately published amendment, not an
    # edit to the original protocol. Original extraction/scoring functions stay exact.
    amended_runtime = json.loads(json.dumps(base))
    amended_runtime["budgets"]["maximum_cpu_seconds"] = protocol["budgets"]["maximum_new_cpu_seconds"]
    fixed_existing_bytes = prior_bytes+args.checkpoint.stat().st_size
    limits = original.Limits(amended_runtime, out, fixed_existing_bytes, start_cpu,
                             forward_count=protocol["prior_forwards"])
    cache = original.DownloadCache(out/"source-cache",
        protocol["budgets"]["cumulative_source_byte_cap"]-protocol["prior_source_bytes"])
    local = LocalLabels(args.prior_run/"source-cache")
    old_donors = {donor["split"]: donor for donor in manifest["donors"]}
    role_map = protocol["logical_roles_to_original_splits"]
    extracted, data = {}, {}
    provenance = {
        "freeze_reference": args.freeze_reference,
        "amendment_runner_sha256": sha(Path(__file__)), "amendment_protocol_sha256": sha(args.protocol),
        "original_runner_sha256": sha(args.original_runner), "original_protocol_sha256": sha(args.original_protocol),
        "manifest_sha256": sha(args.manifest), "prior_failure_sha256": sha(args.prior_run/"failure.json"),
        "roles": role_map, "prior_v1_status": "unassessable training-support failure; no head fit",
        "prior_cpu_seconds_charged": protocol["budgets"]["prior_cpu_seconds_charged"],
        "new_ct_or_label_access_before_readout_freeze": False,
        "checkpoint_sha256": base["model"]["checkpoint_sha256"],
        "source_sha256": base["model"]["source_sha256"],
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
    }
    original.save_json(out/"start_receipt.json", provenance)
    for role in ("train", "calibration"):
        donor = old_donors[role_map[role]["original_split"]]
        if donor["scroll"] != role_map[role]["scroll"]:
            raise ValueError("Amended physical donor role mismatch")
        extracted[role] = prior_extraction(donor, args.prior_run)
        labels = original.read_categorical(donor, local)
        data[role] = {order: original.assemble(extracted[role], labels, order)
                      for order in base["depth_orders"]}
        for field in ("labels", "label_id", "context_index"):
            if not np.array_equal(data[role]["source"][field], data[role]["reverse"][field]):
                raise ValueError("Cached depth orders disagree on the frozen scored pixels")
        counts = [int((data[role]["source"]["labels"] == value).sum()) for value in (0, 1)]
        if counts != protocol["expected_labeled_counts_background_ink"][role]:
            raise ValueError("Already observed support counts do not reproduce exactly")
    heads, thresholds, calibration_scores = {}, {}, {}
    for train_order in base["depth_orders"]:
        heads[train_order] = original.fit_heads(data["train"][train_order], base)
        if heads[train_order]["selected_per_class"] != 2712:
            raise ValueError("Unexpected class-balanced training sample size")
        limits.check()
        thresholds[train_order], calibration_scores[train_order] = {}, {}
        for calibration_order in base["depth_orders"]:
            calibration = data["calibration"][calibration_order]
            scores = original.score_heads(calibration, heads[train_order])
            calibration_scores[train_order][calibration_order] = scores
            thresholds[train_order][calibration_order] = {
                arm: original.calibration_threshold(calibration["labels"], score, 0.01)
                for arm, score in scores.items()}
    if heads["source"]["selected_label_ids_sha256"] != heads["reverse"]["selected_label_ids_sha256"]:
        raise ValueError("Depth orders did not use identical selected training labels")
    freeze = {**provenance, "heads": original.json_plain(heads), "thresholds": thresholds,
        "all_four_head_order_calibration_order_threshold_sets_frozen": True,
        "held_out_814_ct_and_labels_not_accessed": True,
        "new_cpu_seconds_at_freeze": time.process_time()-start_cpu,
        "new_wall_seconds_at_freeze": time.monotonic()-start_wall,
        "no_head_retry_or_score_directed_selection": True}
    freeze_path = out/"readout_freeze.json"
    original.save_json(freeze_path, freeze)
    freeze_sha = sha(freeze_path)
    original.event("role_amendment_readouts_frozen_before_any_814_access", sha256=freeze_sha)
    training_diagnostics = {
        order: original.evaluate(data["train"][order],
            original.score_heads(data["train"][order], heads[order]),
            thresholds[order][order], extracted["train"])
        for order in base["depth_orders"]}
    # The first held-out CT or label access can only occur below this point.
    hecate = import_pinned(args.hecate_source, "pinned_hecate_role_amendment")
    model = hecate.load_model(args.checkpoint, "cpu")
    if any(layer.training for layer in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Backbone is not fully frozen in eval mode")
    buffers = {name: value.clone() for name, value in model.named_buffers()
               if "running_" in name or "num_batches_tracked" in name}
    extracted["test"] = original.extract_donor(old_donors["test"], amended_runtime, model, cache, limits)
    test_labels = original.read_categorical(old_donors["test"], cache)
    data["test"] = {order: original.assemble(extracted["test"], test_labels, order)
                    for order in base["depth_orders"]}
    for field in ("labels", "label_id", "context_index"):
        if not np.array_equal(data["test"]["source"][field], data["test"]["reverse"][field]):
            raise ValueError("Held-out depth orders do not share identical scored labels")
    combinations, primary, original_reversed = [], None, None
    for train_order, calibration_order, test_order in itertools.product(base["depth_orders"], repeat=3):
        frozen_threshold = thresholds[train_order][calibration_order]
        test_scores = original.score_heads(data["test"][test_order], heads[train_order])
        entry = {"training_scroll": "PHerc0841", "calibration_scroll": "PHerc0009B", "test_scroll": "PHerc0814",
            "training_order": train_order, "calibration_order": calibration_order, "test_order": test_order,
            "thresholds_frozen_before_test": frozen_threshold,
            "arm_validity": {arm: heads[train_order][arm]["valid"] for arm in ("baseline", "depth", "features")},
            "baseline_note": "Independent of training order; repeated baseline rows are duplicates, not replications.",
            "calibration": original.evaluate(data["calibration"][calibration_order],
                calibration_scores[train_order][calibration_order], frozen_threshold, extracted["calibration"]),
            "test": original.evaluate(data["test"][test_order], test_scores, frozen_threshold, extracted["test"])}
        combination = (train_order, calibration_order, test_order)
        entry["interpretation"] = "symmetric sensitivity; no best combination selected"
        if combination == ("source", "source", "source"):
            entry["interpretation"] = "amended experiment primary source/source/source comparison"
            primary = entry
        elif combination == ("reverse", "reverse", "reverse"):
            entry["interpretation"] = "amended experiment all-reversed sensitivity"
            original_reversed = entry
        combinations.append(entry)
    if sha(freeze_path) != freeze_sha or not all(torch.equal(value, dict(model.named_buffers())[name]) for name, value in buffers.items()):
        raise ValueError("Frozen heads/thresholds or running normalization buffers changed")
    if primary is None or original_reversed is None or len(combinations) != 8:
        raise ValueError("Incomplete orientation table")
    # Recheck every original cache so this amendment cannot silently replace v1.
    verify_prior(args, protocol)
    limits.check()
    record = {"experiment": "separately published support-only role amendment, not a successful v1",
        "provenance": provenance, "readout_freeze_sha256": freeze_sha,
        "roles": role_map, "support_counts_background_ink": protocol["expected_labeled_counts_background_ink"],
        "new_forwards": limits.forward_count-protocol["prior_forwards"], "cumulative_forwards": limits.forward_count,
        "prior_source_bytes": protocol["prior_source_bytes"], "new_source_bytes": cache.bytes,
        "cumulative_source_bytes": protocol["prior_source_bytes"]+cache.bytes,
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "fixed_checkpoint_plus_v1_directory_bytes": fixed_existing_bytes,
        "new_cpu_seconds": time.process_time()-start_cpu, "new_wall_seconds": time.monotonic()-start_wall,
        "prior_cpu_seconds_charged": protocol["budgets"]["prior_cpu_seconds_charged"],
        "cumulative_cpu_charge_seconds": protocol["budgets"]["prior_cpu_seconds_charged"]+time.process_time()-start_cpu,
        "running_normalization_buffers_unchanged": True,
        "new_source_assets": list(cache.records.values()),
        "new_feature_sha256": {str(path.relative_to(out)): sha(path) for path in sorted((out/"features").glob("*.npz"))},
        "prior_feature_sha256": protocol["prior_feature_sha256"],
        "held_out_exclusions": extracted["test"]["excluded"],
        "training_diagnostics": training_diagnostics,
        "training_diagnostics_interpretation": "In-sample fitting-parent diagnostics on all originally scored PHerc0841 pixels, using each training order's matching calibration-order threshold; not independent validation.",
        "primary": primary, "all_reversed_sensitivity": original_reversed,
        "all_eight_combinations": combinations, "limitations": protocol["limitations"],
        "no_unread_data_access_or_prize_claim": True}
    original.save_json(out/"result.json", original.json_plain(record))
    limits.check()
    original.event("role_amendment_complete", result=str(out/"result.json"),
        cumulative_forwards=limits.forward_count, new_cpu_seconds=record["new_cpu_seconds"],
        cumulative_source_bytes=record["cumulative_source_bytes"])


def main():
    # Include integrity verification and numerical-library imports in new-run time.
    start_wall, start_cpu = time.monotonic(), time.process_time()
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--protocol", type=Path, default=folder/"hecate_role_protocol.json")
    parser.add_argument("--original-runner", type=Path, default=folder/"hecate_probe.py")
    parser.add_argument("--original-protocol", type=Path, default=folder/"hecate_probe_protocol.json")
    parser.add_argument("--manifest", type=Path, default=folder/"hecate_donor_manifest.json")
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hecate-source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze-reference")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    prior_bytes = verify_prior(args, protocol)
    if args.action == "preflight":
        print(json.dumps({"status": "prior code, receipts and64 cache hashes verified", "fits": 0,
            "model_forwards": 0, "decoded_labels": False, "expected_training_pixels_per_class": 2712,
            "negative_calibration_pixels": 4650, "maximum_new_forwards": 32,
            "prior_run_bytes": prior_bytes}))
        return
    if any(getattr(args, name) is None for name in ("checkpoint", "hecate_source", "out", "freeze_reference")):
        parser.error("run needs --checkpoint, --hecate-source, --out and --freeze-reference")
    if re.fullmatch(r"[0-9a-f]{40}", args.freeze_reference) is None:
        parser.error("--freeze-reference must be a complete published commit hash")
    repo = folder.parents[1]
    output = args.out.resolve()
    prior = args.prior_run.resolve()
    if output.exists() or output == repo or repo in output.parents or output == prior or prior in output.parents:
        parser.error("Choose a new output directory outside both the repository and preserved prior run")
    try:
        execute(args, protocol, prior_bytes, start_wall, start_cpu)
    except BaseException as error:
        message = re.sub(r"https?://[^\s]+", "[URL redacted]", str(error))
        failure = {"status": "role_amendment_failed_without_retry", "error_type": type(error).__name__,
            "error_message": message, "protocol_sha256": sha(args.protocol),
            "runner_sha256": sha(Path(__file__)), "freeze_reference": args.freeze_reference,
            "original_v1_remains_unassessable": True}
        if output.is_dir():
            (output/"failure.json").write_text(json.dumps(failure, indent=2)+"\n")
        print(json.dumps(failure), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
