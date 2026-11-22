#!/usr/bin/env python3
"""Prospective orientation sensitivity supplement; reuse frozen Hecate caches.

No network, model inference, feature fitting, context selection or parameter
search occurs here. All eight order combinations must be reported together.
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


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


class LocalAssetsOnly:
    """Interface expected by the frozen reader, with deliberately no HTTP path."""
    def __init__(self, directory):
        self.directory = directory

    def get(self, url, expected_sha256=None, expected_bytes=None):
        path = self.directory/hashlib.sha256(url.encode()).hexdigest()
        if not path.is_file():
            raise ValueError("Original acquired asset missing; supplement does not download replacements")
        if expected_sha256 is None or digest(path) != expected_sha256:
            raise ValueError("Original local label asset hash mismatch")
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise ValueError("Original local label asset byte count mismatch")
        return path


def original_extraction(donor, result, run_dir):
    excluded = {item["id"] for item in result["exclusions"][donor["split"]]}
    retained = []
    for number, context in enumerate(donor["contexts"]):
        if context["id"] in excluded:
            continue
        paths = {order: str(run_dir/"features"/f"{donor['split']}-{number:02d}-{order}.npz")
                 for order in ("source", "reverse")}
        if not all(Path(path).is_file() for path in paths.values()):
            raise ValueError("Original feature cache missing; no replacement extraction allowed")
        retained.append({"context": context, "paths": paths})
    return {"donor": donor, "retained": retained, "excluded": result["exclusions"][donor["split"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("hecate_orientation_protocol.json"))
    parser.add_argument("--original-runner", type=Path, default=Path(__file__).with_name("hecate_probe.py"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("hecate_donor_manifest.json"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze-reference")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if digest(args.original_runner) != protocol["original_runner_sha256"]:
        raise ValueError("Original runner differs from the prospectively frozen version")
    if digest(args.manifest) != protocol["original_manifest_sha256"]:
        raise ValueError("Original donor manifest differs from frozen bytes")
    if args.action == "preflight":
        print(json.dumps({"status": "JSON and code hash checks passed", "order_combinations": 8,
                          "experimental_array_access": False, "model_forwards": 0, "fits": 0}))
        return
    if args.run_dir is None or args.out is None or re.fullmatch(r"[0-9a-f]{40}", args.freeze_reference or "") is None:
        parser.error("run requires --run-dir, --out and a full forty-character --freeze-reference")
    repo = Path(__file__).resolve().parents[2]
    output = args.out.resolve()
    if output.exists() or repo == output or repo in output.parents:
        parser.error("Choose a new output JSON outside the repository")
    start, cpu = time.monotonic(), time.process_time()
    spec = importlib.util.spec_from_file_location("frozen_hecate_probe_v1", args.original_runner)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    import numpy as np
    manifest = json.loads(args.manifest.read_text())
    result_path = args.run_dir/"result.json"
    freeze_path = args.run_dir/"readout_freeze.json"
    result = json.loads(result_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if (result["provenance"]["runner_sha256"] != protocol["original_runner_sha256"]
            or result["provenance"]["protocol_sha256"] != protocol["original_protocol_sha256"]
            or result["provenance"]["manifest_sha256"] != protocol["original_manifest_sha256"]
            or result["readout_freeze_sha256"] != digest(freeze_path)
            or freeze["test_label_array_not_decoded"] is not True):
        raise ValueError("Original execution/freeze provenance does not match the supplement")
    donors = {donor["split"]: donor for donor in manifest["donors"]}
    cache = LocalAssetsOnly(args.run_dir/"source-cache")
    extracted, data = {}, {}
    for split in ("calibration", "test"):
        extracted[split] = original_extraction(donors[split], result, args.run_dir)
        labels = original.read_categorical(donors[split], cache)
        data[split] = {order: original.assemble(extracted[split], labels, order)
                       for order in ("source", "reverse")}
        for field in ("label_id", "labels", "context_index"):
            if not np.array_equal(data[split]["source"][field], data[split]["reverse"][field]):
                raise ValueError("Depth orders do not retain identical scored label pixels")
    combinations = []
    for train_order, calibration_order, test_order in itertools.product(("source", "reverse"), repeat=3):
        heads = freeze["heads"][train_order]
        calibration = data["calibration"][calibration_order]
        testing = data["test"][test_order]
        calibration_scores = original.score_heads(calibration, heads)
        test_scores = original.score_heads(testing, heads)
        thresholds = {arm: original.calibration_threshold(calibration["labels"], score, 0.01)
                      for arm, score in calibration_scores.items()}
        calibration_metrics = original.evaluate(calibration, calibration_scores, thresholds, extracted["calibration"])
        test_metrics = original.evaluate(testing, test_scores, thresholds, extracted["test"])
        annotation = "complete fixed sensitivity table; no preferred result selected"
        if (train_order, calibration_order, test_order) == ("source", "source", "source"):
            annotation = "unchanged original primary comparison"
            if test_metrics != result["results"]["source"]["test"]:
                raise ValueError("Primary source/source/source metrics do not exactly reproduce the original result")
        elif (train_order, calibration_order, test_order) == ("reverse", "reverse", "reverse"):
            annotation = "unchanged original all-reversed sensitivity; not an independently verified physical face"
            if test_metrics != result["results"]["reverse"]["test"]:
                raise ValueError("All-reversed metrics do not exactly reproduce the original sensitivity")
        combinations.append({"training_order": train_order, "calibration_order": calibration_order,
            "test_order": test_order, "annotation": annotation, "thresholds": thresholds,
            "baseline_note": "Baseline does not depend on training order; its two training-order rows are duplicates.",
            "arm_status": {arm: {"valid": heads[arm]["valid"],
                                   "reason_if_invalid": None if heads[arm]["valid"] else "original frozen fit invalid; not retried"}
                           for arm in ("baseline", "depth", "features")},
            "calibration": calibration_metrics, "test": test_metrics})
    if len(combinations) != 8 or digest(freeze_path) != result["readout_freeze_sha256"]:
        raise ValueError("Incomplete combination table or changed frozen heads")
    feature_hashes = {str(Path(path).relative_to(args.run_dir)): digest(Path(path))
        for value in extracted.values() for item in value["retained"] for path in item["paths"].values()}
    record = {"purpose": protocol["purpose"], "supplement_freeze_reference": args.freeze_reference,
        "supplement_protocol_sha256": digest(args.protocol), "supplement_runner_sha256": digest(Path(__file__)),
        "original_result_sha256": digest(result_path), "original_readout_freeze_sha256": digest(freeze_path),
        "original_runner_sha256": digest(args.original_runner), "original_manifest_sha256": digest(args.manifest),
        "model_forwards": 0, "new_fits": 0, "new_downloads": 0, "new_contexts": 0,
        "primary_exactly_reproduced": True, "original_all_reversed_exactly_reproduced": True,
        "same_scored_label_pixels_across_orders": True, "reused_feature_sha256": feature_hashes,
        "wall_seconds": time.monotonic()-start, "cpu_seconds": time.process_time()-cpu,
        "all_eight_combinations": combinations, "limitations": protocol["limitations"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    original.save_json(output, record)
    print(json.dumps({"status": "complete", "combinations": 8, "new_fits": 0, "model_forwards": 0,
                      "result": str(output), "cpu_seconds": record["cpu_seconds"]}))


if __name__ == "__main__":
    main()
