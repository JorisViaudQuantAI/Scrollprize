#!/usr/bin/env python3
"""Paired, public-development Hecate stitch test with a bounded cold start.

Every assembly uses the same 128 new CPU FP32 patch predictions. No new field
selection, model fitting, threshold transfer or unread inference is performed.
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
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = ("run.py", "protocol.json", "metrics.py", "test_metrics.py", "test_stitch.py", "README.md")
ARMS = ("central", "hann", "uniform")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def save_json(path, value):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    partial.replace(path)


def verify_freeze(reference, protocol):
    if not re.fullmatch(r"[0-9a-f]{40}", reference):
        raise ValueError("Full published freeze commit required")
    for name in FROZEN:
        path = HERE / name
        relative = path.relative_to(ROOT).as_posix()
        if path.read_bytes() != subprocess.check_output(["git", "show", f"{reference}:{relative}"], cwd=ROOT):
            raise ValueError("An experiment file differs from its public freeze")
    for relative, expected in protocol["dependency_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("A frozen source dependency changed")


def make_window(torch):
    """The released predictor's FP32 nonperiodic, normalized, floored Hann."""
    window = torch.hann_window(256, periodic=False)
    window = window[:, None] * window[None, :]
    return (window / window.max().clamp_min(torch.finfo(torch.float32).eps)).clamp_min(.001).numpy()


class Stitch:
    """Accumulate full patch probabilities in original row-major order."""
    def __init__(self, window):
        if window.shape != (256, 256) or window.dtype != np.float32:
            raise ValueError("Expected released FP32 Hann window")
        self.window = window
        self.central = np.full((1024, 1024), np.nan, dtype=np.float32)
        self.unmasked = np.full_like(self.central, np.nan)
        self.numerators = {key: np.zeros((1152, 1152), dtype=np.float32) for key in ("hann", "uniform")}
        self.denominators = {key: np.zeros((1152, 1152), dtype=np.float32) for key in ("hann", "uniform")}
        self.count = 0

    def add(self, index, score, support):
        if index != self.count or score.shape != (256, 256) or support.shape != score.shape:
            raise ValueError("Unexpected patch identity, order or shape")
        if score.dtype != np.float32 or support.dtype != bool or not np.isfinite(score).all():
            raise ValueError("Invalid prediction or mask")
        if score.min() < 0 or score.max() > 1:
            raise ValueError("Prediction outside probability range")
        y, x = index // 8 * 128, index % 8 * 128
        masked = score * support
        self.central[y:y+128, x:x+128] = masked[64:192, 64:192]
        self.unmasked[y:y+128, x:x+128] = score[64:192, 64:192]
        self.numerators["hann"][y:y+256, x:x+256] += masked * self.window
        self.denominators["hann"][y:y+256, x:x+256] += self.window
        self.numerators["uniform"][y:y+256, x:x+256] += masked
        self.denominators["uniform"][y:y+256, x:x+256] += np.float32(1)
        self.count += 1

    def finish(self):
        if self.count != 64 or not np.isfinite(self.central).all():
            raise ValueError("Incomplete patch grid")
        maps = {"central": self.central}
        for key in ("hann", "uniform"):
            if not (self.denominators[key] > 0).all():
                raise ValueError("Incomplete weighted source support")
            maps[key] = (self.numerators[key] / self.denominators[key])[64:1088, 64:1088].copy()
        return maps


def field_gate(metrics, metric_module):
    """Descriptive method-selection criterion, never an unread-scan gate."""
    decisions = {name: metric_module.compare(metrics[name]["central"], metrics[name]["hann"])
                 for name in ("float", "uint8")}
    return {"pass": all(value["pass"] for value in decisions.values()),
            "representations": decisions,
            "scope": "Larger published known-glyph replication only; no unread inference"}


def run(args):
    protocol = json.loads((HERE / "protocol.json").read_text())
    verify_freeze(args.freeze_reference, protocol)
    native = module(ROOT / "firstletters-colab/native_resolution/run.py", "blend_native_helpers")
    helper = module(ROOT / "firstletters-colab/transversal_lab/hecate_probe.py", "blend_source_helpers")
    metric_module = module(HERE / "metrics.py", "blend_metrics")
    out = native.outside_git(args.output_dir)
    if out.exists():
        raise FileExistsError("A fresh output directory is required; retain failed attempts")
    out.mkdir(parents=True)
    start = time.monotonic()
    limits = native.Limits(protocol, start, out)
    result = {"schema": "scrollprize.hecate-blending.v1", "status": "started", "fields": [],
              "freeze_reference": args.freeze_reference, "protocol_sha256": sha(HERE / "protocol.json"),
              "runner_sha256": sha(__file__), "training_steps": 0, "unread_CT_requests": 0,
              "letter_reading_claim": False, "old_native_gate_changed": False, "numeric_gate_pass": False}
    save_json(out / "start.json", result)
    stage, cache, code = "cold_acquisition", None, 1
    prior_alarm = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(native.StopExperiment("whole_run_time_cap_reached")))
    signal.setitimer(signal.ITIMER_REAL, protocol["limits"]["whole_run_wall_seconds"])
    try:
        # Empty prior-pairs is intentional and recorded: every asset is newly
        # acquired; no synthetic historical receipt or relaxed historical CLI.
        cache = native.PinnedCache(out / "source_cache", [], protocol, out / "source_receipt.json")
        checkpoint = protocol["fine_checkpoint"]
        weight_path = cache.get(checkpoint["url"], checkpoint["sha256"], checkpoint["bytes"])
        manifest = json.loads((ROOT / "firstletters-colab/transversal_lab/hecate_donor_manifest.json").read_text())
        donors = {donor["scroll"]: donor for donor in manifest["donors"]}
        selection = json.loads((ROOT / "firstletters-colab/native_resolution/selection.json").read_text())
        sources = []
        for entry in selection["results"]:
            limits.check()
            choice = entry["choice"]
            donor = donors[choice["scroll"]]
            label_source = helper.read_categorical(donor, cache)
            geometry = helper.read_categorical(donor, cache, geometry=True)
            for origin in choice["context_grid_8x8"]:
                if not (geometry[helper.geometry_footprint(donor, {"origin_target_yx": origin})] > 0).all():
                    raise ValueError("Incomplete existing field geometry")
            reader = helper.SurfaceReader(donor, cache)
            origin = np.asarray(choice["core_origin_target_yx"]) - 16
            slices, weights = zip(*native.fine_geometry(donor, origin, 288, helper.box_axis_weights))
            raw = native.resample_fine(reader.read(slices), weights)
            labels = native.categorical_core(donor, choice["core_origin_target_yx"], label_source)
            folder = out / choice["scroll"]
            folder.mkdir()
            np.save(folder / "fine_context_uint8.npy", raw, allow_pickle=False)
            np.save(folder / "labels.npy", labels, allow_pickle=False)
            binding = protocol["historical_field_identity"][choice["scroll"]]
            if sha(folder / "fine_context_uint8.npy") != binding["fine_context_sha256"] or sha(folder / "labels.npy") != binding["labels_sha256"]:
                raise ValueError("Reacquired input differs from the historical public development field")
            sources.append((choice, folder))
            print(json.dumps({"stage": "source_identity_verified", "field": choice["scroll"],
                              "forwards": limits.count, "new_body_bytes": cache.bytes}), flush=True)
            del raw, geometry, label_source, labels
        save_json(out / "pre_inference_inputs.json", {"historical_field_identity": protocol["historical_field_identity"],
                  "all_complete_field_hashes_match": True, "forwards_before_identity_check": limits.count,
                  "new_application_body_bytes": cache.bytes, "requests_including_redirects": cache.requests})
        stage = "model_loading"
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        hecate = module(ROOT / "firstletters-colab/ink/third_party/hecate/hecate.py", "blend_hecate")
        model = hecate.load_model(str(weight_path), "cpu")
        if model.patch_size != (64, 256, 256) or model.divisor != 200 or model.training:
            raise ValueError("Unexpected model geometry, input scaling or training state")
        before_buffers = native.buffer_hashes(model)
        window = make_window(torch)
        captured = {}
        hook = model.register_forward_hook(lambda _m, _args, output:
            captured.__setitem__("logits", output.detach().float().cpu().numpy()[0, 0].copy()))
        result["runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
                             "torch": torch.__version__, "device": "cpu", "precision": "fp32", "threads": 4,
                             "machine": platform.machine(), "cpu_affinity_count": len(os.sched_getaffinity(0))}
        result["window_array_sha256"] = native.array_sha(window)
        for choice, folder in sources:
            stage = "field_inference"
            field = np.load(folder / "fine_context_uint8.npy", mmap_mode="r", allow_pickle=False)
            labels = np.load(folder / "labels.npy", allow_pickle=False)
            support_fine = np.any(field, axis=0)[64:1088, 64:1088]
            support_all = support_fine.reshape(256, 4, 256, 4).all((1, 3))
            support_any = support_fine.reshape(256, 4, 256, 4).any((1, 3))
            np.save(folder / "common_support.npy", support_all, allow_pickle=False)
            patches = np.lib.format.open_memmap(folder / "patch_probabilities.npy", mode="w+", dtype=np.float32, shape=(64, 256, 256))
            logits_cache = np.lib.format.open_memmap(folder / "patch_logits.npy", mode="w+", dtype=np.float32, shape=(64, 256, 256))
            stitch = Stitch(window)
            timings = []
            for index in range(64):
                y, x = index // 8 * 128, index % 8 * 128
                raw = np.ascontiguousarray(field[:, y:y+256, x:x+256])
                valid = raw.any(0)
                score, elapsed = native.forward(model, raw, valid, "reverse", "fp32", limits, torch)
                patches[index] = score
                logits_cache[index] = captured.pop("logits")
                stitch.add(index, score, valid)
                timings.append(elapsed)
                save_json(out / "progress.json", {"stage": stage, "field": choice["scroll"], "contexts": index+1,
                          "forwards": limits.count, "wall_seconds": time.monotonic()-start})
                if index % 8 == 0 or index == 63:
                    patches.flush()
                    logits_cache.flush()
                    print(json.dumps({"stage": stage, "field": choice["scroll"], "contexts": index+1,
                                      "forwards": limits.count, "wall_seconds": time.monotonic()-start}), flush=True)
            maps = stitch.finish()
            metrics = {"float": {}, "uint8": {}}
            common = {}
            for arm, fine_map in maps.items():
                np.save(folder / (arm + "_fine.npy"), fine_map, allow_pickle=False)
                # The official export quantizes the fine map BEFORE downsampling.
                exported = np.rint(fine_map.clip(0, 1) * np.float32(255)).astype(np.uint8)
                np.save(folder / (arm + "_uint8.npy"), exported, allow_pickle=False)
                common[arm] = metric_module.common_grid(fine_map)
                quantized_common = metric_module.common_grid(fine_map, uint8_export=True)
                for representation, values in (("float", common[arm]), ("uint8", quantized_common)):
                    metrics[representation][arm] = metric_module.measure(values, labels, support_all, seam_period=32)
                    np.save(folder / f"{arm}_{representation}_common.npy", values, allow_pickle=False)
            historical = native.measure(native.reduce_scores(stitch.unmasked), labels, support_any)
            np.save(folder / "historical_unmasked_central.npy", stitch.unmasked, allow_pickle=False)
            row = {"scroll": choice["scroll"], "metrics": metrics, "gate": field_gate(metrics, metric_module),
                   "historical_unmasked_central_metric": historical,
                   "fully_supported_common_cells": int(support_all.sum()), "any_supported_common_cells": int(support_any.sum()),
                   "fine_support_holes": int((~support_fine).sum()), "forward_seconds": timings,
                   "fine_context_sha256": sha(folder / "fine_context_uint8.npy"), "labels_sha256": sha(folder / "labels.npy"),
                   "patch_probabilities_sha256": sha(folder / "patch_probabilities.npy"),
                   "output_sha256": {path.name: sha(path) for path in sorted(folder.glob("*.npy"))},
                   "core_mm": [2.4576, 2.4576], "full_context_mm": [2.7648, 2.7648]}
            save_json(folder / "result.json", row)
            result["fields"].append(row)
            del field, patches, logits_cache, stitch, maps, raw, score
        stage = "final_integrity"
        hook.remove()
        if native.buffer_hashes(model) != before_buffers:
            raise ValueError("Model buffers changed during inference")
        cache.verify_unchanged()
        result.update({"status": "complete", "numeric_gate_pass": all(row["gate"]["pass"] for row in result["fields"]),
                       "model_buffers_unchanged": True, "all_complete_field_hashes_match": True,
                       "complete_maps_require_visual_review": True})
        code = 0
    except Exception as error:
        result.update({"status": "resource_unassessable" if isinstance(error, native.StopExperiment) else "failed",
                       "stage": stage, "error_type": type(error).__name__,
                       "reason": str(error) if isinstance(error, native.StopExperiment) else "source_or_execution_validation_failed"})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_alarm)
        result.update({"forwards_completed": limits.count, "wall_seconds": time.monotonic()-start,
                       "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        if cache is not None:
            result.update({"new_application_body_bytes": cache.bytes, "requests_including_redirects": cache.requests,
                           "source_receipt_sha256": sha(out / "source_receipt.json") if (out / "source_receipt.json").exists() else None})
        save_json(out / "result.json", result)
    print(json.dumps({"status": result["status"], "forwards": limits.count, "numeric_gate_pass": result["numeric_gate_pass"],
                      "wall_seconds": result["wall_seconds"]}), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-reference", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
