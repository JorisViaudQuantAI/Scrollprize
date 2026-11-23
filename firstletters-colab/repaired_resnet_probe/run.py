#!/usr/bin/env python3
"""Four fixed FP32 passes on retained public CT; run only after a Git freeze."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "4"
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FROZEN_FILES = ("run.py", "protocol.json", "test_geometry.py", "README.md")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def outside(path):
    path = Path(path).resolve()
    if any(os.path.lexists(parent / ".git") for parent in (path, *path.parents)):
        raise ValueError("Raw outputs and model assets must remain outside Git")
    return path


def verify(path, expected, size=None):
    path = Path(path)
    if not path.is_file() or (size is not None and path.stat().st_size != size) or sha(path) != expected:
        raise ValueError("Pinned source identity or size mismatch")


def verify_freeze(reference):
    commit = subprocess.check_output(["git", "rev-parse", "--verify", reference + "^{commit}"], cwd=REPO, text=True).strip()
    for name in FROZEN_FILES:
        path = HERE / name
        original = subprocess.check_output(["git", "show", commit + ":" + str(path.relative_to(REPO))], cwd=REPO)
        if original != path.read_bytes():
            raise ValueError("Package differs from the supplied published freeze")
    return commit


def axis_overlap(source_length, target_length):
    """Exact 79/24 cell widths; returned integer weights each sum to 79."""
    if source_length % 2 or target_length % 2:
        raise ValueError("This fixed symmetric experiment requires even lengths")
    edges = 24 * (source_length // 2) + 79 * (np.arange(target_length + 1, dtype=np.int64) - target_length // 2)
    if edges[0] < 0 or edges[-1] > 24 * source_length:
        raise ValueError("Target cells leave actual source support")
    lo, hi = int(edges[0] // 24), int((edges[-1] + 23) // 24)
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        indices = np.arange(int(lower // 24), int((upper + 23) // 24), dtype=np.int64)
        weights = np.minimum(upper, 24 * (indices + 1)) - np.maximum(lower, 24 * indices)
        if np.any(weights <= 0) or int(weights.sum()) != 79:
            raise ValueError("Incomplete exact cell overlap")
        rows.append((indices - lo, weights))
    return slice(lo, hi), rows


def weighted_axis(values, rows, axis):
    moved = np.moveaxis(values, axis, 0)
    output = np.empty((len(rows), *moved.shape[1:]), dtype=np.int64)
    for j, (indices, weights) in enumerate(rows):
        output[j] = np.sum(moved[indices] * weights.reshape((-1,) + (1,) * (moved.ndim - 1)), axis=0, dtype=np.int64)
    return np.moveaxis(output, 0, axis)


def divide_nearest_even(numerator, denominator):
    quotient, remainder = np.divmod(numerator, denominator)
    increment = (2 * remainder > denominator) | ((2 * remainder == denominator) & ((quotient % 2) == 1))
    return quotient + increment


def resample_input(fine):
    if fine.shape != (64, 1152, 1152) or fine.dtype != np.uint8:
        raise ValueError("Unexpected retained fine input geometry")
    specs = [axis_overlap(n, m) for n, m in zip(fine.shape, (18, 256, 256))]
    values = fine[tuple(spec[0] for spec in specs)]
    for axis, (_, rows) in enumerate(specs):
        values = weighted_axis(values, rows, axis)
    values = divide_nearest_even(values, 79 ** 3)
    if values.shape != (18, 256, 256) or values.min() < 0 or values.max() > 255:
        raise ValueError("Invalid integrated uint8 input")
    return values.astype(np.uint8)


def label_centres(origin):
    j = np.arange(16, 48, dtype=np.int64)
    return (5 * (48 * (int(origin) + 128) + 79 * (2 * j + 1 - 64))) // 234


def read_labels(donor, cache_roots):
    key = hashlib.sha256(donor["label_url"].encode()).hexdigest()
    paths = [Path(root) / key for root in cache_roots]
    path = next((path for path in paths if path.is_file()), None)
    if path is None:
        raise ValueError("Original cached categorical label asset is absent; no fetch permitted")
    verify(path, donor["label_sha256"], donor["label_size_bytes"])
    if donor["label_format"] == "npy":
        values = np.load(path, allow_pickle=False)
    elif donor["label_format"] == "png":
        from PIL import Image
        with Image.open(path) as image:
            if image.mode not in ("L", "P"):
                raise ValueError("Categorical PNG must use direct indices")
            values = np.asarray(image).copy()
    else:
        raise ValueError("Unsupported categorical label format")
    if list(values.shape) != donor["label_shape_yx"] or not np.isin(values, (0, 1, 2)).all():
        raise ValueError("Categorical label schema changed")
    return values


def metrics(categories, probabilities):
    known = categories != 0
    y, scores = (categories[known] == 2), probabilities[known]
    positive, negative = int(y.sum()), int((~y).sum())
    result = {"known": int(known.sum()), "ink": positive, "background": negative,
              "unknown": int((~known).sum()), "ap": None, "auc": None,
              "recall_at_background_5pct": None, "background_exceedance": None,
              "threshold_strict_greater": None, "selected_ink": None, "selected_background": None}
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Invalid model probabilities")
    if positive and negative:
        order = np.argsort(scores, kind="stable")
        ordered, target = scores[order], y[order]
        ends = np.r_[np.flatnonzero(np.diff(ordered) != 0) + 1, len(ordered)]
        starts = np.r_[0, ends[:-1]]
        rank_sum = sum(float(target[a:b].sum()) * (a + 1 + b) / 2 for a, b in zip(starts, ends))
        result["auc"] = (rank_sum - positive * (positive + 1) / 2) / (positive * negative)
        order = np.argsort(-scores, kind="stable")
        ordered, target = scores[order], y[order]
        ends = np.r_[np.flatnonzero(np.diff(ordered) != 0) + 1, len(ordered)]
        true_positives = np.cumsum(target)[ends - 1]
        increments = np.diff(np.r_[0, true_positives])
        result["ap"] = float(np.sum((increments / positive) * (true_positives / ends)))
    if negative:
        threshold = float(np.sort(scores[~y])[::-1][math.floor(0.05 * negative)])
        selected = scores > threshold
        result.update(threshold_strict_greater=threshold,
                      selected_background=int(selected[~y].sum()),
                      selected_ink=int(selected[y].sum()),
                      background_exceedance=float(selected[~y].mean()))
        if positive:
            result["recall_at_background_5pct"] = float(selected[y].mean())
    result["status"] = "mixed" if positive and negative else "ink_only" if positive else "background_only" if negative else "no_known_labels"
    return result


class ResourceStop(RuntimeError):
    pass


class Guards:
    def __init__(self, protocol, started):
        self.rules, self.started = protocol["limits"], started
        self.forward_attempts = 0
        self.forward_seconds = []

    def check(self):
        if time.monotonic() - self.started >= self.rules["whole_run_seconds"]:
            raise ResourceStop("Whole-run wall guard")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > self.rules["RSS_soft_kib"]:
            raise ResourceStop("RSS soft guard")
        if sum(self.forward_seconds) > self.rules["cumulative_inference_seconds_soft"]:
            raise ResourceStop("Cumulative inference soft guard")

    def before_forward(self):
        self.check()
        if self.forward_attempts >= self.rules["maximum_forwards"]:
            raise ResourceStop("Forward count cap")
        self.forward_attempts += 1

    def after_saved_forward(self, seconds):
        self.forward_seconds.append(seconds)
        if seconds > self.rules["per_forward_seconds_soft"]:
            raise ResourceStop("Per-forward soft guard; completed output retained")
        self.check()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Assets:
    def __init__(self, directory, output, protocol, guards):
        self.directory, self.output = outside(directory), output
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rules, self.guards = protocol["limits"], guards
        self.bytes, self.requests, self.rows = 0, 0, []

    def receipt(self):
        save(self.output / "model_asset_receipt.json", {"assets": self.rows, "new_application_body_bytes": self.bytes,
             "requests_including_redirects": self.requests, "CT_requests": 0, "wire_bytes_measured": False})

    def get(self, name, specification):
        self.guards.check()
        path = self.directory / name
        expected = specification["bytes"]
        if path.exists():
            verify(path, specification["sha256"], expected)
            self.rows.append({"name": name, "bytes": expected, "sha256": specification["sha256"], "new_body_bytes": 0, "cached": True})
            self.receipt()
            return path
        if self.bytes + expected + 1 > self.rules["maximum_new_application_body_bytes"]:
            raise ResourceStop("Application-body cap would be exceeded")
        current, response = specification["url"], None
        temporary = path.with_suffix(path.suffix + ".partial")
        try:
            for hop in range(self.rules["maximum_HTTP_requests_per_asset"]):
                parsed = urllib.parse.urlsplit(current)
                if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                    raise ValueError("Invalid model asset URL")
                if self.requests >= self.rules["maximum_requests_including_redirects"]:
                    raise ResourceStop("HTTP request cap")
                self.requests += 1
                try:
                    response = urllib.request.build_opener(NoRedirect).open(
                        urllib.request.Request(current, headers={"Accept-Encoding": "identity", "User-Agent": "Scrollprize-public-ResNet-probe/1"}),
                        timeout=self.rules["HTTP_timeout_seconds"])
                except urllib.error.HTTPError as error:
                    response = error
                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    response.close(); response = None
                    if not location or hop + 1 == self.rules["maximum_HTTP_requests_per_asset"]:
                        raise ValueError("Invalid or excessive model redirect")
                    current = urllib.parse.urljoin(current, location)
                    continue
                if response.status != 200:
                    raise ValueError("Pinned model asset missing or refused")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise ValueError("Transformed model asset response")
                declared = response.headers.get("Content-Length")
                if declared is not None and (not declared.isdecimal() or int(declared) != expected):
                    raise ValueError("Model asset declared size mismatch")
                break
            if response is None:
                raise ValueError("No model asset response")
            received = 0
            with temporary.open("wb") as stream:
                while received < expected + 1:
                    self.guards.check()
                    part = response.read(min(1 << 20, expected + 1 - received))
                    if not part:
                        break
                    stream.write(part)
                    received += len(part); self.bytes += len(part)
            verify(temporary, specification["sha256"], expected)
            temporary.replace(path)
            self.rows.append({"name": name, "bytes": expected, "sha256": specification["sha256"], "new_body_bytes": received, "cached": False})
            self.receipt()
            print(json.dumps({"stage": "model_asset_ready", "name": name, "new_body_bytes": received}), flush=True)
            return path
        finally:
            if response is not None:
                response.close()
            self.receipt()


def module(path):
    spec = importlib.util.spec_from_file_location("pinned_repaired_resnet_common", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def buffer_hashes(model):
    return {name: hashlib.sha256(buffer.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, buffer in model.named_buffers()}


def render_panel(field_directory, scroll, labels, scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), constrained_layout=True)
    extent = (0, 1.0112, 1.0112, 0)
    axes[0].imshow(labels, cmap=ListedColormap(["#b8b8b8", "white", "black"]), norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3), interpolation="nearest", extent=extent)
    axes[0].set_title("Categories: unknown gray\nbackground white / ink black", fontsize=10)
    for axis, order in zip(axes[1:], ("source", "reverse")):
        image = axis.imshow(scores[order], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest", extent=extent)
        axis.set_title(order + " depth order\nfixed probability scale 0–1", fontsize=10)
    for axis in axes:
        axis.set_xlabel("mm"); axis.set_ylabel("mm")
    fig.colorbar(image, ax=axes[1:], fraction=.025, ticks=[0, .5, 1], label="Ink probability")
    fig.suptitle(scroll + " — complete fixed 1.0112 mm central core", fontsize=12)
    fig.savefig(field_directory / "core_review.png", dpi=160)
    plt.close(fig)


def run(args):
    started = time.monotonic()
    output = outside(args.output)
    if output.exists():
        raise ValueError("Output directory already exists")
    output.mkdir(parents=True)
    protocol = json.loads((HERE / "protocol.json").read_text())
    guards = Guards(protocol, started)
    result = {"schema": protocol["schema"], "status": "started", "fields": [], "training_steps": 0,
              "promotion_rule_present": False, "unread_acquisition": False, "reading_claim": False,
              "protocol_sha256": sha(HERE / "protocol.json"), "runner_sha256": sha(__file__),
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "interpretation": protocol["interpretation"]}
    assets = None
    old_alarm = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(ResourceStop("Whole-run alarm")))
    signal.setitimer(signal.ITIMER_REAL, protocol["limits"]["whole_run_seconds"])
    try:
        result["freeze_reference"] = verify_freeze(args.freeze_reference)
        # Fail before acquiring a checkpoint if the execution environment is incomplete.
        for dependency in ("torch", "safetensors", "PIL", "matplotlib"):
            if importlib.util.find_spec(dependency) is None:
                raise ValueError("Required dependency unavailable before acquisition: " + dependency)
        sources = protocol["source_inputs"]
        for relative, expected in ((sources["selection_relative_path"], sources["selection_sha256"]),
                                   (sources["donor_manifest_relative_path"], sources["donor_manifest_sha256"])):
            verify((HERE / relative).resolve(), expected)
        manifest = json.loads((HERE / sources["donor_manifest_relative_path"]).read_text())
        prepared = []
        for field in sources["fields"]:
            guards.check()
            directory = output / field["scroll"]; directory.mkdir()
            fine_path = Path(args.fine_context_root) / field["fine_context_relative_path"]
            verify(fine_path, field["fine_context_sha256"], sources["npy_bytes"])
            fine = np.load(fine_path, mmap_mode="r", allow_pickle=False)
            model_input = resample_input(fine)
            np.save(directory / "input_uint8.npy", model_input, allow_pickle=False)
            del fine
            donor = next(d for d in manifest["donors"] if d["id"] == field["donor_id"])
            original_labels = read_labels(donor, args.label_cache)
            yy, xx = (label_centres(origin) for origin in field["original_coarse_core_origin_yx"])
            if yy.min() < 0 or xx.min() < 0 or yy.max() >= original_labels.shape[0] or xx.max() >= original_labels.shape[1]:
                raise ValueError("Original categorical cell centres out of bounds")
            labels = original_labels[np.ix_(yy, xx)].copy()
            np.save(directory / "labels_core.npy", labels, allow_pickle=False)
            row = {"scroll": field["scroll"], "source_fine_context_sha256": field["fine_context_sha256"],
                   "input_sha256": sha(directory / "input_uint8.npy"), "original_label_sha256": donor["label_sha256"],
                   "labels_core_sha256": sha(directory / "labels_core.npy"),
                   "input_zero_depth_column_count": int((~model_input.any(axis=0)).sum()),
                   "label_counts": {str(i): int((labels == i).sum()) for i in (0, 1, 2)}, "orders": {}}
            result["fields"].append(row)
            prepared.append((row, directory, model_input, labels))
        save(output / "pre_forward_inputs.json", result)
        assets = Assets(args.asset_dir or output / "assets", output, protocol, guards)
        for name, specification in protocol["model_assets"].items():
            assets.get(name, specification)
        guards.check()
        import torch
        import safetensors
        torch.set_num_threads(protocol["model_execution"]["torch_threads"])
        torch.set_num_interop_threads(protocol["model_execution"]["torch_interop_threads"])
        common = module(assets.directory / "common.py")
        model, config = common.load_resnet_model(str(assets.directory), ckpt_path=str(assets.directory / "model.safetensors"), strict=True)
        for name, expected in protocol["model_execution"]["expected_config"].items():
            if config.get(name) != expected:
                raise ValueError("Pinned model config has unexpected input geometry")
        model = model.cpu().float().eval()
        if any(layer.training for layer in model.modules()):
            raise ValueError("Model remains in training mode")
        initial_buffers = buffer_hashes(model)
        result["model_parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        result["versions"] = {"torch": torch.__version__, "numpy": np.__version__, "safetensors": safetensors.__version__}
        for row, directory, model_input, labels in prepared:
            for order in protocol["sampling"]["orders"]:
                value = model_input if order == "source" else model_input[::-1]
                normalized = np.ascontiguousarray(np.minimum(value, 200).astype(np.float32) / np.float32(255.0))
                tensor = torch.from_numpy(normalized[None, None])
                guards.before_forward()
                before = time.monotonic()
                with torch.inference_mode():
                    logits = model(tensor)
                    if tuple(logits.shape) != (1, 1, 64, 64) or logits.dtype != torch.float32:
                        raise ValueError("Unexpected model logit shape or precision")
                    probabilities = logits.sigmoid()[0, 0].cpu().numpy().copy()
                seconds = time.monotonic() - before
                if not np.isfinite(probabilities).all():
                    raise ValueError("Nonfinite complete model output")
                np.save(directory / (order + "_probabilities.npy"), probabilities, allow_pickle=False)
                core = probabilities[16:48, 16:48]
                row["orders"][order] = {"metrics": metrics(labels, core), "forward_seconds": seconds,
                    "probabilities_sha256": sha(directory / (order + "_probabilities.npy")),
                    "normalized_input_sha256": hashlib.sha256(normalized.tobytes()).hexdigest()}
                result["forwards_attempted"] = guards.forward_attempts
                save(output / "result.json", result)
                print(json.dumps({"stage": "public_control_forward", "scroll": row["scroll"], "order": order,
                                  "forwards_completed": sum(len(f["orders"]) for f in result["fields"]), "seconds": seconds}), flush=True)
                if buffer_hashes(model) != initial_buffers:
                    raise ValueError("Running-normalization buffers changed")
                guards.after_saved_forward(seconds)
                del logits, tensor
        result["running_normalization_buffers_unchanged"] = buffer_hashes(model) == initial_buffers
        del model
        for row, directory, _, labels in prepared:
            guards.check()
            scores = {order: np.load(directory / (order + "_probabilities.npy"), allow_pickle=False)[16:48, 16:48]
                      for order in protocol["sampling"]["orders"]}
            render_panel(directory, row["scroll"], labels, scores)
            row["panel_sha256"] = sha(directory / "core_review.png")
        if guards.forward_attempts != 4:
            raise ValueError("Expected exactly four completed model passes")
        result["status"] = "complete"
    except ResourceStop as error:
        result.update(status="resource_unassessable", failure=str(error))
    except Exception as error:
        result.update(status="failed", failure_type=type(error).__name__, failure=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        result.update(forwards_attempted=guards.forward_attempts, completed_forwards=sum(len(f["orders"]) for f in result["fields"]),
                      cumulative_inference_seconds=sum(guards.forward_seconds), wall_seconds=time.monotonic() - started,
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      new_application_body_bytes=assets.bytes if assets else 0, CT_requests=0,
                      requests_including_redirects=assets.requests if assets else 0)
        save(output / "result.json", result)
    print(json.dumps({"status": result["status"], "completed_forwards": result["completed_forwards"],
                      "wall_seconds": result["wall_seconds"], "result_sha256": sha(output / "result.json")}), flush=True)
    return 0 if result["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-reference", required=True)
    parser.add_argument("--fine-context-root", type=Path, required=True)
    parser.add_argument("--label-cache", type=Path, action="append", required=True)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
