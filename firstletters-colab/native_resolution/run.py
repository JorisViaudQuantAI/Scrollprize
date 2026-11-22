#!/usr/bin/env python3
"""Frozen native-fine / degraded-fine / coarse Hecate comparison.

Public label-selected development controls only. Detailed arrays stay outside Git.
No training, target search, adaptive fields, or per-image contrast normalization.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
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
import urllib.error
import urllib.parse
import urllib.request

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "4"

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent / "transversal_lab"
ARMS = ("A_native_fine", "B_degraded_fine", "C_coarse")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def outside_git(path):
    path = Path(path).resolve()
    if any(os.path.lexists(p / ".git") for p in (path, *path.parents)):
        raise ValueError("Detailed outputs must stay outside every Git worktree")
    return path


def verify_freeze(reference):
    if not re.fullmatch(r"[0-9a-f]{40}", reference):
        raise ValueError("A full published freeze commit is required")
    repo = HERE.parents[1]
    for name in ("run.py", "protocol.json", "selection.json"):
        path = HERE / name
        rel = path.relative_to(repo).as_posix()
        if path.read_bytes() != subprocess.check_output(["git", "show", f"{reference}:{rel}"], cwd=repo):
            raise ValueError("Runner, protocol or selection differs from the freeze")


class StopExperiment(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class PinnedCache:
    """Reuse only hash-pinned old bytes; acquire missing public objects once.

    Redirects count as requests. Signed redirect URLs remain in memory only.
    The limit is application-read response bodies, not network wire traffic.
    """
    def __init__(self, root, prior_pairs, protocol, receipt_path):
        self.root, self.receipt_path = Path(root), Path(receipt_path)
        self.root.mkdir()
        self.maximum_bytes = protocol["limits"]["new_body_bytes"]
        self.maximum_requests = protocol["limits"]["requests_including_redirects"]
        self.allowed_receipts = set(protocol["pins"]["prior_receipt_sha256"])
        self.old, self.records, self.failures = {}, {}, []
        self.bytes, self.reserved, self.requests = 0, 0, 0
        self.lock = threading.Lock()
        for receipt_path, cache_path in prior_pairs:
            if sha(receipt_path) not in self.allowed_receipts:
                raise ValueError("Prior acquisition receipt is not pinned")
            receipt = json.loads(Path(receipt_path).read_text())
            for row in receipt["completed_assets"]:
                item = dict(row, path=Path(cache_path) / hashlib.sha256(row["url"].encode()).hexdigest())
                if row["url"] in self.old:
                    if any(self.old[row["url"]][k] != row[k] for k in ("bytes", "sha256")):
                        raise ValueError("Conflicting old source identities")
                else:
                    self.old[row["url"]] = item

    def _save(self):
        save_json(self.receipt_path, {
            "new_application_body_bytes": self.bytes, "requests_including_redirects": self.requests,
            "maximum_new_body_bytes": self.maximum_bytes, "maximum_requests": self.maximum_requests,
            "wire_bytes_measured": False,
            "completed_assets": list(self.records.values()), "failed_assets": self.failures,
            "source_version_limit": "New chunk bytes are individually hash-recorded; mutable bucket objects do not supply an atomic volume snapshot.",
        })

    def get(self, url, expected_sha256=None, expected_bytes=None):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.query or parsed.username or parsed.password:
            raise ValueError("Canonical sources must be credential-free HTTPS URLs")
        key = hashlib.sha256(url.encode()).hexdigest()
        old = self.old.get(url)
        with self.lock:
            recorded = self.records.get(url)
        if recorded is not None:
            path = old["path"] if recorded["reused_old_cache"] else self.root / key
            if sha(path) != recorded["sha256"]:
                raise ValueError("Already used source changed")
        elif old is not None:
            path = old["path"]
            if not path.is_file() or path.stat().st_size != old["bytes"] or sha(path) != old["sha256"]:
                raise ValueError("Pinned old source is absent or changed")
            recorded = {k: old[k] for k in ("url", "bytes", "sha256")}
            recorded["reused_old_cache"] = True
            with self.lock:
                self.records[url] = recorded
                self._save()
        else:
            path = self.root / key
            partial = path.with_suffix(".partial")
            count, digest = 0, hashlib.sha256()
            response = None
            try:
                current = url
                opener = urllib.request.build_opener(NoRedirect)
                for hop in range(4):
                    with self.lock:
                        if self.requests >= self.maximum_requests:
                            raise StopExperiment("request_budget_exhausted")
                        self.requests += 1
                    p = urllib.parse.urlsplit(current)
                    if p.scheme != "https" or not p.hostname or p.username or p.password:
                        raise ValueError("Nonpublic redirect rejected")
                    req = urllib.request.Request(current, headers={"Accept-Encoding": "identity",
                        "User-Agent": "Scrollprize-native-resolution-control/1"})
                    try:
                        response = opener.open(req, timeout=40)
                    except urllib.error.HTTPError as error:
                        response = error
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.headers.get("Location")
                        response.close()
                        response = None
                        if hop == 3 or not location:
                            raise ValueError("Redirect limit or missing location")
                        current = urllib.parse.urljoin(current, location)
                        continue
                    if response.status != 200:
                        raise ValueError("Public object response is not HTTP 200")
                    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                        raise ValueError("Transformed HTTP representation rejected")
                    break
                if response is None:
                    raise ValueError("No object response")
                length = response.headers.get("Content-Length")
                declared = int(length) if length is not None and length.isdecimal() else None
                if expected_bytes is not None and declared is not None and declared != expected_bytes:
                    raise ValueError("Declared object length differs")
                with partial.open("xb") as output:
                    while True:
                        with self.lock:
                            allowance = min(1 << 20, self.maximum_bytes - self.bytes - self.reserved)
                            if allowance <= 0:
                                raise StopExperiment("body_budget_exhausted")
                            self.reserved += allowance
                        block = b""
                        try:
                            block = response.read(allowance)
                        finally:
                            with self.lock:
                                self.reserved -= allowance
                                self.bytes += len(block)
                        if not block:
                            break
                        count += len(block)
                        if expected_bytes is not None and count > expected_bytes:
                            raise ValueError("Object exceeds expected bytes")
                        output.write(block)
                        digest.update(block)
                if declared is not None and count != declared:
                    raise ValueError("Exposed HTTP body is truncated")
                if expected_bytes is not None and count != expected_bytes:
                    raise ValueError("Object length mismatch")
                if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
                    raise ValueError("Object hash mismatch")
                partial.rename(path)
                recorded = {"url": url, "bytes": count, "sha256": digest.hexdigest(), "reused_old_cache": False}
                with self.lock:
                    self.records[url] = recorded
                    self._save()
            except Exception as error:
                with self.lock:
                    self.failures.append({"url": url, "received_bytes": count, "error_type": type(error).__name__})
                    self._save()
                raise StopExperiment("source_acquisition_failed:" + type(error).__name__) from None
            finally:
                if response is not None:
                    response.close()
        if expected_sha256 is not None and recorded["sha256"] != expected_sha256:
            raise ValueError("Source and expected hashes differ")
        if expected_bytes is not None and recorded["bytes"] != expected_bytes:
            raise ValueError("Source and expected lengths differ")
        return path

    def verify_unchanged(self):
        for url, row in self.records.items():
            path = self.old[url]["path"] if row["reused_old_cache"] else self.root / hashlib.sha256(url.encode()).hexdigest()
            if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
                raise ValueError("A used source asset changed")


def fine_geometry(donor, origin_yx, side_coarse, box_axis_weights):
    """Native source voxel footprints for a 2.4um grid at a coarse-grid origin."""
    if side_coarse not in (64, 288) or len(origin_yx) != 2 or min(origin_yx) < 0:
        raise ValueError("Unexpected fixed field geometry")
    ratio = 2.4 / float(donor["source_pitch_um"])
    y, x = map(int, origin_yx)
    crop = donor["crop_xyxy"]
    edges = (
        31.5 + (np.arange(65, dtype=np.float64) - 32) * ratio,
        crop[1] + (4*y + np.arange(4*side_coarse+1)) * ratio - .5,
        crop[0] + (4*x + np.arange(4*side_coarse+1)) * ratio - .5,
    )
    if edges[1][-1] > crop[3]-.5+1e-8 or edges[2][-1] > crop[2]-.5+1e-8:
        raise ValueError("Fine context leaves documented crop")
    return tuple(box_axis_weights(e, int(n)) for e, n in zip(edges, donor["source_shape_zyx"]))


def sparse_axis(values, weights, axis):
    """Apply small-support box weights without a dense field-sized contraction."""
    rows = [np.flatnonzero(row) for row in weights]
    if any(not len(row) for row in rows):
        raise ValueError("Empty resampling footprint")
    shape = list(values.shape)
    shape[axis] = weights.shape[0]
    out = np.zeros(shape, dtype=np.float32)
    broadcast = [1] * values.ndim
    broadcast[axis] = weights.shape[0]
    for slot in range(max(map(len, rows))):
        indices = np.array([row[slot] if slot < len(row) else 0 for row in rows])
        coeff = np.array([weights[i, row[slot]] if slot < len(row) else 0
                          for i, row in enumerate(rows)], dtype=np.float32)
        out += np.take(values, indices, axis=axis) * coeff.reshape(broadcast)
    return out


def resample_fine(raw, weights):
    if np.asarray(raw).dtype != np.uint8:
        raise ValueError("Fine source must be uint8")
    value = raw
    for axis in (2, 1, 0):
        value = sparse_axis(value, weights[axis], axis)
    if not np.isfinite(value).all() or value.min() < -1e-3 or value.max() > 255.001:
        raise ValueError("Fine resampling produced invalid values")
    return np.rint(value).astype(np.uint8)


def degrade(raw):
    if raw.dtype != np.uint8 or raw.shape != (64, 256, 256):
        raise ValueError("Expected native-fine model patch")
    coarse = np.rint(raw.reshape(16, 4, 64, 4, 64, 4).mean((1, 3, 5), dtype=np.float32)).astype(np.uint8)
    expanded = coarse.repeat(4, 0).repeat(4, 1).repeat(4, 2)
    fine_valid = raw.any(0)
    coarse_valid = fine_valid.reshape(64, 4, 64, 4).any((1, 3))
    return expanded, coarse, fine_valid, coarse_valid


def reduce_scores(fine):
    fine = np.asarray(fine)
    if fine.ndim != 2 or any(n % 4 for n in fine.shape):
        raise ValueError("Fine scores need dimensions divisible by four")
    h, w = fine.shape
    return fine.reshape(h//4, 4, w//4, 4).mean((1, 3), dtype=np.float64).astype(np.float32)


def categorical_core(donor, origin_yx, labels, size=256):
    y, x = map(int, origin_yx)
    yy = ((2*(y+np.arange(size, dtype=np.int64))+1)*20)//39
    xx = ((2*(x+np.arange(size, dtype=np.int64))+1)*20)//39
    if yy.max() >= labels.shape[0] or xx.max() >= labels.shape[1]:
        raise ValueError("Labels outside their source grid")
    return labels[np.ix_(yy, xx)].copy()


def measure(score, labels, support):
    score, labels, support = np.asarray(score), np.asarray(labels), np.asarray(support)
    if score.shape != labels.shape or score.shape != support.shape or support.dtype != bool:
        raise ValueError("Score/label/support grid mismatch")
    if not set(np.unique(labels)).issubset({0, 1, 2}) or not np.isfinite(score[support]).all():
        raise ValueError("Invalid labels or scores")
    known = support & (labels != 0)
    truth, values = labels[known] == 2, score[known]
    pos, neg = int(truth.sum()), int((~truth).sum())
    result = {"known": int(known.sum()), "ink": pos, "background": neg,
              "unknown_or_unsupported": int(known.size-known.sum()),
              "ap": None, "auc": None, "recall_at_background_5pct": None,
              "background_exceedance": None, "threshold_strict_greater": None,
              "operating_point": "Descriptive within-region ROC point; labels set this threshold, not a transferable detector threshold."}
    if pos and neg:
        order = np.argsort(-values, kind="stable")
        sorted_values, sorted_truth = values[order], truth[order]
        ends = np.r_[np.flatnonzero(np.diff(sorted_values)), len(values)-1]
        tp = np.cumsum(sorted_truth)[ends]
        fp = ends+1-tp
        recall, precision = tp/pos, tp/(tp+fp)
        result["ap"] = float(np.sum(np.diff(np.r_[0., recall])*precision))
        result["auc"] = float(np.trapezoid(np.r_[0., recall], np.r_[0., fp/neg]))
        budget = int(math.floor(.05*neg))
        threshold = float(np.sort(values[~truth])[::-1][budget])
        positive = values > threshold
        result.update({"recall_at_background_5pct": float(np.mean(positive[truth])),
                       "background_exceedance": float(np.mean(positive[~truth])),
                       "threshold_strict_greater": threshold,
                       "allowed_background_count": budget,
                       "background_positive_count": int(positive[~truth].sum()),
                       "ink_positive_count": int(positive[truth].sum())})
    return result


def field_metrics(maps, labels, support):
    regions = {"field": (slice(None), slice(None)),
               "q00": (slice(0, 128), slice(0, 128)),
               "q01": (slice(0, 128), slice(128, 256)),
               "q10": (slice(128, 256), slice(0, 128)),
               "q11": (slice(128, 256), slice(128, 256))}
    return {name: {arm: measure(maps[arm][region], labels[region], support[region])
                   for arm in ARMS} for name, region in regions.items()}


def gate(metrics):
    quadrants = ("q00", "q01", "q10", "q11")
    if any(metrics[q][a]["recall_at_background_5pct"] is None or metrics[q][a]["ap"] is None
           for q in quadrants for a in ARMS):
        return {"pass": False, "reason": "missing_class_or_support", "macro_recall": None}
    macro = {a: float(np.mean([metrics[q][a]["recall_at_background_5pct"] for q in quadrants])) for a in ARMS}
    a, b, c = (macro[k] for k in ARMS)
    if 1-c < .05 or 1-b <= 0:
        return {"pass": False, "reason": "baseline_ceiling_prevents_meaningful_gain", "macro_recall": macro}
    relative_c, relative_b = (a-c)/(1-c), (a-b)/(1-b)
    ap_nonworse = sum(metrics[q][ARMS[0]]["ap"] >= metrics[q][ARMS[2]]["ap"] for q in quadrants)
    passed = relative_c >= .20 and a-c >= .05 and relative_b >= .10 and ap_nonworse >= 3
    return {"pass": bool(passed), "reason": "all_prespecified_gates_pass" if passed else "prespecified_gain_gate_failed",
            "macro_recall": macro, "relative_miss_reduction_vs_C": relative_c,
            "absolute_macro_recall_gain_vs_C": a-c, "relative_miss_reduction_vs_B": relative_b,
            "quadrants_AP_nonworse_vs_C": ap_nonworse,
            "not_statistical_significance_or_letter_reading": True}


def rectangles_overlap(first, second, first_size, second_size, guard=0):
    return all(first[i]-guard < second[i]+second_size and second[i] < first[i]+first_size+guard for i in (0, 1))


class Limits:
    def __init__(self, protocol, start, out):
        self.protocol, self.start, self.out = protocol, start, out
        self.count = 0

    def check(self):
        limits = self.protocol["limits"]
        if time.monotonic()-self.start >= limits["whole_run_wall_seconds"]:
            raise StopExperiment("whole_run_time_cap_reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > limits["peak_rss_soft_kib"]:
            raise StopExperiment("process_RSS_soft_cap_reached")
        if self.count >= limits["maximum_forwards"]:
            raise StopExperiment("forward_cap_reached")


def buffer_hashes(model):
    return {name: array_sha(value.detach().cpu().numpy()) for name, value in model.named_buffers()}


def forward(model, raw, valid_xy, order, precision, limits, torch):
    limits.check()
    value = raw if order == "source" else raw[::-1].copy()
    image = torch.from_numpy((value.astype(np.float32)/np.float32(model.divisor))[None, None])
    valid = torch.from_numpy(np.ascontiguousarray(valid_xy)[None, None, None]).expand(1, 1, raw.shape[0], *raw.shape[1:])
    start = time.monotonic()
    autocast = torch.autocast("cpu", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    with torch.inference_mode(), autocast:
        logits = model(image, valid)
        score = logits.sigmoid().float()[0, 0].numpy().copy()
    limits.count += 1
    elapsed = time.monotonic()-start
    if not np.isfinite(score).all() or score.min() < 0 or score.max() > 1:
        raise ValueError("Nonfinite or out-of-range model scores")
    # Stop on a post-forward violation before accepting a larger experiment.
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > limits.protocol["limits"]["peak_rss_soft_kib"]:
        raise StopExperiment("process_RSS_soft_cap_reached_after_forward")
    return score, elapsed


def run(args):
    verify_freeze(args.freeze_reference)
    protocol = json.loads((HERE/"protocol.json").read_text())
    selection = json.loads((HERE/"selection.json").read_text())
    if sha(HERE/"selection.json") != protocol["pins"]["selection_sha256"]:
        raise ValueError("Selection changed")
    out = outside_git(args.output_dir)
    if out.exists():
        raise FileExistsError("Use a fresh output directory; failures must be retained")
    out.mkdir(parents=True)
    started = time.monotonic()
    limits = Limits(protocol, started, out)
    result = {"schema": "scrollprize.native-resolution-result.v1", "freeze_reference": args.freeze_reference,
              "runner_sha256": sha(__file__), "protocol_sha256": sha(HERE/"protocol.json"),
              "selection_sha256": sha(HERE/"selection.json"), "training_steps": 0,
              "new_unread_region_access": False, "fields": [], "status": "started"}
    save_json(out/"start.json", result)
    stage, cache = "pins", None
    code = 1
    old_alarm = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(StopExperiment("whole_run_time_cap_reached")))
    signal.setitimer(signal.ITIMER_REAL, protocol["limits"]["whole_run_wall_seconds"])
    try:
        for path, expected in ((LAB/"hecate_probe.py", protocol["pins"]["probe_sha256"]),
                               (LAB/"hecate_donor_manifest.json", protocol["pins"]["donor_manifest_sha256"]),
                               (args.hecate_source, protocol["pins"]["hecate_source_sha256"]),
                               (args.coarse_checkpoint, protocol["coarse_checkpoint"]["sha256"])):
            if sha(path) != expected:
                raise ValueError("Pinned code, manifest or checkpoint changed")
        helper = load_module(LAB/"hecate_probe.py", "native_resolution_probe")
        manifest = json.loads((LAB/"hecate_donor_manifest.json").read_text())
        donors = {d["scroll"]: d for d in manifest["donors"]}
        pilot_donor = donors["PHerc0814"]
        pilot_context = next(c for c in pilot_donor["contexts"] if c["id"] == "814-c03")
        for entry in selection["results"]:
            choice = entry["choice"]
            if choice["scroll"] == "PHerc0814":
                origin = [x-16 for x in choice["core_origin_target_yx"]]
                if rectangles_overlap(pilot_context["origin_target_yx"], origin, 64, 288, guard=1):
                    raise ValueError("Pilot context overlaps a field or its guard")
        cache = PinnedCache(out/"source_cache", args.prior_cache, protocol, out/"source_receipt.json")
        stage = "fine_checkpoint"
        checkpoint = protocol["fine_checkpoint"]
        fine_path = cache.get(checkpoint["url"], checkpoint["sha256"], checkpoint["bytes"])
        stage = "pilot_source"
        pilot_reader = helper.SurfaceReader(pilot_donor, cache)
        pilot_labels = helper.read_categorical(pilot_donor, cache)
        pilot_geometry = helper.read_categorical(pilot_donor, cache, geometry=True)
        if not (pilot_geometry[helper.geometry_footprint(pilot_donor, pilot_context)] > 0).all():
            raise ValueError("Pilot source geometry incomplete")
        slices, weights = zip(*fine_geometry(pilot_donor, pilot_context["origin_target_yx"], 64, helper.box_axis_weights))
        pilot = resample_fine(pilot_reader.read(slices), weights)
        _, pilot_coarse, valid_fine, valid_coarse = degrade(pilot)
        pilot_state = categorical_core(pilot_donor, np.asarray(pilot_context["origin_target_yx"])+16, pilot_labels, 32)
        support = valid_coarse[16:48, 16:48]
        np.savez_compressed(out/"pilot_inputs.npz", fine=pilot, coarse=pilot_coarse,
                            fine_valid=valid_fine, coarse_valid=valid_coarse, labels=pilot_state)
        stage = "model_load"
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        hecate = load_module(args.hecate_source, "native_resolution_hecate")
        fine_model = hecate.load_model(str(fine_path), "cpu")
        coarse_model = hecate.load_model(str(args.coarse_checkpoint), "cpu")
        if fine_model.patch_size != (64, 256, 256) or coarse_model.patch_size != (16, 64, 64):
            raise ValueError("Unexpected checkpoint geometry")
        buffers = {"fine": buffer_hashes(fine_model), "coarse": buffer_hashes(coarse_model)}
        result["runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
                             "torch": torch.__version__, "device": "cpu", "threads": 4}
        stage = "pilot_fp32"
        pilots, pilot_maps = {}, {}
        for model_name, model, raw, valid in (("fine", fine_model, pilot, valid_fine),
                                             ("coarse", coarse_model, pilot_coarse, valid_coarse)):
            pilots[model_name] = {}
            for order in ("source", "reverse"):
                score, seconds = forward(model, raw, valid, order, "fp32", limits, torch)
                common = reduce_scores(score[64:192, 64:192]) if model_name == "fine" else score[16:48, 16:48]
                metric = measure(common, pilot_state, support)
                if metric["auc"] is None:
                    raise ValueError("Calibration c03 lacks mixed supported classes")
                pilot_maps[model_name, order, "fp32"] = score
                pilots[model_name][order] = {"seconds": seconds, "metrics": metric}
                print(json.dumps({"stage": stage, "model": model_name, "order": order,
                                  "seconds": seconds, "forwards": limits.count}), flush=True)
        orders = {name: ("source" if values["source"]["metrics"]["auc"] > values["reverse"]["metrics"]["auc"] else "reverse")
                  for name, values in pilots.items()}
        fine_mean = float(np.mean([v["seconds"] for v in pilots["fine"].values()]))
        fine_fp32_mean = fine_mean
        coarse_mean = float(np.mean([v["seconds"] for v in pilots["coarse"].values()]))
        projected_fp32 = 256*fine_mean + 128*coarse_mean
        precision, parity, bf16_mean = "fp32", [], None
        if projected_fp32 > protocol["pilot"]["bf16_trigger_projected_seconds"]:
            stage = "pilot_bf16"
            for order in ("source", "reverse"):
                score, seconds = forward(fine_model, pilot, valid_fine, order, "bf16", limits, torch)
                baseline = pilot_maps["fine", order, "fp32"]
                common = reduce_scores(score[64:192, 64:192])
                metric = measure(common, pilot_state, support)
                delta = float(np.max(np.abs(score.astype(np.float64)-baseline.astype(np.float64))))
                ap_diff = abs(metric["ap"]-pilots["fine"][order]["metrics"]["ap"])
                parity.append({"order": order, "seconds": seconds, "max_abs_probability_difference": delta,
                               "absolute_AP_difference": ap_diff,
                               "pass": delta <= .005 and ap_diff <= .01})
                pilot_maps["fine", order, "bf16"] = score
            bf16_mean = float(np.mean([row["seconds"] for row in parity]))
            if all(row["pass"] for row in parity) and bf16_mean < fine_fp32_mean:
                precision = "bf16"
                fine_mean = bf16_mean
        projected = 256*fine_mean+128*coarse_mean
        decision = {"orders": orders, "fine_precision": precision, "coarse_precision": "fp32",
                    "pilot_fp32": pilots, "bf16_parity": parity,
                    "fine_fp32_mean_seconds": fine_fp32_mean, "fine_bf16_mean_seconds": bf16_mean,
                    "bf16_faster_than_fp32": bf16_mean < fine_fp32_mean if bf16_mean is not None else None,
                    "projected_field_fp32_seconds": projected_fp32,
                    "projected_selected_field_seconds": projected,
                    "selection_uses_old_calibration_c03_only": True}
        save_json(out/"pilot_decision.json", decision)
        np.savez_compressed(out/"pilot_scores.npz", **{"_".join(key): value for key, value in pilot_maps.items()})
        result["pilot_decision"] = decision
        if projected > protocol["limits"]["whole_run_wall_seconds"]-(time.monotonic()-started):
            raise StopExperiment("selected_backend_cannot_fit_remaining_time")
        # Acquire and materialise each complete field before inspecting any field output.
        stage = "field_sources"
        sources = []
        for entry in selection["results"]:
            choice = entry["choice"]
            donor = donors[choice["scroll"]]
            labels = helper.read_categorical(donor, cache)
            geometry = helper.read_categorical(donor, cache, geometry=True)
            origin = np.asarray(choice["core_origin_target_yx"])-16
            for context_origin in choice["context_grid_8x8"]:
                context = {"origin_target_yx": context_origin}
                if not (geometry[helper.geometry_footprint(donor, context)] > 0).all():
                    raise ValueError("Field geometry incomplete")
            reader = helper.SurfaceReader(donor, cache)
            slices, weights = zip(*fine_geometry(donor, origin, 288, helper.box_axis_weights))
            raw = resample_fine(reader.read(slices), weights)
            state = categorical_core(donor, choice["core_origin_target_yx"], labels)
            counts = {"known": int((state != 0).sum()), "ink": int((state == 2).sum()), "background": int((state == 1).sum())}
            if any(counts[k] != choice["labels"][k] for k in counts):
                raise ValueError("Selected field label counts changed")
            folder = out/choice["scroll"]
            folder.mkdir()
            np.save(folder/"fine_context_uint8.npy", raw, allow_pickle=False)
            np.save(folder/"labels.npy", state, allow_pickle=False)
            sources.append((choice, folder, state))
            del raw
            limits.check()
        if projected > protocol["limits"]["whole_run_wall_seconds"]-(time.monotonic()-started):
            raise StopExperiment("source_acquisition_left_insufficient_inference_time")
        save_json(out/"pre_field_freeze.json", {"pilot_decision_sha256": sha(out/"pilot_decision.json"),
                  "field_inputs": [{"scroll": c["scroll"], "fine_sha256": sha(f/"fine_context_uint8.npy"),
                                    "labels_sha256": sha(f/"labels.npy")} for c, f, _ in sources],
                  "forwards_before_fields": limits.count, "no_field_scores_seen": True})
        for choice, folder, state in sources:
            stage = "field_inference"
            raw_field = np.load(folder/"fine_context_uint8.npy", mmap_mode="r", allow_pickle=False)
            maps = {arm: np.lib.format.open_memmap(folder/(arm+"_common.npy"), mode="w+",
                    dtype=np.float32, shape=(256, 256)) for arm in ARMS}
            fine_maps = {arm: np.lib.format.open_memmap(folder/(arm+"_fine.npy"), mode="w+",
                         dtype=np.float32, shape=(1024, 1024)) for arm in ARMS[:2]}
            for value in (*maps.values(), *fine_maps.values()):
                value[:] = np.nan
                value.flush()
            common_support = np.lib.format.open_memmap(folder/"common_support.npy", mode="w+",
                             dtype=bool, shape=(256, 256))
            common_support[:] = False
            common_support.flush()
            timings = {arm: [] for arm in ARMS}
            for index in range(64):
                y, x = (index//8)*32, (index%8)*32
                a = np.ascontiguousarray(raw_field[:, 4*y:4*y+256, 4*x:4*x+256])
                b, c, fine_valid, coarse_valid = degrade(a)
                region = (slice(y, y+32), slice(x, x+32))
                common_support[region] = coarse_valid[16:48, 16:48]
                for arm, model, value, valid, order, dtype in (
                    (ARMS[0], fine_model, a, fine_valid, orders["fine"], precision),
                    (ARMS[1], fine_model, b, fine_valid, orders["fine"], precision),
                    (ARMS[2], coarse_model, c, coarse_valid, orders["coarse"], "fp32")):
                    score, elapsed = forward(model, value, valid, order, dtype, limits, torch)
                    timings[arm].append(elapsed)
                    if arm in ARMS[:2]:
                        core = score[64:192, 64:192]
                        fine_maps[arm][4*y:4*y+128, 4*x:4*x+128] = core
                        maps[arm][region] = reduce_scores(core)
                    else:
                        maps[arm][region] = score[16:48, 16:48]
                save_json(out/"progress.json", {"stage": stage, "field": choice["scroll"],
                          "completed_contexts_in_field": index+1, "forwards": limits.count,
                          "wall_seconds": time.monotonic()-started})
                if index % 8 == 0 or index == 63:
                    for value in (*maps.values(), *fine_maps.values(), common_support):
                        value.flush()
                    print(json.dumps({"stage": stage, "field": choice["scroll"], "contexts": index+1,
                                      "forwards": limits.count, "wall_seconds": time.monotonic()-started}), flush=True)
            if not all(np.isfinite(v).all() for v in (*maps.values(), *fine_maps.values())):
                raise ValueError("Incomplete field map")
            np.savez_compressed(folder/"scores.npz", **maps, **{k+"_fine_grid": v for k, v in fine_maps.items()},
                                labels=state, support=common_support)
            scores = field_metrics(maps, state, common_support)
            field_result = {"scroll": choice["scroll"], "metrics": scores, "gate": gate(scores),
                            "forward_seconds": timings, "scores_sha256": sha(folder/"scores.npz"),
                            "fine_context_sha256": sha(folder/"fine_context_uint8.npy"),
                            "labels_sha256": sha(folder/"labels.npy"),
                            "scored_support_pixels": int(common_support.sum()),
                            "nominal_core_mm": choice["core_size_mm"], "all_conditions_same_scored_support": True}
            save_json(folder/"result.json", field_result)
            result["fields"].append(field_result)
        stage = "final_integrity"
        if buffer_hashes(fine_model) != buffers["fine"] or buffer_hashes(coarse_model) != buffers["coarse"]:
            raise ValueError("Model buffers changed")
        cache.verify_unchanged()
        result.update({"status": "complete", "numeric_gate_pass": all(row["gate"]["pass"] for row in result["fields"]),
                       "complete_maps_require_visual_review": True, "letter_reading_claim": False,
                       "model_buffers_unchanged": True})
        code = 0
    except Exception as error:
        result.update({"status": "resource_unassessable" if isinstance(error, StopExperiment) else "failed",
                       "stage": stage, "error_type": type(error).__name__,
                       "reason": str(error) if isinstance(error, StopExperiment) else "source_geometry_or_execution_validation_failed",
                       "numeric_gate_pass": False, "letter_reading_claim": False})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        result.update({"forwards_completed": limits.count, "wall_seconds": time.monotonic()-started,
                       "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        if cache is not None:
            result.update({"new_application_body_bytes": cache.bytes, "requests_including_redirects": cache.requests,
                           "source_receipt_sha256": sha(out/"source_receipt.json") if (out/"source_receipt.json").exists() else None})
        save_json(out/"result.json", result)
    print(json.dumps({"status": result["status"], "forwards": limits.count,
                      "numeric_gate_pass": result["numeric_gate_pass"], "wall_seconds": result["wall_seconds"]}), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-reference", required=True)
    parser.add_argument("--hecate-source", required=True, type=Path)
    parser.add_argument("--coarse-checkpoint", required=True, type=Path)
    parser.add_argument("--prior-cache", action="append", nargs=2, metavar=("RECEIPT", "CACHE"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
