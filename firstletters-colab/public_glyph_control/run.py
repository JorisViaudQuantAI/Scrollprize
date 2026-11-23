#!/usr/bin/env python3
"""Frozen complete-published-glyph control; no training or unread inference.

The released Hecate predictor assembles both images without modification. A
four-forward, previously used donor pilot may enable CPU BF16 inside the model
only. Detailed sources, pilot scores, predictions and receipts stay outside Git.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = ("run.py", "protocol.json", "selection.json", "README.md", "test_geometry.py")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                                  ensure_ascii=False) + "\n")
    partial.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def verify_freeze(reference, protocol):
    if not re.fullmatch(r"[0-9a-f]{40}", reference):
        raise ValueError("A full public freeze commit is required")
    for name in FROZEN:
        path = HERE / name
        relative = path.relative_to(ROOT).as_posix()
        if path.read_bytes() != subprocess.check_output(
                ["git", "show", f"{reference}:{relative}"], cwd=ROOT):
            raise ValueError("An experiment file differs from its public freeze")
    if sha(HERE / "selection.json") != protocol["pins"]["selection_sha256"]:
        raise ValueError("The selected published glyph changed")
    for relative, expected in protocol["dependency_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("A pinned dependency changed")


def central_indices(depth, window=64, reverse=False):
    """Exact released full-depth reversal and central-window selection."""
    if depth < window:
        raise ValueError("Insufficient depth")
    start = depth // 2 - window // 2
    if reverse:
        start = depth - start - window
    values = np.arange(start, start + window)
    return values[::-1] if reverse else values


def chunk_plan(shape_zyx, chunk_zyx, origin_yx, extent_yx):
    """Complete XY chunk plan; the source has one full-depth chunk."""
    depth, height, width = map(int, shape_zyx)
    cd, cy, cx = map(int, chunk_zyx)
    y0, x0 = map(int, origin_yx)
    ey, ex = map(int, extent_yx)
    y1, x1 = y0 + ey, x0 + ex
    if cd != depth or min(cd, cy, cx, ey, ex) <= 0:
        raise ValueError("Unexpected depth chunk or empty extent")
    if min(y0, x0) < 0 or y1 > height or x1 > width:
        raise ValueError("The complete context must be inside the source")
    result = []
    for iy in range(y0 // cy, (y1 - 1) // cy + 1):
        for ix in range(x0 // cx, (x1 - 1) // cx + 1):
            sy0, sy1 = max(y0, iy * cy), min(y1, (iy + 1) * cy)
            sx0, sx1 = max(x0, ix * cx), min(x1, (ix + 1) * cx)
            result.append({
                "key": f"0/{iy}/{ix}",
                "source_yx": [sy0 - iy * cy, sy1 - iy * cy,
                              sx0 - ix * cx, sx1 - ix * cx],
                "target_yx": [sy0 - y0, sy1 - y0, sx0 - x0, sx1 - x0],
            })
    return result


def validate_metadata(zarray, zattrs, source):
    if zarray.get("zarr_format") != 2:
        raise ValueError("Only the pinned Zarr v2 source is accepted")
    mapping = {"shape": "shape_zyx", "chunks": "chunks_zyx", "dtype": "dtype",
               "order": "order", "compressor": "compressor", "filters": "filters",
               "fill_value": "fill_value", "dimension_separator": "dimension_separator"}
    for key, reference in mapping.items():
        if key not in zarray or zarray[key] != source[reference]:
            raise ValueError("Unexpected source metadata: " + key)
    scales = zattrs.get("multiscales", [])
    if len(scales) != 1:
        raise ValueError("Ambiguous source multiscale metadata")
    axes = scales[0].get("axes", [])
    if ([a.get("name") for a in axes] != ["z", "y", "x"]
            or any(a.get("unit") != "micrometer" for a in axes)):
        raise ValueError("Unexpected source axes or physical units")
    level0 = [x for x in scales[0]["datasets"] if x.get("path") == "0"]
    transforms = level0[0]["coordinateTransformations"] if len(level0) == 1 else []
    expected = [{"scale": source["spacing_um_zyx"], "type": "scale"},
                {"translation": [0.0, 0.0, 0.0], "type": "translation"}]
    if transforms != expected or zattrs.get("num_slices") != source["shape_zyx"][0]:
        raise ValueError("Unexpected level-zero sampling or depth")


class ResourceStop(RuntimeError):
    pass


class Budget:
    def __init__(self, protocol, started):
        self.limits = protocol["limits"]
        self.started = started
        self.attempts = 0
        self.completed = 0
        self.fixture_completed = 0

    def check(self):
        if time.monotonic() - self.started >= self.limits["whole_run_wall_seconds"]:
            raise ResourceStop("whole_run_time_cap_reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > self.limits["peak_rss_soft_kib"]:
            raise ResourceStop("peak_RSS_soft_cap_reached")

    def begin_forward(self):
        self.check()
        if self.attempts >= self.limits["maximum_forwards"]:
            raise ResourceStop("forward_cap_reached")
        self.attempts += 1
        return time.monotonic()

    def finish_forward(self, started, fixture=False):
        elapsed = time.monotonic() - started
        self.completed += 1
        self.fixture_completed += int(fixture)
        self.check()
        return elapsed


class SurfaceReader:
    """Acquire every raw source chunk; retain only the prescribed 65 planes."""
    def __init__(self, selection, protocol, cache, budget, out):
        self.selection, self.protocol = selection, protocol
        self.source, self.cache = protocol["source"], cache
        self.budget, self.out = budget, Path(out)
        self.base = selection["source_base"].rstrip("/")
        self.plan = chunk_plan(self.source["shape_zyx"], self.source["chunks_zyx"],
                               selection["context"]["origin_yx"],
                               selection["context"]["shape_yx"])
        if len(self.plan) != self.source["complete_chunk_count"]:
            raise ValueError("Unexpected source chunk union")

    def read(self):
        metadata = {}
        for record in self.source["metadata"]:
            self.budget.check()
            path = self.cache.get(self.base + "/" + record["suffix"],
                                  record["sha256"], record["bytes"])
            metadata[record["suffix"]] = json.loads(path.read_text())
        validate_metadata(metadata["0/.zarray"], metadata[".zattrs"], self.source)
        save_json(self.out / "acquisition_plan.json", {
            "source_base": self.base, "chunk_plan": self.plan,
            "chunk_count": len(self.plan), "complete_chunk_reads": True,
            "retained_source_z_halfopen": self.source["retained_source_z_halfopen"],
            "no_resampling": True, "missing_objects_are_fatal": True,
        })
        partial = self.out / "source_union65_uint8.partial.npy"
        volume = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8,
                                          shape=tuple(self.source["retained_shape_zyx"]))
        coverage = np.zeros(volume.shape[-2:], dtype=bool)
        z0, z1 = self.source["retained_source_z_halfopen"]
        workers = self.protocol["limits"]["download_workers"]
        completed = 0

        def acquire(item):
            url = self.base + "/" + self.source["chunk_array_path"] + "/" + item["key"]
            return self.cache.get(url, expected_bytes=self.source["complete_chunk_bytes"])

        # Only one bounded group is in flight. A failed group never queues the
        # remaining source plan. PinnedCache counts all completed/failed bodies.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for offset in range(0, len(self.plan), workers):
                self.budget.check()
                group = self.plan[offset:offset + workers]
                futures = [executor.submit(acquire, item) for item in group]
                for item, future in zip(group, futures):
                    path = future.result()
                    chunk = np.fromfile(path, dtype=np.uint8).reshape(self.source["chunks_zyx"])
                    sy0, sy1, sx0, sx1 = item["source_yx"]
                    ty0, ty1, tx0, tx1 = item["target_yx"]
                    if coverage[ty0:ty1, tx0:tx1].any():
                        raise ValueError("Source chunk footprints overlap")
                    volume[:, ty0:ty1, tx0:tx1] = chunk[z0:z1, sy0:sy1, sx0:sx1]
                    coverage[ty0:ty1, tx0:tx1] = True
                    completed += 1
                    del chunk
                self.budget.check()
                save_json(self.out / "progress.json", {
                    "stage": "source_acquisition", "chunks_complete": completed,
                    "chunks_required": len(self.plan), "new_body_bytes": self.cache.bytes,
                    "forwards_completed": self.budget.completed,
                    "wall_seconds": time.monotonic() - self.budget.started,
                })
                if offset % 40 == 0 or completed == len(self.plan):
                    print(json.dumps({"stage": "source_acquisition", "chunks": completed,
                                      "bytes": self.cache.bytes}), flush=True)
        if completed != len(self.plan) or not coverage.all():
            raise ValueError("Acquisition support is incomplete")
        if self.cache.bytes != self.source["metadata_and_chunk_body_bytes"]:
            raise ValueError("Acquired bodies differ from the complete source plan")
        volume.flush()
        del volume
        final = self.out / "source_union65_uint8.npy"
        partial.replace(final)
        np.save(self.out / "acquisition_support.npy", coverage, allow_pickle=False)
        return final


def make_cpu_bf16_adapter(torch, model):
    """Keep official predict unchanged; override only its model-call scope."""
    class CPUAutocastAdapter(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.patch_size = inner.patch_size
            self.divisor = inner.divisor
            self.margin = inner.margin

        def forward(self, image, valid):
            # A disabled outer autocast in released predict does not disable
            # this explicitly enabled inner CPU autocast scope.
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
                return self.inner(image, valid)

    return CPUAutocastAdapter(model).requires_grad_(False).eval()


class DepthArchive:
    """Archive existing forward values, never features or a new projection."""
    def __init__(self, out, expected_per_order=225):
        self.out = Path(out)
        self.expected = expected_per_order
        self.maps, self.paths = {}, {}
        self.order, self.index, self.pending = None, 0, set()
        self.origin_dtypes = {}
        self.total_bytes = 0

    def begin_order(self, order):
        if self.maps:
            raise ValueError("The previous archive order is still open")
        self.order, self.index, self.pending = order, 0, set()
        self.origin_dtypes = {}
        shapes = {"pre_collapse_logits": (self.expected, 62, 64, 64),
                  "attention_scores": (self.expected, 62, 64, 64),
                  "patch_probability": (self.expected, 256, 256)}
        self.paths = {name: self.out / f"{order}_{name}.npy" for name in shapes}
        self.maps = {name: np.lib.format.open_memmap(self.paths[name], mode="w+",
                                                   dtype=np.float32, shape=shape)
                     for name, shape in shapes.items()}

    def capture(self, name, value):
        if not self.maps or self.index >= self.expected or name in self.pending:
            raise ValueError("Unexpected or repeated archive tensor")
        if name not in ("pre_collapse_logits", "attention_scores"):
            raise ValueError("Unknown archive tensor")
        if tuple(value.shape) != (1, 1, 62, 64, 64):
            raise ValueError("Unexpected pre-collapse tensor shape")
        dtype = str(value.dtype)
        if name in self.origin_dtypes and self.origin_dtypes[name] != dtype:
            raise ValueError("An archived originating dtype changed")
        self.origin_dtypes[name] = dtype
        array = value.detach().float().cpu().numpy()[0, 0]
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite archived tensor")
        self.maps[name][self.index] = array
        self.pending.add(name)

    def finish_forward(self, logits):
        if self.pending != {"pre_collapse_logits", "attention_scores"}:
            raise ValueError("A required same-forward tensor was not captured")
        probability = logits.detach().float().sigmoid().cpu().numpy()[0, 0]
        if probability.shape != (256, 256) or not np.isfinite(probability).all():
            raise ValueError("Invalid archived actual 2D probability")
        self.maps["patch_probability"][self.index] = probability
        self.origin_dtypes["final_model_logits"] = str(logits.dtype)
        self.origin_dtypes["patch_probability"] = str(probability.dtype)
        self.index += 1
        self.pending.clear()
        if self.index % 15 == 0:
            for value in self.maps.values():
                value.flush()

    def finish_order(self):
        if self.index != self.expected or self.pending:
            raise ValueError("The depth archive is incomplete")
        self.close()
        rows = {name: {"file": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
                for name, path in self.paths.items()}
        self.total_bytes += sum(row["bytes"] for row in rows.values())
        return {"patch_count": self.index, "arrays": rows,
                "originating_dtypes": dict(self.origin_dtypes), "stored_dtype": "float32",
                "storage_order": "row-major XY tiles; depth follows each model input order",
                "two_dimensional_reconstruction_review": "pending; archived values must not select this run's output"}

    def close(self):
        for value in self.maps.values():
            value.flush()
        self.maps = {}


def attach_depth_capture(model, archive):
    """Return the original callable and hook handle for explicit restoration."""
    original = model.features_logits

    def capture_features(*inputs, **kwargs):
        values = original(*inputs, **kwargs)
        archive.capture("pre_collapse_logits", values[1])
        return values  # Identical tuple and tensor objects, without rewriting.

    def capture_attention(_module, _inputs, output):
        archive.capture("attention_scores", output)

    model.features_logits = capture_features
    handle = model.canonical.decoder.depth_collapse.attn_conv.register_forward_hook(capture_attention)
    return original, handle


def pilot_decision(records, settings):
    donors = sorted(settings["inputs"])
    comparison = {}
    for donor in donors:
        fp32 = records[(donor, "fp32")]["scores"]
        bf16 = records[(donor, "bf16")]["scores"]
        delta = np.abs(fp32.astype(np.float64) - bf16.astype(np.float64))
        maximum, mean = float(delta.max()), float(delta.mean())
        comparison[donor] = {
            "max_full_score_abs_error": maximum, "mean_full_score_abs_error": mean,
            "numerical_pass": maximum <= settings["full_score_max_abs_error_inclusive"]
            and mean <= settings["full_score_mean_abs_error_inclusive"],
        }
    means = {kind: float(np.mean([records[(d, kind)]["seconds"] for d in donors]))
             for kind in ("fp32", "bf16")}
    speed = means["bf16"] < means["fp32"]
    accepted = all(row["numerical_pass"] for row in comparison.values()) and speed
    return {"by_donor": comparison, "mean_seconds": means, "timing_pass": speed,
            "bf16_accepted": accepted, "selected_precision": "bf16" if accepted else "fp32",
            "not_an_independent_speed_benchmark": True, "warmup_forwards": 0,
            "first_unwarmed_forward": settings["first_unwarmed_forward"]}


def run_pilots(model, fields, protocol, budget, out, torch):
    settings, rows, saved = protocol["precision_pilot"], {}, []
    for index, entry in enumerate(settings["fixed_order"]):
        donor, precision = entry["scroll"], entry["precision"]
        field = np.load(fields[donor], mmap_mode="r", allow_pickle=False)
        raw = np.asarray(field[:, :256, :256])[::-1].copy()
        support = torch.from_numpy(raw.any(0))[None, None, None].expand(1, 1, 64, -1, -1)
        image = torch.from_numpy((raw.astype(np.float32) / np.float32(model.divisor))[None, None])
        begin = budget.begin_forward()
        with torch.inference_mode():
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
                logits = model(image, support)
            # Match the adapter route: the float-logit sigmoid lies outside its
            # enabled model-autocast scope, exactly as in released predict.
            scores = logits.float().sigmoid()[0, 0].cpu().numpy().copy()
        elapsed = budget.finish_forward(begin)
        if scores.shape != (256, 256) or scores.dtype != np.float32 or not np.isfinite(scores).all():
            raise ValueError("Invalid precision pilot output")
        name = f"pilot_{donor}_{precision}.npy"
        np.save(out / name, scores, allow_pickle=False)
        rows[(donor, precision)] = {"scores": scores, "seconds": elapsed}
        saved.append({"index": index, "scroll": donor, "precision": precision,
                      "forward_seconds": elapsed, "full_score_file": name,
                      "full_score_file_sha256": sha(out / name)})
        print(json.dumps({"stage": "precision_pilot", "index": index,
                          "scroll": donor, "precision": precision, "seconds": elapsed}), flush=True)
        del field, raw, image, support, logits
    result = pilot_decision(rows, settings)
    result["fixed_order_results"] = saved
    result["fp32_worst_pilot_seconds"] = max(row["seconds"] for (d, p), row in rows.items() if p == "fp32")
    save_json(out / "precision_pilot.json", result)
    return result


def support_receipt(volume, selection, hecate, out):
    """Separate acquired bytes from CT nonempty columns; never label either ink."""
    supports, counts = {}, {}
    for order in ("source", "reverse"):
        indices = central_indices(volume.shape[0], 64, order == "reverse")
        # Order does not alter any-over-depth support, but the physical slabs
        # differ by one source plane because 109 and 65 are odd.
        low, high = int(indices.min()), int(indices.max()) + 1
        nonempty = np.any(volume[low:high], axis=0)
        for y in hecate.axis_starts(2048, 256, 128):
            for x in hecate.axis_starts(2048, 256, 128):
                if not nonempty[y:y+256, x:x+256].any():
                    raise ValueError("An entirely empty patch would violate the fixed forward count")
        path = out / f"ct_nonempty_support_{order}.npy"
        np.save(path, nonempty, allow_pickle=False)
        supports[order] = sha(path)
        counts[order] = {"context_nonempty_columns": int(nonempty.sum()),
                         "context_total_columns": int(nonempty.size),
                         "core_nonempty_columns": int(nonempty[64:1984, 64:1984].sum()),
                         "core_total_columns": 1920 * 1920}
    return {"file_sha256": supports, "counts": counts,
            "meaning": "any nonzero native CT in the selected slab; not tissue, ink or sheet correctness",
            "acquisition_support_complete": True}


def run(args):
    started = time.monotonic()
    protocol = json.loads((HERE / "protocol.json").read_text())
    selection = json.loads((HERE / "selection.json").read_text())
    verify_freeze(args.freeze_reference, protocol)
    native = module(ROOT / "firstletters-colab/native_resolution/run.py", "glyph_native_cache")
    out = native.outside_git(args.output_dir)
    if out.exists():
        raise FileExistsError("Use a fresh output directory; retain failed attempts")
    out.mkdir(parents=True)
    budget = Budget(protocol, started)
    result = {"schema": protocol["schema"], "status": "started", "orders": [],
              "freeze_reference": args.freeze_reference, "training_steps": 0,
              "unread_CT_requests": 0, "new_checkpoint_downloads": 0,
              "new_reading_claim": False, "automatic_order_selection": False,
              "unread_promotion": False, "public_glyph_review_status": "pending",
              "file_sha256": {name: sha(HERE / name) for name in FROZEN},
              "scroll": selection["scroll"], "segment": selection["segment"],
              "published_reference": selection["human_reference"],
              "core": selection["core"], "context": selection["context"]}
    save_json(out / "start.json", result)
    cache, stage, code = None, "local_input_validation", 1
    handles = []
    archive, original_features, instrumented_model = None, None, None

    def alarm_handler(*_):
        raise ResourceStop("whole_run_time_cap_reached")

    previous = signal.signal(signal.SIGALRM, alarm_handler)
    remaining = max(.001, protocol["limits"]["whole_run_wall_seconds"] - (time.monotonic() - started))
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        if shutil.disk_usage(out).free < protocol["archive"]["minimum_free_disk_bytes"]:
            raise ValueError("Insufficient disk for the complete source and same-forward archive")
        weight = native.outside_git(args.fine_checkpoint)
        expected = protocol["checkpoint"]
        if not weight.is_file() or weight.stat().st_size != expected["bytes"] or sha(weight) != expected["sha256"]:
            raise ValueError("The required local fine checkpoint is absent or has a different identity")
        fields = {"PHerc0841": native.outside_git(args.pilot_0841_context),
                  "PHerc0814": native.outside_git(args.pilot_0814_context)}
        for donor, path in fields.items():
            if not path.is_file() or sha(path) != protocol["precision_pilot"]["inputs"][donor]["fine_context_sha256"]:
                raise ValueError("A required historical pilot field is absent or changed")
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(data.shape) != protocol["precision_pilot"]["input_shape_zyx"] or data.dtype != np.uint8:
                raise ValueError("Unexpected historical pilot array")
            del data
        budget.check()
        stage = "model_loading"
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        hecate = module(ROOT / "firstletters-colab/ink/third_party/hecate/hecate.py", "glyph_released_hecate")
        model = hecate.load_model(str(weight), "cpu")
        if model.patch_size != (64, 256, 256) or model.divisor != 200 or model.training:
            raise ValueError("Unexpected frozen model configuration")
        before_buffers = native.buffer_hashes(model)
        result["runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
                             "torch": torch.__version__, "device": "cpu", "threads": 4,
                             "inter_op_threads": 1, "seed": 0, "machine": platform.machine(),
                             "cpu_affinity_count": len(os.sched_getaffinity(0))}
        stage = "precision_pilot"
        pilot = run_pilots(model, fields, protocol, budget, out, torch)
        result["precision_pilot"] = pilot
        execution_model = make_cpu_bf16_adapter(torch, model) if pilot["bf16_accepted"] else model
        stage = "source_acquisition"
        # A new bounded cache with no prior pairs; no synthetic old receipts.
        cache = native.PinnedCache(out / "source_cache", [], protocol, out / "source_receipt.json")
        reader = SurfaceReader(selection, protocol, cache, budget, out)
        volume_path = reader.read()
        volume = np.load(volume_path, mmap_mode="r", allow_pickle=False)
        if list(volume.shape) != protocol["source"]["retained_shape_zyx"] or volume.dtype != np.uint8:
            raise ValueError("Unexpected reconstructed source union")
        support = support_receipt(volume, selection, hecate, out)
        result["source_union_file_sha256"] = sha(volume_path)
        result["source_support"] = support
        result["conservative_fp32_projection_seconds"] = (
            time.monotonic() - started + 450 * pilot["fp32_worst_pilot_seconds"])
        save_json(out / "pre_fixture_inference.json", {
            "source_union_file_sha256": result["source_union_file_sha256"],
            "source_support": support, "pilot_completed_forwards": budget.completed,
            "fixture_forwards_before_freeze": budget.fixture_completed,
            "new_application_body_bytes": cache.bytes,
            "selected_forward_precision": pilot["selected_precision"],
            "source_z_halfopen_by_order": protocol["inference"]["source_z_halfopen_by_order"],
            "conservative_fp32_projection_seconds": result["conservative_fp32_projection_seconds"],
            "projection_is_not_a_gate": True,
        })
        archive = DepthArchive(out, protocol["inference"]["forwards_per_order"])
        scale = model.depth_coordinate_scale.detach().float().cpu().numpy().copy()
        result["depth_archive_contract"] = {
            "depth_coordinate_scale": float(scale),
            "depth_coordinate_scale_originating_dtype": str(model.depth_coordinate_scale.dtype),
            "depth_coordinate_scale_array_sha256": native.array_sha(scale),
            "raw_attention_scores_before_coordinate_bias": True,
            "attention_z_coordinate": "(arange(62)-31)/31",
            "model_input_depth_margin": 1,
            "source_feature_z_indices_by_order": {"source": list(range(23, 85)),
                                                   "reverse": list(range(85, 23, -1))},
            "spatial_stride": 4,
            "archived_actual_2d_probability": "sigmoid(final_logits.float()), before support/Hann weighting",
            "support_reconstructable_from_source_union_and_fixed_grid": True,
            "same_450_fixture_forwards_only": True, "feature_tensors_archived": False,
            "no_alternative_projection_selected": True,
        }
        save_json(out / "depth_archive_contract.json", result["depth_archive_contract"])
        instrumented_model = model
        original_features, attention_handle = attach_depth_capture(model, archive)
        handles.append(attention_handle)
        state = {"order": None, "started": None, "timings": []}

        def before_forward(_module, inputs):
            if state["started"] is not None or tuple(inputs[0].shape) != (1, 1, 64, 256, 256):
                raise ValueError("Unexpected batch, shape or nested top-level forward")
            state["started"] = budget.begin_forward()

        def after_forward(_module, _inputs, output):
            if tuple(output.shape) != (1, 1, 256, 256) or not torch.isfinite(output).all():
                raise ValueError("Invalid model logits")
            elapsed = budget.finish_forward(state["started"], fixture=True)
            archive.finish_forward(output)
            state["started"] = None
            state["timings"].append(elapsed)
            if len(state["timings"]) % 15 == 0:
                save_json(out / "progress.json", {
                    "stage": "fixture_inference", "order": state["order"],
                    "order_forwards": len(state["timings"]),
                    "all_forwards_completed": budget.completed,
                    "wall_seconds": time.monotonic() - started,
                })

        handles.extend([execution_model.register_forward_pre_hook(before_forward),
                        execution_model.register_forward_hook(after_forward)])
        for order in protocol["inference"]["orders"]:
            stage = "fixture_inference_" + order
            state.update({"order": order, "timings": [], "started": None})
            budget.check()
            archive.begin_order(order)
            path = out / (order + "_context_uint8.npy")
            output = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(2048, 2048))
            hecate.predict(execution_model, volume, output, reverse=order == "reverse",
                           stride=128, batch_size=1, precision="fp32", three_d=False)
            output.flush()
            if len(state["timings"]) != protocol["inference"]["forwards_per_order"]:
                raise ValueError("The fixed full-grid inference was incomplete")
            core = np.asarray(output[64:1984, 64:1984]).copy()
            np.save(out / (order + "_core_uint8.npy"), core, allow_pickle=False)
            hecate.write_png(core, out / (order + "_core.png"))
            hecate.write_png(output, out / (order + "_context.png"))
            archive_result = archive.finish_order()
            if archive.total_bytes > protocol["archive"]["maximum_additional_disk_bytes"]:
                raise ValueError("The fixed archive disk budget was exceeded")
            row = {"order": order, "actual_forward_precision": pilot["selected_precision"],
                   "official_predict_precision_argument": "fp32", "forwards": len(state["timings"]),
                   "forward_seconds": state["timings"],
                   "source_z_halfopen": protocol["inference"]["source_z_halfopen_by_order"][order],
                   "is_pure_reversal_of_same_physical_slab": False,
                   "depth_archive": archive_result,
                   "output_sha256": {p.name: sha(p) for p in [path, out / (order + "_core_uint8.npy"),
                                      out / (order + "_core.png"), out / (order + "_context.png")]},
                   "human_legibility_review": "pending; do not infer from filename or expected letter"}
            result["orders"].append(row)
            save_json(out / (order + "_result.json"), row)
            save_json(out / "partial_result.json", result)
            del output, core
        stage = "final_integrity"
        for handle in handles:
            handle.remove()
        handles = []
        model.features_logits = original_features
        original_features = None
        if budget.attempts != 454 or budget.completed != 454 or budget.fixture_completed != 450:
            raise ValueError("Unexpected complete experiment forward counts")
        if native.buffer_hashes(model) != before_buffers:
            raise ValueError("Model buffers changed")
        if sha(weight) != expected["sha256"] or sha(volume_path) != result["source_union_file_sha256"]:
            raise ValueError("A used checkpoint or reconstructed input changed")
        for donor, path in fields.items():
            if sha(path) != protocol["precision_pilot"]["inputs"][donor]["fine_context_sha256"]:
                raise ValueError("A pilot source changed")
        cache.verify_unchanged()
        budget.check()
        result.update({"status": "complete", "model_buffers_unchanged": True,
                       "all_source_hashes_reverified": True, "both_orders_retained": True,
                       "additional_depth_archive_bytes": archive.total_bytes})
        code = 0
    except (Exception, KeyboardInterrupt) as error:
        result.update({"status": "stopped", "stopped_stage": stage,
                       "error_type": type(error).__name__, "reason": str(error)[:240]})
    finally:
        for handle in handles:
            handle.remove()
        if original_features is not None:
            instrumented_model.features_logits = original_features
        if archive is not None:
            archive.close()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        result.update({"forward_attempts": budget.attempts, "completed_forwards": budget.completed,
                       "completed_fixture_forwards": budget.fixture_completed,
                       "wall_seconds": time.monotonic() - started,
                       "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       "new_application_body_bytes": cache.bytes if cache is not None else 0,
                       "requests_including_redirects": cache.requests if cache is not None else 0,
                       "wire_bytes_measured": False})
        save_json(out / "result.json", result)
        print(json.dumps({key: result[key] for key in ("status", "completed_forwards",
                         "completed_fixture_forwards", "wall_seconds", "new_application_body_bytes")}), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-reference", required=True)
    parser.add_argument("--fine-checkpoint", type=Path, required=True)
    parser.add_argument("--pilot-0841-context", type=Path, required=True)
    parser.add_argument("--pilot-0814-context", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
