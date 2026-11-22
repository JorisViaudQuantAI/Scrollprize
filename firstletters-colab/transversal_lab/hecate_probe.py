#!/usr/bin/env python3
"""A bounded, frozen Hecate readout diagnostic on three public donor surfaces.

The JSON protocol and donor manifest must be published before ``run``. Outputs,
downloaded arrays, features, fitted heads and labels belong outside this repo.
``preflight`` reads JSON only. ``synthetic-check`` reads no experimental data.
This does not run an unread-scroll search or establish independent ink truth.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import threading
import time
from urllib.parse import urlparse
from urllib.request import urlopen
import warnings

# Set before importing numerical libraries; torch is configured explicitly too.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "4"

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def event(stage: str, **values: object) -> None:
    print(json.dumps({"stage": stage, **values}, allow_nan=False), flush=True)


def json_plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_plain(item) for item in value]
    return value


def box_axis_weights(edges: np.ndarray, source_length: int):
    """Integrate piecewise-constant native voxels over exact target cell edges.

    Native voxel i occupies [i-.5, i+.5]. No reflection, padding, interpolation
    outside support, or endpoint-based zoom is allowed.
    """
    edges = np.asarray(edges, dtype=np.float64)
    if (np.any(np.diff(edges) <= 0) or edges[0] < -0.5-1e-8
            or edges[-1] > source_length-0.5+1e-8):
        raise ValueError("Requested physical cell leaves native voxel support")
    first = max(0, int(math.floor(edges[0]+0.5+1e-10)))
    last = min(source_length, int(math.ceil(edges[-1]+0.5-1e-10)))
    native = np.arange(first, last, dtype=np.float64)
    overlap = np.maximum(0.0, np.minimum(edges[1:, None], native[None]+0.5)
                         - np.maximum(edges[:-1, None], native[None]-0.5))
    weights = overlap / np.diff(edges)[:, None]
    if not np.allclose(weights.sum(1), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Incomplete voxel-volume support")
    return slice(first, last), weights.astype(np.float32)


def context_geometry(donor: dict, context: dict, protocol: dict):
    pitch = float(donor["source_pitch_um"])
    ratio = float(protocol["sampling"]["target_pitch_um"])/pitch
    shape = donor["source_shape_zyx"]
    crop = donor["crop_xyxy"]
    y, x = map(int, context["origin_target_yx"])
    if y < 0 or x < 0:
        raise ValueError("Negative target origin")
    z_edges = float(protocol["sampling"]["z_center_source_index"]) + (np.arange(17)-8)*ratio
    y_edges = crop[1] + (y+np.arange(65))*ratio-0.5
    x_edges = crop[0] + (x+np.arange(65))*ratio-0.5
    if y_edges[-1] > crop[3]-0.5+1e-8 or x_edges[-1] > crop[2]-0.5+1e-8:
        raise ValueError("Context leaves the documented donor crop")
    return tuple(box_axis_weights(edges, int(length)) for edges, length in
                 zip((z_edges, y_edges, x_edges), shape))


def resample_box(raw: np.ndarray, weights) -> np.ndarray:
    wz, wy, wx = weights
    value = np.einsum("az,zyx->ayx", wz, raw.astype(np.float32), optimize=True)
    value = np.einsum("by,ayx->abx", wy, value, optimize=True)
    value = np.einsum("cx,abx->abc", wx, value, optimize=True)
    if not np.isfinite(value).all() or value.min() < -1e-3 or value.max() > 255.001:
        raise ValueError("Box-average values outside the source uint8 range")
    # Convex voxel averages stay inside [0,255]. There is no clipping or stretch.
    return np.rint(value).astype(np.uint8)


def label_indices(donor: dict, context: dict, central: bool = True):
    y, x = map(int, context["origin_target_yx"])
    low, high = (16, 48) if central else (0, 64)
    if abs(float(donor["label_pitch_um"])-9.36) > 1e-8:
        raise ValueError("Categorical mapping requires the frozen 9.36 um grid")
    # Exact 9.6/9.36=40/39. Integer arithmetic fixes half-cell tie behavior.
    yy = ((2*(y+np.arange(low, high, dtype=np.int64))+1)*20)//39
    xx = ((2*(x+np.arange(low, high, dtype=np.int64))+1)*20)//39
    h, w = donor["label_shape_yx"]
    if yy.min() < 0 or xx.min() < 0 or yy.max() >= h or xx.max() >= w:
        raise ValueError("Context is outside the documented label grid")
    return yy, xx


def geometry_footprint(donor: dict, context: dict):
    """All mask cells intersecting the physical 64x64 input-cell rectangle."""
    origin = np.asarray(context["origin_target_yx"], dtype=np.int64)
    if abs(float(donor["geometry_mask_pitch_um"])-9.36) > 1e-8:
        raise ValueError("Geometry mapping requires the frozen 9.36 um grid")
    lower = (origin*40)//39
    upper = ((origin+64)*40+38)//39
    shape = np.asarray(donor["geometry_mask_shape_yx"])
    if (lower < 0).any() or (upper > shape).any():
        raise ValueError("Full context footprint leaves the geometry-mask grid")
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def preflight(protocol: dict, manifest: dict) -> dict:
    if protocol["status"] != "prospectively_frozen_before_run":
        raise ValueError("Protocol must explicitly be frozen before execution")
    donors = manifest["donors"]
    expected = {"train": "PHerc0009B", "calibration": "PHerc0841", "test": "PHerc0814"}
    if len(donors) != 3 or {d["split"] for d in donors} != set(expected):
        raise ValueError("Exactly one public donor for each of three fixed splits is required")
    summary = []
    forwards = 0
    for donor in donors:
        if donor["scroll"] != expected[donor["split"]]:
            raise ValueError("Unexpected donor scroll or split")
        if donor["label_values"] != {"ink": 2, "background": 1, "unknown": 0}:
            raise ValueError("Studio labels must be 0 unknown / 1 background / 2 ink")
        if len(donor["contexts"]) != 16:
            raise ValueError("Exactly sixteen predefined contexts per source are required")
        metadata = donor["source_zarr_metadata"]
        if metadata["zarr_format"] != 2 or metadata["dtype"] != "|u1" or metadata["order"] != "C":
            raise ValueError("Only C-order uint8 Zarr v2 source slabs are supported")
        if metadata.get("filters") not in (None, []):
            raise ValueError("Unsupported source filters")
        if metadata["shape"] != donor["source_shape_zyx"] or metadata["shape"][0] != 65:
            raise ValueError("This protocol requires the documented sixty-five fine planes")
        if abs(float(donor["label_pitch_um"])-9.36) > 1e-8:
            raise ValueError("Unexpected Studio label sampling")
        for field in ("source_zarr_url", "label_url"):
            parsed = urlparse(donor[field])
            if parsed.scheme != "https" or parsed.query or parsed.username:
                raise ValueError("Use public canonical HTTPS URLs without credentials or query strings")
        seen_contexts = set()
        for context in donor["contexts"]:
            key = tuple(context["origin_target_yx"])
            if key in seen_contexts:
                raise ValueError("Duplicate predefined context")
            seen_contexts.add(key)
            context_geometry(donor, context, protocol)
            label_indices(donor, context, central=False)
            if donor.get("geometry_mask_url"):
                geometry_footprint(donor, context)
        forwards += 2*len(donor["contexts"])
        summary.append({"id": donor["id"], "scroll": donor["scroll"], "split": donor["split"],
                        "contexts": len(donor["contexts"]), "source_pitch_um": donor["source_pitch_um"]})
    if forwards > protocol["budgets"]["maximum_forwards"]:
        raise ValueError("Forward cap exceeded")
    return {"json_only": True, "maximum_forwards": forwards, "donors": summary,
            "real_image_access": False, "status": "metadata_contract_passed"}


class DownloadCache:
    def __init__(self, root: Path, maximum_bytes: int):
        self.root = root
        self.root.mkdir()
        self.maximum_bytes = maximum_bytes
        self.bytes = 0
        self.reserved_bytes = 0
        self.records = {}
        self.failures = []
        self.lock = threading.Lock()
        self.receipt_path = root.parent/"source_acquisition.json"

    def _save_receipt_locked(self):
        save_json(self.receipt_path, {"transferred_body_bytes": self.bytes,
            "maximum_source_bytes_excluding_checkpoint": self.maximum_bytes,
            "source_chunks": "Mutable public bucket chunks are hash-recorded on acquisition, not pre-pinned.",
            "completed_assets": list(self.records.values()), "failed_assets": self.failures})

    def get(self, url: str, expected_sha256: str | None = None,
            expected_bytes: int | None = None) -> Path:
        name = hashlib.sha256(url.encode()).hexdigest()
        path = self.root/name
        if path.exists():
            if expected_sha256 and sha256_file(path) != expected_sha256:
                raise ValueError("Cached asset hash mismatch")
            return path
        partial = path.with_suffix(".partial")
        digest = hashlib.sha256()
        count = 0
        try:
            with urlopen(url, timeout=45) as response, partial.open("xb") as output:
                while True:
                    with self.lock:
                        allowance = min(1 << 20, self.maximum_bytes-self.bytes-self.reserved_bytes)
                        if allowance <= 0:
                            raise ValueError("Source download byte cap reached")
                        self.reserved_bytes += allowance
                    block = b""
                    try:
                        block = response.read(allowance)
                    finally:
                        with self.lock:
                            self.reserved_bytes -= allowance
                            self.bytes += len(block)
                    if not block:
                        break
                    count += len(block)
                    if expected_bytes is not None and count > expected_bytes:
                        raise ValueError("Source asset is larger than its declared byte count")
                    digest.update(block)
                    output.write(block)
        except Exception as error:
            # Redirected signed URLs must not leak into terminal traces/receipts.
            with self.lock:
                self.failures.append({"url": url, "asset_key": name, "received_bytes": count,
                    "partial_sha256": digest.hexdigest(), "error_type": type(error).__name__})
                self._save_receipt_locked()
            raise RuntimeError(f"Public source download failed ({type(error).__name__}); canonical asset {name}") from None
        error = ("byte_count_mismatch" if expected_bytes is not None and count != expected_bytes
                 else "sha256_mismatch" if expected_sha256 and digest.hexdigest() != expected_sha256 else None)
        if error:
            with self.lock:
                self.failures.append({"url": url, "asset_key": name, "received_bytes": count,
                    "partial_sha256": digest.hexdigest(), "error_type": error})
                self._save_receipt_locked()
            raise ValueError(f"Downloaded source {error}")
        partial.rename(path)
        with self.lock:
            self.records[name] = {"url": url, "bytes": count, "sha256": digest.hexdigest()}
            self._save_receipt_locked()
        return path


class SurfaceReader:
    def __init__(self, donor: dict, cache: DownloadCache):
        self.donor, self.cache = donor, cache
        self.metadata = donor["source_zarr_metadata"]
        self.base = donor["source_zarr_url"].rstrip("/")
        # Manifest URLs point to the level-0 array, already ending in /0/.
        array_url = f"{self.base}/.zarray"
        actual_path = cache.get(array_url, donor["source_zarr_metadata_sha256"])
        if json.loads(actual_path.read_text()) != self.metadata:
            raise ValueError("Live source Zarr metadata differs from frozen manifest")
        self.chunks = self.metadata["chunks"]
        self.shape = self.metadata["shape"]
        self.separator = self.metadata.get("dimension_separator", ".")
        self.compressor = self.metadata.get("compressor")
        if self.compressor:
            import numcodecs
            self.codec = numcodecs.get_codec(self.compressor)

    def read(self, slices) -> np.ndarray:
        output = np.empty(tuple(item.stop-item.start for item in slices), dtype=np.uint8)
        blocks = []
        for z in range(slices[0].start//self.chunks[0], (slices[0].stop-1)//self.chunks[0]+1):
            for y in range(slices[1].start//self.chunks[1], (slices[1].stop-1)//self.chunks[1]+1):
                for x in range(slices[2].start//self.chunks[2], (slices[2].stop-1)//self.chunks[2]+1):
                    index = (z, y, x)
                    key = self.separator.join(map(str, index))
                    url = f"{self.base}/{key}"
                    blocks.append((index, url))
        expected_bytes = int(np.prod(self.chunks)) if not self.compressor else None
        # At most four independent, public chunk reads. Missing chunks fail; no fill.
        with ThreadPoolExecutor(max_workers=4) as pool:
            files = list(pool.map(lambda block: self.cache.get(block[1], expected_bytes=expected_bytes), blocks))
        for (index, _), path in zip(blocks, files):
            data = path.read_bytes()
            if self.compressor:
                data = self.codec.decode(data)
            values = np.frombuffer(data, dtype=np.uint8).reshape(self.chunks)
            native_slices, output_slices = [], []
            for axis in range(3):
                origin = index[axis]*self.chunks[axis]
                lo, hi = max(slices[axis].start, origin), min(slices[axis].stop, origin+self.chunks[axis])
                native_slices.append(slice(lo-origin, hi-origin))
                output_slices.append(slice(lo-slices[axis].start, hi-slices[axis].start))
            output[tuple(output_slices)] = values[tuple(native_slices)]
        return output


def read_categorical(donor: dict, cache: DownloadCache, geometry: bool = False):
    prefix = "geometry_mask" if geometry else "label"
    path = cache.get(donor[f"{prefix}_url"], donor[f"{prefix}_sha256"], donor[f"{prefix}_size_bytes"])
    encoding = donor[f"{prefix}_format"]
    if encoding == "png":
        from PIL import Image
        with Image.open(path) as image:
            if image.mode not in ("L", "P"):
                raise ValueError("Categorical PNG must contain direct L or palette indices")
            values = np.asarray(image).copy()
    elif encoding == "npy":
        values = np.load(path, allow_pickle=False)
    else:
        raise ValueError("Unsupported categorical array format")
    if list(values.shape) != donor["label_shape_yx"] or values.ndim != 2:
        raise ValueError("Categorical array shape differs from the frozen label grid")
    if not geometry and not set(np.unique(values)).issubset({0, 1, 2}):
        raise ValueError("Unexpected Studio category; do not infer a binary threshold")
    return values


def cached_projection(model, features, logits, support):
    """Exact released coarse-model forward projection, reusing one backbone pass."""
    import torch
    import torch.nn.functional as F
    scores = model.canonical.decoder.depth_collapse.attn_conv(features).float()
    half = scores.shape[2]//2
    z = (torch.arange(scores.shape[2], dtype=torch.float32)-half)/half
    scores = scores + model.depth_coordinate_scale*z.view(1, 1, -1, 1, 1)
    valid = support[:, None, None].expand(-1, 1, 16, -1, -1)
    weights = scores.masked_fill(~valid, -1e4).softmax(2)*valid
    weights = weights/weights.sum(2, keepdim=True).clamp_min(1e-8)
    probability = F.interpolate((weights*logits).sum(2).sigmoid(), size=(64, 64),
                                mode="bilinear", align_corners=False)
    eps = torch.finfo(torch.float32).eps
    return torch.logit(probability.clamp(eps, 1-eps))


@dataclass
class Limits:
    protocol: dict
    out: Path
    checkpoint_bytes: int
    start_cpu: float
    forward_count: int = 0

    def check(self):
        if time.process_time()-self.start_cpu > self.protocol["budgets"]["maximum_cpu_seconds"]-10:
            raise RuntimeError("CPU budget reached; no additional extraction or fit")
        disk = self.checkpoint_bytes + sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
        if disk > self.protocol["budgets"]["maximum_data_and_derived_bytes"]:
            raise RuntimeError("Checkpoint + downloaded data + derived-array disk cap reached")
        if self.forward_count > self.protocol["budgets"]["maximum_forwards"]:
            raise RuntimeError("Backbone forward budget exceeded")


def extract_donor(donor, protocol, model, cache, limits):
    import torch
    reader = SurfaceReader(donor, cache)
    geometry = read_categorical(donor, cache, geometry=True) if donor.get("geometry_mask_url") else None
    retained, excluded = [], []
    for number, context in enumerate(donor["contexts"]):
        limits.check()
        if geometry is not None:
            coverage = float(np.mean(geometry[geometry_footprint(donor, context)] > 0))
            if coverage < protocol["sampling"]["minimum_geometry_footprint_coverage"]:
                excluded.append({"id": context["id"], "reason": "geometry_footprint_coverage", "coverage": coverage})
                continue
        mapping = context_geometry(donor, context, protocol)
        slices, weights = zip(*mapping)
        raw = resample_box(reader.read(slices), weights)
        if raw.shape != (16, 64, 64):
            raise ValueError("Unexpected resampled model input shape")
        support = raw.any(axis=0)
        paths = {}
        for order in protocol["depth_orders"]:
            limits.check()
            array = raw if order == "source" else raw[::-1].copy()
            image = torch.from_numpy((array.astype(np.float32)/np.float32(255.0))[None, None])
            start = time.monotonic()
            with torch.inference_mode():
                features, logits = model.features_logits(image)
                baseline = cached_projection(model, features, logits, torch.from_numpy(support[None]))
                pooled = features[0, :, 6:10, 16:48, 16:48].float().mean(1).numpy().copy()
                volume = logits[0, 0, :, 16:48, 16:48].float().numpy().copy()
                baseline_values = baseline[0, 0, 16:48, 16:48].float().numpy().copy()
            limits.forward_count += 1
            path = limits.out/"features"/f"{donor['split']}-{number:02d}-{order}.npz"
            np.savez_compressed(path, features=pooled, volume=volume, baseline=baseline_values,
                                support=support[16:48, 16:48])
            paths[order] = str(path)
            save_json(limits.out/"progress.json", {"stage": "feature_extraction", "donor": donor["id"],
                "context_id": context["id"], "order": order, "forwards_completed": limits.forward_count,
                "cpu_seconds": time.process_time()-limits.start_cpu, "source_bytes": cache.bytes,
                "last_feature_sha256": sha256_file(path)})
            if number % 4 == 0 or number == len(donor["contexts"])-1:
                event("feature_extraction", donor=donor["id"], context=number+1, order=order,
                      forward_wall_seconds=time.monotonic()-start, forwards=limits.forward_count)
            del features, logits, baseline, image
        retained.append({"context": context, "paths": paths,
                         "central_nonzero_column_fraction": float(support[16:48, 16:48].mean())})
    return {"donor": donor, "retained": retained, "excluded": excluded}


def assemble(extracted, labels, order):
    donor = extracted["donor"]
    result = {name: [] for name in ("features", "volume", "baseline", "labels", "context_index", "label_id")}
    seen = set()
    duplicates = 0
    for index, item in enumerate(extracted["retained"]):
        yy, xx = label_indices(donor, item["context"])
        label_id = (yy[:, None]*donor["label_shape_yx"][1]+xx[None]).ravel()
        values = labels[np.ix_(yy, xx)].ravel()
        with np.load(item["paths"][order], allow_pickle=False) as array:
            valid = array["support"].ravel() & (values != donor["label_values"]["unknown"])
            unique = np.array([int(identifier) not in seen for identifier in label_id], dtype=bool)
            duplicates += int(np.sum(valid & ~unique))
            valid &= unique
            seen.update(map(int, label_id[valid]))
            result["features"].append(array["features"].reshape(64, -1).T[valid])
            result["volume"].append(array["volume"].reshape(16, -1).T[valid])
            result["baseline"].append(array["baseline"].ravel()[valid])
            result["labels"].append((values[valid] == donor["label_values"]["ink"]).astype(np.uint8))
            result["context_index"].append(np.full(valid.sum(), index, dtype=np.int16))
            result["label_id"].append(label_id[valid])
    if not result["labels"]:
        raise ValueError("No predefined context retained")
    joined = {name: np.concatenate(value) for name, value in result.items()}
    joined["duplicate_labeled_pixels_removed"] = duplicates
    return joined


def fit_heads(training, protocol):
    from scipy.optimize import minimize
    from scipy.special import expit
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    labels = training["labels"]
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    per_class = min(len(positive), len(negative), protocol["readouts"]["maximum_training_pixels_per_class"])
    if per_class < protocol["readouts"]["minimum_training_pixels_per_class"]:
        raise ValueError(f"Insufficient labeled training support: ink={len(positive)}, background={len(negative)}; no fit or substituted contexts")
    rng = np.random.default_rng(protocol["seed"])
    selected = np.sort(np.r_[rng.choice(positive, per_class, replace=False),
                             rng.choice(negative, per_class, replace=False)])
    y = labels[selected].astype(np.float64)
    x = training["volume"][selected].astype(np.float64)
    penalty = protocol["readouts"]["depth_uniform_l2_penalty"]
    initial = np.r_[np.ones(16)/16, 0.0]

    def objective(theta):
        z = x@theta[:16]+theta[16]
        difference = theta[:16]-1/16
        loss = np.mean(np.logaddexp(0.0, z)-y*z)+0.5*penalty*np.sum(difference**2)
        residual = (expit(z)-y)/len(y)
        gradient = np.r_[x.T@residual+penalty*difference, residual.sum()]
        return float(loss), gradient

    depth = minimize(objective, initial, jac=True, method="SLSQP",
        bounds=[(0.0, 1.0)]*16+[(None, None)],
        constraints={"type": "eq", "fun": lambda t: t[:16].sum()-1,
                     "jac": lambda t: np.r_[np.ones(16), 0.0]},
        options={"maxiter": protocol["readouts"]["depth_max_iterations"],
                 "ftol": protocol["readouts"]["depth_function_tolerance"]})
    depth_valid = bool(depth.success and np.isfinite(depth.x).all()
                       and abs(depth.x[:16].sum()-1) < 1e-6 and depth.x[:16].min() >= -1e-7)
    scaler = StandardScaler().fit(training["features"][selected])
    model = LogisticRegression(C=protocol["readouts"]["feature_logistic_C"], solver="lbfgs",
        max_iter=protocol["readouts"]["feature_max_iterations"],
        tol=protocol["readouts"]["feature_tolerance"], random_state=protocol["seed"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(scaler.transform(training["features"][selected]), labels[selected])
    feature_valid = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    feature_valid &= bool(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all())
    return {
        "training_rows": len(labels), "selected_per_class": per_class,
        "selected_label_ids_sha256": hashlib.sha256(training["label_id"][selected].astype("<i8").tobytes()).hexdigest(),
        "baseline": {"valid": True, "kind": "unmodified_released_2d_logit"},
        "depth": {"valid": depth_valid, "weights": depth.x[:16], "bias": depth.x[16],
                  "success": bool(depth.success), "iterations": int(depth.nit),
                  "message": str(depth.message), "objective": float(depth.fun)},
        "features": {"valid": bool(feature_valid), "mean": scaler.mean_, "scale": scaler.scale_,
                     "weights": model.coef_[0], "bias": model.intercept_[0],
                     "iterations": int(model.n_iter_[0]), "warnings": [str(w.message) for w in caught]},
    }


def score_heads(data, heads):
    scores = {"baseline": data["baseline"].astype(np.float64)}
    if heads["depth"]["valid"]:
        scores["depth"] = data["volume"]@np.asarray(heads["depth"]["weights"])+heads["depth"]["bias"]
    if heads["features"]["valid"]:
        head = heads["features"]
        scores["features"] = ((data["features"]-head["mean"])/head["scale"])@head["weights"]+head["bias"]
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise FloatingPointError("Non-finite frozen readout scores")
    return scores


def calibration_threshold(labels, scores, fraction):
    background = np.sort(scores[labels == 0])[::-1]
    if not len(background):
        return None
    index = min(int(math.floor(fraction*len(background))), len(background)-1)
    # The later prediction comparison is STRICT >; ties therefore cannot raise FPR.
    return float(background[index])


def metrics(labels, scores, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score
    positive, negative = int(labels.sum()), int((labels == 0).sum())
    result = {"pixels": len(labels), "ink": positive, "background": negative,
              "prevalence": positive/len(labels) if len(labels) else None,
              "auroc": None, "average_precision": None, "threshold": threshold,
              "fpr_at_calibration_threshold": None, "recall_at_calibration_threshold": None}
    if positive and negative:
        result["auroc"] = float(roc_auc_score(labels, scores))
        result["average_precision"] = float(average_precision_score(labels, scores))
    order = np.argsort(-scores, kind="stable")
    for fraction in (0.01, 0.05):
        n = max(1, int(math.ceil(fraction*len(labels)))) if len(labels) else 0
        true = int(labels[order[:n]].sum())
        result[f"top_{int(fraction*100)}_percent"] = {
            "reviewed_pixels": n, "precision": true/n if n else None,
            "recall": true/positive if positive else None,
            "tie_break": "stable frozen context order, then row-major label order"}
    if threshold is not None:
        selected = scores > threshold
        result["fpr_at_calibration_threshold"] = float(selected[labels == 0].mean()) if negative else None
        result["recall_at_calibration_threshold"] = float(selected[labels == 1].mean()) if positive else None
    return result


def evaluate(data, scores, thresholds, extracted):
    result = {"parent": {}, "contexts": [],
              "duplicate_labeled_pixels_removed": data["duplicate_labeled_pixels_removed"]}
    for arm, score in scores.items():
        result["parent"][arm] = metrics(data["labels"], score, thresholds[arm])
    for index, item in enumerate(extracted["retained"]):
        selected = data["context_index"] == index
        result["contexts"].append({"id": item["context"]["id"],
            "metrics": {arm: metrics(data["labels"][selected], score[selected], thresholds[arm])
                        for arm, score in scores.items()}})
    return result


def synthetic_check():
    """Small meaningful checks of physical sampling and low-FPR tie handling."""
    for pitch in (2.401, 2.403):
        edges = 31.5+(np.arange(17)-8)*(9.6/pitch)
        interval, weights = box_axis_weights(edges, 65)
        assert interval.start == 0 and interval.stop == 64
        assert np.allclose(weights@np.ones(64), 1.0, atol=1e-6)
    interval, weights = box_axis_weights(np.arange(17)*4-0.5, 65)
    expected = np.arange(64, dtype=np.float32).reshape(16, 4).mean(1)
    assert np.allclose(weights@np.arange(64, dtype=np.float32), expected)
    mapping_donor = {"label_pitch_um": 9.36, "label_shape_yx": [10000,10000],
                     "geometry_mask_pitch_um": 9.36, "geometry_mask_shape_yx": [10000,10000]}
    yy, _ = label_indices(mapping_donor, {"origin_target_yx": [0,0]}, central=False)
    assert yy[19] == 20  # Exact tie picks the next categorical cell, as prescribed.
    footprint = geometry_footprint(mapping_donor, {"origin_target_yx": [2432,1664]})
    assert footprint[0].stop == 2560  # Avoid floating-point ceil yielding 2561.
    labels = np.r_[np.zeros(200, dtype=np.uint8), np.ones(20, dtype=np.uint8)]
    scores = np.r_[np.repeat([0.0, 1.0], 100), np.ones(20)]
    threshold = calibration_threshold(labels, scores, 0.01)
    assert np.mean(scores[labels == 0] > threshold) <= 0.01
    print(json.dumps({"synthetic_checks": "passed", "experimental_data_access": False}))


def run(args, protocol, manifest):
    import torch
    import scipy
    import sklearn
    out = args.out.resolve()
    repo = Path(__file__).resolve().parents[2]
    if out == repo or repo in out.parents:
        raise ValueError("Experimental arrays/labels/heads must be written outside the public repository")
    out.mkdir(parents=True, exist_ok=False)
    (out/"features").mkdir()
    start_wall, start_cpu = time.monotonic(), time.process_time()
    checkpoint_bytes = args.checkpoint.stat().st_size
    if checkpoint_bytes != protocol["model"]["checkpoint_bytes"] or sha256_file(args.checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise ValueError("Pinned checkpoint does not match")
    if sha256_file(args.hecate_source) != protocol["model"]["source_sha256"]:
        raise ValueError("Pinned standalone Hecate code does not match")
    spec = importlib.util.spec_from_file_location("pinned_hecate", args.hecate_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = module.load_model(args.checkpoint, "cpu")
    if any(layer.training for layer in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Backbone must be fully frozen in eval mode")
    normalization_before = {name: value.clone() for name, value in model.named_buffers()
                            if "running_" in name or "num_batches_tracked" in name}
    cache = DownloadCache(out/"source-cache", protocol["budgets"]["maximum_source_bytes_excluding_checkpoint"])
    limits = Limits(protocol, out, checkpoint_bytes, start_cpu)
    provenance = {"protocol_sha256": sha256_file(args.protocol), "manifest_sha256": sha256_file(args.manifest),
        "runner_sha256": sha256_file(Path(__file__)), "checkpoint_sha256": protocol["model"]["checkpoint_sha256"],
        "hecate_source_sha256": protocol["model"]["source_sha256"],
        "versions": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "threads": 4, "interop_threads": 1, "freeze_reference": args.freeze_reference,
        "test_labels_decoded_before_readout_freeze": False}
    save_json(out/"start_receipt.json", provenance)
    donors = {donor["split"]: donor for donor in manifest["donors"]}
    extracted = {}
    labels = {}
    for split in ("train", "calibration"):
        extracted[split] = extract_donor(donors[split], protocol, model, cache, limits)
        labels[split] = read_categorical(donors[split], cache)
    heads, thresholds, prepared, results = {}, {}, {}, {}
    for order in protocol["depth_orders"]:
        prepared[order] = {split: assemble(extracted[split], labels[split], order)
                           for split in ("train", "calibration")}
        heads[order] = fit_heads(prepared[order]["train"], protocol)
        limits.check()
        calibration_scores = score_heads(prepared[order]["calibration"], heads[order])
        thresholds[order] = {arm: calibration_threshold(prepared[order]["calibration"]["labels"], score,
            protocol["evaluation"]["calibration_background_fpr"]) for arm, score in calibration_scores.items()}
        results[order] = {}
        for split in ("train", "calibration"):
            scores = score_heads(prepared[order][split], heads[order])
            results[order][split] = evaluate(prepared[order][split], scores, thresholds[order], extracted[split])
    frozen = {"heads": json_plain(heads), "thresholds": thresholds,
              "test_label_array_not_decoded": True, "source_order_primary": True,
              "no_fit_or_threshold_retries": True, "cpu_seconds_at_freeze": time.process_time()-start_cpu,
              "wall_seconds_at_freeze": time.monotonic()-start_wall, **provenance}
    freeze_path = out/"readout_freeze.json"
    save_json(freeze_path, frozen)
    freeze_hash = sha256_file(freeze_path)
    event("readouts_frozen_before_test_labels", sha256=freeze_hash)
    # Deliberately late: not even the held-out categorical asset is downloaded above.
    extracted["test"] = extract_donor(donors["test"], protocol, model, cache, limits)
    labels["test"] = read_categorical(donors["test"], cache)
    for order in protocol["depth_orders"]:
        test = assemble(extracted["test"], labels["test"], order)
        scores = score_heads(test, heads[order])
        results[order]["test"] = evaluate(test, scores, thresholds[order], extracted["test"])
    unchanged = all(torch.equal(value, dict(model.named_buffers())[name]) for name, value in normalization_before.items())
    if not unchanged or sha256_file(freeze_path) != freeze_hash:
        raise ValueError("Frozen state changed during evaluation")
    limits.check()
    record = {"purpose": protocol["purpose"], "provenance": provenance,
        "readout_freeze_sha256": freeze_hash, "running_normalization_buffers_unchanged": unchanged,
        "forwards": limits.forward_count, "source_download_bytes": cache.bytes,
        "checkpoint_bytes": checkpoint_bytes,
        "source_plus_checkpoint_bytes": cache.bytes+checkpoint_bytes,
        "wall_seconds": time.monotonic()-start_wall, "cpu_seconds": time.process_time()-start_cpu,
        "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "source_assets": list(cache.records.values()),
        "exclusions": {split: value["excluded"] for split, value in extracted.items()},
        "fit_status": {order: {arm: heads[order][arm]["valid"] for arm in ("baseline", "depth", "features")}
                       for order in heads},
        "results": results, "limitations": protocol["limitations"],
        "no_unread_data_access": True, "no_prize_or_readability_claim": True}
    save_json(out/"result.json", json_plain(record))
    event("complete", result=str(out/"result.json"), forwards=limits.forward_count,
          cpu_seconds=record["cpu_seconds"], wall_seconds=record["wall_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "synthetic-check", "run"))
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("hecate_probe_protocol.json"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--hecate-source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze-reference", help="Published git commit that froze this runner, protocol and manifest")
    args = parser.parse_args()
    if args.action == "synthetic-check":
        synthetic_check()
        return
    if args.manifest is None:
        parser.error("--manifest is required")
    protocol = json.loads(args.protocol.read_text())
    if sha256_file(args.manifest) != protocol["donor_manifest_sha256"]:
        raise ValueError("Donor manifest bytes differ from the protocol's pinned manifest")
    manifest = json.loads(args.manifest.read_text())
    checked = preflight(protocol, manifest)
    if args.action == "preflight":
        print(json.dumps(checked, indent=2))
        return
    if any(getattr(args, item) is None for item in ("hecate_source", "checkpoint", "out", "freeze_reference")):
        parser.error("run needs --hecate-source, --checkpoint, --out and --freeze-reference")
    if re.fullmatch(r"[0-9a-f]{40}", args.freeze_reference) is None:
        parser.error("--freeze-reference must be the complete forty-character published commit hash")
    if args.out.exists():
        parser.error("--out must be a new directory; an earlier run must not be overwritten")
    try:
        run(args, protocol, manifest)
    except BaseException as error:
        # Keep an infrastructure failure auditable without exposing redirect tokens.
        message = re.sub(r"https?://[^\s]+", "[URL redacted]", str(error))
        failure = {"status": "failed_without_retry", "error_type": type(error).__name__,
            "error_message": message, "freeze_reference": args.freeze_reference,
            "protocol_sha256": sha256_file(args.protocol), "manifest_sha256": sha256_file(args.manifest),
            "runner_sha256": sha256_file(Path(__file__)),
            "source_acquisition_receipt": "source_acquisition.json", "progress_receipt": "progress.json"}
        if args.out.is_dir():
            save_json(args.out/"failure.json", failure)
        event("failed_without_retry", error_type=type(error).__name__, error_message=message)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
