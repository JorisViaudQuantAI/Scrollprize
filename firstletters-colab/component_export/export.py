"""Write auditable per-pass probabilities without changing legacy inference.

Prediction support describes an executed, weighted prediction. It does not
certify acquisition coverage, tissue, a complete receptive field, or ink.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np


SUPPORT_EPSILON = 1e-6  # The same threshold as infer_ink.sweep's division.
DEPTH_MASS_ATOL = 1e-6
PROVENANCE_FIELDS = (
    "mesh_sha256", "normal_sha256", "normal_convention", "plane_order",
    "frame", "source_content_sha256",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def outside_git(path):
    """Reject ordinary and linked worktrees, including symlink destinations."""
    resolved = Path(path).resolve()
    for ancestor in (resolved, *resolved.parents):
        if os.path.lexists(ancestor / ".git"):
            raise ValueError("Component exports and their image outputs must stay outside every Git worktree")
    return resolved


def read_provenance(path):
    """Keep missing geometry unknown; declarations never override observations."""
    declared = dict.fromkeys(PROVENANCE_FIELDS)
    if path is None:
        return declared, None
    payload = Path(path).read_bytes()
    values = json.loads(payload)
    if not isinstance(values, dict) or set(values) - set(PROVENANCE_FIELDS):
        raise ValueError("Component provenance must contain only: " + ", ".join(PROVENANCE_FIELDS))
    json.dumps(values, allow_nan=False)
    for key, value in values.items():
        if value is None:
            continue
        if key.endswith("sha256"):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{key} must be a lowercase SHA-256 or null")
        elif key == "frame":
            if not isinstance(value, dict) or not value:
                raise ValueError("frame must be a nonempty declared frame object or null")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty description or null")
    declared.update(values)
    return declared, hashlib.sha256(payload).hexdigest()


def prediction_support_on_output_grid(weight_sum, shape_yx):
    """A resize target is supported only if all contributing pixels are.

    Use the same OpenCV linear resize and dimensions as the score output. Testing
    interpolated *invalidity* for exact zero avoids treating a small positive
    invalid contribution as valid through a near-one tolerance.
    """
    weights = np.asarray(weight_sum)
    if weights.ndim != 2 or not weights.size or not np.isfinite(weights).all():
        raise ValueError("Weight sums must be a finite, nonempty two-dimensional array")
    support = weights > SUPPORT_EPSILON
    if support.shape != tuple(shape_yx):
        import cv2

        invalid = cv2.resize(
            (~support).astype(np.float32), (shape_yx[1], shape_yx[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        support = invalid == 0.0
    return support


def _signature(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class ComponentWriter:
    """A new private directory; atomic component directories; receipt last.

    A component directory is published only after its arrays and metadata are
    complete. The run receipt is absent if any model or legacy output fails.
    """

    def __init__(self, output_dir, *, run_metadata, expected_components, provenance_path=None):
        self.root = outside_git(output_dir)
        if os.path.lexists(self.root):
            raise FileExistsError("Use a new private component output directory")
        if not isinstance(expected_components, int) or expected_components < 1:
            raise ValueError("At least one component is required")
        self.declared, self.provenance_sha256 = read_provenance(provenance_path)
        json.dumps(run_metadata, allow_nan=False)
        self.run_metadata = run_metadata
        self.expected = expected_components
        self.components = []
        self.checkpoints = []
        self.finished = False
        self.root.mkdir(parents=True)

    def checkpoint_before_load(self, path):
        """Hash the actual checkpoint and detect ordinary concurrent changes."""
        path = Path(path).resolve()
        before = _signature(path)
        digest = sha256_file(path)
        if _signature(path) != before:
            raise ValueError("Checkpoint changed while hashing; wait for its producer to finish")
        record = {"path": str(path), "sha256": digest, "signature": before}
        self.checkpoints.append(record)
        return record

    def model_sink(self, checkpoint_record, model_index, model_info):
        if _signature(checkpoint_record["path"]) != checkpoint_record["signature"]:
            raise ValueError("Checkpoint changed during model loading")
        # Copy plain metadata now; no model or GPU tensor is retained here.
        model_info = json.loads(json.dumps(model_info, allow_nan=False))

        def write(direction, shift, probability, weight_sum, sampling):
            self.write_component(
                model_index=model_index, model_info=model_info,
                checkpoint_sha256=checkpoint_record["sha256"], direction=direction,
                shift=shift, probability=probability, weight_sum=weight_sum, sampling=sampling,
            )

        return write

    def write_component(self, *, model_index, model_info, checkpoint_sha256, direction,
                        shift, probability, weight_sum, sampling):
        if self.finished:
            raise RuntimeError("A finished component export cannot be extended")
        scores = np.asarray(probability)
        if scores.ndim != 2 or scores.size == 0 or not np.issubdtype(scores.dtype, np.floating):
            raise ValueError("Component scores must be a nonempty floating-point HW array")
        if scores.shape != tuple(self.run_metadata["output_shape_yx"]):
            raise ValueError("Component scores differ from the declared output grid")
        support = prediction_support_on_output_grid(weight_sum, scores.shape)
        if not np.isfinite(scores[support]).all():
            raise ValueError("A prediction-supported component value is nonfinite")
        masses = np.asarray(sampling["model_plane_source_weight_masses"], dtype=np.float64)
        if masses.ndim != 1 or not masses.size or not np.isfinite(masses).all():
            raise ValueError("Finite per-plane depth weight masses are required")
        full_depth = bool(np.all(np.isclose(masses, 1.0, rtol=0.0, atol=DEPTH_MASS_ATOL)))
        complete_depth_support = support & full_depth
        saved_scores = scores.astype(np.float32, copy=True)
        saved_scores[~support] = np.nan
        name = f"component-{len(self.components):04d}"
        staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=self.root))
        try:
            hashes = {}
            for filename, array in (
                ("probability.npy", saved_scores),
                ("prediction_support.npy", support),
                ("complete_depth_support.npy", complete_depth_support),
            ):
                with (staging / filename).open("xb") as stream:
                    np.save(stream, array, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                hashes[filename] = sha256_file(staging / filename)
            metadata = {
                "schema": "scrollprize.inference-component.v1", "status": "complete",
                "model_index": model_index, "model": model_info,
                "checkpoint_sha256": checkpoint_sha256,
                "direction": direction, "shift_model_planes": shift,
                "shape_yx": list(scores.shape), "score_dtype": "float32",
                "score_values": "Original clipped per-pass probabilities before depth/model averaging or display scaling; NaN outside prediction_support",
                "prediction_support_definition": "Model-grid wsum > 1e-6; every linear-resize contributor must be supported",
                "prediction_support_threshold": SUPPORT_EPSILON,
                "prediction_support_pixels": int(support.sum()),
                "full_depth_context": full_depth, "depth_mass_atol": DEPTH_MASS_ATOL,
                "complete_depth_support_definition": "prediction_support AND full_depth_context; does not certify XY/acquisition/tissue coverage",
                "complete_depth_support_pixels": int(complete_depth_support.sum()),
                "acquisition_coverage_status": "unknown: no acquisition-validity input is supplied by infer_ink",
                "sampling": sampling, "declared_geometry": self.declared,
                "declarations_status": "Supplied geometry declarations are recorded, not independently verified; null means unknown",
                "files_sha256": hashes,
            }
            _write_json(staging / "component.json", metadata)
            metadata_hash = sha256_file(staging / "component.json")
            os.replace(staging, self.root / name)
            self.components.append({
                "directory": name, "component_json_sha256": metadata_hash,
                "model_index": model_index, "direction": direction,
                "shift_model_planes": shift,
            })
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def finish(self):
        if self.finished:
            raise RuntimeError("Component export is already finished")
        if len(self.components) != self.expected:
            raise ValueError(f"Expected {self.expected} components, received {len(self.components)}")
        for checkpoint in self.checkpoints:
            if _signature(checkpoint["path"]) != checkpoint["signature"]:
                raise ValueError("Checkpoint changed during inference; no completion receipt will be written")
        receipt = {
            "schema": "scrollprize.inference-components.v1", "status": "complete",
            "run": self.run_metadata, "components": self.components,
            "declared_provenance_file_sha256": self.provenance_sha256,
            "exporter_sha256": sha256_file(__file__),
            "interpretation": "Private review inputs, not an ink finding. Prediction support is not acquisition validity or tissue support. Check depth/XY context and external acquisition support before comparisons.",
        }
        pending = self.root / ".receipt.json.pending"
        _write_json(pending, receipt)
        os.replace(pending, self.root / "receipt.json")
        self.finished = True
        return receipt
