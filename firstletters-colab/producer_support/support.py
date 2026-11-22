"""Conservative source and full-tile context support; no detector calls.

The caller supplies the actual source-availability masks and contribution
geometry. Availability, papyrus membership and executed prediction coverage
are different observations. A positive CT intensity is never used here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

GEOMETRY_KEYS = (
    "mesh_sha256", "normal_sha256", "normal_convention", "plane_order",
    "frame", "source_content_sha256",
)
HEX = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(array):
    """C-order payload hash; dtype and shape must accompany this value."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def outside_git(path):
    resolved = Path(path).resolve()
    for p in (resolved, *resolved.parents):
        if os.path.lexists(p / ".git"):
            raise ValueError("Producer masks and location-bearing receipts must stay outside Git")
    return resolved


def _bool(array, ndim, name):
    a = np.asarray(array)
    if a.dtype != np.bool_ or a.ndim != ndim or not a.size:
        raise ValueError(f"{name} must be a nonempty {ndim}D boolean array")
    return a


def _box(box, name):
    if len(box) != 4 or any(type(v) not in (int, np.int32, np.int64) for v in box):
        raise ValueError(f"{name} must be four integer half-open YXYX bounds")
    y0, x0, y1, x1 = map(int, box)
    if y0 >= y1 or x0 >= x1:
        raise ValueError(f"{name} has empty or reversed bounds")
    return y0, x0, y1, x1


def select_depth_support(source_available_lhw, model_plane_source_contributors):
    """Transport support through the exporter's recorded depth interpolation.

Only strictly positive source weights matter. A missing/partial model plane
has false support everywhere, including a plane with interpolation mass 0.25.
This function does not perform XY resampling or certify original acquisition.
"""
    source = _bool(source_available_lhw, 3, "source_available_lhw")
    if not isinstance(model_plane_source_contributors, (list, tuple)) or not model_plane_source_contributors:
        raise ValueError("An explicit nonempty list of model-plane contributors is required")
    out = np.zeros((len(model_plane_source_contributors), *source.shape[1:]), dtype=bool)
    for k, contributors in enumerate(model_plane_source_contributors):
        if not isinstance(contributors, (list, tuple)):
            raise ValueError("Each model plane must list its source contributors")
        mass, indices = 0.0, []
        for contributor in contributors:
            layer, weight = contributor["source_layer"], contributor["weight"]
            if type(layer) not in (int, np.int32, np.int64) or type(weight) not in (int, float, np.float32, np.float64):
                raise ValueError("Source layers and weights must be explicit numbers")
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("Depth interpolation weights must be finite and nonnegative")
            mass += float(weight)
            if weight > 0:
                if not 0 <= layer < source.shape[0]:
                    raise ValueError("A contributing source layer lies outside the supplied stack")
                indices.append(int(layer))
        if indices and np.isclose(mass, 1.0, rtol=0, atol=1e-6):
            out[k] = np.all(source[indices], axis=0)
    return out


@dataclass(frozen=True)
class TileContribution:
    """One real or planned model call on one common unresampled raster.

    input_box_yxyx describes the complete input, including any artificial
    padding outside the source. output_box_yxyx describes the retained output.
    weights_yx must equal the actual blend weights on that retained output.
    Zero weights make no contribution. Both boxes are in the source XY frame.
    """
    input_box_yxyx: tuple[int, int, int, int]
    output_box_yxyx: tuple[int, int, int, int]
    weights_yx: np.ndarray


@dataclass
class SupportResult:
    context_support_yx: np.ndarray
    contribution_support_yx: np.ndarray
    source_all_required_yx: np.ndarray
    mesh_valid_yx: np.ndarray
    contributor_count_yx: np.ndarray
    tile_context_complete: np.ndarray
    record: dict

    def counts(self):
        return {
            "chart_pixels": int(self.context_support_yx.size),
            "mesh_valid_pixels": int(self.mesh_valid_yx.sum()),
            "source_all_required_pixels": int(self.source_all_required_yx.sum()),
            "mesh_and_source_pixels": int((self.mesh_valid_yx & self.source_all_required_yx).sum()),
            "contribution_supported_pixels": int(self.contribution_support_yx.sum()),
            "context_supported_pixels": int(self.context_support_yx.sum()),
            "tiles": int(len(self.tile_context_complete)),
            "complete_context_tiles": int(self.tile_context_complete.sum()),
        }


def tile_context_support(source_available_lhw, mesh_valid_yx, tiles, *, output_roi_yxyx=None):
    """Require every positive-weight contributor to have complete full-tile input.

    This deliberately uses the full input tile as a conservative receptive
    field. It makes no architectural claim about the minimal receptive field.
    Do not derive the tile list from model scores. Supply actual executed tiles,
    or label the result explicitly as a planned-schedule preflight.
    """
    source = _bool(source_available_lhw, 3, "source_available_lhw")
    mesh = _bool(mesh_valid_yx, 2, "mesh_valid_yx")
    if source.shape[1:] != mesh.shape:
        raise ValueError("Source availability and mesh validity grids differ")
    h, w = mesh.shape
    roi = _box((0, 0, h, w) if output_roi_yxyx is None else output_roi_yxyx, "output_roi_yxyx")
    ry0, rx0, ry1, rx1 = roi
    if not (0 <= ry0 < ry1 <= h and 0 <= rx0 < rx1 <= w):
        raise ValueError("Output ROI must lie within the source grid")
    source_all = source.all(axis=0)
    all_input = source_all & mesh
    count = np.zeros((ry1 - ry0, rx1 - rx0), dtype=np.uint32)
    bad_contributor = np.zeros(count.shape, dtype=bool)
    complete, descriptions = [], []
    for tile in tiles:
        iy0, ix0, iy1, ix1 = _box(tile.input_box_yxyx, "input_box_yxyx")
        oy0, ox0, oy1, ox1 = _box(tile.output_box_yxyx, "output_box_yxyx")
        if not (iy0 <= oy0 < oy1 <= iy1 and ix0 <= ox0 < ox1 <= ix1):
            raise ValueError("Retained output must lie inside the full input tile")
        weights = np.asarray(tile.weights_yx)
        real_numeric = np.issubdtype(weights.dtype, np.integer) or np.issubdtype(weights.dtype, np.floating)
        if weights.shape != (oy1 - oy0, ox1 - ox0) or not real_numeric:
            raise ValueError("Blend weights must match the retained output shape")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("Blend weights must be finite and nonnegative")
        inside = 0 <= iy0 < iy1 <= h and 0 <= ix0 < ix1 <= w
        good = bool(inside and all_input[iy0:iy1, ix0:ix1].all())
        complete.append(good)
        ya, xa, yb, xb = max(oy0, ry0), max(ox0, rx0), min(oy1, ry1), min(ox1, rx1)
        if ya < yb and xa < xb:
            positive = weights[ya - oy0:yb - oy0, xa - ox0:xb - ox0] > 0
            dest = np.s_[ya - ry0:yb - ry0, xa - rx0:xb - rx0]
            count[dest] += positive.astype(np.uint32)
            if not good:
                bad_contributor[dest] |= positive
        descriptions.append({
            "input_box_yxyx": [iy0, ix0, iy1, ix1],
            "output_box_yxyx": [oy0, ox0, oy1, ox1],
            "weights_shape_yx": list(weights.shape), "weights_dtype": str(weights.dtype),
            "weights_array_sha256": array_sha256(weights),
            "full_input_inside_source": bool(inside), "context_complete": good,
        })
    covered = count > 0
    record = {
        "schema": "scrollprize.tile-context-support.v1",
        "source_shape_lhw": list(source.shape), "output_roi_yxyx": list(roi),
        "source_available_array_sha256": array_sha256(source),
        "mesh_valid_array_sha256": array_sha256(mesh),
        "definition": "All source planes and mesh pixels in every positive-weight contributing full input tile must be supported; artificial edge context fails.",
        "context_scope": "conservative-full-input-tile", "xy_resampling": "none",
        "tiles": descriptions,
    }
    return SupportResult(
        covered & ~bad_contributor, covered,
        source_all[ry0:ry1, rx0:rx1].copy(), mesh[ry0:ry1, rx0:rx1].copy(),
        count, np.asarray(complete, dtype=bool), record,
    )


def review_support(context_support_yx, prediction_support_yx, score_yx=None):
    """Intersect with separately observed completed prediction support."""
    context = _bool(context_support_yx, 2, "context_support_yx")
    prediction = _bool(prediction_support_yx, 2, "prediction_support_yx")
    if context.shape != prediction.shape:
        raise ValueError("Context and prediction support grids differ")
    result = context & prediction
    if score_yx is not None:
        score = np.asarray(score_yx)
        if score.shape != result.shape or not np.issubdtype(score.dtype, np.floating):
            raise ValueError("Score must be a matching floating-point map")
        if not np.isfinite(score[result]).all():
            raise ValueError("A supported prediction is nonfinite")
    return result


def chart_areas(counts, spacing_um_yx=None):
    """Uniform chart area, never deduplicated physical papyrus area."""
    scale = None
    if spacing_um_yx is not None:
        spacing = np.asarray(spacing_um_yx, dtype=float)
        if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("Declared chart spacing must be a finite positive YX pair")
        scale = float(np.prod(spacing)) * 1e-8
    areas = {k.replace("_pixels", "_cm2"): None if scale is None else int(v) * scale
             for k, v in counts.items() if k.endswith("_pixels")}
    return {"spacing_um_yx": None if spacing_um_yx is None else list(map(float, spacing_um_yx)),
            "chart_areas": areas, "unique_papyrus_cm2": None, "on_sheet_cm2": None,
            "interpretation": "Chart area only; overlaps, mesh distortion and tissue identity are not resolved. m7/coherence is separate."}


def write_handoff_declaration(output_dir, context_support_yx, *, geometry, source_path,
                              mesh_path, normal_path, producer, definition,
                              shift_to_normal_sign, context_record):
    """Write the existing producer-context-support.v1 interface, with real assets.

    This checks file identity and declaration structure, not anatomical truth.
    The caller must have established acquisition semantics and exact model
    sampling. Planned Studio preflights do not invoke this function.
    """
    mask = _bool(context_support_yx, 2, "context_support_yx")
    if not mask.any():
        raise ValueError("No context-supported pixels")
    root = outside_git(output_dir)
    if root.exists():
        raise FileExistsError("Use a new private support directory")
    if context_record.get("tile_record_kind") != "executed":
        raise ValueError("A handoff declaration requires actual executed tile records")
    if (context_record.get("prediction_completed") is False
            or context_record.get("model_forward_calls") == 0
            or context_record.get("weights_loaded") is False):
        raise ValueError("A preflight with no completed prediction cannot be relabelled as executed")
    if context_record.get("acquisition_semantics_confirmed") is not True:
        raise ValueError("Source presence alone does not confirm acquisition semantics")
    acquisition_definition = context_record.get("acquisition_definition")
    if not isinstance(acquisition_definition, str) or not acquisition_definition.strip():
        raise ValueError("State the actual acquisition/FOV support derivation")
    if set(geometry) != set(GEOMETRY_KEYS):
        raise ValueError("Supply exactly the existing six declared_geometry fields")
    if any(not isinstance(geometry.get(k), str) or not HEX.fullmatch(geometry[k])
           for k in ("source_content_sha256", "mesh_sha256", "normal_sha256")):
        raise ValueError("Complete source, mesh and normal SHA-256 values are required")
    if any(not isinstance(v, str) or not v.strip() for v in (producer, definition, geometry["normal_convention"], geometry["plane_order"])):
        raise ValueError("Producer, sampling and normal definitions must be explicit")
    if type(shift_to_normal_sign) is not int or shift_to_normal_sign not in (-1, 1):
        raise ValueError("An explicit physical shift-to-normal sign is required")
    # Reuse the consumer's exact frame contract without importing any model.
    import importlib.util
    consumer_path = Path(__file__).resolve().parents[1] / "component_export/prepare_handoff.py"
    spec = importlib.util.spec_from_file_location("producer_support_handoff", consumer_path)
    consumer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consumer)
    consumer.validate_frame(geometry["frame"])
    assets = {"source_content_sha256": Path(source_path), "mesh_sha256": Path(mesh_path),
              "normal_sha256": Path(normal_path)}
    for key, path in assets.items():
        if sha256_file(path) != geometry[key]:
            raise ValueError("Actual asset bytes disagree with " + key)
    # Stage the whole directory. No consumable context.json is visible at its
    # final location until final asset checks and the receipt both succeed.
    json.dumps(context_record, allow_nan=False)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
    try:
        mask_path = staging / "acquisition_context.npy"
        with mask_path.open("xb") as stream:
            np.save(stream, mask, allow_pickle=False)
        declaration = {
            "schema": "scrollprize.producer-context-support.v1",
            "support_kind": "acquisition-and-model-context", "producer": producer,
            "definition": definition, "mask_sha256": sha256_file(mask_path),
            **geometry, "shift_to_normal_sign": shift_to_normal_sign,
            "context_record": context_record,
        }
        consumer.validate_producer(declaration, mask_path)
        with (staging / "context.json").open("x") as stream:
            json.dump(declaration, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        for key, path in assets.items():
            if sha256_file(path) != geometry[key]:
                raise ValueError("Source asset changed before completion")
        receipt = {"schema": "scrollprize.producer-support-receipt.v1", "status": "complete",
                   "mask_sha256": declaration["mask_sha256"],
                   "context_json_sha256": sha256_file(staging / "context.json"),
                   "supported_pixels": int(mask.sum()),
                   "interpretation": "Hashes verify byte identity. Geometry and acquisition definitions remain producer assertions; prediction coverage is separate."}
        with (staging / "receipt.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        if os.path.lexists(root):
            raise FileExistsError("Output appeared during producer preparation")
        os.rename(staging, root)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)
