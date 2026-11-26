"""Read-only semantic validation of an existing Studio native evidence bundle.

This module does not execute a model, write a receipt, dispatch a job, or certify
anatomical support. ``valid`` means that the declared arrays and the supported
Studio tiling semantics agree. It never means production COMPLETE or discovery.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


SCHEMA = "funnel.native_evidence.v1"
SHA = re.compile(r"[a-f0-9]{64}\Z")
ROLES = {
    "input": (np.dtype("uint8"), "LYX"),
    "source_depth": (np.dtype("bool"), "LYX"),
    "probability": (np.dtype("float32"), "YX"),
    "output_valid": (np.dtype("bool"), "YX"),
    "source_valid": (np.dtype("bool"), "YX"),
    "context_valid": (np.dtype("bool"), "YX"),
}
RECIPE_HASHES = ("model_sha256", "pipeline_sha256", "runner_sha256", "model_code_sha256")
SCOPE = (
    "Consistency of existing native files and Studio input/output support only; "
    "not worker execution, same-sheet anatomy, calibrated search sensitivity, or confirmed text."
)


class InvalidEvidence(ValueError):
    def __init__(self, code, path, message):
        super().__init__(message)
        self.issue = {"code": code, "path": path, "message": message}


def _require(condition, code, path, message):
    if not condition:
        raise InvalidEvidence(code, path, message)


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(obj):
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _pair(value):
    return isinstance(value, list) and len(value) == 2 and all(_integer(v) for v in value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _metadata(manifest):
    _require(isinstance(manifest, dict), "MANIFEST_TYPE", "manifest", "Expected a JSON object")
    required = {
        "schema", "name", "frame", "recipe", "depth_samples", "depth_units",
        "signed_offsets_um", "depth_unknown_reason", "arrays", "cost",
        "calibration", "scientific_status", "support_basis",
    }
    missing = required - set(manifest)
    _require(not missing, "MANIFEST_FIELDS", "manifest", f"Missing required fields: {sorted(missing)}")
    _require(manifest["schema"] == SCHEMA, "SCHEMA", "schema", "Unsupported native evidence schema")
    _require(_text(manifest["name"]), "NAME", "name", "A nonempty evidence name is required")
    # Reject non-finite metadata independently of any probability-array check.
    try:
        _canonical(manifest)
    except (ValueError, TypeError) as exc:
        raise InvalidEvidence("JSON_METADATA", "manifest", str(exc)) from exc

    frame = manifest["frame"]
    _require(isinstance(frame, dict), "FRAME", "frame", "Expected a frame object")
    for key in ("scroll", "source_volume", "chart_id", "registration_status"):
        _require(_text(frame.get(key)), "FRAME", f"frame.{key}", "Explicit frame value or uncertainty is required")
    _require(_pair(frame.get("origin_yx")), "ORIGIN", "frame.origin_yx", "Expected two nonnegative source-pixel indices")

    recipe = manifest["recipe"]
    _require(isinstance(recipe, dict), "RECIPE", "recipe", "Expected a recipe object")
    for key in RECIPE_HASHES:
        value = recipe.get(key)
        _require(isinstance(value, str) and SHA.fullmatch(value), "RECIPE_HASH", f"recipe.{key}", "Expected a full lowercase SHA256")
    for key in ("normalization", "source_domain"):
        _require(_text(recipe.get(key)) or isinstance(recipe.get(key), dict) and bool(recipe[key]),
                 "RECIPE", f"recipe.{key}", "An explicit nonempty declaration is required")
    _require(recipe.get("precision") in ("fp32", "fp16", "bf16"), "PRECISION", "recipe.precision", "Explicit supported precision required")
    _require(_integer(recipe.get("stride"), 1) and recipe["stride"] <= 96,
             "STRIDE", "recipe.stride", "This adapter supports integer strides 1..96")
    _require(recipe.get("blend") in ("none", "hann"), "BLEND", "recipe.blend", "Unsupported blend")
    # The nonoverlap producer used stride 96; lower strides with overwrite
    # semantics need a separate recipe/validator instead of pretending to blend.
    _require(recipe["blend"] != "none" or recipe["stride"] == 96,
             "OVERWRITE_RECIPE", "recipe", "Unblended overlapping windows are outside this adapter's scope")
    for key, value in (("input_window", 128), ("output_window", 96), ("halo", 16)):
        _require(type(recipe.get(key)) is int and recipe[key] == value,
                 "WINDOW_GEOMETRY", f"recipe.{key}", f"Expected {value} for this Studio adapter")
    _require(recipe.get("tiling_policy") == "studio_v1_end_anchored",
             "TILING_POLICY", "recipe.tiling_policy", "Unsupported tiling policy")
    _require(recipe.get("origin_yx") == frame["origin_yx"],
             "ORIGIN_MISMATCH", "recipe.origin_yx", "Recipe and frame origins must agree")
    expected_phase = [v % recipe["stride"] for v in frame["origin_yx"]]
    _require(_pair(recipe.get("phase_yx")) and recipe["phase_yx"] == expected_phase,
             "PHASE_MISMATCH", "recipe.phase_yx", "Phase must record source origin modulo stride")
    if "direction" in recipe:
        _require(recipe["direction"] in ("asc", "desc", "as_recorded"),
                 "DIRECTION", "recipe.direction", "Unsupported direction label")

    _require(manifest["depth_units"] == "source_plane_index", "DEPTH_UNITS", "depth_units", "Depth indices are not signed physical offsets")
    depths = manifest["depth_samples"]
    if depths is None:
        _require(_text(manifest["depth_unknown_reason"]), "DEPTH_UNKNOWN", "depth_unknown_reason", "Unknown depths need an explicit reason")
    else:
        _require(isinstance(depths, list) and len(depths) == 16 and all(_integer(x) for x in depths)
                 and len(set(depths)) == 16,
                 "DEPTH_SAMPLES", "depth_samples", "Expected 16 distinct explicit source-plane indices")
        _require(manifest["depth_unknown_reason"] is None,
                 "DEPTH_CONTRADICTION", "depth_unknown_reason", "Known indices cannot also be declared unknown")
    offsets = manifest["signed_offsets_um"]
    _require(offsets is None or isinstance(offsets, list) and len(offsets) == 16
             and all(type(x) in (int, float) and math.isfinite(x) for x in offsets),
             "SIGNED_OFFSETS", "signed_offsets_um", "Use null for unmeasured physical offsets")

    cost = manifest["cost"]
    _require(isinstance(cost, dict), "COST", "cost", "Expected accounting object")
    for key in ("cpu_ms", "gpu_ms", "io_bytes"):
        _require(key in cost and (cost[key] is None or _integer(cost[key])),
                 "COST", f"cost.{key}", "Unmeasured cost must be null; measured integer units must be nonnegative")
    _require("map_elapsed_seconds" in cost and (cost["map_elapsed_seconds"] is None or _number(cost["map_elapsed_seconds"])),
             "COST", "cost.map_elapsed_seconds", "Use a measured nonnegative duration or null")
    for key in ("model_windows", "sentinel_forwards"):
        _require(_integer(cost.get(key)), "COST", f"cost.{key}", "Explicit nonnegative forward count is required")

    calibration = manifest["calibration"]
    _require(isinstance(calibration, dict), "CALIBRATION", "calibration", "Expected explicit calibration limitations")
    _require("identity_sentinel_pass" in calibration and
             (calibration["identity_sentinel_pass"] is None or type(calibration["identity_sentinel_pass"]) is bool),
             "CALIBRATION", "calibration.identity_sentinel_pass", "Identity check must be boolean or unknown")
    for key in ("semantic_control_status", "training_exposure"):
        _require(_text(calibration.get(key)), "CALIBRATION", f"calibration.{key}", "Explicit control scope is required")
    _require(isinstance(manifest["scientific_status"], str) and
             manifest["scientific_status"].lower() in ("unconfirmed", "known_public_control"),
             "SCIENTIFIC_STATUS", "scientific_status", "This adapter cannot admit a discovery claim")
    _require(manifest["support_basis"] in ("EXPLICIT_PREPARED_MASK", "IMPLICIT_PREPARED_PRESENCE"),
             "SUPPORT_BASIS", "support_basis", "Prepared input support must state its actual provenance")


def _arrays(manifest, root):
    descriptors = manifest["arrays"]
    _require(isinstance(descriptors, dict) and set(descriptors) == set(ROLES),
             "ARRAY_ROLES", "arrays", "All six native array roles are required, without display-image substitution")
    root = Path(root).resolve(strict=True)
    _require(root.is_dir(), "ROOT", "root", "Expected an existing local bundle directory")
    arrays, files = {}, {}
    for role, (expected_dtype, expected_axes) in ROLES.items():
        row = descriptors[role]
        field = f"arrays.{role}"
        if role == "source_depth" and isinstance(row, dict) and "derivation" in row:
            _require(row["derivation"] == "all_true_as_executed" and
                     manifest["support_basis"] == "IMPLICIT_PREPARED_PRESENCE",
                     "SUPPORT_DERIVATION", field, "Only the explicit legacy all-true execution assumption is supported")
            required = {"derivation", "evidence_path", "evidence_sha256", "dtype", "shape", "axes"}
            _require(required <= set(row) and not {"path", "sha256"} & set(row),
                     "SUPPORT_DERIVATION", field, "Derived support requires the original receipt, not a fabricated array path")
            _require(_text(row["evidence_path"]) and not Path(row["evidence_path"]).is_absolute(),
                     "ARRAY_PATH", field, "Receipt path must be relative to the bundle")
            path = (root / row["evidence_path"]).resolve(strict=True)
            _require(path.is_relative_to(root) and path.is_file(),
                     "ARRAY_PATH", field, "Receipt path escapes the bundle or is not a file")
            _require(isinstance(row["evidence_sha256"], str) and SHA.fullmatch(row["evidence_sha256"])
                     and _file_sha(path) == row["evidence_sha256"],
                     "HASH_MISMATCH", field, "Original execution receipt hash does not match")
            evidence = json.loads(path.read_text(encoding="utf-8"))
            _require(isinstance(evidence, dict) and "source_support_sha256" in evidence
                     and evidence["source_support_sha256"] is None
                     and isinstance(evidence.get("job"), dict) and not evidence["job"].get("source_support"),
                     "SUPPORT_DERIVATION", field, "Original receipt does not document an implicit all-true source assumption")
            _require(evidence.get("input_sha256") == descriptors["input"]["sha256"]
                     and evidence.get("input_shape") == list(arrays["input"].shape),
                     "SUPPORT_DERIVATION", field, "Original receipt belongs to a different prepared input")
            _require(row["axes"] == "LYX" and np.dtype(row["dtype"]) == np.dtype("bool")
                     and row["shape"] == list(arrays["input"].shape),
                     "SUPPORT_DERIVATION", field, "Derived support must align exactly with the native input")
            arrays[role] = np.ones(arrays["input"].shape, dtype=bool)
            files[role] = path
            continue
        _require(isinstance(row, dict) and {"path", "sha256", "dtype", "shape", "axes"} <= set(row),
                 "ARRAY_DESCRIPTOR", field, "Incomplete array descriptor")
        if role == "source_depth":
            _require(manifest["support_basis"] == "EXPLICIT_PREPARED_MASK",
                     "SUPPORT_BASIS", "support_basis", "An explicit support array must not be labeled implicit")
        _require(_text(row["path"]) and not Path(row["path"]).is_absolute(),
                 "ARRAY_PATH", f"{field}.path", "Array paths must be relative to root")
        path = (root / row["path"]).resolve(strict=True)
        _require(path.is_relative_to(root) and path.is_file(),
                 "ARRAY_PATH", f"{field}.path", "Array path escapes the bundle or is not a file")
        _require(isinstance(row["sha256"], str) and SHA.fullmatch(row["sha256"]),
                 "ARRAY_HASH", f"{field}.sha256", "Expected full SHA256")
        _require(_file_sha(path) == row["sha256"], "HASH_MISMATCH", field, "Native file bytes do not match the descriptor")
        _require(row["axes"] == expected_axes, "ARRAY_AXES", f"{field}.axes", f"Expected axes {expected_axes}")
        _require(isinstance(row["shape"], list) and all(_integer(x, 1) for x in row["shape"]),
                 "ARRAY_SHAPE", f"{field}.shape", "Shape must contain positive integer dimensions")
        try:
            declared_dtype = np.dtype(row["dtype"])
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
        except (ValueError, TypeError, OSError) as exc:
            raise InvalidEvidence("ARRAY_LOAD", field, str(exc)) from exc
        if not isinstance(arr, np.ndarray):
            if hasattr(arr, "close"):
                arr.close()
            raise InvalidEvidence("ARRAY_FORMAT", field, "Expected one NPY array, not an archive")
        _require(arr.dtype == expected_dtype and declared_dtype == arr.dtype,
                 "ARRAY_DTYPE", field, f"Actual and declared dtype must both be {expected_dtype}")
        _require(list(arr.shape) == row["shape"], "ARRAY_SHAPE", field, "Loaded shape differs from the descriptor")
        arrays[role], files[role] = arr, path
    shape = arrays["input"].shape
    _require(len(shape) == 3 and shape[0] == 16, "INPUT_SHAPE", "arrays.input", "Studio requires 16 planes in LYX order")
    _require(arrays["source_depth"].shape == shape, "MASK_ALIGNMENT", "arrays.source_depth", "Source support must align with all input planes")
    for role in set(ROLES) - {"input", "source_depth"}:
        _require(arrays[role].shape == shape[1:], "MASK_ALIGNMENT", f"arrays.{role}", "Map/mask must align with native YX input coordinates")
    return arrays, files


def _derive(input_stack, source_depth, recipe):
    """Recompute masks from arrays, not producer mask hashes or copied statistics."""
    h, w = input_stack.shape[-2:]
    empty = np.ones((h, w), dtype=bool)
    for plane in input_stack:
        empty &= plane == 0
    source = np.all(source_depth, axis=0)
    stride, blend = recipe["stride"], recipe["blend"]
    ys = set(range(0, h, stride))
    xs = set(range(0, w, stride))
    if blend == "hann":
        ys.add(max(h - 96, 0))
        xs.add(max(w - 96, 0))
    covered = np.zeros((h, w), dtype=bool)
    bad_context = np.zeros((h, w), dtype=bool)
    windows, unsafe_windows = 0, 0
    for y in sorted(ys):
        for x in sorted(xs):
            region = np.s_[y:y + 96, x:x + 96]
            if empty[region].all():
                continue
            windows += 1
            covered[region] = True
            y0, x0, y1, x1 = y - 16, x - 16, y + 112, x + 112
            safe = 0 <= y0 and 0 <= x0 and y1 <= h and x1 <= w
            if safe:
                safe = bool(source[y0:y1, x0:x1].all())
            if not safe:
                unsafe_windows += 1
                bad_context[region] = True
    output = covered & ~empty
    context = output & source & ~bad_context
    return {"output_valid": output, "source_valid": source, "context_valid": context}, {
        "model_windows": windows,
        "unsafe_context_windows": unsafe_windows,
        "input_shape": list(input_stack.shape),
        "all_zero_input_pixels": int(empty.sum()),
        "covered_pixels": int(covered.sum()),
    }


def _rehash(manifest, files):
    for role, path in files.items():
        descriptor = manifest["arrays"][role]
        expected = descriptor["evidence_sha256"] if "derivation" in descriptor else descriptor["sha256"]
        _require(_file_sha(path) == expected,
                 "ARTIFACT_CHANGED", f"arrays.{role}", "Artifact changed during read-only validation")


def validate_native(manifest, root):
    """Validate one existing evidence observation without changing any file.

    Unknown source depths or physical registration remain explicit limitations.
    They do not prevent a byte/mask audit, but cannot satisfy an exact-depth pair.
    Structured errors are returned instead of manufacturing completion receipts.
    """
    result = {
        "schema": "funnel.native_validation.v1", "valid": False,
        "errors": [], "warnings": [], "derived": {},
        "scientific_admission": "NOT_ESTABLISHED", "production_complete": False,
        "scope": SCOPE,
    }
    try:
        _metadata(manifest)
        arrays, files = _arrays(manifest, root)
        masks, derived = _derive(arrays["input"], arrays["source_depth"], manifest["recipe"])
        for role, expected in masks.items():
            mismatch = int(np.count_nonzero(arrays[role] != expected))
            _require(mismatch == 0, "MASK_SEMANTICS", f"arrays.{role}", f"{mismatch} pixels disagree with recomputed support")
            derived[f"{role}_pixels"] = int(expected.sum())
        _require(manifest["cost"]["model_windows"] == derived["model_windows"],
                 "WINDOW_COUNT", "cost.model_windows", "Reported model forwards disagree with nonempty tiling windows")
        supported = arrays["probability"][masks["output_valid"]]
        _require(bool(np.isfinite(supported).all()), "NONFINITE_SCORE", "arrays.probability", "Supported probabilities must be finite")
        _require(bool(((supported >= 0) & (supported <= 1)).all()),
                 "PROBABILITY_RANGE", "arrays.probability", "Supported probabilities must lie in [0,1]")
        _rehash(manifest, files)
        derived.update({
            "manifest_sha256": _digest(manifest), "recipe_sha256": _digest(manifest["recipe"]),
            "frame_sha256": _digest(manifest["frame"]),
            "depth_indices_known": manifest["depth_samples"] is not None,
            "signed_physical_offsets_known": manifest["signed_offsets_um"] is not None,
            "reported_cost": manifest["cost"],
            "resource_accounting_complete": all(manifest["cost"][k] is not None for k in ("cpu_ms", "gpu_ms", "io_bytes")),
            "identity_sentinel_pass": manifest["calibration"]["identity_sentinel_pass"],
            "semantic_control_status": manifest["calibration"]["semantic_control_status"],
            "support_basis": manifest["support_basis"],
            "native_acquisition_verified": False,
        })
        result["derived"] = derived
        result["warnings"].append({"code": "NO_PRODUCTION_ADMISSION", "message": "Historical import does not attest a newly authorized worker run or matched semantic calibration."})
        if manifest["support_basis"] == "IMPLICIT_PREPARED_PRESENCE":
            result["warnings"].append({"code": "ASSUMED_PREPARED_SUPPORT", "message": "All-true support reproduces the original prepared-input assumption; native CT acquisition support was not measured."})
        if manifest["depth_samples"] is None:
            result["warnings"].append({"code": "UNKNOWN_DEPTHS", "message": manifest["depth_unknown_reason"]})
        if manifest["signed_offsets_um"] is None:
            result["warnings"].append({"code": "UNKNOWN_PHYSICAL_DEPTHS", "message": "Source-plane indices are not independently verified signed physical depths."})
        if not derived["resource_accounting_complete"]:
            result["warnings"].append({"code": "PARTIAL_COST_ACCOUNTING", "message": "Unmeasured costs remain null; elapsed time is not measured CPU consumption."})
        if manifest["calibration"]["identity_sentinel_pass"] is not True:
            result["warnings"].append({"code": "IDENTITY_NOT_PASSED", "message": "Numerical identity sentinel is absent or failed."})
        if derived["context_valid_pixels"] == 0:
            result["warnings"].append({"code": "NO_ADMISSIBLE_CONTEXT", "message": "No output pixel has full real-source context; this is uninformative for text absence."})
        result["valid"] = True
    except InvalidEvidence as exc:
        result["errors"].append(exc.issue)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        result["errors"].append({"code": "INVALID_BUNDLE", "path": "manifest", "message": str(exc)})
    return result


def assess_pair(manifest_a, manifest_b, root):
    """Check exact same-frame reversal, not independent anatomical replication."""
    validations = [validate_native(m, root) for m in (manifest_a, manifest_b)]
    result = {
        "schema": "funnel.native_pair.v1", "valid": False, "errors": [],
        "members_valid": [v["valid"] for v in validations], "derived": {},
        "scientific_admission": "NOT_ESTABLISHED", "production_complete": False,
        "independent_replication": False,
        "scope": "Exact same-input depth reversal only; not recto/verso identity, independent coverage, or search sensitivity.",
    }
    if not all(result["members_valid"]):
        for i, validation in enumerate(validations):
            for issue in validation["errors"]:
                result["errors"].append({**issue, "member": i})
        return result
    try:
        _require(manifest_a["frame"] == manifest_b["frame"], "PAIR_FRAME", "frame", "Pair members must name the exact same source frame and crop")
        _require(manifest_a["support_basis"] == manifest_b["support_basis"],
                 "PAIR_SUPPORT_BASIS", "support_basis", "Prepared source support must have the same meaning in both members")
        recipes = [{k: v for k, v in m["recipe"].items() if k != "direction"} for m in (manifest_a, manifest_b)]
        _require(recipes[0] == recipes[1], "PAIR_RECIPE", "recipe", "Only the recorded direction label may differ between paired recipes")
        a_depth, b_depth = manifest_a["depth_samples"], manifest_b["depth_samples"]
        _require(a_depth is not None and b_depth is not None,
                 "PAIR_DEPTH_UNKNOWN", "depth_samples", "Unknown native depths cannot certify an exact reversal")
        _require(a_depth == b_depth[::-1], "PAIR_DEPTH_SHIFT", "depth_samples", "Depth lists must be exact reversals, without a one-plane shift")
        a_offsets, b_offsets = manifest_a["signed_offsets_um"], manifest_b["signed_offsets_um"]
        _require((a_offsets is None and b_offsets is None) or a_offsets is not None and b_offsets is not None and a_offsets == b_offsets[::-1],
                 "PAIR_PHYSICAL_DEPTH", "signed_offsets_um", "Physical offsets, when known, must reverse consistently")
        arrays_a, files_a = _arrays(manifest_a, root)
        arrays_b, files_b = _arrays(manifest_b, root)
        _require(arrays_a["input"].shape == arrays_b["input"].shape,
                 "PAIR_SHAPE", "arrays.input", "Paired native shapes differ")
        _require(np.array_equal(arrays_a["input"], arrays_b["input"][::-1]),
                 "PAIR_INPUT", "arrays.input", "Actual input arrays are not exact depth reversals")
        _require(np.array_equal(arrays_a["source_depth"], arrays_b["source_depth"][::-1]),
                 "PAIR_SOURCE_SUPPORT", "arrays.source_depth", "Actual depth support masks are not exact reversals")
        _rehash(manifest_a, files_a)
        _rehash(manifest_b, files_b)
        result["derived"] = {
            "exact_indices_reversed": True, "exact_input_reversed": True,
            "exact_source_support_reversed": True,
            "member_manifest_sha256": [_digest(manifest_a), _digest(manifest_b)],
            "context_valid_pixels": validations[0]["derived"]["context_valid_pixels"],
        }
        result["valid"] = True
    except InvalidEvidence as exc:
        result["errors"].append(exc.issue)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        result["errors"].append({"code": "INVALID_PAIR", "path": "pair", "message": str(exc)})
    return result


def main(argv=None):
    """Print an audit result; no artifact or original evidence is written."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Native evidence JSON manifest")
    parser.add_argument("--root", type=Path, required=True, help="Common root of referenced original files")
    parser.add_argument("--pair", type=Path, help="Optional manifest of an exact depth reversal")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.pair is None:
            result = validate_native(manifest, args.root)
        else:
            other = json.loads(args.pair.read_text(encoding="utf-8"))
            result = assess_pair(manifest, other, args.root)
    except (OSError, ValueError) as exc:
        result = {
            "schema": "funnel.native_validation.v1", "valid": False,
            "errors": [{"code": "MANIFEST_LOAD", "path": "manifest", "message": str(exc)}],
            "scientific_admission": "NOT_ESTABLISHED", "production_complete": False,
            "scope": SCOPE,
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
