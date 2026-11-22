#!/usr/bin/env python3
"""Bridge completed Hecate component exports to a three-depth comparison.

No model forwards. External producer acquisition/context support is mandatory;
this script checks bytes and declarations, not the truth of tissue or geometry.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np

HEX = re.compile(r"[0-9a-f]{64}\Z")
MUST = ("mesh_sha256", "normal_sha256", "normal_convention", "plane_order", "frame", "source_content_sha256")
STABLE_SAMPLING = (
    "blend", "stride", "tta", "resampled", "model_patch_zyx", "model_roi_y0_x0_y1_x1",
    "source_roi_y0_x0_y1_x1", "model_grid_shape_yx", "output_shape_yx",
    "source_shape_zyx", "source_layer_pitch_um", "source_layers_per_model_plane",
    "xy_pitch_um", "model_plane_order", "input_resampling_border_mode",
    "output_interpolation", "output_grid", "positive_shift_convention",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def outside_git(path):
    p = Path(path).resolve()
    for ancestor in (p, *p.parents):
        if os.path.lexists(ancestor / ".git"):
            raise ValueError("Private review outputs cannot be written inside a Git worktree")
    return p


def checked_member(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Component path must be relative")
    dest = (root / relative).resolve()
    if not dest.is_relative_to(root.resolve()):
        raise ValueError("Component path escapes export directory")
    return dest


def exact_hash(path, expected):
    if not isinstance(expected, str) or not HEX.fullmatch(expected) or sha(path) != expected:
        raise ValueError("Export payload or metadata SHA256 mismatch")


def validate_frame(frame):
    keys = ("acquisition", "axes", "units", "level", "voxel_um_xyz", "origin_native_xyz")
    if not isinstance(frame, dict) or not all(k in frame for k in keys):
        raise ValueError("Producer frame is incomplete")
    if frame["axes"] != "XYZ" or frame["units"] != "native_voxels" or type(frame["level"]) != int or frame["level"] != 0:
        raise ValueError("Requires level-zero scanner XYZ frame")
    if not isinstance(frame["acquisition"], str) or not frame["acquisition"]:
        raise ValueError("Missing acquisition")
    for key, positive in (("voxel_um_xyz", True), ("origin_native_xyz", False)):
        values = frame[key]
        if not isinstance(values, list) or len(values) != 3 or any(
            type(x) not in (float, int) or not np.isfinite(x) or (positive and x <= 0) for x in values
        ):
            raise ValueError("Invalid scanner frame vector")


def validate_producer(p, mask_path):
    if not isinstance(p, dict) or p.get("schema") != "scrollprize.producer-context-support.v1":
        raise ValueError("External producer declaration required")
    if p.get("support_kind") != "acquisition-and-model-context":
        raise ValueError("A tissue/CT-nonzero or prediction mask is not acquisition+context support")
    if not isinstance(p.get("producer"), str) or not p["producer"].strip() or not isinstance(p.get("definition"), str) or not p["definition"].strip():
        raise ValueError("Actual producer and context definition must be stated")
    for key in (*MUST[0:2], MUST[-1], "mask_sha256"):
        if not isinstance(p.get(key), str) or not HEX.fullmatch(p[key]):
            raise ValueError("Missing verified producer hash: " + key)
    for key in ("normal_convention", "plane_order"):
        if not isinstance(p.get(key), str) or not p[key].strip():
            raise ValueError("Missing producer direction definition")
    if type(p.get("shift_to_normal_sign")) is not int or p["shift_to_normal_sign"] not in (-1, 1):
        raise ValueError("Explicit shift-to-normal sign +1/-1 is required")
    validate_frame(p.get("frame"))
    exact_hash(mask_path, p["mask_sha256"])


def prepare(export_dir, acquisition_mask_path, producer_path, output_dir, model_index, direction):
    export_dir = Path(export_dir).resolve()
    acquisition_mask_path = Path(acquisition_mask_path).resolve()
    producer_path = Path(producer_path).resolve()
    output_dir = outside_git(output_dir)
    if output_dir.exists():
        raise FileExistsError("Review directory must be new")
    receipt_path = export_dir / "receipt.json"
    original_hashes = [(receipt_path, sha(receipt_path)), (producer_path, sha(producer_path))]
    receipt = load_json(receipt_path)
    p = load_json(producer_path)
    validate_producer(p, acquisition_mask_path)
    original_hashes.append((acquisition_mask_path, p["mask_sha256"]))
    if receipt.get("schema") != "scrollprize.inference-components.v1" or receipt.get("status") != "complete":
        raise ValueError("Run receipt absent or incomplete")
    refs = receipt.get("components")
    if not isinstance(refs, list) or not refs:
        raise ValueError("No completed inference components")
    all_metadata = []
    for ref in refs:
        dirname = ref.get("directory")
        component_json = checked_member(export_dir, f"{dirname}/component.json")
        exact_hash(component_json, ref.get("component_json_sha256"))
        original_hashes.append((component_json, ref["component_json_sha256"]))
        m = load_json(component_json)
        if m.get("schema") != "scrollprize.inference-component.v1" or m.get("status") != "complete":
            raise ValueError("Incomplete component")
        if m.get("model_index") != ref.get("model_index") or m.get("direction") != ref.get("direction") or m.get("shift_model_planes") != ref.get("shift_model_planes"):
            raise ValueError("Receipt/component index mismatch")
        directory = component_json.parent
        for filename in ("probability.npy", "prediction_support.npy", "complete_depth_support.npy"):
            a = directory / filename
            expected = m.get("files_sha256", {}).get(filename)
            exact_hash(a, expected)
            original_hashes.append((a, expected))
        all_metadata.append((m, directory))
    chosen = [(m, directory) for m, directory in all_metadata if m["model_index"] == model_index and m["direction"] == direction]
    if len(chosen) != 3:
        raise ValueError("Need exactly three saved depths from one checkpoint and direction")
    offsets = [m["shift_model_planes"] for m, _ in chosen]
    if any(type(x) not in (int, float) or not np.isfinite(x) for x in offsets) or len(set(offsets)) != 3 or offsets.count(0) != 1:
        raise ValueError("Depth offsets must be distinct with exactly one zero")
    # Sorting is descriptive; no adaptive depth selection.
    chosen.sort(key=lambda row: row[0]["shift_model_planes"])
    reference, _ = chosen[0]
    for m, _ in chosen:
        s = m["sampling"]
        if not m.get("full_depth_context") or not s.get("physical_pitch_known") or s.get("xy_tile_padding_present"):
            raise ValueError("Incomplete depth, physical pitch or unresolved XY padding")
        if any(not np.isfinite(x) or not np.isclose(x, 1, rtol=0, atol=1e-6) for x in s.get("model_plane_source_weight_masses", [])) or not s.get("model_plane_source_weight_masses"):
            raise ValueError("Incomplete source-plane interpolation mass")
        if m.get("declared_geometry") != {k: p[k] for k in MUST}:
            raise ValueError("Producer and inference geometry are not the same declared source")
        if m["checkpoint_sha256"] != reference["checkpoint_sha256"] or m["model"] != reference["model"] or m["shape_yx"] != reference["shape_yx"]:
            raise ValueError("Different model checkpoint or output frame")
        if any(s.get(k) != reference["sampling"].get(k) for k in STABLE_SAMPLING):
            raise ValueError("Depths differ in shared inference sampling")
        if s.get("effective_depth_pitch_um") != reference["sampling"].get("effective_depth_pitch_um"):
            raise ValueError("Inconsistent effective physical depth step")
    shape = tuple(reference["shape_yx"])
    external = np.load(acquisition_mask_path, allow_pickle=False)
    if external.dtype != np.bool_ or external.shape != shape:
        raise ValueError("Producer mask must be boolean and on the exact output grid")
    if not external.any():
        raise ValueError("No acquisition/context-supported pixels")
    masks = []
    for m, directory in chosen:
        supported = np.load(directory / "complete_depth_support.npy", allow_pickle=False)
        prediction = np.load(directory / "probability.npy", mmap_mode="r", allow_pickle=False)
        if supported.dtype != np.bool_ or supported.shape != shape or prediction.shape != shape or not np.issubdtype(prediction.dtype, np.floating):
            raise ValueError("Invalid source support or score dtype/shape")
        if not np.isfinite(prediction[supported & external]).all():
            raise ValueError("Nonfinite scores marked valid")
        masks.append(supported & external)
    if not np.logical_and.reduce(masks).any():
        raise ValueError("No shared valid coverage across three depths")
    for path, expected in original_hashes:
        exact_hash(path, expected)
    s = reference["sampling"]
    provenance = {
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "mesh_sha256": p["mesh_sha256"], "normal_sha256": p["normal_sha256"],
        "normal_convention": p["normal_convention"], "plane_order": p["plane_order"],
        "frame": p["frame"], "xy_pitch_um": s["xy_pitch_um"],
        "depth_pitch_um": s["effective_depth_pitch_um"],
        "inference": {"model": reference["model"], "direction": direction, "shared_sampling": {k: s.get(k) for k in STABLE_SAMPLING}},
        "preprocess": {"source_content_sha256": p["source_content_sha256"],
                       "source_layer_pitch_um": s["source_layer_pitch_um"],
                       "model_plane_order": s["model_plane_order"]},
    }
    # Validate the exact downstream contract before writing anything.
    import importlib.util
    consumer = Path(__file__).resolve().parents[1] / "depth_screening" / "combine.py"
    if consumer.exists():
        spec = importlib.util.spec_from_file_location("depth_screening_combine", consumer)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.check_provenance([provenance] * 3)
    output_dir.mkdir(parents=True)
    inputs = {"maps": []}
    hashes = {}
    for i, ((m, directory), mask) in enumerate(zip(chosen, masks)):
        filename = f"valid-depth-{i}.npy"
        with (output_dir / filename).open("xb") as f:
            np.save(f, mask, allow_pickle=False)
        hashes[filename] = sha(output_dir / filename)
        inputs["maps"].append({
            "offset_planes": m["shift_model_planes"] * p["shift_to_normal_sign"],
            "map": str((directory / "probability.npy").resolve()),
            "mask": filename, "provenance": provenance,
        })
    (output_dir / "inputs.json").write_text(json.dumps(inputs, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")
    hashes["inputs.json"] = sha(output_dir / "inputs.json")
    handoff = {"schema": "scrollprize.component-depth-handoff.v1", "status": "complete",
               "selected_model_index": model_index, "direction": direction,
               "offset_sign": p["shift_to_normal_sign"], "offsets_model_planes": [m["shift_model_planes"] for m, _ in chosen],
               "source_receipt_sha256": original_hashes[0][1], "producer_declaration_sha256": original_hashes[1][1],
               "acquisition_context_sha256": p["mask_sha256"], "output_sha256": hashes,
               "valid_pixels_per_depth": [int(v.sum()) for v in masks],
               "common_valid_pixels": int(np.logical_and.reduce(masks).sum()),
               "provenance_claim": "Producer assertions are declared, not independently certified. Verified hashes establish byte identity, not anatomical validity.",
               "next_step": "Run depth_screening/combine.py --manifest inputs.json with private output folder; inspect individual depths, central and median before any max."}
    # A completed handoff must refer to the same immutable bytes checked before writing.
    for path, expected in original_hashes:
        exact_hash(path, expected)
    (output_dir / "handoff.json").write_text(json.dumps(handoff, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")
    return handoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--acquisition-context-mask", required=True)
    parser.add_argument("--producer-declaration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-index", required=True, type=int)
    parser.add_argument("--direction", required=True)
    args = parser.parse_args()
    result = prepare(args.export_dir, args.acquisition_context_mask, args.producer_declaration,
                     args.output_dir, args.model_index, args.direction)
    print(json.dumps({"status": result["status"], "common_valid_pixels": result["common_valid_pixels"]}))


if __name__ == "__main__":
    main()
