#!/usr/bin/env python3
"""CPU-only Studio context preflight from retained render-support arrays.

No weights are loaded, no model is imported, and no inference is performed.
Only the existing Studio Render and deterministic tiling helpers are used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

try:
    from .support import (TileContribution, array_sha256, chart_areas, outside_git,
                          sha256_file, tile_context_support)
except ImportError:
    from support import (TileContribution, array_sha256, chart_areas, outside_git,
                         sha256_file, tile_context_support)

HERE = Path(__file__).resolve().parent
STUDIO_PATH = HERE.parent / "studio_t4/studio_t4.py"


def studio_module():
    spec = importlib.util.spec_from_file_location("producer_support_studio", STUDIO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_studio_context(render, source_available_lhw, mesh_valid_yx, *, mesh_index,
                           order, shift=0, stride=48):
    """Return (SupportResult, plan) using Studio's exact current tile schedule.

    The result is a planned source-context mask. It does not establish actual
    inference coverage or original acquisition validity. Studio's reflect-padded
    tiles are retained in the plan but fail the full-real-context mask.
    """
    studio = studio_module()
    if stride not in (48, 96):
        raise ValueError("Studio supports strides 48 or 96")
    source = np.asarray(source_available_lhw)
    mesh = np.asarray(mesh_valid_yx)
    expected = (render.n_layers, *render.shape)
    if source.dtype != np.bool_ or source.shape != expected:
        raise ValueError("source_available_lhw must be boolean on the complete render LHW grid")
    if mesh.dtype != np.bool_ or mesh.shape != tuple(render.shape):
        raise ValueError("mesh_valid_yx must be boolean on the render YX grid")
    ks = studio.plane_indices(render.n_layers, mesh_index, order, shift)
    stack = render.stack(ks)
    studio.check_planes_nonzero(stack, ks)  # The production CLI has this same gate.
    h, w = render.shape
    blend = "hann" if stride == 48 else "none"
    kernel = studio.hann96() if blend == "hann" else np.ones((96, 96), np.float32)
    ys, xs = studio.window_starts(h, w, stride, blend)
    empty = (stack == 0).all(axis=0)
    tiles = []
    for y in ys:
        for x in xs:
            if empty[y:y + 96, x:x + 96].all():
                continue
            hh, ww = min(96, h - y), min(96, w - x)
            tiles.append(TileContribution(
                (y - 16, x - 16, y + 112, x + 112), (y, x, y + hh, x + ww),
                kernel[:hh, :ww],
            ))
    result = tile_context_support(source[ks], mesh, tiles)
    # Production validity_mask also excludes all-zero central pixels. Keep that
    # historical intensity proxy separate from source and full-context support.
    legacy_validity = studio.validity_mask(stack, stride, blend)
    if not np.array_equal(result.contribution_support_yx, studio.covered_mask(stack, stride, blend)):
        raise ValueError("Preflight contribution geometry differs from the Studio helper")
    plan = {
        "schema": "scrollprize.studio-context-plan.v1", "tile_record_kind": "planned",
        "studio_code_sha256": sha256_file(STUDIO_PATH),
        "studio_release_commit": studio.RELEASE_COMMIT,
        "expected_checkpoint_sha256": studio.WEIGHTS_SHA256,
        "input": {"kind": render.kind, "shape_lhw": list(expected),
                  "sha256": render.input_sha256(), "selected_stack_array_sha256": array_sha256(stack)},
        "planes": {"mesh_index": int(mesh_index), "order": order, "shift": int(shift),
                   "source_layers": ks},
        "tiling": {"window": 128, "kept": 96, "stride": stride, "blend": blend,
                   "skip_empty_windows": True, "edge_padding": "reflect in inference; invalid for real full context"},
        "normalisation": "float32, clip(raw, 0, 200) / 255",
        "planned_windows": len(tiles), "legacy_intensity_proxy_pixels": int(legacy_validity.sum()),
        "model_forward_calls": 0, "weights_loaded": False, "prediction_completed": False,
        "acquisition_semantics_confirmed": False,
        "interpretation": "Deterministic planned source-context coverage; no executed prediction or acquisition/FOV certification. The input/support bytes and Studio code must still match at inference.",
    }
    return result, plan


def write_preflight(output_dir, result, plan, *, support_path, spacing_um_yx=None):
    root = outside_git(output_dir)
    if root.exists():
        raise FileExistsError("Use a new private preflight directory")
    support_path = Path(support_path)
    support_hash = sha256_file(support_path)
    counts = result.counts()
    area = chart_areas(counts, spacing_um_yx)
    root.mkdir(parents=True)
    arrays = {
        "source_context_support.npy": result.context_support_yx,
        "planned_contribution_support.npy": result.contribution_support_yx,
        "source_available_all_required.npy": result.source_all_required_yx,
        "mesh_valid.npy": result.mesh_valid_yx,
        "tile_context_complete.npy": result.tile_context_complete,
    }
    hashes = {}
    for name, array in arrays.items():
        with (root / name).open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        hashes[name] = sha256_file(root / name)
    with (root / "tile_schedule.json").open("x") as stream:
        json.dump(result.record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    hashes["tile_schedule.json"] = sha256_file(root / "tile_schedule.json")
    if sha256_file(support_path) != support_hash:
        raise ValueError("Producer support changed during preflight")
    receipt = {
        "schema": "scrollprize.studio-context-preflight.v1", "status": "complete",
        "stage": "cpu_preflight_only", "handoff_ready": False,
        "tool_sha256": sha256_file(__file__), "support_tool_sha256": sha256_file(HERE / "support.py"),
        "support_npz_sha256": support_hash,
        "plan": plan, "counts": counts, **area, "outputs_sha256": hashes,
        "handoff_blockers": ["Original acquisition/FOV semantics not established by source availability",
                             "No executed model contribution/weight receipt",
                             "No authenticated mesh, normal and physical frame declaration"],
    }
    # The presence of preflight.json signifies only this CPU preflight completed.
    with (root / "preflight.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return receipt


def public_aggregate(receipts, *, scroll, surface_set):
    """An allowlisted aggregate projection; no local names, shapes or hashes.

    Counts sum chart coverage and may double-count overlapping renders. This
    function never copies strings, source paths, tile boxes or arbitrary fields
    from a private receipt.
    """
    import re
    if not re.fullmatch(r"PHerc[0-9]{3,4}[AB]?", scroll):
        raise ValueError("Use a scroll name only, such as PHerc0125")
    if surface_set not in {"spiral_fits", "raw_meshes", "atlas", "generated_windows", "public_control", "mixed"}:
        raise ValueError("Use a broad surface-set category")
    receipts = list(receipts)
    allowed = ("chart_pixels", "mesh_valid_pixels", "source_all_required_pixels",
               "mesh_and_source_pixels", "contribution_supported_pixels", "context_supported_pixels",
               "tiles", "complete_context_tiles")
    counts = {key: sum(int(r["counts"][key]) for r in receipts) for key in allowed}
    areas = {}
    for key in (k.replace("_pixels", "_cm2") for k in allowed if k.endswith("_pixels")):
        values = [r["chart_areas"].get(key) for r in receipts]
        areas[key] = sum(values) if values and all(v is not None for v in values) else None
    return {"schema": "scrollprize.producer-support-public-aggregate.v1", "scroll": scroll,
            "surface_set": surface_set, "renders": len(receipts), "stage": "cpu_preflight_only",
            "counts": counts, "chart_areas": areas, "unique_papyrus_cm2": None, "on_sheet_cm2": None,
            "model_forward_calls": 0, "reading_verdict": "not evaluated",
            "interpretation": "Summed chart coverage may overlap. No unread location, prediction, candidate or acquisition certificate is included."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local retained uint8 render; no remote data are fetched")
    parser.add_argument("--support-npz", required=True, help="Producer NPZ with source_available_lhw and mesh_valid_yx")
    parser.add_argument("--mesh-index", required=True, type=int)
    parser.add_argument("--order", required=True, choices=("asc", "desc"))
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--stride", type=int, choices=(48, 96), default=48)
    parser.add_argument("--spacing-um", type=float, nargs=2, metavar=("DY", "DX"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = outside_git(args.output_dir)
    if root.exists():
        raise FileExistsError("Use a new private preflight directory")
    studio = studio_module()
    render = studio.Render(args.input)
    input_before = render.input_sha256()
    support_before = sha256_file(args.support_npz)
    with np.load(args.support_npz, allow_pickle=False) as support:
        result, plan = prepare_studio_context(
            render, support["source_available_lhw"], support["mesh_valid_yx"],
            mesh_index=args.mesh_index, order=args.order, shift=args.shift, stride=args.stride,
        )
    if render.input_sha256() != input_before or sha256_file(args.support_npz) != support_before:
        raise ValueError("Producer source/support changed during computation")
    receipt = write_preflight(root, result, plan, support_path=args.support_npz, spacing_um_yx=args.spacing_um)
    print(json.dumps({"stage": receipt["stage"], "counts": receipt["counts"],
                      "model_forward_calls": 0, "handoff_ready": False}, indent=2))


if __name__ == "__main__":
    main()
