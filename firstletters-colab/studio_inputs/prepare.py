#!/usr/bin/env python3
"""Select Studio's published 16-plane band without resampling or normalization.

Only NumPy is required. The caller must know the mesh-plane index and the actual
rendered sample spacing; neither is inferred from a stack's length or filename.
See README.md for the convention, coverage limits, and reversal distinction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


STUDIO_REVISION = "bf66f753176792a966bca17822629fb8821f20c8"
STUDIO_SOURCE = (
    "https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/"
    f"blob/{STUDIO_REVISION}/README.md"
)
ORDERS = ("release-descending", "same-band-reversed")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def select_planes(
    stack: np.ndarray,
    *,
    mesh_plane_index: int,
    xy_spacing_um: Sequence[float],
    depth_spacing_um: float,
    index_direction: int,
    order: str,
    coverage_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Return an unchanged-value uint8 [16,H,W] slab and its geometry record.

    ``index_direction`` is +1 or -1 relative to the caller's declared mesh normal.
    ``coverage_mask``, if provided, is a boolean acquisition-coverage array of the
    *same DHW shape* as stack. All selected samples must be covered. A papyrus
    silhouette or an ink mask is not an acquisition-coverage mask.

    A fully zero selected plane is rejected even when coverage is asserted. This
    is a fail-safe input-quality check, not proof that a zero plane was missing.
    Individual zero CT values remain unchanged; they can be legitimate samples.
    """
    if not isinstance(stack, np.ndarray) or stack.ndim != 3:
        raise ValueError("input must be a NumPy [D,H,W] array")
    if stack.dtype != np.uint8:
        raise ValueError("input must be uint8; conversion or intensity scaling is not implicit")
    if any(size == 0 for size in stack.shape):
        raise ValueError("input dimensions must be nonzero")
    mesh_plane_index = _integer(mesh_plane_index, "mesh_plane_index")
    index_direction = _integer(index_direction, "index_direction")
    if index_direction not in (-1, 1):
        raise ValueError("index_direction must be +1 or -1 relative to the declared mesh normal")
    if not 0 <= mesh_plane_index < stack.shape[0]:
        raise ValueError("mesh_plane_index is outside the input stack")
    try:
        if len(xy_spacing_um) != 2:
            raise ValueError("xy_spacing_um must contain the actual [y,x] sample spacings")
        xy = [_positive(v, "xy_spacing_um") for v in xy_spacing_um]
    except TypeError as exc:
        raise ValueError("xy_spacing_um must contain the actual [y,x] sample spacings") from exc
    step = _positive(depth_spacing_um, "depth_spacing_um")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")

    # Published supplied-surface convention: a 31-plane render with mesh at 15
    # uses 23,22,...,8. Do not center the band silently or infer the mesh plane.
    offsets = np.arange(8, -8, -1, dtype=np.int64)
    if order == "same-band-reversed":
        offsets = offsets[::-1]
    indices = mesh_plane_index + offsets
    if indices.min() < 0 or indices.max() >= stack.shape[0]:
        raise ValueError("required 16-plane band is out of bounds; re-render, do not pad")

    if coverage_mask is not None:
        if not isinstance(coverage_mask, np.ndarray) or coverage_mask.dtype != np.bool_:
            raise ValueError("coverage_mask must be a boolean NumPy array; no thresholding is implicit")
        if coverage_mask.shape != stack.shape:
            raise ValueError("coverage_mask must have exactly the input [D,H,W] shape")
        if not np.all(coverage_mask[indices]):
            raise ValueError("required band contains uncovered CT samples; re-render or use a complete crop")

    output = np.ascontiguousarray(stack[indices])
    zero_planes = np.flatnonzero(~np.any(output != 0, axis=(1, 2)))
    if zero_planes.size:
        raise ValueError(
            "required plane is entirely zero; source indices "
            + ",".join(str(int(indices[i])) for i in zero_planes)
        )
    physical_offsets = offsets.astype(np.float64) * step * index_direction
    record = {
        "schema": "studio-inputs-v1",
        "release_revision": STUDIO_REVISION,
        "release_source": STUDIO_SOURCE,
        "input_shape_dhw": list(stack.shape),
        "input_dtype": "uint8",
        "output_shape_dhw": list(output.shape),
        "mesh_plane_index": mesh_plane_index,
        "mesh_plane_provenance": "explicit caller input; never inferred from stack length",
        "xy_spacing_um_yx": xy,
        "depth_spacing_um": step,
        "index_direction_relative_to_declared_mesh_normal": index_direction,
        "order": order,
        "source_plane_indices": indices.tolist(),
        "source_index_offsets_from_mesh": offsets.tolist(),
        "sample_offsets_um_along_declared_mesh_normal": physical_offsets.tolist(),
        "sample_band_midpoint_offset_um": float(physical_offsets.mean()),
        "sample_centers_span_um": float(physical_offsets.max() - physical_offsets.min()),
        "resampling": "none",
        "adapter_intensity_transform": "none",
        "expected_upstream_intensity_transform": "float32; clip(raw,0,200)/255",
        "coverage_check": (
            "all selected samples true in caller-supplied acquisition-coverage mask"
            if coverage_mask is not None
            else "unverified: no acquisition-coverage mask supplied"
        ),
        "zero_plane_check": "all 16 selected planes contain at least one nonzero sample",
        "all_zero_depth_column_count": int(np.count_nonzero(np.all(output == 0, axis=0))),
    }
    return output, record


def prepare_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    mesh_plane_index: int,
    xy_spacing_um: Sequence[float],
    depth_spacing_um: float,
    index_direction: int,
    order: str,
    coverage_mask_path: str | Path | None = None,
) -> dict:
    """Write a new .npy and .npy.json pair, refusing to overwrite either file.

    Hashes are of complete file bytes, including the .npy header. Paths and names
    are omitted from the sidecar so local directory or candidate names do not
    enter provenance automatically. A generated sidecar for an unread scroll is
    still a private experiment artifact and must not be committed without review.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.suffix != ".npy":
        raise ValueError("output_path must end in .npy")
    sidecar_path = output_path.with_suffix(".npy.json")
    if output_path.exists() or sidecar_path.exists():
        raise FileExistsError("output or provenance sidecar already exists")
    # Hash before and after loading so an input replaced during preparation does
    # not produce a seemingly reproducible record. Files should remain immutable.
    input_hash = sha256_file(input_path)
    stack = np.load(input_path, mmap_mode="r", allow_pickle=False)
    coverage = None
    coverage_hash = None
    if coverage_mask_path is not None:
        coverage_mask_path = Path(coverage_mask_path)
        coverage_hash = sha256_file(coverage_mask_path)
        coverage = np.load(coverage_mask_path, mmap_mode="r", allow_pickle=False)
    output, record = select_planes(
        stack,
        mesh_plane_index=mesh_plane_index,
        xy_spacing_um=xy_spacing_um,
        depth_spacing_um=depth_spacing_um,
        index_direction=index_direction,
        order=order,
        coverage_mask=coverage,
    )
    if sha256_file(input_path) != input_hash:
        raise ValueError("input changed during preparation")
    if coverage_mask_path is not None and sha256_file(coverage_mask_path) != coverage_hash:
        raise ValueError("coverage mask changed during preparation")
    record["input_file_sha256"] = input_hash
    record["coverage_mask_file_sha256"] = coverage_hash
    record["adapter_file_sha256"] = sha256_file(Path(__file__))
    made_output = made_sidecar = False
    try:
        with output_path.open("xb") as stream:
            made_output = True
            np.save(stream, output, allow_pickle=False)
        record["output_file_sha256"] = sha256_file(output_path)
        with sidecar_path.open("x", encoding="utf-8") as stream:
            made_sidecar = True
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except Exception:
        # Keep a failed two-file write from appearing ready for inference. Never
        # delete pre-existing files: flags are set only after exclusive creation.
        if made_sidecar:
            sidecar_path.unlink(missing_ok=True)
        if made_output:
            output_path.unlink(missing_ok=True)
        raise
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mesh-plane-index", required=True, type=int)
    parser.add_argument("--xy-spacing-um", required=True, type=float, nargs=2, metavar=("DY", "DX"))
    parser.add_argument("--depth-spacing-um", required=True, type=float)
    parser.add_argument("--index-direction", required=True, type=int, choices=(-1, 1))
    parser.add_argument("--order", required=True, choices=ORDERS)
    parser.add_argument("--coverage-mask", type=Path)
    args = parser.parse_args()
    try:
        record = prepare_file(
            args.input,
            args.output,
            mesh_plane_index=args.mesh_plane_index,
            xy_spacing_um=args.xy_spacing_um,
            depth_spacing_um=args.depth_spacing_um,
            index_direction=args.index_direction,
            order=args.order,
            coverage_mask_path=args.coverage_mask,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")
    print(json.dumps({
        "output_shape_dhw": record["output_shape_dhw"],
        "output_file_sha256": record["output_file_sha256"],
        "coverage_check": record["coverage_check"],
    }, indent=2))


if __name__ == "__main__":
    main()
