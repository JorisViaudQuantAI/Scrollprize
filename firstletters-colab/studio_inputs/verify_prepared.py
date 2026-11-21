#!/usr/bin/env python3
"""Read-only format/provenance check for an already prepared Studio input.

This checks the file's contents, not whether its plane order, geometry, or claimed
origin are correct. It never assigns a mesh position or physical sample offsets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prepare import STUDIO_REVISION, sha256_file


def verify_prepared_file(
    input_path: str | Path,
    *,
    coverage_mask_path: str | Path | None = None,
) -> dict:
    path = Path(input_path)
    input_hash = sha256_file(path)
    volume = np.load(path, mmap_mode="r", allow_pickle=False)
    if (not isinstance(volume, np.ndarray) or volume.ndim != 3
            or volume.shape[0] != 16 or volume.dtype != np.uint8
            or any(size == 0 for size in volume.shape)):
        raise ValueError("prepared input must be nonempty uint8 [16,H,W]")
    if np.any(~np.any(volume != 0, axis=(1, 2))):
        raise ValueError("prepared input contains an entirely zero required plane")
    coverage_hash = None
    if coverage_mask_path is not None:
        coverage_mask_path = Path(coverage_mask_path)
        coverage_hash = sha256_file(coverage_mask_path)
        mask = np.load(coverage_mask_path, mmap_mode="r", allow_pickle=False)
        if (not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
                or mask.shape != volume.shape):
            raise ValueError("coverage mask must be boolean with exactly the prepared [16,H,W] shape")
        if not np.all(mask):
            raise ValueError("prepared input contains uncovered CT samples")
        if sha256_file(coverage_mask_path) != coverage_hash:
            raise ValueError("coverage mask changed during verification")
    if sha256_file(path) != input_hash:
        raise ValueError("prepared input changed during verification")
    return {
        "schema": "studio-prepared-input-v1",
        "release_revision_for_format_contract": STUDIO_REVISION,
        "input_shape_dhw": list(volume.shape),
        "input_dtype": "uint8",
        "input_file_sha256": input_hash,
        "coverage_mask_file_sha256": coverage_hash,
        "verifier_file_sha256": sha256_file(Path(__file__)),
        "geometry_check": "unverified: no mesh index or physical spacing inferred",
        "mesh_plane_index": None,
        "sample_offsets_um_along_declared_mesh_normal": None,
        "depth_order_check": "unverified: expected already in model order; format does not establish order",
        "coverage_check": (
            "all samples true in caller-supplied acquisition-coverage mask"
            if coverage_mask_path is not None
            else "unverified: no acquisition-coverage mask supplied"
        ),
        "zero_plane_check": "all 16 planes contain at least one nonzero sample",
        "all_zero_depth_column_count": int(np.count_nonzero(np.all(volume == 0, axis=0))),
        "intensity_transform": "none",
        "expected_upstream_intensity_transform": "float32; clip(raw,0,200)/255",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--coverage-mask", type=Path)
    args = parser.parse_args()
    try:
        record = verify_prepared_file(args.input, coverage_mask_path=args.coverage_mask)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Verification failed: {exc}\n")
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
