#!/usr/bin/env python3
"""Pre-read, metadata-only chunk transfer bound for the Paris 4 paired-volume preflight.

For Zarr v2 uncompressed chunks, calculate the payload size and best possible
number of chunk objects accessed. No CT pixel data is accessed or interpreted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    for field in ("shape", "chunks"):
        v = spec[field]
        if len(v) != 3 or any(type(x) is not int or x <= 0 for x in v):
            raise ValueError(f"invalid 3D {field}")
    if spec.get("zarr_format") != 2 or spec.get("order") != "C":
        raise ValueError("Only C-order Zarr v2 supported")
    if spec.get("compressor") is not None or spec.get("filters") is not None:
        raise ValueError("Compressed/filtered arrays require empirical transfer sizes")
    if spec.get("dtype") != "|u1":
        raise ValueError("Only confirmed uint8 arrays supported, not implicit conversions")
    if spec.get("dimension_separator") != "/":
        raise ValueError("Expected nested Zarr v2 chunks")
    return spec


def chunk_span(n: int, chunk: int, offset: int) -> int:
    if n <= 0 or chunk <= 0 or offset < 0:
        raise ValueError("positive lengths/chunks and nonnegative offset required")
    return (offset + n - 1) // chunk - offset // chunk + 1


def min_xy_chunks(spec: dict, side_px: int, depth_chunks: int = 1) -> int:
    if side_px > min(spec["shape"][1:]):
        raise ValueError("Patch does not fit in volume")
    if depth_chunks <= 0:
        raise ValueError("depth_chunks must be positive")
    cz, cy, cx = spec["chunks"]
    return depth_chunks * math.ceil(side_px / cy) * math.ceil(side_px / cx)


def max_xy_chunks(spec: dict, side_px: int, depth_chunks: int = 1) -> int:
    # Interior bound for arbitrary sub-chunk origin, not necessarily available
    # at small volume boundaries. Explicit bounding boxes are still required.
    cz, cy, cx = spec["chunks"]
    return depth_chunks * math.ceil((side_px + cy - 1) / cy) * math.ceil((side_px + cx - 1) / cx)


def nominal_chunk_payload(spec: dict) -> int:
    return math.prod(spec["chunks"])  # verified dtype: uint8; interior complete chunks


def evaluate(protocol: dict, specs: list[dict]) -> dict:
    width_px = math.ceil(protocol["side_length_um"] / protocol["pixel_pitch_um"])
    pairs = protocol["preregistered_pairs"]
    if pairs != 3 or len(specs) != 2:
        raise ValueError("Expected exactly three matched two-volume controls")
    specs_bytes = [nominal_chunk_payload(spec) for spec in specs]
    chunk_counts = [min_xy_chunks(spec, width_px) for spec in specs]
    single_pair_bytes = sum(n * b for n, b in zip(chunk_counts, specs_bytes))
    def mb(raw):
        return raw / 2**20
    budget = int(protocol["budget_mib"] * 2**20)
    best_three_bytes = pairs * single_pair_bytes
    valid_widths = [w for w in range(1, width_px+1)
                    if pairs * sum(min_xy_chunks(s, w) * b for s,b in zip(specs,specs_bytes)) <= budget]
    max_fit_px = max(valid_widths) if valid_widths else None
    result = {
        "schema":"scrollprize.paris4-raw-zarr-chunk-budget-result.v1",
        "stage":"metadata_only_no_pixels",
        "source_count":len(specs),
        "controls":pairs,
        "square_side_um":protocol["side_length_um"],
        "pixel_pitch_um":protocol["pixel_pitch_um"],
        "side_pixels_ceil":width_px,
        "budget_mib":protocol["budget_mib"],
        "chunk_shape_zyx":[s["chunks"] for s in specs],
        "shape_zyx":[s["shape"] for s in specs],
        "nominal_complete_chunk_payload_bytes":specs_bytes,
        "min_chunk_objects_one_pair_each_source":chunk_counts,
        "one_pair_best_aligned_no_halo_bytes":single_pair_bytes,
        "one_pair_best_aligned_no_halo_mib":mb(single_pair_bytes),
        "three_disjoint_pairs_best_aligned_no_halo_bytes":best_three_bytes,
        "three_disjoint_pairs_best_aligned_no_halo_mib":mb(best_three_bytes),
        "single_pair_exceeds_budget":single_pair_bytes>budget,
        "three_disjoint_pairs_exceed_budget":best_three_bytes>budget,
        "one_pair_interior_misalignment_ceiling_mib":mb(sum(max_xy_chunks(s,width_px)*b for s,b in zip(specs,specs_bytes))),
        "max_square_side_for_three_disjoint_pairs_aligned_px":max_fit_px,
        "max_square_side_for_three_disjoint_pairs_aligned_um":None if max_fit_px is None else max_fit_px*protocol["pixel_pitch_um"],
        "depth_chunks_per_volume_modeled":1,
        "interpretation":(
            "Nominal full-chunk network payload under wholly present uncompressed interior Zarr chunks; "
            "minimum aligned request counts. HTTP headers/handshakes, chunk-boundary overlaps, 3D alignment "
            "halos and RAM are excluded. Independent nonoverlapping controls were assumed for 3-pair sum. "
            "Missing chunks can reduce transfer but fail source support, not certify zero-valued anatomy. "
            "No actual chunk was requested; this is an I/O feasibility decision, not measured network traffic."
        ),
    }
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--zarray-78",type=Path,required=True)
    p.add_argument("--zarray-137",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    protocol=json.loads(args.protocol.read_text())
    specs=[load_spec(args.zarray_78),load_spec(args.zarray_137)]
    out=evaluate(protocol,specs)
    out["frozen_protocol_sha256"]=hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    out["zarray_metadata_sha256"]=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.zarray_78,args.zarray_137)]
    args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k in (
        "one_pair_best_aligned_no_halo_mib","three_disjoint_pairs_best_aligned_no_halo_mib","single_pair_exceeds_budget","max_square_side_for_three_disjoint_pairs_aligned_um")},indent=2))

if __name__=="__main__":
    main()
