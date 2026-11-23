"""Render a local CT volume through original triangles without intensity scaling.

Example (negative depths are clearest with the equals sign)::

    python firstletters-colab/surface_sampling/cli.py \
        --volume crop.npy --geometry surface.npz --output /tmp/render-example \
        --volume-origin 0 0 0 --origin-uv 0 0 --step-uv 1 1 \
        --shape-yx 64 64 --depths=-2,-1,0,1,2 --samples-per-axis 4

The geometry NPZ must contain ``uv``, ``xyz`` and ``triangles`` arrays. XYZ and
normal depths use native CT voxel units. The volume is a finite float32/float64
ZYX ``.npy`` array; it is opened read-only with NumPy mmap. Standard NPY format
versions 1 and 2 are accepted, including those inside NPZ. Source headers,
declared allocations and the output location are checked before array loading
or rendering. Fixed limits deliberately keep this CLI a bounded local tool.

Each ``depth_NNN.tif`` is a 32-bit floating-point TIFF in unchanged source CT
units. There is NO automatic scaling, clipping, normalisation or integer cast.
``valid_mask.png`` is a one-bit mask: white is retained, black is invalid.
Invalid pixels have value zero in every TIFF and must be excluded by the mask.

The output directory must not exist and must be outside any Git working tree,
checking both lexical and symlink-resolved ancestors for a .git entry. The JSON
report records source basenames and SHA256 hashes, never absolute file paths.
It is written last; an interrupted write can leave incomplete TIFFs without a
report. Outputs may contain private scan coordinates and must still be handled
according to the source dataset's publication rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

if __package__:
    from .render import render
    from .surface import OriginalSurface
else:
    from render import render
    from surface import OriginalSurface


MAX_OUTPUT_SIDE = 2048
MAX_OUTPUT_PIXELS = 262_144
MAX_DEPTHS = 129
MAX_STACK_BYTES = 256 * 1024 * 1024
MAX_VOLUME_BYTES = 1024 * 1024 * 1024
MAX_GEOMETRY_BYTES = 64 * 1024 * 1024
MAX_VERTICES = 250_000
MAX_FACES = 500_000
MAX_HEADER_BYTES = 16_384


def _depth_list(value):
    pieces = value.split(",")
    if not 1 <= len(pieces) <= MAX_DEPTHS:
        raise argparse.ArgumentTypeError(f"depth count must be between 1 and {MAX_DEPTHS}")
    try:
        depths = [float(piece) for piece in pieces]
    except ValueError:
        raise argparse.ArgumentTypeError("depths must be comma-separated finite numbers") from None
    if not all(math.isfinite(depth) for depth in depths):
        raise argparse.ArgumentTypeError("depths must be comma-separated finite numbers")
    return depths


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volume", type=Path, required=True, help="local float32/float64 ZYX .npy volume")
    parser.add_argument("--geometry", type=Path, required=True, help="local .npz containing uv, xyz, triangles")
    parser.add_argument("--output", type=Path, required=True, help="new directory outside every Git working tree")
    parser.add_argument("--volume-origin", type=float, nargs=3, metavar=("X", "Y", "Z"), required=True)
    parser.add_argument("--origin-uv", type=float, nargs=2, metavar=("U", "V"), required=True)
    parser.add_argument("--step-uv", type=float, nargs=2, metavar=("DU", "DV"), required=True)
    parser.add_argument("--shape-yx", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), required=True)
    parser.add_argument("--depths", type=_depth_list, required=True, help="comma-separated voxel depths; e.g. --depths=-2,0,2")
    parser.add_argument("--samples-per-axis", type=int, choices=(1, 4, 8), default=1)
    return parser


def _validate_request(args):
    height, width = args.shape_yx
    if min(height, width) <= 0 or max(height, width) > MAX_OUTPUT_SIDE:
        raise ValueError(f"output dimensions must be between 1 and {MAX_OUTPUT_SIDE}")
    if height * width > MAX_OUTPUT_PIXELS:
        raise ValueError(f"output pixel count exceeds the {MAX_OUTPUT_PIXELS} limit")
    if not 1 <= len(args.depths) <= MAX_DEPTHS:
        raise ValueError(f"depth count exceeds the {MAX_DEPTHS} limit")
    if height * width * len(args.depths) * 4 > MAX_STACK_BYTES:
        raise ValueError(f"output stack exceeds the {MAX_STACK_BYTES}-byte allocation limit")
    for name in ("volume_origin", "origin_uv", "step_uv", "depths"):
        if not all(math.isfinite(value) for value in getattr(args, name)):
            raise ValueError(f"{name} must be finite")
    if min(args.step_uv) <= 0:
        raise ValueError("step_uv values must be strictly positive")
    with np.errstate(over="ignore", invalid="ignore"):
        low = np.asarray(args.origin_uv) - np.asarray(args.step_uv) / 2
        high = np.asarray(args.origin_uv) + np.asarray(args.step_uv) * [width - 0.5, height - 0.5]
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("output footprint exceeds finite coordinate arithmetic")


def _check_output_location(output):
    output = output.expanduser()
    if output.exists() or output.is_symlink():
        raise ValueError("output directory already exists; no files will be overwritten")
    lexical = Path(os.path.abspath(os.fspath(output)))
    resolved = output.resolve(strict=False)
    for location in (lexical, resolved):
        for ancestor in (location, *location.parents):
            marker = ancestor / ".git"
            if marker.exists() or marker.is_symlink():
                raise ValueError("output directory must be outside every Git working tree (.git ancestor found)")
    return resolved


def _input_file(path, suffix, role, byte_limit):
    path = path.expanduser()
    if path.suffix.lower() != suffix or not path.is_file():
        raise ValueError(f"{role} must be an existing local {suffix} file")
    if path.stat().st_size > byte_limit:
        raise ValueError(f"{role} file exceeds the bounded input-size limit")
    return path


def _npy_header(stream, role, total_bytes):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
            stream, max_header_size=MAX_HEADER_BYTES
        )
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
            stream, max_header_size=MAX_HEADER_BYTES
        )
    else:
        raise ValueError(f"{role} must use standard NPY format version 1 or 2")
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in "fiu" or dtype.itemsize > 8:
        raise ValueError(f"{role} must contain ordinary real numeric values, with no objects or structured dtype")
    if not shape or any(not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError(f"{role} has an invalid or empty array shape")
    nbytes = math.prod(shape) * dtype.itemsize
    if stream.tell() + nbytes != total_bytes:
        raise ValueError(f"{role} declared array bytes do not match its file size")
    return {"shape": tuple(shape), "dtype": dtype, "nbytes": nbytes, "fortran_order": bool(fortran_order)}


def _volume_header(path):
    with path.open("rb") as stream:
        header = _npy_header(stream, "volume", path.stat().st_size)
    if len(header["shape"]) != 3 or min(header["shape"]) < 2:
        raise ValueError("volume must be ZYX with at least two voxels on every axis")
    if header["dtype"].kind != "f" or header["dtype"].itemsize not in (4, 8):
        raise ValueError("volume dtype must be float32 or float64")
    if header["nbytes"] > MAX_VOLUME_BYTES:
        raise ValueError(f"volume exceeds the {MAX_VOLUME_BYTES}-byte allocation limit")
    return header


def _geometry_headers(path):
    headers = {}
    required = ("uv", "xyz", "triangles")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(set(names)) != len(names):
            raise ValueError("geometry archive has duplicate member names")
        if sum(member.file_size for member in members) > MAX_GEOMETRY_BYTES:
            raise ValueError(f"uncompressed geometry exceeds the {MAX_GEOMETRY_BYTES}-byte limit")
        for key in required:
            name = key + ".npy"
            if name not in names:
                raise ValueError(f"geometry NPZ must contain {key}")
            info = archive.getinfo(name)
            with archive.open(info) as stream:
                headers[key] = _npy_header(stream, f"geometry {key}", info.file_size)
    uv, xyz, triangles = (headers[key] for key in required)
    if len(uv["shape"]) != 2 or uv["shape"][1] != 2 or uv["shape"][0] > MAX_VERTICES:
        raise ValueError(f"geometry uv must be N by 2, with at most {MAX_VERTICES} vertices")
    if xyz["shape"] != (uv["shape"][0], 3):
        raise ValueError("geometry xyz must be N by 3 and match uv")
    if (len(triangles["shape"]) != 2 or triangles["shape"][1] != 3
            or triangles["shape"][0] > MAX_FACES or triangles["dtype"].kind not in "iu"):
        raise ValueError(f"geometry triangles must be integer F by 3, with at most {MAX_FACES} faces")
    return headers


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    """Validate, render and export one bounded local job; return its summary."""
    _validate_request(args)
    output = _check_output_location(args.output)
    volume_path = _input_file(args.volume, ".npy", "volume", MAX_VOLUME_BYTES + MAX_HEADER_BYTES + 16)
    geometry_path = _input_file(args.geometry, ".npz", "geometry", MAX_GEOMETRY_BYTES)
    volume_header = _volume_header(volume_path)
    geometry_headers = _geometry_headers(geometry_path)
    source_files = {
        "volume": {"filename": volume_path.name, "sha256": _sha256(volume_path)},
        "geometry": {"filename": geometry_path.name, "sha256": _sha256(geometry_path)},
    }
    volume = np.load(volume_path, mmap_mode="r", allow_pickle=False, max_header_size=MAX_HEADER_BYTES)
    with np.load(geometry_path, allow_pickle=False, max_header_size=MAX_HEADER_BYTES) as geometry:
        surface = OriginalSurface(geometry["uv"], geometry["xyz"], geometry["triangles"])
    stack, valid, diagnostics = render(
        volume, args.volume_origin, surface, args.origin_uv, args.step_uv,
        args.shape_yx, args.depths, args.samples_per_axis,
    )
    # Recheck after rendering, and create exclusively. Existing output is never
    # reused even if another process created it during this job.
    output = _check_output_location(args.output)
    output.mkdir(parents=True, exist_ok=False)
    layers = []
    for index, (depth, layer) in enumerate(zip(args.depths, stack)):
        filename = f"depth_{index:03d}.tif"
        Image.fromarray(layer).save(output / filename, format="TIFF", compression="raw")
        layers.append({"index": index, "depth_voxels": float(depth), "filename": filename})
    Image.fromarray(valid).save(output / "valid_mask.png", format="PNG")
    report = {
        "schema_version": 1,
        "source_files": source_files,
        "source_volume": {"shape_zyx": list(volume_header["shape"]), "dtype": str(volume_header["dtype"])},
        "source_geometry": {key: {"shape": list(header["shape"]), "dtype": str(header["dtype"])}
                            for key, header in geometry_headers.items()},
        "spec": {
            "volume_origin_xyz": list(args.volume_origin),
            "origin_uv": list(args.origin_uv),
            "step_uv": list(args.step_uv),
            "shape_yx": list(args.shape_yx),
            "depths_voxels": list(args.depths),
            "samples_per_axis": args.samples_per_axis,
            "output_dtype": "float32",
            "intensity_units": "unchanged source CT units",
            "automatic_intensity_scaling": False,
            "invalid_pixel_value": 0.0,
            "valid_mask": "valid_mask.png",
            "valid_mask_meaning": "one-bit PNG: white retained, black invalid",
        },
        "diagnostics": diagnostics,
        "layers": layers,
    }
    with (output / "report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {
        "output_directory": output.name,
        "depth_layers": len(layers),
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.size),
        "report": "report.json",
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (ValueError, OSError, MemoryError, FloatingPointError, zipfile.BadZipFile) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
