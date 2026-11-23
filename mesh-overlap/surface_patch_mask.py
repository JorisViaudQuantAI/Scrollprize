"""Exclude copied XYZ mesh vertices from UV labels and training patch footprints."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import tifffile
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree


def read_mesh(path, frame):
    metadata = path / "meta.json"
    if metadata.exists():
        declared = json.loads(metadata.read_text()).get("volume_id")
        if isinstance(declared, (str, int)) and str(declared) != frame:
            raise ValueError(f"{path} has volume_id {declared}, expected {frame}")
    xyz = [tifffile.imread(path / f"{axis}.tif") for axis in "xyz"]
    if any(a.dtype.kind != "f" or a.ndim != 2 or a.shape != xyz[0].shape for a in xyz):
        raise ValueError(f"Expected three equally shaped floating-point mesh channels in {path}")
    xyz = np.stack(xyz, axis=-1)
    valid = np.isfinite(xyz).all(axis=-1) & (xyz >= 0).all(axis=-1)
    if not valid.any():
        raise ValueError(f"No valid XYZ vertices in {path}")
    return xyz, valid


def overlapping(target, valid, reference_dirs, frame, radius):
    indices = np.flatnonzero(valid)
    target = target.reshape(-1, 3)
    hit = np.zeros(valid.size, dtype=bool)
    for folder in reference_dirs:
        ref, support = read_mesh(folder, frame)
        tree = cKDTree(ref[support])
        for start in range(0, len(indices), 100_000):
            ids = indices[start:start+100_000]
            distances = tree.query(target[ids], k=1)[0]
            hit[ids] |= distances <= radius
    return hit.reshape(valid.shape)


def window_count(mask, h, w):
    integral = np.pad(mask, ((1, 0), (1, 0))).cumsum(0, dtype=np.int64).cumsum(1)
    return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]


def select_patches(valid, excluded, known, patch, stride):
    h, w = patch
    sy, sx = stride
    empty = dict(candidate=0, center_only=0, center_only_with_overlap=0, retained=0)
    if h > valid.shape[0] or w > valid.shape[1]:
        return np.empty((0, 2), dtype=np.int64), empty
    allowed = (window_count(valid, h, w) == h*w) & (window_count(known, h, w) > 0)
    allowed = allowed[::sy, ::sx]
    contacts = window_count(excluded, h, w)[::sy, ::sx]
    height, width = valid.shape
    center = excluded[h//2:h//2+height-h+1:sy, w//2:w//2+width-w+1:sx]
    center_only = allowed & ~center
    kept = allowed & (contacts == 0)
    stats = dict(candidate=int(allowed.sum()), center_only=int(center_only.sum()),
                 center_only_with_overlap=int((center_only & (contacts > 0)).sum()), retained=int(kept.sum()))
    return np.argwhere(kept) * np.array(stride), stats


def run(target_dir, reference_dirs, frame, radius, patch, stride, label_path=None):
    if not np.isfinite(radius) or radius < 0 or min(*patch, *stride) <= 0:
        raise ValueError("Radius must be nonnegative and finite and grid dimensions positive")
    vertices, valid = read_mesh(target_dir, frame)
    excluded = overlapping(vertices, valid, reference_dirs, frame, radius)
    output = dict(valid=valid, excluded=excluded, eligible_vertices=valid & ~excluded,
                  patch_hw=np.array(patch), stride_yx=np.array(stride))
    known = valid
    sampler_known = valid
    if label_path is not None:
        with np.load(label_path, allow_pickle=False) as labels:
            known, ink = labels["known"], labels["ink"]
            for value in (known, ink):
                if value.shape != valid.shape or not np.isin(value, [False, True, 255]).all():
                    raise ValueError("Expected binary known and ink arrays in the target UV grid")
            for key in labels.files:
                if key not in output and key not in ("ink", "known", "origins_yx"):
                    output[key] = labels[key]
        known = valid & known.astype(bool)
        h, w = patch
        # Any such label can lie in an arbitrarily aligned patch that also contains a copied vertex.
        unsafe = maximum_filter(excluded, size=(2*h-1, 2*w-1), mode="constant", cval=0)
        sampler_known = known & ~unsafe
        output["known"] = sampler_known
        output["ink"] = ink.astype(bool) & output["known"]
    output["origins_yx"], windows = select_patches(valid, excluded, sampler_known, patch, stride)
    stats = dict(frame=frame, radius_voxels=radius, patch_hw=list(patch), stride_yx=list(stride),
                 shape_yx=list(valid.shape), valid_vertices=int(valid.sum()), excluded_vertices=int(excluded.sum()),
                 eligible_vertices=int(output["eligible_vertices"].sum()),
                 input_known=int(known.sum()) if label_path else None,
                 remaining_known=int(output["known"].sum()) if label_path else None, windows=windows)
    return output, stats


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("target", type=Path)
    cli.add_argument("references", type=Path, nargs="+")
    cli.add_argument("--frame", required=True, help="Common CT voxel coordinate frame")
    cli.add_argument("--radius", type=float, default=0)
    cli.add_argument("--patch", type=int, nargs=2, default=(16, 16), metavar=("HEIGHT","WIDTH"))
    cli.add_argument("--stride", type=int, nargs=2, default=(8, 8), metavar=("ROWS","COLS"))
    cli.add_argument("--labels", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    out = args.output.resolve()
    if out.suffix.lower() != ".npz" or (args.labels and args.labels.resolve() == out):
        cli.error("Choose a separate .npz output path")
    fields, stats = run(args.target, args.references, args.frame, args.radius, args.patch, args.stride, args.labels)
    out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=out.parent, suffix=".npz", delete=False) as handle:
        temp = Path(handle.name)
    try:
        np.savez_compressed(temp, **fields)
        temp.replace(out)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
