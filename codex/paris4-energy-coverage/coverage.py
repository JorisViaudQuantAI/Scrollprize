"""Measure public scanner coverage of the complete published Paris 4 segment."""
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import tifffile

ROOT = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/"
SEG = "20231016151002"
DLS = "20230205180739"
SCANS = {"137": "20260323153942", "78": "20260411134726"}
VOLUMES = {
    key: f"volumes/{scan}-2.400um-0.2m-{key}keV-masked.zarr/"
    for key, scan in SCANS.items()
}
REGION_GRID = 6
SUPPORT_STRIDE = 4


def read_bytes(key):
    with urlopen(ROOT + key, timeout=45) as response:
        return response.read()


def read_json(key):
    return json.loads(read_bytes(key))


def mesh(scan, spacing):
    prefix = f"segments/{SEG}/mesh/{SEG}-on-{scan}-{spacing}um.tifxyz/"
    with ThreadPoolExecutor(max_workers=3) as workers:
        raw = list(workers.map(read_bytes, [prefix + axis + ".tif" for axis in "xyz"]))
    xyz = np.stack([tifffile.imread(io.BytesIO(part)) for part in raw], axis=-1)
    valid = np.isfinite(xyz).all(axis=-1) & (xyz >= 0).all(axis=-1) & (xyz != 0).any(axis=-1)
    meta = read_json(prefix + "meta.json")
    actual_bbox = np.stack([xyz[valid].min(axis=0), xyz[valid].max(axis=0)])
    if not np.allclose(actual_bbox, np.array(meta["bbox"]), atol=2, rtol=0):
        raise ValueError(f"{scan} TIFF extrema disagree with the published mesh metadata")
    return xyz, valid, [hashlib.sha256(part).hexdigest() for part in raw]


def fixed_to_moving(scan, fixed):
    tx = read_json(f"transforms/{scan}-to-{DLS}-r1/fit/transform.json")
    if DLS not in tx["fixed_volume"]:
        raise ValueError("unexpected fixed scanner")
    affine = np.asarray(tx["transformation_matrix"], float)
    linear, shift = affine[:, :3], affine[:, 3]
    moving = np.asarray(tx["moving_landmarks"])
    fixed_lm = np.asarray(tx["fixed_landmarks"])
    inv = np.linalg.inv(linear)
    residual = np.linalg.norm((fixed_lm-shift) @ inv.T-moving, axis=1)
    backwards_error = np.linalg.norm(moving @ linear.T+shift-fixed_lm, axis=1)
    if not np.isfinite(residual).all() or np.median(backwards_error) > 100:
        raise ValueError("landmark transformation direction or units unsupported")
    quality = {
        "landmarks": len(moving),
        "forward_rms_DLS_voxels": float(np.sqrt(np.mean(backwards_error**2))),
        "inverse_residual_p50_p95_max_native_voxels": np.percentile(residual, [50,95,100]).round(3).tolist(),
        "affine_linear_singular_values": np.linalg.svd(linear, compute_uv=False).round(6).tolist(),
    }
    return (fixed.astype(float)-shift) @ inv.T, quality


def inside(xyz, shape_zyx, margin=0):
    upper = np.asarray(shape_zyx[::-1], dtype=float)-1
    return ((xyz >= margin) & (xyz <= upper-margin)).all(axis=-1)


def counts_grid(mask):
    row_groups = np.array_split(np.arange(mask.shape[0]), REGION_GRID)
    col_groups = np.array_split(np.arange(mask.shape[1]), REGION_GRID)
    return [[int(mask[np.ix_(rr, cc)].sum()) for cc in col_groups] for rr in row_groups]


def coarse_support(key, xyz):
    """Return non-fill at nearest level-4 voxel in the published masked CT."""
    import numcodecs
    level = 4
    base = VOLUMES[key]
    attrs = read_json(base+".zattrs")
    ds = {str(v["path"]): v for v in attrs["multiscales"][0]["datasets"]}
    coarse = read_json(base+f"{level}/.zarray")
    if coarse.get("filters"):
        raise ValueError("unsupported filter chain")
    if coarse["zarr_format"] != 2 or coarse["fill_value"] not in (0,None):
        raise ValueError("unexpected storage or fill convention")
    def scale(entry):
        return np.asarray(next(t["scale"] for t in entry["coordinateTransformations"] if t["type"] == "scale"), float)
    def translation(entry):
        return sum((np.asarray(t["translation"],float) for t in entry["coordinateTransformations"] if t["type"]=="translation"), np.zeros(3))
    native, low = ds["0"], ds[str(level)]
    s0, sl = scale(native), scale(low)
    if s0.size != 3 or sl.size != 3 or not np.allclose(sl/s0, 2**level, rtol=.01):
        raise ValueError("unexpected pyramid scales")
    p = (xyz[:,::-1]*s0 + translation(native)-translation(low))/sl
    iy = np.floor(p+.5).astype(int)
    if np.any(iy < 0) or np.any(iy >= np.asarray(coarse["shape"])):
        raise ValueError("native bounds do not map to coarse voxel centres")
    chunks = np.asarray(coarse["chunks"])
    unique, ids = np.unique(iy//chunks, axis=0, return_inverse=True)
    out = np.zeros(len(iy),bool)
    separator = coarse.get("dimension_separator", ".")
    dtype = np.dtype(coarse["dtype"])
    def one(group):
        ix, point = group
        members = np.flatnonzero(ids == ix)
        name = separator.join(str(v) for v in point)
        try:
            data = read_bytes(base+f"{level}/"+name)
        except HTTPError as error:
            if error.code == 404:
                return members, np.zeros(len(members),bool)
            raise
        if coarse["compressor"]:
            data = numcodecs.get_codec(coarse["compressor"]).decode(data)
        arr = np.frombuffer(data, dtype=dtype)
        full = tuple(int(v) for v in chunks)
        clipped = tuple(np.minimum(chunks, np.asarray(coarse["shape"])-point*chunks))
        shape = full if arr.size == np.prod(full) else clipped
        arr = arr.reshape(shape, order=coarse.get("order", "C"))
        idx = (iy[members]-point*chunks).T
        return members, arr[tuple(idx)] != 0
    with ThreadPoolExecutor(max_workers=7) as workers:
        for members, flags in workers.map(one, enumerate(unique)):
            out[members] = flags
    return out, len(unique)


def run():
    fixed, valid, sha_dls = mesh(DLS,"7.91")
    coords = {}
    masks = {}
    result = {
        "segment": SEG, "DLS": DLS, "mesh_shape_vu": list(valid.shape),
        "grid_stride_in_published_mesh_nodes": 1,
        "valid_points": int(valid.sum()), "mesh_sha256_xyz": sha_dls,
        "valid_nodes_per_6x6_uv_sector": counts_grid(valid), "scans": {},
    }
    for key, scan in SCANS.items():
        pos, qc = fixed_to_moving(scan, fixed)
        shape = read_json(VOLUMES[key]+"0/.zarray")["shape"]
        mask = valid & inside(pos,shape)
        coords[key], masks[key] = pos, mask
        result["scans"][key] = {
            "id":scan,"shape_zyx":shape,"registration":qc,"inside":int(mask.sum()),
            "percent":round(float(mask.sum()/valid.sum()*100),5), "inside_uv_sectors":counts_grid(mask),
            "inside_eroded_32":int((valid & inside(pos,shape,32)).sum()),
            "inside_dilated_32":int((valid & inside(pos,shape,-32)).sum()),
        }
    overlap = masks["78"] & masks["137"]
    result["joint_inside"] = int(overlap.sum())
    result["joint_inside_uv_sectors"] = counts_grid(overlap)
    direct, direct_valid, sha78 = mesh(SCANS["78"], "2.4")
    check = {"shape_vu":list(direct_valid.shape),"sha256_xyz":sha78}
    if direct_valid.shape == valid.shape:
        common = valid & direct_valid
        err = np.linalg.norm(direct[common]-coords["78"][common],axis=-1)
        check.update(common_nodes=int(common.sum()),p50_p95_max_native_voxels=np.percentile(err,[50,95,100]).round(3).tolist())
    else:
        check["pointwise_check"] = "different official UV raster dimensions"
    result["independent_official_mesh_78"] = check
    selected = np.zeros_like(valid)
    selected[::SUPPORT_STRIDE,::SUPPORT_STRIDE] = True
    selected &= overlap
    support = {"uv_stride":SUPPORT_STRIDE, "level":4,"joint_bounds_nodes_sampled":int(selected.sum())}
    if selected.any():
        try:
            a, n137 = coarse_support("137",coords["137"][selected])
            b, n78 = coarse_support("78",coords["78"][selected])
            support.update(nonfill_137=int(a.sum()),nonfill_78=int(b.sum()),both_nonfill=int((a&b).sum()),chunks_read_or_implicit_fill=[n137,n78])
        except Exception as error:
            support["unverified_reason"] = f"{type(error).__name__}: {error}"
    else:
        support["both_nonfill"] = 0
    result["nonfill_support_on_joint_grid"] = support
    print("COVERAGE_JSON "+json.dumps(result, separators=(",",":"),sort_keys=True))


if __name__ == "__main__":
    run()
