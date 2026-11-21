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
    # A crop can be shorter than the rounded-up native boundary at this level.
    # -1 is unresolved and remains distinct from a verified Zarr fill value.
    usable = np.flatnonzero(((iy >= 0) & (iy < np.asarray(coarse["shape"]))).all(axis=1))
    chunks = np.asarray(coarse["chunks"])
    unique, ids = np.unique(iy[usable]//chunks, axis=0, return_inverse=True)
    out = np.full(len(iy),-1,dtype=np.int8)
    separator = coarse.get("dimension_separator", ".")
    dtype = np.dtype(coarse["dtype"])
    def one(group):
        ix, point = group
        members = usable[np.flatnonzero(ids == ix)]
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
    # Both UV rasters cover the entire official surface, at different resolutions.
    fixed_selected = np.zeros_like(valid)
    fixed_selected[::SUPPORT_STRIDE,::SUPPORT_STRIDE] = True
    fixed_selected &= overlap
    public78 = direct[::SUPPORT_STRIDE,::SUPPORT_STRIDE].astype(float)
    valid78 = direct_valid[::SUPPORT_STRIDE,::SUPPORT_STRIDE]
    t78 = read_json(f"transforms/{SCANS['78']}-to-{DLS}-r1/fit/transform.json")
    a78 = np.asarray(t78["transformation_matrix"],float)
    dls78 = public78 @ a78[:,:3].T + a78[:,3]
    as137, _ = fixed_to_moving(SCANS["137"],dls78)
    inside78 = valid78 & inside(public78,result["scans"]["78"]["shape_zyx"])
    inside137 = valid78 & inside(as137,result["scans"]["137"]["shape_zyx"])
    both = inside78 & inside137
    nfixed = int(fixed_selected.sum())
    ndirect = int(both.sum())
    percent = lambda n, d: round(100*n/d,6) if d else None
    direct_info = {
        "stride_vu":[SUPPORT_STRIDE,SUPPORT_STRIDE],"sampling_origin_vu":[0,0],
        "full_shape_vu":list(direct_valid.shape),"sampled_shape_vu":list(valid78.shape),
        "valid":int(valid78.sum()),"inside_137":int(inside137.sum()),
        "inside_78":int(inside78.sum()),"joint":ndirect,
        "joint_percent_all_valid":percent(ndirect,int(valid78.sum())),
        "joint_uv_sectors":counts_grid(both),"valid_uv_sectors":counts_grid(valid78),
    }
    energies = {
        "137":np.concatenate((coords["137"][fixed_selected],as137[both])),
        "78":np.concatenate((coords["78"][fixed_selected],public78[both])),
    }
    identified = {}
    failures = {}
    source_chunks = {}
    for key, nodes in energies.items():
        identified[key] = np.full(len(nodes),-1,dtype=np.int8)
        if len(nodes):
            try:
                identified[key], source_chunks[key] = coarse_support(key,nodes)
            except Exception as error:
                failures[key] = f"{type(error).__name__}: {error}"
    def summary(a,b):
        known = (a >= 0) & (b >= 0)
        return {"joint_bounds_nodes_sampled":len(a),"both_evaluable":int(known.sum()),
                "unknown_137":int((a<0).sum()),"unknown_78":int((b<0).sum()),
                "nonfill_137":int((a==1).sum()),"nonfill_78":int((b==1).sum()),
                "both_nonfill":int(((a==1)&(b==1)).sum()),
                "both_nonfill_percent_among_evaluable":percent(int(((a==1)&(b==1)).sum()),int(known.sum()))}
    a,b = identified["137"],identified["78"]
    result["coarse_joint_verification"] = {
        "level":4,"uv_stride":SUPPORT_STRIDE,"unique_chunks_by_energy":source_chunks,
        "unresolved_access":failures,"DLS_grid":summary(a[:nfixed],b[:nfixed]),
        "direct_78_grid":summary(a[nfixed:],b[nfixed:]),
    }
    av,bv = a[nfixed:],b[nfixed:]
    categories = np.full(valid78.shape,"N")
    categories[valid78] = "O"
    state = np.full(ndirect,"G")
    state[(av==0)|(bv==0)] = "F"
    state[(av==1)&(bv==1)] = "B"
    categories[both] = state
    direct_info["categories"] = {key:int((categories==key).sum()) for key in "NOGFB"}
    rows = ["v_mesh,category_count_runs_at_u_stride4"]
    for v, row in enumerate(categories):
        cuts = np.r_[0,np.flatnonzero(row[1:] != row[:-1])+1,len(row)]
        runs = "|".join(f"{row[i]}:{j-i}" for i,j in zip(cuts,cuts[1:]))
        rows.append(f"{SUPPORT_STRIDE*v},{runs}")
    csv = "\n".join(rows)+"\n"
    direct_info["rle_schema"] = {
        "v_mesh":"original 78-keV TIFXYZ vertex row, sampled at 0,4,8,...",
        "runs":"row starts at original U=0; each unit advances 4 stored mesh columns",
        "N":"no valid public mesh vertex","O":"outside at least one scanner bounding box",
        "G":"inside both boxes; common non-fill not verified",
        "F":"inside both boxes; fill verified for at least one scanner at level 4",
        "B":"inside both boxes and non-fill verified in both at level 4",
    }
    direct_info["rle_csv_sha256"] = hashlib.sha256(csv.encode("ascii")).hexdigest()
    result["native_78_mesh_full_grid"] = direct_info
    print("COVERAGE_JSON "+json.dumps(result,separators=(",",":"),sort_keys=True))
    for row in rows:
        print("COVERAGE_ROW "+row)


if __name__ == "__main__":
    run()
