"""fl_helpers.py - small helpers for the First Letters Colab runbook (first_letters_colab.ipynb).

Everything here is plain Python (numpy, zarr<3 or zarr>=3, fsspec, requests, tifffile, opencv, optional torch).
It does NOT run any model or VC3D tool itself; the notebook calls those as shell commands.

Conventions (checked against the winners' public repos):
  * coordinates are level-0 voxels, written x, y, z (VC3D order); zarr arrays are (z, y, x)
  * a render made with `vc_render_tifxyz --num-slices N --slice-step 1` has layer k at (k - (N-1)/2) voxels
    along the mesh normal, so with N odd the traced surface is layer (N-1)//2 and depth d = k - (N-1)//2
"""
from __future__ import annotations

import json
import math
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def url(key: str) -> str:
    return f"{BUCKET}/{key.lstrip('/')}"


# ----------------------------------------------------------------------------------------------------------------
# S3 listing / download (anonymous, open-data bucket)
# ----------------------------------------------------------------------------------------------------------------
def s3_ls(prefix: str, recursive: bool = False):
    """(dirs, files[(key, size)]) under prefix. Anonymous ListObjectsV2."""
    import requests
    dirs, files, token = [], [], None
    while True:
        p = {"list-type": "2", "prefix": prefix}
        if not recursive:
            p["delimiter"] = "/"
        if token:
            p["continuation-token"] = token
        r = requests.get(BUCKET, params=p, timeout=60)
        r.raise_for_status()
        t = ET.fromstring(r.content)
        dirs += [e.find("s:Prefix", _NS).text for e in t.findall("s:CommonPrefixes", _NS)]
        files += [(e.find("s:Key", _NS).text, int(e.find("s:Size", _NS).text)) for e in t.findall("s:Contents", _NS)]
        if t.find("s:IsTruncated", _NS).text != "true":
            return dirs, files
        token = t.find("s:NextContinuationToken", _NS).text


def _get(key_size_dest):
    import requests
    key, size, dest = key_size_dest
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(5):
        try:
            r = requests.get(url(key), timeout=300)
            r.raise_for_status()
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(r.content)
            os.replace(tmp, dest)
            return size
        except Exception:  # noqa: BLE001 - retry any network error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed: {key}")


def download_normal_grids(ngrid_key: str, dest: str, z0: int, z1: int, margin: int = 256, threads: int = 32,
                          max_files: int | None = None, dry_run: bool = False):
    """Copy a normal-grid store (metadata.json + xy/xz/yz *.grid) to a local directory for vc_grow_seg_from_seed.

    xy grids are one file per z slice, so only z in [z0-margin, z1+margin] is fetched; xz and yz grids are one file
    per y / x slice and are fetched whole (about 6 GB for a ~8k-wide scroll). The *_img preview folders are skipped.
    Returns (n_files, n_bytes)."""
    pre = ngrid_key.rstrip("/") + "/"
    jobs = []
    _, top = s3_ls(pre)
    for k, s in top:
        jobs.append((k, s, os.path.join(dest, k[len(pre):])))
    for sub in ("xy/", "xz/", "yz/"):
        _, files = s3_ls(pre + sub)
        for k, s in files:
            if sub == "xy/":
                try:
                    zi = int(os.path.basename(k).split(".")[0])
                except ValueError:
                    continue
                if not (z0 - margin <= zi < z1 + margin):
                    continue
            jobs.append((k, s, os.path.join(dest, k[len(pre):])))
    if max_files is not None:
        jobs = jobs[:max_files]
    total = sum(s for _, s, _ in jobs)
    print(f"normal grids: {len(jobs)} files, {total / 1e9:.2f} GB -> {dest}")
    if dry_run:
        return len(jobs), total
    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        done = sum(ex.map(_get, jobs))
    print(f"  fetched {done / 1e9:.2f} GB new in {time.time() - t0:.0f} s")
    return len(jobs), total


# ----------------------------------------------------------------------------------------------------------------
# remote zarr access (OME-Zarr v2 volumes and predictions in the open-data bucket)
# ----------------------------------------------------------------------------------------------------------------
def open_level(key_or_url: str, level: int | str = 0):
    """Open one pyramid level of a remote zarr v2 group read-only (works with zarr 2.x and 3.x)."""
    import fsspec
    import zarr
    u = key_or_url if key_or_url.startswith("http") else url(key_or_url)
    u = u.rstrip("/") + f"/{level}"
    if int(zarr.__version__.split(".")[0]) >= 3:
        return zarr.open_array(u, mode="r", zarr_format=2)
    return zarr.open(fsspec.get_mapper(u), mode="r")


def read_box(arr, xyz, half: int):
    """(z,y,x) crop of half-size `half` around xyz, clipped to the array; returns (block, origin_zyx)."""
    x, y, z = [int(round(v)) for v in xyz]
    lo = [max(0, c - half) for c in (z, y, x)]
    hi = [min(n, c + half + 1) for c, n in zip((z, y, x), arr.shape)]
    return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]), lo


def snap_seed(m7_key: str, xyz, half: int = 48):
    """Nearest voxel of the (binary, thresholded) m7 surface prediction to xyz, at level 0.
    GrowPatch seeds must lie on the predicted sheet. Returns (x, y, z, distance_voxels) or None."""
    arr = open_level(m7_key, 0)
    blk, (oz, oy, ox) = read_box(arr, xyz, half)
    zz, yy, xx = np.nonzero(blk > 0)
    if len(zz) == 0:
        return None
    x, y, z = xyz
    d2 = (xx + ox - x) ** 2 + (yy + oy - y) ** 2 + (zz + oz - z) ** 2
    i = int(np.argmin(d2))
    return int(xx[i] + ox), int(yy[i] + oy), int(zz[i] + oz), float(math.sqrt(d2[i]))


def sheet_profile(ct_key: str, xyz, half: int = 32):
    """Quick CT look at a seed: mean intensity and the share of dark (gap) voxels in a small cube.
    Not a quality score (use the Scan-Quality-Map atlas for that), just a sanity check that the seed is on papyrus."""
    arr = open_level(ct_key, 0)
    blk, _ = read_box(arr, xyz, half)
    b = blk.astype(np.float32)
    med = float(np.median(b))
    return {"mean": float(b.mean()), "median": med, "dark_fraction": float((b < 0.6 * med).mean()) if med > 0 else None,
            "shape": list(blk.shape)}


# ----------------------------------------------------------------------------------------------------------------
# GrowPatch parameters (keys read by vc_grow_seg_from_seed / GrowPatch.cpp in villa c8f3ba4)
# ----------------------------------------------------------------------------------------------------------------
def write_grow_params(path: str, normal_grid_dir: str, voxel_um: float, generations: int = 140, step: int = 20,
                      threads: int = 4, cache_bytes: int = 4_000_000_000, min_area_cm: float = 0.5):
    """The exact parameter set the PHerc. 0306B winners used for their claim surface
    (discovery/fl/sqm/PHerc0306B_f306_w074own/params.json), with the paths replaced."""
    p = {"mode": "seed", "step_size": step, "min_area_cm": min_area_cm, "generations": generations,
         "thread_limit": threads, "normal_grid_path": str(normal_grid_dir), "voxelsize": voxel_um,
         "cache_size": cache_bytes}
    Path(path).write_text(json.dumps(p, indent=1))
    return p


# ----------------------------------------------------------------------------------------------------------------
# tifxyz
# ----------------------------------------------------------------------------------------------------------------
def tifxyz_stats(mesh_dir: str, umbilicus_json: str | None = None, voxel_um: float | None = None):
    import tifffile
    X = tifffile.imread(os.path.join(mesh_dir, "x.tif")).astype(np.float64)
    Y = tifffile.imread(os.path.join(mesh_dir, "y.tif")).astype(np.float64)
    Z = tifffile.imread(os.path.join(mesh_dir, "z.tif")).astype(np.float64)
    v = (X > 0) & (Y > 0) & (Z > 0)
    out = {"grid_hw": list(X.shape), "valid_nodes": int(v.sum()),
           "x": [float(X[v].min()), float(X[v].max())], "y": [float(Y[v].min()), float(Y[v].max())],
           "z": [float(Z[v].min()), float(Z[v].max())]}
    meta = os.path.join(mesh_dir, "meta.json")
    if os.path.exists(meta):
        m = json.load(open(meta))
        sc = m.get("scale")
        if sc and voxel_um:
            step = 1.0 / float(sc[0] if isinstance(sc, list) else sc)
            out["area_cm2_approx"] = float(v.sum() * (step * voxel_um * 1e-4) ** 2)
    if umbilicus_json:
        cp = sorted(json.load(open(umbilicus_json))["control_points"], key=lambda p: p["z"])
        zc = np.array([p["z"] for p in cp]); xc = np.array([p["x"] for p in cp]); yc = np.array([p["y"] for p in cp])
        r = np.hypot(X[v] - np.interp(Z[v], zc, xc), Y[v] - np.interp(Z[v], zc, yc))
        out["radius_vox_median"] = float(np.median(r))
        if voxel_um:
            out["radius_mm_median"] = float(np.median(r) * voxel_um / 1000)
    return out


# ----------------------------------------------------------------------------------------------------------------
# renders: reading, quick looks, depth windows
# ----------------------------------------------------------------------------------------------------------------
def open_render(path: str):
    """A vc_render_tifxyz --zarr-output store: level '0' is (layers, rows, cols) uint8."""
    import zarr
    g = zarr.open(path, mode="r")
    try:
        return g["0"]
    except Exception:  # noqa: BLE001 - a bare array
        return g


def surface_index(n_layers: int) -> int:
    return (n_layers - 1) // 2


def villa_layer_window(n_layers: int, d: int, depth: int = 17):
    """--layer-start/--layer-end (end exclusive) for villa ink inference: a `depth`-layer window centred on
    surface_index + d (the PHerc. 343 run_ink.sh rule: s = N/2 + d - 8 for N=66)."""
    c = surface_index(n_layers) + d
    s = c - depth // 2
    e = s + depth
    if s < 0 or e > n_layers:
        raise ValueError(f"d={d}: layers {s}..{e - 1} outside 0..{n_layers - 1}")
    return s, e


def quicklook(render, out_dir: str, stride: int = 1):
    """Middle layer and max/mean composites of the central +-5 layers, contrast stretched, as PNGs."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    n = render.shape[0]
    c = surface_index(n)
    lo, hi = max(0, c - 5), min(n, c + 6)
    stack = np.asarray(render[lo:hi, ::stride, ::stride]).astype(np.float32)
    outs = {}
    for name, img in (("layer_mid", stack[c - lo]), ("composite_max", stack.max(0)), ("composite_mean", stack.mean(0))):
        p = os.path.join(out_dir, f"{name}.png")
        cv2.imwrite(p, stretch(img))
        outs[name] = p
    return outs


def stretch(img, lo_pct=1.0, hi_pct=99.5, valid=None):
    a = np.asarray(img, dtype=np.float32)
    m = (a > 0) if valid is None else valid
    if not m.any():
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(a[m], [lo_pct, hi_pct])
    return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------------------------
# resampling for Hecate (needs 9.6 um isotropic input; hecate.py refuses spacings >2 % off)
# ----------------------------------------------------------------------------------------------------------------
def resample_stack(arr, src_um: float, dst_um: float = 9.6, band: int = 1024, device: str = "cpu"):
    """Trilinear resample of a (Z, Y, X) uint8 stack by src_um/dst_um in all three axes, in row bands.
    Uses torch if available (GPU if device='cuda'), else scipy.ndimage.zoom."""
    f = src_um / dst_um
    Z, Y, X = arr.shape
    oz, oy, ox = max(1, round(Z * f)), max(1, round(Y * f)), max(1, round(X * f))
    out = np.zeros((oz, oy, ox), np.uint8)
    try:
        import torch
        import torch.nn.functional as F
        have_torch = True
    except ImportError:  # pragma: no cover
        have_torch = False
        from scipy import ndimage
    sy = Y / oy                                         # exact source step along Y (align_corners=False)
    for y0o in range(0, oy, band):
        y1o = min(oy, y0o + band)
        yc = (np.arange(y0o, y1o) + 0.5) * sy - 0.5     # source row coordinate of every output row
        yc = np.clip(yc, 0, Y - 1)
        y0 = int(np.floor(yc.min()))
        y1 = min(Y, int(np.floor(yc.max())) + 2)
        src = np.asarray(arr[:, y0:y1, :]).astype(np.float32)
        # 1) Z and X: full extent in this band, so a plain resize is exact; Y kept at native rows
        if have_torch:
            t = torch.from_numpy(src)[None, None].to(device)
            r = F.interpolate(t, size=(oz, y1 - y0, ox), mode="trilinear", align_corners=False)[0, 0].cpu().numpy()
        else:
            r = ndimage.zoom(src, (oz / Z, 1, ox / X), order=1)
        # 2) Y: linear interpolation at the exact source coordinates
        i0 = np.floor(yc).astype(int) - y0
        i1 = np.minimum(i0 + 1, r.shape[1] - 1)
        w = (yc - np.floor(yc)).astype(np.float32)[None, :, None]
        piece = r[:, i0, :] * (1 - w) + r[:, i1, :] * w
        out[:, y0o:y1o, :] = np.clip(np.rint(piece), 0, 255).astype(np.uint8)
    return out


def hecate_crop(res, n_native: int, src_um: float, d_native: float, nz: int = 16, dst_um: float = 9.6):
    """16 planes of a resampled stack centred on the surface + d_native native layers (forward order)."""
    f = src_um / dst_um
    c = (surface_index(n_native) + d_native) * f      # centre in resampled plane units
    s = int(round(c - nz / 2 + 0.5))
    if s < 0 or s + nz > res.shape[0]:
        raise ValueError(f"d={d_native}: planes {s}..{s + nz - 1} outside 0..{res.shape[0] - 1}")
    return np.ascontiguousarray(res[s:s + nz])


# ----------------------------------------------------------------------------------------------------------------
# ensembles (the two winning rules) and the claim-style image
# ----------------------------------------------------------------------------------------------------------------
def load_map(path: str, shape_hw=None):
    """An 8-bit ink map (PNG/TIF) as float32 probability in [0, 1], optionally resized to shape_hw (rows, cols)."""
    import cv2
    import tifffile
    a = tifffile.imread(path) if path.lower().endswith((".tif", ".tiff")) else cv2.imread(path, cv2.IMREAD_UNCHANGED)
    a = np.asarray(a).astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    a /= 255.0 if a.max() > 1.5 else 1.0
    if shape_hw is not None and a.shape != tuple(shape_hw):
        a = cv2.resize(a, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)
    return a


def zscore_ensemble(maps_by_model: dict, valid=None):
    """PHerc. 343 rule: per model, one mean/sd over ALL its depth maps (pooled), z-score each map, then average the
    models' z-scores at each depth. maps_by_model = {model: {depth: 2-D array}} (all on one grid).
    Returns {depth: ensemble z-map}."""
    stats = {}
    for m, by_d in maps_by_model.items():
        pooled = np.concatenate([(v[valid] if valid is not None else v.ravel()) for v in by_d.values()])
        stats[m] = (float(pooled.mean()), float(pooled.std() + 1e-6))
    depths = sorted(set.intersection(*[set(v) for v in maps_by_model.values()]))
    return {d: np.mean([(maps_by_model[m][d] - stats[m][0]) / stats[m][1] for m in maps_by_model], 0) for d in depths}


def product_rule(a, b):
    """PHerc. 306B rule: per-pixel product of two models' ink probabilities (ink only where both agree)."""
    return np.asarray(a, np.float32) * np.asarray(b, np.float32)


def claim_image(ink, ct_layer, out_png: str, voxel_um: float, threshold_pct: float = 85.0, full_pct: float = 99.5,
                ct_dim: float = 0.45, flip_lr: bool = False, flip_ud: bool = False, title: str | None = None):
    """Ink in yellow over a dimmed CT layer with a 1 cm scale bar (the PHerc. 343 look; percentiles instead of his
    fixed z-score levels). Returns pixels per cm."""
    import cv2
    ink = np.asarray(ink, np.float32)
    ct = np.asarray(ct_layer, np.float32)
    valid = ct > 0
    lo, hi = np.percentile(ink[valid], [threshold_pct, full_pct]) if valid.any() else (0, 1)
    op = np.clip((ink - lo) / max(hi - lo, 1e-6), 0, 1) * 0.9
    g = stretch(ct, 1, 99.5, valid).astype(np.float32) * ct_dim
    col = np.array([120, 235, 255], np.float32)          # BGR of (255, 235, 120)
    rgb = g[..., None] * (1 - op[..., None]) + col * op[..., None]
    rgb[~valid] = 0
    rgb = rgb.astype(np.uint8)
    if flip_lr:
        rgb = rgb[:, ::-1]
    if flip_ud:
        rgb = rgb[::-1]
    rgb = np.ascontiguousarray(rgb)
    px_cm = 10000.0 / voxel_um
    h = rgb.shape[0]
    y = h - 40
    x0 = 40
    cv2.rectangle(rgb, (x0, y - 12), (x0 + int(round(px_cm)), y), (255, 255, 255), -1)
    cv2.putText(rgb, "1 cm", (x0, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    if title:
        cv2.putText(rgb, title, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imwrite(out_png, rgb)
    return px_cm


def depth_shuffle(stack, seed: int = 0):
    """False-positive control used by PHerc. 0306B: shuffle the layer order; letters must disappear."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(stack.shape[0])
    return np.ascontiguousarray(np.asarray(stack)[idx])


if __name__ == "__main__":
    print(__doc__)
