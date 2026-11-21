#!/usr/bin/env python3
"""CPU renderer: flatten a tifxyz surface against a remote OME-Zarr (v2) CT volume.

For every output pixel (u, v) of the flattened surface the volume is sampled with
trilinear interpolation at  P(u, v) + offset_k * N(u, v)  for k = 0..L-1, where P comes
from the tifxyz grid (bicubic resampled) and N = dP/du x dP/dv (unit).  This mirrors
`vc_render_tifxyz --scale 1 --group-idx 0 --num-slices L [--flip-normals]` from VC3D:
layer k sits at (k - (L-1)/2) * step voxels along the (optionally negated) normal.

Only the zarr chunks touched by the surface are fetched over HTTPS (thread pool, LRU
memory cache, optional on-disk cache); a whole volume is never downloaded.

Output layout (consumed by the ink-model scripts):
    <out>/layers/00.tif .. NN.tif     uint8, one file per layer
    <out>/composite_max.png, composite_mean.png, layer_mid.png, mask.png
    <out>/tifxyz/{x,y,z}.tif, meta.json   the exact (cropped) mesh that was rendered
    <out>/render_meta.json            every parameter needed to reproduce the render

Example:
    python render_tifxyz.py \
        --volume https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr \
        --tifxyz <segments_dir>/<seg>/mesh.tifxyz \
        --out <renders_dir>/<seg> --layers 65 --auto-crop
"""
import argparse, json, os, sys, time, threading, shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import cv2
cv2.setNumThreads(2)
import requests, tifffile
from scipy import ndimage as ndi


# ----------------------------------------------------------------------------- volume
class RemoteZarrArray:
    """Minimal read-only zarr-v2 array over HTTP(S) (or a local dir) with chunk caching."""

    def __init__(self, root, level="0", cache_mb=1200, disk_cache=None, workers=8):
        self.root = root.rstrip("/")
        self.level = str(level)
        self.local = not self.root.startswith("http")
        self.sess = threading.local()
        meta = json.loads(self._get(f"{self.level}/.zarray"))
        assert meta["zarr_format"] == 2, "only zarr v2 supported"
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.fill = meta.get("fill_value") or 0
        self.sep = meta.get("dimension_separator", ".")
        self.order = meta.get("order", "C")
        self.codec = None
        if meta.get("compressor"):
            import numcodecs
            self.codec = numcodecs.get_codec(meta["compressor"])
        assert not meta.get("filters"), "filters unsupported"
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_max = cache_mb * 1e6
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(workers)
        self.disk = disk_cache
        self.n_fetch = 0
        self.bytes_fetch = 0
        if self.disk:
            os.makedirs(self.disk, exist_ok=True)

    def _session(self):
        if not hasattr(self.sess, "s"):
            self.sess.s = requests.Session()
        return self.sess.s

    def _get(self, key):
        if self.local:
            p = os.path.join(self.root, key)
            if not os.path.exists(p):
                return None
            with open(p, "rb") as f:
                return f.read()
        for attempt in range(5):
            try:
                r = self._session().get(f"{self.root}/{key}", timeout=120)
                # Only an absent sparse chunk is fill; denied access is not blank CT.
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.content
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1.5 * (attempt + 1))

    def _load_chunk(self, idx):
        key = f"{self.level}/" + self.sep.join(str(i) for i in idx)
        dpath = None
        if self.disk:
            dpath = os.path.join(self.disk, key.replace("/", "_") + ".npy")
            if os.path.exists(dpath):
                return np.load(dpath)
        raw = self._get(key)
        if raw is None:
            arr = None
        else:
            self.n_fetch += 1
            self.bytes_fetch += len(raw)
            if self.codec is not None:
                raw = self.codec.decode(raw)
            arr = np.frombuffer(raw, dtype=self.dtype).reshape(self.chunks, order=self.order)
        if dpath is not None and arr is not None:
            tmp = dpath + f".tmp{threading.get_ident()}.npy"
            np.save(tmp, arr)
            os.replace(tmp, dpath)
        return arr

    def get_chunks(self, idxs):
        """dict idx -> ndarray or None (missing = fill)."""
        out, need = {}, []
        with self.lock:
            for i in idxs:
                if i in self.cache:
                    self.cache.move_to_end(i)
                    out[i] = self.cache[i]
                else:
                    need.append(i)
        if need:
            for i, arr in zip(need, self.pool.map(self._load_chunk, need)):
                out[i] = arr
                with self.lock:
                    self.cache[i] = arr
                    self.cache_bytes += arr.nbytes if arr is not None else 0
            with self.lock:
                while self.cache_bytes > self.cache_max and len(self.cache) > len(idxs):
                    _, a = self.cache.popitem(last=False)
                    self.cache_bytes -= a.nbytes if a is not None else 0
        return out

    def read_box(self, lo, hi):
        """dense block for [lo, hi) (zyx), clipped to the array; returns (block, origin)."""
        lo = [max(0, int(l)) for l in lo]
        hi = [min(s, int(h)) for s, h in zip(self.shape, hi)]
        if any(h <= l for l, h in zip(lo, hi)):
            return np.zeros((1, 1, 1), self.dtype), (0, 0, 0)
        c0 = [l // c for l, c in zip(lo, self.chunks)]
        c1 = [(h - 1) // c for h, c in zip(hi, self.chunks)]
        idxs = [(a, b, c) for a in range(c0[0], c1[0] + 1) for b in range(c0[1], c1[1] + 1)
                for c in range(c0[2], c1[2] + 1)]
        chunks = self.get_chunks(idxs)
        block = np.full([h - l for l, h in zip(lo, hi)], self.fill, self.dtype)
        for idx, arr in chunks.items():
            if arr is None:
                continue
            s = [i * c for i, c in zip(idx, self.chunks)]
            a0 = [max(l, x) for l, x in zip(lo, s)]
            a1 = [min(h, x + c, sh) for h, x, c, sh in zip(hi, s, self.chunks, self.shape)]
            if any(y <= x for x, y in zip(a0, a1)):
                continue
            block[a0[0] - lo[0]:a1[0] - lo[0], a0[1] - lo[1]:a1[1] - lo[1], a0[2] - lo[2]:a1[2] - lo[2]] = \
                arr[a0[0] - s[0]:a1[0] - s[0], a0[1] - s[1]:a1[1] - s[1], a0[2] - s[2]:a1[2] - s[2]]
        return block, tuple(lo)


# ----------------------------------------------------------------------------- tifxyz
def read_tifxyz(path):
    x = tifffile.imread(os.path.join(path, "x.tif")).astype(np.float32)
    y = tifffile.imread(os.path.join(path, "y.tif")).astype(np.float32)
    z = tifffile.imread(os.path.join(path, "z.tif")).astype(np.float32)
    meta = json.load(open(os.path.join(path, "meta.json")))
    pts = np.stack([x, y, z], -1)
    valid = np.isfinite(pts).all(-1) & (z > 0)
    mp = os.path.join(path, "mask.tif")
    if os.path.exists(mp):
        m = tifffile.imread(mp)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape == valid.shape:
            valid &= m >= 255
    sc = meta.get("scale", [1.0, 1.0])
    return pts, valid, (float(sc[0]), float(sc[1])), meta


def write_tifxyz(path, pts, valid, scale, extra_meta=None):
    os.makedirs(path, exist_ok=True)
    p = pts.astype(np.float32).copy()
    p[~valid] = -1.0
    for i, n in enumerate("xyz"):
        tifffile.imwrite(os.path.join(path, f"{n}.tif"), p[..., i])
    v = p[valid]
    meta = {"format": "tifxyz", "type": "seg", "scale": [float(scale[0]), float(scale[1])],
            "bbox": [v.min(0).tolist(), v.max(0).tolist()] if len(v) else None,
            "uuid": os.path.basename(path.rstrip("/"))}
    if extra_meta:
        meta.update(extra_meta)
    json.dump(meta, open(os.path.join(path, "meta.json"), "w"), indent=1)


def grid_normals(pts, valid):
    """N = dP/du x dP/dv on the grid (u = column, v = row); invalid neighbours -> one-sided."""
    P = pts.astype(np.float64)
    H, W = valid.shape

    def diff(axis):
        d = np.zeros_like(P)
        ok = np.zeros((H, W), bool)
        fwd = np.roll(P, -1, axis) - P
        bwd = P - np.roll(P, 1, axis)
        vf = valid & np.roll(valid, -1, axis)
        vb = valid & np.roll(valid, 1, axis)
        sl_last = [slice(None)] * 2; sl_last[axis] = -1
        sl_first = [slice(None)] * 2; sl_first[axis] = 0
        vf[tuple(sl_last)] = False
        vb[tuple(sl_first)] = False
        both = vf & vb
        d[both] = 0.5 * (fwd[both] + bwd[both])
        d[vf & ~vb] = fwd[vf & ~vb]
        d[vb & ~vf] = bwd[vb & ~vf]
        ok = vf | vb
        return d, ok

    du, oku = diff(1)
    dv, okv = diff(0)
    n = np.cross(du, dv)
    ln = np.linalg.norm(n, axis=-1)
    ok = oku & okv & (ln > 1e-9)
    n[ok] /= ln[ok][:, None]
    n[~ok] = 0
    return n.astype(np.float32), ok


def fill_invalid(arr, valid):
    """nearest-valid extrapolation so interpolation near the border does not ring."""
    if valid.all():
        return arr
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return arr[idx[0], idx[1]]


# ----------------------------------------------------------------------------- render
def render(args):
    t0 = time.time()
    pts, valid, (sx, sy), meta = read_tifxyz(args.tifxyz)
    if args.seg_scale != 1.0:
        pts = pts * args.seg_scale
    H, W = valid.shape
    # --- grid crop (in grid cells) --------------------------------------------------
    if args.grid_crop:
        gx0, gy0, gw, gh = args.grid_crop
    elif args.auto_crop:
        ys, xs = np.where(valid)
        gx0, gy0, gw, gh = xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    else:
        gx0, gy0, gw, gh = 0, 0, W, H
    gx0, gy0 = max(0, gx0), max(0, gy0)
    gw, gh = min(gw, W - gx0), min(gh, H - gy0)
    pts = pts[gy0:gy0 + gh, gx0:gx0 + gw]
    valid = valid[gy0:gy0 + gh, gx0:gx0 + gw]
    H, W = valid.shape
    nrm, nok = grid_normals(pts, valid)
    if args.flip_normals:
        nrm = -nrm
    ptsf = fill_invalid(pts, valid)
    nrmf = fill_invalid(nrm, valid & nok)

    # output canvas: 1 px = 1 level-0 voxel * 1/out_scale ; grid index = px * s / out_scale
    S = args.scale
    OW, OH = int(round(W / sx * S)), int(round(H / sy * S))   # same canvas as vc_render_tifxyz
    L = args.layers
    offs = (np.arange(L) - 0.5 * (L - 1)) * args.step
    vol = RemoteZarrArray(args.volume, args.level, cache_mb=args.cache_mb, disk_cache=args.disk_cache,
                          workers=args.workers)
    lvl_scale = 1.0 / (2 ** int(args.level)) if args.level_scale is None else args.level_scale
    out = args.out
    os.makedirs(os.path.join(out, "layers"), exist_ok=True)
    stack = np.zeros((L, OH, OW), np.uint8)
    omask = np.zeros((OH, OW), np.uint8)
    T = args.tile
    interp = cv2.INTER_CUBIC if args.surface_interp == "bicubic" else cv2.INTER_LINEAR
    vmask = valid.astype(np.float32)
    ntiles = ((OH + T - 1) // T) * ((OW + T - 1) // T)
    done = 0
    for ty in range(0, OH, T):
        for tx in range(0, OW, T):
            h, w = min(T, OH - ty), min(T, OW - tx)
            gu = ((np.arange(w, dtype=np.float32) + tx + args.px_offset) * (sx / S))[None, :].repeat(h, 0)
            gv = ((np.arange(h, dtype=np.float32) + ty + args.px_offset) * (sy / S))[:, None].repeat(w, 1)
            m = cv2.remap(vmask, gu, gv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
            done += 1
            if not m.any():
                continue
            P = cv2.remap(ptsf, gu, gv, interp, borderMode=cv2.BORDER_REPLICATE)
            N = cv2.remap(nrmf, gu, gv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
            P = P[m] * lvl_scale + args.vox_offset   # (n,3) xyz in level voxels
            N = N[m]
            ext = np.abs(offs).max() * lvl_scale + 2
            lo = np.floor(P.min(0) - ext).astype(int)[::-1]
            hi = np.ceil(P.max(0) + ext).astype(int)[::-1] + 1
            block, org = vol.read_box(lo, hi)
            res = np.zeros((L, P.shape[0]), np.float32)
            blockf = block if block.dtype == np.uint8 else block.astype(np.float32)
            for k, o in enumerate(offs):
                q = P + (o * lvl_scale) * N
                coords = np.stack([q[:, 2] - org[0], q[:, 1] - org[1], q[:, 0] - org[2]])
                res[k] = ndi.map_coordinates(blockf, coords, order=1, mode="constant", cval=0.0, output=np.float32)
            if block.dtype != np.uint8:
                res = res / float(np.iinfo(block.dtype).max) * 255.0
            tile = np.zeros((L, h, w), np.uint8)
            tile[:, m] = np.clip(np.rint(res), 0, 255).astype(np.uint8)
            stack[:, ty:ty + h, tx:tx + w] = tile
            omask[ty:ty + h, tx:tx + w] = m.astype(np.uint8) * 255
            if done % 10 == 0 or done == ntiles:
                print(f"  tile {done}/{ntiles}  fetched {vol.n_fetch} chunks / {vol.bytes_fetch/1e6:.0f} MB  "
                      f"{time.time()-t0:.0f}s", flush=True)
    # --- write -----------------------------------------------------------------------
    nd = max(2, len(str(L - 1)))
    for k in range(L):
        tifffile.imwrite(os.path.join(out, "layers", f"{k:0{nd}d}.tif"), stack[k], compression="zlib")
    c = L // 2
    hw = min(args.composite_half, c)
    sub = stack[c - hw:c + hw + 1]
    cv2.imwrite(os.path.join(out, "composite_max.png"), sub.max(0))
    cv2.imwrite(os.path.join(out, "composite_mean.png"), np.rint(sub.mean(0)).astype(np.uint8))
    cv2.imwrite(os.path.join(out, "layer_mid.png"), stack[c])
    cv2.imwrite(os.path.join(out, "mask.png"), omask)
    tdir = os.path.join(out, "tifxyz")
    write_tifxyz(tdir, pts / args.seg_scale if args.seg_scale != 1.0 else pts, valid, (sx, sy),
                 {"source_tifxyz": os.path.abspath(args.tifxyz), "source_grid_crop_xywh": [int(gx0), int(gy0), int(W), int(H)]})
    rm = {"volume": args.volume, "level": args.level, "tifxyz_source": os.path.abspath(args.tifxyz),
          "grid_crop_xywh": [int(gx0), int(gy0), int(W), int(H)], "tifxyz_scale": [sx, sy],
          "out_scale_px_per_voxel": S, "layers": L, "step_voxels": args.step, "flip_normals": bool(args.flip_normals),
          "layer_offsets_along_normal": offs.tolist(), "normal_definition": "N = dP/du x dP/dv (u=col, v=row); negated if flip_normals",
          "px_offset": args.px_offset, "vox_offset": args.vox_offset, "surface_interp": args.surface_interp, "volume_interp": "trilinear", "canvas_wh": [OW, OH],
          "voxel_size_um": args.voxel_um, "pixel_size_um": (args.voxel_um / S) if args.voxel_um else None,
          "valid_area_cm2": (float((omask > 0).sum()) * (args.voxel_um / S * 1e-4) ** 2) if args.voxel_um else None,
          "chunks_fetched": vol.n_fetch, "mb_fetched": vol.bytes_fetch / 1e6, "seconds": time.time() - t0,
          "composite_layers": [c - hw, c + hw], "renderer": "survey/render_tifxyz.py"}
    json.dump(rm, open(os.path.join(out, "render_meta.json"), "w"), indent=1)
    print(f"done {out}: {L} x {OH} x {OW}, valid {100*(omask>0).mean():.1f}%, {vol.n_fetch} chunks "
          f"({vol.bytes_fetch/1e6:.0f} MB) in {time.time()-t0:.0f}s")
    return stack, omask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volume", required=True, help="zarr root URL or local path (OME-Zarr v2 multiscale)")
    ap.add_argument("--level", default="0", help="multiscale level to sample (default 0)")
    ap.add_argument("--level-scale", type=float, default=None, help="override coordinate scale for the level (default 2^-level)")
    ap.add_argument("--tifxyz", required=True, help="tifxyz directory (x.tif y.tif z.tif meta.json)")
    ap.add_argument("--out", required=True, help="output dir, e.g. renders/<scroll>/<segment>")
    ap.add_argument("--layers", type=int, default=65)
    ap.add_argument("--step", type=float, default=1.0, help="layer spacing in level-0 voxels")
    ap.add_argument("--scale", type=float, default=1.0, help="output pixels per level-0 voxel")
    ap.add_argument("--seg-scale", type=float, default=1.0, help="multiply tifxyz coordinates (mesh made on another volume)")
    ap.add_argument("--no-flip-normals", dest="flip_normals", action="store_false",
                    help="by default normals are negated like the published renders (--flip-normals in VC3D)")
    ap.add_argument("--grid-crop", type=int, nargs=4, metavar=("X0", "Y0", "W", "H"), help="crop in tifxyz grid cells")
    ap.add_argument("--auto-crop", action="store_true", help="crop to the bbox of valid grid points")
    ap.add_argument("--surface-interp", choices=["bicubic", "linear"], default="linear",
                    help="linear reproduces the published surface volumes (corr 0.9997 vs official PHerc0800 render)")
    ap.add_argument("--px-offset", type=float, default=0.5,
                    help="output pixel u samples grid position (u + px_offset) * scale; 0.5 reproduces the official renders")
    ap.add_argument("--vox-offset", type=float, default=0.0, help="constant added to xyz before sampling (voxel-centre convention)")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--cache-mb", type=int, default=1200)
    ap.add_argument("--disk-cache", default=None, help="optional dir for fetched chunks")
    ap.add_argument("--workers", type=int, default=8, help="concurrent HTTP downloads (I/O bound)")
    ap.add_argument("--composite-half", type=int, default=8, help="composites use mid +- this many layers")
    ap.add_argument("--voxel-um", type=float, default=None, help="level-0 voxel size, recorded in render_meta.json")
    args = ap.parse_args()
    render(args)


if __name__ == "__main__":
    main()
