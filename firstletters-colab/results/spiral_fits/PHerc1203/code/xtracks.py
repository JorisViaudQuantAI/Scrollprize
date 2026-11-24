#!/usr/bin/env python3
"""xtracks.py : surface-track extraction with the PINNED villa extract_surface_tracks.py (@0fb38c45), parallelised.

The per-slab algorithm is the pinned module's own (X.prepare_cc_labels + X.get_skeleton_tracks, i.e. cc3d 6-conn CC at
full res, max-pool by downsample_factor, cc3d.dust [min,max], kimimaro TEASAR scale 1 const 2, interjoint paths with
>= 10 vertices, offsets added). Only the I/O is replaced: the m7 band is first binarised (> 0, as the pinned code does)
and bit-packed along x into one memmap (fits in RAM), then N worker processes each take every N-th slab key and write
their own DBM; the parts are merged into one DBM and the pinned tracks.write_packed_track_store builds the packed store.
Parameters are set to the published stores' extract.json (organisers' 0175A / 0846A stores):
  downsample 2, z_chunk_depth_h 2, z_chunk_stride_h 8, yx_chunk_thickness_v 4, yx_stride_v 16, dust 400,
  max_area 640000, min_path_vertices 10 (hard-coded in pinned code), path_mode interjoint, kimimaro parallel 3.

usage:
  xtracks.py pack   --m7 M7_ZARR_DIR --z0 Z0 --z1 Z1 --bits BITS.npy [--procs 6]
  xtracks.py run    --bits BITS.npy --z0 Z0 --z1 Z1 --work DIR --workers N --kpar 3 [--lo-y A --lo-x B --hi-y C --hi-x D]
  xtracks.py merge  --work DIR --workers N --out OUT.dbm
"""
import argparse, dbm, json, os, pickle, sys, time, types
import numpy as np

SPIRAL = os.environ.get('SPIRAL_DIR', '/content/cache/sg/villa/volume-cartographer/scripts/spiral')
PARAMS = dict(downsample_factor=2, z_chunk_depth_h=2, z_chunk_stride_h=8, yx_chunk_thickness_v=4, yx_stride_v=16,
              dust_threshold=400, max_area_threshold=640_000, path_mode='interjoint')


def load_pinned(kpar):
    for name in ('vesuvius', 'vesuvius.data', 'vesuvius.data.utils'):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules['vesuvius.data.utils'].open_zarr = None
    sys.path.insert(0, SPIRAL)
    import extract_surface_tracks as X
    for k, v in PARAMS.items():
        setattr(X, k, v)
    real = X.kimimaro.skeletonize
    shim = types.SimpleNamespace(skeletonize=lambda *a, **k: real(*a, **{**k, 'parallel': kpar}))
    X.kimimaro = shim
    return X


def cmd_pack(a):
    import zarr
    from concurrent.futures import ProcessPoolExecutor
    arr = zarr.open(a.m7, mode='r')['0']
    Z, Y, Xs = arr.shape
    nz = a.z1 - a.z0; nb = (Xs + 7) // 8
    out = np.lib.format.open_memmap(a.bits, mode='w+', dtype=np.uint8, shape=(nz, Y, nb)); del out
    json.dump(dict(z0=a.z0, z1=a.z1, shape=[Z, Y, Xs]), open(a.bits + '.json', 'w'))
    cy = arr.chunks[1]
    jobs = [(a.m7, a.bits, a.z0, a.z1, y0, min(Y, y0 + cy)) for y0 in range(0, Y, cy)]
    t = time.time()
    with ProcessPoolExecutor(a.procs) as ex:
        for i, r in enumerate(ex.map(_pack_one, jobs)):
            if i % 5 == 0: print(f'pack {i+1}/{len(jobs)} {time.time()-t:.0f}s', flush=True)
    print('packed', a.bits, (nz, Y, nb), f'{time.time()-t:.0f}s')


def _pack_one(job):
    import zarr
    m7, bits, z0, z1, y0, y1 = job
    arr = zarr.open(m7, mode='r')['0']
    out = np.load(bits, mmap_mode='r+')
    blk = np.asarray(arr[z0:z1, y0:y1, :]) > 0
    out[:, y0:y1, :] = np.packbits(blk, axis=-1)
    out.flush(); return y1


def tasks_for(meta, a, X):
    z0, z1 = meta['z0'], meta['z1']; Z, Y, Xs = meta['shape']
    T = [('h', z) for z in range(z0, z1, X.z_chunk_stride_h)]
    T += [('vy', w) for w in range(a.lo_y, a.hi_y, X.yx_stride_v)]
    T += [('vx', w) for w in range(a.lo_x, a.hi_x, X.yx_stride_v)]
    return T


def yx_range(bits, meta):
    """same as pinned find_yx_range: 20 sampled slices of the band, bounds of predictions > 0"""
    nz = bits.shape[0]; Xs = meta['shape'][2]
    lo = np.array([meta['shape'][1], Xs]); hi = np.array([0, 0])
    for k in range(0, nz, max(1, nz // 20)):
        p = np.unpackbits(bits[k], axis=-1, count=Xs)
        yx = np.stack(np.nonzero(p), -1)
        if len(yx): lo = np.minimum(lo, yx.min(0)); hi = np.maximum(hi, yx.max(0))
    return lo, hi


def cmd_run(a):
    meta = json.load(open(a.bits + '.json'))
    if a.lo_y is None:
        lo, hi = yx_range(np.load(a.bits, mmap_mode='r'), meta)
        a.lo_y, a.lo_x, a.hi_y, a.hi_x = int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])
    os.makedirs(a.work, exist_ok=True)
    print('yx range (full-res)', (a.lo_y, a.lo_x), '..', (a.hi_y, a.hi_x), flush=True)
    json.dump(dict(lo=[a.lo_y, a.lo_x], hi=[a.hi_y, a.hi_x], params=PARAMS, z0=meta['z0'], z1=meta['z1']),
              open(os.path.join(a.work, 'run_params.json'), 'w'), indent=1)
    import subprocess
    procs = []
    for i in range(a.workers):
        cmd = [sys.executable, os.path.abspath(__file__), 'worker', '--idx', str(i), '--bits', a.bits, '--work', a.work,
               '--workers', str(a.workers), '--kpar', str(a.kpar), '--lo-y', str(a.lo_y), '--lo-x', str(a.lo_x),
               '--hi-y', str(a.hi_y), '--hi-x', str(a.hi_x)]
        procs.append(subprocess.Popen(cmd, env={**os.environ, 'OMP_NUM_THREADS': '1'}))
    rcs = [p.wait() for p in procs]
    print('worker return codes', rcs, flush=True)
    if any(rcs): sys.exit(1)


def _worker(a, i):
    X = load_pinned(a.kpar)
    meta = json.load(open(a.bits + '.json')); z0, z1 = meta['z0'], meta['z1']; Z, Y, Xs = meta['shape']
    bits = np.load(a.bits, mmap_mode='r')
    T = tasks_for(meta, a, X)
    # interleave families so that every worker gets a mix; worker i takes every N-th
    mine = T[i::a.workers]
    t0 = time.time(); n = 0; ntr = 0
    log = open(os.path.join(a.work, f'worker{i}.log'), 'a')
    with dbm.open(os.path.join(a.work, f'part{i}.dbm'), 'c') as db:
        for fam, w in mine:
            key = f'{fam}:{w}'
            if key in db: continue
            if fam == 'h':
                zz1 = min(w + X.z_chunk_depth_h, z1, Z)
                if zz1 <= w: continue
                p = np.unpackbits(bits[w - z0:zz1 - z0], axis=-1, count=Xs); off = (w, 0, 0)
            elif fam == 'vy':
                wm = min(w + X.yx_chunk_thickness_v, Y)
                if wm - w < X.downsample_factor: db[key] = pickle.dumps([]); continue
                p = np.unpackbits(bits[:, w:wm, :], axis=-1, count=Xs); off = (z0, w, 0)
            else:
                wm = min(w + X.yx_chunk_thickness_v, Xs)
                if wm - w < X.downsample_factor: db[key] = pickle.dumps([]); continue
                b0, b1 = w // 8, (wm - 1) // 8 + 1
                p = np.unpackbits(bits[:, :, b0:b1], axis=-1)[:, :, w - 8 * b0: wm - 8 * b0]; off = (z0, 0, w)
            p = np.ascontiguousarray(p.astype(np.uint8))
            if p.max() == 0:
                db[key] = pickle.dumps([]); continue
            cc = X.prepare_cc_labels(p)
            tr = X.get_skeleton_tracks(cc, offset_zyx=off)
            db[key] = pickle.dumps(tr)
            n += 1; ntr += len(tr)
            if n % 20 == 0:
                log.write(f'{time.strftime("%H:%M:%S")} worker {i}: {n}/{len(mine)} keys, {ntr} tracks, {time.time()-t0:.0f}s\n'); log.flush()
    return f'worker {i}: {n} keys, {ntr} tracks, {time.time()-t0:.0f}s'


def cmd_merge(a):
    sys.path.insert(0, SPIRAL)
    n = 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with dbm.open(a.out, 'n') as out:
        for i in range(a.workers):
            with dbm.open(os.path.join(a.work, f'part{i}.dbm'), 'r') as db:
                for k in db.keys():
                    out[k] = db[k]; n += 1
    print('merged keys', n, 'backend', dbm.whichdb(a.out))
    from tracks import write_packed_track_store
    write_packed_track_store(a.out, force=True, show_progress=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd')
    ap.add_argument('--m7'); ap.add_argument('--z0', type=int); ap.add_argument('--z1', type=int)
    ap.add_argument('--bits'); ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--work'); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--kpar', type=int, default=3)
    ap.add_argument('--lo-y', type=int); ap.add_argument('--lo-x', type=int); ap.add_argument('--hi-y', type=int); ap.add_argument('--hi-x', type=int)
    ap.add_argument('--out'); ap.add_argument('--idx', type=int)
    a = ap.parse_args()
    np.random.seed(0)
    if a.cmd == 'worker': print(_worker(a, a.idx))
    else: {'pack': cmd_pack, 'run': cmd_run, 'merge': cmd_merge}[a.cmd](a)
