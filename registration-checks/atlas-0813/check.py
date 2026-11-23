"""One public winding, 24 geometry-selected sites, one CPU worker. See protocol.txt."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from xml.etree import ElementTree as ET

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'BLOSC_NTHREADS', 'ZARR_ASYNC__CONCURRENCY', 'ZARR_THREADING__MAX_WORKERS'):
    os.environ[key] = '1'

import numpy as np
from numcodecs import blosc
import requests
from scipy import ndimage, signal
import tifffile

blosc.set_nthreads(1)
D = Path(__file__).resolve().parent
BUCKET = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
CT = 'PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr'
M7 = 'PHerc0813/representations/predictions/surfaces/20250821151723-surface-20260413222639-surface-m7-L0-th0.2.zarr'
W = 'z11296_w020'
ATLAS = 'https://raw.githubusercontent.com/pscamillo/vesuvius-eligible-meshes/66000516c97e043b696ce874f145e29638109c91/meshes/PHerc0813/' + W
AUTHOR = 'https://raw.githubusercontent.com/Nieuwlaar/pherc343-first-letters/e8370f1f88a211166056df62ea6fcfffe4a1063e/mesh/snap_meshes.py'
HASH = {'x.tif': '672c6eaeb0e4f4de5b897ed9c2ca2f09c348d23debccc1227b33c798cafce75e',
        'y.tif': '0c444ebe9a1d7556e09800c9d29c0968d19b375f8e74a7abb6d62f0b1ea2b1f1',
        'z.tif': '6a9cd84208207b3ccc38ea0802bbe603452e2c25d92c34d3186fb46ae16079d5',
        'meta.json': '39fe809524ba4efc002f5f7a0f1456f9fb47b71b0f7a6de7f41772c3d164182a',
        'snap_meshes.py': '253e48b31072be8c95b890715b80e90e4efbb34b5b0687a63b7d4a0a7a87857d'}
SESSION = requests.Session()
PITCH = 9.362


def sha(data):
    return hashlib.sha256(data).hexdigest()


def get(url, **kwargs):
    for retry in range(4):
        try:
            r = SESSION.get(url, timeout=(6, 30), **kwargs)
            if r.status_code not in [429, 500, 502, 503, 504] or retry == 3:
                return r
        except requests.RequestException:
            if retry == 3:
                raise
        time.sleep(1 + retry)


def required(url):
    r = get(url)
    r.raise_for_status()
    return r.content


def saved(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)


def verify_sources():
    for filename, digest in HASH.items():
        p = D / filename
        if not p.exists():
            saved(p, required(AUTHOR if filename.endswith('.py') else ATLAS + '/' + filename))
        if sha(p.read_bytes()) != digest:
            raise ValueError('Unexpected source ' + filename)
    raw = D / 'original' / W
    raw.mkdir(parents=True, exist_ok=True)
    for n in ['x.tif', 'y.tif', 'z.tif', 'meta.json']:
        if not (raw / n).exists():
            (raw / n).symlink_to(D / n)
    return raw


def real_normals(P, valid):
    tree = ast.parse((D / 'snap_meshes.py').read_text())
    orig = next(node for node in tree.body if isinstance(
        node, ast.FunctionDef) and node.name == 'normals')
    module = ast.Module(body=[orig], type_ignores=[])
    ctx = {'np': np}
    exec(compile(module, 'upstream:normals', 'exec'), ctx)
    return ctx['normals'](*P.transpose(2, 0, 1), valid)


def sample_sites(P, valid, norms):
    mask = ndimage.binary_erosion(
        valid, structure=ndimage.generate_binary_structure(
            2, 1)) & np.isfinite(norms).all(2)
    uv = np.argwhere(mask)
    ps = P[mask].astype(float)
    center = (np.array(valid.shape) - 1) / 2
    i = int(np.argmin(np.linalg.norm(uv - center, axis=1)))
    chosen = []
    dist = np.full(len(uv), np.inf)
    for _ in range(24):
        if dist[i] < 100:
            break
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(ps - ps[i], axis=1))
        i = int(np.argmax(dist))
    assert len(chosen) > 0
    sites = uv[chosen]
    xyz = ps[chosen]
    diff = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    np.fill_diagonal(diff, np.inf)
    (D / 'selected-public-geometry.json').write_text(json.dumps({'uv_yx': sites.tolist(
    ), 'point_count': len(sites), 'minimum_distance_vox': float(np.min(diff))}, indent=2) + '\n')
    return sites, float(np.min(diff))


def listed(prefix):
    r = get(BUCKET, params={'list-type': 2, 'prefix': prefix, 'delimiter': '/'})
    r.raise_for_status()
    root = ET.fromstring(r.content)
    assert root.findtext('{*}IsTruncated') == 'false'
    return [v.text for v in root.findall(
        './{*}CommonPrefixes/{*}Prefix') + root.findall('./{*}Contents/{*}Key')]


def m7_cache(P, N, ok):
    target = D / 'm7.zarr'
    meta = {}
    for key in ['.zgroup', '0/.zarray']:
        p = target / key
        if not p.exists():
            saved(p, required(BUCKET + '/' + M7 + '/' + key))
        meta[key] = json.loads(p.read_text())
    cfg = meta['0/.zarray']
    sz = np.asarray(cfg['chunks'])
    assert cfg['dimension_separator'] == '/' and np.all(sz == 192)
    xyz = P[ok][:, None, :] + np.arange(-15, 16, dtype=np.float32)[None, :, None] * N[ok][:, None, :]
    keys = np.unique(np.rint(xyz.reshape(-1, 3)[:, ::-1]).astype(int) // sz, axis=0)
    receipts = {}
    missing = []
    t = time.monotonic()
    for i, coords in enumerate(keys):
        tail = '/'.join(map(str, coords))
        key = M7 + '/0/' + tail
        p = target / '0' / tail
        if not p.exists():
            resp = get(BUCKET + '/' + key)
            if resp.status_code == 404:
                if key in listed(key):
                    raise RuntimeError('404 on a listed binary key')
                missing.append(tail)
                continue
            resp.raise_for_status()
            saved(p, resp.content)
        body = p.read_bytes()
        receipts[tail] = [len(body), sha(body)]
        if (i + 1) % 25 == 0:
            print('m7', i + 1, len(keys), round(time.monotonic() - t, 1), 's', flush=True)
    (D / 'm7-chunks.json').write_text(json.dumps({'keys': len(keys),
                                                  'stored': receipts, 'confirmed_absent': missing}, indent=1) + '\n')
    return {'planned': len(keys), 'stored': len(receipts), 'confirmed_absent': len(
        missing), 'bytes': sum(v[0] for v in receipts.values())}


def stock_snap():
    (D / 'windings.txt').write_text(W + '\n')
    log = D / 'upstream-snap.log'
    outfile = D / 'snapped' / 'snap_report.tsv'
    if outfile.exists():
        outfile.unlink()
    env = dict(os.environ, SURF_ZARR=str(D / 'm7.zarr'), RANK_PROCS='1')
    wrapper = "import sys,runpy;from numcodecs import blosc;blosc.set_nthreads(1);sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
    result = subprocess.run([sys.executable,
                             '-c',
                             wrapper,
                             str(D / 'snap_meshes.py'),
                             str(D / 'original'),
                             'PHerc0813',
                             str(D / 'snapped'),
                             '--windings',
                             str(D / 'windings.txt'),
                             '--procs',
                             '1'],
                            capture_output=True,
                            text=True,
                            env=env)
    log.write_text(result.stdout + '\n' + result.stderr)
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return (D / 'snapped' / W)


def ct_values(xyz):
    """Read only the Z slabs of uncompressed chunks needed for these trilinear samples."""
    metadata = required(BUCKET + '/' + CT + '/0/.zarray')
    (D / 'ct-metadata.json').write_bytes(metadata)
    spec = json.loads(metadata)
    assert spec['compressor'] is None and spec['dtype'] == '|u1' and spec['chunks'] == [
        128] * 3 and spec['order'] == 'C'
    pos = xyz.reshape(-1, 3)[:, ::-1]
    base = np.floor(pos).astype('i4')
    frac = pos - base
    corners = np.asarray([[z, y, x] for z in range(2) for y in range(2) for x in range(2)])
    pts = (base[:, None, :] + corners[None, :, :]).reshape(-1, 3)
    assert (pts >= 0).all() and (pts < np.asarray(spec['shape'])).all()
    weights = np.prod(np.where(corners[None, :, :], frac[:, None, :], 1 - frac[:, None, :]), axis=2)
    cidx, ix = np.unique(pts // 128, axis=0, return_inverse=True)
    vox = np.empty(len(pts), 'u1')
    receipts = []
    for i, index in enumerate(cidx):
        pick = ix == i
        local = pts[pick] % 128
        lo, hi = int(local[:, 0].min()), int(local[:, 0].max())
        start = lo * 128**2
        end = (hi + 1) * 128**2 - 1
        key = CT + '/0/' + '/'.join(map(str, index))
        p = D / 'ct-slabs' / ('_'.join(map(str, index)) + '_' + str(lo) + '_' + str(hi) + '.raw')
        if not p.exists():
            r = get(BUCKET + '/' + key, headers={'Range': f'bytes={start}-{end}'})
            if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{128**3}':
                raise ValueError((key, r.status_code, r.headers.get('Content-Range')))
            if len(r.content) != end - start + 1:
                raise ValueError('Short read for ' + key)
            saved(p, r.content)
        buf = p.read_bytes()
        block = np.frombuffer(buf, np.uint8).reshape(hi - lo + 1, 128, 128)
        vox[pick] = block[local[:, 0] - lo, local[:, 1], local[:, 2]]
        receipts.append([key, start, end, sha(buf)])
    (D / 'ct-slabs.json').write_text(json.dumps(receipts, indent=1) + '\n')
    return np.sum(vox.reshape(-1, 8) * weights, axis=1).reshape(xyz.shape[:-1]), {'ranges': len(
        receipts), 'bytes': sum(e - s + 1 for _, s, e, _ in receipts), 'sampled_trilinear_points': len(pos)}


def evaluate(P, N, S, sites):
    points = []
    snaps = []
    offsets = np.arange(-24, 24.5, .5)
    for y, x in sites:
        p = P[y, x].astype(float)
        n = N[y, x].astype(float)
        n /= np.linalg.norm(n)
        t = (P[y, x + 1] - P[y, x - 1]).astype(float)
        t -= n * (t @ n)
        t /= np.linalg.norm(t)
        v = np.cross(n, t)
        delta = S[y, x] - p
        axial = delta @ n
        lateral = np.array([[0, 0, 0], 2 * t, -2 * t, 2 * v, -2 * v])
        points.append(p + lateral[:, None, :] + offsets[None, :, None] * n)
        snaps.append((float(axial), float(np.linalg.norm(delta - axial * n))))
    traces, receipts = ct_values(np.asarray(points))
    rows = []
    for idx, (tr, (d, dt)) in enumerate(zip(traces, snaps)):
        valid = float(np.mean(tr[0] > 3))
        curve = ndimage.gaussian_filter1d(tr.mean(0), 1.2)
        peak_ix, _ = signal.find_peaks(curve, prominence=12, distance=8)
        peaks = offsets[peak_ix]
        rec = {
            'id': idx + 1,
            'material_fraction': valid,
            'n_peaks': len(peaks),
            'snap_vox': d,
            'tangential_roundoff_vox': dt}
        if valid >= .8 and len(peaks):
            raw = np.abs(peaks)
            snapped = np.abs(peaks - d)
            near0 = np.flatnonzero(np.isclose(raw, raw.min(), rtol=0, atol=1e-9))
            near1 = np.flatnonzero(np.isclose(snapped, snapped.min(), rtol=0, atol=1e-9))
            rec.update(
                raw_nearest=float(
                    raw.min()),
                snapped_nearest=float(
                    snapped.min()),
                unique_raw=len(near0) == 1,
                unique_snapped=len(near1) == 1)
            if len(near0) == 1:
                rec['after_same_peak'] = float(abs(peaks[near0[0]] - d))
                rec['signed_raw_peak'] = float(peaks[near0[0]])
                if len(near1) == 1:
                    rec['nearest_peak_changed'] = bool(near0[0] != near1[0])
        rows.append(rec)

    def gauge(sub, pred):
        a = np.array([r['raw_nearest'] for r in sub])
        b = np.array([r[pred] for r in sub])
        gain = a - b
        def median(values):
            return float(np.median(values)) if len(values) else None

        return {
            'n': len(sub),
            'median_before_vox': median(a),
            'median_after_vox': median(b),
            'median_paired_gain_vox': median(gain),
            'closer_over_quarter': int(np.sum(gain > 0.25)),
            'further_over_quarter': int(np.sum(gain < -0.25)),
            'within_quarter': int(np.sum(np.abs(gain) <= 0.25)),
            'within_two_before': int(np.sum(a <= 2)),
            'within_two_after': int(np.sum(b <= 2)),
        }

    known = [r for r in rows if 'raw_nearest' in r]
    primary = [r for r in known if r['unique_raw']]
    certain = [r for r in known if 'nearest_peak_changed' in r]
    result = {'n_sampled': len(rows),
              'n_material_supported': sum(r['material_fraction'] >= .8 for r in rows),
              'n_peaks_and_material': len(known),
              'primary_distance_to_original_peak': gauge(primary,
                                                         'after_same_peak'),
              'distance_to_any_peak': gauge(known,
                                            'snapped_nearest'),
              'unique_nearest_both': len(certain),
              'changed_nearest_peak': sum(r['nearest_peak_changed'] for r in certain),
              'median_absolute_snap_vox': float(np.median([abs(r['snap_vox']) for r in rows])),
              'max_absolute_snap_vox': float(max(abs(r['snap_vox']) for r in rows)),
              'max_tangential_roundoff_vox': float(max(r['tangential_roundoff_vox'] for r in rows)),
              'native_ct': receipts}
    return result, rows


def main():
    raw = verify_sources()
    catalog = {
        key: listed(key) for key in [
            'PHerc0813/volumes/',
            'PHerc0813/transforms/',
            'PHerc0813/representations/predictions/surfaces/']}
    (D / 'public-catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')

    def load_xyz(folder): return np.stack(
        [tifffile.imread(folder / (axis + '.tif')) for axis in 'xyz'], -1)
    P = load_xyz(raw)
    valid = np.isfinite(P).all(2) & (P > -.5).all(2)
    N, ok = real_normals(P, valid)
    sites, min_sep = sample_sites(P, valid, N)
    print('geometry', len(sites), 'min-separation vox', min_sep, flush=True)
    m7 = m7_cache(P, N, ok)
    print('downloaded m7', m7, flush=True)
    snapped = stock_snap()
    S = load_xyz(snapped)
    stats, records = evaluate(P, N, S, sites)
    out = {'winding': W,
           'atlas_commit': '66000516c97e043b696ce874f145e29638109c91',
           'upstream_snap_commit': 'e8370f1f88a211166056df62ea6fcfffe4a1063e',
           'protocol_sha256': sha((D / 'protocol.txt').read_bytes()),
           'pitch_um': PITCH,
           'n_valid_vertices': int(valid.sum()),
           'n_mesh_normals': int(ok.sum()),
           'minimum_evaluation_distance_vox': min_sep,
           'm7': m7,
           'upstream_report': (D / 'snapped' / 'snap_report.tsv').read_text(),
           'source_mesh_and_code_sha256': HASH,
           'snapped_sha256': {axis: sha((snapped / (axis + '.tif')).read_bytes()) for axis in 'xyz'},
           'metrics': stats}
    (D / 'point-metrics.json').write_text(json.dumps(records, indent=2) + '\n')
    (D / 'result.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
