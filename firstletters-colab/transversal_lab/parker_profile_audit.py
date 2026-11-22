#!/usr/bin/env python3
"""Frozen, detector-free depth audit of one already published Parker 2019 letter.

Original implementation; no GPL reference code copied. Data are CC BY 4.0:
Parker et al. (2019), https://doi.org/10.1371/journal.pone.0215775 .
--self-test and the default dry plan never access source CT pixel arrays.
--execute requires an external cache and the published protocol/manifest.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import struct
import threading
import time
import urllib.request
import zlib

# Set before importing NumPy so this process can coexist with other agent work.
NUMERICAL_THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
for variable in NUMERICAL_THREAD_ENV:
    os.environ[variable] = '1'

import numpy as np
import PIL
from PIL import Image


HERE = Path(__file__).resolve().parent


def runtime_metadata():
    return {
        'python_version': sys.version.split()[0], 'numpy_version': np.__version__,
        'pillow_version': PIL.__version__,
        'numerical_thread_environment': {key: os.environ[key] for key in NUMERICAL_THREAD_ENV},
        'declared_dtypes': {'ppm': '<f8', 'ct_header_mode': 'I;16B',
                            'ct_numpy_expected': '>u2', 'profiles': np.dtype(np.uint16).str},
    }


def sanitized_failure(cache, freeze_reference, protocol_hash, state, exc):
    # Never serialize exception strings, redirected URLs, request IDs, or arbitrary headers.
    write_json(cache / 'failure.json', {
        'published_freeze_reference': freeze_reference, 'protocol_sha256': protocol_hash,
        'implementation_sha256': digest(Path(__file__).read_bytes()),
        'stage': state.get('stage'), 'error_type': type(exc).__name__,
        'http_code': getattr(exc, 'code', None), 'runtime': runtime_metadata(),
    })


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def describe(values: np.ndarray, labels: np.ndarray) -> dict:
    """Fixed-sign descriptive AUC, with half credit for tied pairs."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    out = {'n_positive': n1, 'n_negative': n0}
    if min(n1, n0) < 1:
        out['signed_auc'] = None
        return out
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite descriptor; do not silently remove samples')
    positive, negative = values[labels], values[~labels]
    _, group = np.unique(values, return_inverse=True)
    p = np.bincount(group, weights=labels.astype(float))
    n = np.bincount(group, weights=(~labels).astype(float))
    earlier_negative = np.cumsum(n) - n
    auc = np.sum(p * (earlier_negative + 0.5 * n)) / (n1 * n0)
    delta = float(positive.mean() - negative.mean())
    positive_ss = (n1 - 1) * positive.var(ddof=1) if n1 > 1 else 0.0
    negative_ss = (n0 - 1) * negative.var(ddof=1) if n0 > 1 else 0.0
    pooled_variance = (positive_ss + negative_ss) / (n1 + n0 - 2) if n1 + n0 > 2 else 0.0
    out.update({
        'positive_mean': float(positive.mean()),
        'negative_mean': float(negative.mean()),
        'positive_quantiles_0_25_50_75_100': np.quantile(positive, [0, .25, .5, .75, 1]).tolist(),
        'negative_quantiles_0_25_50_75_100': np.quantile(negative, [0, .25, .5, .75, 1]).tolist(),
        'signed_mean_difference': delta,
        'pooled_sd_contrast': delta / np.sqrt(pooled_variance) if pooled_variance > 0 else None,
        'signed_auc': float(auc),
    })
    return out


def common_shift_support(eligible, xx, yy, dx, dy):
    sx, sy = xx + dx, yy + dy
    inside = (sx >= 0) & (sx < eligible.shape[1]) & (sy >= 0) & (sy < eligible.shape[0])
    selected = np.flatnonzero(inside)
    selected = selected[eligible[sy[selected], sx[selected]]]
    return selected, sx[selected], sy[selected]


def sample_negative_x(array_yx, points_xyz, depths):
    """Integer source indices; no interpolation, normal repair, or clipping."""
    return array_yx[points_xyz[:, 1, None], points_xyz[:, 0, None] - depths[None, :]]


def self_test():
    labels = np.array([False, False, True, True])
    assert describe(np.arange(4), labels)['signed_auc'] == 1.0
    assert describe(-np.arange(4), labels)['signed_auc'] == 0.0
    assert describe(np.ones(4), labels)['signed_auc'] == 0.5
    assert describe(np.array([0, 1, 2]), np.array([0, 0, 1]))['signed_auc'] == 1.0
    assert describe(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))['signed_auc'] == 0.5
    # An independently legible synthetic coordinate encoding catches axis/sign errors.
    zz, yy, xx = np.indices((4, 5, 9))
    volume = 100 * zz + 10 * yy + xx
    got = sample_negative_x(volume[3], np.array([[4, 2, 3]]), np.arange(-2, 3))
    np.testing.assert_array_equal(got, [[326, 325, 324, 323, 322]])
    eligible = np.ones((3, 4), dtype=bool)
    eligible[1, 2] = False
    chosen, sx, sy = common_shift_support(eligible, np.array([0, 1, 3]), np.array([1, 1, 1]), 1, 0)
    np.testing.assert_array_equal(chosen, [0])
    np.testing.assert_array_equal(sx, [1])
    np.testing.assert_array_equal(sy, [1])
    print('Generated-array integrity checks passed; no donor data accessed.')


def load_frozen_inputs(protocol_path):
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    manifest_path = protocol_path.parent / protocol['manifest']
    manifest_bytes = manifest_path.read_bytes()
    if digest(manifest_bytes) != protocol['manifest_sha256']:
        raise RuntimeError('Acquisition manifest changed after protocol freeze')
    document_path = protocol_path.parent / protocol['protocol_document']
    if digest(document_path.read_bytes()) != protocol['protocol_document_sha256']:
        raise RuntimeError('Prose protocol changed after protocol freeze')
    manifest = json.loads(manifest_bytes)
    expected = {
        'schema_version': 1,
        'surface_shape_yx': [2530, 1440],
        'source_shape_zyx': [2530, 1440, 286],
        'grid_origin_xy': [8, 8],
        'grid_stride_xy': [16, 16],
        'depth_offsets_voxels': list(range(-16, 17)),
        'normal_xyz': [-1, 0, 0],
        'eligible_source_x_inclusive': [16, 269],
        'label_shift_axes': ['x', 'y'],
        'label_shift_distances_pixels': [-256, -64, -16, 16, 64, 256],
        'cpu_time_cap_seconds': 7200,
    }
    for key, value in expected.items():
        if protocol[key] != value:
            raise RuntimeError(f'Unsupported protocol change: {key}; publish a new implementation prospectively')
    return protocol, manifest, digest(protocol_bytes)


def inspect_tiff_header(data):
    with Image.open(io.BytesIO(data)) as image:
        if image.size != (286, 1440) or image.mode != 'I;16B' or getattr(image, 'n_frames', 1) != 1:
            raise RuntimeError('Unexpected source TIFF dimensions, dtype, endian order, or page count')


class PublicArchive:
    def __init__(self, cache, protocol, manifest):
        self.cache, self.protocol, self.manifest = cache, protocol, manifest
        self.lock = threading.Lock()
        self.receipts = []
        self.allocated = {'source_slice': 0, 'geometry_or_supplied_label': 0}
        self.received_successfully = 0
        self.url = manifest['public_download_url']
        self.etag = manifest['archive_etag_observed']
        # Reuse only the normal public download redirect, in memory. No signed URL is logged.
        request = urllib.request.Request(self.url, method='HEAD')
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.headers.get('ETag') != self.etag or int(response.headers.get('Content-Length', -1)) != manifest['archive_bytes']:
                raise RuntimeError('Public archive version or length changed')
            self.transport_url = response.url
        self.members_dir = cache / 'members'
        self.members_dir.mkdir(exist_ok=True)

    def _validate(self, data, member):
        if len(data) != member['uncompressed_bytes'] or f'{zlib.crc32(data) & 0xffffffff:08x}' != member['crc32']:
            raise RuntimeError('ZIP member CRC32 or length failed')
        value = digest(data)
        expected = member.get('sha256_verified_preflight')
        if expected is not None and value != expected:
            raise RuntimeError('Preflight SHA256 changed')
        if member['role'] == 'source_slice':
            inspect_tiff_header(data)
        return value

    def retrieve(self, member):
        basename = Path(member['name']).name
        destination = self.members_dir / basename
        # Preflight files may already be directly in the external cache.
        local = destination if destination.exists() else self.cache / basename
        if local.exists():
            data = local.read_bytes()
            value = self._validate(data, member)
            receipt = {'source': self.url, 'member': member['name'], 'member_crc32': member['crc32'],
                       'member_sha256': value, 'reused_verified_local_member': True}
            self._record(receipt)
            return local
        start, end = member['range_start'], member['range_end']
        for attempt in range(self.protocol['retries_per_member'] + 1):
            # Reservation is a conservative upper bound, including failed attempts.
            reserve = end - start + 2
            with self.lock:
                self.allocated[member['role']] += reserve
                if (self.allocated['source_slice'] > self.protocol['source_download_cap_bytes'] or
                        sum(self.allocated.values()) > self.protocol['all_archive_download_cap_bytes']):
                    raise RuntimeError('Frozen download cap exceeded')
            began = time.monotonic()
            try:
                request = urllib.request.Request(self.transport_url, headers={
                    'Range': f'bytes={start}-{end}', 'If-Match': self.etag,
                    'User-Agent': 'published-ink-control-audit/1.0',
                })
                with urllib.request.urlopen(request, timeout=30) as response:
                    if (response.status != 206 or response.headers.get('Content-Range') !=
                            f'bytes {start}-{end}/{self.manifest["archive_bytes"]}' or
                            response.headers.get('ETag') != self.etag):
                        raise RuntimeError('HTTP206, exact Content-Range, and pinned ETag are required')
                    packet = response.read(reserve)
                if len(packet) != end - start + 1:
                    raise RuntimeError('Unexpected bounded response length')
                header = struct.unpack_from('<4s5H3L2H', packet)
                name = packet[30:30 + header[-2]].decode('utf-8')
                if header[0] != b'PK\x03\x04' or name != member['name'] or header[3] != 8:
                    raise RuntimeError('Unexpected local ZIP header or member')
                payload = 30 + header[-2] + header[-1]
                data = zlib.decompress(packet[payload:payload + member['compressed_bytes']], -15)
                value = self._validate(data, member)
                destination.write_bytes(data)
                receipt = {
                    'source': self.url, 'member': member['name'], 'first_byte': start, 'last_byte': end,
                    'http_status': 206, 'etag': self.etag, 'range_bytes': len(packet),
                    'range_sha256': digest(packet), 'member_crc32': member['crc32'],
                    'member_sha256': value, 'attempt': attempt + 1, 'seconds': time.monotonic() - began,
                    'transport_reused_normal_public_redirect': True,
                }
                with self.lock:
                    self.received_successfully += len(packet)
                self._record(receipt)
                return destination
            except Exception as exc:
                self._record({'source': self.url, 'member': member['name'], 'first_byte': start,
                              'last_byte': end, 'attempt': attempt + 1, 'failed': True,
                              'error_type': type(exc).__name__, 'http_code': getattr(exc, 'code', None)})
                if attempt == self.protocol['retries_per_member']:
                    raise RuntimeError(f'Public member retrieval failed: {basename}; {type(exc).__name__}') from None
        raise RuntimeError('Unreachable')

    def _record(self, receipt):
        with self.lock:
            self.receipts.append(receipt)
            write_json(self.cache / 'parker_download_receipts.json', {
                'requests': self.receipts,
                'reserved_bytes_upper_bound': self.allocated,
                'successfully_received_bytes': self.received_successfully,
                'whole_archive_sha256_locally_verified': False,
            })


def geometry(paths, protocol):
    ppm_path = paths['lunate-sigma.ppm']
    expected_header = b'width: 1440\nheight: 2530\ndim: 6\nordered: true\ntype: double\nversion: 1\n<>\n'
    with ppm_path.open('rb') as handle:
        if handle.read(73) != expected_header:
            raise RuntimeError('Unexpected PPM header')
    ppm = np.memmap(ppm_path, dtype='<f8', mode='r', offset=73, shape=(2530, 1440, 6))
    mask = np.asarray(Image.open(paths['lunate-sigma_mask.png']))
    with Image.open(paths['lunate-sigma_ink-label.png']) as label_image:
        if label_image.mode != '1':
            raise RuntimeError('Supplied binary label encoding changed; do not threshold grayscale labels')
        labels = np.asarray(label_image)
    if labels.dtype != np.bool_ or set(np.unique(labels).tolist()) != {False, True}:
        raise RuntimeError('Expected exactly the preflight binary label values')
    if mask.shape != (2530, 1440) or labels.shape != mask.shape:
        raise RuntimeError('Supplied mask or label dimensions changed')
    eligible = ((mask == 255) & np.isfinite(ppm).all(axis=2) &
                (ppm[:, :, 0] >= 16) & (ppm[:, :, 0] <= 269) &
                (ppm[:, :, 1] >= 0) & (ppm[:, :, 1] < 1440) &
                (ppm[:, :, 2] >= 0) & (ppm[:, :, 2] < 2530))
    yy, xx = np.meshgrid(np.arange(8, 2530, 16), np.arange(8, 1440, 16), indexing='ij')
    keep = eligible[yy, xx]
    yy, xx = yy[keep], xx[keep]
    points = np.asarray(ppm[yy, xx])
    if (not np.all(points[:, 3:] == [-1., 0., 0.]) or not np.all(points[:, 1] == xx) or
            not np.all(points[:, 2] == yy) or not np.all(points[:, :3] == np.floor(points[:, :3]))):
        raise RuntimeError('PPM axes, normal, or integer source coordinates changed')
    if len(xx) != protocol['expected_grid_points'] or int(labels[yy, xx].sum()) != protocol['expected_positive_labels']:
        raise RuntimeError('Prospective sample eligibility changed')
    if np.unique(yy).tolist() != protocol['source_slice_ids']:
        raise RuntimeError('Source slice list changed')
    return eligible, labels, yy, xx, points[:, :3].astype(int)


def execute(cache, protocol, manifest, protocol_hash, freeze_reference, state):
    # Hard process CPU limit, independent of Python checkpoints and network waits.
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_CPU)
    finite_limits = [v for v in (old_soft, old_hard, protocol['cpu_time_cap_seconds']) if v != resource.RLIM_INFINITY]
    cap = min(finite_limits)
    def cpu_limit_handler(signum, frame):
        error = RuntimeError('CPU time cap reached')
        sanitized_failure(cache, freeze_reference, protocol_hash, state, error)
        raise error
    signal.signal(signal.SIGXCPU, cpu_limit_handler)
    resource.setrlimit(resource.RLIMIT_CPU, (max(1, cap - 1), cap))
    began, cpu0 = time.monotonic(), time.process_time()
    cache.mkdir(parents=True, exist_ok=True)
    for name in ('parker_profile_results.json', 'parker_raw_profiles.npz'):
        if (cache / name).exists():
            raise RuntimeError('Existing outcome/profile output: use a new cache for a separately registered run')
    attempt = 1
    while (cache / f'parker_profile_start_{attempt:03d}.json').exists():
        attempt += 1
    start_name = f'parker_profile_start_{attempt:03d}.json'
    write_json(cache / start_name, {
        'published_freeze_reference': freeze_reference, 'protocol_sha256': protocol_hash,
        'manifest_sha256': protocol['manifest_sha256'],
        'implementation_sha256': digest(Path(__file__).read_bytes()),
        'runtime': runtime_metadata(), 'status': 'started; no outcome yet',
    })
    state['stage'] = 'public_archive_header'
    acquisition = PublicArchive(cache, protocol, manifest)
    paths = {}
    state['stage'] = 'geometry_acquisition'
    for member in manifest['members']:
        if member['role'] == 'geometry_or_supplied_label':
            paths[Path(member['name']).name] = acquisition.retrieve(member)
    state['stage'] = 'geometry_validation'
    eligible, labels, yy, xx, xyz = geometry(paths, protocol)
    sources = [m for m in manifest['members'] if m['role'] == 'source_slice']
    source_paths = {}
    state['stage'] = 'source_member_acquisition'
    with concurrent.futures.ThreadPoolExecutor(max_workers=protocol['download_workers']) as pool:
        futures = {pool.submit(acquisition.retrieve, member): member for member in sources}
        try:
            for future in concurrent.futures.as_completed(futures):
                member = futures[future]
                path = future.result()
                index = int(path.stem.replace('flatfielded-slices', ''))
                source_paths[index] = path
                if len(source_paths) % 10 == 0 or len(source_paths) == len(sources):
                    print(json.dumps({'source_members_ready': len(source_paths), 'total': len(sources)}), flush=True)
        except BaseException:
            for pending in futures:
                pending.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    state['stage'] = 'source_profile_extraction'
    depths = np.asarray(protocol['depth_offsets_voxels'])
    profiles = np.empty((len(xx), len(depths)), dtype=np.uint16)
    observed_ct_dtypes = set()
    for index, path in sorted(source_paths.items()):
        sample_indices = np.flatnonzero(xyz[:, 2] == index)
        with Image.open(path) as image:
            array = np.asarray(image)
            if array.shape != (1440, 286) or array.dtype.kind != 'u' or array.dtype.itemsize != 2:
                raise RuntimeError('Decoded CT dimensions or unsigned16 dtype changed')
            observed_ct_dtypes.add(array.dtype.str)
            profiles[sample_indices] = sample_negative_x(array, xyz[sample_indices], depths)
    np.savez_compressed(cache / 'parker_raw_profiles.npz', surface_x=xx, surface_y=yy, source_xyz=xyz,
                        depths=depths, profiles=profiles, labels=labels[yy, xx])
    state['stage'] = 'frozen_descriptive_statistics'
    features = {f'depth_{int(d):+d}': profiles[:, j].astype(float) for j, d in enumerate(depths)}
    features['profile_mean'] = profiles.mean(axis=1)
    features['front_minus_back'] = (profiles[:, (depths >= 1) & (depths <= 8)].mean(axis=1) -
                                   profiles[:, (depths >= -8) & (depths <= -1)].mean(axis=1))
    features['geometry_x_nuisance'] = xyz[:, 0].astype(float)
    base_labels = labels[yy, xx]
    native = {}
    for name, values in features.items():
        outcome = describe(values, base_labels)
        row_auc = []
        for row in np.unique(yy):
            selected = yy == row
            row_result = describe(values[selected], base_labels[selected])
            if row_result['signed_auc'] is not None:
                row_auc.append(row_result['signed_auc'])
        outcome['rows_with_both_classes'] = len(row_auc)
        outcome['row_macro_signed_auc'] = float(np.mean(row_auc)) if row_auc else None
        native[name] = outcome
    shifts = []
    for axis in protocol['label_shift_axes']:
        for amount in protocol['label_shift_distances_pixels']:
            selected, sx, sy = common_shift_support(eligible, xx, yy,
                                                   amount if axis == 'x' else 0,
                                                   amount if axis == 'y' else 0)
            shifted_labels = labels[sy, sx]
            comparisons = {name: {
                'native_same_intersection': describe(values[selected], base_labels[selected]),
                'shifted_labels_same_intersection': describe(values[selected], shifted_labels),
            } for name, values in features.items()}
            shifts.append({'axis': axis, 'offset_surface_pixels': amount, 'n_common': int(len(selected)),
                           'comparisons': comparisons})
    results = {
        'published_freeze_reference': freeze_reference, 'start_receipt': start_name,
        'runtime': runtime_metadata(), 'observed_ct_numpy_dtypes': sorted(observed_ct_dtypes),
        'profile_dtype': profiles.dtype.str,
        'protocol_sha256': protocol_hash, 'implementation_sha256': digest(Path(__file__).read_bytes()),
        'manifest_sha256': protocol['manifest_sha256'], 'case': manifest['case'], 'independent_cases': 1,
        'approximately_micrometres_per_voxel': protocol['approximately_micrometres_per_voxel'],
        'supplied_normal_axis_xyz': protocol['normal_xyz'], 'unique_sample_count': len(xx),
        'native': native, 'fixed_label_shifts': shifts,
        'cpu_seconds': time.process_time() - cpu0, 'wall_seconds': time.monotonic() - began,
        'reserved_download_bytes_upper_bound': acquisition.allocated,
        'successfully_received_archive_bytes': acquisition.received_successfully,
        'scope': protocol['scope'], 'whole_archive_sha256_locally_verified': False,
    }
    state['stage'] = 'writing_results'
    write_json(cache / 'parker_profile_results.json', results)
    state['stage'] = 'complete'
    print(json.dumps({'status': 'complete', 'sample_points': len(xx), 'cpu_seconds': results['cpu_seconds'],
                      'wall_seconds': results['wall_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=HERE / 'parker_profile_protocol.json')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--freeze-reference', help='40-character published commit containing the frozen code/protocol')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    protocol, manifest, protocol_hash = load_frozen_inputs(args.protocol)
    source = [m for m in manifest['members'] if m['role'] == 'source_slice']
    plan = {
        'protocol_sha256': protocol_hash, 'manifest_sha256': protocol['manifest_sha256'],
        'source_member_count': len(source), 'source_compressed_bytes': sum(m['compressed_bytes'] for m in source),
        'source_exact_range_bytes': sum(m['range_end'] - m['range_start'] + 1 for m in source),
        'source_download_cap_bytes': protocol['source_download_cap_bytes'],
        'cpu_time_cap_seconds': protocol['cpu_time_cap_seconds'], 'execute': args.execute,
    }
    if len(source) != 151 or plan['source_compressed_bytes'] != 107664592:
        raise RuntimeError('Frozen acquisition plan changed')
    print(json.dumps(plan), flush=True)
    if not args.execute:
        return
    if not args.freeze_reference or not re.fullmatch(r'[0-9a-fA-F]{40}', args.freeze_reference):
        parser.error('--execute requires --freeze-reference with a 40-character published commit')
    if args.cache_dir is None:
        parser.error('--execute requires --cache-dir outside the repository')
    cache = args.cache_dir.resolve()
    repository = HERE.parents[1]
    if cache == repository or repository in cache.parents:
        parser.error('Keep source CT, labels, and raw profiles outside the repository')
    cache.mkdir(parents=True, exist_ok=True)
    state = {'stage': 'starting'}
    try:
        execute(cache, protocol, manifest, protocol_hash, args.freeze_reference.lower(), state)
    except Exception as exc:
        sanitized_failure(cache, args.freeze_reference.lower(), protocol_hash, state, exc)
        raise SystemExit('Parker audit failed; see sanitized failure.json in the external cache.') from None


if __name__ == '__main__':
    main()
