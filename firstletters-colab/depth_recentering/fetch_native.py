"""Fetch the already hash-pinned public w00 fixture without depth resampling.

Raw data and download inventory belong outside the repository. Every expected
chunk must return HTTP 200 and match both length and the repository checksum;
missing, forbidden, truncated, and changed objects are errors, never zero fill.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(data):
    return sha256(data).hexdigest()


def fetch(base, key, expected, cache, expected_bytes=None):
    target = cache / key
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = False
    if target.exists():
        payload = target.read_bytes()
        cached = digest(payload) == expected
    if not cached:
        urls = [base + key, (base + key).replace(
            '.s3.us-east-1.amazonaws.com/', '.s3.amazonaws.com/')]
        for attempt in range(4):
            try:
                request = Request(urls[attempt % len(urls)], headers={
                    'User-Agent': 'Scrollprize-depth-control/1.0'})
                with urlopen(request, timeout=25) as response:
                    if response.status != 200:
                        raise RuntimeError(f'{key}: HTTP {response.status}')
                    payload = response.read()
                break
            except (HTTPError, URLError, TimeoutError, ConnectionError):
                if attempt == 3:
                    raise
                time.sleep(0.5)
        if digest(payload) != expected:
            raise ValueError(f'{key}: checksum mismatch; refusing changed source')
        if expected_bytes is not None and len(payload) != expected_bytes:
            raise ValueError(f'{key}: expected {expected_bytes} bytes, got {len(payload)}')
        partial = target.with_name(target.name + '.partial')
        partial.write_bytes(payload)
        partial.replace(target)
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise ValueError(f'{key}: incorrect cached length')
    return key, payload, {'key': key, 'bytes': len(payload),
                          'sha256': digest(payload), 'cached': cached,
                          'expected_sha256_matched': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='new scratch directory outside the repository')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == REPO or REPO in output.parents:
        parser.error('store source CT outside the repository')
    output.mkdir(parents=True, exist_ok=True)
    protocol = (HERE / 'protocol.json').read_bytes()
    frozen = output / 'protocol-before-download.json'
    if frozen.exists() and frozen.read_bytes() != protocol:
        raise RuntimeError('protocol changed after freeze')
    frozen.write_bytes(protocol)
    freeze_info = output / 'protocol-freeze.json'
    if not freeze_info.exists():
        freeze_info.write_text(json.dumps({
            'frozen_utc': datetime.now(timezone.utc).isoformat(),
            'protocol_sha256': digest(protocol),
            'pixels_inspected_before_freeze': False,
        }, indent=2) + '\n')
    manifest_path = REPO / 'evidence' / 'sources.json'
    case = json.loads(manifest_path.read_text())['cases']['w00']
    keys = {k: v for k, v in case['objects'].items() if k.startswith('0/')}
    cache = output / 'objects'
    _, metadata_bytes, metadata_check = fetch(
        case['base_url'], '0/.zarray', keys.pop('0/.zarray'), cache)
    meta = json.loads(metadata_bytes)
    if not (meta['chunks'] == [28, 128, 128] and meta['dtype'] == '|u1'
            and meta['compressor'] is None and meta['order'] == 'C'
            and meta.get('filters') is None):
        raise RuntimeError('source format changed')
    y0, y1, x0, x1 = case['source_bounds_yx']
    expected_keys = {f'0/0/{y}/{x}' for y in range(y0 // 128, y1 // 128)
                     for x in range(x0 // 128, x1 // 128)}
    if set(keys) != expected_keys:
        raise RuntimeError('manifest inventory does not exactly cover the chosen crop')
    native = np.empty(tuple(case['shape']), dtype=np.uint8)
    checks = [metadata_check]
    def get(item):
        key, checksum = item
        return fetch(case['base_url'], key, checksum, cache, 28 * 128 * 128)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for key, payload, check in pool.map(get, sorted(keys.items())):
            _, _, y, x = map(int, key.split('/'))
            ys, xs = y * 128 - y0, x * 128 - x0
            native[:, ys:ys + 128, xs:xs + 128] = np.frombuffer(
                payload, dtype=np.uint8).reshape(28, 128, 128)
            checks.append(check)
    np.save(output / 'pherc0841_native.npy', native, allow_pickle=False)
    inventory = {
        'source_case': case,
        'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_freeze': json.loads(freeze_info.read_text()),
        'source_manifest_file_sha256': digest(manifest_path.read_bytes()),
        'source_case_sha256': digest(json.dumps(case, sort_keys=True).encode()),
        'npy_sha256': digest((output / 'pherc0841_native.npy').read_bytes()),
        'array_bytes_sha256': digest(native.tobytes()),
        'shape_zyx': list(native.shape), 'dtype': str(native.dtype),
        'native_depth_planes_preserved': True,
        'objects_verified': len(checks),
        'object_bytes': sum(c['bytes'] for c in checks),
        'all_objects_required': True, 'fill_substitutions': 0,
        'objects': checks,
    }
    (output / 'source_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps({k: inventory[k] for k in (
        'shape_zyx', 'dtype', 'objects_verified', 'object_bytes',
        'array_bytes_sha256', 'npy_sha256')}, indent=2))
    print(str(output / 'pherc0841_native.npy'))


if __name__ == '__main__':
    main()
