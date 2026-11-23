#!/usr/bin/env python3
"""Frozen one-chunk native Paris 4 S3 exact-value Range control, stdlib only."""
import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_metadata(payload, specification):
    assert digest(payload) == specification['metadata_sha256'], 'metadata SHA256 mismatch'
    data = json.loads(payload)
    for field, wanted in specification['zarray'].items():
        assert data[field] == wanted, 'metadata field mismatch: ' + field


def check_headers(status, headers, expected_status, count, requested=None, etag=None):
    if status != expected_status:
        raise ValueError('HTTP status mismatch: got %s, wanted %s' % (status, expected_status))
    if headers.get('Content-Encoding', 'identity').lower() != 'identity':
        raise ValueError('transformed response')
    if headers.get('Content-Length') != str(count):
        raise ValueError('unexpected Content-Length')
    if requested is not None and headers.get('Content-Range') != requested:
        raise ValueError('unexpected Content-Range')
    if etag is not None and headers.get('ETag') != etag:
        raise ValueError('ETag changed')
    if etag is None and expected_status == 200:
        value = headers.get('ETag', '')
        if not (value.startswith('"') and value.endswith('"') and len(value) > 2):
            raise ValueError('missing strong object ETag')
    return True


class DenyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('redirect refused before following')


class Meter:
    def __init__(self, limits):
        self.opener = urllib.request.build_opener(DenyRedirect())
        self.requests = 0
        self.body_bytes = 0
        self.limits = limits
        self.observations = []

    def get(self, url, cap, headers, status, label, requested=None, etag=None):
        if self.requests >= self.limits['max_requests']:
            raise ValueError('request cap')
        if self.body_bytes + cap > self.limits['max_body_bytes']:
            raise ValueError('body cap including possible response overrun')
        self.requests += 1
        started = time.monotonic()
        request = urllib.request.Request(url, headers=headers, method='GET')
        with self.opener.open(request, timeout=self.limits['timeout_seconds']) as response:
            check_headers(response.status, response.headers, status, cap, requested, etag)
            body = response.read(cap + 1)
            self.body_bytes += len(body)
            if len(body) != cap:
                raise ValueError('response body size mismatch')
            got_etag = response.headers.get('ETag')
        self.observations.append({'label': label, 'status': status, 'bytes': len(body),
                                  'elapsed_s': round(time.monotonic() - started, 6)})
        return body, got_etag


def measure(spec):
    z = spec['z_slice_halfopen']
    plane_bytes = spec['chunk_shape_zyx'][1] * spec['chunk_shape_zyx'][2]
    start, stop = z[0] * plane_bytes, z[1] * plane_bytes
    expected_full = spec['full_chunk_bytes']
    selected = stop - start
    assert z[0] >= 0 and z[1] <= spec['chunk_shape_zyx'][0] and z[0] < z[1]
    assert expected_full == plane_bytes * spec['chunk_shape_zyx'][0]
    assert selected == spec['selected_bytes']
    assert [start, stop - 1] == spec['range_inclusive']
    m = Meter(spec['limits'])
    receipt = {'schema': 'scrollprize.paris4-native-range-result.v1', 'passed': False,
               'requests': 0, 'body_bytes': 0, 'observations': [],
               'selected_bytes': selected, 'full_chunk_bytes': expected_full}
    try:
        meta, _ = m.get(spec['metadata_url'], spec['metadata_bytes'],
                        {'Accept-Encoding': 'identity'}, 200, 'metadata')
        check_metadata(meta, spec)
        whole, etag = m.get(spec['chunk_url'], expected_full,
                            {'Accept-Encoding': 'identity'}, 200, 'whole')
        cr = 'bytes %d-%d/%d' % (start, stop - 1, expected_full)
        part, _ = m.get(spec['chunk_url'], selected,
                        {'Accept-Encoding': 'identity', 'Range': 'bytes=%d-%d' % (start, stop - 1),
                         'If-Match': etag}, 206, 'partial', requested=cr, etag=etag)
        if whole[start:stop] != part:
            raise ValueError('selected bytes differ from whole source')
        receipt['passed'] = True
        receipt['full_sha256'] = digest(whole)
        receipt['selected_sha256'] = digest(part)
        receipt['mismatch_count'] = 0
        receipt['etag_strong'] = True
        receipt['application_body_fraction_partial_to_whole'] = selected / expected_full
        area = spec['fixed_glyph_context_chunks']
        receipt['projection_only'] = {'full_chunk_count': area,
                                       'whole_chunk_body_bytes': area * expected_full,
                                       'one_range_per_chunk_body_bytes': area * selected,
                                       'potential_body_bytes_avoided': area * (expected_full - selected),
                                       'extra_request_count_over_whole_chunk': 0}
    except (ValueError, AssertionError, urllib.error.URLError, TimeoutError, OSError) as error:
        receipt['error'] = '%s: %s' % (type(error).__name__, error)
    finally:
        receipt['requests'] = m.requests
        receipt['body_bytes'] = m.body_bytes
        receipt['observations'] = m.observations
    return receipt


def self_test():
    import email.message
    def h(x):
        headers = email.message.Message()
        for k, v in x.items(): headers[k] = v
        return headers
    check_headers(206, h({'Content-Length':'12','Content-Range':'bytes 3-14/20',
                          'ETag':'"abc"'}), 206, 12, 'bytes 3-14/20', '"abc"')
    cases = [
      (200, {'Content-Length':'12','Content-Range':'bytes 3-14/20','ETag':'"abc"'}, 206, 'bytes 3-14/20','"abc"'),
      (206, {'Content-Length':'12','Content-Range':'bytes 4-15/20','ETag':'"abc"'}, 206, 'bytes 3-14/20','"abc"'),
      (206, {'Content-Length':'12','Content-Range':'bytes 3-14/20','ETag':'"other"'}, 206, 'bytes 3-14/20','"abc"'),
      (200, {'Content-Length':'20','ETag':'W/"weak"'}, 200, None,None),
      (200, {'Content-Length':'20','ETag':'"abc"','Content-Encoding':'gzip'}, 200,None,None),
    ]
    for status, hd, want, cr, etag in cases:
        try:
            check_headers(status, h(hd), want, int(hd['Content-Length']), cr, etag)
        except ValueError:
            continue
        raise AssertionError('negative control incorrectly accepted')
    p = {'zarray':{'shape':[109, 51000, 39980], 'chunks':[109,128,128]},
         'metadata_sha256':digest(b'{"shape":[109,51000,39980],"chunks":[109,128,128]}')}
    check_metadata(b'{"shape":[109,51000,39980],"chunks":[109,128,128]}', p)
    try:
        check_metadata(b'{"shape":[108,51000,39980],"chunks":[109,128,128]}', p)
    except AssertionError:
        pass
    else:
        raise AssertionError('altered metadata passed')
    return 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_test:
        print('offline deterministic tests passed:', self_test())
        return 0
    if args.output is None:
        parser.error('--output required')
    spec = json.loads((HERE / 'paris4_native_range_protocol.json').read_text())
    receipt = measure(spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print('RESULT_JSON=' + json.dumps(receipt, sort_keys=True))
    return 0 if receipt['passed'] else 1

if __name__ == '__main__':
    sys.exit(main())
