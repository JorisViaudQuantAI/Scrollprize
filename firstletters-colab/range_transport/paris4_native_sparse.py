#!/usr/bin/env python3
"""Pinned sparse S3 HEAD+Range control; no whole object GET or fill fallback."""
import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DIR = Path(__file__).resolve().parent

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('redirect prohibited')


def validate_head(code, headers, protocol):
    if code != 200:
        raise ValueError('HEAD status is not 200')
    if headers.get('Content-Length') != str(protocol['full_chunk_bytes']):
        raise ValueError('HEAD size mismatch')
    etag = headers.get('ETag', '')
    if len(etag) < 3 or not (etag.startswith('"') and etag.endswith('"')):
        raise ValueError('missing strong ETag')
    if headers.get('Content-Encoding', 'identity').lower() != 'identity':
        raise ValueError('HEAD encoded object')
    return etag


def validate_part(code, headers, protocol, etag):
    if code != 206:
        raise ValueError('Range status is not 206')
    start, end = protocol['range_inclusive']
    if headers.get('Content-Range') != f'bytes {start}-{end}/{protocol["full_chunk_bytes"]}':
        raise ValueError('Content-Range mismatch')
    if headers.get('Content-Length') != str(end - start + 1):
        raise ValueError('Range body length mismatch')
    if headers.get('ETag') != etag:
        raise ValueError('ETag mismatch between HEAD and Range')
    if headers.get('Content-Encoding', 'identity').lower() != 'identity':
        raise ValueError('encoded Range')


def run(protocol):
    receipt = {'schema': 'scrollprize.paris4-native-sparse-result.v1',
               'passed': False, 'requests': 0, 'body_bytes': 0,
               'whole_get_requests': 0, 'observations': []}
    url = protocol['chunk_url']
    start, end = protocol['range_inclusive']
    expected_bytes = end - start + 1
    if not (expected_bytes == protocol['selected_bytes'] == 65 * 128 * 128):
        raise ValueError('frozen geometry inconsistency')
    opener = urllib.request.build_opener(NoRedirect())
    try:
        t = time.monotonic()
        receipt['requests'] += 1
        req = urllib.request.Request(url, method='HEAD', headers={'Accept-Encoding':'identity'})
        with opener.open(req, timeout=protocol['timeout_seconds']) as response:
            etag = validate_head(response.status, response.headers, protocol)
            if response.read(1):
                raise ValueError('unexpected HEAD response body')
        receipt['observations'].append({'method':'HEAD', 'status':200, 'bytes':0,
                                          'elapsed_s':round(time.monotonic()-t,6)})
        t = time.monotonic()
        receipt['requests'] += 1
        req = urllib.request.Request(url, method='GET', headers={
            'Accept-Encoding':'identity', 'Range':f'bytes={start}-{end}', 'If-Match':etag})
        with opener.open(req, timeout=protocol['timeout_seconds']) as response:
            validate_part(response.status, response.headers, protocol, etag)
            data = response.read(expected_bytes + 1)
        receipt['body_bytes'] += len(data)
        receipt['observations'].append({'method':'GET', 'status':206, 'bytes':len(data),
                                         'elapsed_s':round(time.monotonic()-t,6)})
        if len(data) != expected_bytes:
            raise ValueError('partial response body size mismatch')
        if receipt['requests'] > protocol['max_requests'] or receipt['body_bytes'] > protocol['max_body_bytes']:
            raise ValueError('budget exceeded')
        actual_sha = hashlib.sha256(data).hexdigest()
        if actual_sha != protocol['pinned_selected_sha256']:
            raise ValueError('new partial bytes do not match prior whole-GET reference')
        receipt['passed'] = True
        receipt['selected_sha256'] = actual_sha
        receipt['pinned_reference_match'] = True
        receipt['etag_strong'] = True
        receipt['fraction_vs_full_get_application_body'] = expected_bytes / protocol['full_chunk_bytes']
    except (AssertionError, ValueError, urllib.error.URLError, TimeoutError, OSError) as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
    return receipt


def self_test():
    from email.message import Message
    def headers(p):
        h=Message()
        for a,b in p.items():h[a]=b
        return h
    p={'full_chunk_bytes':1785856, 'range_inclusive':[360448,1425407]}
    base={'Content-Length':'1785856','ETag':'"one"'}
    assert validate_head(200, headers(base), p) == '"one"'
    for hd in [{'Content-Length':'123','ETag':'"one"'},
               {'Content-Length':'1785856','ETag':'W/"one"'},
               {'Content-Length':'1785856','ETag':'"one"','Content-Encoding':'gzip'}]:
        try:validate_head(200,headers(hd),p)
        except ValueError:pass
        else:raise AssertionError('invalid HEAD accepted')
    part={'Content-Length':'1064960','ETag':'"one"','Content-Range':'bytes 360448-1425407/1785856'}
    validate_part(206,headers(part),p,'"one"')
    for status, hd in [(200,part), (206,{**part,'ETag':'"two"'}),
                       (206,{**part,'Content-Range':'bytes 0-1064959/1785856'})]:
        try:validate_part(status,headers(hd),p,'"one"')
        except ValueError:pass
        else:raise AssertionError('invalid partial accepted')
    return 8


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.self_test:
        print('offline sparse tests passed:',self_test())
        return 0
    if not args.output:parser.error('--output required')
    p=json.loads((DIR/'paris4_native_sparse_protocol.json').read_text())
    r=run(p)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
    print('RESULT_JSON='+json.dumps(r,sort_keys=True))
    return 0 if r['passed'] else 1

if __name__=='__main__':sys.exit(main())
