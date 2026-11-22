#!/usr/bin/env python3
"""Final bounded public-transport tranche for the unchanged Parker audit.

No scientific implementation is copied or altered: import the exact frozen v1.
Refresh ordinary public redirects under a lock, never modify signed URLs, and
permit at most two further attempts per still-missing frozen archive member.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace
import urllib.request

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    raw = path.read_bytes()
    amendment = json.loads(raw)
    document = path.parent / amendment['amendment_document']
    if sha(document.read_bytes()) != amendment['amendment_document_sha256']:
        raise RuntimeError('Final transport document changed')
    original = path.parent / amendment['frozen_runner']
    if sha(original.read_bytes()) != amendment['frozen_runner_sha256']:
        raise RuntimeError('Original scientific runner changed')
    spec = importlib.util.spec_from_file_location('parker_frozen_v1_final_transport', original)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    protocol, manifest, protocol_hash = base.load_frozen_inputs(path.parent / amendment['frozen_protocol'])
    if protocol_hash != amendment['frozen_protocol_sha256']:
        raise RuntimeError('Original scientific protocol changed')
    if (amendment['additional_attempts_per_missing_member'] != 2 or
            amendment['redirect_max_local_age_seconds'] != 20 or
            amendment['request_timeout_seconds'] != 45 or
            amendment['download_workers'] != 4):
        raise RuntimeError('Unexpected final transport allowance')
    for key in ['source_download_cap_bytes', 'all_archive_download_cap_bytes']:
        if amendment[key] != protocol[key]:
            raise RuntimeError('Original cumulative byte caps must remain unchanged')
    return amendment, sha(raw), base, protocol, manifest, protocol_hash


def verify_preserved(cache, amendment):
    for filename, expected in amendment['preserved_prior_receipt_sha256'].items():
        path = cache / filename
        if not path.is_file() or sha(path.read_bytes()) != expected:
            raise RuntimeError('A preserved prior receipt is missing or changed')
    prior = json.loads((cache / 'parker_attempt002_download_receipts.json').read_text())
    hashes = {r['member']: r['member_sha256'] for r in prior['requests'] if not r.get('failed')}
    source_count = sum('/slices/' in name for name in hashes)
    if source_count != amendment['historically_verified_source_members']:
        raise RuntimeError('Historical source count changed')
    if prior['reserved_bytes_upper_bound'] != amendment['cumulative_reserved_seed_bytes']:
        raise RuntimeError('Cumulative prior byte ledger changed')
    return prior, hashes


def run(cache, freeze_reference, amendment, amendment_hash, base, protocol, manifest, protocol_hash):
    prior, prior_hashes = verify_preserved(cache, amendment)
    state = {'stage': 'final_transport_start'}
    for name in ['parker_profile_results.json', 'parker_raw_profiles.npz', 'parker_profile_start_003.json']:
        if (cache / name).exists():
            raise RuntimeError('A completed or already-started final attempt cannot be overwritten')
    # Exclusive sentinel prevents a repeated final tranche in the same cache.
    with (cache / 'parker_final_transport_started.json').open('x') as handle:
        json.dump({'published_transport_reference': freeze_reference,
                   'amendment_sha256': amendment_hash,
                   'wrapper_sha256': sha(Path(__file__).read_bytes()),
                   'logical_attempt': 3}, handle, indent=2)
        handle.write('\n')
    transport_state = {
        'scientific_freeze_reference': amendment['scientific_freeze_reference'],
        'previous_transport_reference': amendment['previous_transport_reference'],
        'final_transport_reference': freeze_reference,
        'final_transport_amendment_sha256': amendment_hash,
        'final_transport_wrapper_sha256': sha(Path(__file__).read_bytes()),
        'logical_attempt': 3,
        'redirect_max_local_age_seconds': 20,
        'request_timeout_seconds': 45,
        'additional_attempts_per_missing_member': 2,
        'redirect_refreshes': 0,
        'cumulative_reserved_seed_bytes': amendment['cumulative_reserved_seed_bytes'],
        'successful_byte_counter_scope': 'Successful attempt001+002+003 range bodies; preflight separately reserved in cumulative caps.',
    }
    original_runtime = base.runtime_metadata

    def runtime_metadata():
        value = original_runtime()
        value['final_transport_amendment'] = dict(transport_state)
        return value

    base.runtime_metadata = runtime_metadata
    # Isolate this transport adapter to the imported frozen module; do not mutate
    # the process-wide urllib module or any scientific function.
    def open_with_declared_timeout(request, *args, **kwargs):
        kwargs['timeout'] = 45
        return urllib.request.urlopen(request, *args, **kwargs)

    base.urllib = SimpleNamespace(request=SimpleNamespace(
        Request=urllib.request.Request, urlopen=open_with_declared_timeout))
    original_archive = base.PublicArchive

    class ShortLivedPublicArchive(original_archive):
        def __init__(self, data_cache, execution_protocol, execution_manifest):
            self.cache, self.protocol, self.manifest = data_cache, execution_protocol, execution_manifest
            self.lock = threading.Lock()
            self.receipts = []
            self.allocated = dict(amendment['cumulative_reserved_seed_bytes'])
            self.received_successfully = prior['successfully_received_bytes']
            self.url = execution_manifest['public_download_url']
            self.etag = execution_manifest['archive_etag_observed']
            self.members_dir = data_cache / 'members'
            self.members_dir.mkdir(exist_ok=True)
            self._redirect_lock = threading.Lock()
            self._thread = threading.local()
            self._redirect_url = None
            self._redirect_time = float('-inf')

        @property
        def transport_url(self):
            force = getattr(self._thread, 'refresh_before_retry', False)
            with self._redirect_lock:
                expired_for_this_policy = time.monotonic() - self._redirect_time >= 20
                if force or self._redirect_url is None or expired_for_this_policy:
                    request = urllib.request.Request(self.url, method='HEAD')
                    with urllib.request.urlopen(request, timeout=45) as response:
                        if (response.status != 200 or response.headers.get('ETag') != self.etag or
                                int(response.headers.get('Content-Length', -1)) != self.manifest['archive_bytes']):
                            raise RuntimeError('Public redirect HEAD failed frozen source checks')
                        self._redirect_url = response.url
                    self._redirect_time = time.monotonic()
                    transport_state['redirect_refreshes'] += 1
                self._thread.refresh_before_retry = False
                # Only this normal public response supplies the URL; never edit it.
                return self._redirect_url

        def _validate(self, data, member):
            value = super()._validate(data, member)
            expected = prior_hashes.get(member['name'])
            if expected is not None and value != expected:
                raise RuntimeError('Previously verified member SHA256 changed')
            return value

        def _record(self, receipt):
            if receipt.get('failed'):
                # Original retrieve() invokes this on the failed first attempt;
                # its next property lookup must obtain a fresh ordinary redirect.
                self._thread.refresh_before_retry = True
            receipt = dict(receipt)
            receipt['logical_attempt'] = 3
            receipt['transport_route'] = 'ordinary_public_redirect_cached_under_20_seconds'
            super()._record(receipt)

    base.PublicArchive = ShortLivedPublicArchive
    execution_protocol = dict(protocol)
    execution_protocol['retries_per_member'] = 1  # two attempts total in this final tranche
    execution_protocol['download_workers'] = 4
    try:
        base.execute(cache, execution_protocol, manifest, protocol_hash, freeze_reference, state)
    except Exception as exc:
        base.sanitized_failure(cache, freeze_reference, protocol_hash, state, exc)
        raise SystemExit('Final Parker transport tranche stopped; no more retries this session.') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amendment', type=Path, default=HERE / 'parker_transport_final_amendment.json')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--freeze-reference', help='40-character published final transport commit')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    amendment, amendment_hash, base, protocol, manifest, protocol_hash = load(args.amendment)
    if args.self_test:
        base.self_test()
        return
    print(json.dumps({'amendment_sha256': amendment_hash, 'scientific_protocol_sha256': protocol_hash,
                      'historically_verified_members': 135, 'missing_if_prior_cache_present': 16,
                      'fresh_cache_may_require_all_source_members': 151,
                      'cumulative_reserved_seed_bytes': amendment['cumulative_reserved_seed_bytes'],
                      'maximum_additional_attempts_per_missing_member': 2,
                      'redirect_cache_seconds': 20, 'request_timeout_seconds': 45,
                      'execute': args.execute}), flush=True)
    if not args.execute:
        return
    if args.cache_dir is None or not args.freeze_reference or not re.fullmatch(r'[0-9a-fA-F]{40}', args.freeze_reference):
        parser.error('--execute requires an external cache and 40-character published freeze reference')
    cache = args.cache_dir.resolve()
    repository = HERE.parents[1]
    if cache == repository or repository in cache.parents:
        parser.error('Keep source arrays and outcome files outside the repository')
    cache.mkdir(parents=True, exist_ok=True)
    run(cache, args.freeze_reference.lower(), amendment, amendment_hash, base, protocol, manifest, protocol_hash)


if __name__ == '__main__':
    main()
