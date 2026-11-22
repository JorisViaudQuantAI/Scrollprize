#!/usr/bin/env python3
"""Transport-only resumption of the frozen Parker published-letter audit.

Imports the exact v1 implementation, makes each bounded request through the
original public OSF URL, and preserves all scientific inputs/calculations.
Does not alter signed URLs, credentials, access controls, or source objects.
Default invocation validates the amendment and prints its plan without network.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(amendment_path):
    amendment_bytes = amendment_path.read_bytes()
    amendment = json.loads(amendment_bytes)
    document = amendment_path.parent / amendment['amendment_document']
    if sha(document.read_bytes()) != amendment['amendment_document_sha256']:
        raise RuntimeError('Transport amendment document changed')
    runner = amendment_path.parent / amendment['frozen_runner']
    if sha(runner.read_bytes()) != amendment['frozen_runner_sha256']:
        raise RuntimeError('Frozen v1 runner changed')
    spec = importlib.util.spec_from_file_location('parker_frozen_v1', runner)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    protocol, manifest, protocol_hash = base.load_frozen_inputs(amendment_path.parent / amendment['frozen_protocol'])
    if protocol_hash != amendment['frozen_protocol_sha256']:
        raise RuntimeError('Frozen scientific protocol changed')
    if amendment['additional_attempts_per_uncached_member'] != 1:
        raise RuntimeError('This amendment authorizes only one new attempt per missing member')
    if amendment['source_download_cap_bytes'] != protocol['source_download_cap_bytes'] or amendment['all_archive_download_cap_bytes'] != protocol['all_archive_download_cap_bytes']:
        raise RuntimeError('Original aggregate download caps must remain unchanged')
    return amendment, sha(amendment_bytes), base, protocol, manifest, protocol_hash


def preserved_attempt(cache, amendment):
    for filename, expected in amendment['preserved_attempt001_sha256'].items():
        path = cache / filename
        if not path.is_file() or sha(path.read_bytes()) != expected:
            raise RuntimeError('Preserved attempt001 receipt is missing or changed')
    prior = json.loads((cache / 'parker_attempt001_download_receipts.json').read_text())
    success = [r for r in prior['requests'] if not r.get('failed')]
    source_success = [r for r in success if '/slices/' in r['member']]
    if len(source_success) != amendment['cached_source_members_from_attempt001']:
        raise RuntimeError('Attempt001 cache provenance count changed')
    prior_hashes = {r['member']: r['member_sha256'] for r in success}
    if prior['reserved_bytes_upper_bound']['source_slice'] != amendment['attempt001_source_reserved_upper_bound_bytes']:
        raise RuntimeError('Prior source-byte ledger changed')
    return prior, prior_hashes


def run(cache, freeze_reference, amendment, amendment_hash, base, protocol, manifest, protocol_hash):
    prior, prior_hashes = preserved_attempt(cache, amendment)
    source_seed = (amendment['attempt001_source_reserved_upper_bound_bytes'] +
                   amendment['preflight_source_range_bytes'])
    metadata_seed = amendment['preflight_metadata_reserved_upper_bound_bytes']
    transport_state = {
        'scientific_freeze_reference': amendment['scientific_freeze_reference'],
        'transport_freeze_reference': freeze_reference,
        'transport_amendment_sha256': amendment_hash,
        'transport_wrapper_sha256': sha(Path(__file__).read_bytes()),
        'logical_attempt': 2,
        'route': 'Original public canonical OSF URL for every bounded range; ordinary redirects only.',
        'additional_attempts_per_uncached_member': 1,
        'preflight_source_range_bytes': amendment['preflight_source_range_bytes'],
        'preflight_metadata_reserved_upper_bound_bytes': metadata_seed,
        'attempt001_source_reserved_upper_bound_bytes': amendment['attempt001_source_reserved_upper_bound_bytes'],
        'successful_byte_counter_scope': 'Successful attempt001 plus attempt002 HTTP range bodies; preflight bytes separately reserved in cumulative caps.',
    }
    original_runtime_metadata = base.runtime_metadata

    def runtime_metadata():
        values = original_runtime_metadata()
        values['transport_amendment'] = dict(transport_state)
        return values

    base.runtime_metadata = runtime_metadata
    original_archive = base.PublicArchive

    class CanonicalPublicArchive(original_archive):
        def __init__(self, data_cache, execution_protocol, execution_manifest):
            super().__init__(data_cache, execution_protocol, execution_manifest)
            # This is the published public URL, not an altered/extended signed URL.
            self.transport_url = self.url
            self.allocated = {'source_slice': source_seed,
                              'geometry_or_supplied_label': metadata_seed}
            self.received_successfully = prior['successfully_received_bytes']

        def _validate(self, data, member):
            value = super()._validate(data, member)
            previous = prior_hashes.get(member['name'])
            if previous is not None and value != previous:
                raise RuntimeError('Cached member SHA256 differs from preserved attempt001 receipt')
            return value

        def _record(self, receipt):
            receipt = dict(receipt)
            receipt['logical_attempt'] = 2
            receipt['transport_route'] = 'original_public_canonical_url'
            super()._record(receipt)

    base.PublicArchive = CanonicalPublicArchive
    execution_protocol = dict(protocol)
    # Strictly one additional attempt per missing member; no fresh retries.
    execution_protocol['retries_per_member'] = 0
    state = {'stage': 'transport_resumption_start'}
    try:
        base.execute(cache, execution_protocol, manifest, protocol_hash, freeze_reference, state)
    except Exception as exc:
        base.sanitized_failure(cache, freeze_reference, protocol_hash, state, exc)
        raise SystemExit('Parker transport resumption stopped; see sanitized failure.json.') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amendment', type=Path, default=HERE / 'parker_transport_amendment.json')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--freeze-reference', help='40-character published transport-amendment commit')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    amendment, amendment_hash, base, protocol, manifest, protocol_hash = load(args.amendment)
    if args.self_test:
        base.self_test()
        return
    print(json.dumps({
        'amendment_sha256': amendment_hash, 'scientific_protocol_sha256': protocol_hash,
        'cached_source_members': amendment['cached_source_members_from_attempt001'],
        'remaining_source_members': protocol['source_slice_count'] - amendment['cached_source_members_from_attempt001'],
        'cumulative_source_reserved_seed_bytes': amendment['attempt001_source_reserved_upper_bound_bytes'] + amendment['preflight_source_range_bytes'],
        'preflight_metadata_reserved_upper_bound_bytes': amendment['preflight_metadata_reserved_upper_bound_bytes'],
        'additional_attempts_per_uncached_member': 1, 'execute': args.execute,
    }), flush=True)
    if not args.execute:
        return
    if args.cache_dir is None or not args.freeze_reference or not re.fullmatch(r'[0-9a-fA-F]{40}', args.freeze_reference):
        parser.error('--execute requires an external --cache-dir and 40-character published --freeze-reference')
    cache = args.cache_dir.resolve()
    repository = HERE.parents[1]
    if cache == repository or repository in cache.parents:
        parser.error('Raw donor data and results must remain outside the repository')
    cache.mkdir(parents=True, exist_ok=True)
    run(cache, args.freeze_reference.lower(), amendment, amendment_hash, base, protocol, manifest, protocol_hash)


if __name__ == '__main__':
    main()
