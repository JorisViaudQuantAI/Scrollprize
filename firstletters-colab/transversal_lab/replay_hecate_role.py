#!/usr/bin/env python3
"""Portable fresh reproduction of the fixed Hecate role experiment.

This PACKAGING path was check-tested only; it was NOT fully replayed in the
packaging session. Scientific outputs belong to the separately published live
role amendment, not to this file's generated-array checks.

Usage (from a clone containing the frozen companion JSONs and hecate_probe.py):
  python firstletters-colab/transversal_lab/replay_hecate_role.py check
  python firstletters-colab/transversal_lab/replay_hecate_role.py synthetic-check
  python firstletters-colab/transversal_lab/replay_hecate_role.py run \
      --out /absolute/new/external/hecate-replay \
      --freeze-reference FULL_PUBLISHED_COMMIT

By default run obtains exact hash-pinned public Hecate code/checkpoint assets.
Optional --hecate-source FILE --checkpoint FILE reuse verified model assets.
Optional --reuse-cache DIR --reuse-receipt source_acquisition.json reuse only
hash-verified raw source assets, at the phase where each is allowed. No prior
features, fitted heads, thresholds, runtime receipts or outcomes are reused.
The two published evidence/*source_acquisition.json files are mandatory input
integrity manifests: every acquired/reused source asset must match their bytes
and SHA256. New execution and feature receipts need not match old receipts.

Dependencies: numpy, torch (CPU), scipy, scikit-learn, Pillow, numcodecs. The live
run used Python 3.12; numpy 2.3.5, torch 2.6.0+cpu, scipy 1.17.0,
scikit-learn 1.8.0, Pillow 13.0.0 and numcodecs 0.16.5. Fresh numerical/ZIP
provenance need not reproduce historical feature-file or runtime hashes.

841 fits heads; background-only 9B calibrates; 814 remains inaccessible until
both heads and all four head-order/calibration-order threshold sets are frozen.
All eight orientation combinations are reported, with no selected winner.
During run, a label/mask support guard precedes checkpoint acquisition and CT;
it uses the unchanged mapped cores and does not select or replace contexts.
Check is metadata-only; synthetic-check uses generated arrays, never labels.
The hard CPU guard uses POSIX resource limits (the published run used Linux).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import resource
import shutil
import signal
import sys
import tempfile
import threading
import time
from urllib.parse import urlparse
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROLE_SHA256 = '40340f7d76639ee3ef2d0e8baa68e9c61701342dc5ecd10df19ab2cbb8f3d048'
MAX_CPU_SECONDS = 1800
MAX_FORWARDS = 96
MAX_SOURCE_BYTES = 514000000
MAX_RUN_BYTES = 1073741824
SOURCE_RECEIPTS = {
    'hecate_v1_source_acquisition.json': '8fc138c17ba26453355575d6d1721437ce5f9b9fd705ce37cfb3758c6bb2eb26',
    'hecate_role_source_acquisition.json': '9178251390f8296045d194f50338050d557392ac8aed1fc5ee2cb467146f712a',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def import_exact(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def contract(args):
    if sha(args.role_protocol) != ROLE_SHA256:
        raise ValueError('Published role protocol bytes changed')
    role = json.loads(args.role_protocol.read_text())
    checks = [(args.original_runner, role['original_runner_sha256']),
              (args.original_protocol, role['original_protocol_sha256']),
              (args.manifest, role['original_manifest_sha256'])]
    for path, expected in checks:
        if sha(path) != expected:
            raise ValueError('A frozen scientific companion file changed')
    base = json.loads(args.original_protocol.read_text())
    manifest = json.loads(args.manifest.read_text())
    expected_roles = {'train': ('PHerc0841', 'calibration'),
                      'calibration': ('PHerc0009B', 'train'),
                      'test': ('PHerc0814', 'test')}
    donors = {d['split']: d for d in manifest['donors']}
    if len(donors) != 3 or len(manifest['donors']) != 3:
        raise ValueError('Exactly three frozen physical donors are required')
    for logical, (scroll, original_split) in expected_roles.items():
        mapping = role['logical_roles_to_original_splits'][logical]
        if (mapping['scroll'], mapping['original_split']) != (scroll, original_split):
            raise ValueError('Role mapping changed')
        if donors[original_split]['scroll'] != scroll or len(donors[original_split]['contexts']) != 16:
            raise ValueError('Frozen donor identity or sixteen-context selection changed')
    expected = {'maximum_forwards': MAX_FORWARDS, 'maximum_cpu_seconds': MAX_CPU_SECONDS,
                'maximum_source_bytes_excluding_checkpoint': MAX_SOURCE_BYTES,
                'maximum_data_and_derived_bytes': MAX_RUN_BYTES, 'threads': 4,
                'interop_threads': 1, 'network_workers': 4}
    if any(base['budgets'][key] != value for key, value in expected.items()):
        raise ValueError('Original complete fresh-run budgets changed')
    return role, base, manifest, published_source_expectations()


def canonical(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.query or parsed.username or parsed.password:
        raise ValueError('Only ordinary canonical public HTTPS sources are permitted')


def published_source_expectations():
    """Read only public integrity metadata, never a source CT/label body."""
    index = {}
    for name, expected_sha in SOURCE_RECEIPTS.items():
        path = HERE / 'evidence' / name
        if sha(path) != expected_sha:
            raise ValueError('Published source-integrity receipt changed')
        for record in json.loads(path.read_text())['completed_assets']:
            canonical(record['url'])
            if not re.fullmatch(r'[0-9a-f]{64}', record['sha256']) or int(record['bytes']) < 0:
                raise ValueError('Malformed published source-integrity record')
            if record['url'] in index and index[record['url']] != record:
                raise ValueError('Published source receipts disagree on canonical input bytes')
            index[record['url']] = record
    return index


def fresh_role_mapping(role):
    descriptions = {'train': 'freshly extract 841 features and fit the fixed readouts',
                    'calibration': 'freshly extract 9B background features and freeze all thresholds',
                    'test': 'freshly extract 814 only after both heads and all four threshold sets freeze'}
    return {logical: {**item, 'use': descriptions[logical]}
            for logical, item in role['logical_roles_to_original_splits'].items()}


def disk_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


class TestAccessGate:
    """Reject test-asset byte access, including cache hits, until freeze integrity."""
    def __init__(self, donor):
        self.prefix = donor['source_zarr_url'].rstrip('/') + '/'
        self.exact = {donor['label_url']}
        if donor.get('geometry_mask_url'):
            self.exact.add(donor['geometry_mask_url'])
        self.freeze_path = None
        self.freeze_sha = None
        self.test_requests = 0
        self.lock = threading.Lock()

    def is_test(self, url):
        return url in self.exact or url.startswith(self.prefix)

    def allow(self, url):
        canonical(url)
        if not self.is_test(url):
            return
        if self.freeze_path is None or sha(self.freeze_path) != self.freeze_sha:
            raise ValueError('Held-out CT/geometry/labels are locked until an intact readout freeze')
        with self.lock:
            self.test_requests += 1

    def unlock(self, path, expected_sha):
        if self.freeze_path is not None or sha(path) != expected_sha:
            raise ValueError('Invalid or repeated test-access unlock')
        document = json.loads(Path(path).read_text())
        if not document.get('all_four_head_order_calibration_order_threshold_sets_frozen'):
            raise ValueError('All head-order/calibration-order threshold sets must be frozen')
        self.freeze_path, self.freeze_sha = Path(path), expected_sha


def reuse_index(args):
    if (args.reuse_cache is None) != (args.reuse_receipt is None):
        raise ValueError('Supply both --reuse-cache and --reuse-receipt, or neither')
    if args.reuse_cache is None:
        return {}, None
    receipt = json.loads(args.reuse_receipt.read_text())
    records = receipt.get('completed_assets')
    if not isinstance(records, list):
        raise ValueError('Reuse receipt must describe canonical source assets, not outcomes/features')
    index = {}
    for record in records:
        canonical(record['url'])
        if not re.fullmatch(r'[0-9a-f]{64}', record['sha256']) or int(record['bytes']) < 0:
            raise ValueError('Malformed source-cache integrity record')
        if record['url'] in index and index[record['url']] != record:
            raise ValueError('Conflicting cache provenance for one canonical source')
        index[record['url']] = record
    return index, sha(args.reuse_receipt)


def source_cache(original, root, gate, args, source_pins):
    index, receipt_sha = reuse_index(args)

    class FreshSourceCache(original.DownloadCache):
        def __init__(self):
            super().__init__(root, MAX_SOURCE_BYTES)
            self.reused_bytes = 0
            self.network_bytes = 0

        def _save_receipt_locked(self):
            original.save_json(self.receipt_path, {
                'charged_source_asset_bytes': self.bytes,
                'network_body_bytes': self.network_bytes,
                'verified_local_source_bytes': self.reused_bytes,
                'maximum_source_bytes_excluding_checkpoint': self.maximum_bytes,
                'source_chunks': 'All bytes must match the published source-integrity manifests; mutable changed inputs fail.',
                'source_integrity_manifest_sha256': SOURCE_RECEIPTS,
                'completed_assets': list(self.records.values()), 'failed_assets': self.failures,
                'reuse_receipt_sha256': receipt_sha,
                'no_prior_features_heads_or_outcomes_reused': True,
                'held_out_access_gate_open': gate.freeze_path is not None,
            })

        def get(self, url, expected_sha256=None, expected_bytes=None):
            # The gate is deliberately before local existence checks, hashing or copying.
            gate.allow(url)
            pin = source_pins.get(url)
            if pin is None:
                raise ValueError('Source asset absent from the published reference input manifests')
            if (expected_sha256 is not None and expected_sha256 != pin['sha256'] or
                    expected_bytes is not None and expected_bytes != pin['bytes']):
                raise ValueError('Frozen scientific asset expectation conflicts with reference source manifest')
            expected_sha256, expected_bytes = pin['sha256'], pin['bytes']
            name = hashlib.sha256(url.encode()).hexdigest()
            destination = self.root / name
            if destination.exists():
                record = self.records.get(name)
                if record is None or sha(destination) != record['sha256']:
                    raise ValueError('Current fresh-run source cache changed')
                if expected_sha256 and record['sha256'] != expected_sha256:
                    raise ValueError('Current source hash differs from frozen asset pin')
                if expected_bytes is not None and destination.stat().st_size != expected_bytes:
                    raise ValueError('Current source byte count differs from frozen count')
                return destination
            old = index.get(url)
            if old is not None:
                source = args.reuse_cache / name
                if source.is_file():
                    if args.reuse_cache.resolve() not in source.resolve().parents:
                        raise ValueError('Reuse source escapes the explicitly supplied cache')
                    actual_size, actual_sha = source.stat().st_size, sha(source)
                    if (actual_size != old['bytes'] or actual_sha != old['sha256'] or
                            expected_sha256 is not None and actual_sha != expected_sha256 or
                            expected_bytes is not None and actual_size != expected_bytes):
                        raise ValueError('Verified local source-cache bytes do not match provenance')
                    with self.lock:
                        if self.bytes + self.reserved_bytes + actual_size > self.maximum_bytes:
                            raise ValueError('Fresh source budget exceeded, including reused bytes')
                        self.reserved_bytes += actual_size
                    try:
                        if disk_bytes(args.out) + actual_size > MAX_RUN_BYTES:
                            raise ValueError('Fresh run disk cap exceeded by source-cache copy')
                        shutil.copyfile(source, destination)
                        if sha(destination) != actual_sha:
                            raise ValueError('Source changed during verified copy')
                    finally:
                        with self.lock:
                            self.reserved_bytes -= actual_size
                    with self.lock:
                        self.bytes += actual_size
                        self.reused_bytes += actual_size
                        self.records[name] = {'url': url, 'bytes': actual_size, 'sha256': actual_sha,
                                              'origin': 'verified_local_source_cache'}
                        self._save_receipt_locked()
                    return destination
            try:
                path = super().get(url, expected_sha256, expected_bytes)
            finally:
                # .bytes is concurrency-aware in the original cache. The exact network
                # total is its charged total minus copied local bytes, not per-call deltas.
                with self.lock:
                    self.network_bytes = self.bytes - self.reused_bytes
                    self._save_receipt_locked()
            if disk_bytes(args.out) > MAX_RUN_BYTES:
                raise ValueError('Fresh run disk cap exceeded during source acquisition')
            return path

    return FreshSourceCache()


def model_asset(out, name, url, expected_sha, expected_bytes, local, state):
    """Obtain fixed public code/weights; never import unverified source bytes."""
    canonical(url)
    destination = out / 'model-assets' / name
    maximum = expected_bytes if expected_bytes is not None else (1 << 20)
    if local is not None and (not local.is_file() or sha(local) != expected_sha or
                              expected_bytes is not None and local.stat().st_size != expected_bytes):
        raise ValueError('Explicit local model asset differs from its frozen pin')
    count = 0
    digest = hashlib.sha256()
    stream = local.open('rb') if local is not None else urlopen(url, timeout=45)
    with stream, destination.open('xb') as output:
        while True:
            block = stream.read(min(1 << 20, maximum - count + 1))
            if not block:
                break
            count += len(block)
            if count > maximum or disk_bytes(out) + len(block) > MAX_RUN_BYTES:
                raise ValueError('Pinned model asset or total disk cap exceeded')
            digest.update(block)
            output.write(block)
    if digest.hexdigest() != expected_sha or expected_bytes is not None and count != expected_bytes:
        raise ValueError('Public model asset did not match frozen integrity requirements')
    state.setdefault('model_assets', []).append({'canonical_url': url, 'bytes': count,
        'sha256': digest.hexdigest(), 'origin': 'verified_local_file' if local is not None else 'public_download'})
    save(out / 'model_acquisition.json', state['model_assets'])
    return destination


def same_coordinates(np, data):
    for field in ('labels', 'label_id', 'context_index'):
        if not np.array_equal(data['source'][field], data['reverse'][field]):
            raise ValueError('Input directions do not share the frozen scored label coordinates')


def known_support_upper_bound(original, donor, protocol, labels, geometry):
    """Fixed label/mask-only bound; CT nonzero support can only reduce it."""
    import numpy as np
    seen, counts, excluded = set(), [0, 0], []
    for context in donor['contexts']:
        if geometry is not None:
            coverage = float(np.mean(geometry[original.geometry_footprint(donor, context)] > 0))
            if coverage < protocol['sampling']['minimum_geometry_footprint_coverage']:
                excluded.append(context['id'])
                continue
        yy, xx = original.label_indices(donor, context)
        ids = (yy[:, None] * donor['label_shape_yx'][1] + xx[None]).ravel()
        values = labels[np.ix_(yy, xx)].ravel()
        known = values != donor['label_values']['unknown']
        unique = np.array([int(identifier) not in seen for identifier in ids], dtype=bool)
        valid = known & unique
        seen.update(map(int, ids[valid]))
        counts[0] += int(np.sum(values[valid] == donor['label_values']['background']))
        counts[1] += int(np.sum(values[valid] == donor['label_values']['ink']))
    return {'upper_bound_counts_background_ink': counts,
            'geometry_excluded_context_ids': excluded,
            'scope': 'fixed categorical labels and geometry only; CT nonzero support is not yet known',
            'context_selection_or_replacement': False}


def execute(args, role, protocol, manifest, source_pins, state, start_wall, start_cpu):
    original = import_exact(args.original_runner, 'fresh_replay_exact_hecate_probe')
    original.preflight(protocol, manifest)  # Original JSON-only geometry contract, unchanged.
    out = args.out
    (out / 'features').mkdir()
    (out / 'model-assets').mkdir()
    old_donors = {d['split']: d for d in manifest['donors']}
    mapping = fresh_role_mapping(role)
    donors = {logical: old_donors[item['original_split']] for logical, item in mapping.items()}
    gate = TestAccessGate(donors['test'])
    cache = source_cache(original, out / 'source-cache', gate, args, source_pins)
    state['stage'] = 'run_only_fixed_label_and_geometry_support_guard'
    early_labels, support_guard = {}, {}
    for logical in ('train', 'calibration'):
        donor = donors[logical]
        early_labels[logical] = original.read_categorical(donor, cache)
        geometry = original.read_categorical(donor, cache, geometry=True) if donor.get('geometry_mask_url') else None
        support_guard[logical] = known_support_upper_bound(original, donor, protocol, early_labels[logical], geometry)
        expected = role['expected_labeled_counts_background_ink'][logical]
        support_guard[logical]['required_final_counts_background_ink'] = expected
        support_guard[logical]['adequate_upper_bound'] = all(
            actual >= required for actual, required in zip(
                support_guard[logical]['upper_bound_counts_background_ink'], expected))
        save(out / 'label_support_guard.json', support_guard)
        if not support_guard[logical]['adequate_upper_bound']:
            raise ValueError('Fixed label/geometry support cannot reproduce prescribed counts; no CT or forwards attempted')
    if gate.test_requests:
        raise ValueError('Held-out source access occurred during the fitting/calibration label guard')
    revision = protocol['model']['revision']
    base_url = f'https://huggingface.co/scrollprize/hecate/resolve/{revision}/'
    state['stage'] = 'verified_model_assets'
    hecate_source = model_asset(out, 'hecate.py', base_url + 'hecate.py',
        protocol['model']['source_sha256'], None, args.hecate_source, state)
    checkpoint = model_asset(out, protocol['model']['checkpoint'], base_url + protocol['model']['checkpoint'],
        protocol['model']['checkpoint_sha256'], protocol['model']['checkpoint_bytes'], args.checkpoint, state)
    state['stage'] = 'numerical_environment_and_frozen_model'
    import numpy as np
    import torch
    import scipy
    import sklearn
    import PIL
    import numcodecs
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    hecate = import_exact(hecate_source, 'fresh_replay_exact_hecate_model')
    model = hecate.load_model(checkpoint, 'cpu')
    if any(layer.training for layer in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Backbone must be wholly frozen and in eval mode')
    normalization = {name: value.clone() for name, value in model.named_buffers()
                     if 'running_' in name or 'num_batches_tracked' in name}
    # Checkpoint is inside out and is therefore already counted by Limits' directory sum.
    limits = original.Limits(protocol, out, 0, start_cpu, forward_count=0)
    class CountedBackbone:
        def __getattr__(self, name):
            return getattr(model, name)

        def features_logits(self, *inputs, **kwargs):
            if limits.forward_count >= MAX_FORWARDS:
                raise RuntimeError('No backbone forward beyond the fixed ninety-six is allowed')
            return model.features_logits(*inputs, **kwargs)

    counted = CountedBackbone()
    provenance = {
        'kind': 'fresh technical reproduction, not the original resumed experiment or a new independent case',
        'fresh_replay_fully_executed_during_packaging_session': False,
        'this_execution_is_a_new_fresh_run': True,
        'freeze_reference': args.freeze_reference, 'replay_runner_sha256': sha(Path(__file__)),
        'original_runner_sha256': sha(args.original_runner), 'original_protocol_sha256': sha(args.original_protocol),
        'role_protocol_sha256': sha(args.role_protocol), 'manifest_sha256': sha(args.manifest),
        'roles': mapping, 'checkpoint_sha256': protocol['model']['checkpoint_sha256'],
        'hecate_source_sha256': protocol['model']['source_sha256'],
        'published_input_integrity_manifest_sha256': SOURCE_RECEIPTS,
        'versions': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                     'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
                     'pillow': PIL.__version__, 'numcodecs': numcodecs.__version__},
        'threads': 4, 'interop_threads': 1,
        'prior_feature_head_threshold_outcome_or_runtime_receipts_reused': False,
        'historical_npz_hash_equality_required': False,
        'test_asset_bytes_accessed_before_readout_freeze': False,
        'label_support_guard_sha256': sha(out / 'label_support_guard.json'),
        'support_guard_scope': 'run-only known-label/geometry upper bound before checkpoint or CT; exact nonzero support checked after extraction',
    }
    original.save_json(out / 'start_receipt.json', provenance)
    extracted, data = {}, {}
    for logical in ('train', 'calibration'):
        state['stage'] = f'fresh_{logical}_features_and_labels'
        extracted[logical] = original.extract_donor(donors[logical], protocol, counted, cache, limits)
        labels = early_labels[logical]
        data[logical] = {order: original.assemble(extracted[logical], labels, order)
                         for order in protocol['depth_orders']}
        same_coordinates(np, data[logical])
        counts = [int((data[logical]['source']['labels'] == value).sum()) for value in (0, 1)]
        if counts != role['expected_labeled_counts_background_ink'][logical]:
            raise ValueError('Frozen fitting/calibration support does not reproduce; no substituted contexts')
    state['stage'] = 'fit_fixed_heads_and_all_calibration_thresholds'
    heads, thresholds, calibration_scores = {}, {}, {}
    for train_order in protocol['depth_orders']:
        heads[train_order] = original.fit_heads(data['train'][train_order], protocol)
        if heads[train_order]['selected_per_class'] != 2712:
            raise ValueError('Unexpected fixed class-balanced training count')
        limits.check()
        thresholds[train_order], calibration_scores[train_order] = {}, {}
        for calibration_order in protocol['depth_orders']:
            calibration = data['calibration'][calibration_order]
            scores = original.score_heads(calibration, heads[train_order])
            calibration_scores[train_order][calibration_order] = scores
            thresholds[train_order][calibration_order] = {
                arm: original.calibration_threshold(calibration['labels'], score, 0.01)
                for arm, score in scores.items()}
    if heads['source']['selected_label_ids_sha256'] != heads['reverse']['selected_label_ids_sha256']:
        raise ValueError('Training depth orders selected different label coordinates')
    if gate.test_requests:
        raise ValueError('Unexpected test source access before the readout freeze')
    freeze = {**provenance, 'heads': original.json_plain(heads), 'thresholds': thresholds,
              'all_four_head_order_calibration_order_threshold_sets_frozen': True,
              'held_out_814_ct_geometry_and_labels_not_accessed': True,
              'forwards_at_freeze': limits.forward_count,
              'cpu_seconds_at_freeze': time.process_time() - start_cpu,
              'wall_seconds_at_freeze': time.monotonic() - start_wall,
              'no_head_retry_or_score_directed_selection': True}
    freeze_path = out / 'readout_freeze.json'
    original.save_json(freeze_path, freeze)
    freeze_sha = sha(freeze_path)
    gate.unlock(freeze_path, freeze_sha)
    original.event('fresh_readouts_frozen_before_any_814_access', sha256=freeze_sha)
    training_diagnostics = {order: original.evaluate(data['train'][order],
        original.score_heads(data['train'][order], heads[order]), thresholds[order][order], extracted['train'])
        for order in protocol['depth_orders']}
    state['stage'] = 'held_out_features_after_freeze'
    extracted['test'] = original.extract_donor(donors['test'], protocol, counted, cache, limits)
    test_labels = original.read_categorical(donors['test'], cache)
    data['test'] = {order: original.assemble(extracted['test'], test_labels, order)
                    for order in protocol['depth_orders']}
    same_coordinates(np, data['test'])
    state['stage'] = 'all_eight_fixed_orientation_combinations'
    combinations = []
    for train_order, calibration_order, test_order in itertools.product(protocol['depth_orders'], repeat=3):
        frozen_threshold = thresholds[train_order][calibration_order]
        entry = {'training_scroll': 'PHerc0841', 'calibration_scroll': 'PHerc0009B',
                 'test_scroll': 'PHerc0814', 'training_order': train_order,
                 'calibration_order': calibration_order, 'test_order': test_order,
                 'thresholds_frozen_before_test': frozen_threshold,
                 'arm_validity': {arm: heads[train_order][arm]['valid'] for arm in ('baseline', 'depth', 'features')},
                 'baseline_note': 'Training-order duplicates are not independent tests.',
                 'calibration': original.evaluate(data['calibration'][calibration_order],
                     calibration_scores[train_order][calibration_order], frozen_threshold, extracted['calibration']),
                 'test': original.evaluate(data['test'][test_order],
                     original.score_heads(data['test'][test_order], heads[train_order]), frozen_threshold, extracted['test'])}
        entry['interpretation'] = ('primary source/source/source' if (train_order, calibration_order, test_order) == ('source',)*3
            else 'all-reversed sensitivity' if (train_order, calibration_order, test_order) == ('reverse',)*3
            else 'symmetric sensitivity; no winner selected')
        combinations.append(entry)
    unchanged = all(torch.equal(value, dict(model.named_buffers())[name]) for name, value in normalization.items())
    if not unchanged or sha(freeze_path) != freeze_sha or len(combinations) != 8:
        raise ValueError('Frozen state changed or orientation table is incomplete')
    if limits.forward_count != 2 * sum(len(item['retained']) for item in extracted.values()):
        raise ValueError('Forward ledger differs from retained contexts and two depth orders')
    limits.check()
    record = {'experiment': 'fresh technical replay of the published role experiment',
        'provenance': provenance, 'readout_freeze_sha256': freeze_sha, 'roles': mapping,
        'forwards': limits.forward_count, 'maximum_forwards': MAX_FORWARDS,
        'source_bytes_charged_including_reuse': cache.bytes, 'source_network_bytes': cache.network_bytes,
        'source_verified_local_reuse_bytes': cache.reused_bytes,
        'checkpoint_bytes': checkpoint.stat().st_size, 'run_directory_bytes': disk_bytes(out),
        'cpu_seconds': time.process_time() - start_cpu, 'wall_seconds': time.monotonic() - start_wall,
        'running_normalization_buffers_unchanged': True,
        'source_assets': list(cache.records.values()),
        'fresh_feature_sha256': {str(path.relative_to(out)): sha(path) for path in sorted((out/'features').glob('*.npz'))},
        'exclusions': {logical: item['excluded'] for logical, item in extracted.items()},
        'training_diagnostics': training_diagnostics,
        'training_diagnostics_interpretation': 'In-sample 841 diagnostics, not independent validation.',
        'primary': next(x for x in combinations if x['interpretation'] == 'primary source/source/source'),
        'all_reversed_sensitivity': next(x for x in combinations if x['interpretation'] == 'all-reversed sensitivity'),
        'all_eight_combinations': combinations, 'limitations': role['limitations'],
        'reproduction_note': 'No equality to historical runtime or feature-file hashes is assumed. The same physical parents are not independent replications.',
        'no_unread_data_or_prize_claim': True}
    original.save_json(out / 'result.json', original.json_plain(record))
    limits.check()
    state['stage'] = 'complete'
    original.event('fresh_replay_complete', forwards=limits.forward_count, cpu_seconds=record['cpu_seconds'])


def synthetic_checks(args, role, protocol, manifest):
    original = import_exact(args.original_runner, 'fresh_replay_generated_checks')
    original.preflight(protocol, manifest)
    original.synthetic_check()
    import numpy as np
    generated_donor = {'label_pitch_um': 9.36, 'label_shape_yx': [100, 100],
        'geometry_mask_pitch_um': 9.36, 'geometry_mask_shape_yx': [100, 100],
        'label_values': {'unknown': 0, 'background': 1, 'ink': 2},
        'contexts': [{'id': 'generated-a', 'origin_target_yx': [0, 0]},
                     {'id': 'generated-duplicate', 'origin_target_yx': [0, 0]}]}
    labels = np.ones((100, 100), dtype=np.uint8)
    geometry = np.ones_like(labels)
    yy, xx = original.label_indices(generated_donor, generated_donor['contexts'][0])
    labels[np.ix_(yy[:2], xx)] = 2
    labels[np.ix_(yy[2:3], xx)] = 0
    count = known_support_upper_bound(original, generated_donor, protocol, labels, geometry)
    if count['upper_bound_counts_background_ink'] != [928, 64]:
        raise AssertionError('Generated label support did not deduplicate exact categorical coordinates')
    geometry[:] = 0
    count = known_support_upper_bound(original, generated_donor, protocol, labels, geometry)
    if count['upper_bound_counts_background_ink'] != [0, 0] or len(count['geometry_excluded_context_ids']) != 2:
        raise AssertionError('Generated geometry support did not preserve whole-footprint exclusions')
    donor = {'source_zarr_url': 'https://example.invalid/test/0/',
             'label_url': 'https://example.invalid/test-label.npy',
             'geometry_mask_url': 'https://example.invalid/test-geometry.png'}
    gate = TestAccessGate(donor)
    for url in [donor['label_url'], donor['geometry_mask_url'], donor['source_zarr_url'] + '.zarray']:
        try:
            gate.allow(url)
        except ValueError:
            pass
        else:
            raise AssertionError('Generated held-out request escaped the gate')
    with tempfile.TemporaryDirectory(prefix='hecate-replay-generated-') as temp:
        freeze = Path(temp) / 'freeze.json'
        save(freeze, {'all_four_head_order_calibration_order_threshold_sets_frozen': True,
                      'heads': 'generated fixture', 'thresholds': 'generated fixture'})
        gate.unlock(freeze, sha(freeze))
        gate.allow(donor['label_url'])
        freeze.write_text('{}')
        try:
            gate.allow(donor['label_url'])
        except ValueError:
            pass
        else:
            raise AssertionError('Modified generated freeze did not relock test access')
    print(json.dumps({'generated_gate_checks': 'passed', 'generated_label_support_checks': 'passed', 'model_fits': 0,
                      'model_forwards': 0, 'source_acquisitions': 0,
                      'fresh_full_replay_executed': False}))


def main():
    start_wall, start_cpu = time.monotonic(), time.process_time()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('check', 'synthetic-check', 'run'))
    parser.add_argument('--original-runner', type=Path, default=HERE/'hecate_probe.py')
    parser.add_argument('--original-protocol', type=Path, default=HERE/'hecate_probe_protocol.json')
    parser.add_argument('--role-protocol', type=Path, default=HERE/'hecate_role_protocol.json')
    parser.add_argument('--manifest', type=Path, default=HERE/'hecate_donor_manifest.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--freeze-reference')
    parser.add_argument('--hecate-source', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reuse-cache', type=Path)
    parser.add_argument('--reuse-receipt', type=Path)
    args = parser.parse_args()
    role, protocol, manifest, source_pins = contract(args)
    if args.action == 'check':
        print(json.dumps({'status': 'pinned_json_hash_and_role_contract_passed',
            'roles': fresh_role_mapping(role), 'maximum_forwards': MAX_FORWARDS,
            'maximum_cpu_seconds': MAX_CPU_SECONDS, 'maximum_source_bytes': MAX_SOURCE_BYTES,
            'maximum_run_bytes_including_checkpoint': MAX_RUN_BYTES,
            'prior_feature_or_runtime_hashes_required': False, 'model_or_data_access': False,
            'published_source_integrity_manifests': SOURCE_RECEIPTS,
            'source_assets_pinned': len(source_pins),
            'fresh_full_replay_executed_in_packaging_session': False}, indent=2))
        return
    if args.action == 'synthetic-check':
        synthetic_checks(args, role, protocol, manifest)
        return
    if args.out is None or not args.freeze_reference or not re.fullmatch(r'[0-9a-f]{40}', args.freeze_reference):
        parser.error('run requires --out and a complete lowercase forty-character published --freeze-reference')
    args.out = args.out.resolve()
    repository = HERE.parents[1]
    if args.out.exists() or args.out == repository or repository in args.out.parents:
        parser.error('--out must be a new external directory; no previous output is reused or overwritten')
    if args.reuse_cache is not None:
        args.reuse_cache = args.reuse_cache.resolve()
        if args.reuse_cache == args.out or args.reuse_cache in args.out.parents or args.out in args.reuse_cache.parents:
            parser.error('Fresh output and optional reuse cache must be disjoint')
    reuse_index(args)  # Metadata only; never opens any cached label/CT bytes here.
    args.out.mkdir(parents=True, exist_ok=False)
    state = {'stage': 'starting', 'freeze_reference': args.freeze_reference}
    save(args.out / 'replay_invocation.json', {'freeze_reference': args.freeze_reference,
        'runner_sha256': sha(Path(__file__)), 'role_protocol_sha256': ROLE_SHA256,
        'fresh_replay_packaging_was_not_fully_executed': True})

    def cpu_limit(signum, frame):
        raise RuntimeError('Absolute fresh-run CPU cap reached')

    signal.signal(signal.SIGXCPU, cpu_limit)
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_CPU)
    finite = [x for x in (old_soft, old_hard, MAX_CPU_SECONDS) if x != resource.RLIM_INFINITY]
    hard = min(finite)
    resource.setrlimit(resource.RLIMIT_CPU, (max(1, hard-5), hard))
    try:
        execute(args, role, protocol, manifest, source_pins, state, start_wall, start_cpu)
    except BaseException as error:
        save(args.out / 'failure.json', {'status': 'failed_without_retry_or_substitution',
            'error_type': type(error).__name__, 'stage': state['stage'],
            'freeze_reference': args.freeze_reference, 'runner_sha256': sha(Path(__file__)),
            'role_protocol_sha256': ROLE_SHA256,
            'cpu_seconds': time.process_time()-start_cpu, 'wall_seconds': time.monotonic()-start_wall,
            'no_redirect_url_or_private_path_in_failure_receipt': True})
        raise SystemExit('Fresh replay stopped; inspect the sanitized failure receipt.') from None


if __name__ == '__main__':
    main()
