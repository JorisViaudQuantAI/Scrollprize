#!/usr/bin/env python3
"""Run one frozen, bounded public-control replication, with stage receipts.

Outputs and acquisition cache must be new directories outside Git. The source
reference is a model prediction; no part of this run establishes a reading.
"""
from __future__ import annotations

import os

# Set before numerical imports; no worker pools or GPU are used.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import scipy
from scipy.stats import rankdata

import features
import selector
import source

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('t5_replication_statistics', ROOT / 'statistics.py')
statistics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(statistics)


class StopRun(RuntimeError):
    """Predeclared terminal failure; the message contains no private paths."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False) + '\n'
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(encoded)
    temporary.chmod(0o600)
    temporary.replace(path)


def artifact(path):
    return {'name': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path)}


def prepare_directory(path):
    path = Path(path).expanduser().resolve()
    if any((p / '.git').exists() for p in (path, *path.parents)):
        raise StopRun('Output and cache must be outside Git working trees')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise StopRun('Initial execution requires new empty output and cache directories')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def verify_freeze(commit):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise StopRun('A full published freeze commit SHA is required')
    protocol_path = ROOT / 'protocol.json'
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    repo = Path(subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--show-toplevel'], text=True).strip())
    relative = protocol_path.relative_to(repo).as_posix()
    frozen = subprocess.check_output(['git', '-C', str(repo), 'show', commit + ':' + relative])
    if frozen != raw:
        raise StopRun('Local protocol differs from the published freeze commit')
    for name, expected in protocol['frozen_files_sha256'].items():
        path = ROOT / name
        if digest(path) != expected:
            raise StopRun('Frozen local file hash differs')
        blob = subprocess.check_output(['git', '-C', str(repo), 'show', commit + ':' + path.relative_to(repo).as_posix()])
        if hashlib.sha256(blob).hexdigest() != expected:
            raise StopRun('Frozen Git file hash differs')
    return protocol, hashlib.sha256(raw).hexdigest()


def guarded(mask, width):
    result = mask.copy()
    if width:
        result[:width] = result[-width:] = False
        result[:, :width] = result[:, -width:] = False
    return result


def observed_auc(maps, valid, reference):
    positive = reference >= 179
    negative = (reference > 0) & (reference <= 64)
    include = valid & (positive | negative)
    labels = positive[include]
    n1, n0 = int(labels.sum()), int((~labels).sum())
    auc = {}
    for name, values in maps.items():
        if not n1 or not n0:
            auc[name] = None
            continue
        ranks = rankdata(features.POLARITIES[name] * values[include], method='average')
        auc[name] = float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return {'ink_count': n1, 'blank_count': n0, 'auc': auc}


def execute(args):
    protocol, protocol_sha = verify_freeze(args.freeze_commit)
    output = prepare_directory(args.output)
    cache = prepare_directory(args.cache)
    if output == cache or output in cache.parents or cache in output.parents:
        raise StopRun('Output and cache directories must be separate')
    started = time.monotonic()
    budget = source.Budget(protocol_sha)
    receipt = {
        'schema': 't5-spatial-replication.experiment.v1',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': protocol_sha, 'freeze_commit': args.freeze_commit,
        'endpoint': protocol['endpoint'], 'scope': protocol['scope'],
        'status': 'running', 'stage': 'initialization', 'cases': [], 'artifacts': [],
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'platform': platform.system(),
                        'numeric_threads': 1},
        'gpu_forwards': 0, 'training_updates': 0, 'unread_regions': 0,
        'any_feature_passes_replication_gate': False,
    }

    def checkpoint(stage):
        receipt['stage'] = stage
        receipt['elapsed_seconds'] = round(time.monotonic() - started, 3)
        receipt['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        receipt['acquisition'] = budget.summary()
        write_json(output / 'result.json', receipt)
        print(json.dumps({'stage': stage, 'elapsed_seconds': receipt['elapsed_seconds'],
                          'response_body_bytes': budget.body_bytes, 'requests': budget.requests}), flush=True)
        if receipt['peak_rss_bytes'] > protocol['limits']['peak_rss_bytes']:
            raise StopRun('Stage-boundary peak RSS exceeds the frozen limit')

    def timeout_handler(signum, frame):
        raise StopRun('Frozen wall-time limit reached')

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(protocol['limits']['wall_seconds'])
    try:
        checkpoint('fetch_reference')
        reference = source.fetch_reference(cache, budget)
        checkpoint('select_regions')
        selection = selector.select(reference, source.KNOWN_ROI, **protocol['selection'])
        write_json(output / 'selection.json', selection)
        receipt['artifacts'].append(artifact(output / 'selection.json'))
        receipt['selection_summary'] = {k: v for k, v in selection.items() if k != 'eligible_candidates'}
        if selection['status'] != 'complete':
            raise StopRun('Fewer than three label-eligible separated regions; no fallback')
        bounds = [list(source.KNOWN_ROI)] + [item['bounds_y0_y1_x0_x1'] for item in selection['selected']]
        budget.authorize_regions(bounds)
        crops = [reference[y0:y1, x0:x1].copy() for y0, y1, x0, x1 in bounds]
        del reference
        gc.collect()
        checkpoint('selection_frozen_before_CT')
        cases = []
        for index, (region, reference_crop) in enumerate(zip(bounds, crops)):
            name = 'development_replay' if index == 0 else f'heldout_{index}'
            checkpoint('fetch_' + name)
            stack = source.fetch_region(cache, budget, region)
            stack_sha = hashlib.sha256(stack.tobytes()).hexdigest()
            checkpoint('compute_' + name)
            maps, common_valid, diagnostics = features.compute(stack)
            del stack
            guard = protocol['replay']['border_guard_px'] if index == 0 else protocol['selection']['border_guard_px']
            valid = guarded(common_valid, guard)
            observed = observed_auc(maps, valid, reference_crop)
            potential = int(np.prod([s - 2 * guard for s in common_valid.shape]))
            support = {
                'guard_px': guard, 'potential_guarded_pixels': potential,
                'guarded_common_valid_pixels': int(valid.sum()),
                'guarded_common_valid_fraction': float(valid.sum() / potential),
                'whole_common_valid_fraction_of_sheet': diagnostics['support']['common_valid_fraction_of_sheet'],
                'class_counts_at_least_10000_each': min(observed['ink_count'], observed['blank_count']) >= 10000,
            }
            support['pass'] = bool(support['guarded_common_valid_fraction'] >= .90
                                   and support['whole_common_valid_fraction_of_sheet'] >= .90
                                   and support['class_counts_at_least_10000_each'])
            path = output / (name + '.npz')
            np.savez_compressed(path, **maps, reference=reference_crop,
                                common_valid=common_valid, eval_valid=valid)
            path.chmod(0o600)
            receipt['artifacts'].append(artifact(path))
            case_report = {'name': name, 'bounds_y0_y1_x0_x1': region,
                           'source_stack_shape': [109, 512, 640], 'source_stack_dtype': 'uint8',
                           'source_stack_C_order_bytes_sha256': stack_sha,
                           'diagnostics': diagnostics, 'support': support, 'observed': observed}
            receipt['cases'].append(case_report)
            if index == 0:
                expected = protocol['replay']['expected_auc']
                errors = {f: abs(observed['auc'][f] - expected[f]) if observed['auc'][f] is not None else None for f in expected}
                replay = {'absolute_auc_errors': errors,
                          'expected_ink_blank_counts': protocol['replay']['expected_class_counts'],
                          'observed_ink_blank_counts': [observed['ink_count'], observed['blank_count']],
                          'all_columns_supported': bool(common_valid.all())}
                replay['pass'] = bool(replay['all_columns_supported']
                                      and replay['observed_ink_blank_counts'] == replay['expected_ink_blank_counts']
                                      and all(v is not None and v <= protocol['replay']['auc_tolerance'] for v in errors.values()))
                receipt['development_replay'] = replay
                checkpoint('development_replay_complete')
                if not replay['pass']:
                    raise StopRun('Original published development metrics did not reproduce; no heldout acquisition')
            else:
                cases.append({'name': name, 'features': maps, 'reference': reference_crop, 'valid': valid})
                checkpoint(name + '_complete')
                if not support['pass']:
                    raise StopRun('A fixed heldout field failed support or class counts; no replacement')
            gc.collect()
        checkpoint('evaluate_fixed_features')
        evaluation = statistics.evaluate(cases, **protocol['statistics'])
        write_json(output / 'statistics.json', evaluation)
        receipt['artifacts'].append(artifact(output / 'statistics.json'))
        receipt['statistics_summary'] = {k: evaluation[k] for k in ('status', 'observed', 'gates', 'failures')}
        receipt['statistics_summary']['spatial_rank'] = evaluation['null']['spatial_rank']
        receipt['statistics_summary']['paired_block_interval'] = evaluation['bootstrap']['percentile_interval']
        receipt['any_feature_passes_replication_gate'] = bool(any(g['pass'] for g in evaluation['gates'].values()))
        receipt['status'] = 'complete' if evaluation['status'] == 'assessable' else 'unassessable'
        checkpoint('complete')
    except BaseException as exc:
        receipt['status'] = 'stopped'
        receipt['any_feature_passes_replication_gate'] = False
        receipt['failure'] = {'type': type(exc).__name__, 'at_stage': receipt['stage'],
                              'reason': str(exc) if isinstance(exc, StopRun) else 'Execution or source contract failed; no retry or fallback'}
        receipt['elapsed_seconds'] = round(time.monotonic() - started, 3)
        receipt['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        receipt['acquisition'] = budget.summary()
        write_json(output / 'result.json', receipt)
        print(json.dumps({'status': 'stopped', 'failure': receipt['failure']}), flush=True)
        return 2
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-commit', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    return execute(args)


if __name__ == '__main__':
    sys.exit(main())
