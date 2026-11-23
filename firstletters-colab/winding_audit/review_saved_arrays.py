"""Saved-array review only: no traversal import, network, model or new sweep."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import subprocess
import time

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    repo, source, run = args.repo, args.source, args.run
    lane = repo / 'firstletters-colab/winding_audit'
    result = json.loads((run / 'result.json').read_text())
    protocol = json.loads((lane / 'protocol.json').read_text())
    acquisition = json.loads((source / 'acquisition.json').read_text())
    counts, failures = {}, []

    def check(group, name, actual, expected=True):
        counts[group] = counts.get(group, 0) + 1
        if actual != expected:
            failures.append({'group': group, 'name': name, 'actual': actual, 'expected': expected})

    check('provenance', 'run_complete', result['status'], 'complete')
    check('provenance', 'acquisition_complete', acquisition['status'], 'complete')
    check('provenance', 'protocol_hash', sha(lane / 'protocol.json'), result['protocol_sha256'])
    check('provenance', 'acquisition_protocol_hash', sha(source / 'protocol.json'), result['protocol_sha256'])
    check('provenance', 'acquisition_receipt_protocol_hash', acquisition['protocol_sha256'], result['protocol_sha256'])
    check('provenance', 'run_source_receipt_hash', sha(source / 'acquisition.json'), result['source_receipt_sha256'])
    check('provenance', 'runner_hash', sha(lane / 'run.py'), result['runner_sha256'])
    files = {'sense_chain.py': repo / 'firstletters-colab/methods_ledger/code/sense_chain.py',
             'legacy_adapter.py': lane / 'legacy_adapter.py', 'legacy_snapshot.py': lane / 'legacy_snapshot.py'}
    for name, path in files.items():
        check('provenance', name + '_hash', sha(path), result['implementation_sha256'][name])
    frozen_commit = '4923329'
    frozen_paths = [*files.values(), lane / 'run.py', lane / 'protocol.json', lane / 'acquire.py']
    for path in frozen_paths:
        relative = str(path.relative_to(repo))
        pinned = subprocess.check_output(['git', 'show', frozen_commit + ':' + relative], cwd=repo)
        check('provenance', relative + '_frozen_bytes', path.read_bytes() == pinned)
    old = subprocess.check_output(['git', 'show', protocol['base_repository_commit'] + ':firstletters-colab/methods_ledger/code/sense_chain.py'], cwd=repo)
    check('provenance', 'legacy_snapshot_is_exact_base', old == files['legacy_snapshot.py'].read_bytes())
    check('provenance', 'producer_signature_pair', acquisition['store_metadata']['source_db_signature'], acquisition['crossing_metadata']['db_signature'])

    source_bytes = 0
    for name, record in acquisition['files'].items():
        path = source / name
        check('source_artifacts', name + '_size', path.stat().st_size, record['bytes'])
        check('source_artifacts', name + '_sha256', sha(path), record['sha256'])
        source_bytes += path.stat().st_size
    rr = json.loads((source / 'range_receipt.json').read_text())
    etags, lengths = {}, {}
    charged = 0
    for idx, record in enumerate(rr['ranges']):
        label = f'range_{idx}'
        check('source_ranges', label + '_complete', record['status'], 'complete')
        if record['status'] != 'complete':
            continue
        path = source / 'range-cache' / record['cache_file']
        size = path.stat().st_size
        lo, hi = record['range_halfopen']
        check('source_ranges', label + '_body_size', size, record['bytes'])
        check('source_ranges', label + '_range_size', size, hi - lo)
        check('source_ranges', label + '_body_hash', sha(path), record['sha256'])
        etags.setdefault(record['url'], set()).add(record['etag'])
        lengths.setdefault(record['url'], set()).add(record['object_bytes'])
        charged += record['bytes']
    check('source_ranges', 'single_etag_per_object', all(len(v) == 1 for v in etags.values()))
    check('source_ranges', 'single_object_length_per_object', all(len(v) == 1 for v in lengths.values()))
    check('source_ranges', 'request_count', len(rr['ranges']), rr['request_count'])
    check('source_ranges', 'byte_sum', charged, rr['source_body_bytes'])
    check('source_ranges', 'acquisition_request_count', len(rr['ranges']), acquisition['request_count'])
    check('source_ranges', 'acquisition_byte_count', charged, acquisition['source_body_bytes'])
    check('source_ranges', 'frozen_byte_budget', charged <= protocol['budgets']['maximum_source_body_bytes'])
    check('source_ranges', 'frozen_request_budget', len(rr['ranges']) <= protocol['budgets']['maximum_requests'])

    hids = np.load(source / 'transverse_ids.npy', mmap_mode='r', allow_pickle=False)
    lens = np.load(source / 'lengths.npy', mmap_mode='r', allow_pickle=False)
    coords = np.load(source / 'coordinates.npy', mmap_mode='r', allow_pickle=False)
    nodes = np.load(source / 'node_ids.npy', mmap_mode='r', allow_pickle=False)
    check('source_structure', 'selected_transverse_count', len(hids), acquisition['transverse_tracks'])
    check('source_structure', 'selected_coordinate_count', len(coords), acquisition['transverse_points'])
    check('source_structure', 'retained_node_count', len(nodes), acquisition['source_id_alignment_verified'])
    check('source_structure', 'retained_nodes_decompose', len(nodes), acquisition['transverse_tracks'] + acquisition['linked_longitudinal_tracks'])
    eligible = hids[lens[hids] >= protocol['selection']['minimum_start_points']]
    starts = np.random.default_rng(protocol['selection']['start_seed']).choice(eligible, min(protocol['selection']['nstart'], len(eligible)), replace=False)
    ranks = {int(t): i for i, t in enumerate(starts)}
    check('source_structure', 'eligible_count', len(eligible), result['geometry_diagnostics']['start_eligible_tracks'])
    check('source_structure', 'diagnostic_track_count', len(hids), result['geometry_diagnostics']['transverse_tracks'])
    check('source_structure', 'diagnostic_point_count', len(coords), result['geometry_diagnostics']['point_count'])

    recomputed, loaded, normalized = {}, {}, {}
    legacy_missing = {}
    total_output_bytes = 0
    for algorithm, conditions in result['analysis'].items():
        recomputed[algorithm] = {}
        for condition, summary in conditions.items():
            prefix = algorithm + '_' + condition
            row_path, id_path = run / (prefix + '_rows.npy'), run / (prefix + '_start_ids.npy')
            a = np.load(row_path, allow_pickle=False)
            ids = np.load(id_path, allow_pickle=False)
            loaded[algorithm, condition] = a, ids
            for path, key in ((row_path, 'rows_sha256'), (id_path, 'completed_start_ids_sha256')):
                check('output_artifacts', path.name + '_hash', sha(path), summary[key])
                total_output_bytes += path.stat().st_size
            check('output_artifacts', prefix + '_shape', list(a.shape), [len(ids), 3])
            check('output_artifacts', prefix + '_finite', bool(np.isfinite(a).all()))
            check('output_artifacts', prefix + '_unique_ids', len(set(map(int, ids))), len(ids))
            check('output_artifacts', prefix + '_ids_selected', all(int(t) in ranks for t in ids))
            rank_sequence = [ranks[int(t)] for t in ids if int(t) in ranks]
            check('output_artifacts', prefix + '_start_order', rank_sequence, sorted(rank_sequence))
            check('output_artifacts', prefix + '_hops', all(float(v).is_integer() and 0 <= v < protocol['traversal']['max_hops'] for v in a[:, 2]))
            dr, dz, hops = ([float(v) for v in a[:, j]] for j in range(3))
            median = lambda vals: statistics.median(vals) if vals else None
            stats = {'chains_full_turn': len(a), 'dr_positive': sum(v > 0 for v in dr),
                     'dr_negative': sum(v < 0 for v in dr), 'median_dr': median(dr),
                     'median_abs_dr': median([abs(v) for v in dr]),
                     'n_abs_dr_5_60_pos': sum(5 < v < 60 for v in dr),
                     'n_abs_dr_5_60_neg': sum(-60 < v < -5 for v in dr),
                     'median_abs_dz': median([abs(v) for v in dz]), 'median_hops': median(hops)}
            for key, value in stats.items():
                check('completed_statistics', prefix + '_' + key, value, summary[key])
            same_dr = [x for x, z in zip(dr, dz) if z == 0]
            same_stats = {'chains_full_turn': len(same_dr), 'dr_positive': sum(v > 0 for v in same_dr),
                          'dr_negative': sum(v < 0 for v in same_dr),
                          'n_abs_dr_5_60_pos': sum(5 < v < 60 for v in same_dr),
                          'n_abs_dr_5_60_neg': sum(-60 < v < -5 for v in same_dr), 'median_dr': median(same_dr)}
            stats['exact_same_z'] = same_stats
            stats['nonzero_dz_closures'] = len(a) - len(same_dr)
            check('completed_statistics', prefix + '_started', summary['chains_started'], len(starts))
            if algorithm == 'corrected_v2':
                check('completed_statistics', prefix + '_same_z_summary', same_stats, summary['exact_same_z'])
                check('completed_statistics', prefix + '_nonzero_z_summary', stats['nonzero_dz_closures'], summary['nonzero_dz_closures'])
                check('accounting', prefix + '_exhaustive', summary['chains_full_turn'] + summary['broken'], summary['chains_started'])
                check('accounting', prefix + '_broken_reasons', sum(summary['broken_reasons'].values()), summary['broken'])
                check('interpretation', prefix + '_no_sense_verdict', summary['winding_sense_verdict'], None)
            else:
                missing = summary['chains_started'] - summary['chains_full_turn'] - summary['broken']
                check('accounting', prefix + '_missing_nonnegative', missing >= 0)
                legacy_missing[condition] = missing
            recomputed[algorithm][condition] = stats
            normalized[algorithm, condition] = {k: v for k, v in summary.items() if k not in ('seconds', 'rows_sha256', 'completed_start_ids_sha256')}
            normalized[algorithm, condition]['completed_start_ids'] = list(map(int, ids))

    invariance = {}
    for algorithm in result['analysis']:
        invariance[algorithm] = {}
        base, base_ids = loaded[algorithm, 'original']
        for condition in protocol['conditions'][1:-1]:
            a, ids = loaded[algorithm, condition]
            observed = {
                'complete_summary_equal': normalized[algorithm, condition] == normalized[algorithm, 'original'],
                'ordered_rows_bitwise_equal': a.dtype == base.dtype and a.shape == base.shape and a.tobytes() == base.tobytes(),
                'completed_start_ids_equal': ids.dtype == base_ids.dtype and ids.shape == base_ids.shape and ids.tobytes() == base_ids.tobytes(),
            }
            check('invariance', algorithm + '_' + condition, observed, result['representation_invariance'][algorithm][condition])
            invariance[algorithm][condition] = observed
    fixed_pass = all(all(v.values()) for v in invariance['corrected_v2'].values())
    legacy_fail = any(not all(v.values()) for v in invariance['legacy'].values())
    check('invariance', 'fixed_storage_gate', fixed_pass, result['fixed_storage_invariance_pass'])
    check('invariance', 'legacy_storage_defect', legacy_fail, result['legacy_storage_defect_reproduced'])
    check('interpretation', 'no_physical_verdict', result['physical_winding_sense_verdict'], None)

    review = {
        'schema': 'q12_saved_array_numerical_review_v1', 'status': 'PASS' if not failures else 'FAIL',
        'review_method': 'Separate saved-array arithmetic using statistics.median and explicit counts; no experiment module imported or executed.',
        'reviewer_scope_disclosure': 'Reviewer authored corrected sense_chain.py; this is a separate numerical recomputation, not an independent review of its implementation.',
        'analysis_forwards': 0, 'model_forwards': 0, 'new_source_requests': 0,
        'result_sha256': sha(run / 'result.json'), 'protocol_sha256': result['protocol_sha256'],
        'review_script_sha256': sha(__file__), 'implementation_sha256_verified': result['implementation_sha256'],
        'frozen_commit_checked': subprocess.check_output(['git', 'rev-parse', frozen_commit], cwd=repo, text=True).strip(),
        'check_counts': counts, 'check_count_total': sum(counts.values()), 'failures': failures,
        'source_verification': {'decoded_files': len(acquisition['files']), 'decoded_bytes_verified': source_bytes,
                                'cached_range_bodies': len(rr['ranges']), 'cached_range_body_bytes_verified': charged,
                                'unique_objects_with_consistent_etag_and_size': len(etags)},
        'output_artifacts_verified': 24, 'output_artifact_bytes_verified': total_output_bytes,
        'recomputed_completed_chain_statistics': recomputed,
        'representation_invariance': invariance,
        'fixed_storage_invariance_pass': fixed_pass, 'legacy_storage_defect_reproduced': legacy_fail,
        'legacy_unaccounted_terminal_counts': legacy_missing,
        'interpretation_notes': [
            'The corrected storage-only conditions have identical completed arrays, ordered IDs, normalized summaries and reported per-start terminal digest. This demonstrates the tested representation invariance, not anatomical correctness.',
            'Failure per-start records were not saved. Their digest equality is verified as recorded, but the digest and failure-angle medians cannot be independently reconstructed from the saved completion arrays.',
            'Legacy completed plus broken totals omit hop-limit terminations; the listed count gaps are retained as baseline behavior, not repaired in the comparison.',
            'Four corrected original completions include only two in the declared pitch band, and only one exact-same-z completion. Those are dependent graph outcomes around an unvalidated geometric origin, not independent anatomical votes.',
            'The corrected mirrored run has one negative drift of about -1408.94 voxels, outside the pitch band. Greedy mirrored paths are not paired physical paths; no physical-sense claim follows.',
            'The corrected version also changes deterministic tie handling, forward-index filtering, canonical unwrapping and failure accounting. A real-data aggregate difference cannot be assigned only to the double-sign fix.',
            'All recorded decoded-source and range-cache file hashes were checked. This does not reconstruct every packed-source extraction, independently revalidate all source-ID alignment, verify whole-ZIP CRC, or re-establish scientific source geometry.',
            'No ink detector, manuscript reading, new surface sweep, hidden-field selection or prize qualification was evaluated.'
        ],
        'seconds': time.monotonic() - started,
    }
    args.output.write_text(json.dumps(review, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: review[k] for k in ('status', 'check_count_total', 'failures', 'fixed_storage_invariance_pass', 'legacy_storage_defect_reproduced', 'legacy_unaccounted_terminal_counts', 'source_verification', 'seconds')}, indent=2))


if __name__ == '__main__':
    main()
