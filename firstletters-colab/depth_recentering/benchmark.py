"""Run the frozen, CPU-only public PHerc0841 displacement diagnostic.

This evaluates recovery of artificial offsets relative to a stored surface band,
not the physical recto location. It contains no ink model or text-likeness score.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parent


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    scale = values.std()
    return (values - values.mean()) / scale if scale > 1e-12 else values * 0


def correlation(a, b):
    if min(np.std(a), np.std(b)) < 1e-12:
        return -1.0
    return float(np.mean(normalize(a) * normalize(b)))


def trimmed_profile(cell):
    ordered = np.sort(cell.reshape(cell.shape[0], -1), axis=1)
    cut = int(ordered.shape[1] * 0.1)
    return ordered[:, cut:-cut].mean(axis=1, dtype=np.float64)


def prominent_peaks(profile):
    smoothed = np.convolve(np.pad(profile, (1, 1), mode='edge'),
                           [0.25, 0.5, 0.25], mode='valid')
    span = float(np.ptp(smoothed))
    peaks = []
    if span <= 0:
        return peaks
    for i in range(1, len(smoothed) - 1):
        if not (smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]):
            continue
        left, right = i, i
        while left > 0 and smoothed[left - 1] <= smoothed[i]:
            left -= 1
        while right < len(smoothed) - 1 and smoothed[right + 1] <= smoothed[i]:
            right += 1
        prominence = smoothed[i] - max(smoothed[left:i + 1].min(),
                                       smoothed[i:right + 1].min())
        if prominence >= 0.25 * span:
            peaks.append({'index': i, 'prominence': float(prominence)})
    return peaks


def bounded_match(profile, template, eligible=True):
    offsets = np.arange(-4, 5)
    common = np.arange(4, 16)
    scores = np.array([correlation(template[common], profile[common + c])
                       for c in offsets])
    best = int(np.argmax(scores))
    correction = int(offsets[best])
    rivals = scores[np.abs(offsets - correction) >= 2]
    margin = float(scores[best] - rivals.max())
    peaks = prominent_peaks(profile)
    competing_peaks = any(abs(a['index'] - b['index']) >= 4
                          for i, a in enumerate(peaks) for b in peaks[i + 1:])
    reasons = []
    if not eligible:
        reasons.append('anatomically_ineligible')
    if np.ptp(profile) < 8.0:
        reasons.append('low_profile_contrast')
    if scores[best] < 0.8:
        reasons.append('low_template_correlation')
    if margin < 0.1:
        reasons.append('competing_depth_alignments')
    if abs(correction) == 4:
        reasons.append('search_boundary')
    if competing_peaks:
        reasons.append('possible_neighboring_sheet_peaks')
    return {
        'accepted': not reasons,
        'correction_planes': correction,
        'correlation': float(scores[best]), 'rival_margin': margin,
        'profile_range': float(np.ptp(profile)),
        'prominent_peaks': peaks,
        'abstention_reasons': reasons,
        'candidate_correlations': [float(v) for v in scores],
    }


def self_test():
    # Independent known shifts catch direction/sign and crop indexing errors.
    axis = np.arange(28)
    native = 20 + 60 * np.exp(-((axis - 13.5) / 1.8) ** 2)
    template = native[4:24]
    for injected in range(-4, 5):
        result = bounded_match(native[4 + injected:24 + injected], template)
        assert result['correction_planes'] == -injected, (injected, result)
        if abs(injected) == 4:
            assert not result['accepted'] and 'search_boundary' in result['abstention_reasons']
    axis = np.arange(20)
    two_peaks = 20 + 70 * np.exp(-((axis - 5) / 1.3) ** 2)
    two_peaks += 60 * np.exp(-((axis - 13) / 1.3) ** 2)
    result = bounded_match(two_peaks, two_peaks)
    assert not result['accepted']
    assert 'possible_neighboring_sheet_peaks' in result['abstention_reasons']
    assert not bounded_match(np.ones(20), template)['accepted']
    return {'known_shift_sign': 'passed', 'boundary_rejection': 'passed',
            'two_peak_ambiguity_rejection': 'passed', 'flat_profile_rejection': 'passed'}


def summarize(records, method):
    accepted = [r for r in records if r[method]['accepted']]
    errors = [abs(r[method]['correction_planes'] - r['truth_correction_planes'])
              for r in accepted]
    unsafe = sum(e > 1 for e in errors)
    reasons = Counter(reason for r in records for reason in
                      r[method].get('abstention_reasons', []))
    return {
        'cases': len(records), 'accepted': len(accepted),
        'coverage': len(accepted) / len(records) if records else None,
        'within_one_plane': sum(e <= 1 for e in errors),
        'within_one_plane_given_accepted': (sum(e <= 1 for e in errors) / len(errors)
                                           if errors else None),
        'unsafe_accepted_error_gt_one_plane': unsafe,
        'failure_to_abstain_given_accepted': unsafe / len(accepted) if accepted else None,
        'unsafe_fraction_all_cases': unsafe / len(records) if records else None,
        'mean_absolute_error_planes_given_accepted': float(np.mean(errors)) if errors else None,
        'maximum_error_planes_given_accepted': float(max(errors)) if errors else None,
        'abstention_reason_counts_nonexclusive': dict(sorted(reasons.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    checks = self_test()
    if args.self_test:
        print(json.dumps(checks, indent=2))
        return
    if not args.data or not args.output:
        parser.error('--data and --output are required unless --self-test is used')
    start_cpu, start_wall = time.process_time(), time.monotonic()
    protocol_bytes = (HERE / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    inventory = json.loads((args.data.parent / 'source_inventory.json').read_text())
    if sha256(protocol_bytes).hexdigest() != inventory['protocol_freeze']['protocol_sha256']:
        raise RuntimeError('protocol does not match the pre-download freeze')
    if sha256(args.data.read_bytes()).hexdigest() != inventory['npy_sha256']:
        raise RuntimeError('input .npy hash differs from verified fetch inventory')
    native = np.load(args.data, allow_pickle=False)
    if list(native.shape) != protocol['source_shape_zyx'] or native.dtype != np.uint8:
        raise RuntimeError('unexpected native input shape/dtype')
    if sha256(native.tobytes()).hexdigest() != inventory['array_bytes_sha256']:
        raise RuntimeError('input array byte hash differs from verified fetch inventory')
    cells = []
    for split, rows in [('calibration', protocol['calibration_grid_rows']),
                        ('validation', protocol['validation_grid_rows'])]:
        for row in rows:
            for col in protocol['grid_columns']:
                patch = native[:, row * 128:(row + 1) * 128,
                               col * 128:(col + 1) * 128]
                band = patch[4:24]
                nonzero = float(np.count_nonzero(band) / band.size)
                sd = float(band.std())
                eligible = nonzero >= 0.95 and sd >= 5.0
                cells.append({'id': f'r{row}c{col}', 'split': split,
                              'nonzero_fraction': nonzero, 'intensity_sd': sd,
                              'anatomically_eligible': eligible,
                              'native_profile': trimmed_profile(patch)})
    calibration = [c for c in cells if c['split'] == 'calibration' and
                   c['anatomically_eligible']]
    if not calibration:
        raise RuntimeError('no eligible calibration patches; protocol cannot be fitted')
    peak_bias = float(np.median([np.argmax(c['native_profile'][4:24]) - 9.5
                                for c in calibration]))
    template = np.median([normalize(c['native_profile'][4:24]) for c in calibration], axis=0)
    cases = []
    for cell in cells:
        for injected in protocol['injected_offsets_planes']:
            observed = cell['native_profile'][4 + injected:24 + injected]
            cases.append({
                'cell': cell['id'], 'split': cell['split'],
                'injected_offset_planes': injected, 'truth_correction_planes': -injected,
                'naive_peak': {'accepted': True, 'correction_planes':
                               float(np.argmax(observed) - 9.5 - peak_bias)},
                'bounded_template': bounded_match(observed, template,
                                                   cell['anatomically_eligible']),
            })
    methods = ['naive_peak', 'bounded_template']
    summaries = {}
    for split in ['calibration', 'validation']:
        records = [r for r in cases if r['split'] == split]
        summaries[split] = {m: summarize(records, m) for m in methods}
        summaries[split]['per_offset'] = {
            str(d): {m: summarize([r for r in records if r['injected_offset_planes'] == d], m)
                     for m in methods} for d in protocol['injected_offsets_planes']}
        summaries[split]['per_cell'] = {
            c['id']: {m: summarize([r for r in records if r['cell'] == c['id']], m)
                      for m in methods} for c in cells if c['split'] == split}
    validation = summaries['validation']['bounded_template']
    proceed = (validation['coverage'] >= 0.5 and
               validation['unsafe_accepted_error_gt_one_plane'] == 0)
    result = {
        'experiment': 'public0841_native_plane_offset_recovery_v1',
        'protocol_sha256': sha256(protocol_bytes).hexdigest(),
        'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_array_bytes_sha256': inventory['array_bytes_sha256'],
        'source_npy_sha256': inventory['npy_sha256'],
        'source_objects_verified': inventory['objects_verified'],
        'source_object_bytes': inventory['object_bytes'],
        'source_protocol_freeze': inventory['protocol_freeze'],
        'data_provenance': 'evidence/sources.json case w00, level 0 only, no additional resampling',
        'spacing_um': 9.366,
        'calibration_peak_bias_planes': peak_bias,
        'calibration_template': [float(v) for v in template],
        'anatomical_eligibility_counts': {
            split: {'selected': sum(c['split'] == split for c in cells),
                    'eligible': sum(c['split'] == split and c['anatomically_eligible'] for c in cells)}
            for split in ['calibration', 'validation']},
        'self_tests': checks,
        'summaries': summaries,
        'passed_continuation_gate': proceed,
        'continuation_gate_is_not_deployment_approval': True,
        'independent_validation_units': 12,
        'limitation': 'Twelve adjacent held-out cells on one released surface are spatially correlated. Nine offsets per cell are repeated transforms, not independent observations. Stored band center is not a verified physical recto. No ink result was evaluated.',
        'cells': [{**c, 'native_profile': [float(v) for v in c['native_profile']]} for c in cells],
        'cases': cases,
        'runtime_cpu_seconds': time.process_time() - start_cpu,
        'runtime_wall_seconds': time.monotonic() - start_wall,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    # A compact table keeps the primary measured result easy to inspect.
    concise = {key: result[key] for key in [
        'experiment', 'protocol_sha256', 'source_array_bytes_sha256',
        'source_objects_verified', 'source_object_bytes',
        'anatomical_eligibility_counts', 'calibration_peak_bias_planes',
        'passed_continuation_gate', 'runtime_cpu_seconds', 'runtime_wall_seconds',
        'limitation']}
    concise['validation'] = {m: summaries['validation'][m] for m in methods}
    concise['zero_offset_validation'] = summaries['validation']['per_offset']['0']
    (args.output / 'summary.json').write_text(json.dumps(concise, indent=2, allow_nan=False) + '\n')
    print(json.dumps(concise, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
