#!/usr/bin/env python3
"""Independent offline scoring and optional full-array review of frozen Q33 v2.

Imports no producer module. The portable mode needs the new scored archive and
receipts, frozen protocol, and the previous 1.1 MB Q10 archive. Optional prepared
and source-cache arguments also verify native identity bytes and literal
four-corner probability readback. No model forward or geometry fitting occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

PROTOCOL_SHA256 = 'd56864f88ba32302344c1858f9a53cf4c1f203b64a7b28fab980105535793a80'
HISTORICAL_SHA256 = '04b107ef27c8a17cec25fbc90939d4ccb1b36190240a74f238fa75cf5a860e08'
FROZEN_COMMIT = 'ae294dc44aa949132cb62b9f1b724514786a524d'
ARMS = [('identity', 'x', 0)] + [
    (f'{axis}_{"minus" if angle < 0 else "plus"}{abs(angle)}', axis, angle)
    for axis in ('x', 'y') for angle in (-20, -10, -5, 5, 10, 20)]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metric(probability, states):
    known = states != 0
    truth, score = states[known] == 2, probability[known]
    assert np.isfinite(score).all() and np.all((0 <= score) & (score <= 1))
    pos, neg = int(truth.sum()), int((~truth).sum())
    predicted = score >= .5
    tp, fp = int((predicted & truth).sum()), int((predicted & ~truth).sum())
    return dict(ink_pixels=pos, background_pixels=neg, unknown_pixels=int((~known).sum()),
                true_positive_at_0_5=tp, false_positive_at_0_5=fp,
                recall_at_0_5=tp/pos if pos else None,
                false_positive_rate_at_0_5=fp/neg if neg else None,
                ap=float(average_precision_score(truth, score)) if pos and neg else None,
                auc=float(roc_auc_score(truth, score)) if pos and neg else None)


def select_history(history):
    """Independent implementation of the fixed label-only, within-class maximin."""
    rows = []
    for position, (index, origin, state) in enumerate(zip(
            history['author_indices'], history['core_origins_yx'], history['states'])):
        ink, bg = int((state == 2).sum()), int((state == 1).sum())
        rows.append(dict(index=int(index), y=int(origin[0])-48, x=int(origin[1])-48,
                         ink=ink, background=bg, position=position))
    known = lambda row: row['ink'] + row['background']
    key = lambda row: (row['y'], row['x'], row['index'])
    chosen = [dict(row, category='mixed') for row in rows if known(row) >= 512
              and 10*row['ink'] >= known(row) and 10*row['ink'] <= 9*known(row)]
    orders = {}
    for category, zero in [('ink', 'background'), ('background', 'ink')]:
        pool = [row for row in rows if known(row) >= 512 and row[zero] == 0]
        selected = [min(pool, key=key)]
        while len(selected) < 3:
            remaining = [row for row in pool if row not in selected]
            distances = [min((row['y']-old['y'])**2 + (row['x']-old['x'])**2
                             for old in selected) for row in remaining]
            best = max(distances)
            selected.append(min([row for row, distance in zip(remaining, distances)
                                 if distance == best], key=key))
        chosen.extend(dict(row, category=category) for row in selected)
        orders[category] = [row['index'] for row in selected]
    return sorted(chosen, key=lambda row: row['index']), orders


def literal_readback(field, query_yx):
    """Independent pointwise bilinear sum, with explicit active-corner bounds."""
    q = np.asarray(query_yx, dtype=np.float64)
    assert np.isfinite(q).all()
    lower = np.floor(q).astype(np.int64)
    fraction = q-lower
    upper = lower+(fraction > 0)
    assert (lower >= 0).all() and (upper < np.array(field.shape)).all()
    answer = np.zeros(q.shape[:-1], dtype=np.float64)
    for ay, ax in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        iy = lower[..., 0] if ay == 0 else upper[..., 0]
        ix = lower[..., 1] if ax == 0 else upper[..., 1]
        wy = 1-fraction[..., 0] if ay == 0 else fraction[..., 0]
        wx = 1-fraction[..., 1] if ax == 0 else fraction[..., 1]
        answer += wy*wx*field[iy, ix].astype(np.float64)
    return answer.astype(np.float32)


def independent_raw_crop(tiles_array, tile_origins, y, x):
    """Intersect every source tile with the desired rectangle; no producer helper."""
    answer = np.zeros((128, 128, 16), dtype=np.uint8)
    coverage = np.zeros((128, 128), dtype=np.uint8)
    for tile, (ty, tx) in zip(tiles_array, tile_origins):
        ty, tx = int(ty), int(tx)
        lo_y, hi_y = max(y, ty), min(y+128, ty+512)
        lo_x, hi_x = max(x, tx), min(x+128, tx+512)
        if lo_y >= hi_y or lo_x >= hi_x:
            continue
        region = np.s_[lo_y-y:hi_y-y, lo_x-x:hi_x-x]
        answer[region] = tile[lo_y-ty:hi_y-ty, lo_x-tx:hi_x-tx]
        coverage[region] += 1
    assert np.all(coverage == 1)
    return answer.transpose(2, 0, 1)


def audit(args):
    started = time.monotonic()
    assert sha(args.protocol) == PROTOCOL_SHA256
    protocol = json.loads(args.protocol.read_text())
    assert [(a['name'], a['axis'], a['angle_deg']) for a in protocol['arms']] == ARMS
    assert protocol['historical_archive_sha256'] == HISTORICAL_SHA256 == sha(args.historical)
    assert protocol['geometry_gate']['maximum_physical_residual_native_voxels'] == .01
    assert protocol['geometry_gate']['minimum_known_retention_global'] == .95
    assert protocol['geometry_gate']['minimum_known_retention_each_core'] == .90
    assert 'before any detector inference' in protocol['preflight_adaptation']
    result = json.loads((args.output/'results.json').read_text())
    assert result['protocol_sha256'] == PROTOCOL_SHA256
    archive_path = args.output/'scored_arrays.npz'
    assert sha(archive_path) == result['scored_archive_sha256']
    with np.load(archive_path, allow_pickle=False) as archived:
        assert set(archived.files) == {'probabilities','states','original_states','common_mask',
                                       'author_indices','condition_names'}
        probability, states, original, common, indices, names = [archived[k] for k in
            ('probabilities','states','original_states','common_mask','author_indices','condition_names')]
    names = names.tolist()
    assert names == [a[0] for a in ARMS]
    assert probability.shape == (13, 17, 32, 32) and probability.dtype == np.float32
    assert original.shape == states.shape == common.shape == (17, 32, 32)
    assert common.dtype == bool and np.isin(original, (0,1,2)).all()
    assert np.array_equal(states, np.where(common, original, 0))
    with np.load(args.historical, allow_pickle=False) as history:
        selected, orders = select_history(history)
        positions = [row['position'] for row in selected]
        old_identity = history['probabilities'][0, positions]
        assert history['condition_names'][0] == 'identity'
        assert np.array_equal(original, history['states'][positions])
    assert indices.tolist() == [row['index'] for row in selected]
    assert orders == protocol['pure_maximin_order']
    assert len(set(indices.tolist())) == 17
    occupied = set()
    for actual, expected in zip(protocol['controls'], selected):
        assert actual == {key:value for key,value in expected.items() if key != 'position'}
        cells = {(expected['y']+dy, expected['x']+dx) for dy in range(48,80) for dx in range(48,80)}
        assert not cells.intersection(occupied)
        occupied.update(cells)
    assert len(occupied) == 17*32*32
    assert tuple(int((original == i).sum()) for i in (2,1,0)) == (8416,8640,352)
    original_known = (original > 0).sum(axis=(1,2))
    kept_known = (states > 0).sum(axis=(1,2))
    assert 100*kept_known.sum() >= 95*original_known.sum()
    assert np.all(100*kept_known >= 90*original_known)
    initial_failed = indices[100*kept_known < 95*original_known].tolist()
    assert initial_failed == [326,357,359,430]
    expected_denominator = dict(ink=int((states == 2).sum()), background=int((states == 1).sum()),
        excluded_known=int(((original > 0)&~common).sum()), original_unknown=int((original == 0).sum()))
    assert protocol['denominator'] == expected_denominator
    mixed = [i for i,row in enumerate(selected) if row['category'] == 'mixed']
    assert len(mixed) == 11 and all(np.any(states[i] == 1) and np.any(states[i] == 2) for i in mixed)
    pooled = {name:metric(p, states) for name,p in zip(names, probability)}
    per_core = [{name:metric(p[i], states[i]) for name,p in zip(names, probability)} for i in range(17)]
    checks, maximum_error = 0, 0.0
    def compare(actual, expected):
        nonlocal checks, maximum_error
        if isinstance(expected, dict):
            for key,value in expected.items(): compare(actual[key],value)
        elif isinstance(expected, list):
            assert len(actual) == len(expected)
            for a,b in zip(actual,expected): compare(a,b)
        else:
            checks += 1
            if isinstance(expected, float):
                difference = abs(actual-expected)
                maximum_error = max(maximum_error, difference)
                assert difference < 1e-12, (actual,expected)
            else:
                assert actual == expected, (actual,expected)
    compare(result['pooled'], pooled)
    for i,row in enumerate(selected):
        assert result['per_core'][i]['author_index'] == row['index']
        assert result['per_core'][i]['category'] == row['category']
        compare(result['per_core'][i]['metrics'], per_core[i])
    comparisons = {}
    for name in names[1:]:
        base, current = pooled['identity'], pooled[name]
        differences = [per_core[i][name]['ap']-per_core[i]['identity']['ap'] for i in mixed]
        dap = current['ap']-base['ap']
        dr = current['recall_at_0_5']-base['recall_at_0_5']
        df = current['false_positive_rate_at_0_5']-base['false_positive_rate_at_0_5']
        comparisons[name] = dict(delta_AP=dap, delta_AUC=current['auc']-base['auc'],
            delta_recall=dr, delta_FPR=df, median_mixed_core_AP_delta=float(np.median(differences)),
            mixed_core_AP_deltas=differences, mixed_cores_losing_more_than_0_10_AP=int((np.array(differences)<-.10).sum()),
            material_AP_drop=bool(dap < -.05), material_recall_loss=bool(dr < -.10), material_FPR_increase=bool(df > .03))
    compare(result['comparisons'], comparisons)
    old_metric = metric(old_identity, states)
    identity_difference = float(np.max(np.abs(probability[0]-old_identity)))
    identity_ap_difference = abs(pooled['identity']['ap']-old_metric['ap'])
    assert identity_difference <= protocol['identity_gate']['historical_max_probability_difference']
    assert identity_ap_difference <= protocol['identity_gate']['historical_max_AP_difference']
    identity = json.loads((args.output/'identity_check.json').read_text())
    compare(identity, dict(passed=True, maximum_probability_difference=identity_difference,
                          AP_difference=identity_ap_difference, metrics=pooled['identity'], historical_same_population=old_metric))
    pure_groups = {}
    for category in ('ink','background'):
        group = [i for i,row in enumerate(selected) if row['category'] == category]
        pure_groups[category] = {name:metric(p[group],states[group]) for name,p in zip(names,probability)}
    full_review = None
    if args.prepared is not None:
        assert sha(args.prepared/'protocol.json') == PROTOCOL_SHA256
        checked_hashes = {}
        for name, meta in protocol['arrays'].items():
            path = args.prepared/name
            assert path.stat().st_size == meta['bytes'] and sha(path) == meta['sha256']
            checked_hashes[name] = meta['sha256']
        for name, digest in protocol['receipts'].items():
            assert sha(args.prepared/name) == digest
        assert np.array_equal(np.load(args.prepared/'states.npy'), states)
        assert np.array_equal(np.load(args.prepared/'original_states.npy'), original)
        assert np.array_equal(np.load(args.prepared/'historical_identity.npy'), old_identity)
        receipt = json.loads((args.output/'inference_receipt.json').read_text())
        assert receipt['protocol_sha256'] == PROTOCOL_SHA256 and receipt['code_hashes'] == protocol['code_hashes']
        assert sha(args.output/'full_probabilities.npy') == receipt['full_probabilities_sha256']
        records = receipt['forwards']
        assert len(records) == 221 and all(row['arm'] == 'identity' for row in records[:17])
        assert {(row['arm'],row['author_index']) for row in records} == {(name,int(i)) for name in names for i in indices}
        full = np.load(args.output/'full_probabilities.npy', mmap_mode='r', allow_pickle=False)
        assert full.shape == (13,17,128,128) and full.dtype == np.float32 and np.isfinite(full).all()
        mapping = np.load(args.prepared/'mapping.npz', allow_pickle=False)
        assert np.array_equal(common, mapping['valid'].all(axis=0))
        assert np.array_equal(common, mapping['common_mask'])
        core = np.stack(np.meshgrid(np.arange(48,80),np.arange(48,80),indexing='ij'),axis=-1)
        assert np.array_equal(mapping['query_yx'][0], np.broadcast_to(core,(17,32,32,2)))
        assert np.max(mapping['residual_vox'][:,common]) <= .01
        assert np.all((mapping['depth_vox'][:,common] >= -8) & (mapping['depth_vox'][:,common] <= 7))
        assert np.all((mapping['query_yx'][:,common] >= 16) & (mapping['query_yx'][:,common] <= 111))
        manual_error, manual_exact = 0.0, True
        for ai in range(13):
            for pi in range(17):
                reference = literal_readback(full[ai,pi],mapping['query_yx'][ai,pi][common[pi]])
                actual = probability[ai,pi][common[pi]]
                manual_error = max(manual_error,float(np.max(np.abs(reference.astype(float)-actual))))
                manual_exact &= np.array_equal(reference,actual)
        assert manual_error <= 1.2e-7
        input_parity = None
        if args.source_cache is not None:
            for name,meta in protocol['acquisition'].items():
                path = args.source_cache/name
                assert path.stat().st_size == meta['bytes'] and sha(path) == meta['sha256']
            labels = np.load(args.source_cache/'labels.npy', mmap_mode='r', allow_pickle=False)
            ct = np.load(args.source_cache/'ct-tiles.npy', mmap_mode='r', allow_pickle=False)
            tiles = np.load(args.source_cache/'tiles.npy', allow_pickle=False)
            coordinates = np.load(args.source_cache/'coordinates.npy', allow_pickle=False)
            inputs = np.load(args.prepared/'inputs.npy', mmap_mode='r', allow_pickle=False)
            assert inputs.shape == (13,17,16,128,128) and inputs.dtype == np.uint8
            for i,row in enumerate(selected):
                assert coordinates[row['index']].tolist() == [row['y'],row['x']]
                assert np.array_equal(original[i],labels[row['y']+48:row['y']+80,row['x']+48:row['x']+80])
                raw = independent_raw_crop(ct,tiles,row['y'],row['x'])
                assert np.array_equal(inputs[0,i],raw)
            input_parity = dict(all_17_exact=True, uint8_compared=17*16*128*128)
        if args.checkpoint is not None:
            assert sha(args.checkpoint) == protocol['weights']['sha256']
        full_review = dict(checked_array_sha256=checked_hashes, inference_forwards=221,
            literal_readback_maximum_absolute_error=manual_error, literal_readback_bitwise_equal=bool(manual_exact),
            native_identity_input_parity=input_parity, checkpoint_checked=args.checkpoint is not None)
    return dict(status='pass', protocol_sha256=PROTOCOL_SHA256, producer_frozen_commit=FROZEN_COMMIT,
        archive_sha256=sha(archive_path), archive_bytes=archive_path.stat().st_size,
        scalar_checks=checks, maximum_metric_absolute_error=maximum_error,
        historical_identity_maximum_absolute_error=identity_difference,
        historical_identity_bitwise_equal=bool(np.array_equal(probability[0],old_identity)),
        label_selection_reproduced=True, original_label_denominator=dict(ink=8416,background=8640,unknown=352),
        scored_denominator=expected_denominator, common_known_retention=float(kept_known.sum()/original_known.sum()),
        minimum_core_known_retention=float(np.min(kept_known/original_known)),
        initial_95_percent_per_core_policy_failures=initial_failed,
        excluded_ink=int(((original==2)&~common).sum()),excluded_background=int(((original==1)&~common).sum()),
        pooled=pooled, comparisons=comparisons, pure_class_groups=pure_groups, full_array_review=full_review,
        checker_sha256=sha(__file__), numpy=np.__version__, sklearn=sklearn.__version__, seconds=time.monotonic()-started,
        limitations='Paired scoring on a geometry-selected common subset of one repeatedly used author donor holdout. Per-core retention policy changed after geometry-only preflight; no statistical significance, independent-scroll, isolated-obliquity or known-ink-depth claim. The portable mode does not certify CT acquisition or inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output','protocol','historical','report'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('prepared','source-cache','checkpoint'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    if args.report.exists(): parser.error('Use a fresh review report')
    report = audit(args)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','archive_bytes','scalar_checks','maximum_metric_absolute_error',
                                          'historical_identity_bitwise_equal','seconds')}))
