#!/usr/bin/env python3
"""Differential parity against the actual archived CLI, not a second loop copy."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_fixture(folder):
    off = np.fromfile(folder / 'offsets.i64', '<i8')
    fam = np.fromfile(folder / 'family_codes.i8', 'i1')
    zb = np.fromfile(folder / 'z_bounds.i32', '<i4').reshape(-1, 2)
    xyz = np.fromfile(folder / 'coordinates.i32', '<i4').reshape(-1, 3)
    with np.load(folder / 'crossings.npz', allow_pickle=False) as archive:
        cross = {key: archive[key] for key in archive.files}
    cp = json.loads((folder / 'umbilicus.json').read_text())['control_points']
    return off, fam, zb, xyz, cross, cp


def transform_fixture(folder, mode):
    off, fam, _, xyz, cross, _ = read_fixture(folder)
    lengths = np.diff(off)
    reverse = np.zeros(len(fam), dtype=bool)
    if mode == 'reverse_all':
        reverse[:] = True
    elif mode == 'random_half':
        reverse = np.random.default_rng(20261123).integers(0, 2, len(fam)).astype(bool)
    elif mode not in ('original', 'reverse_crossing_rows'):
        raise ValueError(mode)
    for t in np.flatnonzero(reverse):
        xyz[off[t]:off[t+1]] = xyz[off[t]:off[t+1]][::-1]
    for t in range(len(fam)):
        sl = slice(cross['offsets'][t], cross['offsets'][t+1])
        if reverse[t]:
            cross['self_local'][sl] = lengths[t]-1-cross['self_local'][sl]
        partners = cross['partners'][sl]
        pl = cross['partner_local'][sl]
        cross['partner_local'][sl] = np.where(reverse[partners], lengths[partners]-1-pl, pl)
        if mode == 'reverse_crossing_rows':
            for key in ('partners', 'self_local', 'partner_local'):
                cross[key][sl] = cross[key][sl][::-1]
    xyz.tofile(folder / 'coordinates.i32')
    np.savez(folder / 'crossings.npz', **cross)


def random_fixture(folder, seed):
    """Independent finite tracks covering empty, filtered and full-turn cases."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    tracks = []
    for t in range(9):
        count = int(rng.integers(15, 180))
        z = int(rng.integers(-45, 46))
        theta = np.linspace(rng.uniform(-6, 6), rng.uniform(-6, 6), count)
        if t % 3 == 0:
            theta = np.linspace(-2, 6, count)
        radius = rng.uniform(100, 850) + rng.uniform(-4, 4)*theta
        cy = 1000+0.25*z
        cx = 1200-0.125*z
        xyz = np.rint(np.stack((np.full(count, z), cy+radius*np.sin(theta), cx+radius*np.cos(theta)), axis=1)).astype('<i4')
        if rng.random() < 0.5:
            xyz = xyz[::-1].copy()
        tracks.append(xyz)
    off = np.r_[0, np.cumsum(list(map(len, tracks)))].astype('<i8')
    off.tofile(folder / 'offsets.i64')
    np.concatenate(tracks).tofile(folder / 'coordinates.i32')
    np.array([[v[0, 0], v[0, 0]] for v in tracks], dtype='<i4').tofile(folder / 'z_bounds.i32')
    np.zeros(len(tracks), dtype='i1').tofile(folder / 'family_codes.i8')
    np.savez(folder / 'crossings.npz', offsets=np.zeros(len(tracks)+1, dtype='<i8'),
             partners=np.empty(0, dtype='<i8'), self_local=np.empty(0, dtype='<i8'),
             partner_local=np.empty(0, dtype='<i8'))
    cp = [dict(z=z, y=1000+0.25*z, x=1200-0.125*z) for z in (-64, 64)]
    (folder / 'umbilicus.json').write_text(json.dumps({'control_points': cp}))


def compare(adapter, baseline, folder, options):
    off, fam, zb, xyz, cross, cp = read_fixture(folder)
    ids = np.flatnonzero((fam == 0) & (zb[:, 0] >= options['z0']-options['dz']) &
                         (zb[:, 1] <= options['z0']+options['dz']))
    def coordinates_for(t):
        return xyz[off[t]:off[t+1]]
    def crossings_for(t):
        sl = slice(cross['offsets'][t], cross['offsets'][t+1])
        return tuple(cross[key][sl] for key in ('partners', 'self_local', 'partner_local'))
    observed, rows = adapter.trace_legacy(ids, coordinates_for, crossings_for,
                                         lambda t: fam[t], lambda t: off[t+1]-off[t], cp, **options)
    output = folder / 'archived_cli_result.npy'
    names = ('z0', 'dz', 'nstart', 'rtol', 'front', 'jmax', 'zstep', 'rmax', 'mirror')
    command = [sys.executable, str(baseline), str(folder), str(folder/'crossings.npz'),
               str(folder/'umbilicus.json'), str(output), *(str(options[k]) for k in names)]
    run = subprocess.run(command, text=True, capture_output=True, timeout=30, check=True)
    expected = json.loads(run.stdout.strip().splitlines()[-1])
    expected_rows = np.load(output, allow_pickle=False).reshape(-1, 3)
    assert {key: observed[key] for key in expected} == expected, (folder, options, observed, expected)
    assert set(observed) == set(expected) | {'completed_start_ids'}
    assert np.array_equal(rows, expected_rows), (folder, options, rows, expected_rows)
    assert len(observed['completed_start_ids']) == len(rows)
    assert len(set(observed['completed_start_ids'])) == len(rows)
    assert set(observed['completed_start_ids']).issubset(ids.tolist())
    return {'case': folder.name, 'mirror': options['mirror'],
            'summary_exact': True, 'rows_bitwise_equal': rows.tobytes() == expected_rows.tobytes(),
            'completed': len(rows), 'broken': expected['broken'],
            'nstart': options['nstart'], 'rmax': options['rmax']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    adapter = load_module('legacy_adapter', here/'legacy_adapter.py')
    baseline = here/'legacy_snapshot.py'
    oracle = load_module('independent_taper_oracle', args.oracle)
    options = dict(z0=16, dz=40, nstart=3, rtol=20., front=.3, jmax=200,
                   zstep=16., rmax=1500., mirror=1)
    records = []
    with tempfile.TemporaryDirectory(prefix='legacy-adapter-parity-') as directory:
        root = Path(directory)
        for pitch in (20, -20):
            for taper, z_path, case in ((0, (0, 16, 32), 'untapered'),
                                       (-int(np.sign(pitch)), (0, 16, 32), 'tapered'),
                                       (-int(np.sign(pitch)), (0, 16, 0), 'same_z')):
                for mode in ('original', 'reverse_all', 'random_half', 'reverse_crossing_rows'):
                    folder = root/f'{pitch}_{case}_{mode}'
                    oracle.construct_fixture(folder, pitch, taper, z_path)
                    transform_fixture(folder, mode)
                    for mirror in (1, -1):
                        records.append(compare(adapter, baseline, folder, options | {'mirror': mirror}))
        for seed in range(8):
            folder = root/f'random_{seed}'
            random_fixture(folder, seed)
            for mirror in (1, -1):
                variation = dict(z0=0, nstart=(0, 1, 5, 100)[seed % 4],
                                 rmax=(1e9, 400.)[seed % 2], mirror=mirror)
                records.append(compare(adapter, baseline, folder, options | variation))
    assert all(r['rows_bitwise_equal'] for r in records)
    report = {'schema': 'legacy-callback-differential-parity-v1',
              'legacy_sha256': adapter.verify_legacy_snapshot(),
              'adapter_sha256': hashlib.sha256((here/'legacy_adapter.py').read_bytes()).hexdigest(),
              'check_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'oracle_sha256': hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
              'historical_settings': {'seed': 0, 'max_hops': 60},
              'test_count': len(records), 'all_original_summary_fields_exact': True,
              'all_result_arrays_bitwise_equal': True,
              'nonempty_result_cases': sum(r['completed'] > 0 for r in records),
              'records': records}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}))


if __name__ == '__main__':
    main()
