"""Generated controls only. Does not acquire any scroll data."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest

MODULE = Path(__file__).resolve().parent / 'native_pair_geometry.py'
spec = importlib.util.spec_from_file_location('native_pair_geometry', MODULE)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
F = json.loads((MODULE.parent / 'native_pair_geometry_sources.json').read_text())


def test_known_affine_and_readonly_audit():
    for key in ('frag5', 'frag6'):
        report = g.audit_landmarks(F[key])
        assert len(report['leave_one_out']) == 5
        assert report['published_affine_in_sample_rms_voxels'] > 0
        assert report['leave_one_out_rms_voxels'] > 0
        assert np.isfinite(report['ray_vs_transformed_normal_angle_degrees_max_sampled'])
        A, src, dst, dims = g.matrix_and_data(F[key])
        assert np.allclose(g.transform_xyz(src, A), dst, atol=2)


def test_missing_support_rejected_without_clamping():
    with pytest.raises(ValueError, match='lack complete target CT support'):
        g.chunk_plan(F['frag5'], [[-1000, 1, 2]], [[1, 0, 0]], [0], 'target_normal')


def test_bounded_plan_and_corner_support():
    # Real public source landmark, but synthetic normal. This is not a
    # valid surface ray or donor ink scoring experiment.
    p = [F['frag5']['source_xyz'][0]]
    n = [[1, 0, 0]]
    for mode in ('preserve_ray', 'target_normal'):
        q = g.chunk_plan(F['frag5'], p, n, [-40, 0, 40], mode)
        assert q['decoded_chunks'] <= 8
        assert q['all_trilinear_neighbors_in_bounds']
        assert q['decoded_upper_bound_bytes'] == q['decoded_chunks'] * 4194304
    with pytest.raises(ValueError, match='budget exceeded'):
        g.chunk_plan(F['frag5'], p, n, [-40, 0, 40], 'preserve_ray', max_chunks=0)


def test_no_silent_nonfinite_or_invalid_mode():
    p, n = [[2000, 700, 500]], [[0, 1, 0]]
    with pytest.raises(ValueError, match='Finite'):
        g.chunk_plan(F['frag5'], [[float('nan'), 0, 0]], n, [0], 'target_normal')
    with pytest.raises(ValueError, match='mode must be'):
        g.chunk_plan(F['frag5'], p, n, [0], 'guess_direction')
    with pytest.raises(ValueError, match='zero normals'):
        g.chunk_plan(F['frag5'], p, [[0, 0, 0]], [0], 'preserve_ray')


def test_trilinear_neighbor_on_boundary():
    synthetic = dict(F['frag5'])
    synthetic['affine_xyz'] = np.eye(4).tolist()
    synthetic['target_shape_zyx'] = [256, 256, 256]
    synthetic['source_pitch_um'] = synthetic['target_pitch_um'] = 1
    # 127.5 touches two X-chunks and is rejected if budget below two.
    q = g.chunk_plan(synthetic, [[127.5, 10, 10]], [[0, 0, 1]], [0], 'target_normal')
    assert q['decoded_chunks'] == 2
    with pytest.raises(ValueError, match='budget exceeded'):
        g.chunk_plan(synthetic, [[127.5, 10, 10]], [[0, 0, 1]], [0], 'target_normal', max_chunks=1)


@pytest.mark.parametrize('point, expected', [
    ([127.5, 127.5, 10], [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]),
    ([127.5, 127.5, 127.5], [
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ]),
])
def test_mixed_trilinear_corners_count_toward_chunk_budget(point, expected):
    """One sample can require four/eight chunks, not only two diagonals."""
    synthetic = dict(F['frag5'])
    synthetic['affine_xyz'] = np.eye(4).tolist()
    synthetic['target_shape_zyx'] = [256, 256, 256]
    synthetic['source_pitch_um'] = synthetic['target_pitch_um'] = 1
    result = g.chunk_plan(synthetic, [point], [[0, 0, 1]], [0], 'target_normal')
    assert result['target_chunks_zyx'] == expected
    assert result['decoded_upper_bound_bytes'] == len(expected) * 4194304
    with pytest.raises(ValueError, match='budget exceeded'):
        g.chunk_plan(synthetic, [point], [[0, 0, 1]], [0],
                     'target_normal', max_chunks=len(expected)-1)
