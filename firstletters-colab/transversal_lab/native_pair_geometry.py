#!/usr/bin/env python3
"""Public Frag5/Frag6 affine stability and bounded native-CT chunk planning.

No ink/CT is fetched, inferred, or decoded by this module. A transform based on
five published fitted landmarks is *not* an independently validated registration.
A chunk plan is a bound on support acquisition, not a measurement of image quality.

Sources and fixed numeric contents are in native_pair_geometry_sources.json.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / 'native_pair_geometry_sources.json'


def matrix_and_data(fixture: dict):
    A4 = np.asarray(fixture['affine_xyz'], dtype=np.float64)
    src = np.asarray(fixture['source_xyz'], dtype=np.float64)
    dst = np.asarray(fixture['target_xyz'], dtype=np.float64)
    dims = tuple(fixture['target_shape_zyx'])
    if A4.shape != (4, 4) or src.shape != (5, 3) or dst.shape != (5, 3):
        raise ValueError('Expected one 4x4 affine and exactly five published XYZ pairs')
    if not np.allclose(A4[3], [0, 0, 0, 1], rtol=0, atol=0):
        raise ValueError('Affine must be homogeneous, no perspective division')
    if any(int(x) <= 0 for x in dims) or len(dims) != 3:
        raise ValueError('Invalid target ZYX geometry')
    if not all(np.isfinite(x).all() for x in (A4, src, dst)):
        raise ValueError('Nonfinite fixture')
    return A4, src, dst, dims


def transform_xyz(points_xyz: np.ndarray, A4: np.ndarray):
    p = np.asarray(points_xyz, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('Expected finite Nx3 scanner XYZ indices')
    return p @ A4[:3, :3].T + A4[:3, 3]


def least_squares_affine(src_xyz, dst_xyz):
    """Fit a fresh affine from just the training pairs; avoid published-fit reuse."""
    M = np.column_stack([src_xyz, np.ones(len(src_xyz))])
    if len(M) < 4 or np.linalg.matrix_rank(M) < 4:
        raise ValueError('Insufficient noncoplanar landmarks for 3D affine')
    coeff = np.linalg.lstsq(M, dst_xyz, rcond=None)[0]
    return coeff, float(np.linalg.cond(M))


def audit_landmarks(fixture):
    A4, src, dst, dims = matrix_and_data(fixture)
    pub = transform_xyz(src, A4)
    pub_err = np.linalg.norm(pub - dst, axis=1)
    loocv = []
    for test in range(5):
        train = np.arange(5) != test
        coeff, condition = least_squares_affine(src[train], dst[train])
        predicted = np.r_[src[test], 1] @ coeff
        delta = predicted - dst[test]
        loocv.append({'left_out_landmark': test, 'error_target_voxels': float(np.linalg.norm(delta)),
                      'error_xyz_target_voxels': [float(x) for x in delta],
                      'training_design_condition': condition})
    errors = np.asarray([row['error_target_voxels'] for row in loocv])
    full_coeff, full_condition = least_squares_affine(src, dst)
    pub_to_refit = np.linalg.norm(transform_xyz(src, A4) - (np.column_stack([src, np.ones(5)]) @ full_coeff), axis=1)
    # Synthetic normals only, not empirical sheet normals.
    N = 4096
    i = np.arange(N, dtype=np.float64)
    z = 1 - 2 * (i + .5) / N
    phi = i * (math.pi * (3 - math.sqrt(5)))
    r = np.sqrt(1 - z * z)
    normals = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    A = A4[:3, :3]
    propagated_rays = normals @ A.T
    target_normals = normals @ np.linalg.inv(A)
    propagated_rays /= np.linalg.norm(propagated_rays, axis=1, keepdims=True)
    target_normals /= np.linalg.norm(target_normals, axis=1, keepdims=True)
    angle_deg = np.degrees(np.arccos(np.clip(np.sum(propagated_rays * target_normals, axis=1), -1, 1)))
    return {
        'physical_parent': fixture['parent'],
        'public_transform_source': fixture['transform_url'],
        'public_landmarks_source': fixture['landmarks_url'],
        'comparison_definition': 'Published full-five affine in-sample; new five leave-one-out 4-point fits, each predicted on its omitted fifth point.',
        'published_affine_in_sample_rms_voxels': float(np.sqrt(np.mean(pub_err ** 2))),
        'published_affine_in_sample_max_voxels': float(pub_err.max()),
        'published_affine_in_sample_residuals_voxels': pub_err.tolist(),
        'leave_one_out_rms_voxels': float(np.sqrt(np.mean(errors ** 2))),
        'leave_one_out_max_voxels': float(errors.max()),
        'leave_one_out': loocv,
        'all_five_refit_design_condition': full_condition,
        'published_vs_refit_landmarks_rms_voxels': float(np.sqrt(np.mean(pub_to_refit ** 2))),
        'deterministic_normal_sample_count': N,
        'ray_vs_transformed_normal_angle_degrees_p50': float(np.median(angle_deg)),
        'ray_vs_transformed_normal_angle_degrees_max_sampled': float(angle_deg.max()),
        'normal_angle_interpretation': '4096 synthetic normals; angle alone says nothing about actual sheet normals, local registration or ink',
        'target_shape_zyx': dims,
        'limitations': [
            'The five published points are not independent of the original published affine fit.',
            'LOO four-point fits are exact/interpolating and can be highly sensitive to landmark geometry; this estimates extrapolation sensitivity among those five landmarks only.',
            'These points are distributed over a volume, not held-out landmarks near a particular scoring patch.',
            'No ink labels, image/CT pixels, source PPM, actual normals, or quantitative sheet-identity checks are used.',
        ],
    }


def chunk_plan(fixture, points_xyz, normals_xyz, offsets_um, mode, max_chunks=64):
    """Plan full trilinear support at transformed points; never fill/clamp invalids.

    Source/target scanner voxel indices are XYZ; source and target pitches in
    micrometres. preserve_ray maps a physical source ray by A.
    target_normal uses the inverse-transpose direction at target pitch.
    This is a planner, not a sampler, and cannot determine the physical sheet side.
    """
    A4, _, _, (sz, sy, sx) = matrix_and_data(fixture)
    P = np.asarray(points_xyz, dtype=np.float64)
    N = np.asarray(normals_xyz, dtype=np.float64)
    depths = np.asarray(offsets_um, dtype=np.float64)
    if (P.ndim != 2 or P.shape[1] != 3 or N.shape != P.shape or
            depths.ndim != 1 or len(depths) < 1 or not np.isfinite(depths).all() or
            not np.isfinite(P).all() or not np.isfinite(N).all()):
        raise ValueError('Finite XYZ centers/normals and finite 1D physical offsets required')
    if len(P) == 0 or np.any(np.linalg.norm(N, axis=1) < 1e-12):
        raise ValueError('Missing centers or zero normals')
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    if mode == 'preserve_ray':
        direction = N @ A4[:3, :3].T / fixture['source_pitch_um']
    elif mode == 'target_normal':
        direction = N @ np.linalg.inv(A4[:3, :3])
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        direction /= fixture['target_pitch_um']
    else:
        raise ValueError('mode must be preserve_ray or target_normal')
    centers = transform_xyz(P, A4)
    locations = centers[:, None, :] + direction[:, None, :] * depths[None, :, None]
    if not np.isfinite(locations).all():
        raise ValueError('Invalid transformed ray locations')
    # A trilinear sample needs floor and ceil. Exact terminal integers are valid.
    lo = np.floor(locations).astype(np.int64)
    hi = np.ceil(locations).astype(np.int64)
    ok = ((lo >= 0).all(axis=2) & (hi[:, :, 0] < sx) &
          (hi[:, :, 1] < sy) & (hi[:, :, 2] < sz))
    valid_centers = np.all(ok, axis=1)
    if not np.all(valid_centers):
        raise ValueError(f'{int((~valid_centers).sum())}/{len(P)} centers lack complete target CT support: reject, never clip/fill')
    chunks = set()
    # Z/Y/X chunk keys, whereas transforms and profile samples use X/Y/Z.
    # A trilinear sample needs every Cartesian combination of lower/upper
    # XYZ coordinates. Two diagonal endpoints miss mixed corners when more
    # than one axis crosses a chunk boundary.
    for use_upper in itertools.product((False, True), repeat=3):
        boundary = np.where(use_upper, hi, lo)
        for q in boundary.reshape((-1, 3)):
            chunks.add((int(q[2] // 128), int(q[1] // 128), int(q[0] // 128)))
            if len(chunks) > max_chunks:
                raise ValueError(f'Chunk budget exceeded: >{max_chunks} decoded 128^3 uint16 chunks')
    return {'points': len(P), 'depths': len(depths), 'mode': mode,
            'target_chunks_zyx': [list(c) for c in sorted(chunks)],
            'decoded_upper_bound_bytes': 128**3 * 2 * len(chunks),
            'decoded_chunks': len(chunks),
            'physical_offset_um_min_max': [float(depths.min()), float(depths.max())],
            'all_trilinear_neighbors_in_bounds': True,
            'interpretation': 'Acquisition plan only: no downloaded data; not proof of surface identity or exact compressed transfer bytes'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture', choices=['frag5', 'frag6'])
    ap.add_argument('--audit', action='store_true')
    ap.add_argument('--points-json', type=Path, help='Geometry-only JSON: points_xyz, normals_xyz, offsets_um, mode')
    ap.add_argument('--output', type=Path)
    ns = ap.parse_args(argv)
    if not ns.fixture or not (ns.audit or ns.points_json) or (ns.audit and ns.points_json):
        ap.error('choose --fixture and exactly one of --audit / --points-json')
    raw = FIXTURES.read_bytes()
    fixtures = json.loads(raw)
    f = fixtures[ns.fixture]
    if ns.audit:
        result = audit_landmarks(f)
    else:
        data = json.loads(ns.points_json.read_bytes())
        result = chunk_plan(f, data['points_xyz'], data['normals_xyz'], data['offsets_um'], data['mode'])
    envelope = {'schema': 'public-native-pair-geometry-v1',
                'fixture': ns.fixture, 'fixture_file_sha256': hashlib.sha256(raw).hexdigest(),
                'result': result}
    payload = json.dumps(envelope, indent=2, allow_nan=False) + '\n'
    if ns.output:
        ns.output.write_text(payload)
    else:
        sys.stdout.write(payload)


if __name__ == '__main__':
    main()
