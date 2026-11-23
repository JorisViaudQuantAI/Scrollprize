"""Conservative continuous UV-pixel support from original source triangles.

Finite midpoint samples can miss a narrow source hole. Here each candidate
triangle is clipped against the whole output pixel. Full summed area is required
AND positive-area pair intersections are forbidden, so a hole cannot be hidden
by compensating double coverage. Unmatched touching source boundaries are also
rejected. Calculations use float64 and declared tolerances, not exact arithmetic.
This certifies source-chart support, not anatomical sheet identity or ray tubes.
"""
from __future__ import annotations

import math
import numpy as np

NOT_REQUESTED, NO_COVERAGE, NEGATIVE, DEGENERATE = 1, 2, 4, 8
OVERLAP, INCONSISTENT_BOUNDARY, ZERO_PHYSICAL_AREA = 16, 32, 64


def _area(poly):
    if len(poly) < 3:
        return 0.0
    # Translate to the first vertex to reduce cancellation at large UV origins.
    p = np.asarray(poly, dtype=np.float64) - poly[0]
    return abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                     - np.dot(p[:, 1], np.roll(p[:, 0], -1)))) * .5


def _clip(poly, a, b):
    """Clip a convex polygon/segment by the left closed half-plane of a->b."""
    if not len(poly):
        return np.empty((0, 2), np.float64)
    edge = b - a
    rel = np.asarray(poly) - a
    value = edge[0] * rel[:, 1] - edge[1] * rel[:, 0]
    output = []
    for i, current in enumerate(poly):
        previous = poly[i - 1]
        vc, vp = value[i], value[i - 1]
        ic, ip = vc >= 0, vp >= 0
        if ic != ip:
            output.append(previous + (vp / (vp - vc)) * (current - previous))
        if ic:
            output.append(current)
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def _intersection(poly, clipper):
    result = np.asarray(poly, dtype=np.float64)
    for i in range(len(clipper)):
        result = _clip(result, clipper[i - 1], clipper[i])
        if not len(result):
            break
    return result


def certify(uv, xyz, triangles, origin_uv, step_uv, shape_yx, *, query_mask=None,
            uv_area_epsilon=1e-12, relative_area_tolerance=1e-9,
            boundary_tolerance=1e-9):
    """Return whole-pixel validity and rejection accounting on a regular grid.

    origin_uv is the first pixel CENTER. A pixel spans +/-step_uv/2 around it.
    Positive faces use the original winding; bad faces poison any intersection
    including a boundary. Pairwise overlap and mismatched shared-edge checks are
    independent of area summation. No source triangle is retriangulated.
    """
    uv, xyz = np.asarray(uv, np.float64), np.asarray(xyz, np.float64)
    faces = np.asarray(triangles)
    if uv.ndim != 2 or uv.shape[1] != 2 or xyz.shape != (len(uv), 3):
        raise ValueError('Expected UV[N,2] and XYZ[N,3]')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError('Expected integer original triangles[F,3]')
    if not np.isfinite(uv).all() or not np.isfinite(xyz).all():
        raise ValueError('Nonfinite source geometry')
    if faces.size and (faces.min() < 0 or faces.max() >= len(uv)):
        raise ValueError('Source triangle index out of bounds')
    origin, step = np.asarray(origin_uv, float), np.asarray(step_uv, float)
    if origin.shape != (2,) or step.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(step).all() or (step <= 0).any():
        raise ValueError('Finite origin and positive UV step required')
    if len(shape_yx) != 2 or any(isinstance(v, bool) or int(v) != v or v <= 0 for v in shape_yx):
        raise ValueError('Positive integer grid shape required')
    h, w = map(int, shape_yx)
    if h * w > 262144:
        raise ValueError('Whole-footprint certification is bounded to 262144 pixels per call')
    if not all(math.isfinite(v) and v >= 0 for v in (uv_area_epsilon, relative_area_tolerance, boundary_tolerance)):
        raise ValueError('Nonnegative finite tolerances required')
    requested = np.ones((h, w), bool) if query_mask is None else np.asarray(query_mask)
    if requested.shape != (h, w) or requested.dtype != np.bool_:
        raise ValueError('query_mask must be boolean and match the grid')
    requested = requested.ravel()
    pixel_area = float(np.prod(step))
    area_tol = relative_area_tolerance * pixel_area + uv_area_epsilon
    if not math.isfinite(pixel_area) or pixel_area <= 0 or not math.isfinite(area_tol) or area_tol >= pixel_area:
        raise ValueError('Pixel UV area must exceed a finite positive coverage tolerance')
    distance_tol = boundary_tolerance * max(1., float(step.max()))
    reason = np.where(requested, 0, NOT_REQUESTED).astype(np.uint8)
    area_out = np.zeros(h * w, float)
    pair_area_out = np.zeros(h * w, float)
    pieces_out = np.zeros(h * w, np.uint32)
    candidate_lists = [[] for _ in range(h * w)]
    pu, px = uv[faces], xyz[faces]
    da, db = pu[:, 1] - pu[:, 0], pu[:, 2] - pu[:, 0]
    signed2 = da[:, 0] * db[:, 1] - da[:, 1] * db[:, 0]
    area3 = np.linalg.norm(np.cross(px[:, 1] - px[:, 0], px[:, 2] - px[:, 0]), axis=1)
    if not np.isfinite(signed2).all() or not np.isfinite(area3).all():
        raise ValueError('Overflow in source area calculation')
    # Work only with faces whose bounding boxes intersect the requested lattice.
    lows, highs = pu.min(axis=1), pu.max(axis=1)
    domain_low = origin - step / 2
    domain_high = origin + step * np.array([w - .5, h - .5])
    if not np.isfinite(domain_low).all() or not np.isfinite(domain_high).all() or (domain_high <= domain_low).any():
        raise ValueError('UV domain overflowed or its extent is not represented')
    guard = 16 * np.finfo(float).eps * np.maximum(1., np.maximum(np.abs(domain_low), np.abs(domain_high)))
    relevant = np.flatnonzero((highs >= domain_low - guard).all(axis=1) & (lows <= domain_high + guard).all(axis=1))
    for face_id in relevant:
        # Round OUTWARD, allowing an extra row/column. An inward ceil/floor
        # can miss a touching face after subtraction/division roundoff.
        # Clip in floating UV first. Otherwise a finite, enormous crossing
        # face can overflow the int64 cast and silently disappear.
        local_low = np.maximum(lows[face_id], domain_low - guard)
        local_high = np.minimum(highs[face_id], domain_high + guard)
        lo = np.floor((local_low - origin) / step - .5).astype(int)
        hi = np.ceil((local_high - origin) / step + .5).astype(int)
        x0, y0 = max(0, lo[0]), max(0, lo[1])
        x1, y1 = min(w - 1, hi[0]), min(h - 1, hi[1])
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                index = yy * w + xx
                if requested[index]:
                    candidate_lists[index].append(int(face_id))
    for index in np.flatnonzero(requested):
        yy, xx = divmod(int(index), w)
        center = origin + step * [xx, yy]
        low, high = center - step / 2, center + step / 2
        rect = np.array([[low[0], low[1]], [high[0], low[1]],
                         [high[0], high[1]], [low[0], high[1]]])
        positives = []
        for face_id in candidate_lists[index]:
            clipped = _intersection(pu[face_id], rect)
            if not len(clipped):
                continue
            if abs(signed2[face_id]) <= uv_area_epsilon:
                reason[index] |= DEGENERATE
            elif signed2[face_id] < 0:
                reason[index] |= NEGATIVE
            elif area3[face_id] == 0:
                reason[index] |= ZERO_PHYSICAL_AREA
            else:
                area = _area(clipped)
                # Keep zero-area contacts too: an unrelated surface touching
                # only the closed pixel edge must not evade topology checks.
                positives.append((face_id, clipped))
                area_out[index] += area
        pieces_out[index] = len(positives)
        if not positives or area_out[index] < pixel_area - area_tol:
            reason[index] |= NO_COVERAGE
        if area_out[index] > pixel_area + area_tol:
            reason[index] |= OVERLAP
        for i, (face_a, poly_a) in enumerate(positives):
            for face_b, poly_b in positives[:i]:
                intersection = _intersection(poly_a, pu[face_b])
                if not len(intersection):
                    continue
                overlap_area = _area(intersection)
                pair_area_out[index] += overlap_area
                if overlap_area > area_tol:
                    reason[index] |= OVERLAP
                    continue
                shared = np.intersect1d(faces[face_a], faces[face_b])
                if len(shared) >= 2:
                    continue
                if len(shared) == 1 and np.linalg.norm(intersection - uv[shared[0]], axis=1).max() <= distance_tol:
                    continue
                reason[index] |= INCONSISTENT_BOUNDARY
        # Bonferroni lower bound: area(union) >= sum(single areas) -
        # sum(pair areas). Many individually sub-tolerance overlaps must not
        # compensate a larger hole in the summed-area check.
        if area_out[index] - pair_area_out[index] < pixel_area - area_tol:
            reason[index] |= NO_COVERAGE
    return {
        'valid': (reason == 0).reshape(h, w),
        'reason_bits': reason.reshape(h, w),
        'coverage_area': area_out.reshape(h, w),
        'pairwise_overlap_area': pair_area_out.reshape(h, w),
        'union_area_lower_bound': (area_out - pair_area_out).reshape(h, w),
        'positive_piece_count': pieces_out.reshape(h, w),
        'diagnostics': {
            'requested_pixels': int(requested.sum()), 'valid_pixels': int((reason == 0).sum()),
            'relevant_source_faces': int(len(relevant)), 'pixel_uv_area': pixel_area,
            'area_tolerance': area_tol, 'boundary_distance_tolerance': distance_tol,
            'coverage_contract': 'whole UV rectangle clipped against original faces; union-area Bonferroni lower bound within tolerance of pixel area, no above-tolerance pair overlap, no bad-face intersection, consistent shared source boundaries',
            'limitations': 'Float64 with declared tolerances; chart coverage only, not physical sheet identity or nonintersecting normal rays',
        },
    }
