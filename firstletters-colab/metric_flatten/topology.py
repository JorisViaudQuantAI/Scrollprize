"""Inverse rasterization using only the original triangular surface.

No triangulation is constructed here.  Every accepted pixel is evaluated in an
original input face.  Positive UV orientation is the declared convention;
negative and degenerate UV faces remain in the source population and poison
their covered query pixels, including pixels also covered by a positive face.

Queries are integer pixel centres: ``(x, y) = origin_xy + (column, row)``.
Invalid XYZ/weights are NaN and invalid source-face IDs are -1.  The caller is
responsible for a file format's invalid sentinel, physical units, and resource
budget.  A valid raster does not independently establish correct sheet geometry.
Pointwise support checks do not certify cells of a regular grid reconstructed
from the samples; downstream interpolation must also respect source support.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


NOT_REQUESTED = np.uint8(1)
NO_SOURCE_COVERAGE = np.uint8(2)
NEGATIVE_UV_COVERAGE = np.uint8(4)
DEGENERATE_UV_COVERAGE = np.uint8(8)
AMBIGUOUS_OVERLAP = np.uint8(16)
INCONSISTENT_SHARED_BOUNDARY = np.uint8(32)

REASON_NAMES = {
    int(NOT_REQUESTED): "not_requested",
    int(NO_SOURCE_COVERAGE): "no_source_coverage",
    int(NEGATIVE_UV_COVERAGE): "negative_uv_coverage",
    int(DEGENERATE_UV_COVERAGE): "degenerate_uv_coverage",
    int(AMBIGUOUS_OVERLAP): "ambiguous_overlap",
    int(INCONSISTENT_SHARED_BOUNDARY): "inconsistent_shared_boundary",
}


def _geometry(uv, xyz, triangles):
    uv = np.asarray(uv, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    faces = np.asarray(triangles)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")
    if xyz.shape != (uv.shape[0], 3):
        raise ValueError("xyz must have shape (N, 3), matching uv")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("triangles must be an integer array of shape (F, 3)")
    if not np.isfinite(uv).all() or not np.isfinite(xyz).all():
        raise ValueError("nonfinite source geometry is not silently discarded")
    if faces.size and (faces.min() < 0 or faces.max() >= uv.shape[0]):
        raise ValueError("source face index outside the vertex array")
    if len(faces) > np.iinfo(np.uint32).max:
        raise ValueError("source face count exceeds coverage-counter capacity")
    return uv, xyz, faces.astype(np.int64, copy=False)


def _nonnegative_finite(name, value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def triangle_metrics(uv, xyz, triangles, *, uv_area_epsilon=1e-12):
    """Return F-length metrics; orientation-rejected faces are never dropped.

    ``uv_area_epsilon`` applies to the absolute *twice* signed UV area.  Singular
    values are descending and describe d(XYZ)/d(UV), hence use the caller's
    XYZ-units per UV-unit.  They are computed for either nondegenerate UV
    orientation.  A negative face therefore has meaningful metric values but
    remains separately rejected by the rasterizer.  Degenerate UV faces retain
    their signed UV area and 3D area; their Jacobian metrics are NaN.
    """
    uv, xyz, faces = _geometry(uv, xyz, triangles)
    epsilon = _nonnegative_finite("uv_area_epsilon", uv_area_epsilon)
    pu, px = uv[faces], xyz[faces]
    a, b = pu[:, 1] - pu[:, 0], pu[:, 2] - pu[:, 0]
    e, f = px[:, 1] - px[:, 0], px[:, 2] - px[:, 0]
    with np.errstate(over="ignore", invalid="ignore"):
        determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        area3d = 0.5 * np.linalg.norm(np.cross(e, f), axis=1)
    if not np.isfinite(determinant).all() or not np.isfinite(area3d).all():
        raise ValueError("source geometry overflows finite area arithmetic")
    nondegenerate = np.abs(determinant) > epsilon
    singular = np.full((len(faces), 2), np.nan, dtype=np.float64)
    ratio = np.full(len(faces), np.nan, dtype=np.float64)
    if nondegenerate.any():
        aa, bb = a[nondegenerate], b[nondegenerate]
        ee, ff, det = e[nondegenerate], f[nondegenerate], determinant[nondegenerate]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            du = (ee * bb[:, 1, None] - ff * aa[:, 1, None]) / det[:, None]
            dv = (ff * aa[:, 0, None] - ee * bb[:, 0, None]) / det[:, None]
            jacobian = np.stack((du, dv), axis=2)
        if not np.isfinite(jacobian).all():
            raise ValueError("source geometry overflows finite Jacobian arithmetic")
        singular[nondegenerate] = np.linalg.svd(jacobian, compute_uv=False)
        ratio[nondegenerate] = area3d[nondegenerate] / (0.5 * np.abs(det))
    positive, negative = determinant > epsilon, determinant < -epsilon
    return {
        "signed_uv_area": determinant * 0.5,
        "source_3d_area": area3d,
        "singular_values": singular,
        "area_ratio_3d_to_abs_uv": ratio,
        "positive_uv": positive,
        "negative_uv": negative,
        "degenerate_uv": ~nondegenerate,
        "counts": {
            "source_triangles": len(faces),
            "positive_uv_triangles": int(positive.sum()),
            "negative_uv_triangles": int(negative.sum()),
            "degenerate_uv_triangles": int((~nondegenerate).sum()),
            "zero_3d_area_triangles": int((area3d == 0).sum()),
        },
    }


def _face_components(faces, vertex_count):
    """Components of original vertex/face incidence, including rejected faces."""
    if not len(faces):
        return np.empty(0, dtype=np.int64), 0
    rows = faces[:, [0, 1, 2]].ravel()
    columns = faces[:, [1, 2, 0]].ravel()
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, columns)),
                       shape=(vertex_count, vertex_count)).tocsr()
    _, vertex_labels = connected_components(graph, directed=False)
    # Unreferenced vertices are not source surface components.
    _, face_labels = np.unique(vertex_labels[faces[:, 0]], return_inverse=True)
    return face_labels.astype(np.int64), int(face_labels.max()) + 1


def _carrier(face_vertices, weights, tolerance):
    """Canonical vertex IDs of an edge or vertex carrying a boundary query."""
    return np.sort(np.where(weights > tolerance, face_vertices, -1), axis=1)


def _on_degenerate(points, tri_uv, tolerance):
    """Full footprint of an epsilon-degenerate face, including its boundary.

    A nonzero-area face can fall below the declared epsilon: its interior must
    still poison coverage. Use edge signs without dividing by that tiny area.
    An exactly collinear face has only the union of its closed edge segments.
    """
    distance2 = np.full(len(points), np.inf)
    edge_signs = []
    for first, second in ((0, 1), (1, 2), (2, 0)):
        start, vector = tri_uv[first], tri_uv[second] - tri_uv[first]
        relative = points - start
        edge_signs.append(vector[0] * relative[:, 1] - vector[1] * relative[:, 0])
        length2 = float(vector @ vector)
        if length2 == 0:
            delta = relative
        else:
            t = np.clip((relative @ vector) / length2, 0.0, 1.0)
            delta = points - (start + t[:, None] * vector)
        distance2 = np.minimum(distance2, np.einsum("ij,ij->i", delta, delta))
    covered = distance2 <= tolerance * tolerance
    a, b = tri_uv[1] - tri_uv[0], tri_uv[2] - tri_uv[0]
    determinant = a[0] * b[1] - a[1] * b[0]
    if determinant != 0:
        signs = np.column_stack(edge_signs)
        covered |= (signs >= 0).all(axis=1) if determinant > 0 else (signs <= 0).all(axis=1)
    return covered


def rasterize_triangles(
    uv,
    xyz,
    triangles,
    shape: Sequence[int],
    *,
    query_mask=None,
    origin_xy=(0.0, 0.0),
    xyz_tolerance=1e-6,
    barycentric_tolerance=1e-9,
    uv_area_epsilon=1e-12,
    degenerate_distance_tolerance=1e-9,
    max_working_points=65536,
    max_output_bytes=1_500_000_000,
):
    """Inverse-map the requested integer grid through original source faces.

    Shared edges/vertices are accepted only when both barycentric carriers have
    the same original vertex IDs and interpolated XYZ differ by at most
    ``xyz_tolerance`` in Euclidean distance.  Coincident coordinates with
    distinct vertex IDs do not create adjacency.  Any interior or nonadjacent
    positive overlap is ambiguous, even when XYZ happen to coincide.

    ``reason_bits`` records every rejection; bits are documented in
    ``REASON_NAMES``. ``ambiguity_mask`` denotes incompatible positive-face
    overlaps. Rejected-orientation coverage has its own reason bits and always
    invalidates the pixel. ``coverage_count`` includes all positive, negative
    and degenerate source-face coverage at requested grid points. Per-face and
    per-component incidence/output counts expose components absent from the
    output; they are not implicitly removed from geometry statistics.

    Working point blocks are bounded independently of a triangle's bounding
    box. The output-size estimate is a conservative allocation guard, not a
    substitute for a caller's whole-process RSS/time budget.
    """
    uv, xyz, faces = _geometry(uv, xyz, triangles)
    if len(shape) != 2 or any(isinstance(v, (bool, np.bool_)) or int(v) != v or v <= 0 for v in shape):
        raise ValueError("shape must contain two positive integer dimensions")
    h, w = map(int, shape)
    if max_working_points <= 0 or int(max_working_points) != max_working_points:
        raise ValueError("max_working_points must be a positive integer")
    max_working_points = int(max_working_points)
    if not math.isfinite(float(max_output_bytes)) or max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be positive and finite")
    if h * w * 88 + len(faces) * 128 > max_output_bytes:
        raise MemoryError("requested raster exceeds the declared allocation allowance")
    xyz_tolerance = _nonnegative_finite("xyz_tolerance", xyz_tolerance)
    barycentric_tolerance = _nonnegative_finite("barycentric_tolerance", barycentric_tolerance)
    if barycentric_tolerance >= 1 / 3:
        raise ValueError("barycentric_tolerance must be less than one third")
    degenerate_distance_tolerance = _nonnegative_finite(
        "degenerate_distance_tolerance", degenerate_distance_tolerance)
    origin = np.asarray(origin_xy, dtype=np.float64)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("origin_xy must contain two finite coordinates")
    requested = np.ones((h, w), dtype=bool) if query_mask is None else np.asarray(query_mask, dtype=bool)
    if requested.shape != (h, w):
        raise ValueError("query_mask must match shape")
    requested = requested.ravel()

    metrics = triangle_metrics(uv, xyz, faces, uv_area_epsilon=uv_area_epsilon)
    component_ids, component_count = _face_components(faces, len(uv))
    count = h * w
    output_xyz = np.full((count, 3), np.nan, dtype=np.float64)
    weights_out = np.full((count, 3), np.nan, dtype=np.float64)
    owner = np.full(count, -1, dtype=np.int64)
    coverage = np.zeros(count, dtype=np.uint32)
    ambiguity = np.zeros(count, dtype=bool)
    reasons = np.where(requested, 0, NOT_REQUESTED).astype(np.uint8)
    face_coverage = np.zeros(len(faces), dtype=np.int64)

    for face_id, vertex_ids in enumerate(faces):
        tri_uv, tri_xyz = uv[vertex_ids], xyz[vertex_ids]
        a, b = tri_uv[1] - tri_uv[0], tri_uv[2] - tri_uv[0]
        determinant = 2.0 * metrics["signed_uv_area"][face_id]
        degenerate = bool(metrics["degenerate_uv"][face_id])
        negative = bool(metrics["negative_uv"][face_id])
        # Barycentric tolerance can extend a triangle by a scale-dependent
        # amount; this expansion prevents silently missing those boundary hits.
        pad = (degenerate_distance_tolerance if degenerate else
               barycentric_tolerance * (np.abs(a) + np.abs(b)).max() * 2.0)
        low, high = tri_uv.min(axis=0) - origin - pad, tri_uv.max(axis=0) - origin + pad
        x0, x1 = max(0, math.ceil(low[0])), min(w - 1, math.floor(high[0]))
        y0, y1 = max(0, math.ceil(low[1])), min(h - 1, math.floor(high[1]))
        if x0 > x1 or y0 > y1:
            continue
        width, total = x1 - x0 + 1, (x1 - x0 + 1) * (y1 - y0 + 1)
        for offset in range(0, total, max_working_points):
            local = np.arange(offset, min(total, offset + max_working_points), dtype=np.int64)
            yy, xx = y0 + local // width, x0 + local % width
            indices = yy * w + xx
            take = requested[indices]
            if not take.any():
                continue
            indices = indices[take]
            points = np.column_stack((xx[take], yy[take])).astype(np.float64) + origin
            if degenerate:
                inside = _on_degenerate(points, tri_uv, degenerate_distance_tolerance)
                weights = None
            else:
                relative = points - tri_uv[0]
                second = (relative[:, 0] * b[1] - relative[:, 1] * b[0]) / determinant
                third = (a[0] * relative[:, 1] - a[1] * relative[:, 0]) / determinant
                weights = np.column_stack((1.0 - second - third, second, third))
                inside = (weights >= -barycentric_tolerance).all(axis=1)
            indices = indices[inside]
            if not len(indices):
                continue
            coverage[indices] += 1
            face_coverage[face_id] += len(indices)
            if degenerate or negative:
                reasons[indices] |= DEGENERATE_UV_COVERAGE if degenerate else NEGATIVE_UV_COVERAGE
                continue

            weights = np.maximum(weights[inside], 0.0)
            weights /= weights.sum(axis=1, keepdims=True)
            values = weights @ tri_xyz
            if not np.isfinite(values).all():
                raise ValueError("interpolated XYZ are nonfinite")
            free = owner[indices] < 0
            first = indices[free]
            output_xyz[first], weights_out[first], owner[first] = values[free], weights[free], face_id

            repeated = ~free & ~ambiguity[indices]
            repeated_indices = indices[repeated]
            if not len(repeated_indices):
                continue
            old_ids = owner[repeated_indices]
            old_weights, new_weights = weights_out[repeated_indices], weights[repeated]
            old_boundary = (old_weights <= barycentric_tolerance).any(axis=1)
            new_boundary = (new_weights <= barycentric_tolerance).any(axis=1)
            old_carrier = _carrier(faces[old_ids], old_weights, barycentric_tolerance)
            new_carrier = _carrier(np.broadcast_to(vertex_ids, new_weights.shape),
                                   new_weights, barycentric_tolerance)
            shared = old_boundary & new_boundary & (old_carrier == new_carrier).all(axis=1)
            consistent = np.linalg.norm(output_xyz[repeated_indices] - values[repeated], axis=1) <= xyz_tolerance
            bad = ~shared | ~consistent
            rejected = repeated_indices[bad]
            ambiguity[rejected] = True
            reasons[rejected] |= AMBIGUOUS_OVERLAP
            reasons[repeated_indices[shared & ~consistent]] |= INCONSISTENT_SHARED_BOUNDARY

    reasons[requested & (coverage == 0)] |= NO_SOURCE_COVERAGE
    valid = requested & (owner >= 0) & (reasons == 0)
    component_output = np.bincount(component_ids[owner[valid]], minlength=component_count)
    component_incidence = np.bincount(component_ids, weights=face_coverage,
                                     minlength=component_count).astype(np.int64)
    owner[~valid] = -1
    output_xyz[~valid], weights_out[~valid] = np.nan, np.nan
    stats = dict(metrics["counts"])
    stats.update({
        "requested_pixels": int(requested.sum()),
        "valid_pixels": int(valid.sum()),
        "ambiguous_pixels": int(ambiguity.sum()),
        "negative_uv_covered_pixels": int(((reasons & NEGATIVE_UV_COVERAGE) != 0).sum()),
        "degenerate_uv_covered_pixels": int(((reasons & DEGENERATE_UV_COVERAGE) != 0).sum()),
        "source_components": component_count,
        "source_components_without_output": int((component_output == 0).sum()),
        "source_triangles_without_requested_grid_coverage": int((face_coverage == 0).sum()),
    })
    return {
        "xyz": output_xyz.reshape(h, w, 3),
        "valid": valid.reshape(h, w),
        "source_triangle_id": owner.reshape(h, w),
        "barycentric_weights": weights_out.reshape(h, w, 3),
        "ambiguity_mask": ambiguity.reshape(h, w),
        "reason_bits": reasons.reshape(h, w),
        "coverage_count": coverage.reshape(h, w),
        "source_triangle_covered_query_count": face_coverage,
        "source_component_id": component_ids,
        "source_component_query_incidence_count": component_incidence,
        "source_component_output_pixel_count": component_output,
        "triangle_metrics": metrics,
        "counts": stats,
        "reason_names": dict(REASON_NAMES),
        "origin_xy": origin.copy(),
    }
