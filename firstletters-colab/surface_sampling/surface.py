"""Sample an original triangular surface at arbitrary regular UV query grids.

This extends the frozen ``metric_flatten/topology.py`` point-support rules to
noninteger, anisotropic grids.  All containment tests, area epsilons and distance
tolerances remain in the ORIGINAL UV coordinates.  There is no triangulation,
hole filling, regular-grid interpolation, normal smoothing or orientation fix.

``origin_uv`` is the first query centre, not a pixel corner.  ``sample_grid``
certifies only those points; it does not establish support between them.  Use
``footprint_support`` for the separate whole-pixel support certificate.  Neither
operation establishes that a source mesh follows the correct physical sheet.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np


# Import the frozen implementation, rather than changing it or maintaining a
# second definition of its barycentric-carrier and degenerate-footprint rules.
_spec = importlib.util.spec_from_file_location(
    "_original_surface_topology",
    Path(__file__).resolve().parents[1] / "metric_flatten" / "topology.py",
)
assert _spec is not None and _spec.loader is not None
_topology = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_topology)

NOT_REQUESTED = _topology.NOT_REQUESTED
NO_SOURCE_COVERAGE = _topology.NO_SOURCE_COVERAGE
NEGATIVE_UV_COVERAGE = _topology.NEGATIVE_UV_COVERAGE
DEGENERATE_UV_COVERAGE = _topology.DEGENERATE_UV_COVERAGE
AMBIGUOUS_OVERLAP = _topology.AMBIGUOUS_OVERLAP
INCONSISTENT_SHARED_BOUNDARY = _topology.INCONSISTENT_SHARED_BOUNDARY
DEGENERATE_XYZ_COVERAGE = np.uint8(64)

REASON_NAMES = dict(_topology.REASON_NAMES)
REASON_NAMES[int(DEGENERATE_XYZ_COVERAGE)] = "degenerate_xyz_coverage"


def _pair(name, value, *, positive=False):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain two finite coordinates")
    if positive and not (result > 0).all():
        raise ValueError(f"{name} must contain two strictly positive values")
    return result


def _shape(value):
    try:
        good = len(value) == 2 and all(
            not isinstance(v, (bool, np.bool_)) and int(v) == v and v > 0
            for v in value
        )
    except (TypeError, ValueError, OverflowError):
        good = False
    if not good:
        raise ValueError("shape_yx must contain two positive integer dimensions")
    return tuple(map(int, value))


class OriginalSurface:
    """Immutable source geometry with bounded-block original-face sampling.

    ``uv`` is N×2, ``xyz`` is N×3 and ``triangles`` is an integer F×3 array.
    Positive signed UV area is the declared orientation.  Rejected faces stay
    in the population and poison every query in their footprint, including
    queries also covered by an acceptable face.  The six existing rejection
    bits are unchanged; bit 64 adds zero physical triangle area, for which a
    normal is undefined.  No nonzero physical-area threshold is imposed.

    Normals are the normalized original physical triangle cross products.
    Jacobians are d(XYZ)/d(UV), in the caller's XYZ units per original UV unit.
    Both are constant within a face.  Shared boundaries use the smallest
    ORIGINAL face ID, including when the two physical normals differ.

    The default 1.5 GB allocation guard covers output and bounded bookkeeping,
    not all process memory.  One million requested grid locations allocate
    about 136 MB of output arrays, in addition to source geometry and working
    blocks.  Bounding boxes are filtered once per grid; cost still depends on
    the number and bounding-box sizes of intersecting original faces.
    """

    def __init__(
        self,
        uv,
        xyz,
        triangles,
        *,
        xyz_tolerance=1e-6,
        barycentric_tolerance=1e-9,
        uv_area_epsilon=1e-12,
        degenerate_distance_tolerance=1e-9,
        max_working_points=65536,
        max_output_bytes=1_500_000_000,
    ):
        uv, xyz, triangles = _topology._geometry(uv, xyz, triangles)
        self.uv, self.xyz, self.triangles = (a.copy() for a in (uv, xyz, triangles))
        for array in (self.uv, self.xyz, self.triangles):
            array.setflags(write=False)
        nonnegative = _topology._nonnegative_finite
        self.xyz_tolerance = nonnegative("xyz_tolerance", xyz_tolerance)
        self.barycentric_tolerance = nonnegative(
            "barycentric_tolerance", barycentric_tolerance)
        if self.barycentric_tolerance >= 1 / 3:
            raise ValueError("barycentric_tolerance must be less than one third")
        self.uv_area_epsilon = nonnegative("uv_area_epsilon", uv_area_epsilon)
        self.degenerate_distance_tolerance = nonnegative(
            "degenerate_distance_tolerance", degenerate_distance_tolerance)
        if (isinstance(max_working_points, (bool, np.bool_))
                or not math.isfinite(float(max_working_points))
                or int(max_working_points) != max_working_points
                or max_working_points <= 0):
            raise ValueError("max_working_points must be a positive integer")
        self.max_working_points = int(max_working_points)
        if not math.isfinite(float(max_output_bytes)) or max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive and finite")
        self.max_output_bytes = float(max_output_bytes)

        pu, px = self.uv[self.triangles], self.xyz[self.triangles]
        self._anchor = pu[:, 0].copy()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            self._a, self._b = pu[:, 1] - pu[:, 0], pu[:, 2] - pu[:, 0]
            e, f = px[:, 1] - px[:, 0], px[:, 2] - px[:, 0]
            self._det = self._a[:, 0] * self._b[:, 1] - self._a[:, 1] * self._b[:, 0]
            physical_cross = np.cross(e, f)
            physical_double_area = np.linalg.norm(physical_cross, axis=1)
        if (not np.isfinite(self._det).all()
                or not np.isfinite(physical_double_area).all()):
            raise ValueError("source geometry overflows finite area arithmetic")
        self._degenerate_uv = np.abs(self._det) <= self.uv_area_epsilon
        self._negative_uv = self._det < -self.uv_area_epsilon
        self._degenerate_xyz = physical_double_area == 0
        self._normals = np.full((len(self.triangles), 3), np.nan)
        physical_ok = ~self._degenerate_xyz
        self._normals[physical_ok] = (
            physical_cross[physical_ok] / physical_double_area[physical_ok, None])
        self._jacobian = np.full((len(self.triangles), 3, 2), np.nan)
        nondegenerate = ~self._degenerate_uv
        if nondegenerate.any():
            a, b = self._a[nondegenerate], self._b[nondegenerate]
            ee, ff, det = e[nondegenerate], f[nondegenerate], self._det[nondegenerate]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                du = (ee * b[:, 1, None] - ff * a[:, 1, None]) / det[:, None]
                dv = (ff * a[:, 0, None] - ee * b[:, 0, None]) / det[:, None]
                jacobian = np.stack((du, dv), axis=2)
            if not np.isfinite(jacobian).all():
                raise ValueError("source geometry overflows finite Jacobian arithmetic")
            self._jacobian[nondegenerate] = jacobian

        with np.errstate(over="ignore", invalid="ignore"):
            pad = np.where(
                self._degenerate_uv,
                self.degenerate_distance_tolerance,
                self.barycentric_tolerance
                * (np.abs(self._a) + np.abs(self._b)).max(axis=1) * 2.0,
            )
            self._bbox_min = pu.min(axis=1) - pad[:, None]
            self._bbox_max = pu.max(axis=1) + pad[:, None]
        if (not np.isfinite(self._bbox_min).all()
                or not np.isfinite(self._bbox_max).all()):
            raise ValueError("source geometry overflows finite bounding-box arithmetic")
        # Outward rounding only broadens candidate enumeration, not acceptance.
        self._bbox_min = np.nextafter(self._bbox_min, -np.inf)
        self._bbox_max = np.nextafter(self._bbox_max, np.inf)
        self._source_counts = {
            "source_triangles": len(self.triangles),
            "positive_uv_triangles": int((self._det > self.uv_area_epsilon).sum()),
            "negative_uv_triangles": int(self._negative_uv.sum()),
            "degenerate_uv_triangles": int(self._degenerate_uv.sum()),
            "zero_3d_area_triangles": int(self._degenerate_xyz.sum()),
        }

    def footprint_support(self, origin_uv, step_uv, shape_yx, query_mask=None):
        """Request the separate continuous rectangle-support certificate.

        The sibling ``footprint`` module owns clipping/overlap certification;
        this method never substitutes a finite set of point queries for it.
        """
        spec = importlib.util.spec_from_file_location(
            "_original_surface_footprint", Path(__file__).with_name("footprint.py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.certify(
            self.uv, self.xyz, self.triangles,
            origin_uv, step_uv, shape_yx, query_mask=query_mask,
        )

    def sample_grid(self, origin_uv, step_uv, shape_yx, query_mask=None):
        """Return XYZ, face normals, Jacobians and original-face provenance.

        Queries are ``origin_uv + (column * du, row * dv)``.  All returned
        arrays have leading shape ``shape_yx``.  Invalid vectors, Jacobians and
        weights are NaN; invalid source IDs are -1.  Coverage counts include
        rejected faces.  A false query mask is NOT_REQUESTED with zero coverage.
        """
        origin = _pair("origin_uv", origin_uv)
        step = _pair("step_uv", step_uv, positive=True)
        h, w = _shape(shape_yx)
        count = h * w
        if count * 160 + len(self.triangles) * 8 > self.max_output_bytes:
            raise MemoryError("requested raster exceeds the declared allocation allowance")
        with np.errstate(over="ignore", invalid="ignore"):
            last = origin + step * (w - 1, h - 1)
        if not np.isfinite(last).all():
            raise ValueError("query extent overflows finite UV arithmetic")
        requested = (np.ones((h, w), dtype=bool) if query_mask is None
                     else np.asarray(query_mask, dtype=bool))
        if requested.shape != (h, w):
            raise ValueError("query_mask must match shape_yx")
        requested = requested.ravel()
        output_xyz = np.full((count, 3), np.nan)
        weights_out = np.full((count, 3), np.nan)
        owner = np.full(count, -1, dtype=np.int64)
        coverage = np.zeros(count, dtype=np.uint32)
        ambiguity = np.zeros(count, dtype=bool)
        reasons = np.where(requested, 0, NOT_REQUESTED).astype(np.uint8)
        face_coverage = np.zeros(len(self.triangles), dtype=np.int64)
        candidate_ids = np.flatnonzero(
            ((self._bbox_max >= origin) & (self._bbox_min <= last)).all(axis=1))
        if not requested.any():
            candidate_ids = candidate_ids[:0]

        # flatnonzero preserves ascending GLOBAL source IDs after bbox cropping.
        for face_id in candidate_ids:
            vertices = self.triangles[face_id]
            tri_uv, tri_xyz = self.uv[vertices], self.xyz[vertices]
            a, b, det = self._a[face_id], self._b[face_id], self._det[face_id]
            degenerate = bool(self._degenerate_uv[face_id])
            # Clip in original UV before division, bounding the quotient even
            # when a remote vertex is many grid widths away.  One guard index
            # on each side prevents a floating division from omitting a boundary.
            low = (np.maximum(self._bbox_min[face_id], origin) - origin) / step
            high = (np.minimum(self._bbox_max[face_id], last) - origin) / step
            x0, x1 = max(0, math.ceil(low[0]) - 1), min(w - 1, math.floor(high[0]) + 1)
            y0, y1 = max(0, math.ceil(low[1]) - 1), min(h - 1, math.floor(high[1]) + 1)
            width, total = x1 - x0 + 1, (x1 - x0 + 1) * (y1 - y0 + 1)
            for offset in range(0, total, self.max_working_points):
                local = np.arange(offset, min(total, offset + self.max_working_points), dtype=np.int64)
                yy, xx = y0 + local // width, x0 + local % width
                indices = yy * w + xx
                take = requested[indices]
                if not take.any():
                    continue
                indices = indices[take]
                points = np.column_stack((xx[take], yy[take])) * step + origin
                if degenerate:
                    inside = _topology._on_degenerate(
                        points, tri_uv, self.degenerate_distance_tolerance)
                    weights = None
                else:
                    relative = points - self._anchor[face_id]
                    second = (relative[:, 0] * b[1] - relative[:, 1] * b[0]) / det
                    third = (a[0] * relative[:, 1] - a[1] * relative[:, 0]) / det
                    weights = np.column_stack((1.0 - second - third, second, third))
                    inside = (weights >= -self.barycentric_tolerance).all(axis=1)
                indices = indices[inside]
                if not len(indices):
                    continue
                coverage[indices] += 1
                face_coverage[face_id] += len(indices)
                poison = (int(DEGENERATE_UV_COVERAGE) if degenerate else
                          int(NEGATIVE_UV_COVERAGE) if self._negative_uv[face_id] else 0)
                if self._degenerate_xyz[face_id]:
                    poison |= int(DEGENERATE_XYZ_COVERAGE)
                if poison:
                    reasons[indices] |= np.uint8(poison)
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
                old_boundary = (old_weights <= self.barycentric_tolerance).any(axis=1)
                new_boundary = (new_weights <= self.barycentric_tolerance).any(axis=1)
                old_carrier = _topology._carrier(
                    self.triangles[old_ids], old_weights, self.barycentric_tolerance)
                new_carrier = _topology._carrier(
                    np.broadcast_to(vertices, new_weights.shape),
                    new_weights, self.barycentric_tolerance)
                shared = old_boundary & new_boundary & (old_carrier == new_carrier).all(axis=1)
                consistent = (np.linalg.norm(
                    output_xyz[repeated_indices] - values[repeated], axis=1)
                    <= self.xyz_tolerance)
                rejected = repeated_indices[~shared | ~consistent]
                ambiguity[rejected] = True
                reasons[rejected] |= AMBIGUOUS_OVERLAP
                reasons[repeated_indices[shared & ~consistent]] |= INCONSISTENT_SHARED_BOUNDARY

        reasons[requested & (coverage == 0)] |= NO_SOURCE_COVERAGE
        valid = requested & (owner >= 0) & (reasons == 0)
        output_normals = np.full((count, 3), np.nan)
        output_jacobian = np.full((count, 3, 2), np.nan)
        for offset in range(0, count, self.max_working_points):
            indices = np.flatnonzero(valid[offset:offset + self.max_working_points]) + offset
            output_normals[indices] = self._normals[owner[indices]]
            output_jacobian[indices] = self._jacobian[owner[indices]]
        owner[~valid] = -1
        output_xyz[~valid], weights_out[~valid] = np.nan, np.nan
        counts = dict(self._source_counts)
        counts.update({
            "requested_pixels": int(requested.sum()),
            "valid_pixels": int(valid.sum()),
            "candidate_source_triangles": len(candidate_ids),
            "source_triangles_with_requested_grid_coverage": int((face_coverage > 0).sum()),
            "ambiguous_pixels": int(ambiguity.sum()),
            "negative_uv_covered_pixels": int(((reasons & NEGATIVE_UV_COVERAGE) != 0).sum()),
            "degenerate_uv_covered_pixels": int(((reasons & DEGENERATE_UV_COVERAGE) != 0).sum()),
            "degenerate_xyz_covered_pixels": int(((reasons & DEGENERATE_XYZ_COVERAGE) != 0).sum()),
        })
        return {
            "xyz": output_xyz.reshape(h, w, 3),
            "normals": output_normals.reshape(h, w, 3),
            "jacobian": output_jacobian.reshape(h, w, 3, 2),
            "valid": valid.reshape(h, w),
            "source_triangle_id": owner.reshape(h, w),
            "barycentric_weights": weights_out.reshape(h, w, 3),
            "reason_bits": reasons.reshape(h, w),
            "coverage_count": coverage.reshape(h, w),
            "ambiguity_mask": ambiguity.reshape(h, w),
            "source_triangle_covered_query_count": face_coverage,
            "counts": counts,
            "reason_names": dict(REASON_NAMES),
            "origin_uv": origin.copy(),
            "step_uv": step.copy(),
        }
