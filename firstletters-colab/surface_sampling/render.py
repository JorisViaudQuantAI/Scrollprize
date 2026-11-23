"""Conservative CT rendering through an original-triangle surface.

``origin_uv`` names the centre of the first output pixel. Each output pixel is
the rectangle centred on that point with side lengths ``abs(step_uv)``. An
N-by-N midpoint rule maps that UV footprint through ``surface.sample_grid``
and averages CT values at each requested normal depth independently. N=1 is
the central point rule; N=4 and N=8 use 16 and 64 subqueries respectively.

This is metric-aware footprint quadrature inspired by graphics: the physical
footprint follows the source surface. It is *uniform UV-area* quadrature, not
physical-area Jacobian weighting and not the EWA Gaussian-kernel algorithm.
No CT reconstruction, enhancement, depth smoothing, or learned model is used.

Required surface protocol::

    surface.footprint_support(origin_uv, step_uv, shape_yx,
                              query_mask=None) -> {"valid": bool[H, W], ...}
    surface.sample_grid(origin_uv, step_uv, shape_yx,
                        query_mask=None) -> {"xyz": float[H, W, 3],
                                             "normals": float[H, W, 3],
                                             "valid": bool[H, W], ...}

``footprint_support`` must certify continuous coverage of the *whole* UV
rectangle, including holes between quadrature points and ambiguous overlaps.
The same footprint criterion is required for all three quadrature arms.
Normals must be unit length and coordinates/depths are in native CT voxels.
Volume axes are Z,Y,X; surface coordinates and volume origins are X,Y,Z.

A pixel is retained only when its whole UV footprint is supported, every
subquery has valid geometry, and all eight trilinear vertices exist at every
subquery and every requested depth. At an integer coordinate on the final
voxel, the absent +1 vertex makes the query invalid even if its weight would
be zero. There is no coordinate clamping or partial-footprint renormalisation.
Invalid pixels are zero in *every* depth, and must be excluded with the
returned explicit mask; zero is not evidence of blank papyrus.

The input volume must be finite float32/float64. Arithmetic and sums use
float64; each retained output value is cast to float32 exactly once. Values
outside the finite float32 range raise rather than silently become infinity.
Tiles contain at most 65,536 subqueries (524,288 vertex reads per trilinear
evaluation), below one million. Apart from the caller's input and returned
stack, workspace is bounded independently of output dimensions and depth
count. CT bounds are certified at the quadrature points, not continuously
between them; neither normal-tube injectivity nor sheet identity is certified.
"""

from __future__ import annotations

from collections.abc import Mapping
import operator

import numpy as np


_MAX_TILE_QUERIES = 65_536
_MAX_SOURCE_READS_PER_CALL = 1_000_000
_NORMAL_LENGTH_ATOL = 1e-5


def _vector(value, size, name):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return vector


def _shape(value):
    try:
        values = tuple(value)
        if len(values) != 2 or any(isinstance(v, (bool, np.bool_)) for v in values):
            raise ValueError
        height, width = (operator.index(v) for v in values)
    except (TypeError, ValueError):
        raise ValueError("shape_yx must contain two positive integers") from None
    if height <= 0 or width <= 0:
        raise ValueError("shape_yx must contain two positive integers")
    return height, width


def _finite_volume(volume):
    volume = np.asarray(volume)
    if volume.ndim != 3 or any(size < 2 for size in volume.shape):
        raise ValueError("volume_zyx must be a 3-D array with each dimension >= 2")
    if volume.dtype.kind != "f" or volume.dtype.itemsize not in (4, 8):
        raise ValueError("volume_zyx must have float32 or float64 dtype")
    # Bounded temporary buffers also support a non-contiguous caller view.
    chunks = np.nditer(
        volume,
        flags=["external_loop", "buffered"],
        op_flags=["readonly"],
        buffersize=_MAX_TILE_QUERIES,
    )
    if any(not np.isfinite(chunk).all() for chunk in chunks):
        raise ValueError("volume_zyx must be finite; missing CT values are not filled")
    return volume


def _result_array(result, key, shape, *, boolean=False):
    if not isinstance(result, Mapping) or key not in result:
        raise ValueError(f"surface result must contain {key!r}")
    array = np.asarray(result[key])
    if array.shape != shape:
        raise ValueError(f"surface {key} must have shape {shape}, got {array.shape}")
    if boolean:
        if array.dtype.kind != "b":
            raise ValueError(f"surface {key} must have boolean dtype")
        return array
    if array.dtype.kind not in "fiu":
        raise ValueError(f"surface {key} must be real numeric")
    return np.asarray(array, dtype=np.float64)


def _inside_trilinear(coords_xyz, volume_shape_xyz):
    return (
        np.isfinite(coords_xyz).all(axis=-1)
        & (coords_xyz >= 0).all(axis=-1)
        & (coords_xyz < volume_shape_xyz - 1).all(axis=-1)
    )


def _trilinear_valid(volume_zyx, coords_xyz):
    """Interpolate only fully supported points; refuse any invalid query.

    This helper deliberately has no padding, clipping, or missing-value mode.
    The renderer checks bounds before calling it; this guard is independent
    protection against accidentally passing an unsupported point later.
    """
    coords_xyz = np.asarray(coords_xyz, dtype=np.float64)
    if coords_xyz.ndim != 2 or coords_xyz.shape[1] != 3:
        raise ValueError("trilinear coordinates must have shape (queries, 3)")
    if coords_xyz.shape[0] * 8 > _MAX_SOURCE_READS_PER_CALL:
        raise ValueError("trilinear call exceeds the source-read budget")
    if not _inside_trilinear(coords_xyz, np.asarray(volume_zyx.shape[::-1])).all():
        raise ValueError("trilinear query lacks all eight source vertices")
    lower = np.floor(coords_xyz).astype(np.intp)
    fraction = coords_xyz - lower
    values = np.zeros(coords_xyz.shape[0], dtype=np.float64)
    for dz in (0, 1):
        wz = fraction[:, 2] if dz else 1.0 - fraction[:, 2]
        for dy in (0, 1):
            wy = fraction[:, 1] if dy else 1.0 - fraction[:, 1]
            for dx in (0, 1):
                wx = fraction[:, 0] if dx else 1.0 - fraction[:, 0]
                source = volume_zyx[
                    lower[:, 2] + dz,
                    lower[:, 1] + dy,
                    lower[:, 0] + dx,
                ].astype(np.float64, copy=False)
                values += source * (wx * wy * wz)
    return values


def _by_pixel(array, height, width, samples_per_axis):
    """Group a dense subgrid as (output pixel, subquery[, XYZ])."""
    n = samples_per_axis
    if array.ndim == 3:
        return array.reshape(height, n, width, n, 3).transpose(0, 2, 1, 3, 4).reshape(
            height * width, n * n, 3
        )
    return array.reshape(height, n, width, n).transpose(0, 2, 1, 3).reshape(
        height * width, n * n
    )


def render(
    volume_zyx,
    volume_origin_xyz,
    surface,
    origin_uv,
    step_uv,
    shape_yx,
    depths_voxels,
    samples_per_axis=1,
):
    """Return ``(stack[D,H,W], valid_per_pixel[H,W], diagnostics)``.

    All depths share one conservative mask. The depth sequence is preserved
    exactly in order, including duplicates; it is never averaged or sorted.
    UV steps must be strictly positive, matching OriginalSurface. Diagnostics contain only
    JSON-serialisable scalar values/lists and no candidate locations.
    """
    volume = _finite_volume(volume_zyx)
    volume_origin = _vector(volume_origin_xyz, 3, "volume_origin_xyz")
    origin = _vector(origin_uv, 2, "origin_uv")
    step = _vector(step_uv, 2, "step_uv")
    if (step <= 0).any():
        raise ValueError("step_uv entries must be strictly positive")
    height, width = _shape(shape_yx)
    depths = np.asarray(depths_voxels, dtype=np.float64)
    if depths.ndim != 1 or not depths.size or not np.isfinite(depths).all():
        raise ValueError("depths_voxels must be a nonempty finite 1-D sequence")
    try:
        if isinstance(samples_per_axis, (bool, np.bool_)):
            raise TypeError
        n = operator.index(samples_per_axis)
    except TypeError:
        raise ValueError("samples_per_axis must be one of 1, 4, 8") from None
    if n not in (1, 4, 8):
        raise ValueError("samples_per_axis must be one of 1, 4, 8")
    for method in ("sample_grid", "footprint_support"):
        if not callable(getattr(surface, method, None)):
            raise ValueError(f"surface must provide {method}()")
    substep = step / n
    if not np.isfinite(substep).all() or (substep == 0).any():
        raise ValueError("quadrature step must remain finite and nonzero")

    stack = np.zeros((depths.size, height, width), dtype=np.float32)
    valid_per_pixel = np.zeros((height, width), dtype=bool)
    shape_xyz = np.asarray(volume.shape[::-1], dtype=np.float64)
    tile_side = int(np.sqrt(_MAX_TILE_QUERIES)) // n
    diagnostics = {
        "method": "source-surface uniform-UV-area midpoint footprint quadrature",
        "samples_per_axis": int(n),
        "subqueries_per_pixel": int(n * n),
        "output_shape_dyx": [int(depths.size), height, width],
        "volume_shape_zyx": [int(v) for v in volume.shape],
        "input_dtype": str(volume.dtype),
        "output_dtype": "float32",
        "accumulation_dtype": "float64",
        "total_output_pixels": height * width,
        "planned_surface_subqueries": height * width * n * n,
        "requested_surface_subqueries": 0,
        "interpolated_points_per_depth": 0,
        "tile_count": 0,
        "max_tile_subqueries": 0,
        "max_vertex_reads_per_interpolation_call": 0,
        "rejected_footprint_pixels": 0,
        "invalid_geometry_subqueries": 0,
        "nonfinite_geometry_subqueries": 0,
        "nonunit_normal_subqueries": 0,
        "ct_unsupported_subqueries_per_depth": [0] * int(depths.size),
        "ct_unsupported_pixels_per_depth": [0] * int(depths.size),
        "normal_length_atol": _NORMAL_LENGTH_ATOL,
        "continuous_uv_footprint_support_required": True,
        "all_eight_ct_vertices_required": True,
        "all_requested_depths_required": True,
        "normal_tube_injectivity_checked": False,
        "continuous_between_quadrature_ct_support_checked": False,
        "coordinate_clamping": False,
        "partial_footprint_renormalisation": False,
        "depth_averaging": False,
        "invalid_value": 0.0,
    }

    for y0 in range(0, height, tile_side):
        th = min(tile_side, height - y0)
        for x0 in range(0, width, tile_side):
            tw = min(tile_side, width - x0)
            diagnostics["tile_count"] += 1
            with np.errstate(over="ignore", invalid="ignore"):
                tile_origin = origin + np.asarray([x0, y0]) * step
                query_origin = tile_origin + (0.5 / n - 0.5) * step
            if not np.isfinite(tile_origin).all() or not np.isfinite(query_origin).all():
                raise ValueError("grid origin overflowed finite coordinates")
            footprint = surface.footprint_support(tile_origin, step, (th, tw))
            footprint_valid = _result_array(footprint, "valid", (th, tw), boolean=True)
            diagnostics["rejected_footprint_pixels"] += int((~footprint_valid).sum())
            if not footprint_valid.any():
                continue

            query_shape = (th * n, tw * n)
            query_mask = np.repeat(np.repeat(footprint_valid, n, axis=0), n, axis=1)
            diagnostics["requested_surface_subqueries"] += int(query_mask.sum())
            diagnostics["max_tile_subqueries"] = max(
                diagnostics["max_tile_subqueries"], query_shape[0] * query_shape[1]
            )
            sampled = surface.sample_grid(
                query_origin, substep, query_shape, query_mask=query_mask
            )
            xyz = _result_array(sampled, "xyz", query_shape + (3,))
            normals = _result_array(sampled, "normals", query_shape + (3,))
            geometry_flag = _result_array(sampled, "valid", query_shape, boolean=True)
            geometry_finite = np.isfinite(xyz).all(axis=-1) & np.isfinite(normals).all(axis=-1)
            with np.errstate(over="ignore", invalid="ignore"):
                normal_length = np.sqrt(np.sum(normals * normals, axis=-1))
            normal_unit = np.isclose(normal_length, 1.0, rtol=0.0, atol=_NORMAL_LENGTH_ATOL)
            geometry_valid = query_mask & geometry_flag & geometry_finite & normal_unit
            diagnostics["invalid_geometry_subqueries"] += int(
                (query_mask & ~geometry_valid).sum()
            )
            diagnostics["nonfinite_geometry_subqueries"] += int(
                (query_mask & ~geometry_finite).sum()
            )
            diagnostics["nonunit_normal_subqueries"] += int(
                (query_mask & geometry_finite & ~normal_unit).sum()
            )

            xyz = _by_pixel(xyz, th, tw, n)
            normals = _by_pixel(normals, th, tw, n)
            geometry_valid = _by_pixel(geometry_valid, th, tw, n)
            geometry_pixel_valid = footprint_valid.ravel() & geometry_valid.all(axis=1)
            pixel_valid = geometry_pixel_valid.copy()

            # Support preflight at *every* depth precedes any CT interpolation.
            for di, depth in enumerate(depths):
                with np.errstate(over="ignore", invalid="ignore"):
                    coords = xyz + depth * normals - volume_origin
                supported = _inside_trilinear(coords, shape_xyz)
                diagnostics["ct_unsupported_subqueries_per_depth"][di] += int(
                    (geometry_valid & ~supported).sum()
                )
                diagnostics["ct_unsupported_pixels_per_depth"][di] += int(
                    (geometry_pixel_valid & ~supported.all(axis=1)).sum()
                )
                pixel_valid &= supported.all(axis=1)

            if not pixel_valid.any():
                continue
            valid_count = int(pixel_valid.sum())
            diagnostics["interpolated_points_per_depth"] += valid_count * n * n
            diagnostics["max_vertex_reads_per_interpolation_call"] = max(
                diagnostics["max_vertex_reads_per_interpolation_call"],
                valid_count * n * n * 8,
            )
            retained_xyz = xyz[pixel_valid].reshape(-1, 3)
            retained_normals = normals[pixel_valid].reshape(-1, 3)
            retained_mask = pixel_valid.reshape(th, tw)
            valid_per_pixel[y0:y0 + th, x0:x0 + tw] = retained_mask
            for di, depth in enumerate(depths):
                coords = retained_xyz + depth * retained_normals - volume_origin
                values = _trilinear_valid(volume, coords)
                means = values.reshape(valid_count, n * n).mean(axis=1, dtype=np.float64)
                if not np.isfinite(means).all() or (np.abs(means) > np.finfo(np.float32).max).any():
                    raise FloatingPointError("rendered value exceeds finite float32 range")
                layer = stack[di, y0:y0 + th, x0:x0 + tw]
                layer[retained_mask] = means.astype(np.float32)

    diagnostics["valid_output_pixels"] = int(valid_per_pixel.sum())
    diagnostics["rejected_output_pixels"] = int(valid_per_pixel.size - valid_per_pixel.sum())
    return stack, valid_per_pixel, diagnostics
