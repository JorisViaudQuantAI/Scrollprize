"""Rigid native-XYZ tilt and physical readback for the frozen Q33 experiment.

The supplied Surface is the pinned author's implementation, with transformed
mesh vertices ``a``, original validity ``valid``, XY mesh pitch ``pitch``, native
CT pitch ``ct_um``, normal sign ``sign`` and Cartesian ``points`` method. Input
geometry uses that method unchanged. The inverse uses the same bilinear mesh
and its signed cross(drow,dcol) normals; it does not flatten the curved surface.

An x/y tilt rotates about the positive local chart x/y tangent at the native
patch centre. Angles follow world-XYZ Rodrigues' right-hand rule. The patch's
reference surface centre, not its asymmetric 16-plane midpoint, is the pivot.
"""
import operator

import numpy as np


def _integer(value, name):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _surface_points(surface, ys, xs, pitch_um, dtype):
    """Paired, rather than Cartesian, form of the author's bilinear evaluator."""
    ys, xs = np.broadcast_arrays(np.asarray(ys, dtype=dtype), np.asarray(xs, dtype=dtype))
    vertices = np.asarray(surface.a, dtype=dtype)
    valid_nodes = np.asarray(surface.valid, dtype=bool)
    pitch = np.asarray(surface.pitch, dtype=np.float64)
    ct_um, sign = float(surface.ct_um), float(surface.sign)
    if (vertices.ndim != 3 or vertices.shape[-1] != 3
            or min(vertices.shape[:2]) < 2 or valid_nodes.shape != vertices.shape[:2]
            or pitch.shape != (2,) or not np.isfinite(pitch).all()
            or np.any(pitch <= 0) or not np.isfinite(ct_um) or ct_um <= 0
            or sign not in (-1.0, 1.0) or not np.isfinite(pitch_um) or pitch_um <= 0):
        raise ValueError("Invalid author Surface geometry/pitch/sign")
    ky, kx = dtype(pitch_um / pitch[1]), dtype(pitch_um / pitch[0])
    gy, gx = (ys + dtype(0.5)) * ky, (xs + dtype(0.5)) * kx
    finite = np.isfinite(gy) & np.isfinite(gx)
    iy = np.floor(np.where(finite, gy, 0)).astype(np.int64)
    ix = np.floor(np.where(finite, gx, 0)).astype(np.int64)
    valid = finite & (iy >= 0) & (ix >= 0) & (iy < vertices.shape[0] - 1) & (ix < vertices.shape[1] - 1)
    iy, ix = np.clip(iy, 0, vertices.shape[0] - 2), np.clip(ix, 0, vertices.shape[1] - 2)
    v, u = gy - iy.astype(dtype), gx - ix.astype(dtype)
    a, b, c, d = vertices[iy, ix], vertices[iy, ix + 1], vertices[iy + 1, ix], vertices[iy + 1, ix + 1]
    valid &= valid_nodes[iy, ix] & valid_nodes[iy, ix + 1] & valid_nodes[iy + 1, ix] & valid_nodes[iy + 1, ix + 1]
    xyz = ((1 - u)[..., None] * (1 - v)[..., None] * a
           + u[..., None] * (1 - v)[..., None] * b
           + (1 - u)[..., None] * v[..., None] * c
           + u[..., None] * v[..., None] * d)
    dx = (1 - v)[..., None] * (b - a) + v[..., None] * (d - c)
    dy = (1 - u)[..., None] * (c - a) + u[..., None] * (d - b)
    cross = np.cross(dy, dx)
    length = np.linalg.norm(cross, axis=-1)
    valid &= np.isfinite(xyz).all(-1) & np.isfinite(length) & (length > 1e-6)
    safe_length = np.maximum(length, dtype(1e-6))
    normal = cross.copy()
    normal *= sign / safe_length[..., None]
    max_edge = np.maximum.reduce([np.linalg.norm(b - a, axis=-1), np.linalg.norm(c - a, axis=-1),
                                  np.linalg.norm(d - b, axis=-1), np.linalg.norm(d - c, axis=-1)])
    valid &= max_edge < 5 * max(pitch) / ct_um

    # Analytic derivatives include normal rotation within each bilinear cell.
    mixed = d - b - c + a
    cross_dx = np.cross(mixed, dx) * kx
    cross_dy = np.cross(dy, mixed) * ky
    unit = cross / safe_length[..., None]
    normal_dx = sign * (cross_dx - unit * np.sum(unit * cross_dx, axis=-1)[..., None]) / safe_length[..., None]
    normal_dy = sign * (cross_dy - unit * np.sum(unit * cross_dy, axis=-1)[..., None]) / safe_length[..., None]
    derivative_x, derivative_y = dx * kx, dy * ky
    xyz = np.where(valid[..., None], xyz, dtype(-1))
    normal = np.where(valid[..., None], normal, dtype(0))
    derivative_y = np.where(valid[..., None], derivative_y, dtype(0))
    derivative_x = np.where(valid[..., None], derivative_x, dtype(0))
    normal_dy = np.where(valid[..., None], normal_dy, dtype(0))
    normal_dx = np.where(valid[..., None], normal_dx, dtype(0))
    return dict(xyz=xyz, normal=normal, valid=valid,
                dxyz_dy=derivative_y, dxyz_dx=derivative_x,
                dnormal_dy=normal_dy, dnormal_dx=normal_dx)


def surface_points_paired(surface, ys, xs, pitch_um=9.36):
    """FP64 paired bilinear evaluation and analytic first derivatives.

    ``ys`` and ``xs`` broadcast to one shape. Derivatives are with respect to
    original chart pixel coordinates; positions and normal depths are native
    CT voxels. FP64 uses the author's already transformed FP32 vertices.
    """
    return _surface_points(surface, ys, xs, pitch_um, np.float64)


def rotate_patch(surface, y, x, angle_deg, axis="x", pitch_um=9.36, offsets_um=None):
    """Rotate the complete native sampling tube of a 128-square patch.

    At angle zero, ``coords`` exactly equals the author's pre-quantization
    geometry construction. Nonzero angles rotate anchors and normals together.
    Invalid mesh positions remain sentinel -1 and are separately masked.
    """
    y, x = _integer(y, "y"), _integer(x, "x")
    angle_deg = float(angle_deg)
    if axis not in ("x", "y") or not np.isfinite(angle_deg):
        raise ValueError("Finite angle and axis 'x' or 'y' required")
    if not np.isfinite(pitch_um) or pitch_um <= 0:
        raise ValueError("Positive finite chart pitch required")
    if offsets_um is None:
        offsets_um = np.arange(-8, 8, dtype=np.float64) * float(surface.ct_um)
    offsets = np.asarray(offsets_um, dtype=np.float32) / surface.ct_um
    if offsets.ndim != 1 or len(offsets) == 0 or not np.isfinite(offsets).all():
        raise ValueError("Finite 1D physical depth offsets required")
    xyz, normal, mesh_valid = surface.points(np.arange(y, y + 128), np.arange(x, x + 128), pitch_um)
    if xyz.shape != (128, 128, 3) or normal.shape != xyz.shape or mesh_valid.shape != (128, 128):
        raise ValueError("Unexpected author Surface.points output shape")
    native = xyz[None] + normal[None] * offsets[:, None, None, None]
    native[:, ~mesh_valid] = -1
    pivot_array, pivot_normal, pivot_valid = surface.points([y + 63.5], [x + 63.5], pitch_um)
    if not bool(pivot_valid[0, 0]):
        raise ValueError("Patch pivot lacks valid source geometry")
    pivot = np.asarray(pivot_array[0, 0], dtype=np.float64)
    n = np.asarray(pivot_normal[0, 0], dtype=np.float64)
    n /= np.linalg.norm(n)
    centre = surface_points_paired(surface, np.array([y + 63.5]), np.array([x + 63.5]), pitch_um)
    tangent = np.array(centre["dxyz_dx" if axis == "x" else "dxyz_dy"][0], dtype=np.float64)
    tangent -= np.dot(tangent, n) * n
    norm = np.linalg.norm(tangent)
    if not bool(centre["valid"][0]) or not np.isfinite(norm) or norm <= 1e-10:
        raise ValueError("Patch pivot has no valid tangent axis")
    axis_unit = tangent / norm
    if angle_deg == 0:
        rotation = np.eye(3, dtype=np.float64)
        coords = np.ascontiguousarray(native, dtype=np.float32)
    else:
        ax, ay, az = axis_unit
        skew = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]], dtype=np.float64)
        theta = np.deg2rad(angle_deg)
        rotation = (np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * np.outer(axis_unit, axis_unit)
                    + np.sin(theta) * skew)
        coords = np.ascontiguousarray((native.astype(np.float64) - pivot) @ rotation.T + pivot, dtype=np.float32)
        coords[:, ~mesh_valid] = -1
    return dict(coords=coords, mesh_valid=np.asarray(mesh_valid, dtype=bool),
                pivot_xyz=pivot, rotation=rotation, axis_unit=axis_unit,
                offsets_vox=offsets, angle_deg=angle_deg, axis=axis,
                normal_convention="signed cross(drow,dcol)",
                rotation_convention="world-XYZ right-hand Rodrigues about positive projected chart tangent")


def _tube(surface, q, pitch_um):
    evaluated = surface_points_paired(surface, q[:, 0], q[:, 1], pitch_um)
    depth = q[:, 2, None]
    value = evaluated["xyz"] + depth * evaluated["normal"]
    jacobian = np.stack([evaluated["dxyz_dy"] + depth * evaluated["dnormal_dy"],
                         evaluated["dxyz_dx"] + depth * evaluated["dnormal_dx"],
                         evaluated["normal"]], axis=-1)
    finite = evaluated["valid"] & np.isfinite(value).all(-1) & np.isfinite(jacobian).all(axis=(1, 2))
    determinant = np.linalg.det(np.where(finite[:, None, None], jacobian, 0.0))
    base_determinant = np.einsum("ij,ij->i", np.cross(evaluated["dxyz_dy"], evaluated["dxyz_dx"]), evaluated["normal"])
    # Positive relative orientation excludes a local depth-tube fold, not all
    # possible distant self-intersections of the source surface.
    good = finite & (determinant * base_determinant > 0) & (np.abs(determinant) > 1e-10)
    return value, jacobian, good, determinant, base_determinant


def inverse_readback(surface, y, x, rotation, pivot, targetcore=(48, 80), pitch_um=9.36,
                     max_iters=15, residual_tol=0.01, depth_bounds=(-8.0, 7.0), model_margin=16.0):
    """Map the original physical core into a rigidly rotated curved tube.

    Solve S(u',v') + d' n(u',v') = C + R.T (S(u,v)-C). The returned query is
    LOCAL model (y,x), not global chart coordinates. Depth is native voxels.
    The solver uses an analytic Jacobian, bounded Newton steps and backtracking.
    Validation includes author-FP32 residual, source geometry, a nonfolded local
    Jacobian, model-grid margin, and the declared depth band. This certifies a
    converged local root; it does not prove global tube injectivity.
    """
    y, x = _integer(y, "y"), _integer(x, "x")
    if isinstance(targetcore, slice):
        if targetcore.step not in (None, 1):
            raise ValueError("Unit target-core step required")
        start, stop = targetcore.start, targetcore.stop
    else:
        start, stop = targetcore
    start, stop = _integer(start, "core start"), _integer(stop, "core stop")
    if not (0 <= start < stop <= 128):
        raise ValueError("Core must lie inside model grid")
    max_iters = _integer(max_iters, "max_iters")
    if max_iters < 1 or not np.isfinite(residual_tol) or residual_tol <= 0:
        raise ValueError("Positive iteration budget and residual tolerance required")
    rotation, pivot = np.asarray(rotation, dtype=np.float64), np.asarray(pivot, dtype=np.float64)
    if (rotation.shape != (3, 3) or pivot.shape != (3,) or not np.isfinite(rotation).all()
            or not np.isfinite(pivot).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0)):
        raise ValueError("Proper orthogonal rotation and finite XYZ pivot required")
    depth_lo, depth_hi = map(float, depth_bounds)
    if (not np.isfinite([depth_lo, depth_hi, model_margin]).all()
            or depth_lo > depth_hi or not 0 <= model_margin < 63.5):
        raise ValueError("Valid depth bounds and model margin required")
    size = stop - start
    yy, xx = np.meshgrid(np.arange(y + start, y + stop), np.arange(x + start, x + stop), indexing="ij")
    target, _, target_valid = surface.points(np.arange(y + start, y + stop), np.arange(x + start, x + stop), pitch_um)
    target = target.reshape(-1, 3).astype(np.float64)
    original_valid = np.asarray(target_valid, dtype=bool).ravel()
    inverse_target = (target - pivot) @ rotation + pivot
    q = np.column_stack([yy.ravel().astype(np.float64), xx.ravel().astype(np.float64), np.zeros(size * size)])
    iterations = np.zeros(len(q), dtype=np.int16)
    solver_tol = min(1e-6, float(residual_tol) * 0.01)
    is_identity = np.array_equal(rotation, np.eye(3))
    if not is_identity:
        active = original_valid.copy()
        for iteration in range(max_iters):
            value, jacobian, geometric, _, _ = _tube(surface, q, pitch_um)
            residual = value - inverse_target
            norm = np.linalg.norm(residual, axis=-1)
            active &= geometric & np.isfinite(norm)
            moving = active & (norm > solver_tol)
            if not moving.any():
                break
            ids = np.flatnonzero(moving)
            singular_values = np.linalg.svd(jacobian[ids], compute_uv=False)
            regular = ((singular_values[:, -1] > 1e-6)
                       & (singular_values[:, 0] / singular_values[:, -1] < 1e6))
            active[ids[~regular]] = False
            ids = ids[regular]
            if len(ids) == 0:
                break
            step = np.linalg.solve(jacobian[ids], residual[ids, :, None])[..., 0]
            step /= np.maximum(1.0, np.max(np.abs(step), axis=1) / 8.0)[:, None]
            accepted = np.zeros(len(ids), dtype=bool)
            for factor in (1.0, 0.5, 0.25, 0.125, 0.0625):
                pending = np.flatnonzero(~accepted)
                if len(pending) == 0:
                    break
                candidate = q[ids[pending]] - factor * step[pending]
                candidate_value, _, candidate_geometry, _, _ = _tube(surface, candidate, pitch_um)
                candidate_norm = np.linalg.norm(candidate_value - inverse_target[ids[pending]], axis=-1)
                better = candidate_geometry & np.isfinite(candidate_norm) & (candidate_norm < norm[ids[pending]])
                use = pending[better]
                q[ids[use]] = candidate[better]
                iterations[ids[use]] = iteration + 1
                accepted[use] = True
            active[ids[~accepted]] = False

    # Final physical residual follows the author's actual FP32 coordinate rule,
    # while analytical residual/conditioning describe the smooth inverse solve.
    value, jacobian, geometric, determinant, base_determinant = _tube(surface, q, pitch_um)
    analytical_residual = np.linalg.norm(value - inverse_target, axis=-1)
    source = _surface_points(surface, q[:, 0], q[:, 1], pitch_um, np.float32)
    tube_fp32 = source["xyz"] + np.asarray(q[:, 2], dtype=np.float32)[:, None] * source["normal"]
    physical_residual = np.linalg.norm(tube_fp32.astype(np.float64) - inverse_target, axis=-1)
    if is_identity:
        # Cartesian author points are the identity oracle; never introduce a
        # fractional interpolation or a spurious subvoxel depth at baseline.
        physical_residual[original_valid] = 0.0
        analytical_residual[original_valid] = 0.0
    sv = np.linalg.svd(np.where(geometric[:, None, None], jacobian, np.eye(3)), compute_uv=False)
    regular = (sv[:, -1] > 1e-6) & (sv[:, 0] / sv[:, -1] < 1e6)
    local = q[:, :2] - np.array([y, x])
    inside_model = ((local >= model_margin) & (local <= 127 - model_margin)).all(axis=-1)
    inside_depth = (q[:, 2] >= depth_lo) & (q[:, 2] <= depth_hi)
    converged = np.isfinite(physical_residual) & (physical_residual <= residual_tol)
    valid = original_valid & source["valid"] & geometric & regular & converged & inside_model & inside_depth
    diagnostic = dict(total=int(len(q)), valid=int(valid.sum()),
                      invalid_target_geometry=int((~original_valid).sum()),
                      invalid_solved_geometry=int((~source["valid"] | ~geometric).sum()),
                      invalid_jacobian=int((~regular).sum()),
                      outside_model_margin=int((~inside_model).sum()),
                      outside_depth_band=int((~inside_depth).sum()),
                      residual_failures=int((~converged).sum()),
                      max_iterations_used=int(iterations.max()),
                      residual_tolerance_native_voxels=float(residual_tol),
                      max_valid_residual_native_voxels=float(physical_residual[valid].max()) if valid.any() else None,
                      depth_bounds_vox=[depth_lo, depth_hi], model_margin_px=float(model_margin),
                      uniqueness="Converged nonfolded local inverse; global tube injectivity is not certified")
    return dict(query_yx=local.reshape(size, size, 2), depth_vox=q[:, 2].reshape(size, size),
                valid=valid.reshape(size, size), residual_vox=physical_residual.reshape(size, size),
                analytical_residual_vox=analytical_residual.reshape(size, size),
                iterations=iterations.reshape(size, size),
                jacobian_determinant=determinant.reshape(size, size),
                base_jacobian_determinant=base_determinant.reshape(size, size), diagnostics=diagnostic)
