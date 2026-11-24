"""Factor the frozen Q33 sampler into anchor and ray-direction rotations.

F_ab(u,v,d) = C + R**a (S(u,v)-C) + d R**b n(u,v).
The hybrid rays are intentionally not the normals of the anchor surface.
Identity and rigid cases delegate to the original Q33 implementation exactly.
Hybrids factor the author's already-FP32 tube displacement N32-S32, preserving
the actual discrete coordinate contract rather than silently re-associating it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_OLD_PATH = Path(__file__).resolve().parents[1] / "studio_oblique_stress/geometry.py"
_SPEC = importlib.util.spec_from_file_location("_q33_frozen_geometry_for_factors", _OLD_PATH)
_legacy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_legacy)

FACTORS = {"identity": (0, 0), "normal_only": (0, 1),
           "anchor_only": (1, 0), "rigid": (1, 1)}


def _matrices(patch):
    family = patch["family"]
    if family not in FACTORS:
        raise ValueError("Unknown factor family")
    rotation = np.asarray(patch["rotation"], dtype=np.float64)
    pivot = np.asarray(patch["pivot_xyz"], dtype=np.float64)
    if (rotation.shape != (3, 3) or pivot.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(pivot).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-10, rtol=0)):
        raise ValueError("Finite pivot and proper orthogonal rotation required")
    anchor, ray = FACTORS[family]
    return rotation if anchor else np.eye(3), rotation if ray else np.eye(3), pivot


def build_patch(surface, y, x, angle_deg, axis="x", family="identity",
                pitch_um=9.36, offsets_um=None):
    """Return native XYZ FP32 coordinates [D,128,128,3] and factor metadata.

    ``identity`` ignores a finite requested block angle and has effective angle
    zero. ``rigid`` and every zero-angle call preserve the historical coordinates
    exactly. ``normal_only`` rotates ray displacement around each unchanged
    anchor; ``anchor_only`` rotates anchors and leaves ray displacement fixed.
    """
    if family not in FACTORS:
        raise ValueError("Unknown factor family")
    requested_angle = float(angle_deg)
    if not np.isfinite(requested_angle):
        raise ValueError("Finite angle required")
    effective_angle = 0.0 if family == "identity" else requested_angle
    patch = _legacy.rotate_patch(surface, y, x, effective_angle, axis=axis,
                                 pitch_um=pitch_um, offsets_um=offsets_um)
    patch.update(family=family, requested_angle_deg=requested_angle,
                 anchor_rotation=bool(FACTORS[family][0]),
                 ray_rotation=bool(FACTORS[family][1]),
                 numerical_contract="Hybrids use C+Ra(S32-C)+Rb(N32-S32); identity/rigid delegate to frozen Q33",
                 ray_convention="Hybrid ray directions need not be normals to their anchor surface")
    if family in ("identity", "rigid") or effective_angle == 0:
        return patch
    original = _legacy.rotate_patch(surface, y, x, 0.0, axis=axis,
                                    pitch_um=pitch_um, offsets_um=offsets_um)
    source, _, valid = surface.points(np.arange(y, y+128), np.arange(x, x+128), pitch_um)
    if not np.array_equal(valid, original["mesh_valid"]):
        raise ValueError("Source geometry validity drift")
    ra, rb, pivot = _matrices(patch)
    source64 = np.asarray(source, dtype=np.float64)
    displacement = original["coords"].astype(np.float64) - source64[None]
    # Leaving unrotated anchors as S32 directly avoids pointless cancellation.
    anchors = source64 if family == "normal_only" else (source64-pivot) @ ra.T + pivot
    coords = anchors[None] + displacement @ rb.T
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    coords[:, ~valid] = -1
    patch["coords"] = coords
    patch["mesh_valid"] = np.asarray(valid, dtype=bool)
    return patch


def paired_tube(surface, ys, xs, depth, patch, pitch_um=9.36, fp32=False):
    """Evaluate paired tube points plus analytic local Jacobians.

    ``value`` has broadcast(ys,xs,depth)+[3] shape. Jacobian columns are chart
    y, chart x and native-voxel depth. With fp32=True, value follows the actual
    hybrid coordinate arithmetic; derivatives still describe the continuous
    bilinear surface and its signed normals. Both descriptions are reported.
    """
    ys, xs, depth = np.broadcast_arrays(np.asarray(ys, dtype=np.float64),
                                      np.asarray(xs, dtype=np.float64),
                                      np.asarray(depth, dtype=np.float64))
    ra, rb, pivot = _matrices(patch)
    evaluated = _legacy.surface_points_paired(surface, ys, xs, pitch_um)
    anchors = ((evaluated["xyz"]-pivot) @ ra.T + pivot)
    rays = evaluated["normal"] @ rb.T
    value = anchors + depth[..., None]*rays
    dy, dx = evaluated["dxyz_dy"] @ ra.T, evaluated["dxyz_dx"] @ ra.T
    ny, nx = evaluated["dnormal_dy"] @ rb.T, evaluated["dnormal_dx"] @ rb.T
    jacobian = np.stack((dy+depth[..., None]*ny,
                         dx+depth[..., None]*nx, rays), axis=-1)
    finite = (evaluated["valid"] & np.isfinite(value).all(-1)
              & np.isfinite(jacobian).all(axis=(-2, -1)))
    determinant = np.linalg.det(np.where(finite[..., None, None], jacobian, 0.0))
    base_determinant = np.sum(np.cross(dy, dx)*rays, axis=-1)
    geometric = (finite & (np.abs(base_determinant) > 1e-10)
                 & (np.abs(determinant) > 1e-10)
                 & (determinant*base_determinant > 0))
    if fp32:
        source = _legacy._surface_points(surface, ys, xs, pitch_um, np.float32)
        source32 = source["xyz"]
        native32 = source32 + np.asarray(depth, dtype=np.float32)[..., None]*source["normal"]
        family = patch["family"]
        if family == "identity" or np.array_equal(patch["rotation"], np.eye(3)):
            value = native32
        elif family == "rigid":
            value = np.asarray((native32.astype(np.float64)-pivot) @ ra.T + pivot,
                               dtype=np.float32)
        else:
            source64 = source32.astype(np.float64)
            displacement = native32.astype(np.float64)-source64
            anchors32 = source64 if family == "normal_only" else (source64-pivot) @ ra.T+pivot
            value = np.asarray(anchors32+displacement @ rb.T, dtype=np.float32)
        finite &= source["valid"] & np.isfinite(value).all(-1)
        geometric &= source["valid"] & np.isfinite(value).all(-1)
    return dict(value=value, jacobian=jacobian, valid=geometric,
                source_valid=finite, jacobian_determinant=determinant,
                base_jacobian_determinant=base_determinant,
                ray_direction=rays)


def inverse_readback(surface, y, x, patch, targetcore=(48, 80), pitch_um=9.36,
                     max_iters=15, residual_tol=0.01, depth_bounds=(-8.0, 7.0),
                     model_margin=16.0):
    """Map original reference-surface points into one factorial sampling tube.

    Identity/rigid readback is delegated exactly to frozen Q33. For ray-only,
    reference chart coordinates and depth zero are exact, with conditioning and
    tangence checked. Anchor-only uses bounded Newton/backtracking followed by
    the actual hybrid FP32 residual. A valid local inverse does not certify
    global injectivity or the unknown physical depth of annotated ink.
    """
    ra, rb, pivot = _matrices(patch)
    family = patch["family"]
    if family in ("identity", "rigid") or np.array_equal(patch["rotation"], np.eye(3)):
        result = _legacy.inverse_readback(surface, y, x, patch["rotation"], pivot,
            targetcore=targetcore, pitch_um=pitch_um, max_iters=max_iters,
            residual_tol=residual_tol, depth_bounds=depth_bounds, model_margin=model_margin)
        result["diagnostics"] = dict(result["diagnostics"], family=family,
                                     reference_mapping="Exact frozen Q33 delegation")
        return result
    y, x = _legacy._integer(y, "y"), _legacy._integer(x, "x")
    if isinstance(targetcore, slice):
        if targetcore.step not in (None, 1):
            raise ValueError("Unit target-core step required")
        start, stop = targetcore.start, targetcore.stop
    else:
        start, stop = targetcore
    start, stop = _legacy._integer(start, "core start"), _legacy._integer(stop, "core stop")
    max_iters = _legacy._integer(max_iters, "max_iters")
    if not (0 <= start < stop <= 128) or max_iters < 1:
        raise ValueError("Valid core and positive iteration budget required")
    depth_lo, depth_hi = map(float, depth_bounds)
    if (not np.isfinite([pitch_um, residual_tol, depth_lo, depth_hi, model_margin]).all()
            or pitch_um <= 0 or residual_tol <= 0 or depth_lo > depth_hi
            or not 0 <= model_margin < 63.5):
        raise ValueError("Valid pitch, residual, depth and output-margin bounds required")
    size = stop-start
    yy, xx = np.meshgrid(np.arange(y+start, y+stop), np.arange(x+start, x+stop), indexing="ij")
    target, _, original_valid = surface.points(np.arange(y+start, y+stop), np.arange(x+start, x+stop), pitch_um)
    target = target.reshape(-1, 3).astype(np.float64)
    original_valid = np.asarray(original_valid, dtype=bool).ravel()
    q = np.column_stack((yy.ravel().astype(np.float64), xx.ravel().astype(np.float64), np.zeros(size*size)))
    iterations = np.zeros(len(q), dtype=np.int16)
    solver_tol = min(1e-6, float(residual_tol)*0.01)
    if family == "anchor_only":
        active = original_valid.copy()
        for iteration in range(max_iters):
            current = paired_tube(surface, q[:, 0], q[:, 1], q[:, 2], patch, pitch_um)
            difference = current["value"]-target
            norm = np.linalg.norm(difference, axis=-1)
            active &= current["valid"] & np.isfinite(norm)
            ids = np.flatnonzero(active & (norm > solver_tol))
            if len(ids) == 0:
                break
            sv = np.linalg.svd(current["jacobian"][ids], compute_uv=False)
            regular = (sv[:, -1] > 1e-6) & (sv[:, 0]/sv[:, -1] < 1e6)
            active[ids[~regular]] = False
            ids = ids[regular]
            if len(ids) == 0:
                break
            step = np.linalg.solve(current["jacobian"][ids], difference[ids, :, None])[..., 0]
            step /= np.maximum(1.0, np.max(np.abs(step), axis=1)/8.0)[:, None]
            accepted = np.zeros(len(ids), dtype=bool)
            for factor in (1.0, 0.5, 0.25, 0.125, 0.0625):
                pending = np.flatnonzero(~accepted)
                if len(pending) == 0:
                    break
                candidate = q[ids[pending]]-factor*step[pending]
                trial = paired_tube(surface, candidate[:, 0], candidate[:, 1], candidate[:, 2], patch, pitch_um)
                trial_norm = np.linalg.norm(trial["value"]-target[ids[pending]], axis=-1)
                better = trial["valid"] & np.isfinite(trial_norm) & (trial_norm < norm[ids[pending]])
                chosen = pending[better]
                q[ids[chosen]] = candidate[better]
                iterations[ids[chosen]] = iteration+1
                accepted[chosen] = True
            active[ids[~accepted]] = False

    # Ray-only keeps q=(original_y,original_x,0) exactly. It still passes all
    # ray/anchor tangence and local-invertibility tests below.
    evaluated = paired_tube(surface, q[:, 0], q[:, 1], q[:, 2], patch, pitch_um)
    actual = paired_tube(surface, q[:, 0], q[:, 1], q[:, 2], patch, pitch_um, fp32=True)
    analytical_residual = np.linalg.norm(evaluated["value"]-target, axis=-1)
    physical_residual = np.linalg.norm(actual["value"].astype(np.float64)-target, axis=-1)
    geometric = evaluated["valid"] & actual["valid"]
    jac = evaluated["jacobian"]
    sv = np.linalg.svd(np.where(geometric[:, None, None], jac, np.eye(3)), compute_uv=False)
    regular = (sv[:, -1] > 1e-6) & (sv[:, 0]/sv[:, -1] < 1e6)
    local = q[:, :2]-np.array([y, x])
    inside_model = ((local >= model_margin) & (local <= 127-model_margin)).all(axis=-1)
    inside_depth = (q[:, 2] >= depth_lo) & (q[:, 2] <= depth_hi)
    converged = np.isfinite(physical_residual) & (physical_residual <= residual_tol)
    valid = original_valid & geometric & regular & converged & inside_model & inside_depth
    diagnostics = dict(family=family, total=int(len(q)), valid=int(valid.sum()),
        invalid_target_geometry=int((~original_valid).sum()),
        invalid_solved_geometry=int((~geometric).sum()),
        invalid_jacobian=int((~regular).sum()),
        outside_model_margin=int((~inside_model).sum()),
        outside_depth_band=int((~inside_depth).sum()),
        residual_failures=int((~converged).sum()),
        max_iterations_used=int(iterations.max()),
        residual_tolerance_native_voxels=float(residual_tol),
        max_valid_residual_native_voxels=float(physical_residual[valid].max()) if valid.any() else None,
        depth_bounds_vox=[depth_lo, depth_hi], model_margin_px=float(model_margin),
        reference_mapping="Exact original chart and zero depth" if family == "normal_only" else "General anchor/ray inverse",
        uniqueness="Converged nonfolded local inverse only; global injectivity and true 3D ink depth are not certified")
    return dict(query_yx=local.reshape(size, size, 2), depth_vox=q[:, 2].reshape(size, size),
                valid=valid.reshape(size, size), residual_vox=physical_residual.reshape(size, size),
                analytical_residual_vox=analytical_residual.reshape(size, size),
                iterations=iterations.reshape(size, size),
                jacobian_determinant=evaluated["jacobian_determinant"].reshape(size, size),
                base_jacobian_determinant=evaluated["base_jacobian_determinant"].reshape(size, size),
                diagnostics=diagnostics)
