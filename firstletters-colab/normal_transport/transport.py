"""Local normal transport in physical micrometre coordinates, using only NumPy.

The caller supplies the COMPLETE, deterministic, pointwise map R^3 -> R^3.
Input/output axes must be Cartesian, orthonormal, consistently handed and measured
in micrometres. The output is a target geometric sheet normal, not a mapped ray.
If a map operates on voxel indices, wrap BOTH input and output voxel scales first.
This module never loads transform files, meshes, CT, annotations or model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

PointMap = Callable[[np.ndarray], np.ndarray]


class GeometryError(ValueError):
    """A local normal cannot be certified under the requested numerical contract."""


@dataclass(frozen=True)
class Settings:
    """Numerical tolerances are policies, not estimates of manuscript accuracy.

    step_um is explicitly required. Central differences use h, h/2 and h/4.
    jacobian_atol/rtol compare Frobenius norms; Jacobians have units um/um.
    max_seconds is cooperative: it cannot preempt an unresponsive point_map.
    """

    step_um: float
    jacobian_atol: float = 1e-9
    jacobian_rtol: float = 1e-6
    normal_tolerance_degrees: float = 0.001
    stencil_relative_tolerance: float = 1e-8
    min_stretch: float = 1e-8
    max_condition: float = 1e6
    max_points: int = 4096
    batch_size: int = 256
    max_seconds: float = 10.0

    def validate(self) -> None:
        for name in (
            "step_um", "jacobian_atol", "jacobian_rtol",
            "normal_tolerance_degrees", "stencil_relative_tolerance", "min_stretch", "max_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise GeometryError(f"{name} must be finite and positive")
        if not np.isfinite(self.max_condition) or self.max_condition < 1:
            raise GeometryError("max_condition must be finite and at least one")
        if self.normal_tolerance_degrees >= 90:
            raise GeometryError("normal_tolerance_degrees must be below 90")
        if self.stencil_relative_tolerance >= 0.01:
            raise GeometryError("stencil_relative_tolerance must be below 0.01")
        for name in ("max_points", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise GeometryError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TransportResult:
    """All returned arrays are float64 and read-only; no face is silently flipped."""

    normals: np.ndarray
    jacobian: np.ndarray
    determinant: np.ndarray
    condition: np.ndarray
    jacobian_coarse_change: np.ndarray
    jacobian_fine_change: np.ndarray
    one_sided_gap_fine: np.ndarray
    normal_step_change_degrees: np.ndarray
    map_calls: int
    mapped_point_evaluations: int
    elapsed_seconds: float
    coordinate_unit: str = "um"


def _fail_at(message: str, mask: np.ndarray) -> None:
    if np.any(mask):
        indices = np.flatnonzero(mask)[:8].tolist()
        raise GeometryError(f"{message}; sample indices {indices}")


def _unit(v: np.ndarray) -> np.ndarray:
    # Scaling first avoids squaring huge or tiny nonzero vectors.
    scale = np.max(np.abs(v), axis=-1, keepdims=True)
    _fail_at("zero normal", scale[:, 0] == 0)
    scaled = v / scale
    return scaled / np.linalg.norm(scaled, axis=-1, keepdims=True)


def _check_jacobian(j: np.ndarray, settings: Settings, level: str) -> tuple[np.ndarray, np.ndarray]:
    _fail_at(f"nonfinite Jacobian at {level}", ~np.isfinite(j).all(axis=(1, 2)))
    try:
        singular = np.linalg.svd(j, compute_uv=False)
        determinant = np.linalg.det(j)
    except np.linalg.LinAlgError as exc:
        raise GeometryError(f"Jacobian factorisation failed at {level}") from exc
    _fail_at(f"singular or vanishing stretch at {level}", singular[:, -1] < settings.min_stretch)
    _fail_at(f"nonfinite determinant at {level}", ~np.isfinite(determinant))
    _fail_at(f"nonpositive determinant at {level}", determinant <= 0)
    condition = singular[:, 0] / singular[:, -1]
    _fail_at(f"ill-conditioned Jacobian at {level}", (~np.isfinite(condition)) | (condition > settings.max_condition))
    return determinant, condition


def transport_normals(
    point_map: PointMap,
    points_um: np.ndarray,
    normals: np.ndarray,
    *,
    settings: Settings,
) -> TransportResult:
    """Return the target SHEET normal normalise(J_F(x)^(-T) n).

    A tangent vector or previously chosen sampling ray would use J_F(x) v instead.
    This API deliberately does not perform that distinct operation.

    points_um and normals must both have shape (N,3), with 1 <= N <= max_points.
    Normals are dimensionless physical-coordinate directions, and need not be unit.
    point_map receives float64 arrays (B,3) and must return float64-or-better arrays
    of the same shape. It must be deterministic, pointwise, and side-effect free.

    There are exactly 19 mapped-point evaluations per input point: the centre and
    +/- each axis at three steps. The finest central Jacobian is retained. The
    two finest transported normals must also agree within the angular tolerance.

    Rejects any invalid member of a batch; there is no partial-success mask. Checks
    of determinant/convergence are local at sampled points and scales. They cannot
    certify global injectivity, exclude all unresolved oscillations, or establish
    the correctness of a supplied registration. Caller-owned map costs are unbounded
    by this helper; its timer checks before/after every call and final calculation.
    """
    start = perf_counter()
    settings.validate()
    if not callable(point_map):
        raise GeometryError("point_map must be callable")
    raw_points, raw_normals = np.asarray(points_um), np.asarray(normals)
    if raw_points.dtype.kind not in "fiu" or raw_normals.dtype.kind not in "fiu":
        raise GeometryError("points and normals must be real numeric arrays")
    points = np.asarray(raw_points, dtype=np.float64)
    supplied_normals = np.asarray(raw_normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not (1 <= len(points) <= settings.max_points):
        raise GeometryError("points_um must have shape (N,3) within the point limit")
    if supplied_normals.shape != points.shape:
        raise GeometryError("normals must have the same (N,3) shape as points_um")
    _fail_at("nonfinite points", ~np.isfinite(points).all(axis=1))
    _fail_at("nonfinite normals", ~np.isfinite(supplied_normals).all(axis=1))
    source_normals = _unit(supplied_normals)
    jacobians = np.empty((3, len(points), 3, 3), dtype=np.float64)
    side_gaps = np.empty((3, len(points)), dtype=np.float64)
    calls = 0
    evaluations = 0

    def check_time() -> None:
        if perf_counter() - start > settings.max_seconds:
            raise GeometryError("cooperative runtime limit exceeded")

    def evaluate(x: np.ndarray) -> np.ndarray:
        nonlocal calls, evaluations
        check_time()
        try:
            raw = np.asarray(point_map(x.copy()))
        except Exception as exc:
            raise GeometryError("point_map evaluation failed") from exc
        calls += 1
        evaluations += len(x)
        check_time()
        if raw.shape != x.shape or raw.dtype.kind != "f" or raw.dtype.itemsize < 8:
            raise GeometryError("point_map must return matching float64-or-better (B,3) values")
        out = np.array(raw, dtype=np.float64, copy=True)
        _fail_at("nonfinite mapped points", ~np.isfinite(out).all(axis=1))
        return out

    for begin in range(0, len(points), settings.batch_size):
        end = min(begin + settings.batch_size, len(points))
        x = points[begin:end]
        centre = evaluate(x)
        for level, factor in enumerate((1.0, 0.5, 0.25)):
            step = settings.step_um * factor
            side = np.empty((len(x), 3, 3), dtype=np.float64)
            for axis in range(3):
                plus, minus = x.copy(), x.copy()
                plus[:, axis] += step
                minus[:, axis] -= step
                if not np.isfinite(plus).all() or not np.isfinite(minus).all():
                    raise GeometryError("nonfinite finite-difference stencil")
                _fail_at("finite-difference step is not representable", (plus[:, axis] == x[:, axis]) | (minus[:, axis] == x[:, axis]))
                positive_step = plus[:, axis] - x[:, axis]
                negative_step = x[:, axis] - minus[:, axis]
                # Nonzero steps can still round to a materially wrong spacing.
                # Stable central differences alone do not reveal that bias.
                relative_error = np.maximum(np.abs(positive_step / step - 1), np.abs(negative_step / step - 1))
                _fail_at("actual finite-difference spacing differs from nominal step", relative_error > settings.stencil_relative_tolerance)
                fplus, fminus = evaluate(plus), evaluate(minus)
                jacobians[level, begin:end, :, axis] = (fplus - fminus) / (2 * step)
                # A cusp can have stable central differences but inconsistent
                # one-sided derivatives. These reuse the same point-map calls.
                side[:, :, axis] = ((fplus - centre) - (centre - fminus)) / step
            side_gaps[level, begin:end] = np.linalg.norm(side, axis=(1, 2))

    for level, name in enumerate(("h", "h/2", "h/4")):
        determinant, condition = _check_jacobian(jacobians[level], settings, name)
        _fail_at(f"nonfinite one-sided derivative gap at {name}", ~np.isfinite(side_gaps[level]))
    coarse = np.linalg.norm(jacobians[1] - jacobians[0], axis=(1, 2))
    fine = np.linalg.norm(jacobians[2] - jacobians[1], axis=(1, 2))
    tolerance = settings.jacobian_atol + settings.jacobian_rtol * np.linalg.norm(jacobians[2], axis=(1, 2))
    # Central truncation differences should decrease (normally by about 1/4),
    # unless already within tolerance where floating-point noise can dominate.
    converged = (fine <= tolerance) & ((coarse <= tolerance) | (fine <= 0.5 * coarse))
    _fail_at("Jacobian step convergence failed", ~converged)
    one_sided_ok = (side_gaps[2] <= 2 * tolerance) | (side_gaps[2] <= 0.75 * side_gaps[1])
    _fail_at("one-sided derivatives do not converge", ~one_sided_ok)

    # cofactor(J) n = det(J) J^(-T) n. Positive determinant was required at
    # every step, so solving the transpose system preserves the oriented face.
    try:
        fine_normals = _unit(np.linalg.solve(jacobians[2].transpose(0, 2, 1), source_normals[..., None])[..., 0])
        previous_normals = _unit(np.linalg.solve(jacobians[1].transpose(0, 2, 1), source_normals[..., None])[..., 0])
    except np.linalg.LinAlgError as exc:
        raise GeometryError("normal transport solve failed") from exc
    cross = np.linalg.norm(np.cross(previous_normals, fine_normals), axis=1)
    dot = np.sum(previous_normals * fine_normals, axis=1)
    angles = np.degrees(np.arctan2(cross, dot))
    _fail_at("transported normal step convergence failed", angles > settings.normal_tolerance_degrees)
    check_time()
    arrays = (fine_normals, jacobians[2], determinant, condition, coarse, fine, side_gaps[2], angles)
    for array in arrays:
        array.setflags(write=False)
    return TransportResult(*arrays, calls, evaluations, perf_counter() - start)
