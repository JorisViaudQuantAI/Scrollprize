"""Small generated-only correctness receipt; no CT, network or trained model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
import unittest

import numpy as np

from transport import Settings, transport_normals
from test_transport import TransportTests, angle_degrees, mapped_surface_normal, rotation, smooth_example, unit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise ValueError("Refusing to overwrite an existing receipt")
    if any((p / ".git").exists() for p in out.parents):
        raise ValueError("Write the receipt outside a Git repository")
    started = perf_counter()
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(TransportTests))
    if not tests.wasSuccessful():
        raise RuntimeError(log.getvalue())
    rng = np.random.default_rng(20261123)
    points = rng.uniform(-2, 2, (24, 3))
    varied_normals = unit(rng.normal(size=(24, 3)))
    z_normals = np.tile([0., 0., 1.], (len(points), 1))
    rigid = rotation()
    shear = np.array([[2., 0.5, 1.], [0., 1., 0.], [0., 0., 0.5]])
    nonlinear_a, _, _, nonlinear = smooth_example()
    cases = (
        ("positive_similarity", lambda x: 1.7 * (x @ rigid.T) + np.array([2., -3., 0.4]), 1.7 * rigid, varied_normals),
        ("anisotropic_scale_and_shear", lambda x: x @ shear.T, shear, z_normals),
        ("affine_plus_generated_smooth_displacement", nonlinear, nonlinear_a, varied_normals),
    )
    measured = []
    for name, point_map, affine, normals in cases:
        result = transport_normals(point_map, points, normals, settings=Settings(step_um=0.002))
        independent = mapped_surface_normal(point_map, points, normals)
        corrected_error = angle_degrees(result.normals, independent)
        old_error = angle_degrees(unit(normals @ affine.T), independent)
        measured.append({
            "case": name,
            "points": len(points),
            "corrected_max_angular_error_degrees": float(corrected_error.max()),
            "old_affine_vector_max_angular_error_degrees": float(old_error.max()),
            "old_affine_vector_median_angular_error_degrees": float(np.median(old_error)),
            "minimum_jacobian_determinant": float(result.determinant.min()),
            "maximum_jacobian_condition": float(result.condition.max()),
            "maximum_normal_step_change_degrees": float(result.normal_step_change_degrees.max()),
            "mapped_point_evaluations": result.mapped_point_evaluations,
        })
    elapsed = perf_counter() - started
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if elapsed > 10 or peak_rss_kib > 256 * 1024:
        raise RuntimeError("Generated review exceeded its ten-second/256-MiB budget")
    here = Path(__file__).resolve().parent
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.iterdir()) if p.suffix in {".py", ".json", ".md"}}
    receipt = {
        "schema_version": 1,
        "status": "passed_generated_geometry_only",
        "scope": "Geometric target sheet normals; not a transported-ray implementation, manuscript validation or ink improvement.",
        "seed": 20261123,
        "test_count": tests.testsRun,
        "test_failures": len(tests.failures),
        "test_errors": len(tests.errors),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "elapsed_check_seconds": elapsed,
        "peak_rss_kib_linux": peak_rss_kib,
        "cases": measured,
        "package_file_sha256": code_hashes,
        "ct_payload_bytes": 0,
        "model_forwards": 0,
        "network_requests": 0,
        "real_transform_parameters_loaded": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "tests": tests.testsRun, "elapsed_seconds": elapsed, "peak_rss_kib": peak_rss_kib, "cases": measured}, indent=2))


if __name__ == "__main__":
    main()
