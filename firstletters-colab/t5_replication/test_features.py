"""Synthetic parity and support tests for the frozen T5 feature adapter.

No CT, image, or label file is read. The reference implementation is obtained
only from two hash-verified source-code blobs in the local Git object database.
AST extraction executes selected function definitions, never module top-level
code, loaders, experiment runners, network code, or command-line entry points.

Run from the repository root:
``PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s firstletters-colab/t5_replication -p test_features.py -v``
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import unittest
from typing import Any, Sequence

import numpy as np
from scipy import linalg as sla
from scipy import ndimage as ndi
from scipy import signal as ssig


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location("t5_replication_features_under_test", HERE / "features.py")
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def _load_pinned_reference() -> dict[str, Any]:
    """Load the exact upstream functions from local, SHA-256-checked Git blobs."""
    requested = {
        "gemini_lane/pherc0139_transverse_physics.py": {
            "_fill_nearest_2d", "_normalized_gaussian", "extract_subvoxel_height_field",
        },
        "gemini_lane/transverse_wavelet_oct_physics.py": {
            "extract_peak_aligned_depth_profiles", "_build_log_gabor_bank_fft",
            "_local_blended_rx_mahalanobis", "_antisymmetric_mnf_and_pca",
            "compute_oct_depth_waveform_features", "compute_wavelet_scattering_features",
        },
    }
    namespace: dict[str, Any] = {
        "np": np, "ndi": ndi, "sla": sla, "ssig": ssig, "math": math,
        "Any": Any, "Sequence": Sequence, "OCT_WAVEFORM_DIM": 31,
        "FEATURE_FAMILIES": {"wavelet_scattering_2d": (
            "wst_s1_isotropic", "wst_s1_anisotropy", "wst_s1_scale_slope",
            "wst_s2_cross_par_ratio", "wst_s2_normalized_coupling",
        )},
    }
    for path, names in requested.items():
        raw = subprocess.check_output(
            ["git", "show", f"{adapter.SOURCE_COMMIT}:{path}"], cwd=REPO_ROOT,
        )
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash != adapter.SOURCE_SHA256[path]:
            raise AssertionError(f"Pinned source SHA-256 mismatch for {path}: {actual_hash}")
        tree = ast.parse(raw.decode("utf-8"), filename=path)
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in selected} != names:
            raise AssertionError(f"Missing pinned reference functions in {path}")
        module = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *selected,
        ], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, f"{adapter.SOURCE_COMMIT}:{path}", "exec"), namespace)
    return namespace


def _synthetic_stack(*, boundary_peaks: bool = False) -> np.ndarray:
    """A deterministic native-depth field with weave, waviness, and a shoulder."""
    rng = np.random.default_rng(2046306)
    y, x = np.indices((32, 40), dtype=np.float64)
    z = np.arange(109, dtype=np.float64)[:, None, None]
    centres = (30.0 if boundary_peaks else 49.0) + 0.5 * np.sin(x / 4.0) + 0.4 * np.cos(y / 5.0)
    if boundary_peaks:
        centres[7:16, 5:15] = 8.25
    amplitude = 95.0 + 18.0 * np.sin(x * 1.2) + 12.0 * np.cos(y * 0.7)
    widths = 2.4 + 0.4 * np.sin((x + y) / 8.0)
    sheet = amplitude[None] * np.exp(-0.5 * ((z - centres[None]) / widths[None]) ** 2)
    shoulder_strength = 15.0 + 12.0 * np.sin(x / 5.0) * np.cos(y / 4.0)
    shoulder = shoulder_strength[None] * np.exp(-0.5 * ((z - centres[None] - 5.0) / 2.0) ** 2)
    return np.maximum(0.0, 17.0 + sheet + shoulder + rng.normal(0.0, 0.5, sheet.shape)).astype(np.float32)


class TestFrozenFeatureAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = _load_pinned_reference()

    def _reference(self, stack: np.ndarray, support: np.ndarray | None = None) -> dict[str, Any]:
        vm = stack.max(axis=0) > 0
        if support is not None:
            vm &= support
        hf = self.ref["extract_subvoxel_height_field"](
            stack, voxel_um=2.399, valid_mask=vm,
            search_half_slices=24, z_smooth_sigma=1.0,
            detrend_sigma_um=200.0, band_half_width=1,
        )
        oct_features, _ = self.ref["compute_oct_depth_waveform_features"](
            stack, hf["n_peak"], hf["sheet_valid"], voxel_um=2.399,
            half_span_vx=15.0, waveform_dim=31,
        )
        wst_features, _ = self.ref["compute_wavelet_scattering_features"](
            hf["surface_ct"], hf["sheet_valid"], voxel_um=2.399,
        )
        _, _, bounds = self.ref["extract_peak_aligned_depth_profiles"](
            stack, hf["n_peak"], half_span_vx=15.0, waveform_dim=31,
        )
        return {
            "features": {
                "raw_ct_surface": hf["surface_ct"],
                "oct_hilbert_env_peak": oct_features["oct_hilbert_env_peak"],
                "wst_s1_scale_slope": wst_features["wst_s1_scale_slope"],
            },
            "sheet_valid": hf["sheet_valid"], "bounds": bounds, "hf": hf,
        }

    def test_exact_feature_parity_with_pinned_original(self) -> None:
        for use_mask in (False, True):
            with self.subTest(explicit_source_mask=use_mask):
                stack = _synthetic_stack()
                stack[:, 0:2, 0:3] = 0.0
                support = np.ones(stack.shape[1:], dtype=bool)
                if use_mask:
                    support[8:12, 15:19] = False
                ref = self._reference(stack, support)
                features, mask, diagnostics = adapter.compute(stack, valid_mask=support)
                for name in adapter.POLARITIES:
                    # Full maps, not just scored pixels: support policy cannot
                    # silently perturb feature arithmetic outside the common mask.
                    np.testing.assert_array_equal(features[name], ref["features"][name])
                    self.assertEqual(features[name].dtype, np.float32)
                np.testing.assert_array_equal(mask, ref["sheet_valid"] & ref["bounds"])
                self.assertEqual(diagnostics["height_field"]["z_center_slice"], ref["hf"]["z_center_slice"])
                json.dumps(diagnostics, allow_nan=False)

    def test_full_depth_support_excludes_clamped_profiles_without_changing_features(self) -> None:
        stack = _synthetic_stack(boundary_peaks=True)
        ref = self._reference(stack)
        features, common_valid, diagnostics = adapter.compute(stack)
        clamped = ref["sheet_valid"] & ~ref["bounds"]
        self.assertGreater(int(clamped.sum()), 0)
        self.assertGreater(int(common_valid.sum()), 0)
        self.assertFalse(common_valid[clamped].any())
        # Same mask serves the baseline and both proposed features. Values of
        # excluded profiles are preserved for inspection, never silently repaired.
        np.testing.assert_array_equal(common_valid, ref["sheet_valid"] & ref["bounds"])
        for name in adapter.POLARITIES:
            np.testing.assert_array_equal(features[name], ref["features"][name])
        self.assertEqual(
            diagnostics["support"]["sheet_valid_but_profile_out_of_bounds"], int(clamped.sum()),
        )

    def test_all_zero_input_abstains(self) -> None:
        features, common_valid, diagnostics = adapter.compute(np.zeros((109, 12, 16), dtype=np.uint8))
        self.assertFalse(common_valid.any())
        self.assertEqual(diagnostics["support"]["legacy_sheet_valid_columns"], 0)
        for feature in features.values():
            self.assertTrue(np.isfinite(feature).all())
            self.assertFalse(feature.any())
        json.dumps(diagnostics, allow_nan=False)

    def test_input_and_polarity_contract(self) -> None:
        expected = {"raw_ct_surface": 1, "oct_hilbert_env_peak": 1, "wst_s1_scale_slope": -1}
        self.assertEqual(dict(adapter.POLARITIES), expected)
        with self.assertRaises(TypeError):
            adapter.POLARITIES["wst_s1_scale_slope"] = 1
        with self.assertRaisesRegex(ValueError, "shape"):
            adapter.compute(np.ones((25, 12, 16), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "voxel_um"):
            adapter.compute(np.ones((109, 12, 16), dtype=np.uint8), voxel_um=8.64)
        bad = np.ones((109, 12, 16), dtype=np.float32)
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            adapter.compute(bad)
        with self.assertRaisesRegex(ValueError, "boolean"):
            adapter.compute(np.ones((109, 12, 16), dtype=np.uint8), valid_mask=np.ones((12, 16)))


if __name__ == "__main__":
    unittest.main()
