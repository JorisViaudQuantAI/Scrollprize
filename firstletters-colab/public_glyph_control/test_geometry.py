"""Generated contract checks; no CT download, checkpoint load or trained model."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("public_glyph_control_test_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
SELECTION = json.loads((HERE / "selection.json").read_text())


class GeometryTests(unittest.TestCase):
    def test_odd_depth_union_preserves_exact_released_inputs(self):
        union = np.arange(109)[22:87]
        for reverse in (False, True):
            expected = runner.central_indices(109, 64, reverse)
            actual = union[runner.central_indices(65, 64, reverse)]
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(union[:64], np.arange(22, 86))
        np.testing.assert_array_equal(union[1:][::-1], np.arange(86, 22, -1))
        self.assertFalse(np.array_equal(union[:64][::-1], union[1:][::-1]))

    def test_all_289_full_chunks_cover_context_exactly_once(self):
        source, context = PROTOCOL["source"], SELECTION["context"]
        plan = runner.chunk_plan(source["shape_zyx"], source["chunks_zyx"],
                                 context["origin_yx"], context["shape_yx"])
        self.assertEqual(len(plan), 289)
        self.assertEqual(plan[0]["key"], "0/79/106")
        self.assertEqual(plan[-1]["key"], "0/95/122")
        coverage = np.zeros((2048, 2048), dtype=np.uint8)
        for record in plan:
            sy0, sy1, sx0, sx1 = record["source_yx"]
            ty0, ty1, tx0, tx1 = record["target_yx"]
            self.assertEqual((sy1 - sy0, sx1 - sx0), (ty1 - ty0, tx1 - tx0))
            self.assertTrue(0 <= sy0 < sy1 <= 128 and 0 <= sx0 < sx1 <= 128)
            coverage[ty0:ty1, tx0:tx1] += 1
        self.assertTrue((coverage == 1).all())
        self.assertEqual(289 * 109 * 128 * 128, 516112384)
        self.assertEqual(516112384 + 208 + 3399, 516115991)

    def test_invalid_context_bounds_fail_without_clipping(self):
        with self.assertRaises(ValueError):
            runner.chunk_plan([109, 51000, 39980], [109, 128, 128], [-1, 0], [2048, 2048])
        with self.assertRaises(ValueError):
            runner.chunk_plan([109, 51000, 39980], [109, 128, 128], [50000, 0], [2048, 2048])

    def test_core_halo_and_physical_sizes(self):
        context, core = SELECTION["context"], SELECTION["core"]
        np.testing.assert_array_equal(np.subtract(core["origin_yx"], context["origin_yx"]), [64, 64])
        np.testing.assert_array_equal(np.subtract(context["end_yx_exclusive"], core["end_yx_exclusive"]), [64, 64])
        np.testing.assert_allclose(np.asarray(core["shape_yx"]) * .0024, [4.608, 4.608])
        np.testing.assert_allclose(np.asarray(context["shape_yx"]) * .0024, [4.9152, 4.9152])

    def test_rejects_wrong_native_sampling_and_layout(self):
        source = PROTOCOL["source"]
        zarray = {"zarr_format": 2, "shape": source["shape_zyx"], "chunks": source["chunks_zyx"],
                  "dtype": "|u1", "order": "C", "compressor": None, "filters": None,
                  "fill_value": 0, "dimension_separator": "/"}
        attrs = {"num_slices": 109, "multiscales": [{
            "axes": [{"name": x, "unit": "micrometer"} for x in ("z", "y", "x")],
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"scale": [2.4, 2.4, 2.4], "type": "scale"},
                {"translation": [0., 0., 0.], "type": "translation"}]}]}]}
        runner.validate_metadata(zarray, attrs, source)
        wrong = copy.deepcopy(attrs)
        wrong["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][1] = 9.6
        with self.assertRaises(ValueError):
            runner.validate_metadata(zarray, wrong, source)
        with self.assertRaises(ValueError):
            runner.validate_metadata(dict(zarray, compressor={"id": "zstd"}), attrs, source)

    def test_pilot_requires_both_donors_and_strict_speed_gain(self):
        settings = PROTOCOL["precision_pilot"]
        records = {(d, p): {"scores": np.zeros((2, 2), dtype=np.float32),
                           "seconds": 2. if p == "fp32" else 1.}
                   for d in settings["inputs"] for p in ("fp32", "bf16")}
        self.assertEqual(runner.pilot_decision(records, settings)["selected_precision"], "bf16")
        records[("PHerc0814", "bf16")]["scores"][0, 0] = .006
        self.assertEqual(runner.pilot_decision(records, settings)["selected_precision"], "fp32")
        records[("PHerc0814", "bf16")]["scores"][:] = 0
        for d in settings["inputs"]:
            records[(d, "bf16")]["seconds"] = 2.
        self.assertEqual(runner.pilot_decision(records, settings)["selected_precision"], "fp32")


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Torch is needed only for generated replay checks")
class GeneratedTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.hecate = runner.module(HERE.parent / "ink/third_party/hecate/hecate.py", "glyph_test_released_hecate")

    def test_official_grid_has_225_forwards(self):
        positions = self.hecate.axis_starts(2048, 256, 128)
        self.assertEqual(len(positions), 15)
        self.assertEqual((positions[0], positions[-1]), (0, 1792))
        self.assertEqual(len(positions) ** 2, 225)

    def test_inner_cpu_autocast_adapter_preserves_official_assembly(self):
        torch, hecate = self.torch, self.hecate

        class FakeLogits(torch.nn.Module):
            def __init__(self, force_quantized=False):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
                self.patch_size, self.divisor, self.margin = (64, 256, 256), 200., 1
                self.force_quantized, self.seen = force_quantized, []

            def forward(self, image, valid):
                active = torch.is_autocast_enabled("cpu")
                self.seen.append(active)
                values = torch.linspace(-3, 3, 256 * 256).reshape(1, 1, 256, 256)
                return values.to(torch.bfloat16) if active or self.force_quantized else values

        # Analytic generated arrays; no CT and no Hecate weights or backbone.
        volume = np.ones((65, 16, 17), dtype=np.uint8)
        volume[:, 0, 0] = 0
        for reverse in (False, True):
            inner, reference = FakeLogits(), FakeLogits(force_quantized=True)
            adapter = runner.make_cpu_bf16_adapter(torch, inner)
            self.assertFalse(adapter.training)
            self.assertFalse(inner.training)
            self.assertIs(next(adapter.parameters()), inner.anchor)
            actual, expected = np.zeros((16, 17), dtype=np.uint8), np.zeros((16, 17), dtype=np.uint8)
            hecate.predict(adapter, volume, actual, reverse=reverse, precision="fp32")
            hecate.predict(reference, volume, expected, reverse=reverse, precision="fp32")
            self.assertEqual(inner.seen, [True])
            self.assertEqual(reference.seen, [False])
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual[0, 0], 0)

    def test_archive_copies_emitted_values_and_native_dtypes(self):
        torch = self.torch
        with tempfile.TemporaryDirectory(prefix="glyph-archive-contract-") as directory:
            archive = runner.DepthArchive(directory, expected_per_order=1)
            archive.begin_order("source")
            logits = torch.zeros((1, 1, 62, 64, 64), dtype=torch.float32)
            attention = torch.full_like(logits, .25, dtype=torch.bfloat16)
            returned_tuple = (object(), logits)
            fake = SimpleNamespace(
                features_logits=lambda _: returned_tuple,
                canonical=SimpleNamespace(decoder=SimpleNamespace(
                    depth_collapse=SimpleNamespace(attn_conv=torch.nn.Identity()))))
            original, handle = runner.attach_depth_capture(fake, archive)
            self.assertIs(fake.features_logits(None), returned_tuple)
            self.assertIs(fake.canonical.decoder.depth_collapse.attn_conv(attention), attention)
            archive.finish_forward(torch.zeros((1, 1, 256, 256)))
            handle.remove()
            fake.features_logits = original
            result = archive.finish_order()
            self.assertEqual(result["patch_count"], 1)
            self.assertEqual(result["originating_dtypes"]["attention_scores"], "torch.bfloat16")
            path = Path(directory) / "source_patch_probability.npy"
            self.assertTrue((np.load(path, allow_pickle=False) == .5).all())
            self.assertTrue((logits == 0).all())
            self.assertTrue((attention == .25).all())
            self.assertLess(archive.total_bytes, 3_000_000)


if __name__ == "__main__":
    unittest.main()
