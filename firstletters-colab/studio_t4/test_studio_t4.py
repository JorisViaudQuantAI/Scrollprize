"""CPU tests for studio_t4.py: no network, no GPU, no real weights.

The one test with the real network uses random weights and needs a checkout of the release repository
(set STUDIO_RELEASE_DIR); it is skipped otherwise.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("studio_t4", HERE / "studio_t4.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)

spec = importlib.util.spec_from_file_location("studio_inputs_prepare", HERE.parent / "studio_inputs" / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

try:
    import torch
except ImportError:  # the tiling tests need torch; plane and file tests do not
    torch = None
needs_torch = unittest.skipUnless(torch is not None, "torch not installed")


def plane_coded(n_layers=41, h=8, w=9):
    """Layer k holds the value k + 1 everywhere."""
    return (np.arange(n_layers, dtype=np.uint8)[:, None, None] + 1) * np.ones((1, h, w), np.uint8)


def surface(h=230, w=310, seed=0):
    """Random 16-plane surface with no CT data outside an ellipse."""
    rng = np.random.default_rng(seed)
    v = rng.integers(1, 255, (16, h, w), dtype=np.uint8)
    yy, xx = np.mgrid[:h, :w]
    v[:, (yy - h / 2) ** 2 / (h / 2.5) ** 2 + (xx - w / 2) ** 2 / (w / 2.5) ** 2 > 1] = 0
    return v


if torch is not None:
    class Pointwise(torch.nn.Module):
        """Stand-in for the U-Net: a per-pixel function of the 16 planes (no spatial context)."""

        def __init__(self):
            super().__init__()
            self.windows = 0

        def forward(self, x):            # x: [B,1,16,128,128] -> [B,1,128,128]
            self.windows += x.shape[0]
            return (x[:, :, :8].mean(2) - x[:, :, 8:].mean(2)) * 6 + x.std(2) - 0.3

    class Constant(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 1, 128, 128), 0.7, dtype=x.dtype, device=x.device)

    def direct(volume, model):
        """The stand-in model on the whole image at once, release normalisation."""
        x = torch.from_numpy(volume[None, None].astype(np.float32)).clamp(0, 200).div(255)
        with torch.inference_mode():
            p = model(x)[0, 0].sigmoid().numpy()
        p[(volume == 0).all(0)] = 0
        return p


class PlaneSelection(unittest.TestCase):
    def test_documented_orders(self):
        self.assertEqual(st.plane_indices(31, 15, "desc"), list(range(23, 7, -1)))
        self.assertEqual(st.plane_indices(31, 15, "asc"), list(range(7, 23)))
        self.assertEqual(st.plane_indices(41, 20, "desc"), list(range(28, 12, -1)))
        self.assertEqual(st.plane_indices(41, 20, "asc"), list(range(12, 28)))

    def test_shifts(self):
        self.assertEqual(st.plane_indices(41, 20, "desc", 2), list(range(30, 14, -1)))
        self.assertEqual(st.plane_indices(41, 20, "desc", -2), list(range(26, 10, -1)))
        self.assertEqual(st.plane_indices(41, 20, "asc", 2), list(range(14, 30)))
        self.assertEqual(st.plane_indices(41, 20, "asc", -6), list(range(6, 22)))

    def test_out_of_range(self):
        for args in ((31, 15, "desc", 8), (31, 15, "asc", -8), (16, 8, "desc", 0), (41, 20, "desc", 13),
                     (41, 20, "asc", -13), (41, 41, "desc", 0), (41, -1, "asc", 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                st.plane_indices(*args)
        self.assertEqual(st.plane_indices(16, 8, "asc"), list(range(16)))   # exactly fits
        with self.assertRaises(ValueError):
            st.plane_indices(41, 20, "forward")

    def test_matches_studio_inputs_convention(self):
        """desc = studio_inputs 'release-descending'; asc = 'release-descending' on the whole-stack-reversed render."""
        stack = plane_coded()
        kw = dict(xy_spacing_um=(9.36, 9.36), depth_spacing_um=9.36, index_direction=1, order="release-descending")
        for m in (12, 20, 28):
            ours = stack[st.plane_indices(41, m, "desc")]
            theirs, _ = prepare.select_planes(stack, mesh_plane_index=m, **kw)
            np.testing.assert_array_equal(ours, theirs)
            ours = stack[st.plane_indices(41, m, "asc")]
            theirs, _ = prepare.select_planes(np.ascontiguousarray(stack[::-1]), mesh_plane_index=40 - m, **kw)
            np.testing.assert_array_equal(ours, theirs)


class Inputs(unittest.TestCase):
    def test_npy_and_tiff_dir(self):
        import tifffile
        stack = plane_coded(20, 6, 7)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.save(d / "r.npy", stack)
            (d / "tif" / "layers").mkdir(parents=True)
            for k in range(20):
                tifffile.imwrite(d / "tif" / "layers" / f"{k:02d}.tif", stack[k])
            (d / "flat").mkdir()
            for k in range(20):
                tifffile.imwrite(d / "flat" / f"{k:02d}.tif", stack[k])
            for path in (d / "r.npy", d / "tif", d / "flat"):
                r = st.Render(path)
                self.assertEqual((r.n_layers, r.shape), (20, (6, 7)))
                np.testing.assert_array_equal(r.stack([9, 3]), stack[[9, 3]])
            self.assertEqual(st.Render(d / "tif").input_sha256(), st.Render(d / "flat").input_sha256())
            (d / "flat" / "05.tif").unlink()
            with self.assertRaisesRegex(ValueError, "without gaps"):
                st.Render(d / "flat")

    def test_zarr(self):
        try:
            import zarr
        except ImportError:
            self.skipTest("zarr not installed")
        stack = plane_coded(20, 6, 7)
        with tempfile.TemporaryDirectory() as d:
            g = zarr.open_group(str(Path(d) / "r.zarr"), mode="w")
            g.create_array("0", shape=stack.shape, dtype=stack.dtype) if hasattr(g, "create_array") \
                else g.create_dataset("0", shape=stack.shape, dtype=stack.dtype)
            g["0"][:] = stack
            r = st.Render(Path(d) / "r.zarr")
            self.assertEqual(r.kind, "zarr")
            np.testing.assert_array_equal(r.stack([4, 5]), stack[[4, 5]])

    def test_rejects_non_uint8_and_zero_planes(self):
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d) / "f.npy", plane_coded(20).astype(np.uint16))
            with self.assertRaisesRegex(ValueError, "uint8"):
                st.Render(Path(d) / "f.npy")
        stack = plane_coded(20)
        stack[4] = 0
        with self.assertRaisesRegex(ValueError, r"\[4\]"):
            st.check_planes_nonzero(stack[[3, 4, 5]], [3, 4, 5])


@needs_torch
class Tiling(unittest.TestCase):
    def test_constant_is_reconstructed(self):
        v = np.full((16, 230, 310), 90, np.uint8)
        want = 1 / (1 + np.exp(-0.7))
        for stride, blend in ((48, "hann"), (96, "none")):
            with self.subTest(stride=stride):
                p = st.predict(Constant(), v, device="cpu", batch_size=5, amp="fp32", stride=stride, blend=blend)
                np.testing.assert_allclose(p, want, rtol=0, atol=1e-6)

    def test_equals_direct_forward(self):
        v = surface()
        for stride, blend, tol in ((48, "hann", 2e-6), (96, "none", 0)):
            with self.subTest(stride=stride):
                m = Pointwise()
                p = st.predict(m, v, device="cpu", batch_size=7, amp="fp32", stride=stride, blend=blend)
                np.testing.assert_allclose(p, direct(v, Pointwise()), rtol=0, atol=tol + 1e-7)

    def test_stride96_covers_every_pixel_once(self):
        h, w = 300, 410
        counts = np.zeros((h, w), int)
        sy, sx = st.window_starts(h, w, 96, "none")
        for y in sy:
            for x in sx:
                counts[y:y + 96, x:x + 96] += 1
        np.testing.assert_array_equal(counts, 1)
        m = Pointwise()
        st.predict(m, np.full((16, h, w), 50, np.uint8), device="cpu", batch_size=4, amp="fp32", stride=96,
                   blend="none")
        self.assertEqual(m.windows, len(sy) * len(sx))
        self.assertEqual(m.windows, 4 * 5)
        # stride 48 Hann: four windows per interior pixel, at least one everywhere
        counts[:] = 0
        sy, sx = st.window_starts(h, w, 48, "hann")
        for y in sy:
            for x in sx:
                counts[y:y + 96, x:x + 96] += 1
        self.assertTrue((counts >= 1).all())
        self.assertTrue((counts[96:-96, 96:-96] == 4).all())

    def test_validity_mask(self):
        v = surface()
        v[:8, 100, 100] = 0                 # some planes zero: still valid
        v[:, 101, 101] = 0                  # all planes zero: invalid
        for stride, blend in ((48, "hann"), (96, "none")):
            valid = st.validity_mask(v, stride, blend)
            np.testing.assert_array_equal(valid, ~(v == 0).all(0))
            self.assertTrue(valid[100, 100])
            self.assertFalse(valid[101, 101])
            p = st.predict(Pointwise(), v, device="cpu", batch_size=8, amp="fp32", stride=stride, blend=blend)
            self.assertTrue((p[~valid] == 0).all())
        # a window that predict() skips leaves its pixels uncovered
        cov = st.covered_mask(np.zeros((16, 200, 200), np.uint8), 96, "none")
        self.assertFalse(cov.any())


class Weights(unittest.TestCase):
    def test_sha_mismatch_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "w.safetensors"
            with self.assertRaisesRegex(ValueError, "not found"):
                st.verify_weights(w)
            w.write_bytes(b"not the release")
            with self.assertRaisesRegex(ValueError, "does not match"):
                st.verify_weights(w)
            with mock.patch.object(st, "WEIGHTS_SHA256", hashlib.sha256(b"not the release").hexdigest()):
                self.assertEqual(st.verify_weights(w), st.WEIGHTS_SHA256)

    def test_fetch_verifies_and_leaves_nothing_on_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "w.safetensors"
            with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(b"tampered")):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    st.fetch_weights(w)
            self.assertEqual(list(Path(d).iterdir()), [])
            good = b"release bytes"
            with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(good)), \
                    mock.patch.object(st, "WEIGHTS_SHA256", hashlib.sha256(good).hexdigest()):
                st.fetch_weights(w)
            self.assertEqual(w.read_bytes(), good)


class FakeRelease:
    """A temporary 'release checkout', dummy weights and a stand-in loader; pins patched to match."""

    def __init__(self, d):
        self.dir = Path(d) / "release"
        (self.dir / Path(st.MODEL_PY).parent).mkdir(parents=True)
        (self.dir / st.MODEL_PY).write_text("# stand-in\n")
        self.weights = Path(d) / "w.safetensors"
        self.weights.write_bytes(b"dummy weights")
        self.loads = 0

    def patches(self):
        return [mock.patch.object(st, "MODEL_PY_SHA256", st.sha256_file(self.dir / st.MODEL_PY)),
                mock.patch.object(st, "WEIGHTS_SHA256", st.sha256_file(self.weights))]

    def loader(self, release_dir, weights, device):
        self.loads += 1
        return Pointwise()

    def argv(self, inp, prefix, *extra):
        return ["--input", str(inp), "--weights", str(self.weights), "--release-dir", str(self.dir),
                "--out-prefix", str(prefix), "--device", "cpu", *extra]


@needs_torch
class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.rel = FakeRelease(self.d)
        self.stack = np.concatenate([surface(150, 170, seed=k)[:1] for k in range(41)])   # 41 layers
        self.inp = self.d / "render41.npy"
        np.save(self.inp, self.stack)
        self._p = [p.start() for p in self.rel.patches()]

    def tearDown(self):
        mock.patch.stopall()
        self.tmp.cleanup()

    def test_receipt_and_outputs(self):
        prefix = self.d / "out" / "seg"
        r = st.run(self.rel.argv(self.inp, prefix, "--mesh-index", "20", "--order", "desc", "--shift", "1",
                                 "--stride", "96"), model_loader=self.rel.loader)
        paths = st.output_paths(prefix)
        on_disk = json.loads(paths["receipt"].read_text())
        self.assertEqual(on_disk, json.loads(json.dumps(r)))
        for key in ("schema", "tool_sha256", "release", "weights", "input", "planes", "normalisation", "tiling",
                    "precision", "batch_size", "device", "memory", "seconds", "windows", "pixels", "outputs"):
            self.assertIn(key, on_disk)
        self.assertEqual(on_disk["planes"]["source_layers"], list(range(29, 13, -1)))
        self.assertEqual(on_disk["planes"]["order"], "desc")
        self.assertEqual(on_disk["input"]["sha256"], st.sha256_file(self.inp))
        self.assertEqual(on_disk["input"]["stack_sha256"], st.sha256_array(self.stack[list(range(29, 13, -1))]))
        self.assertEqual(on_disk["weights"]["sha256"], st.WEIGHTS_SHA256)
        self.assertEqual((on_disk["precision"], on_disk["batch_size"], on_disk["tiling"]["stride"],
                          on_disk["tiling"]["blend"]), ("fp32", 1, 96, "none"))
        self.assertEqual(on_disk["device"]["device"], "cpu")
        self.assertIn("device_name", on_disk["device"])
        self.assertIn("peak_allocated_mib", on_disk["memory"])
        self.assertIn("peak_reserved_mib", on_disk["memory"])
        self.assertGreaterEqual(on_disk["seconds"]["total"], on_disk["seconds"]["inference"])
        prob = np.load(paths["prob"])
        valid = np.load(paths["valid"])
        self.assertEqual((prob.dtype, prob.shape, valid.dtype), (np.float32, (150, 170), np.bool_))
        self.assertEqual(on_disk["pixels"], dict(height=150, width=170, total=150 * 170, valid=int(valid.sum())))
        from PIL import Image
        png = np.asarray(Image.open(paths["preview"]))
        self.assertEqual((png.dtype, png.shape), (np.uint8, (150, 170)))
        self.assertTrue((png[~valid] == 0).all())
        for k in ("prob", "valid", "preview"):
            self.assertEqual(on_disk["outputs"][k]["sha256"], st.sha256_file(paths[k]))
        self.assertNotIn(self.tmp.name, paths["receipt"].read_text())     # no local paths in the receipt
        np.testing.assert_allclose(prob, direct(self.stack[list(range(29, 13, -1))], Pointwise()), atol=1e-7)

    def test_prepared_input(self):
        np.save(self.d / "p16.npy", self.stack[12:28])
        r = st.run(self.rel.argv(self.d / "p16.npy", self.d / "p", "--prepared"), model_loader=self.rel.loader)
        self.assertEqual(r["planes"]["source_layers"], list(range(16)))
        self.assertEqual(r["tiling"]["blend"], "hann")

    def test_refuses_to_overwrite(self):
        prefix = self.d / "seg"
        st.output_paths(prefix)["receipt"].write_text("{}")
        with self.assertRaises(FileExistsError):
            st.run(self.rel.argv(self.inp, prefix, "--mesh-index", "20", "--order", "asc"),
                   model_loader=self.rel.loader)
        self.assertEqual(self.rel.loads, 0)
        self.assertFalse(st.output_paths(prefix)["prob"].exists())
        with contextlib.redirect_stderr(io.StringIO()) as err, \
                mock.patch.object(st, "load_release_model", self.rel.loader):
            self.assertEqual(st.main(self.rel.argv(self.inp, prefix, "--mesh-index", "20", "--order", "asc")), 2)
        self.assertIn("already exist", err.getvalue())

    def test_weights_mismatch_runs_nothing(self):
        self.rel.weights.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "does not match"):
            st.run(self.rel.argv(self.inp, self.d / "x", "--mesh-index", "20", "--order", "asc"),
                   model_loader=self.rel.loader)
        self.assertEqual(self.rel.loads, 0)
        self.assertFalse(any(p.exists() for p in st.output_paths(self.d / "x").values()))

    def test_release_model_py_mismatch(self):
        (self.rel.dir / st.MODEL_PY).write_text("# edited\n")
        with self.assertRaisesRegex(ValueError, "differs from the pinned file"):
            st.run(self.rel.argv(self.inp, self.d / "y", "--mesh-index", "20", "--order", "asc"),
                   model_loader=self.rel.loader)

    def test_cpu_rejects_half_precision(self):
        for prec in ("fp16", "bf16"):
            with self.subTest(prec=prec), self.assertRaisesRegex(ValueError, "fp32"):
                st.run(self.rel.argv(self.inp, self.d / prec, "--mesh-index", "20", "--order", "asc",
                                     "--precision", prec), model_loader=self.rel.loader)

    def test_auto_precision_is_fp16_on_any_cuda(self):
        for capability in ((7, 5), (8, 9)):
            with self.subTest(capability=capability), \
                    mock.patch.object(torch.cuda, "is_available", return_value=True), \
                    mock.patch.object(torch.cuda, "get_device_capability", return_value=capability):
                self.assertEqual(st.resolve_device_and_precision("auto", "auto"), ("cuda", "fp16"))
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)):
            with self.assertRaisesRegex(ValueError, "no native bf16"):
                st.resolve_device_and_precision("cuda", "bf16")

    def test_plane_band_outside_render(self):
        with self.assertRaisesRegex(ValueError, "outside the stack"):
            st.run(self.rel.argv(self.inp, self.d / "z", "--mesh-index", "20", "--order", "desc", "--shift", "13"),
                   model_loader=self.rel.loader)


RELEASE_DIR = os.environ.get("STUDIO_RELEASE_DIR")


@needs_torch
@unittest.skipUnless(RELEASE_DIR and (Path(RELEASE_DIR) / st.MODEL_PY).is_file(),
                     "set STUDIO_RELEASE_DIR to a checkout of the release repository")
class ReleaseArchitecture(unittest.TestCase):
    """The real network with random weights: our tiling vs the release's own predict(), CPU fp32."""

    def test_matches_release_predict(self):
        try:
            import safetensors  # noqa: F401  (the release inference.py imports it)
        except ImportError:
            self.skipTest("safetensors not installed")
        release = Path(RELEASE_DIR)
        module = st.import_release_model(release)       # checks the pinned model.py hash
        torch.manual_seed(0)
        model = module.DeepResidualUNet(deeper=False).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 223_355_921)
        sys.path.insert(0, str(release / Path(st.MODEL_PY).parent))
        try:
            spec = importlib.util.spec_from_file_location("studio_release_inference",
                                                          release / Path(st.MODEL_PY).parent / "inference.py")
            inference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(inference)
        finally:
            sys.path.pop(0)
        v = surface(96, 150, seed=3)
        with tempfile.TemporaryDirectory() as d:
            for stride, blend in ((96, "none"), (48, "hann")):
                vol = v if stride == 96 else v[:, :, :96]
                with self.subTest(stride=stride):
                    ours = st.predict(model, vol, device="cpu", batch_size=2, amp="fp32", stride=stride, blend=blend)
                    theirs = np.array(inference.predict(model, vol, Path(d) / f"{stride}.npy", device="cpu",
                                                        batch_size=2, amp="fp32", stride=stride, blend=blend))
                    self.assertGreater(float(ours[(vol != 0).any(0)].std()), 0)
                    self.assertEqual(float(np.abs(ours - theirs).max()), 0.0)


if __name__ == "__main__":
    unittest.main()
