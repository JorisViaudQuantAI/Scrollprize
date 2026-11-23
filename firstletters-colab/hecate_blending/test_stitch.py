"""Check alignment and the actual released export using deterministic fake logits.

No trained checkpoint, CT source or actual detector forward is used by these tests.
"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runner = load(HERE / "run.py", "test_blend_runner")
hecate = load(HERE.parent / "ink/third_party/hecate/hecate.py", "test_blend_official")


class FakeLogits(torch.nn.Module):
    patch_size = (64, 256, 256)
    divisor = 200.

    def __init__(self, logits):
        super().__init__()
        self.stub = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.logits = logits
        self.calls = 0

    def forward(self, image, valid):
        if image.shape != (1, 1, 64, 256, 256):
            raise ValueError("Unexpected predictor patch")
        value = self.logits[self.calls][None, None]
        self.calls += 1
        return value


class TestStitch(unittest.TestCase):
    def test_official_predict_export_alignment_support_and_quantization(self):
        torch.set_num_threads(2)
        y, x = np.mgrid[:256, :256]
        base = (np.sin(x / 19) + np.cos(y / 31)).astype(np.float32)
        logits = torch.from_numpy(np.stack([base + np.float32((i-31.5)/24) for i in range(64)]))
        # Holes include one complete common cell and one partial common cell.
        volume = np.ones((64, 1152, 1152), dtype=np.uint8)
        volume[:, 100:104, 100:104] = 0
        volume[:, 218:221, 313:316] = 0
        stitch = runner.Stitch(runner.make_window(torch))
        for index in range(64):
            yy, xx = index // 8 * 128, index % 8 * 128
            probability = logits[index].sigmoid().numpy()
            stitch.add(index, probability, volume[:, yy:yy+256, xx:xx+256].any(0))
        maps = stitch.finish()
        yy, xx = np.mgrid[:1024, :1024]
        expected_central = logits.sigmoid().numpy()[(yy // 128) * 8 + xx // 128, 64 + yy % 128, 64 + xx % 128]
        expected_central *= volume.any(0)[64:1088, 64:1088]
        np.testing.assert_array_equal(maps["central"], expected_central)
        fake = FakeLogits(logits)
        official = np.empty((1152, 1152), dtype=np.uint8)
        hecate.predict(fake, volume, official, reverse=True, batch_size=1, precision="fp32")
        self.assertEqual(fake.calls, 64)
        ours = np.rint(maps["hann"].clip(0, 1) * np.float32(255)).astype(np.uint8)
        np.testing.assert_array_equal(ours, official[64:1088, 64:1088])
        self.assertTrue(np.all(ours[36:40, 36:40] == 0))

    def test_incomplete_or_reordered_grid_fails(self):
        stitch = runner.Stitch(runner.make_window(torch))
        probability = np.zeros((256, 256), dtype=np.float32)
        support = np.ones((256, 256), dtype=bool)
        with self.assertRaises(ValueError):
            stitch.add(1, probability, support)
        stitch.add(0, probability, support)
        with self.assertRaises(ValueError):
            stitch.finish()


if __name__ == "__main__":
    unittest.main()
