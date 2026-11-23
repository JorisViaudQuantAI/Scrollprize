"""Small deterministic tests of blend normalization and real seam coordinates."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hann_panel import seam_ratio
import r50_stack
from r50_model import DEPTH, WIDTH
from r50_stack import depth_order, mosaic, tile_starts


def mock_tile(tile):
    x = np.arange(WIDTH, dtype=np.float32)
    return np.asarray((x[:, None]+x[None, :]+tile[0, 0, 0]) / 768, dtype=np.float32)


class BlendTest(unittest.TestCase):
    def setUp(self):
        plane = np.random.default_rng(841).integers(0, 256, (509, 543), dtype=np.uint8)
        self.stack = np.broadcast_to(plane, (DEPTH, *plane.shape))
        self.order = depth_order('asc')

    def test_uniform_default_is_unchanged_and_hann_matches_explicit_sum(self):
        for stride in (128, 256):
            unweighted, meta = mosaic(self.stack, self.order, stride, mock_tile)
            explicit, explicit_meta = mosaic(self.stack, self.order, stride, mock_tile, blend='uniform')
            np.testing.assert_array_equal(unweighted, explicit)
            self.assertEqual(meta, explicit_meta)
            self.assertEqual(meta['blend'], 'arithmetic_mean')
            value, desc = mosaic(self.stack, self.order, stride, mock_tile, blend='hann')
            self.assertEqual(desc['blend'], 'hann')
            win = np.sin(np.pi * (np.arange(WIDTH)+1)/(WIDTH+1)) ** 2
            weights = win[:, None] * win[None, :]
            numerator = np.zeros(self.stack.shape[1:])
            denominator = np.zeros_like(numerator)
            for y in tile_starts(value.shape[0], stride):
                for x in tile_starts(value.shape[1], stride):
                    select = np.s_[y:y+WIDTH, x:x+WIDTH]
                    numerator[select] += weights * mock_tile(self.stack[:, select[0], select[1]])
                    denominator[select] += weights
            self.assertTrue(np.all(denominator > 0))
            np.testing.assert_allclose(value, numerator/denominator, rtol=0, atol=6e-8)
            self.assertTrue(np.isfinite(value).all())

    def test_single_tile_and_nonoverlapping_tiles(self):
        for height, width in ((256, 256), (256, 512)):
            s = self.stack[:, :height, :width]
            u, _ = mosaic(s, self.order, 256, mock_tile)
            h, _ = mosaic(s, self.order, 256, mock_tile, blend='hann')
            np.testing.assert_array_equal(h, u)
        with self.assertRaises(ValueError):
            mosaic(self.stack, self.order, 128, mock_tile, blend='other')

    def test_cli_manifest_and_default(self):
        with tempfile.TemporaryDirectory() as scratch:
            folder = Path(scratch)
            source = folder / 'synthetic.npy'
            small = self.stack[:, :272, :278]
            np.save(source, small)
            for mode in ('uniform', 'hann'):
                target = folder / (mode+'.npz')
                argv = ['r50_stack.py', 'predict', str(source), str(target), '--cache', str(folder), '--threads', '1']
                if mode == 'hann':
                    argv += ['--blend', 'hann']
                with (patch.object(sys, 'argv', argv), patch.object(r50_stack, 'load_model', return_value=None),
                      patch.object(r50_stack, 'predict_tile', side_effect=lambda model, tile: mock_tile(tile)),
                      patch.object(r50_stack.torch, 'set_num_threads'), patch.object(r50_stack.torch, 'set_num_interop_threads'),
                      contextlib.redirect_stdout(io.StringIO())):
                    r50_stack.main()
                with np.load(target, allow_pickle=False) as bundle:
                    manifest = json.loads(str(bundle['manifest']))
                    image = bundle['probability']
                expected, geometry = mosaic(small, self.order, 128, mock_tile, blend=mode)
                np.testing.assert_array_equal(image, expected)
                self.assertEqual(manifest['blend'], geometry['blend'])
                self.assertEqual('hann_window' in manifest, mode == 'hann')

    def test_seams_use_coordinate_changes_between_adjacent_pixels(self):
        y, x = np.indices((512, 512))
        mask = np.ones_like(x, dtype=bool)
        grid = [[0, 128, 256, 384]]*2
        smooth = (x+y).astype(np.float32)
        smooth_result = seam_ratio(smooth, mask, grid, 64)
        self.assertEqual(smooth_result['ratio'], 1)
        self.assertEqual(smooth_result['lines_yx'], [[64, 192, 320, 448]]*2)
        edge = smooth + 10*(x >= 192) + 20*(y >= 448)
        marked = seam_ratio(edge, mask, grid, 64)
        self.assertAlmostEqual(marked['ratio'], 1+(10+20)/8)
        self.assertEqual(marked['pairs'], 8*512)


if __name__ == '__main__':
    unittest.main()
