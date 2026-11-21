import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import tifffile

from surface_patch_mask import run, select_patches


def write_mesh(path, pts, frame):
    path.mkdir()
    (path / "meta.json").write_text(json.dumps({"volume_id": frame}))
    for i, axis in enumerate("xyz"):
        tifffile.imwrite(path / f"{axis}.tif", pts[..., i].astype("float32"))


class PatchMaskTests(unittest.TestCase):
    def test_footprints_match_direct_slicing(self):
        rng = np.random.default_rng(42)
        valid = rng.random((13, 19)) > .04
        known = rng.random(valid.shape) > .78
        excluded = valid & (rng.random(valid.shape) > .93)
        for patch, stride in [((3, 4), (1, 2)), ((4, 3), (2, 1)), ((1, 1), (3, 4)), ((20, 5), (1, 1))]:
            h, w = patch
            expected = []
            for y in range(0, valid.shape[0]-h+1, stride[0]):
                for x in range(0, valid.shape[1]-w+1, stride[1]):
                    region = np.s_[y:y+h, x:x+w]
                    if valid[region].all() and known[region].any() and not excluded[region].any():
                        expected.append((y, x))
            origins, _ = select_patches(valid, excluded, known, patch, stride)
            self.assertEqual(list(map(tuple, origins)), expected)

    def test_labels_center_and_frame(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            y, x = np.mgrid[:6, :6]
            coords = np.stack((x, y, np.full_like(x, 9)), axis=-1)
            write_mesh(root/"target", coords, "scan")
            write_mesh(root/"reference", coords[:1, :1], "scan")
            write_mesh(root/"wrong", coords[:1, :1], "different")
            labels = np.zeros((6, 6), bool)
            labels[0, 0] = labels[2, 2] = labels[5, 5] = True
            ink = labels.copy()
            ink[5, 5] = False
            path = root/"labels.npz"
            np.savez(path, known=labels, ink=ink)
            args = (root/"target", [root/"reference"], "scan", 0, (3, 3), (3, 3))
            fields, stats = run(*args, path)
            self.assertEqual(stats["windows"], dict(candidate=1, center_only=1, center_only_with_overlap=0, retained=1))
            self.assertEqual(fields["origins_yx"].tolist(), [[3, 3]])
            self.assertFalse(fields["known"][0, 0])
            self.assertTrue(fields["known"][5, 5])
            self.assertFalse(fields["ink"][5, 5])
            self.assertFalse(fields["known"][2, 2])
            self.assertFalse(fields["ink"][2, 2])
            self.assertEqual(run(*args)[1]["windows"], dict(candidate=4, center_only=4, center_only_with_overlap=1, retained=3))
            self.assertNotIn("known", run(*args)[0])
            with self.assertRaisesRegex(ValueError, "volume_id"):
                run(root/"target", [root/"wrong"], "scan", 0, (3, 3), (3, 3))


if __name__ == "__main__":
    unittest.main()
