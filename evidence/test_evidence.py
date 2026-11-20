from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import zarr

from fetch_fixture import build_w00, fetch


class EvidenceTests(unittest.TestCase):
    def test_verified_cache_can_be_used_offline_and_corruption_is_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / 'source.bin'
            destination = folder / 'cache' / 'source.bin'
            source.write_bytes(b'known bytes')
            checksum = sha256(source.read_bytes()).hexdigest()
            fetch(source.as_uri(), checksum, destination)
            source.unlink()
            self.assertEqual(fetch(source.as_uri(), checksum, destination).read_bytes(), b'known bytes')
            source.write_bytes(b'changed upstream bytes')
            destination.write_bytes(b'corrupt cache bytes')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                fetch(source.as_uri(), checksum, destination)

    def test_crop_preserves_fine_and_coarse_voxels_with_different_origins(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / 'source'
            objects, arrays = {}, {}
            shape = (28, 128, 128)
            for level, iy, ix in [('0', 1, 1), ('3', 0, 0)]:
                z, y, x = np.indices(shape, dtype=np.uint16)
                voxels = ((z * 13 + (y + iy * 128) * 7 + x + ix * 128) % 251).astype('u1')
                arrays[level] = voxels
                metadata = {'chunks': list(shape), 'compressor': None, 'order': 'C',
                            'dtype': '|u1', 'fill_value': 0}
                files = [(level + '/.zarray', json.dumps(metadata).encode()),
                         (f'{level}/0/{iy}/{ix}', voxels.tobytes())]
                for key, data in files:
                    path = source / key
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    objects[key] = sha256(data).hexdigest()
            case = {'source_bounds_yx': [128, 256, 128, 256], 'objects': objects,
                    'base_url': source.as_uri() + '/'}
            output = folder / 'output.zarr'
            build_w00(case, folder / 'cache', output)
            root = zarr.open_group(output, mode='r')
            np.testing.assert_array_equal(root['0'][:], arrays['0'])
            np.testing.assert_array_equal(root['3'][:], arrays['3'][:, 16:32, 16:32])


if __name__ == '__main__':
    unittest.main()
