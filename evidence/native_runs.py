"""Time ROI planning in fresh Villa processes and compare the ordered tile lists."""

import argparse
from dataclasses import astuple
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
from time import perf_counter


def run_stage(case, volume, patched):
    from vesuvius.ink_detection.inference import infer
    from vesuvius.ink_detection.volume_io import open_volume_root

    root = open_volume_root(str(volume.resolve()))
    array = root['0']
    backend = type(array).__module__ + '.' + type(array).__name__
    if not backend.startswith('vc_reader.'):
        raise RuntimeError(f'Native benchmark requires vc_reader, selected {backend}')
    _, _, height, width, _, _ = infer._volume_axes(array)
    roi = tuple(case['roi_yx'])
    mask_options = dict(height=height, width=width, user_mask=None)
    tile_options = dict(roi_bounds=roi)
    start = perf_counter()
    if patched:
        bounds = infer._occupancy_context(roi, 384, height, width)
        mask_options['scan_bounds'] = bounds
    mask, scale, level = infer.build_lowres_block_mask(root, **mask_options)
    if patched:
        tile_options['mask_origin'] = (bounds[0] // scale[0], bounds[2] // scale[1])
    blocks = infer.iter_blocks((height, width), 384, 192, mask, scale, **tile_options)
    seconds = perf_counter() - start
    rows = [astuple(block) for block in blocks]
    # ru_maxrss is measured in bytes on macOS and KiB on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        'seconds': seconds,
        'peak_rss_bytes': rss if sys.platform == 'darwin' else rss * 1024,
        'backend': backend,
        'occupancy_level': level,
        'mask_shape': list(mask.shape),
        'tiles': len(rows),
        'ordered_tiles_sha256': sha256(json.dumps(rows).encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('patched', type=Path)
    parser.add_argument('case', choices=['w058', 'w00'])
    parser.add_argument('surface', type=Path)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--worker', choices=['before', 'after'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('sources.json').read_text())
    case = manifest['cases'][args.case]
    if args.worker:
        print(json.dumps(run_stage(case, args.surface, args.worker == 'after')))
        return
    if args.runs < 1 or args.baseline.resolve() == args.patched.resolve():
        parser.error('use distinct checkouts and at least one run')
    trials = {'before': [], 'after': []}
    for _ in range(args.runs):
        for name, checkout in [('before', args.baseline), ('after', args.patched)]:
            environment = os.environ.copy()
            py_path = str(checkout.resolve() / 'vesuvius' / 'src')
            if environment.get('PYTHONPATH'):
                py_path += os.pathsep + environment['PYTHONPATH']
            environment.update(PYTHONPATH=py_path, OPENBLAS_NUM_THREADS='1',
                               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            command = [sys.executable, __file__, str(args.baseline), str(args.patched),
                       args.case, str(args.surface), '--worker', name]
            output = subprocess.check_output(command, env=environment, text=True)
            trials[name].append(json.loads(output.splitlines()[-1]))
    hashes = {row['ordered_tiles_sha256'] for rows in trials.values() for row in rows}
    if len(hashes) != 1:
        raise AssertionError('Ordered tile lists differ')
    report = {'case': args.case, 'roi_yx': case['roi_yx'],
              'identical_ordered_tiles': True, 'stages': {}}
    for name, rows in trials.items():
        seconds = [row['seconds'] for row in rows]
        rss = [row['peak_rss_bytes'] for row in rows]
        report['stages'][name] = {
            **rows[0],
            'seconds': seconds,
            'peak_rss_bytes': rss,
            'median_seconds': statistics.median(seconds),
            'median_peak_rss_bytes': statistics.median(rss),
        }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
