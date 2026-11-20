"""Run two Villa CLIs on one region and compare decoded TIFF pixels and geometry."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import tifffile


def run_cli(checkout, output, args):
    environment = os.environ.copy()
    path = str(checkout.resolve() / 'vesuvius' / 'src')
    if environment.get('PYTHONPATH'):
        path += os.pathsep + environment['PYTHONPATH']
    environment['PYTHONPATH'] = path
    command = [sys.executable, '-m', 'vesuvius.ink_detection.inference.infer', str(args.surface.resolve()),
               str(args.checkpoint.resolve()), str(output), '--roi', *map(str, args.roi),
               '--direction', args.direction, '--resolution', args.resolution, '--device', 'cpu',
               '--no-compile', '--num-workers', '0', '--batch-size', '1', '--blend-mode', 'hann',
               '--overlap', '0.5']
    subprocess.run(command, env=environment, check=True, cwd=output.parent)
    geometry = json.loads(output.with_suffix('.tif.roi.json').read_text())
    return tifffile.imread(output), geometry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'patched', 'surface', 'checkpoint'):
        p.add_argument(name, type=Path)
    p.add_argument('--roi', nargs=4, required=True, type=int, metavar=('R0', 'R1', 'C0', 'C1'))
    p.add_argument('--direction', choices=('forward', 'reverse'), default='forward')
    p.add_argument('--resolution', default='0')
    args = p.parse_args()
    if args.baseline.resolve() == args.patched.resolve():
        p.error('pass two distinct checkouts')
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        before, meta_before = run_cli(args.baseline, directory / 'before.tif', args)
        after, meta_after = run_cli(args.patched, directory / 'after.tif', args)
    r0, r1, c0, c1 = args.roi
    if before.shape != after.shape or before.shape != (r1 - r0, c1 - c0) or before.dtype != after.dtype:
        raise AssertionError('Inconsistent output geometry or dtype')
    for key in ('parent_shape_yx', 'roi_row_start_stop_col_start_stop', 'patch_size', 'stride',
                'input_resolution', 'direction', 'layer_indices', 'parent_grid_retained'):
        if meta_before[key] != meta_after[key]:
            raise AssertionError(f'ROI sidecar fields differ at {key}')
    mismatch = int(np.count_nonzero(before != after))
    print(json.dumps({'pixels': int(before.size), 'shape_yx': list(before.shape), 'dtype': str(before.dtype),
                      'different_pixels': mismatch, 'geometry_matches': True}, indent=2))
    if mismatch:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
