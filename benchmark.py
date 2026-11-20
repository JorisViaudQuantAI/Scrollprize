"""Compare Villa's baseline and patched ROI schedulers on a local surface Zarr."""
import argparse
import ast
from dataclasses import astuple, dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import statistics
import subprocess
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import zarr

BASE = 'c8f3ba45304f0a317a49a155929e30d294f201eb'
INFER = 'vesuvius/src/vesuvius/ink_detection/inference/infer.py'
FUNCTIONS = {
    'Block', '_sliding_positions_1d', 'resolve_roi_bounds', 'iter_blocks',
    'downsample_mask_any', 'choose_pyramid_array', 'compute_nonempty_mask_from_lowres_array',
    'build_lowres_block_mask', '_occupancy_layout', '_occupancy_context', '_volume_axes',
}


def load_scheduler(checkout, baseline):
    path = checkout / INFER
    if baseline:
        source = subprocess.check_output(['git', '-C', str(checkout), 'show', f'{BASE}:{INFER}'], text=True)
    else:
        source = path.read_text()
    source_tree = ast.parse(source)
    nodes = [node for node in source_tree.body if getattr(node, 'name', None) in FUNCTIONS]
    nodes += [n for n in source_tree.body if isinstance(n, ast.Assign)
              and any(getattr(t, 'id', '') == 'DEFAULT_OCCUPANCY_SCAN_LEVEL' for t in n.targets)]
    tree = ast.Module(body=nodes, type_ignores=[])
    scope = dict(np=np, Any=Any, Sequence=Sequence, dataclass=dataclass,
                 math=math, LOGGER=logging.getLogger('villa-roi-benchmark'))
    exec(compile(tree, str(path), 'exec'), scope)
    if not baseline and '_occupancy_context' not in scope:
        raise ValueError('The working checkout does not contain flat-roi.patch')
    return scope


class TrackedArray:
    def __init__(self, array, ledger):
        self.array = array
        self.shape = array.shape
        self.chunks = array.chunks
        self.ledger = ledger

    def __getitem__(self, key):
        if isinstance(key, slice):
            key = (key,) * len(self.shape)
        parts = [s.indices(n) for s, n in zip(key, self.shape, strict=True)]
        voxels = math.prod(len(range(a, b, step)) for a, b, step in parts)
        chunks = math.prod((max(0, b - 1) // chunk - a // chunk + 1) if b > a else 0
                           for (a, b, _), chunk in zip(parts, self.chunks, strict=True))
        self.ledger['logical_bytes'] += voxels * self.array.dtype.itemsize
        self.ledger['chunk_positions'] += chunks
        return self.array[key]


class TrackedGroup:
    def __init__(self, group, ledger):
        self.group = group
        self.ledger = ledger

    def array_keys(self):
        return self.group.array_keys()

    def __getitem__(self, name):
        return TrackedArray(self.group[name], self.ledger)


def plan(scheduler, node, args, baseline):
    counts = {'logical_bytes': 0, 'chunk_positions': 0}
    root = TrackedArray(node, counts) if hasattr(node, 'shape') else TrackedGroup(node, counts)
    _, level = scheduler['choose_pyramid_array'](root, preferred_key=args.resolution, purpose='input')
    _, _, height, width, _, _ = scheduler['_volume_axes'](level)
    roi = scheduler['resolve_roi_bounds'](args.roi, height=height, width=width)
    mask_options = dict(height=height, width=width, user_mask=None)
    options = dict(roi_bounds=roi)
    start = perf_counter()
    if not baseline:
        bounds = scheduler['_occupancy_context'](roi, args.patch_size, height, width)
        mask_options['scan_bounds'] = bounds
    mask, scale, mask_level = scheduler['build_lowres_block_mask'](root, **mask_options)
    if not baseline:
        options['mask_origin'] = (bounds[0] // scale[0], bounds[2] // scale[1])
    blocks = scheduler['iter_blocks']((height, width), args.patch_size, args.stride, mask, scale, **options)
    elapsed = perf_counter() - start
    ordered = [astuple(block) for block in blocks]
    digest = hashlib.sha256(json.dumps(ordered).encode()).hexdigest()
    return {**counts, 'mask_level': mask_level, 'blocks': len(ordered), 'schedule_sha256': digest,
            'seconds': elapsed}, ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path, help='patched Villa checkout at pinned base commit')
    parser.add_argument('surface', type=Path, help='local flat Zarr group or rank-3 surface array')
    parser.add_argument('--roi', required=True, nargs=4, type=int, metavar=('R0', 'R1', 'C0', 'C1'))
    parser.add_argument('--resolution', default='0')
    parser.add_argument('--patch-size', type=int, default=384, help='square model patch width in pixels')
    parser.add_argument('--stride', type=int, help='parent-grid stride, default half the patch size')
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if args.stride is None:
        args.stride = max(1, args.patch_size // 2)
    if min(args.runs, args.stride, args.patch_size) < 1:
        parser.error('runs, patch size and stride must be positive')
    node = zarr.open(str(args.surface), mode='r')
    schedulers = {name: load_scheduler(args.checkout, name == 'before') for name in ('before', 'after')}
    runs = {'before': [], 'after': []}
    for _ in range(args.runs):
        baseline = None
        for name, scheduler in schedulers.items():
            result, blocks = plan(scheduler, node, args, name == 'before')
            runs[name].append(result)
            if name == 'before':
                baseline = blocks
            elif blocks != baseline:
                raise AssertionError('Ordered schedules differ')
    report = {'base': BASE, 'runs': {}, 'identical_ordered_schedules': True}
    for name, values in runs.items():
        times = [row['seconds'] for row in values]
        report['runs'][name] = {k: v for k, v in values[0].items() if k != 'seconds'}
        report['runs'][name].update(seconds=times, median_seconds=statistics.median(times))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
