"""Compare tile blending on prospectively selected public PHerc. 841 annotations."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import requests
import torch

from r50_model import MODEL, REVISION, WEIGHT_SHA, WIDTH, load_model, predict_tile
from r50_stack import array_sha256, depth_order, mosaic
from r50_tile import LABELS, VOLUME, array, as_mask, metrics

BASE = '329e9552c35d06a52720c8fbb4077005bef29bdc'
PROTOCOL = 'f70ea2d3f8f24ea8e28e9af5b446ce7e09cdadf9'
PREVIOUS = Path(__file__).with_name('pherc0841-depth-result.json')
CORE, HALO, PARENT, STRIDE = 512, 64, 640, 128
TARGETS = (0.35, 0.65)
THRESHOLDS = (0.5, 0.8)


def compact(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':'))


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def integral(mask):
    table = np.zeros((mask.shape[0]+1, mask.shape[1]+1), dtype=np.int32)
    table[1:, 1:] = mask.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    return table


def count_square(table, y, x, side):
    return int(table[y+side, x+side] - table[y+side, x] - table[y, x+side] + table[y, x])


def rectangle(y, x, h, w):
    return y, x, y+h, x+w


def overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def select_fields():
    urls = [LABELS + name + '.zarr' for name in ('inklabels', 'supervision', 'validation')]
    grids = [array(url, 0) for url in urls]
    overview = [as_mask(array(url, 4)[:]) for url in urls]
    shape = tuple(grids[0].shape)
    if len(shape) != 2 or any(tuple(a.shape) != shape for a in grids):
        raise ValueError('Full annotation grids disagree')
    expected_shape = tuple((size+15)//16 for size in shape)
    if any(any(abs(x-y) > 1 for x, y in zip(o.shape, expected_shape)) for o in overview):
        raise ValueError('Unexpected level-4 overview geometry')
    ink, supervision, validation = overview
    trusted = supervision & validation
    known_sums, ink_sums = integral(trusted), integral(ink & trusted)
    previous = [rectangle(11264, 12544, 272, 278)]
    previous.extend(rectangle(p['y'], p['x'], WIDTH, WIDTH)
                    for p in json.loads(PREVIOUS.read_text())['tiles'])
    side = CORE // 16
    candidates = []
    for y in range(256, shape[0]-CORE-HALO+1, 256):
        for x in range(256, shape[1]-CORE-HALO+1, 256):
            parent = rectangle(y-HALO, x-HALO, PARENT, PARENT)
            if any(overlap(parent, p) for p in previous):
                continue
            n = count_square(known_sums, y//16, x//16, side)
            if n >= 0.70*side**2:
                pos = count_square(ink_sums, y//16, x//16, side)
                candidates.append((y, x, pos/n, n, parent))

    selected, annotations = [], {}

    def read_labels(y, x):
        box = np.s_[y:y+CORE, x:x+CORE]
        ink, supervision, validation = (as_mask(g[box]) for g in grids)
        return ink, supervision & validation

    for minimum_trust, low, high in ((0.9, 0.2, 0.8), (0.75, 0.1, 0.9)):
        for target in TARGETS:
            if any(row['target_ink_fraction'] == target for row in selected):
                continue
            ranked = sorted(candidates, key=lambda v: (abs(v[2]-target), -v[3], v[0], v[1]))
            eligible = (row for row in ranked if not any(
                overlap(row[4], s['parent_yxyx']) for s in selected))
            for attempt, (y, x, _, _, parent) in enumerate(eligible):
                if attempt >= 100:
                    break
                ink, mask = read_labels(y, x)
                n, pos = int(mask.sum()), int((ink & mask).sum())
                if n < minimum_trust*CORE**2 or not low <= pos/n <= high:
                    continue
                selected.append(dict(core_y=y, core_x=x, parent_yxyx=parent, trusted=n, ink=pos,
                                     target_ink_fraction=target, minimum_trust=minimum_trust,
                                     ink_fraction=pos/n, ink_sha256=array_sha256(ink),
                                     trusted_sha256=array_sha256(mask)))
                annotations[(y, x)] = ink, mask
                break
    return selected, annotations, shape


def load_previous_mosaic(directory):
    source = 'https://raw.githubusercontent.com/JorisViaudQuantAI/Scrollprize/' + BASE + '/public-controls/r50_stack.py'
    response = requests.get(source, timeout=25)
    response.raise_for_status()
    data = response.content
    digest = hashlib.sha256(data).hexdigest()
    if digest != 'adc803e12cfa2dce66159d843e76dec87bcefa62d8f54e159a9a5bdb8583ae83':
        raise ValueError('Unexpected historical mosaic implementation ' + digest)
    file = Path(directory) / 'previous_r50_stack.py'
    file.write_bytes(data)
    spec = importlib.util.spec_from_file_location('previous_r50_stack', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mosaic


def phase_maps(stack, model, legacy):
    order = depth_order('asc')
    probabilities = []

    def run_once(tile):
        result = predict_tile(model, tile)
        probabilities.append(result)
        return result

    uniform, geometry = mosaic(stack, order, STRIDE, run_once)

    def replay(function, **options):
        stream = iter(probabilities)
        result, metadata = function(stack, order, STRIDE, lambda tile: next(stream), **options)
        if next(stream, None) is not None:
            raise AssertionError('Reassembly skipped tile predictions')
        return result, metadata

    historical, historical_geometry = replay(legacy)
    if geometry != historical_geometry or not np.array_equal(uniform, historical):
        raise AssertionError('Default uniform output diverged from the frozen implementation')
    hann, _ = replay(mosaic, blend='hann')
    return {'uniform': uniform, 'hann': hann}, geometry, array_sha256(historical)


def binary_counts(prob, ink, mask, threshold):
    detected = prob[mask] >= threshold
    truth = ink[mask]
    tp = int((detected & truth).sum())
    fp = int((detected & ~truth).sum())
    fn, tn = int(truth.sum()) - tp, int((~truth).sum()) - fp
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=divide(tp, tp+fn), fpr=divide(fp, fp+tn),
                dice=divide(2*tp, 2*tp+fp+fn))


def phase_agreement(first, second, mask):
    a, b = first[mask], second[mask]
    differences = np.abs(a.astype(np.float64) - b)
    overlap = {}
    for threshold in THRESHOLDS:
        aa, bb = a >= threshold, b >= threshold
        overlap[str(threshold)] = divide(int(2*(aa & bb).sum()), int(aa.sum()+bb.sum()))
    return dict(n=len(a), mean_absolute=float(differences.mean()) if len(a) else None,
                p95_absolute=float(np.quantile(differences, .95)) if len(a) else None, dice=overlap)


def seam_ratio(image, mask, starts, origin):
    boundary_sum = control_sum = 0.0
    count = 0
    lines = []
    for axis in range(2):
        view = np.moveaxis(image, axis, 1)
        support = np.moveaxis(mask, axis, 1)
        candidates = {s+off-origin for s in starts[axis] for off in (0, WIDTH)}
        edges = sorted(s for s in candidates if s-17 >= 0 and s+16 < view.shape[1])
        lines.append(edges)
        for s in edges:
            needed = [s+k for k in (-17, -16, -1, 0, 15, 16)]
            ok = support[:, needed].all(1)

            def jump(column):
                return np.abs(view[ok, column].astype(np.float64) - view[ok, column-1]).sum()
            boundary_sum += jump(s)
            control_sum += (jump(s-16) + jump(s+16)) / 2
            count += int(ok.sum())
    return dict(pairs=count, lines_yx=lines, seam_mean=divide(boundary_sum, count),
                control_mean=divide(control_sum, count), ratio=divide(boundary_sum, control_sum))


def measure(maps, grids, ink, mask, border):
    span = slice(border, CORE-border)
    ink, mask = ink[span, span], mask[span, span]
    result = dict(trusted=int(mask.sum()), positives=int((ink & mask).sum()), blends={})
    for blend in ('uniform', 'hann'):
        a, b = (maps[name][blend][span, span] for name in ('A', 'B'))
        row = {'phase_agreement': phase_agreement(a, b, mask), 'by_phase': {}}
        for name, probability in (('A', a), ('B', b)):
            ranks = metrics(probability, ink, mask)
            row['by_phase'][name] = dict(
                auc=ranks.get('auc'), thresholds={str(t): binary_counts(probability, ink, mask, t) for t in THRESHOLDS},
                seam=seam_ratio(probability, mask, grids[name], HALO+border),
                probability_sha256=array_sha256(probability))
        result['blends'][blend] = row
    return result


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--cache', type=Path, required=True)
    args.add_argument('--output', type=Path, required=True)
    args = args.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    selected, annotations, shape = select_fields()
    print('FROZEN_HANN_FIELDS ' + compact(selected), flush=True)
    selection_hash = hashlib.sha256(compact(selected).encode()).hexdigest()
    output = dict(protocol=PROTOCOL, base=BASE, source_code=os.environ.get('GITHUB_SHA'),
                  model=MODEL, model_revision=REVISION, weight_sha256=WEIGHT_SHA,
                  source_labels=LABELS, source_surface=VOLUME, pitch_um=2.403, depth=[23, 85],
                  order='ascending', selected_sha256=selection_hash, fields=[],
                  targets_missing=[v for v in TARGETS if not any(s['target_ink_fraction'] == v for s in selected)])
    if selected:
        surface = array(VOLUME, 0)
        if tuple(surface.shape) != (109, *shape) or surface.dtype != np.uint8:
            raise ValueError('Unexpected public CT array')
        model = load_model(args.cache, download=True)
        with tempfile.TemporaryDirectory() as tmp:
            legacy = load_previous_mosaic(tmp)
            for field in selected:
                y, x = field['core_y'], field['core_x']
                y0, x0, y1, x1 = field['parent_yxyx']
                raw = np.asarray(surface[23:85, y0:y1, x0:x1])
                if raw.shape != (62, PARENT, PARENT):
                    raise ValueError('Incomplete CT field')
                maps, grids, parity, passes = {}, {}, {}, 0
                for phase, inset in (('A', 0), ('B', HALO)):
                    stack = raw[:, inset:PARENT-inset, inset:PARENT-inset]
                    panels, geometry, historical_sha = phase_maps(stack, model, legacy)
                    cut = HALO-inset
                    maps[phase] = {m: p[cut:cut+CORE, cut:cut+CORE] for m, p in panels.items()}
                    grids[phase] = [[s+inset for s in geometry['starts_'+axis]] for axis in ('y', 'x')]
                    parity[phase] = historical_sha
                    passes += geometry['tiles']
                labels = annotations[(y, x)]
                output['fields'].append(dict(**field, raw_sha256=array_sha256(raw),
                                             native_tile_starts=grids, forward_passes=passes, bitwise_uniform_reference_sha256=parity,
                                             common_512=measure(maps, grids, *labels, 0), fully_covered_256=measure(maps, grids, *labels, 128)))
                print(f'Scored public field y={y}, x={x} in {passes} forward passes', flush=True)
    output['forward_passes'] = sum(f['forward_passes'] for f in output['fields'])
    data = (json.dumps(output, indent=2, allow_nan=False)+'\n').encode()
    args.output.write_bytes(data)
    print('R50_HANN_RESULT '+compact(output), flush=True)
    print('R50_HANN_RESULT_SHA256 '+hashlib.sha256(data).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
