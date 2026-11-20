"""Build either of two hash-verified public Vesuvius surface fixtures locally."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from numcodecs import Blosc
import tifffile
import zarr

HERE = Path(__file__).resolve().parent


def digest(path):
    result = sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def fetch(url, checksum, destination):
    if destination.is_file() and digest(destination) == checksum:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    urls = [url]
    if 'huggingface.co/' in url:
        urls.append(url + '?download=true')
    if '.s3.us-east-1.amazonaws.com/' in url:
        urls.append(url.replace('.s3.us-east-1.amazonaws.com/', '.s3.amazonaws.com/'))
    with TemporaryDirectory(dir=destination.parent) as temp:
        partial = Path(temp) / 'download.partial'
        for attempt in range(24):
            try:
                request = Request(urls[attempt % len(urls)],
                                  headers={'User-Agent': 'Scrollprize-ROI-benchmark/1.0'})
                with urlopen(request, timeout=50) as source, partial.open('wb') as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                if digest(partial) != checksum:
                    raise ValueError(f'Content differs from the pinned checksum for {url}')
                partial.replace(destination)
                return destination
            except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
                if attempt == 23:
                    raise RuntimeError(f'Could not retrieve {url}') from exc
                time.sleep(min(4, 0.4 + attempt * 0.3))


def build_w058(case, cache, output):
    def get(item):
        key, checksum = item
        return fetch(case['base_url'] + key, checksum, cache / key)

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(get, sorted(case['objects'].items())))
    height, width = case['shape'][1:]
    with TemporaryDirectory(dir=output.parent) as temp:
        raw = np.memmap(Path(temp) / 'planes.raw', dtype='u1', mode='w+', shape=case['shape'])
        for index, path in enumerate(files):
            with tifffile.TiffFile(path) as tif:
                page = tif.pages[0]
                if page.shape != (height, width) or page.dtype != np.uint8:
                    raise ValueError(f'Unexpected plane shape or dtype in {path.name}')
                raw[index] = page.asarray(maxworkers=1)
        root = zarr.open_group(output, mode='w-', zarr_format=2)
        target = root.create_array('0', shape=raw.shape, dtype='u1', chunks=(len(files), 128, 128),
                                   compressors=Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE))
        for y in range(0, height, 128):
            target[:, y:y + 128, :] = raw[:, y:y + 128, :]
        del raw


def build_w00(case, cache, output):
    def get(item):
        key, checksum = item
        fetch(case['base_url'] + key, checksum, cache / key)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(get, case['objects'].items()))
    root = zarr.open_group(output, mode='w-', zarr_format=2)
    y0, y1, x0, x1 = case['source_bounds_yx']
    for level, step in [('0', 1), ('3', 8)]:
        metadata = json.loads((cache / level / '.zarray').read_text())
        chunks = tuple(metadata['chunks'])
        if (chunks != (28, 128, 128) or metadata['compressor'] is not None
                or metadata['order'] != 'C' or metadata['dtype'] != '|u1'
                or metadata['fill_value'] != 0):
            raise ValueError('Unexpected source chunk format')
        y_start, y_end, x_start, x_end = (v // step for v in (y0, y1, x0, x1))
        target = root.create_array(level, shape=(28, y_end - y_start, x_end - x_start),
                                   chunks=chunks, dtype='u1', compressors=None)
        for iy in range(y_start // 128, (y_end - 1) // 128 + 1):
            for ix in range(x_start // 128, (x_end - 1) // 128 + 1):
                key = f'{level}/0/{iy}/{ix}'
                chunk = np.frombuffer((cache / key).read_bytes(), np.uint8).reshape(chunks)
                py, px = iy * 128, ix * 128
                top, bottom = max(py, y_start), min(py + 128, y_end)
                left, right = max(px, x_start), min(px + 128, x_end)
                target[:, top - y_start:bottom - y_start, left - x_start:right - x_start] = (
                    chunk[:, top - py:bottom - py, left - px:right - px]
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=['w00', 'w058'])
    parser.add_argument('output', type=Path, help='new local .zarr directory')
    parser.add_argument('--cache', default=Path('scroll-data-cache'), type=Path)
    parser.add_argument('--checkpoint', type=Path, help='optional destination for Reader v3 weights')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'sources.json').read_text())
    case = manifest['cases'][args.case]
    if args.output.exists():
        parser.error('output already exists; choose a fresh directory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    zarr.config.set({'async.concurrency': 2, 'threading.max_workers': 2})
    if args.checkpoint:
        model = manifest['model']
        fetch(model['url'], model['sha256'], args.checkpoint)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        stage = Path(temp) / 'result.zarr'
        builder = build_w00 if args.case == 'w00' else build_w058
        builder(case, args.cache / args.case, stage)
        stamp = sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
        zarr.open_group(stage, mode='a').attrs.update(
            {'fixture_case': args.case, 'source_manifest_sha256': stamp}
        )
        stage.rename(args.output)
    print(json.dumps({'case': args.case, 'surface': str(args.output), 'shape': case['shape'],
                      'roi_yx': case['roi_yx']}))


if __name__ == '__main__':
    main()
