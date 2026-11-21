"""Prepare a half-scale copy of a whole published DLS intensity layer for the recipient."""
import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tifffile


SOURCE = 'https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/20231016151002/layers'
LAYERS = (29, 30, 31, 33, 34, 35)
CHUNK = 24 * 1024**2


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layer', choices=LAYERS, type=int)
    parser.add_argument('certificate', type=Path)
    args = parser.parse_args()
    original = Path(f'{args.layer}.tif')
    url = f'{SOURCE}/{original.name}'
    subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                    '--retry', '2', '--max-time', '210', '--output', str(original), url], check=True)
    image = tifffile.memmap(original)
    if image.dtype != np.uint16 or image.shape != (10065, 15954):
        raise ValueError(f'Unexpected published TIFF dimensions or type {image.shape} {image.dtype}')
    h, w = (size // 2 for size in image.shape)
    result = Path(f'layer-{args.layer}.npy')
    reduced = np.lib.format.open_memmap(result, mode='w+', dtype='<u2', shape=(h, w))
    for start in range(0, h, 128):
        end = min(h, start + 128)
        block = image[2 * start:2 * end, :2 * w].reshape(end - start, 2, w, 2)
        reduced[start:end] = (block.sum(axis=(1, 3), dtype=np.uint32) + 2) // 4
    del reduced
    compressed = result.with_suffix('.npy.gz')
    with result.open('rb') as src, gzip.open(compressed, 'wb', compresslevel=2) as dst:
        shutil.copyfileobj(src, dst)
    encrypted = Path('delivery.der')
    subprocess.run(['openssl', 'cms', '-encrypt', '-binary', '-outform', 'DER', '-aes-256-gcm',
                    '-in', str(compressed), '-out', str(encrypted), str(args.certificate)], check=True)
    receipt = {'source': url, 'source_sha256': sha256(original), 'source_shape': list(image.shape),
               'output_shape': [h, w], 'npy_sha256': sha256(result), 'ciphertext_sha256': sha256(encrypted),
               'reduction': 'uint16 rounded arithmetic mean of each complete 2x2 block'}
    parts = []
    with encrypted.open('rb') as stream:
        while data := stream.read(CHUNK):
            directory = Path(f'piece-{len(parts)}')
            directory.mkdir()
            (directory / 'data.bin').write_bytes(data)
            parts.append({'index': len(parts), 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    if not 1 <= len(parts) <= 4:
        raise ValueError('Unexpected delivery size')
    receipt['parts'] = parts
    for part in parts:
        (Path(f"piece-{part['index']}") / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
