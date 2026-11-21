"""Run one CPU inference on the public 2.403 µm PHerc. 841 w00 annotations."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
import zarr
from transformers import AutoModel
from zarr.storage import FsspecStore


SEGMENT = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
           "PHerc0841/segments/20260220213127-w00")
LABELS = SEGMENT + "/ink-labels/2.403um-volume-20260319124803/20260918/"
VOLUME = SEGMENT + "/surface-volumes/2.403um-0.22m-77keV-volume-20260319124803.zarr"
MODEL = "scrollprize/PHerc.1667-iteration-0"
REVISION = "06c306449b39df42c745608ceade243498d24243"
GIT_BLOBS = {
    "config.json": "32accd0facbc9424458e5d268460de0f07e6bd7d",
    "configuration_inkdetection.py": "24ac6d12668bfb5cf4b448c8f61723e97264ee25",
    "modeling_inkdetection.py": "c75143f25652f1e5fe9f7f7dc7b5183f00d37fb5",
}
WEIGHT_SHA = "62c43b98f55c169ae874c684a59953b5ce1cb45140129107f861501cce3edaef"
WIDTH = 256
DEPTH = 62


def public_weights(directory):
    directory.mkdir(parents=True, exist_ok=True)
    names = list(GIT_BLOBS) + ["model.safetensors"]
    for name in names:
        path = directory / name
        if not path.exists():
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
            with requests.get(url, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                partial = path.with_suffix(path.suffix + ".partial")
                with partial.open("wb") as f:
                    for chunk in response.iter_content(1024 * 1024):
                        f.write(chunk)
                partial.replace(path)
        if name in GIT_BLOBS:
            data = path.read_bytes()
            header = f"blob {len(data)}".encode() + bytes([0])
            assert hashlib.sha1(header + data).hexdigest() == GIT_BLOBS[name], name
        else:
            with path.open("rb") as f:
                assert hashlib.file_digest(f, "sha256").hexdigest() == WEIGHT_SHA
            assert path.stat().st_size == 333531996


def array(url, level):
    store = FsspecStore.from_url(f"{url}/{level}", read_only=True,
                                 storage_options={"client_kwargs": {"trust_env": True}})
    return zarr.open_array(store=store, mode="r")


def as_mask(values):
    a = np.asarray(values)
    return a > (0 if a.max(initial=0) <= 1 else 127)


def cell_counts(values, cell):
    rows, cols = (d // cell for d in values.shape)
    return values[:rows*cell, :cols*cell].reshape(rows, cell, cols, cell).sum((1, 3))


def choose_public_tile():
    ink_url, sup_url, val_url = (LABELS + f"{name}.zarr"
                                  for name in ("inklabels", "supervision", "validation"))
    ink, sup, val = (array(url, 0) for url in (ink_url, sup_url, val_url))
    print("Declared mask geometry", ink.shape, sup.shape, val.shape, flush=True)
    if not (len(ink.shape) == 2 and ink.shape == sup.shape == val.shape):
        raise ValueError("The public label grids must share their declared geometry")
    for level in (4, 3, 2, 1):
        try:
            small = [as_mask(array(url, level)[:]) for url in (ink_url, sup_url, val_url)]
        except (FileNotFoundError, ValueError, KeyError):
            continue
        if small[0].shape == small[1].shape == small[2].shape:
            break
    else:
        raise RuntimeError("No common public overview level")
    scale = 2**level
    shrink = tuple((d + scale - 1) // scale for d in ink.shape)
    if any(abs(a-b) > 1 for a, b in zip(shrink, small[0].shape)):
        raise ValueError(f"Unexpected overview geometry {small[0].shape}, expected {shrink}")
    cell = WIDTH // scale
    si, ss, sv = small
    # Labels alone determine the tile before the CT or the detector is opened.
    # Try the released validation area first, then the supervised area if empty.
    for method, trusted in (("validation_and_supervision", ss & sv), ("supervision", ss)):
        count_p = cell_counts(trusted & si, cell)
        count_n = cell_counts(trusted & ~si, cell)
        score = np.minimum(count_p, count_n)
        rank = np.argsort(-score.ravel().astype(float), kind="stable")[:64]
        for index in rank:
            if score.ravel()[index] == 0:
                break
            row, col = np.unravel_index(index, score.shape)
            y, x = int(row*WIDTH), int(col*WIDTH)
            sl = np.s_[y:y+WIDTH, x:x+WIDTH]
            i, s, v = (as_mask(m[sl]) for m in (ink, sup, val))
            mask = s & v if method.startswith("validation") else s
            if min((i & mask).sum(), (~i & mask).sum()) >= 512:
                return (x, y), i, mask, v, method, level, tuple(ink.shape)
    raise RuntimeError("No well-supported mixed tile in the published annotations")


def metrics(probability, ink, mask):
    pos = probability[ink & mask]
    neg = probability[~ink & mask]
    out = {"positives": int(len(pos)), "negatives": int(len(neg))}
    if not len(pos) or not len(neg):
        return out
    y = ink[mask].astype(np.int64)
    unique, group = np.unique(probability[mask], return_inverse=True)
    positives = np.bincount(group, weights=y, minlength=len(unique))
    negatives = np.bincount(group, weights=1-y, minlength=len(unique))
    rank_auc = (positives * (np.cumsum(negatives) - .5*negatives)).sum() / (len(pos)*len(neg))
    p, n = positives[::-1], negatives[::-1]
    precision = np.cumsum(p) / (np.cumsum(p) + np.cumsum(n))
    out.update(auc=float(rank_auc), average_precision=float(np.sum(p*precision)/len(pos)),
               base_rate=float(y.mean()), median_positive=float(np.median(pos)),
               median_negative=float(np.median(neg)), positive_ge_half=int((pos >= .5).sum()),
               negative_ge_half=int((neg >= .5).sum()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".r50-weights"))
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.set_num_interop_threads(1)
    xy, ink, mask, validation, method, level, refshape = choose_public_tile()
    x, y = xy
    surface = array(VOLUME, 0)
    print("Full scan surface grid", surface.shape, surface.dtype, "tile XY", xy, flush=True)
    if len(surface.shape) != 3 or surface.shape[0] < DEPTH or surface.shape[0] > 512:
        raise ValueError("Expected a depth-first surface volume with at least 62 layers")
    if tuple(surface.shape[-2:]) != refshape:
        raise ValueError(f"Surface and annotations have different geometry {surface.shape} {refshape}")
    start = (surface.shape[0]-DEPTH)//2
    raw = np.asarray(surface[start:start+DEPTH, y:y+WIDTH, x:x+WIDTH])
    if raw.shape != (DEPTH, WIDTH, WIDTH) or raw.dtype != np.uint8:
        raise ValueError(f"Bad CT tile shape or dtype {raw.shape} {raw.dtype}")
    print("CT acquired", hashlib.sha256(raw.tobytes()).hexdigest(), flush=True)
    public_weights(args.cache)
    model = AutoModel.from_pretrained(str(args.cache.resolve()), local_files_only=True,
                                      trust_remote_code=True).eval()
    # The paper specifies clip [0,200], then Albumentations Normalize(0,1),
    # whose default max_pixel_value is 255. Keep public storage depth order.
    tensor = torch.from_numpy(np.minimum(raw, 200).astype(np.float32)/255.0)[None, None]
    begin = time.monotonic()
    with torch.inference_mode():
        logits = model(tensor).logits
        if tuple(logits.shape) != (1, 1, 64, 64):
            raise ValueError(f"Unexpected output shape {tuple(logits.shape)}")
        prob = F.interpolate(logits.sigmoid(), size=(WIDTH, WIDTH), mode="bilinear",
                             align_corners=False)[0, 0].numpy()
    if not np.isfinite(prob).all():
        raise ValueError("Nonfinite model output")
    core = np.zeros_like(mask)
    core[32:-32, 32:-32] = True
    out = dict(segment="PHerc0841/20260220213127-w00", scan_pitch_um=2.403,
               tile_xy=xy, model=MODEL, revision=REVISION, weight_sha256=WEIGHT_SHA,
               source_surface=VOLUME, source_labels=LABELS, labels=method,
               selection_overview_level=level, volume_shape=tuple(surface.shape),
               depth_range_half_open=[start, start+DEPTH], depth_order="ascending",
               input_shape=list(raw.shape), tile_width_mm=WIDTH*2.403/1000,
               output_shape=list(logits.shape),
               input_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
               positive_mask_sha256=hashlib.sha256(ink.tobytes()).hexdigest(),
               trusted_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
               validation_pixels=int((validation & mask).sum()), threads=torch.get_num_threads(),
               elapsed_cpu_inference_seconds=round(time.monotonic()-begin, 3),
               tile=metrics(prob, ink, mask), central_192=metrics(prob, ink, mask & core))
    print("R50_PUBLIC_CONTROL " + json.dumps(out, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
