#!/usr/bin/env python3
"""Prepare checksum-pinned public 306B model/crop; field CT is an optional addition."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

from common import FILES, HF_DATA, PROTOCOL, field_plan, outside_git, sha, write_json


def fetch(url, path, expected):
    if path.exists():
        if sha(path) != expected:
            raise ValueError("Existing source differs from pinned checksum: " + path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "public-scrollprize-phase-control"})
    with urllib.request.urlopen(req, timeout=90) as source, part.open("wb") as out:
        shutil.copyfileobj(source, out, 1024*1024)
    if sha(part) != expected:
        raise ValueError("Downloaded source differs from pinned checksum: " + path.name)
    part.replace(path)


def prepare_field(root):
    """Download only complete, verified source chunks needed by both phases."""
    import numcodecs
    import numpy as np
    zpath = "results/claim_surface_render_41layers.zarr/0/"
    metadata_path = root / "render-zarray.json"
    fetch(HF_DATA+zpath+".zarray", metadata_path, "6e9dfd265acd7a74f56d3ec82a82fd7de1715ec473cd966c5d3d141c3cd94d51")
    metadata = json.loads(metadata_path.read_text())
    if metadata["shape"] != [41, 4900, 7780] or metadata["chunks"] != [41, 128, 128] or metadata["dtype"] != "|u1":
        raise ValueError("Unexpected pinned public render geometry")
    sums_path = root / "dataset-SHA256SUMS"
    fetch(HF_DATA+"SHA256SUMS", sums_path, "8c114c6359efd95c30508c7b9731ebfab3ee799bf2ec6dcf84cf92e841eb07cf")
    checks = {}
    for line in sums_path.read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields)==2: checks[fields[1].lstrip(" *").removeprefix("./")] = fields[0]
    y0, x0, y1, x1 = field_plan()["required_native_source_box_yxyx"]
    chunks = []
    for cy in range(y0//128, (y1-1)//128+1):
        for cx in range(x0//128, (x1-1)//128+1):
            key = zpath + f"0/{cy}/{cx}"
            if key not in checks:
                raise ValueError("Required source chunk absent from published checksum manifest")
            chunks.append((cy, cx, key, root/"chunks"/f"0-{cy}-{cx}", checks[key]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda row: fetch(HF_DATA+row[2], row[3], row[4]), chunks))
    volume = np.zeros((24, y1-y0, x1-x0), np.uint8)
    seen = np.zeros((y1-y0, x1-x0), bool)
    decoder = numcodecs.get_codec(metadata["compressor"])
    receipts = []
    for cy, cx, key, path, expected in chunks:
        raw = np.frombuffer(decoder.decode(path.read_bytes()), np.uint8).reshape(41,128,128)
        ya, xa = max(y0, cy*128), max(x0, cx*128)
        yb, xb = min(y1, (cy+1)*128), min(x1, (cx+1)*128)
        volume[:, ya-y0:yb-y0, xa-x0:xb-x0] = raw[8:32, ya-cy*128:yb-cy*128, xa-cx*128:xb-cx*128]
        seen[ya-y0:yb-y0, xa-x0:xb-x0] = True
        receipts.append(dict(source=key, sha256=expected, bytes=path.stat().st_size))
    if not seen.all():
        raise ValueError("Incomplete source chunks; never fill missing coverage with zero")
    output = root/"field_input.npz"
    if output.exists(): raise FileExistsError("Field input already exists; use its receipt or a new preparation directory")
    np.savez_compressed(output, st=np.ascontiguousarray(np.minimum(volume,200).transpose(1,2,0)), origin_yx=np.array([y0,x0],np.int32))
    write_json(root/"field_input.json", dict(source_box_yxyx=[y0,x0,y1,x1], input_sha256=sha(output),
        published_chunk_bytes_verified=True, native_CT_acquisition_coverage_certified=False,
        metadata_sha256=sha(metadata_path), published_checksum_manifest_sha256=sha(sums_path), chunks=receipts))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--with-field", action="store_true")
    args=ap.parse_args(); outside_git(args.output_dir); args.output_dir.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item: fetch(item[1][0],args.output_dir/item[0],item[1][1]),FILES.items()))
    if args.with_field: prepare_field(args.output_dir)
    print(json.dumps({"verified_files":len(FILES),"field_prepared":args.with_field,"weights_bytes":(args.output_dir/"seed0.safetensors").stat().st_size}))


if __name__=="__main__":main()
