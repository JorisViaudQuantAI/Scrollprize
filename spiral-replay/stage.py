"""Stage whole published M7 slices with exact HTTP ranges and pinned Villa helpers."""
import argparse
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PIN = "0fb38c45b488e192566ea83ae36369cb6db92652"
CODE = f"https://raw.githubusercontent.com/ScrollPrize/villa/{PIN}/volume-cartographer/scripts/spiral"
META = json.loads((ROOT / "public-inputs.json").read_text())
TRACKS = META["public_tracks"]
NORMALS = META["public_normals"]
UMBILICUS = META["public_umbilicus"]
GEOMETRY_MODULES = ("geom_utils.py", "sample_spiral.py", "umbilicus.py", "config.py",
                    "transforms.py", "gap_triton.py", "flow_triton.py", "flow_fields.py", "fit_session.py")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def acquire(url, dst, byte_range=None):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and (byte_range is None or dst.stat().st_size == byte_range[1] - byte_range[0]):
        return
    headers = {}
    if byte_range:
        lo, hi = byte_range
        headers["Range"] = f"bytes={lo}-{hi-1}"
    req = urllib.request.Request(url, headers=headers)
    tmp = dst.with_name(dst.name + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, tmp.open("wb") as out:
                if byte_range and (resp.status != 206 or not resp.headers.get("Content-Range", "").startswith(f"bytes {lo}-{hi-1}/")):
                    raise ValueError(f"incorrect byte range for {url}")
                shutil.copyfileobj(resp, out)
            if byte_range and tmp.stat().st_size != hi - lo:
                raise ValueError(f"truncated response for {url}")
            tmp.replace(dst)
            return
        except OSError:
            tmp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def git_blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\x00" + data).hexdigest()


def _cached_range(src, dst, lo, hi, base=0):
    with src.open("rb") as stream:
        stream.seek(lo-base)
        data = stream.read(hi-lo)
    if len(data) != hi-lo:
        raise ValueError(f"incomplete local range in {src}")
    dst.write_bytes(data)


def stage_range(name, interval, target, cache):
    lo, hi = interval
    if not target.is_file() and cache:
        if (cache/name).is_file():
            _cached_range(cache/name, target, lo, hi)
        else:
            for file in sorted(cache.glob(name + ".*-*")):
                a, b = map(int, file.name.split(".")[-1].split("-"))
                if a <= lo and hi <= b and file.stat().st_size == b-a:
                    _cached_range(file, target, lo, hi, a)
                    break
    acquire(TRACKS + "/" + name, target, interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["replay","six"], default="replay")
    parser.add_argument("--index-cache", type=Path, help="optional original packed-store directory or directory of ranges")
    parser.add_argument("--upstream", type=Path, help="pinned checkout's volume-cartographer/scripts/spiral directory")
    args = parser.parse_args()
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    code_hashes = json.loads((ROOT / "upstream-blobs.json").read_text())
    for name in GEOMETRY_MODULES:
        dest = ROOT / "upstream" / name
        if args.upstream and (args.upstream/name).resolve() != dest.resolve():
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(args.upstream/name, dest)
        else:
            acquire(CODE+"/"+name, dest)
        if git_blob(dest) != code_hashes[name]:
            raise ValueError(f"{name} does not match the pinned Villa revision")
    acquire(UMBILICUS, ROOT/"umbilicus.json")
    if digest(ROOT/"umbilicus.json") != META["umbilicus_sha256"]:
        raise ValueError("umbilicus checksum differs from the recorded annotation")
    observed = []
    for row in META["planes"]:
        z = row["z"]
        if z not in META["profiles"][args.profile]:
            continue
        offset_path = data/f"plane-{z}-offset-range.i64"
        coords = data/f"plane-{z}.i32"
        stage_range("offsets.i64", row["offset_range"], offset_path, args.index_cache)
        stage_range("coordinates.i32", row["coordinate_range"], coords, args.index_cache)
        if digest(offset_path) != row["offsets_sha256"]:
            raise ValueError(f"incorrect producer offsets at z={z}")
        sha = digest(coords)
        if row["coordinates_sha256"] is not None and sha != row["coordinates_sha256"]:
            raise ValueError(f"incorrect producer coordinates at z={z}")
        offsets = np.fromfile(offset_path,dtype="<i8")
        local = offsets - offsets[0]
        points = np.memmap(coords,"<i4","r").reshape(-1,3)
        if len(local) != row["tracks"]+1 or local[-1]!=len(points) or np.any(np.diff(local)<=0) or not np.all(points[:,0]==z):
            raise ValueError(f"source integrity check failed at z={z}")
        np.save(data/f"plane-{z}-offsets.npy", local, allow_pickle=False)
        observed.append({"z":z,"tracks":row["tracks"],"points":len(points),"sha256":sha})
    # A failed fetch never relabels an incomplete plan as a full six-plane replay.
    (ROOT/"sources.json").write_text(json.dumps({"profile":args.profile,"planes":observed},indent=2)+"\n")
    print(json.dumps(observed))


if __name__ == "__main__":
    main()
