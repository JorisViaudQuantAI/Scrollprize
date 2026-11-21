"""Deterministic geometry and receipts for the published 306B phase experiment."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
RELEASE = PROTOCOL["source_commit"]
DATA = PROTOCOL["dataset_revision"]
MODEL = PROTOCOL["model_revision"]
REPO = "https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/" + RELEASE + "/"
HF_DATA = "https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/" + DATA + "/"
HF_MODEL = "https://huggingface.co/Prodigle/pherc0306b-ink-ahn/resolve/" + MODEL + "/"
FILES = {
    "modeling.py": (REPO + "submission/code/third_party/v8in/ink8um/modeling.py", "0656550dc10556ad09f5b6c21b36c97aa97a5696f71ef4b384eada124095c109"),
    "config.json": (REPO + "submission/code/third_party/v8in/config.json", "30f46a55d61dc3f4e9bf7e9ca261c1486575f1ce05a27f28d09f10ee57597018"),
    "seed0.safetensors": (HF_MODEL + "checkpoints/hn4_s0_final/model.safetensors", PROTOCOL["weights_sha256"]),
    "smoke.npz": (HF_DATA + "docker/build/smoke_crop.npz", "4a81e193e253e8b6234eaa4c1b274e33c08b214183951ed8e573c9ae3f8f639d"),
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def outside_git(path):
    path = Path(path).resolve()
    if any((p / ".git").exists() for p in (path, *path.parents)):
        raise ValueError("Generated inputs and outputs must stay outside every Git worktree")


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def gaussian():
    """Same normal-CDF Gaussian construction as pinned upstream gkern(64, 3)."""
    edges = np.linspace(-3., 3., 65)
    cdf = .5 * (1. + np.vectorize(math.erf)(edges / math.sqrt(2.)))
    one = np.diff(cdf)
    two = np.outer(one, one)
    two /= two.sum()
    return (two / two.max()).astype(np.float32)


def positions(roi, phase):
    """All globally anchored 64px tiles intersecting a half-open native ROI."""
    y0, x0, y1, x1 = roi
    gy, gx = (a+b for a, b in zip(PROTOCOL["source_global_grid_origin_yx"], phase))
    stride, tile = 21, 64
    def axis(lo, hi, origin):
        a = math.ceil((lo-tile+1-origin)/stride)
        b = math.floor((hi-1-origin)/stride)
        return range(origin+a*stride, origin+(b+1)*stride, stride)
    # This frozen interior field needs no artificial padding or outer-grid clipping.
    pts = np.array([(y, x) for y in axis(y0, y1, gy) for x in axis(x0, x1, gx)], np.int32)
    if np.any(pts < [115, 694]) or np.any(pts+64 > [4166, 7638]):
        raise ValueError("The fixed field's tile halo leaves the published carrier")
    return pts


def field_plan():
    roi = PROTOCOL["field"]["native_render_box_yxyx"]
    ps = [positions(roi, p) for p in PROTOCOL["phases_yx"]]
    allp = np.concatenate(ps)
    lo = allp.min(0); hi = (allp+64).max(0)
    counts = []
    for p in ps:
        count = np.zeros((roi[2]-roi[0], roi[3]-roi[1]), np.uint16)
        for y, x in p:
            ya, xa = max(y, roi[0])-roi[0], max(x, roi[1])-roi[1]
            yb, xb = min(y+64, roi[2])-roi[0], min(x+64, roi[3])-roi[1]
            count[ya:yb, xa:xb] += 1
        counts.append({"tiles_before_CT_admission": int(len(p)), "contributors_min_before_CT_admission": int(count.min()), "contributors_max_before_CT_admission": int(count.max())})
    return dict(native_roi_yxyx=roi, phase_offsets_yx=PROTOCOL["phases_yx"],
                required_native_source_box_yxyx=[int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])],
                phases=counts, interpretation="Geometry-only counts; CT admission and actual common valid coverage remain to be measured")


def compare(a, b, valid):
    av, bv = a[valid].astype(np.float64), b[valid].astype(np.float64)
    if not av.size:
        raise ValueError("No common valid pixels")
    delta = np.abs(av-bv)
    corr = float(np.corrcoef(av, bv)[0, 1]) if av.std() > 0 and bv.std() > 0 else None
    return dict(valid_pixels=int(av.size), mean_absolute_difference=float(delta.mean()),
                maximum_absolute_difference=float(delta.max()),
                absolute_difference_quantiles_0_25_50_75_95_99_100=np.quantile(delta, [0, .25, .5, .75, .95, .99, 1]).tolist(),
                pearson_r=corr,
                binary_disagreement={str(t): {"pixels": int(((av>=t)!=(bv>=t)).sum()), "fraction": float(((av>=t)!=(bv>=t)).mean())} for t in (.25, .5, .75)},
                interpretation="Same-physical-pixel stability, not text accuracy or false-positive rate")
