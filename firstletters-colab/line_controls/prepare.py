#!/usr/bin/env python3
"""Retrieve and extract a frozen, public, already-awarded line-control corpus.

Binary source data and arrays must remain outside the repository. This does not
train a model, inspect unread scrolls, choose candidates, or calibrate a gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HF = "https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/f435998b55269762942a979052044309b29650ac/"
GH306 = "https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/"
GH343 = "https://raw.githubusercontent.com/Nieuwlaar/pherc343-first-letters/e8370f1f88a211166056df62ea6fcfffe4a1063e/"
SOURCES = [
    ("306_claim.npy", HF + "results/claim_map_fwd_fin.npy", "sha256", "4de6da3b5533408c81abf2fe3358e12e637b94ccdaf0a59d490b9bd1ddc88301"),
    ("306_shuffle.npy", HF + "results/depth_shuffle_map_fwd_fin.npy", "sha256", "b270740bcf84a4700f3a12d6abee64f5ad45efb816faa776579e613a7203a60c"),
    ("306_ct.png", HF + "results/ct_surface_layer20.png", "sha256", "e28222dd5d67b3ca10ad863f7426342e744618d9a3b5549aac7a03a47bc1ea63"),
    ("343_claim.png", GH343 + "outputs/concat_w047-w048_R5B2_z9500-11000.png", "sha256", "59307c84ea65091de1bea14a3b75d44069bfe1fb498b3445f698ec3d1b8c1c78"),
    ("306_sidecar.json", GH306 + "submission/PHerc0306B_20250521133212_w074own_touchup_V5Rc.json", "git_blob_sha1", "2210df2f8a0a1d585c35957f6729719e7c2e0aa5"),
    ("343_sidecar.json", GH343 + "outputs/concat_w047-w048_R5B2_z9500-11000.json", "git_blob_sha1", "e48ad7dca5c2a56ea9519594fc5f4f20660dbeac"),
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_verified(root, record):
    name, url, algorithm, expected = record
    path = root / name
    if not path.exists():
        temporary = path.with_name(path.name + ".part")
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        checked = temporary
    else:
        checked = path
    actual = digest(checked)
    if algorithm == "git_blob_sha1":
        payload = checked.read_bytes()
        comparison = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    else:
        comparison = actual
    if comparison != expected:
        raise ValueError(f"Source checksum mismatch: {name}")
    if checked != path:
        checked.replace(path)
    return dict(name=name, url=url, bytes=path.stat().st_size, sha256=actual,
                source_verification=dict(algorithm=algorithm, expected=expected))


def crop306(array, bounds):
    """Copy native samples in reading order; no interpolation or stretch."""
    u0, v0, u1, v1 = bounds
    k = 1000.0 / 8.64
    x0, x1 = int(round((u0 + 36) * k)), int(round((u1 + 36) * k))
    y0, y1 = int(round((v0 + 21) * k)), int(round((v1 + 21) * k))
    if not (0 <= x0 < x1 <= array.shape[1] and 0 <= y0 < y1 <= array.shape[0]):
        raise ValueError("Requested public crop is outside source")
    return np.array(array[y0:y1, x0:x1][::-1], copy=True), [x0, y0, x1, y1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out == REPO or REPO in out.parents:
        parser.error("Keep binary data and generated arrays outside the repository")
    out.mkdir(parents=True, exist_ok=True)
    target = out / "cases"
    if target.exists():
        parser.error("cases/ already exists; use a new output directory")
    protocol = json.loads((HERE / "protocol.json").read_text())
    (out / "protocol.json").write_bytes((HERE / "protocol.json").read_bytes())
    records = [fetch_verified(out, item) for item in SOURCES]
    target.mkdir()
    ct = np.asarray(Image.open(out / "306_ct.png"))
    if ct.shape != (4900, 7780) or ct.dtype != np.uint8:
        raise ValueError("Unexpected public CT-layer shape or encoding")
    support = ct > 0
    native = np.load(out / "306_claim.npy", mmap_mode="r", allow_pickle=False)
    shuffled = np.load(out / "306_shuffle.npy", mmap_mode="r", allow_pickle=False)
    for array in [native, shuffled]:
        if array.shape != ct.shape or array.dtype != np.float16:
            raise ValueError("Unexpected score map shape or encoding")
        finite = array[np.isfinite(array)]
        if not len(finite) or finite.min() < 0 or finite.max() > 1:
            raise ValueError("Available source scores not within [0,1]")
    claim_support = support & np.isfinite(native)
    common_support = claim_support & np.isfinite(shuffled)
    cases = []

    def save_case(name, score, mask, kind, role, geometry, targets=None):
        score, mask = score.astype(np.float32), mask.astype(bool)
        if not np.isfinite(score[mask]).all():
            raise ValueError("A required model score is not available")
        score[~mask] = 0  # Only a storage fill: these samples remain explicitly invalid.
        if score.shape != mask.shape:
            raise ValueError("Control score and mask disagree")
        folder = target / name
        folder.mkdir()
        np.save(folder / "score.npy", score, allow_pickle=False)
        np.save(folder / "validity.npy", mask, allow_pickle=False)
        rec = dict(id=name, map="score.npy", mask="validity.npy", shape=list(score.shape),
                   map_kind=kind, role=role, spacing_um_yx=[8.64, 8.64],
                   source_geometry=geometry, score_sha256=digest(folder / "score.npy"),
                   mask_sha256=digest(folder / "validity.npy"), valid_pixels=int(mask.sum()),
                   validity_fraction=float(mask.mean()), targets=targets or [],
                   all_unlabelled_pixels="unknown; not certified text-free")
        (folder / "manifest.json").write_text(json.dumps(rec, indent=2) + "\n")
        cases.append(rec)

    side306 = json.loads((out / "306_sidecar.json").read_text())
    left, top = side306["image_px"]["map_left"], side306["image_px"]["map_top"]
    x0, x1 = side306["claim_box_image_px"]["x0"] - left + 8, side306["claim_box_image_px"]["x1"] - left - 8
    radius = 2.0 * 1000 / 8.64
    rows = [dict(id=f"published_row_{n}", box_xyxy=[x0, y-top-radius, x1, y-top+radius],
                 evidence="author's coarse row-center annotation; not pixel truth")
            for n, y in enumerate(side306["rows"]["centre_y_image_px"], 1)]
    context = [-30, -20, 30, 15]
    map306, raw_bounds = crop306(native, context)
    mask306, _ = crop306(claim_support, context)
    geometry306 = dict(uv_bounds_mm=context, raw_bounds_xyxy=raw_bounds, flip_ud=True,
                       no_resampling=True, label_alignment_limit="Source display uses pixel-center resampling; source image boxes are coarse targets with about one native pixel rounding uncertainty",
                       support_limit="CT layer20>0 AND finite released claim score; tissue/model-support proxy, not audited acquisition coverage")
    save_case("pherc306b_context_raw", map306, mask306, "model_score", "published-positive-context", geometry306, rows)
    blank = [25, -20, 30, 3]
    score, bounds = crop306(native, blank)
    mask, _ = crop306(claim_support, blank)
    save_case("pherc306b_blank_raw", score, mask, "model_score", "author-described-blank-strip",
              dict(uv_bounds_mm=blank, raw_bounds_xyxy=bounds, flip_ud=True, no_resampling=True,
                   support_limit="Author's negative-control strip; not independently labelled text-free. CT layer20>0 AND finite claim score."))
    score, _ = crop306(shuffled, context)
    matched_mask, _ = crop306(common_support, context)
    matched_geometry = dict(geometry306, support_limit="Common CTlayer20>0 AND finiteclaim AND finiteshuffle; missing model coverage is never background")
    save_case("pherc306b_shuffle_raw", score, matched_mask, "model_score", "matched-depth-shuffle-process-control", matched_geometry, rows)
    save_case("pherc306b_context_matched_raw", map306, matched_mask, "model_score", "published-positive-context-common-coverage", matched_geometry, rows)

    side343 = json.loads((out / "343_sidecar.json").read_text())
    panel = side343["outputs"]["image"]["data_panel"]
    rgb = np.asarray(Image.open(out / "343_claim.png").convert("RGB"))
    x, y, w, h = (panel[k] for k in ["x", "y", "width", "height"])
    rgb = rgb[y:y+h, x:x+w].astype(np.float32)
    red, green, blue = rgb.transpose(2, 0, 1)
    score = np.clip((red-blue)/(135 * 0.9), 0, 1)
    mask = (rgb.max(axis=2) > 0) & (np.abs((red-green)-(20/135)*(red-blue)) <= 2.0)
    boxes = [dict(id=str(row["id"]), row=row["row"], box_xyxy=row["box"], evidence="published author letter-location box; not an ink segmentation")
             for row in side343["letters"]["letters_area_px"]]
    save_case("pherc343_context_display", score, mask, "display_signal", "published-positive-context-display-only",
              dict(source_panel_xywh=[x,y,w,h], no_spatial_resampling=True,
                   signal="normalized displayed yellow opacity; raw scores and subthreshold information are not recoverable",
                   mask_limit="Display color-consistency mask excludes cyan annotation and black pixels; not CT/acquisition validity"), boxes)
    inventory = dict(schema="public-line-controls-v1", protocol_sha256=digest(HERE / "protocol.json"),
                     coverage_amendment_sha256=digest(HERE / "amendment-001.json"),
                     extractor_sha256=digest(__file__), license="CC BY-NC 4.0 for source/derived maps and images",
                     source_files=records, cases=cases,
                     excluded_claims=["independent glyphCNN validation", "text-free labels outside supplied controls", "empirical false-positive rate on unread scrolls", "raw probability from a display overlay"])
    (out / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"protocol_sha256":inventory["protocol_sha256"],
                      "cases":[{k:c[k] for k in ["id","shape","map_kind","validity_fraction"]} for c in cases]},indent=2))


if __name__ == "__main__":
    main()
