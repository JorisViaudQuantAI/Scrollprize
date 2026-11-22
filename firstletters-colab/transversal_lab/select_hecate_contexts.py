"""Prospective selection from released coordinates and calibration support only.

This script never opens CT, predictions, or PHerc0814 held-out labels.
All coordinates refer to publicly released labeled donor surfaces.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
TARGET = 9.6
LABEL = 9.36
WIDTH = 64
COUNT = 16


def farthest(candidates):
    """Centroid-nearest first, then maximin; exact ties use Y/X lexical order."""
    c = np.unique(np.asarray(candidates, dtype=np.int64), axis=0)
    if len(c) < COUNT:
        raise ValueError(f"Only {len(c)} eligible contexts, fewer than {COUNT}")
    center = c.mean(axis=0)
    selected = [int(np.argmin(((c - center) ** 2).sum(axis=1)))]
    dist = ((c - c[selected[0]]) ** 2).sum(axis=1)
    while len(selected) < COUNT:
        dist[selected] = -1
        idx = int(np.argmax(dist))
        selected.append(idx)
        dist = np.minimum(dist, ((c - c[idx]) ** 2).sum(axis=1))
    return c[selected]


def from_coordinates(name, shape):
    a = np.load(ROOT / f"coordinates_{name}.npy", allow_pickle=False)
    # Published coordinates are origins of 128-pixel contexts at 9.36 um.
    # Preserve their physical center, snapping to the global 9.6 um grid.
    origins = np.floor((a + 64) * LABEL / TARGET - WIDTH / 2 + 0.5).astype(int)
    edge = np.floor(np.asarray(shape) * LABEL / TARGET).astype(int)
    inside = ((origins >= 0) & (origins + WIDTH <= edge)).all(axis=1)
    return farthest(origins[inside]), int(np.unique(origins[inside], axis=0).shape[0])


def calibration():
    # Permission: calibration annotation support only. Do not select by class.
    state = np.asarray(Image.open(ROOT / "calibration_support/refined_state.png"))
    geometry = np.asarray(Image.open(ROOT / "calibration_support/geometry-valid.png"))
    if state.ndim != 2 or state.shape != (2661, 2136):
        raise ValueError("Unexpected calibration categorical array shape")
    if geometry.shape != state.shape or not set(np.unique(state)) <= {0, 1, 2}:
        raise ValueError("Unexpected calibration geometry or label encoding")
    known = state != 0
    geometry = geometry != 0
    edge = np.floor(np.asarray(state.shape) * LABEL / TARGET).astype(int)
    eligible = []
    for y in range(0, int(edge[0]) - WIDTH + 1, WIDTH):
        for x in range(0, int(edge[1]) - WIDTH + 1, WIDTH):
            iy = np.floor((y + np.arange(WIDTH) + 0.5) * TARGET / LABEL).astype(int)
            ix = np.floor((x + np.arange(WIDTH) + 0.5) * TARGET / LABEL).astype(int)
            # Conservative full physical XY footprint on the label/geometry grid.
            # Include every geometry cell intersecting the 64*9.6 um input box.
            lo = np.floor(np.array([y, x]) * TARGET / LABEL).astype(int)
            hi = np.ceil((np.array([y, x]) + WIDTH) * TARGET / LABEL).astype(int)
            if not geometry[lo[0]:hi[0], lo[1]:hi[1]].all():
                continue
            # At least 128 known central pixels. Ink/background are pooled.
            if int(known[np.ix_(iy[16:48], ix[16:48])].sum()) < 128:
                continue
            eligible.append((y, x))
    return farthest(eligible), len(eligible)


def main():
    selections = {}
    for name, shape in [("814", (2628, 3378)), ("9b", (6665, 5450))]:
        selected, eligible = from_coordinates(name, shape)
        selections[name] = {"eligible_contexts": eligible, "selection_source": "published_coordinates", "origins_target_yx": selected.tolist()}
    selected, eligible = calibration()
    selections["841"] = {"eligible_contexts": eligible, "selection_source": "calibration_known_label_support_only", "origins_target_yx": selected.tolist()}
    for name, result in selections.items():
        a = np.asarray(result["origins_target_yx"])
        d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)
        d[np.diag_indices_from(d)] = np.inf
        result["minimum_center_distance_um"] = float(d.min() * TARGET)
        input_overlap = sum(bool(np.all(np.abs(a[i] - a[j]) < 64)) for i in range(len(a)) for j in range(i))
        core_overlap = sum(bool(np.all(np.abs(a[i] - a[j]) < 32)) for i in range(len(a)) for j in range(i))
        result["overlapping_input_pairs"] = input_overlap
        result["overlapping_scored_core_pairs"] = core_overlap
    out = {"status": "metadata_selected_before_new_CT_or_detector_access", "target_pitch_um": TARGET, "input_shape_zyx": [16, WIDTH, WIDTH], "score_core_yx": [16, 16, 48, 48], "selector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "donors": selections}
    (ROOT / "selected_contexts.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
