#!/usr/bin/env python3
# ---------------------------------------------------------------------------------------------------
# methods_ledger/code/onsheet_gate_m7.py
# Provided for REVIEW of our arithmetic, not as a packaged tool. No support, no tests, no stable interface.
# Sanitised extract of two private lane scripts (the window check and the first lines of the ink job):
# local paths were removed, and the cross-section and en-face drawing code was dropped. The pixel
# classification, the 1 mm block rule and the connected-core gate are unchanged.
# Ledger entry: 01_surfaces.md S1 (patch-growing windows and the on-sheet gate)
# ---------------------------------------------------------------------------------------------------
"""On-sheet test of a patch-grown window BEFORE any ink, against the m7 surface prediction.

Inputs: a 41-layer render of the CT on the mesh (mesh at layer 20) and a render of the m7 surface-prediction
volume on the SAME mesh with the SAME call (41 layers, one voxel per layer along the mesh normal).

Per pixel (evaluated at 1/4 resolution), from the m7 render thresholded at > 127:
  class 1 'off'      no m7 band within +-3 voxels of the mesh
  class 2 'crowded'  a band within +-3 voxels AND another band within +-9 voxels (more than one rising edge)
  class 3 'clean'    a band within +-3 voxels and no other band within +-9 voxels
1 mm block: 'on-sheet' when >= 60 % of its valid pixels are class 2 or 3.
on_sheet_fraction = on-sheet block area / valid area;  on-sheet cm2 = that fraction x valid cm2.
core               = largest 4-connected region of on-sheet blocks.
Gate as used: ink the window only when core >= 4 cm2 (this replaced "on_sheet_fraction >= 0.5" before any window
was inked). A secondary arm inked windows with >= 5 cm2 on-sheet in total; it was later restricted by an eye check
of the mid layer, because this pixel test overstates usable area in crushed zones.

usage: onsheet_gate_m7.py CT_RENDER.npy M7_RENDER.npy UM_PER_PX     (both arrays (41, H, W) uint8)
"""
import sys, json, math
import numpy as np, cv2


def classify(m7r, step=4):
    """m7 render (L, H, W) -> class map at 1/step res."""
    L = m7r.shape[0]; c = (L - 1) // 2
    H, W = m7r.shape[1:]
    out = np.zeros((math.ceil(H / step), math.ceil(W / step)), np.uint8)
    rows = 256
    for y0 in range(0, H, rows):
        b = np.asarray(m7r[:, y0:y0 + rows, :])[:, ::step, ::step] > 127   # (L, h, w)
        k = np.arange(L)[:, None, None] - c
        on = (b & (np.abs(k) <= 3)).any(0)
        sub = b[c - 9:c + 10]                                                # runs inside |k| <= 9: count rising edges
        edges = (np.diff(np.concatenate([np.zeros((1,) + sub.shape[1:], bool), sub], 0).astype(np.int8), axis=0) == 1).sum(0)
        cls = np.where(on, np.where(edges <= 1, 3, 2), 1).astype(np.uint8)
        out[y0 // step:y0 // step + cls.shape[0], :cls.shape[1]] = cls
    return out


def gate(R, M, um, q=4):
    L, H, W = R.shape; c = (L - 1) // 2
    valid = np.asarray(R[c]) > 0
    validq = cv2.resize(valid.astype(np.uint8), (math.ceil(W / q), math.ceil(H / q)), interpolation=cv2.INTER_NEAREST) > 0
    valid_cm2 = float(valid.sum() * (um * 1e-4) ** 2)
    cls = classify(M, q)[:validq.shape[0], :validq.shape[1]]
    cls[~validq] = 0
    bpx = max(1, int(round(1000 / um / q)))                                  # 1 mm blocks
    on_px = (cls >= 2).astype(np.float32); cl_px = (cls == 3).astype(np.float32); v_px = validq.astype(np.float32)

    def block_frac(m):
        hb, wb = math.ceil(m.shape[0] / bpx), math.ceil(m.shape[1] / bpx)
        pad = np.zeros((hb * bpx, wb * bpx), np.float32); pad[:m.shape[0], :m.shape[1]] = m
        return pad.reshape(hb, bpx, wb, bpx).sum((1, 3))
    vb = block_frac(v_px); ob = block_frac(on_px); cb = block_frac(cl_px)
    on_blk = (ob / np.maximum(vb, 1)) >= 0.6
    cl_blk = (cb / np.maximum(vb, 1)) >= 0.6
    on_frac = float((vb * on_blk).sum() / max(vb.sum(), 1))
    cl_frac = float((vb * cl_blk).sum() / max(vb.sum(), 1))
    onmask = np.kron(on_blk.astype(np.uint8), np.ones((bpx, bpx), np.uint8))[:validq.shape[0], :validq.shape[1]] & validq
    n, _, st, _ = cv2.connectedComponentsWithStats(onmask.astype(np.uint8), connectivity=4)
    core_px = int(st[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    core_cm2 = core_px * (q * um * 1e-4) ** 2
    # layer profile -> brightest layer near the mesh ("peak offset" used to centre the detector window)
    prof = [float(np.asarray(R[k])[::4, ::4][valid[::4, ::4]].mean()) if valid.any() else 0.0 for k in range(L)]
    pk = int(np.argmax(prof[c - 6:c + 7])) - 6
    return dict(valid_cm2=round(valid_cm2, 3), on_sheet_fraction=round(on_frac, 3), clean_fraction=round(cl_frac, 3),
                on_sheet_cm2=round(on_frac * valid_cm2, 3), core_cm2=round(core_cm2, 3), peak_offset=pk,
                gate='ink' if core_cm2 >= 4.0 else 'uninformative')


if __name__ == '__main__':
    R = np.load(sys.argv[1], mmap_mode='r'); M = np.load(sys.argv[2], mmap_mode='r')
    print(json.dumps(gate(R, M, float(sys.argv[3]))))
