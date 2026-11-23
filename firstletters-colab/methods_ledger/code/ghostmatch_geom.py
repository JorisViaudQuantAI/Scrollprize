# ---------------------------------------------------------------------------------------------------
# methods_ledger/code/ghostmatch_geom.py
# Provided for REVIEW of our arithmetic, not as a packaged tool. No support, no tests, no stable interface.
# Sanitised copy of a private lane script: local paths were removed and the hard-coded canvas boxes of the
# region under test became arguments (the ledger gives the region in general terms only). The geometry
# (normal orientation, nearest point on the neighbouring turn, Gauss-Newton refinement) is unchanged.
# Ledger entry: 04_review_and_false_leads.md F5 (ghost text from the neighbouring wrap, PHerc. Paris 4)
# ---------------------------------------------------------------------------------------------------
"""Ghost letter-match check, step 1 (geometry), on a published multi-turn segment of a READ scroll.

For a grid of points of the tested strip (canvas coordinates, every 40 canvas px):
  P = mesh point, n = unit normal oriented OUTWARD (away from the umbilicus),
  Q = P + SHIFT_UM along n  (= centre of the detector window in which the letter rows appeared).
Find the nearest point S on the NEXT OUTER TURN of the same mesh (canvas nodes with x < x_P - MIN_DX),
refined to sub-node accuracy by Gauss-Newton on the bilinear surface. Store the canvas (u, v) of S,
|Q-S|, |P-S| and the signed normal offset of S from P. The outer turn is the same published segment
(its earlier turn), so its 'UV' is its own canvas.

usage: ghostmatch_geom.py MESH_TIFXYZ_DIR UMBILICUS.json OUT.npz  X0 X1 Y0 Y1  CAND_X0 CAND_X1 CAND_Y0 CAND_Y1
                          [--shift-um 125] [--voxel-um 2.4] [--min-dx 4000] [--node-step 20]
  X0..Y1        tested strip, canvas px;  CAND_*  canvas box in which outer-turn nodes are searched
"""
import sys, json
import numpy as np
import tifffile
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates


def opt(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


d, umb_json, out_npz = sys.argv[1].rstrip('/') + '/', sys.argv[2], sys.argv[3]
BX0, BX1, BY0, BY1, CX0, CX1, CY0, CY1 = [int(v) for v in sys.argv[4:12]]
SHIFT_UM, VOX, MIN_DX, STEP = opt('--shift-um', 125.0), opt('--voxel-um', 2.4), opt('--min-dx', 4000.0), int(opt('--node-step', 20))

X = tifffile.imread(d + 'x.tif').astype(np.float64); Y = tifffile.imread(d + 'y.tif').astype(np.float64)
Z = tifffile.imread(d + 'z.tif').astype(np.float64)
V = (X > 0) & np.isfinite(X)
u = json.load(open(umb_json))
U = np.array([[q['x'], q['y'], q['z']] for q in u['control_points']], float); U = U[np.argsort(U[:, 2])]


def surf(xc, yc):
    """bilinear xyz at canvas coords (arrays)"""
    co = [np.asarray(yc) / STEP, np.asarray(xc) / STEP]
    return np.stack([map_coordinates(A, co, order=1) for A in (X, Y, Z)], -1)


def jac(xc, yc, h=10.0):
    dx = (surf(xc + h, yc) - surf(xc - h, yc)) / (2 * h)
    dy = (surf(xc, yc + h) - surf(xc, yc - h)) / (2 * h)
    return dx, dy


gy, gx = np.mgrid[BY0:BY1 + 1:40, BX0:BX1 + 1:40].astype(np.float64)
P = surf(gx.ravel(), gy.ravel())
dx, dy = jac(gx.ravel(), gy.ravel(), 20.0)
n = np.cross(dx, dy); n /= np.linalg.norm(n, axis=1, keepdims=True)
ux = np.interp(P[:, 2], U[:, 2], U[:, 0]); uy = np.interp(P[:, 2], U[:, 2], U[:, 1])
flip = np.einsum('ij,ij->i', n[:, :2], P[:, :2] - np.stack([ux, uy], 1)) < 0
n[flip] *= -1
print('normal flipped to outward for %.1f %% of points' % (100 * flip.mean()))
Q = P + (SHIFT_UM / VOX) * n

# outer-turn candidate nodes
r0, r1, c0, c1 = CY0 // STEP, CY1 // STEP, CX0 // STEP, CX1 // STEP
rr, cc = np.mgrid[r0:r1, c0:c1]
ok = V[r0:r1, c0:c1]
nodes = np.stack([X[r0:r1, c0:c1][ok], Y[r0:r1, c0:c1][ok], Z[r0:r1, c0:c1][ok]], 1)
ncx = cc[ok] * STEP; ncy = rr[ok] * STEP
tree = cKDTree(nodes)
dd, ii = tree.query(Q, k=64, distance_upper_bound=800 / VOX)
xo = np.full(len(Q), np.nan); yo = np.full(len(Q), np.nan)
for k in range(len(Q)):
    for j, i in zip(dd[k], ii[k]):
        if not np.isfinite(j):
            break
        if ncx[i] < gx.ravel()[k] - MIN_DX:
            xo[k] = ncx[i]; yo[k] = ncy[i]; break
good = np.isfinite(xo)
print('points with an outer-turn node within 800 um: %d / %d' % (good.sum(), len(Q)))
# Gauss-Newton refinement on the bilinear surface
for it in range(6):
    S = surf(xo[good], yo[good]); ax, ay = jac(xo[good], yo[good])
    r = Q[good] - S
    A = np.stack([ax, ay], 2)                      # (N,3,2)
    AtA = np.einsum('nij,nik->njk', A, A); Atr = np.einsum('nij,ni->nj', A, r)
    delta = np.linalg.solve(AtA + 1e-9 * np.eye(2), Atr[..., None])[..., 0]
    delta = np.clip(delta, -200, 200)
    xo[good] += delta[:, 0]; yo[good] += delta[:, 1]
S = np.full_like(Q, np.nan); S[good] = surf(xo[good], yo[good])
dQS = np.linalg.norm(Q - S, axis=1) * VOX
dPS = np.linalg.norm(P - S, axis=1) * VOX
offPS = np.einsum('ij,ij->i', S - P, n) * VOX
# angle between the two sheets' normals
ax, ay = jac(np.nan_to_num(xo), np.nan_to_num(yo), 20.0); no = np.cross(ax, ay); no /= np.linalg.norm(no, axis=1, keepdims=True)
cosang = np.abs(np.einsum('ij,ij->i', n, no))
sh = gx.shape
np.savez(out_npz, gx=gx, gy=gy, xo=xo.reshape(sh), yo=yo.reshape(sh), dQS=dQS.reshape(sh), dPS=dPS.reshape(sh),
         offPS=offPS.reshape(sh), cosang=cosang.reshape(sh), P=P.reshape(sh + (3,)), Q=Q.reshape(sh + (3,)))
g = good
print('|Q-S| um: median %.0f p10 %.0f p90 %.0f max %.0f' % tuple(np.percentile(dQS[g], [50, 10, 90, 100])))
print('|P-S| um (mesh to outer turn): median %.0f p10 %.0f p90 %.0f' % tuple(np.percentile(dPS[g], [50, 10, 90])))
print('signed normal offset of S from P (um, + outward): median %.0f p10 %.0f p90 %.0f' % tuple(np.percentile(offPS[g], [50, 10, 90])))
print('dx = xo - x: median %.0f p10 %.0f p90 %.0f ; dy = yo - y: median %.0f p10 %.0f p90 %.0f' % (
    *np.percentile((xo - gx.ravel())[g], [50, 10, 90]), *np.percentile((yo - gy.ravel())[g], [50, 10, 90])))
print('|cos| between normals: median %.3f p10 %.3f' % tuple(np.percentile(cosang[g], [50, 10])))
# orientation of the map: Jacobian of (x,y)->(xo,yo) on the grid (det > 0 = orientation-preserving, no mirror)
XO = xo.reshape(sh); YO = yo.reshape(sh)
dxo_dx = np.nanmedian(np.diff(XO, axis=1)) / 40; dyo_dy = np.nanmedian(np.diff(YO, axis=0)) / 40
dxo_dy = np.nanmedian(np.diff(XO, axis=0)) / 40; dyo_dx = np.nanmedian(np.diff(YO, axis=1)) / 40
det = np.diff(XO, axis=1)[:-1] * np.diff(YO, axis=0)[:, :-1] - np.diff(XO, axis=0)[:, :-1] * np.diff(YO, axis=1)[:-1]
print('Jacobian medians: dxo/dx %.3f dxo/dy %.3f dyo/dx %.3f dyo/dy %.3f ; det>0 for %.1f %% of cells' % (
    dxo_dx, dxo_dy, dyo_dx, dyo_dy, 100 * np.nanmean(det > 0)))
