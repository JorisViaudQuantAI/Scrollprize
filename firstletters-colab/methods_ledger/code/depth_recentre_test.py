#!/usr/bin/env python3
# ---------------------------------------------------------------------------------------------------
# methods_ledger/code/depth_recentre_test.py
# Provided for REVIEW of our arithmetic, not as a packaged tool. No support, no tests, no stable interface.
# Sanitised merge of a private lane script and the helper functions it imports: local paths were removed
# and the stack loader was reduced to a .npy reader. The estimator, the re-centring and the injected field
# are unchanged.
# Ledger entry: 01_surfaces.md S9 (depth re-centring attempt: failed its gate, dropped)
# ---------------------------------------------------------------------------------------------------
"""Injected-displacement test of a stack-based sheet-offset estimator.

From a correctly placed 41-layer render (mesh at layer 20) of a PUBLIC positive:
  ref31     layers 5..35 (what a correct 31-layer render looks like)
  disp31    synthetic mis-placement: disp[k] = render41[5 + k + d(y, x)],
            d = mean + amp * sin(2 pi x / P) * cos(2 pi y / (0.8 P)), mean 1, amp 5 voxels, P = 12 mm
  rec31     disp31 re-centred from the stack itself (two passes; bias measured on ref31)
  recref31  ref31 re-centred the same way (does the correction hurt a good render?)
Gate fixed before the numbers were read: >= 90 % of the valid area within +-1 voxel of the injected offset.
Result on the two positives: 24.5 % and 30.5 %  ->  estimator dropped.

Conventions
  stack  (D, H, W) uint8, layer k sits at (k - (D-1)/2) voxels along the render normal.
  offset field e(y, x) in voxels: the sheet (CT intensity peak) is found at layer (D-1)/2 + bias + e.
  recentre(stack, e)[k] = stack[k + e]  (linear in depth, edge-clamped).

usage: depth_recentre_test.py RENDER41.npy OUTDIR --um UM [--amp 5 --period-mm 12 --mean 1]
"""
import os, sys, json, time, argparse
import numpy as np, cv2
from scipy import ndimage as ndi


def valid_mask(stack, mid=None):
    """on-surface pixels: rendered (non-zero) in the middle layers"""
    D = stack.shape[0]; mid = D // 2 if mid is None else mid
    m = (stack[mid] > 0) & (stack[max(0, mid - 3)] > 0) & (stack[min(D - 1, mid + 3)] > 0)
    return m


def block_profile(stack, valid, blk=8):
    """block-mean depth profiles: (D, h, w) float32 over valid px, plus block validity fraction"""
    D, H, W = stack.shape; h, w = H // blk, W // blk
    v = valid[:h * blk, :w * blk].astype(np.float32)
    vb = v.reshape(h, blk, w, blk).sum((1, 3))
    prof = np.zeros((D, h, w), np.float32)
    for k in range(D):
        a = stack[k, :h * blk, :w * blk].astype(np.float32) * v
        prof[k] = a.reshape(h, blk, w, blk).sum((1, 3))
    prof /= np.maximum(vb, 1)[None]
    return prof, vb / (blk * blk)


def estimate_offset(stack, valid, bias=0.0, kmax=7.0, blk=8, sig_lat_px=24, sig_z=1.2, jump=3.0,
                    sig_field_mm=1.0, um=9.362, min_prom=0.04):
    """Sheet-centre offset field from the stack.  Returns dict(e_full (H,W) float32 voxels, flag_full (H,W) bool,
    stats).  Steps: block means (blk px) -> lateral Gaussian (sig_lat_px) -> depth Gaussian (sig_z) -> per block the
    strongest local maximum within +-kmax of the expected sheet layer, sub-voxel parabolic -> reject blocks with low
    prominence or deviating > jump from the 7x7 median -> fill by normalised convolution -> Gaussian sig_field_mm ->
    clip to +-kmax.  flag = no supported peak nearby (fill distance large) or |e| at the clip limit."""
    D = stack.shape[0]; c = (D - 1) / 2.0 + bias
    prof, vf = block_profile(stack, valid, blk)
    good = vf > 0.5
    s = sig_lat_px / blk
    num = ndi.gaussian_filter(prof * good[None], (0, s, s)); den = ndi.gaussian_filter(good.astype(np.float32), s)
    P = num / np.maximum(den, 1e-3)[None]
    P = ndi.gaussian_filter1d(P, sig_z, axis=0)
    ks = np.arange(D, dtype=np.float32)[:, None, None]
    win = np.abs(ks - c) <= kmax + 0.5
    # local maxima inside the window
    lm = np.zeros_like(P, bool); lm[1:-1] = (P[1:-1] >= P[:-2]) & (P[1:-1] > P[2:])
    lm &= win
    Pw = np.where(lm, P, -np.inf)
    kb = np.argmax(Pw, 0); has = np.isfinite(Pw.max(0))
    kb1 = np.clip(kb, 1, D - 2)
    yy, xx = np.indices(kb.shape)
    a, b_, cc = P[kb1 - 1, yy, xx], P[kb1, yy, xx], P[kb1 + 1, yy, xx]
    den2 = a - 2 * b_ + cc
    den2 = np.where(np.isfinite(den2), den2, 0.0)
    with np.errstate(all='ignore'):
        frac = np.where(np.abs(den2) > 1e-6, 0.5 * (a - cc) / np.where(np.abs(den2) > 1e-6, den2, 1.0), 0.0).clip(-0.5, 0.5)
    pk = kb1 + frac
    lo = np.where(win, P, np.inf).min(0); hi = b_
    scale = np.maximum(np.percentile(P[:, good], 99) - np.percentile(P[:, good], 1), 1e-3) if good.any() else 1.0
    prom = (hi - lo) / scale
    ok = has & good & (prom >= min_prom)
    e = np.where(ok, pk - c, np.nan).astype(np.float32)
    med = _nanmedian_filter(e, 7)
    ok2 = ok & (np.abs(e - med) <= jump)
    sig = sig_field_mm * 1000.0 / um / blk
    w0 = ok2.astype(np.float32)
    nf = ndi.gaussian_filter(np.where(ok2, e, 0), sig); df = ndi.gaussian_filter(w0, sig)
    field = nf / np.maximum(df, 1e-6)
    support = df  # ~ local fraction of supported blocks
    field = np.clip(np.nan_to_num(field), -kmax, kmax).astype(np.float32)
    flag_b = good & ((support < 0.25) | (np.abs(field) >= kmax - 0.5))
    H, W = stack.shape[1:]
    e_full = cv2.resize(field, (W, H), interpolation=cv2.INTER_LINEAR)
    f_full = cv2.resize(flag_b.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) & valid
    vv = good
    st = dict(blocks_valid=int(vv.sum()), frac_blocks_peak=float(ok[vv].mean()) if vv.any() else 0.0,
              frac_blocks_kept=float(ok2[vv].mean()) if vv.any() else 0.0,
              e_raw_pct=[round(float(x), 2) for x in np.nanpercentile(e[ok2], [5, 25, 50, 75, 95])] if ok2.any() else None,
              e_field_pct=[round(float(x), 2) for x in np.percentile(field[vv], [5, 25, 50, 75, 95])] if vv.any() else None,
              frac_abs_le2_raw=float((np.abs(e[ok2]) <= 2).mean()) if ok2.any() else None,
              frac_flagged=float(f_full[valid].mean()) if valid.any() else None,
              params=dict(bias=bias, kmax=kmax, blk=blk, sig_lat_px=sig_lat_px, sig_z=sig_z, jump=jump, sig_field_mm=sig_field_mm, um=um))
    return dict(e_full=e_full, flag_full=f_full, e_blocks=field, e_raw=e, ok=ok2, stats=st)


def _nanmedian_filter(a, size):
    """nan-aware median filter (small arrays, block grid)"""
    pad = size // 2
    ap = np.pad(a, pad, constant_values=np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    v = sliding_window_view(ap, (size, size))
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.nanmedian(v.reshape(a.shape + (-1,)), -1)


def recentre(stack, e, rows=512):
    """out[k, y, x] = stack[k + e(y, x)] (linear, edge-clamped).  e: (H,W) float."""
    D, H, W = stack.shape
    out = np.empty((D, H, W), np.uint8)
    ks = np.arange(D, dtype=np.float32)[:, None, None]
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        s = np.asarray(stack[:, y0:y1]).astype(np.float32)
        idx = np.clip(ks + e[None, y0:y1], 0, D - 1)
        i0 = np.floor(idx).astype(np.int64); i1 = np.minimum(i0 + 1, D - 1); f = idx - i0
        v = np.take_along_axis(s, i0, 0) * (1 - f) + np.take_along_axis(s, i1, 0) * f
        zero = s[D // 2] == 0  # keep the render's invalid pixels at 0
        v[:, zero] = 0
        out[:, y0:y1] = np.clip(v + 0.5, 0, 255).astype(np.uint8)
    return out


def displace_window(stack41, d, n_out=31):
    """synthetic mis-placed render: out[k] = stack41[(D41-n_out)/2 + k + d(y,x)] (linear, clamped)"""
    D = stack41.shape[0]; base = (D - n_out) / 2.0
    full = recentre(stack41, d.astype(np.float32))  # out[k] = s[k + d]
    # take n_out layers starting at base (integer when D-n_out even)
    b = int(round(base))
    return np.ascontiguousarray(full[b:b + n_out])


def shrink(a, f, interp=cv2.INTER_AREA):
    h, w = a.shape[:2]
    return cv2.resize(a, (max(1, int(round(w / f))), max(1, int(round(h / f)))), interpolation=interp)


def estimate_iter(stack, valid, n=2, **kw):
    """estimate -> recentre -> estimate the residual, n times; returns (total field dict, recentred stack)"""
    tot = None; cur = stack; flag = None; hist = []
    for i in range(n):
        r = estimate_offset(cur, valid, **kw)
        tot = r['e_full'] if tot is None else np.clip(tot + r['e_full'], -kw.get('kmax', 7.0) - 2, kw.get('kmax', 7.0) + 2)
        flag = r['flag_full'] if flag is None else (flag | r['flag_full'])
        hist.append(r['stats'])
        cur = recentre(stack, tot)
    vv = valid
    st = dict(passes=hist, e_total_pct=[round(float(x), 2) for x in np.percentile(tot[vv], [1, 5, 25, 50, 75, 95, 99])] if vv.any() else None,
              frac_abs_gt4=float((np.abs(tot[vv]) > 4).mean()) if vv.any() else None,
              frac_flagged=float(flag[vv].mean()) if vv.any() else None)
    return dict(e_full=tot.astype(np.float32), flag_full=flag, stats=st), cur


ap = argparse.ArgumentParser()
ap.add_argument('render'); ap.add_argument('out'); ap.add_argument('--um', type=float, required=True)
ap.add_argument('--amp', type=float, default=5.0); ap.add_argument('--period-mm', type=float, default=12.0)
ap.add_argument('--mean', type=float, default=1.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); t0 = time.time()
s41 = np.load(a.render, mmap_mode='r')
D, H, W = s41.shape; assert D == 41, D
valid = valid_mask(s41)
cv2.imwrite(f'{a.out}/mask.png', valid.astype(np.uint8) * 255)
ref = np.ascontiguousarray(s41[5:36]); np.save(f'{a.out}/ref31.npy', ref)
# bias: where the CT peak sits in a well-placed render
r0 = estimate_offset(ref, valid, bias=0.0, um=a.um, sig_field_mm=0.5)
bias = float(np.nanmedian(r0['e_raw'][r0['ok']])) if r0['ok'].any() else 0.0
# known displacement
yy, xx = np.indices((H, W), dtype=np.float32)
P = a.period_mm * 1000.0 / a.um
d = (a.mean + a.amp * np.sin(2 * np.pi * xx / P) * np.cos(2 * np.pi * yy / (0.8 * P))).astype(np.float32)
disp = displace_window(s41, d); np.save(f'{a.out}/disp31.npy', disp)
r1, rec = estimate_iter(disp, valid, n=2, bias=bias, um=a.um, sig_field_mm=0.5); np.save(f'{a.out}/rec31.npy', rec)
r2, recref = estimate_iter(ref, valid, n=2, bias=bias, um=a.um, sig_field_mm=0.5); np.save(f'{a.out}/recref31.npy', recref)
err = (r1['e_full'] + d)[valid]
err2 = (r1['e_full'] + d - r2['e_full'])[valid]  # relative to what the estimator says about the good render
st = dict(shape=[D, H, W], valid_px=int(valid.sum()), valid_cm2=round(float(valid.sum()) * (a.um * 1e-4) ** 2, 3), bias=round(bias, 2),
          ref_offset_stats=r0['stats'], d_pct=[round(float(x), 2) for x in np.percentile(d[valid], [1, 5, 50, 95, 99])],
          disp_est=r1['stats'], recref_est=r2['stats'],
          est_err_abs_pct=[round(float(x), 2) for x in np.percentile(np.abs(err), [50, 75, 90, 95, 99])],
          est_err_mean=round(float(err.mean()), 3), frac_err_le1=round(float((np.abs(err) <= 1).mean()), 3),
          frac_err_le2=round(float((np.abs(err) <= 2).mean()), 3),
          rel_err_abs_pct=[round(float(x), 2) for x in np.percentile(np.abs(err2), [50, 75, 90, 95, 99])], seconds=round(time.time() - t0, 1))
f8 = lambda x: shrink(x.astype(np.float32), 8, cv2.INTER_AREA)
np.savez_compressed(f'{a.out}/fields.npz', d=f8(d), e_disp=f8(r1['e_full']), e_ref=f8(r2['e_full']), flag_disp=f8(r1['flag_full']))
json.dump(st, open(f'{a.out}/prep.json', 'w'), indent=1)
print('POSPREP', json.dumps({k: st[k] for k in ('valid_cm2', 'bias', 'd_pct', 'est_err_abs_pct', 'est_err_mean', 'frac_err_le1', 'frac_err_le2', 'rel_err_abs_pct', 'seconds')}))
