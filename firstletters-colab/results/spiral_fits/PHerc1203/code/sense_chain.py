"""Winding sense from chains of transverse M7 tracks linked through the published exact crossings.
Image convention: slice z of the zarr (z, y, x); x = column (right), y = row (down); theta = atan2(y - cy, x - cx)
with (cy, cx) the official umbilicus at that z.  A chain starts at the low-theta end of a transverse track near z0,
walks towards increasing theta; at the front end of the current transverse track it hops along a longitudinal
(family-1) track that crosses it to another transverse track within +-DZ of z0 that continues further in theta.
After the accumulated theta reaches 2*pi the radius there is compared with the radius at the start:
dr > 0 means the radius grows with increasing theta = villa spiral_outward_sense 'CW' (transforms.py: y = r sin t,
x = r cos t, r increasing with t; 'ACW' mirrors x)."""
import sys, os, json, numpy as np
D = os.environ['SC_TRACKS'] + '/'
z0 = int(sys.argv[1]); DZ = int(sys.argv[2]) if len(sys.argv) > 2 else 40; NSTART = int(sys.argv[3]) if len(sys.argv) > 3 else 3000; RTOL = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0; FRONT = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3; JMAX = int(sys.argv[6]) if len(sys.argv) > 6 else 200; ZSTEP = float(sys.argv[7]) if len(sys.argv) > 7 else 1e9
off = np.fromfile(D + 'offsets.i64', '<i8'); zb = np.fromfile(D + 'z_bounds.i32', '<i4').reshape(-1, 2)
fam = np.fromfile(D + 'family_codes.i8', 'i1')
coords = np.memmap(D + 'coordinates.i32', '<i4', 'r').reshape(-1, 3)
X = np.load(os.environ['SC_CROSS'], mmap_mode='r')
coff, cpart, cself, cploc = X['offsets'], X['partners'], X['self_local'], X['partner_local']
cp = sorted(json.load(open(os.environ['SC_UMB']))['control_points'], key=lambda p: p['z'])
uz = np.array([p['z'] for p in cp], float); uy = np.array([p['y'] for p in cp], float); ux = np.array([p['x'] for p in cp], float)
H = np.nonzero((fam == 0) & (zb[:, 0] >= z0 - DZ) & (zb[:, 1] <= z0 + DZ))[0]
Hset = set(H.tolist())
MIR = float(sys.argv[9]) if len(sys.argv) > 9 else 1.0
geo = {}
def g(h):
    if h not in geo:
        a = np.asarray(coords[off[h]:off[h + 1]]).astype(float)
        z = a[0, 0]; cy, cx = np.interp(z, uz, uy), np.interp(z, uz, ux)
        th = np.unwrap(np.arctan2(a[:, 1] - cy, MIR * (a[:, 2] - cx))); r = np.hypot(a[:, 1] - cy, a[:, 2] - cx)
        geo[h] = (z, th, r, 1.0 if th[-1] >= th[0] else -1.0)
    return geo[h]
def crossings(t):
    s, e = coff[t], coff[t + 1]
    return cpart[s:e], cself[s:e], cploc[s:e]
rng = np.random.default_rng(0)
lens = off[H + 1] - off[H]
cand = H[lens >= 40]
RMAX = float(sys.argv[8]) if len(sys.argv) > 8 else 1e9
cand = np.array([h for h in cand if np.median(g(h)[2]) < RMAX]) if RMAX < 1e8 else cand
starts = rng.choice(cand, min(NSTART, len(cand)), replace=False)
res = []; brk = []
for h0 in starts:
    z, th, r, sg = g(h0)
    i0 = 0 if sg > 0 else len(th) - 1
    start = (th[i0], r[i0], z); acc = 0.0; h = h0; ientry = i0; ok = False; zs = [z]; hops = 0
    while hops < 60:
        z, th, r, sg = g(h)
        front = len(th) - 1 if sg > 0 else 0
        span = (th[front] - th[ientry]) * sg
        if acc + span >= 2 * np.pi:  # finish inside this track
            need = 2 * np.pi - acc
            tt = (th - th[ientry]) * sg
            idx = np.arange(len(th))
            sel = idx >= ientry if sg > 0 else idx <= ientry
            o = np.argsort(tt[sel]); r1 = np.interp(need, tt[sel][o], r[sel][o])
            res.append((r1 - start[1], z - start[2], hops)); ok = True; break
        # hop: crossings of h in the front 25% of its theta range (beyond entry)
        P, SL, PL = crossings(h)
        tt = (th - th[ientry]) * sg
        best = None
        for v, il, jv in zip(P, SL, PL):
            if fam[v] != 1 or tt[il] < FRONT * span: continue
            P2, SL2, PL2 = crossings(v)
            for h2, jv2, i2 in zip(P2, SL2, PL2):
                if h2 == h or h2 not in Hset: continue
                if abs(int(jv2) - int(jv)) > JMAX: continue
                z2, th2, r2, sg2 = g(h2)
                if abs(z2 - z) > ZSTEP: continue
                dth = np.angle(np.exp(1j * (th2[i2] - th[il])))
                if abs(dth) > 0.05 or abs(r2[i2] - r[il]) > RTOL: continue
                front2 = len(th2) - 1 if sg2 > 0 else 0
                adv = (th2[front2] - th2[i2]) * sg2
                gain = tt[il] + dth + adv - span
                cost = abs(r2[i2] - r[il]) + 0.1 * abs(z2 - z0)
                if gain > 0.02 and (best is None or cost < best[0]):
                    best = (cost, h2, i2, tt[il] + dth)
        if best is None: brk.append((hops, acc)); break
        acc += best[3]; h, ientry = best[1], best[2]; hops += 1; zs.append(g(h)[0])
dr = np.array([x[0] for x in res]); dzz = np.array([x[1] for x in res])
out = dict(z0=z0, DZ=DZ, transverse_tracks_in_slab=int(len(H)), chains_started=int(len(starts)), chains_full_turn=int(len(res)),
           dr_positive=int((dr > 0).sum()), dr_negative=int((dr < 0).sum()),
           median_dr=float(np.median(dr)) if len(dr) else None, median_abs_dr=float(np.median(np.abs(dr))) if len(dr) else None,
           n_abs_dr_5_60_pos=int(((dr > 5) & (dr < 60)).sum()), n_abs_dr_5_60_neg=int(((dr < -5) & (dr > -60)).sum()),
           median_abs_dz=float(np.median(np.abs(dzz))) if len(dzz) else None, median_hops=float(np.median([x[2] for x in res])) if res else None)
out['broken']=len(brk); out['broken_median_acc_deg']=float(np.degrees(np.median([b[1] for b in brk]))) if brk else None; out['broken_max_acc_deg']=float(np.degrees(max(b[1] for b in brk))) if brk else None
print(json.dumps(out)); np.save(os.environ.get('SC_OUT', '.') + f'/sense_chain_z{z0}_m{int(MIR)}.npy', np.array(res))
