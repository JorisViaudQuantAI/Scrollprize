#!/usr/bin/env python3
"""Fit a moving->fixed transform to point correspondences and write it in the layout the
open-data bucket uses for <Sample>/transforms/<moving>-to-<fixed>-r1/ (fit/transform.json, schema 1.0.0:
a 3x4 matrix mapping moving level-0 voxel xyz to fixed level-0 voxel xyz, plus the landmarks).

Models: 'scale' (isotropic scale + translation, 4 params), 'similarity' (scale + rotation +
translation, 7 params, Umeyama) and 'affine' (12 params). For each, the in-sample RMS and the
leave-one-out RMS (each point predicted by a fit to the others) are printed, in fixed (coarse)
voxels and in moving (fine) voxels.

usage: fit_landmarks.py landmarks.json --sample PHerc1203 [--model similarity] [--out DIR]
"""
import argparse
import datetime
import json
import os

import numpy as np


def fit(model, P, Q):
    """Return 3x4 M with Q ~ M @ [P, 1]."""
    if model == "affine":
        A = np.hstack([P, np.ones((len(P), 1))])
        X, *_ = np.linalg.lstsq(A, Q, rcond=None)
        return X.T
    mp, mq = P.mean(0), Q.mean(0)
    Pc, Qc = P - mp, Q - mq
    if model == "scale":
        s = np.sqrt((Qc ** 2).sum() / (Pc ** 2).sum()) * np.sign((Pc * Qc).sum())
        R = np.eye(3) * s
    else:  # similarity, Umeyama 1991, reflections excluded
        U, S, Vt = np.linalg.svd(Qc.T @ Pc / len(P))
        d = np.sign(np.linalg.det(U @ Vt))
        D = np.diag([1, 1, d])
        rot = U @ D @ Vt
        s = (S * np.diag(D)).sum() / ((Pc ** 2).sum() / len(P))
        R = s * rot
    t = mq - R @ mp
    return np.hstack([R, t[:, None]])


def apply(M, P):
    return P @ M[:, :3].T + M[:, 3]


def rms(E):
    return float(np.sqrt((E ** 2).sum(1).mean()))


def report(model, P, Q, scale):
    M = fit(model, P, Q)
    E = apply(M, P) - Q
    loo = []
    for i in range(len(P)):
        k = np.arange(len(P)) != i
        loo.append(apply(fit(model, P[k], Q[k]), P[i:i + 1])[0] - Q[i])
    loo = np.array(loo)
    out = {
        "model": model,
        "landmark_pairs": len(P),
        "in_sample_rms_fixed_voxels": round(rms(E), 3),
        "leave_one_out_rms_fixed_voxels": round(rms(loo), 3),
        "leave_one_out_max_fixed_voxels": round(float(np.sqrt((loo ** 2).sum(1)).max()), 3),
        "leave_one_out_rms_moving_voxels": round(rms(loo) / scale, 2),
        "per_point_leave_one_out_xyz_fixed_voxels": [[round(float(v), 2) for v in r] for r in loo],
    }
    return M, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("landmarks", help="landmarks.json (see the file in this folder)")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--model", choices=("scale", "similarity", "affine"), default="similarity",
                    help="model written to transform.json (all three are reported)")
    ap.add_argument("--out", help="folder to write <moving>-to-<fixed>-r1/ into (omit to only report)")
    a = ap.parse_args()
    L = json.load(open(a.landmarks))[a.sample]
    P = np.array([p["moving"] for p in L["points"]], float)
    Q = np.array([p["fixed"] for p in L["points"]], float)
    scale_guess = np.linalg.norm(Q - Q.mean(0)) / np.linalg.norm(P - P.mean(0))
    results = {}
    for m in ("scale", "similarity", "affine"):
        if m == "affine" and len(P) < 5:
            continue
        M, r = report(m, P, Q, scale_guess)
        results[m] = (M, r)
        print(json.dumps(r))
    M, r = results[a.model]
    R = M[:, :3]
    sv = np.linalg.svd(R, compute_uv=False)
    rot = R / np.cbrt(np.linalg.det(R))
    angle = np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    geo = {"scale": round(float(np.cbrt(np.linalg.det(R))), 6),
           "anisotropy": round(float(sv.max() / sv.min()), 5),
           "rotation_deg": round(float(angle), 3),
           "reflected": bool(np.linalg.det(R) < 0)}
    print("chosen", a.model, json.dumps(geo))
    if not a.out:
        return
    mid = L["moving_volume"].split("-")[0]
    fid = L["fixed_volume"].split("-")[0]
    d = os.path.join(a.out, f"{mid}-to-{fid}-r1")
    os.makedirs(os.path.join(d, "fit"), exist_ok=True)
    tr = {
        "schema_version": "1.0.0",
        "fixed_volume": L["fixed_volume"],
        "transformation_matrix": M.tolist(),
        "fixed_landmarks": Q.tolist(),
        "moving_landmarks": P.tolist(),
    }
    json.dump(tr, open(os.path.join(d, "fit", "transform.json"), "w"), indent=1)
    prov = {
        "source": "community registration (not reviewed by the organisers)",
        "moving_volume": L["moving_volume"],
        "fixed_volume": L["fixed_volume"],
        "method": "per-point NCC of band-passed CT z-slices (reg_point.py), then a "
                  f"{a.model} least-squares fit (fit_landmarks.py)",
        "fit": r,
        "geometry": geo,
        "all_models": {k: v[1] for k, v in results.items()},
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "caveat": "Landmarks are 2-D slice matches with coarse-z step 2; no hold-out patch review and no "
                  "deformable part, unlike the organisers' volreg records. See README.",
    }
    json.dump(prov, open(os.path.join(d, "provenance.json"), "w"), indent=1)
    print("wrote", d)


if __name__ == "__main__":
    main()
