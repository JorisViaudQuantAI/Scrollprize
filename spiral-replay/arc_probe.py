"""Measure a two-parameter affine spiral initialization on real transverse M7 tracks.

The common z-axis is the published umbilicus. The two parameters in a
symmetric traceless matrix control ellipticity and tilt. All winding-specific
translations remain fixed; a trace receives no absolute winding number.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from stage import ROOT

sys.path.insert(0, str(ROOT / "upstream"))
from sample_spiral import get_spiral_yxs, get_theta_and_radii, unwrap_shifted_radii
from umbilicus import json_umbilicus_z_to_yx


def representative_arcs(planes, center):
    rows, midpoint_rows = [], []
    selected = {}
    for zi, z in enumerate(planes):
        xyzw = np.memmap(ROOT / "data" / f"plane-{z}.i32", "<i4", "r").reshape(-1, 3)
        offsets = np.load(ROOT / "data" / f"plane-{z}-offsets.npy", allow_pickle=False)
        counts = np.diff(offsets)
        long = np.flatnonzero(counts >= 96)
        midpoints = xyzw[offsets[long] + counts[long]//2].astype(float)
        centered = midpoints[:, 1:] - center(z)
        octants = np.floor((np.arctan2(centered[:, 0], centered[:, 1]) % (2*np.pi)) / (np.pi/4)).astype(int)
        selected[z] = [0]*8
        for octant in range(8):
            in_octant = long[octants == octant]
            if not len(in_octant):
                continue
            # Fixed producer-order quantiles for coverage; independent of fit or ink.
            chosen = in_octant[np.linspace(0, len(in_octant)-1, min(32, len(in_octant)), dtype=int)]
            selected[z][octant] = len(chosen)
            for track in chosen:
                lo, count = offsets[track], counts[track]
                positions = lo + np.linspace(0, count-1, 24, dtype=int)
                points = xyzw[positions, 1:].astype(np.float64) - center(z)
                rows.append(points)
                mid = lo + count//2
                tangent = xyzw[mid+8,1:] - xyzw[mid-8,1:]
                midpoint_rows.append((int(z),int(track),int(octant),*xyzw[mid,1:].tolist(),*tangent.tolist()))
    return torch.from_numpy(np.stack(rows)), np.asarray(midpoint_rows, dtype=np.int64), selected


def transform(params):
    # Determinant one fixes the common scale. Shrinking every track is forbidden.
    a, b = params.unbind()
    return torch.matrix_exp(torch.stack((torch.stack((a, b)),torch.stack((b, -a)))))


def discrepancies(points, jacobian, hand):
    relative = points @ jacobian
    yx = torch.stack((hand * relative[..., 0], relative[..., 1]), dim=-1)
    pitch = yx.new_tensor(16.)  # Villa's model_initial_dr_per_winding at 0fb38c45.
    theta, _, shifted = get_theta_and_radii(yx, pitch)
    radii, _ = unwrap_shifted_radii(theta, shifted, pitch)
    centered = radii - radii.median(dim=-1,keepdim=True).values
    return centered


def describe(x):
    return {name: float(torch.quantile(x, q)) for name,q in [('p50',.5),('p90',.9)]}


def measured(error, mask):
    mae = error[mask].abs().mean(dim=-1)
    return {"n": len(mae), "arc_mean_abs_deviation_in_initial_voxels": describe(mae),
            "within_five_voxels_mean_per_arc": float((mae <= 5).double().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, nargs="+", default=[9252,9852,10452])
    args = parser.parse_args()
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = json.loads((ROOT / "sources.json").read_text())
    planes = [row["z"] for row in manifest["planes"]]
    center = json_umbilicus_z_to_yx(ROOT / "umbilicus.json", coordinate_scale=1.0)
    points, sites, groups = representative_arcs(planes, center)
    train = torch.from_numpy(np.isin(sites[:,0], args.train))
    held = ~train
    if not train.any() or not held.any():
        raise ValueError("both training and unseen axial slices are required")
    result = {"data_restriction": "horizontal M7 curves with at least 96 stored points; up to 32 per axial octant",
              "z_train": sorted(set(sites[train.numpy(),0].tolist())), "z_held": sorted(set(sites[held.numpy(),0].tolist())),
              "track_counts_by_octant": groups, "pitch_native_voxels": 16, "samples_along_arc":24,
              "native_umbilicus_control_points": len(json.loads((ROOT / "umbilicus.json").read_text())["control_points"]),
              "results": {}}
    core = get_spiral_yxs(32,16.,8.,group_by_winding=True,device="cpu")
    result["official_sampler_check"] = {"windings_generated":len(core),"finite_yx":all(torch.isfinite(a).all().item() for a in core),
                                        "samples":sum(len(a) for a in core)}
    for hand in [1.,-1.]:
        params = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([params], lr=.025)
        with torch.no_grad():
            init = discrepancies(points,transform(params),hand)
        for step in range(100):
            opt.zero_grad()
            error = discrepancies(points[train], transform(params), hand)
            # Smooth absolute loss, equal weight per sampled track.
            objective = (torch.sqrt(error.square()+1.)-1.).mean()
            objective.backward()
            opt.step()
        with torch.no_grad():
            jac = transform(params)
            final = discrepancies(points,jac,hand)
            result["results"][("unreflected_y" if hand>0 else "reflected_y")] = {
                "det": float(torch.det(jac)), "matrix_yx": jac.tolist(), "train_initial":measured(init,train),
                "train_affine":measured(final,train),"held_initial":measured(init,held),"held_affine":measured(final,held)}
    result["cpu"] = {"torch":torch.__version__, "cuda_build":torch.version.cuda, "cuda_available":torch.cuda.is_available(),
                     "cuda_device_count":torch.cuda.device_count(),"torch_threads":torch.get_num_threads(),
                     "elapsed_s":time.monotonic()-start,"max_rss_mib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    np.save(ROOT / "data" / "normal-probe-sites.npy",sites,allow_pickle=False)
    (ROOT / "arc-results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
