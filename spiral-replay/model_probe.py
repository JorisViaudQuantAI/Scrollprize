"""One-thread geometry-only component test of Villa's actual SpiralAndTransform.

Requires the six geometry modules from the pinned checkout. The standalone
driver is deliberately smaller than the hosted FitContext. On CPU-only hosts
with no Pyro/torchdiffeq, the optional fallback supplies PyTorch's native
Transform re-exports; its unavailable ODE entry point fails if ever used.
Stationary RK4 uses the actual eager implementation in flow_fields.py.
"""
import argparse
import json
import os
import resource
import sys
import time
import types

import numpy as np
import torch

from arc_probe import ROOT, representative_arcs
from stage import PIN, acquire, CODE
from umbilicus import json_umbilicus_z_to_yx
from sample_spiral import get_theta_and_radii, unwrap_shifted_radii


def get_fitter(fallback):
    reports = []
    if fallback:
        try:
            import pyro.distributions
        except ImportError:
            from torch import distributions
            package = types.ModuleType("pyro")
            package.distributions = distributions
            sys.modules["pyro"] = package
            sys.modules["pyro.distributions"] = distributions
            reports.append("PyTorch real_vector, Transform, AffineTransform and ComposeTransform via its own distributions module")
        try:
            import torchdiffeq
        except ImportError:
            def unreachable(*args, **kwargs):
                raise RuntimeError("unavailable optional time-varying solver was reached")
            module = types.ModuleType("torchdiffeq")
            module.odeint = unreachable
            sys.modules["torchdiffeq"] = module
            reports.append("time-varying torchdiffeq solver guarded by an error; stationary RK4 only")
    from config import Config
    from transforms import SpiralAndTransform
    return Config, SpiralAndTransform, reports


def residuals(model, xyz, pitch):
    canonical = model.get_slice_to_spiral_transform()(xyz.reshape(-1,3)).reshape(xyz.shape)
    theta, _, radii = get_theta_and_radii(canonical[..., 1:], pitch)
    shifted, _ = unwrap_shifted_radii(theta, radii, pitch)
    return shifted - shifted.median(dim=1,keepdim=True).values


def metrics(array):
    vals = np.abs(array).mean(axis=1)
    return {"n":len(vals), "median":float(np.median(vals)), "p90":float(np.quantile(vals,.9)),
            "within_5_mean_per_arc":float(np.mean(vals<=5))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-minimal-cpu-imports",action="store_true")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--steps",type=int,default=40)
    args=parser.parse_args()
    os.environ.setdefault("FIT_SPIRAL_TRITON","0")
    os.environ.setdefault("FIT_SPIRAL_COMPILE","0")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(358)
    device=torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA is unavailable")
    if device.type == "cuda" and args.allow_minimal_cpu_imports:
        raise SystemExit("CUDA replay uses the supported Pyro and torchdiffeq imports")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for name in ["flow_triton.py","gap_triton.py","transforms.py","flow_fields.py"]:
        acquire(CODE+"/"+name,ROOT/"upstream"/name)
    Config, SpiralAndTransform, imports = get_fitter(args.allow_minimal_cpu_imports)
    manifest=json.loads((ROOT/"sources.json").read_text())
    planes=[p["z"] for p in manifest["planes"]]
    center=json_umbilicus_z_to_yx(ROOT/"umbilicus.json",coordinate_scale=1)
    points, sites, _=representative_arcs(planes,center)
    pcs=points.float().numpy()+center(sites[:,0])[:,None,:]
    world=np.concatenate((np.broadcast_to(sites[:,0,None,None],(*pcs.shape[:2],1)),pcs),axis=2).astype("float32")
    world=torch.from_numpy(world).to(device)
    ztrain=np.isin(sites[:,0], [9252,9852,10452])
    train=world[ztrain]
    val=world[~ztrain]
    zg=np.arange(9200-160,10700+160+1,dtype=np.float32)
    umb=torch.from_numpy(np.column_stack((zg,center(zg)))).float().to(device)
    cfg=Config().as_dict()
    # Coarse 3-D lattice and half-size mini-batches for this limited component test.
    cfg.update(model_flow_voxel_resolution=192, model_gap_expander_logit_resolution=192,
               model_linear_z_resolution=192,model_num_flow_integration_steps=1,
               model_flow_field_direct_lr=False,model_gap_expander_num_windings=260)
    model=SpiralAndTransform(1,"rk4",torch.tensor([9040,-4096,-4096],device=device),
                             torch.tensor([10860,4096,4096],device=device),umb,cfg,"CW").to(device)
    # Freeze gap, physical pitch and linear warp deliberately. Only the official
    # continuous spatial velocity field may change the within-arc dispersion.
    for par in model.parameters():par.requires_grad_(False)
    for par in model.flow_field.parameters():par.requires_grad_(True)
    opt=torch.optim.Adam(model.flow_field.parameters(),lr=.00010)
    pitch=torch.tensor(16.,device=device)
    def evaluates(rows):
        with torch.no_grad():
            return metrics(torch.cat([residuals(model,block,pitch).cpu() for block in rows.split(16)]).numpy())
    start=time.monotonic()
    initial={"train":evaluates(train),"unseen":evaluates(val)}
    generator=torch.Generator().manual_seed(358)
    for i in range(args.steps):
        sample=torch.randperm(len(train),generator=generator)[:16]
        opt.zero_grad(set_to_none=True)
        # The custom adjoint routes field gradients to its own accumulator only
        # when points require gradients, even with the fixed umbilicus.
        r=residuals(model,train[sample].detach().requires_grad_(True),pitch)
        loss=(torch.sqrt(r.square()+1.)-1.).mean()
        loss.backward()
        model.flow_field.apply_accumulated_field_grad()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.flow_field.parameters()):
            raise RuntimeError("missing or nonfinite field gradient")
        opt.step()
    after={"train":evaluates(train),"unseen":evaluates(val)}
    if device.type == "cuda":torch.cuda.synchronize()
    result={"upstream":PIN,"device":str(device),"conditional_imports":imports,"train_z":np.unique(sites[ztrain,0]).tolist(),
            "unseen_z":np.unique(sites[~ztrain,0]).tolist(),"steps":args.steps,"mini_batch_arcs":16,
            "parameters_trained":sum(p.numel() for p in model.flow_field.parameters()),"pitch_vox":16,
            "initial":initial,"after":after,"elapsed_s_with_evaluation":time.monotonic()-start,
            "max_rss_mib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            "max_cuda_allocated_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None}
    (ROOT/f"model-results-{device.type}.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
