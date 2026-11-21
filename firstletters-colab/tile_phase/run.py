#!/usr/bin/env python3
"""Two actual tile placements on checksum-pinned, already published PHerc.306B CT."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
from common import (FILES, HERE, PROTOCOL, compare, field_plan, gaussian,
                    outside_git, positions, sha, write_json)


def verified_inputs(root):
    result = {}
    for name, (_, expected) in FILES.items():
        got = sha(root/name)
        if got != expected:
            raise ValueError("Pinned input checksum mismatch: " + name)
        result[name] = got
    return result


def load_model(root, device):
    # Execute only the byte-verified public model module. Its full checkpoint
    # strictly replaces every initialized state entry; no base weights are used.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Full field requires CUDA; there is no CPU fallback")
    spec = importlib.util.spec_from_file_location("published_306b_model", root/"modeling.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = json.loads((root/"config.json").read_text())
    model = module.InkDetector(**config)
    model.load_state_dict(load_file(str(root/"seed0.safetensors"), device="cpu"), strict=True)
    return model.to(device=device, dtype=torch.float32).eval(), torch


def predict(model, torch, tile, device):
    if tile.shape != (64,64,24) or tile.dtype != np.uint8 or tile.max()>200:
        raise ValueError("Expected the native clipped 64x64x24 uint8 tile")
    tensor = torch.from_numpy(np.ascontiguousarray(tile.transpose(2,0,1)[None])).to(device).float().mul_(1/255.)
    with torch.inference_mode():
        logits = model(tensor)
        if tuple(logits.shape) != (1,1,64,64):
            raise ValueError("Unexpected model output geometry; never resize an output")
        result = torch.sigmoid(logits.float())[0,0].cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite model scores")
    return result


def runtime(torch, device):
    import safetensors
    import huggingface_hub
    return dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__,
        safetensors=safetensors.__version__, huggingface_hub=huggingface_hub.__version__,
        device=device, device_name=torch.cuda.get_device_name(0) if device=="cuda" else "CPU",
        cpu_threads=torch.get_num_threads(), autocast=False, precision="FP32",
        cuda_matmul_tf32=False, cudnn_tf32=False, deterministic_algorithms=True)


def code_receipt():
    return {p.name:sha(p) for p in (HERE/"run.py", HERE/"common.py", HERE/"protocol.json")}


def micro(args):
    if args.resume:
        raise ValueError("A two-forward micro run cannot resume or overwrite an outcome")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = verified_inputs(args.inputs)
    with np.load(args.inputs/"smoke.npz", allow_pickle=False) as z:
        st = z["st"]
        origin = [int(z["y0"]), int(z["x0"])]
        n = int(z["n"])
    if st.shape != (169,169,24) or st.dtype != np.uint8 or origin != [2110,4495] or n != 5:
        raise ValueError("Unexpected published smoke geometry")
    tops = PROTOCOL["micro"]["tile_origins_in_crop_yx"]
    tiles = [st[y:y+64,x:x+64] for y,x in tops]
    admission = [int((tile[:,:,12]>0).sum()) for tile in tiles]
    if min(admission)<1024:
        raise ValueError("A prespecified micro tile fails the published 25% CT admission rule")
    valid = st[10:64,10:64,12]>0
    if not valid.any():
        raise ValueError("No same-physical-pixel CT coverage")
    load_start = time.perf_counter()
    model, torch = load_model(args.inputs,"cpu")
    load_seconds = time.perf_counter()-load_start
    started = dict(mode="micro", started_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=code_receipt(), verified_input_sha256=inputs, runtime=runtime(torch,"cpu"),
        model_load_elapsed_seconds=load_seconds, actual_forward_calls_planned=2,
        native_crop_origin_yx=origin, tile_origins_in_crop_yx=tops,
        full_tile_positive_CT_pixels=admission,
        maximum_elapsed_inference_seconds=PROTOCOL["micro"]["maximum_elapsed_inference_seconds"])
    write_json(args.output_dir/"run_started.json",started)
    def expired(_signum, _frame):
        raise TimeoutError("Frozen micro inference budget exhausted; no completed result")
    old_handler=signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, PROTOCOL["micro"]["maximum_elapsed_inference_seconds"])
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    results, durations = [], []
    try:
        for i,tile in enumerate(tiles):
            t=time.perf_counter(); results.append(predict(model,torch,tile,"cpu")); durations.append(time.perf_counter()-t)
            print(json.dumps({"micro_forward_completed":i+1,"elapsed_seconds":durations[-1]}),flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_handler)
    a,b = results[0][10:64,10:64], results[1][:54,:54]
    # These two arrays refer to the SAME 54x54 physical source samples. Input
    # tile origins changed; neither output image is translated or interpolated.
    report=dict(started, status="completed", actual_forward_calls=2,
        inference_elapsed_seconds=time.perf_counter()-wall_start,
        process_CPU_seconds=time.process_time()-cpu_start, forward_elapsed_seconds=durations,
        common_source_crop_box_yxyx=[10,10,64,64], common_shape_yx=[54,54],
        common_side_mm=.46656, output_orientation="native render orientation",
        common_CT_fraction=float(valid.mean()), score_comparison=compare(a,b,valid),
        interpretation="Two actual input placements; a small tile-context sensitivity check, not line ranks, letter recovery, accuracy, or a false-positive estimate")
    artifacts={"micro_phase0.npy":a,"micro_phase10.npy":b,"common_valid.npy":valid}
    for name,array in artifacts.items():np.save(args.output_dir/name,array)
    report["output_sha256"]={name:sha(args.output_dir/name) for name in artifacts}
    write_json(args.output_dir/"report.json", report)
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    return 0


def add_tile(num, den, count, q, blend, origin, roi):
    """Clip a real tile's contribution to a fixed physical ROI, never its input."""
    y,x=map(int,origin); y0,x0,y1,x1=roi
    ya,xa,yb,xb=max(y,y0),max(x,x0),min(y+64,y1),min(x+64,x1)
    dest=np.s_[ya-y0:yb-y0,xa-x0:xb-x0]
    src=np.s_[ya-y:yb-y,xa-x:xb-x]
    num[dest] += q[src]*blend[src]
    den[dest] += blend[src]
    count[dest] += 1


def checkpoint(path, signature, next_indices, num, den, count, elapsed):
    temporary=path.with_name(path.name+".part.npz")
    np.savez(temporary, signature=np.array(json.dumps(signature,sort_keys=True)),
        next_indices=next_indices, num=num, den=den, count=count,
        inference_elapsed_seconds=np.array(elapsed))
    temporary.replace(path)


def field(args):
    # This check precedes model loading: even a fresh CPU environment fails fast.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Full field requires CUDA; there is no CPU fallback")
    if args.maximum_seconds<=0:
        raise ValueError("A positive bounded inference budget is required")
    args.output_dir.mkdir(parents=True,exist_ok=args.resume)
    if (args.output_dir/"report.json").exists():
        raise FileExistsError("Completed field result already exists")
    inputs=verified_inputs(args.inputs)
    prep=json.loads((args.inputs/"field_input.json").read_text())
    input_hash=sha(args.inputs/"field_input.npz")
    if input_hash != prep["input_sha256"] or not prep["published_chunk_bytes_verified"]:
        raise ValueError("Field input receipt mismatch or incomplete verification")
    plan=field_plan(); roi=plan["native_roi_yxyx"]; source=plan["required_native_source_box_yxyx"]
    if prep["source_box_yxyx"]!=source:
        raise ValueError("Field source does not include the exact prescribed real CT halo")
    with np.load(args.inputs/"field_input.npz",allow_pickle=False) as z:
        st=z["st"]; origin=z["origin_yx"]
    if list(origin)!=source[:2] or st.shape!=(source[2]-source[0],source[3]-source[1],24) or st.dtype!=np.uint8 or st.max()>200:
        raise ValueError("Unexpected prepared field input")
    points=[]
    for phase in PROTOCOL["phases_yx"]:
        allp=positions(roi,phase)
        admitted=[(int(y),int(x)) for y,x in allp if np.count_nonzero(st[y-origin[0]:y-origin[0]+64,x-origin[1]:x-origin[1]+64,12])>=1024]
        if not admitted:raise ValueError("No admitted source tiles for a phase")
        points.append(admitted)
    model,torch=load_model(args.inputs,"cuda")
    signature=dict(mode="field", code_sha256=code_receipt(), verified_input_sha256=inputs,
        field_input_sha256=input_hash, runtime=runtime(torch,"cuda"),
        source_box_yxyx=source, output_native_box_yxyx=roi,
        field_box_yxyx=PROTOCOL["field"]["prepared_context_box_yxyx"],
        phase_offsets_yx=PROTOCOL["phases_yx"], batch_size=1,
        actual_admitted_tile_counts=[len(p) for p in points])
    shape=(2,roi[2]-roi[0],roi[3]-roi[1]); state=args.output_dir/"checkpoint.npz"
    if args.resume:
        with np.load(state,allow_pickle=False) as z:
            if json.loads(str(z["signature"]))!=signature:
                raise ValueError("Resume would change the model, inputs, code, runtime, or recipe")
            num,den,count=z["num"],z["den"],z["count"]
            next_indices=z["next_indices"]; previous_elapsed=float(z["inference_elapsed_seconds"])
        if num.shape!=shape or den.shape!=shape or count.shape!=shape:
            raise ValueError("Malformed checkpoint")
        if any(i<0 or i>len(p) for i,p in zip(next_indices,points)):
            raise ValueError("Malformed checkpoint positions")
    else:
        num=np.zeros(shape,np.float32);den=np.zeros(shape,np.float32);count=np.zeros(shape,np.uint16)
        next_indices=np.zeros(2,np.int64);previous_elapsed=0.
        write_json(args.output_dir/"run_started.json",dict(signature,started_utc=datetime.now(timezone.utc).isoformat(),geometry_plan=plan))
    blend=gaussian(); start=time.perf_counter(); last=0.; current_calls=0
    for p,phase_points in enumerate(points):
        for i in range(int(next_indices[p]),len(phase_points)):
            if time.perf_counter()-start+max(last,.1)>args.maximum_seconds:
                elapsed=previous_elapsed+time.perf_counter()-start
                checkpoint(state,signature,next_indices,num,den,count,elapsed)
                print(json.dumps({"status":"incomplete_budget_checkpoint","completed_tiles":next_indices.tolist(),"required_tiles":[len(v) for v in points],"resume":"rerun the same command with --resume"}),flush=True)
                return 3
            y,x=phase_points[i];t=time.perf_counter()
            tile=st[y-origin[0]:y-origin[0]+64,x-origin[1]:x-origin[1]+64]
            q=predict(model,torch,tile,"cuda")
            add_tile(num[p],den[p],count[p],q,blend,(y,x),roi)
            next_indices[p]=i+1;last=time.perf_counter()-t;current_calls+=1
            if current_calls%32==0:
                print(json.dumps({"phase":p,"completed_tiles":i+1,"required_tiles":len(phase_points),"invocation_elapsed_seconds":time.perf_counter()-start}),flush=True)
            if current_calls%64==0:
                checkpoint(state,signature,next_indices,num,den,count,previous_elapsed+time.perf_counter()-start)
    elapsed=previous_elapsed+time.perf_counter()-start
    checkpoint(state,signature,next_indices,num,den,count,elapsed)
    ct=st[roi[0]-origin[0]:roi[2]-origin[0],roi[1]-origin[1]:roi[3]-origin[1],12]>0
    maps=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
    masks=(den>0)&ct[None,:,:]
    common=masks.all(axis=0)
    if not common.any():raise ValueError("No valid common field after actual inference")
    phase_reports=[]
    for p,tag in enumerate(("0","10")):
        np.save(args.output_dir/f"phase{tag}.npy",np.ascontiguousarray(maps[p,::-1]))
        np.save(args.output_dir/f"mask{tag}.npy",np.ascontiguousarray(masks[p,::-1]))
        phase_reports.append(dict(phase_offset_yx=PROTOCOL["phases_yx"][p],
            tiles_before_CT_admission=plan["phases"][p]["tiles_before_CT_admission"],
            tiles_after_CT_admission=len(points[p]),actual_forward_calls=int(next_indices[p]),
            full_field_minimum_contributors=int(count[p].min()),
            common_valid_minimum_contributors=int(count[p][common].min()),
            common_valid_maximum_contributors=int(count[p][common].max()),
            field_pixels_with_model_support=int((den[p]>0).sum()),
            valid_CT_and_model_pixels=int(masks[p].sum())))
    np.save(args.output_dir/"common_valid.npy",np.ascontiguousarray(common[::-1]))
    files=["phase0.npy","phase10.npy","mask0.npy","mask10.npy","common_valid.npy"]
    outputs={name:sha(args.output_dir/name) for name in files}
    recipe=dict(model={"checkpoint_sha256":PROTOCOL["weights_sha256"],"member":"A-hn seed0"},
        source_frame={"acquisition":"published PHerc306B claim-surface render", "raster":"released native306B", "native_box_yxyx":roi,"output_orientation":"vertical flip of native crop"},
        depth_order=list(range(8,32)),preprocess={"input_dtype":"uint8","clip_max":200,"scale":1/255,"resampling":"none"},
        fusion={"recipe":"single seed0 member, no TRACE product", "blend":"Gaussian CDF bins +/-3 sigma, float32 weighted sum divided by weight sum"},
        precision="fp32",tta="TTA1",tile_shape_yx_px=[64,64],stride_yx_px=[21,21],
        coverage_policy="fully supported native inputs and scored common field",
        source_global_grid_origin_yx=[115,694],field_input_sha256=input_hash,
        protocol_sha256=signature["code_sha256"]["protocol.json"])
    pair=dict(reference_case="pherc306b_context_raw",experiment_kind="tile_grid_pair",
        field_box_yxyx=PROTOCOL["field"]["prepared_context_box_yxyx"],
        maps=[dict(map=f"phase{tag}.npy",mask=f"mask{tag}.npy",
            map_sha256=outputs[f"phase{tag}.npy"],mask_sha256=outputs[f"mask{tag}.npy"],
            tile_origin_yx_px=[base+offset for base,offset in zip(PROTOCOL["source_global_grid_origin_yx"],PROTOCOL["phases_yx"][p])],
            tile_phase_offset_yx_px=PROTOCOL["phases_yx"][p],run_signature=recipe) for p,tag in enumerate(("0","10"))])
    write_json(args.output_dir/"pair.json",pair)
    report=dict(signature,status="completed",inference_elapsed_seconds=elapsed,phases=phase_reports,
        common_valid_pixels=int(common.sum()),field_pixels=int(common.size),source_CT_positive_pixels=int(ct.sum()),
        common_valid_fraction_of_field=float(common.mean()),common_valid_fraction_of_source_CT=float(common.sum()/ct.sum()),
        published_render_chunk_bytes_verified=True,native_CT_acquisition_coverage_certified=False,
        output_orientation="reading orientation: native crop flipped vertically",output_sha256=outputs,
        score_comparison=compare(maps[0],maps[1],common),
        ranking="Run row_context/diagnostic.py on pair.json with unchanged Config; do not tune thresholds or compare these member maps directly to the earlier fused product")
    write_json(args.output_dir/"report.json",report)
    print(json.dumps({"status":"completed","phases":phase_reports,"common_valid_pixels":int(common.sum()),"pair_manifest":"pair.json"},indent=2),flush=True)
    return 0


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",choices=("micro","field"),required=True)
    ap.add_argument("--inputs",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--maximum-seconds",type=float,default=900,help="Full-field inference wall budget per invocation; micro uses the frozen 180-second budget")
    ap.add_argument("--resume",action="store_true",help="Resume only the exact same full-field recipe/runtime")
    args=ap.parse_args()
    outside_git(args.inputs);outside_git(args.output_dir)
    return micro(args) if args.mode=="micro" else field(args)


if __name__=="__main__":
    raise SystemExit(main())
