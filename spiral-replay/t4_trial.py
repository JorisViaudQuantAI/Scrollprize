"""Prepare public M7 inputs and optionally time a bounded Villa FitContext on CUDA."""
import argparse
import dbm
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from arc_probe import ROOT, representative_arcs
from stage import META, PIN, git_blob
from umbilicus import json_umbilicus_z_to_yx


def prepare():
    from fit_session import conventional_input_paths, parse_scroll_spec

    plans=json.loads((ROOT/"sources.json").read_text())
    center=json_umbilicus_z_to_yx(ROOT/"umbilicus.json")
    z_list=[p["z"] for p in plans["planes"] if p["z"] in META["train_z"]]
    if not z_list:
        raise ValueError("no training slice is present")
    _, rows, _=representative_arcs(z_list,center)
    folder=ROOT/"fixture"/plans["profile"]
    folder.mkdir(parents=True,exist_ok=True)
    spec={"schema_version":1,"name":"PHerc0358","voxel_size_um":9.362,
          "spiral_outward_sense":"CW","umbilicus":{"coordinate_scale":1.0},
          "normal_zarr_group":"2","lasagna_scale":4}
    (folder/"spiral-scroll.json").write_text(json.dumps(spec,indent=2)+"\n")
    copied=folder/"umbilicus.json"
    copied.write_bytes((ROOT/"umbilicus.json").read_bytes())
    path=folder/"tracks.dbm"
    totals={}
    with dbm.open(str(path),"n") as store:
        for z in z_list:
            xyzw=np.memmap(ROOT/"data"/f"plane-{z}.i32","<i4","r").reshape(-1,3)
            offsets=np.load(ROOT/"data"/f"plane-{z}-offsets.npy",allow_pickle=False)
            selected=[np.asarray(xyzw[offsets[t]:offsets[t+1]]).copy() for t in rows[rows[:,0]==z,1]]
            store[f"h:{z}"]=pickle.dumps(selected,protocol=4)
            totals[z]={"tracks":len(selected),"vertices":sum(len(p) for p in selected)}
    with dbm.open(str(path),"r") as verified:
        for z in z_list:
            curves=pickle.loads(verified[f"h:{z}"])
            assert len(curves)==totals[z]["tracks"] and all(np.all(v[:,0]==z) for v in curves)
    # Verify that the unmodified CLI would supply a truthy outer_shell path even
    # here. The official fit loader tries to open it whenever tracks are active.
    actual_spec=parse_scroll_spec(spec,dataset_root=folder)
    default=conventional_input_paths(folder,actual_spec)
    assert default.outer_shell and not Path(default.outer_shell).exists()
    print(json.dumps({"profile":plans["profile"],"train":totals,"dbm_backend":dbm.whichdb(str(path)),
                      "empty_shell_detected":True,"default_cli_shell":str(Path(default.outer_shell).name)}))
    return folder


def trial(args,folder):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required by Villa FitContext; fixture preparation succeeded")
    code=args.villa.resolve()
    if not (code/"fit_spiral.py").is_file():
        raise SystemExit("--villa must be the pinned volume-cartographer/scripts/spiral directory")
    hashes=json.loads((ROOT/"upstream-blobs.json").read_text())
    for name in ["fit_spiral.py","fit_session.py","config.py","transforms.py","tracks.py","lasagna_data.py"]:
        if git_blob(code/name)!=hashes[name]:
            raise ValueError(f"unexpected upstream blob for {name}")
    sys.path.insert(0,str(code))
    from config import Config,FitConfig
    from fit_session import ScrollSpec,SpiralInputPaths
    from fit_spiral import FitContext

    torch.cuda.reset_peak_memory_stats()
    if dbm.whichdb(str(folder/"tracks.dbm")) is None:
        raise ValueError("fixture has no recognized DBM")
    spec=ScrollSpec(name="PHerc0358",voxel_size_um=9.362,spiral_outward_sense=args.sense,
                    umbilicus_coordinate_scale=1.0,normal_zarr_group="2",lasagna_scale=4)
    inputs={"dataset_root":str(folder),"umbilicus":str(folder/"umbilicus.json"),
            "tracks_dbm":str(folder/"tracks.dbm"),"outer_shell":""}
    if args.normal_root:
        nr=args.normal_root.resolve()
        for field, name in [("normal_x","nx"),("normal_y","ny"),("gradient_magnitude","grad_mag")]:
            inputs[field]=str(nr/f"PHerc0358_{name}.ome.zarr")
            if not (Path(inputs[field])/"2"/".zarray").is_file():
                raise ValueError(f"{field} local level 2 has not been staged")
    config=Config().as_dict()
    overrides={
        "z_begin":9200,"z_end":10700,"optimizer_num_training_steps":args.steps,
        "optimizer_random_seed":358,"model_flow_bounds_radius":4096,"model_flow_voxel_resolution":192,
        "model_linear_z_resolution":192,"model_gap_expander_logit_resolution":192,
        "model_gap_expander_num_windings":260,"model_num_flow_integration_steps":1,
        "model_flow_field_direct_lr":False,"input_disable_patches":True,"loss_weight_shell_outer":0,
        "loss_weight_shell_patch_radius":0,"dense_spacing_mode":"grad_mag",
        "loss_weight_dense_normals":float(100 if args.normal_root else 0),
        "loss_weight_dense_spacing":float(12 if args.normal_root else 0),
        "loss_weight_dense_spacing_density":0,"loss_weight_min_spacing":0,
        "sample_count_dense_normal_points":128,"sample_count_dense_spacing_pairs":64,
        "sample_count_tracks_per_step":32,"sample_count_track_points_per_step":24,
        "sample_count_regularisation_points":32,"track_crossing_mode":"count",
        "track_crossing_precompute_max":0,"track_max_track_crossing_per_step":0,
        "track_length_bin_weights":[1.,1.,1.],"track_exclusion_radius":0.0}
    if overrides.keys()-config.keys():
        raise ValueError("configuration differs from the pinned commit")
    config.update(overrides)
    out=(ROOT/"t4-output").resolve()
    out.mkdir(exist_ok=True)
    ctx=FitContext(FitConfig(config),scroll=spec,paths=SpiralInputPaths(**inputs),
                   out_base_dir=str(out),cache_dir=str(out/"cache"),run_tag=f"358smoke-{args.sense}")
    start=time.monotonic()
    try:
        ctx.load_host_inputs()
        ctx.resolve_output_path()
        ctx.build_device_state()
        losses=[]
        for step in range(args.steps):
            loss,_,_,_=ctx.step(step)
            value=float(loss.detach().cpu()) if isinstance(loss,torch.Tensor) else float(loss)
            if not np.isfinite(value):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            losses.append(value)
        ctx._save_model("smoke",args.steps)
        torch.cuda.synchronize()
        record={"villa_sha":PIN,"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
                "cuda_version":torch.version.cuda,"sense_hypothesis":args.sense,"steps":args.steps,
                "actual_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                "actual_peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                "elapsed_s":time.monotonic()-start,"loss_history":losses,"normals_used":bool(args.normal_root)}
        (out/f"measure-{args.sense}.json").write_text(json.dumps(record,indent=2)+"\n")
        print(json.dumps(record))
    finally:
        ctx.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-cuda",action="store_true")
    p.add_argument("--villa",type=Path,help="pinned volume-cartographer/scripts/spiral")
    p.add_argument("--steps",type=int,default=3)
    p.add_argument("--sense",choices=["CW","ACW"],default="CW")
    p.add_argument("--normal-root",type=Path,help="optional staged Lasagna folder with resident sidecars")
    args=p.parse_args()
    if args.run_cuda and (args.villa is None or not 1<=args.steps<=100):
        p.error("the bounded CUDA trial requires --villa and 1–100 steps")
    os.environ.setdefault("FIT_SPIRAL_TRITON","0")
    os.environ.setdefault("FIT_SPIRAL_COMPILE","0")
    os.environ.setdefault("WANDB_MODE","disabled")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    folder=prepare()
    if args.run_cuda:
        trial(args,folder)


if __name__=="__main__":
    main()
