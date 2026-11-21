#!/usr/bin/env python3
"""Bounded real Hecate component-export regression on an already public CT crop.

Use --prepare-only first; execute two inference subprocesses only with --run.
All generated data and raw receipts must remain outside Git. This harness does
not change inference code, monkeypatch a predictor, or run a review consumer.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import shutil

import numpy as np
from export import outside_git, sha256_file

HERE=Path(__file__).resolve().parent
COLAB=HERE.parent
REPO=COLAB.parent
PROTOCOL_PATH=HERE/"public_smoke_protocol.json"
PROTOCOL=json.loads(PROTOCOL_PATH.read_text())
PRODUCER=COLAB/"ink/infer_ink.py"
HECATE=COLAB/"ink/third_party/hecate/hecate.py"
EXPECTED={PRODUCER:PROTOCOL["new_source_sha256"],HECATE:PROTOCOL["hecate_source_sha256"],
          HERE/"export.py":PROTOCOL["exporter_sha256"]}


def json_write(path,value):
    with Path(path).open("x") as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")


def verify_source():
    for path,want in EXPECTED.items():
        if sha256_file(path)!=want:raise ValueError("Source changed after protocol freeze: "+path.name)


def prepare(root,smoke):
    verify_source()
    if sha256_file(smoke)!=PROTOCOL["source_smoke_sha256"]:
        raise ValueError("Unexpected public smoke crop")
    folder=root/"input";folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/PROTOCOL["checkpoint_filename"]
    if not checkpoint.exists():
        url="https://huggingface.co/scrollprize/hecate/resolve/"+PROTOCOL["hecate_revision"]+"/"+checkpoint.name
        partial=checkpoint.with_suffix(".part")
        request=urllib.request.Request(url,headers={"User-Agent":"public-scrollprize-export-smoke"})
        started=time.perf_counter()
        with urllib.request.urlopen(request,timeout=60) as stream,partial.open("wb") as out:
            shutil.copyfileobj(stream,out,1024*1024)
        if partial.stat().st_size!=PROTOCOL["checkpoint_bytes"] or sha256_file(partial)!=PROTOCOL["checkpoint_sha256"]:
            raise ValueError("Downloaded checkpoint differs from published hash/length")
        partial.replace(checkpoint)
        print(json.dumps({"checkpoint_download_seconds":time.perf_counter()-started,"checkpoint_bytes":checkpoint.stat().st_size}),flush=True)
    if checkpoint.stat().st_size!=PROTOCOL["checkpoint_bytes"] or sha256_file(checkpoint)!=PROTOCOL["checkpoint_sha256"]:
        raise ValueError("Existing checkpoint checksum/length mismatch")
    with np.load(smoke,allow_pickle=False) as data:st=data["st"]
    if st.shape!=(169,169,24) or st.dtype!=np.uint8 or st.max()>200:
        raise ValueError("Unexpected published source stack")
    cropped=np.ascontiguousarray(st[:71,:71,:].transpose(2,0,1))
    prepared=folder/"published_smoke_71_zyx.npy"
    if prepared.exists():
        if not np.array_equal(np.load(prepared,allow_pickle=False),cropped):raise ValueError("Prepared CT changed")
    else:np.save(prepared,cropped,allow_pickle=False)
    baseline=folder/"legacy_infer_ink.py"
    blob=subprocess.check_output(["git","show",PROTOCOL["legacy_git_blob"]],cwd=REPO)
    if hashlib.sha256(blob).hexdigest()!=PROTOCOL["legacy_source_sha256"]:raise ValueError("Unexpected legacy source")
    if baseline.exists():
        if baseline.read_bytes()!=blob:raise ValueError("Legacy source changed")
    else:baseline.write_bytes(blob)
    provenance=folder/"provenance.json"
    declared=dict(mesh_sha256=None,normal_sha256=None,normal_convention=None,plane_order=PROTOCOL["source_plane_order"],
                  frame=None,source_content_sha256=sha256_file(prepared))
    if provenance.exists():
        if json.loads(provenance.read_text())!=declared:raise ValueError("Provenance changed")
    else:json_write(provenance,declared)
    return dict(input_sha256=sha256_file(prepared),checkpoint_sha256=sha256_file(checkpoint),
        legacy_source_sha256=sha256_file(baseline),provenance_sha256=sha256_file(provenance),
        input_shape_zyx=list(cropped.shape),source_preclipped_at_200=True)


def command(root,kind):
    folder=root/"input"
    source=folder/"legacy_infer_ink.py" if kind=="legacy" else PRODUCER
    args=[sys.executable,str(source),str(folder/"published_smoke_71_zyx.npy"),"-c",str(folder/PROTOCOL["checkpoint_filename"]),
          "--hecate-py",str(HECATE),"-o",str(root/kind/"ink.png"),*PROTOCOL["shared_cli_options"]]
    if kind=="new":args += ["--export-components",str(root/"new/components"),"--component-provenance",str(folder/"provenance.json")]
    return args


def run_cli(root,kind):
    folder=root/kind;folder.mkdir(exist_ok=False)
    env=os.environ.copy();env.update(OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
    t=time.perf_counter()
    with (folder/"stdout.log").open("wb") as out,(folder/"stderr.log").open("wb") as err:
        process=subprocess.Popen(command(root,kind),stdout=out,stderr=err,env=env,start_new_session=True,cwd=REPO)
        try:code=process.wait(timeout=PROTOCOL["subprocess_wall_timeout_seconds"])
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
            return dict(status="timeout",elapsed_subprocess_seconds=time.perf_counter()-t,returncode=process.returncode)
    return dict(status="completed" if code==0 else "failed",returncode=code,elapsed_subprocess_seconds=time.perf_counter()-t)


def canonical_metadata(value,root,kind):
    omitted=set(PROTOCOL["excluded_metadata_runtime_fields"])
    if isinstance(value,dict):return {k:canonical_metadata(v,root,kind) for k,v in value.items() if k not in omitted}
    if isinstance(value,list):return [canonical_metadata(v,root,kind) for v in value]
    if isinstance(value,str):return value.replace(str(root/kind),"{RUN}").replace(str(root),"{WORK}")
    return value


def disk_inventory(folder):
    return {str(p.relative_to(folder)):dict(bytes=p.stat().st_size,sha256=sha256_file(p))
            for p in sorted(folder.rglob("*")) if p.is_file()}


def check_outcomes(root):
    old=json.loads((root/"legacy/ink.json").read_text());new=json.loads((root/"new/ink.json").read_text())
    for meta in (old,new):
        model=meta["models"][0]
        if meta["source_shape_zyx"]!=[24,71,71] or model["patch_zyx"]!=[16,64,64] or model["tiles_run"]!=6 or model["passes"]!=6 or meta["tta"]:
            raise ValueError("Actual inference did not match frozen geometry or call count")
    old_files={Path(p).name:Path(p) for group in old["outputs"].values() for p in group}
    new_files={Path(p).name:Path(p) for group in new["outputs"].values() for p in group}
    if old_files.keys()!=new_files.keys():raise ValueError("Legacy output inventory changed")
    pairs={name:dict(legacy_sha256=sha256_file(old_files[name]),new_sha256=sha256_file(new_files[name]),
                    bytes=old_files[name].stat().st_size,byte_equal=old_files[name].read_bytes()==new_files[name].read_bytes()) for name in sorted(old_files)}
    normalized_old=canonical_metadata(old,root,"legacy");normalized_new=canonical_metadata(new,root,"new")
    metadata_equal=normalized_old==normalized_new
    packet=root/"new/components"
    receipt=json.loads((packet/"receipt.json").read_text())
    if receipt["status"]!="complete" or len(receipt["components"])!=6:raise ValueError("Incomplete six-component receipt")
    components=[];direction_arrays={d:[] for d in ("forward","reverse")}
    for index in receipt["components"]:
        folder=packet/index["directory"];metadata_path=folder/"component.json"
        if sha256_file(metadata_path)!=index["component_json_sha256"]:raise ValueError("Component metadata hash mismatch")
        meta=json.loads(metadata_path.read_text())
        arrays={}
        for name,want in meta["files_sha256"].items():
            if sha256_file(folder/name)!=want:raise ValueError("Component array hash mismatch")
            arrays[name]=np.load(folder/name,allow_pickle=False)
        scores=arrays["probability.npy"];support=arrays["prediction_support.npy"];depth=arrays["complete_depth_support.npy"]
        if scores.dtype!=np.float32 or support.dtype!=np.bool_ or depth.dtype!=np.bool_ or scores.shape!=(71,71) or support.shape!=scores.shape or depth.shape!=scores.shape:
            raise ValueError("Component output grid/dtype mismatch")
        if not np.isfinite(scores[support]).all() or not np.isnan(scores[~support]).all():raise ValueError("Unsupported output encoding mismatch")
        if np.any(depth&~support):raise ValueError("Depth support exceeds prediction support")
        geo=meta["declared_geometry"]
        if any(geo[k] is not None for k in ("mesh_sha256","normal_sha256","normal_convention","frame")):
            raise ValueError("Unknown geometry was fabricated")
        sampling=meta["sampling"]
        if sampling["model_grid_shape_yx"]!=[64,64] or sampling["model_roi_y0_x0_y1_x1"]!=[0,0,64,64] or not sampling["resampled"]:
            raise ValueError("The expected one-tile resampled model grid was not used")
        direction_arrays[meta["direction"]].append((meta["shift_model_planes"],scores,support))
        components.append(dict(directory=index["directory"],direction=meta["direction"],shift_model_planes=meta["shift_model_planes"],
            shape_yx=list(scores.shape),score_dtype=str(scores.dtype),full_depth_context=meta["full_depth_context"],
            prediction_support_pixels=int(support.sum()),complete_depth_support_pixels=int(depth.sum()),
            sampling=sampling,declared_geometry=geo,acquisition_coverage_status=meta["acquisition_coverage_status"],
            component_json_sha256=index["component_json_sha256"],files_sha256=meta["files_sha256"]))
    observed={(v["direction"],v["shift_model_planes"]) for v in components}
    if observed!={(d,float(s)) for d in ("forward","reverse") for s in (-1,0,1)}:raise ValueError("Component identity mismatch")
    reconstructed={}
    for direction,items in direction_arrays.items():
        items.sort(key=lambda item:item[0])
        valid=np.logical_and.reduce([item[2] for item in items])
        derived=np.stack([item[1] for item in items]).astype(np.float32).mean(0).astype(np.float16)
        saved=np.load(root/"new"/f"ink_{direction}_prob.npy",allow_pickle=False)
        reconstructed[direction]=dict(compared_prediction_supported_pixels=int(valid.sum()),
            output_pixels=int(valid.size),float16_aggregate_equal_on_supported_pixels=bool(np.array_equal(derived[valid],saved[valid])),
            interpretation="Arithmetic export invariant only; shared prediction support is not acquisition validity")
    inventories={kind:disk_inventory(root/kind) for kind in ("legacy","new")}
    return dict(legacy_output_comparisons=pairs,all_legacy_output_bytes_equal=all(p["byte_equal"] for p in pairs.values()),
        nonruntime_legacy_metadata_equal=metadata_equal,normalized_legacy_metadata=normalized_old,
        normalized_new_metadata=normalized_new,excluded_runtime_metadata_fields=PROTOCOL["excluded_metadata_runtime_fields"],
        components=components,component_count=6,real_forward_calls_per_run=[6,6],
        component_mean_reconstruction=reconstructed,
        forward_count_basis="Pinned producer reports six tiles, batch size one, TTA disabled; each tile invokes the real Hecate model once",
        disk_inventory=inventories,actual_output_bytes={k:sum(v["bytes"] for v in inv.values()) for k,inv in inventories.items()},
        component_export_bytes=sum(p.stat().st_size for p in packet.rglob("*") if p.is_file()),
        source_preclipped_at_200=True,quality_evaluation=False,consumer_run=False,
        consumer_blocker="No verified mesh, normal, anatomical frame, or external acquisition/context-validity mask")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir",type=Path,required=True)
    ap.add_argument("--smoke",type=Path,required=True)
    mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument("--prepare-only",action="store_true");mode.add_argument("--run",action="store_true")
    args=ap.parse_args();root=outside_git(args.work_dir);outside_git(args.smoke)
    root.mkdir(parents=True,exist_ok=True)
    if (root/"run_started.json").exists():raise FileExistsError("This frozen two-run attempt has already started; no automatic retry")
    prepared=prepare(root,args.smoke)
    if args.prepare_only:
        print(json.dumps(dict(status="prepared_no_inference",**prepared),indent=2));return 0
    import torch,cv2
    started=dict(started_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha256_file(PROTOCOL_PATH),
        harness_sha256=sha256_file(__file__),source_sha256={str(p.relative_to(REPO)):sha256_file(p) for p in EXPECTED},
        inputs=prepared,commands={k:command(root,k) for k in ("legacy","new")},
        environment={"python":sys.version.split()[0],"numpy":np.__version__,"torch":torch.__version__,"opencv":cv2.__version__,"OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4"})
    json_write(root/"run_started.json",started)
    runs={}
    for kind in ("legacy","new"):
        verify_source();runs[kind]=run_cli(root,kind)
        print(json.dumps({"run":kind,**runs[kind]}),flush=True)
        if runs[kind]["status"]!="completed":
            json_write(root/"result.json",dict(status="blocked_incomplete",runs=runs,protocol_sha256=started["protocol_sha256"],no_extra_attempts=True));return 3
    verify_source();outcome=check_outcomes(root)
    # Public result omits real local paths; private raw receipts remain intact.
    public_started=dict(started,commands={k:[s.replace(str(root),"{WORK}").replace(str(REPO),"{REPO}").replace(sys.executable,"{PYTHON}") for s in cmd] for k,cmd in started["commands"].items()})
    outcome.update(status="completed",runs=runs,preregistration=public_started)
    json_write(root/"result.json",outcome)
    print(json.dumps({"status":"completed","components":outcome["component_count"],"legacy_bytes_equal":outcome["all_legacy_output_bytes_equal"],"metadata_equal":outcome["nonruntime_legacy_metadata_equal"],"component_export_bytes":outcome["component_export_bytes"]}),flush=True)
    return 0 if outcome["all_legacy_output_bytes_equal"] and outcome["nonruntime_legacy_metadata_equal"] else 2


if __name__=="__main__":raise SystemExit(main())
