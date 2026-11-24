#!/usr/bin/env python3
"""Q33 follow-up: factor anchor placement and sampling-ray direction.

Eight fixed 2x2 blocks; three fresh identity sentinels gate 272 new forwards.
Historical identity/rigid predictions are reused only after native-input hash
reproduction. Four cells share a physical mask within each block. This is a
mechanistic follow-up on a previously inspected donor control, not transfer.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = HERE.parent / "studio_oblique_stress"
sys.path.insert(0,str(LEGACY))
spec = importlib.util.spec_from_file_location("q33_parent_runner", LEGACY/"run.py")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
from assets import acquire_geometry
from native import load_surface, SupportedNative
from factor_geometry import build_patch, inverse_readback

sha256,write_json,metrics = parent.sha256,parent.write_json,parent.metrics
PARENT_PROTOCOL_SHA = "d56864f88ba32302344c1858f9a53cf4c1f203b64a7b28fab980105535793a80"
PARENT_ARCHIVE_SHA = "9558cfd463a3d3450e0b7499321df8fa49b64fa83286c984666da2dfac5fe6ad"
FAMILIES = ["identity","normal_only","anchor_only","rigid"]
BLOCKS = [dict(name=f"{axis}_{'minus' if angle<0 else 'plus'}{abs(angle)}",axis=axis,angle_deg=angle)
          for axis in ("x","y") for angle in (-20,-5,5,20)]


def code_hashes():
    result = parent.code_hashes()
    for name in ("run.py","factor_geometry.py","test_factor_geometry.py"):
        path = HERE/name
        result[str(path.relative_to(ROOT))] = sha256(path)
    return result


def load_parent():
    pp = LEGACY/"evidence/protocol.json"
    archive = LEGACY/"evidence/scored_arrays.npz"
    result = LEGACY/"evidence/results.json"
    receipt = LEGACY/"evidence/inference_receipt.json"
    if sha256(pp)!=PARENT_PROTOCOL_SHA or sha256(archive)!=PARENT_ARCHIVE_SHA:
        raise ValueError("Parent public evidence changed")
    protocol = json.loads(pp.read_text())
    if protocol["code_hashes"] != parent.code_hashes():
        raise ValueError("Parent code changed")
    if json.loads(result.read_text())["scored_archive_sha256"] != PARENT_ARCHIVE_SHA:
        raise ValueError("Parent result/archive mismatch")
    with np.load(archive,allow_pickle=False) as z:
        arrays = {k:z[k].copy() for k in z.files}
    expected = [r["index"] for r in protocol["controls"]]
    if arrays["author_indices"].tolist()!=expected:
        raise ValueError("Parent author indices differ")
    return protocol,arrays,dict(protocol_sha256=PARENT_PROTOCOL_SHA,archive_sha256=PARENT_ARCHIVE_SHA,
        results_sha256=sha256(result),inference_receipt_sha256=sha256(receipt),
        source_commit="bfc9372ded39809fc79388a6017ef768e91bc17a",
        reuse="17 identity and136 rigid aligned predictions; original archive contains unmasked32x32 fields",
        full_parent_audit="For this execution, independently recompute reused values on the new masks from hash-verified parent full128 outputs and mapping, including pixels excluded by the old global mask")


def readback(p,q):
    return map_coordinates(p,np.asarray(q).transpose(2,0,1),order=1,mode="constant",cval=np.nan,prefilter=False)


def prepare(args):
    if args.prepared.exists():
        raise ValueError("Use a fresh prepared directory")
    started = time.monotonic()
    pp,old,parent_receipt = load_parent()
    source = parent.verify_source(args.source_repo)
    for name,(digest,size) in parent.FILES.items():
        parent.verified_file(parent.BASE+name,args.author_cache/name,digest,size)
    parent.verified_file(parent.WEIGHTS_URL,args.checkpoint,parent.WEIGHTS_SHA256,893443724)
    labels = np.load(args.author_cache/"labels.npy",allow_pickle=False)
    controls = pp["controls"]
    original = np.stack([labels[c["y"]+48:c["y"]+80,c["x"]+48:c["x"]+80] for c in controls])
    if not np.array_equal(original,old["original_states"]):
        raise ValueError("Original parent labels differ")
    n = len(controls)
    if n!=17 or sum(c["category"]=="mixed" for c in controls)!=11:
        raise ValueError("Parent panel changed")
    mesh,recipe,acquisition = acquire_geometry(args.geometry_cache)
    surface = load_surface(args.source_repo,mesh,recipe)
    query=np.empty((8,4,n,32,32,2),np.float64)
    depth=np.empty((8,4,n,32,32),np.float64)
    valid=np.empty((8,4,n,32,32),bool)
    residual=np.empty((8,4,n,32,32),np.float64)
    diagnostics=[]
    for bi,block in enumerate(BLOCKS):
        for fi,family in enumerate(FAMILIES):
            for pi,c in enumerate(controls):
                patch=build_patch(surface,c["y"],c["x"],block["angle_deg"],block["axis"],family,
                    pitch_um=recipe["xy_pitch_um"],offsets_um=recipe["offsets_um"])
                if not patch["mesh_valid"].all():
                    raise ValueError("Patch lacks full mesh support")
                inv=inverse_readback(surface,c["y"],c["x"],patch,pitch_um=recipe["xy_pitch_um"])
                query[bi,fi,pi]=inv["query_yx"];depth[bi,fi,pi]=inv["depth_vox"]
                valid[bi,fi,pi]=inv["valid"];residual[bi,fi,pi]=inv["residual_vox"]
                diagnostics.append(dict(inv["diagnostics"],block=block["name"],family=family,author_index=c["index"]))
    common=valid.all(axis=1)
    states=np.where(common,original[None],0).astype(np.uint8)
    known=original>0
    retention=(states>0).sum(axis=(2,3))/known.sum(axis=(1,2))[None]
    overall=(states>0).sum(axis=(1,2,3))/known.sum()
    args.prepared.mkdir(parents=True)
    geo=dict(blocks=BLOCKS,per_block_known_retention=overall.tolist(),per_block_core_known_retention=retention.tolist(),
        original_ink=int((original==2).sum()),original_background=int((original==1).sum()),
        common_ink=(states==2).sum(axis=(1,2,3)).tolist(),common_background=(states==1).sum(axis=(1,2,3)).tolist(),
        per_cell_core=diagnostics,minimum_global_retention_per_block=.95,minimum_core_retention_per_block=.90)
    write_json(args.prepared/"geometry_diagnostics.json",geo)
    if (overall<.95).any() or (retention<.90).any():
        write_json(args.prepared/"failed_geometry_gate.json",dict(model_forwards=0,failed=True,**geo))
        raise ValueError("Per-block geometry gate failed; no model load or core replacement")
    global_common=common.all(axis=0)
    old_names=old["condition_names"].tolist()
    parent_aligned=np.stack([np.stack([old["probabilities"][old_names.index("identity")],
                                      old["probabilities"][old_names.index(b["name"])]]) for b in BLOCKS])
    for bi in range(8):
        if not np.isfinite(parent_aligned[bi,:,common[bi]]).all():
            raise ValueError("Nonfinite parent prediction on new common support")
    np.save(args.prepared/"original_states.npy",original,allow_pickle=False)
    np.save(args.prepared/"states.npy",states,allow_pickle=False)
    np.save(args.prepared/"parent_aligned.npy",parent_aligned,allow_pickle=False)
    np.savez_compressed(args.prepared/"mapping.npz",query_yx=query,depth_vox=depth,valid=valid,
        residual_vox=residual,common_masks=common,global_common=global_common)
    sentinel_ids=[min((i for i,c in enumerate(controls) if c["category"]==category),key=lambda i:controls[i]["index"])
                  for category in ("background","ink","mixed")]
    if [controls[i]["index"] for i in sentinel_ids]!=[250,288,326]:
        raise ValueError("Sentinel selection changed")
    sentinel=np.empty((3,16,128,128),np.uint8)
    np.save(args.prepared/"sentinel_reference.npy",old["probabilities"][old_names.index("identity"),sentinel_ids],allow_pickle=False)
    inputs=np.lib.format.open_memmap(args.prepared/"inputs.npy",mode="w+",dtype=np.uint8,shape=(16,n,16,128,128))
    volume=SupportedNative(recipe["url"],args.native_cache)
    old_hash={(r["arm"],r["author_index"]):r["sha256"] for r in pp["input_hashes"]}
    hashes,reuse_checks=[],[]
    for fi,family in enumerate(FAMILIES):
        for bi,block in enumerate(BLOCKS):
            if family=="identity" and bi>0:
                continue
            for pi,c in enumerate(controls):
                patch=build_patch(surface,c["y"],c["x"],block["angle_deg"],block["axis"],family,
                    pitch_um=recipe["xy_pitch_um"],offsets_um=recipe["offsets_um"])
                raw=np.rint(volume.sample_exact(patch["coords"])).clip(0,255).astype(np.uint8)
                digest=hashlib.sha256(raw.tobytes()).hexdigest()
                item=dict(block=block["name"],family=family,author_index=c["index"],sha256=digest)
                if family in ("identity","rigid"):
                    parent_arm="identity" if family=="identity" else block["name"]
                    if digest!=old_hash[(parent_arm,c["index"])]:
                        raise ValueError("Historical input bytes changed: "+str(item))
                    reuse_checks.append(dict(**item,parent_arm=parent_arm,exact_hash_match=True))
                    if family=="identity" and pi in sentinel_ids:
                        sentinel[sentinel_ids.index(pi)]=raw
                else:
                    inputs[bi*2+fi-1,pi]=raw
                    hashes.append(item)
            inputs.flush()
            print("PREPARED",family,block["name"],"seconds",round(time.monotonic()-started,2),flush=True)
    np.save(args.prepared/"sentinel_inputs.npy",sentinel,allow_pickle=False)
    write_json(args.prepared/"parent_reuse_input_checks.json",dict(exact=True,checked_inputs=len(reuse_checks),
        compared_bytes=len(reuse_checks)*16*128*128,records=reuse_checks))
    write_json(args.prepared/"native_receipt.json",volume.receipt())
    write_json(args.prepared/"geometry_acquisition.json",acquisition)
    arrays=["inputs.npy","sentinel_inputs.npy","sentinel_reference.npy","original_states.npy","states.npy","parent_aligned.npy","mapping.npz"]
    protocol=dict(version="q33-anchor-ray-factorial-v1",base_commit="bfc9372ded39809fc79388a6017ef768e91bc17a",
        purpose="Mechanistic2x2 ablation of reference-anchor placement and sampling-ray direction",
        scope="Follow-up informed by Q33 results on the same previously inspected author donor control; no independent-text or transfer claim",
        source_commit=parent.SOURCE_COMMIT,source_files=source,code_hashes=code_hashes(),parent=parent_receipt,
        weights=dict(url=parent.WEIGHTS_URL,sha256=parent.WEIGHTS_SHA256),controls=controls,blocks=BLOCKS,families=FAMILIES,
        numerical_geometry="Let N32=FP32(S32+d32*n32). Hybrids use C+R^a(S32-C)+R^b(N32-S32), rounded once toFP32; identity/rigid delegate exactly to parent. Pivot/axis/R shared from reference geometry.",
        factors={"identity":[0,0],"normal_only":[0,1],"anchor_only":[1,0],"rigid":[1,1]},
        family_semantics="normal_only means ray direction only; hybrid rays are generally not normals of their anchor surfaces",
        inverse="Reference labels attach to S(u,v), not known ink depth. Identity/ray-only use exact u,v andd0. Anchor-only solves its general tube and checks actual FP32 world residual. Rigid uses unchanged parent inverse.",
        mask_policy="Primary fixed common mask across four cells within each axis-angle block, formed before new inference. Denominators vary between blocks; report within-block contrasts, not cross-block AP as identical populations. Also report global intersection as a secondary descriptive population.",
        geometry_gate=dict(minimum_global_known_retention_each_block=.95,minimum_core_known_retention_each_block=.90,
            maximum_physical_residual_native_voxels=.01,depth_bounds_vox=[-8,7],model_margin_px=16,no_core_replacement=True),
        denominator_per_block=[dict(block=b["name"],ink=int((states[bi]==2).sum()),background=int((states[bi]==1).sum()),
            excluded_known=int(((original>0)&~common[bi]).sum())) for bi,b in enumerate(BLOCKS)],
        secondary_global_known_retention=float(((original>0)&global_common).sum()/known.sum()),
        primary=dict(population="Eleven mixed cores",metric="Recall at frozen probability0.5",material_loss_below=-.10),
        secondary=dict(metrics=["Mixed pooled AP","Mixed FPR@.5","All-label pooled AP/AUC/recall/FPR","Pure-ink recall","Pure-background FPR","Per-core metrics"],
            material_AP_loss_below=-.05,material_FPR_increase_above=.03),
        contrasts={"ray_at_fixed_anchors":"M01-M00","ray_at_rotated_anchors":"M11-M10", "anchor_at_fixed_rays":"M10-M00",
            "anchor_at_rotated_rays":"M11-M01","interaction":"M11-M10-M01+M00"},
        interpretation="Descriptive paired contrasts of the sampling operator; no IID-pixel significance or percentage causal attribution. AP is nonadditive and subgroup-dependent.",
        sentinel_gate=dict(author_indices=[controls[i]["index"] for i in sentinel_ids],patch_indices=sentinel_ids,
            maximum_probability_difference=1e-7,action="Abort before all272 new factorial forwards if any sentinel differs"),
        inference=dict(new_factorial_forwards=272,fresh_identity_sentinels=3,total_executed_forwards=275,
            reused_unique_parent_predictions=153,normalization="FP32clip[0,200]/255 once; original depth order",model="Pinned CPU FP32 eval batch1"),
        runtime=dict(workers=args.workers,threads_per_worker=args.threads,interop_threads=1),
        arrays={n:dict(sha256=sha256(args.prepared/n),bytes=(args.prepared/n).stat().st_size) for n in arrays},
        input_hashes=hashes,receipts={n:sha256(args.prepared/n) for n in ["parent_reuse_input_checks.json","native_receipt.json","geometry_acquisition.json","geometry_diagnostics.json"]},
        caveats=["Reference-surface labels do not localize true3D ink", "Anchor changes also alter reference depth, context and parametrization",
                 "Hybrid cells need not be anatomically valid surfaces", "Rotating sampling is not a new CT acquisition or rotated scanner PSF",
                 "The local inverse does not prove global injectivity", "No angle correction or best condition is selected"],
        preparation_seconds=time.monotonic()-started)
    write_json(args.prepared/"protocol.json",protocol)
    print("PROTOCOL_FROZEN",sha256(args.prepared/"protocol.json"),"retention",overall.tolist(),flush=True)


_MODEL=_INPUTS=_SENTINELS=None


def init_worker(source_repo,checkpoint,prepared,threads):
    global _MODEL,_INPUTS,_SENTINELS
    import torch
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(Path(source_repo)/"studio-multiscroll-cpt-20261023/hf"))
    from inference import load_model
    _MODEL=load_model(Path(checkpoint),"cpu")
    _INPUTS=np.load(Path(prepared)/"inputs.npy",mmap_mode="r",allow_pickle=False)
    _SENTINELS=np.load(Path(prepared)/"sentinel_inputs.npy",allow_pickle=False)


def forward(task):
    import torch
    phase,ai,pi=task
    raw=_SENTINELS[pi] if phase=="sentinel" else _INPUTS[ai,pi]
    x=torch.from_numpy(np.array(raw,dtype=np.float32,copy=True)[None,None]).clamp_(0,200).div_(255)
    started=time.monotonic()
    with torch.inference_mode():
        p=_MODEL(x)[0,0].float().sigmoid().numpy()
    if p.shape!=(128,128) or not np.isfinite(p).all():
        raise ValueError("Invalid model output")
    return phase,ai,pi,p,time.monotonic()-started,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,os.getpid()


def checked_protocol(args):
    p=json.loads((args.prepared/"protocol.json").read_text())
    if p["code_hashes"]!=code_hashes():
        raise ValueError("Code changed after freeze")
    for n,meta in p["arrays"].items():
        if sha256(args.prepared/n)!=meta["sha256"]:
            raise ValueError("Prepared array changed: "+n)
    load_parent()
    return p


def infer(args):
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    p=checked_protocol(args)
    if parent.verify_source(args.source_repo)!=p["source_files"] or sha256(args.checkpoint)!=parent.WEIGHTS_SHA256:
        raise ValueError("Source/checkpoint changed")
    args.output.mkdir(parents=True)
    full=np.lib.format.open_memmap(args.output/"full_probabilities.npy",mode="w+",dtype=np.float32,shape=(16,17,128,128))
    full[:]=np.nan
    refs=np.load(args.prepared/"sentinel_reference.npy",allow_pickle=False)
    sentinel=np.empty((3,128,128),np.float32)
    started=time.monotonic();records=[];rt=p["runtime"]
    with concurrent.futures.ProcessPoolExecutor(max_workers=rt["workers"],mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,initargs=(str(args.source_repo),str(args.checkpoint),str(args.prepared),rt["threads_per_worker"])) as pool:
        for phase,tasks in [("sentinel",[("sentinel",-1,i) for i in range(3)]),
                            ("factorial",[("factorial",ai,pi) for pi in range(17) for ai in range(16)])]:
            for future in concurrent.futures.as_completed([pool.submit(forward,t) for t in tasks]):
                kind,ai,pi,prob,seconds,rss,pid=future.result()
                if kind=="sentinel":sentinel[pi]=prob
                else:full[ai,pi]=prob
                records.append(dict(phase=kind,input_arm=ai,patch=pi,seconds=seconds,peak_rss_kib=rss,pid=pid))
                if len(records)%10==0 or len(records) in (3,275):
                    full.flush();print("PROGRESS",len(records),"/275 seconds",round(time.monotonic()-started,2),flush=True)
            if phase=="sentinel":
                diff=np.abs(sentinel[:,48:80,48:80]-refs).max(axis=(1,2))
                passed=bool((diff<=p["sentinel_gate"]["maximum_probability_difference"]).all())
                np.save(args.output/"sentinel_probabilities.npy",sentinel,allow_pickle=False)
                check=dict(passed=passed,author_indices=p["sentinel_gate"]["author_indices"],maximum_probability_differences=diff.tolist())
                write_json(args.output/"sentinel_check.json",check);print("SENTINELS",json.dumps(check),flush=True)
                if not passed:raise ValueError("Fresh identity sentinel gate failed")
    full.flush()
    import torch,scipy
    receipt=dict(protocol_sha256=sha256(args.prepared/"protocol.json"),code_hashes=code_hashes(),
        parent_archive_sha256=PARENT_ARCHIVE_SHA,checkpoint_sha256=parent.WEIGHTS_SHA256,
        full_probabilities_sha256=sha256(args.output/"full_probabilities.npy"),sentinel_probabilities_sha256=sha256(args.output/"sentinel_probabilities.npy"),
        wall_seconds=time.monotonic()-started,summed_forward_seconds=sum(r["seconds"] for r in records),forwards=records,
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,runtime=rt)
    write_json(args.output/"inference_receipt.json",receipt)
    print("INFERENCE_COMPLETE",receipt["wall_seconds"],flush=True)
    summarize(args)


def grouped(prob,state,controls):
    groups={"all":np.ones(17,bool),"mixed":np.array([c["category"]=="mixed" for c in controls]),
            "pure_ink":np.array([c["category"]=="ink" for c in controls]),
            "pure_background":np.array([c["category"]=="background" for c in controls])}
    return {name:metrics(prob[sel],state[sel]) for name,sel in groups.items()}


def contrasts(cells):
    output={}
    pairs={"ray_at_fixed_anchors":(1,0),"ray_at_rotated_anchors":(3,2),
           "anchor_at_fixed_rays":(2,0),"anchor_at_rotated_rays":(3,1)}
    for group in cells[0]:
        output[group]={}
        for metric in ("ap","auc","recall_at_0_5","false_positive_rate_at_0_5"):
            vals=[c[group][metric] for c in cells]
            if any(v is None for v in vals):continue
            output[group][metric]={name:vals[a]-vals[b] for name,(a,b) in pairs.items()}
            output[group][metric]["interaction"]=vals[3]-vals[2]-vals[1]+vals[0]
    return output


def summarize(args):
    p=checked_protocol(args)
    receipt=json.loads((args.output/"inference_receipt.json").read_text())
    if receipt["protocol_sha256"]!=sha256(args.prepared/"protocol.json") or sha256(args.output/"full_probabilities.npy")!=receipt["full_probabilities_sha256"]:
        raise ValueError("Output/receipt mismatch")
    full=np.load(args.output/"full_probabilities.npy",mmap_mode="r",allow_pickle=False)
    if not np.isfinite(full).all():raise ValueError("Incomplete new predictions")
    original=np.load(args.prepared/"original_states.npy",allow_pickle=False)
    states=np.load(args.prepared/"states.npy",allow_pickle=False)
    old=np.load(args.prepared/"parent_aligned.npy",allow_pickle=False)
    with np.load(args.prepared/"mapping.npz",allow_pickle=False) as m:
        query=m["query_yx"];common=m["common_masks"];global_common=m["global_common"]
    aligned=np.empty((8,4,17,32,32),np.float32)
    rows,per_core,global_rows=[],[],[]
    global_state=np.where(global_common,original,0).astype(np.uint8)
    for bi,b in enumerate(BLOCKS):
        aligned[bi,0],aligned[bi,3]=old[bi,0],old[bi,1]
        for fi in (1,2):
            for pi in range(17):aligned[bi,fi,pi]=readback(full[bi*2+fi-1,pi],query[bi,fi,pi])
        if not np.isfinite(aligned[bi,:,states[bi]>0]).all():raise ValueError("Nonfinite scored probability")
        cells=[grouped(aligned[bi,fi],states[bi],p["controls"]) for fi in range(4)]
        delta=contrasts(cells)
        flags={}
        for fi in (1,2,3):
            base,cur=cells[0]["mixed"],cells[fi]["mixed"]
            flags[FAMILIES[fi]]=dict(mixed_recall_material_loss=bool(cur["recall_at_0_5"]-base["recall_at_0_5"]<-.10),
                mixed_AP_material_loss=bool(cur["ap"]-base["ap"]<-.05),mixed_FPR_material_increase=bool(cur["false_positive_rate_at_0_5"]-base["false_positive_rate_at_0_5"]>.03))
        rows.append(dict(**b,metrics=dict(zip(FAMILIES,cells)),contrasts=delta,engineering_flags=flags))
        global_cells=[grouped(aligned[bi,fi],global_state,p["controls"]) for fi in range(4)]
        global_rows.append(dict(**b,metrics=dict(zip(FAMILIES,global_cells)),contrasts=contrasts(global_cells)))
        per_core.append(dict(block=b["name"],cores=[dict(author_index=c["index"],category=c["category"],
            metrics={family:metrics(aligned[bi,fi,pi],states[bi,pi]) for fi,family in enumerate(FAMILIES)}) for pi,c in enumerate(p["controls"])]))
    np.savez_compressed(args.output/"scored_arrays.npz",probabilities=aligned,states=states,original_states=original,
        common_masks=common,global_common=global_common,block_names=np.array([b["name"] for b in BLOCKS]),
        families=np.array(FAMILIES),author_indices=np.array([c["index"] for c in p["controls"]],np.int64))
    result=dict(protocol_sha256=sha256(args.prepared/"protocol.json"),blocks=rows,per_core=per_core,
        secondary_global=dict(ink=int((global_state==2).sum()),background=int((global_state==1).sum()),
            known_retention=p["secondary_global_known_retention"],blocks=global_rows),
        scored_archive_sha256=sha256(args.output/"scored_arrays.npz"),scope=p["scope"],statistics=p["interpretation"])
    write_json(args.output/"results.json",result)
    print("RESULTS_COMPLETE",result["scored_archive_sha256"],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","infer","summarize"))
    for name in ("source-repo","checkpoint","author-cache","geometry-cache","native-cache","prepared","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=2);parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    if args.workers<1 or args.threads<1:parser.error("Positive worker/thread counts required")
    {"prepare":prepare,"infer":infer,"summarize":summarize}[args.command](args)


if __name__=="__main__":main()
