#!/usr/bin/env python3
"""Independent archive scoring and optional complete-array audit of Q33 factors.

No producer, sampler or geometry module is imported. Portable mode checks the
published archives, masks, groups and factorial contrasts. Full mode additionally
recomputes bilinear readback from both fresh and historical full128 outputs,
including newly scored parent pixels outside the old Q33 common mask.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

PARENT_PROTOCOL_SHA = "d56864f88ba32302344c1858f9a53cf4c1f203b64a7b28fab980105535793a80"
PARENT_ARCHIVE_SHA = "9558cfd463a3d3450e0b7499321df8fa49b64fa83286c984666da2dfac5fe6ad"
EXPECTED_IDS = [250,288,326,357,358,359,360,361,362,395,399,430,432,469,471,472,509]
FAMILIES = ["identity", "normal_only", "anchor_only", "rigid"]
BLOCKS = [(f"{axis}_{'minus' if angle<0 else 'plus'}{abs(angle)}",axis,angle)
          for axis in ("x","y") for angle in (-20,-5,5,20)]
METRICS = ("ap","auc","recall_at_0_5","false_positive_rate_at_0_5")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def metric(probability, states):
    known = states != 0
    truth, score = states[known] == 2, probability[known]
    assert np.isfinite(score).all() and np.all((score >= 0) & (score <= 1))
    pos, neg = int(truth.sum()), int((~truth).sum())
    selected = score >= .5
    tp, fp = int((selected & truth).sum()), int((selected & ~truth).sum())
    return dict(ink_pixels=pos,background_pixels=neg,unknown_pixels=int((~known).sum()),
        true_positive_at_0_5=tp,false_positive_at_0_5=fp,
        recall_at_0_5=tp/pos if pos else None,false_positive_rate_at_0_5=fp/neg if neg else None,
        ap=float(average_precision_score(truth,score)) if pos and neg else None,
        auc=float(roc_auc_score(truth,score)) if pos and neg else None)


def groups(original):
    ink = (original==2).sum(axis=(1,2))
    bg = (original==1).sum(axis=(1,2))
    known = ink+bg
    assert (known>=512).all()
    mixed = (10*ink>=known) & (10*ink<=9*known)
    pure_ink, pure_background = (bg==0), (ink==0)
    assert mixed.sum()==11 and pure_ink.sum()==3 and pure_background.sum()==3
    assert np.all(mixed.astype(int)+pure_ink.astype(int)+pure_background.astype(int)==1)
    return dict(all=np.ones(17,bool),mixed=mixed,pure_ink=pure_ink,pure_background=pure_background)


def grouped(probability, states, masks):
    return {group:metric(probability[mask],states[mask]) for group,mask in masks.items()}


def factorial(cells):
    result = {}
    for group in cells[0]:
        result[group] = {}
        for key in METRICS:
            v = [cell[group][key] for cell in cells]
            if any(x is None for x in v):
                continue
            result[group][key] = dict(
                ray_at_fixed_anchors=v[1]-v[0],ray_at_rotated_anchors=v[3]-v[2],
                anchor_at_fixed_rays=v[2]-v[0],anchor_at_rotated_rays=v[3]-v[1],
                interaction=(v[3]-v[2])-(v[1]-v[0]))
    return result


class Comparisons:
    def __init__(self):
        self.count = 0
        self.maximum_absolute_error = 0.0

    def compare(self, observed, expected, path="root"):
        if isinstance(expected, dict):
            assert set(observed)==set(expected), path
            for key,value in expected.items():
                self.compare(observed[key],value,path+"."+str(key))
        elif isinstance(expected, list):
            assert len(observed)==len(expected), path
            for i,value in enumerate(expected):
                self.compare(observed[i],value,path+f"[{i}]")
        elif isinstance(expected, float):
            assert observed is not None and np.isfinite(observed), path
            error = abs(observed-expected)
            self.maximum_absolute_error = max(self.maximum_absolute_error,error)
            assert error<=1e-12, (path, observed, expected)
            self.count += 1
        else:
            assert observed==expected, (path,observed,expected)
            self.count += 1


def bilinear(field, query_yx):
    """Four literal corner terms; does not call the producer or scipy sampler."""
    query = np.asarray(query_yx,np.float64)
    assert np.isfinite(query).all()
    lower = np.floor(query).astype(np.int64)
    fraction = query-lower
    upper = lower+(fraction>0)
    assert (lower>=0).all() and (upper<np.asarray(field.shape)).all()
    out = np.zeros(query.shape[:-1],np.float64)
    for ay,ax in ((0,0),(0,1),(1,0),(1,1)):
        iy = lower[...,0] if ay==0 else upper[...,0]
        ix = lower[...,1] if ax==0 else upper[...,1]
        wy = 1-fraction[...,0] if ay==0 else fraction[...,0]
        wx = 1-fraction[...,1] if ax==0 else fraction[...,1]
        out += wy*wx*field[iy,ix].astype(np.float64)
    return out.astype(np.float32)


def source_crop(tiles, tile_origins, y, x):
    out = np.zeros((128,128,16),np.uint8)
    coverage = np.zeros((128,128),np.uint8)
    for tile,(ty,tx) in zip(tiles,tile_origins):
        ty,tx = int(ty),int(tx)
        ya,yb,xa,xb = max(y,ty),min(y+128,ty+512),max(x,tx),min(x+128,tx+512)
        if ya>=yb or xa>=xb:
            continue
        out[ya-y:yb-y,xa-x:xb-x] = tile[ya-ty:yb-ty,xa-tx:xb-tx]
        coverage[ya-y:yb-y,xa-x:xb-x] += 1
    assert np.all(coverage==1)
    return out.transpose(2,0,1)


def complete_array_audit(args,p,pp,new,old,common,states,original):
    assert args.parent_prepared and args.parent_output, "Full mode requires both parent array directories"
    for name,meta in p["arrays"].items():
        path = args.prepared/name
        assert path.stat().st_size==meta["bytes"] and sha(path)==meta["sha256"],name
    for name,digest in p["receipts"].items():
        assert sha(args.prepared/name)==digest,name
    np.testing.assert_array_equal(np.load(args.prepared/"states.npy"),states)
    np.testing.assert_array_equal(np.load(args.prepared/"original_states.npy"),original)
    np.testing.assert_array_equal(np.load(args.prepared/"parent_aligned.npy"),new["probabilities"][:,[0,3]])
    root = Path(__file__).resolve().parents[2]
    for name,digest in p["code_hashes"].items():
        assert sha(root/name)==digest,name
    receipt = read_json(args.output/"inference_receipt.json")
    assert receipt["protocol_sha256"]==sha(args.protocol)
    assert receipt["code_hashes"]==p["code_hashes"]
    assert receipt["parent_archive_sha256"]==PARENT_ARCHIVE_SHA
    assert receipt["checkpoint_sha256"]==p["weights"]["sha256"]
    full_path = args.output/"full_probabilities.npy"
    sentinel_path = args.output/"sentinel_probabilities.npy"
    assert sha(full_path)==receipt["full_probabilities_sha256"]
    assert sha(sentinel_path)==receipt["sentinel_probabilities_sha256"]
    full = np.load(full_path,mmap_mode="r",allow_pickle=False)
    assert full.shape==(16,17,128,128) and full.dtype==np.float32 and np.isfinite(full).all()
    assert np.all((full>=0)&(full<=1))
    records = receipt["forwards"]
    assert len(records)==275
    assert all(r["phase"]=="sentinel" for r in records[:3])
    assert sorted(r["patch"] for r in records[:3])==[0,1,2]
    assert all(r["phase"]=="factorial" for r in records[3:])
    assert {(r["input_arm"],r["patch"]) for r in records[3:]}=={(a,i) for a in range(16) for i in range(17)}
    assert p["sentinel_gate"]["author_indices"]==[250,288,326]
    ids = [EXPECTED_IDS.index(i) for i in [250,288,326]]
    sentinels = np.load(sentinel_path,allow_pickle=False)
    assert sentinels.shape==(3,128,128) and sentinels.dtype==np.float32
    assert np.isfinite(sentinels).all() and np.all((sentinels>=0)&(sentinels<=1))
    reference = old["probabilities"][old["condition_names"].tolist().index("identity"),ids]
    np.testing.assert_array_equal(np.load(args.prepared/"sentinel_reference.npy"),reference)
    maximum_diffs = np.max(np.abs(sentinels[:,48:80,48:80]-reference),axis=(1,2))
    assert np.all(maximum_diffs<=1e-7)
    check = read_json(args.output/"sentinel_check.json")
    assert check["passed"] and check["author_indices"]==[250,288,326]
    np.testing.assert_array_equal(check["maximum_probability_differences"],maximum_diffs)
    with np.load(args.prepared/"mapping.npz",allow_pickle=False) as a:
        query,valid,depth,residual = (a[k] for k in ("query_yx","valid","depth_vox","residual_vox"))
        np.testing.assert_array_equal(a["common_masks"],common)
        np.testing.assert_array_equal(a["global_common"],common.all(axis=0))
    assert query.shape==(8,4,17,32,32,2) and valid.shape==(8,4,17,32,32)
    np.testing.assert_array_equal(common,valid.all(axis=1))
    assert np.isfinite(residual[valid]).all() and np.max(residual[valid])<=.01
    assert np.all((depth[valid]>=-8)&(depth[valid]<=7))
    assert np.all((query[valid]>=16)&(query[valid]<=111))
    core_grid = np.stack(np.meshgrid(np.arange(48,80),np.arange(48,80),indexing="ij"),axis=-1)
    for fi in (0,1):
        np.testing.assert_array_equal(query[:,fi],np.broadcast_to(core_grid,(8,17,32,32,2)))
        np.testing.assert_array_equal(depth[:,fi],np.zeros((8,17,32,32)))
    assert sha(args.parent_prepared/"protocol.json")==PARENT_PROTOCOL_SHA
    for name in ("mapping.npz","inputs.npy"):
        assert sha(args.parent_prepared/name)==pp["arrays"][name]["sha256"]
    old_receipt = read_json(args.parent_output/"inference_receipt.json")
    assert sha(args.parent_output/"inference_receipt.json")==p["parent"]["inference_receipt_sha256"]
    assert old_receipt["protocol_sha256"]==PARENT_PROTOCOL_SHA
    assert sha(args.parent_output/"full_probabilities.npy")==old_receipt["full_probabilities_sha256"]
    old_full = np.load(args.parent_output/"full_probabilities.npy",mmap_mode="r",allow_pickle=False)
    assert old_full.shape==(13,17,128,128) and old_full.dtype==np.float32 and np.isfinite(old_full).all()
    assert np.all((old_full>=0)&(old_full<=1))
    with np.load(args.parent_prepared/"mapping.npz",allow_pickle=False) as a:
        old_query,old_valid = a["query_yx"],a["valid"]
    old_names = old["condition_names"].tolist()
    max_error,max_parent_error = 0.0,0.0
    new_evaluated,parent_evaluated,additional_parent_evaluated = 0,0,0
    for bi,(name,_,_) in enumerate(BLOCKS):
        oi = old_names.index(name)
        np.testing.assert_array_equal(query[bi,3],old_query[oi])
        np.testing.assert_array_equal(valid[bi,3],old_valid[oi])
        for pi in range(17):
            scored = states[bi,pi]>0
            for fi in (1,2):
                actual = bilinear(full[2*bi+fi-1,pi],query[bi,fi,pi][scored])
                expected = new["probabilities"][bi,fi,pi][scored]
                max_error = max(max_error,float(np.max(np.abs(actual-expected))))
                np.testing.assert_array_equal(actual,expected)
                new_evaluated += len(actual)
            for fi,old_i in ((0,old_names.index("identity")),(3,oi)):
                actual = bilinear(old_full[old_i,pi],old_query[old_i,pi][scored])
                expected = new["probabilities"][bi,fi,pi][scored]
                max_parent_error = max(max_parent_error,float(np.max(np.abs(actual-expected))))
                np.testing.assert_array_equal(actual,expected)
                parent_evaluated += len(actual)
                additional_parent_evaluated += int((scored & ~old["common_mask"][pi]).sum())
    inputs = np.load(args.prepared/"inputs.npy",mmap_mode="r",allow_pickle=False)
    assert inputs.shape==(16,17,16,128,128) and inputs.dtype==np.uint8
    expected_hashes = {(r["block"],r["family"],r["author_index"]):r["sha256"] for r in p["input_hashes"]}
    assert len(expected_hashes)==272
    for bi,(name,_,_) in enumerate(BLOCKS):
        for fi,family in ((1,"normal_only"),(2,"anchor_only")):
            for pi,author_index in enumerate(EXPECTED_IDS):
                digest = hashlib.sha256(inputs[2*bi+fi-1,pi].tobytes()).hexdigest()
                assert digest==expected_hashes[(name,family,author_index)]
    old_inputs = np.load(args.parent_prepared/"inputs.npy",mmap_mode="r",allow_pickle=False)
    reuse = read_json(args.prepared/"parent_reuse_input_checks.json")
    assert reuse["exact"] and reuse["checked_inputs"]==153 and len(reuse["records"])==153
    assert reuse["compared_bytes"]==153*16*128*128
    old_input_names = [a["name"] for a in pp["arms"]]
    expected_old = {(r["arm"],r["author_index"]):r["sha256"] for r in pp["input_hashes"]}
    seen = set()
    for row in reuse["records"]:
        key = row["parent_arm"],row["author_index"]
        assert key not in seen and row["exact_hash_match"]
        assert row["family"]==("identity" if key[0]=="identity" else "rigid")
        assert row["block"] in [b[0] for b in BLOCKS]
        assert key[0]==("identity" if row["family"]=="identity" else row["block"])
        seen.add(key)
        raw = old_inputs[old_input_names.index(key[0]),EXPECTED_IDS.index(key[1])]
        assert row["sha256"]==expected_old[key]==hashlib.sha256(raw.tobytes()).hexdigest()
    assert seen=={(arm,index) for arm in ["identity"]+[b[0] for b in BLOCKS] for index in EXPECTED_IDS}
    source_review = None
    if args.author_cache:
        for name,meta in pp["acquisition"].items():
            path = args.author_cache/name
            assert path.stat().st_size==meta["bytes"] and sha(path)==meta["sha256"]
        labels = np.load(args.author_cache/"labels.npy",allow_pickle=False)
        tiles = np.load(args.author_cache/"ct-tiles.npy",mmap_mode="r",allow_pickle=False)
        origins = np.load(args.author_cache/"tiles.npy",allow_pickle=False)
        sentinel_inputs = np.load(args.prepared/"sentinel_inputs.npy",allow_pickle=False)
        for pi,c in enumerate(p["controls"]):
            raw = source_crop(tiles,origins,c["y"],c["x"])
            np.testing.assert_array_equal(labels[c["y"]+48:c["y"]+80,c["x"]+48:c["x"]+80],original[pi])
            np.testing.assert_array_equal(old_inputs[0,pi],raw)
            for bi in range(8):
                np.testing.assert_array_equal(inputs[2*bi,pi,8],raw[8])
            if pi in ids:
                np.testing.assert_array_equal(sentinel_inputs[ids.index(pi)],raw)
        source_review = dict(author_identity_bytes=17*16*128*128,
                            ray_only_zero_depth_planes_exact=8*17,sentinel_input_bytes=3*16*128*128)
    if args.checkpoint:
        assert sha(args.checkpoint)==p["weights"]["sha256"]
    return dict(new_output_maps=272,parent_unique_output_maps=153,new_forwards=275,
        new_scored_values_checked=new_evaluated,parent_scored_values_checked_with_block_repetition=parent_evaluated,
        newly_scored_parent_values_checked_outside_old_mask_with_block_repetition=additional_parent_evaluated,
        maximum_new_readback_error=max_error,maximum_parent_readback_error=max_parent_error,
        new_readback_bitwise_equal=True,parent_readback_bitwise_equal=True,
        maximum_sentinel_probability_differences=maximum_diffs.tolist(),
        reused_input_receipt_records_matching_materialized_parent_inputs=153,
        native_ct_acquisition_repeated_by_auditor=False,source_review=source_review,
        scope="Input receipts match materialized, hash-verified prior arrays; native CT reacquisition is not independently rerun. Full readback is checked on every new block's scored support, including newly admitted historical pixels.")


def audit(args):
    started = time.monotonic()
    protocol_hash = sha(args.protocol)
    if args.expected_protocol_sha:
        assert protocol_hash==args.expected_protocol_sha
    p = read_json(args.protocol)
    assert p["version"]=="q33-anchor-ray-factorial-v1"
    assert [(b["name"],b["axis"],b["angle_deg"]) for b in p["blocks"]]==BLOCKS
    assert p["families"]==FAMILIES and [c["index"] for c in p["controls"]]==EXPECTED_IDS
    assert p["factors"]==dict(identity=[0,0],normal_only=[0,1],anchor_only=[1,0],rigid=[1,1])
    gate = p["geometry_gate"]
    assert gate["minimum_global_known_retention_each_block"]==.95
    assert gate["minimum_core_known_retention_each_block"]==.90
    assert gate["maximum_physical_residual_native_voxels"]==.01
    assert p["primary"]["metric"]=="Recall at frozen probability0.5" and p["primary"]["material_loss_below"]==-.10
    assert p["secondary"]["material_AP_loss_below"]==-.05 and p["secondary"]["material_FPR_increase_above"]==.03
    parent_protocol_path = args.parent_evidence/"protocol.json"
    parent_archive_path = args.parent_evidence/"scored_arrays.npz"
    assert sha(parent_protocol_path)==PARENT_PROTOCOL_SHA==p["parent"]["protocol_sha256"]
    assert sha(parent_archive_path)==PARENT_ARCHIVE_SHA==p["parent"]["archive_sha256"]
    pp = read_json(parent_protocol_path)
    assert p["controls"]==pp["controls"]
    with np.load(parent_archive_path,allow_pickle=False) as z:
        old = {k:z[k].copy() for k in z.files}
    result = read_json(args.output/"results.json")
    archive_path = args.output/"scored_arrays.npz"
    assert result["protocol_sha256"]==protocol_hash and sha(archive_path)==result["scored_archive_sha256"]
    with np.load(archive_path,allow_pickle=False) as z:
        new = {k:z[k].copy() for k in z.files}
    assert set(new)=={"probabilities","states","original_states","common_masks","global_common","block_names","families","author_indices"}
    probability,states,original,common = [new[k] for k in ("probabilities","states","original_states","common_masks")]
    assert probability.shape==(8,4,17,32,32) and probability.dtype==np.float32
    assert states.shape==common.shape==(8,17,32,32) and original.shape==(17,32,32)
    assert common.dtype==bool and np.isin(original,[0,1,2]).all()
    assert new["block_names"].tolist()==[b[0] for b in BLOCKS] and new["families"].tolist()==FAMILIES
    assert new["author_indices"].tolist()==EXPECTED_IDS==old["author_indices"].tolist()
    np.testing.assert_array_equal(original,old["original_states"])
    np.testing.assert_array_equal(states,np.where(common,original[None],0))
    np.testing.assert_array_equal(new["global_common"],common.all(axis=0))
    known = (original>0).sum(axis=(1,2))
    kept = (states>0).sum(axis=(2,3))
    assert np.all(100*kept>=90*known[None])
    assert np.all(100*kept.sum(axis=1)>=95*known.sum())
    group_masks = groups(original)
    category = np.where(group_masks["mixed"],"mixed",np.where(group_masks["pure_ink"],"ink","background"))
    assert category.tolist()==[c["category"] for c in p["controls"]]
    old_names = old["condition_names"].tolist()
    for bi,(name,_,_) in enumerate(BLOCKS):
        for fi,old_name in ((0,"identity"),(3,name)):
            np.testing.assert_array_equal(probability[bi,fi],old["probabilities"][old_names.index(old_name)])
    check = Comparisons()
    check.compare(p["denominator_per_block"],[dict(block=b[0],ink=int((states[bi]==2).sum()),
        background=int((states[bi]==1).sum()),excluded_known=int(((original>0)&~common[bi]).sum())) for bi,b in enumerate(BLOCKS)])
    global_state = np.where(new["global_common"],original,0).astype(np.uint8)
    global_retention = float((global_state>0).sum()/known.sum())
    check.compare(p["secondary_global_known_retention"],global_retention)
    check.compare(result["secondary_global"]["ink"],int((global_state==2).sum()))
    check.compare(result["secondary_global"]["background"],int((global_state==1).sum()))
    check.compare(result["secondary_global"]["known_retention"],global_retention)
    summaries = []
    for bi,(name,axis,angle) in enumerate(BLOCKS):
        cells = [grouped(probability[bi,fi],states[bi],group_masks) for fi in range(4)]
        flags = {}
        for fi,family in enumerate(FAMILIES[1:],1):
            a,b = cells[fi]["mixed"],cells[0]["mixed"]
            flags[family] = dict(mixed_recall_material_loss=bool(a["recall_at_0_5"]-b["recall_at_0_5"]<-.10),
                mixed_AP_material_loss=bool(a["ap"]-b["ap"]<-.05),
                mixed_FPR_material_increase=bool(a["false_positive_rate_at_0_5"]-b["false_positive_rate_at_0_5"]>.03))
        expected = dict(name=name,axis=axis,angle_deg=angle,metrics=dict(zip(FAMILIES,cells)),
                        contrasts=factorial(cells),engineering_flags=flags)
        check.compare(result["blocks"][bi],expected,"blocks."+name)
        global_cells = [grouped(probability[bi,fi],global_state,group_masks) for fi in range(4)]
        check.compare(result["secondary_global"]["blocks"][bi],dict(name=name,axis=axis,angle_deg=angle,
            metrics=dict(zip(FAMILIES,global_cells)),contrasts=factorial(global_cells)),"global."+name)
        cores = [dict(author_index=author_index,category=str(category[pi]),
            metrics={family:metric(probability[bi,fi,pi],states[bi,pi]) for fi,family in enumerate(FAMILIES)})
            for pi,author_index in enumerate(EXPECTED_IDS)]
        check.compare(result["per_core"][bi],dict(block=name,cores=cores),"per_core."+name)
        summaries.append(dict(block=name,mixed={family:cells[fi]["mixed"] for fi,family in enumerate(FAMILIES)},
                              mixed_contrasts=factorial(cells)["mixed"]))
    full = complete_array_audit(args,p,pp,new,old,common,states,original) if args.prepared else None
    return dict(status="pass",protocol_sha256=protocol_hash,expected_protocol_sha256=args.expected_protocol_sha,
        parent_protocol_sha256=PARENT_PROTOCOL_SHA,parent_archive_sha256=PARENT_ARCHIVE_SHA,
        archive_sha256=sha(archive_path),archive_bytes=archive_path.stat().st_size,
        scalar_checks=check.count,maximum_metric_absolute_error=check.maximum_absolute_error,
        controls=EXPECTED_IDS,blocks=[b[0] for b in BLOCKS],families=FAMILIES,
        original_ink=int((original==2).sum()),original_background=int((original==1).sum()),
        per_block_known_retention=(kept.sum(axis=1)/known.sum()).tolist(),
        minimum_core_retention_each_block=(kept/known[None]).min(axis=1).tolist(),
        newly_scored_known_pixels_outside_parent_mask_each_block=(((states>0)&~old["common_mask"][None]).sum(axis=(1,2,3))).tolist(),
        secondary_global_known_retention=global_retention,mixed_results=summaries,complete_array_review=full,
        checker_sha256=sha(__file__),numpy=np.__version__,sklearn=sklearn.__version__,seconds=time.monotonic()-started,
        scope="Paired four-cell masks per axis-angle block; cross-block AP has different populations. Portable scoring certifies archive arithmetic and exact reuse of stored parent fields, not original network inference or CT acquisition. Full mode extends readback checking to newly scored historical pixels. No IID-pixel significance or anatomical causal attribution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output","protocol","parent-evidence","report"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("prepared","parent-prepared","parent-output","author-cache","checkpoint"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--expected-protocol-sha")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh review report")
    result = audit(args)
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:result[key] for key in ("status","archive_bytes","scalar_checks","maximum_metric_absolute_error","seconds")}))
