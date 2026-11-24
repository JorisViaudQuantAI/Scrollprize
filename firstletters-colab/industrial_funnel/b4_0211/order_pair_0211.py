#!/usr/bin/env python3
"""B4 regional depth-order diagnostic on nine *public* PHerc0211 map pairs.

These downsampled, quantized PNG activations are not human ink labels.
No automatic blank/skip or scientific promotion is permitted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Claude's independently preselected rng-211 nine paired windings.
PAIRED_WINDINGS = (14,18,21,30,33,41,47,49,52)
THRESHOLD_BYTE = 179  # ceil(0.70*255)
MIN_VALID_PIXELS = 100
MIN_ACTIVATIONS = 25
MARGIN_FRACTION = .10

def git_blob_sha1(data:bytes)->str:
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()

def image8(path:Path)->np.ndarray:
    with Image.open(path) as im:
        if im.mode!="L":raise ValueError(f"Not an original grayscale L PNG: {path} mode={im.mode}")
        return np.asarray(im).copy()

def region_stats(desc:np.ndarray,asc:np.ndarray,mask:np.ndarray)->dict:
    if desc.shape!=asc.shape or desc.shape!=mask.shape or desc.ndim!=2:
        raise ValueError("Map/support frames disagree")
    if desc.dtype!=np.uint8 or asc.dtype!=np.uint8 or mask.dtype!=np.bool_:
        raise ValueError("Expected uint8 probability PNG maps and bool support")
    n=int(mask.sum())
    dh=(desc>=THRESHOLD_BYTE)&mask
    ah=(asc>=THRESHOLD_BYTE)&mask
    d,a=int(dh.sum()),int(ah.sum())
    both=int((dh&ah).sum())
    either=d+a-both
    return {
        "supported_px":n,
        "desc_ge07":d,
        "asc_ge07":a,
        "both_ge07":both,
        "either_ge07":either,
        "activation_jaccard":both/either if either else None,
        "dominance_desc_minus_asc":(d-a)/(d+a) if d+a else None,
        "enough_support":n>=MIN_VALID_PIXELS,
    }

def pair_report(desc:np.ndarray,asc:np.ndarray,support:np.ndarray)->dict:
    if desc.shape!=asc.shape or desc.shape!=support.shape or desc.ndim!=2:
        raise ValueError("Unaligned pair")
    h,w=desc.shape
    if h<2 or w<4:raise ValueError("Small frame")
    regs={
      "whole":region_stats(desc,asc,support),
      "left":region_stats(desc[:,:w//2],asc[:,:w//2],support[:,:w//2]),
      "right":region_stats(desc[:,w//2:],asc[:,w//2:],support[:,w//2:])}
    l,r=regs["left"],regs["right"]
    strong=(l["enough_support"] and r["enough_support"] and
            min(l["desc_ge07"]+l["asc_ge07"],r["desc_ge07"]+r["asc_ge07"])>=MIN_ACTIVATIONS)
    ld=l["dominance_desc_minus_asc"]
    rd=r["dominance_desc_minus_asc"]
    opposite=bool(strong and ld is not None and rd is not None and
                  ((ld<-MARGIN_FRACTION and rd>MARGIN_FRACTION) or
                   (ld>MARGIN_FRACTION and rd<-MARGIN_FRACTION)))
    return {
      "shape_px":[h,w],"regions":regs,"regional_order_disagreement":opposite,
      "diagnostic_only":True,"production_skip_allowed":False}

def run(public_maps:Path,hashes_path:Path)->dict:
    hashes=json.loads(hashes_path.read_text())
    expected=[]
    for n in PAIRED_WINDINGS:
        w=f"fw{n:03d}"
        expected.extend([f"raw20/{w}__s96desc.png",f"raw20/{w}__s96asc.png",
                         f"maps_s96/{w}_valid_mask20.png",
                         f"maps_s96/{w}_parallel_mask20.png"])
    if set(hashes)!=set(expected):raise ValueError("Changed set of nine paired maps and masks")
    for name,sha in hashes.items():
        actual=git_blob_sha1((public_maps/name).read_bytes())
        if actual!=sha:raise ValueError(f"Changed PNG bytes: {name} {actual}!={sha}")
    out=[]
    for n in PAIRED_WINDINGS:
        w=f"fw{n:03d}"
        desc=image8(public_maps/"raw20"/(w+"__s96desc.png"))
        asc=image8(public_maps/"raw20"/(w+"__s96asc.png"))
        valid=image8(public_maps/"maps_s96"/(w+"_valid_mask20.png"))
        parallel=image8(public_maps/"maps_s96"/(w+"_parallel_mask20.png"))
        if not (desc.shape==asc.shape==valid.shape==parallel.shape):
            raise ValueError(f"Different geometry on {w}")
        support=(valid>0)&(parallel>0)
        out.append({"winding":w,**pair_report(desc,asc,support)})
    return {
      "schema":"b4.0211.regional_order.v1",
      "source_commit":"1770a7b9f223a016167fda440f1867848373119d",
      "source_sha1_git_blobs":hashes,
      "public_source":"Claude awarded-text PHerc0211 stride96 uint8 map export",
      "paired_windings":[f"fw{x:03d}" for x in PAIRED_WINDINGS],
      "threshold_byte":THRESHOLD_BYTE,"margin":MARGIN_FRACTION,
      "minimum_activations_in_each_half":MIN_ACTIVATIONS,
      "cases":out,
      "regional_order_disagreement_cases":sum(q["regional_order_disagreement"] for q in out),
      "scientific_admission":"NOT_ESTABLISHED","production_skip_allowed":False,
      "limitations":[
         "Quantized and downscaled display PNGs, not native float32 probabilities.",
         "Nine paired windows belong to one awarded manuscript and include no independent new-text truth.",
         "A global ratio near unity can hide spatially complementary model activations.",
         "Opposite local activations do not prove ink, different writing sides or anatomical sheet identity.",
         "All calculations restricted to common validity and parallel support, not a full source-presence guarantee.",
         "Descriptive regional warning only; a safe skip policy requires independent annotated control recall."]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--maps",type=Path,required=True)
    p.add_argument("--hashes",type=Path,default=Path(__file__).with_name("source_hashes.json"))
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    data=json.dumps(run(a.maps,a.hashes),sort_keys=True,indent=2)+"\n"
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(data)
    else:print(data,end="")

if __name__=="__main__":main()
