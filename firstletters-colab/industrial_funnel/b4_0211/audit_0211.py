#!/usr/bin/env python3
"""Finite-population audit of published PHerc0211 windows, not an ink verdict.

A result here describes a selected, already-awarded positive control. It
must not be used to automatically discard windows of unread scrolls.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

SOURCE_COMMIT = "1770a7b9f223a016167fda440f1867848373119d"
WINDOWS_GIT_SHA1 = "d6a337f299fa5b893dbc84f6de23542adbbd83a6"
SEALED_GIT_SHA1 = "7aa64aa902924bff95bdec3f5cb9360154e8e0f5"
RANDOM_REVIEW_RANKS = (27,29,37,43,52,64,76,88,91,108,122,127,129,138,144)
TOP_K = 25
FLAGS = {"A19": ("fw033_x1", "clear_known_location_not_blind"),
         "A16": ("fw030_x1", "clear_stroke_group_identity_unverified"),
         "A02": ("fw035_x1", "weak_known_location_not_blind")}
LOOKS = {"A17": "fw057_x3", "A36": "fw055_x1", "A06": "fw047_x0"}
ALPHA = Fraction(1,20)

def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

def read_pinned_json(path: Path, expected_sha1: str):
    raw = path.read_bytes()
    sha = git_blob_sha1(raw)
    if sha != expected_sha1:
        raise ValueError(f"Changed source: {path}; expected {expected_sha1}, got {sha}")
    return json.loads(raw)

def p_no_flagged(population: int, sample: int, missed: int) -> Fraction:
    """Exact SRSWOR probability of zero flagged units in a fixed population."""
    if not all(type(z) is int for z in (population,sample,missed)):
        raise TypeError("Counts must be integers")
    if population < 0 or sample < 0 or sample > population or missed < 0 or missed > population:
        raise ValueError("Invalid finite-population counts")
    if sample > population - missed:
        return Fraction(0)
    return Fraction(math.comb(population-missed,sample),math.comb(population,sample))

def max_missed_after_zero(population:int, sample:int, alpha:Fraction=ALPHA) -> int:
    """Largest missed count not excluded at one-sided alpha, given zero in sample."""
    if not 0 < alpha < 1: raise ValueError("Invalid alpha")
    if sample == 0:return population
    return next((m-1 for m in range(population+1) if
                 p_no_flagged(population,sample,m)<alpha),population)

def minimum_zero_audit(population:int,misses_to_exclude:int,alpha:Fraction=ALPHA)->int:
    if not 1 <= misses_to_exclude <= population:raise ValueError("Invalid target")
    return next(n for n in range(population+1)
                if p_no_flagged(population,n,misses_to_exclude)<alpha)

def validate_and_summarize(windows:list,sealed:dict)->dict:
    if len(windows)!=149 or len(sealed)!=40:raise ValueError("Expected 149/40")
    ids=[w.get("window") for w in windows]
    ranks=[w.get("rank") for w in windows]
    if len(set(ids))!=149 or set(ranks)!=set(range(1,150)):
        raise ValueError("Duplicate/missing window names or ranks")
    by_id={w["window"]:w for w in windows}
    for w in windows:
        if w.get("det")!="st96desc":raise ValueError("Changed detector/order recipe")
        # 54 genuine records have line=null, lr=-99: abstention is not no ink.
        if w.get("line") is None:
            if w.get("lr")!=-99:raise ValueError("Unexplained line sentinel")
        elif not math.isfinite(float(w["line"])) or w.get("lr")!=w["line"]:
            raise ValueError("Invalid line statistic")
        if not all(math.isfinite(float(w[f])) for f in
                   ("row","hot","glyph","valid_share","valid_cm2")):
            raise ValueError("Nonfinite score/area")
        if not 0<w["valid_share"]<=1 or not 0<w["valid_cm2"]<=2.81:
            raise ValueError("Invalid coverage/area")
        if w["x1"]<=w["x0"] or not isinstance(w["winding"],str):
            raise ValueError("Invalid chart crop")
    expected=set(range(1,TOP_K+1))|set(RANDOM_REVIEW_RANKS)
    if set(sealed)!={f"A{i:02d}" for i in range(1,41)}:
        raise ValueError("Sealed-review keys changed")
    if len(set(sealed.values()))!=40 or set(sealed.values())-set(ids):
        raise ValueError("Invalid sealed references")
    if {by_id[wid]["rank"] for wid in sealed.values()}!=expected:
        raise ValueError("Not the published top25 plus random15 review")
    for k,(wid,_) in FLAGS.items():
        if sealed[k]!=wid or by_id[wid]["rank"]>TOP_K:
            raise ValueError("Flag assignment contradicts published sealed review")
    for k,wid in LOOKS.items():
        if sealed[k]!=wid:raise ValueError("Look assignment changed")
    sample_ids=set(sealed.values())
    top=[w for w in windows if w["rank"]<=TOP_K]
    rand=[w for w in windows if w["rank"] in RANDOM_REVIEW_RANKS]
    rest=[w for w in windows if w["window"] not in sample_ids]
    null=[w for w in windows if w["line"] is None]
    null_sample=[w for w in rand if w["line"] is None]
    scored_low=[w for w in windows if w["rank"]>TOP_K and w["line"] is not None]
    scored_sample=[w for w in rand if w["line"] is not None]
    if [len(null),len(null_sample),len(scored_low),len(scored_sample)]!=[54,6,70,9]:
        raise ValueError("Missing-score stratum changed")
    area={k:round(sum(w["valid_cm2"] for w in group),6) for k,group in
          (("total",windows),("top_25",top),("random_15",rand),("unreviewed_109",rest))}
    if abs(area["total"]-293.369)>0.002:raise ValueError("Total area mismatch")
    upper=max_missed_after_zero(124,15)
    return {
      "schema":"b4.0211.finite_review.v1",
      "source_commit":SOURCE_COMMIT,
      "source_git_blobs":{"windows":WINDOWS_GIT_SHA1,"sealed_key":SEALED_GIT_SHA1},
      "already_awarded_control":"PHerc0211",
      "coarse":{"windows":149,"valid_area_cm2":area["total"],"s96_gpu_seconds_69_maps":5801},
      "fine":{"reviewed":40,"top25":25,"random15":15,"s48_gpu_seconds_40_windows":4400,
              "observed_visual_flags_top25":3,"observed_visual_flags_random15":0,
              "flags":[{"window":wid,"rank":by_id[wid]["rank"],"status":s}
                       for wid,s in FLAGS.values()]},
      "area_partition_cm2":area,
      "unreviewed":{"windows":109,"area_cm2":area["unreviewed_109"],
                    "fraction_area":area["unreviewed_109"]/area["total"]},
      "no_line_score":{"windows":54,"area_cm2":round(sum(w["valid_cm2"] for w in null),6),
                       "randomly_reviewed":6,"unreviewed":48,
                       "one_stratum_95pct_upper_if_zero":max_missed_after_zero(54,6),
                       "status":"ABSTAIN_OR_NO_LINE_SCORE_NOT_BLANK"},
      "scored_lower_rank":{"windows":70,"randomly_reviewed":9,
                           "one_stratum_95pct_upper_if_zero":max_missed_after_zero(70,9)},
      "audit":{"randomization":"SRSWOR 15 of ranks 26..149 after top25 census",
               "frozen_random_ranks":list(RANDOM_REVIEW_RANKS),
               "published_sealed_key_matches":True,
               "one_sided_alpha":float(ALPHA),
               "lower_rank_population":124,
               "max_additional_visual_flags_not_excluded_95pct":upper,
               "p0_upper_exact":str(p_no_flagged(124,15,upper)),
               "p0_upper_plus_one_exact":str(p_no_flagged(124,15,upper+1)),
               "conditional_min_flag_recall_top25":3/(3+upper),
               "fisher_p_3_of_top25_zero_of_random15_descriptive":
                   math.comb(25,3)/math.comb(40,3),
               "zero_event_audits_to_exclude_at_least_m_flags":
                   {str(m):minimum_zero_audit(124,m) for m in (1,3,6,12,21)}},
      "scientific_admission":"NOT_ESTABLISHED","production_skip_allowed":False,
      "limitations":[
          "Three labels are visual review flags, not independent ground-truth ink annotations; one weak, two known-location.",
          "All windows are from a text-selected band of one previously awarded scroll.",
          "SRSWOR inference requires the documented random draw and perfect counting of the chosen review flags.",
          "Counts are windows, not independent physical sheets or ink sites.",
          "Null line score in 54 windows does not certify absence of text.",
          "One-stratum 95pct bounds are not jointly 95pct and cannot be added at that confidence.",
          "No fine-screen recall, end-to-end GPU savings or cross-scroll transfer can be certified here."]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows",type=Path,required=True)
    p.add_argument("--sealed-key",type=Path,required=True)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    r=validate_and_summarize(read_pinned_json(a.windows,WINDOWS_GIT_SHA1),
                             read_pinned_json(a.sealed_key,SEALED_GIT_SHA1))
    out=json.dumps(r,indent=2,sort_keys=True)+"\n"
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(out)
    else:print(out,end="")

if __name__=="__main__":main()
