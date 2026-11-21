#!/usr/bin/env python3
"""Descriptive feasibility check of the already chosen published 306B field.

This is not a field search, a blind validation, or a tile-placement experiment.
The fixed field predates this check. It is never amended by this script.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from diagnostic import Config, digest, outside_git, rank_map, reference_case, validate_inputs


FIELD_BOX_YXYX = [1268, 3356, 2618, 4514]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-controls-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    outside_git(output)
    if output.exists():
        raise FileExistsError("Use a new output file outside Git")
    reference = reference_case()
    base = args.public_controls_root / "cases" / reference["id"]
    paths = [base/"score.npy", base/"validity.npy"]
    expected = [reference["score_sha256"], reference["mask_sha256"]]
    if [digest(path) for path in paths] != expected:
        raise ValueError("Prepared source bytes differ from the pinned reference")
    score, valid = [np.load(path, allow_pickle=False) for path in paths]
    validate_inputs(score, valid, reference["spacing_um_yx"], "model_score")
    if list(score.shape) != reference["shape"]:
        raise ValueError("Prepared reference shape differs")
    y0, x0, y1, x1 = FIELD_BOX_YXYX
    score, valid = score[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    cv2.setNumThreads(4)
    result = rank_map(score, valid, reference["spacing_um_yx"], "model_score", Config())
    record = dict(status="postselection_pre_gpu_feasibility_diagnostic_only",
                  source="Released TRACE-times-mean-A-hn product, not new A-hn inference",
                  reference_case=reference["id"], field_box_yxyx=FIELD_BOX_YXYX,
                  shape=list(score.shape), spacing_um_yx=reference["spacing_um_yx"],
                  source_score_sha256=expected[0], source_mask_sha256=expected[1],
                  script_sha256=digest(__file__),
                  ranker_sha256=digest(Path(__file__).parent.parent/"line_review/review.py"),
                  config=asdict(Config()), valid_pixels=int(valid.sum()),
                  proposals=len(result["proposals"]), search_truncated=result["search_truncated"],
                  seconds=result["seconds"],
                  limitations=["Chosen field and unchanged ranker only; no coordinate search or tuning",
                               "This is not validation or a genuine phase comparison",
                               "The released fused product differs from planned single A-hn inference"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status":record["status"], "proposals":record["proposals"], "search_truncated":record["search_truncated"]}))


if __name__ == "__main__":
    main()
