# B9 — geometry rescue: complementarity bounds and calibrated fusion

2026-11-26. Follow-up to Q33/PR #5, B8/PR #11 and the November 24 First Letters announcement. No unread regions, private candidates, new inference, or production gate.

## Completed real-data calculation

Pinned source: Q33 factor results at 8f2e576ff961a4f247fb545b5317b223988e57be; blob 403fd08e9619783438b67a73634850456b23ac94. Eight paired blocks, 17 cores (11 mixed), four sampling families, per-block common support and threshold .5. The binary NPZ was unavailable through the connector; use published per-core counts, not inferred overlap.

For each core/class, baseline count b, additional detection counts a_i and support N, the possible number of new detections in the OR union is exactly bounded by max(0,max_i a_i-b) and min(N-b,sum_i a_i). Both endpoints are attainable for some sets consistent with the counts; sum across cores, not across correlated angle blocks. Separate ink and background. This is not a confidence interval or a new measured ensemble.

Mixed cores, identity + all three alternatives, absolute percentage-point changes at .5:

| Block | Ink recall gain | Background FPR increase |
|---|---:|---:|
| x -20 | 1.34–5.58 | 3.54–14.25 |
| x -5 | 2.53–4.98 | 4.31–22.12 |
| x +5 | .69–5.61 | 7.13–16.18 |
| x +20 | 0–.74 | .57–12.02 |
| y -20 | .02–2.61 | 1.11–11.32 |
| y -5 | 3.12–7.36 | 20.98–26.02 |
| y +5 | .49–4.02 | 6.13–17.45 |
| y +20 | 0–2.34 | 5.30–15.97 |

Thus uncalibrated OR necessarily increases FPR in all eight blocks. In the y-20 rigid baseline, 0/5291 ink detections become exactly 2975/5291 with the hybrid OR, but 2077/5848 background positives also appear. The hybrids require reference geometry: this is an oracle intervention, not a deployed recovery algorithm. No generalization or verified reading claim.

[Full 224-row results](result.json); [replay](audit.mjs). 2807 exhaustive marginal-cardinality oracle comparisons, four invalid-count rejections and 2272 source consistency checks ran in the JavaScript connector runtime. Standalone Node run and binary replay still need independent execution.

## Reproduce

From repository root:
```sh
node firstletters-colab/geometry_review/rescue_bounds/audit.mjs firstletters-colab/studio_pose_factors/evidence/results.json > /tmp/b9.json
python firstletters-colab/geometry_review/rescue_bounds/score_fusion.py --selftest
python firstletters-colab/geometry_review/rescue_bounds/score_fusion.py firstletters-colab/studio_pose_factors/evidence/scored_arrays.npz > /tmp/b9-fusion.json
```

The Python evaluator is prepared for a separate real replay, not an executed real-data result. It compares baseline/max/mean/z-score mean under leave-one-core-out calibration to the baseline background false-positive budget; report held-core recall/FPR and per-fold AP, with tied scores handled conservatively. Calibration folds are spatially correlated and already exposed; this is exploratory, not independent scroll validation. Z-score fusion is inspired by the 846A winner, but here only different renders of one checkpoint, not three distinct models. No threshold promotion. See the winner/control registry in the companion handoff.
