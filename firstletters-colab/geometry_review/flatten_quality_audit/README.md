# B8 — paired chart quality and admission-gate sensitivity

Already-awarded PHerc0211, Claude-029 source commit 1770a7b9f223a016167fda440f1867848373119d. Four original source blobs pinned: raw and flat gridq tables, 149 windows and sealed review key. No new inference or labels.

## Actual paired chart comparison

Join the 60 fitted and flat charts by winding ID, not by array row or vertex. Report equally weighted chart macros because gridq.py's cell count is the one-step validity count, not the exact shear8 or roughness denominator, and flatboi changes the chart sampling lattice.

| Metric | Fitted | Flat | Improved charts |
|---|---:|---:|---:|
| Share with absolute 8-cell shear <= 8 degrees | .443667 | .956150 | 60/60 |
| Central shear span (p95-p5 degrees) | 50.879 | 8.019 | 60/60 |
| Turn stretch p95-p5 | 2.3605 | .09033 | 60/60 |
| Z stretch p95-p5 | .87917 | .06133 | 60/60 |
| Share roughness angle >20 degrees | .15557 | .07375 | 60/60 |

Total reported one-step valid cells increase 821653 to 1188985 (1.447x). This is **not** a 44.7% increase in distinct physical papyrus coverage: chart sampling and masks differ, and per-cell correspondence is not provided. All statements concern the released summary tables, not a fresh mesh flattening run.

## Gate sensitivity: preserve repair and exploration queues

Replay hypothetical whole-chart rejection gates at the three descriptive shear-share cutoffs 0.90, 0.95, 0.99, plus the rounded central (p5,p95) stretch range 0.95–1.05 in both axes. These are sensitivity checks on an already-exposed positive control, not preregistered admission thresholds.

| Gate | Excluded charts | Excluded windows | Excluded published visual flags |
|---|---:|---:|---:|
| shear share >= .90 | 0 | 0 | 0/3 |
| shear share >= .95 | 12 | 17 | 2/3 |
| shear share >= .99 | 60 | 149 | 3/3 |
| central turn + z stretch in .95–1.05 | 26 | 72 | 2/3 |

The .95 shear gate would remove 33.134 cm² summed window area, including nine unreviewed windows. Stretch gate would remove 143.054 cm², including 54 unreviewed windows. Stretch bounds are rounded to 0.01; these are not exact joint cell-level acceptance fractions. The three visual flags are weak/partly pre-exposed, one unverified, not validated labels or measured ink recall. All six entirely line-abstaining charts actually have high mean shear share (.961833), vs .955519 on charts with any score; geometry quality alone does not explain ranker abstention.

**Do not use any of these aggregate quality thresholds to declare blank or automatically reject a whole chart.** Route suspect geometry to local repair and preserve both native masks and unresolved windows. Require same physical footprints and paired inference against independent labels before deciding whether flattening changes ink sensitivity or GPU cost.

## Reproduce

```sh
node firstletters-colab/geometry_review/flatten_quality_audit/audit.mjs firstletters-colab/results/pherc0211_calibration
```

5 validation/partition checks and real cohort calculation executed in the JavaScript runtime on connector-returned source files; Node CLI/GitHub Actions replay still requested. No GPU, new CT, unread candidate or promotion.
