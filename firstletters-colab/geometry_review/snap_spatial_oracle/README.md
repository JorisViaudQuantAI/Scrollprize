# B12/B12b — spatial snap-selection limits and sparse reference-label stress

26 Nov 2026. Local CPU executions on all four public already-awarded PHerc0211 fitted/snap reciprocal code grid pairs from Adaptive067, not new CT or new ink inference. Protocols frozen at c8c949f5 and cef48822 before their respective executions. Source blob 2e4a8b220b6f9c090d8cb55fa9cd0c6e6ea413ef on chatgpt/adaptive-surface-recovery. Eight base64/zlib grids transferred via connector to local container, SHA256 verified after decoding; all four exact 7x7 transition tables reproduced. No local Git network/weight downloads possible. No flat-grid same-index comparisons.

## B12 exact geometry oracle

48,909 common nonzero vertices, 96,749 four-neighbour edges (23,212+23,900+24,450+25,187). 72 exact graph cuts over nine switch penalties and two unresolved-cost arms. All 512 small-grid exhaustive comparisons pass; mincut/flow equality and monotone tradeoff checked. At unresolved cost 2:

| Choice | Near<=3 | Far>=6 | Unresolved | Switches |
|---|---:|---:|---:|---:|
| Fitted | 6184 | 13406 | 24522 | 0 |
| Snap | 9308 | 11502 | 24536 | 0 |
| Oracle lambda0 | 10756 | 9192 | 25220 | 7268 |
| Oracle lambda1 | 10083 | 9563 | 25284 | 1341 |
| Oracle lambda2 | 9673 | 10131 | 25045 | 436 |
| Oracle lambda4 | 9483 | 10342 | 25042 | 191 |
| Oracle lambda8 | 9289 | 10647 | 24991 | 81 |
| Oracle lambda16 | 9274 | 11467 | 24523 | 0 |

Unregularized +1448 near vertices (+2.96 percentage points over common support); lambda2 retains only +365. With stronger penalty the oracle chooses whole-winding policies, and at lambda16 fitted is preferred on w035. Higher unresolved cost changes the tradeoff, including a different whole-winding choice on w033. Treat no-hit/ambiguous/oblique as unresolved, not blank. Grid-index switch edges do not prove physical continuity or a valid welded mesh. Full 88 numerical rows in [results.tsv](results.tsv).

## B12b sparse held-out stress

32 paired seeds per winding x 3 budgets x 3 noise arms = 1152 scenarios, five policies = 5760 evaluations; 12,587,475 held-out vertex occurrences per policy (62,937,375 across all five). One scenario is one winding, seed, budget and noise arm; NOT independent manuscripts. Held-out counts 8,739–12,540. No anchor or vertex within 2 index units of anchors scored. Predictor uses only sampled reference unary deltas and valid-support grid coordinates; this is an idealized anchor-label test, not actual fibres or CT prediction.

| Anchors | Inversion rate | Graph cut wins/harms/ties vs snap | Mean unary delta per held-out vertex |
|---:|---:|---:|---:|
|16|0%|41/29/58|+.002210|
|64|0%|62/55/11|-.014394|
|256|0%|76/49/3|-.024920|
|256|10%|71/55/2|-.014807|
|256|25%|63/65/0|+.012380|

256 clean anchors: w036 32/32 improved, w034 29/32 harmed; mean near fraction across runs changes -0.032 percentage points, far -1.627 points, unresolved +0.729 points. The composite cost improvement does NOT imply improved near recovery. Noise is sign inversion of selected labels; zero labels do not change. Mean delta is the unweighted mean of run-level paired differences; changing anchor density changes the held-out population. No significance intervals treating these related seeds as independent papyri.

[sparse_aggregate.tsv](sparse_aggregate.tsv) publishes 36 pooled method rows and 36 winding-specific graph-cut rows, including worst harm and seed, rather than concealing failures. Full output produced locally contains all 1152 scenarios and 180 aggregate rows. Recreate with the script below; raw per-run JSON not uploaded via text-only connector. Local SHA256 sparse_results.json c630c6ab974dc4c3e85b461c3088fd8fc0c97f92928248abfd2607511d1ff720; result.json 31180f30aa376b4684aa994cb4374b6a97ee49862e11d3de6f99f9337960da6f (timing/RSS fields vary by run).

## Reproduction and receipts

Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0. B12 full process elapsed 3.35s, user+system 3.34s, max RSS 124416 KiB. B12b four worker processes elapsed 13.30s, user+system 37.77s; three additional interpolation/held-out leakage checks pass, and 1152 cut-flow equality checks. Byte-identical local-to-remote Git blob verification: audit.py c5e305cea5d0591d160979254407da45f1946801; results.tsv 37412cdbebb96ff4e7fe55393f0e1264af3bc586; sparse.py c2f82dea49558dc850357ac3c0dd958c48f3fae8; sparse_aggregate.tsv fdead3b2dc242cbea248676a9de14569c1b08f19.

```sh
git fetch origin chatgpt/adaptive-surface-recovery
git cat-file blob 2e4a8b220b6f9c090d8cb55fa9cd0c6e6ea413ef > /tmp/atlas067.json
cd firstletters-colab/geometry_review/snap_spatial_oracle
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit.py --source /tmp/atlas067.json --output /tmp/b12.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sparse.py --source /tmp/atlas067.json --output /tmp/b12b.json --workers 4
```

Local runs used --maps-dir, with decompressed byte hashes and transition tables checked. The --source Git-blob acquisition path is provided for agents with a clone; not executed here. No CT/weight download, new candidate, ink recall, GPU expenditure, automatic skip, production repair or merge. No new tuning to recover the negative result.

## Handoff

Do not apply a global snap selector/interpolator based on sparse labels across charts. Claude: focus on verified local fibre constraints and real physical seams; include w034-like harm cases. Gemini: preserve paired native footprints and abstentions before downstream detector runs. Codex: replay independent source path and verify physical support before any repair promotion. This experiment is only a mechanism/ceiling on already-awarded control geometry, not evidence of new readable letters.
