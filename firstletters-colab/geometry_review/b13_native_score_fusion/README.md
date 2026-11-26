# B13 / B14 — actual local replay and new CPU fits

26 Nov 2026. Follow-up to #12, B9 and Q33. Owner supplied the three scored archives and four prepared CT/label/index arrays. All seven SHA256 values match their published identities; all source labels and crop coordinates agree. No Studio weight file supplied. Public exposed PHerc0841 donor region only. B13 protocol committed at 191df215; B14 protocol at 8fad2ab1.

## B13: resolve B9 bounds with actual pixelwise overlap

39 experiment rows, 429 attempted held-core calibration folds, 428 admitted; 19 exact overlap comparisons. Original B9 calibration convention replayed and extended with spatial guard. Every per-core actual OR increment lies within the sharp count-only bound. At y-5 identity, OR adds 199/5327 ink positives (+3.74 pp) and 1237/5892 background positives (+20.99 pp). At y-20 rigid, actual 2975/5291 and 2077/5848 exactly reproduce the earlier degenerate bounds. All eight identity OR arms increase FPR; two add zero ink. The first preflight stopped at a bitwise check of the archived mean-five using FP64 accumulation; FP32 accumulation reproduces all values exactly (FP64 max difference 1.1920929e-7). Corrected only source check before scoring; no outcome-based change.

Guarded pose results (unweighted mean of eight within-block recall/FPR deltas in percentage points, 11 mixed cores, guard192 chart pixels):

| Baseline | Fusion | Recall delta | FPR delta | Joint recall gain/nonincreased FPR blocks |
|---|---|---:|---:|---:|
| identity | max | -8.668 | -1.553 | 0/8 |
| identity | mean | -11.094 | -3.912 | 0/8 |
| identity | zmean | -11.978 | -4.620 | 0/8 |
| rigid | max | +10.993 | +2.862 | 1/8 |
| rigid | mean | +6.422 | +0.941 | 1/8 |
| rigid | zmean | +4.907 | +1.472 | 1/8 |

With rigid y-20 and a calibration FP budget of zero, guarded max still produces 731/5291 true positives and 35/5848 false positives on held cores. Matched calibration FP is NOT matched held-out FP. Rigid hybrids use oracle reference geometry, not deployable correction. Guard192 is a chart-distance proxy, not verified native CT nonoverlap.

Oblique mild5 guard192: max recall -13.62 pp, FPR -3.41; mean -11.59/-3.91; zmean -11.65/-3.85. Full 13-arm similarly poor. On scale all5 known-core leave-one-out, mean improves recall +3.44 pp and reduces FPR -2.69 pp, but with guard272 on admitted 10/11 cores it loses recall -2.97 pp and reduces FPR -5.71 pp. Guarded max +0.34/-3.32 pp on 5227 ink/4993 background labels; core360 abstains for lack of calibration support (117 ink,907 bg). No promotion from this retrospective donor control. Macro per-core AP is not pooled AP. No tuning or claim of generalization.

[fusion_summary.tsv](fusion_summary.tsv) has all 39 rows, [or_overlap.tsv](or_overlap.tsv) all 19 overlaps. Complete per-fold JSON generated locally; SHA256 2524d333a0ed85131aad2a33438728959ca858c22ed7593b54b27e23d5c4112b. Seven synthetic validation checks; actual source shape/order, labels, support, bitwise reused probabilities, ensemble, FP budget and overlap bounds checked. Internal wall 1.896s, full process elapsed 3.79s, peak RSS 174720 KiB.

## B14: new CPU CT model fits

178,423 known labels (all exactly one-tile supported over each 5x5 neighbourhood). Four test/calibration quadrants with 16-pixel guards, other two quadrants used for training, 43,475–108,380 training pixels/fold. Eight actual newly fitted models (4 logistic regression, 4 gradient boosting); 53 held cores and 45,349 known labels. Four CPU processes, 4.442s internal elapsed, 10.841s summed worker CPU; full process 6.62s. All 53 evaluated once.

| Model | Ink TP / 15791 | Bg FP / 29558 | Recall | FPR | Macro AP (23 two-class cores) |
|---|---:|---:|---:|---:|---:|
| Studio identity (existing maps) | 10718 | 5765 | .678741 | .195040 | .572089 |
| Logistic 16 intensity values | 1854 | 4804 | .117409 | .162528 | .373547 |
| HistGB 65 profile/local features | 1666 | 4404 | .105503 | .148995 | .350602 |

Mixed 11 cores (5344 ink,5900 bg): Studio recall .677208, FPR .492373, macro AP .639194; logistic .100674/.138644/.477936; HistGB .076160/.094915/.442835. Lower held FPR does not compensate for severe recall loss; held-out AP also worse on average. Calibration quadrant0 has very low baseline FP budget (200/6983), and applying its thresholds to quadrant3 suppresses most ink; this is genuine spatial calibration shift, not a reason to choose a better threshold on test labels. Local features are not a satisfactory replacement for Studio on this exposed donor. Not a fair equal-training-budget architecture comparison, not a proof against other local models, not independent manuscript validation.

[ct_summary.tsv](ct_summary.tsv): all 24 fold/group/model rows including within-fold pooled AP; [ct_per_core.tsv](ct_per_core.tsv): all 53 core rows. 387 count/AP checks recomputed from saved float64 predictions and frozen thresholds. Full local CT results JSON SHA256 6718092c22b73f2a7ee7b6484ab4b56cf010bdcb3b877459e0bcd4d4c8fc70bb; prediction NPZ SHA256 d4134bb10bd60f4e7a8103169eead1f6c5f7ca1ce5033b00837f82987e2ca73e. Full JSON and NPZ retained locally rather than committed through text connector; owner receives ZIP. Unknown predictions NaN, masked; no independent ink labels or new reading.

## Reproduction

Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0. Source filenames and exact SHA256 in replay.py. Obtain the three published scored arrays (rename pose_factors.npz, oblique_stress.npz, scale_stress.npz) and four original author NPYs from Q33 pinned source URLs. These are the same author donor data, not new observations.

```sh
cd firstletters-colab/geometry_review/b13_native_score_fusion
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python replay.py /path/to/seven-files /tmp/b13.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ct_baseline.py /path/to/seven-files /tmp/b14 --workers 4
```

Both scripts and all four TSVs verified byte-identical local/remote by Git blob hashes. Full runs already completed locally, no agent needs to rerun them except for independent verification.

## Next action

Use geometric diagnostics to trigger repair rather than blanket fusion; keep native masks and regional errors. Do not spend resources deploying local radiometric classifiers on unread papyri based on this result. When Studio checkpoint is available, prioritize new context/geometry-conditioned forwards and an independent optical-control replication rather than another same-donor threshold sweep. No new Studio inference, raw volume CT sampling, GPU spending, candidate publication, automated skip or merge.
