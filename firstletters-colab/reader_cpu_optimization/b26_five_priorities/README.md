# B26–B31 — five-priority CPU research: completed benchmarks

26 Nov 2026; stacked on #19. All tests on already public/exposed PHerc0841/1447. No production merge, unread coordinates, candidate publication or GPU work. This is a research contribution, not a completed production Reader v4 or corrected anatomical mesh.

## P0: CT acquisition and transfer (B26)

[Completed CPU run 45863091634](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45863091634) closes B22's cancelled inference. Stage.py saves an independently reusable native panel before inference; infer_841.py starts a fresh process with two CPU threads and emits every forward timing. Prior three B22 raw hashes reproduced exactly; 7,077,888 overlapping author CT samples bit-identical, all geometry/stencils valid. Cold vs fresh transport with persistent cache, one paired acquisition+render measurement:

| | Cold | Warm |
|---|---:|---:|
| wall seconds | 15.283837 | 4.773319 |
| new native HTTP body bytes | 241172719 | 0 |
| 21-layer CT values | 9289728 | 9289728 |

All cold/warm arrays identical. 3.20x measured on this panel, not full-volume or network wire throughput. Uses existing SupportedNative parallel transport and hash-checked disk cache; no new parallel transport scheduler claimed. Separate render and model stages, both success. B22 root cause not established, no assumption that thread count alone fixed it.

**36 actual Reader forwards** on three 0841 contexts, both depth orders, 3 bands and FP32/BF16. Source thresholds frozen on B21 1447 only. Native support labels 15,054 ink/28,721 bg, one blank crop (4,893 known bg).

| 0841 orientation | center recall/FPR % | median3 recall/FPR % | median3 delta pp |
|---|---:|---:|---:|
| ascending | 100.00/95.48 | 99.22/95.20 | -.784/-.286 |
| descending | 92.91/97.31 | 98.49/99.10 | +5.573/+1.790 |

Both fail predeclared transfer gate. Blank FPR 100% at pooled transferred center thresholds. Ascending/descending center AP .79085/.47640; target-label oracle FPR80 .26893/.84813, not deployable and **not** authorization to choose orientation by target scores. Center FP32 median1.127s, BF16 .594s on hosted two-thread CPU; BF16 differs at transferred thresholds on 190/110592 ascending and 242/110592 descending, not equivalent. Fixed source labels and different parent, but small exposed control, not blind independent validation.

Input/prediction artifacts (expiring): [841 panel](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45863091634/artifacts/15138643992), [1447 panel](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45863091634/artifacts/15137874754), [841 predictions](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45863091634/artifacts/15139875260). All were downloaded locally and exact individual hashes and label/origin identity checked. Runner metrics persisted in runner_receipts.json.

## P0: geometry (B27)

28 real prepared CT tiles, same artificial shifts/decoys as B20, 420 evaluation rows, six checks, 23.63s local. Baseline m15 is forward ZNCC .6/.1; conservative rescue requires agreement from two or more of four one-sided 8x8 windows, no conflicting admitted vote, and 5x5 bidirectional agreement. Adds only to abstained baseline sites.

| | m15 | combined |
|---|---:|---:|
| exact near step boundary / 40167 | 16991 | 20171 |
| noisy ramp exact/admitted | 242122/242342 | 251207/251427 |
| XY decoy admitted / 318319 | 290 | 290 |
| different-tile decoy admitted / 318319 | 958 | 963 |

+7.917pp exact seam coverage; ramp precision .999125; gate passes. 3180 added seam matches all exact in this synthetic scene. 12 synthetic structure-tensor normal cases (four normals x three noise levels): worst p95 .6994deg clean, 1.6659deg noisy SD .1; ideal smooth planes, not actual normal ground truth. No sheet-ID certification, fiber-constrained real interpolation, or new anatomical mesh. Directional rescue is a candidate to integrate with held-out physical fibre constraints; no blind smoothing across seams.

## P1: lightweight Reader student (B28)

Six actual new local CPU fits, 80 updates each, 32 held-target evaluations; both training objectives (known-label BCE vs mixture of label BCE and real Reader soft targets) on 1447 train crops, separate calibration crops; held other winding and pooled-source to 0841 both signs. Full 384 robust normalization; tiny 17->16->16->1 Conv2D student (5x5 receptive field), no large-context Reader architecture equivalence. Magnitude structured pruning to 12 channels, no retraining; actual static x86 INT8 with training-only observers. All quantization conversions pass. Median full384 CPU latency across 8 target rows per representation (one thread, no CT/normalization):

| | Parameters | MACs | Approx ms |
|---|---:|---:|---:|
| FP32 |4801|703070208|32.3|
| pruned FP32 |3169|463601664|26.6|
| INT8 |4801|703070208|8.2|
| pruned INT8 |3169|463601664|7.8|

No quality gate passes; 0841 asc AP float distill .34162 vs teacher .79085. Distilled student FP32 AP .30078/.15860 on held 060/058 vs teacher .54774/.37257. Mean absolute INT8 score drift .001763, max .084422 across student comparisons. Results do not measure full Reader INT8/pruning, and cannot fairly compare student local timing with teacher hosted timing. No deployment. Retain trained weights JSON in local archive. Future student needs multiscale spatial context, more parent-separated training data and proper validation.

## P1: topology and fibre-like cues (B29/B31)

B29 12 real held-winding rows, 144 artificial line/open-C tests. Median3 alone passes same-parent aggregate plus artificial stroke gate (identical input across synthetic depth bands, thus trivial synthetic median invariance; not proof of true multi-depth stroke preservation). One-pixel retention: spatial median/local mean product/erosion zero; line average 100% but fails held FPR gate. All artificial aperture FP zero. Artificial stroke scores on clipped real blank scores are not actual letters; coarse annotation components not certified strokes.

B31 uses true 21-plane native arrays, chart-plane structure-tensor coherence averaged across center17. Coherence alone AUC (negative coherence as ink score) .52330, .45274, .53887 on W058/W060/0841, sign varies; not actual fibre segmentation. Score center*(1-.5*C) vs center, source-only threshold:
| target | recall delta pp | FPR delta pp |
|---|---:|---:|
| W058 |+12.252|+13.166|
| W060 |-16.805|-8.031|
| 0841 ascending |-1.926|-14.557|
| 0841 descending |+5.268|-2.806|
Distinct-parent gate passes for this *one target panel* and both direction hypotheses, but same-parent gate fails badly; FPR remains high (80.93/94.50%). Do not promote universally, use as candidate confidence feature only on new predeclared independent parent controls. No fiber veto or letter discovery.

## P0: radiometric transfer and donor selection (B30)

[Completed CPU run 45863745735](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45863745735), 24 actual new Reader forwards. Original outputs bitwise reproduce B26 across both orders. Target-only unlabeled CDF matching to pooled source1447 worsens asc/desc FPR at fixed source threshold: 95.48->95.89%, 97.31->97.53%; gain1.15+bias8 or gamma.8 also fail to solve transfer. Gain-bias robust-MAD mean drift <.00084, as expected under near-affine intensity changes, not exact equivalence due clipping/rounding. Descriptive oracle results in runner receipts. Do not tune held target threshold to claim transfer.

Two candidate donor windings belong to same parent; both raw histogram Wasserstein and robust normalized quantile distance select W058 (raw distances 3.135 vs4.359, normalized .02683 vs .07982). But on 0841 asc FPR with W058-calibrated threshold 99.25% vs W060 91.85% (recalls 100/99.90%). Distribution similarity fails as sole donor selector. More independent manuscripts needed for DINO/gradient donor-selection optimization; not executed here.

## Reproduce

Hosted workflow artifacts can expire; runner receipts preserve numerical rows and all forward timings. Archive source panel SHA256s in local_receipts.json; owner supplied complete local result archive including weights and native panels. Locally: Python3.13.5, NumPy2.3.5, SciPy1.17.0, torch2.10.0+cpu, sklearn1.8.0. Copy/download artifacts into panel841/, panel1447/, pred841/ under DATA. Original B21 predictions SHA256 baf48b9301c9697c0dd8f4efbb7ffde9c2f891305f08d39194323eb860bd2d81. Execute from this folder:
```bash
python geometry27.py /path/to/original-npy-data geometry27.json
python topology29.py /path/to/B21/predictions.npz topology29.json
python student28.py DATA student28.json /path/to/B21/predictions.npz
python donor30.py DATA /path/to/B21/predictions.npz donor30.json
python fibre31.py DATA /path/to/B21/predictions.npz fibre31.json
```
Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 for local geometry/topology/donor/fibre; student sets Torch thread counts internally. All five local script blobs verified byte-identical to actually executed files, standalone helpers copied exactly from prior branches. Compact tables and JSON receipts include all aggregate result rows; full local JSONs and weights in owner archive. No claim all production priorities are solved. 

**Decision:** adopt isolated staging/cache reuse in review workflow only, request independent replay of directional geometry gate, prioritize real sheet/fibre constraints. No blanket multi-depth median, histogram matching, student deployment, morphology or fibre-based veto. Next data need is additional parent-separated labeled native fields for robust operating points and larger-context student training. No merge.
