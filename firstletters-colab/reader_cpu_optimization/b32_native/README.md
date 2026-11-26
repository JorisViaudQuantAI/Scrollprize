# B32–B36 — actual Reader kernels, random access, real fibre holdout and transfer

26 Nov 2026. Stacked on #20; B26–31 had already addressed the five requested priorities. This continuation executes distinct tests. All inputs are public exposed or already-awarded controls. No GPU, private/unread inference, production promotion or merge. Protocols frozen in PROTOCOL.md (750206c5) and LOCAL_PROTOCOL.md (ee7ca104) before outcome analysis. Local Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, sklearn 1.8.0, torch 2.10.0+cpu; hosted Torch 2.8.0 CPU.

## B32: actual Reader v3 quantization, not student inference

[Successful run 45875857844](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45875857844), [full results and prediction arrays](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45875857844/artifacts/15144449387). SHA-pinned actual Reader weights and Villa revision. Six W058 contexts for source activation min/max calibration, source threshold 0.2621695101261139. Six held W060 and three held 0841 each orientation, 12 held context-direction pairs, two alternating repetitions per method, 72 timed forwards. One thread; preprocessing not timed. 63/63 eligible Conv2D layers actually replaced with quantized x86 kernels; 33,588,256 weight elements of 34,546,498 float parameters (97.23% ratio, not whole-model quantization or measured file reduction). Float 3D stem, norms, residuals, unsupported layers, dequant at each boundary. No conversion failure.

| Method | Median forward s | AP worst loss | Changed decisions at transferred threshold | Speed gate |
|---|---:|---:|---:|---|
| FP32 |1.987707|0|0|baseline|
| channels-last |2.756972|4.2e-8|0/442368|fail|
| selective INT8 |2.003218|.002826|4804/442368|fail|

INT8 maximum score difference .0547931, 543/442368 decisions changed at .5. All held group recall/FPR changes remain within the predeclared 2pp gate and AP loss <=.01; speedup 0.9923x fails >=1.25x. Channels-last is not bitwise but both evaluated decision thresholds identical. The source threshold itself badly overcalls background on 0841 (as in B26); equivalence to an inferior operating point is not improved ink detection. Quant/dequant overhead is a hypothesis, not profiled causal evidence. Need fused quantized islands / profiling before reattempt. BF16 remains a separately measured candidate, not a safe blanket veto. The result artifact was downloaded and locally replayed: 72 count/AP/decision-change checks pass (AP tolerance 1e-5 across sklearn versions). Full rows in reader_quant_receipt.json. Peak RSS is combined benchmark process, not per-arm memory comparison.

## B33: exact storage of already-rendered native CT

15 real B26 panel contexts, 46,448,640 uint8 values, 48 deterministic queries x2 repeats x4 methods = 384 equality checks plus five integrity/bounds checks. Warm filesystem, fresh per-method application cache. Naive NPZ deliberately inflates the entire array on each read; a retained resident array is a fair in-memory alternative. Blocks 21x64x64 zlib1, offset index, SHA256 per uncompressed block, bounded 16-entry LRU; corruption and truncation checks. This is a research prototype for rendered immutable panels, not production persistent native Zarr transport. Hashes validate against the supplied index, not an adversarially replaced index; Python asserts require non-optimized Python.

| Method | Storage bytes | Median ms | Full384 median ms |
|---|---:|---:|---:|
| NPZ inflate each query |37983898|232.613|234.304|
| NPY mmap |46448768|.0900|.1664|
| independent zlib blocks |37000300|4.852|19.720|
| resident | RAM 46448640 |.0580|.1274|

NPZ 4,459,069,440 logical inflated bytes vs block 126,271,488 over both reps, no actual wire traffic measured. Block storage slightly smaller than NPZ on this fixture; no claim always better. Recommend NPY mmap for repeatedly reused small panels, compressed index if storage constrained. Persistent CT native transport and parallel fetch already measured by B26; no new speed claim for remote rendering.

## B34: real fibre held-out geometry

[Public fixture acquisition run 45875961150](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45875961150), [fixture artifact](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45875961150/artifacts/15144264108). Original work by Jeff Chen and contributors, [fiber-sheet pinned source](https://github.com/jeff-j-chen/fiber-sheet/tree/e1f05571b095c41dded6121a784b794ba308c510/data/grid1_final26). 116 real curated traces (38 h/78 v), 2964 published crossing-table rows. 118 file Git blobs checked against the pinned tree; known ply9 and layer0 grouping supplied from publication. All held-out trace samples excluded from fitting; two five-fold designs, 30 surface fits, train-only ply direction and smoothing, held-held crossing normals from independent trace tangents. Common training hull + <=400 vx distance support, 6409/6503 interleaved, 6289/6503 spatial. Three synthetic checks. Approximate normal distance, not certified closest surface distance or sheet identity.

| Split / method | MAE vx | p95 vx | far>=6 | median crossing normal error |
|---|---:|---:|---:|---:|
| interleave affine |147.116|321.798|6244/6409|18.625°|
| interleave TPS |5.518|15.663|2068/6409|4.091°|
| interleave robust TPS |6.304|18.363|2378/6409|4.507°|
| spatial affine |158.977|348.887|6151/6289|20.411°|
| spatial TPS |6.832|21.665|2340/6289|6.330°|
| spatial robust TPS |7.605|24.220|2584/6289|6.498°|

Global TPS greatly outperforms a single affine plane on real traces; robust reweighting harms both held designs, p95 +2.70/+2.55 vx and severe-error +4.84/+3.88pp. 111/105 supported held-held crossings. Spatial p95 angular 24.30° for TPS. Training sampling cap 16/fibre and no held-crossing span selection means not a reproduction of the winner's full mesh. Published curated trace and crossing/ply prior exposed, not independent anatomical truth. No new CT or ink evaluation on 0846A, no sheet-hop verdict. Claude can use this held-fibre baseline to compare local TPS, normal-field integration and abstention on real constraints. Preserve difficult tail, do not blindly smooth or infer blankness from unsupported regions.

## B35: weak target blank supervision and topology

Same B21/B26 actual scores, 4 target domains (0841 both orientation hypotheses), source-only source80 threshold vs first trusted target blank crop used exclusively for negative quantile transport, plus 8-connected hysteresis using source50 high seeds. Exclude calibration blank from all arms; same held labels. 16 metric rows and three synthetic checks (one-pixel line/open C, isolated distractor, conservative tie). No target mixed label used for thresholds; trusted target blank not guaranteed available on an unread scroll. Hysteresis cannot add or reconnect pixels outside low mask.

| Target | source recall/FPR % | blank transport recall/FPR % | source hysteresis recall/FPR % |
|---|---:|---:|---:|
| W058 |65.25/36.67|99.62/82.35|62.75/33.70|
| W060 |98.94/79.98|94.31/73.35|89.06/70.65|
| 0841 asc |100/94.56|84.80/36.58|100/94.53|
| 0841 desc |92.91/96.76|57.34/32.89|92.69/96.70|

All advancement gates fail. W060 source hysteresis worst positive crop loses 94.72pp recall despite synthetic thin-stroke checks passing. Blank transport improves 0841 false positives with unacceptable recall loss; W058 background worsens greatly. Neither is a safe universal veto. Both 0841 hypotheses retained; no target-label orientation choice.

## B36: larger-context student / distillation

Six actual local fits (3 source folds x2 objectives, 192 updates each = 1152), 55,553 parameters, 11 Conv2D converted to actual INT8 per model, 16 held evaluations. Encoder-decoder skip connections, batch norm; known hard labels only, optional teacher soft BCE on center64. Held calibration crops separate, no target training. Median full384 one-thread forward 124.060ms float vs 40.253ms INT8 (3.082x on same student and CPU; not comparison with hosted Reader timing). All 16 fail quality gate, zero conversion failure. Example 0841 ascending label/student AP .31296, distill .33829 vs teacher .79085. Teacher larger RF and data budget incomparable. No claim more training would solve it; no production/student pruning claim in this experiment (B28 previously measured pruning). All teacher/student counts, fixed-float-threshold outputs and trained weights retained locally. Local wall 96.95s, fit time 66.71s. Need multiple parent-separated labeled fields and broad teacher contexts before another distillation sweep.

## Reproduce and evidence

Artifact IDs: B21 predictions 15137042105, B26 native 1447 15137874754, native841 15138643992, 841 predictions 15139875260. Unpack as DATA/b21, DATA/panel1447, DATA/panel841, DATA/pred841. Four SHA256 and aligned label/origin checks in panels.py. Download B34 fixture and unpack into FIBRES. All executed local script Git blobs verified:
- panels.py 5cbf9357e13de220e5ea23530079f290169f0551
- storage33.py 0c955ac4f987b6b042c687ef3a5ade1f1a921532
- fibres34.py a6189387adafcdeca2cb1d9a986765ca87c5d19c
- transfer35.py 7c20e28794519acda812118a155e49a5bc3a3e7a
- student36.py c6cc1c3bc6c65d0f7b1a8d28c8326da662aebab3

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python storage33.py DATA /tmp/b33
python fibres34.py FIBRES /tmp/b34.json
python transfer35.py DATA /tmp/b35.json
PYTHONPATH=.. python student36.py DATA /tmp/b36
```

B32 full hosted runner script/workflow committed, local_receipts.json stores all aggregate rows and full output hashes. Full per-query/fibre/crop JSON and six generated student weight files, B32 predictions and results in owner archive SHA256 08b7cf90d48d85afea1461b9a921d9df3acbec8dedb5efa91c6ebf94b774ec50. Original data and contributor credits/terms preserved. Artifacts expire; pinned source acquisition or local files can regenerate. No coordinates for unread eligible scrolls. No main merge.

**Decision:** use staged memory-mapped rendered fixtures and the actual held-fibre benchmark for further review; reject robust TPS, current standalone quantized Reader wrapper for speed, student deployment, negative anchor threshold transport and hysteresis veto. For future work, isolate true geometric normals/sheet identity on physical controls; expand independent parent labels; profile quantized islands before modifying production.
