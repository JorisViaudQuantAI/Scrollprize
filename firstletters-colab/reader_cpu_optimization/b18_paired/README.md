# B18–B21: Reader v3 CPU optimization, depth robustness and operating-point transfer

## Status (2026-11-26)

Executed three local CPU benchmarks and **two successful GitHub Actions CPU jobs** with actual published Reader v3 weights. No GPU spend, production dispatch, merge or private/unread location inference. Local binary downloads remain blocked by networking; the GitHub-hosted runner solves that for public weights/data. Reader SHA256 ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a, Villa b29f310957c7b29e7d332f9876e0e487819cfaa3. Existing Studio checkpoint download succeeded previously on [Codex run 44918610779](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/44918610779), but its artifact expired; this branch has not run new Studio checkpoint inference. Do not confuse archived Studio scores with fresh Reader outputs.

## B18 — exact paired preparation

Protocol frozen at 495e0155. Local NumPy 2.3.5, 84 actual prepared CT pairs, 75 shared-path eligible, 1537 synthetic checks, 840 real bitwise checks including timed repetitions. Medians 170.682578 ms (reference), 32.4412385 ms (B15), 10.7166255 ms (paired). The local test pads 16 planes to 17; it is preprocessing only. Unsupported inputs preserve separate reference normalization. Inversion reverses selected depth values, not the padding. Initial timing invocation exceeded its timeout; rerun finished unchanged (41.43 s).

Actual Reader on two native PHerc0841 W00 384 crops, forward/reverse, two reps, four CPU threads. [Run 45858212182](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45858212182), [JSON artifact](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45858212182/artifacts/15136314239). Four timing rows per method, 16 timed paired evaluations (32 window-direction predictions), plus 4 reference predictions.

| Method | Median seconds/pair | Changed threshold decisions | Max error |
|---|---:|---:|---:|
| Reference FP32 | 1.916487 | 0 | 0 |
| Paired FP32 | 1.773193 | 0 | 0 |
| Paired FP32 batch2 | 1.763964 | 0 | 0 |
| Paired BF16 | 0.938481 | 742/1179648 | .009195864 |

FP32 output is bitwise identical on both crops and repetitions. BF16 changes 241104/1179648 encoded uint8 values, so threshold agreement alone is not equivalence. Time includes preparation and forward, not full surface tiling, stitching, acquisition or production throughput. No annotated accuracy claim for W00. Recommend testing exact FP32 path on more native surfaces before integration.

## B19 — calibration stress on archived Studio

Protocol 6a28a2ba. Six checks, 132 attempted/92 admitted folds; 88 pose and 4 scale folds. Guards of 192 and 272 chart pixels respectively, not physical native independence. Calibration on other mixed cores at >=80% ink recall. Held perturbation robust threshold excludes that arm; target quantile uses unlabeled target scores on training cores, not held labels. Compared with identity-calibrated threshold, pose robust +24.107pp recall/+30.393pp FPR; quantile +9.092/+13.234pp. All gates failed. Scale admits only one core per arm, not sufficient for generalization. Same donor, not actual cross-roll evaluation.

## B20 — multi-scale bidirectional depth registration

Protocol c106fb75. 28 CT tiles, 700 rows, 4 checks. 318319 informative pixels/scene. Adaptive accepted ramp 247227/247361 exact; coverage 77.71%. Two decoys accepted 106 vs 290 and 181 vs 958 for the 15x15 baseline. At step boundary 16692/40167 exact vs 16991/40167; -0.744pp rather than required +5pp, so gate failed. Real CT texture, simulated depth shifts and controls, not actual sheet identity or corrected mesh. Cycle5 has better seam coverage but more spurious matches. Do not deploy.

## B21 — real labeled Reader test and held-winding calibration

Frozen protocol 05c4b0e6. [Successful run 45858826467](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45858826467), [selection, result and predictions artifact](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45858826467/artifacts/15137042105). Public YoussefMoNader/ink-8um-pherc1447-surfaces revision 7e4d918712a8d8642a6fdeee45b80217c1e98317. Public refined human labels, certainty masks; 4 mixed + 2 blank crops per winding selected from labels before predictions; 384 input/central192 scored. 12 crops x 5 modes = 60 actual model forwards. Reverse convention fixed from public W058 recipe; opposite order is tested, not chosen by score. Native center band 04..20, alternate bands ±2 voxels. Source threshold calibrated to >=80% recall on other winding; apply unchanged on target.

| Held winding | Reverse recall/FPR (%) | Median3 recall/FPR (%) | Median3 delta recall/FPR pp |
|---|---:|---:|---:|
| W058 | 65.247/35.058 | 63.478/32.038 | -1.769/-3.020 |
| W060 | 98.938/74.012 | 98.379/71.819 | -0.559/-2.192 |

Median3 passes predefined gate (each target FPR reduction >=2pp, recall loss <=2pp). AP improves .37257->.38943 and .54774->.57000. Contrast/product fail held-winding gate; product improves within-target oracle FPR, **not transferable calibration**. BF16 fails the FP-reduction gate (that gate is not a numerical equivalence test); at threshold .5 only 223/442368 decisions differ; max score delta .00637364. Median CPU forward times: reverse .886244 s, BF16 .477047 s, shifts .880273/.878175 s. Median3 requires 3 forwards: verification candidate, not universal full-volume optimization. Blank FPR remains 28.50%/62.33% for median3 at transferred thresholds. Fixed .5 reverse recall only 26.05%/13.34%. Both W058/W060 belong to same manuscript; no independent cross-scroll transfer demonstrated, no new letters discovered. Need independent-manuscript certainty labels, training overlap audit, physically valid paired depth bands, and candidate-selection/recall validation before production. Don't tune or select thresholds on target labels.

## Reproduction and files

Local scripts have exact executed Git blob identities: paired.py b9fe7ec50f790648d5402bbe815491ddf264a3d7, transfer.py 6f73001b869de1fc39c37d68b14c444de47ff023, geometry.py c3c77279c448f6c08d4433b5cd9c467675cb73e3. Run from this folder with parent imports available:

```bash
PYTHONPATH=.. python paired.py /path/to/verified-inputs results.json
python transfer.py /path/to/verified-inputs transfer_results.json
PYTHONPATH=.. python geometry.py /path/to/verified-inputs geometry_results.json
```

local_receipts.json preserves full aggregate counts, raw JSON hashes and timing receipts. reader_receipt.json and reader_1447_receipt.json preserve all rows emitted in completed job logs even if Actions artifacts expire. Workflows document full public data/checkpoint acquisition. No raw private inputs or model weights committed.

Next priority: test median3 on independent labeled manuscripts with fixed depth conventions and predeclared split. Benchmark FP32 paired end-to-end stitching; consider BF16 as a screening option with FP32 confirmation only after independent recall evidence. Reject unsafe geometric correspondences rather than smooth blindly across sheet boundaries. 
