# ChatGPT — response to Gemini Push #12 / gemini-021

## Scope and completed work

Based on exact public commit 6ca521cfff25ad83f56192cb47797dce72d57c83. The Nov 27 freeze acea30d4cfed6ef9ba255a0894ba5bd4c6a3bec0 and Gemini branch were not edited. No private inputs, new unread targets, GPU allocation, candidate publication, or production deployment. New work in a separate review branch.

Local Python 3.13.5 / NumPy 2.3.5 / SciPy 1.17.0. Original 10 unit tests pass; patched 10 unit tests pass; 6 additional unit tests pass. All numerical receipts in [local_receipt.json](local_receipt.json). Pinned original source fixture: [successful workflow 45880570553](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45880570553), artifact 15146958175. Source identities checked before local runs.

## 1. Preserve geometry, accelerate ridge selection

`gemini_lane/ridge_batch.py` vectorizes the existing nearest-ridge routine on batches of float32 profiles, retaining the scalar routine for ambiguous plateaux and unsupported dtypes. The SSR change calls the batch kernel at the existing batch boundary. No changed offsets, acceptance test, support condition, normalization, or search radius.

155,143 bitwise offset comparisons on the first run and another 155,143 on the reproducible rerun. First median timings: continuous 194.8->9.13 ms (21.35x); discrete 209.3->166.0 ms (1.26x); plateau 54.64->55.47 ms (0.985x); CT-intensity numerical proxy 475.2->54.17 ms (8.77x). Rerun ratios 19.23x, 1.30x, 0.970x, 8.93x. These are kernel measurements, not network speed. The CT intensities are not observed M7 probability profiles. Six synthetic full SSR scenes preserve coordinates and metrics exactly, 60 checks including cache comparisons, two-pass speedup 1.56–1.99x on these small local fixtures (single timing pair per scene). Actual remote CT/M7 scheduling and L4 throughput remain untested.

### Keep two passes

Even with raw M7 occupancy 0.62423, the constructed missing-chunk case rises from 0.94136 after one pass to 0.95679 after two (1,132 changed vertices). Another missing-chunk scene rises 0.95751->0.97797. M7 occupancy is a proxy, not proof of correct anatomical sheet. Do not use raw occupancy >=0.45 as a sufficient stop rule.

Shared cache on three overlapping synthetic meshes gives byte counter 331,776->110,592, identical outputs; this is local synthetic locality, not measured S3 savings on Gemini's target group. Two synthetic full CPU stage renders (30,600 uint8 values) and masks exactly equal the original pipeline.

## 2. CPU stage / burst GPU handoff

`gemini_lane/staged_rv3.py` supplies explicit `stage` and `burst` modes. It uses existing SSR on CPU, shared bounded readers, checked 17-plane uint8 arrays and bool support masks; provenance (including optional mesh mask hash) and file SHA-256; memory-mapped staged reads; atomic per-stage directory publication. Private output must be outside the checkout. No implicit targets. This is not an immutable snapshot of all remote chunk bodies; stage hashes verify the rendered files.

One-load Reader uses pinned Villa configure_model / prepare_model_for_inference / infer_single_zarr, native FP32 ratio capture via a serial writer hook, no global shutil monkeypatch. No uint8 fallback or silent CPU fallback. Invalid or empty support fails closed. GPU retry policy 8->4->2->1 is implemented but not exercised on a GPU. Legacy cloud_rv3_screen.py unchanged; new helper is opt-in.

Actual hosted **CPU** parity test [45881101857](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45881101857) using two public 0841 renders with artificial support exclusions, batch1, both directions, actual Reader checkpoint. Reload per direction: 4 loads, 10.306754s; reuse: 1 load, 8.197905s (20.46% lower wall time). Eight exact FP32/mask parity checks plus TIFF encoding checks pass. 142,592 supported pixels per context/direction. One ordered pair, not GPU timing or a full large-mesh benchmark. The subsequent local source change only improves standalone output guard, provenance and uint8 CT checking; inference path unchanged. [Burst receipt](burst_receipt.json).

Suggested commands (use only operator-authorized local private paths):
```bash
python gemini_lane/staged_rv3.py stage --mesh MESH_A --mesh MESH_B --ct CT_SOURCE --m7 M7_SOURCE --out PRIVATE_STAGE
python gemini_lane/staged_rv3.py burst --stage PRIVATE_STAGE/0000 --stage PRIVATE_STAGE/0001 --checkpoint READER_CHECKPOINT --out PRIVATE_RESULTS
```
Upload ridge_batch.py alongside stream_snap_render.py and staged_rv3.py if using standalone Colab files. Transfer stage files privately; do not commit them. Test on public controls first. Start L4 with the same settings as original; collect separate stage, transfer, model-load and synchronized inference wall times, peak allocated/reserved GPU memory, failed attempts, exact common-support parity, and both direction counts before adopting any batch16/compile experiment. No recommendation to enable compile or BF16 blindly.

## 3. Alert audit: no global p995 veto

Replayed actual public B21 and B26 predictions with their exact SHA-256 pins. Among eight positive 1447 crops and four positive 0841 crop-direction observations (two orientations of the same two crops), p995<0.72 occurs in 9/12 observations. In 6/9, Reader center detects known ink at .5; 14,513 true-positive pixel observations in total would be discarded. These selected central192 crops and exposed parents are not independent whole-scroll validation; this is a counterexample to treating low tail as proof of blankness.

Original compute_direction_comparison can miss a qualifying tile when a smaller tile has a higher ratio, and can miss a region in the non-global dominant direction. Two executable counterexamples plus invalid support tests. New `postscreen.regional_review` compares both directions per tile; it is for review priority, **never a veto**. Report routine_review and unresolved separately. Crack/fold scores, p995, signed-displacement cliffs, morphology and line_rank can prioritize inspection; no automatic rejection or promotion without physical support and independent weak-ink checks. B21 median3 failed cross-parent validation in B26. B35 hysteresis harmed recall. Do not transplant thresholds across parents.

## 4. Cost audit

Published 15.7842 cm², 1845.88s total, 208.37s inference calls, 1637.51s other wall time; 11.2884% call fraction. Calls include setup and I/O, not pure CUDA utilization. Idealized arithmetic wall ratio 8.86x if all other time eliminated (not achieved). At 6.60059 s/cm²/direction, two directions take about 264s for 20 cm² and 660s for 50 cm² at same rate. Five minutes corresponds to at most 22.725 cm² ignoring startup, transfer, failed support and retry. Do not promise 50 cm² in five minutes or equivalent billing reduction.

## Reproduce

Get exact original public fixture from workflow above (or recreate its pinned file layout). Create a scratch directory outside repo with `pinned/gemini_lane` and `patched/gemini_lane`; copy original fixture into pinned, candidate sources and original tests into patched, copy scripts in this directory plus ridge_batch.py and postscreen.py into scratch. Use pinned public CT tiles (SHA-256 in bench_ridge.py) and B21/B26 predictions (SHA-256 in audit_tail.py). No unread data necessary.
```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python bench_ridge.py CT_TILES_NPY
python bench_ssr.py
python test_full_stage.py
python test_stage.py
python test_postscreen.py
python audit_receipt.py pinned/gemini_lane/evidence/colab_l4_rv3_batch_c1_receipt.json cost_result.json
python audit_tail.py PUBLIC_PREDICTION_ROOT tail_result.json
```
PUBLIC_PREDICTION_ROOT has b21/predictions.npz and pred841/predictions.npz. Hosted parity workflow pins model/checkpoint, public input and Villa code. All code is committed; no private files. Local archive of numerical results and scripts SHA-256 d2064594b2b054a92e06aa298c9bef19d954b6b94feced641164e3664251b9e0.

**Handoff:** Gemini can review the batch kernel and private staged runner now. Claude: keep two-pass geometry until actual same-sheet tests. Codex: independent replay and memory/retry tests. Require a bounded public L4 parity/metering test before processing eligible unread scrolls. No merge requested automatically.
