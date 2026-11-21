# Actual tile-placement control on published PHerc. 306B

This module prepares a two-grid experiment on an already published positive
control. It does not search an unread scroll, establish a reading, or estimate
a false-positive rate. Generated CT inputs, model weights, score arrays and
review packets must stay outside Git. Both executables enforce that boundary.

## Why this experiment

**From a source:** Codex asked whether line proposal ranks change when the tile
grid changes on the *same physical field and mask*
([codex-010](../../collab/codex-010.md)). A translation of an existing score
image cannot answer that question. The published A-hn recipe runs a network on
each input tile, then blends probabilities with a Gaussian and divides by the
weight sum
([source](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/code/ink_v8.py)).

**Method decision:** Run the same A-hn seed-0 checkpoint twice on different
native input patches. Phase A uses the author's global native grid origin
`(y=115,x=694)`, stride 21. Phase B adds `(10,10)` to that origin, the integer
floor of half the stride. All shared settings remain fixed. `protocol.json`
was written before these new model predictions; the old published images and
the earlier line-control outcomes were already known. This is a prospective
comparison conditional on that prior knowledge, not an unseen-data benchmark.

**From sources:** The finalist uses identity TTA, unlike the generic base-v8
script's mirror4 default. The source smoke test describes the same 24 planes,
clip value, grid, and Gaussian, and distinguishes its FP32 CPU calculation
from the FP16 shipped maps
([finalist metadata](https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/blob/f435998b55269762942a979052044309b29650ac/results/member_maps/fwd_hn4_s0.json),
[smoke implementation](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/code/smoke_v7.py)).
Our two phases use identity TTA and FP32, with TF32 disabled. These are
individual **A-hn seed-0 model scores**, not the previous TRACE × mean-A-hn
product. The old ranker thresholds are uncalibrated for this representation.
Do not tune them if this experiment returns no proposals.

## Frozen physical field

**From a source:** The source geometry and the two coarse row annotations come
from the already frozen
[public control inventory](https://github.com/JorisViaudQuantAI/Scrollprize/blob/faba98beebe3fdc3fe6948a5b3e2ce1ea926831a/firstletters-colab/line_controls/inventory.json).
The labels are row locations, not ink-pixel truth, and have about one native
pixel of source-display/rounding uncertainty.

**Method decision and exact geometry calculation:** Choose a 10 mm horizontal
interval centred on the published claim-box midpoint, extending vertically to
include both source row centres and at least 3.5 mm of outward support on each
side. This gives a useful interval longer than the line helper's 8 mm minimum
chain length, without paying for the entire released context. It includes a
10 mm section of each published row, not the entire approximately 38 mm source
annotation. No new score map selects the field.

| Quantity | Frozen value |
|---|---|
| Native sampling | 8.64 µm/pixel in both axes |
| Prepared context box, half-open `y0,x0,y1,x1` | `[1268,3356,2618,4514]` |
| Native render output box, half-open `y0,x0,y1,x1` | `[1549,4050,2899,5208]` |
| Output dimensions | 1,350 × 1,158 pixels; 11.664 × 10.00512 mm |
| Row centres in output | y=406 and y=943 |
| Output orientation | Native crop flipped vertically, matching the fixed reading context |
| Real CT halo required by both grids | `[1490,3991,2961,5262]` |
| Tiles before CT admission | Phase A: 3,886; phase B: 3,944 |
| Contributions before CT admission | Every field pixel: 9–16 per phase |
| Source depth | Native render planes 8 through 31, ascending stored order |
| Tile admission | At least 1,024 positive layer-20 pixels in a 64 × 64 tile |
| Input normalization | Clip uint8 to 200, convert to FP32, multiply by 1/255 |
| Blend | Normal-CDF Gaussian bins over ±3 sigma, normalized to maximum 1; FP32 probability-weight sum divided by weight sum |

**Method decision:** The prepared input contains every complete real CT tile
covering the field, including the halo. Missing chunks fail preparation;
padding is never used to manufacture coverage. The scored mask is positive
source layer 20 intersected with positive model weight support in both phases.
Actual admitted counts, contributor minima, and common-mask fractions are
reported after execution. Before then, the table's counts are geometry only.
Verified published render bytes do not certify completeness of the underlying
original CT acquisition.

## Two different execution scales

### Small CPU numerical check

**Method decision:** Use the released 169 × 169 × 24 smoke crop. Evaluate exactly
two 64 × 64 tiles, with local origins `(0,0)` and `(10,10)`. Compare the same
physical 54 × 54 pixels: first output `[10:64,10:64]` against second output
`[:54,:54]`. No output is translated or resampled. With one contributor per
phase, a normalized Gaussian would cancel; the comparison directly uses the
two probabilities.

The common side length is only **0.46656 mm**, far shorter than an 8 mm line.
This check can show tile-context sensitivity and verify real forward execution.
It **cannot** measure full-grid line-rank stability. The fixed endpoints are
mean/max absolute difference, quantiles at 0/25/50/75/95/99/100 percent,
Pearson correlation when defined, and binary disagreement at 0.25/0.5/0.75.
There is no accuracy pass threshold, AUC, letter verdict or false-positive rate.
The correlated pixels are not independent samples.

Run from the repository root, with a Python environment containing `numpy`,
`torch`, `safetensors` and `huggingface_hub`:

```bash
python firstletters-colab/tile_phase/prepare.py --output-dir /tmp/scrollprize-phase-input
python firstletters-colab/tile_phase/run.py --mode micro --inputs /tmp/scrollprize-phase-input --output-dir /tmp/scrollprize-tile-microphase-v1
```

Preparation downloads four checksum-pinned public files, including a
333,532,180-byte seed-0 checkpoint. It does not download redundant base weights:
the pinned model constructor is instantiated, then the complete finalist state
dict must load with `strict=True`. The output directory must be new. The run
records input hashes, code hashes, protocol hash, runtime versions, and exact
geometry before either model prediction. CPU inference uses four threads and
batch size one. A 180-second Unix alarm bounds the inference attempt; Python
may deliver that alarm only after an in-flight native operation returns.
Model loading is timed separately. No new experimental outcome is claimed by
the protocol alone; the runtime receipt is the record of an actual execution.

### Measured CPU result

**Measured here:** The frozen two-forward check completed on 2026-11-21. The
[complete sanitized receipt](evidence/microphase-v1.json) records the exact
runtime, script/protocol/input/output hashes, and all prespecified statistics.
The protocol SHA-256 is
`b329bc7bd44eed17946827a1aa7a2c365fa4f5433177eaa58c089ffaf57cf3d5`.

| Endpoint | Observed value |
|---|---:|
| Real model forward calls | 2 |
| Elapsed inference-loop time | 13.5424 s |
| Process CPU time during inference | 50.2929 s, across four threads |
| Separately measured model-load time | 2.9743 s |
| Same-physical-pixel comparison | 2,916 pixels, all with positive source CT |
| Mean absolute score difference | 0.0188780 |
| Median absolute score difference | 0.0166469 |
| 95th percentile absolute difference | 0.0440920 |
| Maximum absolute score difference | 0.0957395 |
| Pearson correlation | 0.261472 |
| Disagreement at threshold 0.25 | 39/2,916 = 1.33745% |
| Disagreement at thresholds 0.5 and 0.75 | 0/2,916 at each threshold |

**My inference:** The model output on this particular small patch depends on
the input tile context. The result does not diagnose text quality. Correlation
on such a small patch can be sensitive to limited score variation. Each phase
has one tile contributor: **there is no Gaussian multi-tile averaging in this
micro check**. The full field has 9–16 geometrical contributors per pixel before
CT admission and could behave differently after averaging. Neither low
micro-patch correlation nor zero disagreement at a high threshold establishes
line-rank failure or success. Do not revise the frozen field, thresholds or
primary endpoints in response to these numbers.

**Source/code audit, without further inference:** The pinned upstream
`ink_v8.py` and this runner both use `.float().mul_(1 / 255.)`. Shorthand
descriptions of this as “divide by 255” describe the real-valued normalization,
not necessarily identical floating-point instructions. For all 201 possible
clipped input intensities, the two actual expressions are bitwise equal in the
recorded CPU runtime. Explicit floating-point division instead differs for
72 values, with maximum absolute difference `5.960464477539063e-08`; that
division is not used in this experiment. The
[arithmetic audit](evidence/normalization-arithmetic.json) records the source
hash, runtime and reproduction expression. No additional model calls were
made for this check.

### Full same-field comparison: CUDA only

**Recorded pre-field correction:** Independent review found that the future
`pair.json` labelled relative phase offsets as absolute native tile origins.
The manifest now records native origins `[115,694]` and `[125,704]`, with
`[0,0]` and `[10,10]` retained separately as phase offsets. The
[amendment receipt](evidence/pre-field-origin-metadata-amendment.json) records
old/new runner hashes. The [exact runner used for the completed micro
experiment](evidence/run-at-microphase-v1.py.txt) is archived and matches its
execution receipt. Tile extraction, model computation, stitching, field,
protocol and endpoints did not change; no extra forward was run for this
metadata correction. Full-field inference had not started.

**From a source and my inference:** The earlier public CPU replay reports
371.6 under `cpu_seconds` for 36 seed-0 tiles
([receipt](https://github.com/JorisViaudQuantAI/Scrollprize/blob/codex/public-306b-cpu-replay/public-controls/pherc0306b-replay.json)).
The upstream implementation calculates that value with `time.time()`, so it is
elapsed wall time, not process CPU time. Linear extrapolation to 7,830 tiles is
approximately 22.45 elapsed hours on that recorded machine before allowing for
different hardware and CT admission. That is a budget warning, not a runtime
measurement of this experiment. I do not know the FP32 T4 runtime yet.

Full-field mode refuses to run without CUDA; it has no CPU fallback. In a CUDA
environment, add `numcodecs` for public render-chunk decompression:

```bash
python firstletters-colab/tile_phase/prepare.py --output-dir /tmp/scrollprize-phase-input --with-field
python firstletters-colab/tile_phase/run.py --mode field --inputs /tmp/scrollprize-phase-input --output-dir /tmp/scrollprize-tile-field-v1 --maximum-seconds 900
```

Field preparation requests 143 complete published render chunks and verifies
their hashes against the pinned checksum manifest. It builds only the needed
24-plane halo. **Measured preparation:** all 143 required chunks were fetched
and checksum verified, totalling 96,017,622 compressed source bytes. The
prepared input is 39,271,157 bytes; its hash and per-chunk receipts are in
[field-input-preparation.json](evidence/field-input-preparation.json).
No full-field model forward was run during preparation. CPU refusal was
verified before an output directory or model execution was created.
The [input-only admission record](evidence/field-input-admission.json) retains
all 7,830 tiles under the fixed central-plane CT rule. It identifies 1,563,295
positive central-plane pixels among 1,563,300 field pixels. This is planned
scoring coverage conditional on successful model execution, not a completed
prediction mask or an acquisition-validity certificate.
The runner checkpoints every 64 model calls and before an
estimated budget overrun. Exit 3 means an incomplete checkpoint, never a
completed result. Resume the same command with `--resume`. Runtime, input,
protocol and code signatures must still match. The loop budget includes
stitch/checkpoint work and can overrun by an in-flight forward or filesystem
write; it is not a process-killing scheduler. The first completed run must
report actual device, memory/runtime behaviour and CT coverage. FP32 T4
execution and resumption remain to be verified on the GPU.

When complete, the runner produces `phase0.npy`, `phase10.npy`, their separate
masks, `common_valid.npy`, and `pair.json`. Send that manifest to the fixed
row-context comparator:

```bash
python firstletters-colab/row_context/diagnostic.py --pair-manifest /tmp/scrollprize-tile-field-v1/pair.json --output-dir /tmp/scrollprize-tile-field-review-v1
```

**Method decision:** Primary endpoints are each published row's coverage by the
top 1/3/5/10 proposals, centreline and coarse box separately, and the first
individual proposal rank covering at least half the observed centreline. Report
not attained explicitly. Score-difference statistics are secondary. Show both
fixed row contexts and the common physical field regardless of rank. Report
how much of each complete source row was observed; uncomputed portions remain
unknown. No inference about text absence follows from zero proposals. These
are descriptive results on two known positive rows, not a generalization test.

The field matches the published phase-0 tile geometry, model inputs and
weighting rule. It does not claim bitwise identity with the author's full map:
that map used FP16 and the upstream code grouped accumulations into parts.
Our field streaming order is fixed and FP32 for both phases. The two-forward
micro check also does not replace the author's 36-tile, stitched-reference
smoke test or Codex's already completed replay.

## Pins and verification

| Source | Revision or SHA-256 |
|---|---|
| Public submission source | `feabff6c54f95bd8e52a90e0a7a76881ee67d4cc` |
| Dataset revision | `f435998b55269762942a979052044309b29650ac` |
| Finalist model revision | `e57c4d541c239e06450a98c1cf2015d3de8d887e` |
| Seed-0 checkpoint | `9a2812843924c50cafaa2b598f62c92ceca9f0d1f5d05b7bbf25764814ec7009` |
| Model class source | `0656550dc10556ad09f5b6c21b36c97aa97a5696f71ef4b384eada124095c109` |
| Model config | `30f46a55d61dc3f4e9bf7e9ca261c1486575f1ce05a27f28d09f10ee57597018` |
| Published smoke input | `4a81e193e253e8b6234eaa4c1b274e33c08b214183951ed8e573c9ae3f8f639d` |
| Dataset checksum manifest | `8c114c6359efd95c30508c7b9731ebfab3ee799bf2ec6dcf84cf92e841eb07cf` |

`python -m unittest discover -s firstletters-colab/tile_phase -p 'test_*.py' -v`
checks global-grid equivalence, real halo containment, exact micro physical
alignment, weighted stitching of a synthetic coordinate ramp, and exclusion of
unknown pixels. These are numerical-plumbing tests; they provide no ink-model
accuracy evidence.

**From sources:** The public code carries an MIT licence and the released data
carry CC BY-NC 4.0
([code licence](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/code/LICENSE),
[data licence](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/LICENSE-DATA)).
The Gaussian construction and inference convention are adapted from that
code; see `THIRD_PARTY_NOTICE`. This module writes new predictions and changed
grid coordinates, not an unmodified published map. It makes no assertion that
the released weights may be used for a new prize entry under every applicable
licence or contest rule.
