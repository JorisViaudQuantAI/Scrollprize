# Studio release replay on public PHerc. 0841

**Purpose:** give Claude Code and Codex a small, reproducible execution check for
the detector used in the published PHerc. 211 submission. This folder contains
real inference measurements on an author's released PHerc. 0841 validation data.
It contains no unread-scroll search output. It is not an independent accuracy
benchmark and does not establish that the model will transfer to the atlas.

## Sources and exact input

**From a source.** The pinned upstream revision is
[`bf66f753176792a966bca17822629fb8821f20c8`](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/tree/bf66f753176792a966bca17822629fb8821f20c8).
The final [soft-loss release](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-softloss-20261023/README.md)
identifies checkpoint SHA-256
`96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90`.
The scripts verify all **893,443,724 bytes** against that hash. They load the
unchanged released network and apply float32 `clip(raw,0,200)/255` exactly once.
They do not retrain, resample, interpolate missing planes or normalize each crop.

**From a source.** The fixture is the author's
`841-expanded-layer-03-approved-r000001` in
[`paranke/vesuvius-814-841-9b-studio-training-data`](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/tree/ad919d288ede1c4daf92725288bb6fe29c558591/donor-studio-resunet-scratch-20260926/prepared/841-expanded-layer-03-approved-r000001),
pinned to revision `ad919d288ede1c4daf92725288bb6fe29c558591`. The downloaded
`ct-tiles.npy` contains 28 prepared tiles, each `512×512×16`, already in the
author's model order. Its matching row in the pinned
[data record](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/donor-studio-resunet-scratch-20260926/data.json)
records native low-resolution mesh-derived affine sampling, XY pitch **9.36 µm**,
and depth pitch **9.366 µm**. This is a recipe attribution; array shape alone does
not establish geometry. We do not infer a mesh-center index for these prepared
inputs. No additional resampling is performed.

**From the measured artifacts.** Four source files have fixed size and SHA-256
pins in both runners: CT tiles, categorical labels, sample coordinates and tile
coordinates. Their total is **124,686,677 bytes**. Unknown label `0` is excluded;
`1` means background and `2` means ink. Input retrieval is content-verified, but
the underlying volumetric acquisition coverage has not been independently
re-audited. Full source-byte checksums are in the saved protocols. Source CT,
weights and prediction arrays are downloaded outside this repository; only
compact protocols and measured results are committed.

## First diagnostic: three cells, two depth orders

**From the measured artifacts.** Before loading the model,
`run_public_control.py` selects three nonoverlapping 128-pixel cells using labels
and fixed rules: mixed, sparse and labelled background. It takes the
lexicographically first eligible coordinate in each class. Those cells happen
to be neighbors. All selection details and input hashes are in
[the frozen protocol](results/three_cell_protocol.json). The model runs each cell
in stored order and with exactly those same 16 planes reversed. This is an
order-only comparison, not a second face or a newly placed slab.

Central 96-pixel results, scored only on known labels:

| Cell | Ink / background pixels | Stored AP / AUC | Reversed AP / AUC | Stored TP / FP at 0.5 | Reversed TP / FP at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed | 1,670 / 4,305 | 0.6065 / 0.8232 | 0.3211 / 0.6466 | 0 / 0 | 0 / 183 |
| Sparse | 22 / 668 | 0.6585 / 0.9833 | 0.01684 / 0.0000 | 0 / 0 | 0 / 0 |
| Background | 0 / 2,560 | Undefined | Undefined | 0 / 0 | 0 / 0 |

Source: [complete results](results/three_cell_results.json). AP/AUC are undefined
when a class is absent. In particular, the central 32-pixel cores of these three
cells contain no labelled ink, so those cores cannot provide an ink-ranking
test. Unknown pixels are not treated as negatives to manufacture a denominator.

**From the measured artifacts.** Six forwards took **26.52 seconds** total on
this host, using CPU FP32, batch one and four Torch threads. Individual forwards
took **3.77–5.21 seconds**; process peak RSS was **1,987,952 KiB**. Model loading
took **1.16 seconds** after downloads and hash checks. These are local CPU
measurements, not T4 throughput or memory measurements.

**My inference.** The stored order gives better ranking on the two cells with
ink labels, but all 1,692 labelled ink pixels remain below 0.5 in both trials.
This tiny diagnostic does not establish a successful reading or a useful atlas
threshold. Reversal can introduce false positives: the reversed mixed cell
flags 183 labelled background pixels at 0.5. Do not promote a reversal rule into
a candidate veto, and do not retune a threshold on these cells and call that
independent validation.

## Follow-up: every labelled pixel in the author's holdout

**From the measured artifacts.** The full labeled-pixel replay succeeded:
**AP 0.7536042209**, which rounds to the release's **0.7536**, and **ROC AUC
0.8250587480**. It includes exactly **15,791 ink** and **29,558 background**
pixels, from **53** central32 cores. The other **1,281** of the author's 1,334
published coordinates have unknown-only central32 cores and cannot change
these AP/AUC statistics. The released network uses GroupNorm rather than
batch statistics, so omitting those empty-label examples does not change
another example's normalization.

| Full author-holdout replay | Measured result |
| --- | ---: |
| Average precision | 0.7536042209 |
| ROC AUC | 0.8250587480 |
| Ink recall at fixed 0.5 | 10,718 / 15,791 = 67.87% |
| Labelled-background false-positive rate at 0.5 | 5,765 / 29,558 = 19.50% |
| Direct forwards, stored order | 53 |
| Total inference time, CPU FP32, four threads | 222.54 seconds |
| Model loading, after verification | 1.80 seconds |
| Process peak RSS | 1,981,860 KiB (about 1.89 GiB) |

Sources: [follow-up protocol](results/author_holdout_protocol.json) and
[complete results](results/author_holdout_results.json). Before loading the
model, this follow-up recorded an engineering tolerance of 0.02 absolute AP
against the published rounded value; the observed result matches at all four
reported decimal places. The tolerance is ours, not the author's validated
criterion. Exact CPU/CUDA bit identity is not claimed, and a difference from
a four-decimal reference is not a measurement of the original full-precision
value. No model, threshold, depth order or data was tuned after the initial
three-cell diagnostic.

**My inference.** The earlier tiny cells were unrepresentative of the full
holdout's fixed-threshold sensitivity. Retain both outcomes. The full replay
supports the checkpoint/input/scoring integration; the 19.50% labelled-background
false-positive rate at 0.5 also makes it clear why raw thresholding is not a
reading or a candidate-validation procedure.

**From a source.** The published evaluation uses the central **32×32** of
128-pixel windows, pools known labels and reports **15,791 ink** and **29,558
background** pixels. The final soft-loss release reports AP **0.7536**. See the
[upstream evaluator](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/donor-studio-resunet-scratch-20260926/verify_holdout.py)
for crop/metric conventions and the
[soft-loss result](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-softloss-20261023/README.md)
for the checkpoint-specific reference value. The evaluator's epoch-nine headline
is for different weights; it must not be used as this release's target.

**My inference, limitation.** This holdout has been repeatedly examined during
model development; PHerc. 0841 is also a training donor. Successful replay checks
our implementation and the released input/model combination. It does not give
an independent estimate of another scroll's sensitivity, confirm faint ink or
replace the larger published 343/306B controls requested by Claude. The follow-up
was chosen after the three-cell diagnostic, and is explicitly labelled that way.

## Reproduce on CPU

Use Python 3.12 with NumPy 2.4.6, safetensors 0.8.0 and CPU PyTorch 2.13.0. The
measured environment used `torch==2.13.0+cpu`. Installation can use a dedicated
virtual environment; no CUDA training stack or 96 GB training machine is needed
for these small inference checks.

```bash
git clone https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt studio-source
git -C studio-source checkout bf66f753176792a966bca17822629fb8821f20c8

python firstletters-colab/studio_control/run_public_control.py \
  --source-repo studio-source \
  --checkpoint /tmp/studio-control/studio-softloss-20261023-ema.safetensors \
  --cache /tmp/studio-control/source \
  --output-dir /tmp/studio-control/three-cells \
  --threads 4

python firstletters-colab/studio_control/run_author_holdout.py \
  --source-repo studio-source \
  --checkpoint /tmp/studio-control/studio-softloss-20261023-ema.safetensors \
  --cache /tmp/studio-control/source \
  --output-dir /tmp/studio-control/author-holdout \
  --threads 4
```

The runners fetch missing fixed inputs, reject changed hashes, require the pinned
upstream commit and unchanged imported model/inference files, and refuse to reuse
an existing output directory. The protocol is written before model loading.
Runtime fields and floating-point last bits can vary across hardware; no
cross-hardware bit identity is claimed. The prepared input format can be checked
separately with [verify_prepared.py](../studio_inputs/verify_prepared.py).

## Handoff to Claude's T4

**From a source.** Upstream's
[inference CLI](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/hf/inference.py)
defaults to BF16 and explicitly rejects CUDA devices without BF16 support.
Use **`--device cuda --batch-size 1 --amp fp32`** for the first T4 check. FP16 is
not an exposed precision option in this pinned release; silently substituting it
would introduce a separate numerical change. Record actual T4 allocated and
reserved peak memory and timing before estimating a survey budget.

For a new surface, use [the explicit native-input adapter](../studio_inputs/README.md)
with verified mesh-plane and normal conventions. Keep same-band reversal separate
from whole-stack reversal and re-extraction. For a full map, the release recommends
stride 48 with Hann blending; the submission used stride 64. These runners test
direct crop forwards, not full-canvas blending, reflection at canvas edges or
empty-column masking. A production-wrapper test remains necessary on the T4.
