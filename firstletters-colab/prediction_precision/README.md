# Prediction precision and one fixed logit display

**Status: a new analysis prepared for public freeze, not an executed result.**

**From the team's completed Hann comparison:** the PHerc. 0814 float map retains
discrimination that the actual uint8 export loses at the strict background
operating point. This observation motivates the new comparison. The two
controls have already been examined; this is an **exploratory serialization
analysis**, not an untouched validation set. The protocol is frozen before the
new uint16 calculation and before either forthcoming whole-glyph image is
inspected. It does not modify an earlier experiment or its decision.

## Fixed uint16 comparison

**Design choice:** retain the actual float and uint8 outputs. Quantize the fine
probability map to uint16 with nearest-even rounding of `float64(P) * 65535`,
after clipping to the probability interval. Using float64 makes the product
exact for the retained float32 inputs. This is done **before** common-grid
reduction. Each 4 × 4 uint16 block is summed in uint64, divided once by
`16 * 65535` in float64, and cast to float32. Equal integer sums remain tied.
The actual uint8 baseline retains its original export operation; it is not
replaced with a retrospectively modified quantizer.

The primary assembly is **Hann**, on the same PHerc. 0841 and PHerc. 0814 fields.
Central-crop and uniform-overlap assemblies are diagnostic only. All three use
the same labels and fully supported cells as the original run. Their AP, AUC,
recall, actual background counts, strict cutoff, and ties are retained. The
runner also counts saturated fine pixels and supported background/ink common
cells at the maximum code.

The descriptive cutoff remains one whole-field background order statistic per
representation: descending background score at `floor(Nbackground / 20)`, with
strict `score > cutoff` and no tie-breaking. Labels determine these cutoffs;
they are not thresholds for an unread scroll. The same nominal background
budget can produce different realised FPR when quantization introduces ties.

**Predeclared serialization criterion:** each Hann field's uint16 recall must
be within **0.01 absolute** of its float recall, and AP within **0.005 absolute**.
Both fields must satisfy both conditions. The criterion is fidelity to the
retained float scores, not a new ink-detection gate. Diagnostic assemblies cannot
replace Hann after results are known. A failure is retained without trying
other bit depths or selecting a more favourable cutoff.

## One global display transform

**Design choice:** for every complete fixed core, use

\[
L=\log(p)-\log(1-p),\qquad
p=\operatorname{clip}(P,\epsilon_{32},1-\epsilon_{32}),\qquad
g=\operatorname{clip}((L+8)/24,0,1).
\]

The calculation uses float64; the display array is stored as float32 and the
PNG uses nearest-even rounding of `255*g`. Every image uses those same constants.
There is no fitted contrast range, smoothing, morphology, sharpening, crop
search, stroke drawing, or expected-letter input. Display values do not enter
the probability metrics. A clearer rendering would expose existing numerical
differences, not establish a new detector, improve calibration, or supply ink
that was absent from the prediction.

Near zero and one, a small probability difference can correspond to a large
logit difference. Consequently a BF16 probability-parity check would not bound
logit-display error. The actual source precision is retained in the result;
this analysis does not make an independent precision claim.

## Published whole-glyph mode

After the separately frozen Paris 4 control finishes **both depth orders**, the
same fixed display can be applied to both. The runner reconstructs the canonical
float Hann map from the actual retained full-patch probabilities and exact
saved nonempty-column support. It uses the released float32 window and the
same row-major accumulation order. The **entire 2048 × 2048 context must reproduce
every original uint8 pixel exactly** before a derived image is accepted.

The fixed 1920 × 1920 core is then saved as float, uint16, the unchanged original
uint8 pixel copy, and the global logit display. Both source and reverse orders
remain in the result. No pixel answer key, automatic reading, preferred-order
selection, unread target, or source modification is introduced. The original
fixture's one-layer physical depth difference and training-exposure limitations
continue to apply. See its [protocol](../public_glyph_control/protocol.json) and
[README](../public_glyph_control/README.md).

## Reproduction and limits

The runner verifies all four files here against a full public freeze commit
**before opening any retained result or score array**. Consumed artifacts are
checked against their original receipts and rechecked afterwards. Every new
array and image is saved in a fresh directory outside Git; original outputs
are never overwritten.

Each invocation has a **60-second wall-time cap**, **1 GiB soft RSS cap**, zero
network requests, zero new CT acquisition, zero detector forwards, and zero
training. Whole-glyph mode imports Torch only to construct the exact window;
it does not load a checkpoint or instantiate a detector. Native library calls
may defer a Python timeout, and RSS is checked between processing steps.

```bash
python firstletters-colab/prediction_precision/test_run.py

python firstletters-colab/prediction_precision/run.py \
  --freeze-reference FULL_PUBLIC_FREEZE_SHA \
  --mode controls \
  --input-dir /tmp/completed-hann-control-run \
  --output-dir /tmp/prediction-precision-controls-v1

python firstletters-colab/prediction_precision/run.py \
  --freeze-reference FULL_PUBLIC_FREEZE_SHA \
  --mode glyph \
  --input-dir /tmp/completed-public-glyph-control-run \
  --output-dir /tmp/prediction-precision-glyph-v1
```

Use the original Torch runtime for exact whole-glyph replay. The
[released Hecate predictor](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/hecate.py)
defines the baseline probability assembly and uint8 export. The retained
[metric implementation](../hecate_blending/metrics.py) defines the control
operating point; neither is changed by this analysis.
