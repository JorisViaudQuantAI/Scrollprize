# One depth-bias counterfactual, after exact original replay

**Status: prospective protocol and implementation, not an executed result.**
Freeze all four files before inspecting the running published-glyph experiment's
images, archive arrays, or recorded scalar. There is one counterfactual only:
replace `depth_coordinate_scale` by zero. Keep the raw attention scores, 3D logits,
source support, precision, grid, depth range and assembly unchanged. Both physical
orders and all 225 patches per order remain in the analysis.

## What this tests

**From the [pinned Hecate source](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/hecate.py):**
the projection adds a scalar times normalized depth to learned attention scores,
then averages volumetric **logits** under a depth softmax. Its fine model excludes
the two outer input planes and operates on a coarser internal XY grid before
interpolation. The [model card](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/README.md)
describes intended middle-sheet localization, possible fibre/neighboring-sheet
responses, and Paris 4 teacher exposure. A known Paris 4 control is not an
independent generalization test.

**My derivation:** with fixed raw attention `a`, logits `l`, and normalized depth
`u=(arange(62)-31)/31`, the weights are `softmax(a+alpha*u)`. The derivative of
the weighted logit with respect to `alpha` is the weighted covariance of `l`
and `u`. The scalar can enhance or suppress signal depending on that association.
Its removal has no guaranteed benefit. If the recorded scalar is zero, preserve
the resulting no-op rather than adding another intervention.

**My inference:** diffuse raw attention can separately dilute narrow positive
responses among negative depths. This experiment does not isolate or rule out
that mechanism. Entropy and effective-plane count are descriptive diagnostics,
not justification for trying a different temperature, uniform attention, maxima,
depth windows, scalar search, or a preferred order.

## Strict replay first

The input is exactly the [separately frozen published-glyph control](../public_glyph_control/README.md).
The runner verifies its completed receipt and every consumed archive/input hash.
It derives nonempty CT columns from the original 65-plane source union, checks
them against the saved masks, and validates the recorded scalar bytes and depth
mapping. No checkpoint is loaded and no CNN is executed.

It then reconstructs **all 450 original patch probabilities bit for bit**, and
both complete 2048 × 2048 original Q8 contexts byte for byte. Both baseline orders
must finish successfully before **any** zero-bias projection is calculated.
A mismatch makes the diagnostic unassessable, with no tolerance relaxation.
Q8 agreement alone is inadequate because serialization can hide float differences.

The exact path preserves masked fill `-1e4`, support multiplication and weight
renormalization, weighted-logit sum, sigmoid, bilinear interpolation, the epsilon
clamp, `logit`, and final float32 sigmoid. Omitting the final round trip would
change the reproduction contract. Unsupported columns remain zero after assembly.

**From the frozen fixture:** source feature planes are 23 through 84; reverse
feature planes are 85 down to 24, at native 2.4 µm spacing. The physical slabs
differ by one source plane. The archive's depth order must not be reversed again.
Mean depth is reported relative to source plane 54, the stack midpoint; that
reference is not an independently established papyrus surface or ink depth.

## Outputs and interpretation

Both baseline and zero-bias maps use the already-frozen
[prediction-precision display](../prediction_precision/README.md): clip to float32
epsilon, take logit, and map globally with `clip((logit+8)/24,0,1)`. There is no
contrast fitting, crop search, spatial processing, glyph score, or automatic
preference. The original source files are never overwritten.

For each complete fixed core, report mean signed, mean absolute and maximum
absolute probability change, changed Q8 pixels, and upward/downward crossings
of **strict P > 0.5**. The 0.5 boundary is descriptive; it does not identify ink
or background. Results are provided for the whole core and its nonempty CT
columns, which likewise are not a tissue/ink answer key.

Attention diagnostics report the population count and fixed 5th, 50th and 95th
percentiles of entropy, `exp(entropy)` effective-plane count, and weighted mean
physical depth. They use all supported internal positions in all fixed tiles;
overlap repeats physical positions. For these diagnostics alone, weights are
normalized in float64 to avoid numerical normalization artefacts. That operation
does not feed the prediction. These are model-weighting summaries, not independent
samples, ink localization truth, uncertainty intervals, or significance tests.

**My inference:** a more complete reference-consistent glyph could support the
narrow claim that this scalar suppressed useful archived signal on this known
control. It would not validate every 3D response, identify the intended physical
sheet, or establish a better detector for unread scrolls. Removing a learned
preference can expose fibres or neighbouring-sheet responses. Brightness and
larger connected blobs are therefore not sufficient evidence of improvement.
Retain unchanged, worsened and ambiguous outcomes. No earlier gate changes.

## Execution and resource contract

The script checks the new freeze before opening source results, parameters or
arrays. It requires the original Torch runtime and uses four CPU threads.
There are exactly 450 original projection replays followed by 450 zero-bias
projections, with zero CNN forwards, downloads, new CT requests or training.
The wall limit is **300 seconds** and soft RSS limit **1.5 GiB**. Native operations
may defer a Python timeout; RSS is checked between bounded groups of patches.
Arrays and images go to a fresh directory outside Git, with source hashes
rechecked at the end. Root reviews and decides execution after the original run.

```bash
python firstletters-colab/depth_bias_control/test_run.py

python firstletters-colab/depth_bias_control/run.py \
  --freeze-reference FULL_PUBLIC_FREEZE_SHA \
  --input-dir /tmp/completed-public-glyph-control-run \
  --output-dir /tmp/depth-bias-control-v1
```

No image or parameter from the running glyph experiment was inspected to choose
this intervention. The separate source-only planning memo preceded implementation.
