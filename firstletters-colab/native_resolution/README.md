# Native fine detail versus matched degradation

**Completed experiment:** see [RESULTS.md](RESULTS.md) for all 388 executed CPU
forwards, the positive within-control information signal, and the failed
prespecified two-field advancement decision. The frozen runner, protocol and
selection below are unchanged.

This is a real-model experiment designed to test whether a fine-resolution
pipeline recovers coherent ink structure lost by the current coarse pipeline.
At protocol preparation, no new model or CT outcome is known.

The [protocol](protocol.json), [runner](run.py) and [label-only selection](selection.json)
must be published before execution. The runner verifies their exact bytes against
a full Git commit. All raw arrays and score maps are written outside Git.

## Why this comparison

**From a source:** the released
[Hecate checkpoint family](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/README.md)
includes a 2.4 µm model with a 64 × 256 × 256 input and a 9.6 µm model with a
16 × 64 × 64 input. The finer model has changed 3D supervision and learned
depth aggregation relative to the original canonical detector. It divides
uint8 input by 200 without clipping; the coarse model divides by 255.
The fine model excludes its two outer depth planes and interpolates its
coarser internal prediction grid. Its output raster is not proof of 2.4 µm
physical resolving power.

**My inference:** this is a useful next test because it changes the image
information presented to a real detector, while retaining a controlled
degradation arm. It does not repeat the tiny-core frozen-head or context-ablation
experiments in [ChatGPT 012](../../collab/chatgpt-012.md) and
[020](../../collab/chatgpt-020.md). Native canonical 2 µm inference was already
tried on some PHerc. 1203 regions; this is not the team's first fine-scan test.

## Two coherent public fields

The same published PHerc. 0841 and PHerc. 0814 donor products used by the earlier
probe supply the inputs and model-assisted categorical labels. Their exact
source metadata, code and label hashes are pinned through the existing
[donor manifest](../transversal_lab/hecate_donor_manifest.json).

A label-only lexicographic rule selects one 256 × 256 common-grid core per
donor, with complete 288 × 288 context, at 9.6 µm. Each core spans 2.4576 mm;
the complete input context spans 2.7648 mm. The rule requires at least 95%
known labels and 10–60% ink among known labels, with complete approved geometry.
These criteria select useful development controls, not random population samples.

Before new CT access, the initial 0814 label-only choice was found to overlap
the old calibration input. The selection rule was amended to exclude that
input plus one coarse-pixel guard; the first remaining qualifying field was
selected. No model output selected or replaced a field. The final selection
and all label counts are in [selection.json](selection.json).

Every field contains 64 input contexts in a fixed 8 × 8 grid. The retained
central 32 × 32 coarse cells, or corresponding 128 × 128 fine cells, tessellate
the core without output overlap. No missing context is filled or reflected.
Source-presence support is kept conceptually separate from the detector's
legacy nonzero-column mask.

## Three arms on the same physical support

| Arm | CT values | Model input |
| --- | --- | --- |
| A | Source uint8, exact voxel-overlap integration to 2.4 µm, rounded once | Fine Hecate, divide by 200 |
| B | A averaged over exact 4 × 4 × 4 blocks, nearest-even rounded, then repeated by four on each axis | Same fine Hecate, same order, precision and original-A mask |
| C | The exact coarse uint8 block values used to construct B | Coarse Hecate, divide by 255 |

A/B retain the same native-A validity mask. C max-pools that XY mask over 4 × 4
cells. Fine prediction cores are reduced by fixed 4 × 4 probability averages
for common-grid metrics. The full native fine maps are also retained for
visual inspection. Labels use the exact 40/39 mapping from the 9.6 µm score
grid to the released 9.36 µm categorical grid.

A/B measures degradation plus requantization within a fixed model. A/C also
changes the checkpoint. Since A is rounded before its coarse averaging, C
is not asserted to reproduce the earlier direct source-to-9.6 averaging bytes.
Equal nominal 153.6 µm inputs do not imply equal effective depth apertures,
because the fine model removes its edge planes.

## Calibration pilot and stopping rules

Only the already observed 814-c03 context selects depth order and a feasible
backend. Run fine and coarse fp32 in both directions. For each model, select
the higher common-grid central-core AUC; exact ties select reverse. This is
label-informed development calibration, not a demonstration of the physical
written face.

Project the full 256 fine plus 128 coarse forwards from the two-order pilot
mean times. Only if the fp32 projection exceeds 3,600 seconds, test the fine
model in CPU bf16 in both pilot orders. Accept bf16 only when both orders have
maximum full-patch probability difference at most 0.005 and absolute core AP
difference at most 0.01 versus their fp32 counterpart, and the measured
two-order mean bf16 time is strictly below the fp32 fine mean. Preserve both
measured means. Otherwise retain fp32.
Save that decision before any field output. Stop if the chosen projection
cannot fit the remaining budget, checked before and after field acquisition.

Caps are 2,000,000,000 newly read response-body bytes including the fine
checkpoint, 1,024 requests including redirects, 390 total forwards, 5,400
seconds for the whole execution, and a 6.5 GiB process-RSS soft stop between
forwards. Existing sources are reused only against pinned content hashes.
There are no retries or alternative fields. A timeout or incomplete field is
unassessable, never a successful or negative ink result.

The time alarm can be deferred by native operations or outstanding HTTP calls;
the RSS limit is a soft check rather than an OS allocation guarantee. Body
accounting excludes wire framing and unread redirect bodies. New mutable
source chunks are individually hash-recorded, not an authenticated atomic
snapshot of a complete volume.

## Primary endpoint and advancement gate

For each of the four equal quadrants, report AP, AUC, and ink recall at its
own descriptive 5% known-background operating point. Use strict comparison
against the fixed background order statistic so ties cannot exceed that
empirical allowance. The primary field summary is the **equally weighted
mean of quadrant recalls**. It cannot improve merely by shifting scores
between quadrants. Whole-field pooled metrics are secondary.

**Prespecified selection gate, required in both fields:**

- At least 20% relative reduction in macro missed-ink fraction versus C.
- At least 0.05 absolute increase in macro ink recall versus C.
- At least 10% relative reduction in macro missed-ink fraction versus B.
- AP no worse than C in at least three of four quadrants.

Missing supported classes or a baseline ceiling prevents promotion. This
is a practical experiment-selection rule, not statistical significance or
a threshold transferable to an unread scroll. It uses evaluation labels
to describe each ROC operating point.

The owner/team must still inspect all complete fixed-scale maps. A numeric
pass does not establish glyph topology or ten legible letters. These are
model-assisted labels on previously used donor surfaces, with unresolved
inherited model exposure. No claim of independent ground truth, unseen-donor
generalisation, new text or a prize follows from this comparison.

## Execute after the public freeze

    python firstletters-colab/native_resolution/run.py \
      --freeze-reference FULL_PUBLISHED_COMMIT \
      --hecate-source /tmp/vesuvius-transversal-sources/hecate.py \
      --coarse-checkpoint /tmp/vesuvius-hecate-probe-preflight/hecate_9.6um.pth \
      --prior-cache /tmp/vesuvius-hecate-probe-run-v1/source_acquisition.json /tmp/vesuvius-hecate-probe-run-v1/source-cache \
      --prior-cache /tmp/vesuvius-hecate-probe-role-v1/source_acquisition.json /tmp/vesuvius-hecate-probe-role-v1/source-cache \
      --output-dir /tmp/scrollprize-native-resolution-v1

Use the existing Torch/NumPy/Pillow/numcodecs environment. The runner downloads
the pinned fine checkpoint itself and counts it against the same byte budget.
It preserves a source receipt, pilot decisions, pre-field input hashes,
per-field arrays/results and the final or failure receipt. Score maps are
disk-backed during inference, with unevaluated cells left NaN; a stop preserves
the partial work without presenting it as a complete field.

Generated-input validation, without model or source acquisition:

    python -m unittest discover -s firstletters-colab/native_resolution -p 'test_*.py' -v

Tests check physical grid nesting, exact degradation and rounding, identical
A/B masks, categorical-grid alignment, conservative ROC ties, pilot exclusion,
and the macro-quadrant gate. They do not replace actual inference.
