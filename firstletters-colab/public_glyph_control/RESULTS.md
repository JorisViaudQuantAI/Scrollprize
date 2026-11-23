# A complete published control is recovered; this is not a new reading

Date: 2026-11-23. **My inference from the disclosed-reference visual review:**
the original source order recovers the complete broad stroke structure of the
preselected published Nu. The reverse order retains fragments but lacks a
continuously supported diagonal bridge. Both complete outputs are retained.
This is a successful qualitative pipeline control from a scroll with documented
teacher exposure, not a held-out accuracy estimate, a title, or a qualifying
new reading.

## The fixed reference and the observed result

**From a source:** the [frozen selection](selection.json) identifies the fourth
letter of the published word γείνεϲθαι at column −8, line 5, using the
[scholarly publication](https://www.iris.unina.it/handle/11588/985203), its
[author copy](https://www.academia.edu/125007883/THE_FINAL_COLUMNS_OF_PHERC_PARIS_4_REVEALED_THROUGH_VIRTUAL_UNWRAPPING)
and the official published image. This is not the physical final text column.
Our visual collation supplies the region mapping; the inspected sources do not
give a publisher-authored per-letter bounding box. The
[pinned Hecate card](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/README.md)
documents Paris 4 volumetric teacher exposure. The sources inspected do not
establish a per-letter training ledger for this exact selected glyph. Here,
"training-exposed control" means a development control from that scroll; it
does not assert that this exact letter was independently traced to a training
sample.

**From a source — execution and review records:** the
[five-file freeze](https://github.com/JorisViaudQuantAI/Scrollprize/commit/cada2a5497a47868d8dfde68f04e1a333a7723fb)
preceded acquisition and inference. The core is **1920 × 1920 native pixels,
4.608 × 4.608 mm**, with a 2048 × 2048 context. There was no region replacement,
letter prior in the model, per-image contrast fit, training or preferred-order
selection. The [public aggregate](evidence/result.json) binds the original
producer receipt and every retained output to SHA256 hashes. A
[separate visual-review record](evidence/primary_visual_review.json) supplements
the producer's original, unchanged `pending` review fields.

**My inference from inspecting both complete original Q8 cores and the
published reference:** source order contains two broad upright light strokes
with a continuous diagonal descending from left to right. Their positions,
endpoints and connection agree with the reference. Thin dark crack-like gaps,
horizontal texture and fragments at the upper boundary remain visible; the
edge fragments are not counted as more letters. Reverse order contains
isolated lighter regions but does not visibly support the complete connecting
stroke. The known reference is not used to fill that gap.

The reviewer is **ChatGPT, with the expected reference disclosed**. This is an
AI visual comparison, not an independent human papyrological reading. The
scholarly reading itself came from virtually unwrapped imagery, not optical
access to the closed roll. No pixel-level answer key, sensitivity, specificity
or statistical legibility score is manufactured from this comparison.

## What actually ran

**From a source — [run aggregate](evidence/result.json):**

| Quantity | Completed measurement |
| --- | ---: |
| Fixture CNN forwards | 450: 225 per order |
| Precision-pilot forwards | 4 |
| Total producing forwards | 454 |
| Actual fixture arithmetic | FP32 on CPU, four threads |
| Full-run wall time | 5981.135613 s, about 99.69 min |
| Peak resident memory | 2,465,212 KiB |
| New CT/metadata application-body bytes | 516,115,991 |
| Requests, including redirects | 291 |
| Native CT chunks | 289, all complete and hash-verified |
| Archived depth/attention/probability arrays | 1,032,192,768 B including headers |
| New checkpoint downloads / training steps / unread CT requests | 0 / 0 / 0 |

These are application-body bytes; headers and transport overhead were not
metered. The CPU timings include this environment's workload and are not GPU
performance estimates. The completed run stayed within its published resource
limits.

**From a source — precision pilot:** BF16 failed the predeclared conditions.
Maximum absolute differences against FP32 were **0.00808823** and **0.08822867**
for the two old development controls, exceeding the **0.005** limit. The second
mean error also exceeded **0.0005**, and the two BF16 calls were slower. The
runner therefore used FP32 for all fixture forwards. The pilot was not repeated
or turned into a speed benchmark.

## Independent numerical verification

**From a source — [independent review](evidence/independent_review.json):**
[verify_result.py](verify_result.py) reconstructed all **450** archived patch
probabilities from the actual 3D logits, raw attention, recorded depth scalar
and source-derived support. All **29,491,200** probability values were
**bitwise identical**, with maximum absolute difference **0** and maximum ULP
difference **0**. Independent Hann assembly reproduced both full **2048² Q8
contexts**, both **1920² cores**, and all four PNGs exactly.

The same review checked all 289 original chunks and compared **272,629,760**
retained native union voxels bytewise with those sources. It verified 291
source assets, 18 arrays, five frozen files, two dependencies, checkpoint
identity without loading weights, and the four pilot outputs. It took
**19.524267 seconds**, with **zero CNN forwards, checkpoint loads, training
steps or network requests**. It did not interpret images.

**My inference:** this verifies native-source reconstruction and the retained
post-CNN calculations and exports, including the integrity of the assets
actually rehashed and compared. It does not independently rerun the CNN or
exclude a shared source/geometry error. It does not establish independent
glyph truth or certify the geometry of another scroll. The source chunks are
individually pinned; mutable bucket objects do not establish an atomic scan
snapshot.

## Limits that stay in force

**From a source — [frozen geometry](README.md):** source input uses planes
22:86 and reverse input uses 23:87 followed by reversal. Their physical slab
centres differ by one **2.4 µm** plane. This is not a pure reversal of identical
CT bytes, and the comparison cannot identify recto or prove a physical-face
effect.

**My inference:** a working complete-glyph control makes a universal failure
of this CPU pipeline less plausible. It does not distinguish absent ink,
wrong sheet/depth, registration error, acquisition differences or model domain
shift in the team's unread targets. One reference-selected control from a
training-exposed scroll supplies no false-positive rate or evidence of transfer.

The [fixed precision display](../prediction_precision/GLYPH_RESULTS.md) and
[single zero-depth-bias intervention](../depth_bias_control/RESULTS.md) were
separately frozen and reported. Neither changes the original images or earlier
failed gates. No organiser submission or prize claim follows from this run.
