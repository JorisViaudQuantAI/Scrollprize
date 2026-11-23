# High background-labelled scores extend beyond nearby annotation boundaries

Date: 2026-11-23. Completed descriptive diagnostic; no new promotion gate.

**Measured:** PHerc. 0814's high-score disagreement is not confined to the
frozen 76.8 µm neighbourhood of recorded ink or unknown labels. Among its
**3,334 background-labelled cells with float score ≥ 0.99**, **1,399
(41.9616%)** lie in the predefined interior category, away from both kinds of
label centres and the scoring edge. Likewise **1,016 / 2,603 (39.0319%)** of
its maximum-score Q8 background cells lie in that category. See the
[complete execution receipt](evidence/result.json).

**My inference:** nearby recorded annotation boundaries alone do not account
for all of this disagreement at the tested radius. This observation does not
establish label correctness, omitted real ink, wrong-sheet response, or model
hallucination. All earlier failed gates remain failed.

## What was fixed and measured

**From the [public freeze](https://github.com/JorisViaudQuantAI/Scrollprize/commit/5b48d79fde5a41f8239b8548f1609d4349de70ff)
and [protocol](protocol.json):** the diagnostic uses the two complete existing
Hann fields, their original labels and common support, and their unchanged own
and cross-donor cutoffs. It also records the fixed descriptive endpoints
`float score >= 0.99` and `Q8 common score == 1.0`. It selects no replacement
field, changes no label, and refits no threshold.

Distances are Euclidean distances from the **9.6 µm scoring-cell centres** to
the original **9.36 µm label-cell centres**, separately for known ink and unknown
labels. They are not measurements from a physical ink edge. Complete 282 × 282
label-only halos supply the nearest-label search. Fixed upper-inclusive bands
end at **9.6, 19.2, 38.4 and 76.8 µm**. Integer coordinates in 0.04 µm units
avoid approximate equality decisions at the band edges. Each scoring field is
256 × 256 cells, or **2.4576 mm per side**.

**Measured in the [receipt](evidence/result.json):** all partition denominators
and endpoint counts reconcile to the original totals. No background-labelled
cell or positive is removed. The run took **2.303913 seconds**, with **178,864 KiB
peak RSS**, and verified **13 unchanged consumed assets**. It made **zero
network requests, new CT requests, CT-array reads, detector forwards or training
steps**.

**From a source — the separate [independent review](evidence/independent_review.json):**
an independently written checker reconstructed every fixed-radius bin by
enumerating 19 × 19 original-label-cell offsets per scoring centre, using
rationally derived coordinates in 0.02 µm units. It did not import the runner's
distance or partition helpers or use its nearest-neighbour tree. The finite
enumeration covers every possible label centre within the largest tested
radius; the closest omitted axial distance is 88.92 µm, beyond 76.8 µm.
All **1,103 scalar checks matched exactly**, including denominators, all six
endpoint counts/rates, edge censorship, inherited gates and both fields. The
review took **0.823 seconds**, used **151,304 KiB peak RSS**, and performed no
CNN forwards, CT reads or network requests. It verifies the calculation and
source identities, not the physical correctness of the labels.

## Whole-field denominators and the saturation context

**Measured:** the supported known-background populations are **33,575** cells
on PHerc. 0841 and **38,818** on PHerc. 0814. Counts below are exact; percentages
are rounded for display and use each field's complete background-labelled
population as denominator. [Source: result](evidence/result.json).

| Endpoint | PHerc. 0841 count / denominator (rate) | PHerc. 0814 count / denominator (rate) |
| --- | ---: | ---: |
| Own-field float cutoff | 1,678 / 33,575 (4.9978%) | 1,940 / 38,818 (4.9977%) |
| Own-field Q8 cutoff | 1,678 / 33,575 (4.9978%) | 0 / 38,818 (0%) |
| Other-donor cutoff, float | 0 / 33,575 (0%) | 4,684 / 38,818 (12.0666%) |
| Other-donor cutoff, Q8 | 0 / 33,575 (0%) | 4,684 / 38,818 (12.0666%) |
| Float score ≥ 0.99 | 0 / 33,575 (0%) | 3,334 / 38,818 (8.5888%) |
| Q8 common score = 1.0 | 0 / 33,575 (0%) | 2,603 / 38,818 (6.7057%) |

**Measured:** the unchanged own-field float cutoffs are
**0.7052721381187439** on 0841 and **0.9997147917747498** on 0814. The own-field
Q8 cutoff on 0814 is **1.0**. Its 2,603 tied maximum-score background cells exceed
the 1,940-cell allowance; strict `score > cutoff` therefore selects nothing.
The zero row is the inherited conservative tie outcome, not evidence of an
absence of problematic predictions. Conversely, transferring 0814's float
cutoff to 0841 leaves only **101 / 30,442** known-ink cells positive. These cutoffs
are label-informed operating points, not calibrated ink probabilities.
[Source: result](evidence/result.json); see the original
[blending analysis](../hecate_blending/RESULTS.md).

## Disagreement is enriched near ink labels, but extends farther away

**Measured:** PHerc. 0814 has **4,959** background-labelled cells within 76.8 µm
of a known-ink label centre and **33,859** beyond that radius. The table retains
all five bands and their denominators. [Source: result](evidence/result.json).

| Distance to a known-ink label centre | Background denominator | Float ≥ 0.99: count (within-band rate) | Q8 = 1: count (within-band rate) | Own float cutoff: count (within-band rate) |
| --- | ---: | ---: | ---: | ---: |
| ≤ 9.6 µm | 422 | 145 (34.3602%) | 139 (32.9384%) | 116 (27.4882%) |
| > 9.6 to 19.2 µm | 650 | 223 (34.3077%) | 203 (31.2308%) | 159 (24.4615%) |
| > 19.2 to 38.4 µm | 1,321 | 411 (31.1128%) | 372 (28.1605%) | 267 (20.2120%) |
| > 38.4 to 76.8 µm | 2,566 | 804 (31.3328%) | 618 (24.0842%) | 441 (17.1863%) |
| > 76.8 µm | 33,859 | 1,751 (5.1714%) | 1,271 (3.7538%) | 957 (2.8264%) |

**Measured:** the nearby 4,959 cells are **12.7750%** of 0814's background-labelled
population but contain **1,583 / 3,334 (47.4805%)** of its float ≥ 0.99 cells.
The within-band rate is **31.9218%** nearby versus **5.1714%** farther away.
Thus proximity is associated with a substantially higher rate, while **52.5195%**
of these high-score cells still lie beyond the radius. Similarly **1,271 / 2,603
(48.8283%)** of Q8-saturated background cells lie beyond it. The larger far-away
denominator is visible; counts alone would conceal that difference in rates.
[Source: result](evidence/result.json).

**Measured comparison:** at its own, much lower float cutoff, 0841 also has a
near-label concentration: **1,195 / 5,308 (22.5132%)** nearby versus **483 / 28,267
(1.7087%)** farther away. Nearby cells contain **71.2157%** of its own-cutoff
background positives. It has no background-labelled float ≥ 0.99 or Q8 = 1 cells
in this field. [Source: result](evidence/result.json).

**My inference:** a boundary-related component is compatible with both fields.
The pattern does not show that boundary errors explain every high 0814 score,
and it does not distinguish model spread across an edge from imperfect labels
or alignment. The fields' different own cutoffs also prevent interpreting the
near/far comparison as a shared calibrated detector error rate.

## The predefined interior category retains substantial disagreement

**From the protocol:** the code's `robust_interior_status` category means more
than 76.8 µm from known-ink centres, unknown-label centres and any **observed**
unsupported common-cell centre, and more than 76.8 µm from the continuous outer
scoring boundary. It is an interpretation category, not a certified correctly
labelled region. All other cells remain in the reported partitions and totals.
[Source: protocol](protocol.json).

**Measured:** this interior category contains **27,979** of 0814's
background-labelled cells. Its recorded endpoint counts are:
[source: result](evidence/result.json).

| Endpoint | Total positives in the full field | Positives in the interior category | Rate among its 27,979 background cells | Share of all endpoint positives |
| --- | ---: | ---: | ---: | ---: |
| Own-field float cutoff | 1,940 | 733 | 2.6198% | 37.7835% |
| Transferred 0841 float cutoff | 4,684 | 2,203 | 7.8738% | 47.0325% |
| Float score ≥ 0.99 | 3,334 | 1,399 | 5.0002% | 41.9616% |
| Q8 common score = 1.0 | 2,603 | 1,016 | 3.6313% | 39.0319% |

**Measured:** both fields have **zero unsupported common cells** in their
inherited masks. This diagnostic therefore observes no interior support holes.
It does not inspect support outside the scoring crop or reread CT. The scoring
edge is flagged separately: **5,094** background cells on 0814 are edge-censored,
containing **437** of its float ≥ 0.99 cells and **323** of its Q8-saturated cells.
Those cells remain counted. On 0841, the interior category contains **483**
own-cutoff positives among **23,321** background cells, or **2.0711%**.
[Source: result](evidence/result.json).

**Measured:** all 0814 background-labelled positives under the recorded
definitions are more than 76.8 µm from an unknown-label centre. Its **1,582**
background cells within that distance of an unknown label contain none of
those positives. Thus proximity to the existing unknown-label mask does not
explain their location. This does not establish that the mask identifies all
annotation uncertainty. [Source: result](evidence/result.json).

## What this does and does not resolve

**My inference:** the retained disagreement extends beyond nearby recorded
label boundaries at the frozen 76.8 µm radius, even after separately accounting
for the scoring edge. That is the useful narrower conclusion. The radius is
finite; the diagnostic does not rule out larger registration errors, wider
model responses, omitted annotations, or signals from another sheet.

**From the [donor manifest](../transversal_lab/hecate_donor_manifest.json) and
[result](evidence/result.json):** the labels are model assisted, the donors and
fields were already used in development, and exact Hecate training/teacher
exposure for these fields remains unresolved. This analysis provides no
independent optical or papyrological adjudication. Spatially related cells
and model outputs are dependent; the reported rates are descriptive finite-field
measurements, with no generalization or significance claim.

**My inference:** preserve the float maps and the disagreement as evidence.
The [precision analysis](../prediction_precision/RESULTS.md) shows that export
format can destroy ranking information; higher-precision storage does not decide
whether a high background-labelled score corresponds to real ink. Erasing nearby
labels, excluding difficult boundary cells, or shifting labels to improve the
metrics would answer a different question and cannot repair the earlier failed
gates. This diagnostic contains no new detector, promotion rule, unread scan,
glyph reading, title claim or prize claim.
