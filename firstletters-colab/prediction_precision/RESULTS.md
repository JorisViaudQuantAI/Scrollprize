# Retain float predictions: uint16 restores one operating point, not full ranking

Date: 2026-11-23. **Measured outcome: the frozen two-field uint16 fidelity
criterion failed.** On PHerc. 0814, uint16 restores almost all of the float
recall that the ordinary uint8 export lost, but its average precision remains
more than the allowed 0.005 away from float. PHerc. 0841 passes. This is a
serialization result on already seen, model-assisted public controls, not a
new detector, reading or unread-scroll validation.

## What was fixed and executed

**From the public protocol:** the
[four-file freeze](https://github.com/JorisViaudQuantAI/Scrollprize/commit/5cdfe2de2deee727808d9258edb20875185ba8ca)
was published before new uint16 values or derived displays were calculated.
The earlier [blending experiment](../hecate_blending/RESULTS.md) had already
revealed uint8 saturation, so this hypothesis is explicitly exploratory.
Hann is primary; central and uniform assemblies are diagnostic. Fine-grid
quantization uses nearest-even `round(float64(P) * 65535)`, followed by exact
integer 4-by-4 sums and a single division by `16 * 65535`. Actual prior float
and uint8 maps remain unchanged. Both Hann fields must preserve recall within
0.01 absolute and AP within 0.005 absolute. No favourable diagnostic arm or
alternative bit depth can rescue a failure.

**Measured:** the [unchanged control-run receipt](evidence/controls_result.json)
records **6.155824 seconds**, **181,688 KiB peak RSS**, 28 unchanged consumed
source arrays and **zero network requests, new CT acquisition, CNN forwards or
training steps**. It consumes retained predictions, labels and support, not CT
payloads. Original method gates were not modified.

## Primary Hann results

**Measured:** all rows below use the same supported known-label population.
Each representation's cutoff is the predeclared whole-field background order
statistic, with strict `score > cutoff` and no breaking of tied scores. The
nominal allowance is five percent; realised background exceedance can be lower
because of ties. These label-informed cutoffs are not deployment thresholds.

| Field | Representation | AP | AUC | Ink recall | Actual background exceedance |
| --- | --- | ---: | ---: | ---: | ---: |
| PHerc. 0841 | Float | 0.936147 | 0.948011 | 62.4269% | 4.9978% |
| PHerc. 0841 | Actual uint8 | 0.936009 | 0.947947 | 62.4269% | 4.9978% |
| PHerc. 0841 | Uint16 | 0.936147 | 0.948011 | 62.4269% | 4.9978% |
| PHerc. 0814 | Float | 0.796289 | 0.857805 | 46.1043% | 4.9977% |
| PHerc. 0814 | Actual uint8 | 0.757998 | 0.861942 | 0% | 0% |
| PHerc. 0814 | Uint16 | 0.785618 | 0.865482 | 46.0415% | 4.9951% |

For 0814 Hann, float recovers **11,018 / 23,898** known-ink cells at **1,940 /
38,818** background cells. Uint16 recovers **11,003** ink cells at **1,939**
background cells: only 15 fewer ink cells, a **0.06277 percentage-point** recall
difference. Uint8's 2,603 maximum-score background cells exceed the 1,940-cell
budget, forcing a strict cutoff of 1.0 and zero positives.

Uint16 reduces those maximum-score background cells to **573**, so its
operating point becomes usable. Nevertheless its AP loss is **0.0106710314**,
over twice the **0.005** fidelity allowance. It therefore fails the combined
criterion. The 0841 Hann AP difference is only **0.0000001274**, with identical
recall, and that field passes. All central/uniform outcomes remain in the
receipt; their 0814 AP fidelity checks fail too.

**My inference:** uint16 is materially less destructive here than uint8, but
it is not a sufficient replacement for the retained float map. Keep float
predictions for quantitative screening and recalibration; treat the PNG as a
display/export product. This is a deterministic fidelity recommendation for
these arrays, not a claim that the float scores are calibrated probabilities
or true physical ink.

## Why the AUC increase is not evidence that the model improved

**Measured:** the
[independent review](evidence/controls_independent_review.json) confirms that
0814 Hann's unique known score values decrease **48,543 → 15,975** after
uint16 serialization. Mixed-class tied positive–negative pairs increase
**1,798,251 → 55,939,233**. Exact twice-AUC ordering credit increases
**1,591,524,769 → 1,605,768,397** over the same **927,672,564** positive–negative
pairs, while independently grouped AP decreases.

**My reasoning:** AUC averages ordering credit across all positive–negative
pairs and gives a tie half credit. AP instead averages precision as successive
groups of positive examples enter the retrieved set. Coarsening scores can
change those two quantities in opposite directions. In addition, quantization
is applied before 4-by-4 averaging; it can reorder nearly equal block means,
not merely tie an already reduced score. The observed AUC increase cannot
override the preset AP fidelity failure. No new CT or model inference was
involved, so it is especially inappropriate to call this a detection gain.

## Fixed displays and their limits

**From the protocol:** every full core receives the same pointwise transform,
`clip((logit(clip(P, float32_eps, 1-float32_eps)) + 8) / 24, 0, 1)`, calculated
in float64. No range is fitted to an image, and no spatial filter or expected
letter is supplied. These displays never enter AP, AUC or threshold selection.

**My visual observation:** both complete Hann control displays were inspected
after computation. On 0814, the fixed logit display reveals texture and
context/tile structure inside regions that appear almost uniformly white in
uint8. On 0841, it retains the crack-like gaps and reduces the original broad
foreground/background contrast. Neither small partial-glyph view establishes
an additional letter or better legibility. Exposing numerical variation can
also make model and papyrus texture more visible; it is not evidence of ink.

The separately frozen whole-public-glyph display mode has **not been executed
at the time of this report**. It must first reproduce every original uint8
context pixel exactly from the archived patches, then retain both full depth
orders. A later result will report that work without revising this record.

## Independent verification and provenance

**Measured:** [verify_controls.py](verify_controls.py) imports neither the
production analysis nor its metric implementation. It checks all six fine
quantizations using independent nearest-even arithmetic, sixteen strided
integer reductions, all 18 common-grid metric sets, the fixed displays and
decoded PNGs. It checks 28 consumed source arrays, 60 output assets, four
precision freeze files, six primary freeze files and three dependency hashes.
Manual grouped AP and exact-integer AUC agree with scikit-learn checks.

The [review receipt](evidence/controls_independent_review.json) records **478
scalar checks**, maximum difference **2.22e-16**, **2.143689 seconds**, and no
CNN, checkpoint load or network request. The review confirms both field
decisions and the failed aggregate fidelity criterion. Frozen files and
source arrays were unchanged.

**From sources:** the
[donor manifest](../transversal_lab/hecate_donor_manifest.json) records
model-assisted labels and unresolved Hecate pretraining exposure. These
fields were selected and previously evaluated in the team's earlier work.
Exact Hecate training/teacher exposure for the two fields is not established.
No statistical generalisation or pixel-truth claim follows from the precision
results. All earlier failed gates remain failed.
