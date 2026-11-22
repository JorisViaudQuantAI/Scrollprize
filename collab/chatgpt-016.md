# ChatGPT 016 — weak local structure, not a justified registration correction

22 November 2026. Continuing [ChatGPT 015](chatgpt-015.md), using only the
public opened PHerc.1667 Cornice1 Fragment3 control. This is an executed,
adaptive diagnostic on already observed data. There is no unread candidate,
new detector result or prize claim.

## 1. What I ran and found

**From a source / measured.** I implemented and ran the fixed analysis locally.
It completed **52,488** translation correlations, **96** site-pair comparisons
and **36** within-scan structure records, with no new CT download. A separate
agent independently reconstructed every filtered array and recomputed every
translation score by direct dot products. The largest difference was
**3.997 × 10⁻¹⁵**. [Result](../firstletters-colab/transversal_lab/frag5_local_structure/evidence/result.json),
[independent review](../firstletters-colab/transversal_lab/frag5_local_structure/evidence/independent_result_review.json).

**From a source / measured.** The poorly matched record 25 did not acquire
a convincing alignment under the wider search. Its maximum unfiltered
footprint/native Pearson correlation was **0.0619**; after one and four
fixed smoothing passes the maxima were **0.1402** and **0.2023**. All three
maxima lie on the search boundary. Its original zero-shift scores remained
near zero. These are optimized descriptive maxima over the declared shifts,
not evidence that the maximizing displacement is correct.
[All primary and secondary summaries](../firstletters-colab/transversal_lab/frag5_local_structure/evidence/matched_summary.csv).

**From a source / measured.** The other three locations behaved differently:

| Public record | Zero shift, unfiltered | Zero shift, one pass | Zero shift, four passes | Four-pass variance retained, fine footprint | Four-pass variance retained, native coarse |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.7621 | 0.9001 | 0.9481 | 26.29% | 39.07% |
| 25 | -0.0169 | -0.0202 | -0.0035 | 1.39% | 6.27% |
| 65 | 0.7289 | 0.8725 | 0.9646 | 25.98% | 62.69% |
| 70 | 0.6653 | 0.8235 | 0.9262 | 20.26% | 53.41% |

These values use the same central **17³** samples in every row and scale,
with the footprint fine representation as primary. They should not be
confused with the previous 33³ correlations. The point-sampled sensitivity,
all intermediate measurements and every original result remain available.
[Structure table](../firstletters-colab/transversal_lab/frag5_local_structure/evidence/structure_summary.csv),
[figure](../firstletters-colab/transversal_lab/frag5_local_structure/evidence/local_structure_summary.svg).

**My inference.** In these small neighborhoods, record 25 has much less
variation that persists through the fixed filters, and the tested translation
family does not establish corresponding anatomy. This strengthens the case
for investigating local anatomical identifiability before fitting an offset.
It does **not** identify the material, prove the location is air, exclude
papyrus or ink, or distinguish all acquisition and registration failures.
The maxima at the boundary also leave larger displacements unresolved.

## 2. Why self-coherence and smoothing can mislead

**From a source / measured.** Before smoothing, record 25's native coarse
lag-one correlations were **0.7343 in X, 0.7911 in Y and 0.0797 in Z**.
After four passes they were **0.9052, 0.8831 and 0.8949**, while its cross-scan
zero-shift agreement remained approximately zero. Thus a visually or
numerically smooth local field is not sufficient evidence of reproducible
anatomy. These are descriptive measurements; the axis pattern alone does
not establish a reconstruction-noise mechanism.
[All within-scan measurements](../firstletters-colab/transversal_lab/frag5_local_structure/evidence/result.json).

**From a source / generated controls.** Two independent fixed-seed noise
fields also became strongly self-correlated after filtering: one field's
X lag-one correlation rose from **0.0259** to **0.8797**. Searching 729 shifts
then gave a maximum cross-field correlation of **0.3741**, despite the
fields being independently generated. A synthetic flat slab had **81**
shifts within 0.01 of its maximum because tangential position was ambiguous;
periodic bands had **243**. These are counterexamples, not a calibrated
false-positive rate for real CT.
[All generated controls](../firstletters-colab/transversal_lab/frag5_local_structure/generated_checks.json).

**From a source / measured.** The all-site comparison also shows how
smoothing can raise matches between different locations. In the footprint
arm, record 65 against native record 70 rose from **0.0915** to **0.3874**
at zero shift, while the corresponding same-site score rose from **0.7289** to
**0.9646**. This mismatch is a stress comparison within one fragment,
not an exchangeable null sample.
[All 96 comparisons](../firstletters-colab/transversal_lab/frag5_local_structure/evidence/site_pair_zero_shift.csv),
[complete matrices](../firstletters-colab/transversal_lab/frag5_local_structure/evidence/site_pair_matrices.svg).

**My inference / operational consequence.** Do not use smoothed coherence,
a structure-tensor eigenvalue or a maximized cross-correlation by itself
to approve a local label transfer or surface correction. Keep an absolute
physical support, report the unfitted transform, retain displaced and
mismatched comparisons, and show what information survives across scales.
This four-location diagnostic supplies no universal acceptance threshold.

## 3. Design and selection limits

**From a source / prospective publication.** The runner, protocol, generated
tests and static review were published at
[32a1393](https://github.com/JorisViaudQuantAI/Scrollprize/commit/32a13933a4e811973a79809f20c6dd54de19488f)
before computing these additional CT metrics. All four CT arrays and earlier
outcomes had already been observed. This is therefore an adaptive development
analysis, not a new independent test fragment.

**From a source / protocol.** The analysis uses separable `[1,2,1]/4`
filtering zero, one or four times, removing touched boundaries. All 729
translations in `{-4,…,+4}³` are retained. The sign convention is
`fine(q)` against `coarse(q+s)`. At nominal 7.91 µm, the search reaches
**31.64 µm per axis**, and the common 17³ support spans **134.47 µm** in
voxel support, or **126.56 µm** between extreme sample centers. The original
33³ cache spans **261.03 µm** in nominal support. These are calculations
from the [frozen specification](../firstletters-colab/transversal_lab/frag5_local_structure/protocol.json),
not claims of a complete detector context or a correctly identified sheet.

**From a source / protocol.** Two four-column slabs at opposite ends of the
central X interval provide a descriptive fit/check calculation in both
directions. Their contributing cached-grid supports are disjoint at a fixed
shift after the largest filter, but the full shift search can access
overlapping support. The fine arrays also already contain interpolation
and footprint integration. Neither slab is an independent acquisition or
held-out fragment. Every selected shift is reported; none is applied.

**From a source / measured.** For record 25, all six footprint slab-selected
shifts lie on the search boundary. The two four-pass fit scores reached
**0.4223** and **0.4230**, but the opposite-slab scores at those selected
shifts were only **0.1365** and **0.1659**. This is an explicit example of
why a fitted score cannot serve as its own confirmation.
[Complete fit/check records](../firstletters-colab/transversal_lab/frag5_local_structure/evidence/result.json).

**From a source / metadata.** The recorded reconstructed-zero values are
20,244 in the fine scan and 13,506 in the native coarse scan.
[Fine window](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/metadata/window.json),
[coarse window](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_7.91um/metadata/window.json).
**My inference:** the relatively low contrast of record 25 to those values
motivates a material-context check, but does not calibrate attenuation or
establish an empty location. I do not know the physical cause from this test.

## 4. Verification, deliverables and next finite experiment

**From a source / implementation review.** The independent review found an
exact-constant floating-point edge case before the public freeze: mean
rounding could give a mathematically constant non-binary field a tiny
nonzero variance. The corrected implementation explicitly checks constants
before correlation and reports undefined values as null/NaN. The generated
checks include non-binary constants, finite support, known displacement with
positive affine intensity conversion and exact gradient tensors.
[Review receipt](../firstletters-colab/transversal_lab/frag5_local_structure/independent_code_review.json).

**From a source / measured.** The CT diagnostic recorded **0.7542 process CPU
seconds** from its main entry point, excluding import time, tests and review.
It used zero new network bytes and zero new raw chunks. Independent checking
took **2.2835 CPU seconds**. All 52,488 saved translation scores, 324 directional
lag correlations, 96 site pairs, 72 peak summaries and 48 slab-direction
summaries passed. The checker rebuilt filters independently and imported
none of the analysis runner's scientific functions.
[Reusable checker](../firstletters-colab/transversal_lab/frag5_local_structure/review_local_structure.py),
[result manifest](../firstletters-colab/transversal_lab/frag5_local_structure/evidence/artifact_manifest.json).

**My inference / next experiment.** The next useful question is whether a
larger fixed anatomical neighborhood resolves record 25's local ambiguity.
Use the same center and published affine; enlarge context without choosing
a new center or fitting a displacement. A contrasting previously correlated
record provides an execution control. An 81³ grid at the same nominal pitch
would have **640.71 µm** voxel support. The pair must be explicitly selected
adaptively, and a high outer-context correlation must not be presented as
proof that the original central patch contains a sheet or transferable ink.
Freeze any new pixel acquisition separately and keep its measured cost.

## Open questions

- **ChatGPT:** Does the same-center larger-context comparison reveal stable
  corresponding structure around record 25, or does the ambiguity persist?
- **Codex:** Can the distinction between missing local information and a
  fitted offset be retained in the real native surface-validation path?
- **Claude Code:** The full-field 306B two-placement test remains with your
  existing GPU queue; this CPU work supplies no replacement inference result.
- **Owner:** No candidate or award is reported. No organiser contact,
  submission, upstream PR or merge to main has been made.

## Next actions

- **ChatGPT:** Preserve these complete results and execute one bounded,
  separately frozen larger-context check if its geometry and transport plan
  fit the remaining source budget. Do not reopen a model sweep from these data.
- **Codex:** Continue owned geometry work, keeping the original transform
  and local sensitivity separate from any chosen correction.
- **Claude Code:** Continue owned GPU work and provide the actual paired
  tile-placement outputs when available for the requested rank analysis.
- **Owner:** Decide on any later submission only when there is a concrete,
  eligible result to review. No action is requested for this public-control work.
