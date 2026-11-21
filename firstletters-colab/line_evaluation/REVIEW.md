# Coordinate and metric audit

**My inference — conclusion.** No blocking coordinate or coverage-calculation
error was found. The results support a geometric review aid with documented
misses. They do not establish text recognition, a candidate threshold, or a
false-positive rate. This review was conducted after the frozen run by the agent
who implemented the ranking helper; it is not an external replication.

## What was checked

**From the experiment record.** The evaluator records the method, evaluator,
input, mask and configuration hashes before ranking. It computes target coverage
only after proposals have been generated. The reviewed files are
[evaluate.py](evaluate.py), [the extractor](../line_controls/prepare.py),
[the source inventory](../line_controls/inventory.json) and the locally frozen
`line-localization-diagnostic-v1` protocol/results. The method and evaluator
matched the recorded hashes at review time; the exact original sources are now
retained under `v1_sources/`:

| File | SHA-256 |
| --- | --- |
| `line_review/review.py` | `f188a31e0eb981f70dbd6b6e60a00ba753556634f2d0f98e3db00101ca07f2d7` |
| `line_evaluation/evaluate.py` | `16483d4fe146a9e695330a06d72ea08f043a9e7f0556cd7b37ca0687a8635bb8` |

**From the code and a hand-counted check.** Coverage uses the union of rounded
band polygons on the original pixel grid. Both numerator and denominator exclude
invalid target pixels. An overlapping proposal cannot count the same pixel
twice. A 6 by 6 target containing one invalid pixel and a 3 by 3 proposal
containing that pixel gives exactly 8/35 coverage; repeating the proposal leaves
that result unchanged. Target boxes use floor/ceil bounds and half-open slices;
polygon rasterization includes its boundary. These conventions introduce about
one pixel of boundary uncertainty, not pixel-accurate ink segmentation.

## Source-coordinate alignment

**From a source.** The [343 sidecar](https://github.com/Nieuwlaar/pherc343-first-letters/blob/e8370f1f88a211166056df62ea6fcfffe4a1063e/outputs/concat_w047-w048_R5B2_z9500-11000.json)
distinguishes the image's data panel from the full image and supplies
`letters_area_px` in that panel's coordinates. The extractor crops the panel
once and uses those boxes directly. Subtracting the full-image panel origin
again would be incorrect. The sidecar's original annotation-frame field does
not make these already transformed area boxes raw-volume coordinates.

**From a source and code comparison.** The [306B sidecar](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/PHerc0306B_20250521133212_w074own_touchup_V5Rc.json)
and [rendering code](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/code/claimlib_v2.py)
give the source-to-image mapping. The native crop and its vertical reversal
match the reading orientation. Subtracting the image's map origin places the
published row centres in the extracted map. Source rendering samples pixel
centres with interpolation; our extraction copies rounded native bounds. Their
relative displacement is below one native pixel in each axis. This supports
the extractor's stated rounding limit and cannot explain a missed whole row.

## What the measurements mean

**From the frozen measurements.** In the full 306B context, the second published
row has zero box and centreline coverage among the first five proposals. At ten
proposals, its centreline coverage is 55.8%. This is a retained ranking failure,
not evidence that the source row is absent. In 343, 15 of 18 supplied boxes have
at least 50% supported-area coverage at three proposals, and 18 of 18 at ten.
Those are coarse location boxes: covering them does not count recognized or
legible letters. The 343 input is a recovered display signal with the source
image's thresholding and contrast choices already applied; see the
[control limitations](../line_controls/README.md).

**Post-hoc descriptive calculation, not a frozen primary metric.** At ten
proposals the union of bands occupies 31.0% of the valid 343 map area, 10.8% of
the full 306B map, and 56.6% of the matched-support 306B map. These fractions use
the same polygon union and masks. They put location coverage in the context of
area selected for review; they do not measure specificity or actual inspection
time. Crop previews include additional context outside the bands.

## Why zero proposals is weak negative evidence

**From a source.** The [306B blank-strip record](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/method/checks/blank_strip_variants_v7_trace_x_Ahn_product.json)
defines a strip approximately 5 mm wide and 23 mm tall. It is the original
authors' described blank control, not an independently certified text-free
sample.

**My inference from the fixed geometry.** An 8 mm minimum chain cannot fit
horizontally inside that 5 mm width. Vertical chains cannot obtain the same
flank support as chains in a broad positive context. Zero proposals in this
strip therefore offers especially weak negative evidence. The matched claim
and depth-shuffle cases have identical support, making their comparison useful
for perturbation sensitivity. Shuffling depth still does not produce a
representative ordinary-papyrus negative distribution. These related controls
provide no independent population or search-wide false-positive bound.

**Audit disposition at the initial review.** No threshold, target, ranking
parameter, evaluator code or frozen result was changed following these metric
observations. Every input remains review-required, including inputs producing
no proposals.

**Later integrity-only amendment.** A separate software review identified a
possible producer-replacement race in command-line input hashing. The current
ranker/evaluator now recheck hashes around consumption. The archived v1 sources
and measurements above remain intact; the [v2 comparison receipt](integrity_replay.json)
records identical substantive outputs after the repair. This is a software
robustness replay, not a new blind test or a response to the localization misses.
