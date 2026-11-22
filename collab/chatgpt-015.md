# ChatGPT 015 — separate native-source and raw-grid follow-up

22 November 2026. Continuing the preserved source-render failure in
[ChatGPT 014](chatgpt-014.md). This note reports a separate, adaptive public
Frag5 control experiment. It does not reclassify the earlier `neither` result,
read ink labels or photographs, run a detector, or report an unread candidate.

## 1. New result

**From a source / measured.** All **32 of 32 native fine-TIFF integers**
matched their cached raw-Zarr counterparts exactly, using 64 new intensity
bytes. The secondary trilinear calculation at the four original PPM centers
reproduced the old surface-TIFF discrepancies exactly: **1,530, 236, 0 and
904 uint16 counts**. The fine native-TIFF/Zarr interface therefore does
not explain those sampled center errors. The legacy surface-rendering
hypothesis remains failed.
[Full result](../firstletters-colab/transversal_lab/frag5_native_grid/evidence/result.json),
[all native integers](../firstletters-colab/transversal_lab/frag5_native_grid/evidence/native_source_values.csv),
[all center diagnostics](../firstletters-colab/transversal_lab/frag5_native_grid/evidence/native_center_diagnostic.csv).

**From a source / measured.** The separate anatomy stage completed all
**216 prescribed comparison rows**, containing **864 correlations**. The
primary comparison below uses the unchanged published affine and zero
shift. Each intensity metric has 35,937 samples; each Laplacian metric has
29,791 interior samples. These counts are voxel support, not independent
statistical replicates.

| Public geometry record | Intensity Pearson, point | Intensity Pearson, footprint | Intensity Spearman, point | Intensity Spearman, footprint |
| --- | ---: | ---: | ---: | ---: |
| 20 | 0.6687 | 0.7537 | 0.4849 | 0.5535 |
| 25 | 0.0142 | 0.0267 | -0.0005 | 0.0062 |
| 65 | 0.6448 | 0.7338 | 0.5009 | 0.5761 |
| 70 | 0.6187 | 0.7030 | 0.5411 | 0.6099 |

| Public geometry record | Laplacian Pearson, point | Laplacian Pearson, footprint | Laplacian Spearman, point | Laplacian Spearman, footprint |
| --- | ---: | ---: | ---: | ---: |
| 20 | 0.1751 | 0.2531 | 0.1302 | 0.1854 |
| 25 | -0.0015 | -0.0003 | -0.0061 | -0.0029 |
| 65 | 0.2063 | 0.2833 | 0.1602 | 0.2131 |
| 70 | 0.1034 | 0.1556 | 0.0913 | 0.1356 |

**From a source / measured.** All 27 shifts, both representations and both
metrics are retained in the
[complete CSV](../firstletters-colab/transversal_lab/frag5_native_grid/evidence/native_grid_all_shifts.csv)
and [complete figure](../firstletters-colab/transversal_lab/frag5_native_grid/evidence/native_grid_all_shifts.svg).
Record 25 stays near zero over the entire registered shift set: footprint
intensity Pearson ranges from 0.0066 to 0.0389 and Spearman from -0.0025
to 0.0118. This is a descriptive range of already registered conditions,
not a chosen alignment or confidence interval.
[All registered ranges](../firstletters-colab/transversal_lab/frag5_native_grid/evidence/registered_shift_ranges.json).

## 2. Why this was a separate legitimate question

**From a source.** The pinned
[VC volume loader](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/src/Volume.cpp)
reads native TIFF slices. Our failed surface-profile test used the released
raw Zarr representation. Their equality at the selected points was not
previously established. The
[static follow-up audit](../firstletters-colab/transversal_lab/frag5_contract/reaudit/vc226_sampling_chain_reaudit.md)
found no missed interpolation arithmetic, but retained uncertainty about the
historical executable, input source, transform, radius, interval, output
history and PPM association.

**My inference.** Comparing two raw scanner grids at fixed scanner coordinates
does not logically require reproducing a legacy surface render. The PPM
coordinates can serve as already-selected geometric locations for that
narrow question. This does not authenticate their photographic association,
physical ink face or same-sheet identity. The distinction must be explicit
before new data, because silently dropping an inconvenient acceptance gate
would invalidate the original claim.

**From a source / prospective publication.** The separate
[follow-up protocol](../firstletters-colab/transversal_lab/frag5_native_grid/frag5_native_grid_followup_protocol.json)
and [runner](../firstletters-colab/transversal_lab/frag5_native_grid/frag5_native_grid_followup.py)
were published at
[1dae1cb](https://github.com/JorisViaudQuantAI/Scrollprize/commit/1dae1cbf3c9eeb8c7713fa0e50795608afc823ec)
before new native TIFF or coarse CT pixel access. They pin the earlier
failed result, six cached fine chunks, all 14 original raw-object identities,
32 fixed integer positions and every calculation. The earlier fine data
had already been observed; this was an adaptive engineering investigation,
not a fresh independent evaluation of an untouched donor.

**From a source / protocol.** Each of the same four fixed PPM centers supplies
its eight Cartesian integer corners. All 32 native TIFF values must equal
the cached Zarr values exactly. The secondary central-value comparison
interpolates those native corners at the original PPM point and compares
with the already-observed surface TIFF index 32, regardless of the source
gate. This secondary diagnostic cannot change the gate or select a phase.
Only exact 32/32 equality permits the separate scanner-grid anatomy stage.

**From a source / protocol.** The optional anatomy stage retains the original
published affine, four rounded coarse centers, 33³ grids, point-sampled fine
arm and 125-midpoint approximate voxel-footprint arm. It keeps all 27 coarse
XYZ offsets in `{-1,0,+1}³`, with zero offset primary. It reports Pearson and
Spearman correlations of intensities and the six-neighbor Laplacian on the
31³ valid interior. No offset, registration, blur, intensity conversion or
model parameter is fitted. The source metadata and entire old raw-object
identity snapshot must match before old and new data can be combined.

## 3. Interpretation

**My inference.** The observed footprint effect is real for this narrow
comparison: at the three locations with appreciable raw structural
correlation, the fixed averaging operation raises zero-shift intensity
Pearson by about 0.084–0.089 and Spearman by 0.069–0.075. It also raises
the corresponding zero-shift Laplacian correlations. The fourth location
remains uninformative for correspondence under these metrics. There is no
basis for replacing this four-location outcome by a pooled success number.

**My inference.** This supports retaining physical voxel support in a
fine-to-coarse simulation and treating residual alignment uncertainty
locally. It does not show that the approximate footprint reproduces the
actual scanner, preserves the thin ink signal, or improves a detector.
**I do not know the resulting gain in legible letters or ink accuracy.**
No ink labels or predictions were used to assess that question.

**From a source / measured qualification.** Record 25 is neither empty nor
constant: every one of its 35,937 zero-shift samples is nonzero, with standard
deviations of 4,265.7 in fine-point values, 2,576.0 in the fine footprint,
and 3,505.1 in native coarse values. **My inference:** nonzero intensity and
unit PPM normals do not establish a well-imaged corresponding sheet. These
statistics cannot distinguish noise, weak informative texture, acquisition
differences, off-sheet geometry or local correspondence error. In particular,
the data do not support declaring the published transform universally wrong
or universally accurate.

**My inference / decision.** Keep the original affine and all sensitivities
as evidence. Do not select a favorable shift, fit a blur to these outcomes,
or promote these locations to a photographic-label benchmark. The next
useful control should establish identifiable local anatomy and the actual
physical interface at a detection-sized context before attributing any
fine/native-coarse detector difference to information loss.

**My inference / limits.** Local source equality covers exactly the measured
integer values. Anatomy correlations, if available, describe four small
neighborhoods of one physical fragment. They do not establish ink sensitivity,
photographic-label independence, the historical surface phase, correct
physical face, or population registration accuracy. A nominal voxel footprint
of the fine field does not model the complete native scanner resolution,
noise and reconstruction pipeline. Separate uint16 reconstruction windows
also prevent interpreting raw equality across scans as calibrated attenuation.
No automatic decision authorizes a label or adaptation stage.

## 4. Verification, cost and practical handoff

**From a source / independent review.** The
[separate local-cache checker](../firstletters-colab/transversal_lab/frag5_native_grid/review_frag5_native_grid.py)
reconstructed the 32 integers and all four center diagnostics, verified all
398 prior and 50 new cached object hashes/lengths, and independently
recomputed all 864 correlations from the saved numerical arrays. The largest
correlation difference was **7.77×10⁻¹⁶**. It checked source identity,
acquisition order and cumulative caps. It did not reacquire data, fit
parameters or repeat the 125-point integration; the saved integration
arrays were hash-verified. The original stage-1 checker remained unchanged.
[Review receipt](../firstletters-colab/transversal_lab/frag5_native_grid/evidence/independent_result_review.json).

**From a source / measured.** The follow-up used **8.285 process CPU seconds**
and **226.595 elapsed seconds**, plus 3.430 CPU seconds for independent review.
All 24 HEAD checks and 50 body responses passed. It acquired **32,492,758
new raw chunk bytes** and **2,609 new TIFF/metadata bytes**. The cumulative
raw-source total across both experiments is **57,064,081 bytes** with
**14 unique decoded chunks**. The declared cumulative TIFF/metadata total,
including the specified preflights, is **1,076,780 bytes**. The two CT execution
processes together used approximately **13.891 measured CPU seconds**;
that sum excludes reviews and earlier source/header preflights. The separate
68.285-second cumulative CPU ledger includes a conservative 60-second prior
reservation and must not be presented as exact measured runtime.

**From a source / provenance.** The final result SHA-256 is
`c0e57d1deea4dabaea850ee7e0f111cf48a83f5df73482507c55d1647d7a6c7c`;
the independent review is
`881e8d8081a146bb7d9f004d6a63637a7fcaf6be8d445cd1e9de15c1cd88e2d6`.
[Artifact manifest](../firstletters-colab/transversal_lab/frag5_native_grid/evidence/artifact_manifest.json)
links the exact numerical records, source checks, complete ledger and
presentation files. Neither run used the GPU or changed Claude's queue.

**From a source / implementation.** The bounded resumer verifies five
canonical historical JSON records separately from six individually pinned
raw chunk objects. A newly repeated first stage has different timed receipt
hashes; those are not silently treated as the original run. The
[reproduction guide](../firstletters-colab/transversal_lab/frag5_native_grid/README.md)
explains the minimal external cache and the distinction. Scientific helpers
are imported from the unchanged, hash-checked first-stage executable.
The original failed files remain intact.

**My inference / next decision.** This work is most useful as support for
the team's real rendering and inference path. A future detector experiment
needs explicit physical sampling and radiometry, an appropriate reference
with honestly stated annotation lineage, and separated evaluation tissue.
It should answer whether actual native coarse CT retains useful ink evidence
beyond fine-derived synthetic coarse controls. Neither a raw anatomy
correlation nor a better match to a legacy render is a prize result.

## Open questions

- **Codex:** Which independent anatomical checks can resolve same-sheet
  correspondence and normal/tangential error at useful detection-context
  scale, beyond these already-observed small neighborhoods?
- **Claude Code:** Can the next existing-control inference manifest pin the
  entire renderer/tensor depth convention and a defined intensity conversion,
  while retaining the current GPU queue and released-model baseline?
- **ChatGPT:** Which remaining ambiguity is most valuable to resolve with one
  small prospective experiment: legacy render provenance, physical local
  correspondence, or a defined model-input conversion on a separate control?
- **Owner:** No qualifying letters, candidate or award is reported here.
  Submission and discovery disclosure remain your decisions; no action is
  requested for these public-control experiments.

## Next actions

- **ChatGPT:** Preserve both experiments and complete numerical evidence;
  use this outcome to specify one further bounded method question. Treat
  all observed Frag5 locations as development evidence for later choices.
- **Codex:** Use the source and geometry distinctions when verifying native
  inputs. Keep a fitted or selected correction separate from untouched
  evaluation, and continue the already-owned native geometry work.
- **Claude Code:** Continue existing GPU work. Do not promote a new detector,
  calibration or input-order choice solely from these unlabeled measurements.
- **Owner:** Review a concrete eligible result if one emerges. Nothing in
  this work contacts organisers, opens upstream PRs or merges to `main`.
