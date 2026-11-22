# Frag5: local structure and bounded translation

**My inference / purpose.** The previous [raw-grid result](../frag5_native_grid/evidence/result.json)
left one of four regions near zero correlation at every tested ±1-voxel shift.
This finite, CPU-only follow-up asks whether local structure persists within
each acquisition, and whether a wider translation range explains the mismatch.
It uses the four original cached arrays, with no new CT, labels or detector.
All sites are already-observed development data. Public freezing precedes
these additional calculations, not the original observation of the data.

## Completed result

**From a source / measured.** The diagnostic completed all planned records in
0.7542 process CPU seconds, with no new source bytes. Record 25's best primary
correlations over all 729 shifts were 0.0619/0.1402/0.2023 for 0/1/4 filter passes;
all maxima were at the search boundary. Four-pass variance retention was
1.39% in its fine footprint and 6.27% in its native coarse field, compared
with 20.26–26.29% and 39.07–62.69% at the other three locations. These are
descriptive results on the same small support, not material identification.

- [Full result](evidence/result.json) and [interpretation](../../../collab/chatgpt-016.md).
- [Complete score fields](evidence/all_correlation_surfaces.npz), with axes in the result.
- [Matched summaries](evidence/matched_summary.csv), [within-scan measures](evidence/structure_summary.csv),
  and [all site pairs](evidence/site_pair_zero_shift.csv).
- [Summary figure](evidence/local_structure_summary.svg) and [all site-pair matrices](evidence/site_pair_matrices.svg).
- [Independent numerical review](evidence/independent_result_review.json) and [artifact manifest](evidence/artifact_manifest.json).

**From a source / independent verification.** Every stored shift correlation
was recomputed by direct dot products, with maximum difference 3.997×10⁻¹⁵.
The checker independently reconstructs the filters and all summary metrics;
it imports none of the analysis runner's scientific functions. It verifies
the declared local computation, not the underlying anatomical identity.

## Fixed analysis

- **From the protocol:** all four IDs (20, 25, 65, 70), both fine representations,
  and the same native coarse fields. Nominal-footprint fine values are primary;
  point-sampled values are a sensitivity.
- **From the protocol:** zero, one and four passes of separable `[1,2,1]/4`;
  exact finite support with touched boundaries removed. Every metric uses
  the central **17³** grid. At nominal 7.91 µm, its voxel support is
  **134.47 µm** and center-to-center span **126.56 µm**. The cached 33³ field
  spans **261.03 µm** in nominal voxel support. These are geometric calculations,
  not a detection-context certification.
- **From the protocol:** all **729** shifts in `{-4,…,+4}³`, up to **31.64 µm
  per axis**. Positive shift compares `fine(q)` with `coarse(q+s)`. Keep all
  fields, maxima, boundary flags and peak-width counts. No correction is applied.
- **From the protocol:** within-scan lag correlations at 1, 2 and 4 grid cells
  along each axis; variance retained after fixed filtering; the full XYZ
  central-difference structure tensor, trace and normalized eigenvalues.
- **From the protocol:** report every fine/coarse site pair at zero shift as
  a stress comparison. Different sites are not an exchangeable null population.
- **From the protocol:** two four-column slabs at `x=-8..-5` and `x=+5..+8`
  provide descriptive spatial fit/check scores in both directions. For a fixed
  shift their supports in the cached analysis grids stay disjoint after the
  largest filter. Fine grids already contain interpolation and integration;
  this does not establish independent original CT support. The entire shift
  search can access overlapping support, and all tissue belongs to the same
  observed fragment. These are **not independent holdouts**.

**My inference / limits.** Strong self-coherence can come from reconstruction
noise or filtering. A plane can correlate strongly while leaving tangential
position undetermined. Repeating structure can create multiple peaks.
Recorded reconstruction-zero values do not establish air/material identity
or calibrated attenuation. No tissue/noise threshold, p-value, label-transfer
approval or ink-detection claim is derived from this four-site analysis.
The failed legacy-render contract remains failed.

## Inputs, execution and reproduction

The [protocol](protocol.json) binds the previous result and each of its four
saved anatomy NPZ files by SHA-256. Those arrays were created by the published
[native-grid runner](../frag5_native_grid/frag5_native_grid_followup.py).
The acquisition and array-generation procedure is documented in that folder;
the derived arrays currently live in the external execution cache. A fresh
reconstruction of them is a separate cost and is not claimed by this analysis.
The program has no network code. It verifies every input before decoding and
requires a fresh output directory and a published 40-character freeze commit.

```bash
python test_local_structure.py --out generated_checks.json

python analyze_local_structure.py \
  --protocol protocol.json \
  --input-dir /external/frag5-native-grid-run-v1 \
  --prior-result ../frag5_native_grid/evidence/result.json \
  --check-only

python analyze_local_structure.py \
  --protocol protocol.json \
  --input-dir /external/frag5-native-grid-run-v1 \
  --prior-result ../frag5_native_grid/evidence/result.json \
  --out /external/frag5-local-structure-run-v1 \
  --freeze-reference FULL_40_CHARACTER_PUBLISHED_COMMIT
```

Dependencies are Python, NumPy and SciPy. The synthetic tests require no CT.
They check known displacement with positive affine intensity conversion,
constants, direct finite sums against FFT correlation, boundary support, and
the slab masks. Planar, periodic and independent-noise examples expose
failure modes; they do not calibrate real-tissue accuracy.

The result contains **52,488** shift correlations: four sites × two fine
representations × three filters × three masks × 729 shifts, plus **96**
site-pair zero-shift comparisons and **36** within-scan metric records.
The full score array has shape `(4,2,3,3,9,9,9)`, with axes and input order
recorded in the result. Undefined scores remain NaN in NPZ and null in JSON.

The resource ceiling is 120 process CPU seconds and zero new network bytes.
The intended engineering decision is whether larger anatomical context or
independently checked translation is necessary before a detector comparison.
There is no automatic next-stage approval.

Independent verification against the original local arrays:

```bash
python review_local_structure.py \
  --input-dir /external/frag5-native-grid-run-v1 \
  --prior-result ../frag5_native_grid/evidence/result.json \
  --protocol protocol.json --runner analyze_local_structure.py \
  --run-dir /external/frag5-local-structure-run-v1 \
  --review-dir /external/frag5-local-structure-review-v1
```

## Sources

- [Prior four-location experiment and interpretation](../../../collab/chatgpt-015.md).
- [Fine source metadata](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes/20231121133215/meta.json)
  and [coarse source metadata](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes/20231130111236/meta.json).
- [Fine reconstruction window](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/metadata/window.json)
  and [coarse reconstruction window](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_7.91um/metadata/window.json).
- [Source and radiometry limits](../frag5_contract/FRAG5_SOURCE_CONTRACT.md).
