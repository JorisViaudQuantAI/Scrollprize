# Native-pair geometric sensitivity and bounded acquisition: frozen plan

22 November 2026. Public metadata and five already published registration landmarks per donor; **no CT, labels, images, photographs or unread scrolls** in this experiment. The original preflight and published in-sample RMS are already known: [photographic_donor_preflight.md](photographic_donor_preflight.md). This is a prospective freeze of the *additional calculations*, not a blind new donor study. Results cannot be called held-out local image-registration validation.

## Frozen two-donor calculation

Physical donors, without changing the public points or affine: PHerc. 1667 Cornice 1 Fragment 3 (Frag5), fine 3.24 µm / actual coarse 7.91 µm, and PHerc. 51 Cornice 4 Fragment 8 (Frag6), fine 3.24 µm / actual coarse 7.91 µm. Use the five exact XYZ voxel-index source/target pairs and the public 4×4 fine-to-coarse affine for each. Their primary URLs and numeric values are frozen in native_pair_geometry_sources.json; keep the raw publisher transforms intact.

1. Compute each published affine's full-five in-sample Euclidean residual; report RMS and max in **coarse voxels**. This is a source-consistency check, not an independent estimate.
2. For each donor, fit five *new* unweighted three-dimensional affine least-squares models using four of the five paired landmarks; predict the omitted fifth. Report all five signed XYZ residual vectors, Euclidean residuals, their RMS and max, and each fit's design-matrix condition number. No outlier deletion, robust fitting, regularization, spatial reweighting, model-selection retries or revised fit after seeing the output. A 4-point affine is an exactly interpolating model, not a statistically reliable generalization estimator. These are leave-one-landmark-out sensitivity measurements inside the original five-point population, not independent local sheet validation.
3. Compare the original public transform against the fresh five-point least-squares refit only at those five sites. For orientation sensitivity, take *4096 synthetic unit normal directions*, ordered by a Fibonacci sphere using z=1-2(i+0.5)/4096 and azimuth=i*pi*(3-sqrt(5)), i=0..4095. Report median and sampled maximum angle between transformed source rays A n and re-rendered target normals A^{-T} n. Do not call these empirical sheet-normal angles or flip a side based on the outcome.
4. Record no correlation with ink, no p-value and no new prize claim. Regardless of result, both native-pair labels and any subsequent model outputs must remain excluded until independent same-sheet/normal checks. For resource allocation only, consider >1 coarse voxel LOO error an **engineering caution** (not a probability bound, accuracy threshold or prize rule).

## Frozen reusable sampling guard

For subsequent geometrically verified source-surface centers, the optional planner accepts finite scanner XYZ voxel positions/normals and **physical offsets in micrometres**. It has two explicit modes: (a) preserve a source physical ray with A(n / 3.24 µm), or (b) use the target geometric normal A^{-T}n with spacing of 7.91 µm. It projects to coarse scanner XYZ and requires *every* requested trilinear corner to fall within raw coarse ZYX bounds. No clipping, inferred fill, boundary reflection or normalization is allowed. Count distinct 128³ uint16 chunks over both trilinear corners; refuse plans exceeding 64 chunks, or 268,435,456 decoded bytes. The planner does **not** fetch compressed chunks or imply a compressed-byte cap. Exact transfer, ETag, headers, source-surface identity, native radiometry and independent local anatomical controls must be verified by a later executor before CT or ink work.

This is deliberately limited to a reusable data-acquisition and transfer prerequisite, **not** a novel ink detector or standalone prize entry. No L4/GPU job, model-forward run, private CT read, organiser contact, upstream PR, main merge or publication of unread findings is authorized here.

## Open questions

- **Codex:** can independent unlabeled, same-sheet anatomical points near an actual Frag5 surface tile constrain local sheet identity and 3D error? The five original registration landmarks cannot.
- **Claude Code:** which depth-order and input radiometry convention would a native-coarse model replay require, without choosing a flip or intensity scale on held-out labels?
- **ChatGPT:** are HTTP range headers and PPM source identity sufficient to enable a future bounded native-coarse test without full-volume acquisition?

## Next actions

- **ChatGPT:** publish implementation/fixture/tests with this freeze, run exactly this two-donor calculation, retain all results and limitations, and send a separate immutable numbered results note.
- **Codex:** continue independently owned native geometry; treat leave-one-out sensitivity as a planning diagnostic only.
- **Claude Code:** continue the active Colab L4 queue without overlapping this CPU metadata check.
- **Owner:** no action is needed for the experiment. Any eventual prize submission or discovery disclosure remains yours.
