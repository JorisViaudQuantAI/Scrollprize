# ChatGPT 017 — larger-context agreement does not validate the center

22 November 2026. Continuing [ChatGPT 016](chatgpt-016.md). I implemented and
executed the separately frozen context-enlargement experiment on the public
opened PHerc.1667 Cornice1 Fragment3 control. The centers, published affine
and sampling procedure stayed fixed. This note reports an anatomical
diagnostic, not ink detection, a candidate or a prize result.

## 1. The new measured result

**From a source / measured.** Both enlarged reconstructions reproduced every
value in the original central arrays exactly: maximum error **zero** for
both 33³ fine representations and the entire 35³ native coarse neighborhood,
at both sites. The protocol kept each center fixed; the overlap checks also
confirm that its previously sampled central values did not change.
[Overlap gates](../firstletters-colab/transversal_lab/frag5_context81/evidence/overlap_gates.json).

**From a source / measured.** Enlarging the weak record 25 context increased
aggregate correlation, but its central region remained weak. The primary
footprint/native comparison at **zero shift** is:

| Grid side | Nominal voxel support | Record 20 Pearson | Record 20 Spearman | Record 25 Pearson | Record 25 Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.13447 mm | 0.7621 | 0.5217 | -0.0169 | -0.0170 |
| 33 | 0.26103 mm | 0.7537 | 0.5535 | 0.0267 | 0.0062 |
| 65 | 0.51415 mm | 0.7282 | 0.4768 | 0.3123 | 0.0604 |
| 81 | 0.64071 mm | 0.7333 | 0.5250 | 0.3762 | 0.0836 |

These are nested supports of the same two previously observed locations.
They are not independent samples, search area estimates or oriented detector
contexts. Both fine arms, both metrics and all 27 original shifts are retained.
[All 432 nested rows](../firstletters-colab/transversal_lab/frag5_context81/evidence/nested_contexts_all_shifts.csv),
[complete context figure](../firstletters-colab/transversal_lab/frag5_context81/evidence/context_dependence.svg).

**From a source / measured.** The separately specified outer shell—the 81³
cube excluding its central 33³ cube—gave record 25 footprint Pearson **0.3897**
and Spearman **0.0890** at zero shift. Its original central 33³ values remained
**0.0267** and **0.0062**. The shell contains 495,504 voxel comparisons, not
495,504 independent anatomical observations.
[Every shell comparison](../firstletters-colab/transversal_lab/frag5_context81/evidence/outer_shell_zero_shift.csv).

**My inference.** The larger field contains information that raises the
aggregate score. This does not establish useful correspondence or observable
material at the central point. A single context-wide registration score
would conceal that distinction. I do not know whether the central ambiguity
is due to weak material contrast, a local hole, a sheet-location error,
acquisition differences, or another physical cause.

## 2. A high smoothed score is particularly easy to misread

**From a source / measured.** On the identical 65³ support, with no shift or
fit, the fixed smoothing sequence produced:

| Binomial passes | Record 20 Pearson | Record 20 Spearman | Record 25 Pearson | Record 25 Spearman |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.7282 | 0.4768 | 0.3123 | 0.0604 |
| 1 | 0.8768 | 0.6209 | 0.6646 | 0.0877 |
| 4 | 0.9469 | 0.7740 | 0.8687 | 0.1556 |

The last record 25 Pearson value looks strong in isolation, while its rank
agreement and the unchanged center tell a much less reassuring story.
The table is a description of the prescribed filters, not a choice of an
optimal smoothing level.
[All smoothing rows](../firstletters-colab/transversal_lab/frag5_context81/evidence/fixed65_smoothing_zero_shift.csv),
[Pearson and rank figure](../firstletters-colab/transversal_lab/frag5_context81/evidence/smoothing_and_rank.svg).

**My inference.** Shared high-contrast or slowly varying structures may
dominate the aggregate covariance while much of the field remains poorly
matched. That is a possible explanation of the Pearson/Spearman divergence,
not a measured identification of its cause. This run did not measure
covariance concentration, label identity or physical surface occupancy.
The independently generated noise counterexamples in [ChatGPT 016](chatgpt-016.md)
also show why smoothing-derived coherence needs careful interpretation.

**My inference / decision.** Do not promote a high smoothed whole-window
score to a local label-transfer certificate. Keep these distinct outputs
in a real registration or rendering report:

1. **Context agreement:** how a fixed surrounding field compares under the
   supplied transform, including declared shifts and scale sensitivities.
2. **Evidence at the target:** whether the actual center and required model
   support contain identifiable, physically appropriate anatomy.
3. **Geometric uncertainty:** independent same-sheet checks or a defensible
   local displacement bound, separate from any score used to fit alignment.

This experiment provides the first output and a warning about the second;
it does not complete the third. The center should remain unresolved for
photographic label transfer. No new threshold is inferred from this pair.

## 3. What was frozen before new data

**From a source / prospective publication.** The protocol, runner, tests and
independent static review were published at
[7f20a77](https://github.com/JorisViaudQuantAI/Scrollprize/commit/7f20a7796431af44e922ffa0af82d0810f1eb189)
before new raw CT bodies. Record 20 was selected as a previously correlated
contrast and record 25 as the previously weak case, after the outcomes in
ChatGPT 016. This is explicitly adaptive development work. The pair must not
be described as an unbiased sample or a new independent validation fragment.

**From a source / protocol.** Each unchanged center supplies an 81³ integer
coarse-grid field. Fine point values use the original inverse affine and
trilinear sampler. The footprint arm retains all 125 midpoint samples per
nominal coarse voxel. Native coarse support is 83³, preserving all 27 offsets
in `{-1,0,+1}³`. The analysis reports 17³, 33³, 65³ and 81³ supports, a separate
65³ fixed-filter sequence and the predefined outer shell. Positive offsets
sample the native coarse field at `q+s`. No offset, blur, source window,
normal, center or model parameter was fitted.
[Full frozen protocol](../firstletters-colab/transversal_lab/frag5_context81/protocol.json).

**From a source / geometry.** At nominal 7.91 µm, an 81³ field has 640.71 µm of
voxel support and 632.8 µm between extreme sample centers. Its width is a
scanner-axis quantity. It does not establish coverage of an arbitrarily
oriented model slab, physical sheet identity or ink-facing direction.
[Original source/geometry specification](../firstletters-colab/transversal_lab/frag5_native_pair/frag5_native_pair_stage1_protocol.json),
[official coarse metadata](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes/20231130111236/meta.json).

**From a source / acquisition record.** All 51 HEAD checks and 45 body responses
completed successfully. The runner checked object identity and sizes before
new bodies, checked both Zarr metadata hashes before new chunks, and
reused six individually pinned source objects. It then fetched 43 new chunks.
The original scientific helpers were imported only after their hashes
passed. Both old-overlap gates passed before the new metric stage.
[HEAD snapshot](../firstletters-colab/transversal_lab/frag5_context81/evidence/heads_before_new_data.json),
[complete ledger](../firstletters-colab/transversal_lab/frag5_context81/evidence/acquisition.json).

## 4. Cost, verification and practical limit

**From a source / measured.** This run used **41.4822 process CPU seconds**
and **151.0277 elapsed seconds**. It acquired **175,385,570 new raw bytes** and
**801 metadata bytes**. The cumulative raw-source total for this Frag5
sequence is **232,449,651 bytes**, below the frozen 256 MiB ceiling. The run's
working source set was 49 decoded chunks; the historical union is 57 unique
chunks, below 64. These are execution resources, excluding development,
plotting and independent review. No GPU was used.
[Full result and resource ledger](../firstletters-colab/transversal_lab/frag5_context81/evidence/result.json).

**From a source / independent verification.** A separate checker recomputed
all **896 correlations**, all **18 structure records** and both overlap
gates. The maximum correlation difference was **1.9984 × 10⁻¹⁵**. It used
independently implemented average ranks, centered dot products and finite
binomial kernels, and imported none of the primary scientific functions.
It verified all 51 HEAD records, 45 new bodies and six retained source
bodies against the declared identities, hashes, sizes and admission limits.
[Full independent review](../firstletters-colab/transversal_lab/frag5_context81/evidence/independent_result_review.json).

**From a source / independently sampled raw verification.** Before reading
new outcomes, the checker fixed 15 geometric positions per site: eight cube
corners, six face centers and the center. At these **30 positions**, manual
eight-corner raw interpolation differed from the saved fine point values
by at most **1.3606 × 10⁻⁸ counts**; the independently computed 125-midpoint
means differed by at most **2.6084 × 10⁻⁹ counts**. All 30 native coarse
integers matched exactly. These raw checks used a separately predeclared
**10⁻⁶-count** numerical tolerance; they did not relax the original
10⁻⁸/exact overlap requirements, whose measured errors were zero.
[Every fixed raw check](../firstletters-colab/transversal_lab/frag5_context81/evidence/fixed_raw_sample_checks.json).

**From a source / verification scope.** The independent review used **10.2543
CPU seconds**, decoded 33 already cached chunks and made 30,270 integer
reads. It checked all metric calculations but sampled only 30 output
positions for direct raw integration; it did not repeat the full 81³
integration independently. It performed no new network acquisition, label
access, model inference or input mutation.
[Reusable checker](../firstletters-colab/transversal_lab/frag5_context81/review_context81.py).

**From a source / provenance.** The result SHA-256 is
`24c1e6d1aec3d265bf87509b84033d78da29e23151a401b025a7de7192bebf9a`.
The two saved 81³ derived-array packages, source-object identities, complete
metric tables and presentation files are hash-bound by the execution
records and [artifact manifest](../firstletters-colab/transversal_lab/frag5_context81/evidence/artifact_manifest.json).
Large source objects and derived CT arrays stay outside Git; the bounded
reproduction procedure and canonical cache requirements are documented in
the [README](../firstletters-colab/transversal_lab/frag5_context81/README.md).

**My inference / limit.** This closes the narrow question of whether a
larger fixed context changes the aggregate comparison: it does for record 25,
and it does not resolve the original central ambiguity. It establishes no
ink-model improvement, label independence, detection sensitivity, calibrated
registration accuracy or claim about unread scrolls. The earlier failed
legacy surface-render contract also remains failed.

## Open questions

- **ChatGPT:** Can the existing data distinguish a nearby interface from
  a true central surface using a finite, physically directed anatomical
  check, without selecting a depth from model or label outcomes?
- **Codex:** Can a native geometry report keep central support and surrounding
  agreement separate instead of treating one aggregate score as local validity?
- **Claude Code:** When the paired 306B GPU outputs arrive, can the existing
  physical mask and exact tile origins accompany them for the outstanding
  line-rank comparison? This CPU control work does not replace that test.
- **Owner:** No candidate, qualifying letters or award is reported. Submission
  and discovery disclosure remain your decisions.

## Next actions

- **ChatGPT:** Publish the complete two-stage results and independent checks;
  keep the affine unchanged and the central record 25 interpretation unresolved.
  The next method question should resolve physical support at the target,
  rather than extend a correlation or detector sweep.
- **Codex:** Continue owned native geometry work. Use the center/context
  distinction when assessing whether a transferred surface or training label
  is usable, with independent physical checks where available.
- **Claude Code:** Continue the existing GPU queue and preserve released-model
  controls. No new smoothing or adaptation is promoted by these measurements.
- **Owner:** Review any later eligible result before deciding on submission.
  This work makes no organiser contact, upstream PR or merge to main.
