# Preserve the comparison when a surface moves

**My inference:** the next useful step is to compare the raw and adjusted
surfaces while keeping their geometry and inference provenance explicit. A
closer material peak is not sufficient evidence that a snap preserved the
intended sheet. This folder supplies an audit, not a replacement snapper or a
mesh-editing command.

## What the current evidence supports

**From a source — reproduced on 21 November 2026.** Codex has now published the
[per-site paired geometry](https://github.com/JorisViaudQuantAI/Scrollprize/tree/1ed75191bbef1c0fa7834c1249f4b0bb8c0eaeaf/registration-checks/atlas-0813/paired).
Applying the unchanged audit to its original-ray peaks and measured displacement
gives **10 passes and 14 holds across 24 sites**. The holds comprise nine changed
nearest crests, four unchanged crests whose distance worsens by more than 0.25
native voxel, and one initially tied crest with no unique reference. All nine
switches also worsen distance to the original crest. The full aggregate,
source-file hashes and code hashes are in
[published_pair_audit.json](published_pair_audit.json).

**My inference.** These are passes of a narrow fixed-original-ray criterion,
not successful sheet corrections. The original crest may be the wrong interface;
the adjusted surface also changes its normals and thus its actual sampling rays.
No ink, reading, anatomical sheet identity or whole-winding guarantee follows.
The rule was frozen before these per-site records became available but after
their original aggregate results were known, so this remains exploratory.

### Reproduce the newly available per-site audit

```bash
python firstletters-colab/geometry_review/replay_published_pair.py
```

**From a source — implementation.** The
[adapter](replay_published_pair.py) reads only pinned Git objects already present
in the local repository. It runs Codex's reviewed offline `verify.py` against
temporary copies of the published files, then reconstructs the legacy point
schema from the unchanged original ray. It never downloads CT or resamples a
profile. The verifier passed its coordinate identities and prior aggregate
checks. The adapter additionally compares the published switch flag with the
unique indices it derives itself. It writes aggregate JSON to standard output;
it never emits site IDs, chart indices or XYZ coordinates.

**From a source — exact conventions.** In Codex's
[producer](https://github.com/JorisViaudQuantAI/Scrollprize/blob/1ed75191bbef1c0fa7834c1249f4b0bb8c0eaeaf/registration-checks/atlas-0813/check.py),
nearest-crest uniqueness uses `rtol=0, atol=1e-9` native voxel. The separate
**0.25-voxel** convention concerns **before/after distance changes**, not nearest
identity. The adapter preserves both. It cannot assign `after_same_peak` or a
changed-identity flag to an initially tied crest. Consequently the frozen legacy
audit emits one `missing_per_site_evidence` hold; the receipt explicitly identifies
this as complete source evidence of ambiguity, not a missing CT record.

### Changed versus unchanged original-ray crest

**From a source — calculated aggregate.** The following values come from the
[published paired records](https://github.com/JorisViaudQuantAI/Scrollprize/blob/1ed75191bbef1c0fa7834c1249f4b0bb8c0eaeaf/registration-checks/atlas-0813/paired/geometry.json)
and are reproducibly aggregated in our [receipt](published_pair_audit.json).
Each cell gives **median / maximum**. The initially tied site is excluded from
both columns; its physical reference and reference-ray misses are unknown.

| Measurement | Changed original-ray crest, 9 sites | Unchanged original-ray crest, 14 sites |
| --- | ---: | ---: |
| Normal rotation, degrees | 2.698 / 8.643 | 5.261 / 19.723 |
| Reference transverse miss from snapped same-UV ray, µm | 3.004 / 19.168 | 2.214 / 43.130 |
| Reference transverse miss from ray through the projected original centre, µm | 4.347 / 7.060 | 6.570 / 48.786 |

**From a source — definitions.** As specified in Codex's
[paired README](https://github.com/JorisViaudQuantAI/Scrollprize/blob/1ed75191bbef1c0fa7834c1249f4b0bb8c0eaeaf/registration-checks/atlas-0813/paired/README.md),
the reference is the initially unique original-ray crest position. Transverse
miss measures perpendicular distance from this fixed physical point to the
new central ray; source values in native voxels are multiplied by 9.362 µm.
The projected comparison uses the snapped surface point closest to the old
centre and its local frame. It does not necessarily pass through the reference.

**My inference.** The unchanged group still contains substantial normal changes
and misses. Keeping a fixed-ray crest index does not establish that the new
renderer follows the same physical location. These descriptive splits cannot
establish a causal relationship or support a new angle/miss threshold; no
threshold was fitted. Preserve the same-UV comparison and separately map fixed
physical reference points, reporting ray miss and correspondence validity.

### Historical aggregate-only result

**From a source.** Codex's
[pinned PHerc. 0813 report](https://github.com/JorisViaudQuantAI/Scrollprize/blob/898b5052f40bd711fb6732861b6cd8b05a9e2bc8/registration-checks/atlas-0813/README.md)
compares 24 geometry-selected sites using the original surface normal. Median
distance to any detected CT crest decreases from 4.00 to 2.62 native voxels;
distance to the same initially nearest crest increases from 4.00 to 6.13 voxels
on 23 unambiguous sites. The nearest detected crest changes at 9 of those 23.
Those are local CT criteria; the report explicitly does not claim that a crest
identifies the physical sheet or ink.

**From a source.** The published
[result.json](https://github.com/JorisViaudQuantAI/Scrollprize/blob/898b5052f40bd711fb6732861b6cd8b05a9e2bc8/registration-checks/atlas-0813/result.json)
contains aggregates. Its
[producer](https://github.com/JorisViaudQuantAI/Scrollprize/blob/898b5052f40bd711fb6732861b6cd8b05a9e2bc8/registration-checks/atlas-0813/check.py)
also writes a local `point-metrics.json`, but that file and the underlying
per-site profiles were not present at that pinned revision. At that point we
did not know which individual sites would pass the conjunctive audit. A numerical
acceptance count cannot be reconstructed from those published marginals alone.

**From our execution record.** Applying the new audit to the actual public
aggregate returned **`insufficient_evidence`**, with no fabricated acceptance
count. The input and protocol hashes are retained in
[existing_aggregate_audit.json](existing_aggregate_audit.json), preserved unchanged.
The `current_public_evidence` text in frozen `protocol.json` describes that
historical state; its bytes and SHA256 remain unchanged. The newly published
records resolve the missing per-site evidence as reported above. Neither replay
performed new CT sampling, ink inference or candidate inspection.

## 1. Audit the saved point records before interpreting improvements

[protocol.json](protocol.json) states the fixed rule. It was designed after
reading Codex's aggregate outcomes and before access to the unavailable point
records. Applying it to those same sites is **exploratory**, not an untouched
validation or external preregistration. The physical uncertainty of crest
location is not estimated here.

**My inference — audit rule.** A point may pass this narrow audit only when:

1. The record is complete, with material support at least 0.8 and at least one
   detected crest.
2. Both original and adjusted nearest crests are unique.
3. Their detected identity is unchanged.
4. Distance to the original crest has not increased by more than 0.25 native
   voxel.
5. Stored distances agree with the claimed unchanged identity.

The 0.8 support threshold and 0.25-voxel distance-change tie convention come from Codex's
[existing frozen protocol](https://github.com/JorisViaudQuantAI/Scrollprize/blob/898b5052f40bd711fb6732861b6cd8b05a9e2bc8/registration-checks/atlas-0813/protocol.txt).
The latter is a scoring convention, **not a measured anatomical error bound**.
Missing evidence, ties, changed association and regression hold the proposal
for review. The command never edits a mesh or quietly clips a displacement.

```bash
python firstletters-colab/geometry_review/audit.py points \
  --input /path/outside/repo/point-metrics.json
```

For this direct command the input is the existing producer's list; there is no
conversion or rerun of its sampling. The separate pinned replay above supplies
the explicit adapter for the newly published paired schema. Output contains
aggregate counts and nonexclusive reasons only.
It does not echo input paths, IDs, coordinates or other record fields. Exit
status 2 means the supplied evidence is insufficient or invalid, not that a
particular physical sheet was disproven.

**My inference — unavoidable limit.** Preserving the original detected crest
can preserve the wrong physical interface. Conversely, a necessary correction
from an initially wrong mesh could change the nearest crest. This audit makes
that decision visible; it cannot resolve it. The 24 sampled sites also cannot
certify all vertices of the winding. A fresh public surface with independently
reviewed correspondence is required before claiming a production improvement.

### A conditional preventive constraint, when uncertainty is available

**My inference, algebra.** Suppose the initially nearest detected crest has
offset `p`, its adjacent crests have offsets `l` and `r`, and their location
errors are bounded by `e_p`, `e_l`, `e_r`. All offsets use the unchanged original
normal. A displacement `d` preserves that nearest-crest association for every
location allowed by those bounds if both the original location zero and `d`
lie strictly inside

```text
((l + p + e_l + e_p)/2, (p + r - e_p - e_r)/2).
```

The crest intervals must also remain ordered and disjoint. Both neighbors must
be observed. `audit.py interval` implements this conditional calculation and
requires explicit error bounds and their stated basis. It abstains if the
initial association itself is not robust. It supplies **no default bound**:
half the profile sample step is a discretization quantity, not a defensible
bound on blurred CT-crest localization. Neither the initial aggregate nor the
new paired records supply an empirical localization-error bound, so this
formulation remains unevaluated. Passing it would still preserve only a detected
crest, not certify a sheet.

## 2. Validate the raw-versus-adjusted rendering contract

```bash
python firstletters-colab/geometry_review/audit.py pair \
  --input /path/outside/repo/paired-render-manifest.json
```

[example_pair_manifest.synthetic.json](example_pair_manifest.synthetic.json)
is an explicitly synthetic schema example. Its repeated-digit hashes are
placeholders; it is not evidence that any artifact exists.

The validator requires two arms and records the following distinctions:

| Requirement | Reason |
| --- | --- |
| Same acquisition, volume-metadata hash, level-zero XYZ frame, origin and physical voxel sizes | A mesh and a CT probe must refer to the same coordinate system. |
| Same reference chart, shape, physical sampling and pixel-center convention | UV comparison requires a declared common chart. Moving a mesh does not validate UV distortion or area. |
| Separate mesh/input/validity hashes and a common-valid-mask hash | The experiment must retain both input geometries and state its common denominator. The validator does not inspect the masks. |
| Same checkpoint, preprocessing, inference settings and ordered depth offsets | Model, threshold, stride, precision and order changes are additional interventions. |
| Explicit normal convention and policy | Fixed original normals isolate placement; recomputing normals changes both position and sampling direction. |
| Explicit correspondence basis | Same UV coordinates and same native anatomical positions are different claims. |

**My inference — same chart.** `comparison_basis: common_chart` compares two
predictions at corresponding chart coordinates, with a common validity mask.
If the mesh moved, these inputs generally sample different CT locations. This
mode must explicitly set `claims_same_physical_positions: false`. It can show
the result of the geometry intervention, but cannot establish pixelwise
anatomical agreement just by intersecting masks.

**My inference — same physical anchors.** `comparison_basis: fixed_native_anchors`
requires a shared native-XYZ anchor hash and frame, plus a separate mapping from
those anchors into each chart. Use the original CT audit locations as fixed
anchors if their native coordinates are available. A nearest point on another
sheet is not a valid anatomical mapping simply because it is close. Mapping
validity and sheet identity still require inspection; a successful manifest
check explicitly reports `anatomical_alignment_certified: false`.

**My inference — normals.** With `normal_policy: fixed_original_field`, both
normal-field hashes must match. With `recomputed_per_mesh`, the validator calls
the experiment **geometry and normal change**, rather than a pure depth-placement
comparison. This permits Claude's existing renderer to be used honestly when
it recomputes normals; it does not require pretending the renderer supports a
fixed-normal option it lacks.

Manifest success means **declared metadata is internally consistent**. This
tool does not read model weights, mesh TIFFs or map arrays to verify the declared
hashes. It does not calculate area, measure correspondence, certify a sheet or
prove a reading. Keep private manifests and generated experiment outputs outside
the public repository; this tool never submits or publishes them.

## 3. If three depth outputs are combined with a pixelwise maximum

**My inference.** A maximum over depth can be useful for triage but introduces
selection at every pixel. A displayed stroke may be assembled from different
depths, including different sheets or unrelated noise. A per-pixel maximum is
not equivalent to choosing the best of three complete images: it permits a
different choice at every pixel. A three-way correction to a whole-image score
therefore does not automatically calibrate the composite's false-positive rate.

Keep the three original maps, the maximum, and the winning-depth-index map.
Inspect offset coherence along each proposed stroke and across neighboring
strokes. Coherence is supporting evidence only; correlated artifacts can also
be coherent. A candidate needs support on an anatomically coherent surface,
with the individual-depth evidence retained. The composite alone is insufficient.

Any score threshold or null distribution must be recalibrated through the
**same three-depth maximum and the same full proposal search** on matched public
positive, sparse and artifact controls. A null calibration from one depth cannot
be reused unchanged. Freeze the depth set before looking at candidate output.
If a local depth field is later fitted, constrain it from CT geometry and audit
sheet continuity; optimizing it solely for ink appearance creates another
opportunity to manufacture a convincing pattern.

## Tests and handoff

```bash
python -m unittest discover -s firstletters-colab/geometry_review -p 'test_*.py' -v
```

**From our test run:** 23 tests passed. They cover changed-crest acceptance
failure, missing/ambiguous records, the stated tie boundary, numeric validation,
conditional interval boundaries, frame/scale/order/model mismatches, normal
policy and UV-versus-native correspondence. They verify software contracts and
algebra on synthetic examples, not anatomical performance. Five adapter tests
add the exact uniqueness tolerance, original-ray switch identity, abstention on
initial ties, rejection of contradictory published switch flags, and aggregate
privacy. The separate published coordinate verifier also passed on the pinned
real records; it verifies their internal consistency, not sheet identity.

## Open questions

- **Codex:** what fresh public surface can supply independently reviewed sheet
  correspondence before this audit is used to justify production geometry?
- **Claude Code:** do the paired render jobs retain the original chart and
  recompute normals, or use a fixed original normal field? Which comparison
  basis is actually supported by their saved coordinates?

## Next actions

- **Codex:** retain the source geometry and independently review ambiguous
  correspondence; the public records have now been audited without new CT.
  Do not tune the rule on these 23 unambiguous sites.
- **Claude Code:** retain raw and adjusted outputs separately, record the normal
  policy, use common validity for chart comparison and native anchors for any
  claim of physical correspondence. Treat depth maxima as review aids.
- **Owner:** keep interpretation and submission decisions separate from this
  metadata check; no claimed find or external action is produced here.
