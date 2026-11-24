# Studio pose factors — Q33 mechanistic follow-up

This module separates two interventions that changed together in the preceding
[native-CT tilt experiment](../studio_oblique_stress/RESULTS.md): moving the
surface anchors and rotating the directions along which CT is sampled. It
uses the same public PHerc0841 author annotation panel and fixed Studio
checkpoint. The experiment is complete: **275 new CPU forwards**, with all
eight four-cell blocks retained. The [results and interpretation](RESULTS.md)
include mixed-core outcomes, conditional contrasts, full and portable audit
receipts, and the limits of the mechanistic comparison.

The design is informed by the observed Q33 failures. It is a mechanistic
development study, not an independent validation, a search for the best angle,
or a demonstration of new readable text.

## Factorial construction

Let `S(u,v)` be the original curved surface in native XYZ, `n(u,v)` its signed
unit normal, `C` the original reference-surface centre of the 128-square
patch, and `R` a signed rigid rotation around one positive local chart
tangent at that centre. Define the two independent binary factors:

- **A — anchors:** leave the anchors at `S`, or rotate them to
  `C + R(S-C)`.
- **D — sampling directions:** leave the directions at `n`, or rotate them
  to `R n`.

For depth `d` in native CT voxels, the sampling coordinate is

```text
F(A,D; u,v,d) = C + R^A (S(u,v) - C) + d R^D n(u,v),
where R^0 = I and R^1 = R.
```

| Cell | Anchor positions | Sampling directions | Meaning |
|---|---|---|---|
| A0 D0 | Original | Original | Parent identity |
| A0 D1 | Original | Rotated | Direction-only intervention |
| A1 D0 | Rotated | Original | Anchor-only intervention |
| A1 D1 | Rotated | Rotated | Parent rigid-tilt intervention |

The off-diagonal cells deliberately use directions that need not be normal
to their anchor surface. They are controlled sampling interventions, not
claims that the modified directions are anatomically correct mesh normals.
The rotation centre remains the reference surface point, rather than the
midpoint of the asymmetric sixteen-plane input stack.

Eight fixed blocks were executed:

| Rotation axis | Signed angles |
|---|---|
| Positive local chart x tangent | -20, -5, +5, +20 degrees |
| Positive local chart y tangent | -20, -5, +5, +20 degrees |

Every block contains the four factorial cells on the same 17 parent cores.
The axes describe the local mesh chart, not certified handwriting directions.
These are subsets of the angles already tested in Q33; their choice is
informed by that parent experiment. The executable protocol must record the
exact arm names, ordering, source identities and numerical tolerances before
any new detector forward.

## Fixed panel and portable reuse

The original panel contains 11 mixed cores, three pure-ink controls and three
pure-background controls. The parent selected it from annotations before its
tilted inference. This follow-up retains all 17 author indices and their
original categorical annotations; it does not replace a core after inspecting
geometry or predictions.

The parent code and evidence are part of this repository at
[`bfc9372ded39809fc79388a6017ef768e91bc17a`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/bfc9372ded39809fc79388a6017ef768e91bc17a).
The relevant portable files are
[protocol.json](../studio_oblique_stress/evidence/protocol.json) and
[scored_arrays.npz](../studio_oblique_stress/evidence/scored_arrays.npz).
The parent archive includes aligned probabilities, original and masked
labels, condition names, author indices and its original common mask.

Parent identities:

- Protocol SHA256:
  `d56864f88ba32302344c1858f9a53cf4c1f203b64a7b28fab980105535793a80`.
- Probability archive SHA256:
  `9558cfd463a3d3450e0b7499321df8fa49b64fa83286c984666da2dfac5fe6ad`.
- Author source commit:
  `bf66f753176792a966bca17822629fb8821f20c8`.
- Checkpoint SHA256:
  `96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90`.

Reuse is portable: the parent experiment's temporary prepared and inference
directories are not required. Before accepting any reused prediction, the
follow-up reconstructs and verifies the **153 relevant parent input hashes
from native CT**. Those references comprise 17 unique identity inputs and
136 rigid-tilt inputs (eight blocks times 17 cores). The same 17 identity
predictions appear in every block but count only once as unique references.

The actual new forward budget is:

| Work | Count |
|---|---:|
| Identity sentinels, executed before the new factor cells | 3 |
| A0 D1: eight blocks times 17 cores | 136 |
| A1 D0: eight blocks times 17 cores | 136 |
| **New model forwards executed** | **275** |
| Unique parent prediction references reused | 153 |

The sentinels are author indices **250** (background), **288** (ink) and
**326** (mixed). Their fresh worker predictions must satisfy the frozen
comparison against the parent identity before the 272 new factor-cell
forwards are submitted. A source, input-hash, checkpoint or sentinel mismatch
must stop the experiment. The results must distinguish newly executed
forwards from reused references; repeated use of identity in eight blocks is
not eight independent measurements or eight additional forwards per core.

## Native data and fixed numerical conventions

Use the parent's pinned author geometry, affine recipe, source CT and
categorical labels. CT is sampled directly at the factor-specific native XYZ
coordinates. Rewarping an already prepared sixteen-plane slab would define
a different experiment and could miss the required source context.

The original physical conventions remain:

- Native CT pitch: 9.366 micrometres; annotation XY pitch: 9.36 micrometres.
- Sixteen depth offsets: `[-8,...,+7]` native voxels. The reference surface
  has depth zero at input index 8.
- Same signed normal convention and stored depth order as the parent.
- Same native CT interpolation, uint8 conversion and model normalization.
- Same checkpoint in evaluation mode, with the frozen runtime and no training.

Every positive-weight CT interpolation corner must be supported by a present
source object. Missing source data cannot be silently converted into a zero
background. Received source hashes and the exact input hashes belong in the
preparation receipts. Source support certifies availability of stored CT
values, not anatomical sheet identity.

## Physical readback and populations

Each scored label refers to its original reference-surface point `X=S(u,v)`.
For each cell, the geometric readback must find `(u',v',d')` satisfying

```text
F(A,D; u',v',d') = X.
```

The corresponding output probability is read at `(u',v')`, with the declared
bilinear rule. Categorical labels remain unchanged. At fixed anchors the
reference point has depth zero; when anchors move, correspondence generally
requires a different depth and chart position. A numerical solution must meet
the frozen residual, source support, depth, output-margin and local geometric
checks. No claim of global tube injectivity follows from a converged local
inverse.

### One common mask per angle/axis block

The **primary mask is the intersection of valid readback/support across all
four cells within that block**, applied to the original known annotations.
It is fixed before new model inference. The original parent mask intersected
thirteen tilt conditions; it must not silently replace this new blockwise
definition.

Every block must retain at least:

- **95% of original known labels globally** across the 17 cores;
- **90% of original known labels in each core**.

The experiment retains geometry failures rather than replacing cores or
changing the mask after scores are seen. Report original unknown labels and
geometry-excluded known labels separately, including ink/background counts
and retention for each block and each core.

**Different blocks may retain different populations.** The A0 D0 reference
therefore needs to be scored separately on each block's mask, even though its
underlying predictions are reused. A difference between raw AP values from
two angles could reflect a population difference. The primary comparisons
are paired contrasts among the four cells on one block's identical labels.
Any across-block summary must expose the denominator variation and must not
present all blocks as one common pixel population.

Do not compare these blockwise identity APs directly with the parent's global
0.806586 identity AP or its earlier 53-core result. Those are different masks
or populations. Any secondary stricter common-population analysis must be
identified separately and cannot replace the primary outcomes after inspection.

## Outcomes and descriptive factorial contrasts

The **primary outcome is recall on the eleven parent mixed cores**, pooled
over their retained ink labels within each block, at the fixed probability
threshold 0.5. This addresses the parent result in which aggregate AP and
pure controls obscured the loss of detections inside mixed zones.

Secondary reporting includes AP, ROC AUC, background FPR, the underlying
confusion counts, per-core measures and separate pure-ink/pure-background
responses. A pure control means its known annotations have one class; unknown
pixels remain unknown. Metrics with no corresponding denominator remain
undefined rather than being replaced with a favourable numeric score.
Pure-ink AP does not measure ink/background separation.

For a metric `M` evaluated on one block's common population, report the
conditional contrasts:

```text
Direction effect with original anchors:  M01 - M00
Direction effect with rotated anchors:   M11 - M10
Anchor effect with original directions:  M10 - M00
Anchor effect with rotated directions:   M11 - M01
Interaction contrast:                    M11 - M10 - M01 + M00
```

The interaction is a descriptive contrast in the chosen metric's units.
It is not a percentage of a physical failure mechanism explained. AP is
nonlinear and not additive over groups, so aggregate AP, pooled mixed-core
AP and the distribution of separate per-core APs must remain distinct.
Keep all eight blocks; do not promote whichever angle or cell gives the
best score. Do not infer statistical independence of pixels, cores or blocks.

## Freeze and verification order

Before the 275 new forwards, freeze the experiment's code, panel, angles,
four-cell masks, prepared-input hashes, reuse mapping, metrics, sentinel
criterion and any decision thresholds. The freeze should identify both this
follow-up and the exact parent evidence being reused.

The intended execution order is:

1. Verify the pinned source, model and portable parent evidence.
2. Reconstruct the 153 parent input references from native CT and check hashes.
3. Construct the new factor cells, qualify inverse readback and blockwise
   common support, and save all masks and source receipts.
4. Commit the frozen protocol and preparation identities.
5. Execute the three identity sentinels and enforce their comparison gate.
6. Execute all 272 new factor-cell forwards without score-based changes.
7. Independently reproduce the readback, labels, denominators, metrics and
   interaction contrasts, then publish every outcome and retained failure.

The final machine-readable protocol and receipts take precedence over these
interface descriptions. Any change made after preflight must be recorded
explicitly with its timing; the fact that the study is informed by Q33 must
remain visible even when the new prediction phase is prospectively frozen.

## Reproduction interface

The runner has `prepare`, `infer` and `summarize` commands. Each accepts
the same seven mandatory path arguments. The parent evidence is resolved
inside the repository; there are no `--parent-prepared` or `--parent-output`
arguments.

```bash
python firstletters-colab/studio_pose_factors/run.py prepare \
  --source-repo /path/to/pinned-studio-source \
  --checkpoint /path/to/studio-softloss-20261023-ema.safetensors \
  --author-cache /path/to/author-data-cache \
  --geometry-cache /path/to/geometry-cache \
  --native-cache /path/to/native-ct-cache \
  --prepared /path/to/pose-factors-prepared \
  --output /path/to/pose-factors-results
```

Use the same arguments with `infer` after the protocol freeze, then with
`summarize` to replay retained predictions. Use fresh output directories and
consult `run.py --help` for the final implemented options. Large CT objects,
weights and prepared inputs remain outside git; source receipts, protocol,
small score archives and analysis code are the portable deliverables.

### Replay the checked-in results without weights or CT downloads

```bash
python firstletters-colab/studio_pose_factors/audit_results.py \
  --output firstletters-colab/studio_pose_factors/evidence \
  --protocol firstletters-colab/studio_pose_factors/evidence/protocol.json \
  --parent-evidence firstletters-colab/studio_oblique_stress/evidence \
  --expected-protocol-sha c502a6bad514c4bd8eb63656f93cdcd82d62b58b11250ea290c8cf7690b7ef5c \
  --report /tmp/studio_pose_factors_replay.json
```

Choose a fresh `--report` filename; the auditor refuses to overwrite an
existing review.

This mode independently recalculates archive metrics and contrasts; it does
not execute the detector or reacquire CT. See [RESULTS.md](RESULTS.md) for
the separately completed full-array readback audit. The
[optical-control feasibility note](next_control.md) documents a further
public-data route and the checks still needed before any new model run.

## Interpretation limits

The interventions change how a model samples a known reference surface.
They do not establish a true 3D location for the ink: annotations describe
surface columns, and an ink feature at an unknown normal offset may project
into another column after a direction or anchor change. Moving anchors also
changes reference depth and surrounding context. The off-diagonal cells help
distinguish these constructions, but do not isolate every anatomical or
network mechanism.

The 17 cores come from one spatially related, repeatedly evaluated author
region. Parent results informed the angle set and primary emphasis. Any
favourable conditional contrast requires independent validation before use
as a mesh correction or a negative-screening guarantee. This experiment
neither completes Q33's dot-row/readability question nor authorizes work on
unread candidate locations.
