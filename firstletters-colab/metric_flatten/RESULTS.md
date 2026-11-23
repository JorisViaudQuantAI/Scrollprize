# Measured result: correction works locally; whole-output replacement fails

Date: 2026-11-23. Public, already-announced PHerc. 306B control only.

**From this experiment:** the unchanged published ARAP calculation reproduced
the released float32 XYZ arrays **exactly**, with an identical validity mask.
Its large local distortion correction also survived inverse sampling through
the original source triangles. However, the new sampler retained **97.3905%**
of the wider baseline support, below the frozen **99%** gate. **The experiment
therefore fails the adoption gate.** The 99% threshold has not been reduced.
See the [complete receipt](results/geometry_result.json) and
[frozen protocol](https://github.com/JorisViaudQuantAI/Scrollprize/blob/22c31018be255874601b5944459757a43ff36a09/firstletters-colab/metric_flatten/protocol.json).

This is an implemented, measured geometry result. It is not a new ink reading,
an improvement to an ink model, or evidence of prize qualification. The ARAP
optimizer and its original correction are the winners' work. Our contributions
are the portable replay, explicit input contracts, original-triangle query
provenance, adversarial support test, and measured failure analysis.

## 1. What was actually run

**From source and measured:** the control reconstructs the real pre-ARAP
surface from the [published 306B inputs](https://github.com/SirProdigle/pherc0306b-first-letters/tree/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/code/surface/inputs),
preserving touch-up arithmetic and the float32 TIFF roundtrip. It then runs the
publisher's 60-iteration solver, fold weight 0.02, threshold 1.25, fixed vertex,
region and rigid anchor alignment. B uses the original Delaunay inverse
interpolation; C uses the **same** solved UV coordinates and original surface
triangles. No CT or ink-model output was used.

| Measurement | Result |
| --- | ---: |
| Source vertices selected | 77,185 |
| Original source triangles | 153,224 |
| Source connected components | 1 |
| Isolated vertices silently discarded | 0 |
| Degenerate source 3D triangles | 0 |
| Source/body transfer, once | 14,205,351 bytes in 17 requests |
| Completed cached run | 49.532 seconds |
| ARAP within that run | 18.092 seconds |
| Completed-run peak RSS | 362,446,848 bytes |
| CT requests / CNN forwards | 0 / 0 |

The protocol was published at commit
[`22c31018be255874601b5944459757a43ff36a09`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/22c31018be255874601b5944459757a43ff36a09)
**before the real arrays were acquired or inspected**. The defect and the
publisher's summary before/after results were already known: this remains a
retrospective engineering control, not an independent efficacy test.

An initial invocation rejected a cache/output location inside Git ancestry
before acquisition. The first actual acquisition/compute run subsequently
stopped without a final receipt; the cause is unknown. Its last saved stage
was at 94.404 seconds. The
[interruption record](results/interrupted_run.json) preserves its failed
retention gate. An unchanged run from the verified cache completed without
further network requests. The three completed large NPZ artifacts of the first
attempt are [byte-identical to the completed run's counterparts](results/interrupted_artifact_comparison.json); no result was
selected by changing a parameter. The original inputs and both attempts'
records are retained outside this public repository.

## 2. Controlled local benefit, credited to the published method

**From the measured receipt:** the fixed published right group contains 1,150
cells with all four corners valid in A, B and C. The table uses exactly those
target-grid cells in every arm.

| Metric | A: before ARAP | B: published inverse interpolation | C: original-triangle inverse interpolation |
| --- | ---: | ---: | ---: |
| Median absolute vertical stretch error | 13.6008% | 0.2971% | 0.2971% |
| Cells with either axis outside ±5% | 100% | 0% | 0% |
| 95th-percentile absolute shear | 2.804° | 0.891° | 0.891° |

**My inference:** the control substantiates a cheap correction for the kind of
metric distortion raised in [Claude 017](../../collab/claude-017.md) and
[ledger Q10](../methods_ledger/09_open_questions.md). It does not show that our
sampler improves the already-corrected letters: B and C give identical metrics
in this group. The local relative reduction in median absolute vertical error
is 97.82%, measured against A. It must not be called a 97.82% reading improvement.

On the **entire identical original-triangle population**, the fraction with
either metric singular value outside ±5% fell from 58.5124% to 29.9927%.
Source-area-weighted fractions were 61.7631% and 34.7459%. The source 3D area
was unchanged at 2,475.8506 mm². These full-population statistics keep the hard
faces rather than improving the denominator by dropping them. They also show
that the larger context is far from uniformly isometric.

Target-grid positions are not exact material correspondences after
reparameterization. The original-triangle measurements and the common-cell
measurements answer different questions; neither is optical ground truth.

## 3. Why the wider-output gate fails

**From the receipt and saved rejection bits:** B contains 74,343 valid target
nodes, while C contains 72,403. The 1,940 lost nodes partition as follows:

| Reason at the target query | Nodes |
| --- | ---: |
| No original-triangle coverage | 1,039 |
| Ambiguous positive-face overlap, without negative-face coverage | 669 |
| Ambiguous overlap and negative-UV-face coverage | 232 |
| Total B-only nodes | 1,940 |

The optimizer produces **1,334 negatively oriented UV triangles**, compared
with zero initially. Its fold weighting reduces their influence on the metric
objective; it does not enforce an injective chart. C rejects ambiguous and
negative-covered queries, rather than selecting whichever triangle a new
Delaunay tessellation happens to supply. The reported loss is a sample-count
loss, not a measurement of missing anatomical papyrus area.

**From the receipt:** all **20,172** nodes in the published padded claim-region
crop remain valid. That crop is 82 × 246 nodes and includes 2 mm of context; it
is not the exact qualifying claim rectangle. Keeping this crop intact does
not override the independent whole-output gate.

The common B/C nodes also deserve attention. Their median XYZ difference is
zero, but the 95th percentile is **0.743742 source voxel = 6.42593 µm**, and the
maximum is **519.916 voxels = 4.49207 mm**. Differences exceed 1, 20 and 100
source voxels at 3,068, 206 and 58 common nodes respectively. **My inference:**
agreement in the known letter group cannot validate interpolation throughout
the wider mesh. A large difference does not by itself prove which XYZ is
anatomically correct; C has the narrower, checkable claim of interpolation
through the supplied original surface triangles.

**From the [post-run support analysis](results/support_analysis.json):**
rebuilding the baseline Delaunay tessellation from saved UV again reproduces B
exactly. All **45,189** common queries using an original source triangle agree
with C exactly. All 58 discrepancies above 100 voxels, and 202 of the 206 above
20 voxels, use Delaunay triangles spanning more than one original grid step.
The remaining four above 20 voxels use a different diagonal within one source
cell. Every one of the 1,039 unsupported baseline queries uses a nonlocal
original-grid connection. This attributes the discrepancy to connectivity,
with no new optimization or anatomical-truth claim.

The [scientific figure](results/geometry_summary.svg) shows the full-field
losses, common-node displacement and fixed-region metric correction. This is
a public-control geometry visualization, not an unread-scroll ink image.

## 4. A real support-preservation difference under a designed perturbation

**From the frozen test:** removing a 7 × 7 square of source vertices around
the published fixed vertex removes 49 vertices and 126 original triangles.
There are 36 strictly interior test queries whose former source triangle has
all three vertices removed.

| Inverse sampler after the deliberate removal | Interior queries still filled |
| --- | ---: |
| Rebuilt Delaunay interpolation | 36 / 36 |
| Original-triangle interpolation | 0 / 36 |

This demonstrates a functional difference in preserving missing source
support. The missing square was deliberately constructed on real geometry;
it is **not an observed manuscript hole**. No optimizer was rerun for this
comparison. It does not establish better ink prediction.

## 5. Remaining contracts and decision

All other frozen gates pass: exact published replay, accepted-sample
provenance, local metric correction, identity reconstruction, and the designed
support-removal check. Maximum float64 barycentric reconstruction error is
3.64e-12 voxel. Identity reconstruction of the published grid is exact.
Float32 serialization is separate: maximum displacement is 0.0005774 voxel,
about 0.004989 µm.

**From the [independent saved-array review](results/independent_review.json):**
58 checks pass, including all 23 artifact hashes/byte counts, exact B/target
arrays, C barycentric XYZ and UV reconstruction, signed face orientation,
right-group metrics, support accounting, coordinate differences and all
serialized XYZ/masks. This review uses NumPy/Pillow directly and does not import
the experiment's numerical functions. It does not rerun the optimizer or
rasterizer. Identity output was not saved, so its independent identity check
is limited to expected source-triangle incidence and receipt consistency.
The removed-support counts are verified against the saved arrays without
rerunning either interpolator.

The [generic CLI integration check](results/cli_smoke.json) also passes on a
generated 11 × 11 planar grid: all 121 output nodes and float32 coordinates
match the input, including source metadata with float32-rounded scale. This
checks the runnable handoff, not manuscript effectiveness.

**Decision:** use the code as a reproducible geometry diagnostic and a guarded
experimental exporter. **Do not adopt C as an unattended replacement, and do
not start a broad unread-scroll comparison on the strength of this result.**
The frozen decision permitted a public CT/ink comparison only if every gate
passed. That condition was not met, so no such comparison was started here.

There is also a renderer integration limit: sampled-node provenance does not
certify every point between nodes. Regular-grid interpolation can cross a
subpixel slit even when its corner nodes are individually valid. Before an
export is used for reading, its renderer must honor the original support at
the cell/edge level, as well as the invalid-node mask. No output here claims
to solve that remaining problem.

The current [Python renderer](../ink/render_tifxyz.py) checks an interpolated
node-validity mask and then interpolates XYZ and normals. It does not consume
the original-triangle provenance. **My implementation recommendation for
Codex:** evaluate original triangles at the requested render samples, or
validate the intervening rendering cells against that source support before
using interpolated XYZ. A node-only mask is insufficient for the subpixel
case. This recommendation has not been implemented or measured here.

## Open questions

- **Claude Code:** can the new standard flattening path expose signed triangle
  areas, overlap/support counts and true metric summaries before inference?
  This is the useful continuation of Q10/Q49; the present experiment did not
  measure every fitted winding or detector scale tolerance.
- **Codex:** can the renderer consume original-triangle provenance or a cell
  support mask so interpolation and normal estimation never bridge invalid
  source support?
- **Gemini:** can the source-support check be used as a diagnostic alongside
  your current filters, without converting either into an uncalibrated ink veto?

## Next actions

- **ChatGPT:** retain the failed gate, publish the source-only and saved-array
  review evidence, and deliver the reproducible control outputs.
- **Claude Code:** keep Q7/Q1 running; apply geometry diagnostics to the existing
  flattening implementation before requesting another rendering comparison.
- **Codex:** identify the smallest renderer change needed to enforce source
  cell/edge support; no organiser or upstream submission is authorized here.
- **Owner:** review this as a possible documented toolkit improvement. A prize
  submission remains your decision; this result is not a qualifying reading.
