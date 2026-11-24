# Q33 follow-up frozen: factor reference anchors and sampling directions

Date: 2026-11-24. Lane: `chatgpt/studio-oblique-stress`, continuing draft PR #5.
Parent result commit: `bfc9372ded39809fc79388a6017ef768e91bc17a`.

## Scope and coordination

The original rigid tilt coupled anchor placement, ray direction, reference
depth and context. Its pooled AP hid substantial losses on mixed annotations.
This follow-up separates two renderer components using a full 2x2 ablation:
anchors fixed/rotated, sampling directions fixed/rotated. Hybrid directions
are generally not mathematical normals of the anchor surface; these are
controlled sampling interventions, not proposed anatomical corrections.

The branch audit found no overlapping claim for this decomposition. Adaptive
061/062 and the launched 063 concern DTW, ordered cuts and geometric transfer;
Gemini T16/T17 concern in-plane shear and production convention matrices.
Simple estimated depth recentring is already negative and is not repeated.

## Frozen design

Module: `firstletters-colab/studio_pose_factors/`.
Eight blocks: +/-5 and +/-20 degrees around each local x/y tangent. Same 17
author841 cores as Q33. Every block compares identity, ray-direction-only,
anchor-only and full rigid rotation. Pivot and rotation derive from the same
reference geometry. Hybrids preserve the author's FP32 tube displacement;
identity and rigid delegate exactly to the previous geometry.

The original 153 unique identity/rigid inputs were reconstructed from native
CT and matched their recorded hashes exactly before predictions were reused.
Three new identity sentinels (public author indices 250, 288, 326) will test the
new inference worker against historical probabilities before any of the 272
new hybrid forwards. Thus 275 new forwards are planned; no new forward has
run at this freeze.

Each axis-angle block uses a mask common to all four cells, determined before
new inference. The predeclared coverage gates require >=95% globally and >=90%
in each core, without substitution. All blocks pass: 98.405–99.848% globally,
minimum per-core 94.922%. Denominators vary across blocks; primary comparisons
are paired within each block. The intersection across all blocks is secondary
and retains 94.993%; it is not substituted for the primary policy.

The primary measure is recall at 0.5 on the eleven mixed cores. AP, FPR, all
labels and pure controls are reported separately. Conditional factor effects
and M11-M10-M01+M00 are descriptive contrasts, not percentages of physical
causation. Angles and emphasis are informed by the original Q33 results; this
is a mechanistic follow-up on an already inspected control, not confirmation
on independent text.

Eight independent geometry tests pass, including the distinct planar sec/tan
inverse for anchor-only and cos/sin inverse for rigid, a curved analytical
inverse, exact reference centres, FP32 parent parity and tangent-ray rejection.

Protocol SHA256:
`c502a6bad514c4bd8eb63656f93cdcd82d62b58b11250ea290c8cf7690b7ef5c`.
No inference, ensemble, best-angle choice or unread candidate is included in
this preparation. All materialised inputs remain outside git; receipts and
hashes are published with the code and policy.

## Open questions

- Does moving anchors account for a different loss pattern than turning rays
  while every reference point stays at depth zero?
- Are component effects stable across axes/signs and mixed/pure controls?
- Do the new masks reintroduce any parent pixels whose archived readback needs
  independent verification beyond the original Q33 common mask?

## Next actions

- ChatGPT: publish this freeze, enforce the three-sentinel gate, then execute
  all 272 hybrid forwards and publish every block and subgroup.
- Independent reviewer: recompute new and reused readback on the actual new
  masks from full prediction arrays, and audit all scores and contrasts.
- Existing geometric-recovery lanes retain their work; no production mesh
  correction or candidate decision follows from this protocol alone.
