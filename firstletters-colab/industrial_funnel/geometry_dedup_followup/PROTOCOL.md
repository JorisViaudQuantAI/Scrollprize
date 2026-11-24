# Geometry correspondence is a proposal, not an inference cache key

Frozen on 2026-11-24 before loading the twelve mesh arrays for this follow-up.
This CPU experiment follows Gemini-009; it does not repeat its mean overlap
claim as a new finding. No CT, model, ink label, unread papyrus, or GPU is used.

## Question and fixed inputs

Does the published normal-correspondence predicate establish identical input
sampling, and how often do its accepted correspondences disagree with an
independent geometric reference? The reference is the public PHerc. 0211 winner
surface, not anatomical truth or an independent ink annotation.

Use all twelve public fitted/snap/flat w033–w036 grids, pinned to Claude
`ab5b297de8cf696e098852ccb9b3be766317acb2`. Verify every input against the
hashes in adaptive-067's result at adaptive commit
`2cba2f7a8a316ac9e1d42ed939d226ec255e5d22`. Use Gemini's exact helpers at
`da94312727c7e95f7f7fdc2d7481d502265434c5` (source SHA-256
`7a43443989eaaa08d0c9bf1f8f9b98fac95bf768fbaa48ccebd3772589c57bec`).
Reference ray implementation and code-grid archive are pinned to adaptive-067,
with source SHA-256 `1130ba0bb36a6b877dd6e4f410c12367b247c6633ecea014c0750b9ce2676962`.

## Measurements fixed in advance

1. All 36 directed non-self winding pairs, plus 12 self-pairs. No selected-pair
   reporting only. Preserve the upstream criterion: target bilinear factor 4,
   absolute normal displacement <=3 native voxels, lateral displacement <=4,
   absolute normal dot product >=0.7. Normals and source denominators exactly
   follow Gemini's implementation. Record per-vertex masks and source IDs.
2. For each accepted correspondence, align the sign of the target normal to
   the source normal. For symmetric offsets t in [-8,+8] native voxels, compute
   max ||(Q+t*nB) - (P+t*nA)||. The norm of this affine difference is convex, so
   its maximum is attained at an endpoint. This is a geometry-only diagnostic
   of a common 17-position depth span, not the renderer's complete sampling
   contract, not a 16-layer byte comparison, and not a measured ink error.
3. Use the exact adaptive-067 finite-ray classifier against the public winner
   at both matched positions, with the same respective normals used by G4.
   Recompute source labels when needed rather than interpolating categorical
   labels or silently substituting the area's averaged normal. Also cross-tab
   the source label against the published 067 label on that original vertex.
   Codes: 0 invalid, 1 no hit, 2 ambiguous, 3 oblique/invalid normal,
   4 near (<=3 vox), 5 intermediate, 6 far (>=6 vox). Unknown never means absent
   ink. Count near-to-far and near-to-unknown separately; neither is ground-truth
   sheet misidentification or confirmed missed text.
4. Explain any self-pair failures. Test the minimal sampling correction of
   including every usable original target vertex alongside the bilinear samples.
   This addresses boundary support only; it does not establish scientific
   equivalence of two distinct surfaces. Report its effect on all pairs.

## Controls, interpretation and budget

Synthetic controls cover identity with missing grid cells, parallel offset
surfaces, reversed normals, and nonparallel normals. Check exact parity with
the upstream summary helper and adaptive-067 reference coordinate hashes.
Store source hashes, full per-pair aggregates and compact numeric maps.
Visually inspect an atlas of actual source classifications, accepted matches,
and reference disagreements. Keep all outcomes, including zero disagreements.

No threshold tuning. No whole-chart deletion policy is inferred from a local
point predicate. A cache hit requires the actual ordered input bytes and the
full renderer/model/context contract; these data are not available in this
experiment. Approximate pruning still needs a recall audit on native maps.
CPU cap: ten minutes; downloaded public geometry <=16 MiB; no persistent CT.
