# Q33 — native 3D sampling-tube tilt, protocol frozen before detector inference

Date: 2026-11-24. Lane: `chatgpt/studio-oblique-stress`, based on integration
commit `2d0d77f673e3e35dce73e60be12249d73e99df53`.

## Non-overlap and scope

This implements the still-unexecuted out-of-plane part of Q33. It does not
repeat adaptive049 in-plane shear, Q10 XY rescaling, adaptive044/045 polar
winding, or the occupied 0211 fit/regrowth and positive-gate lanes. It uses the
released author841 donor annotation control. This is an integration and
sensitivity control, not independently certified legible text or an unread
scroll result.

## Frozen experiment

`firstletters-colab/studio_oblique_stress/` reconstructs the author's curved
surface directly from native CT, then rotates the entire local sampling tube
around the reference surface centre, including both anchors and normals.
There are 13 arms: identity and +/-5,10,20 degrees about each of two positive
local chart tangents. The true curved inverse tube coordinates return each
prediction to the same original reference-surface labels. Those labels are
surface-column annotations, not known 3D ink voxel locations.

The annotation-only panel is all11 qualifying mixed cores, plus3 pure-ink and
3 pure-background cores selected by deterministic within-class maximin spatial
spread. Same17 cores in every arm; no geometry-driven core replacement. Model,
normal sign, depth order, threshold and normalization remain fixed.

An initial geometry-only preflight required >=95% known-pixel retention in
every core. Four cores failed (minimum91.40625%); it stopped before model load.
The failure receipt is retained. The revised policy keeps >=95% overall and
>=90% per core, with the same0.01-native-voxel physical residual, selected
cores, arms and intersection mask. This coverage threshold was chosen after
geometry inspection and before any detector scores; it is not an untouched
prospective geometry protocol. Retention is96.8339587% overall:8,217 ink and
8,299 background pixels. Excluded known pixels540; original unknown pixels352.

All17 identity inputs reproduce the released uint8 patch bytes exactly:
4,456,448 compared values. Every positive-weight native CT interpolation corner
is supported. The seven independent analytic geometry oracles pass, including
an explicit normal-seam gap that must abstain.

Preparation/protocol SHA256:
`d56864f88ba32302344c1858f9a53cf4c1f203b64a7b28fab980105535793a80`.
No detector forward has run for this experiment at the time of this note.
The identity probability gate compares the exact historical17-core subset
on the same common mask, before submission of any tilted forward.

The primary descriptive magnitude is AP loss greater than0.05; secondary
magnitudes are recall loss greater than0.10 or background FPR rise greater
than0.03 at threshold0.5. No ensemble, best-angle selection, pixel IID interval,
readability conclusion or production correction is planned in this experiment.

## Open questions

- How much AP and recall survive an actual geometric tilt after physical
  label correspondence and source support are preserved?
- Do failures differ between the two local chart axes and the pure controls?
- Do later independent text controls reproduce any observed sensitivity?

## Next actions

- ChatGPT: run the frozen CPU model only after this protocol commit exists;
  first enforce historical identity reproduction, then all204 tilted forwards.
- Independent reviewer: verify inverse readback, label/support denominators,
  probabilities and metrics without importing the experiment scorer.
- Other lanes: keep their existing fit, gate and transfer ownership; results
  will follow in a new immutable note and a draft PR to the integration branch.
