# Q33 pose-factor follow-up: completed, audited, all eight blocks retained

2026-11-24. Follows immutable [chatgpt-oblique-003](chatgpt-oblique-003.md).
Branch: `chatgpt/studio-oblique-stress`; draft PR #5, targeting
`claude/firstletters-colab`.

## New contribution

The previous rigid tilt moved anchors and sampling directions together.
The new 2x2 native-CT experiment separates those two sampler components
on the same 17 author841 cores, for +/-5 and +/-20 degrees around each
local chart tangent. Hybrids need not use normals of their anchor surface.

The protocol, native input/reuse hashes, masks and numerical gates were
frozen before new inference at
`e2641df5f799bfc8dc7f8cd0e065520034db694e`. Protocol SHA256:
`c502a6bad514c4bd8eb63656f93cdcd82d62b58b11250ea290c8cf7690b7ef5c`.
No core, angle, gate or scored mask was changed after new model outputs.

**275 new forwards completed in 646.061 seconds on CPU:** 3 identity
sentinels followed by 272 hybrid inputs. The 153 unique parent outputs
were reused after reconstructing their exact input hashes from native CT.
The three fresh sentinels had zero probability difference.

## Main result

Primary outcome: pooled ink recall on the 11 mixed cores at fixed threshold
0.5, with a common four-cell mask within each axis/angle block.

| Intervention | Blocks losing >10 recall points versus paired identity |
| --- | ---: |
| Sampling directions only | 3/8 |
| Anchor positions only | 8/8 |
| Both / rigid | 8/8 |

At **y -20 degrees**, anchor-only and rigid each have **0/5,291 true-positive
mixed ink pixels** and **0/5,848 false-positive mixed background pixels**.
Direction-only retains 56.2276% recall, versus identity 67.4353%.
Its mixed AP nevertheless rises 0.638070 -> 0.699044. This ranking gain is
real on the mixed panel but does not preserve fixed-threshold detections.

The direction effect conditional on already rotated anchors ranges from
-4.418 to +4.869 recall points. At original anchors it can lose 23.674
points. Interactions are descriptive and can reflect a recall floor; they
are not percentages of a physical failure mechanism explained.

**Equal angles do not equalize spatial displacement:** direction changes
act over the 16-plane depth extent, while anchor changes have a lever arm
over the 128-square context. Anchor motion still combines depth, context
and chart readback. These results support separate logging/testing of the
two operators, not an intrinsic normalized sensitivity or a mesh correction.

## Coverage and verification

- Every primary block passes the frozen >=95% global / >=90% per-core
  known-label retention gates: observed global 98.4053%–99.8476%, minimum
  core 94.921875%.
- Masks differ between blocks. All primary contrasts are paired inside
  each block; their identity scores and denominators are reported.
- The predeclared secondary global intersection retains 94.992964% of
  known labels. It is descriptive, not the primary gate population. The
  3/8, 8/8, 8/8 recall flag counts persist on that stricter intersection.
- Eight geometry tests pass, including independent planar and cylindrical
  analytic oracles, FP32 compatibility and degenerate/support cases. One
  independent auditor in full and portable modes passes 8,436 scalar checks
  in each, maximum metric error 3.33e-16.
- Manual readback from full new and historical probability maps is bitwise
  exact on new block masks: 270,854 values each. This includes 7,356
  historical values outside the parent's older scoring mask, with repeated
  uses across blocks counted explicitly.
- The full audit checks materialized sources and receipts; it does not
  independently repeat CT acquisition. The runner did the native
  reconstruction. Portable mode checks archive arithmetic and provenance
  represented by the archive, not original detector execution.

## Files and reproducibility

See [the completed report](../firstletters-colab/studio_pose_factors/RESULTS.md),
[the runner and protocol](../firstletters-colab/studio_pose_factors/README.md),
and [the independent auditor](../firstletters-colab/studio_pose_factors/audit_results.py).
The 1,924,035-byte [score archive](../firstletters-colab/studio_pose_factors/evidence/scored_arrays.npz)
has SHA256 `ba7e0f6a1032f8eb00cdb83d75b07613ce5c8cd496880943aaaf5652e3c372c2`.
All 32 cells, their 128 group-level rows, per-core scores, paired contrasts,
secondary global scores, figures and receipts are retained.

The branch audit still finds distinct work in adaptive DTW/ordered cuts,
gap-aware transfer and multi-anchor consensus, and Gemini snap/stride/order
controls. This lane does not duplicate them. No candidate location, new
reading or production policy is inferred from these public author controls.

The separate [next-control note](../firstletters-colab/studio_pose_factors/next_control.md)
locates official Frag4 native CT, native PPM and optical annotations and
checks the published checkpoint ancestry. Frag4 is excluded from declared
gradient updates but exposed during development and shares the PHercParis1
parent manuscript with trained Frag3. It is an external optical control
relative to author841, not independent validation. No Frag4 forward is
claimed; native identity registration, scale, uint16 normalization and
depth orientation still need closure.

## Open questions

- Which parts of the anchor effect arise from depth versus context or
  chart correspondence, at a controlled physical displacement?
- Can the Frag4 native identity renderer and normal/depth convention be
  established without choosing them using detector scores?
- Does the direction-only AP gain recur under a separately frozen optical
  control protocol while preserving the desired operating-point recall?

## Next actions

1. Other agents / owner: review the report and portable replay before
   integration of PR #5 into the shared branch.
2. ChatGPT lane: complete the four native Frag4 acquisition/render checks
   in `next_control.md` before defining a new model experiment.
3. Surface-recovery lanes: retain separate anchor/direction metadata and
   mixed-core recall in proposed evaluations; avoid treating a quiet map
   or aggregate AP gain as sufficient evidence of preserved ink detection.
