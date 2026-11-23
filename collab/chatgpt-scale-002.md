# ChatGPT scale002 — actual Q10 CPU result and failed scale ensemble

Consumer: Claude/adaptive surface QA and detector-screen interpretation. Branch: `chatgpt/studio-scale-stress`. This continues immutable [scale001](chatgpt-scale-001.md); it does not replace the Q12 winding audit. The experiment was frozen in commit `cb3d5f44f7a82b3fcaa4e7e62f6e707f48b7c9ba` before model inference.

## Completed

We downloaded and hash-verified 1.018GB of public Studio checkpoint/author PHerc0841 data and ran265 actual CPU forwards: all 53 labelled cores, five fixed conditions, same physical labels. All265 contexts have measured prepared-tile support; identity matches the existing adapter exactly. Two CPU workers completed in638.18 seconds. No new training or unread data.

| Condition | AP | Change | Recall at 0.5 | Background FPR at 0.5 |
| --- | ---: | ---: | ---: | ---: |
| Identity | 0.753604 | — | 67.87% | 19.50% |
| X ×0.5 | 0.693695 | −0.059909 | 58.44% | 21.15% |
| X ×2 | 0.737682 | −0.015922 | 54.85% | 12.32% |
| Y ×0.5 | 0.612890 | −0.140714 | 41.95% | 17.21% |
| Y ×2 | 0.753363 | −0.000242 | 52.47% | 7.69% |
| Uniform mean-five | 0.736428 | −0.017176 | 52.08% | 14.77% |

Both compression arms cross the prospective AP-loss>0.05 harm flag. Y enlargement demonstrates why near-constant AP alone does not establish unchanged fixed-threshold sensitivity. **Mean-five fails** its AP-gain and median mixed-core gates; do not advance it. All conditions keep15,791 ink/29,558 background labels;8,923 unknowns remain excluded.

Readback interpolation alone changes AP by at most0.00004673, much less than the compression losses. We do not subtract it or claim to isolate architectural scale invariance: anti-aliasing, physical field of view and network context change together. Image X/Y are not asserted to follow text lines. The author donor holdout is exposed, spatially compact and not a new-scroll benchmark.

## Deliverables and verification

- Full account: `firstletters-colab/studio_scale_stress/RESULTS.md`; all gates, per-core metrics, runtime and source receipts remain available.
- 17 sampling tests pass, including6 separately authored analytic checks.
- Independent rational-coordinate/sklearn reviewer passes 3,029 scalar checks with maximum error3.33e-16 and bitwise identical physical prediction readback; original label population and file hashes verified.
- A 1.12 MB `evidence/scored_arrays.npz` contains all scored probabilities/labels and can be checked with `audit_scored_archive.py`, without the checkpoint or a rerun.
- Source audit `next_control.md` identifies the original PHerc0306B CT plus human tracings for a transfer follow-up. November organizer “ink maps” are three model predictions, not gold labels. Human tracing black pixels are untraced, not certified background; the newer organizer canvas also differs from the original winner raster.

## Open questions

- Q10 remains open on PHerc0343 and on the production renderer/tiling path. This is direct detector evidence that scale can matter, not evidence of a missed PHerc0175A reading.
- A new-scroll validation needs verified annotation-to-native-CT registration and defensible negative labels if AP/FPR are to be reported. PHerc0306B raw human tracing alone supplies only weak positive constraints.

## Next actions

- Claude/adaptive: consume this sensitivity result alongside existing metric/coverage QA; retain ownership of the atlas/shared-renderer model comparison. Expected output is a separately pinned public-positive check before interpreting corrected unread renders. Do not treat geometry-only PASS as a detector PASS.
- Detector lane: retain the five-scale mean as NO-GO under its frozen gate; do not tune weights on this donor and report them as independent validation.
- ChatGPT scale lane: provide the reproducible code, scored-core archive and draft integration PR. Owner retains any main merge or prize submission.
