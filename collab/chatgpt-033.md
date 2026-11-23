# ChatGPT 033 — fixed-budget depth hypotheses do not beat a uniform sweep (Q8/Q33)

Date: 2026-11-23. Branch: `chatgpt/adaptive-surface-recovery`. No candidate. Owner requested sequential work in the same session rather than a scheduled task. The periodic task is disabled.

[Protocol](../firstletters-colab/adaptive_surface_recovery/depth_hypotheses/protocol.md) frozen in 665fbf5f4e26e670b703886a826e9d29b5fd6354; [code](../firstletters-colab/adaptive_surface_recovery/depth_hypotheses/run.mjs) committed in feadec275b34185a8e8007d24ab0ec96a84fdbbf before numerical evaluation; [all 204 cases](../firstletters-colab/adaptive_surface_recovery/depth_hypotheses/result.json) in 043e7557bbebbd38677b95e9e9b2c0c59a9b035c.

Direct S3 acquisition was blocked by DNS in this environment. We used the exact archived native 28-plane CT mean profiles from the published PHerc0841 control, 12 calibration cells and 12 evaluation cells, 17 artificial offsets per evaluation cell. This is a retrospective geometry-proxy test: no new CT, no model forwards, no evidence of ink recovery. The original stored plane is not physical ground truth.

At identical three-window count, the fixed offsets [-6,0,6] versus adaptive posterior-coverage triplet and single best correlation:

| Metric | Fixed 3 | Adaptive 3 | Single best |
|---|---:|---:|---:|
| Within 1 plane | 52.94% | 59.31% | 27.45% |
| Within 2 planes | 88.24% | 74.02% | 35.78% |
| Mean nearest error (planes) | 1.412 | 1.623 | 5.515 |
| Worst nearest error | 3 | 6 | 16 |

82 paired wins, 87 losses, 35 ties. The frozen 10-point improvement, 90% within 2 and no bad cell-specific decline all fail. Two cells fall to 2/17 within one plane. Template likelihood focuses compute on the wrong modes in some cells; the robust fixed sweep is preferable in this comparison, not a proved optimal depth strategy. Do not tune on these profiles.

Reproduce from branch root: `node firstletters-colab/adaptive_surface_recovery/depth_hypotheses/run.mjs firstletters-colab/depth_recentering/results/results.json`. CLI checks source blob. Actual execution used the identical evaluate() function in a V8 isolate (119 ms); Node wrapper was not exercised. All 204 cases and basic self-tests recorded.

## Open questions
- Claude: can the pending Q7/Q1 jobs provide a real positive missed at the current depth/order, with a public releasable control?
- Codex/Gemini: preserve both raw and snapped alternatives; do not infer sheet identity from a bright peak or adopt this triplet selection.

## Next actions
- ChatGPT: close this retrospective depth-template line, pivot to a distinct tractable Q21/Q37 review problem using archived public controls; do not rerun the same geometry sweep.
- No other agents' jobs, branches, prize submissions or unpublished candidate data were modified.
