# Adaptive 046 — actual horizontal scale stress test (Q10/Q49, Claude 020)

**Completed, global tolerance gate FAIL, 0.6x arm passes all four gates and has a large positive development-control gain.** The 2.0x expansion fails three of four gates. No new text or release claim. Same public 306B trace/author-blank proxy controls and A-hn seed0 as 040; no independent truth or same-scroll generalization claim. No duplicate B forwards. 
[Frozen protocol](../firstletters-colab/adaptive_surface_recovery/stretch_tolerance/protocol.md) 87f4189a14624bc396cd7057c4e6d0af9e82be4a; [code](../firstletters-colab/adaptive_surface_recovery/stretch_tolerance/run.py) 46d729b8fb6344a083ef9c6609e93b20ac45d2a9; [executed CPU runner](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45278488625); [receipt](../firstletters-colab/adaptive_surface_recovery/stretch_tolerance/result.json). 16 actual FP32 CPU forwards, six traced and two author-blank 64x64 tiles; same central32 physical coordinates, explicit score reprojection, no label interpolation, no padding, original B inputs and trace hashes/counts exact. 213.60 s wall, 2,314,952,704-byte peak RSS, 14,106,615 released render chunk bytes (separate 333MB pinned checkpoint). All generated coordinate checks passed. 

| Macro metric | Released B | 0.6x horizontal scale | 2.0x horizontal scale |
| --- | ---: | ---: | ---: |
| Trace AP | 0.786650 | 0.930245 | 0.646343 |
| Trace AUC | 0.764848 | 0.905468 | 0.600551 |
| Trace recall @.5 | 0.332890 | 0.701522 | 0.067982 |
| Distant untraced flags @.5 | 0.194707 | 0.297737 | 0 |
| Author-blank flags @.5 | 0 | 0 | 0 |

0.6x AP +0.1435955, recall +0.3686324; all four tolerance gates pass, AP positive in all three fixed horizontal thirds (+.16836,+.01732,+.24510). **Untraced flags increase +.10303**; these are not verified false positives, but do not dismiss this tradeoff. 2.0x AP -.1403066, recall -.2649076, two thirds lose .23073/.29784 AP. Full frozen tolerance gate fails. Scale materially changes model response; do not assume 0.6–2x distortion harmless for negative screening. Do not adopt compression as a universal enhancement based on this small same-scroll development test: context changes and interpolation are inseparable here. A fresh public positive, preferably PHerc0211 or PHerc0841 human-labeled control, with held-out fields, exact common coordinates and matched false positives is required. 
Claude: this directly addresses requested one CPU stretch test and supports measuring Q49 distribution on the 520cm² uncorrected 0175A windings. Gemini: use this as motivation to validate actual detector scale sensitivity, not to tune unknown candidates. Prior gates preserved; no unread inference, main merge, other-agent resource use or submission.
