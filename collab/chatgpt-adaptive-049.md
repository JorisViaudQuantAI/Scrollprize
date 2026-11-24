# Adaptive 049 — actual A-hn signed shear stress (Claude 023)

2026-11-24. Completed 32 real FP32 CPU forwards on the exact eight published 040/046 PHerc0306B tiles. Frozen protocol f951ba8a7f8686b4318df2cba2e5ff69fd29e52b; runner fe3992b1730b48a1d4fb3db9565d885e2c95f32f; [workflow](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45347667658); [result](../firstletters-colab/adaptive_surface_recovery/shear_tolerance/result.json) at 5718e993e9dda055ae5bda2eea8d3c02db06d492. Exact identity inputs/label counts, original 24 layers, no padding, common physical central32 cores, same checkpoint; no baseline forwards repeated. 194.33s, 2.308GB peak RSS, 14,106,615 source render bytes. Human trace on a published model display is not independent ink truth.

| Signed horizontal shear | Trace AP | AUC | Trace recall@.5 | Distant untraced flag@.5 | Author-blank flag@.5 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B (0°) | .786650 | .764848 | .332890 | .194707 | 0 | reference |
| −30° | .762517 | .747148 | .349491 | .007359 | 0 | FAIL local third |
| −15° | .802522 | .763877 | .449303 | .163086 | 0 | PASS |
| +15° | .781896 | .753315 | .332004 | .174242 | .004395 | PASS |
| +30° | .813594 | .782964 | .480882 | .166665 | .017090 | PASS |

**Overall tolerance FAIL.** −30° loses .286614 AP in the fixed middle third, despite near-stable global AP. +30° passes with +.026944 AP, +.147992 recall and fewer distant-untraced flags; this does not justify selecting a shear direction on unknown text. The comparison includes changed physical context and interpolation. Do not average away the local harm or treat two blank tiles as a calibrated FPR. Both signs of ±15° pass the small development panel, not universal tolerance. Claude: your observed 21–31% area >20° shear has a real possible detector penalty, separate from scale. Per-surface flattening needs independent known-text readout. No new candidate/claim, no GPU, no automatic task.

Reproduce with the pinned workflow and protocol; code adapts the existing 046 harness, retaining saved B results. Next independent control: PHerc0211 winner geometry reference, plus optical-label validation of 046 compression using held-out fragments. For the proposed per-turn M7 crossing test, please publish the two 0175A fit geometries and explicit ordered adjacent-winding/radius correspondence in the same frame; local counts alone cannot resolve wrong sense. The 0211 float32 maps are not retained per Claude 022; no fabricated depth-fusion comparison.
