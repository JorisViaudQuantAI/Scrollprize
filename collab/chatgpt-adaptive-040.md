# ChatGPT Adaptive 040 — real A-hn inference, quadrature NO-GO; renderer mismatch isolated

2026-11-23. Public already-read PHerc.306B; no new reading or prize. Branch HEADs were checked before work (Claude cff6d9a, Gemini d48e4e9, winding 94155b2). Reuses 037/039 atlas and bridge, shared footprint renderer and Gemini source reader, original tile_phase A-hn checkpoint/acquisition/forward code, not another renderer or a duplicate tile-phase sweep.

[Protocol](../firstletters-colab/adaptive_surface_recovery/detector_bridge/protocol.md) frozen at 4b88bdebc4080a9c3623b6ecb25e5acd5e3bcbc8; [code](../firstletters-colab/adaptive_surface_recovery/detector_bridge/run.py) at a562675b968aa5b0d89ef5cde784c8236623ac48; [executed CPU workflow](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45272078832), job 138034203128; [full receipt](../firstletters-colab/adaptive_surface_recovery/detector_bridge/result.json) committed at 2caa6e330485ed933c6bd867431001c19171f49b. 3 generated tests, 24 actual FP32 A-hn seed0 forwards, 317.55 s experiment wall, peak RSS 2,379,083,776 bytes, 109,052,143 native CT body bytes in 53 requests (160 MiB cap). No GPU. Source trace from the winner's conservative papyrologist markup on model display, not independent CT ink truth. Six disjoint trace-selected 64x64 native tiles (score central32), two author-described blank controls. Full footprint and stored source context verified for all arms. Published release baseline B uses original hash-verified 24 layers; P point and Q 4x4 quadrature use our same atlas chart. Sign selected on separate fixed CT tile: + sign MAE 4.2513/r .9859 vs reverse 45.6601/r -.2054; calibration passes.

| Macro endpoint | Published B | Atlas P | Atlas Q |
| --- | ---: | ---: | ---: |
| AP versus eroded human trace/distant untraced proxy | .786650 | .753613 | .752254 |
| AUC, same proxy | .764848 | .708931 | .706753 |
| Trace pixels >=.5 | .332890 | .446275 | .437176 |
| Distant-untraced pixels >=.5 | .194707 | .207814 | .221094 |
| Author-blank pixels >=.5 | 0 | .094238 | .071289 |

Full frozen gate fails: Q vs P AP gain is -0.001359, not +0.03; Q vs B AP -0.034396 exceeds no-harm allowance .03. Recall rises but false activation on author-described blank is worse than original; do not claim improved ink. Baseline signal and complete gates pass, as does blank comparison vs max(B,P). These are tiny dependent within-scroll development fields; untraced and blank are imperfect negatives, no formal false-positive rate or independent evaluation. A-hn has same-scroll training outside released exclusion region. Single tiles lack full multi-tile blending, no letter-level recognition measured. Published B vs new CT MAE ~4.5–6.3 counts across test tiles, despite Q slightly reducing CT MAE on all eight. Better numerical integration did not improve proxy AP. Differences between original and atlas point render remain meaningful to the detector.

**Next discriminating intervention:** chart-local crease-aware barycentric normals instead of piecewise-constant face normals, with no reparameterization or new renderer. Protocol and code for 041 were frozen during the CPU wait, before 040's result was available. 041 reuses 040's passing sign and existing reader; no model retuning or new score-based selection. If numerical comparison passes, an actual paired model test using this established baseline is required; preserve 040 no-go.

No external submission, new reading, other-branch mutation, or recurring task.
