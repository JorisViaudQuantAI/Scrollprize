# ChatGPT Adaptive 039 — PASS: atlas, shared footprint renderer and Gemini reader interoperate on native CT

Date 2026-11-23. This is a cross-swarm integration result on already-published PHerc.306B, not a new detector or new reading. The owner requested that each agent read the others' branches, avoid duplicate work, and publish every meaningful advance. The [cross-swarm index](../firstletters-colab/adaptive_surface_recovery/INTEGRATION.md) records the prior 032–038 experiments and other lanes. This note uses a lane-specific filename because shared `chatgpt-032.md` describes the separate winding-order correction.

## Dependencies and decision

- Our 037 material atlas, 038 indexed original-face point sampler; exact atlas SHA256 `43c12969b32c42b978e9d1cbb3bd70fb645a998100062d889d66228a130fd8d0`.
- Existing shared [surface_sampling implementation at f7fc619](https://github.com/JorisViaudQuantAI/Scrollprize/tree/f7fc619ec11f9dc95caa0c90fb2148cbf7c86ebd/firstletters-colab/surface_sampling). **No new point/footprint/quadrature renderer written, and no duplicate NIST phantom experiment.**
- Existing Gemini [source-supported reader and precise sampler at 0b92a3e](https://github.com/JorisViaudQuantAI/Scrollprize/tree/0b92a3e06d686562f0f8bfb7baa264c3e98a9ca6/gemini_lane). Its support mask must be true for every voxel in the dense local crop before calling the shared all-eight-neighbor finite-volume renderer. No missing-data filling, radiometric conversion to normalized model units, smoothing of normals, or stitching across charts.

[Frozen protocol](../firstletters-colab/adaptive_surface_recovery/shared_renderer_bridge/protocol.md) at 215f9bdc25575cddcd9cffa443af36ae9fc8d1da. [Adapter code](../firstletters-colab/adaptive_surface_recovery/shared_renderer_bridge/run.py) at 638f8d1b615a7288243bb9725a1f151ded4d2916. [Pinned one-shot workflow](../.github/workflows/chatgpt-adaptive-shared-bridge.yml) at e3e5de8b3ec4e240bc527d0ce95c52f6bd0669c5; [executed run 45269199729](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45269199729), job 138024163781. [Numerical receipt](../firstletters-colab/adaptive_surface_recovery/shared_renderer_bridge/result.json) at e3987d4698893b89ea01b67348631d832f9e59c2. Four generated checks pass; all eight real gates pass. Exact source blobs were checked before execution.

## Actual measured result

Two fixed 32x32 native ROIs at 8.64 um pitch, one on each chart, same phase family as 038. Thirty-one ordered depth samples -15 through +15. Center sampling vs whole-UV-pixel clipping; existing shared renderer N=1 vs N=4; Gemini precise point sampler as numerical CT parity reference.

| Measure | Primary chart | Fallback chart |
| --- | ---: | ---: |
| Point-valid pixels | 1024 | 512 |
| Whole-footprint-valid pixels | 1024 | 480 |
| Point-only pixels excluded by continuous footprint | 0 | 32 |
| Whole-footprint samples valid at all 31 depths (N=1 and N=4) | 1024 | 480 |
| Compared actual CT values (N=1 vs Gemini precise) | 31,744 | 14,880 |
| Maximum absolute CT difference | 0 | 0 |
| Max absolute XYZ difference from 038 | 3.64e-12 voxel | 3.64e-12 voxel |
| N=4 vs N=1 CT RMS change (native intensity units) | 0.683 | 2.844 |
| N=4 vs N=1 maximum CT change | 7.690 | 32.444 |

Point validity, global original face IDs, coverage counts, reason bits and ambiguity masks are exactly equal across both point implementations. The stricter whole-pixel contract removes 32/512 (6.25%) point-valid fallback pixels, all due to incomplete UV coverage; does not invent support. All retained CT values and availability agree in N=1, and N=4 uses exactly the same retained mask. 46,624 actual CT values compared, no differences. 16,777,455 charged response-body bytes in 9 requests; 2.994 s measured experiment wall time (not workflow setup/download); 187,224,064 bytes peak RSS. Dense crops contain 575,016 and 267,300 voxels, all stored-source available.

**Limits:** two small public development fields, not whole-atlas or independent-scroll generalization. N=4 changes CT values; no labels, numerical integration truth or ink inference measured here, so no efficacy conclusion. Face normals are piecewise constant; no anatomical sheet identity or normal-ray injectivity claim. Whole-pixel support is a stricter geometric criterion than point support, not a direct improvement in recall. Dense crops may overfetch and can fail where precise remote sampling would retain partial source; this bridge refuses such cases rather than manufacture data.

## Reuse the adapter

```python
from run import load_atlas, ChartBridge
from surface import OriginalSurface
atlas = load_atlas(atlas_path, expected_sha256)
chart = ChartBridge(atlas, 0, OriginalSurface)  # chart 1 separately
point = chart.sample_grid(origin_uv, (0.05, 0.05), (h, w))
# point["original_face_id"] maps every valid query to original source face.
footprint = chart.footprint_support(origin_uv, (0.05, 0.05), (h, w))
# chart.export_npz(new_path) writes uv, xyz, triangles, original_face_ids.
```

Reuse `surface_sampling/render.py::render` with the verified finite dense CT block, origin in XYZ native voxels, explicit UV center/step, ordered depths, and this ChartBridge. Do not merge the UV charts. Read the exact workflow for pinned checkout/download commands. The public atlas artifact expires after 30 days; regenerate from 037 if needed and verify source/payload identity. New outputs on unread scrolls remain outside Git.

## Open questions / next actions

- **Claude**: Q1/Q7 verdicts and wrong-sense fit remain yours; Q10 use the same atlas/footprint interface for future flatten QA. Shared winding-order Q12 correction remains on its separate accepted lane. For the requested corrected Q12 replay, coordinate with that lane rather than duplicate it; its protocol needs complete selected transverse z-band tracks, all directly linked longitudinal tracks and crossing rows with original global IDs/point-index mapping, not arbitrary truncated chains. No private locations requested in this public note.
- **Gemini**: reuse the original-face bridge with the existing source-supported reader; this result does not alter your RGT/T2/FP16/T8 code or duplicate your Progress dossier. Joint claim only with appropriate attribution and owner approval.
- **Codex**: indexed point sampling is already available and now validated against the shared footprint implementation. Use explicit chart separation, global face IDs and continuous UV coverage rather than interpolating a coarse node mask.
- **Adaptive lane**: no further parallel reimplementation of source rendering. The next discriminating test is a frozen, bounded *actual detector* comparison on an independently annotated published control, identical model and support, N=1 vs N=4 and chart strategy as appropriate. Coordinate its scope first with the shared surface_sampling and active GPU lanes; assess letter/ink recall, artifact controls and context support, not merely pixel smoothness. No unread launch yet.
- No main merge, other-branch mutation, organizer submission, new GPU resources, or recurring automation.
