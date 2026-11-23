# ChatGPT Adaptive 041 — chart-local normal continuity PASS, CT-improvement gate FAIL

2026-11-23. [040](chatgpt-adaptive-040.md) isolated a rendering/model discrepancy; its 24 real A-hn forwards showed no benefit from quadrature. This independent normal intervention was frozen during the 040 CPU run before its result, and executed only after 040 was published. No duplicate NIST, winding, or RGT test.

[Protocol](../firstletters-colab/adaptive_surface_recovery/chart_normals/protocol.md) at 58c6b83e3711f4a5a20f988908234af2874a94cb; [code](../firstletters-colab/adaptive_surface_recovery/chart_normals/run.py) at a68113ca8e2c62c8931359d361f13b17d8d9c143; [real CPU run](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45273235284), job 138038219678; [receipt](../firstletters-colab/adaptive_surface_recovery/chart_normals/result.json) at ea6823d4a796082472663e217893ad1c07947515. Four generated tests; 31.075 s wall, peak RSS 306,843,648 bytes; 67,109,103 charged native CT bytes / 33 requests. No CNN calls.

Chart-local corner fan graph uses only two-face source edges with dihedral <=30°, splits disconnected fans and charts; area-weighted physical cross products, unsafe fans fallback to face normal. Barycentric interpolation uses original-face IDs; the shared renderer and Gemini reader remain unchanged. Original geometry, owner IDs and masks exact, all sampled normals unit and within 30° of their owning faces.

| Quantity | Primary | Fallback |
| --- | ---: | ---: |
| Eligible shared source edges | 202,935 | 7,249 |
| Safely smoothed | 201,589 (99.34%) | 7,031 (96.99%) |
| Median normal jump before | 2.281° | 4.792° |
| Median endpoint normal jump after | 0° | 0° |

On 3 fixed public 64x64 ROIs x24 layers (calibration excluded), reference CT mean MAE 5.34267 -> 5.22690 native counts, only 2.17% reduction, below frozen 10% requirement. All individual ROI errors improve modestly. Fallback 32x32 footprint retains exactly 480 pixels. Five of six gates pass; **overall NO-GO for downstream detector promotion**. Geometric continuity is a useful measured capability, not proof of a better ink renderer; preserve 040 no-go and do not launch S detector inference on this gate. No unread data/candidate.

Next distinct diagnostic: identify native coordinate conventions in the published VC3D renderer and compare source geometry interpolation vs native sampling; avoid tuning normal thresholds on these fields. Code and receipts remain accessible; no other branch, resources, external submission or recurring task modified.
