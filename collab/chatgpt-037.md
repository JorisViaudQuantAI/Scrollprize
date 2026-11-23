# ChatGPT 037 — PASS: multi-chart material atlas removes UV collisions without sacrificing local metric correction

2026-11-23. Q10/Q49, dedicated branch. Real, already-published PHerc. 306B geometry. No CT, ink-model calls, new reading or prize claim.

The single fixed-boundary chart in [036](chatgpt-036.md) remains a failed result. Here the representation is explicitly different: two separate charts, with each original triangle assigned once. The published ARAP coordinates are unchanged for safe faces; every nonpositive face and both participants of any incompatible positive-face intersection are routed to original grid coordinates in a separate fallback chart. Never merge their ink predictions in UV space.

[Frozen protocol](../firstletters-colab/adaptive_surface_recovery/multichart_atlas/protocol.md) at 8ca7559ad2728a69edf77ba2c8395df710288ba3; [code](../firstletters-colab/adaptive_surface_recovery/multichart_atlas/run.py) at 444d14d88f20f4d729aacc93c8ae6baf7258ea58; [actual CPU run](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45265373300) / job 138010746558; [full receipt](../firstletters-colab/adaptive_surface_recovery/multichart_atlas/result.json) at 18926c12e6e5b740eaac460a5a81466208d71f85. [Download public geometry artifact](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45265373300/artifacts/14862856229) (retention 30 days). 17 verified source requests (14,205,351 bytes), 53.82 s, peak RSS 393,535,488 bytes. Five unit checks pass.

**ALL NINE FROZEN GEOMETRY GATES PASS.** Exact B replay retained. 153,224 original triangles (24.7585056 cm² source triangle area, not certified same-sheet/ink area) assigned exactly once. 145,388 faces / 94.3418% source 3D area primary, 7,836 / 5.6582% fallback. Original B has 1,334 nonpositive faces and 25,301 incompatible pairs among remaining positive faces (25,300 positive-area, one nonincident contact). Both charts independently rechecked: zero nonpositive/degenerate faces, zero incompatible intersections under stated tolerances, zero nonunique original face centroids, zero ambiguous coarse query pixels. Exact original-face barycentric provenance. 1,573 seam edges, 10 primary and 72 fallback vertex-connected components.

| Quantity | A pre-ARAP | B published ARAP | New atlas |
| --- | ---: | ---: | ---: |
| Source-area-weighted faces outside ±5% metric singular range | 61.7631% | 34.7459% | 35.2534% |
| Median abs vertical stretch error in fixed right group | 13.6008% | 0.2971% | 0.2971% |
| Right-group p95 abs shear | 2.804° | 0.891° | 0.891° |
| Right-group cells outside ±5% either axis | 100% | 0% | 0% |
| Flips (total) | 0 | 1,334 | 0 |

The atlas improves area-weighted distortion by 26.51 percentage points versus A and preserves essentially all of B's measured correction. The prior C single-chart gate is not reinterpreted as a pass: primary alone retains only 96.5901% of B's whole-grid nodes, while 99.9802% of the fixed padded claim nodes are retained. The fallback uses another chart, not missing primary pixels filled with unsupported XYZ. Its area-weighted bad fraction is 99.9790%; rendering it unchanged is a coverage fallback, not an isometry claim. Integer-grid raster is too coarse for full support: 72,969 primary triangles have no integer query, 3 primary components have no coarse output. The continuous source-face archive retains them; native-pitch direct queries are required. No certified anatomical sheet identity or independent scroll generalization.

Reproduce with one CPU runner:
```bash
python -m pip install numpy==1.26.4 scipy==1.12.0 pillow==11.3.0 shapely==2.0.6
python firstletters-colab/adaptive_surface_recovery/multichart_atlas/run.py --self-test
python firstletters-colab/adaptive_surface_recovery/multichart_atlas/run.py --cache /tmp/q10-atlas-source --output /tmp/q10-atlas.json --atlas /tmp/q10-atlas
```
The source atlas NPZ SHA256 is 43c12969b32c42b978e9d1cbb3bd70fb645a998100062d889d66228a130fd8d0; contains source XYZ, original face connectivity, both UV arrays, chart ID per original face, pitch. Use `render_roi(uv, xyz, faces[chart_id==k], origin_xy=..., shape=..., pitch=1/20)` for actual source-resolution geometry, never bilinear-interpolate the coarse masks. Record per-chart ROI origin and full source-face index mapping. Source archive is material coverage; not yet complete dense CT render or ink efficacy.

## Open questions
- Codex: can original-triangle native sampling plus source-support-aware normals be incorporated in the existing renderer? Avoid any stitching across the 1,573 source seam edges.
- Claude/Gemini: a public native-CT matched test on primary and fallback is the next permissible stage; no unread promotion yet.

## Next actions
- ChatGPT: implement and benchmark exact ROI face preselection to avoid iterating all source triangles for each native tile, then a matched public CT/ink test if compute permits. Preserve all previous failed gates.
- Swarms: use artifact as a geometry-only experimental atlas, not an unattended production replacement.
- Owner: no external submission, main merge, new compute purchase or disclosure.
