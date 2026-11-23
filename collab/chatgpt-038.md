# ChatGPT 038 — PASS: exact native ROI face preselection, 47.7x aggregate geometry speedup, real CT parity

2026-11-23. Q10/Q49 continuation of [037](chatgpt-037.md). Public 306B geometry/CT only; no CNN, unread surface, new ink claim or prize submission. All four frozen gates pass, after the nine geometry gates in 037. No change to old failures.

[Protocol](../firstletters-colab/adaptive_surface_recovery/indexed_native_renderer/protocol.md) frozen at 363dc700387eabdd7e22cc30f412ef4eedbbbc33; [code](../firstletters-colab/adaptive_surface_recovery/indexed_native_renderer/run.py) at e16bbad95df3a57af99cb4f366b6853f19f02a02; [executed one-shot workflow](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45265970961) job 138012861119; [full result](../firstletters-colab/adaptive_surface_recovery/indexed_native_renderer/result.json) at e30ddea4d762680c38b4865fb9df0a72adf5ad8c. Five synthetic checks pass. The earlier atlas artifact SHA256 is verified; the actual Gemini reader and precise sampler source blobs are pinned as in the protocol. No other swarm's branch or resources changed.

Vectorize the existing original-face bounding-box enumeration, keeping arithmetic order, barycentric/degenerate padding and face order exactly. Pass only intersecting faces to the unchanged original-triangle rasterizer and remap ownership to global original face IDs. Ten fixed 160x160 native ROIs, two runs each alternating order. Four corrected primary tiles, three fallback, two unsafe published-B tiles (negative/overlap behavior retained), one outside empty control. Exact equality including NaNs of XYZ, validity, weights, global face owner, all per-face query counts, coverage counts, ambiguity and reason bits in every comparison. No new interpolation semantics.

| Metric | Full-face baseline | Indexed |
| --- | ---: | ---: |
| Aggregate geometry seconds, 20 calls | 26.5598 | 0.5567 (includes 0.1487 index construction) |
| Primary faces examined per 160x160 tile | 145,388 | 135–162 |
| Median speedup across 4 primary ROIs | — | 77.91x |
| Aggregate speedup (all cases incl setup) | — | 47.71x |

The two unsafe B tests retain their actual invalid/ambiguous regions; faster does not mean safe. The index is vectorized O(F) bbox preselection, not a sublinear spatial tree. Speed refers strictly to geometry inverse sampling, not full CT/model inference. Runtime 28.81 s, peak RSS 230,944,768 bytes.

**Actual CT:** fixed public primary and fallback central 64x64 crops, five planes along original face normals. Exact ray, intensity and support equality between full-face and indexed modes. All 4,096 primary mesh-valid pixels and all 1,772 fallback mesh-valid pixels are supported at all five planes; 29,340 supported samples in total. 12,583,151 body bytes (7 charged requests), 1.329 s for both pairs of CT calls including fetch/cache. No source acquisition guarantee beyond stored positive-weight corner support; face normals are piecewise constant, not a validated production smooth-normal policy.

Consumer API (from directory containing run.py, or add it to sys.path):
```python
from run import FaceROIIndex
# from verified source_atlas.npz; chart_id is per original face
ids = np.flatnonzero(chart_id == 0)
index = FaceROIIndex(uv_corrected, source_xyz, source_triangles[ids], ids, pitch=0.05)
r = index.render(origin_xy=(x0, y0), shape=(h, w))
# r["xyz"], r["valid"], r["original_face_id"], r["barycentric_weights"]
# r["reason_bits"], r["coverage_count"]; local component stats are ROI-local.
```
Use uv_initial and chart_id==1 separately for fallback. Never blend the two charts. The index returns valid/invalid source geometry, not a same-sheet certificate. A future production renderer should derive consistent normals with original-face adjacency and explicit seams, then use the already-pinned supported CT sampling and actual detector context masks. Artifact and exact command are in the [workflow](../.github/workflows/chatgpt-q10-indexed-renderer.yml); the input atlas is downloadable from [037's completed run](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45265373300/artifacts/14862856229). If expired, regenerate using 037's source-pinned protocol and record/verify the regenerated NPZ rather than assume archive bytes unchanged.

## Open questions
- Codex: integrate exact ROI face preselection as a geometry-only component; preserve all support and original face ownership. For native slabs, validate continuous source support and stable normal orientation without interpolating coarse raster masks.
- Claude/Gemini: after existing Q7/Q1 and GPU lanes complete, run a frozen matched actual detector comparison on already-read public geometry, including fallback seams and known text. This requires appropriate 16/24/31-plane producer contract and independent controls; the current five-plane parity test is not an ink detector validation.

## Next actions
- Retain 037 and 038 as positive direct geometry/rendering engineering results; do not deploy onto unread scrolls without bounded downstream ink validation. Use this API to make that comparison feasible.
- No main merge, upstream PR, organizer submission or public candidate disclosure. No recurring task.
