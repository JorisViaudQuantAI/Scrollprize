# P13 — CPU/S3 Zarr optimization for Gemini Néon (public controls only)

**26 November 2026.** This is the next independent contribution after [Gemini P12 / PR #22](https://github.com/JorisViaudQuantAI/Scrollprize/pull/22); stacked on that branch. The original Gemini Push #12 `6ca521cfff25ad83f56192cb47797dce72d57c83` and Nov 27 freeze `acea30d4cfed6ef9ba255a0894ba5bd4c6a3bec0` remain unmodified. No unread eligible-scroll scans, private locations, GPU calls, new readings or production merges.

## Main result: actual remote/native CT, not just a local proxy

[Successful GitHub Actions run 45892230045](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45892230045), [complete machine-readable receipt artifact](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45892230045/artifacts/15151508988), [per-operation rows retained in Git](native_receipt.json). Public awarded **PHerc. 0846B** mesh: source `brkzak/Pherc0846B_submission` commit `daabe53b90654f9aa46e1512bf4bef251fd3975b`, four source members checked by Git blob hash; only public CT `20250804142305-9.362um-1.2m-113keV-masked.zarr/0`. No winner inference code executed. Two deterministic full-geometry 16x16 mesh contexts, each rendered to a 17x301x301 uint8 stack and support masks, plus 24 fixed geometry-only rays of 25 depths. First and repeat measurements for both methods. **No Reader model, M7 network snap, ink ranking, unseen location or winner-reading reproduction.**

| Actual hosted CPU/S3 observations | Gemini reference sampler | P13 stencil sampler |
|---|---:|---:|
| CT chunk bodies received (excludes ~239 metadata bytes) | 387,973,120 B = 370 MiB | 188,743,680 B = 180 MiB |
| Decoded chunks over 6 operations | 185 | 90 |
| Requested chunk references | 288 | 192 |
| Sum of 6 sample/render wall times | 9.62545 s | 3.20934 s |
| Bitwise differences across every shared rendered array, source-support mask and ray score | — | **0** |

**51.35% fewer received CT-body bytes** and a **3.00x observed timing ratio on one ordered run**; do **not** forecast a 3x speedup across PHerc. 0125, full scans, network endpoints, or L4. Network cache/order, object availability and source geometry differ. All 42 actual native array comparisons (values + support and per-layer mask fields across arms/phases) passed. Results rechecked from the artifact separately: every repeated and cross-method SHA-256 matches, and every pixel support count matches. Original CT body bytes include re-fetches in a bounded 128MiB LRU per arm; metadata receipts are recorded separately. Both arms have 512MiB post-response per-arm transport guard, not a hard on-wire cap.

## Code changes and behavior

1. **`gemini_lane/stencil_reader.py`**: Opt-in `StencilReader(RemoteZarrV2)` for **uint8, fill-zero, Zarr v2**. It plans only the chunk cells touched by positive-weight trilinear corners and prefetches their union in bounded batches. SciPy `map_coordinates`, float32 output, exact-zero corner rules, missing-chunk support and physical XYZ inputs remain the same. Small dense boxes use a bounded direct path. For unsupported source dtypes/fills, calls the untouched reference sampler. No GPU/model dependency.
2. **`gemini_lane/stream_snap_render.py`**: Optional `reuse_center_samples=True` on the existing two-pass M7 snap. The **default is False**. It reuses source values/support from the initial center and accepted rounded displacement candidates instead of re-sampling centers/final points; requires an immutable M7 volume during the call. No change to accepted vertices, normals, ridge-picking, source-corner validity or two-pass logic.
3. **`gemini_lane/staged_rv3.py`**: Opt-in CPU preprocessing flags `--reader-mode stencil`, `--prefetch-mb 64`, and `--reuse-center-samples`. The default remains `legacy` with no center reuse. One shared reader per M7/CT volume spans all explicit meshes, saving re-fetches. SHA-256 stage payload receipts, private output guard, immutable stage paths and one-load burst Reader v3 from P12 are preserved. No implicit target list or production routing.

### Example commands

Use **only** operator-authorized local mesh and CT/M7 sources, with outputs outside every Git checkout. Test first on public controls, and enable M7 sample reuse **only** when the source snapshot is immutable:

```bash
python gemini_lane/staged_rv3.py stage \
  --mesh /public/control/mesh_A --mesh /public/control/mesh_B \
  --ct /public/control/native_ct.zarr --m7 /public/control/m7.zarr \
  --out /scratch/public_stage \
  --reader-mode stencil --prefetch-mb 64 --reuse-center-samples
# Existing one-model-load burst, separately from staging:
python gemini_lane/staged_rv3.py burst \
  --stage /scratch/public_stage/0000 --stage /scratch/public_stage/0001 \
  --checkpoint /scratch/reader-v3.pth --out /scratch/public_reader_output
```

The output paths and URLs above are placeholders, not proof of user data availability. Keep stage files/results private for any eligible unread source; they must never enter public Git. The CLI rejects outputs inside a repository.

## Independently tested CPU controls

The supplied code passes **10 original Gemini tests** and all new checks; the same tests passed on the hosted job before native downloads:

- [`stencil_regression.py`](stencil_regression.py): **18 cases / 182,376 bitwise comparisons**, varying 3D shapes, missing chunks, exact integer/near-integer coordinates, nonfinite/out-of-range points, workers/cache limits, direct versus sparse mode and non-uint8 fallback.
- Original extensive local [source pinned CT-texture stress](local_receipt.json): **120 cases / 623,280** bitwise value/support checks, 84 fallback calls; five sparse/dense boundary cases. On synthetic local Zarr made with real 0841 CT-texture *values*, decoded chunks are 36→22 (dense), 320→229 (oblique), 68→28 (diagonal), 49→49 (sparse), and 512→343 (exact chunk boundaries). There is no claim these are real surface/winner geometry.
- [`snap_reuse_test.py`](snap_reuse_test.py): **28 tests / 109,396 exact coordinate/metric checks**, seven synthetic M7 scenes, 0–3 passes, missing chunk, two-surface and no-sheet cases; sampling calls 182→112. A standalone local repetition gave 1.66→1.54s total, variability expected; center reuse does **not** bypass required ray searches or establish sheet identity.
- [`test_stage_modes.py`](test_stage_modes.py): **64,876 exact CT, support-mask and snap-metric comparisons** on two synthetic staged meshes, three configurations. Same 30,600 CT values per stage arm and matched support, with no semantic change.

[Local per-case numbers and source hashes](local_receipt.json) and the full [hosted native receipt](native_receipt.json) are committed. Local complete JSONs and scripts can be packed for independent audit; no CT/source binaries need to be committed.

## Negative results / required next validation

- A previous native run failed because the public TIFF required `imagecodecs` for LZW; the next failed an exact `0.05 == 0.050000000745...` comparison in published metadata; a third failed because offline receipt output already existed. Those errors were corrected **before** [the successful run](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45892230045). Preserve these failure receipts rather than interpreting them as model failures.
- The actual benchmark is **one ordered legacy→stencil comparison** on one public winner geometry, not an order-balanced GPU cost trial. Report received bytes as the strongest evidence; do not extrapolate L4 cost or whole-manuscript speed from this wall-time ratio.
- Validate immutable M7 snapshot and re-fetch consistency before turning on center reuse for remote data. Defaults deliberately unchanged.
- Gemini / Codex can independently replicate on an alternative awarded public geometry with alternated execution order and two matched cache budgets, then measure CPU stage, network, GPU model-load, inference, model probability equality, failure/retry and VRAM separately. No need to rerun our already completed synthetic exactness unless auditing correctness.
- **No automatic exclusion** from low `p995`, fibrosis heuristics, line_rank, clipped M7 support, low on-sheet or other unverified score: P12 evidence shows weak actual ink would be lost.

**Decision:** an opt-in, numerically exact CT I/O optimization and a cache-aware stage interface are ready for independent public-control review. They are **not** production-enabled or incorporated into Gemini's frozen branch.
