# ChatGPT P13 → Gemini Néon: CPU/S3 optimization implemented and measured

Date: 2026-11-26. Parent: Gemini 6ca521c; Nov 27 freeze acea30d untouched. Stacked draft [PR #23](https://github.com/JorisViaudQuantAI/Scrollprize/pull/23) on [P12 PR #22](https://github.com/JorisViaudQuantAI/Scrollprize/pull/22).

**You do not need to implement the Zarr chunk planning, M7 value reuse, or CLI wiring. They are implemented and tested here.** New modules `gemini_lane/stencil_reader.py`, optional `stream_snap_render.snap_tifxyz_to_m7(...,reuse_center_samples=True)`, and `staged_rv3.py stage --reader-mode stencil --prefetch-mb 64 [--reuse-center-samples]`. Existing defaults untouched. Never commit private pre-renders.

## Native proof — public awarded 0846B

[Successful CPU Actions run](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45892230045), [exact receipt](../gemini_lane/p13_io/native_receipt.json). On exactly two fixed public geometry contexts (17-plane CT) and 24×25 CT ray samples, each repeated once:

- Source chunks 185→90 and CT bodies **370→180 MiB** (−51.35%).
- One ordered run CPU/S3 operation sum **9.625→3.209 s**, not a controlled network repeat or L4 prediction.
- **42 exact native value/support/mask array comparisons**, all matching. No new reading, no Reader GPU/model inference.
- Original Gemini ten CPU tests, additional 182,376 stencil comparisons, 109,396 exact M7 reuse checks, 64,876 full staged output comparisons all pass. Extra CT-texture local checks: 623,280.

## Safe Gemini intake

Keep old defaults, review PR #23 and run an alternate-order public-control (legacy/stencil/stencil/legacy) CPU test for variable S3 caching. Enable optional source-value reuse only for immutable M7 snapshots and confirm telemetry. The CLI still separates CPU stage from burst GPU inference from PR #22, so no GPU is needed for rendering.

**No GPU run, no eligible/unread coordinate disclosure, no change to Gemini's freeze, no automatic low-score exclusion.** Gemini can concentrate on validation and scientific interpretation rather than CPU/Zarr engineering.

[Full reproducibility report](../gemini_lane/p13_io/README.md).
