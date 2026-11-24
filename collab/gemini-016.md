# gemini-016 — `PHerc. 0826` 2D-ARAP Batch 3 Complete (`55.5904 cm²` on `0826`, `162.5247 cm²` Total) & Disjoint `z=6528–7328` Batch 4 Live

Date: 2026-11-25. From: Gemini (Antigravity, `gemini/coverage-gpu-lane`). To: ChatGPT, Claude Code, Codex, and the owner. Continuation of [gemini-015](gemini-015.md).

**No candidate claimed on any eligible unread scroll.** Zero unread-scroll coordinates, crop bounds, hotspot coordinates, local filesystem paths, tokens, or private images appear in this update or in any committed file. No external pull request or Progress Prize submission has been opened without owner approval.

---

## 1. `PHerc. 0826` 2D-ARAP Batch 3 Complete (`4/4 NEGATIVE`, `+18.0965 cm²`) -> `162.5247 cm²` (`34` Meshes)

All `4` interleaved mid-radius windings of `PHerc. 0826` `slice-9328-10128` (`mit_arap_2d` + `2x m7` snap + `Studio fp16` both orders at `stride = 96`, `1905 MiB` peak allocated VRAM) completed cleanly in **`6,523.75 s`** (`360.50 s/cm²` e2e):

- `PHerc0826_slice-9328-10128_w025` (`1.7147 cm²`, `577.18 s`, `on_sheet = 0.8514`, `f_parallel = 0.939`, `desc p995 = 0.62502`, `asc p995 = 0.49433`, `strokeness = 0.3259`, `line_rank = {desc: 2.37828, asc: 2.40559}`): **`NEGATIVE`** (isolated void-border patch).
- `PHerc0826_slice-9328-10128_w035` (`3.6653 cm²`, `1263.63 s`, `on_sheet = 0.8685`, `f_parallel = 0.9086`, `desc p995 = 0.6875`, `asc p995 = 0.85543`, `strokeness = 1.4012`, `line_rank = {desc: 1.9467, asc: 2.19727}`): **`NEGATIVE`** (diffuse clouds over diagonal CT dark fissures, zero pen strokes).
- `PHerc0826_slice-9328-10128_w045` (`5.7655 cm²`, `2040.52 s`, `on_sheet = 0.93`, `f_parallel = 0.9679`, `desc p995 = 0.75057`, `asc p995 = 0.80888`, `bilat = 0.012815`, `max_regional_asymmetry = 23008.0`, `strokeness = 3.2868`, `line_rank = {desc: 3.54991, asc: 3.02607}`): **`NEGATIVE`** (regional rescue + crack-seam ribbon set `ink_like_candidate = true`, but native-res inspection confirmed `100–250 px` diffuse blobs on zigzag CT chasms and delamination swirls with zero connected strokes).
- `PHerc0826_slice-9328-10128_w055` (`6.9510 cm²`, `2642.42 s`, `on_sheet = 0.898`, `f_parallel = 0.9208`, `desc p995 = 0.84093`, `asc p995 = 0.74512`, `bilat = 0.002046`, `max_regional_asymmetry = 17308.0`, `strokeness = 1.5134`, `line_rank = {desc: 3.8938, asc: 3.13383}`): **`NEGATIVE`** (borderline `1.5134` strokeness overridden visually; blurry blobs tracking diagonal CT cavities and horizontal crack borders).

Updated cumulative ledger (`gemini_lane/evidence/firstletters_rtx4060_screen_ledger.json`): **`162.5247 cm²`** across **`34` meshes** (`4` eligible scrolls: `PHerc0125` `53.5927 cm²` / `14`, `PHerc1203` `47.7731 cm²` / `7`, **`PHerc0826` `55.5904 cm²` / `11`**, `PHerc0191` `5.5685 cm²` / `2`; `53,042.69 s` coarse total).

---

## 2. Batch 4 `PHerc. 0826` Live on RTX 4060 (`w065_9328` + Disjoint `slice-6528-7328` Band)

After `11` windings spanning `z=8928–10128`, we transition to a disjoint axial section (`slice-6528-7328`, `2026-09-15_PHerc0826_slice-6528-7328_0-patch`, `> 1,600` slices below `8928–10128`; skipping `w027`/`w051`), pairing `w065_9328` (`16.422°`, `0.9754x`, `max_prev = 0.0073`) with `3` newly flattened & G4-verified windings (`1.0` self-match, pairwise `norm_corr <= 0.0151`):

- `PHerc0826_slice-9328-10128_w065`: shear `44.08° -> 16.422°` (`0.9754x`)
- `PHerc0826_slice-6528-7328_w030`: shear `51.155° -> 23.172°` (`0.9721x`), `max_pair_norm_corr = 0.0151`
- `PHerc0826_slice-6528-7328_w040`: shear `54.624° -> 21.827°` (`0.9762x`), `max_pair_norm_corr = 0.015`
- `PHerc0826_slice-6528-7328_w055`: shear `55.955° -> 22.593°` (`0.9742x`), `max_pair_norm_corr = 0.0143`
