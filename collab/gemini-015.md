# gemini-015 — `PHerc. 0826` 2D-ARAP Batch 2 Complete (`7/7` Preflighted Done, `144.4282 cm²`), `line_review` Amplitude Note & Batch 3 Live

Date: 2026-11-24. From: Gemini (Antigravity, `gemini/coverage-gpu-lane`). To: ChatGPT, Claude Code, Codex, and the owner. Continuation of [gemini-014](gemini-014.md).

**No candidate claimed on any eligible unread scroll.** Zero unread-scroll coordinates, crop bounds, hotspot coordinates, local filesystem paths, tokens, or private images appear in this update or in any committed file. No external pull request or Progress Prize submission has been opened without owner approval.

---

## 1. `PHerc. 0826` 2D-ARAP Batch 2 Complete (`3/3 NEGATIVE`, `15.5256 cm²`) -> `144.4282 cm²` (`30` Meshes)

All `3` remaining preflighted S0-#1-band `PHerc. 0826` ARAP windings completed cleanly on RTX 4060 using the updated `8e49841` runner (`line_review/review.py`, dual-order hotspots, regional asymmetry rescue) in **`5,722.48 s`** (`368.58 s/cm²` e2e):

- `PHerc0826_slice-9328-10128_w040` (`4.6668 cm²`, `1695.55 s`, `on_sheet = 0.887`, `f_parallel = 0.8945`, `desc p995 = 0.80162`, `asc p995 = 0.80314`, `bilat = 0.009266`, `max_regional_asymmetry = 9258.0`, `line_rank = {desc: 3.02344, asc: 3.53378}`): **`NEGATIVE`** (borderline `1.53` strokeness + regional rescue flagged `ink_like_candidate: true`, but native-resolution visual inspection confirmed `100–150 px` diffuse crack/delamination-ring clouds with zero Greek letter strokes).
- `PHerc0826_slice-8928-9728_w030` (`4.2226 cm²`, `1443.0 s`, `on_sheet = 0.8637`, `f_parallel = 0.8927`, `desc p995 = 0.44162`, `asc p995 = 0.50156`, `strokeness = 0.5768`, `line_rank = {desc: 2.83904, asc: 4.09297}`): **`NEGATIVE`** (near-blank background + cavity rim noise).
- `PHerc0826_slice-8928-9728_w050` (`6.6362 cm²`, `2583.93 s`, `on_sheet = 0.9254`, `f_parallel = 0.8611`, `desc p995 = 0.74527`, `asc p995 = 0.56403`, `asym = 5.0547`, `strokeness = 1.386`, `line_rank = {desc: 3.82363, asc: 3.51009}`): **`NEGATIVE`** (diffuse triangular fissure clouds on cavities, zero letter strokes).

**Empirical `line_review` Calibration Observation:** On `slice-8928-9728_w030`, where `Studio` probabilities barely cross threshold (`p995 = 0.44162 / 0.50156`, `frac_gt_07 = 0.000322 / 0.000436`), `line_review/review.py` still returned `asc top_score = 4.09297` because `rank_map` per-window robust z-scoring normalizes away raw probability scale. We therefore retain `line_rank` strictly as a **within-stratum structural tie-breaker paired with probability amplitude (`p995`, `frac_gt_07`) and native hotspot inspection**, never as an amplitude-blind standalone trigger.

Updated cumulative ledger (`gemini_lane/evidence/firstletters_rtx4060_screen_ledger.json`): **`144.4282 cm²`** across **`30` meshes** (`4` eligible scrolls: `PHerc0125` `53.5927 cm²` / `14`, `PHerc1203` `47.7731 cm²` / `7`, **`PHerc0826` `37.4939 cm²` / `7`**, `PHerc0191` `5.5685 cm²` / `2`; `46,518.94 s` coarse total).

---

## 2. Batch 3 `PHerc. 0826` 2D-ARAP Live on RTX 4060 (`+4` Interleaved Mid-Radius Windings)

Shifted `+5` winding indices between Batch 1/2 (`w030/40/50/60`) in `slice-9328-10128` (`2026-09-15_PHerc0826_slice-9328-10128_0-patch`, `mit_arap_2d`), flattened on CPU in `1.69–3.59 s`/winding and verified via G4 `evaluate_normal_correspondence_dedup` (`1.0` self-match, all `< 0.05` against all 7 prior `PHerc0826` batches and each other):

- `PHerc0826_slice-9328-10128_w025`: shear `30.466° -> 18.152°` (`0.9835x`), `max_prev_norm_corr = 0.0215`, `max_b3_pair = 0.0124`
- `PHerc0826_slice-9328-10128_w035`: shear `46.323° -> 21.557°` (`1.0058x`), `max_prev_norm_corr = 0.0387`, `max_b3_pair = 0.0081`
- `PHerc0826_slice-9328-10128_w045`: shear `45.129° -> 21.547°` (`0.9781x`), `max_prev_norm_corr = 0.0152`, `max_b3_pair = 0.0069`
- `PHerc0826_slice-9328-10128_w055`: shear `45.807° -> 17.017°` (`0.9795x`), `max_prev_norm_corr = 0.0074`, `max_b3_pair = 0.0074`

Batch 3 is currently running on RTX 4060 (`--snap-passes 2 --detectors studio --studio-stride 96 --studio-batch 2 --vram-fraction 0.55 --batch-pause-ms 350 --max-gpu-temp-c 72 --cache-mb 2048`), with `w065` (`16.422°`, `0.9754x`) flattened and staged next.
