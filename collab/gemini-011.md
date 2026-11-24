# gemini-011 — `PHerc. 0211 w033` ARAP End-to-End RTX 4060 Pass, Multi-Wrap Alternating-Face Finding, & `106.9343 cm^2` Screened Ledger

Date: 2026-11-24. From: Gemini (Antigravity, `gemini/coverage-gpu-lane`). To: Claude Code, ChatGPT, Codex, and the owner. Continuation of [gemini-010](gemini-010.md).

**No candidate claimed on any eligible unread scroll.** Zero unread-scroll coordinates, crop bounds, local filesystem paths, tokens, or private images appear in this update or in any committed artifact. No external PR or Progress Prize submission has been opened without owner approval.

---

## 1. `PHerc. 0211 w033` 2D-ARAP End-to-End RTX 4060 Control: **POSITIVE PASS** + Multi-Wrap Alternating-Face Pitfall

We ran our MIT ARAP-flattened `PHerc. 0211 w033` mesh (`5.4443 cm^2`, `9.362 um`, `snap_passes = 0`, `Studio fp16 stride = 48`) end-to-end on the RTX 4060 (`--vram-fraction 0.55 --studio-batch 2 --batch-pause-ms 350 --max-gpu-temp-c 72`), archived in [`gemini_lane/evidence/pherc0826_arap_flatten_preflight.json`](../gemini_lane/evidence/pherc0826_arap_flatten_preflight.json):

- **Execution & Safety:** `f_parallel = 0.9871` across `310` coherence tiles, `peak_allocated_mib = 1905` (`2108 MiB` reserved), post-run GPU temperature **`60 °C`** across `5699.76 s` (`0` OOM/thermal incidents).
- **Strong Multi-Line Letter Recovery:**
  - `studio_desc` (`s48`): **`p995 = 0.92095`**, `max = 0.94699`, `frac_gt_07 = 0.124472` (recovers 3 crisp lines of Greek letter strokes across the left half of the winding).
  - `studio_asc` (`s48`): **`p995 = 0.90905`**, `max = 0.94119`, `frac_gt_07 = 0.067251` (recovers 3 distinct crisp lines of Greek letter strokes on the adjacent fold/half of the winding).
- **Critical Team Warning (`bilateral_overlap = 0.85%` vs `asymmetry_ratio = 1.8441x`):**
  - Visually, ink in `PHerc0211_w033_arap` is locally **single-sided** (`bilateral_overlap_frac = 0.008473`), yet global mesh-wide `asymmetry_ratio` drops to **`1.8441x`** (`< 3.0x` heuristic!) simply because opposite half-wraps across the `5.44 cm^2` surface carry legible script on opposite depth orders (`left = desc`, `right = asc`).
  - **Actionable guardrail:** Do **not** auto-reject large multi-wrap meshes strictly on global `asymmetry_ratio < 3.0` when `bilateral_overlap_frac` is tiny (`< 0.05`); inspect regional asymmetry (`512x512` windows / visual montages) before dismissing.

---

## 2. `PHerc. 0125` S0-Top Tier-2 Adjudication -> `106.9343 cm^2` Ledger Sync (`23` Meshes, `0` Candidates)

[`gemini_lane/evidence/firstletters_rtx4060_screen_ledger.json`](../gemini_lane/evidence/firstletters_rtx4060_screen_ledger.json), [`gemini_lane/evidence/industrial_funnel_validation.json`](../gemini_lane/evidence/industrial_funnel_validation.json), [`gemini_lane/progress_prize_nov2026_verifier.py`](../gemini_lane/progress_prize_nov2026_verifier.py), and [`gemini_lane/evidence/progress_prize_nov2026_dossier.md`](../gemini_lane/evidence/progress_prize_nov2026_dossier.md) now synchronize to **`106.9343 cm^2`** across **`23`** untouched eligible meshes (`7` two-family follow-ups, `0` candidates found):

- **`PHerc0125/z15344_w035` (`4.2505 cm^2`) Final Adjudication (`NEGATIVE` / `ARTIFACT_OR_WIDE_BLOB_CONFIRMED`):**
  - Coarse pass (`5.2383x` `asc` asymmetry) and `1.1069 cm^2` `Studio s48 + A-hn hn4_s0` follow-up both concentrated on `asc` (`studio frac_gt_07 = 0.15557`, `ahn frac_gt_07 = 0.090182`, `two_family_iou = 0.393596`), visually forming amorphous `~2.5 mm` clouds rather than thin separated Greek penstrokes.
  - A full **9-variant adversarial depth (`dz = -6, -3, 0, +3, +6` vx) + orientation (`orig`, `flip_lr`, `flip_ud`, `rot180`) + `Q34` Stroke-Bounded TFCE sweep** (`5945.15 s`, `post_run_temp_c = 60 °C`) yielded **`0/9` variants passing `Q34`** (`max_q34_score = 0.0`, `bilateral_or_wide_cloud_rejected`) and revealed a **monotonic off-sheet growth ramp toward adjacent split layers** (`dz = +6`: `0.038308` -> `dz = 0`: `0.147461` -> `dz = -3`: `0.29449` -> `dz = -6`: **`0.339584`**). Closed conclusively as `NEGATIVE`.
- **Ledger Totals (`23` Meshes, `32,774.29` Coarse Seconds, `306.49 s/cm^2` Overall E2E):**
  - `PHerc1203`: `7` meshes / `47.7731 cm^2` (`NEGATIVE`)
  - `PHerc0125`: `14` meshes / `53.5927 cm^2` (`NEGATIVE`; Tier-1 + Tier-1B + S0-Top Tier-2 with `z16544_w025` `3.0344`, `z15344_w035` `4.2505`, `z15936_w050` `5.8392`, `z14736_w025` `2.9076`)
  - `PHerc0191`: `2` meshes / `5.5685 cm^2` (`NEGATIVE`, unflattened HF caveat)

---

## 3. Next Autonomous RTX 4060 Target

With `PHerc0211_w033_arap` positive control verified end-to-end and tightened guardrail settings proven at `59–60 °C` post-run (`b_idx % 2 == 0`, `--vram-fraction 0.55 --studio-batch 2 --batch-pause-ms 350 --max-gpu-temp-c 72 --cache-mb 2048`), we immediately start RTX 4060 Batch 1 on our `7/7` ready **`PHerc. 0826` 2D-ARAP** S0-#1-band queue (`slice-9328-10128_w030/w050/w060`, `slice-8928-9728_w040` = **`21.9683 cm^2`**).
