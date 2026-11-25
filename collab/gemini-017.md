# gemini-017 — `PHerc. 0826` 2D-ARAP Batch 4 Closed (`184.2065 cm²`, `38` Meshes), `w055` 2-Family + `Q34` FP Check & Batch 5 Live

Date: 2026-11-25. From: Gemini (Antigravity, `gemini/coverage-gpu-lane`). To: ChatGPT, Claude Code, Codex, and the owner. Continuation of [gemini-016](gemini-016.md).

**No candidate claimed on any eligible unread scroll.** Zero unread-scroll coordinates, crop bounds, hotspot coordinates, local filesystem paths, tokens, or private images appear in this update or in any committed artifact. No external pull request or Progress Prize submission has been opened without owner approval.

---

## 1. `PHerc. 0826` 2D-ARAP Batch 4 Closed (`4/4 NEGATIVE`, `+21.6818 cm²`) -> `184.2065 cm²` (`38` Meshes)

Batch 4 (`w065_9328` plus the disjoint `slice-6528-7328` windings `w030`, `w040`, `w055`; `mit_arap_2d` + `2x m7` snap + `Studio fp16` both orders at `stride = 96`) completed in **`8,243.77 s`** (`380.22 s/cm²` coarse e2e):

- `PHerc0826_slice-9328-10128_w065` (`8.1441 cm²`, `3159.54 s`, `on_sheet = 0.9346`, `f_parallel = 0.9197`, `desc p995 = 0.81562`, `asc p995 = 0.84325`, `strokeness = 1.236`, `line_rank = {desc: 3.78214, asc: 3.01499}`): **`NEGATIVE`** (isolated blobs over tangled delamination rings and voids).
- `PHerc0826_slice-6528-7328_w030` (`2.9245 cm²`, `1037.42 s`, `on_sheet = 0.9166`, `f_parallel = 0.9139`, `desc p995 = 0.62543`, `asc p995 = 0.86692`, `asymmetry = 17.0086x`, `strokeness = 2.5946`, `line_rank = {desc: 2.95751, asc: 2.57729}`): **`NEGATIVE`** (broad `~150 px` diffuse `asc` cotton-cloud blotches matching sibling `w055`).
- `PHerc0826_slice-6528-7328_w040` (`3.9983 cm²`, `1470.92 s`, `on_sheet = 0.9001`, `f_parallel = 0.9474`, `desc p995 = 0.56807`, `asc p995 = 0.85489`, `asymmetry = 23.9149x`, `strokeness = 1.7001`, `line_rank = {desc: 3.86423, asc: 3.4428}`): **`NEGATIVE`** (diffuse unilateral `asc` blobs over delamination swirls and cavities).
- `PHerc0826_slice-6528-7328_w055` (`6.6149 cm²`, `2575.89 s`, `on_sheet = 0.8947`, `f_parallel = 0.8832`, `desc p995 = 0.87062`, `asc p995 = 0.89645`, `asymmetry = 2.6404x`, `max_regional_asymmetry = 55077.0`, `strokeness = 4.0156`, `line_rank = {desc: 4.75979, asc: 4.20129}`): **`NEGATIVE` after 2-family `Studio s48 + A-hn` follow-up + 9-variant `Q34` adversarial verification** (details in Section 2).

Updated cumulative ledger ([`firstletters_rtx4060_screen_ledger.json`](../gemini_lane/evidence/firstletters_rtx4060_screen_ledger.json)): **`184.2065 cm²`** across **`38` meshes** (`4` eligible scrolls: `PHerc0125` `53.5927 cm²` / `14`, `PHerc1203` `47.7731 cm²` / `7`, **`PHerc0826` `77.2722 cm²` / `15`**, `PHerc0191` `5.5685 cm²` / `2`; **`8` two-family follow-ups**; `61,286.46 s` coarse total / `332.71 s/cm²`).

---

## 2. Cross-Scroll FP Validation on `PHerc0826_slice-6528-7328_w055` (`IoU = 0.4654` Yet `0/9` Q34 Off-Sheet Cloud)

Every `slice-6528-7328` winding showed unilateral `studio_asc` dominance (`2.64x–23.91x`), with `w055` reaching `frac_gt_07 = 0.065762` and our highest coarse `line_rank` (`4.75979` / `4.20129`). High-resolution two-family and adversarial verification cleanly resolved the mechanism:

1. **Two-Family `Studio s48 + A-hn hn4_s0` Follow-Up (`0.6891 cm²`, `1672.76 s`, `post_temp = 58 °C`):**
   - `Studio asc` (`p995 = 0.91359`, `frac_gt_07 = 0.209758`) and `A-hn hn4_s0 asc` (`p995 = 0.86783`, `frac_gt_07 = 0.170984`) strongly agreed spatially (**`two_family_iou = 0.465391`**, **`studio_covered_by_ahn = 0.633435`**, `asymmetry_ratio = 6.6505x`, `bilateral_overlap = 0.004671`).
   - Crucially, **`line_rank` `asc` returned `ABSTAIN` (`proposal_count = 0`)** because the signal formed an unbroken `~3.75 mm` amorphous cloud (`126,118 px`, `equiv_diameter_mm = 3.7516`), and native `1:1` inspection showed diffuse cotton-clouds with zero Greek penstrokes.
2. **9-Variant Adversarial Depth (`dz = -6..+6` vx) & Orientation (`orig, flip_lr, flip_ud, rot180`) + `Q34` Check (`2267.33 s`, `post_temp = 59 °C`):**
   - **`0/9` variants passed `Q34`** (`adversarial_verdict = ARTIFACT_OR_WIDE_BLOB_CONFIRMED`, `max_q34_score = 0.046812`). All 4 `s48` orientations remained nearly invariant (`min pearson_r = 0.971977`, `iou_ge_05 = 0.8504–0.8936`, `width_median_mm = 0.600–0.634 mm`, `q34_score = 0.0`).
   - Depth sweep showed an off-sheet growth gradient matching [`PHerc0125/z15344_w035`](gemini-011.md): `studio_asc frac_gt_07` peaked off-sheet at **`dz = -3` (`0.271662`)** and **`dz = -6` (`0.266922`)** vs `dz = 0` (`0.201559`), `dz = +3` (`0.129001`), and `dz = +6` (`0.071759`).
   - **Operational takeaway:** Even `two_family IoU > 0.45` with clean papyrus support does **not** establish ink when `line_rank` abstains or native `1:1` visuals lack thin stroke topology; depth sweeps and visual inspection remain essential before any claim.

---

## 3. Pipeline Bugfix (`screen_scroll_rtx4060.py`) & Next Live Batch 5

- **Fixed `contrast=None` Crash in [`screen_scroll_rtx4060.py`](../gemini_lane/screen_scroll_rtx4060.py#L541):** `firstletters-colab/line_review/review.py` intentionally emits `"contrast": null` when a line chain borders a crop edge (`incomplete_flank_support_no_contrast_bonus`). Guarded `p.get("contrast") is not None` -> `None`, added flushed `STUDIO_DONE`/`AHN_DONE` progress markers, and added a regression test in [`test_screen_scroll_rtx4060.py`](../gemini_lane/test_screen_scroll_rtx4060.py#L234).
- **Live RTX 4060 Batch 5 Launched (`PHerc0826_slice-6528-7328_w020` & `w060`):** Both windings are pre-flattened (`mit_arap_2d`) and G4-verified (`max_pair_norm_corr <= 0.0167`), completing radial coverage `w020..w060` of `slice-6528-7328` before rotating into remaining flat `PHerc0125` `pscamillo` windings to surpass `200 cm²` cumulative.
