# Consolidated Monthly Progress Prize Technical Dossier (Deadline: November 30, 2026)

**Branch:** `gemini/coverage-gpu-lane`  
**Verification Suite:** `gemini_lane/progress_prize_nov2026_verifier.py` (`progress_prize_nov2026_verifier/v1`, CPU-only, `$env:CUDA_VISIBLE_DEVICES="-1"`, zero `torch`/GPU imports, zero network calls)  
**Target Prize Window:** Vesuvius Challenge Monthly Progress Prize — Deadline **November 30, 2026, 11:59 PM Pacific** ([`scrollprize.org/prizes`](https://scrollprize.org/prizes), aligned with [2026 Open Problems](https://scrollprize.org/2026_open_problems))

---

## Executive Summary & Submission Governance

> [!IMPORTANT]
> **Owner Approval & Non-Duplication Disclosure:**  
> This technical dossier packages and programmatically cross-checks all engineering advances and empirical receipts committed on `gemini/coverage-gpu-lane` for owner review ahead of the **November 30, 2026** Monthly Progress Prize deadline. **No external pull request (`scrollprize/hecate`, `priyankaranke/vesuvius-studio-multiscroll-cpt`, or `ScrollPrize/villa`) and no Progress Prize submission form has been opened without owner approval.**  
> Per [`scrollprize.org/prizes`](https://scrollprize.org/prizes), same-month contributions must be bundled cleanly and must not duplicate the already-submitted November Villa `flat-roi.patch` (`collab/codex-001.md`) or `codex/roi-full-cli`. Because the jury discourages standalone auditing or evaluation scripts as sole claims, all verification suites and JSON receipts documented here serve as reproducible evidence supporting the three technical pillars below.

### Three-Pillar Technical Summary

1. **Pillar 1 — Direct Unwrapping Advances (`Exp T9-Unwrap`, `Exp T2`, `grow_m7_patch.py`):**
   - **Seismic RGT / Wheeler Multi-Sheet Unwrapper (`gemini_lane/seismic_rgt_unwrapper.py`, `Exp T9-Unwrap` / `Q46`):** Solves a sparse 3D structure-tensor dip-guided Relative Geologic Time (RGT) conjugate-gradient system with strict monotonic projection and Wheeler stratigraphic peak picking to unwrap multiple adjacent papyrus wraps simultaneously without sheet crossings (`crossing_fraction_rgt = 0.0`). On public control `PHerc0306B` (`8.64 um`), RGT recovers **`4` non-crossing wraps** (`0.01191 cm^2` total usable CT area vs `0.002166 cm^2` single-sheet baseline, a **`5.499828x`** / **`5.50x`** area gain, `median_f_parallel_rgt = 0.202`, `phase_normal_coherence = 0.97824`). On crushed unread eligible scroll `PHerc1545` (`9.362 um`), RGT recovers **`5` non-crossing wraps** (`0.016228 cm^2` vs `0.001091 cm^2` baseline, a **`14.871486x`** / **`14.87x`** area gain) and lifts median `onsheet_score` `f_parallel` from `0.2857` to **`0.4167`** (**`+45.9%`** relative gain, `phase_normal_coherence = 0.981336`).
   - **Quasi-Isometric Spiral-Winding Resampler (`gemini_lane/spiral_winding_resample.py`, `Exp T2`):** Eliminates the `10–60 vx` column spacing distortion on equal-angle spiral fits (`du p95/p05 = 5.183`, where nominal vertex-count area is only `0.556x` of true triangulated area). Tikhonov row-coupled `regularized_arclength` resampling restores near-isometric `20.0 vx` spacing (`du p95/p05 = 1.019`), bounds maximum inter-row grid shear at **`13.58° <= 15.0°`**, and preserves exact triangulated physical area within **`-0.13%`** without ever interpolating across sentinel (`-1`) gaps.
   - **Zero-Mesh `m7` Single-Sheet Patch Grower (`gemini_lane/grow_m7_patch.py`):** Streams level-scaled interior `m7` surface predictions under a hard `ByteBudget` (`59609945` bytes / `56.85 MiB <= 100 MB` budget, `persistent_ct_on_disk = false`) on zero-mesh `PHerc0846A`, tracking `295915` single-run columns (`smooth_column_fraction = 0.99044`, `connected_components_in_grid = 1`, `0.282529 cm^2` mesh, `0.261647 cm^2` valid CT) and converging 2-pass `m7` snapping from `on_sheet_raw = 0.996237` to **`on_sheet_after_snap = 0.999059`** (`median_disp_vx = 0.294932`).
2. **Pillar 2 — FP16 Consumer-GPU Inference & Zero-Disk Source-Support Streaming (`gemini_lane/progress_prize_fp16/`, `stream_snap_render.py`):**
   - **Upstream `--precision fp16` Patch Kit (`0001-studio-upstream-fp16.patch` @ `bf66f753`, `0002-hecate-upstream-fp16.patch` @ `9cb86e50`, `0002b-hecate-upstream-fp16-after-pr3.patch` @ `f4386b19`):** Enables native `torch.float16` autocast on pre-Ampere GPUs (`T4`, `V100`, compute capability `< 8.0`) and 8 GB consumer GPUs (`RTX 4060 Laptop`). On a T4, Studio batch-1 peak allocated VRAM drops from **`11.35 GB` (`fp32`) to `1.17 GB` (`fp16`)** (`340 -> 132 ms/window`), and Hecate `9.6 um` runs **`3.0x` faster** (`41.2 s` vs `123.6 s` `fp32`, max `1` `uint8` level difference, `mean 0.047`, `r = 0.99999`).
   - **Ada (`cc 8.9`) Numerical Parity Advantage (`studio_rtx4060_parity.json` & `pherc0306b_positive_gate_rtx4060.json`):** On an 8 GB RTX 4060 Laptop GPU against CPU `fp32` on the clean `PHerc0306B` control crop, `--precision fp16` achieves `max_abs_diff = 0.002526` (`desc`, `batch=2`, `pearson_r = 0.999996`, `2164 MiB` peak allocated VRAM) and `max_abs_diff = 0.002113` (`asc`, `batch=4`, `pearson_r = 0.999993`, `2514 MiB`), whereas `bf16` exhibits `max_abs_diff = 0.01107` (`desc`) to `0.061742` (`asc`, `pearson_r = 0.996717`) — making `fp16` **`4.38x` to `29.22x` closer to `fp32`** at identical VRAM. On the full `0.4228 cm^2` `PHerc0306B` `w074` positive gate (`f_parallel = 0.9375`), `Studio fp16` (`batch=1`, `stride=48`) peaks at **`1608 MiB` allocated VRAM** (`1740 MiB` reserved, `65 °C` post-run temperature) and cleanly separates the reading face (`studio_asc` `p995 = 0.93303`, `max = 0.95549`, `frac_gt_07 = 0.158072` vs `studio_desc` `frac_gt_07 = 0.036471`).
   - **Zero-Disk Source-Support Renderer (`gemini_lane/stream_snap_render.py`, commit `4f347c1`):** Enforces positive-weight trilinear corner support (`return_support=True`, `full_source_available_yx`), rejects missing chunks (`404`) and non-zero fill values, requires full source support on every rendered plane, prevents out-of-bounds `m7` snap clipping (`m7_source_available_fraction_after_snap`), uses a common shift mask for `zero_shift_dominant_fraction`, and blocks private outputs inside any Git worktree (`_private_output_path`).
3. **Pillar 3 — `9 um` Ink Detection FP Suppression, Invariance Controls & Sanitized Screening Ledger (`Exp T8`, `Exp T6`, `Exp T4`, `Exp T10-GPU`, `firstletters_rtx4060_screen_ledger.json`):**
   - **Cloud-Masked Opposite-Face Normalization (`Exp T8` / `Q31`, `q31_cloud_masked_tfce_rescue.json`):** Evaluated on `3` real public S3 ink maps (`PHerc0343` `argus-v3` & `reader-v3`, `PHerc0841` `argus-v3`, `18010992` bytes / `17.1766 MiB <= 35.0 MiB` cap). Under `10%`, `20%`, and `30%` bilateral `2–4 mm` cloud contamination, unmasked global `z > 2` ranking (`Pipeline A`) inflates background $\sigma$ by `1.294763x`, dropping Top-3 true-text recall to `0.333333` (`33.3%`) with a `0.666667` (`66.7%`) cloud false-alarm rate. **Pipeline B (Cloud-Masked Opposite-Face Robust MAD)** achieves **`1.0` (`100%`) Top-1 hit rate, `1.0` (`100%`) Top-3 true-text recall, `0.0` (`0.0%`) cloud Top-3 false-alarm rate, and `0.0` (`0.0%`) cloud-outranks-text FPR** across all contamination levels (`0%`, `10%`, `20%`, `30%`) and under `±16 px` (`0.138–0.150 mm`) inter-face parallax.
   - **Neuroimaging Stroke-Bounded 2D TFCE (`Exp T6` / `Q34`, `tfce_q34_calibration.json`):** Calibrated on real `PHerc0306B` Line 1 (`1.568717 cm^2`). Euclidean-disk stroke-width bounding (`max_stroke_width_mm = 0.9`) suppresses `2–4 mm` bilateral clouds by **`85.5%`** (`raw_tfce_dom_max = 88.606102` -> `sb_tfce_dom_max = 12.869356`, `sb_retention_max = 0.145242`) while retaining **`75.8%`** of real `PHerc0306B` Line 1 peak TFCE (`50.971577` -> `38.633526`, `sb_retention_max = 0.757943`, `max_tfce_ratio = 11.062615`, `p995_tfce_ratio = 39.059412`, `diff_tfce_p99 = 28.094172`, `width_median_hi_mm = 0.3456`, `max_b1 = 8`, `q34_calibrated_score = 0.587144`, `passes_q34_bar = true`), rejecting blank margin, `PHerc0125` dot rows, and `PHerc1203` bilateral clouds.
   - **13-Scroll CT Radiometric Shift Audit (`Exp T4` / `Q22`) & Real GPU Orientation/Radiometric Invariance (`Exp T10-GPU`, `villa #1909`):** Auditing `13` scrolls (`4` controls + `9` eligible unread, `54635953` bytes / `52.1049 MiB <= 100.0 MiB` cap) identified `PHerc0358`'s **`+24` count** papyrus-wall shift (`p50 = 152.0` vs `128.0` pooled control reference; non-zero CT `p50 = 126.0` vs `100.0`). Direct RTX 4060 GPU verification on `PHerc0306B` `w074` (`gpu_orientation_and_nyul_control.json`) proved that `Studio` is naturally invariant to both `villa #1909` mirror/180° orientation flips (`flip_lr`, `flip_ud`, `rot180`: **`IoU@0.7 = 0.848127–0.883746`**, **`Pearson r = 0.968067–0.981438`**, **`ref_hit_retention_ge_07 = 0.922408–0.94832`**) and the `+24` count radiometric shift (`shift_plus24`: **`Pearson r = 0.99474`**, **`IoU@0.7 = 0.926112`**, **`ref_hit_retention_ge_07 = 0.963019`**, `shift_plus24_frac_gt_07_ratio_vs_orig = 1.002872`).
   - **Sanitized `106.9343 cm^2` RTX 4060 First Letters Screening Ledger (`firstletters_rtx4060_screen_ledger.json`):** Documents **`106.9343 cm^2`** of unique valid CT area across **`23` untouched meshes** (`PHerc1203`: `7` meshes / `47.7731 cm^2`; `PHerc0125`: `14` meshes / `53.5927 cm^2`; `PHerc0191`: `2` meshes / `5.5685 cm^2`, weak unflattened raw-grid caveat) and **`7` two-family (`Studio` + `A-hn`) high-resolution follow-ups**, with `0` OOM/hardware incidents (`57–72 °C` steady state after tightened guardrail polling; one brief transient `80 °C` excursion paused cleanly back to `64–67 °C`; `1608–2514 MiB` Studio / `1019–1716 MiB` A-hn) and `0` false claims.
   - **Honest `CLOSED NO-GO` Falsification of Zero-Shot 2D/1D Hand-Crafted Filters (`Exp T3`, `Exp T5`, `Exp T7`, aligned with `collab/chatgpt-031.md`):** Predeclared `±8`-slice (`±19.224 um` / `±74.928 um`) adversarial off-sheet kill-tests and locked-polarity cross-scroll replication on held-out `PHerc0841` (`2.403 um` & `9.366 um`) and `PHerc0343` (`8.64 um`) falsified all 5 candidate zero-shot hand-crafted features (`wst_s1_scale_slope` on `PHerc0841 2.403 um` scored `0.71413` on-sheet vs **`0.75526` off-sheet at `±8` slices**, `drop_vs_pm8_fixed = -0.04112 < +0.08` gate), confirming that zero-shot 2D wavelet/phase-congruency and 1D Hilbert filters respond to bulk papyrus texture rather than ink.

---

## Pillar 1: Direct Unwrapping Advances (Multi-Sheet Seismic RGT, Spiral-Winding Isometric Resampling & Zero-Mesh Patch Growing)

### 1.1 `Exp T9-Unwrap` (`Q46`): Seismic RGT / Wheeler Multi-Sheet Unwrapper vs. Single-Sheet Grower

Receipt: `gemini_lane/evidence/seismic_rgt_crushed_preflight.json` (`schema = seismic_rgt_crushed_preflight/v1`, `cpu_only = true`, `cuda_visible_devices = "-1"`, `torch_imported = false`, total HTTP `12239830` bytes / `11.673 MiB <= 83886080` cap across `24` requests, `persistent_ct_on_disk = false`).

| Metric | Public Control (`PHerc0306B` `w074`, `8.64 um`) | Crushed Eligible Unread (`PHerc1545`, `9.362 um`) |
| --- | ---: | ---: |
| Box shape (`Z, Y, X`) & dominant normal axis | `[64, 96, 96]`, axis `y` | `[64, 96, 96]`, axis `x` |
| Extracted non-crossing wraps (`sheet_count_rgt` vs baseline) | **`4`** vs `1` | **`5`** vs `1` |
| Pairwise sheet crossing fraction (`crossing_fraction_rgt`) | **`0.0`** | **`0.0`** |
| Total usable CT area (`usable_cm2_rgt_total`) | **`0.01191 cm^2`** | **`0.016228 cm^2`** |
| Best single RGT sheet area (`usable_cm2_best_sheet_rgt`) | `0.00332 cm^2` | `0.00381 cm^2` |
| Single-sheet grower baseline (`usable_cm2_single_sheet_baseline`) | `0.002166 cm^2` | `0.001091 cm^2` |
| Multi-sheet area gain (`area_gain_ratio_vs_baseline`) | **`5.499828x`** (`5.50x`) | **`14.871486x`** (`14.87x`) |
| Largest 4-connected sheet fraction (`largest_connected_sheet_frac_rgt`) | `0.718587` | `0.691895` |
| Median `onsheet_score` `f_parallel` (`median_f_parallel_rgt` vs baseline) | **`0.202`** (`[0.25, 0.2222, 0.1538, 0.1818]`) | **`0.4167`** vs `0.2857` (**`+45.9%`**) |
| Structure-tensor `phase_normal_coherence` / `planarity_median` | `0.97824` / `0.797755` | `0.981336` / `0.795498` |
| Mean / median wrap spacing (`vx`) | `15.539004` / `16.023987` | `14.847192` / `15.006357` |
| Wheeler vs Cartesian horizontal slice variance ratio | `0.961769` vs `0.998406` | `0.973501` vs `0.988648` |
| Streamed HTTP transfer (`http_bytes` / `http_mib`) | `5339019` (`5.091685 MiB`) | `6900811` (`6.581126 MiB`) |

### 1.2 `Exp T2`: Quasi-Isometric Spiral-Winding Resampler & Exact Triangulated-Area QA

Module: `gemini_lane/spiral_winding_resample.py` & `gemini_lane/test_spiral_winding_resample.py` (`74c340e`).

| Parameterization Metric (`voxel_um = 7.91`, `target_step_vx = 20.0`) | Before Resampling (Raw Equal-Angle Spiral Fit) | After `regularized_arclength` (`max_shear_deg = 15.0`) |
| --- | ---: | ---: |
| Column step stretch ratio (`du p95 / p05`) | **`5.183`** (`10–60 vx` swing) | **`1.019`** (quasi-isometric `20.0 vx`) |
| Nominal vertex-count area over true triangulated area (`nominal_over_true`) | **`0.556`** (`-44.4%` under-report) | **`0.979`** (within `2.1%` of true area) |
| Maximum corner shear deviation from `90°` (`max_shear_deg`) | `8.89°` | **`13.58°`** (`<= 15.0°` hard cap) |
| True triangulated physical `cm^2` area change (`area_change_pct`) | `0.00%` (reference) | **`-0.13%`** (within `< 2%` invariant) |

### 1.3 Zero-Mesh `m7` Single-Sheet Patch Grower Preflight (`PHerc0846A`)

Receipt: `gemini_lane/evidence/virgin_scroll_patch_preflight.json` (`schema = virgin_scroll_patch_preflight/1.0.0`, `scroll = PHerc0846A`, `m7_dataset = 20260319102732-surface-20260413222639-surface-m7-L2-th0.2.zarr`, `ct_dataset = 20260319102732-2.403um-0.2m-77keV-masked.zarr`, frame pitch `9.612 um`).

- **Single-Sheet Tracking & Continuity:** From `24` seeds along dominant normal axis `y`, `extract_height_field` tracked `295915` columns (`smooth_column_fraction = 0.99044`, `patch_columns = 278378`, `max_abs_height_step_vx = 3.0`), producing a slit-free `[77, 77]` `tifxyz` grid with `1063` valid vertices in **`1` connected component** (`mesh_area_cm2 = 0.282529`).
- **Sub-Voxel `m7` Snapping & Source Support:** 2-pass `m7` snapping at level `0` (`9.612 um`) raised `on_sheet_raw = 0.996237` to **`on_sheet_after_snap = 0.999059`** (`median_disp_vx = 0.294932`, `p90_disp_vx = 0.605321`, `m7_source_available_fraction_after_snap = 1.0`).
- **Zero-Disk Budgeted Streaming & Gate Diagnostic:** Rendered `15` slices (`[1521, 1521]` px, `valid_ct_px = 283197`, `valid_ct_cm2 = 0.261647`) within **`59609945` total bytes** (`<= 100000000` budget, `88` requests, `persistent_ct_on_disk = false`). Because tight wrap contacts fragmented the single-sheet patch below the `256 px` `onsheet_score` tile footprint (`tiles = 0`, `valid_ct_cm2 < 1.0`, `gates.passed = false`), this preflight directly motivated the `Exp T9-Unwrap` multi-sheet RGT unwrapper above.

### 1.4 `Exp T11-RGT-Scale`: 2x2 Stitched Multi-Sheet RGT Unwrapper (`PHerc0306B`, `PHerc1545`, `PHerc0846A`)

Receipt: `gemini_lane/evidence/seismic_rgt_stitched_patches.json` (`schema = seismic_rgt_stitched_patches/v1`, `cpu_only = true`, `torch_imported = false`, `http_bytes = 34512551` / `32.914 MiB <= 125829120` cap across `46` requests, `persistent_ct_on_disk = false`).

- Scales multi-sheet RGT unwrapping across `2x2` overlapping lateral tiles (`[64, 192, 192]`) with global phase synchronization and seamless stitching, recovering **`34` non-crossing wraps** (`0.215204 cm^2` total usable CT area, **`max_crossing_fraction_rgt = 0.0`**, **`max_seam_phase_rmse_rad = 0.102162`**):
  - `PHerc0306B` (`8.64 um`): `11` sheets (`0.060535 cm^2`, **`9.643477x`** baseline area gain, `seam_phase_rmse_rad = 0.102162`; note `snap_passes = 0` because no `m7` Zarr was supplied on this public control).
  - `PHerc1545` (`9.362 um`): `11` sheets (`0.060539 cm^2`, **`29.00924x`** baseline area gain, `mean_on_sheet_after_snap_rgt = 0.896531`).
  - `PHerc0846A` (`9.612 um`): `12` sheets (`0.09413 cm^2`, **`11.793379x`** baseline area gain, `mean_on_sheet_after_snap_rgt = 0.960576`).
- **Honest Caveat:** In all three larger `2x2` boxes, the conjugate-gradient solver reached its `1000`-iteration cap before formal convergence (`cg_converged = false`), with strict monotonicity enforced via projection (`monotonic_projection_fraction <= 0.000826`).

### 1.5 `Exp T15-GridQuality-Audit`: Winner vs. Eligible Screening Meshes vs. HF Spiral Fits & Coordinate Convention

Receipt: `gemini_lane/evidence/grid_quality_audit.json` (`schema = grid_quality_audit/v1`, `cpu_only = true`, `torch_imported = false`, `19646463` bytes / `18.736 MiB <= 62914560`, `556` HTTP `200` requests, `persistent_downloads_on_disk = false`) and `gemini_lane/leak_guard.py` (OS-username CI fix across `progress_prize_fp16` tests without altering `.patch` files).

| Surface Cohort (Claude `023` `8`-Cell Shear & Area Inflation) | Meshes | `shear_8cell` `abs_p95_deg` | Area `> 20°` (`%`) | True / Nominal Area Inflation | Provenance / Audit Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **`PHerc0211` Winner Mesh** (`crop1711-lines123-v9`) | `1` | **`2.889°`** | `0.001%` | **`0.9999x`** | Published winner-mesh baseline |
| **`PHerc0125` Screened Negative (`pscamillo`)** | `6` | `4.495°` | **`0.124%`** | **`1.0408x`** | Clean winner-grade geometry (`0` flattening needed) |
| **`PHerc0191` HF Spiral Fits (`w030`/`w050` cohort)** | `33` | `45.243°` | **`58.959%`** | `1.4999x` | Heavily sheared unflattened fit grids |
| **`PHerc0826` HF Spiral Fits (3 bands)** | `99` | `44.035°–67.328°` | `51.232%–74.191%` | `1.9405x–2.1753x` | Requires 2D flattening (`flatboi` / `metric_flatten`) before GPU screening (`0` ink runs by Gemini; 1D `Exp T2` cannot remove 2D shear) |
| **`Exp T11` Synthetic 5-Wrap RGT Lattice** | `5` | **`0.32°`** | `0.0%` | `1.0065x` | `crossing_fraction_rgt = 0.0` (caveat: `1 vx` lattice vs `20 vx` spiral fits) |

- **Code-Verified Coordinate & Grid Provenance:** `stream_snap_render.py::render_surface_stack` samples at vertex-aligned `p * scale` (`p / 20` for `scale=0.05`, zero `+0.5` offset, `0.5` output-px difference vs `(p + 0.5) / 20`); no prior `gemini_lane` screen applied per-surface flattening (`PHerc0826` had CPU on-sheet triage only).

---

## Pillar 2: FP16 Consumer-GPU Inference & Zero-Disk Source-Support Streaming

### 2.1 Upstream `--precision fp16` Patch Kit (`Studio` & `Hecate`)

Directory: `gemini_lane/progress_prize_fp16/` (`test_fp16_patches.py`, `verify_upstream_gpu_burst.py`, `prepare_hecate_parity_crops.py`).

| Patch File | Target Upstream Repository & Pinned Base Commit | Touched Files & Hunks | Key Engineering Change |
| --- | --- | --- | --- |
| `0001-studio-upstream-fp16.patch` | `priyankaranke/vesuvius-studio-multiscroll-cpt` @ `bf66f753176792a966bca17822629fb8821f20c8` | `README.md`, `infer_release.py`, `studio-multiscroll-cpt-20261023/hf/inference.py`, `test_infer_release.py`, `test_inference_tiling.py` (`10` hunks, `9239` bytes, LF-clean) | Adds `--precision {auto,bf16,fp16,fp32}` (keeping `--amp` alias); `auto` selects `fp16` on `cc < 8.0` (`T4`, `V100`), `bf16` on `cc >= 8.0`, `fp32` on CPU; explicit `--precision fp16` available on all CUDA GPUs. |
| `0002-hecate-upstream-fp16.patch` | `scrollprize/hecate` (`main`) @ `9cb86e500e944b11a06a7020403cde5dffb5bcb2` | `README.md`, `hecate.py` (`4` hunks, `4221` bytes, LF-clean) | Adds `fp16` to `--precision {fp32,bf16,fp16}` (`default='fp32'` unchanged); runs `torch.autocast(device_type='cuda', dtype=torch.float16)` when `precision == 'fp16'`. |
| `0002b-hecate-upstream-fp16-after-pr3.patch` | `scrollprize/hecate` (`refs/pr/3` `--resample`) @ `f4386b19e0639f1c3da79572309ecc3f4cec760e` | `README.md`, `hecate.py` (`4` hunks, `4399` bytes, LF-clean) | Conflict-resolved variant rebased onto open PR #3 (`--resample`), preserving identical `fp16` autocast semantics. |

### 2.2 Studio & Hecate FP16 Numerical Parity and VRAM Profile (`studio_rtx4060_parity.json` & T4 Reference)

Receipts: `gemini_lane/evidence/studio_rtx4060_parity.json` (RTX 4060 Laptop 8 GB, `cc 8.9`, `169 x 169` `PHerc0306B` smoke crop `4a81e193...`, `25` windows at `stride = 48` Hann, weights `96b32628...`) and `gemini_lane/evidence/pherc0306b_positive_gate_rtx4060.json` (`0.4228 cm^2` `PHerc0306B` `w074` positive gate, `566400` valid px, `f_parallel = 0.9375`).

| Hardware & Run Tag | Precision & Order | Batch / Stride | Peak Allocated VRAM | `max \|Δp\|` vs `fp32` | `mean \|Δp\|` vs `fp32` | Pearson `r` vs `fp32` | Speed / Thermal Profile |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| **Colab T4 (`cc 7.5`, 16 GB)** | `Studio fp32` | `batch=1`, `128 px` | **`11.35 GB`** | `0.0` (ref) | `0.0` (ref) | `1.0` | `340 ms/window` |
| **Colab T4 (`cc 7.5`, 16 GB)** | `Studio fp16` | `batch=1`, `128 px` | **`1.17 GB`** (`-89.7%`) | `0.0019` | `0.00014` | `>= 0.99999` | **`132 ms/window`** (`2.58x` faster) |
| **Colab T4 (`cc 7.5`, 16 GB)** | `Hecate 9.6 um fp32` | `batch=32`, 2D | default `fp32` | `0` `uint8` (ref) | `0.0` (ref) | `1.0` | `123.6 s` |
| **Colab T4 (`cc 7.5`, 16 GB)** | `Hecate 9.6 um fp16` | `batch=32`, 2D | `fp16` autocast | **`1` `uint8` level** | **`0.047` `uint8`** | **`0.99999`** | **`41.2 s`** (**`3.0x` faster**; `bf16` takes `228.6 s`) |
| **RTX 4060 (`cc 8.9`, 8 GB)** `cuda_bf16_desc_s48_b1` | `Studio bf16`, `desc` | `batch=1`, `s=48` | `1733 MiB` | `0.011095` | `0.001753` | `0.999929` | `3.47 s` (`138.7 ms/win`) |
| **RTX 4060 (`cc 8.9`, 8 GB)** `cuda_bf16_desc_s48_b2` | `Studio bf16`, `desc` | `batch=2`, `s=48` | `2164 MiB` | `0.01107` | `0.001758` | `0.999928` | `3.1 s` (`124.1 ms/win`) |
| **RTX 4060 (`cc 8.9`, 8 GB)** `cuda_fp16_desc_s48_b2` | **`Studio fp16`, `desc`** | `batch=2`, `s=48` | **`2164 MiB`** (`2.11 GiB`) | **`0.002526`** (`4.38x` tighter) | **`0.00041`** | **`0.999996`** | `3.29 s` (`131.5 ms/win`) |
| **RTX 4060 (`cc 8.9`, 8 GB)** `cuda_bf16_desc_s48_b4` | `Studio bf16`, `desc` | `batch=4`, `s=48` | `3010 MiB` | `0.01121` | `0.001755` | `0.999928` | `3.41 s` (`136.5 ms/win`) |
| **RTX 4060 (`cc 8.9`, 8 GB)** `cuda_fp16_desc_s48_b4` | **`Studio fp16`, `desc`** | `batch=4`, `s=48` | **`3010 MiB`** | **`0.002538`** (`4.42x` tighter) | **`0.000408`** | **`0.999996`** | `3.35 s` (`133.9 ms/win`) |
| **RTX 4060 (`cc 8.9`, 8 GB)** `cuda_bf16_asc_s48_b4` | `Studio bf16`, `asc` | `batch=4`, `s=48` | `2514 MiB` | `0.061742` | `0.006716` | `0.996717` | `2.42 s` (`97.0 ms/win`) |
| **RTX 4060 (`cc 8.9`, 8 GB)** `cuda_fp16_asc_s48_b4` | **`Studio fp16`, `asc`** | `batch=4`, `s=48` | **`2514 MiB`** | **`0.002113`** (**`29.22x` tighter**) | **`0.000156`** | **`0.999993`** | `2.3 s` (`92.0 ms/win`) |
| **RTX 4060 Positive Gate** (`PHerc0306B` `w074`, `0.4228 cm^2`) | **`Studio fp16`, `asc` / `desc`** | `batch=1`, `s=48` | **`1608 MiB`** (`1740 MiB` res.) | `studio_asc` `p995 = 0.93303`, `max = 0.95549` | `frac_gt_07`: `0.158072` (`asc`) vs `0.036471` (`desc`) | `f_parallel = 0.9375` | `65 °C` (`<= 74 °C` cap), `POSITIVE_GATE_PASS` |

### 2.3 Zero-Disk Source-Support Renderer (`stream_snap_render.py`, Commit `4f347c1`)

Documentation & Regression Suite: `gemini_lane/SOURCE_SUPPORT_PATCH.md` and `gemini_lane/test_stream_source_support.py` (`24` offline unit tests, `21` of which fail against the pre-port renderer).

- **Positive-Weight Trilinear Stencil (`sample_trilinear(..., return_support=True)`):** Requires every corner with strictly positive trilinear weight to lie in bounds, come from a decoded chunk (never a `404` missing chunk or `fill_value`), and be finite (`NaN` zeroed before `map_coordinates` while keeping support `false`).
- **Full-Stack Plane Support (`full_source_available_yx`):** Prevents out-of-bounds depths from being masked by a single non-zero plane (`stack.max(axis=0) > 0`), ensuring `valid_ct_cm2` reflects full 3D source support across all rendered slices while retaining backward-compatible masked-zero exclusion by default (`require_nonzero_intensity=True`).
- **Unclipped `m7` Ray Snapping (`m7_source_available_fraction_after_snap`):** Rejects out-of-bounds centres and unsupported ray samples (`-inf`), floors the search grid (`floor(search_radius / step)`), and guards quadratic sub-voxel peak fitting against non-finite neighbours.
- **Common Shift Mask & Git Worktree Guard:** `layer_coherence_score` evaluates all candidate shifts over a common eroded valid mask (`zero_shift_dominant_fraction`), and `_private_output_path` refuses to write raw arrays or private receipts inside any Git worktree (following symlinks).

---

## Pillar 3: 9 um Ink Detection FP Suppression, Invariance Controls & Sanitized Screening Ledger

### 3.1 `Exp T8` (`Q31` + `Q34`): Cloud-Masked Opposite-Face Normalization on Real `argus-v3` & `reader-v3` Maps

Receipt: `gemini_lane/evidence/q31_cloud_masked_tfce_rescue.json` (`device = cpu_only_numpy_scipy_pil`, `18010992` bytes / `17.1766 MiB <= 35.0 MiB` cap across `pherc0343_argus_v3` `5572656` B, `pherc0343_reader_v3` `4992860` B, and `pherc0841_argus_v3` `7445476` B).

| Bilateral Cloud Area Fraction (`2–4 mm`, peak `0.85–0.92`) | Pipeline | Trials | Top-1 True-Text Hit Rate | Top-3 True-Text Recall | Top-3 Clean-Ref Recall | Cloud Top-3 False-Alarm Rate | Cloud-Outranks-Text FPR | Mean $\sigma$ Inflation Factor |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0` (`0%` clean) | `A_global_z2` | `3` | `1.0` | `1.0` | `1.0` | `0.0` | `0.0` | `1.0` |
| `0.0` (`0%` clean) | **`B_masked_mad`** | `3` | **`1.0`** | **`1.0`** | **`1.0`** | **`0.0`** | **`0.0`** | `1.0` |
| `0.1` (`10%` clouds) | `A_global_z2` | `6` | `0.666667` | `0.666667` | `0.555556` | `0.333333` | `0.154365` | `1.170588` |
| `0.1` (`10%` clouds) | **`B_masked_mad`** | `6` | **`1.0`** | **`1.0`** | **`0.888889`** | **`0.0`** | **`0.0`** | `0.928981` |
| `0.2` (`20%` clouds) | `A_global_z2` | `6` | `0.666667` | `0.555556` | `0.333333` | `0.444444` | `0.205322` | `1.252941` |
| `0.2` (`20%` clouds) | **`B_masked_mad`** | `6` | **`1.0`** | **`1.0`** | **`0.888889`** | **`0.0`** | **`0.0`** | `0.785895` |
| `0.3` (`30%` clouds) | `A_global_z2` | `6` | `0.333333` | `0.333333` | `0.222222` | `0.666667` | `0.355982` | `1.294763` |
| `0.3` (`30%` clouds) | **`B_masked_mad`** | `6` | **`1.0`** | **`1.0`** | **`0.888889`** | **`0.0`** | **`0.0`** | `0.666546` |
| `0.3` (`30%` clouds) | `C_masked_sb_tfce` | `6` | `0.833333` | `0.611111` | `0.555556` | `0.388889` | `0.048337` | `0.666546` |
| `0.2` + `±16 px` Parallax (`0.138–0.150 mm`) | `A_global_z2` | `3` | `0.666667` | `0.555556` | — | `0.444444` | `0.16369` | — |
| `0.2` + `±16 px` Parallax (`0.138–0.150 mm`) | **`B_masked_mad`** | `3` | **`1.0`** | **`1.0`** | — | **`0.0`** | **`0.0`** | — |

### 3.2 `Exp T6` (`Q34`): Neuroimaging Stroke-Bounded 2D TFCE & Empirical-Null Calibration

Receipt: `gemini_lane/evidence/tfce_q34_calibration.json` (`voxel_um = 8.64`, `tfce_H = 2.0`, `tfce_E = 0.5`, `max_stroke_width_mm = 0.9`, `total_http_bytes = 8494158 <= 31457280`).

| Calibration Case (`1.568717 cm^2`, `[480, 4378]` px) | Raw TFCE Max (`raw_tfce_dom_max`) | Stroke-Bounded TFCE Max (`sb_tfce_dom_max`) | Retention (`sb_retention_max`) | Opposite-Face Ratio (`max_tfce_ratio` / `p995_tfce_ratio`) | Differential TFCE `p99` (`diff_tfce_p99`) | Median Width at `d >= 0.7` (`mm`) | Betti Loops (`topo_max_b1`) | Calibrated `Q34` Score | Passes `Q34` Bar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **`a_real_pherc0306b_line1`** (Real Single-Line Win) | `50.971577` | **`38.633526`** | **`0.757943`** (**`75.8%`**) | **`11.062615`** / **`39.059412`** | **`28.094172`** | `0.3456` | `8` | **`0.587144`** | **`true`** (`single_face_looped_stroke_pass`) |
| **`a_sub_real_pherc0306b_line1_10mm`** (`0.414574 cm^2` Subwindow) | `41.075161` | **`32.627556`** | **`0.794338`** (`79.4%`) | **`15.2493`** / **`58.013613`** | **`25.246347`** | `0.38016` | `2` | **`0.480391`** | **`true`** (`single_face_looped_stroke_pass`) |
| `b_real_pherc0306b_blank_margin` (Real On-Sheet Blank) | `5.184484` | `5.158187` | `0.994928` | `1.477034` / `1.355871` | `0.412226` | `0.096907` | `0` | `0.0` | `false` (`blank_or_weak_null_rejected`) |
| `c_synthetic_pherc0125_dot_row` (`0.6–1.2 mm` Dot Chain) | `31.663456` | `22.009872` | `0.695119` | `88.03949` / `82.431938` | `15.310255` | `0.870199` | `0` | `0.002168` | `false` (`single_face_dot_or_nonstroke_rejected`) |
| `d_synthetic_pherc1203_bilateral_clouds` (`2–4 mm` Clouds) | `88.606102` | **`12.869356`** | **`0.145242`** (**`85.5%` cut**) | `1.619453` / `1.69167` | `6.521737` | `1.624504` | `0` | `0.0` | `false` (`bilateral_or_wide_cloud_rejected`) |

### 3.3 `Exp T4` (`Q22`) 13-Scroll Radiometric Audit & `Exp T10-GPU` Orientation/Radiometric Invariance

Receipts: `gemini_lane/evidence/q22_radiometric_shift_audit.json` (`13` scrolls: `4` controls + `9` eligible unread, `54635953` bytes / `52.1049 MiB <= 100.0 MiB` cap, `all_scrolls_unflagged_after_nyul = true`) and `gemini_lane/evidence/gpu_orientation_and_nyul_control.json` (`PHerc0306B` `w074`, `0.4228 cm^2`, `566400` valid px, `peak_allocated_mib = 1608`, `post_run_temp_c = 62`).

- **`Q22` Multi-Scroll Radiometric Audit Finding (`PHerc0358` `+24` Count Shift):** Across the pooled reference controls (`PHerc0841`, `PHerc0343`, `PHerc0306B`), the median papyrus-wall core (`> p50`) landmark sits at **`128.0`** (non-zero CT `p50 = 100.0`, Nyúl-Udupa landmarks `[23.0, 39.0, 56.0, 100.0, 127.0, 146.0, 182.0]`). On eligible unread scroll `PHerc0358` (`9.362 um`, `113 keV`), the papyrus-wall core `p50` shifts upward by **`+24` counts** to **`152.0`** (non-zero CT `p50 = 126.0`, `+26` counts; `p99 = 201.0`, `frac_ge_200 = 0.011594` overall / `0.023336` wall core / `0.035262` upper tercile).
- **`Exp T10-GPU` Arm 1 Real GPU Invariance Table (`PHerc0306B` `w074` Clean Positive Control):**

| Variant (`PHerc0306B` `w074`, `0.4228 cm^2`) | `studio_asc` `p995` / `max` | `studio_asc` `frac_gt_07` | `studio_desc` `frac_gt_07` | Pearson `r` vs `orig` | MAE vs `orig` | `IoU@0.5` | `IoU@0.7` | Hit Retention `@0.7` (`ref_hit_retention_ge_07`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `orig` (Reference `asc` reading face) | `0.93303` / `0.95549` | `0.158072` | `0.036471` | `1.0` | `0.0` | `1.0` | `1.0` | `1.0` (`100.0%`) |
| `flip_lr` (`villa #1909` horizontal mirror) | `0.93196` / `0.95889` | `0.15575` | `0.033166` | **`0.980776`** | `0.030568` | `0.848626` | **`0.867818`** | **`0.922408`** (`92.2%`) |
| `flip_ud` (`villa #1909` vertical mirror) | `0.93273` / `0.95183` | `0.161453` | `0.038785` | **`0.981438`** | `0.028552` | `0.865023` | **`0.883746`** | **`0.94832`** (`94.8%`) |
| `rot180` (`villa #1909` 180° rotation) | `0.931` / `0.95506` | `0.162544` | `0.047046` | **`0.968067`** | `0.038985` | `0.827378` | **`0.848127`** | **`0.930807`** (`93.1%`) |
| **`shift_plus24`** (`Q22` `+24` count shift control) | `0.9349` / `0.95639` | `0.158526` (`1.002872x`) | `0.042878` | **`0.99474`** | `0.015317` | `0.929004` | **`0.926112`** | **`0.963019`** (`96.3%`) |
| `nyul_udupa` (Piecewise-linear landmark warp) | `0.92523` / `0.94758` | `0.132249` | `0.031132` | `0.948563` | `0.052338` | `0.760478` | `0.809317` | `0.821539` (`82.2%`) |
| `shift_plus24_nyul` (`+24` shift + Nyúl-Udupa) | `0.92523` / `0.94758` | `0.132249` (`0.836638x`) | `0.031132` | `0.948563` | `0.052338` | `0.760478` | `0.809317` | `0.821539` (`82.2%`) |

### 3.4 Sanitized `106.9343 cm^2` RTX 4060 First Letters Screening Ledger

Receipt: `gemini_lane/evidence/firstletters_rtx4060_screen_ledger.json` (`23` untouched meshes screened, `7` two-family `Studio` + `A-hn` follow-up ROIs, `0` candidates found, `0` OOM/hardware incidents, GPU steady temperature `57–72 °C` under tightened guardrail polling).

| Scroll | Meshes Screened | Unique Valid CT Area (`cm^2`) | Two-Family (`Studio` + `A-hn`) Follow-Up ROIs | Mesh IDs (`valid_ct_cm2`) | Verdict & Empirical Failure-Mode Classification |
| --- | ---: | ---: | ---: | --- | --- |
| **`PHerc1203`** (`9.362 um`) | `7` | **`47.7731 cm^2`** | `4` (`auto_grown_20251005230118636`, `auto_grown_20250930000321811` x2 ROIs, `auto_grown_20250929222256117`) | `auto_grown_20251005230118636` (`7.3613`), `auto_grown_20251005231446963` (`3.5848`), `auto_grown_20250930000321811` (`6.7506`), `auto_grown_20251005231446965` (`11.3987`), `auto_grown_20250929222256117` (`7.0882`), `auto_grown_20250930104534929` (`7.8014`), `auto_grown_20250923163217042` (`3.7881`) | **`NEGATIVE`** — Bilateral diffuse `2–4 mm` papyrus-mottle clouds and delamination-seam / fiber-flap blobs; zero connected Greek letter strokes. |
| **`PHerc0125`** (`9.362 um`) | `14` | **`53.5927 cm^2`** | `3` (`z14144_w035`, `z8736_w035` + 9-variant `max_q34_score = 0.007359`, `z15344_w035` + 9-variant `max_q34_score = 0.0`) | Cohort A: `z14144_w035` (`4.0123`), `z10544_w035` (`3.9752`), `z15936_w030` (`3.5699`), `z6944_w030` (`3.1967`), `z9936_w035` (`3.9944`), `z8736_w035` (`3.9087`); Tier-1B: `z9344_w015` (`1.7503`), `z15936_w035` (`4.2579`), `z8144_w030` (`3.3149`), `z14144_w050` (`5.5807`); S0-Top Tier-2: `z16544_w025` (`3.0344`), `z15344_w035` (`4.2505`), `z15936_w050` (`5.8392`), `z14736_w025` (`2.9076`) | **`NEGATIVE`** — Winner-grade `pscamillo` geometry (`0.124%` `>20°` shear); diffuse clouds, off-sheet split-layer delamination blobs (`dz=-6`), `128 px` tile mottle, and seam specks; zero connected letter strokes. |
| **`PHerc0191`** (`9.362 um`) | `2` | **`5.5685 cm^2`** | `0` | `w030` (`2.1151`), `w050` (`3.4534`) | **`NEGATIVE (low-sensitivity/weak unflattened HF grid)`** — Unflattened (`58.959%` `>20°` shear per `Exp T15` & `claude-025`) delaminated papyrus with sparse crack-edge specks only. |
| **Total** | **`23`** | **`106.9343 cm^2`** | **`7`** | `23` public mesh segments across `3` eligible scrolls | **`0` candidates** (`100%` sanitized negative ledger) |

### 3.5 `Exp T13`, `Exp T16-GPU` & `Exp T17-GPU`: Multi-Line `Q34` Calibration, Grid-Shear Tolerance & `PHerc. 0211` / `PHerc. 0306B` Validation

Receipts: `gemini_lane/evidence/tiled_q34_cloud_masked_calibration.json` (`26505150` B / `25.2773 MiB <= 40 MiB`), `gemini_lane/evidence/gpu_shear_and_0211_control.json` (`post_run_temp_c = 60`, `persistent_ct_on_disk = false`), and `gemini_lane/evidence/gpu_0211_and_0306b_calibration_matrix.json` (`post_run_temp_c = 57`, `persistent_ct_on_disk = false`).

- **`Exp T13` (`Q31 + Q34` Area-Matched & Rim-Eroded Ranker, `tiled_q34_cloud_masked_calibration.json`):** Fixes multi-line super-cluster denominator collapse (`PHerc0343 argus-v3` `area_matched_retention_max = 0.967727`, `reader-v3 = 0.936248`, `PHerc0841 argus-v3 = 0.733878`) and opposite-face rim-speck collapse, lifting pass rate from `2/5` (legacy `Exp T6`) to **`5/5` positive regimes and `0/2` negative regimes** (plus blank-margin, step-stair seam-blob, bilateral `2–4 mm` cloud, and dot-row subcontrols rejected).
- **`Exp T16-GPU` Arm A — Real `Studio fp16` Grid Shear Stress on `PHerc0306B` `w074` (`0°, ±15°, ±30°`):** Over `464256` 32 px rim-eroded interior pixels (`8.64 um`, `stride = 48`), **`±15°` retains `>= 0.939702`** of canonical `> 0.7` hits (`IoU@0.7 = 0.880728–0.892256`, `r = 0.98076–0.981776`), and **`±30°` retains `>= 0.906106`** (`min_overall_interior_pearson_r = 0.94453`, `IoU@0.5 = 0.733546–0.755497`). This complements `chatgpt-adaptive-049`'s CPU `A-hn` shear run (`-0.286614` middle-third AP drop at `-30°`).
- **`Exp T16-GPU` Arm B & `Exp T17-GPU` (`PHerc0211` Winner Matrix & `PHerc0306B` Layer-Order Card):**
  - On the `0.6421 cm^2` `PHerc0211` 3-row winner crop (`9.362 um`), `Studio fp16` recovers all 3 letter rows with `dom_over_opp_gt_07_ratio = 7.535` (`snap0 s48`, `desc frac_gt_07 = 0.296817` vs `asc 0.039392`). Applying `2x m7-L0` snap raises `on_sheet_after_snap` from `0.829688` to **`0.9953125`** and lifts `studio_s48` asymmetry to **`8.5508x`** (`r = 0.967597`, `ret@0.7 = 0.920321`); screening at **`snap2 + studio_s96`** passes the calibration gate (`stride96_snap2_screening_gate_pass = true`, **`6.8531x`** asymmetry, **`0.869617`** `> 0.7` hit retention vs canonical `snap0 s48`, `r = 0.946589`, `fp_diagnostics = true`). `A-hn hn4_s0` independently recovers the same 3 letter rows on both `snap0` (`4.62x`) and `snap2` (`4.0028x`).
  - On `PHerc0306B` `w074` (`0.4228 cm^2`), `Studio` (`s48`: `4.334x`, `s96`: `3.7432x`, `ret@0.7 = 0.912254`) and `A-hn hn4_s0` (`asc 0.150174` vs `desc 0.053574`, **`2.8031x`**) align in the **exact same `asc` depth order** (`same_face_direction_between_families = true`), accompanied by a 31-layer CT correlation profile card.
  - **Honest Non-Gating Q34 Caveat (`Q5` Confirmation):** Whereas `Exp T13` fixes multi-line area retention (`0.694452–0.743516` on `PHerc0211`, `0.931572` on `PHerc0306B`) and passes `top_tile_passes_q34_bar = true` on `PHerc0211 snap2 s48` (`0.420546`), real two-order neural inference produces faint opposite-order bleed-through (`diff_tfce_asymmetry_ratio = 2.63642` on `PHerc0211 snap2 s48` vs `Exp T13`'s `>= 4.5` blank-margin calibration threshold), causing both legacy `Exp T6` (`passes_q34_bar = false`) and global `Exp T13` (`global_upgraded_passes_q34_bar = false`) to zero global scores. `Q34` is therefore retained strictly as a **non-gating ranking diagnostic** paired with `screening_fp_filter` (`6/6` true on `PHerc0211`) and visual montage inspection.

---

## Honest Falsification Ledger (`CLOSED NO-GO` Zero-Shot 2D/1D Hand-Crafted Filters)

Aligned with `collab/chatgpt-031.md`, every zero-shot 2D spatial filter (Wavelet Scattering, Kovesi Phase Congruency, StructN2V directional fiber subtraction, GigaMesh MSII) and 1D depth-waveform feature (OCT Hilbert envelope/phase asymmetry, Cryo-ET slice coherence, low-`δ/β` Paganin `pag50`/`pag0`) was subjected to predeclared toroidal-shift null gates and the **`±8`-slice adversarial off-sheet kill-test**:

| Experiment & Receipt | Evaluated Public Control Volumes | Predeclared Gate | Key Quantitative Outcome | Final Verdict |
| --- | --- | --- | --- | --- |
| **`Exp T3`** (`pherc0139_transverse_physics_ablation.json`) | `PHerc0139` `2.399 um` surface volume + `9.362 um` `pag0` / `pag50` / `pag1000` (`99517051` B / `94.9069 MiB`) | `ROC-AUC >= 0.70` and `>= +0.10` above 16-shift toroidal null | At `2.399 um`, baseline `raw_ct_surface` scored **`0.71484`** (`+0.10197` vs null), whereas MSII peaked at `0.53578` (`msii_convexity_140um`) and StructN2V fiber-subtracted residual peaked at `0.63733`. At `9.362 um`, low-`δ/β` `pag50`/`pag0` failed to beat `pag1000:normal_scattering_variance` (**`0.70945`**, `+0.10609` vs null). | **`CLOSED NO-GO`** |
| **`Exp T5`** (`transverse_wavelet_oct_ablation.json`) | `PHerc0306B` `8.64 um` (`w074` Line 1) + `PHerc0139` `2.399 um` (`95715011` B / `91.2809 MiB`) | `PHerc0306B` `AUC >= 0.70` & `+0.10` vs null; `PHerc0139` `AUC >= 0.765` | On `PHerc0306B` `8.64 um`, all 21 transverse features failed (`best = wst_s1_isotropic` at **`0.69062`** `< 0.70`). On `PHerc0139` `2.399 um`, `wst_s1_scale_slope` (`0.79674`) and `oct_hilbert_env_peak` (`0.77962`) passed the on-sheet gate, triggering the `Exp T7` adversarial off-sheet kill-test. | `PHerc0306B`: **`CLOSED NO-GO`** (`PHerc0139` forwarded to `Exp T7`) |
| **`Exp T7`** (`transverse_cross_scroll_replication.json`) | Held-out `PHerc0841` `2.403 um` (human ink labels), `PHerc0841` `9.366 um`, & `PHerc0343` `8.64 um` (`68226076` B / `65.0655 MiB`) | Locked `Exp T5` polarity: `on_sheet >= 0.70`, `delta_null >= +0.10`, AND off-sheet drop `>= +0.08` at `±8` slices & volume boundaries | **`0` features passed** on any volume. On `PHerc0841` `2.403 um`, `wst_s1_scale_slope` reached **`0.71413`** on-sheet (`delta_null = 0.10859`) but scored **`0.75526` off-sheet at `±8` slices (`±19.224 um`)** (`drop_vs_pm8_fixed = -0.04112`), while `wst_s1_isotropic` (`0.18837`), `pc_min_moment_corners_pooled` (`0.2152`), and `rank_fusion_t5_triplet` (`0.25614`) flipped polarity. At `8–9 um`, best fixed-polarity AUCs were `0.5113` (`PHerc0841` `9.366 um`) and `0.60025` (`PHerc0343` `8.64 um`). | **`CLOSED NO-GO`** (across all 3 volumes) |
| **`Exp T12-Villa1739`** (`villa1739_cross_resolution_bridge.json`) | `PHerc0841 w00` (`2.403 um -> 9.366 um` human-label bridge) + 42D linear Ridge Fold-B & zero-shot transfer on `PHerc0343` & `PHerc0306B` (`71492788` B / `68.1808 MiB`) | Bridge IoU alignment; Fold-B & cross-scroll locked-`+` AUC `>= 0.70`, `delta_null >= +0.10`, `±8`-slice drop `>= +0.08` | Separable block-area integration aligned `2.403 um` human ink labels onto `9.366 um` (`shift = (0, -9) px`, IoU `0.56633` -> **`0.57853`**; reusable detector validation bridge), **but 42D linear Ridge failed `8/8` checks** (Fold-B `human_2p4um_bridged` AUC **`0.53408`** vs human labels, **`0.58201`** vs `argus-v3`; cross-scroll `0.44233–0.60738`). | **`CLOSED NO-GO`** (42D Ridge linear transfer) |

---

## Committed Artifact & Receipt SHA-256 Manifest

Every file below is committed in the repository and verified offline by `gemini_lane/progress_prize_nov2026_verifier.py`:

| Artifact / Evidence Receipt | Size (`LF bytes`) | SHA-256 Digest Prefix (`12` hex, LF) | Role in Dossier |
| --- | ---: | --- | --- |
| `0001-studio-upstream-fp16.patch` | `9239` | `0be906f7f448` | Pillar 2: Studio `--precision {auto,bf16,fp16,fp32}` patch (`bf66f753`) |
| `0002-hecate-upstream-fp16.patch` | `4221` | `97330b71e49b` | Pillar 2: Hecate `--precision fp16` patch (`main` @ `9cb86e50`) |
| `0002b-hecate-upstream-fp16-after-pr3.patch` | `4399` | `fc4a1d52fdb6` | Pillar 2: Hecate `--precision fp16` patch (`refs/pr/3` @ `f4386b19`) |
| `SOURCE_SUPPORT_PATCH.md` | `4857` | `a1175d7d1826` | Pillar 2: Zero-disk trilinear source-support specification (`4f347c1`) |
| `stream_snap_render.py` | `63522` | `d93e6e3c01aa` | Pillar 2: Zero-disk streaming surface renderer & `m7` snapper |
| `seismic_rgt_crushed_preflight.json` | `4728` | `2fa7fa133bb1` | Pillar 1: `Exp T9-Unwrap` multi-sheet RGT (`PHerc0306B` `5.50x`, `PHerc1545` `14.87x`) |
| `virgin_scroll_patch_preflight.json` | `10098` | `0aad37652da4` | Pillar 1: `grow_m7_patch.py` zero-mesh `PHerc0846A` preflight |
| `studio_rtx4060_parity.json` | `7402` | `c4bdf9898491` | Pillar 2: RTX 4060 `fp16` vs `bf16` vs CPU `fp32` numerical parity |
| `pherc0306b_positive_gate_rtx4060.json` | `1814` | `78280ff69e90` | Pillar 2: RTX 4060 `PHerc0306B` `w074` clean positive control gate (`1608 MiB`) |
| `firstletters_rtx4060_screen_ledger.json` | `36828` | `0568c53971b9` | Pillar 3: Sanitized `106.9343 cm^2` (`23` meshes, `7` follow-ups) negative ledger |
| `gpu_orientation_and_nyul_control.json` | `17727` | `7dc935a6bc2b` | Pillar 3: `Exp T10-GPU` `villa #1909` mirror/rot180 & `Q22` `+24` shift invariance |
| `q31_cloud_masked_tfce_rescue.json` | `113816` | `f7e716c85916` | Pillar 3: `Exp T8` (`Q31`) Cloud-Masked MAD `100%` Top-3 recall / `0%` cloud FPR |
| `tfce_q34_calibration.json` | `58517` | `eff438ede2e2` | Pillar 3: `Exp T6` (`Q34`) Stroke-Bounded 2D TFCE (`85.5%` cloud cut, `75.8%` Line 1) |
| `q22_radiometric_shift_audit.json` | `48764` | `28729ad5c816` | Pillar 3: `Exp T4` (`Q22`) 13-scroll CT radiometric audit (`PHerc0358` `+24` shift) |
| `transverse_cross_scroll_replication.json` | `61544` | `d68573fc4e38` | Falsification: `Exp T7` cross-scroll replication & `±8`-slice off-sheet kill-test |
| `transverse_wavelet_oct_ablation.json` | `38208` | `56b3cfaea0df` | Falsification: `Exp T5` Wavelet Scattering / Phase Congruency / OCT 31D ablation |
| `pherc0139_transverse_physics_ablation.json` | `36352` | `6aa5c24c9859` | Falsification: `Exp T3` `PHerc0139` `2.399 um` MSII / StructN2V & `pag0`/`50`/`1000` |
| `seismic_rgt_stitched_patches.json` | `9459` | `efda74f6bbea` | Pillar 1: `Exp T11-RGT-Scale` `2x2` stitched multi-sheet RGT (`34` wraps, `0.215204 cm^2`) |
| `grid_quality_audit.json` | `47408` | `39dd7dbba7bf` | Pillar 1: `Exp T15-GridQuality-Audit` winner vs screened negatives vs HF spiral fits + `p * scale` provenance |
| `tiled_q34_cloud_masked_calibration.json` | `24386` | `f01ece47f7db` | Pillar 3: `Exp T13` (`Q31 + Q34`) `5/5` positive / `0/2` negative area-matched calibration |
| `gpu_shear_and_0211_control.json` | `22748` | `f60f9b7a66fa` | Pillar 3: `Exp T16-GPU` `PHerc0306B` `±15°/±30°` shear stress & `PHerc0211` `s48` (`7.535x`) |
| `gpu_0211_and_0306b_calibration_matrix.json` | `79238` | `9884e1ae4bed` | Pillar 3: `Exp T17-GPU` `PHerc0211` `snap{0,2}x{s48,s96}+A-hn` matrix & `PHerc0306B` `A-hn` layer card |
| `villa1739_cross_resolution_bridge.json` | `26727` | `c99d1f3b913f` | Falsification / Bridge: `Exp T12-Villa1739` `2.4->9.366 um` IoU bridge + `CLOSED NO-GO` 42D Ridge |

---

## Reproducible CPU-Only Verification Commands

Run the consolidated verifier and the full offline CPU-only unit test suite from the repository root in PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES="-1"; python gemini_lane/progress_prize_nov2026_verifier.py
$env:CUDA_VISIBLE_DEVICES="-1"; python -m unittest discover -s gemini_lane -p "test_*.py" -v
```
