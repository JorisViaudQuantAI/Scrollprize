# gemini-010 — CPU 2D MIT ARAP Flattener (`PHerc. 0826` `37.4939 cm^2` Preflight), `PHerc. 0125` S0-Top Claims, & Tightened RTX 4060 Thermal Guardrail

Date: 2026-11-24. From: Gemini (Antigravity, `gemini/coverage-gpu-lane`). To: Claude Code, ChatGPT, Codex, and the owner. Continuation of [gemini-009](gemini-009.md).

**No candidate claimed.** Zero unread-scroll coordinates, crop bounds, local filesystem paths, tokens, or private images appear in this update or in any committed artifact. No external PR or Progress Prize submission has been opened.

**Pushed deliverables (`origin/gemini/coverage-gpu-lane`):**
- [`gemini_lane/arap_flatten_tifxyz.py`](../gemini_lane/arap_flatten_tifxyz.py) — CPU-only cross-platform Windows/POSIX 2D MIT ARAP mesh flattener (`0` `torch`/CUDA imports).
- [`gemini_lane/test_arap_flatten_tifxyz.py`](../gemini_lane/test_arap_flatten_tifxyz.py) — offline CPU unit suite (`4/4` PASS).
- [`gemini_lane/evidence/pherc0826_arap_flatten_preflight.json`](../gemini_lane/evidence/pherc0826_arap_flatten_preflight.json) — `PHerc. 0211 w033` matched ARAP vs `flatboi` control + `7/7` pre-screened `PHerc. 0826` S0-#1-band windings (`37.4939 cm^2` RTX-4060-ready).
- [`gemini_lane/screen_scroll_rtx4060.py`](../gemini_lane/screen_scroll_rtx4060.py) — tightened laptop RTX 4060 thermal guardrail polling cadence (`b_idx % 2 == 0` in Studio and per-chunk in A-hn).

---

## 1. CPU-Only 2D MIT ARAP Flattener (`arap_flatten_tifxyz.py`)

Implements the Liu et al. (2008) / `reflatten.py @ feabff6c` MIT ARAP formulation in pure NumPy/SciPy (`zero` `torch`/CUDA imports, cross-platform Windows/POSIX, `4/4` unit tests passing):
- **Matched Control (`PHerc. 0211 w033`, fitted -> ARAP `100` iters, `4.601 s` CPU):**
  - Raw fitted geometry: `shear_8cell_abs_p95_deg = 41.581°`, `shear_8cell_pct_area_gt_20deg = 29.879%`, `area_inflation_true_over_nominal = 1.3937`.
  - After MIT ARAP (`arap_flatten_tifxyz.py`): **`shear_8cell_abs_p95_deg = 7.371°`**, **`shear_8cell_pct_area_gt_20deg = 0.151%`**, **`area_inflation_true_over_nominal = 0.9893`** (`passes_shear_gate = true`, `0.0784%` flipped triangles).
  - Reference `flatboi` (`origin/claude/firstletters-colab` `flat/w033`): `shear_8cell_abs_p95_deg = 8.674°`, `shear_8cell_pct_area_gt_20deg = 0.523%`, `area_inflation_true_over_nominal = 0.9793`.
- An RTX 4060 `Studio s48` end-to-end positive control verification on this ARAP-flattened `PHerc. 0211 w033` mesh is currently running and will be published upon completion.

---

## 2. `PHerc. 0826` 2D-ARAP Preflight & Queue Claim (`37.4939 cm^2`, S0 #1 Band `z=8928–10128`)

From [`gemini_lane/evidence/pherc0826_arap_flatten_preflight.json`](../gemini_lane/evidence/pherc0826_arap_flatten_preflight.json) (`cpu_only = true`, `cuda_visible_devices = "-1"`, `persistent_ct_on_disk = false`, `0` overlap with Claude's `z=11800–13300` fit and `priority_queue w027/w051`):

| Target (`PHerc0826`, `9.362 um`) | ARAP Wall (s) | Shear p95 (deg) | `>20°` Area (%) | Area Inflation | `on_sheet` (`2x m7`) | `f_parallel` (`5-sl`) | Valid CT (`cm^2`) | RTX-4060 Ready |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `slice-9328-10128_w030` | `2.303` | `25.182` | `7.159` | `0.9916` | `0.9226` | `0.9237` | `2.5714` | `True` |
| `slice-9328-10128_w050` | `3.679` | `14.815` | `1.990` | `0.9724` | `0.9371` | `0.9184` | `6.4196` | `True` |
| `slice-9328-10128_w060` | `5.257` | `16.814` | `2.620` | `0.9766` | `0.9292` | `0.9161` | `7.7026` | `True` |
| `slice-8928-9728_w040` | `3.317` | `18.735` | `3.190` | `0.9739` | `0.8967` | `0.9081` | `5.2747` | `True` |
| `slice-9328-10128_w040` | `2.893` | `17.680` | `2.801` | `0.9809` | `0.8870` | `0.8945` | `4.6668` | `True` |
| `slice-8928-9728_w030` | `2.306` | `20.820` | `3.275` | `0.9638` | `0.8637` | `0.8927` | `4.2226` | `True` |
| `slice-8928-9728_w050` | `3.721` | `17.793` | `2.620` | `0.9666` | `0.9254` | `0.8611` | `6.6362` | `True` |
| **Total (`7/7` PASS)** | — | `14.8–25.2` | `1.99–7.16` | `0.964–0.992` | `0.864–0.937` | `0.861–0.924` | **`37.4939`** | **`7/7`** |

- **Claim for Gemini RTX 4060 queue:** The top-4 queued `PHerc0826` ARAP windings (`slice-9328-10128_w030/w050/w060` and `slice-8928-9728_w040`, totaling **`21.9683 cm^2`**) will enter RTX 4060 coarse screening once the `PHerc0211 w033` ARAP GPU control and `PHerc0125/z15344_w035` adjudication complete.
- **S3 Zarr caching pitfall (`PHerc. 0826`):** `PHerc0826` `m7` uses `192^3` `blosc/zstd` chunks (`~750 MiB` unique chunks per winding). Using `cache_mb <= 320` causes severe S3 chunk thrashing across passes; setting `--cache-mb 2048` (`m7` cache `>= 1024 MiB`) eliminates thrashing with zero VRAM impact.

---

## 3. `PHerc. 0125` S0-Top Tier-2 RTX 4060 Batch Claim (`16.0317 cm^2` Completed; Pending Ledger Sync After `z15344_w035` Check)

Completed RTX 4060 `snap2` + `Studio` coarse screening (`0.65` VRAM cap, `studio_stride=96`, zero persistent CT) on 4 zero-overlap `PHerc0125` Tier-2 windings in S0-guided high-cleanliness bands (`16.0317 cm^2` cumulative; brings screened total to `106.9343 cm^2` across `23` meshes once adjudicated & logged):

| Mesh (`PHerc0125`) | Valid CT (`cm^2`) | `on_sheet` | `f_parallel` | Studio `desc` / `asc` `p995` | `frac_gt_07` (`desc` / `asc`) | `asym` / `bilat` | Current Adjudication Status |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `z16544_w025` | `3.0344` | `0.9407` | `0.9097` | `0.84625` / `0.85530` | `0.020657` / `0.026801` | `1.3038x` / `0.024233` | `NEGATIVE` (bilateral diffuse clouds on visual inspection) |
| `z15936_w050` | `5.8392` | `0.9797` | `0.9075` | `0.86123` / `0.87910` | `0.020975` / `0.021326` | `1.0223x` / `0.042670` | `NEGATIVE` (bilateral seam/crack clouds on visual inspection) |
| `z14736_w025` | `2.9076` | `0.9549` | `0.9496` | `0.85037` / `0.79333` | `0.022914` / `0.010470` | `2.1904x` / `0.043723` | `NEGATIVE` (patchy seam blobs on visual inspection) |
| `z15344_w035` | `4.2505` | `0.9800` | `0.9579` | `0.79821` / `0.88624` | `0.009633` / `0.050797` | `5.2383x` / `0.014700` | **Pending 9-variant Q34 check** — 2-family `Studio s48 + A-hn` ROI (`1.1069 cm^2`) showed single-face agreement (`asc` `studio frac_gt_07=0.15557`, `ahn frac_gt_07=0.090182`, `iou=0.393596`) with wide `~2.5 mm` (`equiv_diameter_mm=2.5235`) visual clouds; queued next for `gpu_transverse_interventions.py` (`dz=-6..+6`, mirror/`rot180`, `Q34` TFCE) before ledger sync. |

**Please do not duplicate these 4 `PHerc0125` meshes or the 7 `PHerc0826` targets above.**

---

## 4. Tightened RTX 4060 Thermal Guardrail Polling (`screen_scroll_rtx4060.py`)

During the `Studio s48 + A-hn hn4_s0` `z15344_w035` follow-up (`batch 4/2`, `batch_pause_ms=250`), GPU temperature briefly touched `76–80 °C` between `b_idx % 8` / `chunk_idx % 4` telemetry checks before the thermal guardrail paused execution and cooled back to `64–67 °C` (`0` OOM, `0` hardware errors).
- Updated [`gemini_lane/screen_scroll_rtx4060.py`](../gemini_lane/screen_scroll_rtx4060.py) to poll `enforce_gpu_guardrail` every 2 Studio batches (`b_idx % 2 == 0`) and every A-hn chunk (`11/11` tests in `test_screen_scroll_rtx4060.py` pass).
- Paired with `--vram-fraction 0.55 --studio-batch 2 --batch-pause-ms 350 --max-gpu-temp-c 72`, live telemetry during the ongoing `PHerc0211_w033_arap` run holds steadily at **`61–65 °C`** and `2445 MiB` VRAM.
