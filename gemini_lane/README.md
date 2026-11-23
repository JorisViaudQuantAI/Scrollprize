# Gemini Local GPU & Coverage Lane (`gemini_lane/`)

MIT License for scripts in this folder (`../LICENSE`); upstream checkpoints, CT volumes, and mesh datasets retain their respective licences (CC BY-NC 4.0 / MIT as noted below).

This directory adds a resource-bounded **consumer 8 GB GPU lane** (local **NVIDIA GeForce RTX 4060 Laptop GPU**, 8,188 MiB VRAM with native BF16/FP16, `torch 2.10.0+cu128`, CUDA 12.8) plus a live-verified **19-scroll First Letters public asset & coverage registry** to coordinate cleanly with Claude Code, Codex, and ChatGPT.

---

## 1. Resource Guardrails (Local Workstation)

* **Zero persistent data bloat:** All checkpoints, crops, and reference maps are downloaded into `tempfile.TemporaryDirectory` outside the repository, SHA-256 verified, and unlinked immediately after each seed. Net disk delta for the 3-seed parity replay was ~2 MB (JSON receipt only).
* **Serialized, VRAM-capped GPU execution:** Only one process touches CUDA at a time, `torch.cuda.set_per_process_memory_fraction(0.65)` caps VRAM at ~5.2 GiB, `--batch 2` is the default, and `gc.collect()` + `torch.cuda.empty_cache()` run between seeds.

---

## 2. Measured Deliverable 1 — PHerc. 306B A-hn CUDA Parity Replay (Answers `codex-010`)

Reproduces Codex's pinned CPU smoke ([`public-controls/pherc0306b-replay.json`](https://github.com/JorisViaudQuantAI/Scrollprize/blob/59f56be9d6d1feed9fa749dce072b697f79c5770/public-controls/pherc0306b-replay.json)) on **RTX 4060 CUDA (`float16` autocast, `batch=2`, `vram_fraction_cap=0.65`, `tta=1`, 36 tiles, 1,849 interior pixels)** against the three published finalist checkpoints at `SirProdigle/pherc0306b-first-letters` commit `feabff6c54f95bd8e52a90e0a7a76881ee67d4cc`:

| Seed | Verdict | Max \|diff\| (CUDA fp16) | Max \|diff\| (Codex CPU fp32) | Mean \|diff\| (CUDA) | Pearson *r* (CUDA) | Inference time (CUDA vs Codex CPU) | Peak VRAM alloc / reserved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hn4_s0` | `PASS` | `0.00042` | `0.00149` | `0.000120` | `0.999999` | **4.338 s** vs 371.6 s (**85.7x**) | 1715.7 / 2446.0 MiB |
| `hn4_s1` | `PASS` | `0.00033` | `0.00174` | `0.000079` | `1.000000` | **3.677 s** vs 371.5 s (**101.0x**) | 1715.7 / 2446.0 MiB |
| `hn4_s2` | `PASS` | `0.00068` | `0.00141` | `0.000162` | `0.999999` | **3.770 s** vs 372.6 s (**98.8x**) | 1715.7 / 2446.0 MiB |

Receipt: [`evidence/pherc0306b_rtx4060_replay.json`](evidence/pherc0306b_rtx4060_replay.json).

### Scope and limits

* This verifies local CUDA `float16` arithmetic parity and VRAM safety on the author-selected 36-window PHerc. 306B smoke field in one pose (`tta=1`).
* **It does not measure generalisation or transfer to any unread First Letters scroll.**

### Reproduce

```bash
python gemini_lane/pherc0306b_cuda_replay.py \
  --device cuda --batch 2 --vram-fraction 0.65 \
  --output gemini_lane/evidence/pherc0306b_rtx4060_replay.json
```

Dependencies beyond Python stdlib + NumPy: `torch` (CUDA), `safetensors`, `huggingface-hub`, `zarr`, `tifffile`, `opencv-python-headless`.

---

## 3. Measured Deliverable 2 — 19-Scroll Coverage & Public Asset Inventory

[`coverage_inventory.py`](coverage_inventory.py) uses only the Python standard library to query the official public endpoints (`vesuvius-challenge-open-data` S3, `pscamillo/vesuvius-eligible-meshes`, `rodriguescarson/eligible-scroll-atlas-renders`, `rodriguescarson/eligible-scroll-spiral-fits`, `IyanDopico/vesuvius-sheet-tools`, and `AlexeyDrobkovStrikesBack/herculaneum-umbilici`) across all 19 eligible First Letters scrolls listed on `https://scrollprize.org/prizes` (checked 2026-11-22) and merges recorded team work through `collab/chatgpt-017.md`, `claude-010.md`, and `codex-010.md`.

Key live-verified facts in [`evidence/coverage_matrix.json`](evidence/coverage_matrix.json):

* **19 / 19** eligible scrolls have S3 `m7` surface predictions and `normal-grids` (`s3_has_m7_and_normal_grids = true`).
* **2 / 19** eligible scrolls have 2.403 µm rescans on S3: `PHerc0846A` (zero ink screening recorded) and `PHerc1203`.
* **Major untouched public mesh sets ready for m7-snapped GPU screening:**
  * `PHerc0191`: 1 HF spiral-fit run (`90` windings, `415.63 cm²` raw surface) — zero ink screening recorded.
  * `PHerc0826`: 3 HF spiral-fit bands (`270` windings total raw) — only 3 old atlas windings + 1 small 1.16 cm² window queued/screened.
  * `PHerc0125`: `340` meshes in `pscamillo/vesuvius-eligible-meshes` (`62` in HF atlas + `278` new; Codex claimed 2 -> **276 new windings unscreened**).
  * `PHerc1218`: `25` verified patches in `IyanDopico/vesuvius-sheet-tools` (staged on CPU by Codex, zero GPU ink runs recorded).
  * `PHerc0800` (`6` S3 `auto_grown` segments) and `PHerc1203` (`22` S3 `segments/raw/` entries, most unscreened outside Codex A/B/C and Claude N6–N10).
* **9 / 19** scrolls have zero public meshes in the above repos (`PHerc0175A`, `0175B`, `0483A`, `0483B`, `0490A`, `0490B`, `0846A`, `0846B`, `1545`).

**Caveat:** Raw mesh/winding counts are **not** unique on-sheet papyrus area; spiral fits overlap across bands and can drift off-sheet. Always apply m7 support and layer-coherence checks before counting coverage.

### Reproduce & Test

```bash
python gemini_lane/coverage_inventory.py --out gemini_lane/evidence/coverage_matrix.json
python -m unittest discover -s gemini_lane -p "test_*.py" -v
```

Quality-ranked Top-10 execution queue from public asset metadata: [`evidence/priority_queue_scout.md`](evidence/priority_queue_scout.md).

First Letters donor-overlap compliance matrix (6 detector families × 19 eligible scrolls + controls) & Nov 30 Progress Prize upstream-PR blueprint: [`evidence/compliance_and_progress_audit.md`](evidence/compliance_and_progress_audit.md).

PHerc. Paris 4 (Scroll 1) Title Prize energy/segment map & `< 180 MB` dual-energy different-ink streaming protocol: [`evidence/paris4_title_recon.md`](evidence/paris4_title_recon.md).

---

## 4. Zero-Disk Streaming Surface Pipeline (`stream_snap_render.py`)

[`stream_snap_render.py`](stream_snap_render.py) provides the zero-disk-bloat preprocessing and on-sheet filtering pipeline for our RTX 4060 screening queue:

* **`RemoteZarrV2`:** HTTP(S) and local Zarr v2 chunk streaming with strict HTTP 404-only empty-chunk handling (403/5xx raise immediately) and a hard in-memory LRU cache (`--cache-mb 512`, zero disk cache).
* **`snap_tifxyz_to_m7`:** Three-channel vertex sentinel validation (`finite & >= 0`), neighbor-safe unit normals, and two-pass normal-directed snapping onto S3 `m7` surface predictions with before/after `on_sheet` and displacement metrics.
* **`render_surface_stack` & `layer_coherence_score`:** Pure NumPy/SciPy surface-volume renderer separating mesh validity from CT acquisition support, plus patch-wise depth-shift correlation to reject oblique cut-throughs before GPU inference.

Offline synthetic tests (`6/6` tests in [`test_stream_snap_render.py`](test_stream_snap_render.py), `12/12` total in `gemini_lane/`):

```bash
python -m unittest discover -s gemini_lane -p "test_*.py" -v
```

---

## 5. Measured Deliverable 3 — Studio (`PHerc0211` Detector) RTX 4060 `fp16` vs `bf16` Parity & VRAM Profile

[`studio_rtx4060_bench.py`](studio_rtx4060_bench.py) runs Claude's packaged [`../firstletters-colab/studio_t4/studio_t4.py`](../firstletters-colab/studio_t4/studio_t4.py) on the 169 × 169 PHerc. 306B smoke crop (`mesh_index=12`, `25` windows at stride 48 Hann) with a 65% VRAM cap (`~5.2 GiB`):

| Run | Precision | Order | Batch | Time (s) | ms / window | Peak alloc / reserved (MiB) | Max \|diff\| vs CPU `fp32` | Pearson *r* vs CPU `fp32` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cpu_fp32_desc_s48_b1` | `fp32` | `desc` | 1 | 48.88 | 1955.0 | — | — | — |
| `cpu_fp32_asc_s48_b1` | `fp32` | `asc` | 1 | 49.25 | 1969.9 | — | — | — |
| `cuda_fp16_desc_s48_b2` | `fp16` | `desc` | 2 | **3.29** | 131.5 | **2,164 / 2,354** | `0.002526` | **`0.999996`** |
| `cuda_fp16_desc_s48_b4` | `fp16` | `desc` | 4 | **3.35** | 133.9 | 3,010 / 3,364 | `0.002538` | **`0.999996`** |
| `cuda_fp16_asc_s48_b4` | `fp16` | `asc` | 4 | **2.30** | 92.0 | 2,514 / 2,842 | **`0.002113`** | **`0.999993`** |
| `cuda_bf16_desc_s48_b2` | `bf16` | `desc` | 2 | 3.10 | 124.1 | 2,164 / 2,354 | `0.011070` | `0.999928` |
| `cuda_bf16_desc_s48_b4` | `bf16` | `desc` | 4 | 3.41 | 136.5 | 3,010 / 3,364 | `0.011210` | `0.999928` |
| `cuda_bf16_asc_s48_b4` | `bf16` | `asc` | 4 | 2.42 | 97.0 | 2,514 / 2,938 | **`0.061742`** | **`0.996717`** |

**Key technical finding:** Although `studio_t4.py --precision auto` selects `bf16` on Ada/Ampere GPUs (`cc >= 8`), `bf16` (7-bit mantissa) exhibits up to `0.061742` probability drift (`r = 0.996717`) on `asc`, whereas explicit `--precision fp16` (10-bit mantissa) achieves **4.4x–29.2x lower max error** (`0.002113–0.002538`, `r >= 0.999993`) at identical peak VRAM (`2.11 GiB` at `batch=2`, `2.94 GiB` at `batch=4`). Receipt: [`evidence/studio_rtx4060_parity.json`](evidence/studio_rtx4060_parity.json).

### Reproduce

```bash
python gemini_lane/studio_rtx4060_bench.py \
  --vram-fraction 0.65 \
  --output gemini_lane/evidence/studio_rtx4060_parity.json
```

---

## 6. Colab T4 notebook & `PHerc1203` / `PHerc0125` / `PHerc0191` CPU on-sheet triage

[`colab_t4_firstletters_screener.ipynb`](colab_t4_firstletters_screener.ipynb) is a zero-setup Colab T4 wrapper around `screen_scroll_rtx4060.py` (Studio fp16 stride 48 + A-hn fp16, both depth orders). It clones `gemini/coverage-gpu-lane`, installs only `numcodecs safetensors tifffile opencv-python-headless requests`, and has pre-filled cells for: (i) `PHerc0306B` positive gate (`--uv-crop 80:140,120:345`, must be reviewed before (ii)-(iv) unlock), (ii) `PHerc0125` Tier-1 (`z10544_w035`, `z14144_w035`, `z8736_w035`, `z6944_w030`, `z15936_w030`; skips `z14144_w045`), (iii) the 6 clean `PHerc1203` S3 sheets (`46.66 cm²`, `--snap-passes 2`), (iv) 4 `PHerc0191` spiral-fit windings (`w020`, `w030`, `w040`, `w050` from `rodriguescarson/eligible-scroll-spiral-fits` `slice-11600-12400`, pinned HF revision `63c641ba`, author-flagged *unvalidated geometry*), (v) `verify_upstream_gpu_burst.py` (skips cleanly if the script is not on the branch). Outputs go to `/content/scrollprize_private` (outside git); only `montage.png`, `hotspots.png`, `summary.json` are zipped / copied to Drive.

[`evidence/pherc1203_onsheet_triage.json`](evidence/pherc1203_onsheet_triage.json) is the sanitized CPU-only result (`m7` snap + `onsheet_score` + `valid_ct_cm2`, no ink detector, no images/paths), built by `build_pherc1203_onsheet_triage.py` from the private per-sheet summaries.

[`evidence/pherc0125_0191_onsheet_triage.json`](evidence/pherc0125_0191_onsheet_triage.json) is the sanitized CPU-only triage (same allowlist and method, 5 slices, 2x `m7` snap) of the 5 `PHerc0125` Tier-1 windings (pscamillo commit `66000516`) and 4 `PHerc0191` windings, built by `build_pherc0125_0191_onsheet_triage.py` (schema/privacy tests: `test_pherc0125_0191_triage.py`). Ranked by `f_parallel` (post-snap `on_sheet`, `valid_ct_cm2`): `PHerc0125` `z14144_w035` (0.985, 0.991, 4.01 cm²), `z10544_w035` (0.984, 0.996, 3.98 cm²), `z8736_w035` (0.954, 0.983, 3.91 cm²), `z15936_w030` (0.949, 0.955, 3.57 cm²), `z6944_w030` (0.885, 0.958, 3.20 cm²); `PHerc0191` `w030` (0.815, 0.945, 2.12 cm²), `w020` (0.813, 0.900, 1.44 cm²), `w050` (0.811, 0.923, 3.45 cm²), `w040` (0.754, 0.915, 2.80 cm²). `PHerc0191` windings show markedly higher `f_drift` (0.16-0.21 vs <= 0.05 for all five `PHerc0125`), consistent with the author's unvalidated winding density; the `onsheet_score` is a coherence prior, not an ink or coverage claim.

```bash
CUDA_VISIBLE_DEVICES="" python gemini_lane/screen_scroll_rtx4060.py --scroll PHerc1203 --mesh <sheet URL> \
  --ct-zarr <CT zarr> --m7-zarr <m7 zarr> --detectors none --num-slices 5 --snap-passes 2 --cache-mb 256 \
  --voxel-um 9.362 --out-private-dir <private dir outside repo>
python gemini_lane/build_pherc1203_onsheet_triage.py --private-dir <private dir> \
  --output gemini_lane/evidence/pherc1203_onsheet_triage.json
python -m unittest discover -s gemini_lane -p "test_*.py" -v
```

---

## 7. Zero-mesh m7 patch grower (`PHerc0846A`, `PHerc1545`)

[`grow_m7_patch.py`](grow_m7_patch.py) grows one single-sheet `tifxyz` patch from public S3 `m7` predictions, CPU-only, on top of `stream_snap_render.py` (source-support version):

1. Stream one interior `m7` box (`BudgetedRemoteZarrV2`, hard `ByteBudget` cap on every byte, metadata included).
2. Track one sheet column-by-column (`track_sheet`): the unambiguous nearest `m7` run within `max_step`, runs at most `max_run` long, no hops between wraps. Seams are cut (`cut_seams`). The grid drops nodes whose edges cross a slit, keeps one connected component, writes `-1.0` on all three channels for invalid vertices, and uses `scale=[0.05, 0.05]`.
3. Apply a 2x `snap_tifxyz_to_m7`, render CT through `ScaledReader` (exact positive-weight-corner chunk fetches, so no neighbour 2 MiB CT chunks), and score `onsheet_score`.
4. Gates: `f_parallel >= 0.70` (needs at least one scored tile) and `valid_ct_cm2 >= 1.0`.

The tifxyz goes only to a private directory outside every Git worktree. The public receipt keeps whitelisted scalars only: no coordinates, bbox, images, ink or local paths.

[`evidence/virgin_scroll_patch_preflight.json`](evidence/virgin_scroll_patch_preflight.json) records the first live run on `PHerc0846A`:
- Frame: `m7-L2` level 0 (9.612 µm). Extraction at `m7-L2` level 1, snap at level 0, CT from the 2.403 µm scan's pyramid level 3. Streamed 59.6 MB.
- Result: **gate not passed**. Mesh area 0.28 cm², valid CT 0.26 cm². The snap converged (on-sheet 0.999). No 256 px tile reached 90 % valid, so `f_parallel` was not measured.
- The receipt also holds the public `m7` / CT / `normal-grids` / `lasagna` metadata summary for both scrolls.

```bash
CUDA_VISIBLE_DEVICES="" python gemini_lane/grow_m7_patch.py --scroll PHerc0846A --budget-mb 100   --extract-level 1 --snap-level 0 --box-target Z0 Y0 X0 Z1 Y1 X1 --height-axis y --num-slices 15   --private-out <private dir outside repo> --evidence-out gemini_lane/evidence/virgin_scroll_patch_preflight.json
```

Needs `numcodecs` for the blosc-zstd `m7` arrays. The offline tests (`test_grow_m7_patch.py`) use uncompressed synthetic Zarr stores and do not need it.

---

## 8. Attribution & Licences

* Scripts & tests in `gemini_lane/`: MIT License (`../LICENSE`).
* Studio soft-loss weights & `model.py`: Priyank Aranke and contributors (`vesuvius-studio-multiscroll-cpt`, code MIT, weights CC BY-NC 4.0).
* PHerc. 306B A-hn weights, smoke crop, and upstream `ink_v8.py` / `smoke_v7.py` / `v8in`: Liam Warner and Nicola Bodill (`SirProdigle/pherc0306b-first-letters`, code MIT, model/data CC BY-NC 4.0; underlying `v8-in` base weights by `YoussefMoNader`).
* Scroll CT data, S3 `m7` surface predictions, `pscamillo/vesuvius-eligible-meshes`, `rodriguescarson/eligible-scroll-*`, and `IyanDopico/vesuvius-sheet-tools`: CC BY-NC 4.0 / respective author licences.
