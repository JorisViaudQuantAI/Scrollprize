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

---

## 4. Measured Deliverable 3 — Studio (`PHerc0211` Detector) RTX 4060 `fp16` vs `bf16` Parity & VRAM Profile

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

## 5. Attribution & Licences

* Scripts & tests in `gemini_lane/`: MIT License (`../LICENSE`).
* Studio soft-loss weights & `model.py`: Priyank Aranke and contributors (`vesuvius-studio-multiscroll-cpt`, code MIT, weights CC BY-NC 4.0).
* PHerc. 306B A-hn weights, smoke crop, and upstream `ink_v8.py` / `smoke_v7.py` / `v8in`: Liam Warner and Nicola Bodill (`SirProdigle/pherc0306b-first-letters`, code MIT, model/data CC BY-NC 4.0; underlying `v8-in` base weights by `YoussefMoNader`).
* Scroll CT data, S3 `m7` surface predictions, `pscamillo/vesuvius-eligible-meshes`, `rodriguescarson/eligible-scroll-*`, and `IyanDopico/vesuvius-sheet-tools`: CC BY-NC 4.0 / respective author licences.
