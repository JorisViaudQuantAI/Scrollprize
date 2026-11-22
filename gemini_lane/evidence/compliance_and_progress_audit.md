# First Letters Donor-Overlap Compliance Matrix & Nov 30 Progress Prize Audit

Prepared from the Cursor / Claude Opus 5.5 compliance audit (`2026-11-22`), verified against official model cards, training manifests, and [`scrollprize.org/prizes`](https://scrollprize.org/prizes).

---

## 1. First Letters Training-Overlap & Licence Compliance Matrix

Legend:
- **TRAIN**: Present in published training data.
- **VAL**: Used for validation or checkpoint selection.
- **INDIRECT**: Indirectly exposed via base weights or teacher models.
- **BUCKET**: Present in published training bucket/corpus; train/val split not disclosed.
- **CLEAN**: Absent from published training sources.

| Model Family (Pinned Rev.) | Licence (Code / Weights) | 19 Eligible Scrolls | PHerc. 0343 | PHerc. 0306B | PHerc. 0841 | PHerc. 0814 | PHerc. 0009B | PHerc. 1447 W058 | PHerc. Paris 4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Studio** ([`bf66f753`](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/README.md)) | MIT / CC BY-NC 4.0 | **CLEAN** (`0211` excluded, non-eligible) | **TRAIN** (`0343P`, 5 segs) | **CLEAN** | **TRAIN** (incl. `841-w00`) | **TRAIN** | **TRAIN** (incl. `132115`) | **CLEAN** | **TRAIN** (29 rows) |
| **A-hn** ([`feabff6c`](https://github.com/SirProdigle/pherc0306b-first-letters/tree/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc) / [`e57c4d54`](https://huggingface.co/Prodigle/pherc0306b-ink-ahn/blob/e57c4d541c239e06450a98c1cf2015d3de8d887e/README.md)) | MIT / CC BY-NC 4.0 | **CLEAN** | **INDIRECT** (Reader teachers) | **TRAIN** (5 surfaces; ±2.9 mm claim zone excluded) | **INDIRECT** (`TRACE v3B` ckpt selection) | **INDIRECT** | **INDIRECT / VAL** | **TRAIN** (`w058`, `w060`) | **INDIRECT** |
| **v8-in** ([`d89166b4`](https://huggingface.co/YoussefMoNader/ink-8um-v8in/blob/d89166b41a3f5fad7749b3d7c0fdd1bd3695d844/README.md)) | MIT / MIT (data CC BY-NC) | **CLEAN** | **CLEAN** | **CLEAN** | **CLEAN** | **TRAIN** | **VAL** (`ag132115`) | **CLEAN** | **TRAIN** (2 segs) |
| **reader-v3** ([`aee1d64b`](https://huggingface.co/domenicor046/reader-v3/blob/aee1d64bced4eb35546348f3d276b9e2a0b99c46/README.md)) | MIT / MIT | **CLEAN** | **TRAIN** (6 segs, 2 held out) | **CLEAN** | **CLEAN** (held out) | **TRAIN** | **TRAIN** (14 segs, 4 held out) | **CLEAN** | **INDIRECT** (via `ink_9um`) |
| **ink_9um** ([`7109667e`](https://huggingface.co/scrollprize/ink_9um/blob/7109667e2607db1b90c37c8b09cb876ea7fe7bb1/README.md)) | MIT / MIT | **CLEAN** | **CLEAN** | **CLEAN** | **CLEAN** | **TRAIN** (`46527`) | **CLEAN** | **CLEAN** | **TRAIN** (8 segs) |
| **Hecate** ([`9cb86e50`](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/README.md)) | MIT / MIT | **CLEAN** in published dirs *(confirm 1203/0846A 9.6 distillation)* | **CLEAN** (`ink/unused/P343p`) | **CLEAN** | **BUCKET** (`ink/841/w00`) | **TRAIN** | **TRAIN** | **CLEAN** | **TRAIN** (+ 3D Paris 4 teacher) |

### Operational Compliance Rules for RTX 4060 Lanes (G1–G4)

1. **Eligible Scrolls (19):** All 6 families are cleared for screening the 19 eligible scrolls (`G1`–`G4`), subject to:
   - **Hecate 9.6 µm:** Resample inputs to `9.6 µm` (eligible scans are `9.362 µm` or `8.640 µm`) and request HF clarification on whether any `1203` / `0846A` `2.4 µm` pairs entered 9.6 µm distillation.
   - **Fine-tuning / pseudo-labels:** Exclude a >= 3D buffer (e.g. `A-hn` used ±2.9 mm) around any evaluated region, and publish weights + datasets under CC BY-NC 4.0 (compatible with First Letters open-source requirements for derivatives of `Studio` or `A-hn`).
   - **TRACE v3B:** Audit its model card separately before using `A-hn × TRACE v3B` products in any claim.
2. **Clean Positive Controls (Critical Correction!):**
   - **`PHerc. 0841 w00`** is **contaminated** for `Studio` (training donor) and `Hecate` (`ink/841/w00` in bucket); it remains clean ONLY for `reader-v3`, `ink_9um`, `v8-in`, and `A-hn`. Studio's holdout must use `841-expanded-layer-03`, `eval211.py` (`PHerc0211`), or **`PHerc. 0306B`**.
   - **`PHerc. 0306B`** is **CLEAN** for `Studio`, `reader-v3`, `v8-in`, `ink_9um`, and `Hecate` (contaminated for `A-hn`).
   - **`PHerc. Paris 4` (Scroll 1)** is **contaminated across all 6 carbon-ink CNN families** (`Studio`, `A-hn`, `v8-in`, `reader-v3`, `ink_9um`, `Hecate`).

---

## 2. Nov 30 Progress Prize Strategy (Non-Duplicate)

Per [`scrollprize.org/prizes`](https://scrollprize.org/prizes):
- Small same-month contributions must be bundled into one submission; Joris already submitted the root `flat-roi.patch` Villa work for November (`collab/codex-001.md`), so `codex/roi-full-cli` is excluded here.
- **Jury expressly discourages standalone analytic, auditing, or evaluation tools.** Therefore do **NOT** submit `coverage_inventory.py`, `score_text.py`, `line_controls`, `tile_phase`, `component_export`, or raw parity JSONs alone, nor `render_tifxyz.py` (duplicates `vc_render_tifxyz`).

### Recommended Single Bundle (For Owner Decision — No Upstream PR/Submission Without Approval)

**"FP16 Inference for Pre-Ampere & 8 GB GPUs across Upstream Studio and Hecate"** (~$250–$1,000 realistic tier, aligned with [Open Problems](https://scrollprize.org/2026_open_problems) *Data scale: cheaper compute/storage paths*):
1. **Upstream `--precision fp16` PR to [`scrollprize/hecate`](https://huggingface.co/scrollprize/hecate/discussions)** (`firstletters-colab/hecate_fp16`, 3-block patch, 3.0× faster end-to-end on T4, max diff 1/255, `r = 0.999992`; close gap by running 2D+3D parity on both 2.4 µm and 9.6 µm checkpoints in both depth orders, coordinating with open PR #3 `--resample`).
2. **Upstream `--precision fp16` PR to [`priyankaranke/vesuvius-studio-multiscroll-cpt`](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt)** (integrating `studio_t4` `fp16` into `hf/inference.py` + `infer.sh` rather than keeping a standalone wrapper). Strengthened directly by our RTX 4060 measurement (`studio_rtx4060_parity.json`) showing `--precision fp16` delivers **4.4×–29.2× tighter parity** (`max |diff| 0.0021–0.0025`, `r >= 0.999993`) than `auto`/`bf16` even on Ada `cc 8.9` GPUs while slashing T4 batch-1 VRAM from `11.35 GB` to `1.17 GB` (and `2.11 GiB` at `batch=2` on RTX 4060).
   - **Required doc fix before submission:** Drop `PHerc. 0841 w00` AUC `0.916` as a quality claim for `Studio` (since `w00` is in `Studio`'s training set); cite `fp16`-vs-`fp32` parity + `841-expanded-layer-03` / `eval211.py` / `PHerc. 0306B` clean controls instead.
3. **Hold for December (unless ported to Python `volume-registration` in Villa):** Node.js cross-scan annular registration predictor (`1203`/`0846A`).
