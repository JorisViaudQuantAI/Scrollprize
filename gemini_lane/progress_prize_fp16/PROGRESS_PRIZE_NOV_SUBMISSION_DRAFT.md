# DRAFT (not submitted): November 2026 Progress Prize

Rules: [scrollprize.org/prizes](https://scrollprize.org/prizes). Theme: [Open Problems](https://scrollprize.org/2026_open_problems), *data scale: cheaper compute*.

## Non-duplication

The November submission already sent covers the Villa `flat-roi.patch` (ROI flattening). This entry shares no code,
repository or result with it: it changes the inference precision of two released ink models, in their own
repositories. The CLI ROI work (`codex/roi-full-cli`) and standalone audit or evaluation tools are not included.
If the rules require one bundle per month, this text is the addendum to that bundle.

## Title

fp16 inference for the Studio and Hecate ink models on pre-Ampere and 8 GB GPUs

## What

Two small upstream pull requests (links added once posted, after owner approval):

1. **Studio** ([vesuvius-studio-multiscroll-cpt](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt)):
   `--precision {auto,bf16,fp16,fp32}` in `hf/inference.py` and `infer_release.py`. `auto` picks fp16 on GPUs with
   compute capability < 8.0, which today stop with a BF16 error; `--amp` is kept as an alias. Link: TBD.
2. **Hecate** ([scrollprize/hecate](https://huggingface.co/scrollprize/hecate)): `--precision fp16` in `hecate.py`,
   default unchanged, compatible with the open `--resample` pull request (#3). Link: TBD.

## Why it matters

Free Colab and most donated or older GPUs are T4 / V100 class, and many laptops have 8 GB. Both models were usable
there only in slow, memory-hungry fp32.

## Evidence (numerical parity, memory and time; not accuracy claims)

**Studio, fp16 vs fp32:**

- T4: peak GPU memory at batch 1 **11.35 GB (fp32) → 1.17 GB (fp16)**; 340 → 132 ms per window; on a
  512 × 512 px crop max |Δp| 0.0019, mean |Δp| 0.00014.
- RTX 4060 Laptop (8 GB, cc 8.9), clean PHerc. 0306B control crop (absent from Studio's and Hecate's training
  lists), each GPU run against CPU fp32: fp16 at **batch 2 peaks at 2.11 GiB** (2164 MiB allocated); max |Δp|
  0.0021–0.0025, Pearson r ≥ 0.999993. bf16 on the same crop: max |Δp| 0.0111–0.0617, i.e. fp16's max |Δp| is
  **4.4× to 29.2× smaller**, at the same peak memory and a similar speed.
- Colab L4 (cc 8.9), 41 × 1024 × 1024 PHerc. 0306B crop, against GPU fp32: fp16 max |Δp| 0.0031, bf16 0.035, at
  the same speed and memory (measured with our packaged CLI, same code path). On both Ada GPUs bf16 is about ten
  times further from fp32 than fp16; our CLI now uses fp16 for `auto` on every CUDA GPU, and the Studio PR offers
  the same change upstream.

**Hecate (9.6 µm checkpoint, T4), fp16 vs the fp32 default:** 3.0× faster end to end (41.2 s vs 123.6 s), output
within one uint8 level of fp32 (mean 0.047, r 0.99999). bf16 is slower than fp32 on a T4 (228.6 s).

Known-text AUCs are deliberately not used: the PHerc. 0841 controls are training data for Studio and sit in Hecate's
training bucket, so they cannot show accuracy for these models.

## Reproduction

`gemini_lane/progress_prize_fp16/` in our repository: the three patches with their pinned bases, CPU-only tests
(`python -m unittest discover -s gemini_lane -p "test_*.py" -v`; with `STUDIO_UPSTREAM_DIR` and
`HECATE_UPSTREAM_DIR` set to upstream clones, the patches are also applied and their blob hashes checked),
`prepare_hecate_parity_crops.py` + `verify_upstream_gpu_burst.py` for the remaining GPU checks below, and these
drafts. Measurement receipts: `gemini_lane/evidence/studio_rtx4060_parity.json`, `firstletters-colab/studio_t4/`,
`firstletters-colab/hecate_fp16/`.

## Licence

Code changes MIT, as both upstream repositories. Model weights untouched (Studio CC BY-NC 4.0, Hecate MIT).

## AI assistance

Code, runs and text produced with AI coding agents, directed and reviewed by the repository owner.

---

## Remaining checks before submission

All GPU checks below run in one unattended 2-step Colab T4 pass (`prepare_hecate_parity_crops.py` streams two public
uint8 `[64, 256, 256]` CT parity crops — `PHerc0306B` 8.640 µm control + `PHerc0846A` 2.403 µm public CT, `< 15 MB`
total HTTP transfer — into a temporary directory strictly outside the git repository; `verify_upstream_gpu_burst.py`
clones and patches the pinned upstream bases in a temporary directory, deletes it afterwards, and writes a JSON receipt
without local paths; defaults `--vram-fraction 0.35 --batch 1 --max-gpu-temp-c 68`, safe on an 8 GB laptop and on a
Colab T4; delete `/tmp/hecate_parity_crops` once the receipt is saved):

```bash
python gemini_lane/progress_prize_fp16/prepare_hecate_parity_crops.py --out-dir /tmp/hecate_parity_crops
python gemini_lane/progress_prize_fp16/verify_upstream_gpu_burst.py \
  --stages studio,hecate --hecate-patch main \
  --hecate-input-9p6 /tmp/hecate_parity_crops/hecate_9p6um_crop.npy \
  --hecate-input-2p4 /tmp/hecate_parity_crops/hecate_2p4um_crop.npy \
  --output /tmp/hecate_parity_crops/burst_receipt.json
```

- [ ] Hecate: patched `hecate.py`, fp16 vs fp32, on **both 2.4 µm and 9.6 µm** checkpoints, 2D and `--output-3d`,
      both depth orders (`--reverse`), on a T4; preferably on the PHerc. 0306B crop (`--stages hecate`).
- [ ] Studio: patched upstream CLI end to end on a T4 (`auto` → fp16) and the RTX 4060 (`--precision fp16`) on the
      169 × 169 PHerc. 0306B smoke crop (`--stages studio`); the GPU numbers above were measured through a wrapper
      or our packaged CLI with the same code path.
- [x] Patches re-checked with `git apply --check` and blob hashes against fresh upstream clones (Studio `bf66f753`,
      Hecate `9cb86e50` and `refs/pr/3` @ `f4386b19`).
- [ ] Studio upstream tests green on Linux (4 tests fail on Windows before and after the patch).
- [ ] Both pull requests posted with owner approval; links filled in above.
- [ ] Confirm with the rules page that this entry is bundled correctly with the November Villa submission (the
      rules ask for small related contributions to be bundled and discourage standalone audit tools; the burst
      script and tests are reproduction material, not a separate claim). Deadline: November 30, 2026, 11:59 pm
      Pacific.
