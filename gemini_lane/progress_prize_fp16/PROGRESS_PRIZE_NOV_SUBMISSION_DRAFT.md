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

**Hecate (9.6 µm checkpoint, T4), fp16 vs the fp32 default:** 3.0× faster end to end (41.2 s vs 123.6 s), output
within one uint8 level of fp32 (mean 0.047, r 0.99999). bf16 is slower than fp32 on a T4 (228.6 s).

Known-text AUCs are deliberately not used: the PHerc. 0841 controls are training data for Studio and sit in Hecate's
training bucket, so they cannot show accuracy for these models.

## Reproduction

`gemini_lane/progress_prize_fp16/` in our repository: the three patches with their pinned bases, CPU-only tests
(`python -m unittest discover -s gemini_lane -p "test_*.py" -v`), and these drafts. Measurement receipts:
`gemini_lane/evidence/studio_rtx4060_parity.json`, `firstletters-colab/studio_t4/`, `firstletters-colab/hecate_fp16/`.

## Licence

Code changes MIT, as both upstream repositories. Model weights untouched (Studio CC BY-NC 4.0, Hecate MIT).

## AI assistance

Code, runs and text produced with AI coding agents, directed and reviewed by the repository owner.

---

## Remaining checks before submission

- [ ] Hecate: patched `hecate.py`, fp16 vs fp32, on **both 2.4 µm and 9.6 µm** checkpoints, 2D and `--output-3d`,
      both depth orders (`--reverse`), on a T4; preferably on the PHerc. 0306B crop.
- [ ] Studio: patched upstream CLI end to end on a T4 (`auto` → fp16) and the RTX 4060 (`--precision fp16`); the
      GPU numbers above were measured through a wrapper with the same code path.
- [ ] Studio upstream tests green on Linux (4 tests fail on Windows before and after the patch).
- [ ] Both pull requests posted with owner approval; links filled in above.
- [ ] Confirm with the rules page that this entry is bundled correctly with the November Villa submission.
