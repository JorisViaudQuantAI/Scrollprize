# DRAFT (not posted): `--precision fp16` for `hecate.py`

Target: [huggingface.co/scrollprize/hecate](https://huggingface.co/scrollprize/hecate) (HF pull request).

| patch | base | use |
| --- | --- | --- |
| `0002-hecate-upstream-fp16.patch` | `main` @ `9cb86e500e944b11a06a7020403cde5dffb5bcb2` | if [discussion #3](https://huggingface.co/scrollprize/hecate/discussions/3) (`--resample`) is still open |
| `0002b-hecate-upstream-fp16-after-pr3.patch` | `refs/pr/3` @ `f4386b19e0639f1c3da79572309ecc3f4cec760e` | if #3 is merged first |

Both make the same change. They differ only in the context of the argparse hunk: #3 adds `--resample` on the line
right after `--precision`, so `0002` and #3 touch adjacent lines and git reports a one-line conflict whichever
merges second (checked with `git merge-file`). `0002b` is that conflict resolved (both arguments kept).
Both were re-checked against fresh clones of `main` @ `9cb86e50` and `refs/pr/3` @ `f4386b19`: `git apply --check`
passes, blob hashes match the patches' `index` lines, and the patched `predict()` runs fp32 on CPU and refuses
fp16/bf16 there.

---

## Title

Add `--precision fp16` for GPUs without fast bf16 (T4, V100)

## Summary

`hecate.py` offers `--precision fp32` (default) and `bf16`. GPUs before Ampere, such as the free-Colab T4, have no
fast bf16 path: on a T4, bf16 is slower than fp32. This PR adds an `fp16` choice:

- `predict()` accepts `precision='fp16'` and runs `torch.autocast('cuda', dtype=torch.float16)`;
- fp16 and bf16 are both refused on CPU, with the message "bf16 and fp16 require CUDA; otherwise use fp32";
- `--precision` gains `fp16` and a help string; the default stays `fp32`;
- one README sentence.

3 hunks in `hecate.py`, no other behaviour change.

## Measurements

Colab T4, torch 2.14.1+cu130, `hecate_9.6um.pth`, 2D `--output`, batch 32, one 27 × 999 × 999 crop of the public
PHerc. 0841 surface volume resampled to 9.6 µm, fresh process per run (model load included):

| precision | seconds | vs fp32 output (uint8 levels) |
| --- | ---: | --- |
| **fp16** | **41.2** | **max 1, mean 0.047, r 0.99999** |
| fp32 (default) | 123.6 | — |
| bf16 | 228.6 | max 11, mean 0.321, r 0.999893 |

fp16 is 3.0× faster than the default end to end, and its output differs from fp32 by at most one uint8 level.
On CPU, the patched script's default (`fp32`) output is byte-identical to the unpatched script, and
`--precision fp16` is refused.

This is a numerical parity check, not an accuracy claim (that crop's segment is in Hecate's published training
bucket).

## Limits

- One crop of one scroll, one batch size, one run per precision.
- Only the 9.6 µm checkpoint and the 2D output were measured so far (see checklist).
- fp16 has a smaller range than bf16; no overflow seen here. On Ampere or newer, bf16 remains a fine choice.

## Licence

MIT, as the repository.

---

## Pre-posting checklist (owner)

- [ ] Run patched `hecate.py` fp16 vs fp32 with **both** checkpoints (`hecate_2.4um.pth` and `hecate_9.6um.pth`),
      2D `--output` and `--output-3d`, with and without `--reverse`, on a Colab T4 in two unattended steps (crops are
      written strictly outside the git repository into `/tmp/hecate_parity_crops` and deleted afterwards; public CT
      structural crops for numerical parity only, not an accuracy claim):
      1. Stream both uint8 `[64, 256, 256]` parity crops (`PHerc0306B` 8.640 µm positive control + `PHerc0846A`
         2.403 µm public CT, `< 15 MB` total HTTP transfer):
         `python gemini_lane/progress_prize_fp16/prepare_hecate_parity_crops.py --out-dir /tmp/hecate_parity_crops`
      2. Run the burst (use `--stages studio,hecate` to verify both PRs in one pass, or `--stages hecate` for Hecate
         only; checkpoints hash-checked against their LFS pointers; gate max ≤ 3 and mean ≤ 0.25 uint8 levels, 2D and
         3D; defaults `--vram-fraction 0.35 --batch 1 --max-gpu-temp-c 68`):
         `python gemini_lane/progress_prize_fp16/verify_upstream_gpu_burst.py --stages studio,hecate --hecate-patch main --hecate-input-9p6 /tmp/hecate_parity_crops/hecate_9p6um_crop.npy --hecate-input-2p4 /tmp/hecate_parity_crops/hecate_2p4um_crop.npy --output /tmp/hecate_parity_crops/burst_receipt.json`
      Copy the sanitized JSON receipt, remove `/tmp/hecate_parity_crops`, and add the receipt's numbers above as a
      separate table (batch 1; the existing table is batch 32). fp32 memory at batch 1 is unmeasured; if a run stops on
      the cap on an 8 GB laptop GPU, use a T4.
- [ ] Check the state of discussion #3 and pick `0002` or `0002b` (`--hecate-patch pr3`); re-run
      `git apply --check` on the chosen base.
- [ ] Optionally mention the overlap in a short comment on #3.
- [ ] Owner approval before opening the HF pull request.
