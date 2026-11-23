# DRAFT (not posted): `--precision fp16` for the released inference CLI

Target: [priyankaranke/vesuvius-studio-multiscroll-cpt](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt), base `main` @ `bf66f753176792a966bca17822629fb8821f20c8`.
Patch: `0001-studio-upstream-fp16.patch` (5 files, +55 / −6). Apply with `git apply 0001-studio-upstream-fp16.patch`.

---

## Title

Add `--precision {auto,bf16,fp16,fp32}` so the released weights run on GPUs without BF16 (T4, V100)

## Summary

`hf/inference.py` and `infer_release.py` offer `--amp bf16|fp32`, default `bf16`. On a GPU without native BF16
(compute capability < 8.0, e.g. the free-Colab T4) the CLI stops with "GPU does not support BF16; use --amp fp32",
and fp32 needs 11.35 GB of GPU memory at batch 1 on a T4.

This PR:

- adds `--precision` with choices `auto`, `bf16`, `fp16`, `fp32`; `--amp` stays as an alias, so existing commands
  are unchanged;
- adds `fp16` autocast (`torch.autocast("cuda", dtype=torch.float16)`) next to the existing bf16 path in `predict()`;
- makes `auto` the default: **bf16 on compute capability ≥ 8.0 (unchanged behaviour), fp16 below, fp32 on CPU**.
  Pre-Ampere GPUs now run instead of erroring out;
- keeps the existing `is_bf16_supported()` refusal for an explicit `--precision bf16`;
- adds 4 CPU tests (`resolve_amp` per compute capability, explicit choices kept, fp16 on CPU equals fp32, the
  option passed on by `infer_release.py`) and one README sentence.

Tiling, normalisation, model and weights are untouched.

## Measurements

Same network, normalisation and tiling as `hf/inference.py` (run through a wrapper that imports the release's
`model.py` at this commit, SHA-256 pinned, and the released soft-loss weights, SHA-256 `96b32628…0e90`).

**NVIDIA T4 (cc 7.5), torch 2.14.1+cu130, 128 px windows:**

| setting | ms per window | peak allocated GPU memory |
| --- | ---: | ---: |
| fp32, batch 1 | 340 | 11.35 GB |
| fp16, batch 1 | 132 | 1.17 GB |
| fp16, batch 16 | 105 | 5.7 GB |

fp16 vs fp32 on a 512 × 512 px crop, stride 64 Hann: max |Δp| 0.0019, mean |Δp| 0.00014.

**RTX 4060 Laptop (Ada, cc 8.9, 8 GB), torch 2.10.0+cu128**, public PHerc. 0306B crop (169 × 169 px, 25 windows,
stride 48 Hann), each precision compared with CPU fp32 on the same crop:

| run | max \|Δp\| vs CPU fp32 | mean \|Δp\| | Pearson r | peak allocated |
| --- | ---: | ---: | ---: | ---: |
| bf16, desc, batch 1 | 0.011095 | 0.001753 | 0.999929 | 1733 MiB |
| bf16, desc, batch 2 | 0.011070 | 0.001758 | 0.999928 | 2164 MiB |
| **fp16, desc, batch 2** | **0.002526** | **0.000410** | **0.999996** | **2164 MiB** |
| bf16, desc, batch 4 | 0.011210 | 0.001755 | 0.999928 | 3010 MiB |
| **fp16, desc, batch 4** | **0.002538** | **0.000408** | **0.999996** | **3010 MiB** |
| bf16, asc, batch 4 | 0.061742 | 0.006716 | 0.996717 | 2514 MiB |
| **fp16, asc, batch 4** | **0.002113** | **0.000156** | **0.999993** | **2514 MiB** |

On Ada, fp16 has the same peak memory as bf16 at equal batch and a similar speed (92–139 ms per window across
both), and a 4.4× (descending plane order) to 29.2× (ascending order) smaller max |Δp| against fp32.

**Colab L4 (Ada, cc 8.9)**, 41 × 1024 × 1024 crop of the same PHerc. 0306B render, stride 48 Hann, 529 windows,
GPU fp32 reference at batch 4:

| precision, batch | ms per window | peak allocated | max \|Δp\| vs fp32 | mean \|Δp\| |
| --- | ---: | ---: | ---: | ---: |
| fp32, 4 | 95.6 | 15.7 GB | — | — |
| fp16, 16 | 82.7 | 7.6 GB | 0.0031 | 0.00028 |
| bf16, 16 | 82.0 | 7.6 GB | 0.035 | 0.0027 |

On both Ada GPUs, bf16 is no faster than fp16 and about ten times further from fp32. This is why
`--precision fp16` is also offered on cc ≥ 8.0. `auto` still picks bf16 there so existing outputs do not change.
Given these numbers we would suggest `auto` → fp16 on every CUDA GPU (our own CLI now does this); happy to make
that one-line change in this PR if you agree.

## What these numbers are not

- Parity and memory only. They are not an accuracy claim: the PHerc. 0306B crop (not in this model's training
  lists) is sub-letter scale, and no AUC is reported here.
- The T4 and RTX 4060 numbers come from a wrapper with the same code path, the L4 numbers from our packaged
  re-implementation of it; this exact patched CLI has been tested on CPU only so far (see checklist).
- One GPU of each kind, one torch version each, no repeats.

## Tests

`python -m unittest test_infer_release test_inference_tiling -v` on CPU: the 4 new tests pass. On Windows, 4
existing tests fail before and after the patch (backslash path in `test_loaded_weights_invoke_documented_codepath`,
memmap file lock in `test_same_output_fewer_windows`); to be confirmed green on Linux before posting.

The patch was re-checked against a fresh clone of `main` @ `bf66f753`: `git apply --check` passes, pre- and
post-image blob hashes match the patch's `index` lines, and the patched `predict()` gives identical output for
fp32, fp16 and bf16 on CPU (autocast is CUDA-only).

## Licence

Code changes under the repository's MIT licence. Weights untouched (CC BY-NC 4.0).

---

## Pre-posting checklist (owner)

- [ ] Run the patched CLI end to end on a T4 (`auto` → fp16) and on the RTX 4060 (`--precision fp16`), weights
      hash-checked: `python verify_upstream_gpu_burst.py --stages studio --output studio_burst.json` (clones and
      patches `main`, runs `infer_release.py` on the 169 × 169 PHerc. 0306B smoke crop: CPU fp32 reference, CUDA
      fp16, `auto`, and bf16 on cc ≥ 8.0; defaults `--vram-fraction 0.35 --batch 1 --max-gpu-temp-c 68`).
- [ ] Decide whether `auto` should mean fp16 on every CUDA GPU before posting.
- [ ] Run the upstream tests on Linux.
- [ ] Confirm `main` is still `bf66f753` (`git apply --check` again if not).
- [ ] Owner approval before opening the PR.
