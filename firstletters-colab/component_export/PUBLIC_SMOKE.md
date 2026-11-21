# Real Hecate component export on a tiny published CT input

**Measured result:** Two actual CPU inference subprocesses completed on
2026-11-21: the historical producer without this feature, then the current
producer with component export. Each executed six one-tile Hecate forwards.
All **ten legacy PNG/NPY output files were byte-identical**. The new run wrote
six complete component packets, and their metadata/array hashes verified.
Every nonruntime field of the legacy JSON sidecar also matched after the
explicit normalization of the two output-directory prefixes.

Evidence is in [public-smoke-v1.json](evidence/public-smoke-v1.json); its SHA-256
is `07ecad1dd71df8f27994cf1adce697ff40d55300fcba806162a71e02ad4a1692`.
This proves a bounded producer/export regression on the specified input.
It is not an ink-accuracy result, a reading, or a completed downstream review.

## Frozen input and model

**From sources:** The input is the already published PHerc. 306B smoke crop
([pinned dataset](https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/blob/f435998b55269762942a979052044309b29650ac/docker/build/smoke_crop.npz)).
Its construction uses stored render planes 8 through 31 and clips intensities
at 200
([source implementation](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/code/smoke_v7.py)).
The Hecate checkpoint is the published 9.6 µm model at revision
`9cb86e500e944b11a06a7020403cde5dffb5bcb2`
([pinned release](https://huggingface.co/scrollprize/hecate/tree/9cb86e500e944b11a06a7020403cde5dffb5bcb2)).

**Method decision, made before outputs:** Take only `st[:71,:71,:]`, then
transpose the 24-plane array to contiguous ZYX order. This deliberately reuses
the published crop's pre-existing clip at 200. It is **not a standard raw-input
Hecate quality evaluation**. No threshold, region or depth was selected from
these new outputs.

| Setting | Frozen value |
|---|---|
| Source artifact hash | `4a81e193e253e8b6234eaa4c1b274e33c08b214183951ed8e573c9ae3f8f639d` |
| Derived input shape | 24 × 71 × 71, uint8 ZYX |
| Derived input file hash | `641aa8c04fe1273f55c6280afda52b08f399a5b66ed76d225f172ed64ef79080` |
| Input pitch | 8.64 µm for Z, Y and X |
| Model checkpoint bytes | 522,576,086 |
| Model checkpoint hash | `809f4f10f7cb7afa19b4bee0f7d2ab31edd7e11b664f9f210cc9c17f22fcfe5d` |
| Runtime model source | Repository `ink/third_party/hecate/hecate.py`, hash `c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59` |
| Model input | 16 × 64 × 64, one patch per component |
| Resampling | `auto`, target 9.6 µm, model XY grid 64 × 64 |
| Surface layer and shifts | Centre 11.5; shifts −1, 0, +1 model planes |
| Directions | Forward and reverse |
| Inference | CPU, FP32, no AMP, no TTA, batch size 1, four threads |
| Tile stitching | Stride 32, Hann blend |
| Mask input | None; acquisition support remains unknown |
| Execution bound | Two subprocesses total, each with a 300-second hard timeout |

The protocol was frozen before inference:
[public_smoke_protocol.json](public_smoke_protocol.json), SHA-256
`1a8dc4ab721a35d9dc1dfc5fd5c29dfa2f86b056e562c3da276956a646d0dfe5`.
The pre-run receipt also records the exact private command paths, input hashes,
source hashes and runtime. The public evidence replaces those local path
prefixes with `{WORK}`, `{REPO}` and `{PYTHON}`.

## Actual outputs and support

**Measured here:** Each of the following six packets contains a 71 × 71
float32 probability array, a boolean prediction-support mask, a boolean
depth-complete-support mask, and metadata. The completion receipt and all
payload hashes verified.

| Direction | Shift, model planes | Recorded physical shift | Prediction-supported pixels | Depth-complete pixels | XY zero padding |
|---|---:|---:|---:|---:|---|
| Forward | −1 | −9.6 µm | 5,041 | 5,041 | None |
| Forward | 0 | 0 µm | 5,041 | 5,041 | None |
| Forward | +1 | +9.6 µm | 5,041 | 5,041 | None |
| Reverse | −1 | −9.6 µm | 5,041 | 5,041 | None |
| Reverse | 0 | 0 µm | 5,041 | 5,041 | None |
| Reverse | +1 | +9.6 µm | 5,041 | 5,041 | None |

**Measured here:** The exporter records source-layer interpolation positions,
weights and applied order, effective depth pitch 9.6 µm, the 64 × 64 model
grid, the 71 × 71 output grid, linear output resizing, and input
`BORDER_REPLICATE`. Every model plane has complete depth interpolation mass.
No XY tile zero padding was needed. The mean of each direction's three saved
float32 components, cast to float16, exactly reconstructs its saved legacy
float array on all 5,041 prediction-supported pixels.

**Interpretation:** These support checks describe numerical prediction/depth
support. They do not certify the acquisition, anatomical alignment, tissue, or
the complete receptive field near a resampling boundary. Positive and negative
shifts refer to increasing/decreasing source index before reversal; they do
not assert which way is outward from the papyrus.

The mesh identity, surface normal, normal convention and anatomical frame are
still `null`. No acquisition/context-validity mask was fabricated from CT
intensity or score confidence. **The `depth_screening` consumer was not run and
remains blocked on those missing geometry/support inputs.** This is a real
producer/export test, not a full producer-to-consumer handoff.

The ten legacy files compared byte for byte comprise the two direction PNGs,
six per-direction/per-shift PNGs, and two float16 direction arrays. Their
individual legacy/new hashes are retained in the evidence. Runtime exclusions
from metadata comparison are enumerated in the protocol; no inference or
geometry fields were excluded.

## Observed time and storage

**Measured here:** The historical CLI subprocess took 13.8827 seconds; the new
CLI with export took 14.6417 seconds. Network-forward time reported by the two
producers was 11.63 and 11.38 seconds, respectively. These are **one ordered
observational pair**, with the historical run first. Cache state, process
startup and ordinary runtime variation are uncontrolled. They do not establish
a general export-overhead percentage or speedup. There was no extra trained
model run beyond the six forwards in each subprocess.

**Measured here:** The component packet contains 232,383 logical bytes,
including its arrays and metadata. The historical output directory contains
38,007 logical bytes; the new directory contains 270,459. Those directory
totals include logs and sidecars as well as images/arrays, so their difference
is not exactly the component payload. File-specific output sizes and hashes
are in the main evidence; filesystem allocation measurements are in
[public-smoke-storage-v1.json](evidence/public-smoke-storage-v1.json).

The four prepared input files occupy 522,741,910 logical bytes, dominated by
the 522,576,086-byte checkpoint. Model weights, CT arrays and output imagery
remain outside Git. Only code, frozen protocol and sanitized numerical
receipts are included here.

## Reproduce the bounded check

Use the existing inference environment with NumPy, Torch and OpenCV. Obtain the
pinned public `smoke.npz` with `tile_phase/prepare.py`, or supply that exact
artifact from an existing public-control checkout. The harness verifies its
hash, downloads the pinned Hecate checkpoint, extracts the frozen CT input,
and recovers the historical producer from Git blob
`891013ad0ae4f51dc5d1657f23ade30009ef65d1`.

```bash
python firstletters-colab/component_export/public_smoke.py --prepare-only --work-dir /tmp/scrollprize-hecate-component-smoke-v1 --smoke /tmp/scrollprize-phase-input/smoke.npz
python firstletters-colab/component_export/public_smoke.py --run --work-dir /tmp/scrollprize-hecate-component-smoke-v1 --smoke /tmp/scrollprize-phase-input/smoke.npz
```

`--prepare-only` makes no inference calls. `--run` writes its preregistration
before executing either real CLI. The harness checks the frozen producer,
exporter and Hecate module hashes before execution and at completion. It kills
a subprocess after 300 seconds, stops after any failed/incomplete run, and
refuses an automatic retry once the attempt has begun. Use a new directory for
an explicitly planned new experiment; do not silently replace this result.

The exercised harness SHA-256 is
`fa2cfaa0bfc24706bc903a488f864f7591343d07f7507cf675898cc28e07e7ce`.
No production inference code was changed for this smoke experiment.
