# Studio ink detector on a Colab T4 (fp16)

## What it is

`studio_t4.py` runs the released ink detector that won the PHerc0211 First Letters prize
([vesuvius-studio-multiscroll-cpt](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt),
soft-loss weights `studio-softloss-20261023-ema.safetensors`) on one rendered surface.

The release CLI offers bf16 or fp32. It refuses a GPU without bf16, such as a T4. In fp32 it needs
11.35 GB at batch 1 on a T4. This tool runs the same network, normalisation and tiling with fp16 autocast.
fp16 needs 1.17 GB at batch 1.

It takes a rendered stack, picks the 16 planes, runs the model and writes four files:

| file | content |
| --- | --- |
| `PREFIX.prob.npy` | float32 `[H,W]` ink probability |
| `PREFIX.valid.npy` | bool `[H,W]`: the 16 selected planes are not all zero, and a window covered the pixel |
| `PREFIX.preview.png` | 8-bit preview, `round(255·p)`, invalid pixels 0 |
| `PREFIX.receipt.json` | input hash, weights hash, planes, order, stride, precision, batch, device name, peak allocated/reserved GPU memory, seconds, pixels, output hashes. No paths. |

It never overwrites an output. It never downloads anything unless you pass `--fetch-weights`. The weights are
always checked against the SHA-256 that the release's `infer_release.py` pins
(`96b32628…0e90`, 893,443,724 bytes). A missing or mismatching file is an error.

**Model definition.** The network is imported from your own checkout of the release repository
(`--release-dir`). Its `studio-multiscroll-cpt-20261023/hf/model.py` must match the SHA-256 of that file at
release commit `bf66f753176792a966bca17822629fb8821f20c8`. We chose this over copying `model.py`, so this
folder contains no copy of the network. The tiling and precision functions are adapted from the release's
`hf/inference.py` (MIT); the notice is in the script header.

Normalisation is exactly the release's: uint8 → float32, `clip(raw, 0, 200) / 255`. Tiling is the release's:
128 px windows, central 96 px kept, stride 48 with Hann blending (default) or stride 96 without blending.
Windows with no CT data in their central 96 px are skipped. Pixels with all 16 planes zero are written as 0.

## Measured numbers on a T4

All numbers in this section and the next three were measured on one Colab T4, torch 2.14.1+cu130, with a
private wrapper. This CLI is the same code path as the measured private wrapper, re-packaged. The packaged
CLI itself has since run on a Colab L4 and, independently, on an RTX 4060 (section "Measured on an L4").

| setting | time per 128 px window | peak allocated GPU memory |
| --- | ---: | ---: |
| fp32, batch 1 | 340 ms | 11.35 GB |
| fp16, batch 1 | 132 ms | 1.17 GB |
| fp16, batch 16 (default here) | 105 ms | 5.7 GB |

- fp16 against fp32, 512 × 512 px crop, stride 64 Hann: max |Δp| 0.0019, mean |Δp| 0.00014. AUC on the
  control region 0.9158 in both.
- Batch sweep, fp16, stride 48: 130 ms (b1), 121 (b2), 116 (b4), 111 (b8), 105 (b16), 105 (b24), 105 (b32)
  per window. Peak allocated 1.2 / 1.5 / 2.1 / 3.3 / 5.7 / 8.2 / 10.6 GB.
- Scan speed, fp16, batch 16, stride 48 Hann: 45 s per megapixel of data, about 52 s/cm² per pass at 9.37 µm.
- Stride 96 without blending is about 4× faster: 12.1–12.6 s per megapixel, about 17 s/cm² per pass at
  8.64 µm. It leaves a visible 96 px tile grid.

## Measured on an L4 and on an RTX 4060 (packaged CLI)

Colab L4, 41 × 1024 × 1024 crop of the PHerc. 0306B render, stride 48 Hann, 529 windows, reference fp32 at
batch 4:

| precision | batch | ms per window | peak allocated | peak reserved | max \|Δp\| vs fp32 | mean \|Δp\| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 4 | 95.6 | 15.7 GB | | | |
| fp16 | 16 | 82.7 | 7.6 GB | 9.2 GB | 0.0031 | 0.00028 |
| fp16 | 32 | 87.8 | 14.3 GB | 17.5 GB | 0.0031 | 0.00028 |
| bf16 | 16 | 82.0 | 7.6 GB | 9.2 GB | 0.035 | 0.0027 |
| bf16 | 32 | 86.6 | 14.3 GB | 17.5 GB | 0.036 | 0.0027 |

- bf16 is no faster than fp16 and sits about ten times further from fp32. A second measurement on an
  RTX 4060 (Gemini, `collab/gemini-002.md`) found the same: max |Δp| 0.0025 for fp16 and 0.011–0.062 for bf16
  against CPU fp32. **`--precision auto` therefore now means fp16 on every CUDA GPU.** bf16 runs only when
  asked for.
- On an 8 GB card, batch 4 at stride 48 peaked at 3.0 GB allocated (same measurement).
- Stride 96 at 9.36 µm: about 10 s/cm² per order on the L4.
- Do not rank stride-96 maps with a line finder. Without blending, the tile seams break chains: on the known
  PHerc. 0343 letters the best chain was 3.8 at stride 96 and 9.5 at stride 48.

## Plane convention

`desc` takes planes c+8 down to c−7. This is the documented order ("render planes 00–30, stack 23 down to 8").
`asc` takes c−8 up to c+7. Here c is the mesh layer index plus `--shift`.

In the vocabulary of [`../studio_inputs/`](../studio_inputs/README.md): `desc` is `release-descending`. `asc`
is **not** `same-band-reversed` (that is c−7 up to c+8). `asc` is the release band taken from the whole
stack reversed, i.e. from the same render with the opposite normal sign. That is why it reads the same tissue on
a flipped-normal render.

Measured on the PHerc0841 w00 known-text control (AUC against the 2.4 µm answer key, text region, centre shift 0):

| render | order | planes | stride | AUC |
| --- | --- | --- | --- | ---: |
| `vc_render_tifxyz`, 41 layers, mesh at 20, no `--flip-normals` | `desc` | 28→13 | 96 | 0.916 |
| same | `desc` | 28→13 | 48 Hann | 0.924 |
| same | `asc` | 12→27 | 96 | 0.568 |
| same mesh, our renderer with normals flipped, 65 layers, mesh at 32 | `asc` | 24→39 | 96 | 0.908 |
| same | `desc` | 40→25 | 96 | 0.573 |

- A render without `--flip-normals` needs `desc`. A flipped-normal render needs `asc` for the same planes.
  The wrong order gives 0.57, close to chance.
- For the published 31-layer renders made with flipped normals (mesh at 15), the same rule gives `asc`,
  planes 7→22.
- The normal sign says which planes form the same stack. It does not say which face of the sheet carries
  writing. On the two public positives below, the better order was `asc` on one and `desc` on the other. On a
  new surface, run both orders.

## Depth tolerance

Same control, `desc`, stride 96, `--shift` in layers (AUC 0.916 at shift 0):

| shift | −1 / +1 | −2 / +2 | −4 / +4 | −6 / +6 |
| --- | --- | --- | --- | --- |
| AUC | 0.912 / 0.913 | 0.901 / 0.896 | 0.867 / 0.860 | 0.808 / 0.806 |
| loss | < 0.005 | ≈ 0.02 | ≈ 0.05 | down to 0.81 |

## What the control shows, and what it does not

- It shows that the plane order, the normalisation, the tiling and the fp16 path are right: the map matches
  a known-text answer key (AUC 0.92), and the wrong order does not.
- PHerc0841 is one of the three training donors of this model. The control says nothing about accuracy on an
  unseen scroll.
- The answer key is itself another model's output, not hand-labelled ink.
- For a replay of the release's own PHerc0841 holdout on CPU, see [`../studio_control/`](../studio_control/README.md).

## Two public positives from awarded scrolls

fp16, stride 96 without blending, shift 0. 41-layer renders with flipped normals, mesh at 20, 8.64 µm.
Letter-box AUC: pixels in the published letter boxes against far background.

| positive | better order | Studio | Studio, other order | reader-v3, same order as Studio |
| --- | --- | ---: | ---: | ---: |
| PHerc0343 | `asc` (12→27) | 0.736 | 0.699 | 0.684 |
| PHerc0306B | `desc` (28→13) | 0.601 | 0.541 | 0.572 |

- Studio reads in the same order as reader-v3 on both, with a higher letter-box AUC. Its PHerc0343 map shows
  both text rows, with the 96 px tile grid visible.
- The PHerc0306B box is mostly blank papyrus. This may explain the lower AUC of both models there.
- The release's training list includes PHerc0343P label rows. We did not check whether they overlap this
  surface.

## Colab quick start

Runtime: T4 GPU. Colab's preinstalled torch is used as is.

```bash
git clone -b claude/firstletters-colab https://github.com/JorisViaudQuantAI/Scrollprize.git repo
git clone https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt studio-release
git -C studio-release checkout bf66f753176792a966bca17822629fb8821f20c8
pip install safetensors tifffile      # add "zarr" for a zarr input

python repo/firstletters-colab/studio_t4/studio_t4.py \
  --input my_render/ --mesh-index 20 --order desc \
  --weights weights/studio-softloss-20261023-ema.safetensors --fetch-weights \
  --release-dir studio-release --out-prefix out/my_segment_desc
```

- `--input`: a `.npy` uint8 `[L,H,W]`, a directory of per-layer TIFFs (`00.tif`, `01.tif`, …, or a `layers/`
  folder of them) as written by `vc_render_tifxyz`, or a zarr store (array `0` if it is a group). Only
  uint8 is accepted; nothing is rescaled.
- `--mesh-index` is the layer of the mesh surface (20 for a 41-layer `vc_render_tifxyz` render, 15 for 31).
  It is not guessed from the stack length.
- Run `--order asc` with a new `--out-prefix` too (see above).
- `--prepared` takes a `[16,H,W]` `.npy` already in model order, e.g. from `../studio_inputs/prepare.py`.
- `--stride 96` for a quick first pass; `--batch-size` (default 16 on CUDA); `--precision` (default fp16 on a
  GPU, fp32 on CPU; bf16 only on request); `--device cpu` works but is slow (about 3 s per window
  in fp32 with 4 threads on our CPU test host).
- Without `--fetch-weights`, download the asset yourself from the
  [release page](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/tag/studio-softloss-20261023).

## Tests

```bash
python -m unittest discover -s firstletters-colab/studio_t4 -p 'test_*.py' -v
```

CPU only, no network, no real weights. They cover plane selection for both orders and shifts (including
out-of-range errors and agreement with `studio_inputs/prepare.py`), the three input formats, Hann tiling
reconstructing a constant and equalling a direct forward of a small stand-in model, stride 96 covering every
pixel once, the validity mask, refusal to overwrite, weights and `model.py` hash mismatches, and the receipt.
With `STUDIO_RELEASE_DIR=<release checkout>`, one more test builds the real network with random weights and
checks that this tiling equals the release's own `predict()` (max difference 0.0, CPU fp32, stride 96 and
stride 48 Hann). Torch-dependent tests are skipped if torch is not installed.

On CPU we also ran the CLI end to end with the real weights (fetched with `--fetch-weights`, hash verified) on
a synthetic 41-layer stack: its map equals the release's `hf/inference.py` output, max difference 0.0
(fp32, stride 96).

## Limits and what was not measured

- The packaged CLI has run on two GPUs (L4, RTX 4060), not on a T4. The T4 numbers above are from the
  wrapper it was re-packaged from.
- The T4 numbers come from one GPU, one torch version and one control. fp16 against fp32 was compared on one
  512 × 512 px crop.
- The zarr reader was tested on a store written by zarr 3.4, not on `vc_render_tifxyz --zarr-output`.
- The whole map is held in memory: about 35 bytes per pixel (16 input planes, the float32 map and weights,
  masks). This is arithmetic, not a measurement; 100 megapixels need about 3.5 GB of RAM.
- A probability map is not a reading. There is no threshold here, and none was calibrated.
- Stride 64 (the submission tiling) is not offered; use the release CLI for it.

## Licence and credit

The detector, its training and its weights are the work of the release authors:
[github.com/priyankaranke/vesuvius-studio-multiscroll-cpt](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt).
Their code is MIT. Their weights, and the outputs they released (including probability maps), are CC BY-NC 4.0,
the licence of the scan data. Check the prize rules before you use output of these weights in a submission. This folder is MIT, like the rest of the
repository.
