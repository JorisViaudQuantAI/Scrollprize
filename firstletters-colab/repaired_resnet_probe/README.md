# Fixed central-field probe of the repaired ResNet50 detector

**Status: prepared before probe acquisition or inference.** The owner/root must publish the exact package freeze before running it. No model output was used to select these locations, the depth slab, or an orientation. There is no automatic promotion rule or unread-scroll acquisition.

This is a four-pass architecture-transfer probe on two public, model-assisted control fields. It asks whether a different released detector produces usable ink agreement on a fixed small portion of each field. It performs no training on these donors; inherited checkpoint exposure still requires audit. It is not a reading, a native-acquisition benchmark replication, or an independent truth test.

## Fixed physical inputs

Use the centre of each retained PHerc. 0841 and PHerc. 0814 fine context from the earlier frozen experiment. Their exact uint8 tensor hashes are pinned in [protocol.json](protocol.json). Each tensor has shape `64 × 1152 × 1152` at 2.4 µm. The original selection and donor manifest are separately hash pinned. No Hecate score file or result is read.

The model input is `18 × 256 × 256` at **7.9 µm**, symmetrically centred within each retained context:

- Depth aperture: **142.2 µm**, leaving **5.7 µm** on either side.
- XY input extent: **2.0224 mm** per side.
- Decoder output: `64 × 64`, with a **31.6 µm** cell pitch.
- Scored core: the complete central `[16:48, 16:48]`, **1.0112 mm** per side.

The pitch ratio is exactly `79/24`. Fine source cell `i` occupies `[i, i+1]`. Depth edges are `19/8 + j*79/24`; XY edges are `464/3 + j*79/24`. Integer overlap numerators sum to 79 on every axis. The runner integrates all intersecting cells with int64 sums, divides once by `79³`, and rounds nearest even to uint8. It never pads, extrapolates, or invents missing cells. It then clips the resampled input at 200 and divides by 255 in float32.

The author's standard selection is global layers `24:42` in a 65-layer stack, an upper-centred even-depth crop. This probe instead fixes a symmetric physical slab about the retained fine context's centre. The retained context itself came from the earlier fixed source-centre-31.5 recipe. This is an explicitly recentered, fine-derived transfer probe; it does not reproduce the author's native stack appearance or depth crop.

## Categorical labels and endpoints

Read the **original** category PNG/NPY through the pinned donor manifest and cached asset hash. Do not reuse the previously resampled 9.6 µm label array. Unknown remains category 0, background is 1, and ink is 2.

For original coarse core origin `o` and decoder cell `j = 16..47`, the physical centre is

`(o + 128)*9.6 + (j + 0.5 - 32)*31.6` µm.

Its original 9.36 µm categorical index is exactly

`floor(5*(48*(o+128) + 79*(2*j+1-64))/234)`.

Apply this independently to y and x. This is a **centre-point categorical comparison** at 31.6 µm. The model was trained against area-downsampled labels; these centre categories do not replicate that training target. No interpolated or majority-vote label threshold is introduced.

All known centre categories are scored, with no intensity-dependent veto. The complete acquired input support is required; stored zero intensities remain values. Zero-column counts are diagnostic only.

Report **both source and reverse order**, without choosing the higher-scoring order: AP, ROC-AUC, and descriptive ink recall at a 5% known-background operating point. For each donor/order, let `k = floor(0.05*Nbackground)` and select scores strictly greater than descending background score `k`. Ties cannot exceed the budget. This is not a transferred threshold or an inferential false-positive guarantee. Missing classes produce undefined endpoints; the field is retained and never replaced.

Each donor gets a panel showing its entire categorical core and both probability maps on the identical **0–1** scale. No per-image min/max normalization is used. Complete 64-square probabilities, inputs, labels and panels stay outside Git.

## Model and execution pins

The [pinned repair wrapper](https://github.com/Nieuwlaar/resnet50-scroll1-checkpoint-repair/blob/1e47872dcac8e26e4ca7e1befeec7dbb8be5200d/scripts/common.py) loads the exact backbone and repaired decoder with strict state matching. The architecture uses a 3D ResNet50, per-stage maximum over depth, and a 2D decoder. Its direct path needs Torch, NumPy and safetensors; Pillow and Matplotlib support labels and panels.

The [repaired checkpoint revision](https://huggingface.co/Nieuwlaar/resnet50_7.9um_scroll1_frags-repaired/tree/1c35111679394a8f0c9c0eee033edf8e9dfa29e4) supplies the 342,867,696-byte safetensors file. Configuration and backbone come from [the pinned original model revision](https://huggingface.co/scrollprize/resnet50_7.9um_scroll1_frags/tree/f3975de34d4734dd507360304f828808b7182139). Every URL, byte count and SHA256 is in the protocol.

Execution is CPU, batch one, float32, four Torch threads. `model.eval()` and `torch.inference_mode()` are mandatory; running-normalization buffers are hashed before and after predictions. Exactly four passes are permitted, in order: PHerc. 0841 source/reverse, then PHerc. 0814 source/reverse. There is no warmup, backend search, retry, or orientation selection.

Caps are **400,000,000 new application-body bytes**, **zero CT requests**, **four forwards**, **300 seconds per forward** and **1,200 seconds total inference** as soft guards, **1,800 seconds whole run**, and **6.5 GiB peak-RSS soft guard**. Model assets allow at most 16 HTTP requests including redirects, four requests per asset, and no retry. Completed outputs survive a resource stop. Native operations and blocking HTTP calls can defer the whole-run alarm; wire bytes are not measured.

## Running after publication

Use an environment with the dependencies above already available. Dependency presence is checked before acquiring model assets. Supply the exact retained fine arrays and original label caches:

```bash
python firstletters-colab/repaired_resnet_probe/run.py \
  --freeze-reference PUBLISHED_COMMIT \
  --fine-context-root ../retained-public-fine-run \
  --label-cache ../old-public-run/source-cache \
  --label-cache ../old-public-role-run/source-cache \
  --asset-dir ../repaired-resnet-assets \
  --output ../repaired-resnet-probe-output
```

The output path must be fresh and outside Git. Preseeded model assets are accepted only if their exact size and hash match. Otherwise only the four protocol-listed model assets may be fetched; CT and labels are never fetched.

`result.json` retains each completed pass, its timing and output hash, undefined endpoints, resource accounting, and interpretation limits. `pre_forward_inputs.json` records both prepared inputs before any prediction. The runner verifies all four package files against the supplied Git freeze before proceeding.

Geometry-only checks use generated arrays and exact rational arithmetic:

```bash
python firstletters-colab/repaired_resnet_probe/test_geometry.py
```

They do not load real CT, labels, weights or model code. The separate Hecate experiment and its gates remain unchanged.
