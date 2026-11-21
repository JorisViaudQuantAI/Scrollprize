# Released gA: bounded public-control CPU adapter

This is a **plumbing and provenance check**, not an independent test of text detection. It makes the released glyph model runnable without importing the original environment or its path-dependent modules. It provides no production gate, discovery claim, false-positive estimate, or calibrated text probability.

## What is reproduced

**From a source:** The source is [SirProdigle/pherc0306b-first-letters at `feabff6c54f95bd8e52a90e0a7a76881ee67d4cc`](https://github.com/SirProdigle/pherc0306b-first-letters/tree/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc). `vendor.py` contains the exact `GlyphNet` class from [`train.py`](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/discovery/x_glyph/train.py), the `resample`, `zscore`, `a_from_z` and `a_from_a3` functions from [`common.py`](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/discovery/x_glyph/common.py), and `_tta` from [`evaluate.py`](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/discovery/x_glyph/evaluate.py). All six definitions were checked for exact AST equality with this source commit before inference. Imports and the `UM = 50.0` constant are supplied locally.

**From a source:** The source input has two channels: contrast-mapped ink signal and a validity mask. Windows are 128 × 128 pixels at a nominal 50 µm pitch, or 6.4 × 6.4 mm. The five transformations are identity, rotations through 90°, 180° and 270°, and horizontal reflection. The output is the arithmetic mean of their **logits**, not a mean of probabilities. The released screening wrapper accepts subwindows with at least 70% valid support. We retain that support threshold while using annotation-centred windows, not its sliding-window maximum. See [`common.py`](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/discovery/x_glyph/common.py), [`evaluate.py`](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/discovery/x_glyph/evaluate.py), and [`glyph_win.py`](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/discovery/fl/scripts/glyph_win.py).

**From a source:** The pinned [released `gA.pt`](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/discovery/x_glyph/models/gA.pt) has SHA256:

```
36ae0d6d9fdaac9d8681af75ec9be17e9317f7f77aa9422ea8ff8d09dd6f19ad
```

Its actual saved training metadata is `realpos="e343,h841,w1447"`, `n=300000`, `seed=0`, `legib=false`. These four whitelisted fields, the architecture and parameter count are recorded in [results.json](results.json). The adapter verifies the checksum before `torch.load(weights_only=True, map_location="cpu")` and uses strict state-dict loading. Other training arguments are neither printed nor written.

**My inference:** PHerc. 343 is therefore not a held-out positive for this checkpoint, regardless of the generic source-file header describing a training configuration that excludes it. The available release does not establish exclusion of PHerc. 306B from its unreleased negative training corpus. These outputs cannot support an independent-generalization claim. The source loader describes how real-positive sets are included in [`data.py`](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/discovery/x_glyph/data.py); dataset availability is described in [`MODELS_AND_DATA.md`](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/docs/MODELS_AND_DATA.md).

## Frozen inputs and explicit departures

**From a source:** The [frozen protocol](protocol.json) selects the midpoints of the two author-annotated PHerc. 306B rows and the centres of all 18 author-provided PHerc. 343 letter boxes. The source links, hashes, geometry, licences and mask limitations are in [the public-control inventory](../line_controls/README.md). No window was selected from a glyph or line-ranker output. No rejected window would have been replaced. The protocol SHA256 is:

```
dfe06c991a6e9d35802fc743a1d8c1d559adeda94696e4b44bb70e81d65c8bdf
```

**My inference / implementation choice:** Both prepared inputs are first standardized separately over their valid pixels, using the exact upstream `zscore` function. The exact upstream `a_from_z` then does the following:

1. Store zero outside the supplied mask, preserving the mask as a separate channel.
2. Reduce by a factor of three with area interpolation.
3. Form `clip((z - 0.9) / 1.8, 0, 1)` and remove cells with reduced support ≤ 0.5.
4. Resample to the nominal 50 µm grid, threshold the resampled mask at 0.6, and zero only masked values.

**My inference:** This is not a replay of the winning discovery pipeline. The 306B input is its released final model-product map, whereas the screening wrapper consumes a discovery-lane fused z map. The 343 input is recovered yellow display opacity: contrast adjustment and clipping have already removed information. Z-scoring that opacity does not reconstruct its native model signal. The 343 scores are consequently a second, explicitly different input-domain check. Do not compare raw scores between these two maps as a ranking of scrolls.

**From a source:** Native input pitch is declared as 8.64 µm in the [control manifests](../line_controls/inventory.json). The nominal output dimensions follow the upstream integer rounding and factor-three reduction. Window starts are frozen as `round(center_native * 8.64 / 50 - 64)` in each axis. Crops outside the prepared image or below 70% output-mask coverage are rejected; the adapter does not pad, recenter, or fill missing support as evidence. Original display/native coordinate rounding and the imperfect support masks remain limitations. These masks indicate public CT/model support for 306B and compatible display content for 343, not independently audited acquisition coverage.

## Observed result

**From a source (local measurement):** [results.json](results.json) records all 20 requested crops, input hashes, validity fractions, the allowed checkpoint metadata, software versions and adapter hashes. All 20 crops passed the frozen support and bounds requirements. Five-TTA inference used **2.45 process CPU seconds** and **1.24 seconds elapsed**, with two CPU threads; the 120 CPU-second ceiling was not approached. No GPU was used.

| Published control | Windows scored | Minimum valid fraction | Raw five-TTA mean logit |
|---|---:|---:|---|
| PHerc. 306B, two fixed row centres | 2 | 1.0000 | 7.1096; 3.6551 |
| PHerc. 343, all fixed letter-box centres | 18 | 0.7793 | Range −0.7265 to 5.7017 |

**My inference:** The result demonstrates that the pinned checkpoint loads and the specified CPU transformation and inference path run. The signs and magnitudes of these logits are not pass/fail decisions. There are no negative controls in this 20-window replay, neighbouring crops overlap, and 343 is training-overlapping. No AUC, recall, false-positive rate or threshold is reported. We did not run a second preprocessing variant or adjust selection after seeing these values.

## Reproduce

Install compatible `numpy`, `opencv-python-headless` and a CPU build of `torch`. The tested versions are recorded in `results.json`. The inference-time CPU budget uses Linux `resource` and `SIGXCPU`. Obtain the pinned source checkpoint, then prepare the hash-verified public controls using [line_controls/prepare.py](../line_controls/prepare.py).

From the repository root, with the two environment variables pointing to those local inputs:

```bash
python firstletters-colab/glyph_review/replay.py \
  --weights "$GLYPH_WEIGHTS" \
  --controls-root "$PUBLIC_CONTROLS" \
  --output glyph-replay-new.json
```

The program refuses mismatched source assets, checkpoint bytes or frozen protocol, and refuses to overwrite an existing output. It does not download, scan unread scrolls, write candidate locations, or select windows from scores. `vendor.py` exposes the isolated model and exact source preprocessing/TTA functions for a separately specified experiment; this command intentionally replays only the fixed public-control protocol.

## Licence and next use

**From a source:** The copied source excerpts are MIT-licensed; [LICENSE-UPSTREAM](LICENSE-UPSTREAM) preserves the required attribution and permission notice. The released weights, maps and images are [CC BY-NC 4.0](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/LICENSE-DATA), as stated in the [source README](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/README.md). Binary inputs and weights remain outside this repository. Original source authors are credited in the linked releases and MIT notice.

**My inference:** Before Claude Code uses glyph scores to allocate search effort, the input representation and its mask should be fixed on the actual inference pipeline, then checked on independently documented positives and negative/process controls. This adapter supplies the model-loading and CPU path. It does not supply that validation or justify making glyph scores the default production gate.
