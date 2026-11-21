# Explicit input preparation for the released Studio detector

## Native first: answer to Claude's sampling question

**From a source.** The inspected release revision is
[`bf66f753176792a966bca17822629fb8821f20c8`](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/tree/bf66f753176792a966bca17822629fb8821f20c8).
Its [inference instructions](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/README.md#run-inference-with-the-released-weights)
require raw `uint8 [16,H,W]` CT in model depth order. For the supplied surface they say
to render planes `00–30` and stack `23` down to `8`.
[The actual inference code](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/hf/inference.py)
does **no depth or XY resampling**. It converts the raw input to float32, clamps to
`[0,200]`, and divides by **255**, not 200. Do not normalize the adapter's output a
second time. The wrapper's `--amp` choices are `bf16` and `fp32`; the documented CPU
path is `--device cpu --batch-size 1 --amp fp32`.

**From a source.** The source [training record](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/donor-studio-resunet-scratch-20260926/data.json)
and [rendering code](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/donor-studio-resunet-scratch-20260926/render.py)
include native low-resolution donor recipes at 9.362, 9.366 and 8.640 µm through
depth. Some use a 9.36 µm XY pitch with 8.64 µm depth pitch; some historical source
rows are explicitly resampled high-resolution data. The
[continued-training description](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/README.md)
also lists native 9.362 µm Reader rows. The actual rendered XY and depth spacing
must therefore be recorded separately; "all training data were isotropic native
9.36 µm" would be wrong.

**My inference.** Native 9.36 µm renders are an appropriate first input. I found
no release instruction to resample them to 9.6 µm or to one donor's voxel size.
This does not prove optimality on another scroll. Do not introduce unmeasured
interpolation before the public-positive feasibility check. Input validity and
surface placement remain separate from sampling pitch.

## What this helper does

`prepare.py` selects planes from an already rendered `.npy` stack. It needs NumPy;
it does not need PyTorch, a GPU, CT volumes, labels, or a mesh. It writes a new
`uint8 [16,H,W]` `.npy` and a `.npy.json` provenance sidecar. It changes no voxel
values, resamples nothing, and has no implicit intensity conversion.

Provide these metadata explicitly; none has a guessed default:

- The **integer mesh-plane index**, from the renderer or an independently checked
  geometry record. The stack length alone does not establish this index.
- Actual rendered **XY sample spacing**, in `[y,x]` order and micrometres.
- Actual positive **depth sample spacing**, in micrometres. This is not necessarily
  the source CT voxel size if the renderer used a non-unit slice step.
- `--index-direction 1` when increasing stored plane index follows the declared
  mesh normal; `-1` when it opposes that normal. This is a geometric convention,
  not a claim about which papyrus face carries writing.
- `--order release-descending` or `--order same-band-reversed`.

An optional `--coverage-mask` must be a **boolean `[D,H,W]` acquisition-coverage
array** matching the entire input. All selected samples must be true. A float,
integer mask, 2D silhouette, or ink mask is rejected; there is no thresholding or
broadcasting. Uncovered samples require a re-render or a complete spatial crop.
If no mask is supplied, the sidecar explicitly records **coverage unverified**.
The array alone cannot establish that every source chunk was fetched. A mask
created by setting all entries true does not establish that fact either; its
meaning must come from the downloader/renderer or another coverage audit.

Out-of-bounds bands and entirely zero required planes stop preparation. This is
a conservative guard: an entirely zero plane could also be genuine air, so the
error is not proof of a missing download. Individual zero values are preserved.
The sidecar counts pixels that are zero through all 16 selected planes; it does
not interpret them as blank papyrus. The release itself zeroes predictions at
such pixels. Missing planes must never be inserted as zeros to satisfy the shape.

The helper does not detect a nonzero plane duplicated to hide a missing plane,
an incorrect caller-supplied mesh position, or an incorrect coverage assertion.
Those require renderer provenance. Existing outputs and sidecars are never
overwritten. Complete input, output, optional coverage-mask, and adapter-file
SHA-256 hashes make the preparation reviewable. Input names and paths are omitted
from the sidecar. Keep generated artifacts from unread scrolls private: do not
commit candidate images, locations, or manifests automatically.

## Exact plane convention and the half-plane offset

For an explicitly known mesh index `m`, the supplied-surface release convention
selects source indices `m+8, m+7, ..., m−7`. Therefore:

| Render metadata | `release-descending` | `same-band-reversed` |
| --- | --- | --- |
| 31 planes; mesh index **known to be 15** | 23, 22, ..., 8 | 8, 9, ..., 23 |
| 41 planes; mesh index **known to be 20** | 28, 27, ..., 13 | 13, 14, ..., 28 |

**My inference, by arithmetic.** The mean sample-center position is `m+0.5`, not
`m`. With depth spacing 9.36 µm and increasing index along the normal, offsets run
from +74.88 to −65.52 µm, midpoint +4.68 µm. The first-to-last sample-center span is
140.4 µm. These describe sampling, not an ink depth or a physical sheet thickness.
The helper reports these quantities for the supplied spacing and normal direction;
it does not interpolate by half a voxel to hide the asymmetry.

Reversing the **selected 16 planes** preserves the same physical band. Reversing
the **entire input stack**, updating its mesh index to `D−1−m` and index direction,
then extracting the release band selects original indices `m−8, ..., m+7`.
Those bands differ by one source plane at each end, and their midpoints differ
by one depth step. A whole-stack reversal is therefore an order **and** placement
change. Use `same-band-reversed` for the controlled order comparison. To test the
other band, prepare it deliberately and label it as a separate placement trial.
Neither operation creates a new render on the opposite sheet interface.

**From a source.** The release's own training geometry is not one universal
depth-sign rule: [campaign rendering](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/donor-studio-resunet-scratch-20260926/render.py)
renders 17 planes, sometimes reverses them, then keeps the first 16; recorded
model offsets include both `−8..+7` and `+8..−7`. A released 28-plane donor source
has its own explicitly recorded `6:22` selection. This helper does **not** invent
its mesh position or treat that donor selection as equivalent to the documented
31-plane supplied-surface convention. A verified, already prepared 16-plane
release tile should go directly to upstream inference with its own provenance.

## Use on a public control

This example assumes independently confirmed metadata for a 41-plane public
control. Replace every metadata argument with that control's actual record.
The filenames are examples and no control data are bundled.

```bash
python firstletters-colab/studio_inputs/prepare.py \
  --input public_control_41.npy \
  --output public_control_studio16.npy \
  --mesh-plane-index 20 \
  --xy-spacing-um 9.36 9.36 \
  --depth-spacing-um 9.36 \
  --index-direction 1 \
  --order release-descending \
  --coverage-mask public_control_coverage.npy
```

Run the same command with a new output and `--order same-band-reversed` for an
order-only comparison. If a coverage record is unavailable, omitting the mask
is allowed for a **technical feasibility check with unverified acquisition
coverage**, and is marked as such. It does not certify a search negative.

The adapter records native intensities. Supply the resulting file to the pinned
release's `infer_release.py` or `infer.sh`; let upstream apply `clip200/div255`.
Its README recommends stride 48 for new maps to reduce tiling artifacts, while
the reproduced submission uses stride 64. Record the choice, checkpoint hash,
runtime, peak memory, and public-control result alongside the output.

For an **already prepared** public release fixture, use the separate read-only
format verifier:

```bash
python firstletters-colab/studio_inputs/verify_prepared.py --input released_planes16.npy
```

It prints a hash and format/coverage record to standard output; it does not
reorder or rewrite the file. Its mesh position, physical offsets, and depth
order remain explicitly unverified. A `[16,H,W]` shape alone does not establish
any of them. The fixture's upstream preparation record is still required to
attribute its geometry or model order. Its optional coverage mask has the same
strict boolean, full-shape, all-selected-samples-covered contract.

## Tests

```bash
python -m unittest discover -s firstletters-colab/studio_inputs -p 'test_*.py' -v
```

The tests use synthetic plane-coded values. They verify 31-to-41 translation,
same-band versus whole-stack reversal, physical offsets and anisotropic spacing,
bounds and dtype failures, zero-plane and coverage guards, file hashes, unchanged
voxel values, and no-overwrite behavior. They do not establish ink accuracy.
