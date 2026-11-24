# Studio oblique sampling stress — Q33

**Executed results:** [RESULTS.md](RESULTS.md) reports all 221 CPU forwards,
the native reconstruction and independent audits. Nine of twelve tilted
conditions lose more than 10 points of recall. The x +5 degree pooled AP gain
does not translate to a comparable improvement on mixed cores.

This module tests how the released Studio ink detector responds when the **actual 3D sampling geometry** of a known labeled surface is tilted. It uses a public PHerc0841 author holdout, the original native CT acquisition, and a fixed checkpoint. Predictions are mapped back to the same physical surface points before comparison.

The experiment extends the preceding XY scale stress test to a different error: rotation of the complete curved sampling patch in 3D. It does not establish an improvement, a usable tilt correction, or a new reading. This README describes the protocol, the geometry-only preflight amendment below, and reproduction interfaces; it makes no detector-result claim.

## Fixed experiment

The source is `841-expanded-layer-03-approved-r000001`, using the same pool of 53 labeled central cores as `studio_scale_stress`. Select a small panel from its labels and spatial coordinates before inspecting any tilted predictions:

- **11 mixed cores:** at least 512 known pixels and ink fraction among known pixels in `[0.1, 0.9]`.
- **3 pure-ink controls:** at least 512 known pixels, all known labels ink.
- **3 pure-background controls:** at least 512 known pixels, all known labels background.

All 11 qualifying mixed cores are retained. Within each pure class, the frozen maximin rule starts at the lexicographically smallest `(y,x,author_index)` and then maximizes the minimum squared XY distance to previously selected members of that class, breaking ties lexicographically. Final order is by author index. Unknown pixels never become background. The selection receipt retains the chosen cores, their label counts and the selection settings. This label-conditioned panel is a development control; it does not estimate performance over every possible surface region.

For each selected core, use a 128×128 input with 16 depth planes. The 32×32 central core defines the original physical evaluation locations. There are **13 arms**:

| Arm group | Rotation axis | Angles |
|---|---|---|
| Identity | None | 0° |
| Local chart x | Positive local x tangent at the patch center | −20°, −10°, −5°, +5°, +10°, +20° |
| Local chart y | Positive local y tangent at the patch center | −20°, −10°, −5°, +5°, +10°, +20° |

The axes come from the actual transformed mesh, not the scanner's global coordinate axes. The rotation follows the right-hand rule in native XYZ space about the surface point at chart index `(63.5, 63.5)` within the patch. It rotates **surface anchors and their normals together**. The pivot is the reference surface center, not the midpoint of the asymmetric `[-8,…,7]` depth stack. The two local chart tangents need not be mutually orthogonal on a distorted chart.

Sample each resulting coordinate set directly from the native CT. Do not simulate this experiment by rotating or interpolating an already rendered 16-plane slab: the tilted patch can require source voxels outside that slab.

## Preserve physical label correspondence

A prediction pixel after 3D rotation generally refers to a different physical point. `geometry.py` therefore solves an inverse mapping into the rotated **curved** sampling tube for each original core point. It uses the same bilinear mesh and signed `cross(drow,dcol)` normal convention as the baseline; a flat-plane cosine correction is insufficient for a curved surface.

The inverse must have valid mesh support, acceptable physical residual and conditioning, retained local orientation, and coordinates inside both the usable model output and the 16-plane depth interval. Missing source chunks and unsupported samples remain explicitly unknown. Scoring must use the frozen common support across arms and original categorical labels, with no label interpolation or arm-specific favorable subset.

Before interpreting any detector comparison, the zero-angle reconstruction must reproduce the fixed author inputs under the declared numerical tolerance. Keep the checkpoint, layer order, CT interpolation, uint8 conversion, normalization, model window and score readback conventions unchanged across arms. Preserve the control receipts and any failed gate; do not tune thresholds, centers, angles, or sample selection to recover a preferred score.

### Geometry policy amendment before predictions

The initial `initial95` preflight required at least 95% of known labeled pixels to remain supported in every core, as well as at least 95% globally. **Four cores failed the per-core gate, retaining approximately 91.4–93.8%.** This was a geometry-only preflight: no detector forward pass was run. The original failed-gate receipt is preserved in the experiment evidence.

The subsequent `retained90` policy changes only the minimum per-core retention:

| Requirement | Final `retained90` policy |
|---|---|
| Global retained known pixels | At least 95% |
| Retained known pixels in each core | At least 90% |
| Maximum physical inverse residual | 0.01 native voxel, unchanged |
| Selected controls and perturbations | Same 17 cores and 13 arms; no replacement |
| Scoring support | Same fixed intersection of validity across all arms, determined before model loading |

This coverage decision is **data-informed by the observed geometry, before any detector predictions**. It is not a fully prospective choice made before geometry inspection. No detector score or model output motivated it. The original categorical labels, common-mask rule, inverse solver and residual tolerance remain unchanged; pixels excluded by the common mask stay excluded for every arm. A failure of the amended gate still stops preparation and inference.

## Pinned model and data

| Component | Identity |
|---|---|
| Author source | `priyankaranke/vesuvius-studio-multiscroll-cpt`, commit `bf66f753176792a966bca17822629fb8821f20c8` |
| Author dataset | `paranke/vesuvius-814-841-9b-studio-training-data`, revision `ad919d288ede1c4daf92725288bb6fe29c558591` |
| Holdout | `841-expanded-layer-03-approved-r000001` |
| Released checkpoint | `studio-softloss-20261023-ema.safetensors` |
| Checkpoint SHA-256 | `96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90` |
| Native CT | PHerc0841, volume `20250821151531`, 9.366 µm, level 0 |
| Original label pitch | 9.36 µm; pixel centers `(index+0.5) × pitch` |
| Depth sampling | `[-8,…,7] × 9.366 µm`, normal sign +1, no runtime reversal |

Public sources:

- [Pinned author code](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/tree/bf66f753176792a966bca17822629fb8821f20c8)
- [Pinned data release](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/tree/ad919d288ede1c4daf92725288bb6fe29c558591)
- [Holdout selection and geometry record](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/ad919d288ede1c4daf92725288bb6fe29c558591/donor-studio-resunet-scratch-20260926/prepared/841-expanded-layer-03-approved-r000001/selection.json)
- [Completed author render record](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/ad919d288ede1c4daf92725288bb6fe29c558591/donor-studio-resunet-scratch-20260926/prepared/841-expanded-layer-03-approved-r000001/render.json)
- [Checkpoint release](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/tag/studio-softloss-20261023)

The original native array is:

```text
https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0841/volumes/20250821151531-9.366um-1.2m-113keV-masked.zarr/0/
```

Its recorded shape is `[19400,7703,7703]` in ZYX order, with uncompressed uint8 C-order chunks `[128,128,128]`. `native.py` handles bounded source acquisition and explicit observation support. Source array metadata must be validated before interpreting its values.

The original mesh uses a 9.612 µm geometry grid. The pinned recipe's affine matrix maps its stored XYZ coordinates directly to native level-0 XYZ. **Do not apply the separate HR-to-native transform again:** its factor-four relationship to the stored geometry is already included in the recipe matrix.

## Bounded geometry acquisition

`assets.acquire_geometry(cache_dir)` returns:

```python
mesh_dir, recipe_dict, provenance_dict = acquire_geometry(cache_dir)
```

It verifies the pinned selection, render and archive-index JSON, then obtains only the first **8 MiB** of the 1.377 GB geometry archive when the mesh is not already cached. It requires HTTP 206, exact range and body length, and a fixed prefix hash. Five whitelisted regular members—four mesh files plus the transform receipt—are read and checked against their independent source-index hashes. It never performs unrestricted archive extraction. The whole archive hash is recorded as an expected identity, not claimed to have been verified from a partial transfer.

Verified cached assets can be reused. Returned provenance contains public source URLs, hashes and scientific settings; raw author metadata should stay in the external cache. CT data, model weights and prepared arrays also belong outside git.

## Reproduction command templates

Run from the Scrollprize repository root. Replace every `/path/to/...` value with an explicit local path. `--source-repo` is the author source checkout at the commit above, and `--checkpoint` is the already downloaded, hash-verified soft-loss checkpoint. **All seven path flags are mandatory for every command**, including stages that do not use every asset. The `prepare` command explicitly selects the amended `retained90` policy; the runner's default remains `initial95`.

Prepare the frozen selection, exact source geometry, native CT samples, identity checks and inverse readback:

```bash
python firstletters-colab/studio_oblique_stress/run.py prepare \
  --source-repo /path/to/pinned-studio-source \
  --cache /path/to/author-data-cache \
  --geometry-cache /path/to/geometry-cache \
  --native-cache /path/to/native-ct-cache \
  --checkpoint /path/to/studio-softloss-20261023-ema.safetensors \
  --prepared /path/to/q33-prepared \
  --output /path/to/q33-results \
  --geometry-policy retained90
```

Run the fixed checkpoint on the prepared 13-arm inputs:

```bash
python firstletters-colab/studio_oblique_stress/run.py infer \
  --source-repo /path/to/pinned-studio-source \
  --cache /path/to/author-data-cache \
  --geometry-cache /path/to/geometry-cache \
  --native-cache /path/to/native-ct-cache \
  --checkpoint /path/to/studio-softloss-20261023-ema.safetensors \
  --prepared /path/to/q33-prepared \
  --output /path/to/q33-results
```

Summarize retained predictions on the frozen common physical labels:

```bash
python firstletters-colab/studio_oblique_stress/run.py summarize \
  --source-repo /path/to/pinned-studio-source \
  --cache /path/to/author-data-cache \
  --geometry-cache /path/to/geometry-cache \
  --native-cache /path/to/native-ct-cache \
  --checkpoint /path/to/studio-softloss-20261023-ema.safetensors \
  --prepared /path/to/q33-prepared \
  --output /path/to/q33-results
```

Preparation should record source hashes, selected-core label counts, geometry and CT support, baseline reproduction, and inverse-mapping diagnostics. Inference should retain the checkpoint identity and per-arm outputs. Summarization should preserve all arms and distinguish ranking on mixed labels from responses on pure controls. ROC-AUC is undefined for a single-class core; an all-ink AP is trivial and does not measure ink/background discrimination. Use the specified class-response measurements for pure controls.

## Interpretation limits

- This is one labeled surface from one author holdout. PHerc0841 also occurs elsewhere in Studio's training data, and the author inspected this holdout during model development. It is not independent new-scroll validation.
- Labels are sparse author annotations; they are not an exhaustive optical ink mask. A pure-ink or pure-background core means its **known** labels have that class, not that every pixel was annotated.
- The perturbation deliberately changes the sampling geometry and surrounding tissue. It measures sensitivity of this checkpoint and rendering path; it does not validate arbitrary surface corrections or establish same-sheet identity.
- The inverse is a local geometric solve. A small residual and a positive local Jacobian do not prove global tube injectivity or eliminate all neighboring-sheet ambiguities.
- The final per-core retention threshold was amended after the initial geometry-only failure. Report this adaptation and the preserved failure alongside the final experiment; do not describe all geometry gates as selected before observing the data.
- Patches and pixels are spatially dependent. Do not treat them as independent scrolls or construct an independent-pixel significance claim.
- A favorable angle on this panel would not authorize selecting angles on unread candidates. Any such use requires a separate prospective validation.

See [research.md](research.md) for the primary sampling references and the distinction between whole-patch rotation and normal-ray perturbation. [RESULTS.md](RESULTS.md) records the completed experiment; the executable protocol and numerical receipts specify the exact run.

## Replay the published scores without CT or weights

The small archive contains aligned probabilities and original/masked labels.
The independent checker imports neither the producer nor its metric helper:

```bash
python firstletters-colab/studio_oblique_stress/audit_results.py \
  --output firstletters-colab/studio_oblique_stress/evidence \
  --protocol firstletters-colab/studio_oblique_stress/evidence/protocol.json \
  --historical firstletters-colab/studio_scale_stress/evidence/scored_arrays.npz \
  --report /tmp/q33-archive-review.json
```

Use a fresh report filename. This mode verifies selection, masks, labels,
historical identity, metrics and gates. To additionally verify the full local
prediction readback and prepared-source identity, supply `--prepared`,
`--source-cache` and `--checkpoint` with the paths used for the complete run.
The published full audit and portable audit are both in `evidence/`.
