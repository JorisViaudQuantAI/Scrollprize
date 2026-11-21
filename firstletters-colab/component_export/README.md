# Retain raw inference components for private review

`ink/infer_ink.py --export-components DIR` retains every **checkpoint × direction
× depth shift** as an original float32 probability map, with explicit support
masks. This is an opt-in extension to the existing inference command. It does
not change normalization, model calls, depth interpolation, tiling, blending,
depth/model averaging, or legacy image/float outputs.

**Measured software evidence:** 15 synthetic contract tests pass using the real
tiler, depth interpolation, OpenCV and output writers, with a deterministic
NumPy predictor instead of an ink model. They include byte comparison against
the pre-feature inference source, byte comparison with export enabled/disabled,
and identical predictor-call counts. A fixed clock makes timing fields
comparable in those tests. This is not a real-scroll accuracy or runtime result;
no prize or reading claim is made.

**Measured real-data regression:** A separate frozen Hecate smoke experiment
now runs the real historical and exporter-enabled CLIs on a tiny published
PHerc. 306B CT input. Both execute six forwards; all ten legacy PNG/NPY outputs
are byte-identical, nonruntime metadata match, and six exported components
verify and reconstruct the legacy direction averages. The source crop was
already clipped at 200, so this is not a Hecate quality benchmark. See
[the protocol, measured results and limitations](PUBLIC_SMOKE.md). Required
mesh/normal/frame and acquisition support remain unknown; the downstream
depth consumer has not been run.

## Run it once, keep every pass

Example paths below are placeholders, not manuscript locations. Supply the
actual rendered-stack pitch through its metadata or `--input-um` /
`--input-um-z`; a native CT voxel size is not automatically a rendered layer
step.

```bash
python firstletters-colab/ink/infer_ink.py /private/published-control/layers \
  -c /private/model.pth \
  -o /private/new-run/ink.png \
  --z-shifts=-1,0,1 --direction both \
  --export-components /private/new-run/components \
  --component-provenance /private/render-geometry.json
```

The provenance option is optional. Missing geometry is recorded as `null`, not
invented. The component directory must be **new and outside every Git
worktree**, including linked worktrees and symlink destinations. When export is
enabled, the legacy `--output` destination must also be outside Git. Generated
unread-scroll maps and location-bearing receipts remain private under
[the collaboration rules](../../collab/README.md).

The existing `--save-float` still saves the direction's final aggregate as
float16. The existing `--save-each` still saves display-scaled PNGs. Neither
changes meaning when the new option is enabled. Components are kept before
those aggregates and display transformations, including for a single-model,
single-depth run.

## Files and support semantics

`receipt.json` is written last, after all components and all legacy outputs
succeed. It indexes `component-0000/`, `component-0001/`, and so on; **use the
receipt's model/direction/shift fields rather than assuming a directory order**.
Each component contains:

| File | Meaning |
| --- | --- |
| `probability.npy` | Float32 per-pass probabilities after the existing `[0,1]` clip and any existing output resize. NaN where `prediction_support` is false. Finite computed scores from an incomplete depth window remain available and are identified as such. |
| `prediction_support.npy` | Boolean mask from **model-grid blending weight sum > 1e-6**. With output resizing, every contributing model-grid pixel must pass that test. |
| `complete_depth_support.npy` | `prediction_support AND full_depth_context`; this still does not certify XY context, acquisition validity or papyrus tissue. |
| `component.json` | Checkpoint SHA-256, model/shift/direction, exact grid and sampling metadata, context limits, declarations, and hashes of all three arrays. |

“Raw” here means an original probability component, not logits. Existing
within-pass tile blending and optional flip TTA still apply. No per-component
contrast scaling, threshold, depth maximum or new detector inference is added.

**Prediction support is not acquisition validity.** A tile with some nonzero CT
can produce scores where individual input pixels are missing. An input mask
used for tile selection/blending is not automatically an acquisition mask or a
tissue mask. The exporter makes no such inference from nonzero CT, finite
probabilities, or model confidence.

The output grid is exactly the current source ROI, after any `--xy-bin`, with
shape `[y1-y0, x1-x0]`. The receipt also retains the original pre-bin shape/pitch,
the binned shape, the model grid/ROI, resampling settings and output dimensions.
Conservative support transport uses the same OpenCV linear resize dimensions as
the probability map. A target receiving any positive interpolation contribution
from an unsupported model-grid pixel is excluded, even if its numerical score
is finite. This deliberately excludes more pixels than nearest-neighbor mask
resizing near support boundaries.

## Physical depth and context

Every component records the actual source-layer coordinate and contributing
source layers/weights for each model plane, in its applied forward or reversed
order. Its physical model-plane step is

```text
effective_depth_pitch_um = source_layers_per_model_plane * source_layer_pitch_um
physical_shift_um = shift_model_planes * effective_depth_pitch_um
```

These quantities are not automatically the checkpoint's nominal pitch. A run
that skips resampling uses the rendered stack's step. Positive shift means
increasing source Z index before direction reversal; its physical relation to
the papyrus normal remains unknown unless supplied by the renderer's geometry.
When the CLI had to assume a native pitch, physical-pitch/shift fields remain
`null` and the numerical assumptions used by inference are recorded separately.

`full_depth_context` requires every model plane's source interpolation weights
to sum to one within absolute tolerance `1e-6` and zero relative tolerance. The
individual masses are retained. It detects partial planes as well as wholly
empty ones: a synthetic 41-layer input, 16-plane model, center 19.5 and shift
13.75 has a last-plane mass of 0.25 even though no plane is wholly empty.
Legacy inference is preserved; the new depth-complete mask rejects that
component's context without erasing its actually computed scores.

Zero padding of model tiles when the full XY image is smaller than a patch is
recorded separately, including the padded bottom/right extent. Resampling's
`BORDER_REPLICATE` convention is recorded too. **Neither mask certifies a fully
valid model receptive field or correct sheet geometry.**

The optional geometry JSON accepts only these declaration fields:

```json
{
  "mesh_sha256": null,
  "normal_sha256": null,
  "normal_convention": null,
  "plane_order": null,
  "frame": null,
  "source_content_sha256": null
}
```

Replace a null only with information from the actual input producer. Hashes
must be lowercase SHA-256 values; normal/plane-order descriptions must be
nonempty. These are declarations, not independently checked mesh or normal
identities. The actual checkpoint is hashed before model loading and checked
for ordinary concurrent file changes through completion. The exporter does
not hash an entire remote CT volume, infer a normal from a filename, or turn
unknown geometry into a valid depth-comparison manifest.

## Handoff to depth and row review

The intended use is to keep these components during an already planned
inference run, then supply appropriate float maps to
[depth_screening](../depth_screening/README.md) or row review. This removes the
need to rerun the model solely to recover individual depths. The remaining gate
is **a real producer-to-consumer handoff with correct acquisition/geometry
support**, followed by a public-control demonstration of the intended review.

Before treating a component's mask as comparison validity:

1. Wait for `receipt.json`; verify its component-metadata hashes and each
   component's array hashes. A partially completed directory is not a completed
   run, even if some components were atomically published.
2. Require `full_depth_context == true`. Review the XY padding/border metadata;
   conservatively reject a component with unresolved XY tile padding.
3. Obtain an **external boolean acquisition/context support mask** on this exact
   output grid, produced from the actual sampled input and its interpolation /
   model context. It must account for missing inputs, not merely nonzero
   intensity or a surface-outline mask. Keep any required tissue/sheet mask
   distinct and intersect it explicitly.
4. Use `complete_depth_support & external_acquisition_context_support`, plus any
   separately justified tissue/sheet mask, as comparison validity. Never rename
   `prediction_support.npy` to “acquisition valid”.
5. For three-depth comparison, retain one checkpoint and direction, three
   distinct offsets including zero, the same mesh and normal, exact output grid,
   frame and preprocessing/inference settings. Populate its required provenance
   from actual producer records. Unknown mesh/normal/frame fields should keep
   that comparison blocked. `effective_depth_pitch_um` is the correct physical
   step for its `offset_planes` field.

Minimal mask gate, after verifying the receipt and payload hashes:

```python
if not component["full_depth_context"]:
    raise ValueError("Incomplete depth context")
if component["sampling"]["xy_tile_padding_present"]:
    raise ValueError("Resolve padded XY context before comparison")
if not component["sampling"]["physical_pitch_known"]:
    raise ValueError("Physical sampling pitch is unknown")
if acquisition_context.dtype != np.bool_ or acquisition_context.shape != scores.shape:
    raise ValueError("A producer-supplied boolean acquisition/context mask is required")
comparison_valid = complete_depth_support & acquisition_context
```

This gate does not by itself verify anatomical correspondence or allow a
three-depth maximum to inherit a single-depth detection threshold.

## Cost, completion and checks

There are no additional CT reads or model calls for export. There is one
sequential checkpoint hash read per loaded checkpoint and the additional disk
writes/hash reads. Uncompressed array payload is **6 bytes × H × W per
component**: four for the probability and one for each mask. A 1024² ROI with
one model, two directions and three shifts adds 36 MiB, plus small headers and
metadata. Additional working arrays are limited to one component at a time;
resampled support also needs temporary arrays on the model/output grids. The
existing inference accumulators are unchanged. No real T4/CPU runtime overhead
benchmark is claimed yet.

Each component is written in a private staging directory, its arrays are
hashed and metadata written, then the directory is atomically renamed into
place. The run receipt is atomically published last. A failure leaves no run
completion receipt; a failed component is not published. Finished component
payloads are auditable if a later pass fails. Use a new directory for a retry;
the writer does not overwrite an existing export or silently resume it.

Run tests in the existing inference environment:

```bash
python -m unittest discover -s firstletters-colab/component_export -p 'test_*.py' -v
```

The historical-source regression uses Git blob
`891013ad0ae4f51dc5d1657f23ade30009ef65d1`, the pre-feature `infer_ink.py`. It is
skipped if a source archive omits that history. Inference contract tests require
the inference environment's existing OpenCV and Torch dependencies; they use
real Torch runtime setup but no trained weights. Exporter-only tests need
NumPy, with OpenCV required for the resampling test. A skipped runtime test is
not evidence of byte equivalence.
