# Producer support and a CPU Studio preflight

This adapter closes one specific integration gap: a renderer can retain source
availability at every sampled depth, then check the full input context needed
by the next model **before spending GPU time**. It uses the existing Studio
input reader, plane selector, tile starts, Hann weights and empty-window rule.
It loads no weights and performs no model forward.

**Measured software result:** the 21 offline tests exercise generated arrays
only, including the real Studio scheduling helpers and the existing producer
declaration validator. They are not an end-to-end ink-control result. No new CT,
unread render or model output was acquired for this work.

## Minimal producer handoff from Gemini's renderer

Keep these artifacts in a new private directory outside Git:

| Artifact or NPZ key | Shape/type | Exact meaning |
| --- | --- | --- |
| `render.npy` | `uint8 [L,H,W]` | The actual encoded stack that Studio will read, without another intensity conversion. |
| `source_available_lhw` | `bool [L,H,W]` | Every strictly positive-weight interpolation contributor exists, is in bounds and has a finite encoded source value. |
| `mesh_valid_yx` | `bool [H,W]` | Coordinate and normal interpolation is supported by valid mesh vertices under the renderer's recorded rule. |
| `nonzero_intensity_yx` | optional `bool [H,W]` | An intensity diagnostic only; it is neither of the preceding masks. |

The support arrays can be stored together as `render-support.npz`. These key
names match the optional `return_support=True` renderer API and
`--out-support-npz` artifact on the reviewed Gemini lane. A shape match alone
does not authenticate a mask: bind the stack and masks to the same renderer
receipt and file hashes.

**Source availability does not automatically certify acquisition/FOV support.**
A present chunk may contain synthetic masked-out values. Zero can mean a valid
encoded value, a masked value or unknown acquisition; intensity cannot resolve
that distinction. Missing chunks are conservatively unavailable, including
when the Zarr numerical fill convention permits reading them as zero. Require
an independently supplied acquisition mask or an explicit, justified producer
derivation before calling the result `acquisition-and-model-context` support.
Papyrus/tissue membership and m7/coherence estimates remain separate.

## Use the actual Studio schedule before GPU execution

From the repository root, on an already retained local render:

```bash
python firstletters-colab/producer_support/prepare_studio.py \
  --input /private/render/render.npy \
  --support-npz /private/render/render-support.npz \
  --mesh-index 15 --order desc --shift 0 --stride 48 \
  --spacing-um 9.36 9.36 \
  --output-dir /private/render/studio-context-preflight
```

The paths, central index and sampling above are examples, not manuscript
locations or defaults to assume. Use the actual rendered chart pitch; native
CT voxel size does not by itself establish chart scale. Omit `--spacing-um`
when the chart scale is unknown: areas are then `null`.

The pure entry point is
`prepare_studio_context(render, source_available_lhw, mesh_valid_yx,
mesh_index=..., order=..., shift=0, stride=48)`, where `render` is the existing
`studio_t4.py` `Render` object. It returns a `SupportResult` and a plan. The CLI
uses this function, checks input hashes and writes these private outputs:

| Output | Meaning |
| --- | --- |
| `source_context_support.npy` | Conservative source and mesh context for every positively weighted planned tile contributor. |
| `planned_contribution_support.npy` | At least one planned positive-weight contribution; **not** executed prediction support. |
| `source_available_all_required.npy` | Availability at the output pixel over all selected planes, before XY context. |
| `mesh_valid.npy` | The separate mesh mask on the output grid. |
| `tile_context_complete.npy` | One boolean per planned tile, in the stored schedule order. |
| `tile_schedule.json` | Every input/output box and weight hash, with full-input support verdicts. |
| `preflight.json` | Written last: hashes, selected planes, Studio code identity, planned settings, counts and chart areas. |

The receipt is explicitly `cpu_preflight_only`, with `model_forward_calls: 0`,
`prediction_completed: false` and `handoff_ready: false`. It does not create a
`context.json` acquisition declaration or an `inference-components.v1` receipt.
Studio's historical validity mask is an intensity-and-coverage proxy; its pixel
count is reported separately as `legacy_intensity_proxy_pixels`.

For each planned retained output `(y,x)`, Studio reads the full 128×128 input
box `[y-16,x-16,y+112,x+112]` and keeps the central 96×96 output. The adapter
uses `plane_indices`, `window_starts`, `hann96`, and the exact central-empty
skip predicate. Record the actual `source_layers`: at mesh index 15, `desc`
selects 23 through 8, while `asc` selects 7 through 22. These are not identical
physical source-plane sets in reverse order.

**The required real render halo is larger than the nominal 16-pixel crop
margin in some schedules.** A reflect-padded edge tile can contribute positive
weight across its whole retained 96-pixel extent. Every pixel it contributes
fails the conservative all-contributors context rule, even where a second,
fully supported tile also contributes. In the generated 256×256 stride-48
fixture, all 65,536 chart pixels have planned contribution coverage, but only
the central 64×64 have full real tile context. This is a schedule effect, not
a measured unread-scroll result. For a planned 20 mm review window, render
additional surrounding context and calculate the supported interior from the
actual schedule. A 16-pixel halo is not a guarantee; a halo of at least one
retained tile is a practical starting point that still needs this check.

## Reusable support calculation

`tile_context_support(source_available_lhw, mesh_valid_yx, tiles,
output_roi_yxyx=None)` accepts `TileContribution` records containing:

```python
TileContribution(
    input_box_yxyx=(...),   # full model input in one common raster
    output_box_yxyx=(...),  # retained output in that same raster
    weights_yx=...,         # actual nonnegative real blend weights
)
```

Every full input tile must have all required source planes and mesh pixels.
An out-of-bounds input fails, even when production inference reflects or pads
it. An output is supported only if it receives a positive weight and **every**
positive-weight contributor has complete context. Zero blend weights do not
contaminate a neighboring valid contributor. The full input tile is a
conservative context bound; no smaller architectural receptive field is
claimed. For an overwrite operation, provide only contributions that survive
in the final result, not every overwritten intermediate call.

`select_depth_support` accepts the exporter's exact
`model_plane_source_contributors` lists (`source_layer`, `weight`). It rejects
negative/nonfinite weights and makes a partially supported depth plane invalid
everywhere. Its unit-mass tolerance is the exporter's absolute `1e-6`.
Zero-weight contributors are ignored. It does not handle XY filtering or
resampling. The present generic tile adapter requires one common unresampled
XY raster. Transport acquisition support through every real blur/interpolation
contributor before using it with a resampled Hecate pass; do not resize this
mask with nearest neighbor or assume the Studio mask fits another model.

After a completed inference run, `review_support(context, prediction, score)`
intersects the context mask with the independently retained boolean prediction
support and rejects a nonfinite score marked valid. Confirm the executed tile
schedule, input hashes and inference settings agree with the preflight. The
preflight does not itself establish that correspondence.

## Compatibility with existing consumers

For `line_review/review.py`, use a completed float map with the resulting
boolean review mask, actual `--spacing-um DY DX`, and `--map-kind model_score`.
Missing scores remain unknown. `line_controls/prepare.py` is a pinned published
control extractor, not a general arbitrary-scroll importer.

`row_context/diagnostic.py` accepts the pinned published PHerc. 306B reference
frame only. Its `pair.json` contract includes equal `run_signature` dictionaries
and actual `tile_origin_yx_px` values on one physical field. Do not relabel an
unread-winding map as that control. Compare origin pairs on their common
independently supported pixels, with the same precision, depth planes,
normalization and blending.

The separate `write_handoff_declaration` function can write the **existing**
`scrollprize.producer-context-support.v1` schema and an
`acquisition_context.npy`. It requires actual executed tile records,
`acquisition_semantics_confirmed: true` plus a nonempty
`acquisition_definition`, explicit physical sign and the six exact geometry
fields accepted by `component_export`:

```text
mesh_sha256, normal_sha256, normal_convention, plane_order,
frame, source_content_sha256
```

The source, mesh and normal hashes must match actual supplied local files.
Use the same hash convention in the inference geometry declaration: Gemini's
raw array `stack_sha256` is not the `.npy` file SHA-256 used by Studio's
`input.sha256`. The writer reuses `prepare_handoff.validate_frame` and
`validate_producer`, verifies source assets again before completing, and
publishes its staged directory only after all checks and writes succeed. A
failed final asset check leaves no consumable declaration. It refuses output
within Git or into an existing directory. Hashes establish
byte identity, not correct anatomy, normal direction or acquisition truth.
The caller must still establish that the context calculation describes the
same model, selected planes and exact sampling as the inference components.

The current completed-component adapter is
`component_export/prepare_handoff.py`. It accepts exactly three depths from
one checkpoint and direction, with distinct offsets including zero. If its
one external mask is to cover several depths, use the intersection of their
separately established context masks. A production two-shift run must remain
two shifts; do not invent a zero-depth prediction. Studio and A-hn outputs do
not become component-export receipts by changing their filenames.

## Area accounting and public projection

`SupportResult.counts()` distinguishes chart, mesh, all-required-source,
mesh-and-source, planned contribution, and full-context pixels. Uniform chart
areas are `pixel_count * dy_um * dx_um * 1e-8` cm². These are chart areas;
overlaps and mesh distortion prevent summing them as unique papyrus area.
`unique_papyrus_cm2` and `on_sheet_cm2` remain `null`. Report m7 predictions,
connected cores and coherence diagnostics separately with their definitions.

`public_aggregate(receipts, scroll="PHerc0125", surface_set="spiral_fits")`
copies only allowlisted aggregate counts, chart areas and stage labels. It
never copies local paths, mesh names, crop/tile bounds, selected planes,
per-window hashes, score maps or candidate fields. Aggregate categories are
restricted to broad surface sets. Full receipts and all arrays remain private;
publish no find before the organisers' announcement.

## Offline verification

```bash
python -m unittest discover -s firstletters-colab/producer_support -p 'test_*.py' -v
```

The suite covers valid zero values, missing context away from a valid target,
mesh holes, depth interpolation mass and order, exact zero blend weights,
reflected boundaries, cropped review regions, independent prediction coverage,
Studio empty-tile admission and exact planned geometry, absence of Torch
execution, file identity, unknown-acquisition blocking, private output paths,
non-overwrite and allowlisted public summaries. The real producer-to-model
execution remains a separate gate; these tests cannot certify it.
