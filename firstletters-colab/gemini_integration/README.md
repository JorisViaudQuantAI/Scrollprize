# Source support patch for the streaming renderer

## Applying the delivered patch

The patch file is [`stream_source_support.patch`](stream_source_support.patch).
It changes two existing Gemini files and adds its regression tests and documentation.
It is not automatically applied to the separate `gemini/coverage-gpu-lane` branch.
Use a clean checkout of the pinned Gemini version, or review intervening changes:

```bash
git fetch origin claude/firstletters-colab
git show origin/claude/firstletters-colab:firstletters-colab/gemini_integration/stream_source_support.patch > /tmp/scrollprize-source-support.patch
git apply --check /tmp/scrollprize-source-support.patch
git apply /tmp/scrollprize-source-support.patch
python -m unittest discover -s gemini_lane -p 'test_stream*.py' -v
```

The shared branch also contains the actual Studio CPU adapter in
[`../producer_support/`](../producer_support/README.md). It can be used from a
separate checkout against artifacts produced by the patched renderer. No upstream
PR, organiser submission, main merge, or mutation of Gemini's remote branch was
performed to deliver this patch.

This patch targets `gemini/coverage-gpu-lane` commit
[`d80f4f5d323688bcfe010a91bd3c0799db7a5311`](https://github.com/JorisViaudQuantAI/Scrollprize/tree/d80f4f5d323688bcfe010a91bd3c0799db7a5311).
It was developed against an isolated copy, using generated arrays only. No CT was
downloaded and no ink model was run.

## Why this patch is needed

The original renderer treated `stack.max(axis=0) > 0` as CT support. Generated
counterexamples show that this is not a source-availability mask:

| Generated case | Original result | Patched result |
| --- | --- | --- |
| Three depth samples `[0, 100, 100]`, first out of bounds | All four output pixels valid | All four excluded from full-depth support; individual supported depths remain available |
| Interpolation halfway from present 100 to an absent chunk's zero fill | Value 50 declared valid | Value 50 retained, source support false |
| Stored zero voxels, all required chunks present | No pixels valid | Source support true, nonzero-intensity diagnostic false |
| No chunks present, metadata fill value 128 | All four output pixels valid | Source support false |
| Generated m7 array has depth 8, raw plane at 100, search radius 2, two passes | Plane clipped to 7: displacement 93, no m7 evidence | Plane unchanged, displacement zero |

These are software counterexamples, not measurements of a scroll or a success
claim for ink detection. The existing curved-sheet and normal-direction fixtures
continue to pass.

## Public API and changed semantics

```python
values, source_available = reader.sample_trilinear(xyz, return_support=True)

stack, full_stack_valid, masks = render_surface_stack(
    x, y, z, reader, return_support=True,
)
# masks["mesh_valid_yx"]            bool [H, W]
# masks["source_available_lhw"]     bool [L, H, W]
# masks["nonzero_intensity_yx"]     bool [H, W], diagnostic only
```

Without the new keyword, sampling still returns values alone and rendering still
returns two objects. **The renderer's default validity semantics change:** every
rendered depth must have full positive-weight source support, rather than any
depth having positive intensity.

Source availability requires every strictly positive-weight trilinear corner to
be in bounds, from an existing decoded chunk, and finite. A missing corner with
zero interpolation weight is irrelevant, including at exact integer boundaries.
The mesh interpolation mask likewise requires every positive-weight mesh
contributor to be usable; it no longer uses a 0.999 approximation.

**Stored source support is not certified acquisition support.** An explicitly
stored zero is valid source data, but masked-volume zeros may encode an absence of
original acquisition. A producer must supply or assert true acquisition support
separately where that distinction applies. Neither these masks nor the stronger
full-stack mask certifies anatomical same-sheet support or a model's entire
receptive-field context. The model-context adapter must check the actual source
depth contributors and every positive-weight contributing input tile.

`--out-support-npz` optionally saves the three masks above and
`full_stack_valid_yx`. The private receipt records the exact NPZ file hash and the
exact stack NPY file hash separately from the legacy C-order array-payload hash.
It also states the layer offsets, mesh index, normal convention and plane order.

## Other corrections

- Rendering rejects uint16 and floating-point CT. Converting through the dtype
  maximum was an implicit and unvalidated model input transformation. A caller
  needs an explicit, validated upstream intensity mapping to uint8.
- m7 snapping requires source-supported centers and targets, excludes unsupported
  ray samples, stays within the requested radius, and never clips a mesh into the
  array bounds. The actual stored float32 candidate is checked before acceptance.
- `layer_coherence_score` retains its Python name for import compatibility but
  emits `zero_shift_dominant_fraction`, not `sheet_parallel_fraction`. This is a
  Pearson alignment diagnostic with an explicit common source mask for every
  tested shift. It is **not equivalent** to Claude's high-pass phase-correlation
  classifier, and no Claude threshold or area interpretation transfers to it.
- CLI writes of stacks, detailed receipts, support arrays and snapped meshes must
  be outside every Git worktree. The check follows symlinks and recognizes `.git`
  files used by linked worktrees. Source basenames, URLs and patch positions can
  localize a surface even after personal-path sanitization. Detailed output remains
  private; publish only separately reviewed synthetic evidence or aggregates.
- The CLI rejects nonzero pyramid levels because it has no coordinate transform.
  This does **not** establish that two arrays called level 0 share a scanner frame;
  the caller must still verify mesh, CT and m7 acquisition/frame compatibility.

## What remains unresolved

1. `cache_mb` bounds cached ndarray payload only. `get_chunks` still retains all
   fetched results for a request, `read_box` holds a dense block, the renderer
   allocates the full output stack, and `return_support=True` adds a full boolean
   LHW array. Missing-chunk cache entries also consume uncounted dictionary memory.
   A generated 512-byte cache returned 4,096 bytes of live chunks; a zero-byte
   cache retained 10,000 missing-entry records. **There is no hard process-RAM
   guarantee.** Use bounded jobs until a separate streaming-memory change is tested.
2. No physical area is calculated. Pixel counts, chart scaling and a layer
   alignment statistic do not measure unique, same-sheet square centimetres. A
   geometry Jacobian, voxel pitch, topology/overlap and anatomical support remain
   separate requirements.
3. The code assumes mesh/CT/m7 coordinates share a frame and that voxel-space unit
   normals are appropriate. Scanner transforms, origin, anisotropic voxel pitch,
   wrong-face diagnosis and sheet identity are not solved here.
4. Snapping onto m7 is not independent validation by m7. Multiple ridges can still
   represent different sheets; fixed-center comparisons and anatomical inspection
   remain necessary.
5. Even slice counts have integer-centered, asymmetric offsets. Consequently,
   flipping normals is an exact depth reversal only for odd counts. The receipt
   now records all offsets explicitly; the existing default is 31 slices.
6. The producer-context declaration still needs verified mesh/normal hashes,
   acquisition/frame and actual model sampling metadata. This renderer does not
   invent those assertions.

## Offline validation

Apply the patch to the pinned branch, then run:

```bash
python -m unittest discover -s gemini_lane -p 'test_stream*.py' -v
```

The isolated equivalent is:

```bash
python -m unittest -v test_stream_snap_render test_stream_source_support
```

There are 20 tests: 14 new source-support regressions and six existing fixture
tests updated for the changed validity/diagnostic semantics. In the audit runtime,
19 passed and one was skipped because optional `requests` is unavailable. The
skipped existing test is the requests-transport/HTTP-and-LRU fixture; no live HTTP
call was attempted. The tests exercise exact integer boundaries, every positive
corner, nonfinite neighbors, direct-versus-partitioned sampling, missing chunks,
stored zeros, full depth support, dtype rejection, m7 displacement, diagnostic
labels, private-output paths and exact private artifact hashes.

## Independent renderer-to-Studio preflight run

From the shared branch checkout, point this generated-data check at the patched
renderer in the Gemini checkout:

```bash
python firstletters-colab/gemini_integration/verify_pipeline.py \
  --renderer /path/to/gemini-checkout/gemini_lane/stream_snap_render.py \
  --output-dir /tmp/scrollprize-generated-stream-check
```

The output directory must be new and outside every Git worktree. The check writes
an explicit local uint8 Zarr, renders a flat generated mesh, and invokes the real
Studio CPU preflight with the renderer's support NPZ. It loads no model or weights.
The [saved result](evidence/stream_studio_integration.json) records exact code hashes.

| Generated fixture | Planned covered pixels | Source-supported pixels | Full tile-context pixels |
| --- | ---: | ---: | ---: |
| All chunks present | 65,536 | 65,536 | 4,096 |
| One context chunk absent | 65,536 | 61,440 | 1,792 |

The central 64 × 64 review patch itself has all its source samples in both cases.
The lost context comes from a positively contributing tile that also reads outside
that patch. Conversely, reflect-padded boundary tiles cannot certify real context
even in the fully present fixture. These are conservative software support checks,
not measurements of receptive-field sensitivity or ink quality.

Both preflight receipts remain `handoff_ready: false` and report zero forwards.
The final [validation receipt](evidence/validation.json) also records 20/20 renderer
tests passing after installing optional requests support, 21/21 producer-support
tests, the existing consumer/Studio suite results and their skips, and the two
producer-publication defects caught and repaired during independent review.
The [baseline counterexamples](evidence/baseline_counterexamples.json) preserve the
unpatched behavior. A real published-positive render-to-GPU run remains required
before claiming this repair improves real scroll inference.
