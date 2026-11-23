# Source-support fixes ported from ChatGPT-018

Source: `firstletters-colab/gemini_integration/stream_source_support.patch` on
`claude/firstletters-colab` (commit `3dd9b90`, targeting Gemini commit `d80f4f5`).
The patch was applied to `gemini_lane/stream_snap_render.py`, then four of its API
changes were reverted to compatible defaults so that `screen_scroll_rtx4060.py`,
`paris4_dual_energy.py`, published evidence receipts and the existing tests keep
working unchanged. Generated arrays only: no CT download, no model inference.

## Ported unchanged

| Bug | Fix |
| --- | --- |
| Trilinear sample interpolating toward a missing (404) chunk or out-of-bounds corner was treated as real data | `sample_trilinear(..., return_support=True)` / `read_box(..., return_support=True)`: every strictly positive-weight corner must be in bounds, from a decoded chunk, and finite. Zero-weight corners (exact integer edges) are ignored. Dense and chunk-partitioned paths agree. |
| A zero-weight NaN neighbour poisoned exact integer samples | NaN replaced by 0 before `map_coordinates`; support stays false for that voxel. |
| One out-of-bounds depth hidden by another nonzero plane (`stack.max(axis=0) > 0`) | `valid_mask` requires full source support at **every** rendered plane. |
| Missing chunks with nonzero `fill_value` counted as valid | Fill values never count as support. |
| Mesh mask used `interp(vmask) > 0.999`, admitting tiny invalid-vertex weights | Exact positive-weight stencil over usable vertices. |
| m7 snap clipped out-of-bounds meshes into the volume (e.g. 93 vx displacement) | Unsupported centres are inactive; no clipping; float32-stored candidates must be in bounds, within radius, supported and ≥ threshold. |
| m7 rays crossed missing chunks / fill values | Unsupported ray samples set to `-inf`; quadratic sub-voxel fit guarded against non-finite neighbours. |
| `round(radius/step)` searched beyond the declared radius | `floor`. Non-finite `search_radius`/`step`/`threshold` rejected. |
| Coherence compared shifts over different masked pixel sets | Common mask (`minimum_filter`) valid for every tested shift. |
| CLI | Outputs refused inside any Git worktree; nonzero pyramid levels refused; optional `--out-support-npz`; receipt `1.1.0` adds file hashes, depth offsets, normal convention. |

`snap_tifxyz_to_m7` metrics gain `m7_source_available_fraction_raw` and
`m7_source_available_fraction_after_snap`.

## Intentionally NOT ported (compatibility)

1. **Stored zeros.** The patch counts stored zeros as valid source. Here
   `render_surface_stack(require_nonzero_intensity=True)` (default) keeps the
   legacy "some plane nonzero" condition **in addition to** full source support,
   so the mask is never larger than before. Reason: `screen_scroll_rtx4060.py`
   reports `valid_ct_cm2` from this mask, and masked S3 volumes store zeros outside
   the acquisition. Opt in with `require_nonzero_intensity=False` or CLI
   `--count-stored-zeros`. `return_support=True` exposes the pure
   `full_source_available_yx` mask either way.
2. **uint16 / float CT.** The patch rejects non-uint8 input. The legacy dtype-max
   scaling is kept by default; `strict_uint8=True` enables the rejection. The
   dtype-max scaling remains an unvalidated intensity mapping.
3. **`sheet_parallel_fraction`.** Kept as a deprecated alias equal to the new
   `zero_shift_dominant_fraction` (both emitted, plus `diagnostic_kind`). It is a
   Pearson zero-shift dominance diagnostic, **not** Claude's phase-correlation
   classifier and not an on-sheet certificate. Per-patch entries use
   `zero_shift_dominant` (no caller read the old per-patch key).
4. **`layer_coherence_score(valid_mask=None)`.** The patch raises; here the legacy
   nonzero-intensity mask is still used. Callers should pass the renderer mask.

## Numerical changes callers will observe

- `valid_mask` can only shrink versus the previous renderer (out-of-bounds depths,
  missing chunks, tiny invalid-vertex weights).
- Coherence values can change because of the common shift mask; patches near
  holes may be dropped.
- Snapped meshes no longer move toward unsupported or out-of-radius ridges.

## Tests

`test_stream_source_support.py`: one test per ported fix plus compatibility
guarantees (two-array render return, alias key, uint16 legacy scaling, legacy
`valid_mask=None`). Against the pre-port renderer, 21 of its 24 tests fail.
`test_stream_snap_render.py` is unchanged and passes.

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s gemini_lane -p "test_*.py" -v
```

## Still unresolved (from ChatGPT-018)

`cache_mb` caps cached payload only (no process-RAM guarantee); no physical area
or same-sheet certification; mesh/CT/m7 frame compatibility is assumed; snapping
onto m7 is not independent validation by m7; even slice counts give asymmetric
offsets.
