# Region-bounded occupancy reads for Villa flat ink inference

Villa’s flat `--roi` command checks a coarse occupancy map before selecting model patches. On single-level surfaces the check falls back to the input level. This patch limits the read to the requested region plus the context of every parent-grid patch that can overlap it. It preserves the original patch order and top-left alignment when pooling an optional user mask.

## Apply the patch

Use a clean Villa checkout at `c8f3ba45304f0a317a49a155929e30d294f201eb`. From the directory containing this README, set `VILLA` to its absolute path.

```bash
VILLA=/path/to/villa
git -C "$VILLA" apply --check "$PWD/flat-roi.patch"
git -C "$VILLA" apply "$PWD/flat-roi.patch"
```

Use the usual flat inference command in the configured Villa environment. Coordinates are half-open `(row_start, row_stop, col_start, col_stop)` at the selected input level. Replace the example region with one inside your surface.

```bash
python -m vesuvius.ink_detection.inference.infer surface.zarr weights.pth region.tif \
  --roi 400 528 600 728 --direction forward
```

## Measured example

On the publicly annotated PHerc. 1447 W058 surface, the occupancy check requested 16,783,956 voxel bytes for one 128 × 128 region. The baseline requested 775,450,368. The two schedulers selected the same four 384 × 384 inference patches in the same order. Five fresh processes per version gave median planning times of 0.095 and 3.885 seconds respectively. A PyTorch replay produced identical probabilities on all 16,384 region pixels. These timing figures cover patch selection. [Public W058 dataset](https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces)

## Measure your own surface

`benchmark.py` uses Python 3.10+, NumPy and Zarr-Python. It compares the scheduling functions from the pinned Git commit with those in the patched working tree on your local input. Set the square patch size and stride to match your model. By default the script uses 384 pixels and half-patch overlap.

```bash
python benchmark.py "$VILLA" surface.zarr --roi 400 528 600 728 --patch-size 384 --stride 192
```

The JSON reports timing, selected logical bytes, positions on the chunk grid, and a hash of the ordered patch list. A differing patch list fails the comparison. Times cover occupancy and scheduling only. Chunk positions describe the selection and can include missing or cached chunks. Compressed traffic and model execution time depend on the storage backend and model. The baseline run needs enough memory to materialize the occupancy level. A small existing pyramid or a region covering most of the input can yield little reduction. A bare array without an occupancy group or user mask already schedules every patch in the relevant region.

`compare_native.py` runs both real Villa CLIs using the same installed `vc_reader` and CPU interpreter, checks the resulting TIFF pixels and ROI geometry, and discards its temporary outputs. It requires a Villa environment with its native extensions, model extras, tifffile, and a compatible checkpoint. The comparison uses CPU inference with one patch per batch and the same half-patch overlap in each run.

```bash
git -C "$VILLA" worktree add --detach /path/to/villa-baseline c8f3ba45304f0a317a49a155929e30d294f201eb
python compare_native.py /path/to/villa-baseline "$VILLA" surface.zarr weights.pth \
  --roi 400 528 600 728
python -m pytest "$VILLA/vesuvius/tests/ink_detection/test_roi_inference.py"
```

Choose your usual directional convention with `--direction reverse` when appropriate. Scope is restricted to flat single-region inference. Existing output-area selection, model weights and native reader requirements are inherited from Villa. [Villa installation](https://github.com/ScrollPrize/villa/blob/c8f3ba45304f0a317a49a155929e30d294f201eb/vesuvius/README.md) · [Flat ink detection](https://github.com/ScrollPrize/villa/blob/c8f3ba45304f0a317a49a155929e30d294f201eb/vesuvius/docs/ink_detection.md)

The patch builds on the parent-context ROI implementation contributed to ScrollPrize/Villa in PR 2169. This standalone, AI-assisted software and its patch follow Villa’s MIT license. Input data and checkpoints retain the terms of their respective authors. [Upstream ROI implementation](https://github.com/ScrollPrize/villa/pull/2169) · [MIT license](LICENSE)
