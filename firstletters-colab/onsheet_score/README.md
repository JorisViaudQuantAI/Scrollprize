# On-sheet score for a rendered surface (layer coherence)

## What it is

`onsheet_score.py` estimates how much of a rendered surface stack runs parallel to the papyrus sheet and how
much cuts obliquely through wraps. It needs only the render: no labels, no surface prediction, no model.

It compares two planes of the stack, 2 layers either side of the mesh layer. Each plane is high-passed
(Gaussian, sigma 6 px) and the pair is phase-correlated in 256 px tiles on a 128 px grid.

| tile class | rule | meaning |
| --- | --- | --- |
| parallel | response < 0.10 | the two planes share no pattern: they sample different fibre layers of one sheet |
| drift | response ≥ 0.10 and shift ≥ 1.5 px | the same sheet/gap bands appear in both planes, moved sideways: a cut-through |
| still | response ≥ 0.10 and shift < 1.5 px | a shared pattern that does not move |

`f_parallel` is the share of scored tiles in the first class. A tile is scored when at least 90 % of its
pixels are non-zero in all three planes.

## Use

```bash
python onsheet_score.py --layers RENDER/layers --out score.json --map tiles.npy
```

`--layers` takes a directory of per-layer TIFFs (the output of `vc_render_tifxyz`). `--npy` and `--zarr` take a
`[layers, H, W]` stack. `--mid` is the mesh layer and defaults to the central one. Only three planes are read.
`--map` writes the class of every tile (0 parallel, 1 drift, 2 still, 255 not scored), which is what you need to
mask the cut-through part of a window before ranking it. Outputs are never overwritten.

Needs numpy and opencv-python, plus tifffile or zarr for the matching input. `test_onsheet_score.py` runs on
synthetic planes (7 tests, CPU, no data).

## Measured

All numbers are for renders at the native voxel size (8.6–9.4 µm per pixel and per layer). The thresholds were
fixed on the three controls and five meshes judged by eye, before the comparison with the atlas labels.

| render | tiles | f_parallel | f_drift | f_still |
| --- | --- | --- | --- | --- |
| PHerc. 0841 control window (published text) | 273 | 0.985 | 0.015 | 0.000 |
| PHerc. 0306B claim crop (published text) | 546 | 0.951 | 0.040 | 0.009 |
| PHerc. 0343 awarded windings (published text) | 567 | 0.848 | 0.102 | 0.049 |

The packaged script reproduces these three values exactly from the original renders.

On the 250 published windings of `rodriguescarson/eligible-scroll-atlas-renders` (31-layer stacks, planes 13
and 17), `f_parallel` separates the windings the atlas author approved by eye from those rejected: AUC 0.83,
31 against 21, Mann–Whitney p = 0.0001. Per scroll, weighted by valid area:

| scroll | windings | valid cm² | f_parallel | windings below 0.7 |
| --- | --- | --- | --- | --- |
| PHerc. 0358 | 3 | 10.8 | 0.86 | 0 |
| PHerc. 0826 | 3 | 13.8 | 0.87 | 0 |
| PHerc. 0257 | 6 | 22.1 | 0.87 | 0 |
| PHerc. 0268 | 6 | 14.8 | 0.31 | 6 |
| PHerc. 0125 | 62 | 363.8 | 0.87 | 2 |
| PHerc. 0813 | 75 | 407.3 | 0.78 | 19 |
| PHerc. 0800 | 95 | 476.8 | 0.77 | 25 |

Per-winding values are in `atlas_scores.csv`.

## Limits

- **Upper bound.** A tile with no CT structure (air, noise, blur) also has no shared pattern and counts as
  parallel.
- **No depth information.** A render can be parallel to the sheet and still sit several voxels off its surface,
  or on the wrong face.
- **Not one sheet.** A render that is parallel everywhere can still step from one wrap to the next between
  tiles. On crushed scrolls we saw windows with a high pixel share on a predicted sheet whose en-face CT was
  mostly a "wood-grain" star of crossed folds; looking at the mid layer remains necessary.
- **Scale.** Thresholds are in pixels and were set at 8.6–9.4 µm. They were not tested at 2.4 µm or on
  downsampled renders.
- **Validation is thin.** The eye labels are another person's, on 59 windings. On PHerc. 0257 the score
  (0.81–0.93) is more generous than our own eye. Use it to rank and to qualify negatives, not as a gate on its
  own.
