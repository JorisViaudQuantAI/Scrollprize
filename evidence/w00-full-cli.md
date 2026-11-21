# Full Villa inference time on PHerc. 841 W00

Bounding the occupancy scan to a 128 × 128 ROI reduced its logical read size by 49.8% on this public W00 crop. The complete command took a median of about 15–16 seconds in both versions. The patch finished sooner in two of four alternating pairs and used nearly the same peak memory. This adds a one-tile case alongside the four-tile W058 case, where seven of the eight published pairs finished sooner with the patch. [W00 measurements](w00-full-cli.json) · [W058 measurements](w058-full-cli.md)

| Median of four processes per version | Original Villa | Region-bounded occupancy |
| --- | ---: | ---: |
| Elapsed time | 15.511 s | 15.898 s |
| User and system CPU time | 7.540 s | 7.831 s |
| Peak resident memory | 1,038.813 MiB | 1,039.002 MiB |

| Pair, in chronological order | Original (s) | Patched (s) | Original minus patched (s) |
| --- | ---: | ---: | ---: |
| 1 · original first | 19.566 | 17.400 | +2.166 |
| 2 · patched first | 16.209 | 17.007 | −0.798 |
| 3 · original first | 14.813 | 14.088 | +0.725 |
| 4 · patched first | 14.358 | 14.789 | −0.431 |

The median paired difference was +0.147 seconds. Each fresh process imported Villa, loaded Reader v3, selected tiles, ran inference and wrote the result and geometry sidecar. The timer stopped when the command exited. The eight TIFFs had identical encoded bytes and 16,384 identical decoded pixels, spanning 109 distinct values from 57 to 165. Every field of the original sidecars matched across all eight runs, including the input path. The JSON displays its basename for portability. [Per-process measurements, geometry and SHA-256 hashes](w00-full-cli.json)

## What the occupancy scan read

The public 9.366 µm surface `20260220213127-w00` supplies 28 layers. The published fixture contains source rows and columns [2688, 3456), giving a 768 × 768 crop. The timed ROI was [32, 160) on both fixture axes, or [2720, 2848) in the source. Both commands used the native reader, level 0 for inference, forward layers 6 through 22, Hann blending and one 384 × 384 tile on the 192-pixel parent grid. [Hash-verified source manifest](sources.json)

| W00 region | Level-3 logical bytes, original → patched | Distinct chunk positions | Tiles scheduled by each version |
| --- | ---: | ---: | ---: |
| Timed 128 × 128 ROI | 258,048 → 129,472 | 1 → 1 | 1 |
| Full 768 × 768 fixture, planning only | 258,048 → 258,048 | 1 → 1 | 9 |

The entire 28 × 96 × 96 level-3 array occupies a single chunk. For the small ROI, the requested mask contracted from 96 × 96 to 68 × 68 within that chunk. For the full-field control, both native schedulers returned the same nine tiles in the same order. Logical byte counts describe the extent of the requested arrays. Physical storage traffic was outside the measurement. One shared CPU host, a retained filesystem cache and four pairs limit the timing precision.

## Reproduce

From the repository root, use a Linux environment with native `vc_reader`, the Villa inference dependencies, and baseline and patched checkouts of the pinned revision. Create a new fixture and an empty trial directory. The measurement used one available logical CPU and one math thread. Choose a CPU number allowed on your host. [Villa revision](https://github.com/ScrollPrize/villa/tree/c8f3ba45304f0a317a49a155929e30d294f201eb) · [Patch and setup](../README.md)

```bash
python evidence/fetch_fixture.py w00 w00.zarr --checkpoint reader-v3.pth
taskset -c 0 python evidence/end_to_end.py /path/to/baseline-villa /path/to/patched-villa \
  w00.zarr reader-v3.pth w00-timing --roi 32 160 32 160 --direction forward \
  --case-label "PHerc. 841 W00, public 28-layer 768 x 768 crop, native occupancy level 3"
python benchmark.py /path/to/patched-villa w00.zarr --roi 0 768 0 768 --runs 1
```
