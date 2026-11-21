# Full Villa inference time on PHerc. 1447 W058

Eight fresh native Villa processes ran the same 128 × 128 region of the full public W058 surface using Reader v3 on CPU. Each process loaded its model, selected patches, ran inference and wrote the TIFF and its ROI geometry. The patched code finished sooner in all four paired comparisons. Every run produced the same 16,384 decoded pixels and the same complete geometry sidecar. [Public source layers](https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces/tree/7e4d918712a8d8642a6fdeee45b80217c1e98317/w058)

| Median across four processes per version | Original Villa | Region-bounded occupancy |
| --- | ---: | ---: |
| Elapsed time | 26.47 s | 19.28 s |
| User and system CPU time | 24.43 s | 16.29 s |
| Peak process resident memory | 2,050.6 MiB | 1,189.1 MiB |

| Pair in chronological order | Baseline wall time (s) | Patched wall time (s) |
| --- | ---: | ---: |
| 1 · baseline first | 22.69 | 20.39 |
| 2 · patched first | 26.46 | 17.79 |
| 3 · baseline first | 26.77 | 25.37 |
| 4 · patched first | 26.48 | 18.18 |

The median paired wall-time saving was 5.30 seconds. The two median elapsed times differ by 27.2%. These summaries answer different questions. The machine was an Intel Xeon Platinum 8573C with other work running. The local filesystem cache stayed available between processes. Wall-time savings across the four pairs ranged from 1.39 to 8.67 seconds. ROI size, storage and model time can change the gain on another job. [Exact timings, input and output hashes](w058-full-cli.json)

## Reproduce

Use the pinned Villa revision and a separate checkout with the published patch applied. Prepare the W058 native Zarr with the repository's input manifest. Its 21 verified source planes cover the full 4,512 × 8,184 surface. Use a Linux Python environment with the native vc_reader, Villa inference dependencies and Reader v3. The trial directory must be empty.

```bash
python evidence/end_to_end.py /path/to/baseline-villa /path/to/patched-villa \
  /path/to/w058.zarr /path/to/reader-v3.pth /path/to/new-timing-directory
```

The command runs four pairs with alternating order, one math thread, four 384 × 384 inference tiles, half overlap, Hann blending and reverse depth order. It checks the complete sidecars and decoded images before writing the comparison. The median planning times recorded separately in the root README refer to the occupancy and patch-selection stage. [Patch and original benchmark](../README.md) · [Source manifest](sources.json)
