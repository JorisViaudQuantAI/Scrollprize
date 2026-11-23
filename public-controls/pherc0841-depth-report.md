# Depth order and false positives on PHerc. 841

Six disjoint 256 × 256 windows yielded 86,172 trusted ink pixels and 306,247 trusted background pixels. All six met their prespecified ink-fraction quotas. Five tiles were entirely within the published validation and supervision masks. The sixth had 64,739 trusted pixels. At 2.403 µm, each tile spans 0.615 mm per side. The 18 CPU inferences used the same frozen ResNet3D-50 weights, the same 62 CT layers and three predetermined layer orders. [Preregistered protocol](https://github.com/JorisViaudQuantAI/Scrollprize/blob/125d1c45a20cf6e80bf1a362f6fe4d6478245a36/public-controls/pherc0841-depth-protocol.md) · [Code](https://github.com/JorisViaudQuantAI/Scrollprize/blob/3106029c59a7be854d3bfca8548652df8d6fcac1/public-controls/depth_panel.py) · [Run and logs](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/44905165954/job/136849072602)

## Pooled results

The false-positive rate is the fraction of trusted background pixels called ink. Recall is the fraction of trusted ink pixels detected. Each pixel receives equal weight within the pooled results. The sampling quotas deliberately give these six windows a range of ink coverage.

| Depth order | False-positive rate at 0.5 | Recall at 0.5 | False-positive rate at 0.8 | Recall at 0.8 | False-positive islands at 0.5 | Largest island, pixels |
|---|---:|---:|---:|---:|---:|---:|
| Ascending | 12.79% | 38.65% | 0.80% | 11.32% | 10 | 20,441 |
| Descending | 13.58% | 8.68% | 0.00% | 0.00% | 16 | 18,746 |
| Fixed permutation | 48.00% | 57.63% | 17.62% | 3.02% | 17 | 49,639 |
| Intersection of all three | 1.25% | 1.22% | 0.00% | 0.00% | 8 | 3,151 |

In the two tiles containing only annotated background, ascending depth produced zero detections at 0.5 across 131,072 pixels. The fixed permutation produced 82,904 false positives in five connected islands across these same tiles. Four islands contained at least 16 pixels. The largest contained 49,639. At 0.8 the permuted model still marked 50,003 blank pixels.

On the whole panel at 0.5, the permutation retained 83.53% of the ascending true positives and 37.39% of its false positives. Requiring agreement of all three orders delivered 1,049 true positives and 3,819 false positives. Independent ink and background labels are needed to interpret a response that persists after reordering.

## Individual windows

Each cell gives false-positive rate / recall. A dash denotes a tile with zero annotated ink pixels. Coordinates are the public surface grid's native upper-left X and Y. Eight-connected false-positive components are confined to the annotated background of one tile.

### Threshold 0.5

| Public tile X, Y | Stratum | Ink pixels | Background pixels | Ascending | Descending | Fixed permutation | Islands, asc / desc / perm |
|---|---|---:|---:|---:|---:|---:|---:|
| 12544, 7936 | Near blank | 0 | 65,536 | 0.00% / — | 2.51% / — | 50.51% / — | 0 / 1 / 2 |
| 11776, 9984 | Near blank | 0 | 65,536 | 0.00% / — | 0.00% / — | 75.99% / — | 0 / 0 / 3 |
| 14336, 2048 | Sparse | 10,615 | 54,921 | 39.84% / 84.20% | 9.46% / 9.86% | 32.00% / 71.50% | 7 / 7 / 6 |
| 14080, 9472 | Sparse | 7,926 | 56,813 | 0.00% / 0.00% | 33.04% / 73.90% | 34.60% / 30.95% | 0 / 3 / 2 |
| 12288, 11520 | Mixed | 34,766 | 30,770 | 56.17% / 56.62% | 38.72% / 0.00% | 19.75% / 54.33% | 1 / 2 / 2 |
| 13568, 7168 | Mixed | 32,865 | 32,671 | 0.06% / 14.26% | 12.39% / 1.75% | 63.60% / 63.07% | 2 / 3 / 2 |

### Threshold 0.8

| Public tile X, Y | Stratum | Ink pixels | Background pixels | Ascending | Descending | Fixed permutation |
|---|---|---:|---:|---:|---:|---:|
| 12544, 7936 | Near blank | 0 | 65,536 | 0.00% / — | 0.00% / — | 29.43% / — |
| 11776, 9984 | Near blank | 0 | 65,536 | 0.00% / — | 0.00% / — | 46.87% / — |
| 14336, 2048 | Sparse | 10,615 | 54,921 | 0.00% / 4.64% | 0.00% / 0.00% | 0.20% / 0.00% |
| 14080, 9472 | Sparse | 7,926 | 56,813 | 0.00% / 0.00% | 0.00% / 0.00% | 6.29% / 0.00% |
| 12288, 11520 | Mixed | 34,766 | 30,770 | 7.92% / 26.64% | 0.00% / 0.00% | 0.00% / 3.59% |
| 13568, 7168 | Mixed | 32,865 | 32,671 | 0.00% / 0.00% | 0.00% / 0.00% | 0.86% / 4.11% |

The earlier single public control had 95.10% recall at 0.5 in ascending order. Across the four new ink-bearing windows it ranged from 0% to 84.20%. The panel uses one scroll, two background-only windows and a single fixed depth permutation. [Earlier public control](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/44903548520/job/136844271398)

## Reproduction

Use the pinned CPU environment specified in the workflow. Set the producing commit to reproduce the provenance field. The machine-readable output includes per-tile and per-stratum confusion counts, connected-component measures, the exact layer indices and hashes. [Numerical output](./pherc0841-depth-result.json) · [Workflow](https://github.com/JorisViaudQuantAI/Scrollprize/blob/3106029c59a7be854d3bfca8548652df8d6fcac1/.github/workflows/public-depth-panel.yml) · [Model weights](https://huggingface.co/scrollprize/PHerc.1667-iteration-0/tree/06c306449b39df42c745608ceade243498d24243)

```sh
GITHUB_SHA=3106029c59a7be854d3bfca8548652df8d6fcac1 python public-controls/depth_panel.py --cache ./r50-cache --output pherc841-result.json
sha256sum pherc841-result.json
```

Result SHA-256 is `9ad6c0a0c8b0f61441f696eed22b1d7be6c6e92be0e6e4c7fe4b6fec23f6a981`.
