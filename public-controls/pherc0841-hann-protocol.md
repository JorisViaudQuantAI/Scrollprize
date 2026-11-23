# PHerc. 841 public test of Hann tile blending

## Data and selection fixed before inference

Use only the official PHerc. 841 w00 2.403 µm surface (native layers 23 through 84 inclusive in ascending order) and the dated 20260918 ink, supervision and validation masks. A trusted pixel belongs to both supervision and validation. A parent is 640 × 640 native pixels, with an evaluated central core of 512 × 512. The two parents must be disjoint throughout their full footprints. Exclude the 272 × 278 public parity test and every tile in the preceding six-tile public depth study, also by full parent footprint. [Source surface and labels](https://scrollprize.org/data_browser/PHerc0841) · [Earlier public windows](pherc0841-depth-result.json)

1. Place potential 512-pixel core origins on a 256-pixel lattice anchored at the source origin. Keep parents fully inside the array. Use the public level-4 masks to estimate coverage and ink prevalence inside each core.
2. Aim for two mixed fields, with target ink fractions of 0.35 and 0.65. For each target, order candidates by the absolute difference from that target in the overview, then descending trusted coverage and increasing native Y and X. Require overview trusted coverage of at least 70%. Inspect at most 100 geometrically available ranked candidates per target and confirm counts on the full-resolution masks. First require at least 90% trusted coverage and 20–80% ink among trusted pixels. If a target remains empty, relax to 75% trust and 10–90% ink using the same ranking. Report missing targets and selection tiers. Freeze and print selected regions, label counts and hashes before opening the CT or weights. No fields are replaced after scoring.

## Tile phases and endpoints

Use the published frozen R50 and 256 × 256 tiles with stride 128. Phase A uses all 16 tiles in the 640-pixel parent. Phase B uses nine tiles on the inset 512 × 512 field, shifting both native-axis grid origins by 64 pixels. Both score the same physical 512 × 512 central region. A secondary 256 × 256 center has four contributing tiles at every pixel in both phases. The study requires 25 model calls per parent. For each phase, uniform and Hann consume the identical per-tile probabilities.

Uniform is the previous arithmetic mean. Hann uses the product of two symmetric positive windows, each obtained as `np.hanning(258)[1:-1]`, and divides by the sum of weights per pixel. The previous uniform implementation is loaded from commit `329e9552c35d06a52720c8fbb4077005bef29bdc` and tested pixel-for-pixel on each phase. Positive edge weights preserve border pixels and allow a one-tile or nonoverlapping use case.

* **Seams.** Get the internal start and end coordinates from the actual tiles in each phase. Across each boundary measure the mean absolute neighboring-pixel change in the corresponding direction. Compare it with the mean at offsets −16 and +16 along the same axis, for the same scanlines. Require trusted labels for both pixels in each of the three differences. Pool sums across horizontal and vertical edges. Report their ratio, both means and edge counts. Larger than one means the grid boundary changes more sharply than its nearby control.
* **Grid stability.** On the same trusted pixels compare phase A to phase B using mean and 95th-percentile absolute probability change and binary Dice at fixed thresholds 0.5 (primary) and 0.8. Dice is undefined when both positive sets are empty.
* **Ink labels.** For each field, view and blend report ROC AUC, true and false positives, true and false negatives, recall, false-positive rate and Dice against the annotations at 0.5 and 0.8. Keep all favorable and unfavorable outcomes. Whole-core and four-tile-center results are reported separately.

The fields are annotation-selected samples of one surface. Adjacent pixels are correlated and a higher phase Dice can arise from either stabilized predictions or a change in the amount of ink predicted.
