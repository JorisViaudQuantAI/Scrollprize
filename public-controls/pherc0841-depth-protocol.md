# Public PHerc. 841 depth-order benchmark

## Sampling rule fixed before inference

The sampling frame is the official PHerc. 841 w00 surface volume at 2.403 µm and its dated 20260918 ink, supervision and validation masks. A trusted pixel belongs to the intersection of supervision and validation. Ink and background counts use only those pixels. [Public scan](https://scrollprize.org/data_browser/PHerc0841) · [Label conventions](https://scrollprize.org/tutorial5)

1. Partition the surface into non-overlapping, fully in-bounds 256 × 256 windows, anchored to the image origin. Exclude overlap with the previously benchmarked public window and its 272 × 278 border test.
2. Use the public level-4 mask overview to rank candidate windows. Require estimated trusted coverage of at least 75%. For each stratum rank by distance to the target ink fraction, then by SHA-256 of the ASCII string `841-w00-panel-v1:y:x`, with y and x the native pixel indices. Inspect at most the first 72 eligible candidates for each stratum. All selection uses labels alone.
3. Aim for two windows per stratum. The full-resolution trusted ink fraction must be 0–1% (target 0%), 5–20% (target 12.5%) or 35–65% (target 50%). Require full-resolution trusted coverage of at least 90%. First seek a gap of at least one complete tile on either axis between selected windows. Relax the gap to simple disjointness for an unfilled quota.
4. If quotas remain unfilled, rank remaining overview windows by descending trusted coverage then the same hash. Inspect at most 120 candidates to fill up to six windows with at least 75% full-resolution trust. Identify these windows as additional coverage and report their actual ink fractions. Report any unfilled quotas. Freeze and print the selection and its hash before opening the detector.

This is a prevalence-stratified control sample. Results characterize these windows. Pixel counts within a window are spatially correlated.

## Inference and analysis

Use the previously verified, frozen `scrollprize/PHerc.1667-iteration-0` model, central public depth indices 23–84 inclusive, original clipping and normalization, and a single CPU prediction per tile and order. Use the same three orders for every tile: ascending, descending and the single 62-index permutation already committed in `r50_stack.py`. Keep thresholds fixed at 0.5 and 0.8.

Report confusion counts, false-positive rate and recall for each window and order, by stratum and pooled across tiles. A rate with an empty denominator is undefined. At 0.5 report the number of 8-connected false-positive islands confined to annotated background, the number with at least 16 pixels, largest area and the number of distinct annotated background regions containing a false detection. Components are never joined across tile boundaries. Also report the simultaneous response across all three depth orders and the fraction of ascending detections retained under permutation.

Only public control inputs, numerical summaries and reproducible input/model/output hashes are used.
