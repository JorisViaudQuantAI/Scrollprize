# Single-line proposals for mandatory review

This CPU helper complements the dense-row scorer. It looks for one chain of
separate marks in a model map and writes a review packet containing the whole
map and unchanged original-resolution crops. It emits **no text, blankness,
candidate or prize verdict**. Every map remains review-required, including a map
with no proposals or a truncated search.

## Source and scope

**From a source.** The released PHerc. 306B
[single-line finder](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/discovery/fl/scripts/line_find.py)
was introduced because a spectral row score needs multiple rows, while a reader
may first notice one line. It links nearby fragments into marks, searches chains
of separated marks in physical units, and compares a line band with its flanks.
Its inputs are recipe-specific fused z-maps, not raw probabilities. It also uses
a learned glyph model and recipe-specific null statistics; the source explicitly
labels its pooled-null correction as post-hoc.

**From our implementation.** We adapted the mark/chain geometry and geometric
score, with attribution under [LICENSE.source](LICENSE.source). The inspected
upstream revision is `feabff6c54f95bd8e52a90e0a7a76881ee67d4cc`; the complete
upstream `line_find.py` has SHA-256
`91b0410894220ac1ddde99cb38fa93c5b1385ea01fc3d130eaa6255a051bbb7a`.
We did not copy its machine-specific paths, lane discovery, candidate printing,
glyph checkpoint, old gates, glyph null constants or claimed false-positive
rates. This implementation is a new, uncalibrated ranking variant, not an exact
reproduction of the winning detector.

**My inference.** A sparse line merits its own review path. A high score still
can come from cracks, parallel fibres, regularly spaced blobs, model tiling or
unrelated aligned marks. Four geometric slots do not mean four letters. There
is no statistical confidence attached to this score and no First Letters
eligibility decision here.

## Explicit input contract

- `score`: a floating NumPy `[H,W]` array. Values at valid pixels must be finite
  and in `[0,1]`. There is no implicit integer-image scaling, percentile stretch,
  normalization, polarity change or model-family conversion.
- `valid`: a **boolean** array with the same shape, provided explicitly. A zero
  map value remains an observed zero when its mask is true. Nonfinite values at
  unobserved pixels are ignored; unobserved pixels never become background.
- `spacing_um_yx`: actual positive physical `[dy,dx]` pixel spacings in micrometres.
  Anisotropic sampling is supported. A physical scale cannot be inferred from
  image dimensions, and a display screenshot without scale is not suitable.
- `map_kind`: **`model_score`**, **`probability`** or **`display_signal`**, required.
  Use `model_score` for uncalibrated model products. `display_signal` identifies
  values reconstructed from a stretched or composited public illustration.
  Their thresholds are display thresholds, not detector probability thresholds.
  Even a model called a probability detector is not assumed calibrated for text.

CT acquisition coverage and surface support are separate facts. The supplied
mask must describe which map values are valid for the intended comparison;
a silhouette or a CT-nonzero mask alone does not establish all source chunks
were acquired or that the mesh follows the correct sheet. Record its provenance.

## Fixed default search

**From our implementation, proposed defaults rather than calibrated optima.**
The finite defaults are written in the frozen `Config` dataclass. Set or record
them before looking at a control outcome; do not choose whichever control-specific
parameter list happens to give the most appealing ranking.

| Parameter | Default |
| --- | --- |
| Input thresholds | 0.25, 0.50, 0.75 |
| Directions | 0, 5, …, 175 degrees in image coordinates |
| Target work-grid sampling | Approximately 35 µm; no upsampling |
| Fragment-link diameter | 0.15 mm |
| Minimum mark area / extent | 0.06 mm² / 0.4 mm |
| Chain band half-width / placement step | 1.0 mm / 0.25 mm |
| Maximum mark extent across / along chain | 4.0 mm / 8.0 mm |
| Slot-merging centroid gap | 1.2 mm, or overlapping along-line extents |
| Split a chain at a gap above | 6.5 mm |
| Minimum slots / line length | 4 / 8.0 mm |
| Flanks from the line centre | 1.75–3.5 mm on each side |
| Required observed share of each flank for contrast | 0.50 |
| Long-chain score penalty starts above | 60 mm |
| Unique mark-set resource limit | 4,000 per threshold, reported if reached |
| Maximum work-grid pixels | 8,000,000; larger jobs must be partitioned |
| Crops included in the review packet | First 12; whole map always included |

Most physical geometry constants follow the source. The finite probability-scale
threshold family, broad direction search and resource limits are our choices.
They are **not** the source's z-map threshold or a transferred calibration.

### Mask and coordinate behavior

**From our implementation.** Downsampling uses area means over observed samples.
Work cells containing any missing support are marked unobserved conservatively.
No row or column is removed to close a tear. Distances and physical extents
therefore remain in the original chart frame. Fragment linking is constrained
by the validity mask. Missing support remains visible in every packet.

Flank coverage uses the full expected geometric flank area, including the part
outside the image. If either flank lacks sufficient support, its contrast is
`null` and the score gets **no contrast bonus**. It is not treated as observed
blank papyrus. Incomplete band support is reported and penalized. Such proposals
are still available for review rather than silently discarded.

The rank score is geometric: log2 contrast when available, plus log2 slots/4,
minus flank crowding, incomplete-band and long-chain penalties. No neural glyph
score or OCR output enters it. Greedy overlap suppression reduces duplicate
packet crops across neighboring parameter settings; the full map is preserved.
The JSON records the threshold, angle, features, coverage, actual search counts
and truncation state. Different parameter trials are not independent evidence.

## Use

Dependencies are NumPy, OpenCV and Pillow. This tool uses no Torch, GPU, model
download or network connection. From the repository root:

```bash
python firstletters-colab/line_review/review.py \
  --map /tmp/private-input/model_map.npy \
  --mask /tmp/private-input/map_validity.npy \
  --spacing-um 9.36 9.36 \
  --map-kind model_score \
  --output-dir /tmp/private-review/run-001
```

The output directory must be new and **outside every Git worktree**. No explicit
override is provided: unread-scroll locations and images belong in private
experiment artifacts. The command prints only proposal count, runtime, review
requirement and truncation; it prints no source name, path, score, location or
reading. Input paths are not copied into the JSON, but the JSON's bounding boxes
and the packet images still make the packet sensitive for an unread scroll.
Do not publish a generated packet merely because it is easy to generate.

**Integrity-only amendment after the first control run.** The command now hashes
both source files before loading and checks both hashes again after ranking,
before writing any packet. If a producer replaces the map or mask meanwhile,
the run fails rather than attaching a new file's hash to an older loaded array.
Wait for producers to finish before invoking review. This change affects only
the command's input-integrity checks; `rank_map`, `Config`, search parameters and
ranking outcomes are unchanged. Any replay after this amendment must be labelled
as a post-outcome robustness replay, retaining the original frozen run and source.

For a source-backed public illustration, supply a separately derived `[0,1]`
signal and use `--map-kind display_signal`. Preserve the derivation, original
image hash and its physical-scale evidence in the control's provenance. A result
on a stretched public illustration only tests geometric review behavior.

The Python interface is:

```python
from review import Config, rank_map, write_packet

config = Config()  # Record this before measuring control outcomes.
report = rank_map(score, valid, [9.36, 9.36], "model_score", config)
write_packet(score, valid, report, "/tmp/private-review/run-001")
```

`bbox_yxyx` and `band_polygon_xy` refer to the **original** map. The packet saves
the full original map and each selected crop unchanged as `.npy`, with matching
boolean masks. PNGs use the supplied `[0,1]` scale with no new contrast stretch;
purple means unobserved. Overviews and HTML previews can be reduced, so inspect
the native arrays at full resolution before recording review completion.

## Tests and remaining validation

**From our measured public-control run.** The [frozen evaluation](../line_evaluation/README.md)
now records localization on released 306B maps and a 343 display signal, including
the failure to cover the second published 306B row within the first five bands.
The control set supplies no independent false-positive rate or skip-review gate.
The full-map review requirement remains in force.

```bash
python -m unittest discover -s firstletters-colab/line_review -p 'test_*.py' -v
```

**From our tests.** Eleven tests pass: a single line without a second row; physical
scale invariance under replicated isotropic and anisotropic sampling; preserved
distance across a mask gap; unavailable flanks receiving no blank-background
bonus; ignored invalid values; a complete review packet for zero proposals;
unchanged original crops; no overwriting; Git-worktree exclusion; and explicit
search truncation/resource and input failures; and rejection before packet writing
when a producer atomically replaces either source file during ranking. Synthetic marks in these tests
are geometry fixtures, not demonstrated Greek text or independent accuracy data.

**My inference.** Before using scores to prioritize a large queue, run the exact
frozen family on published known-text, sparse-text and non-text controls at their
actual physical scales. Perturbed depth order is a perturbation control, not a
certified negative. If a future false-positive rate is needed, calibrate the
**maximum over the entire searched family per parent** against appropriately
matched null parents, including model, preprocessing, area, validity and display
scale. Do not count hundreds of correlated angles or overlapping bands as
independent trials. Until that work is done, retain mandatory full-map review
and report these scores only as review ordering.
