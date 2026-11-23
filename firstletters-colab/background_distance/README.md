# Where do background-labelled Hann positives occur?

This is one **exploratory disagreement diagnosis**, using the two complete,
previously observed PHerc. 0841 and PHerc. 0814 development fields. The broad
saturation result was already known when this protocol was written. Freezing
these files protects the new spatial cross-tabulation from adjustment; it does
not turn an already observed development experiment into an independent test.

**No new result is claimed by this implementation.** Publish the four files
before opening retained predictions for this diagnostic. The runner requires
their exact bytes to match a complete freeze commit supplied by the operator.
Publication itself is the operator's responsibility; the offline runner does
not contact GitHub.

## Scientific question and limits

Do background-labelled positive and saturated predictions concentrate near
annotated ink or unknown regions, or persist in annotated interiors? Near-edge
concentration can be consistent with boundary placement, mixed cells, model
bleeding, or imperfect labels. It does not distinguish those explanations.
Interior disagreement can reflect omitted ink, larger geometry errors,
neighbouring-sheet response, or model artefacts. It does not establish true ink.

The unchanged score is the primary experiment's saved **Hann** output, in its
float and actual fine-export-derived uint8 representations. No consensus,
central-crop, or alternative assembly is evaluated here. Every original
background-labelled positive stays in the totals. There is no erosion-derived
AP/recall, alignment search, label repair, fitted threshold, model-selection
decision, or promotion gate. Previous gates and their exact result identities
are retained in the output.

These are model-assisted public labels on previously used development donors.
Exact Hecate training/teacher exposure for these fields remains unresolved. The
label publisher's training counts must not be attributed to Hecate. See the
[pinned donor manifest](../transversal_lab/hecate_donor_manifest.json),
[0841 label metadata](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/a63c3796ebd13e1c70282433cdeb5c34aaa45c07/donor-continued-training-20260910/labels/source-841-grow-a-approved-r000001/label.json),
[0814 label metadata](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/a63c3796ebd13e1c70282433cdeb5c34aaa45c07/donor-continued-training-20260910/labels/source-814-approved-r000001/label.json),
and [Hecate model card](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/README.md).

## Original physical label geometry

The common scoring grid has a 9.6 µm pitch; the categorical source labels have
a 9.36 µm pitch. Labels encode 0 unknown, 1 background, and 2 ink. The original
source-label centres are used for distances, without first resampling a
boundary mask onto the scoring grid. The existing centre-category mapping is
reproduced exactly and must match the pinned saved label file.

Coordinates are integers in units of **0.04 µm**:

| Quantity | Integer expression |
|---|---|
| Common pitch | 240 |
| Original label pitch | 234 |
| Scoring centre for global common index `g` | `(2*g + 1)*120` |
| Original label centre for index `i` | `(2*i + 1)*117` |
| Original category at scoring centre | `floor((2*g + 1)*20/39)` |

A nearest-centre tree selects the closest original category centre. Integer
`int64` coordinate differences then give exact squared distances for assigning
bands; floating-point square roots do not decide boundary membership. The
separate reference populations are original **ink** and **unknown** label
centres. The code never substitutes unknown for background.

Each fixed core has a complete **282 × 282 original-label-pixel halo crop**.
The crop includes the core's outer cell boundaries plus **76.8 µm and one
9.36 µm source-pixel guard** on each side. Cached label identities, shapes,
category encoding, bounds, and complete core mapping were checked using
labels only before prediction access. The runner repeats those checks. A
missing or out-of-bounds halo aborts; no clipping or synthetic label fill is
permitted.

The five bands are:

| Band | Physical distance to a reference centre |
|---|---|
| 1 | At most 9.6 µm |
| 2 | More than 9.6 µm, at most 19.2 µm |
| 3 | More than 19.2 µm, at most 38.4 µm |
| 4 | More than 38.4 µm, at most 76.8 µm |
| 5 | Beyond 76.8 µm |

An exact upper-edge tie stays in the lower band. If a reference category is
absent from the complete halo, the statement is **no such centre within the
tested radius**. It is classified beyond 76.8 µm, not described as having
infinite global distance. These are distances between categorical pixel
centres, not established physical annotation-error bounds.

## Support and censorship are separate

The inherited common support mask requires all sixteen underlying fine
columns to be nonzero. It is a detector-input mask, not independent proof of
acquisition completeness, sheet identity, or correct segmentation. The complete
primary acquisition receipt is preserved separately; no CT array is reread.

Observed unsupported common cells inside the saved 256 × 256 core form a
third reference population. Their distances use the same physical bands. This
diagnostic does not observe support beyond that scoring mask. A scoring cell
whose centre is at most 76.8 µm from the continuous outer core boundary is
therefore flagged **support-distance censored**, independently of its label
distances or observed support-hole distance. This is the outer eight-cell
ring: 7,936 of 65,536 scoring cells before restricting to labelled background.

The flag removes nothing. Ink/unknown distances within the tested radius are
not censored by the scoring boundary because the complete original-label halo
is available. A separately reported interior category requires all three
observed distances beyond 76.8 µm and no support-distance censorship. Its
complement and all censored cells remain explicitly accounted for. No improved
metric is computed by discarding them.

## Six fixed endpoints; no new calibration

Every background population and band receives counts and rates for:

1. Float score strictly above its unchanged own-field cutoff.
2. Actual uint8-derived common score strictly above its unchanged own-field cutoff.
3. Float score strictly above the other donor's unchanged cutoff.
4. Actual uint8-derived common score strictly above the other donor's unchanged cutoff.
5. Float score at least **0.99**.
6. Actual uint8-derived common score exactly **1.0**.

The own-field/cross-donor cutoffs are copied from hash-pinned completed results
and checked against both result chains. They are not recalculated or refitted.
In particular, a strict cutoff of 1.0 remains 1.0 and yields no exceedances.
The saved uint8-derived common score is float32 on [0,1], obtained previously
by averaging sixteen actual fine uint8 pixels. This diagnostic does not export
or quantize predictions again. The saturation summaries supplement the old
endpoints and cannot replace their decisions.

Each band reports its supported labelled-background denominator, each positive
count, positive count divided by that denominator, and share of the endpoint's
whole-field positives. Empty denominators or zero total positives yield `null`
rates/shares where undefined. Every partition must sum to the same complete
background and positive totals. Old own-field and transferred background/ink
counts and own-field tie counts must reproduce the inherited receipts exactly.
The endpoints overlap; their counts must not be added together.

## Execution contract

The only required dependencies are NumPy, SciPy and Pillow. No Torch or trained
model is imported. The cap is **60 seconds** for the entire process, including
imports and checks, with a **1 GiB soft RSS** stop between bounded stages. The
wall timer aborts work; a native call can delay final process exit. Network
requests, CT reads, model forwards, training steps, and retries are all zero.

After publication of the four files, the operator can run:

```bash
python firstletters-colab/background_distance/run.py \
  --freeze-reference FULL_PUBLISHED_40_CHARACTER_COMMIT \
  --primary-dir /tmp/scrollprize-hecate-blending-v1 \
  --consensus-result /tmp/scrollprize-context-consensus-v1/result.json \
  --output-dir /tmp/scrollprize-background-distance-v1
```

Do not use this command before the freeze. The output directory must be fresh,
outside every Git worktree and outside the inherited primary tree. Failed
attempts are retained. Every consumed input is hashed before use and again
after analysis. Original primary source/result receipts and the secondary
result are copied byte-for-byte into the output directory. The final receipt
contains the aggregate partitions, untouched old gates, and consumed asset
identities. There are no target searches, unread-scroll inputs, images, or new
reading claims.

## Generated verification

```bash
python -m unittest discover -s firstletters-colab/background_distance -p test_run.py -v
```

These tests use generated arrays and temporary files only. They check rational
coordinate mapping, complete halo bounds, nearest-centre agreement with brute
force, `int64` distances, exact edge inclusivity, absent categories, independent
support censorship, strict ties, complete count reconciliation, source mutation
rejection, and refusal before array access when the freeze check fails. They
do not open the retained prediction arrays, CT, checkpoints, or glyph outputs.
