# A complete published glyph at native 2.4 µm

**Status: protocol and runner prepared for review; no execution result is included here.**

**Design choice.** This is a separate development control for recovery of a whole
published letter. It does not depend on whether the earlier stitch comparison
passes its descriptive gate. It uses one fixed public Paris 4 field, both released
depth orders, and the unchanged released Hecate Hann predictor. There is no
training, region replacement, automatic reading, order selection or unread-scroll
promotion. Both images remain part of the result even if neither is legible.

## The human reference and its limits

**From a source.** Nicolardi et al., *The final columns of PHerc.Paris. 4 revealed
through virtual unwrapping*, **Cronache Ercolanesi 54 (2024), 9–27**, publish the
human transcription of sixteen columns. Their orthography discussion explicitly
locates **γείνεϲθαι at column −8, line 5**. The reference is the fourth letter of
that word, Greek **Ν**. The printed word has no restoration brackets or underdots.
See the [primary publication record](https://www.iris.unina.it/handle/11588/985203)
and the [accessible author copy](https://www.academia.edu/125007883/THE_FINAL_COLUMNS_OF_PHERC_PARIS_4_REVEALED_THROUGH_VIRTUAL_UNWRAPPING).

**Our source collation.** Several surrounding words and the line containing the
reference word match the official gallery image identified as segment
`20231210121321` in the [Paris 4 catalogue](https://scrollprize.org/data_browser/PHercParis4).
The matched column is **−8, not the physical final text column**. The publication
does not provide a machine-readable per-glyph bounding box; this letter selection
is a documented visual collation with the already-published transcription.

**Measured source comparison, before our inference.** The official gallery JPEG
is 1800 × 2296; the canonical public `ds8` JPEG is 4998 × 6375. Resizing the latter
to the former in the identity orientation gives interior Pearson correlation
**0.9996624643** and mean absolute difference **1.2231851/255** with bicubic
resizing. The best reflected alternative gives **0.1679570**. The comparison
rectangle in gallery pixels was `[x=400:1600, y=120:2200]`. This supports the same
displayed raster and orientation. Nominal factor 8 from thumbnail to native
surface is supported by the `ds8` filename and image dimensions; the exact
thumbnail pixel-center convention was not recovered. The selected glyph has
margin around its strokes. All image URLs and hashes are in [selection.json](selection.json).

**Limits.** The human reading was made from historical machine-produced images
and scholarly review, not optical access to the closed roll. The selected public
canonical detector image guides this development fixture's location; it is not a
pixel-level answer key, and no pixel AUC is calculated against it. The
[pinned Hecate model card](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/README.md)
documents Paris 4 volumetric teacher supervision. This is a test of complete
letter recovery on training-exposed data, not independent-scroll generalization.

## Fixed source and geometry

**From source metadata.** This is the original public `20231210121321` surface,
not its separately catalogued `v8` variant:

```text
https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/segments/20231210121321/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/
```

The [L0 `.zarray`](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/segments/20231210121321/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/0/.zarray)
has ZYX shape **109 × 51000 × 39980**, chunk shape **109 × 128 × 128**,
uncompressed C-order uint8, and `/` chunk separators. Its
[`.zattrs`](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/segments/20231210121321/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zattrs)
declare isotropic **2.4 µm** at level zero. Higher levels downsample XY only.
The runner pins and validates both metadata bodies and uses native L0 integers
without interpolation, regridding, pooling or CT intensity normalization beyond
the model's prescribed division by 200.

| Region | Native Y interval | Native X interval | Raster | Physical size |
|---|---|---|---|---|
| Complete-glyph core | `[10240,12160)` | `[13696,15616)` | 1920 × 1920 | 4.608 × 4.608 mm |
| Inference context | `[10176,12224)` | `[13632,15680)` | 2048 × 2048 | 4.9152 × 4.9152 mm |

**Design choice.** The square contains **one complete nu**, with possible
neighboring mark fragments at its edges. Those fragments are not counted as
additional expected letters. The core is a 64-pixel inset on every context edge.
Coordinates identify an already-published control, not an unread-scroll find.

Every one of the **289** intersected chunks must be delivered in full. The
context starts halfway through an XY chunk and therefore spans **17 × 17**
chunks. Each contains **1,785,856 bytes**, giving **516,112,384 chunk-body bytes**
and **516,115,991 bytes including the two metadata bodies**. Missing, truncated,
transformed or wrong-sized responses abort; there is no fill or skip fallback.
Every acquired chunk gets a SHA256 receipt. Mutable bucket objects still do not
provide an atomic volume snapshot.

The reader retains native source planes **22:87**, a 65-plane union. On the
original 109-plane source, released Hecate uses **22:86** for source order and
**23:87 followed by reversal** for reverse order. Applying unchanged `predict`
to the retained 65 planes gives precisely those same two input sequences.
They have different physical slab centres by one source plane: this is **not a
pure reversal of identical CT bytes**. Neither order is asserted to be recto.
The historical render command, physical normal sign and exact sheet correctness
are not established by the metadata.

Acquisition coverage and nonempty CT columns are saved separately. All selected
source bytes must exist; a column whose acquired CT is zero remains zero. Each
of the 225 patches in each order must contain at least one nonzero CT column,
otherwise the experiment stops before fixture inference rather than silently
changing the declared forward count. Nonempty CT support is not an ink mask.

## Predictor and the four-forward precision pilot

The runner requires the locally available, hash-pinned **522,656,918-byte** fine
Hecate checkpoint; it does not download weights. Both precision candidates use
the same frozen model, input scaling and source support. The old public PHerc0841
and PHerc0814 contexts must match their complete historical `.npy` SHA256s.

**Design choice.** Before acquiring the Paris 4 chunks, the runner evaluates only
tile zero from each old context, in reverse layer order. There are exactly four
pilot forwards in this fixed order:

1. PHerc0841 BF16.
2. PHerc0841 FP32.
3. PHerc0814 FP32.
4. PHerc0814 BF16.

There are no additional warmup forwards. The first unprimed model call is BF16;
the timings are a small development comparison, not a stable speed benchmark.
The full 256 × 256 score uses `sigmoid(logits.float())`, matching released
prediction, before source-support masking. BF16 is accepted only when **both
donors** have maximum absolute difference **≤ 0.005** and mean absolute
difference **≤ 0.0005**, and the two BF16 calls' mean measured duration is
strictly less than the two FP32 calls' mean. Otherwise FP32 is retained. The
gates are not adjusted and pilots are not repeated. An execution or resource
exception aborts; a completed numerical or timing gate failure selects FP32.
Both pilot score arrays per donor are retained for independent checking.

When BF16 passes, a small `nn.Module` adapter enters CPU BF16 autocast only inside
the model call. The original `hecate.predict` still receives `precision='fp32'`;
its assembly, support masking, FP32-logit sigmoid and export code are unchanged.
The adapter preserves parameters, patch size, divisor, margin and evaluation
mode. Generated tests exercise this nested-autocast behavior against fixed fake
logits without loading a trained model.

Both fixture orders use the released **256-pixel tile, stride 128, batch 1**
predictor. Its 15 × 15 grid produces **225 forwards per order**, **450 fixture
forwards**, and **454 including the precision pilot**. Exports use the unchanged
fixed probability scale: `round(clip(p,0,1)*255)`. There is no per-image contrast
normalization, expected-letter prior, redrawing, threshold search or selection
of a preferred order.

## Archive for later depth diagnosis

**Design choice.** The same 450 fixture forwards also save pre-collapse 3D logits
and raw attention-convolution scores as float32 memory maps, each with per-order
shape **225 × 62 × 64 × 64**. A third map saves the actual full-patch 2D
probabilities, **225 × 256 × 256**, before support/Hann weighting. The raw array
payload across both orders is **1,032,192,000 bytes**, plus small `.npy` headers.
Originating tensor dtypes, the model's depth-coordinate scale, tile order and
source-plane mapping are recorded. No feature tensor is archived.

A nonmutating wrapper copies logits and returns the identical `features_logits`
tuple; a forward hook copies attention scores before the coordinate bias is
added. These do not change weights or introduce further model forwards. Model
buffers and the checkpoint are verified unchanged. The archive is reserved for
later, separately declared analysis: first independently reconstruct the actual
2D scores from the saved values, support and depth coordinate. This run does not
select another depth projection or use the archive to decide which output wins.

## Resource limits and execution

The limits are **550,000,000 newly read application-body bytes**, **400 HTTP
requests including redirects**, **no network retries**, **454 forward attempts**,
**6.5 GiB soft peak RSS**, and **7200 seconds including source reads**. The
archive has a **2,000,000,000-byte extra disk allowance**; startup requires at
least **3,000,000,000 bytes free** for source data, outputs and the archive.
The fresh source cache reuses the frozen native reader's bounded `PinnedCache`
with an empty list of prior receipts. No historical receipt is fabricated.

**Runtime planning, not a new measurement.** At the current approximate
10.5 seconds per FP32 forward, 450 forwards take **4725 seconds, about 79 minutes**.
Normal acquisition, loading, pilot checks, integrity checks and exports suggest
roughly **85–100 minutes total**. BF16 may reduce this if the fixed pilot passes;
no speedup is promised. A descriptive conservative projection uses the worst
observed FP32 pilot duration, regardless of selected precision, and does not
change the fixed two-hour limit or select the outcome.

All five files in this directory and the declared source dependencies must be
reviewed and committed before acquisition or any trained-model forward. The
runner checks its exact bytes against the supplied full public commit SHA.
The owner/root executor decides whether to run it; preparation is not automatic
execution authorization.

```bash
PYTHONPATH=/tmp/scrollprize-hecate-blend-deps python -m unittest discover \
  -s firstletters-colab/public_glyph_control -p test_geometry.py -v

PYTHONPATH=/tmp/scrollprize-hecate-blend-deps python \
  firstletters-colab/public_glyph_control/run.py \
  --freeze-reference FULL_40_CHARACTER_PUBLIC_COMMIT_SHA \
  --fine-checkpoint /tmp/verified-public-assets/hecate_2.4um.pth \
  --pilot-0841-context /tmp/previous-public-run/PHerc0841/fine_context_uint8.npy \
  --pilot-0814-context /tmp/previous-public-run/PHerc0814/fine_context_uint8.npy \
  --output-dir /tmp/scrollprize-public-glyph-control-v1
```

Use the actual already-verified cache paths; the paths above are examples.
Outputs must be in a fresh directory outside every Git worktree. Failed attempts
and partial files remain available for review. The runner saves source receipts,
the retained 65-plane input and support, pilot scores, both fixed-scale core and
context PNGs, array hashes, per-forward timing, archived depth values and
aggregate results. Detailed arrays are not automatically added to the repository.

## Decision after execution

The primary evaluation is **visual recovery of the whole published nu**, with
the expected reading disclosed and uncertainty preserved. Review both outputs
against the published reference, including breaks, absent strokes and unwanted
connections. Permit an unreadable or ambiguous outcome. Knowledge of the
expected letter is not permission to fill in an unsupported stroke. Numeric
execution success and precision parity are not a finding of legibility.

No numeric AUC against the canonical model's image, automatically preferred
order, new reading, title discovery, prize claim or unread deployment follows
from this control. Its practical purpose is to separate pipeline-level failure
to recover a complete known letter from negative results on other scrolls.
