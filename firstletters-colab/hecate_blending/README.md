# Hecate: does overlapping reconstruction recover labelled strokes?

This is a new, prospective reconstruction diagnostic on the **same two already
observed public development fields** used in [native resolution](../native_resolution/RESULTS.md).
It tests a possible tile-assembly defect. It is not an unseen-scroll validation,
an independent pixel-truth benchmark or a qualifying reading. The prior native
experiment's failed advancement gate remains failed.

## Comparison fixed before new outcomes

The two complete 2.7648 mm square CT contexts produce 2.4576 mm square scored
cores. They retain the prior exact box integration to 2.4 micrometres, fixed
reverse layer order, /200 normalization without clipping, released fine Hecate
checkpoint, CPU FP32, and 8 by 8 grid of 256 pixel patches at 128 pixel stride.
There are exactly **128 new detector forwards**. Each full probability patch is
retained once and feeds all three reconstruction methods:

- **Central:** the historical central 128 by 128 pixel tessellation.
- **Hann:** the released predictor's full-patch, FP32, nonperiodic separable Hann,
  normalized to its maximum and floored at .001, divided by summed weights.
- **Uniform:** full-patch equal-weight overlap, a mechanism diagnostic only.

The final `raw.any(Z)` mask is applied identically to every arm, as in the released
predictor. Primary scoring uses only 9.6 micrometre cells whose entire 4 by 4 fine
footprint is valid. Historical unmasked central scores and any-supported-cell
metrics are retained separately so masking cannot masquerade as stitching gain.
No new smoothing, binarization filter, sharpening or morphological closing is used.

Float maps are area-averaged 4 by 4 onto the original categorical-label grid.
The export sensitivity analysis first applies the official fine-pixel
`round(255*p)` uint8 quantization, then averages; rounding after averaging is a
different operation. Both representations are scored. All maps retain fixed
0–1 display scaling; no per-image contrast stretch is allowed.

For each field, method and representation, one descriptive 5% known-background
operating point is calculated using all its supported known background. The
strict threshold and conservative tie rule are fixed in the protocol. This is
evaluation-label conditioning, **not a deployable cutoff, search-wide false
positive bound or statistical test**.

The main endpoint is recovery of adjacent labelled-ink pixel pairs straddling
the original 32-common-pixel tile boundaries. A pair counts only when both
pixels exceed that one field threshold. Matching background-pair exceedance,
within-tile ink-pair recovery, whole-field recall, AP and AUC expose the case
where an image becomes smoother without recovering ink. Pair observations are
correlated and are not independent experimental units.

The prospective selection gate requires **both fields and both representations**
to gain at least 5 percentage points in seam ink-pair recall; lose no more than
1 point in AP, whole-field recall or within-tile ink-pair recall; increase seam
background-pair exceedance by no more than 1 point; and have at least 20 pairs
in each relevant pool. Uniform overlap cannot replace a failed primary Hann arm.
A numeric pass only warrants a separately selected larger **published known-glyph
replication**, after complete-map inspection. It cannot open an unread follow-up
or change the original native-resolution gate.

## Cold-start recovery and execution

The earlier transient runtime and CT cache were lost. This runner explicitly
starts a fresh bounded cache, retaining the original frozen code unchanged.
It never fabricates an old acquisition receipt. Both newly reconstructed full
CT inputs and categorical-label NPY files must match their **published previous
SHA-256 hashes before the first model forward**. That checks the effective input
despite the mutable CT bucket; chunk metadata alone would not establish identity.
Missing objects, incomplete geometry, changed hashes or failed downloads stop
the run. No external fill, patch substitution or retry is allowed.

The [protocol](protocol.json) pins checkpoint, source, helper, field identity,
analysis rules and caps: 900,000,000 new response-body bytes, 512 HTTP requests
including redirects, 128 forwards, 4,500 seconds including acquisition, and a
6.5 GiB between-forward soft RSS stop. Download bytes are application-body reads,
not complete network-wire traffic. Dependency installation is separately reported.
The Unix alarm may be deferred by a native operation; RSS is not an OS reservation.

```bash
PYTHONPATH=/tmp/scrollprize-hecate-blend-deps \
  python -m unittest discover -s firstletters-colab/hecate_blending -p 'test_*.py'
PYTHONPATH=/tmp/scrollprize-hecate-blend-deps \
  python firstletters-colab/hecate_blending/run.py \
    --freeze-reference FULL_PUBLISHED_COMMIT \
    --output-dir /tmp/scrollprize-hecate-blending-v1
```

Use a new output directory for every attempt. The full code, protocol, tests and
this README must match the public freeze commit. Tests use artificial logits to
check core alignment, masking and bytewise equality to the **actual released
predictor's** uint8 export without any real detector forward. Arrays and source
receipts stay outside Git; only reviewed aggregate evidence is published.

## Provenance and limits

**From sources:** the exact Hecate implementation is vendored at
[`hecate.py`](../ink/third_party/hecate/hecate.py), SHA-256
`c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59`,
matching [the pinned release](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/hecate.py).
The [donor manifest](../transversal_lab/hecate_donor_manifest.json) links the
released CT and model-assisted Studio labels. These labels and the known model
exposure prevent a claim of independent truth or generalisation. The
[prior result](../native_resolution/evidence/result.json) fixes the already
observed fields, reverse order and exact reconstructed bytes. The model's
internal XY stride and depth-margin removal still limit physical detail.

**From a source:** Codex's [earlier R50 Hann test](https://github.com/JorisViaudQuantAI/Scrollprize/blob/43a5365692be70f37258dbb3c6e29fef08f340b1/public-controls/pherc0841-hann-report.md)
reduced seams without consistent AUC gain. **Inference:** this is why labelled
stroke continuity and background controls, rather than visual smoothness, must
decide whether this Hecate reconstruction change merits further work.

**Inference:** because visible seams in the prior outcomes motivated this test,
the new public freeze is prospective for this new comparison but cannot erase
its retrospective hypothesis selection. Even a clear gain here remains a
development result on two small fragments, not ten legible letters.
