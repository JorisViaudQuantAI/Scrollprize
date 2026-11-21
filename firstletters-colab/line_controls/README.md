# Public controls for sparse-line review

This is a small, fixed corpus for checking whether a line-review tool can
localize already published writing. It is **not** a new ink model, an independent
glyph-classifier test, a representative negative set, or a candidate gate.
Source maps and generated arrays remain outside git. Only retrieval/extraction
code, the frozen protocol, its coverage amendment and compact provenance are
committed.

## Why these controls are public

**From a source.** The organisers announced the
[PHerc. 343 award on 8 October](https://scrollprize.substack.com/p/50k-first-letters-prize-awarded-for)
and the [PHerc. 306B award on 2 November](https://scrollprize.substack.com/p/writing-seen-in-pherc-306b-for-the).
The selected images, annotations and source maps are in those winners' released
repositories. No held-back atlas candidate image or unannounced find is used.

**From a source.** The
[343 release](https://github.com/Nieuwlaar/pherc343-first-letters/tree/e8370f1f88a211166056df62ea6fcfffe4a1063e)
contains a yellow-ink-over-CT image, physical scale, display recipe and 18
letter-location boxes in two rows. It does not include the original raw
probability maps in that Git tree. The
[306B release](https://github.com/SirProdigle/pherc0306b-first-letters/tree/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc)
links to individual native maps in its
[Hugging Face dataset](https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/tree/f435998b55269762942a979052044309b29650ac/results).
The [data guide](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/docs/MODELS_AND_DATA.md)
identifies the raw raster as 4,900×7,780 pixels. We use the released TRACE × A-hn
product map, its depth-shuffle counterpart and the public CT layer used for
support. These product scores are not calibrated probabilities of legible text.

## Frozen selection and the coverage correction

**From the experiment record.** [protocol.json](protocol.json) fixes the source
revisions, crops, physical scale, source-based targets, extraction and reporting
scope. It was written before downloading image pixels or seeing new ranker/glyph
outputs. Its SHA-256 is
`ff7164e51aae2d2dcffce021fac13ed0e5df9f65d86b638aa491323dcdb49efa`.
This is a local recorded freeze, not an externally timestamped preregistration.

**From the source-byte inspection.** Both released 306B score maps contain NaNs
outside available prediction coverage. The claim has 27,520,437 finite pixels;
the shuffle has only 4,864,188. Nonzero CT alone would wrongly assert model
coverage at 1,877,549 and 24,439,248 pixels, respectively. Before any ranking,
[amendment-001.json](amendment-001.json) added the required finite-score support
and a matched-coverage claim case. It leaves source/crop choices and all ranking
thresholds unchanged. Missing scores are stored as zero **only behind a false
validity mask**; they remain unknown and supply no negative evidence.

## Corpus

**From the extracted artifacts.** Every array has float32 scores, an explicit
boolean mask and separate metadata. The sampling is 8.64 µm in each displayed
axis. Raw 306B values are exact float16-to-float32 promotions with a crop and
vertical reversal, without spatial interpolation or contrast stretching.

| Case | Shape H×W | Kind | Valid fraction | Interpretation |
| --- | ---: | --- | ---: | --- |
| `pherc306b_context_raw` | 4051×6945 | `model_score` | 97.4823% | Published positive context, two coarse row targets |
| `pherc306b_blank_raw` | 2662×579 | `model_score` | 97.5820% | Original authors' blank control strip; not independently labelled text-free |
| `pherc306b_shuffle_raw` | 4051×6945 | `model_score` | 17.2891% | Corrupted-input process control on common support |
| `pherc306b_context_matched_raw` | 4051×6945 | `model_score` | 17.2891% | Unchanged claim scores on exactly the same mask as the shuffle |
| `pherc343_context_display` | 2455×5641 | `display_signal` | 63.3713% | Normalized displayed yellow opacity; 18 published letter boxes |

Source/output hashes, exact extraction bounds and targets are in
[inventory.json](inventory.json). Validity fractions are measurements of these
masks, not unique-papyrus area or audited acquisition coverage. We checked that
all valid 306B output samples equal the corresponding native source samples,
that matched claim/shuffle masks are identical, and that all output scores are
finite in [0,1] with boolean masks of identical shape.

### Physical and label conventions

**From a source.** 306B's
[image sidecar](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/PHerc0306B_20250521133212_w074own_touchup_V5Rc.json)
specifies the source frame, 8.64 µm scale, claim box and two row centers. Its
[blank-strip record](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/method/checks/blank_strip_variants_v7_trace_x_Ahn_product.json)
specifies the distant strip. Our line targets use the claim's horizontal bounds
inset by eight source pixels and ±2 mm around each published row center. They
are coarse localization targets, not pixel-accurate text segmentation. The
source display samples pixel centers while our extraction preserves native
pixels; about one source pixel of rounding uncertainty is immaterial to these
millimetre-scale boxes but must not be hidden in a pixel-accuracy claim.

**From a source.** 343's
[sidecar](https://github.com/Nieuwlaar/pherc343-first-letters/blob/e8370f1f88a211166056df62ea6fcfffe4a1063e/outputs/concat_w047-w048_R5B2_z9500-11000.json)
defines a grayscale CT background, yellow `(255,235,120)` ink color, 0.9 maximum
opacity, and cyan baseline annotations. The derived signal is
`clip((R-B)/(135*0.9),0,1)`. A fixed color-consistency mask excludes cyan and
unrelated colors. This recovers only normalized displayed opacity: the source
contrast stretch, thresholding, clipping and quantization are already baked in.
It does **not** recover raw model probabilities or faint signals hidden below
the original display threshold. Black content and annotation pixels are invalid.
No data-dependent stretch is added by this extractor.

**My inference — interpretation.** Use the two positive controls to expose
missed sparse lines and localization failures, and use the blank strip as one
documented challenge case. Compare original and shuffled 306B scores only on
their common support. A successful shuffle rejection is not a false-positive
bound on ordinary papyrus. Never label the rest of the positive context as
background merely because it is outside the supplied boxes.

### Glyph-model leakage limitation

**From the inspected checkpoint, recorded in [the CPU replay](../glyph_review/results.json).** The
released `gA.pt` checkpoint's saved arguments list `realpos='e343,h841,w1447'`.
Thus the 343 control is a **training-family control** for that glyph classifier,
not held-out evidence. The 306B training-window corpus is not released, so
independence of every 306B negative cannot be certified either. The frozen
protocol initially recorded overlap as unknown; the checkpoint inspection
clarifies that limitation without changing any control selection. The source
checkpoint is in the
[released discovery tree](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/discovery/x_glyph/models/gA.pt).

## Reproduce

Dependencies are NumPy and Pillow. Approximately 180 MB of source data are
downloaded once. The generated float32/mask arrays need additional disk space.
Hashes are mandatory, and an existing `cases/` directory is never overwritten.

```bash
python firstletters-colab/line_controls/prepare.py \
  --output-dir /tmp/public-line-controls
```

Each `/tmp/public-line-controls/cases/CASE/` contains `score.npy`, `validity.npy`
and `manifest.json`. With the sibling review tool:

```bash
python firstletters-colab/line_review/review.py \
  --map /tmp/public-line-controls/cases/pherc306b_context_raw/score.npy \
  --mask /tmp/public-line-controls/cases/pherc306b_context_raw/validity.npy \
  --spacing-um 8.64 8.64 --map-kind model_score \
  --output-dir /tmp/public-line-review
```

Use `display_signal` for the 343 case. The manifest's target boxes are for
evaluation only; do not pass them to proposal generation or fit thresholds to
them. This corpus alone cannot calibrate a search-wide false-positive rate.

## Licence and attribution

**From a source.** The
[343 README](https://github.com/Nieuwlaar/pherc343-first-letters/blob/e8370f1f88a211166056df62ea6fcfffe4a1063e/README.md#licence)
and [306B data licence](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/LICENSE-DATA)
place derived images/maps under **CC BY-NC 4.0**; their code is MIT. Credit the
original source repositories and Vesuvius Challenge data when sharing derived
controls. Our operations are cropping, reading-order reversal, validity-mask
construction and, for 343 only, explicit display-opacity extraction. No source
image pixels are presented as raw CT or independently annotated truth.
