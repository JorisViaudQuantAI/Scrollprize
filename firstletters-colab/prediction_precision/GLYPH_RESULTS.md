# Complete-glyph precision exports preserve the original calculation

Date: 2026-11-23. **From a source — [completed glyph-mode receipt](evidence/glyph_result.json):**
the separately frozen precision analysis completed on both orders of the
[published whole-glyph control](../public_glyph_control/RESULTS.md). It exactly
reproduced both original 2048 × 2048 Q8 contexts before exporting the complete
1920 × 1920 cores as retained FP32 probabilities, original Q8, Q16 and the
predeclared global logit display. The original files were unchanged.

This is the completed glyph mode of the
[5cdfe2d freeze](https://github.com/JorisViaudQuantAI/Scrollprize/commit/5cdfe2de2deee727808d9258edb20875185ba8ca),
not a revision to the [failed two-field Q16 fidelity result](RESULTS.md).
There are no pixel labels for this glyph fixture, so no AP, AUC, pixel recall,
threshold validation or Q16 fidelity pass is inferred here.

## Calculation and measured cost

**From a source — [protocol](protocol.json):** Q16 uses nearest-even rounding
of `float64(P) * 65535`. The display is the same pointwise function for every
view: `clip((logit(clip(P, float32_eps, 1-float32_eps)) + 8) / 24, 0, 1)`.
No image-specific contrast range, spatial filter, letter shape or chosen crop
enters that function. These displays never feed back into the detector.

**From a source — [run receipt](evidence/glyph_result.json):** runtime was
**6.247890 s**, peak RSS **501,504 KiB**, with eight consumed source artifacts
unchanged. There were **zero CNN forwards, training steps, network requests or
new CT reads**. Both physical depth orders remain in every representation.

| Complete core | Minimum retained FP32 probability | Maximum retained FP32 probability |
| --- | ---: | ---: |
| Source order | 0.0000189152 | 0.999177814 |
| Reverse order | 0.00000740496 | 0.992227912 |

These extrema are descriptive model outputs, not calibrated ink confidence.
PNG files produced by different serialization calls may have different file
hashes even when their decoded Q8 pixels are identical; pixel parity is checked
directly.

## What the images establish

**My inference from the complete original and fixed-display views:** the
original Q8 source-order image already shows the published control's broad
connected stroke structure. The fixed display reveals more low-score texture
while reducing the broad light/dark contrast. It does not reveal an additional
independently supported letter or repair the reverse-order discontinuity.
More visible texture is not evidence that the detector has improved.

The reference-informed visual assessment and its exposure limitations are
recorded with the [whole-glyph result](../public_glyph_control/RESULTS.md).
This is an already-published Paris 4 development control from a scroll with
documented teacher exposure; exact per-letter training membership is unresolved.
The fixed source and reverse physical slabs differ by one native plane;
neither display establishes the written physical face.

**From a source:** [verify_glyph.py](verify_glyph.py) independently reconstructs
both complete Hann contexts and checks the Q8 precondition, FP32 cores, Q16
arithmetic, display arrays and decoded PNGs. Its completed receipt is
[evidence/glyph_independent_review.json](evidence/glyph_independent_review.json).
The review reproduced both complete contexts and both core float/Q16/display
arrays exactly, including all six decoded PNGs. It took **4.557080 s** with
**560,956 KiB** peak RSS and checked eight source assets, fourteen outputs,
four frozen files and three dependency pins. It uses no trained model or
network access and makes no effectiveness claim. All earlier failed gates
stay unchanged.
