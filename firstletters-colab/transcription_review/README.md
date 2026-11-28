# An auditable journal from visible strokes to word hypotheses

This small, CPU-only tool records **visible-stroke observations, alternative Greek letters, abstentions and lexical hypotheses as separate things**. It validates their declared relationship and renders a deterministic Markdown review. It never calls an OCR, language model, network service or ink detector, and never certifies text, ground truth, anatomical correspondence or independent evidence.

The included fixture is entirely synthetic. Its hashes, sites, boxes and strokes do not identify any real manuscript or result.

## Use

Requires Python 3.10 or later and only the standard library.

```bash
python transcription_review.py validate example.synthetic.json
python transcription_review.py report example.synthetic.json --output /tmp/review.md
python -m unittest discover -s . -p 'test_*.py' -v
```

Output files must be new, with an existing parent directory outside any Git worktree. stdout contains aggregate contract results only. Generated journals, reports and lexical results can contain sensitive readings and locations; keep them in the intended private workspace. Code and synthetic examples can be public. The validator does not change inputs, overwrite previous reviews or decide publication.

## Evidence contract

The root schema is `scrollprize.transcription-review.v1`. See the complete synthetic example for required fields. Unknown fields are rejected, so an extra `ground_truth` or training-label field cannot silently promote an observation.

| Object | Required meaning |
| --- | --- |
| `sources` | Visual raster ID, exact lowercase SHA256, native `[height,width]`, kind, declared physical-site ID, parent source IDs and relation. A regrown view retains the site's identity and names its parent view. |
| `lines` | A declared common reading orientation, the basis for segmentation, and ordered contiguous reading units. Units are hypotheses about segmentation, not automatically letters. |
| `observations` | One reviewer's visible strokes, raster hash, bounded native raster box, unit, common orientation, prior-verdict exposure and alternatives or explicit abstention. |
| `lexical_hypotheses` | A compatible form on an explicit contiguous span. Its only status is `lexical_hypothesis_only`; `ground_truth` and `independent_evidence` must both be false. |

All boxes are half-open native-source `[y0,x0,y1,x1]` coordinates **before** the orientation transform. `identity`, quarter turns, horizontal reflection and reflected quarter turns describe a common reading-orientation hypothesis; the tool records but does not execute them. A reflected quarter turn means reflect horizontally first, then rotate clockwise. An alternative orientation belongs in a separate line hypothesis. This bookkeeping does not establish that two regrown surfaces follow the same sheet or that their boxes are anatomically aligned.

Every unit needs an observation, including an explicit `abstain` when unreadable. An abstention has an empty alternative list and `allow_unknown:true`. A visual hypothesis needs one or more distinct single Greek-character alternatives. On an explicitly segmented one-character unit these are alternative letters. On an unsegmented block these are **conditional glyph hints only**, not a character sequence or a statement that the block contains exactly one letter; `allow_unknown:true` is then mandatory. A plausible shape must still be described in `strokes`; a word suggestion cannot be its visual source.

`length_min` and `length_max` retain uncertain segmentation. `length_max:null` means unbounded/unknown, not one missing character. For example a damaged region may have `length_min:0,length_max:null` with a conditional crescent-or-bar hint; it remains visible in the report and blocks lexical matching across that region. Do not replace such a region with one `?` or assert a single-character segmentation to make a dictionary word fit. Absence of recognizable ink does not establish a missing letter.

Reviewers are recorded separately, but there is **no vote count or automatic consensus**. Alternative letters are combined by union; unknown flags by logical OR. Thus a second reviewer who abstains cannot disappear behind a confident first reviewer. A reviewer who has seen earlier verdicts must set `prior_verdict_seen:true`; their independent production does not make their reading blind.

Sources with the same byte hash, same declared site, or connected ancestry are grouped as correlated. A second detector, contrast view or regrow of the same place cannot become an independent vote. Distinct groups still do not certify statistical independence. Cyclic ancestry and duplicate same-reviewer/same-source-byte annotations are rejected. The validator cannot detect dishonest source declarations or an undeclared common ancestor.

## Optional local lexicon

Supply a UTF-8 text file containing one Greek wordform per line. No lexicon is downloaded or fabricated. The caller fixes a contiguous half-open span, a common known orientation, and **one character per unit** before comparison.

```bash
python transcription_review.py suggest example.synthetic.json \
  --lexicon /tmp/user-wordforms.txt --line line-1 --start 0 --stop 5 \
  --limit 25 --output /tmp/lexical-hypotheses.json
```

The command refuses unresolved-length units, absent units, unknown orientation or mixed orientations. It accepts each lexicon form only if its normalized length equals the fixed span and each character fits a visual alternative or an explicitly allowed unknown. It performs no fuzzy matching, insertion, deletion, reconstruction of word boundaries or semantic completion.

Greek case and diacritics are normalized; ordinary, final and lunate sigmas are unified. A letter-bearing iota subscript is preserved as `Ι` rather than silently removed. Latin lookalikes, punctuation, multiword entries and nonstandard Greek alphabet symbols are rejected. Original dictionary spellings are retained with their normalized form. Exact deduplication, alphabetical order, a maximum of 100 results, the full matching-form count and truncation flag make the result deterministic. The exact lexicon bytes and canonical journal content are pinned by SHA256 receipts. Dictionary spelling is not evidence that accents or breathing marks were visible.

The result is a set of **compatible lexical hypotheses**, never words read on the image. A lexicon can contain an erroneous wordform, and a compatible word can be absent from the manuscript. A zero-result query proves neither blankness nor that the frozen visual alternatives were wrong. Revising alternatives after inspecting matches creates a new, exposed review; retain the original journal.

## Reuse the existing repository's evidence tools

This journal complements the following existing code at integration commit [`5afc846f015926497f296c95f1ba8e2db7580ade`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/5afc846f015926497f296c95f1ba8e2db7580ade):

- [`line_review/review.py`](https://github.com/JorisViaudQuantAI/Scrollprize/blob/5afc846f015926497f296c95f1ba8e2db7580ade/firstletters-colab/line_review/review.py) retains full-map review, unchanged native crops, explicit masks and physical sampling. Proposal slots are not letters. Its proposal score must not be promoted to lexical evidence.
- [`component_export/prepare_handoff.py`](https://github.com/JorisViaudQuantAI/Scrollprize/blob/5afc846f015926497f296c95f1ba8e2db7580ade/firstletters-colab/component_export/prepare_handoff.py) checks completed export receipts and payload hashes, external acquisition/context support, geometry declarations and matched model settings. Use its completed components and producer records before making a depth-comparison claim.
- [`geometry_review/audit.py`](https://github.com/JorisViaudQuantAI/Scrollprize/blob/5afc846f015926497f296c95f1ba8e2db7580ade/firstletters-colab/geometry_review/audit.py) distinguishes common chart coordinates from fixed physical anchors. A metadata pass does not certify a sheet; proximity to a crest alone cannot establish the same physical text.
- [`glyph_review/README.md`](https://github.com/JorisViaudQuantAI/Scrollprize/blob/5afc846f015926497f296c95f1ba8e2db7580ade/firstletters-colab/glyph_review/README.md) documents that the released glyph score is uncalibrated and training-overlapping on some public controls. It is not a character recognizer or an independent reading.

Use raw components, masks, same-sheet checks, scale and full-row context when available. The new journal intentionally **does not duplicate an image viewer or pretend to verify asset bytes**: it validates declared hashes and boxes, not source-file contents. Its output states `source_assets_verified:false` and `anatomical_alignment_certified:false`. Before trusting a real journal, compare its hashes to actual immutable files and inspect the original pixels.

## Concrete failure cases covered by tests

Tests cover duplicate evidence and source correlations; preservation of abstentions; rejection of lexical hypotheses promoted to ground truth; mixed orientations and gapped/variable-length segmentation; exact Greek normalization; deterministic local lexical matching. They test the bookkeeping contract, not whether the synthetic marks resemble Greek or whether any real letters have been recovered.
