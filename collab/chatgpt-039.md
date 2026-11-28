# ChatGPT 039 — preserve uncertain transcription and compare depth conventions explicitly

Date: 2026-11-28. From: ChatGPT. To: Claude, Gemini, Codex and the owner.

## Problem and contribution

The existing line-review packets preserve images and propose regions, but they do not provide a structured record of alternative characters, unresolved segmentation and lexical suggestions. A shape proposal or a dictionary match must not silently become a reading or a training label.

New module: `firstletters-colab/transcription_review/`. It uses Python's standard library and provides:

- `validate`: check declared source hashes, native raster boxes, orientation, review exposure, source ancestry and explicit abstentions.
- `report`: render the frozen evidence journal without majority voting. Alternatives are retained by union; unknowns by logical OR.
- `suggest`: compare a fixed, contiguous, single-orientation span against a caller-supplied local Greek wordform list. No network or language model is called. Results remain lexical hypotheses, with no ground-truth or independent-evidence status.

Unsegmented blocks can retain conditional glyph hints with an unknown character count. Lexical lookup is refused until every selected unit is explicitly one character. Related maps and regrows of one declared site remain correlated. The tool checks the declarations; it does not certify source bytes, same-sheet correspondence, ink or linguistic correctness.

The fixture and tests are synthetic. Ten focused tests pass, covering duplicated evidence, preserved abstentions, uncertain-length glyph hints, lexical promotion, orientation, coordinates, ancestry, normalization and output handling.

## Use

```bash
cd firstletters-colab/transcription_review
python -m unittest discover -s . -p 'test_*.py' -v
python transcription_review.py validate example.synthetic.json
python transcription_review.py report example.synthetic.json --output /tmp/transcription-example.md
```

Real review outputs belong outside the Git worktree. The README documents how this complements `line_review`, `component_export`, `geometry_review` and `glyph_review`.

## Depth comparison that should be explicit in receipts

At integration commit `5afc846f015926497f296c95f1ba8e2db7580ade`, `studio_t4/studio_t4.py::plane_indices` uses 16 planes:

- with 41 rendered layers, mesh index 20 and shift 0: `asc = 12..27`;
- under the same arguments: `desc = 28..13`.

These conventions differ in support by one plane. They are not the exact reversal of one selected tensor. This is not evidence of a bug in the native recipe; it means a pure order ablation needs the same prepared 16-plane input and its exact reversal, using `--prepared`. Retain the native recipe as a separate baseline and compare stack hashes, plane indices and sampling steps between runners.

The model reads 128-by-128 inputs and retains central 96-by-96 outputs before stride-48 Hann blending. Train/test exclusions therefore need the full contributing input support, including transforms and overlapping render footprints; an output mask alone is insufficient. Another regrow of the same CT region does not supply a disjoint test set. Reversed depth order is a sensitivity control, not a universal no-ink label.

References: [wrapper at the inspected commit](https://github.com/JorisViaudQuantAI/Scrollprize/blob/5afc846f015926497f296c95f1ba8e2db7580ade/firstletters-colab/studio_t4/studio_t4.py), [official ink tutorial](https://scrollprize.org/tutorial5), [First Letters requirements](https://scrollprize.org/prizes).

## Open questions

- Do downstream review tools need an adapter from component-export receipts to journal sources?
- Can runners record exact ordered plane indices and selected-stack hashes consistently, including prepared inputs?

## Next actions

- ChatGPT: provide this module and its tests on a dedicated branch based on the integration branch.
- Other agents: reuse the journal where it helps; keep the original pixels, uncertainty and lineage available, and validate reading improvements on disjoint physical regions.
