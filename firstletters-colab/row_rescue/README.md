# Published-line row-rescue: a **fourth** review proposal, not a text detector

**Decision on 2026-11-23:** A label-free physical-row diversity heuristic brings an overlooked second known text row into a four-band public review shortlist, **without dropping any of the first three scored bands**. It adds a second proposed viewing order; it must not replace full-map review, establish blankness, or count legible letters. This is retrospective development against previously seen sources, not a held-out validation.

The observed failure was concrete: the archived public PHerc. 306B four-best bands concentrated on its stronger first line and gave **zero** centreline coverage to the second published line. We asked whether a fourth proposal chosen for independently supported row diversity rather than score alone can expose the missed known line, under the **same four-proposal count**.

## Method and actual test

[compare.py](compare.py) operates on an archived `line_review` report's `top10_proposals`, using only score, polygon centroid, and physical length. Keep original top three. If their vertical centres span less than **3 mm**, select a fourth proposal at least 3 mm from their score-weighted centroid, maximizing score × physical length × Gaussian neighboring-proposal vote (1.25 mm sigma). Otherwise return the original fourth. No text boxes, target masks, or labels enter the choice. See [the pre-execution freeze](freeze.json) for the fixed selection, comparison, resource budget and stop criteria.

The [successful CPU GitHub Actions run](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45206993318) used the original source-verified public controls from `line_controls/prepare.py`, archived unchanged proposal lists and original bitwise-checked validity masks. The [immutable run artifact](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45206993318/artifacts/14833004749) contains the full JSON receipt (`row-rescue-result.json`, SHA256 `575ddfc6f69bcac6157011bf1ad48e3bd7eef28dcf47c80cd2cf696af54539f5`; ZIP SHA256 `5123e876ea8b561b8aaca68377b19419ec43753c3f3ce86dcdd33127dbea2645`). This was **one** completed control comparison, not two independent text sources on 306B.

| Published control, centreline coverage | Top 4 by score | Top 3 + row-rescue fourth |
| --- | ---: | ---: |
| 306B full support, line 1 | 88.93% | 88.93% |
| 306B full support, **line 2** | **0%** | **42.47%** (rank 8 inserted at slot 4) |
| 306B shuffle-common support, line 1 | 52.81% | 52.81% |
| 306B shuffle-common support, **line 2** | **21.82%** | **92.31%** (rank 6 inserted at slot 4) |
| 343 displayed positive, author boxes >=50% covered | 15 / 18 | 15 / 18 (unchanged list) |
| 306B author-described blank and depth shuffle | 0 proposals | 0 proposals |

**Trade-off:** the total fraction of valid map selected in the 306B full-support case increased from 4.71% to 5.66%; on the shuffle-common support from **24.96% to 38.22%**. An equal number of bands is **not** equal inspection area or speed. On the matched support the first line's coarse box coverage declined from 34.58% to 33.96%, although its centerline remained identical. No human-reader timing, false-positive reduction, new ink-model gain, or unseen-scroll improvement was measured.

## Reproduce

No checkpoint download, model inference, GPU, unread CT, or candidate disclosure is involved. Use Python 3.11 and NumPy 1.26.4, OpenCV 4.10.0-headless and Pillow 11.1.0.

```bash
python firstletters-colab/row_rescue/test_compare.py -v
python firstletters-colab/line_controls/prepare.py --output-dir /tmp/public-row-rescue-controls
python firstletters-colab/row_rescue/compare.py \
  --data /tmp/public-row-rescue-controls \
  --output /tmp/row-rescue-result.json
```

The public preparer acquires approximately 180 MB of externally version-pinned source material outside Git and checks its source digests. The comparator verifies the archived Git blob hashes `a614144401fefec6ca49fd124de2a28b2a28dc07` (previous results) and `a36c130e8d2d00c660ff2e5d6fbd8449d970645a` (protocol), plus every prepared case's score/validity hash. The CPU workflow passed five independent-of-scroll synthetic behavior tests and the real exact masked comparison. Original input arrays, new artifacts and any unread material remain outside this repository.

## Decision and integration boundary

The frozen development gate passed: on both variants of the single published 306B claim the second line's geometric centreline coverage increased while first-line centreline was preserved; 343's existing shortlist and both control outcomes were unchanged. This is **not a new model result**: 306B is a previously read/published scroll and may overlap detector training; 343 is an already-processed display signal, not comparable calibrated probability. It is a review-ordering heuristic, **not a false-positive test or a prize submission**.

**Next independent gate for Codex or Claude Code:** on a different *already published and independently annotated* scroll with a cached full 2D model map and a documented ordinary blank papyrus control, freeze the map hash and proposed shortlist before any new scores. Compare full-map inspection and top-four score-only vs this optional alternative using author-specified row/letter coverage and selected valid area. Record whether any true row is lost and whether nontext bands proliferate. A failure ends this heuristic; a success makes it an optional second view in the existing screeners. **Do not run a new GPU experiment just to test this reviewer.**

Original 306B authors: [pherc0306b-first-letters](https://github.com/SirProdigle/pherc0306b-first-letters); 343 authors: [pherc343-first-letters](https://github.com/Nieuwlaar/pherc343-first-letters). Public control sources are reused under their original CC BY-NC 4.0 data terms.
