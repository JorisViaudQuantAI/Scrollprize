# Retrospective ablation: simpler rules select the same four bands

Date: 2026-11-23. **From a source — [measured receipt](result.json):** on all
five frozen public proposal lists used in [ChatGPT 029](../../../collab/chatgpt-029.md),
two simpler selectors give exactly the same indices as the published row-rescue
rule. Both preserve the top three and the same fixed 3 mm eligibility condition;
one then chooses the longest eligible proposal, the other maximum score × length.
No parameter was changed or selected in this retrospective arithmetic check.

| Case | Published rule | Longest eligible | Score × length eligible |
| --- | --- | --- | --- |
| 306B full support | 1, 2, 3, 8 | 1, 2, 3, 8 | 1, 2, 3, 8 |
| 306B matched support | 1, 2, 3, 6 | 1, 2, 3, 6 | 1, 2, 3, 6 |
| 343 | 1, 2, 3, 4 | 1, 2, 3, 4 | 1, 2, 3, 4 |
| Blank and shuffle | Empty | Empty | Empty |

The table uses **one-based** ranks; the JSON records **zero-based** indices.
The [script](run.py) independently expresses the selection arithmetic rather
than importing the production selector. It reads only the archived proposals,
declared spacing and source/freeze files. It verifies the two original Git blob
identities and records all consumed SHA256s and its own hash. It makes no
network request, reads no score map or validity mask, and performs no CNN work.

**My reasoning:** identical selected lists imply identical coverage and selected
area under any fixed evaluation mask. **No coverage was recomputed here.** This
is not a second independent replay of the reported masked coverage measurements.

**From the new arithmetic:** in full-support 306B, the original fourth proposal
is already **8.628965 mm** from the top-three weighted centre, satisfying the
3 mm condition. Its replacement is **18.239719 mm** long versus **8.548704 mm**
for the original fourth. The matched-support replacement is **34.938618 mm**
versus **10.727254 mm**. The chosen proposal contributes about **69.09%** and
**77.27%** of its own neighborhood vote, respectively. Proposal votes are not
independent model evidence.

**My inference:** the reported recovery establishes a useful shortlist
replacement on one known public context, but these cases do not demonstrate
that the Gaussian neighborhood vote helps. They also do not isolate diversity
alone as the explanation in full-support 306B. The earlier result, its selected
area trade-off and its development gate remain unchanged. This check introduces
no new efficacy gate and does not select a new production algorithm.

Further row-selector adoption still requires a different already-published
full map with independently documented positive and ordinary-blank annotations.
Neither the same 306B field under a different mask nor a small partial-glyph
control supplies that validation. No qualifying reading or prize claim follows.

## Reproduce

Python standard library only. Use a fresh output filename; the script refuses
to overwrite an earlier receipt.

```bash
python firstletters-colab/row_rescue/ablation_review/run.py \
  --output /tmp/row-rescue-ablation-result.json
```

This is an appendix to the next geometry handoff, not a new numbered experiment
or a request for GPU work.
