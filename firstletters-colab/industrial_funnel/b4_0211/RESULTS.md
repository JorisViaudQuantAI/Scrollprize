# B4 — audit of Claude's newly published PHerc0211 benchmark

Executed 2026-11-24 on public, previously awarded PHerc0211 only.
Frozen code commit: 779ebf0d5767709b26ce366a2e7f265106460958.
Actual GitHub Actions run: https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45457378308
Eight CPU synthetic/adversarial tests pass; the actual 149-window input/key and
nine two-order PNG pairs match their pinned Git SHA1 source blobs.

## 1. Actual frozen review-population replay

Claude's windows.json contains 149 uniquely ranked windows, 293.369 cm2.
The exact sealed 40-window review is the top25 and the 15 separately drawn
ranks 27,29,37,43,52,64,76,88,91,108,122,127,129,138,144.
Top25: 52.222 cm2, three visual flags at ranks 2,6,7.
Random15: 29.469 cm2, zero visual flags.
Unreviewed 109: 211.678 cm2, NO FINE-REVIEW OUTCOME.

Of the 149 entries, 54 (103.256 cm2) have line=null and ranking
sentinel lr=-99. These are not verified blank. Six were independently
selected in the 15-window low-rank random review; 48 are unreviewed.

The exact one-sided 95% SRSWOR hypergeometric zero-event bound on the
remaining 124 lower-ranked windows is 21 additional visual flags,
not zero. Conditional on the three selected review flags, the top25
could therefore account for only 3/(3+21)=12.5% of all flags consistent
with that audit at the stated confidence. This is NOT an ink recall
estimate. It assumes the stated randomization and correctly counted
visual flags; one flagged result is weak, and two were seen before
unblinding. A descriptive Fisher 2x2 probability for three of top25
versus zero of random15 is 0.2327935.

If no flags are observed, auditing a simple random sample from the 124
requires 118 reviewed units to exclude even one missed flag at the
same exact one-sided 95% level, 78 to exclude >=3, 48 for >=6 and
27 for >=12. This illustrates the poor power of a small zero-event
survey at very sparse yield, not a justification for skipping.

## 2. Actual paired-order public maps

On the nine independent preselected winding pairs with common published
valid/parallel pixels and uint8 >=179 activation threshold, 3/9 exhibit
opposite signed left/right activation dominance (>=25 activations in
both halves, support >=100 and dominance magnitude >=0.10):

| Winding | Whole desc | Whole asc | Left dominance | Right dominance | Whole Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: |
| fw014 | 2,632 | 1,624 | +0.6817 | -0.3018 | 0.0462 |
| fw030 | 7,606 | 17,291 | -0.9183 | +0.8091 | 0.00753 |
| fw033 | 11,062 | 20,327 | -0.9112 | +0.5294 | 0.01196 |

The other six do not meet the frozen opposite-halves definition.
For fw033 specifically, left desc/asc activation counts are 798/17,169,
while right desc/asc are 10,264/3,158. The whole-map single score
would hide which part is visible under each order. Visual inspection of
the original fw033 maps supports this regional model-activation pattern;
it does not confirm any new letters.

The original published PNGs were spatially reduced and quantized; these
measurements DO NOT certify original float32 probability, actual ink,
anatomical sidedness, cross-manuscript transfer or a safe skip rule.
The valid/parallel masks are useful but do not certify every native
CT plane's acquisition support.

## 3. Scientific and engineering decisions

- DO NOT promote the top25 (or a fixed top sixth/half) as a universal
  high-recall stop rule from this positive, text-selected and partially
  unreviewed single-scroll cohort.
- Treat null line scores as ranker abstentions; maintain a dedicated
  unknown/exploration stratum with a frozen audit.
- Avoid global depth-order asymmetry as an exclusion rule for multi-wrap
  charts. Use region-wise evidence and inspect both orders when feasible.
- Keep coarse ranking as a prioritization tool only, not a blank verdict.
- R1/R2 industrial validation still needs the same native coarse and
  fine float32 footprints, support/recipe lineage, independent labels
  across held-out scrolls and actual end-to-end CPU/GPU/human costs.

No new CT acquisition, CNN forwards, unread locations, prize candidate,
production promotion or measured GPU savings in this contribution.
The quantitative receipts are in ci_summary.json. The result follows
the already-failed Adaptive-071 50%-cutoff gate and does not overwrite it.
