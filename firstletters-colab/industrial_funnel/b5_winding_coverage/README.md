# B5/B6 — public 0211 chart coverage and finite-population audit design

Scope: already-awarded PHerc0211 only. Source `1770a7b9f223a016167fda440f1867848373119d` (Claude-029); identical input blob pins as B4. No unread material, new inference, visual labels, cost measurement or ink-recall claim.

## B5: actual chart coverage

The top 25 windows touch 20/60 windings (4 complete). The published sealed 40-window review touches 30/60 (6 complete); 63 windows, 122.129 cm² summed valid-window area, belong to the 30 entirely unreviewed winding charts. 25 of those 63 have no line score. Three untouched winding charts have no scored window whatsoever. The 40 reviewed windows represent 27.1091% of the summed within-winding x-interval union, not 27% of physical papyrus. The chart spans do not overlap in the published 149-window partition. None of these statistics identifies an anatomical sheet or proves the absence/presence of ink. One additional review per untouched winding (30 total) guarantees chart touch only, not full chart coverage.

## B6: design for a new blinded audit, not an executed review

Freeze the remaining 109 windows after the existing 40 reviews. Partition by whether the winding had been reviewed at all and whether the line score abstained. Counts (touched-scored, touched-abstain, untouched-scored, untouched-abstain): (23,23,38,25). Within each stratum independently draw a uniform sample without replacement. Label each draw with the same prespecified blinded visual-flag rule; never replace a failed or difficult review, never stop early or adapt the allocation. Report unresolved separately, never as negative.

For fixed stratum sizes N_h and draws n_h, maximize the zero-event probability product C(N_h-m_h,n_h)/C(N_h,n_h) over all distributions of m flags across strata. The decreasing marginal ratio (N_h-n_h-m_h)/(N_h-m_h) makes greedy allocation exact; the implementation compares rational probabilities using BigInt and enumerates every feasible four-stratum allocation for each fixed budget/target. The oracle checked 1316 small cases by exhaustive enumeration.

| Additional reviews | Optimal allocation for target 6 | Worst P(zero given >=6 flags) | Reject at 5% if zero |
|---:|---|---:|---|
| 30 | 6,6,11,7 | 0.148608 | No |
| 45 | 9,10,16,10 | 0.041648 | Yes |
| 60 | 13,13,20,14 | 0.008045 | Yes |

45 additional zero-flag reviews under that exact design would exclude >=6 residual visual flags, NOT establish high ink recall. The simple random sample comparator needs 42 reviews to exclude six; stratification ensures the four explicit risk buckets are represented, not every winding. Even an SRS needs 104 of 109 reviews to exclude a single flag. The initial three weak/partly pre-exposed visual flags cannot serve as independent ink ground truth. Preserve the failed 071/B4 cutoff; no production exclusion.

## Reproduce

Requires Node.js (built-in fs and crypto only). From the repository root:

```sh
BASE=firstletters-colab/results/pherc0211_calibration
B=firstletters-colab/industrial_funnel/b5_winding_coverage
node "$B/coverage.mjs" --self-test
node "$B/coverage.mjs" "$BASE/maps_export211/windows.json" "$BASE/look/P1/KEY_sealed.json"
node "$B/stratified_design.mjs" "$BASE/maps_export211/windows.json" "$BASE/look/P1/KEY_sealed.json"
```

Receipts: coverage_result.json (6/6 synthetic checks plus real cohort); stratified_design_result.json (1316 oracle comparisons plus actual stratum-size design). Calculations executed in the JavaScript runtime on connector-returned pinned source blobs, not GitHub Actions or a local clone. A fresh Node CLI replay is still requested. No source images, candidate coordinates, new CT or GPU jobs published.
