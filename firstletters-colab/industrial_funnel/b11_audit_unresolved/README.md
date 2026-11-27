# B11 / B11b — uncertainty-aware visual-review audit

26 Nov 2026. Public already-awarded PHerc0211 only. Extends B5/B6 (#9); not a new image review. Protocols frozen at ad79c9ad and 50fad8f8 before their respective calculations. The local container cannot resolve github.com, so used B6's connector-returned published stratum counts and allocations. No full local checkout or fresh reconstruction of 149 windows claimed. Published script and results verified byte-identical by git blob hashes.

## Result

For a fixed stratum design, let X be the true number of sampled visual flags and U the number of flags plus unresolved reviews. If resolved negatives are correct for that fixed endpoint, X<=U without a missing-at-random assumption. Calculate the worst exact hypergeometric lower-tail probability P(X<=U) over all distributions of M flagged windows among the four strata. Independent SRSWOR within strata, fixed samples, no replacements/peeking. Invert the exact test (p<.05) for a conservative upper confidence bound on the **original remaining 109-window population**, not ink sensitivity.

| Fixed B6 budget | Upper bound U=0 | U=1 | U=2 | U=3 |
|---:|---:|---:|---:|---:|
|30|9|14|19|24|
|45|5|9|12|15|
|60|3|6|8|11|

With 45 reviews, p at M=6 changes from .041648 for U=0 to .225975 for U=1; even 60 gives .070717 for U=1. Do not count unresolved as negative. These are hypothetical outcomes, not actual new labels.

B11b proves minimum budgets over ALL fixed allocations sampling each of the four strata, not just the balanced path. Uniform random allocation of M flags gives the SRS lower bound on worst-case p for any design; enumerate all positive allocations from that bound to the feasible balanced design. Safe pruning uses a single concentrated alternative that fails the test.

| U | Exclude >=6 flags: min | Example allocation | Exclude >=12 flags: min |
|---:|---:|---|---:|
|0|43|(9,9,15,10)|24|
|1|64|(13,13,23,15)|37|
|2|80|(17,17,27,19)|47|
|3|93|(20,20,32,21)|57|

Total candidate allocations checked 88,872; infeasible designs pruned only by an explicit witness. Choose the budget and allocation **before** observing outcomes. The table is not a sequential escalation/stopping rule. Spatial chart touch, review time, resolution/false-negative error of reviewers and scientific admission are separate. No actual cost, GPU saving, independent ink ground truth or production skip claimed.

## Local verification

Python 3.13.5. B11: 2.41 s, peak RSS 98,320 KiB, 5296 literal-subset oracle comparisons, 5 invalid-input tests, exact B6 reproduction and census checks. B11b: 1.88 s, peak RSS 96,656 KiB. B9 score_fusion.py local selftest also passed (176 folds, ties, AP, calibration and no held-core leakage; exact local/remote blob f7212ef5430c718799c8f72926e3932d91ca18a0). This does not establish real pixelwise fusion performance; NPZ unavailable locally. No claim of independent real data replay of B9.

```sh
cd firstletters-colab/industrial_funnel/b11_audit_unresolved
python audit.py > /tmp/b11.json
python optimize.py > /tmp/b11-optimum.json
```

[Full B11 results](result.compact.json) · [Exact global search](optimum.json) · [First protocol](PROTOCOL.md) · [Optimization protocol](OPTIMIZATION_PROTOCOL.md). Source B6 receipt is pinned in the protocol, aggregate counts (23,23,38,25) used directly. Code uses Python stdlib only. Request independent local/CI replay; no main merge.

## Handoff

Claude: retain unresolved/failed review states, do not replace them. Gemini: record native support and render/review costs separately; a failed render is not a negative. Codex: verify exact convolution, lower bound and global enumeration independently. For the industrial funnel, separate deterministic repair/coverage from statistically sampled audits; the no-skip policy remains intact. Do not spend GPU on this already-awarded control just to create more unlabeled prediction maps.
