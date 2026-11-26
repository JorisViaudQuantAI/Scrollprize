# Optional exact bound for the total missed count

## Result and status

The existing A02 synthetic fixture has two known-order strata of 20 units with
3 paired audits each, plus 2 unknown-order units that always receive both orders.
If every sampled known-order review is zero, the existing summed Bonferroni
upper bound is **26 missed units among 40** at 95% confidence. Direct inversion
of the worst-case joint zero-event probability gives **14 among 40** under the
same fixed stratified sampling design. The 2 unknown-order units are excluded
from this estimand; they are not extra zero-event observations.

This is an improvement to an optional **mathematical comparator**. It is not a
correction of an invalid existing bound: the existing Bonferroni bound is valid
and conservative. No admission gate, default survey reducer, A02 result, or
promotion rule is changed. The comparator always labels its output
`evidence_verified=false` and `promotion_allowed=false`.

All examples below impose zero outcomes for design comparison. They are not
observed papyrus results, new model inference, evidence of new letters, or a
validation of single-order sensitivity. There were **zero new model inferences**.

## What is estimated

Each eligible, physically deduplicated unit has a fixed binary outcome under a
frozen review recipe: whether using the selected single depth order would miss
qualifying detectable text that the paired comparison reveals. The target is
the **total count of these units** in the frozen known-order cohort.

For stratum h, let N_h be its population size, n_h its sampled paired-audit size,
and m_h the unknown fixed number of missed units. The total is M = sum(m_h).
The result assumes:

1. The population, strata, sample sizes, normal-order rule and review recipe
   were frozen before inspecting the outcomes used in this audit.
2. Sampling is simple random sampling without replacement within each stratum,
   and draws in different strata are independent. The randomization must be
   independent of the outcomes; searching seeds for convenient samples does
   not meet this premise.
3. Unit identities refer to disjoint, physically deduplicated units. Multiple
   views, rerenders or overlapping declarations are not independent extra
   units merely because their filenames differ.
4. Every sampled outcome is resolved under the frozen recipe and equals zero.
   For the A03 workflow, the entire planned cohort, receipts and paired reviews
   must also pass `evidence_bound_audit`, including `evidence_complete` and
   `cohort_complete`. Passing mathematical inputs to this module establishes
   none of those facts.

There is no IID assumption about pixels, image windows or the fixed labels
inside a stratum. Validity comes from the sampling design. Unknown normal-order
units follow a separate certainty paired path and do not enter this single-order
missed-count population. An all-unknown A02 cohort must retain its existing
absent bound; the mathematical empty-population value zero is not an admission
result.

## Exact calculation

For a fixed allocation m across strata, the probability of all sampled reviews
being zero is

$$
p_0(m)=\prod_h\frac{\binom{N_h-m_h}{n_h}}{\binom{N_h}{n_h}}.
$$

For a proposed total M, maximize this probability over the nuisance allocation:

$$
P_0^*(M)=\max_{\substack{\sum_h m_h=M\\0\leq m_h\leq N_h}}p_0(m).
$$

The comparator returns

$$
U_\alpha=\max\{M:P_0^*(M)\geq\alpha\},\qquad\alpha=1/20.
$$

Equality is retained. A count is excluded only when its maximum zero-event
probability is **strictly below** alpha. All probability comparisons use exact
`Fraction` arithmetic; decimal `.05` is interpreted as exactly `1/20`.

### Why the heap gives the maximum

Adding one missed unit to a stratum containing m missed units multiplies its
zero-event probability by

$$
r_h(m)=\frac{N_h-n_h-m}{N_h-m}.
$$

For fixed h this ratio is nonincreasing in m. Each stratum therefore offers a
nonincreasing list of multiplicative factors. Selecting the largest available
factor at each step merges those lists in decreasing order. The selected factors
form a prefix in each stratum, so the selection is feasible. An exchange with
any smaller selected factor cannot increase the product; this proves that the
merged prefixes maximize the product for every total M. Ties may be resolved
arbitrarily; the implementation uses stratum position for reproducibility.

With n=0 every factor is one, so all N entirely unobserved units are credited to
the bound immediately. This makes the arithmetic conservative and avoids a loop
over large nonaudited strata. It does not make an n=0 stratum eligible for the
current A02 sampler. With n=N any positive makes zero observations impossible;
a census has no capacity for hidden positives. Once every stratum's capacity
N_h-n_h is exhausted, all greater totals have probability zero. Empty strata
and an empty mathematical population are defined without division by zero.

For a fixed population with M greater than U_alpha, its actual allocation has
probability no larger than P_0^*(M), which is below alpha. Define the full
confidence procedure to return U_alpha after all-zero observations and the
trivial total population size otherwise. Its noncoverage probability is thus
below alpha for every fixed population. **This is a frequentist coverage
statement, not a posterior probability that M is at most U_alpha after seeing
zero.** Repeated cohorts, selective reporting, or stopping adaptively after an
apparently satisfactory result require a separately specified error-control
procedure. This comparator does not provide one.

The running time is O((U_alpha - sum(N_h for n_h=0) + 1) log H) heap operations,
in addition to exact integer arithmetic and input processing. Very large strata
with extremely small positive sampling fractions can still be expensive. No
large-population performance claim is made.

## Comparison with the current code

The baseline calls the repository's actual `survey.zero_upper_count(N,n,alpha/H)`
and sums the H stratum bounds. The first row is also checked against the current
`order_audit.audit_summary` on the existing A02 test fixture. Source SHA-256
hashes are recorded in `comparison_result.json`.

| Synthetic known-order design | Known units | Paired audits | Current summed Bonferroni upper | Exact joint total upper |
|---|---:|---:|---:|---:|
| Existing fixture: 2 strata of 20, 3 audited each | 40 | 6 | 26 | **14** |
| 1 stratum of 120, 18 audited | 120 | 18 | 17 | **17** |
| 2 strata of 60, 9 audited each | 120 | 18 | 36 | **17** |
| 6 strata of 20, 3 audited each | 120 | 18 | 90 | **17** |
| 12 strata of 10, 2 audited each after rounding | 120 | 24 | 96 | **13** |
| Strata 100/15 and 20/3 | 120 | 18 | 33 | **17** |

For the existing fixture, a maximizing allocation at M=14 is (7,7), giving
P0*=0.06293936595875654. At M=15, (8,7) gives
P0*=0.048414896891351186. These rational certificates bracket alpha=0.05.

The Bonferroni calculation offers simultaneous per-stratum coverage as well as
its derived total bound. **The new result covers only the total.** Its maximizing
allocation is a nuisance-parameter certificate, not an estimate of which strata
contain text. Per-stratum Bonferroni bounds must not be replaced with components
of that allocation.

The new total bound cannot exceed the sum of exact Bonferroni bounds: a total
above that sum forces at least one stratum above its bound, whose zero-event
probability is below alpha/H. Every allocation's full product is then below
alpha. The larger difference in highly stratified examples comes from avoiding
this conservative union-bound construction for a question solely about the
total; it is not a change to the observations or an assumption of equal risk.

### Prospective sample-size implication

These are **planning calculations**, with equal integer n in each equal-sized
stratum fixed before collecting outcomes. The target is to exclude M>=6 among
120 known-order units at 95% confidence **if** all sampled outcomes are zero.
They do not guarantee that a real experiment will observe zero.

| Equal strata | Smallest total audits, joint total bound | Smallest total audits, summed Bonferroni |
|---|---:|---:|
| 1 x 120 | 47 | 47 |
| 2 x 60 | 48 | 84 |
| 6 x 20 | 48 | 120 |
| 12 x 10 | 48 | 120 |

This suggests a way to retain useful scientific strata without making a total
risk criterion depend excessively on their number. It does not authorize
reducing the frozen audit fraction, changing an ongoing draw, or releasing
single-order inference into the standard production chain.

### Why pooling the samples is unsafe

Strata (N=100,n=1) and (N=20,n=20) contain 120 units and 21 audits in total.
Suppose the first stratum has 50 missed units and the second has none. The real
probability of observing zero is **1/2**. Pretending these were an SRS of 21 from
120 gives an upper bound of 14 and would exclude the true total 50 half the time.
The correct stratified joint calculation gives **95**, with maximum zero-event
probability exactly 0.05 at M=95. The joint algorithm preserves the heterogeneous
sampling design; it does not pool N and n into a single hypergeometric model.

## Verification and reproduction

`comparison_result.json` records **9 passing tests**, no failures or skips, and
the Python version and elapsed time of the run. The exhaustive test considers
**5,511 ordered plans**: H=0; H=1 with N<=9; H=2 with N<=6; H=3 with N<=4; H=4
with N<=2. Every legal n from 0 through N is included.

For each design, an independent oracle enumerates **every allocation** of
missed counts and evaluates `comb(N-m,n)/comb(N,n)` directly with rational
arithmetic. It checks the heap result at every attainable probability threshold
strictly between zero and one
and every interval midpoint, including exact equality and both boundary
certificates. Separate tests check unconditional noncoverage for all fixed
allocations in selected designs, monotonicity with more sampling and greater
confidence, order permutations, ties, census/empty/n=0 cases, a trillion-unit
n=0 stratum, invalid inputs, and the unsafe-pooling counterexample.

An independent agent also derived the greedy exchange argument and evaluated
all 5,511 small designs against a separate rational allocation enumeration.
That mathematical review found no discrepancy. This is code and mathematics
verification; it supplies no real audit outcomes.

From the repository root:

```bash
cd firstletters-colab/industrial_funnel
python -m followup_a03.run_comparison --out followup_a03/comparison_result.json
```

The command reruns the focused tests before writing the report. The optional
comparator remains separate from A03 admission and all production gates. A later
integration would need an explicit contract that accepts only a fully admitted,
complete, zero-event known-order cohort and preserves all existing missingness,
unknown-order, receipt, cost and no-promotion conditions.

## Independent review of the A03 producer correction

`test_admission_policy_compatibility.py` is a separate admission regression; it
is not included in the 9 statistical tests recorded in `comparison_result.json`.
It checks a concrete compatibility risk introduced by revalidating producer
envelopes against the shared policy: legitimate A02 arms must remain admissible
as experiments, and costs from earlier partial attempts must still be counted.

The test creates complete synthetic REGISTER/GEOMETRY/CONTROL/COARSE DAGs with
one PARTIAL attempt declaring 3 CPU milliseconds followed by COMPLETE declaring
4, plus 3 declared CPU milliseconds for the ancestors. These are synthetic receipt
values, not measurements of hardware use. It checks four arms through both the actual shared
`FileReceipts` reader and a reader that performs no semantic validation:

| Arm | COARSE state | Standard RANK relation | Declared CPU cost (ms) |
|---|---|---|---:|
| Legacy paired | COMPLETE | READY_FOR_OPERATOR | 10 |
| A02 known-order single | EXPERIMENT_COMPLETE | Parentage rejected | 10 |
| A02 known-order audited pair | EXPERIMENT_COMPLETE | Parentage rejected | 10 |
| A02 unknown-order certainty pair | EXPERIMENT_COMPLETE | Parentage rejected | 10 |

Both readers produce identical inspections. All valid cases retain
`costs_complete=true`, `halted=false` and `dispatch_authorized=false`.
This additional test passed. It complements `test_producer_interop.py`, which
checks that incompatible producer envelopes are rejected. If validation fails,
the `spent` vector may contain only earlier validated costs; the existing
`costs_complete=false` and halted batch correctly prevent treating that partial
vector as a complete or budget-safe total.

```bash
python -m unittest followup_a03.test_admission_policy_compatibility -v
```
