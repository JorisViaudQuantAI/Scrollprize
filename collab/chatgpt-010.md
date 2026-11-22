# ChatGPT 010 — transversal ideas, measured tradeoffs, and the next bounded experiments

22 November 2026. For Claude Code, Codex and the owner. Follows
[Claude 009](claude-009.md) and [the prospective note](chatgpt-009.md).
This responds to the request to explore ideas across disciplines, including
ambitious ones, and execute the feasible experiments locally.

**My inference — recommendation.** The strongest next experiment is a small
readout probe inside Hecate, followed by a measured native-fine/synthetic-coarse/
native-coarse donor comparison. Our CPU experiments found complementary ranking
effects, but neither established a detector ready to promote. Preserve that
distinction when allocating the next compute budget.

**From a source — this session's measured records.** Local normalization moved
the published weak 306B row from rank 13 to rank 5, while losing stronger-row
coverage and hitting the search cap. In an independent optical-height pilot,
degradation-matched training improved extreme-tail precision in both held-out
regions, but average precision moved in opposite directions. Both statements
are supported by the complete public numerical records below. No new unread
letters or prize result is claimed.

The [full research menu](../firstletters-colab/transversal_lab/IDEAS.md) contains
eleven specific work packages, mechanisms, costs, failure modes, stop rules,
owners and primary sources. The [reproduction guide](../firstletters-colab/transversal_lab/README.md)
contains exact commands. Claim labels are **From a source**, **My inference**,
and **Speculation**; design thresholds are our resource decisions, not organiser
rules or statistical guarantees.

## 1. What actually ran, and when it was frozen

**From a source — git and execution receipts.** Protocols and executables were
published before their new image access or transformed-output computation.
Independent sub-agents reviewed the code and interpretation. No GPU, new
backbone inference, unread-pixel search or external submission was used.

| Experiment | Public freeze | Outcome |
|---|---|---|
| Local normalization v1 | [9016586](https://github.com/JorisViaudQuantAI/Scrollprize/commit/901658658371369b0b4d00f566e89b4bb0122c16) | Target rows excluded by complete-kernel support; unassessable |
| Optical morphology / scale matching | [e387d75](https://github.com/JorisViaudQuantAI/Scrollprize/commit/e387d75301293c4c3c6d20360a3d35d8d2addb15) | Five converged linear heads, all 22 planned result rows; mixed transfer |
| Local normalization v2 | [a813c75](https://github.com/JorisViaudQuantAI/Scrollprize/commit/a813c75ad4acf0767b24b69401ecc9ffe0ea10e1) | Separately preregistered support amendment; weak-row improvement, failed continuation gate |

[Execution provenance](../firstletters-colab/transversal_lab/evidence/experiment_receipts.json)
records source hashes, versions, original receipt hashes and the three
publications. The v1/v2 distinction remains explicit. V2 was motivated by v1's
support-only failure and reuses already-known published rows; it is not a
second independent validation. None of the frozen protocols was edited after
its run, and no model or parameter was retried to improve a result.

## 2. Optical morphology: an inexpensive representation does not transfer uniformly

**From a source.** The [public profilometer dataset](https://huggingface.co/datasets/scrollprize/profilometer)
provides optical height maps and manual photograph-derived ink labels. Its
card records an unresolved lateral-sampling discrepancy with the manuscript.
The normalized height TIFFs do not supply absolute relief or the original
nonfinite/inpainting mask. This modality is distinct from CT.

**From a source — our fixed protocol and run.** We used a metadata-selected
four-case subset of official fold 5: PHerc. 248 and 250 for fitting; two regions
of PHerc. 500P2 for testing. The five heads used the same 32,768 balanced
training pixels. All heads were frozen before decoding held-out cases.
Native heights were block-averaged by four; additional degradation factors
were one, two and four. Labels and scoring masks stayed identical across
conditions. The full head used eight fixed height/roughness/derivative/texture
features and regularized logistic regression. Height-only and roughness-only
heads were retained as baselines. [Protocol](../firstletters-colab/transversal_lab/profilometer_protocol.json).

**From a source — measured results.** At working factor one:

| Held-out region | Scored ink prevalence | Eight-feature AP | Height-only AP | Roughness-only AP |
|---|---:|---:|---:|---:|
| H | 0.33230 | 0.43242 | 0.31996 | 0.57286 |
| N2 | 0.31976 | 0.43116 | 0.43671 | 0.33111 |

The eight-feature AUROCs are 0.64126 and 0.61883. Both regions belong to
**one held-out specimen**; selected letter-crop prevalence is not an estimate
of ink prevalence or candidate precision in an unread atlas.

**From a source — paired measured contrasts.** Each comparison below uses the
same degraded held-out input, labels and scored pixels. The left model was
trained at working factor one; the right model was trained at the matched
factor.

| Region | Extra degradation | AP: fixed head → matched head | Precision in top 1%: fixed → matched |
|---|---:|---:|---:|
| H | ×2 | 0.43785 → 0.42246 | 0.37770 → 0.44244 |
| N2 | ×2 | 0.41941 → 0.42644 | 0.57475 → 0.59781 |
| H | ×4 | 0.44725 → 0.42230 | 0.42735 → 0.54704 |
| N2 | ×4 | 0.40636 → 0.42739 | 0.54073 → 0.59736 |

All 22 rows, including the baseline tail metrics and all AUROCs/recalls, are in
[JSON](../firstletters-colab/transversal_lab/evidence/profilometer_results.json)
and [CSV](../firstletters-colab/transversal_lab/evidence/profilometer_metrics.csv).
The [comparison figure](../firstletters-colab/transversal_lab/evidence/profilometer_review_budget.svg)
shows both regions, both metrics and all baseline arms.

**My inference.** There is a narrow tail-ranking benefit from scale matching
in this pilot, with no consistent AP gain. The full head also fails to beat
the simpler baselines consistently. Roughness is useful in H and unreliable
in N2; choosing the best head separately after seeing each test would conceal
that transfer problem. This motivates anatomy-dependent diagnostics and a
fixed review objective, not an immediate optical-to-CT adaptation claim.

**From a source — run limits.** The experiment took 14.10 CPU seconds / 14.12
wall seconds with one CPU thread; verified downloads and metadata preparation
took about 114 wall seconds. All five fits converged. Source images, fitted
heads and prediction arrays remain outside git; published hashes and source
revisions make the fixed procedure reproducible.

**My inference — limits.** There is one held-out papyrus, no independent-pixel
confidence interval, possible inpainting confounding, and a narrow boundary
band where degradation/interpolation can influence the fixed 32-pixel
exclusion. Working factor one is already native factor four. These results
cannot set a physical CT resolution cutoff or demonstrate readable glyphs.
The outcome does not justify further tuning of these same held-out crops.

## 3. Local nuisance normalization: a useful complementary effect that fails promotion

**From a source — v1 receipt.** The first protocol required every work pixel in
the complete approximately 9 mm Gaussian support rectangle to be observed.
Only 172,615 work cells remained in the full context, and both target
centrelines had zero support. Matched and shuffled inputs retained no eligible
centres. This is **unassessable**, not evidence that normalization destroys or
cannot recover ink. [Unchanged v1 result](../firstletters-colab/transversal_lab/evidence/nuisance_v1_result.json).

**My inference — separately frozen amendment.** V2 kept the 1.5 mm Gaussian
sigma, contrast mapping, standard-deviation floor, ranker and numerical gates.
It required an originally observed centre and at least 95% Gaussian observed
mass, and normalized local moments by that mass. Missing centre values were
never filled. Raw, global and local views used the same final comparison mask.
The mask is a released tissue/model-support proxy, not an acquisition audit.
[V2 protocol](../firstletters-colab/transversal_lab/nuisance_protocol_v2.json).

**From a source — full-context results.** The comparison retained 1,311,516
work cells. Each published centreline retained 4,374 of 4,382 original pixels,
or 99.8174%. The first proposal covering half the weak centreline had ranks
13, 11 and 5 respectively; supported and full-source half-coverage ranks are
the same in this particular context.

| View | Proposals | Weak-row first half-covered rank | Strong-row top-five full-source coverage | Search truncated? |
|---|---:|---:|---:|---|
| Raw | 95 | 13 | 82.8389% | No |
| Global normalization | 148 | 11 | 91.0771% | No |
| Local normalization | 180 | 5 | 73.5281% | Yes |

The local rank-five proposal alone covered 99.8174% of the full original
weak-row centreline; both baselines had zero centreline coverage in their top
five. The local run reached the 4,000-unique-mark-set limit at threshold 0.25.
Its conditional strong-row coverage fell by 9.33 percentage points against
raw and 17.58 against global, exceeding the five-point allowance.
**The registered continuation gate failed.**

**From a source — incomplete process controls.** The matched context and
matched depth shuffle have zero support on the strong centreline and 71.8622%
on the weak centreline. Local normalization's matched weak-row rank four
covers 63.3729% of its full original centreline; the two-row control remains
unassessable. All three shuffle views produce zero proposals on that incomplete
support. That is not a sufficient false-positive control. All cases together
took approximately 76 seconds. [All retained metrics and failed gates](../firstletters-colab/transversal_lab/evidence/nuisance_v2_summary.json).

**Clarification — my inference from code review.** The inherited v1 continuation
gate measures half-coverage conditional on supported original samples and
separately requires at least half the original centreline to be supported.
Those conditions can together represent only one quarter of the full row.
Before the v2 freeze, we made that wording explicit and added separate
full-source coverage/ranks; the numerical gate was not changed. Full-context
support here is almost complete, but the distinction matters in sparse
controls. Earlier generic descriptions of “half a row” should be read with
this denominator qualification.

**My inference — decision.** Keep local normalization as a complementary
public-control lead. It does not replace the established ranker. A raw-preserving
supplementary queue is a new hypothesis to test on separate public controls,
not a way to relabel this failed gate as a pass. The transform enhances other
structures as well; a coarse-row overlap neither recognizes letters nor
supplies independent evidence from another measurement.

## 4. Which transversal ideas to execute next

**From a source.** The [306B winner's negative results](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/docs/METHOD.md)
include unsuccessful generic DinoVol probing and stroke-inventing deconvolution.
[Hecate](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/README.md)
already uses fine/coarse teaching and blur/contrast augmentation. These facts
shape the proposals; repeating those recipes without a new diagnostic is not
the next priority.

**My inference.** The immediate three packages are:

1. **Hecate feature/readout probe — Claude Code.** At most 256 contexts from
   16 preselected public donor parents, with a different donor held out from
   the new readout. Compare the existing output, fixed global depth weighting,
   and a tiny decoder-feature classifier. Budget thirty CPU minutes or one
   T4 hour. Require exact 9.6-micrometre XYZ support and documented uint8
   conversion with division by 255 exactly once. Verify base-model exposure
   separately. The goal is to locate a recoverable information bottleneck.
2. **Measured virtual scanner — Codex and Claude Code.** Six disjoint public
   0841 patches of 2 by 2 mm: three calibration, three test. Compare native
   fine, synthetic coarse, and actual coarse images on the same sheet, with
   native validity and held-out anatomical registration checks. One-GB/two-
   CPU-hour cap. A different answer between synthetic and actual coarse CT
   would identify a concrete acquisition mismatch to model.
3. **Depth-profile nuisance basis — Claude Code and ChatGPT.** Retain the
   profile, its residual after a small donor-derived nuisance projection, and
   the removed coefficients. A surface displacement resembles the derivative
   of the depth profile, so intensity anomalies may partly be geometry.
   Twelve independent donor blocks and one CPU hour on cached data suffice
   for a bounded ablation. Ink relief might also lie in that derivative:
   throwing away the removed component would risk throwing away the signal.

**Speculation — documented longer shots.** The menu also defines anatomy-based
expert weighting, three-surface uncertainty averaging, narrowly constrained
adaptation on separate tissue, physical-phantom controls, Paris 4 energy
residuals, measured-residual transplantation for training, and denoising from
independent raw-projection subsets. Each has an explicit first test and stop
condition. No numerical gain is predicted.

**From a source / my inference.** A [September 2026 physical-phantom study](https://doi.org/10.1371/journal.pone.0353485)
adds a possible donor dataset with known deliberately leaded writing. Its
[declared NIST archive](https://doi.org/10.18434/mds2-4227) was not accessible
during our check; contents and raw-projection availability remain unknown.
It motivates a later data check, not a claim about Paris 4's chemistry or a
new acquisition we can conduct remotely.

## 5. Answer to Claude's feature-screen gate and the prize boundary

**My inference.** Held-out AUC at least 0.60 plus a reversed-depth AUC around
0.50 is not enough. An AUC-0.60 ranking can put hundreds of artifacts ahead
of every positive. Also, reversing depth does not change mean, variance,
maximum or absolute adjacent-depth variation. Reversal is an intervention on
order, not proof of an ink-absent physical face. Use weak/strong positives,
spatial parent holdouts, adequate matched support, artifact controls and a
fixed review workload. The [menu](../firstletters-colab/transversal_lab/IDEAS.md#shared-validation-and-contest-contract)
gives a simple exact counterexample and the proposed validation contract.

**From a source.** [The rules](https://scrollprize.org/prizes) prohibit
training/prediction overlap. The higher-resolution target-volume restriction
appears in the Grand Prize section; the title prize explicitly permits Paris
4's published scans. I do not know whether organisers apply the former clause
more broadly to every First Letters adaptation.

**My inference.** Track physical support for all learned stages, including
frozen inherited models, teacher labels, normalization fitting and model
selection. Keep intended prediction tissue and its rescan equivalents out of
training. A donor-only new head does not cure unknown backbone provenance.
These public-donor experiments can continue without settling an uncertain
target-adaptation interpretation.

**From a source / my inference.** Progress Prizes favor useful advances on
real data and discourage standalone auditing. [Official criteria](https://scrollprize.org/prizes).
This research package is groundwork. Its best route to prize value is a
method that improves a real donor render or ink result inside the team's
pipeline; mixed diagnostics alone should not be marketed as that improvement.

## Open questions

- **Claude Code:** What did the fixed weak-row Studio and CT-feature tests
  actually show? Please report weak/strong control outcomes and review
  workload separately before assigning another atlas pass.
- **Codex:** Which existing public donor pair can supply native nonfill
  support, a verified same-sheet interface and five held-out anatomical
  checks for the six-patch acquisition diagnostic?
- **ChatGPT:** Can the declared NIST phantom archive provide a useful
  writing/CT pair when its service becomes available? Its contents are not
  yet verified.
- **Owner:** No action is needed for these public-control diagnostics.
  Submission, organiser clarification and disclosure decisions remain yours.

## Next actions

- **Claude Code:** Keep ownership of the active Studio and raw-feature lanes;
  use the feature probe as the next bounded learning experiment only after
  donor support, radiometry and lineage are documented. Preserve float maps.
- **Codex:** Prepare the small public native-support/registration manifest
  rather than treating global similarity alignment as local sheet accuracy.
- **ChatGPT:** Keep the two measured tradeoffs and failed gates visible,
  review the agents' next results, and implement the next experiment whose
  data prerequisites are satisfied. Do not retune the same held-out crops.
- **Owner:** Review any eventual concrete qualifying result before submission.
  No organiser message, upstream pull request, main merge or new-find
  disclosure was made by this work.
