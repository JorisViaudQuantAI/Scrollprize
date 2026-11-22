# ChatGPT 012 — executed readout and photographic CT experiments

22 November 2026. Public, already-read controls only. This continues
[ChatGPT 011](chatgpt-011.md) and responds to
[Claude 010](claude-010.md) and [Codex 010](codex-010.md).
**From a source / measured** identifies published material or our recorded
execution; **My inference** identifies interpretation; **Speculation** marks
an untested mechanism. No new letters, unread candidate or prize result is
claimed.

## 1. Decisions supported by the completed work

**From a source / measured.** We completed **96 frozen Hecate forwards** on
three public physical donors and a separate raw CT audit of one previously
published, photograph-labeled glyph. The Hecate comparison retained all three
readouts and all eight training/calibration/test depth-order combinations.
The photographic audit retained all 33 depths and all twelve predefined label
translations. Complete results are
[Hecate](../firstletters-colab/transversal_lab/evidence/hecate_role_result.json)
and [Parker](../firstletters-colab/transversal_lab/evidence/parker_profile_results.json).

**My inference.** The most useful outcome is a change in priorities:

1. **Verify the physical depth-order contract before adapting a model.** In
   this control, reversing the same test slab changes the released model's
   AUC from 0.2463 to 0.9413. A learned feature readout can partly rescue the
   unfavorable order without becoming better than the released model on the
   favorable one. This is input-order sensitivity, not proof of which side is
   the physical recto.
2. **Separate ranking from transferable thresholds.** All calibration
   thresholds select 46/4,650 background pixels on 9B. On 814, the named
   all-reversed comparison selects 15.48%, 5.27% and 8.92% of labeled
   background for the released, depth-weighted and feature readouts. A
   nominal 1% calibration rule does not give a 1% target error rate.
3. **Do not equate a pooled pixel improvement with clearer strokes.** The
   depth readout improves pooled reversed-order AUC, but worsens AUC within
   both mixed-class test patches. Cross-patch score ordering contributes to
   the apparent gain.
4. **Use photographic labels and actual paired acquisitions for the next
   mechanism test.** The Parker control demonstrates a bounded access path
   and exposes geometry/alignment effects. It does not establish an
   ink-specific raw-intensity detector.

These decisions follow from the fixed results below; they are not new tests.

## 2. What actually ran in the Hecate experiment

### Design, failure and prospective amendment

**From a source.** The
[pinned Hecate implementation](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/hecate.py),
[published Studio donors](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/tree/a63c3796ebd13e1c70282433cdeb5c34aaa45c07),
[physical mapping preflight](../firstletters-colab/transversal_lab/hecate_donor_preflight.md),
and [frozen protocol](../firstletters-colab/transversal_lab/hecate_probe_protocol.json)
define the inputs. Sixteen contexts per donor were selected before CT
outcomes. Fine source slabs were box-averaged at 9.6 µm, rounded once to
uint8 and divided by 255 once. A 16×64×64 input contributes a central 32×32
scoring core. All backbone parameters and normalization buffers remain frozen.
The scientific code and manifest were published at
[`0dd77c7`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/0dd77c76effa551e3de99c1af87b62e553413939).

**From a source / measured.** The original 9B fitting cores contained **zero
ink labels and 4,650 background labels**. Their full input contexts also
contained no positive labels. The published source sampler admits all four
children of an ink-containing parent, including empty children; our smaller
fixed contexts missed the parent's positive pixels. The run stopped before
fitting any head or accessing 814 CT/labels. This is an **unassessable
training-support failure**, not a failed detector. The original output and
its causal audit remain intact:
[failure summary](../firstletters-colab/transversal_lab/evidence/hecate_v1_outcome_summary.json),
[support audit](../firstletters-colab/transversal_lab/evidence/hecate_v1_training_support_audit.json).

**My inference / ownership.** I should have checked the fitting labels'
class support before paying for 64 feature forwards. Blinding the test does
not require ignoring training labels. The source's parent-level enrichment
was insufficient assurance for our smaller contexts.

**From a source / measured.** We then published a separate
[role amendment](../firstletters-colab/transversal_lab/hecate_role_protocol.json)
at
[`fc1c16b`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/fc1c16b0486dd8fba9313046a724f2464abf0621),
before any fitting, held-out CT extraction or held-out label decoding. It reused all 64 existing feature
caches and changed only the fitting/calibration donor roles:

| Role | Physical donor | Known background | Known ink | Use |
| --- | --- | ---: | ---: | --- |
| Fitting | PHerc. 0841 | 8,927 | 2,712 | Fit each learned readout with 2,712 pixels of each class, fixed seed |
| Calibration | PHerc. 0009B | 4,650 | 0 | Freeze the background operating point; positive-class validation is unavailable |
| Test | PHerc. 0814 | 3,566 | 1,554 | Unchanged sixteen contexts; labels decoded after all heads and thresholds froze |

Counts and execution ordering are from the
[complete result](../firstletters-colab/transversal_lab/evidence/hecate_role_result.json)
and [readout freeze](../firstletters-colab/transversal_lab/evidence/hecate_role_readout_freeze.json).
The freeze hash is
`a4da6a372999ff7fc8456423ce72146f1b4ebdb6f9c030a720ffa4e3f64be6ec`.
The original failed attempt and its feature hashes were verified again after
completion. No contexts were replaced; no geometry exclusions occurred.
Public 814 metadata, coordinate lists and publisher-supplied label hashes
were read during preflight. Its label-image payload was not acquired or
decoded then; comparing published LFS hashes is distinct from reading labels.

**From the protocol.** The three readouts are the released projection;
sixteen global nonnegative depth weights summing to one plus a bias; and a
regularized logistic readout of 64 frozen decoder channels averaged over
fixed depth planes 6:10, the central four planes. There is one fit
per learned readout per training order, without tuning or retries. All four
fits converged. Both head sets and all four training-order × calibration-order
threshold sets were written before any 814 CT extraction or label decoding/use. The threshold
is calibration background score 46 in descending zero-based order; strict
`score > threshold` prevents ties from exceeding the specified empirical
fraction. The source/source/source comparison and the all-reversed
sensitivity were named in advance.

### Results: preserve the primary comparison and its sensitivity

**From a source / measured.** All values below are conditional on the
released labels, from the
[complete parent table](../firstletters-colab/transversal_lab/evidence/hecate_parent_metrics.csv).
Top 1% reviews 52 labeled pixels; top 5% reviews 256. These are pixel ranking
budgets, not numbers of legible letters or reviewed windows.

| Comparison | Readout | AUC | AP | Precision, top 1% | Precision, top 5% | Test background selected at frozen threshold | Test ink recall at that threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary: all source | Released | 0.2463 | 0.2291 | 13.46% | 44.53% | 41.19% | 14.16% |
| Primary: all source | Depth weights | 0.2001 | 0.1934 | 0% | 0% | 28.38% | 0.39% |
| Primary: all source | Feature readout | 0.5928 | 0.3994 | 67.31% | 57.42% | 0.53% | 2.45% |
| All-reversed sensitivity | Released | 0.9413 | 0.8474 | 100% | 100% | 15.48% | 95.88% |
| All-reversed sensitivity | Depth weights | 0.9600 | 0.8625 | 100% | 85.94% | 5.27% | 82.43% |
| All-reversed sensitivity | Feature readout | 0.7157 | 0.6465 | 100% | 92.97% | 8.92% | 48.97% |

**From a source / measured.** Every other orientation combination is retained
in the [complete orientation table](../firstletters-colab/transversal_lab/evidence/hecate_orientation_metrics.csv)
and [figure](../firstletters-colab/transversal_lab/evidence/hecate_readout_comparison.svg).
For completeness, these are all test background-selection rates. S means
ascending source depth; R reverses it. The baseline repeats across training
orders because it is not trained here; those repeats are not replications.

| Training / calibration / test order | Released | Depth weights | Feature readout |
| --- | ---: | ---: | ---: |
| S / S / S | 41.19% | 28.38% | 0.53% |
| S / S / R | 7.82% | 0% | 3.06% |
| S / R / S | 79.78% | 75.80% | 2.36% |
| S / R / R | 15.48% | 4.71% | 5.30% |
| R / S / S | 41.19% | 13.21% | 21.20% |
| R / S / R | 7.82% | 3.39% | 6.20% |
| R / R / S | 79.78% | 57.43% | 30.34% |
| R / R / R | 15.48% | 5.27% | 8.92% |

**My inference.** No order is selected from this table for deployment. A
low background rate alone can hide near-zero sensitivity: the primary
feature readout retains only 2.45% of labeled ink at its frozen threshold.
Conversely, a useful ranking can coexist with a badly transported threshold.
These are separate properties.

### The small denominator and aggregation effect matter

**From a source / measured.** Only **five of sixteen test cores** have known
labels; eleven have none. Of the five, two contain both classes, two contain
only background, and one contains only ink. The all-ink core contributes
1,024 of the 1,554 positives. The protocol's pitch and sizes imply a
0.6144 mm input edge and 0.3072 mm scoring edge. All sixteen scoring cores
cover about **1.510 mm²**; the known test labels occupy about **0.472 mm²**
of that nominal flattened grid. This is a tiny representation diagnostic,
not a field-scale legibility test. Counts are in the result's per-context
records; physical sizes are arithmetic from its frozen sampling contract.

**From a source / measured.** On the two mixed-class cores, the reversed
released model's AUCs are **0.9960 and 0.9977**. The all-reversed depth head
reduces them to **0.9409 and 0.9518**, despite its higher pooled AUC. Its
within-core APs also fall, from **0.9915 / 0.9909** to **0.8616 / 0.8729**.
Even the source-trained depth head's attractive reversed-test pooled AP of
0.9153 coexists with lower AP in both mixed cores: 0.9044 / 0.8435.
The primary feature readout improves one mixed core's AP, 0.3046 to 0.5010,
and worsens the other's, 0.4266 to 0.1566. See the complete result's
`contexts` records; no new feature or alternative experiment was run to
obtain these already-declared diagnostics.

**My inference.** Cross-context comparisons must account for the pooled AUC
gain when within-core ordering worsens in both mixed cores. This is not
necessarily useless: correcting cross-patch score offsets could help a
global ranking. It is also not evidence that strokes have become clearer.
The class-homogeneous patches and concentration of positives make that
distinction especially consequential here.

**From a source / measured.** Feature-readout fitting-parent AP is 0.7831
for source order and 0.9515 for reverse order, versus 0.3994 and 0.6465 on
the matching test orders. Those fitting metrics use all scored 0841 pixels
and are explicitly in-sample; they are not an additional validation donor.

**My inference.** This demonstrates that fitting the source labels is
possible and that a good fitting score is insufficient for transfer. It does
not isolate whether the gap comes from learned nuisance structure,
annotation provenance, context composition or acquisition differences.
There is only one held-out physical parent for the new heads. The inherited
backbone's exposure is unknown, and the Studio annotations are model-assisted.
The inputs are fine-derived 9.6 µm arrays, not native coarse CT. Small
64×64 contexts also do not establish behavior under full-page tiling.

### Orientation correction and immediate engineering use

**From a source.** The
[orientation protocol](../firstletters-colab/transversal_lab/hecate_orientation_protocol.json)
records a correction made before any donor experiment: the publisher's
earlier `normal_sign` operation and later `runtime_reverse` must be composed.
9B's `normal_sign=-1` with `runtime_reverse=false`, and 814's later reversal,
both produce a net reversed legacy stream relative to ascending source Z.
The flag alone does not indicate opposite physical directions. 0841's
physical face remains unverified.

**My inference.** Make this contract explicit in the rendering/inference
handoff: source axis order, source surface normal, accumulated flips, sample
centers in physical units, and final model depth order. Test the composition
on a fixed public control. Do not choose a flip on an unread target because
it looks more textual. Maximizing over orientations would also invalidate a
threshold calibrated for a single orientation unless the entire maximization
procedure were separately calibrated.

**My inference / decision.** Do not deploy either learned head as an improved
ink detector on this evidence. Retain the frozen readouts as diagnostics.
Use the available L4 for Claude's already active work while fixing provenance
and control design. This depth-order experiment **does not answer Codex's
two-tile-placement question**; that spatial-context question remains open.

## 3. Independent photographic-label CT audit: completed

**From a source.** [Parker et al. (2019)](https://doi.org/10.1371/journal.pone.0215775),
their [methods appendix](https://doi.org/10.1371/journal.pone.0215775.s001),
and [public archive](https://doi.org/10.17605/OSF.IO/ZDKN4) supply one already
published lunate sigma from PHerc. Paris 2 Fragment 96, photographic ink
labels, and approximately 1.3 µm, 53 keV CT. The independent implementation
and [exact protocol](../firstletters-colab/transversal_lab/parker_profile_protocol.md)
were published before CT intensities were decoded. No learned head or
threshold was fitted in this audit.

**From a source / measured.** A fixed 16-pixel lattice, complete support and
the supplied mask retain **11,693 points: 4,678 ink and 7,015 background**.
We sampled depths -16 through +16 along the supplied direction, plus a
fixed profile mean, positive-minus-negative depth contrast and PPM-height
comparator. All signs were declared beforehand: larger descriptor values
predict ink. No below-chance result was flipped.

| Fixed descriptor | Pooled signed AUC | Mean within mixed-class rows |
| --- | ---: | ---: |
| Center intensity | 0.4630 | 0.4915 |
| Mean over 33 depths | 0.4089 | 0.4379 |
| Positive-depth minus negative-depth contrast | 0.5754 | 0.6090 |
| PPM source-X height | 0.5161 | 0.3344 |

Measured values, all depths and all paired shifts are in the
[complete report](../firstletters-colab/transversal_lab/parker_results.md),
[native CSV](../firstletters-colab/transversal_lab/evidence/parker_native_metrics.csv),
[shift CSV](../firstletters-colab/transversal_lab/evidence/parker_shift_metrics.csv)
and [figure](../firstletters-colab/transversal_lab/evidence/parker_depth_comparison.svg).
The 33 individual-depth pooled AUCs range from 0.3971 to 0.5339; we retain
the whole curve rather than select a winning depth.

**My inference.** An AUC below 0.5 means an association opposite to the fixed
sign, not absence of signal. Here the material is known to carry ink. Weak
raw descriptors cannot establish that ink is absent, or that a learned 3D
representation has no useful information.

**From a source / measured.** The fixed depth contrast beats eight of the
twelve shifted-label comparisons and loses four. Each comparison uses its
own identical native/shifted support. For X -256, native/shifted AUCs are
0.5929/0.4752; for Y +256, they are 0.5602/0.5799. Comparing every shifted
score against the full-sample 0.5754 would ignore these changing supports.
Within-row summaries use 126 of 151 sampled rows, those containing both
classes. The rows are not independent specimens.

**My inference.** These shifts are alignment stress tests, not exchangeable
randomizations, p-values or false-positive bounds. The pooled height AUC near
0.5 does not exonerate geometry: its within-row value is 0.3344. Pooled and
within-row weighting answer different questions. The contrast's modest
association could involve ink, texture, height, alignment or several of
these together; this experiment does not identify the mechanism.

**From a source / measured.** Every supplied direction is **(-1,0,0)** and
the runner follows `(X-d,Y,Z)`. These are fixed-axis profiles around a height
map, not independently estimated local surface normals. Before intensity
analysis, 1,057 of 3,016,367 fully masked surface centers were outside the
source bounds, and 21,827 lacked full 33-depth support. They were rejected,
never clipped to a nearby valid voxel. All 151 declared source TIFF members
passed CRC/header checks; local SHA-256 hashes were recorded, and available
prior hashes were checked.

**My inference / decision.** Keep the bounded loader, source-bound guard and
independent-photo endpoint. Do not promote the raw contrast as a detector or
call the fixed axis verified recto/verso depth. NIST phantom payload access
remains unresolved; no NIST experiment was executed in this tranche.

## 4. Transversal follow-ups with a sharper question

### A. Separate sampling loss from acquisition mismatch

**From a source.** The metadata-only
[photographic-donor preflight](../firstletters-colab/transversal_lab/photographic_donor_preflight.md)
identifies these actual scan pairs and associated public fine-scan IR/label
bundles:

| Parent | Fine scan | Actual coarse scan | Why it is useful |
| --- | --- | --- | --- |
| [PHerc. 1667 Cr1 Fr3](https://scrollprize.org/data_browser/PHerc1667Cr1Fr3) | 3.24 µm, 70 keV | 7.91 µm, 70 keV | Same stated energy; published fine-to-coarse affine and five point pairs |
| [PHerc. 51 Cr4 Fr8](https://scrollprize.org/data_browser/PHerc51Cr4Fr8) | 3.24 µm, 53 keV | 7.91 µm, 53 keV | Another physical parent with the same kinds of assets |

The linked preflight supplies exact primary metadata URLs and hashes. It
contains no CT arrays or decoded labels. It also identifies
[Paris 1 Fragment 34](https://scrollprize.org/data_browser/PHercParis1Fr34)
as a further donor with photographic label production directly documented in
the [original dataset paper](https://arxiv.org/html/2304.02084v4).
I do not know the full original tracing/approval history for the two Naples
labels. An IR image and a label file alone do not prove that history.

**My inference.** The next comparison should hold the physical ink and
photographic endpoint fixed while changing acquisition. Compare fine CT
averaged to the target physical support against actual coarse CT of the
same surface. This addresses a question the current Hecate experiment cannot:
whether missing performance is explained by sampling alone or by native
acquisition/reconstruction differences. Equal stated energy helps, but does
not eliminate registration, motion, radiometry or reconstruction differences.

**My inference — proposed gates, not an executed or frozen new trial.**
Start with Frag5. Verify headers, PPM shape, source-volume identity, exact
byte-range access and raw Zarr bounds before pixel analysis. A first bounded
preflight should cap itself at **64 decoded 128³ uint16 chunks, 256 MiB**;
this is a working-set cap, not a guarantee that the selected geometry fits.
Record actual compressed transfer separately. Do not download multi-gigabyte
PPMs or all 65 surface TIFFs to obtain a small control. Resolve the exact
uint16-to-model radiometry before Hecate inference; arbitrary per-window
min/max scaling would change the experiment.

**From a source / arithmetic on metadata.** The published five-point affine
fits have in-sample RMS residuals about 0.2682 and 0.8841 coarse voxels for
Frag5 and Frag6 respectively. **My inference:** these are not held-out local
registration errors. A low residual near a few landmarks cannot certify the
sheet identity and depth of an ink patch. Inspect independent geometric
features and enforce no-fill/bounds checks before transferring labels.

**My inference — coordinate detail that matters.** For affine linear part
`A`, a transformed geometric surface normal is proportional to `A^-T n`
with the coordinate metric handled correctly. A previously sampled ray maps
instead by `A(p+d n)+t = Ap+t+d A n`. Under shear or anisotropic scaling these
directions differ. State whether a renderer preserves the original ray or
re-renders along the transformed normal; do not apply one formula to both.

### B. Treat patch offsets as a separate nuisance, not a reason to invent ink

**My inference.** The Hecate aggregation reversal suggests testing global
score transport separately from within-patch discrimination. A future
comparison can report both without changing the ink model: fixed whole-area
ranking, mixed-patch ranking, and a donor-disjoint operating point. Hold out
physical parents, not neighboring fragments or patches. Paris 1 fragments
share one parent; Paris 2 fragments, including the Parker case, share another.
The [dataset paper](https://arxiv.org/html/2304.02084v4) documents those
fragment origins.

**Speculation.** A nuisance readout fitted from label-independent CT
structure might correct material-dependent score offsets while preserving
the released stroke ranking. This is worth a later, separately frozen test
only if it improves whole-window ranking on a new parent. Local percentile
normalization can force apparent positives into every blank window, so it
cannot itself be a calibrated ink probability. Our earlier 306B nuisance
pilot did not pass its continuation gate, and its second coarse row box is
not a recovered legible-line control; see [ChatGPT 011](chatgpt-011.md).

### C. Make orientation uncertainty explicit without searching for plausible letters

**Speculation.** A readout trained to use paired forward/reverse features
could be less sensitive to arbitrary depth indexing. Averaging or taking
the maximum of current scores is not automatically safe: either can change
the background tail, and averaging can suppress a real face-specific signal.
Such a method needs another parent, its own frozen threshold calibration and
a clear physical-order contract. The now-observed 814 outcomes cannot be
used to tune this proposal and then counted as its held-out test.

**My inference / priority.** Do A before B or C. We already have a strong
released-model ranking in one fixed order on this small known control.
Another adaptation sweep would not resolve the more important native-scan,
label-independence and geometry uncertainties.

## 5. Reproducibility, costs and prize relevance

**From a source / measured.** Hecate used four CPU threads and no GPU. The
amendment added 32 forwards, **344.65 process CPU seconds** and **460.52 wall
seconds**. Including the preserved original attempt gives 96 forwards and
**425,405,380 source bytes**, plus the **522,576,086-byte checkpoint**.
The conservative cumulative CPU charge is **944.65 seconds**, including a
600-second charge for the first attempt whose exact final CPU time was not
persisted. That charge must not be presented as an exact total measurement.
The result hash is
`4c34b349e697e760456ff915be1cebc4079f61eeb1be90e7a9715b4ba56d0913`.

**From a source / measured.** Parker's two incomplete network attempts and
their separately published transport repairs are preserved. The successful
final tranche took **3.64 CPU seconds / 37.17 wall seconds**, obtained the
last sixteen members and analyzed the case. Those times exclude earlier
acquisition work. The complete archive was not downloaded or locally
whole-archive-hash-verified. The exact result hash is
`1948c244e6ebdcc6da2a189587acfdfbb5aea7ba1de92213e7943b548e786350`.
Full receipts, generated plots and reproduction commands are in the
[laboratory README](../firstletters-colab/transversal_lab/README.md).

**From the implementation records.** The historical Hecate resumer requires
this run's exact cached features and receipts. A separate fresh-run wrapper
imports the frozen scientific helpers and records fresh provenance; its
generated-array and access-order checks are packaging verification, not a
second full donor experiment. See
[replay_hecate_role.py](../firstletters-colab/transversal_lab/replay_hecate_role.py).

**From a source.** The current
[Progress Prize guidance](https://scrollprize.org/prizes#progress-prizes)
prioritizes useful integrated work, measured improvement and adoption, and
discourages standalone evaluation/auditing tools. First Letters requires a
legible image, appropriate mesh and nonoverlapping training/prediction regions.
The explicit higher-resolution-data exclusion appears under the Grand Prize
conditions; that wording should not be silently transplanted into a different
prize. No rule interpretation here certifies an adaptation scheme's eligibility.

**My inference / decision.** This is useful experimental support for the
team's existing pipeline, not sufficient grounds for another isolated prize
entry. A stronger contribution would integrate the physical-order and
training-support guards into a real render/inference path, demonstrate the
before/after failure on public controls, and make the bounded photographic
native-pair loader reusable. No submission, organiser contact, upstream PR,
Discord post or main-branch merge was made.

## Open questions

- **Claude Code:** What are the complete composed depth conventions in each
  active renderer/model path? Can the requested two-tile-placement comparison
  on the same public control preserve identical physical field, support and
  ranking denominators? This experiment did not answer that question.
- **Codex:** Can the Frag5 fine/coarse pair pass independent local
  sheet-identity, source-bound and nonfill checks within the compact budget?
  The published affine's five-point residual is insufficient.
- **ChatGPT:** Can the photographic native-pair experiment resolve radiometry
  and local geometry well enough to separate acquisition mismatch from
  resampling loss? The data route is identified, but that comparison has not
  yet been run.
- **Owner:** Existing-model exposure and exact Naples annotation lineage
  remain unknown. They constrain any claim of independent generalization or
  prize-ready validation.

## Next actions

- **ChatGPT:** Keep these outcomes fixed, retain the failed attempts, and
  move the next bounded preflight toward the native fine/coarse photographic
  control. Do not reuse 814 or the Parker glyph as a fresh held-out test after
  method selection from this note.
- **Claude Code:** Keep the released model as the working baseline. Record
  the composed orientation contract, preserve the quality strata requested
  in ChatGPT 011, and resolve the public control's tile-placement sensitivity
  before promoting new learned readouts.
- **Codex:** Reuse the strict invalid-coordinate rejection and support audit;
  verify the proposed compact Frag5 mapping with independent geometry.
  Report physical bounds and transform uncertainty separately from ink scores.
- **Owner:** No intervention is needed for these public-data experiments.
  Decide any later submission or discovery disclosure; this note makes no
  new prize claim.
