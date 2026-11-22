# Bounded transversal CPU experiments

This directory contains bounded CPU experiments on public, already-read
material: optical-height features, local standardization of released PHerc.
306B scores, frozen Hecate readouts, and photograph-labeled raw CT profiles.
No unread-scroll inference or GPU is needed. Methods, results and decisions are
explained in [ChatGPT 010](../../collab/chatgpt-010.md),
[the interpretation corrections in ChatGPT 011](../../collab/chatgpt-011.md),
and [the new execution results in ChatGPT 012](../../collab/chatgpt-012.md).
None of these controls establishes new letters or a prize result.

## Evidence and interpretation

| Record | Contents |
| --- | --- |
| [Optical results](evidence/profilometer_results.json) | Exact sanitized output: all 22 result rows, fitted-head details and limitations |
| [Optical metrics CSV](evidence/profilometer_metrics.csv) | The same complete numerical comparison in tabular form |
| [Original nuisance v1 result](evidence/nuisance_v1_result.json) | Exact original receipt, also the required provenance input to v2 |
| [Nuisance v2 summary](evidence/nuisance_v2_summary.json) | Reduced record retaining the comparison metrics, coverage denominators and failed gates |
| [Experiment receipts](evidence/experiment_receipts.json) | Publication history, source/code/result hashes and execution provenance |
| [Optical comparison figure](evidence/profilometer_review_budget.svg) | AP and top-one-percent precision for both regions and every registered model arm |

**From the experiment records.** The optical pilot uses manual ink labels and
an explicit four-case subset of the official leave-one-papyrus-out fold. Both
held-out cases belong to **one** papyrus, PHerc. 500P2. Scale matching has mixed
average-precision effects while improving the fixed top-one-percent ranking
metric in both cases. The full feature set does not consistently beat the
simple baselines. Read the complete results rather than selecting one metric.

**From the experiment records.** Nuisance v1 is unassessable for its target
rows because complete-kernel support excludes their centrelines. V2 is a
separately published, adaptive support amendment on the same known rows, not
independent validation. It improves the second coarse row-box rank but fails its continuation
gate, including strong-row preservation and a search-truncation check. The
matched process control still lacks the strong row's support. The matched
depth-shuffle views produced zero proposals; that result does not establish a
population false-positive rate. **Correction:** [Claude 010](../../collab/claude-010.md)
and [ChatGPT 011](../../collab/chatgpt-011.md) establish that this second box
contains almost no published readable text. Its higher rank is not recovery of
a missed legible row. The original records remain intact.

**My inference.** These are bounded diagnostics. Optical height is a different
modality from CT; selected letter-crop prevalence is not atlas prevalence.
The normalized optical TIFFs lack absolute relief and the original missingness
mask; the source's lateral-calibration discrepancy remains unresolved. Report
degradation **factors**, not a CT resolution cutoff. The fixed optical boundary
exclusion can retain a narrow band influenced by padding. For the nuisance
pilots, masks are tissue/model-support proxies, coarse row boxes are
localization targets rather than ink truth, and standardized scores are display
signals rather than calibrated probabilities. V2 reports conditional and
full-source coverage separately; do not interchange their denominators.

## Environment

Run commands from the repository root on Linux with Python 3.11 or newer
(`hashlib.file_digest` and Unix interval/resource limits are used). The recorded
runs used Python 3.12.14. Dependencies are:

- Optical: `numpy`, `scipy`, `scikit-learn`, `Pillow`, `joblib`, `threadpoolctl`.
- Nuisance v1/v2: `numpy`, `scipy`, `Pillow`, OpenCV (`opencv-python-headless`
  is sufficient) and `matplotlib` for the saved comparison figures.
- Hecate: CPU `torch`, `numpy`, `scipy`, `scikit-learn`, `Pillow`, and
  `numcodecs`; no GPU or backbone training. The recorded run used Torch
  2.6.0+cpu, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0 and
  numcodecs 0.16.5. See its execution receipt for the complete environment.
- Parker: `numpy`, `Pillow`; HTTP range handling, archive-member integrity
  checks and tied-rank AUC are implemented in the frozen runner.
- New numerical-result plots: `numpy`, `matplotlib`.

For an isolated environment outside the repository:

```bash
python3 -m venv /tmp/vesuvius-transversal-env
. /tmp/vesuvius-transversal-env/bin/activate
python -m pip install numpy scipy scikit-learn Pillow joblib threadpoolctl opencv-python-headless matplotlib
```

The receipts record the versions actually used. The package list above is a
dependency installation example, not a claim of an identical locked binary
environment. Preserve each replay's version record; floating-point ties and
rendered-figure hashes may vary across environments.

To regenerate the optical figure from the committed numerical results only:

```bash
python firstletters-colab/transversal_lab/plot_results.py
```

This reads no images and performs no fitting or inference. It writes the SVG
and PNG alongside the public evidence.

## Frozen publications and file hashes

**From git and the experiment records.** These commits preceded the respective
new image access or transformed-output computation:

| Procedure | Publication commit |
| --- | --- |
| Nuisance v1 and initial prospective note | [`901658658371369b0b4d00f566e89b4bb0122c16`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/901658658371369b0b4d00f566e89b4bb0122c16) |
| Exact optical protocol and runner | [`e387d75301293c4c3c6d20360a3d35d8d2addb15`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/e387d75301293c4c3c6d20360a3d35d8d2addb15) |
| Nuisance v2 support amendment | [`a813c75ad4acf0767b24b69401ecc9ffe0ea10e1`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/a813c75ad4acf0767b24b69401ecc9ffe0ea10e1) |

The retained files have these SHA-256 hashes. Their code and protocols are not
edited to improve observed outcomes.

| File | SHA-256 |
| --- | --- |
| [profilometer_protocol.json](profilometer_protocol.json) | `d545a59735797a2f6f3cbe022a4c1a6ad527f338374f2d894544d3f22fc7b78e` |
| [profilometer_pilot.py](profilometer_pilot.py) | `b59be0ff688d770cfa9239e3962b94223f97fef0e43a01cfcda63e18fd1f7d2b` |
| [nuisance_protocol.json](nuisance_protocol.json) | `534dff0132056f364e65985d219fdc4614655e1cebddf1f41b9cbb51e994f14f` |
| [nuisance_pilot.py](nuisance_pilot.py) | `e7db21c5f203a52102bc546849db85dace573c7cb700fde2044da4a038399507` |
| [nuisance_protocol_v2.json](nuisance_protocol_v2.json) | `489132d889913cd7e27e9901032ac45f0f04e8d6a664d8d1d07603ad5bc32f73` |
| [nuisance_pilot_v2.py](nuisance_pilot_v2.py) | `e16d01f95c90a9f357fb96bc5e5bb042573aba09ba426d1bc0ec0dd63860f6f0` |

## Reproduce the optical pilot

The original experiments are complete. The commands below describe **one exact
verification** in fresh external directories, not repeated trials or retuning.
Keep failures and incomplete outcomes. Any changed method needs a separately
recorded protocol; these commands do not authorize deployment on unread tissue.

First download and checksum the fixed subset, without decoding any images:

```bash
python firstletters-colab/transversal_lab/profilometer_pilot.py \
  --prepare-only \
  --frozen-protocol-sha256 d545a59735797a2f6f3cbe022a4c1a6ad527f338374f2d894544d3f22fc7b78e \
  --work-dir /tmp/transversal-optical-verification
```

Then perform the single registered fit/evaluation in that same directory:

```bash
python firstletters-colab/transversal_lab/profilometer_pilot.py \
  --run \
  --frozen-protocol-sha256 d545a59735797a2f6f3cbe022a4c1a6ad527f338374f2d894544d3f22fc7b78e \
  --work-dir /tmp/transversal-optical-verification
```

The protocol pins the source revision, file sizes/checksums, split, features,
balanced sampling rule and seed, degradation, heads and metrics. Source images and
labels total 134,620,886 bytes, plus small metadata files. The runner uses one
CPU thread, enforces its 900-second run CPU cap, checks preparation provenance
and refuses to repeat a run in the same directory. It retains models,
prediction arrays, masks, source/fit hashes, receipts and sanitized JSON/CSV
outside git. It freezes all five heads before decoding held-out cases. No
hyperparameter or threshold search is part of this procedure.

## Reproduce nuisance v1 and v2

Prepare the already-awarded public controls once, following
[the source preparation documentation](../line_controls/README.md):

```bash
python firstletters-colab/line_controls/prepare.py \
  --output-dir /tmp/transversal-public-controls
```

This extractor downloads the pinned public corpus and writes `cases/` with
`score.npy`, `validity.npy` and manifests. Reuse an existing verified external
corpus if available. The nuisance runners verify the selected score/mask hashes
against the committed inventory; they perform no new model inference.

```bash
python firstletters-colab/transversal_lab/nuisance_pilot.py \
  --public-controls-root /tmp/transversal-public-controls \
  --output-dir /tmp/transversal-nuisance-v1-verification

python firstletters-colab/transversal_lab/nuisance_pilot_v2.py \
  --public-controls-root /tmp/transversal-public-controls \
  --v1-result firstletters-colab/transversal_lab/evidence/nuisance_v1_result.json \
  --output-dir /tmp/transversal-nuisance-v2-verification
```

**Use the exact committed original v1 receipt for `--v1-result`.** V2 checks its
SHA-256, `b3bbc83a192420d778fc17e7f93f74cac416deab49e142241ae6d932c729c081`,
to identify the recorded motivation for the amendment. A newly reproduced v1
receipt contains new timestamps and runtimes and will not have that byte hash.
V2 also verifies the frozen v1 runner/protocol it imports.

Each nuisance run requires a fresh output directory outside git. The frozen
comparison contains three input cases, caps each at 180 seconds and saves `result.json`,
per-case reports, masks, score arrays and comparison plots. Neither completion
nor a better isolated rank is a passing operational gate. Do not lower the
support threshold, shrink the kernel, enlarge the search cap or tune line
thresholds after inspecting these outcomes. Follow
[the recorded decisions](../../collab/chatgpt-010.md) and
[the proposed next experiments](IDEAS.md).

## Deeper runs: complete evidence

The interpretation and decision record is [ChatGPT 012](../../collab/chatgpt-012.md).
These files retain every registered arm, including failures and sensitivities.

| Record | Contents |
| --- | --- |
| [Hecate original support failure](evidence/hecate_v1_outcome_summary.json) | The original fitting donor supplied zero positive labels; no head fit or test access occurred |
| [Training-support audit](evidence/hecate_v1_training_support_audit.json) | How the published parent sampler admitted empty child contexts |
| [Hecate role amendment](hecate_role_protocol.json) | Prospective reuse of 0841 for fitting, 9B for negative calibration, and unchanged 0814 for testing |
| [Hecate complete result](evidence/hecate_role_result.json) | In-sample diagnostics, all eight depth-order combinations, every context and acquisition/hash provenance |
| [Frozen heads and thresholds](evidence/hecate_role_readout_freeze.json) | Exact readouts and all four calibration threshold sets, written before any 0814 CT extraction or label decoding/use |
| [Independent result review](evidence/hecate_role_result_review.json) | Integrity, support, threshold consistency and completeness checks on 1,122 existing metric objects; no additional experiment |
| [Fresh replay packaging checks](evidence/hecate_replay_packaging_checks.json) | Pinned runner hash, offline checks and independent code review; complete replay explicitly not run |
| [Hecate parent metrics](evidence/hecate_parent_metrics.csv) | Primary and all-reversed train/calibration/test tables, with training explicitly marked in-sample |
| [Hecate orientation metrics](evidence/hecate_orientation_metrics.csv) | Complete eight-combination calibration/test table |
| [Hecate comparison figure](evidence/hecate_readout_comparison.svg) | All unique train/test AP comparisons and every transferred operating point |
| [Parker report](parker_results.md) | All 33 depths, fixed descriptors, twelve paired shifts, limitations and acquisition history |
| [Parker complete result](evidence/parker_profile_results.json) | Exact complete output, including all 900 native/shifted statistical bundles |
| [Parker native metrics](evidence/parker_native_metrics.csv) | All frozen native descriptors; prescribed signs retained |
| [Parker shift metrics](evidence/parker_shift_metrics.csv) | Every same-support native/shifted comparison |
| [Parker figure](evidence/parker_depth_comparison.svg) | Full depth curves and descriptive shift envelope; no selected depth or inferred confidence interval |
| [Photographic-donor preflight](photographic_donor_preflight.md) | Metadata-only feasibility of another photographed parent and actual native fine/coarse pairs |

**From the frozen publications.** The original Hecate code, physical sampling
contract and manifest were published at
[`0dd77c7`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/0dd77c76effa551e3de99c1af87b62e553413939),
and the full orientation supplement at
[`8561904`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/8561904425118c7ed8d8c9aa93b15e83c8418203).
The original run stopped for insufficient positive training support. That
orientation executable was therefore not run. The separately published
role amendment at
[`fc1c16b`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/fc1c16b0486dd8fba9313046a724f2464abf0621)
preserved the failed attempt and reused its 64 cached forwards. It retained
the same readout forms, contexts, scoring pixels, full orientation comparison
and budgets. It is a new prospective amendment, not a successful original run.

**My inference.** Only the new readout heads have a documented held-out
physical donor. The inherited Hecate backbone's exposure is unknown; Studio
labels are model-assisted; one test donor and correlated pixels do not
establish population generalization. All Hecate inputs here are box-averaged
fine CT at a nominal 9.6 µm pitch, not actual coarse-acquisition CT. A threshold
that selects at most 1% of sampled 9B background is not a guaranteed target
false-positive rate or a bound on false text proposals. See the complete
limitations beside each result.

**From the frozen publications.** Parker scientific code, protocol and archive
manifest were published at
[`bcd67e3`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/bcd67e3005132b74e295964cea99f4a23d63608c).
Two documented transport failures preceded numerical analysis. The
[first](parker_transport_amendment.json) and
[final](parker_transport_final_amendment.json) transport amendments preserved
the scientific procedure. Final transport was published at
[`abc3298`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/abc3298d0c6f7e73a2b21794488214dd106f8bc1).
All original failures and acquisition receipts remain in `evidence/`.

## Reproduce the final Hecate role experiment

The historical [role-amendment executable](hecate_role_amendment.py) is a strict
resumer of this session's exact 64 feature caches and receipts. Its pinned
runtime/receipt hashes cannot be reproduced by rerunning the failed original
experiment at another time. Use [replay_hecate_role.py](replay_hecate_role.py)
for a fresh execution of the final scientific design.

That packaging script imports the exact original scientific helpers, retains
all contexts and labels, and assigns the amended logical roles directly. It
pins all 438 source assets to the two published acquisition receipts and
records its own feature hashes and execution provenance. A run-only check
of the fixed fitting/calibration labels and geometry rejects impossible class
support before checkpoint acquisition or CT extraction. It does not change
contexts or replace any labels; the original extraction still enforces the
exact final support. Both readouts for each depth order
and every calibration threshold must be frozen before any 0814 CT, geometry
or label asset is accessed, including optional verified cache reuse.

From a clone of this published branch:

```bash
python firstletters-colab/transversal_lab/replay_hecate_role.py check
python firstletters-colab/transversal_lab/replay_hecate_role.py synthetic-check

python firstletters-colab/transversal_lab/replay_hecate_role.py run \
  --out /tmp/transversal-hecate-role-verification \
  --freeze-reference "$(git rev-parse HEAD)"
```

The fresh wrapper obtains the pinned public model source and checkpoint unless
verified existing assets are provided with `--hecate-source` and `--checkpoint`.
Optional `--reuse-cache` together with `--reuse-receipt` permits individually
hash-verified source assets only. It never imports fitted heads, feature arrays
or outcomes from another run. The budget is 96 forwards, 1,800 process CPU
seconds, 514,000,000 source bytes excluding the 522,576,086-byte checkpoint,
and 1 GiB of checkpoint plus run files. All failures are retained.

**Verification scope:** the actual result used the original runner followed
by the preregistered cache resumer. The fresh wrapper was checked with frozen
hashes, generated arrays and an access-order guard, and independently reviewed;
its complete fresh acquisition/inference path was not rerun in this session.
No second experiment or independent replication is claimed by packaging it.

## Reproduce the Parker published-letter audit

Use a fresh external cache and the final transport wrapper. Copy the six
historical receipts exactly; their hashes identify the original failure and
the cumulative byte ledger, not a new scientific result:

```bash
mkdir -p /tmp/transversal-parker-verification
cp firstletters-colab/transversal_lab/evidence/parker_attempt001_failure.json \
  firstletters-colab/transversal_lab/evidence/parker_attempt001_download_receipts.json \
  firstletters-colab/transversal_lab/evidence/parker_profile_start_001.json \
  firstletters-colab/transversal_lab/evidence/parker_attempt002_failure.json \
  firstletters-colab/transversal_lab/evidence/parker_attempt002_download_receipts.json \
  firstletters-colab/transversal_lab/evidence/parker_profile_start_002.json \
  /tmp/transversal-parker-verification/

python firstletters-colab/transversal_lab/parker_transport_final.py --self-test
python firstletters-colab/transversal_lab/parker_transport_final.py \
  --execute \
  --cache-dir /tmp/transversal-parker-verification \
  --freeze-reference abc3298d0c6f7e73a2b21794488214dd106f8bc1
```

An empty cache acquires all 151 declared source TIFF members and the geometry/
label packet. The final wrapper's 135 previously verified members and 16
remaining members describe the original partly populated cache. The fresh
run retains the historical ledger conservatively; it does not pretend those
old source bytes were downloaded by the new run. Every reused/downloaded
member is checked against the archive manifest and available earlier hashes.
The whole 1.8 GB archive was not downloaded or locally hash-verified.

Raw profiles and source CT stay outside git. The supplied fixed `(-1,0,0)`
directions are not independently validated local surface normals. This is
technical reproduction of one public photograph-labeled glyph, not a new
glyph, donor, alignment estimate, or ink-depth measurement. The artifact
manifest records the original report name `PARKER_AUDIT_RESULTS.md`; its
unchanged bytes are published here as `parker_results.md`.

To regenerate both new figures and their CSVs from the committed numerical
results, with no source acquisition, fitting or model inference:

```bash
python firstletters-colab/transversal_lab/plot_deep_probe_results.py
```
