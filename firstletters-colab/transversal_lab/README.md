# Bounded transversal CPU experiments

This directory contains two completed public-data experiments: optical-height
features with scale-matched linear classifiers, and local standardization of
released PHerc. 306B model scores. No unread-scroll inference or GPU is needed.
Methods, results and decisions are explained in
[ChatGPT 010](../../collab/chatgpt-010.md); the next proposed experiments are in
[IDEAS.md](IDEAS.md). Neither pilot establishes new letters or a prize result.

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
independent validation. It improves the weak-row rank but fails its continuation
gate, including strong-row preservation and a search-truncation check. The
matched process control still lacks the strong row's support. The matched
depth-shuffle views produced zero proposals; that result does not establish a
population false-positive rate.

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
