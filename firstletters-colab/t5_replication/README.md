# T5: conditional spatial replication on a published control

This CPU experiment tests a specific new lead from Gemini: **Hilbert depth-envelope
amplitude and wavelet scale slope** on the public PHerc. 0139 native 2.399 µm
surface volume. It evaluates agreement with a **model-generated reference**, not
optical ink truth, readable letters or an unread-scroll discovery.

**From the [upstream result](https://github.com/JorisViaudQuantAI/Scrollprize/blob/62b2c02c6b43675131afdf642fcf5ba6221f0970/gemini_lane/evidence/transverse_wavelet_oct_ablation.json):**
the two selected features scored AUC 0.77962 and 0.79674, against raw surface
intensity 0.71484, on one development field. The wavelet shifted-label maximum
was 0.81802. Features and signs were selected using that field. The coarser
306B arm remained negative. Credit for the original feature implementations
and the exploratory lead belongs to Gemini.

**My inference:** an inexpensive spatial replication is justified; another
feature sweep or immediate GPU expenditure is not yet supported. This experiment
freezes the selected features and their signs, verifies the original result,
and tests three separate fields without choosing them by their feature scores.

## Frozen design

The complete authority is [protocol.json](protocol.json). Its file hashes cover
the implementation, tests, this README and [source manifest](source_manifest.json).
Publish the protocol before running; the runner checks every hashed file against
the supplied published Git commit. No reference pixels or CT chunks were read
while preparing this protocol. The source preflight was one metadata GET and one
reference HEAD.

- Development replay: the exact published 512 × 640 field, original 32-pixel
  guard, exact class counts and AUC tolerance 0.00005.
- Replication: three 512 × 640 fields with a 64-pixel guard; label-only selection
  on a 128-pixel lattice. Full field and retained core each require at least 20%
  high reference, 20% low reference and 90% nonzero reference. Greedy maximum
  minimum rectangle-gap distance gives spread, with at least 1,024 pixels of
  Euclidean gap from the development field and previously selected fields.
- Fixed high class: reference 179–255. Fixed low class: 1–64. Zero and 65–178
  are excluded. These are proxy classes, not confirmed ink and papyrus.
- Fixed signs: raw intensity positive, Hilbert envelope positive, wavelet slope
  negative. No new fitting or sign selection.
- Common support: full source chunks, valid sheet, all 31 depth samples in
  bounds, finite features and the lateral guard. A field must retain at least
  90% of guarded potential pixels and 90% of legacy sheet support, with at least
  10,000 pixels of each proxy class after masking.
- Equal-field macro AUC; 199 paired spatial-block bootstrap replicates; 199
  independently drawn toroidal label translations per field. The null compares
  with the maximum of the two features' macro AUCs, not the null mean.
- A feature must reach AUC 0.70 in every field, macro improvement at least 0.05,
  positive lower paired-block interval and spatial rank at most 0.05. All
  replay, support and assessability checks must also pass.

No field is replaced after seeing CT or features. No threshold is relaxed. A
missing class in any statistical draw is retained and prevents passage.

At nominal scale, a field spans 1.228288 × 1.535360 mm; the minimum rectangle gap
is 2.456576 mm. These dimensions are arithmetic from the source's nominal voxel
size, not an independent calibration of the surface geometry. They are small
local fields, not independent scrolls or complete text lines.

## What the controls do and do not establish

Synthetic parity tests run actual pinned upstream functions and check exact
feature equality, including excluded/clamped profiles. The development replay
tests the real acquisition/feature/evaluation chain. It is a reproduction,
not independent evidence of efficacy.

The heldout fields are selected using the teacher map's class balance. The
result therefore concerns **label-eligible regions of the same public control**,
not a random scroll sample. Teacher error, model/CT correlation, shared physical
material, local geometric ambiguity and possible upstream model training overlap
remain. This experiment is not a model-training or First Letters submission.

The spatial ranks are **misalignment diagnostics**, not valid familywise error
probabilities or proof of significance. Toroidal wrapping, nonstationarity and
conditional selection remain. The paired intervals describe variation across
128-pixel blocks within these three fields, not independent donor uncertainty.

## Run and inspect

Required installed packages are NumPy, SciPy and Pillow. No Torch, GPU, model
weights, CT decoder extension or annotation team is used. Run from this directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python experiment.py \
  --freeze-commit <published-protocol-commit> \
  --cache /tmp/t5-replication-source \
  --output /tmp/t5-replication-result
```

Both directories must be new or empty and outside every Git working tree. The
single process uses one numerical thread, a 1,200-second alarm, a 2 GiB peak-RSS
stage check, and source limits of 176 MiB and 100 requests, including preflight.
The complete four-field plan is 155,477,363 response-body bytes and 84 requests.
Redirects, retries, missing-chunk fill and adaptive replacement are forbidden.
These byte counts are HTTP response bodies, not total on-wire traffic.

`result.json` is updated after stages, preserving terminal failures. Each case
has an NPZ containing original-sign feature maps, proxy reference, common mask
and guarded evaluation mask. `statistics.json` preserves all offsets, block
draws, counts and numerical results. `selection.json` records every label-eligible
candidate and the deterministic chosen fields. Source receipts contain whole-file
hashes. Nonversioned S3 hashes identify the acquired bytes retrospectively; they
do not establish an atomic snapshot of the remote dataset.

Generated tests, with the pinned Gemini commit available in the local Git object
database:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s . -p 'test_*.py' -v
```

Results and any independent review will be added separately. This frozen code
and protocol will remain unchanged after the first source-array acquisition.
