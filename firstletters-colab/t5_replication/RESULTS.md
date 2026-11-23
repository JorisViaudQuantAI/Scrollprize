# T5 replication: original result reproduced, both transfer gates failed

Date: 2026-11-23. Public PHerc. 0139 control only.

**From the [measured receipt](results/result.json) and
[independent recomputation](results/independent_review.json):** the run completed;
all acquisition, replay and support checks passed. Neither selected feature
passed the frozen spatial-replication gate. The wavelet feature retains a
conditional association with the published model map, but the development
score does not transfer uniformly to the three fixed test fields.

**My inference:** do not promote either feature to the unread-scroll inference
pipeline on this evidence. Do not reinterpret the failed gate as absence of all
signal: the wavelet association is measurable, but it is association with a
model reference and its absolute performance is inconsistent.

## 1. What was fixed before acquisition

**From the [public protocol freeze](https://github.com/JorisViaudQuantAI/Scrollprize/commit/b5079245a4e0965a36b6f99ea50921f5c9bc4f6a):**
the feature arithmetic, positive/positive/negative signs, selection rule,
support rules, seed, statistical calculations and acceptance thresholds were
published before reading reference pixels or CT chunks. The original
development field and its reported scores were already known. A 207-byte
metadata GET and a reference HEAD preceded the freeze, as disclosed in the
[source manifest](source_manifest.json).

**From [selection.json](results/selection.json):** 8,109 fully contained,
128-aligned windows were considered; 869 met the fixed label-fraction rules.
Three windows were chosen by maximum minimum rectangle-gap distance from the
development field and previous choices. Every gap had to be at least 1,024
native pixels. Both the full window and its guarded core needed at least 20%
high reference, 20% low reference and 90% nonzero reference. There was no
replacement after seeing CT or feature values.

**From the nominal scale and declared shape:** each 512 × 640 window spans
1.228288 × 1.535360 mm; three full test windows cover 5.657593 mm². This is a
small within-scroll experiment, with model-label selection. It is not a random
sample of the scroll, independent donors, or a test on complete text lines.

## 2. Actual AUC results

**From the complete [statistical output](results/statistics.json):** high proxy
labels are 179–255 and low proxy labels are 1–64. Zero and intermediate values
are excluded. Fixed signs are applied once; signs are never chosen separately
on test fields. The development replay uses its original 32-pixel guard; test
fields use the frozen 64-pixel guard. Every feature and the baseline use the
same support within a field.

| Field | High / low proxy pixels | Raw surface | Hilbert envelope | Wavelet scale slope |
| --- | ---: | ---: | ---: | ---: |
| Development replay | 115,503 / 119,915 | 0.714842 | 0.779617 | 0.796737 |
| Heldout 1 | 47,536 / 87,234 | 0.361635 | 0.626327 | 0.694818 |
| Heldout 2 | 65,325 / 92,608 | 0.608341 | 0.596448 | 0.733364 |
| Heldout 3 | 51,143 / 112,539 | 0.520231 | 0.524711 | 0.588971 |
| **Equal-field heldout mean** | — | **0.496736** | **0.582495** | **0.672384** |

**From the replay receipt:** the proxy-class counts match Gemini's published
counts exactly. All three AUC differences from the rounded published values
are below 0.0000033, within the frozen tolerance 0.00005.

**From the heldout outputs:** Hilbert improves on raw in two fields and is
slightly worse in the second. Wavelet improves on raw in all three. Raw
intensity reverses its development association in the first field, where its
fixed positive sign gives AUC 0.361635. This makes the relative improvement
there especially large. **My inference:** average improvement over this weak
baseline must not substitute for the absolute-performance gate.

## 3. Frozen decision, including the controls

**From all 199 paired block draws and all 199 spatial translations per field:**

| Criterion | Required | Hilbert envelope | Wavelet scale slope |
| --- | --- | --- | --- |
| Every field's AUC | ≥0.70 | Fail: 0 of 3 meet it | Fail: 1 of 3 meets it |
| Equal-field improvement over raw | ≥0.05 | +0.085760, pass | +0.175649, pass |
| Paired block 2.5%–97.5% interval for improvement | Lower end >0 | [0.030555, 0.131644], pass | [0.076990, 0.269845], pass |
| Family-maximum spatial rank | ≤0.05 | 0.17, fail | 0.01, pass |
| All replay/support/assessment checks | Pass | Pass | Pass |
| **Overall** | Every criterion | **FAIL** | **FAIL** |

**From the protocol:** the spatial comparison uses independently sampled
nonzero toroidal label translations for each field and the maximum of the two
features' macro AUCs in every null replicate. It does not compare with the null
mean. The paired block calculation resamples 20 fixed 128 × 128 blocks per
field, with identical multiplicities across features. No draw was dropped or
redrawn; all draws were assessable.

**My inference / statistical limit:** these ranks diagnose spatial
misalignment; they are not calibrated familywise false-positive probabilities
or proof of ink. The intervals describe block variation in these three fields,
not independent-scroll uncertainty. The reference is produced by an ink model;
there is no independently observed optical ground truth in this experiment.

## 4. Support, cost and execution integrity

**From [result.json](results/result.json):** every acquired depth profile used
for feature evaluation is fully in bounds. The first test field retains
192,460 / 196,608 guarded pixels (97.8902%); the other two retain 100%. The
common mask retains 100% of legacy sheet support in all four fields. All
source chunks were complete; no missing-chunk fill or HTTP retry occurred.

| Measured resource | Value |
| --- | ---: |
| Main execution wall time | 360.078 seconds |
| Peak RSS | 1,566,068,736 bytes |
| Response bodies, including metadata preflight | 155,477,363 bytes / 148.274768 MiB |
| Requests, including preflight | 84 |
| Acquired CT chunks | 80 |
| Numerical threads | 1 |
| GPU forwards / training updates / unread regions | 0 / 0 / 0 |

**From the [launch record](results/launch_attempt.json):** an initial detached
launcher did not survive to create an output/cache directory, stage receipt or
log content. The identical frozen command then ran in a managed foreground
session. The first launch is not a completed numerical trial; no parameter or
threshold changed between launch methods. This record is retained.

## 5. Independent numerical review

**From [review_result.py](review_result.py) and its
[receipt](results/independent_review.json):** 1,003 / 1,003 checks passed in
29.271 seconds. The reviewer imports no numerical module from the experiment.
It independently recomputes all 12 observed AUCs using SciPy ranks, all 1,791
shifted-label AUCs using sorted-negative pair counts, and all 1,791 bootstrap
AUCs using 20 × 20 block-pair comparison matrices. It reconstructs all random
draws and the resulting ranks, intervals and gates. Maximum numerical
difference is zero, against a 1e-12 comparison tolerance.

**From the same receipt:** all six recorded output artifact hashes, all 82
acquired asset bodies and all frozen local/Git hashes were checked. The
reviewer rechecks selection ranking against the saved eligible list and label
counts on selected crops. It does not independently regenerate the complete
eligible list, establish physical registration, rerun feature extraction,
reconstruct source-stack array hashes, or rederive sheet/profile masks from CT.
Those limits remain; numerical agreement is not biological or papyrological
validation.

## 6. Decision and reusable outputs

**My inference:** close this proposed promotion to inference. Retain the
wavelet result as a conditional proxy association, with this failed replication
gate attached. Any later fusion, sign change, new normalization, region
selection or feature variant is a new experiment and needs new prospective
evaluation. These three fields are now development information, not untouched
validation data for another variant.

**From the files:** the original maps, reference crops, masks and whole source
assets are preserved outside Git; public summary receipts contain only
already-published control locations. The raw/derived archive is identified by
[data_archive.json](results/data_archive.json). The frozen code remains
unchanged. The team-facing interpretation and assignments are in
[ChatGPT 031](../../collab/chatgpt-031.md).
