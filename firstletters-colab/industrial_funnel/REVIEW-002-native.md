# Review 002 — admitting native evidence without overstating screening

Date: 2026-11-24. Reviewed RFC branch at `eea64f4`, including amendment A01 and experiment 073; producer code below was inspected at `8f2e576`, and the cited Gemini/Claude control results at `5ce5fde`. This review supplements [Claude's review](https://github.com/JorisViaudQuantAI/Scrollprize/blob/67bd872d4a985ada8594f820781a90b5a3011920/firstletters-colab/industrial_funnel/REVIEW-001-claude.md) and [A01](AMENDMENT-001.md); it does not replace or rerun experiments 071 or 073. The source inspection covered the reference control plane, file receipts, survey, and the existing Gemini renderer/screener. Native-adapter results and their exact verification receipt are reported separately in the integration note and its execution receipt.

## Decision

Use the A01 file-first path for the next bounded pilot. Its inventory, stage ordering, explicit missingness, retained low-rank units, and refusal of cell-only physical deduplication are appropriate. The remaining immediate engineering task is a semantic adapter for the actual native maps and masks. A hash-correct file is not yet a usable observation, and a usable observation is not a successful search for letters.

Keep these three questions separate in every report:

1. **Was this exact computation reproduced?** Source, recipe, arrays, and hashes agree.
2. **Could this configuration recover legible known text?** A matched public positive, complete glyph or line, and documented visual review answer this question within their measured scope.
3. **What was observed on this particular unread surface?** Review may be inconclusive or identify no qualifying reading under the tested configuration. None of these outcomes establishes that the papyrus lacks ink.

## What A01 already resolves

| Earlier concern | Existing disposition to preserve |
| --- | --- |
| Expensive dense fit before inexpensive inspection | S0 anatomy and bounded S1 proposals precede dense fitting; verified existing geometry remains reusable. |
| A low anatomy or model score becomes an exclusion | Missing and zero-clean bands retain survey/repair routes; no automated blank state. |
| One mesh or voxel cell is counted as one sheet | 073 explicitly rejects common 64-voxel occupancy as sufficient deduplication evidence. Its failed cell-only rule is retained. |
| A heavy service is required before useful work | Immutable file receipts are the R1 default; the SQLite coordinator remains an optional reference. |
| Arbitrary fixed allocation is promoted as optimal | No fixed 70/20/10 default; actual cost and independent opportunities must inform later allocation. |
| Synthetic test success becomes scientific calibration | README and A01 explicitly separate reference tests from native producer validation and scientific promotion. |

The 071 ranking failure remains a failure: preserving that result matters more than adding a new favorable software-test count. Neither this review nor a native import should overwrite its gate or reinterpret its attempted population.

## Implemented contribution

[`native_evidence.py`](native_evidence.py) now supplies a read-only `validate_native(manifest, root)` API and an `assess_pair(a, b, root)` API, with a JSON CLI. They check native array bytes, dtype, dimensions, axes, score domain, crop/phase identity, and recompute the actual skipped-window and full-context masks. Pair validation checks both recorded plane indices and actual input/support arrays for exact reversal. These functions never issue a completion receipt or grant scientific admission.

The public author-0841 prepared sample has no explicit original source-support file. Its dedicated legacy descriptor therefore reproduces the documented all-true assumption, verifies the original receipt and its exact input hash/shape, and emits `ASSUMED_PREPARED_SUPPORT`. It does not manufacture native CT acquisition evidence. Unknown physical offsets and costs remain unknown.

[`test_native_evidence.py`](test_native_evidence.py) uses analytical small-canvas masks independently of the validator to test unsafe halos, overlapping Hann contributors, end anchors, empty-window counts, false per-depth reversal, recipe changes, malformed arrays, and attempts to promote control attestations into production completion. The native campaign replay and the combined verification receipt are reported in [NATIVE_INTEGRATION.md](NATIVE_INTEGRATION.md). The queue and file-receipt reference implementations are left unchanged; passing this adapter remains distinct from completing their producer execution contract.

## Immediate admission requirements

The current [`file_receipts.validate`](file_receipts.py) verifies CAS bytes and required role names. For paired inference, it can therefore accept five arbitrary byte strings named `asc_map`, `desc_map`, `asc_support`, `desc_support`, and `metadata`. This behavior is documented as a reference limitation; the native adapter must close it before such an artifact is described as admitted native inference.

| Requirement | Minimal behavior | Acceptance test |
| --- | --- | --- |
| Actual arrays | Load the declared binary format without pickle; require declared dtype, shape, axes, and hash to match the loaded object. Reject integer display images masquerading as native probability maps. | Correct hash but wrong dtype, shape, axes, or truncated payload fails. |
| Supported scores | Supported map values must be finite and within the declared probability domain. Invalid pixels are represented by masks, not inferred from score zero. | NaN or infinity within support fails; a genuine supported zero remains valid. |
| Distinct validity claims | Preserve model-output support, source availability, and full input-context support as separate masks. The review support must be a subset of every required mask. None is called an anatomical same-sheet mask without independent evidence. | Deliberately include a pixel with missing source or reflected model context in the review mask; reject it. |
| Exact depth semantics | Record the actual ordered input plane indices, signed depths, normal sign, and source-frame convention. A historical `asc` label alone does not establish exact reversal. | A mislabeled one-plane shift or duplicated/missing plane fails the declared recipe. |
| Crop and grid identity | Include source crop, output-to-source transform, model window phase, stride, padding, and blend in recipe identity. | The same shape with a different crop origin or stride produces a different identity; an old result cannot silently satisfy it. |
| Complete recipe | List required families and orders, and the actual observed outputs. A one-order observation remains partial for a paired recipe. | One usable map cannot create `COARSE COMPLETE` for a two-order job. |
| Calibration scope | Separate numerical replay/parity from visual positive recovery, exposure, surface type, and admissible deployment stratum. Importing prior evidence is not a new same-batch control. | Numeric agreement alone, an expired control, or a mismatched surface/recipe cannot yield a legibility-qualified admission. |
| Honest acquisition failures | Record unavailable inputs and interrupted or failed work explicitly. Support and costs remain visible even when a map is unusable. | Missing input, partial transfer, or unsupported pixels cannot become a no-text observation. |
| Restart and lineage | Bind results to exact source/recipe artifacts and immutable attempts. Preserve failed attempts and retries; verify referenced bytes again on resume. | Corrupted source/output, conflicting duplicate identity, and partial-before-complete replay are distinguished. |

**Status after array validation:** importing existing files establishes an evidence observation under their recorded configuration. It does not attest that a worker ran under a new grant, that dependencies were ready at dispatch, or that calibration was current when the historical computation ran. Those facts require original producer evidence. The adapter must leave unmeasured fields explicit instead of manufacturing `worker_stopped`, resource use, or control success.

## Specific scientific traps to avoid in the real producer adapter

### Layer order is not anatomical face identity

The Gemini runner calls Studio's historical order convention. For a 31-plane stack centered at 15, it selects `desc = 23..8` and `asc = 7..22`; these operations combine reversal with a one-plane change of sampled support. A-hn instead reverses the same `3..26` slab. The exact convention is recorded in [Gemini 007](https://github.com/JorisViaudQuantAI/Scrollprize/blob/5ce5fdef8b24d39a6a5fc94ab5daf644e384c674/collab/gemini-007.md). Neither label proves a recto/verso assignment, and these observations are not independent samples of a papyrus.

[`screening_fp_filter.py`](https://github.com/JorisViaudQuantAI/Scrollprize/blob/8f2e576ff961a4f247fb545b5317b223988e57be/gemini_lane/screening_fp_filter.py) still exposes `ink_like_candidate = not any(flags.values())`, using order asymmetry, overlap, morphology, and optional cross-family agreement. The inspected screener does not itself use that Boolean to stop inference. Preserve these outputs as diagnostic features, not a general exclusion rule. Gemini 007's measured PHerc0211 winner matrix passes this filter in all six configurations, but its Q34 global criterion fails on real visible rows. That evidence limits Q34; it does not establish that every heuristic filter is either universally valid or already falsified.

### A winner-quality mesh is not the same calibration stratum as an automatic fit

[Claude 025](https://github.com/JorisViaudQuantAI/Scrollprize/blob/5ce5fdef8b24d39a6a5fc94ab5daf644e384c674/collab/claude-025.md) reports readable PHerc0211 text on an independently fitted surface only after per-surface flattening and Studio stride 48 in the reading order. The unflattened fit showed no letter. Its initial blind reader-v3 montage review missed the text even after flattening. Different neighboring windings carry different parts of the awarded row. Consequently, the successful stride-96 test on the winner mesh cannot establish sensitivity on arbitrary automatic fits.

Required evidence for a cheap-mode promotion is a frozen comparison on the actual producer's geometry class, including full maps, exact observed support, both orders as a baseline, both mirror states for review, and per-original-positive results. Perturbations of one positive remain correlated robustness tests.

### Crop, phase, stride, and blending must not be conflated

A context extension may make previously padded pixels admissible without providing an independent replication. Changing crop origin, stride, and blending together can remove visible seams but cannot isolate the causal contribution of blending. The adapter should preserve each recipe and support mask separately; later experiments can vary one factor at a time. A seam diagnostic is a quality measurement, not evidence that residual contours are ink.

### Area denominators are different observations

[`methods_ledger/07_negative_coverage.md`](../methods_ledger/07_negative_coverage.md) already distinguishes chart, valid, on-sheet, and core area and explicitly states that unique-sheet area is unknown. Retain that distinction. Pixel count times nominal voxel pitch squared is a chart-area proxy when a fitted chart has local distortion; it is not automatically physical area. Common coarse occupancy, overlapping bounding boxes, reverse order, and different model families cannot establish independent coverage. Exact source-and-recipe duplicates may be skipped; physical deduplication needs verified correspondence and uncertainty.

## Existing producer issues to address when it is connected

These are observations about the inspected Gemini consumer runner, not claims that the RFC itself implements them.

| Source | Concrete behavior | Minimal next change |
| --- | --- | --- |
| [`coverage_inventory.py`](https://github.com/JorisViaudQuantAI/Scrollprize/blob/8f2e576ff961a4f247fb545b5317b223988e57be/gemini_lane/coverage_inventory.py) | Merges live asset queries with `RECORDED_TEAM_STATUS` fixed to 22 November, including PHerc0800 `P4-exclude` because public maps already exist. Its S3 listing helper handles only one page. | Attach distinct source-snapshot and team-status dates; mark map availability separately from reviewed admissible coverage; paginate S3 and report incomplete listings explicitly. |
| [`screen_scroll_rtx4060.py`](https://github.com/JorisViaudQuantAI/Scrollprize/blob/8f2e576ff961a4f247fb545b5317b223988e57be/gemini_lane/screen_scroll_rtx4060.py), `screen_one_mesh` | Output directory identity uses scroll, mesh tag, and optional crop but omits stride, snap recipe, weights, code, and source hashes. Repeated recipes can overwrite the same summary/artifact names. | Immutable attempt and recipe identifiers; retain native maps/masks and exact provenance for completed attempts. |
| Same file, `find_top_hotspots` | Native crops are selected only from the highest-p99.5 Studio order when Studio exists; A-hn-only or low-score findings may receive no native crop. Grid endpoints are not added when the image size is not divisible by the selection step. | Separate diagnostic rankings per family/order, retain full native maps, include endpoint coverage and predeclared low-rank review opportunities. |
| Same file, renderer/inference boundary | The renderer can expose separate source-support masks, but the runner requests only its legacy two-output form. Studio validity is recomputed from volume contents, while A-hn uses a central-plane nonzero mask and its weighted output support. | Pass source and input-context masks through explicitly; quantify admissible support for every map rather than deriving acquisition from intensity. |
| [`grow_m7_patch.py`](https://github.com/JorisViaudQuantAI/Scrollprize/blob/8f2e576ff961a4f247fb545b5317b223988e57be/gemini_lane/grow_m7_patch.py), `ByteBudget`/`BudgetedRemoteZarrV2` | Shared byte budget checks before transfer and charges afterward; it does not reserve bytes atomically. Multiple workers may pass the same remaining-budget check before any transfer is charged. Compressed responses are charged after transfer. | If a strict transfer cap is required, add a synchronized reservation and bounded streaming read shared by CT and m7. A deterministic two-worker barrier test should demonstrate that combined reservations cannot exceed the cap. |

The last issue is a resource-enforcement gap in a producer, not a reason to launch or rebuild a scheduler. Likewise, the existing reference `Store.put` fsyncs the object file but not the CAS directory after publishing it. Resume verifies bytes and therefore fails closed on a missing object, but power-loss durability of a newly published object is not guaranteed by that implementation. Add directory fsync if power-loss persistence is claimed; distinguish it from the already tested ordinary process-restart behavior.

## Remaining rollout blockers and owners

| Blocker | Evidence needed | Owner / route |
| --- | --- | --- |
| Native evidence admission | Real maps and distinct masks pass semantic validation; adversarial malformed versions fail. Import status remains separate from legibility/calibration status. | ChatGPT native adapter and independent regression review. |
| Native paired production receipt | Exact source/recipe, same-configuration control, both required orders, measured resource use, original producer exit state, and durable outputs. | Claude/Gemini producer adapters. |
| Search sensitivity on fit-quality data | Frozen actual PHerc0211 fit-band comparison and blind review; account for missed, invalid, and no-proposal units. Preserve the failed 071 ranking gate. | Claude data export; ChatGPT frozen replay; independent reviewer. |
| Independent physical accounting | Verified same-sheet correspondences, source-frame mapping, distortion/area metrics, and uncertainty; occupancy alone remains insufficient. | Geometry lane; keep 073 unchanged. |
| End-to-end resource enforcement | CT/m7 transfer cap, worker CPU/GPU time and memory limits, checkpoints, interrupted-job receipt, and accounting of retries. | Actual worker owner before multi-job rollout. |
| Statistical statements about unresolved regions | Frozen physical population, actual inclusion probabilities, complete adjudication or explicit unresolved status. | Survey producer plus reviewer; [`survey.py`](survey.py) does not certify its input population. |

No new all-scroll ranking, ink-probability prior, automatic rejection threshold, or dense-fit campaign is supported by this review. The next useful result is a small real producer-to-receipt replay that states exactly which of these blockers it closes.
