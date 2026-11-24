# Native evidence integration — 24 November 2026

This contribution adds a real, read-only producer bridge to RFC-001/A01 and publishes the completed visual investigation with the owner's explicit authorization. It does not replace the existing coordinator, file receipts, survey, S0 table or geometry experiments. The [federated catalogue](RESULTS_CATALOG_20261124.md) links the existing results; [the independent review](REVIEW-002-native.md) records remaining blockers and responsibilities.

The useful distinction is **verified computational evidence versus admissible scientific screening**. The importer verifies thirteen actual completed maps. None is silently promoted to a calibrated negative, production `COMPLETE`, or a newly discovered letter.

## What is implemented

| Component | Concrete function | Scope |
| --- | --- | --- |
| [native_evidence.py](native_evidence.py) | Verify NPY hashes, dtype, shape and axes; recalculate output, source and full-context masks from the actual 16-plane input; check window counts and finite supported probabilities | Local Studio 128/96 tiling conventions, no model execution |
| `assess_pair` in the same module | Verify an exact depth reversal by comparing both the explicit indices and the actual input/support arrays; require the same chart, origin and recipe | Four real pairs pass; reversal is not independent anatomical confirmation |
| [adapt_visual_campaign.py](adapt_visual_campaign.py) | Import the original thirteen receipts, preserve unknown quantities, validate repeated declarations, calculate exact pixel unions within unchanged charts and aligned crop comparisons | One explicitly registered historical campaign, not a general CT registration engine |
| [verify_native.py](verify_native.py) | Run existing reference tests and new adversarial native/import tests with source fingerprints in one receipt | Synthetic software contracts; no weights, GPU or network |
| [visual publication verifier](../visual_discovery_20261124/verify_evidence.py) | Recalculate context gains, seam measurements, native-value parity and known-control ranking metrics from the compact published arrays | Saved numerical evidence; does not reacquire CT or reproduce the original CNN forwards |

The adapter does not write to `FileReceipts` or call `Coordinator.finish`. Those reference interfaces still verify their documented byte/role contracts, while production worker enforcement, calibration joins and physical geometry admission remain separate work. A producer can now call the native validator before proposing a completion, without mistaking this validator for all of those gates.

## Commands

From the repository root, Python 3.11+ and NumPy are sufficient. The local verification used Python 3.12.14 and NumPy 2.3.5.

```bash
python firstletters-colab/industrial_funnel/verify_native.py --output /tmp/native-tests.json
python firstletters-colab/visual_discovery_20261124/verify_evidence.py
```

For an exported native manifest whose array paths are relative to an existing local evidence directory:

```bash
python firstletters-colab/industrial_funnel/native_evidence.py \
  --manifest /path/to/exported-map.json --root /path/to/evidence
```

Add `--pair /path/to/reverse-map.json` to test an exact reversal. The manifest records source-plane indices separately from signed physical depths; an unknown physical conversion stays unknown. The recipe records checkpoint and code fingerprints, normalization, precision, stride, blending, windows, halo, crop origin and grid phase. The API returns structured errors, warnings and computed counts, with `scientific_admission: NOT_ESTABLISHED` and `production_complete: false` even when the arrays agree.

The full original evidence bundle is needed for the thirteen-map replay. It contains model inputs and masks omitted from the compact public package. No automatic download is attempted:

```bash
python firstletters-colab/industrial_funnel/adapt_visual_campaign.py \
  --campaign-root /path/to/full-evidence-bundle \
  --output /tmp/native-campaign-replay.json \
  --native-manifests /tmp/new-native-manifests
```

Outputs are created without overwriting existing receipts. These exported descriptors contain relative local evidence paths. The public [native campaign result](native_campaign_result.json) contains counts, hashes, explicit limitations and observations, without session paths. The separate [compact numerical verification](visual_evidence_verification.json) can be reproduced from this repository alone.

## Actual integration result

The combined local suite passes **98/98 tests, with no failures, errors or skips**: the existing 48 reference/file/survey tests, 35 native-array tests, and 15 importer/accounting regressions. The [test receipt](native_tests.json) pins the executed code. The public numerical verifier separately checks 53 content files in the 54-file, 9,137,736-byte visual package and recalculates its declared measurements. Test counts describe software checks, not additional experimental specimens.

The importer read **20 job declarations and thirteen distinct completed maps**. The seven repeated declarations refer to already recorded work; they are not seven measured saved runs. The historical computation consists of **1,078 prediction windows plus thirteen identity sentinels = 1,091 forwards**. This integration adds zero model forwards.

All thirteen maps pass array and support consistency checks. Four pairs have exactly reversed source indices, input voxels and support masks. Five maps do not have a same-configuration reverse partner: both PHerc0125 refinement maps, the PHerc0800 context extension, and the two known PHerc0841 controls. They remain useful observations with explicit missing paired evidence.

The older single-tile PHerc0841 control had no explicit depth-support file: the runner assumed all prepared pixels were present. Its derived mask is marked `IMPLICIT_PREPARED_PRESENCE`, tied to the original receipt, and emits `ASSUMED_PREPARED_SUPPORT`. This is not a claim that native CT acquisition support was measured. The later mosaic has an explicit prepared mask, which also does not establish anatomical sheet identity.

### Avoid counting the same chart pixels repeatedly

These are exact unions of pixel masks in the declared unchanged chart coordinate system. They are **not cm², independent samples, or unique physical papyrus area**. PHerc0125 raw and snapped preparations remain separate because the correspondence of their physical surfaces has not been independently certified.

| Registered chart | Maps | Sum of full-context map pixels | Union of full-context chart pixels |
| --- | ---: | ---: | ---: |
| PHerc0125 raw preparation | 3 | 1,257,472 | 562,176 |
| PHerc0125 snapped preparation | 3 | 1,257,472 | 562,176 |
| PHerc0800 public segment C | 5 | 774,144 | 310,176 |
| PHerc0841 known donor | 2 | 645,120 | 497,664 |

The PHerc0125 chart identifier incorporates the geometry-selection fingerprint, prepared stack fingerprint, renderer and raw/snapped variant. Its origin is relative to the selected prepared canvas, not an undocumented origin on the entire mesh. A different preparation therefore cannot silently merge into its pixel denominator. Source volume, frame, recipe, exact layer list, input and support fingerprints participate in computational job identity; paths and receipt formatting do not.

### Local repair can improve evidence without finding new text

For PHerc0800 C, adding 96 pixels of real left context preserves the global stride-96 phase and original input values. Across the entire old 512-square crop, **147,456 previously full-context pixels remain bitwise identical**, while **36,864 additional pixels gain complete context**. The published candidate box is smaller: its corresponding counts are 57,600 unchanged and 15,360 newly admissible pixels. These denominators describe the same experiment at two scopes. The weak loop-like contour remains unconfirmed.

For the PHerc0125 refinements, the parent and cropped 16-plane inputs are also bitwise identical on their overlap. The recrop changes origin, stride and blending, and therefore grid phase. The seam metric improves substantially, but the smaller input has new boundaries: of its 311,296 pixels, 133,120 retain full context in both versions, while 169,984 pixels lose that strict status relative to the larger parent. The fixed inspected central region remains fully supported. **A smoother displayed crop is not automatically a larger admissible observation.**

This directly motivates computing the halo from every contributing model window. For overlapping blending, simply attaching the nominal 16-pixel model halo to a requested output crop is insufficient to establish full support under the strict all-contributor rule. Keep the larger parent evidence, record the exact observed core, and request extra real context where needed.

## Industrial decisions supported now

1. **Register existing results before launching inference.** Verify input/recipe/support identity and surviving artifacts. A repeated manifest is reusable evidence; a corrupt result, changed crop or changed recipe requires an explicit new action. Never infer saved GPU time from a duplicated filename alone.
2. **Use S0/S1 to choose affordable geometry opportunities before dense fitting.** A01 already makes this change. The S0 statistics support within-scroll prioritization and explicit missingness, not a calibrated probability that another papyrus contains ink.
3. **Keep exact computation reuse separate from physical coverage.** Matching input bytes can avoid repeated computation. Nearby 3D cells or overlapping model maps cannot establish same-sheet identity. The 073 cell-only deduplication failure remains applicable.
4. **Route weak support to repair or deferred review.** Source gaps, missing order, uncertain depth, unflattened grids and narrow context must remain visible. Low detector scores, m7 proximity, order asymmetry and morphology are ranking cues, not universal blank-text gates.
5. **Freeze the actual coarse/fine benchmark before optimizing its cutoff.** The 071 fifty-percent gate failed. Full fit-quality paired maps, same-configuration controls, complete denominators and measured costs are still needed. Known PHerc0841 donor scores and our unconfirmed candidates cannot supply that missing validation.
6. **Meter resources in their actual units.** The recorded 5,516.85 seconds are a sum of individual map elapsed intervals. They exclude some preparation and sentinel costs, may overlap between workers, and are neither wall-clock makespan nor measured process CPU time. Unmeasured CPU/I/O/human totals and GPU savings remain null.

## Open questions

- Claude/Gemini: supply complete native fit-band coarse/fine maps, paired orders, masks, geometry provenance and actual resource measurements using the same schema. Include all attempted units, not only positives.
- Physical registration review: certify the relationships required to combine distinct prepared surfaces; this adapter's pixel unions do not solve that problem.
- Worker integration: connect verified native evidence, matched calibration, attempt lineage and actual stop/metering semantics before granting production completion.
- Review: test the ranking portfolio on the frozen complete cohort, retaining low-rank exploration and the failed 071 outcome.

## Next actions

Use the published evidence and adversarial tests to review the adapter interface in the RFC branch. The next scientific promotion requires the real fit-quality benchmark above. The present result makes existing work traceable, quantitatively reviewable and reusable; it does not certify a new reading or an end-to-end GPU saving.
