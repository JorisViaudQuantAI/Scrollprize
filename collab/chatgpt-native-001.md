# Native 001 — visual evidence published and RFC native adapter verified

2026-11-24. Contribution based on industrial RFC/A01 `eea64f485ba59fd6b924edc3ce1150381dd8b25d`; the cross-lane catalogue also reviews Claude integration `67bd872d4a985ada8594f820781a90b5a3011920` and Adaptive `1289d99ba96c34be325dc00522c468e1861a7d5c`. The owner explicitly requested publishing this lane's recent visual results and reviewing/implementing useful industrial-pipeline improvements. This contribution has no confirmed new reading and does not merge main or start a GPU search.

## Published recent results

The [visual evidence package](../firstletters-colab/visual_discovery_20261124/README.md) contains seven original figures, seven frozen agent readings, thirteen inference receipts, selected float32 probability/mask crops, the known PHerc0841 labelled subset, and 7,776 saved native-value comparisons. Its 54 files total 9,137,736 bytes. Original and sanitized-file hashes are distinct; the complete CT/model inputs and checkpoint are not included.

- PHerc0800 C: the O-like contour remains a morphological hypothesis, not a validated letter or new-text claim. Adding real left context makes 15,360 additional pixels in the fixed candidate box admissible; 57,600 already admissible pixels remain bitwise identical. The latter are not an independent replication.
- PHerc0125: recropping/stride-48/Hann reduces discontinuities at the original grid from approximately 34 to 1 display levels, but yields no robust glyph. Crop, phase, stride and blending change together.
- PHerc0800 renderer check: 7,744/7,776 recorded native values match exactly and the remaining 32 differ by one uint8 level. This validates sampled renderer conventions, not anatomical sheet identity or text.
- Known public PHerc0841: AUC 0.833212667 and AP 0.788154413 on 67,113 known ink and 111,310 known background pixels; 608,009 unknown pixels excluded. No independent-holdout claim.

The [public verifier](../firstletters-colab/visual_discovery_20261124/verify_evidence.py) recalculates those numerical results without a model; [executed receipt](../firstletters-colab/industrial_funnel/visual_evidence_verification.json). Historical total: 1,078 map windows plus thirteen identity sentinels = 1,091 forwards. This publication/integration adds zero forwards.

## RFC review and implementation

The [federated catalogue](../firstletters-colab/industrial_funnel/RESULTS_CATALOG_20261124.md) indexes results across the reviewed snapshots. The [independent review](../firstletters-colab/industrial_funnel/REVIEW-002-native.md) and [implementation note](../firstletters-colab/industrial_funnel/NATIVE_INTEGRATION.md) retain A01's file-first S0/S1 approach. The 071 fifty-percent stopping gate still fails; the 073 cell-only deduplication failure is unchanged.

New code validates actual NPY arrays and recomputes source/output/full-context masks, skipped windows and exact reversed-input pairs. The importer audits all thirteen actual maps, identifies four same-configuration exact reversal pairs and seven repeated job declarations, and preserves the five observations without a paired reverse. It records the older 0841 all-present support assumption explicitly instead of inventing a CT acquisition mask. It also rejects contradictory crop extents, duplicate completed records and recipe/accounting conflicts.

Exact pixel union within the unchanged PHerc0800 chart gives **310,176 full-context chart pixels**, against 774,144 summed across its five maps. These are pixel accounting quantities, not unique physical papyrus area. Raw and snapped PHerc0125 preparations remain separate. Narrower PHerc0125 refinement crops improve seams while losing full context near their new boundaries; preserve the larger parent map and compute all-contributor halos before follow-up.

[Native campaign receipt](../firstletters-colab/industrial_funnel/native_campaign_result.json); [98/98 local tests](../firstletters-colab/industrial_funnel/native_tests.json): 48 existing reference tests +35 native-array tests +15 importer regressions, no failures/errors/skips. All successful audits still report `scientific_admission: NOT_ESTABLISHED`, `production_complete: false`. This is an actual historical producer-data integration, not a newly admitted worker execution or measured GPU saving.

## Open questions

- Claude/Gemini: the full fit-quality PHerc0211 native coarse/fine export, masks, matched controls, actual costs and complete attempted-unit population remain the key R1/R2 inputs. Existing winner/display controls cannot replace them.
- Geometry reviewers: distinct chart preparations and nearby 3D cells still require physical correspondence before area union or same-sheet deduplication.
- Worker owners: native semantic checks must be joined with original attempt/dependency/control/resource/stop receipts before production completion. The existing reference queue is unchanged.
- Visual review: the candidate is an unconfirmed contour with a material-edge alternative; it must not become a positive training label or prize claim.

## Next actions

Review the native manifest API and adversarial fixtures in the contribution branch, then connect one real producer export to that interface. Freeze the complete coarse/fine comparison before selecting a budget cutoff. Reuse verified historical outputs for exact requests and keep missing support, calibration and review visible as unresolved work.
