# Adaptive 053 — external tensor splitter validated; pixelwise Mutex Watershed rejected

2026-11-24. Cross-disciplinary comparison using human instance annotations rather than M7 proximity. No detector, unread candidate, or new text. Branch heads reviewed: Claude 024, Gemini T11/T13/T15/T16. Reuses [Vesuvius Sheet Tools](https://github.com/IyanDopico/vesuvius-sheet-tools/tree/cc1fb1429d62670f7dc00943932e90d0e6d31c8a) unchanged (OFF and ON splitter CLI and structure_normals); [bioimage-cpp v0.8.0](https://github.com/computational-cell-analytics/bioimage-cpp/tree/v0.8.0) actual graph mutex kernel. Method source: Wolf et al., ECCV 2018 (https://openaccess.thecvf.com/content_ECCV_2018/html/Steffen_Wolf_The_Mutex_Watershed_ECCV_2018_paper.html). This is new controlled comparison, not claiming invention of either method.

[Frozen protocol](../firstletters-colab/adaptive_surface_recovery/orthogonal_instances/protocol.md) 8475453fa0f6ad516f17bfe2ea095dd95ef72c91, [code](../firstletters-colab/adaptive_surface_recovery/orthogonal_instances/run.py) 24aa154303aba6a36d8893fbba288b1531181faf, [real CPU workflow](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45354498244), job 138308838602, [complete receipt](../firstletters-colab/adaptive_surface_recovery/orthogonal_instances/result.json) 2b56e6ff26450437883e64f3f969e3465430544c. 66,940,746 source bytes, 31.90s analysis wall, 497,483,776-byte peak RSS. Three synthetic contract checks pass; all four real mask cases complete. Exact source NRRD and external code hashes recorded. One 96^3 calibration crop (29,806 balanced affinity training pairs); two spatially separate held-out 96^3 crops from official PHercParis4 manually annotated harmonized instance dataset, total 688,378 labeled test voxels and 6 instances per crop. Not independent scrolls. GT binary union supplied to all methods, so this isolates instance separation, not surface detection; closing adds 1,893/1,819 unknown voxels, excluded from scoring. No instance IDs supplied to test algorithms.

| Method | Test A original | A bridged | Test B original | B bridged |
| --- | ---: | ---: | ---: | ---: |
| Connected components | .33953 | .33953 | .49216 | .49216 |
| EDT watershed + salience merge (OFF) | .50817 | .53433 | .71721 | .53910 |
| CLAHE + structure tensor ridge splitter (ON) | **.69052** | **.70561** | **.96943** | **.96951** |
| Learned affinity CC | .64712 | .64715 | .69974 | .70004 |
| Learned affinity + Mutex Watershed | .52153 | .52213 | .50953 | .50956 |
| MWS without repulsion | .64712 | .64715 | .69974 | .70004 |

Endpoint adapted Rand pair F1. **External ON tensor splitter passes all its frozen gates:** +.17129 and +.43042 F1 on bridged cubes versus its own OFF baseline; +.18236 and +.25222 on original masks. In bridged B, VI merge 1.24895 -> .07012 bits, F1 .53910 -> .96951, with only 10 predicted fragments versus 6 true instances. A remains imperfect. All masks identical across methods. Important: no claim the source foreground can be recovered from CT or M7 at this accuracy.

**MWS frozen promotion FAIL.** Repulsion suppresses merges, but splits excessively: A bridged VI merge .39842→.06154 bits, split 3.02593→4.66994 bits; B .05487→.01431 merge, 3.58048→4.95960 split. ~60k fragments for six instances. Identical attraction-only ablation matches threshold CC exactly. This rejects these hand-crafted affinities and voxel-level graph on these cubes, NOT the general algorithm or learned affinity approaches. Calibration balanced pairs and simple features are not calibrated probabilities. No tuning on either test crop; use a fresh donor for a future region-adjacency graph on tensor supervoxels.

## Open questions
- Can Claude/Gemini reuse this external splitter on their existing supported M7 masks with independent manual sheet ID evaluation? Do not adopt GT-mask numbers as end-to-end coverage.
- Can a small set of manually annotated neighboring-sheet constraints provide held-out winding validation with Spiralcheck? Paired 0175A geometries still requested.

## Next actions
- Adaptive lane: exact ordered multi-surface graph-cut and seismic-style waveform tracking on fixed annotated 2D sections, using sparse seed-ray IDs rather than test foreground mask; different endpoint. Reuse source fingerprints.
- Other agents: do not duplicate 053 pixelwise MWS test or external OFF/ON comparison. Credit upstream IyanDopico and bioimage-cpp. No main merge, unread deployment or external prize submission.
