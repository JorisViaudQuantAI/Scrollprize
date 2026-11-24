# Adaptive 057 — true 3D ordered hybrid: positive feasible case, global gate not met

2026-11-24. [Frozen protocol](../firstletters-colab/adaptive_surface_recovery/dtw_order_3d/protocol.md) cda47c460f801e079c32978e86715c50ab2d87ef (before 056 outcomes); [code](../firstletters-colab/adaptive_surface_recovery/dtw_order_3d/run.py) 084e1d89a7dedd820433db5dbeebc06cf0be885b; [executed workflow](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45361701649), job 138333111652; [complete numerical result](../firstletters-colab/adaptive_surface_recovery/dtw_order_3d/result.json) 83cc1bddbd96250d3158b567dbdfa7c057a612eb. Three fresh official manual Paris4 instance cubes, 46,158,321 source bytes, 19.941s analysis wall, peak RSS 381,988,864 bytes. All four synthetic gates pass including 2 exact exhaustive 3D and row-only objective pairs, lateral transposition, crossing control, two-axis translated waveform. All executed cuts verify exact mincut energy and active hard constraints. No GT foreground except central seed-ray IDs/depths; actual CT elsewhere.

**Global gate not met**: first and third predeclared cubes each have only one eligible seed run (minimum two); no replacement. Middle cube has 3 seed sheets on 32×32 lateral grid over 96 depths; 2787/2997 eligible sheet-ray comparisons. Four manual sheet IDs each >=64 voxels exist in that volume: the unseeded fourth contributes 531 eligible ray pairs (16.00% of seeded+unseeded unique-run population), not discovered by any arm. No claim to recover all sheets.

| Arm (only assessable cube) | Within2 | Correct ID | MAE vox | Gap violations | Axis1 / axis2 slope violations |
|---|---:|---:|---:|---:|---:|
| Local | .65124 | .60926 | 4.807 | 79 | 539 / 711 |
| Independent 3D cut | .77933 | .71654 | 1.201 | 0 | 0 / 0 |
| Joint 3D cut | .77933 | .71654 | 1.201 | 0 | 0 / 0 |
| Tree DTW | .61751 | .65913 | 2.346 | 0 | 0 / 127 |
| Projection only | .63114 | .66344 | 2.294 | 0 | 0 / 0 |
| Fixed prior K | .93254 | .85791 | 1.036 | 0 | 0 / 0 |
| Reciprocal prior H | .93147 | .84822 | .947 | 0 | 0 / 0 |
| Row-only Hrow | .92070 | .84177 | .972 | 0 | 14 / 0 |

H improves within2 by 31.396 pp, correct ID by 18.909 pp and MAE by 59.66% vs W. Zero wrong-ID hits for all regularized methods, but background misses remain. H passes all per-cube gates on this one cube. Additional cross-row coupling yields only +1.076 pp within2 vs Hrow, below +3 pp criterion; it enforces the previously violated first-axis slopes. K slightly better within2 and ID, H lower MAE, no superiority of weighting established. Reversing tree propagation axis order changes 78.81% of predictions, median 2 vox/p95 7, proving traversal dependence. This is a real two-lateral-axis surface optimization, **not** a validated whole-scroll reconstruction or generalization result. 056 remains negative.

## Open questions
- Sparse anchors limit surface discovery and multi-sheet test eligibility; an independent protocol with a different fixed anchor budget is needed.
- Transfer to independent manuscripts requires verified sheet meshes; a single-sheet reference can validate geometry only, not adjacent-sheet ID, and cannot substitute for the multi-instance benchmark.

## Next actions
- Reuse exact 057 solver and 055/056/057 negative results. Do not repeat 3D control or change thresholds to promote.
- Test independent already-published winner meshes on other scrolls with native CT and geometry withheld except at one seed; distinguish single-surface placement from multi-sheet ID. No unread data, merge or prize submission.
