# Adaptive 058 — cross-manuscript single-sheet native CT transfer: global gate not met

2026-11-24. [Frozen protocol](../firstletters-colab/adaptive_surface_recovery/dtw_cross_scroll/protocol.md) 2aa70fa37e002271241b4bb079ef428427504a50; [runner](../firstletters-colab/adaptive_surface_recovery/dtw_cross_scroll/run.py) 7c7ea6a66bfb35739cdf359e57da7229acb16edf; [executed workflow](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45362220084) job 138334858325; [full receipt](../firstletters-colab/adaptive_surface_recovery/dtw_cross_scroll/result.json) 3066b29a7874a10dd0728ceec3712e36ece94382. All synthetic solver/reference checks pass. Two distinct already-read manuscripts, PHerc0211 and PHerc0343, native CT and exact published winner mesh hashes. 23,075,664 CT bytes, 17 charged requests, 35.7219s analysis, peak RSS 242,290,688 bytes. No ink, CNN, GPU or unread locations.

6 fixed reference ROIs, 3 pass single-sheet projection coverage (999 scored rays each). The other 3 have two disconnected local mesh components projecting onto the same native rays: 1024/1024 ambiguous in 0211 q=.5 and 0343 q=.2, 1006/1024 in 0343 q=.5. **No CT was read for those rejected ROIs.** Retain these as reference-model failures, not failed ink or CT predictions. No replacements, no loosening of single-sheet gate. This is why a single-valued reference is insufficient even on a winner's mesh.

| Assessable ROI | W DTW within2 / MAE vox | I independent cut | H hybrid |
| --- | --- | --- | --- |
| 0211 q=.2 | .26426 / 4.255 | .41942 / 3.029 | .37137 / 3.174 |
| 0211 q=.8 | .48248 / 3.212 | .52152 / 2.038 | .57758 / 2.721 |
| 0343 q=.8 | .34334 / 3.391 | .41041 / 3.096 | .37738 / 3.229 |
| Descriptive equal-ROI mean | .36336 / 3.619 | .45045 / 2.721 | .44211 / 3.041 |

On 2997 assessable rays H improves vs W within2 by 7.875pp and MAE by 15.97%, removes all 596 W slope violations. But I has slightly better within2 and lower MAE overall on assessable fields; the hybrid's incremental advantage is NOT established. Global gate fails (3/6 assessable, no full scroll macro); no cross-manuscript single-sheet promotion. Each ROI only one integer mesh seed, CT elsewhere; the published mesh is a geometric reference, not independent material-center annotation. 0343 has a published interpolated seam. DTW traversal changes 77.34–85.06% of depths in these cases.

## Open questions
The three ambiguous reference ROIs contain multiple local surface crossings. Can the existing ordered solver recover them jointly, using the original connectivity to define separate reference graphs rather than forcing single-sheet projection? Local disconnected components need not be distinct global windings.

## Next actions
Freeze a separate multi-component reference protocol on precisely the previously unassessable 058 ROIs, before their CT access. Use all source components and exact central anchors; do not retroactively promote 058 or cherry-pick assessable results. Compare to the simpler ordered cut. No unread inference or external submission.
