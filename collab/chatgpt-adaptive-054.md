# Adaptive 054 — seismic DTW positive; coupled graph cut nonpromotion

2026-11-24. Orthogonal tracking benchmark distinct from 053 oracle-mask instance partition. Source: two same spatially separated manually annotated PHercParis4 test cubes, NRRD SHA256 and normalized CT bytes verified against 053. 12 preplanned sections, 11 evaluable; one section has only one unique seed run and is explicitly unassessable. Each predictor gets one annotated ray with seed depths, otherwise **only CT**, not the ground-truth foreground. 1,793 evaluated labeled sheet/column pairs (A 1,055 / B 738), excluding the seed neighborhood; 755 additional planned pairs without a unique GT run are unknown and excluded. Same scroll and same cubes as 053, different endpoint, no independent-scroll inference.

[Protocol](../firstletters-colab/adaptive_surface_recovery/ordered_tracking/protocol.md) a58ebfe6dccb63425c9b8f872aa9aded0aeac65a; [code](../firstletters-colab/adaptive_surface_recovery/ordered_tracking/run.py) 973f627828991780c47ef5207993c9d9044a1c3d; [actual CPU run](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45355019698), job 138310576733; [full receipt](../firstletters-colab/adaptive_surface_recovery/ordered_tracking/result.json) f33984ade67160e3233385993bdea7d853b37508. 50,643,380 source bytes, 4.6433s analysis wall, 234,872,832-byte peak RSS, no CNN/GPU. Four tiny exact-cut energies equal exhaustive enumeration and reflected energies; planted crossing blocked by joint constraints; dtw-python 1.5.3 identity and shifted waveform checks pass. Every real graph solution independently verifies feasibility and integer objective = mincut + unary constant.

Inspired by Li et al. TPAMI 2006 doi:10.1109/TPAMI.2006.19 (ordered graph search), Jin et al. JGE 2017 doi:10.1088/1742-2140/aa7309 (seismic DTW), and Giorgino JSS 2009 doi:10.18637/jss.v031.i07 (DTW implementation). These are bounded adaptations, not reproductions of complete methods.

| Arm | Within 2 vox A / B | Correct manual ID A / B | Wrong manual ID A / B | MAE vox A / B |
| --- | ---: | ---: | ---: | ---: |
| P local waveform template | .22938 / .16938 | .38009 / .27236 | .41611 / .19241 | 17.692 / 15.611 |
| I independent regularized cut | .34408 / .24932 | .61137 / .39837 | .23128 / .08266 | 8.298 / 9.801 |
| J joint ordered cut | .35450 / .24932 | .58957 / .39837 | .22844 / .08266 | 8.757 / 9.801 |
| W adjacent trace DTW | .31090 / .31030 | **.71280 / .72764** | **.16398 / 0** | **6.168 / 4.182** |

**W passes all frozen advancement gates in both cubes.** Correct ID +33.27 and +45.53 percentage points versus P; MAE decreases 65.14% / 73.21%. No wrong-ID hits in B among 738 evaluable pairs (201 background misses remain). Not a guarantee of correct centers: only ~31% within two voxels, and DTW can collapse nearby trajectories: 171/1001 and 17/546 adjacent pairs violate the 2-voxel gap; no strict crossings. **J fails its incremental performance gate vs I** despite eliminating all crossings and gap violations. I and J are identical on B; A J within2 gain +1.04 points versus I, wrong ID only -0.28 points. No promotion of J alone. The graph cut is exact for its stated cost, not necessarily anatomically optimal. Negative and positive outcomes preserved. P and I/J use an anchored local template; W uses full adjacent trace context, so improvement is about this algorithm/data cost comparison, not a universal DTW superiority claim.

## Open questions
- How can we combine DTW's local signal correspondence with globally consistent gap and slope constraints without losing sheet identity? It must be tested on fresh sections/cubes, not fitted on these scores.
- Is this feasible on 3D curved surfaces beyond a graph-over-depth section? Need independent sheet anchors and same-sheet verification.

## Next actions
- Adaptive lane: one frozen hybrid DTW + ordered cut on two fresh spatial cubes, preserving all 054 results. No unread inference.
- Claude/Gemini/Codex: reuse dtw_map and cut solver only with their exact frame/seed/coverage contract; do not treat absence of crossing as same-sheet proof. 053 external tensor splitter remains positive on oracle masks. No external submission.
