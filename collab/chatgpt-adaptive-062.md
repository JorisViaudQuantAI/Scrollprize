# Adaptive 062 — exact path-free DTW edge objective: mathematical PASS, efficacy NO-GO

2026-11-24. [Frozen protocol](../firstletters-colab/adaptive_surface_recovery/dtw_edge_cut/protocol.md) 648d7f2b9532a551df776e53da0d733daa985799, [solver](../firstletters-colab/adaptive_surface_recovery/dtw_edge_cut/edge_solver.py) 61463837b076f9a1ef3af7b5d05e6283dcb92eaf, [runner](../firstletters-colab/adaptive_surface_recovery/dtw_edge_cut/run.py) d0f0ef581ee43a7a359689bb834f5c7d34b9d417, [executed workflow](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45364502407), job 138342618182; [receipt](../firstletters-colab/adaptive_surface_recovery/dtw_edge_cut/result.json) e34fefe7bc7b854d6ea3162fd7569826fc5bc23d. Frozen before 061 outcome. Exact ordered-label XOR construction motivated by Ishikawa (doi:10.1109/TPAMI.2003.1233908) and general submodular ordered-label priors (doi:10.1016/j.dam.2009.02.026). New objective sums mismatch to each of 3968 directed local DTW maps per cube, no propagation tree. Retains exact unary, gap and slope. All synthetic gates pass; independent exhaustive energies 115/85 exactly reproduced, identity reduces exactly to TV, actual transposition energy 389562=389562 and zero decoded differences. Proof and measured parity, not an anatomical guarantee.

Four fresh disjoint Paris4 cubes, 4,302 evaluable pairs, only one cube with two seeded sheets (other three one each), no gap<16 labeled pair in multisheet cube. Therefore multi-sheet population gate fails. No replacement. All 10 arms/4 cases complete. Source 81,423,223 bytes, analysis wall 47.021s, peak RSS 408,440,832. 0.959–1.809% of map states clipped to allowed depth domain, all recorded.

| Arm | Equal-cube within2 | Correct ID | MAE vox |
| --- | ---: | ---: | ---: |
| W | .65851 | .85324 | 2.1963 |
| J | .74215 | .88226 | 2.2094 |
| K | **.84226** | **.99295** | **1.4300** |
| D | .81811 | .98648 | 1.5076 |
| G all-edge | .75136 | .93012 | 2.0990 |
| G0 no TV | .74899 | .92238 | 2.0734 |
| JM | .74496 | .89895 | 2.1255 |
| GM all-edge extra anchors | .75170 | .94170 | 2.0481 |

All arms wrong-ID share 0 in these fields (background misses remain). All cut arms satisfy hard constraints. G fails efficacy gates: on second cube, single-sheet within2 .07493 vs K .43460 and ID .74251 vs W .95504, despite exact objective. G has no >=3pp gain vs both J/K; GM all per-cube nonharm gates pass against JM but no >=3pp gain and missing multi-sheet population. No promotion. The mathematical formulation solves a real traversal-dependence defect but does not fix a poor unary/ambiguous local correspondence; enforcing a globally optimum incorrect data cost does not improve physical placement. No tuning of the coefficient or selecting a favorable arm. No evidence about tight-gap behavior because no eligible compressed pair in this sample. Keep 061 multi-anchor positive and compressed-stratum negative. No unread inference or CNN.

## Next actions
Keep exact solver available for future improved correspondence costs, not adopted in current production. Prefer fresh gap-stratified, multi-anchor reference test of 061, with a spacing-sensitive prior designed from *predicted* baseline geometry rather than held-out truth. No threshold fit on these fields. No external submission/main merge.
