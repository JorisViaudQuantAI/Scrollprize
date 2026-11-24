# Adaptive 060 — measured two-component recovery on other manuscripts; global gate FAIL

2026-11-24. 059 stopped before CT on literal floating equality. [Numeric amendment](../firstletters-colab/adaptive_surface_recovery/dtw_component_transfer/numeric_retry.md) b07fe82236da9e832dc27b18a17a6d371914c7b1, [wrapper](../firstletters-colab/adaptive_surface_recovery/dtw_component_transfer/numeric_retry.py) 47b502d2d0d09151b0a25688b10b362fcbf2a26c, [executed run](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45362919538), job 138337211526, [complete receipt](../firstletters-colab/adaptive_surface_recovery/dtw_component_transfer/retry_result.json) f81b3b01ce97e782670b55d646d8fb3d90f0407c. Maximum synthetic planar interpolation residual is 7.105427357601002e-15 voxel, below the independently frozen 1e-10 tolerance. All five synthetic gates pass; the original strict failure is preserved. Real protocol, code, mesh/reference selection, solver, thresholds and caps unchanged from 059. Exact 058 mesh and CT metadata hashes verified.

All three previously projection-ambiguous fixed ROIs become assessable when separating the two original-incidence local components. Two central integer depth anchors only, CT elsewhere. 5890 scored reference-component/ray pairs (1998 + 1998 + 1894); common coverage 100%, 100%, 94.795%. This is a real cross-manuscript comparison of multiple local surface crossings, not independent globally annotated sheet identity or a new reading. 23,075,664 CT bytes, 17 charged requests, 44.976 s analysis wall, 344,825,856-byte peak RSS. No CNN, GPU, unread inference.

| ROI | Min reference gap vox | W within2 | J within2 | H within2 | W / J / H MAE vox | H gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0211 q=.5 | 17.847 | .46146 | .49650 | .68218 | 3.418 / 4.880 / 2.206 | PASS |
| 0343 q=.2 | 3.647 | .25225 | .30330 | .23423 | 5.363 / 4.500 / 5.096 | FAIL vs J within2 and wrong-reference hits |
| 0343 q=.5 | 23.143 | .34213 | .29831 | .42767 | 4.098 / 6.471 / 2.968 | PASS |
| Equal-ROI descriptive macro | — | .35195 | .36604 | .44803 | 4.293 / 5.283 / 3.423 | NOT a pooled promotion gate |

H improves equal-ROI within2 vs W by 9.608 pp and vs J by 8.199 pp; mean MAE decreases 20.26% vs W and 35.21% vs J. H wrong-reference hits macro .02102 vs W .03837 / J .01001. Zero gap and both-axis slope violations for H in all 3 ROIs; W has 308 gap violations and 923 slope violations in total. Both assessable wider-gap fields pass; on the tight-gap field, J wrong-reference hits .03003 vs H .06306 exceeds +.03 allowance and H within2 is 6.907 pp lower than J. **All population gates pass, global efficacy gate fails.** K fixed-weight macro within2 .44223 and MAE 3.674, no universal superiority of reciprocity. Hrow .43418, no independent 3D coupling claim. Changing DTW propagation order changes 72.66–78.32% of predictions. This is a signal of fragility.

## Scientific conclusion
Distinct local mesh-incidence components resolve the ambiguity that prevented evaluation in 058; this is a geometry reference change, not an improvement in input CT or a new anatomical segmentation. The same 057/055 hybrid improves both widely separated reference crossings but cannot be trusted in the compressed field. The observed minimum gaps suggest a mechanism worth investigating; three ROIs do not establish a spacing threshold. No tuning or post-hoc switching between H and J. A fresh preregistered gap-stratified evaluation with independent material/instance labels and sparse multi-anchor or uncertainty-aware constraints is needed before deployment. Preserve all 056–060 negative decisions. 

## Open questions / coordination
Claude/Gemini/Codex can reuse the exact two-component evaluator and 057 cut with pinned source hashes; do not conflate local disconnected geometry with true global winding IDs. This experiment does not certify a new unread surface. No external submission or main merge.
