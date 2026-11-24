# Adaptive 063 — predicted-gap weighting fails transfer; gap-estimation bottleneck exposed

2026-11-24. [Frozen protocol](../firstletters-colab/adaptive_surface_recovery/gap_aware_transfer/protocol.md) 5191d02cd0013060b85cac1647c7fb3829468960; [runner](../firstletters-colab/adaptive_surface_recovery/gap_aware_transfer/run.py) 6d65d3358a96e200d947708ebdf4636b48358ae9; [actual workflow](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45365104064), job 138344676014; [full receipt](../firstletters-colab/adaptive_surface_recovery/gap_aware_transfer/result.json) e04f644e8b14efc94c19cfbcd5a5b01408db1385. Five synthetic gates pass, including 061 exact anchor tests, separate planes, attenuation endpoints and lateral transposition. All six selected actual ROIs assessed: 3 fresh (4,914 pairs) and 3 repeated 060 fields (5,290 pairs), kept separate. 52,435,792 source CT bytes/31 charged requests, 151.535s wall, 357,875,712-byte peak RSS. Exact published reference mesh/CT source metadata hashes and all 126 geometry candidate receipts recorded. No ink map, model or unread source.

Fresh population was preselected by geometry only: 0211 one wide field, no admissible tight field; 0343 one tight and one wide. The missing fourth field is retained as missing, not substituted. Each field has two locally disconnected original-incidence components, common reference support 80.20–100%, and at least two extra observations per component. Not independently labeled global winding identities. Same exact anchors for JM, M and S; no ground-truth distances fed to the optimizer.

| Fresh equal-ROI macro | Within2 | MAE vox | Wrong reference |
| --- | ---: | ---: | ---: |
| W tree DTW | .25029 | 6.44967 | .02614 |
| J ordered cut | .24879 | 7.55001 | .02362 |
| K single-path prior | .36297 | 5.62156 | .02270 |
| D symmetric two-path prior | .35144 | 5.29944 | .01457 |
| JM same-anchor ordered cut | .29866 | 5.23236 | .02549 |
| M 061 multi-anchor consensus | .38456 | 4.24884 | .01593 |
| S predicted-gap-weighted M | .38515 | 4.35296 | .02668 |

M descriptively improves on the matched-anchor JM in these three fields, not a new validated promotion. S provides essentially no near2 gain (+.058 pp), increases mean error and wrong-reference hits. On the fresh compressed 0343 field, overall wrong-reference share rises .04779→.08005 (fails +.03 bound); within2 .22162→.22282. Across the 550 fresh true gap<8 component/ray pairs, M/S within2 .34364/.34727 (+.364 pp, not required +3 pp), JM .31091. W .41818 remains better geometrically on this stratum; no switching by held-out truth. In repeated stress fields S macro within2 .47780 vs M .47020, but true gap<8 within2 .07796 vs M .08065. Do not combine repeated and fresh results.

**Critical diagnostic: predicted spacing is not a reliable compression detector.** On the fresh tight field, true gap<8 occurs at 275 rays, JM predicts 46 close rays, only 6 are true positives: precision 13.04%, recall 2.18%, 269 false-safe rays, gap MAE 5.453 vox. On the repeated tight field, 186 true close rays, 198 predicted close, zero overlap. Thus the proposed gate cannot be trusted when based on the reconstructed baseline gap. This is a measured failure of this estimator, not a universal statement about local CT evidence. No parameter search or retrospective correction. All S slope/gap and exact-anchor checks pass, but efficacy/population gates FAIL; no promotion. 

## Scientific handoff
061 remains a positive development result on independently selected spatial manual cubes, with tight-gap limitation; 062 exact path-free objective remains rejected for efficacy. 063 identifies a specific circularity: using an inaccurate baseline geometry to infer where to distrust waveform correspondences fails in precisely the compressed region. Next independent discriminator needs local CT evidence of unresolved/multiple sheets, or external verified anchors with uncertainty, evaluated against held-out instance IDs or separately projected public mesh reference. A 3D method cannot certify anatomical identity via its own inferred spacing. Retain all 056–063 failed gates. No unread deployment, prize claim, external submission or main merge.
