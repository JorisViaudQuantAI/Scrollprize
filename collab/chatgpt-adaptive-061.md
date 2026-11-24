# Adaptive 061 — sparse multi-anchor and symmetric-path hybrid: frozen development gate PASS

2026-11-24. [Protocol](../firstletters-colab/adaptive_surface_recovery/multi_anchor_consensus/protocol.md) frozen a739c088707331ac341ba865fe2c71a1463cee67; [anchor solver](../firstletters-colab/adaptive_surface_recovery/multi_anchor_consensus/anchored_cut.py) 4029b610446366bd958535c8dd620862d0add228; [runner](../firstletters-colab/adaptive_surface_recovery/multi_anchor_consensus/run.py) 5a130d2793f1b7a37cdc92cdd0593e54a463491d; [executed workflow](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45364132693), job 138341365455; [full numerical receipt](../firstletters-colab/adaptive_surface_recovery/multi_anchor_consensus/result.json) 8cf707568c186356be2de554eb4a9980dc3266ad. Reviewed Claude 024 and Gemini 007; no duplicate T17/RGT. Four fresh nonoverlapping manual Paris4 cubes, 81,827,509 source bytes, 116.056s analysis, peak RSS 456,335,360. All five synthetic checks pass: original single-anchor solver exact, two-anchor brute-force energy 66, contradictory anchor rejected, path-order permutation exact, cached DTW identical to original. 8,689 held-out sheet/ray pairs, 11 central sheet seeds, 38 extra exact observations on four fixed rays/cube. All four multi-sheet and all seeded IDs receive extra anchors. Explicitly extra manual inputs, not automatic discovery.

| Arm | Equal-cube mean within2 | Correct ID | Wrong ID | MAE vox |
| --- | ---: | ---: | ---: | ---: |
| W central DTW | .56074 | .77432 | .01764 | 2.8130 |
| J central ordered cut | .71693 | .77390 | .01445 | 2.6349 |
| K single-path fixed prior | .77916 | .84378 | .01010 | 1.8413 |
| H reciprocal prior | .78441 | .84212 | .00691 | 1.8530 |
| D central two-path equal prior | .80779 | .87231 | .00475 | 1.5340 |
| A D + extra exact anchors | .80850 | .87320 | .00475 | 1.5303 |
| JM ordered cut + extra anchors | .72911 | .79269 | .01445 | 2.4949 |
| M both paths from all admitted anchors | **.82725** | **.88172** | **.00406** | **1.4427** |

All primary gates pass: M vs K within2 +4.81 pp, vs same-anchor JM +9.81 pp; vs W +26.65 pp, correct ID +10.74 pp, MAE -48.71%, 33 gap and 854 slope violations reduced to zero. D alone (one anchor, symmetric two path costs) improves K by 2.86 pp. Extra hard anchors alone A add only .072 pp vs D, while incorporating their waveform trajectories M adds 1.87 pp vs A. Cube B JM remains slightly better than M in near2 (.82428 vs .81507) and identity (.93320 vs .89602). M not universally superior. No replacement of prior 056–060 failed gates.

**Compression remains unresolved:** pooled descriptive strata (actual true seeded midpoint gaps, not observable deployment criteria). For 628 pairs with gap <8 vox, W/JM/M within2 .35510/.85032/.72134; correct ID .38057/.85669/.72293; wrong ID .20064/0/.04140. M beats W but loses to simple same-budget JM, especially B. For 5662 pairs gap>=16, W/JM/M within2 .61233/.76704/.86630. No threshold deployment from these data. Additional anchors identify 16 of 18 cube-local manual IDs (>=64 vox) vs 11 central, but only the 11 central IDs are reconstructed. No claim of automatic coverage expansion.

Uncertainty gate passes 3/4 cubes: at 80% coverage M error improves while wrong-ID does not rise. Pooled (descriptive) MAE 1.42985→1.15381, ID .87064→.90432, within2 .82587→.88029; wrong-ID .005294→.005755 slightly increases. Thus spread is a useful error-ranking signal, **not a validated identity-confidence or deployment abstention rule**. Do not report pooled gate success as calibrated FPR. Stable tie ranking and common eligible label denominator preserved.

## Next actions
062 independently frozen before these scores: exact all-adjacent-edge DTW mismatch penalties, no propagation tree in the objective; four new cubes. Preserve 061 result and compressed-stratum limitation. Claude/Gemini/Codex can reuse multi-anchor solver, but only with genuine verified anchors; no unattended ink screening, main merge or submission.
