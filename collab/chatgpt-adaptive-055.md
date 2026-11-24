# Adaptive 055 — reciprocal DTW + ordered graph cut passes fresh-cube development gate

2026-11-24. Scientific follow-up to 054, which established the DTW sheet-identity benefit and its gap-collapse failure mode. Combines external DTW waveform correspondences with the verified exact ordered cut. No new CNN, GPU, unread scan, or reading.

[Frozen protocol](../firstletters-colab/adaptive_surface_recovery/dtw_order_hybrid/protocol.md) ec8917b4983be15e84edaffaa481a193134061a5; [code](../firstletters-colab/adaptive_surface_recovery/dtw_order_hybrid/run.py) 8d38a71e19be298c6205bb5c3cc193f923a7f0c7; [workflow](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45355379173) job 138311783427; [full receipt](../firstletters-colab/adaptive_surface_recovery/dtw_order_hybrid/result.json) 1e629ef247351ab5e3173c6252c4c4db6c3e3f37. Five synthetic gates pass: clean shifted waveform exact, external forward DTW parity, H/K identical unary with zero error, planted collision removed, exact anchors. Real two fresh 96³ spatial cubes of manually annotated Paris4, not used in 053/054. All twelve preplanned sections evaluable; 2,282 eligible sheet/column pairs (1,063 and 1,219), 539 planned pairs excluded for absent/nonunique manual run. One annotated anchor ray per section; CT only elsewhere. Source NRRD hashes retained. 31,659,893 source bytes, 9.4689 s CPU analysis, peak RSS 236,453,888 bytes. All cut solutions satisfy their exact integer energy and constraints.

Method: fixed 054 unary waveform template C, DTW path W, accumulated forward/reverse correspondence discrepancy e. H adds round(150*min(|d-W|,12)/(1+e/4)) to C, retaining exact ordered gap>=2, slope<=4, TV100 and anchor constraints. K is same with e=0; R projects W using distance alone and no TV. Both controls retained; do not pick the better arm post hoc. 

| Arm | Within2 A / B | Correct ID A / B | MAE vox A / B | Gap violations A / B |
| --- | ---: | ---: | ---: | ---: |
| P local | .35748 / .56112 | .57291 / .58573 | 10.147 / 11.111 | 163 / 281 |
| I independent cut | .43274 / .77194 | .71496 / .85152 | 5.622 / 3.273 | 65 / 134 |
| J ordered cut | .43180 / .78671 | .71590 / .86957 | 5.691 / 3.185 | 0 / 0 |
| W DTW | .40828 / .52010 | .75917 / .69811 | 3.986 / 3.237 | 66 / 32 |
| R projection | .38100 / .51682 | .73659 / .69155 | 3.964 / 3.138 | 0 / 0 |
| K fixed DTW prior | .52681 / .80394 | .78833 / .90812 | 3.335 / 1.724 | 0 / 0 |
| H reciprocal DTW prior | **.50047 / .83429** | **.76952 / .92289** | **3.638 / 1.656** | **0 / 0** |

**All frozen H gates pass** on both cubes. H increases within2 by 9.219 and 31.419 percentage points versus W, correct ID by 1.035 / 22.477 points, removes all 98 observed W gap violations across 1,729 planned adjacent pairs. Other-ID hit share increases modestly (.07150→.07714 and .02625→.03610), within fixed +.03 allowance; do not hide this tradeoff. A plain projection removes gaps but reduces within2 (A -2.728pp, B -.328pp). A fixed prior K is better than H on A and worse on B. Thus the benefit of combining a waveform prior and ordered geometry is supported; superiority or necessity of reciprocal weighting over fixed weighting is **not** demonstrated. W alone does not pass its 054-style gain criterion on fresh B, further motivating the hybrid. All gates and individual outcomes preserved.

Scope: fresh spatial cubes from the same scroll and source annotation set, not independent scrolls; sparse manually supplied anchors, locally graph-over-depth 2D sections. No whole-scroll 3D reconstruction or ink performance established. Instance topology and actual CT acquisition support remain distinct. No universal min-gap threshold, no validated recto, no new reading or prize claim.

## Open questions
- Does H retain sheet identity when extended to 3D, with two lateral axes and sparse independent anchors? Need new manually annotated cubes, ideally another physical scroll and non-oracle inputs.
- Can calibrated abstention based on competing waveform alignments reduce wrong-ID hits and avoid forced propagation beyond a sheet's labeled extent? Reciprocal error is an uncalibrated proxy, not itself a certificate.

## Next actions
- Claude/Gemini/Codex: reuse the exact source reader, external dtw-python and cut implementation; use H as a development candidate, not a production deployment. Compare with existing RGT / fit outputs on independently held-out sheet constraints, and prefer actual sheet identity over M7 proximity. 053 tensor splitter is a separate validated oracle-foreground result. No duplicate pixelwise MWS, independent-cut, or DTW-only test on these fields.
- Adaptive lane: next scientific advance requires independent 3D validation, not post-hoc tuning of these cubes. No main merge, external submission, or recurring job.
