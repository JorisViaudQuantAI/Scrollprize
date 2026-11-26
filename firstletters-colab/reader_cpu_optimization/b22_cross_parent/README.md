# B22–B25: distinct-parent test, cascade limits and local retention

2026-11-26. Stacked on PR #18. Public already-awarded controls only. No new unread candidate, production deployment, GPU use or main merge.

## B22: native 0841 transfer experiment

The complete B21 prediction archive was acquired directly from [Actions artifact 15137042105](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45858826467/artifacts/15137042105); all three file SHA256 identities were verified locally. This removes the need to upload large checkpoints into the chat. Pooled 1447 thresholds at >=80% source ink recall frozen in calibration.json before native 0841 inference. The panel comprises three disjoint 384 contexts (two mixed, one blank) selected solely from existing annotation labels, central192 scored. Both normal orientation hypotheses, three native 17-plane bands (center and ±2) and FP32/BF16, exact native reconstruction check against author prepared16; fail closed if any full-context geometry, stencil or identity check fails. Native CT body budget 512 MiB (post-response stop threshold, not exact transport limit).

[Run 45860224825](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45860224825): dependency install and source/checkpoint acquisition succeeded; as of this report the native rendering/inference step is in progress. No 0841 outcome or completed distinct-parent transfer claimed. The run is limited to 15 minutes and preserves available receipts even on failure. Author donor labels and Reader development exposure are not untouched independent validation; normal order must not be selected using held-out scores.

## B23: actual archived pixelwise cascade replay

Source archive matches B21 published SHA256. All 340 scalar metrics reproduced exactly, 81 exhaustive majority checks passed. 12 crops, 442368 central192 pixels including unknown. For a fixed median threshold, after two depth bands, the third can be omitted **only** if the two threshold maps agree on every scored pixel. This preserves binary decisions, not continuous scores. Whole-crop model cost, not hypothetical per-pixel CNN savings.

| Setting | Method | Extra crops | Estimated seconds | Reference seconds | Changed decisions |
|---|---|---:|---:|---:|---:|
| fixed .5 | center, −2, conditional +2 | 8/12 | 28.224 | 31.736 | 0 |
| other-winding threshold | same | 12/12 | 31.736 | 31.736 | 0 |
| fixed .5 | BF16, FP32 if any pixel within .001 | 8/12 | 12.815 | 10.635 | 0 |
| other-winding threshold | BF16, same fallback | 12/12 | 16.359 | 10.635 | 0 |
| other-winding threshold | BF16 only | 0/12 | 5.725 | 10.635 | 4313 |

BF16-only at the transferred threshold loses 441 annotated ink detections, adds none, and changes 2355 known-label decisions. At .5 the corresponding counts are 36 lost and 29 added. The .001 empirical margin is not a certified numerical bound. The whole-crop fallback is slower than FP32 at both regimes in this panel; no deployment. Timings are estimates using measured B21 arm medians, not measured cascade wall times.

## B24: component/crop retention

All 12 crops, 8 contain positives. At transferred thresholds, no positive crop loses >5pp recall, no >=50%-detected component loses all detection; 10->9 components retain >=10% detection, 9->9 >=50%, 7->7 >=80%. At fixed .5, >=50% changes 2->1. Only ten connected annotation components exist (nine >=1024 pixels, one 720), all touch crop boundary or unknown labels. These are coarse connected annotation regions, not individual strokes or letters. This benchmark cannot certify preservation of fine strokes. Two synthetic checks pass.

## B25: learned depth-score combinations

54 actual CPU logistic fits, three synthetic checks. Fixed feature sets and leave-source-crop-out thresholds followed by full-source refit. Both fail the predeclared FP/recall transfer gate. At 80% OOF source recall, full refit recall shifts to 96–100%; calibration does not transfer. W058/W060 share a parent.

| Model | Held W058 recall / FPR | Held W060 recall / FPR |
|---|---:|---:|
| Center | 65.25 / 35.06 | 98.94 / 74.01 |
| Median3 | 63.48 / 32.04 | 98.38 / 71.82 |
| Depth features D | 78.09 / 62.18 | 100 / 100 |
| Depth + opposite direction DO | 68.12 / 41.11 | 100 / 89.96 |

No learned combination promoted. The pooled models and exact coefficients frozen in learned_receipt.json for diagnostic transfer to 0841, not selected after held-out inspection.

## Reproduction

Download and unpack the B21 public artifact (predictions.npz, results.json, selection.json). Run cascade.py, components.py, and learned.py with input directory and output JSON arguments. Local code Git blobs exactly match executed files: cascade dd21c14ff6ee5444ad43b2c5b5874431ecce282d, components 0fc4c5c0d7749e6198103f40deccf989802ec31b, learned 46639a3b3a9de5053635506046e22f9cdea3fdf6. Full local receipts retained; aggregate data and models committed here. B22 workflow and all source hashes are in the branch. Do not treat ongoing native runner as a successful experiment until its outcome/receipt is verified.

Next decision: preserve exact FP32 preprocessing and three-depth median as bounded confirmation method pending distinct-parent/geometry verification. Reject whole-crop BF16->FP32 uncertainty fallback for speed on this high-recall panel; reject learned D/DO variants under the current calibration policy.