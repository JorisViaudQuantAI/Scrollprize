# ChatGPT 035 — Q43 native CT bright-deposit injection: NO-GO

Date: 2026-11-23. Branch: `chatgpt/adaptive-surface-recovery`. No candidate, no unread CT, no inference/GPU.

[Protocol](../firstletters-colab/adaptive_surface_recovery/highz_injection/protocol.md) committed at 7b7afd01e773f5c75ee7662a63a10d78e9bcdc0a; [Python implementation](../firstletters-colab/adaptive_surface_recovery/highz_injection/run.py) at cc85899ca49fa8fdca6a801fcc34e98f80fd893e; [one-shot CPU workflow](../.github/workflows/chatgpt-q43-public-ct.yml) at a7b79223321a356891c371926c8db3f84dbb4751. [Completed GitHub Actions run](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45263065022), job 138002653883, success including dependency installation, self-tests, hash-verified data acquisition, evaluation and artifact upload. [Complete machine-readable receipt](../firstletters-colab/adaptive_surface_recovery/highz_injection/result.json) committed at 4fefc997f3f638e0a07c7cb71b0b4983f39c7dd3, recovered from the completed job log.

This is an executed synthetic-deposit test on genuine public native PHerc0841 W00 CT (28×768×768 uint8, array SHA256 fb536e0be0f5e0d382d05584bd070b560031b1c880a46412439cb2e953026b0d), not a real chemical deposit or an independent scroll. Twelve calibration and twelve separate validation cells; 24 injected test cases. Fixed 99th-percentile thresholds calibrated on original cells. Digitally added 10% and 30% of positive column mass in a straight stroke over central three planes, subject to rounding and saturation. Non-injected background is not verified blank.

Median results:

| Score | AUC, 10% | Sensitivity, 10% | AUC, 30% | Sensitivity, 30% | Original flagged |
|---|---:|---:|---:|---:|---:|
| Raw max | 0.52385 | 0.318% | 0.66015 | 7.245% | 0.906% |
| Local max | 0.52382 | 0.318% | 0.64934 | 8.709% | 0.993% |
| Local positive column mass | 0.46608 | 0.284% | 0.50026 | 1.997% | 0.385% |

Median realized fraction of requested digital dose 75.11% (nominal 10%) and 82.77% (nominal 30%). At 30% only 8/12 cells reached 80% of the requested dose (gate: 10/12). The column-mass gate also fails 50% sensitivity and +10 pp advantage over local max; original and off-stroke flagged fractions pass the 2% limits. **Do not promote this method or launch unread-scroll bright-deposit screening from it.** The result does not rule out actual high-Z material: input quantization, dose clipping, uncertain sheet/gap alignment and a synthetic nonphysical signal constrain the conclusion. Earlier Q42 spectral disagreement is not resolved.

Reproduce, outside repository scratch for CT:
```bash
python -m pip install numpy==1.26.4 scipy==1.12.0
python firstletters-colab/adaptive_surface_recovery/highz_injection/run.py --self-test
python firstletters-colab/depth_recentering/fetch_native.py --output /tmp/q43-public-ct
python firstletters-colab/adaptive_surface_recovery/highz_injection/run.py --data /tmp/q43-public-ct/pherc0841_native.npy --output /tmp/q43-receipt.json
```

## Open questions
- A new real material control with independently verified high-Z content is required before judging lead detection. Neither the low-contrast carbon controls nor digital injections establish that.
- Do not reinterpret the 1% calibration threshold as an actual false-positive guarantee on unknown scrolls.

## Next actions
- Close this Q43 digital injection version, no post-hoc tuning. For future work prioritize genuine fresh controls or a consequential measured failure in current Q1/Q7/geometry lanes over more archive sweeps.
- The immediate workflow completed; no schedule or recurring automation was created. No other branch or agent job changed.
