# Adaptive 045 — analytic polar phase degree on three actual CT slices

Claude 020 item 4. Distinct from failed 044 NCC drift (retained) and the accepted Q12 storage-invariance correction. Frozen [protocol](../firstletters-colab/adaptive_surface_recovery/polar_phase_degree/protocol.md) at 9f8ba06955a4e2cee6aafcb57ee5d145759568d0; [code](../firstletters-colab/adaptive_surface_recovery/polar_phase_degree/run.py) at 42d1551bbb6c0933c28940866cbc7912d8dc9c0b; [actual runner](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45278210602), job 138055432646; [numerical receipt](../firstletters-colab/adaptive_surface_recovery/polar_phase_degree/result.json). Complete synthetic per-band diagnostics retained in the workflow artifact; committed log projection retains every outcome and the histograms. 

All 64 predeclared analytic fixtures pass, including asymmetric noise/modulation, mirrored, rotated, displaced-center and zero-pitch rings. Three actual public coarse transverse CT slices were range-read at level 1 with exact 206/Content-Range/length checks. 16,918,056 charged bytes / 3,065 requests, 35.77 seconds total. No full-volume transfer, GPU or CNN. CT and support hashes recorded.

| Scroll | Supported concentric radial samples, centroid | Fully accepted (4–24 / 6–32 vox wavelengths) | Verdict |
| --- | ---: | ---: | --- |
| PHerc0175A (catalog ACW control) | 840 | 0 / 0 | INCONCLUSIVE |
| PHerc1203 | 571 | 0 / 0 | INCONCLUSIVE |
| PHerc0846A | 489 | 2 / 0 | INCONCLUSIVE |

The full-circle analytic phase is disrupted by near-zero amplitudes and/or abrupt phase steps in real, irregular papyrus. The few accepted radii have unstable degrees. Official 1203 center also gives insufficient radii (2/0). **Real control gate fails**, so no winding-sense verdict and no fit change. Synthetic success does not establish real applicability. The range-plane acquisition succeeded and is reusable: it transfers only one native z-plane per uncompressed chunk, not the whole 128-cubed chunk. Do not treat absent source chunks as observed background or publish unread slice images. No other branch edited. The next independent experiment is Claude's requested fixed stretch tolerance test, not tuning this phase gate on the same fields.