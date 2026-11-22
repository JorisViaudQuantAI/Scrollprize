# Completed Parker published-letter audit

## Outcome

**From a source (the frozen execution receipts):** All 151 declared source members passed the original integrity/header checks. The audit evaluated 11,693 geometry-selected points: 4,678 supplied ink labels and 7,015 background labels. It used one already published lunate sigma from PHerc. Paris 2 Fragment 96. No unread-scroll pixels, candidate locations, training, selected-depth search, threshold tuning, image-based patch choice or organiser submission was involved.

**My inference:** These simple raw descriptors show limited, sometimes inverse, association with the supplied ink labels. The positive-minus-negative depth contrast has a modest positive association, but it does not outperform every misregistration control. This does not establish an ink detector, prove native registration, or imply that learned 3D models cannot detect ink. The tested material is known to carry ink; weak raw-descriptor association must not be reported as absence of ink.

## Source and prospective record

- **From a source:** [Parker et al. (2019)](https://doi.org/10.1371/journal.pone.0215775), [public data](https://doi.org/10.17605/OSF.IO/ZDKN4), and [S1 methods appendix](https://doi.org/10.1371/journal.pone.0215775.s001). The appendix describes photographic registration and manual binary labels, and approximately 1.3 micrometre voxels at 53 keV for this fragment. The data license is CC BY 4.0; original reference code is GPLv3. This runner was independently implemented.
- Scientific freeze: `bcd67e3005132b74e295964cea99f4a23d63608c`.
- First transport amendment: `410f233f624efbc3689bd3e33253ff662e7cc0bc`.
- Final transport amendment, published before source completion and any numerical result: `abc3298d0c6f7e73a2b21794488214dd106f8bc1`.
- Scientific protocol SHA256: `30eabd9cf4762e290840db1af5ebaf9f6ee07189212b2984c41608c3af902ecb`.
- Result JSON SHA256: `1948c244e6ebdcc6da2a189587acfdfbb5aea7ba1de92213e7943b548e786350`.

**From the preflight receipts:** The supplied normals are all (-1,0,0); this is an audit along that supplied source axis, not validation of true local normals. The source TIFFs are unsigned16 big-endian, 1440 rows by 286 columns. The PPM supplies integer source positions and a 1440 by 2530 surface. Out-of-volume PPM points were masked rather than clipped. Among 3,016,367 fully masked surface points, 1,057 centers were outside the source volume; 21,827 lacked the full frozen 33-depth support. These exclusions were set before CT intensity inspection.

## Fixed descriptors

**From a source (result JSON):** Every AUC keeps its preregistered orientation: larger descriptor values mean positive ink labels. Values below 0.5 have not been flipped. The center value is also depth zero in the full table, not an additional independent model. Means are stored-intensity units, except the source-x comparator which is in voxel coordinates.

| Descriptor | Signed AUC | Row-macro signed AUC | Ink mean | Background mean | Standardized mean contrast |
|---|---:|---:|---:|---:|---:|
| Center intensity | 0.463041 | 0.491502 | 30883.972 | 31160.914 | -0.115526 |
| Mean of all 33 depths | 0.408879 | 0.437888 | 33359.545 | 33744.669 | -0.263981 |
| Positive-axis minus negative-axis contrast | 0.575421 | 0.608964 | 1384.672 | 374.850 | 0.269030 |
| PPM source-x nuisance comparator | 0.516062 | 0.334387 | 120.187 | 120.301 | -0.004649 |

**My inference:** The overall and row-macro source-x values differ markedly (0.516 versus 0.334). Pooling points and averaging within-row associations answer different descriptive questions. This reinforces the need to keep geometric and spatial structure visible; neither pixels nor the 126 rows containing both label classes are independent specimens. No confidence interval or p-value is inferred from their count.

## All 33 frozen depths

**From a source (result JSON):** All native depths are listed below, in frozen order. Positive depth follows the supplied (-1,0,0) axis. The approximate physical offsets are voxel offsets multiplied by the source-reported 1.3 micrometre pitch; they are not a measurement of effective imaging resolution. No depth is chosen as a winner.

| Offset (voxels) | Approx. offset (µm) | Signed AUC | Row-macro AUC | Ink minus background mean | Standardized mean contrast |
|---:|---:|---:|---:|---:|---:|
| -16 | -20.8 | 0.533881 | 0.584558 | 638.084 | 0.117827 |
| -15 | -19.5 | 0.526322 | 0.571514 | 503.424 | 0.093604 |
| -14 | -18.2 | 0.511477 | 0.547363 | 242.259 | 0.045036 |
| -13 | -16.9 | 0.491611 | 0.514811 | -125.941 | -0.023006 |
| -12 | -15.6 | 0.469436 | 0.477869 | -550.266 | -0.097339 |
| -11 | -14.3 | 0.448847 | 0.446449 | -973.875 | -0.167533 |
| -10 | -13.0 | 0.432216 | 0.420157 | -1319.241 | -0.224544 |
| -9 | -11.7 | 0.418266 | 0.400537 | -1557.110 | -0.270210 |
| -8 | -10.4 | 0.407058 | 0.386667 | -1660.300 | -0.304165 |
| -7 | -9.1 | 0.399496 | 0.376058 | -1626.334 | -0.324166 |
| -6 | -7.8 | 0.397054 | 0.376450 | -1485.318 | -0.329399 |
| -5 | -6.5 | 0.402074 | 0.386303 | -1271.986 | -0.318475 |
| -4 | -5.2 | 0.412928 | 0.406703 | -1031.666 | -0.292690 |
| -3 | -3.9 | 0.426304 | 0.431044 | -793.072 | -0.253129 |
| -2 | -2.6 | 0.440389 | 0.453964 | -582.251 | -0.206290 |
| -1 | -1.3 | 0.452965 | 0.473419 | -408.009 | -0.157819 |
| +0 | +0.0 | 0.463041 | 0.491502 | -276.942 | -0.115526 |
| +1 | +1.3 | 0.471737 | 0.508557 | -187.508 | -0.084188 |
| +2 | +2.6 | 0.476885 | 0.519952 | -135.508 | -0.065081 |
| +3 | +3.9 | 0.479147 | 0.525099 | -106.839 | -0.054660 |
| +4 | +5.2 | 0.478519 | 0.527782 | -93.088 | -0.050448 |
| +5 | +6.5 | 0.477025 | 0.530116 | -83.432 | -0.047817 |
| +6 | +7.8 | 0.476125 | 0.530865 | -73.651 | -0.044360 |
| +7 | +9.1 | 0.476494 | 0.531980 | -57.425 | -0.035917 |
| +8 | +10.4 | 0.476313 | 0.531865 | -42.907 | -0.027693 |
| +9 | +11.7 | 0.477874 | 0.534560 | -21.600 | -0.014175 |
| +10 | +13.0 | 0.481424 | 0.538978 | -0.741 | -0.000494 |
| +11 | +14.3 | 0.486005 | 0.542736 | 25.125 | 0.016762 |
| +12 | +15.6 | 0.489543 | 0.545633 | 41.233 | 0.027815 |
| +13 | +16.9 | 0.492929 | 0.549794 | 58.751 | 0.039844 |
| +14 | +18.2 | 0.495198 | 0.552230 | 72.781 | 0.049595 |
| +15 | +19.5 | 0.497253 | 0.551794 | 84.604 | 0.058024 |
| +16 | +20.8 | 0.497574 | 0.547046 | 89.645 | 0.061756 |

## Every fixed label-shift control for the fixed descriptors

**From a source (result JSON):** Each cell is **native AUC → shifted-label AUC on identical common support** for that shift. Different rows can have different support; compare the two entries within a cell. There is no wraparound or zero padding. Full per-depth results for every shift, class counts, distributions and native same-support comparators are retained in `parker_profile_results.json`; no declared depth or shift was dropped.

| Shift (surface pixels) | Common points | Center intensity | Profile mean | Axis contrast | Source-x comparator |
|---|---:|---:|---:|---:|---:|
| x -256 | 9281 | 0.458494 → 0.477975 | 0.396365 → 0.457117 | 0.592854 → 0.475236 | 0.604353 → 0.598477 |
| x -64 | 11088 | 0.468064 → 0.471114 | 0.412500 → 0.413679 | 0.577667 → 0.562949 | 0.543486 → 0.531674 |
| x -16 | 11537 | 0.463932 → 0.464396 | 0.411617 → 0.413831 | 0.574737 → 0.572073 | 0.523218 → 0.518713 |
| x +16 | 11537 | 0.464399 → 0.467249 | 0.409875 → 0.406730 | 0.574444 → 0.576497 | 0.511944 → 0.517227 |
| x +64 | 11088 | 0.465788 → 0.477116 | 0.413979 → 0.416577 | 0.571520 → 0.571715 | 0.498337 → 0.522585 |
| x +256 | 9281 | 0.478464 → 0.457128 | 0.429736 → 0.428894 | 0.556634 → 0.536478 | 0.486048 → 0.530458 |
| y -256 | 10253 | 0.494123 → 0.499972 | 0.436621 → 0.487614 | 0.579821 → 0.541464 | 0.496859 → 0.473380 |
| y -64 | 11324 | 0.471886 → 0.481045 | 0.413751 → 0.423192 | 0.574769 → 0.566265 | 0.510629 → 0.515384 |
| y -16 | 11595 | 0.464366 → 0.467092 | 0.408797 → 0.408970 | 0.574071 → 0.575334 | 0.515505 → 0.515209 |
| y +16 | 11595 | 0.464848 → 0.465827 | 0.409104 → 0.410169 | 0.575901 → 0.572022 | 0.516945 → 0.517588 |
| y +64 | 11324 | 0.463796 → 0.469619 | 0.410347 → 0.411238 | 0.572841 → 0.568830 | 0.521359 → 0.526466 |
| y +256 | 10253 | 0.463612 → 0.493044 | 0.416020 → 0.443983 | 0.560163 → 0.579923 | 0.567023 → 0.558517 |

**From the recorded comparisons:** The axis-contrast descriptor is higher on native labels in 8 of 12 paired comparisons, and higher on shifted labels in 4 of 12. In particular, y+256 gives native 0.560163 versus shifted 0.579923 on 10,253 shared points.

**My inference:** These translations are stress tests, not exchangeable randomizations or calibrated null samples. Their differences are not p-values and cannot bound a First Letters false-positive rate. They do not justify choosing a better alignment, sign, depth or threshold after seeing this case. No such follow-up was run.

## Resource use and acquisition history

**From the execution receipts:**

- Successful final tranche: 3.641306 process CPU seconds and 37.170116 wall seconds. This is the final run only, not the complete elapsed time of the earlier failed network attempts.
- Final tranche obtained 16 new source members / 11,105,233 response-body bytes, with 0 failures and one ordinary redirect refresh.
- Successful HTTP range bodies across attempts001–003: 107,239,412 bytes. This counter excludes preflight ranges; those are separately included in conservative cumulative reservations.
- Cumulative reserved upper bound: 117,007,583 source bytes plus 13,000,000 metadata bytes. These are budget reservations including failed requests, not all received bytes. The original 300 MB source and 350 MB total caps were retained.
- The locally acquired slice members passed full uncompressed-member CRC32, source header checks, and local SHA256 recording. The entire 1.8 GB archive was not downloaded or locally SHA256-verified; its advertised whole-archive hash remains an upstream claim.
- Attempt001 stopped at 53 members after HTTP403 while reusing an ordinary redirect whose observed lifetime was 60 seconds; expiry is an inference, not a recorded server rejection reason. Attempt002 stopped at 135 members on three 30-second timeouts while routing requests through the canonical URL. Both failures remain preserved. The final tranche used normal redirects cached for less than 20 seconds with 45-second requests and a bounded retry allowance. No signed URL or credential was changed or logged.
- Runtime: Python 3.12.14, NumPy 2.3.5, Pillow 13.0.0; one numerical thread. Decoded CT dtype is `>u2`; extracted profile dtype is `<u2`.

## Files and complete output

All complete numerical outcomes are in `parker_profile_results.json`; it includes all native depths, fixed descriptors and every native/shifted same-support comparison. `parker_attempt003_download_receipts.json` contains every final acquisition/reuse integrity record. `parker_profile_start_003.json` and `parker_final_transport_started.json` identify the successful frozen execution. The raw profiles remain outside the repository. The old `failure.json` still describes attempt002; use the explicitly named historical failure receipts and the completed result to avoid mistaking a preserved old failure for a current one.

## Open questions

- **My inference:** A second independently photographed fragment is required before any transfer or generalization claim. This one published glyph cannot supply that evidence.
- **My inference:** The present outputs do not identify ink chemistry, verify the supplied fixed normal against local geometry, establish registration accuracy, or test learned 3D morphology models. Those remain separate questions.

## Next actions

- **Codex:** reuse the demonstrated input guard: check coordinate bounds and source axes/endianness, count invalid PPM points, and never clip them into a valid voxel.
- **Claude Code:** treat this as a frozen public control. Any learned-model evaluation needs its own training-support audit and prospective protocol; do not tune to these outputs and call the same letter held out.
- **Owner:** decide whether the reusable public donor loader, independent-photo control and integrity receipts form a useful Progress Prize contribution. This audit alone is not a First Letters discovery or evidence of a prize entitlement.
