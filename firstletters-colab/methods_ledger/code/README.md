# code/ — scripts for checking our arithmetic

These 15 files are sanitised copies of private lane scripts.
They are provided for **review**, not as a packaged tool: no support, no stable interface, no tests beyond a smoke run.
Each file starts with a header that says what was changed for publication.
In every case local paths, job names and every location of unread material were removed; hard-coded crops became arguments.
The numerical part is unchanged unless the header says otherwise.

Dependencies: numpy, scipy, opencv-python; some need tifffile, zarr, fsspec, scikit-image, Pillow or xraylib.
All inputs are public data (the organisers' open-data bucket, public meshes and model outputs) or your own maps.

| script | what it computes | ledger entry |
|---|---|---|
| `onsheet_gate_m7.py` | pixel classes against the surface prediction, 1 mm block rule, connected core, the window gate | S1 |
| `spiral_oncheck.py` | per-winding on-sheet fractions of a spiral fit against a random-shift control; CT z-score; normal sign | S3, S4, S7 |
| `sense_chain.py` | winding sense from chains of tracks, plain or mirrored | S5 |
| `cmp_tracks.py` | comparison of our track store with a published one | S6 |
| `isometric_resample.py` | resampling of a fitted winding to equal 3-D spacing with common row and column maps | S8 |
| `depth_recentre_test.py` | sheet-offset estimator, re-centring, and the injected-displacement test it failed | S9 |
| `letterbox_auc.py` | letter-box AUC and hot shares of one map on a public positive | C2, C3 |
| `row_criterion.py` | the dot-row criterion (version 2) and its event count per order | F1 |
| `strokeness.py` | strokeness and skeleton length of the marks in a row band | F1, F6 |
| `row_pair_criterion.py` | the row-pair criterion fixed for the second row-like place | F2 |
| `ghostmatch_geom.py` | correspondence between a strip and the neighbouring turn of the same segment | F5 |
| `ghostmatch_compare.py` | cross-correlation of the warped maps against 19 shifted or mirrored nulls | F5 |
| `linefold_stat.py` | the frozen line-pitch folding statistic: profile, periodogram, null fit, trials correction | L5 |
| `ink_signature_fold.py` | line-folded CT difference per depth and blank-papyrus noise | L6 |
| `pb_kedge_spectra.py` | effective lead-to-papyrus attenuation ratios for the two filtered beams | L8 |

Already public elsewhere in this folder tree, and therefore not copied: the layer-coherence score (`../../onsheet_score/`),
the Studio wrapper (`../../studio_t4/`), the triage row score (`../../controls/score_text.py`), the line proposal helper (`../../line_review/`),
the registration tools (`../../registration/`), and the headless window pipeline (`../../colab_pipeline/`).

Smoke checks made on the sanitised copies with synthetic inputs (not a validation):
`row_criterion.py` counts one event for eight dots at a 3 mm pitch in one map and none in the other;
`strokeness.py` returns about 0 for filled discs and about 1.6 for bars 4 mm long;
`linefold_stat.py` recovers a 6.75 mm pitch from a noisy synthetic profile;
`onsheet_gate_m7.py` and `depth_recentre_test.py` run to completion on synthetic stacks.
The first two results are worth a reviewer's attention: a perfectly round mark has a skeleton of one pixel, so the strokeness of real "dots" (0.6–1) reflects their irregular outline.

Not included: the label-free feature extractor (L1, L3, L4, L7), the per-row evaluation of PHerc. 0306B (C3), the 2.4 µm planning and stitching code (H2),
and the title-lane scripts other than the ghost match. Ask in `collab/` for a specific one.
