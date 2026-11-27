# B4 / PHerc0211 — frozen public calibration audit

Date: 2026-11-24. Scope: already-awarded PHerc0211 only. No new ink,
unread coordinates, prize submission or productive screening.

## Sources and frozen choices

Claude's published firstletters-colab/results/pherc0211_calibration at
1770a7b9f223a016167fda440f1867848373119d: 149 windows of 60 fitted
windings, 293.369 cm2; top 25 selected by published line rank, and the
15 SRSWOR ranks 27,29,37,43,52,64,76,88,91,108,122,127,129,138,144
(from report Task P, seed 2110). Full A01-A40 blinded-label key is pinned
by Git blob SHA1; original windows.json by Git blob SHA1.

At the fine-review stage, human visually flagged fw033_x1 (known location,
not blind), fw030_x1 (stroke-like, unverified different sheet), fw035_x1
(weak, previously seen known location). These are NOT independent ink GT.
The low-rank sample yielded no flags. No labels are imputed to other 109.

Primary diagnostic, fixed before actual public replay: exact one-sided 95%
finite-population hypergeometric upper bound on the number of potentially
flagged windows in the 124 lower-ranked windows after zero in 15 SRSWOR.
Report the conditional floor 3/(3+upper) for review-flag recall, not ink
recall. The Fisher probability for a 2x2 comparison of the 40 reviewed
windows is descriptive only and not a calibrated ranker gate.
Audit separately the 54 windows with null line score (lr=-99, NOT BLANK),
but do not add individual 95% stratum bounds as though simultaneous.

Acceptance: a >=90% high-recall scientific skip rule would require an
independent 95% lower recall bound >= .90 on a frozen eligibility cohort
and independent validated labels. The current archive cannot pass that
scientific gate. Never auto-exclude unreviewed or unscored windows.

## Nine paired-order public maps

The paired windings were already chosen by Claude with rng 211: fw014,
fw018, fw021, fw030, fw033, fw041, fw047, fw049, fw052.
Use byte-pinned original downscaled uint8 stride96 maps for desc and asc,
plus the common published valid_mask20 and parallel_mask20 intersection.
Do not resample and do not inspect unsupported pixels.

Compare both orders at uint8 threshold >=179 (about 0.70) on the whole
image, left half and right half. Fix MIN_VALID_PIXELS=100 and
MIN_ACTIVATIONS=25 per half with 0.10 opposite signed regional dominance.
Report counts and Jaccard, without post-hoc threshold selection.

A whole-map ratio near unity can hide strong complementary activations.
The regional test is descriptive only: activations may arise from paper,
folds, neighboring sheets, or detector artifacts. No anatomical identity
or independent ink ground truth is established. Absence of regional
opposition does not prove a negative. The published PNGs are quantized
display products rather than native float32 model output.

## Reproduction, CPU only

From a checkout with the published Claude results:

  python -m pip install numpy==2.3.5 pillow==11.3.0
  python -m unittest discover -s firstletters-colab/industrial_funnel/b4_0211 -p 'test_*.py' -v
  python firstletters-colab/industrial_funnel/b4_0211/audit_0211.py --windows firstletters-colab/results/pherc0211_calibration/maps_export211/windows.json --sealed-key firstletters-colab/results/pherc0211_calibration/look/P1/KEY_sealed.json
  python firstletters-colab/industrial_funnel/b4_0211/order_pair_0211.py --maps firstletters-colab/results/pherc0211_calibration/maps_export211

Output JSON reports scientific_admission=NOT_ESTABLISHED and
production_skip_allowed=false. No weights, CT access, model forwards,
GPU time, or private images.

This does not replace the failed Adaptive-071 50%-cutoff gate nor R1/R2
requirements in RFC-001: native paired coarse/fine float maps, equal
footprints, verified labels and end-to-end costs across manuscripts.
