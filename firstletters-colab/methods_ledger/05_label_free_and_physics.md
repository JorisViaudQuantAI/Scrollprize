# 05 — Label-free and physics ideas

These are attempts to find ink, or text structure, without the neural detectors, or to change what the detectors are given.
Most are negative. L8–L11 are design studies on paper: nothing was run for them unless stated.
Grey levels are those of the published uint8 volumes. One grey level is about 6.9 × 10⁻⁴ cm⁻¹ for the usual export window of −0.03…0.145.

---

<a id="l1"></a>
## L1 — Direct CT features at the sheet surface

**Question.** Does any simple, label-free statistic of the CT values around the surface separate known ink from blank papyrus at 8–9 µm?

**What was done.**
- Controls re-rendered as 41-layer stacks: PHerc. 0841 w00 (4.53 cm²), the PHerc. 0343 windings (8.09 cm²), the PHerc. 0306B crop (7.66 cm²), and a 14.6 cm² crop of a published PHerc. 172 surface volume.
- Per pixel, at half lateral resolution, the sheet is found in that pixel's own depth profile: brightest layer within ±10 of the mesh, penalised by 1 grey level per voxel of distance; then the steepest rise below and the steepest fall above, each within 8 voxels (the two faces).
- Features per face: `surf` (outermost sheet voxel), `crust` (outer 2 voxels minus voxels 4–7 inside), `edge` (face step), `tex` (high-pass energy at the surface against 5 voxels in: the "crackle" idea), `relief` (face height minus its 1.5 mm smooth).
- Depth-window features: 5-voxel white top-hat along depth, maximum over ±3 or ±6 layers; maximum projection over ±3 or ±6 layers.
- Sheet maps: thickness, contrast, raw face gradients, gap.
- Each map in three forms: locally normalised over 1.5 mm; fibre-notched (per 4.5 mm tile the two dominant orientations removed with a ±7° wedge); band-passed at stroke scale (Gaussian 0.1 mm minus 0.5 mm).
- Gate set beforehand: AUC ≥ 0.60 on a control the feature was not tuned on, with nulls near 0.5.

**Controls and nulls.** (i) Depth-shifted: the same feature sampled 10 voxels outward, or the window moved by ±13 layers. (ii) The other face. (iii) Spatial shift: labels rolled by at least 4 mm, 30 draws; this matters, because region-scale features reach 0.57–0.78 by chance on the box controls.

**Numbers.** AUC of the best form of each feature. Reference values from the detectors on the same pixels are in the last row.

| feature | PHerc. 0841 | PHerc. 0343 boxes | PHerc. 0306B box | PHerc. 172 |
|---|---|---|---|---|
| maximum projection ±6, stroke scale | 0.55 | 0.49 | 0.50 | 0.64 (nulls 0.55 / 0.54 / 0.51) |
| surface brightness, the two faces | 0.53 / 0.51 | 0.49 / 0.49 | 0.50 / 0.50 | 0.64 / 0.60 |
| depth top-hat ±3 | 0.53–0.55 | 0.52–0.56 | 0.50–0.52 | 0.55 |
| crust, sharpness, relief, thickness | 0.48–0.54 | 0.49–0.53 | 0.50 | 0.49–0.59 |
| crackle texture | 0.47 | 0.43 | 0.47 | 0.46 |
| raw face gradient (geometry) | 0.53 | 0.60–0.63 | 0.58–0.59 | 0.56–0.61 |
| reader-v3 / Studio | 0.863 / 0.92 | 0.684 / 0.736 | 0.572 / 0.601 | not run |

- The geometry rows sit inside their own label-shift null (up to 0.60): they mark well-imaged sheet, not ink.
- Within PHerc. 0841, spatial half-splits give 0.45–0.49: no stable signal even inside one scroll.
- No letter of any control is visible to the eye in the raw layers, PHerc. 172 included at 40 px/mm.
- Earlier, on Scroll 1: face-relative features at 4.8 and 9.6 µm gave 0.49–0.60; a depth top-hat at **2.4 µm** gave 0.73–0.76 on the control and 0.54–0.56 on a blank mark.

**Verdict.** Negative at 8–9 µm on the three scrolls that resemble the unread ones. Resolution is the lever, not feature design.

**How this could be wrong or miss ink.**
- The sheet-centre rule was changed once after a first pass on PHerc. 0841 (it was picking the wrong thin layer).
- Labels on PHerc. 0343 and 0306B are boxes (diluted positives); the key on PHerc. 0841 and PHerc. 172 is a model output.
- Only hand-made scalar features were tried. The detectors read multi-layer texture that these do not capture.
- The 2.4 µm result on Scroll 1 was never tried on the 2.4 µm bands of the unread PHerc. 1203 and 0846A.

**Not done.** The top-hat at 2.4 µm on an unread scroll. Learned but label-free representations.

**Where the code is.** Not included (about 600 lines of feature code); ask in `collab/` if you want a specific feature.

---

<a id="l2"></a>
## L2 — Layer-coherence on-sheet score

**Question.** Can a rendered stack alone tell whether the surface runs parallel to the sheet or cuts through wraps?

**What was done.** Two planes of the stack, 2 layers either side of the mesh layer, are high-passed (Gaussian, σ 6 px) and phase-correlated in 256 px tiles on a 128 px grid.
On a render parallel to the sheet the two planes sample different fibre layers and share no pattern. On an oblique cut the same bands appear in both planes, shifted.
Tile classes: **parallel** (response < 0.10); **drift** (response ≥ 0.10 and shift ≥ 1.5 px); **still** (response ≥ 0.10, shift < 1.5 px). A tile is scored when ≥ 90 % of its pixels are non-zero.
`f_parallel` = share of scored tiles that are parallel. Sheet-parallel area = valid area × `f_parallel`.
Parameters were fixed on three controls and five windings judged by eye, before the comparison below.

**Controls and nulls.** The atlas author's own eye verdicts on 59 windings (31 approved, 21 rejected, 7 partial). The three positives.

**Numbers.**

| item | value |
|---|---|
| approved against rejected, `f_parallel` | 0.893 against 0.730; AUC 0.83; Mann–Whitney p = 0.0001 |
| positives | 0.985 (0841), 0.951 (0306B), 0.848 (0343) |
| the same 0343 winding in another lane's render | 0.728 |
| atlas, area-weighted, by scroll | 0813 0.778; 0800 0.771; 0125 0.865; 0257 0.867; 0358 0.858; 0826 0.868; 0268 0.306 |
| atlas total | 1,044 of 1,309 valid cm² sheet-parallel; 52 of 250 windings below 0.7 |
| a block-profile alternative (CT peak near the mesh layer) | AUC 0.64, p = 0.09; not used |

**Verdict.** Worked as a ranking aid and as an upper bound on usable area. Published with tests in `firstletters-colab/onsheet_score/`.

**How this could be wrong or miss ink.**
- A tile with no structure at all (noise, air) also counts as parallel: the score is an upper bound.
- It is blind to depth offset: a render can be parallel to the sheet and several voxels off it.
- It was validated against someone else's eye labels on 59 windings and nothing else.
- It gives 0.848 and 0.728 for the same positive in two renders; we have not found why.
- It disagrees with our own eye on some small-scroll windings (0.81–0.93 where parts show wood-grain).
- Used as a mask (PHerc. 0175A lane), it removes 23 % of a positive's letter-box area.

**Not done.** Validation against a second set of labels. A depth-aware companion measure.

**Where the code is.** `firstletters-colab/onsheet_score/onsheet_score.py` (already public).

---

<a id="l3"></a>
## L3 — A tiny transparent classifier, trained on one control and tested on another

**Question.** If no single feature works, does a small combination transfer between scrolls?

**What was done.** Logistic regression (C = 0.1, balanced, 120,000 training pixels) on about 70 of the L1 features, or on the intensity-only subset; gradient-boosted trees (depth 4, 150 iterations) as a second model. Trained on each control, tested on each other.

**Controls and nulls.** Spatial-shift null (97.5th percentile) and a face-swapped null (the two faces' features exchanged).

**Numbers.** All features: 0.56–0.73 (quoted as 0.57–0.73 in `collab/claude-010`; the lowest pair is 0.564), but the weight sits on face position and thickness and the scores lie inside or at the edge of the spatial null (0.52–0.78). Intensity only: 0.51–0.58, with one pair at 0.635 (PHerc. 0306B → 0343) whose face-swapped null is 0.58. Trained on PHerc. 172, intensity only: 0.51–0.53 on the other three. Trees: 0.52–0.59.

**Verdict.** Negative: no small transparent model transfers usefully.

**How this could be wrong or miss ink.** The face-swap null is not valid for the all-features model (swapping flips the sign of its thickness term). One open point: the geometry model transfers to PHerc. 172 at 0.57–0.61 with a tight null, so "where ink is labelled" correlates with local sheet geometry across scrolls; this could be a real ink effect or a bias of where neural keys fire.

**Not done.** Models on raw depth profiles instead of frozen features.

**Where the code is.** Not included.

---

<a id="l4"></a>
## L4 — The PHerc. 172 feature over the whole atlas

**Question.** One feature (maximum projection ±6 at stroke scale) passed the gate on PHerc. 172. Does it flag anything on the 250 atlas windings?

**What was done.** The stacks were being streamed for L2 anyway, so the feature was computed on all 250 windings (1,309.4 valid cm², 890 cm² after a papyrus mask, no failures). Window statistic: 14 × 3 mm windows at 1.5 mm steps with ≥ 90 % papyrus; hot = robust z > 1.5; score = (hot fraction − winding median) / MAD.

**Controls and nulls.** The same window statistic on the four controls.

**Numbers.** Top window score: 3.75 for the PHerc. 172 text row, **3.87 for a PHerc. 0841 window** (where the ink is not CT-visible), 3.53 for PHerc. 0306B, 1.82 for PHerc. 0343. Eleven atlas windings reach 3.87 or more; all were looked at at 20 px/mm and show halos on cracks and gap edges, speckle, and cut-through grain.

**Verdict.** `uninformative` for every scroll: the statistic cannot rank known CT-visible text above other texture, so a quiet map says nothing.

**How this could be wrong or miss ink.** Not applicable as a negative; it is recorded so nobody reads the pass as coverage.

**Not done.** Nothing further.

**Where the code is.** Not included.

---

<a id="l5"></a>
## L5 — Line-pitch folding ("is there writing here"): NO-GO

**Question.** Text lines repeat at a constant pitch. Can that periodicity be detected in a map, or in raw CT, from an area too faint to read, and so point at a scroll, face and depth worth a deep read?

**What was done.** Statistic and thresholds were written before any data were looked at.
- Input: a 2-D map on a canvas where x runs along the lines and y across them, with a validity mask. A box is a rectangle.
- Profile r(y) = mean of the map over valid x in the box; rows with under 50 % valid pixels are interpolated.
- Detrend by subtracting a Gaussian-smoothed copy (σ 8 mm); standardise to unit variance.
- Periodogram with two harmonics: Z²₂(P) = (2/n) Σ_{k=1,2} |Σ_y r(y) e^{2πiky/P}|², for trial pitches P of 3–9 mm.
- Boxes at the same y-range are combined coherently (the profile is averaged over the full width); boxes at different y or on different windings incoherently (sum of Z²₂).
- Null: Z²₂ at every trial pitch in blank-papyrus boxes of the same size, fitted by a scaled χ² (scale and degrees of freedom by moments).
- Local p from that fit; trials-corrected p = 1 − (1 − p)^N with N = 2 × H × (1/3 − 1/9), H the box height in mm; z from the normal quantile.
- **GO criteria**: trials-corrected z ≥ 5 at the true pitch from a box of ≤ 20 cm², on at least two different scrolls, at least once where letters are not legible by eye, with every null below 3.
- **NO-GO**: it works only where letters are already legible, or a null reproduces the peak.

**Controls and nulls.** Wrong layer order; depth-shifted window; the statistic with x and y exchanged; blank papyrus of the same sheet; raw CT brightness on the text area.

**Numbers.** z at the true pitch; where two values are given the second is trials-corrected at the best pitch.

| control | resolution | feature | 2 cm² | 4 cm² | 12 cm² | 20 cm² | 66–80 cm² | legible by eye |
|---|---|---|---|---|---|---|---|---|
| Scroll 1, column 3 | 19 µm | published 2.4 µm ink map | — | 6.0 | — | 8.4 | 11.7 | yes |
| Scroll 1, column 3 | 19.2 µm | reader-v3, reading order | 2.7 | 3.9 | 4.4 | **5.7 / 5.5** | — | yes |
| same | | reader-v3, wrong order | 0.9 | 0.9 | 1.1 | 1.8 / 1.0 | — | no |
| same | | reader-v3, window −48 µm / +48 µm | 2.4 / 2.7 | 3.1 / 4.0 | 3.9 / 4.3 | — | — | yes |
| same | | reader-v3, window −77 µm | 2.3 | 3.1 | 3.6 | 4.2 / 3.7 | — | fragments only |
| same | | reader-v3, window **+77 µm** | 1.6 | 2.4 | 3.3 | **4.8 / 4.2** | — | **no** (isolated blobs) |
| same | | rotated 90° (fold along the lines) | 0.3 | 0.4 | 0.4 | — | — | — |
| Scroll 1, final column (5 lines) | 19.2 µm | reader-v3 fwd / rev; published map; raw CT | — | — | 4.1 / 1.7; 6.1; 0.5 (9 cm²) | 4.1 / 1.8 (17 cm²) | — | fwd yes |
| Scroll 1, columns 2 and 3 | 19.2 µm | raw CT, mesh plane | — | 0.2–0.6 | — | 0.9–1.3 | 1.4–1.6 | — |
| same | | raw CT, best of 27 pooled planes | — | — | — | ≤ 1.5 | ≤ 2.3 | — |
| same | | depth top-hat, best of 27 planes | — | — | — | ≤ 1.1 | ≤ 2.3 | — |
| same | | cross-fibre stroke energy, best of 27 planes | — | — | — | ≤ 2.4 | 3.9 and 3.7 | — |
| same | | sheet column mass (L7) | — | — | — | ≤ 1.3 | ≤ 3.4 (79 cm²) | — |
| Scroll 1, column 3 | 38.4 µm fed as 9.6 µm | reader-v3, reading order | — | 1.0 | — | 0.7–1.2 | 3.5 (79 cm²) | no |
| PHerc. 1447, three windings stacked in absolute z | 8.64 µm | base-model map | — | — | — | — | 0.1 corrected; 1.4 at the label pitch (69 cm²) | partly |
| PHerc. 1447 | | human labels / CT layer / column mass | — | — | — | — | 1.1 / 0.5 / 1.5 | — |
| PHerc. 0841 w00 | 19 µm | published canonical map | — | — | 1.4 corrected (7 cm²) | — | — | yes |

- Pitch and phase on Scroll 1 column 3 (true 6.77 mm): 6.60–6.64 mm and within 0.010 cycle in every reading-order condition, including the illegible +77 µm one; the wrong order does not lock (8.77 mm).
- The scaled-χ² null fits give a scale of 15–30 and 2.5–4 degrees of freedom; white noise would give 1 and 4. Blank boxes calibrated this way exceeded a nominal 5 % level in 3–17 % of cases.
- At 66–80 cm² the nulls rest on 4–7 boxes, so those z values are indicative only.

**Verdict.** NO-GO by the criteria as written. It fails the bar where the eye fails (4.8 against 5), it has no second scroll, and it never works on label-free CT features.
Stage 2 (scoring the maps of unread scrolls) was therefore not run.

**How this could be wrong or miss ink.**
- The NO-GO is partly a lack of controls: PHerc. 1447 has a few short rows, and PHerc. 0841 offers 7 cm² with no blank region to calibrate a null. The statistic was not shown to fail on a real second column.
- It did recover pitch and phase in a condition where the eye sees only blobs. A reviewer may judge that worth an exploratory pass over the 520 cm² of PHerc. 0175A maps and the 364 cm² of PHerc. 0125 maps, with the false-alarm rate measured on those maps.
- All Scroll 1 detectors were trained on Scroll 1, so its numbers are optimistic.
- The 20 cm² boxes used for the degraded runs were 3 × 6.7 cm; a taller, narrower box with 20 line periods might score higher. Not tested, to avoid tuning after the fact.
- Hazards if it is ever applied to unread maps: an oblique cut through layers 5–8 voxels apart gives bands every few mm; scan stitching can be periodic along the axis; blank papyrus is asymmetric between orders (D7).

**Not done.** Stage 2. Wrong-order maps for PHerc. 1447 and 0841. Pooling across windings with a per-winding phase.

**Where the code is.** `code/linefold_stat.py`.

---

<a id="l6"></a>
## L6 — Ink signature in CT units

**Question.** Averaged over many lines of known text, how much brighter or darker is the CT under ink than between lines, at each depth?

**What was done.** Scroll 1, a text column of the published 2.4 µm surface volume, folded at the line grid measured on the published ink map.
For each sample (one line × one 1 mm strip) and each layer: mean CT in the line-centre band (phase within ±0.1 of the centre) minus mean CT in the interline band (phase 0.4–0.6).
Two depth references: the mesh layer; and, per pixel, the brightest layer of the depth-smoothed profile within ±30 layers.
Per-pixel noise on blank papyrus after removing trends longer than about 0.25 mm.

**Controls and nulls.** The 109-layer search itself (a difference must survive the number of depths tried).

**Numbers.**
- Mesh-referenced: within ±3 grey levels at every 2.4 µm layer (standard error 0.5–1.6); largest −3.0 ± 1.5 at +31 µm; within ±1.8 pooled to 9.6 µm in depth.
- Sheet-aligned: largest −2.2 ± 0.75 grey at +17 µm from the brightest layer; everything else within 2 standard errors.
- The centre band holds 30 % ink pixels and the interline band 2.7 %, so per ink pixel this is about −8 ± 3 grey levels: a 2.9σ hint, sign "darker", after a depth search.
- Blank-papyrus pixel noise: 31–33 grey at 9.6 µm laterally, 21–27 grey at 2.4 µm.

**Verdict.** The mean imprint of ink on CT brightness is at most about 0.3σ per pixel. Ink at this protocol is not a brightness offset, which is why averaging, stacking and simple features do not find it.

**How this could be wrong or miss ink.** One column of one scroll at 2.4 µm and 78 keV; the unread scrolls are at 9 µm and 113–116 keV. The sign and size may differ for another ink. Samples overlap in the lateral smoothing, so the standard errors are somewhat optimistic.

**Not done.** The same on PHerc. 0841 at 9.37 µm.

**Where the code is.** `code/ink_signature_fold.py`.

---

<a id="l7"></a>
## L7 — Sheet column mass

**Question.** Does ink add measurable mass to the column of CT through the whole sheet, as a heavy element or a 10–20 µm deposit would?

**What was done.** Scroll 1, a 1.9 × 1.9 cm crop of a text column at 9.6 µm laterally. Column mass = mean over layers of (CT − per-pixel gap level), over the whole sheet (layers 24–84 of 109), and separately over each half and over windows outside the sheet. Labels: published 2.4 µm map > 0.5 against < 0.15.
Kill criterion set beforehand: AUC ≤ 0.55 and an excess under ink below 2 %.

**Controls and nulls.** Labels rolled by 3.5 × 3.0 mm. A window of the outermost 30 layers, outside the sheet.

**Numbers.**

| column window | AUC | AUC with rolled labels | excess under ink |
|---|---|---|---|
| whole sheet | 0.516 | 0.462 | +1.6 % |
| lower half | 0.436 | 0.506 | −9.0 % |
| upper half | 0.591 | 0.447 | +18.2 % |
| outermost 30 layers, outside the sheet | 0.579 | 0.447 | +19.6 % |

**Verdict.** Negative. The upper-half value is reproduced by a window outside the sheet, so it is geometry (more material on that side under text rows), not ink mass.

**How this could be wrong or miss ink.** This was measured where ink is easiest (2.4 µm, 78 keV, Scroll 1), not on a 9.36 µm scan; the 9.36 µm measurement on PHerc. 0841 was planned and not made. PHerc. 172 shows about +7 % column mass under ink, so the result is scroll-specific.

**Not done.** PHerc. 0841 and PHerc. 0139 at 9.36 µm.

**Where the code is.** Not included (it depends on the L1 feature code).

---

<a id="l8"></a>
## L8 — Dual energy and the lead K-edge: our two studies disagree

**Question.** PHerc. 1203 and 0846A each have a 9.36 µm scan labelled 113 keV and a 2.4 µm scan labelled 77 keV. The K-edge of lead is at 88 keV. Can the pair show lead in the ink?

**What was done.** Two paper studies, by two sub-agents, with different assumptions. Nothing was measured on the volumes.
- **Study A (literature survey).** Used tabulated mass-attenuation values at the nominal energies, and the published statement that the 0.1 mm gold filter of the 2.4 µm protocol cuts the spectrum above the gold K-edge (80.7 keV). Result: the excess of a lead deposit relative to papyrus changes by a factor of about 1.7 between the two scans, against 1.0 for carbon and 0.4–0.7 for iron or calcium minerals. Three classes, separable with one pair.
- **Study B (scan physics).** Modelled both beams as filtered polychromatic spectra: an on-axis wiggler spectrum, the filters named in the operator comment of each volume's metadata record, the scintillator's absorption, an energy-integrating detector. Result: effective lead-to-papyrus ratios of 24.1 (high beam) and 25.0 (low beam). The difference, +0.9 to +1.3 over the range of source parameters tried, keeps about 1/25 of the single-energy signal. Detection limit of the pair about 80 µg/cm² per 0.3 mm patch.

**Why they disagree.** Study A treats each beam as its nominal energy. Study B says each spectrum is 30–40 keV wide and straddles the edge: the low beam has 82 % of its detected weight below 88 keV, but lead's attenuation at 50–80 keV is as high as just above the edge, so the averages cancel. For monochromatic beams study B's own table gives 15.2 against 26.7, which is study A's factor.
The spectra are not published. Study B's source and filter model is an assumption; so is study A's "cut".

**Controls and nulls.** None run. Study B proposes mineral grains (bright inclusions) as a free calibration of any dual-energy residual.

**Numbers.** Points on which both studies agree:
- Published lead levels for Herculaneum ink are 16 and 84 µg/cm² on two opened fragments; no study has measured lead in situ in a scanned scroll.
- Lead at those levels would already be visible in single-energy CT: study B computes +0.87 grey per voxel and +0.7 % of a sheet's column mass per µg/cm², so +11–55 % column mass; study A estimates +7 % to +35 % of papyrus brightness.
- The read controls show nothing of that size (L1, L7), so their ink lead is far below those levels.
- A cleaner test exists on a read scroll: PHerc. 0139 has two 2.4 µm scans at the same distance with different filter sets (labelled 77 and 78 keV); study B computes lead ratios of about 25 and 35 for them.

**Verdict.** Not run. If study B is right the pair on PHerc. 1203 and 0846A is useless for lead; if study A is right it is a three-way composition test. Either way single-energy contrast is the stronger lead detector, and it shows nothing on the controls.

**How this could be wrong or miss ink.** "The controls show no lead" was inferred from surface features and column mass on four read scrolls, not measured on the unread ones. An unread scroll with leaded ink would look different from all our controls, and our screens assume it does not.

**Not done.** The spectra check on mineral grains. The PHerc. 0139 pair. Any single-energy "bright deposit" screen of the unread scrolls.

**Where the code is.** `code/pb_kedge_spectra.py` (study B's calculation, with its assumptions listed in the header).

---

<a id="l9"></a>
## L9 — Phase and the two propagation distances

**Question.** Are the 0.22 m and 1.2 m scans of one scroll a two-distance phase measurement, and can phase reveal a thin ink layer?

**What was done.** Read the acquisition record published with every volume (`metadata.json`), and worked through the optics.

**Controls and nulls.** None.

**Numbers.**
- Every BM18 volume is Paganin phase-retrieved with δ/β = 1000 and then unsharp-masked (coefficient 4, σ 1.2 px). The 9.36 µm volumes are 4.68 µm detector pixels binned 2 × 2. Only uint8 volumes are published; no projections.
- Fresnel scale: 3.6 µm at 113 keV and 1.2 m, against 9.36 µm voxels; 1.9 µm at 77 keV and 0.22 m, against 2.4 µm voxels.
- Measured edge overshoot at 9.36 µm: +18 % and −10 % of the step; the unsharp mask alone predicts ±14 %.
- One read scroll (PHerc. 0139) is also published with δ/β = 50 and with almost no phase retrieval; that version has 14 % of voxels clipped at 0.

**Verdict.** Ruled out on paper. The two scans differ in energy, are already retrieved and filtered, and are 8-bit; the fringe is below one voxel in both.

**How this could be wrong or miss ink.** Not a screen. One test remains reasonable on a read scroll: compare features on the unretrieved and the retrieved reconstruction of PHerc. 0139 against its ink map.

**Not done.** That test.

**Where the code is.** None.

---

<a id="l10"></a>
## L10 — Super-resolution and deconvolution of the 9 µm volumes

**Question.** Can fine detail that the 2.4 µm models use be recovered from the 9.36 µm volumes?

**What was done.** Measured the edge-spread on the outer papyrus-to-air surface of PHerc. 1203 at the same place in both scans (thousands of profiles along the normal), and derived the system transfer.

**Controls and nulls.** None.

**Numbers.** 10–90 % edge rise: 17.5 µm in the 9.36 µm volume, 11.4 µm in the 2.4 µm volume. Transfer at 9.36 µm: 0.31 at a 40 µm period, 0.11 at 33 µm, about 1.5 × 10⁻⁴ at 20 µm. Noise in air: 3.25 grey per voxel; papyrus voxel texture about 20 grey.

**Verdict.** Ruled out on paper. Detail at 5–20 µm is transferred at 10⁻³–10⁻⁴ and lies far below the noise; a network can only invent it. What survives at 9 µm is low-frequency, and a detector trained at 9 µm on labels from 2.4 µm (which the released models are) uses it directly.

**How this could be wrong or miss ink.** The fine-scan figure is an upper bound (surface roughness adds to it). The 0306B winners report the same from experience: upsampling the render lost ink, and deconvolving the ink map invented strokes.

**Not done.** A coherence spectrum between registered coarse and fine volumes.

**Where the code is.** None.

---

<a id="l11"></a>
## L11 — Training or adapting a detector ourselves

**Question.** Would a home-made 9 µm detector, or an adaptation of a released one, find ink where Studio and reader-v3 find none?

**What was done.** A design study from the public record: the winners' repositories, model cards, community measurements. Nothing was trained.

**Controls and nulls.** None.

**Numbers.** Options and the prior we gave each of a real gain. The priors are judgements, not measurements.

| option | cost | why it ranks where it does | prior |
|---|---|---|---|
| pooled line-pitch statistic on existing maps | CPU | the only item that needs no training; tested at once (L5) | 10 %; now NO-GO |
| product or ensemble of released models | inference only | the baseline any new model must beat | a bar, not a bet |
| intensity audit and per-scroll input map for Studio; batch-norm statistics for Hecate | ≤ 2 L4-h | reader-v3 is immune to intensity shift and also sees nothing | 5 % |
| 2.4 µm band to locate, 9 µm to show | 3 L4-h per 100 cm² | two physical gains at once; became the fine-resolution lane | 5 % |
| fine-tune reader-v3 with labels of formerly unread scrolls, two positives held out | about 4 L4-h | the 0306B winners report that fine-tunes, self-training and students "all drew the same letters"; PHerc. 1447 fine-tunes did not beat the base model on a held-out winding | 3–5 % |
| self-training on the target scroll with pseudo-labels | 3–4 L4-h per scroll | its authors measured +0.003–0.005 AUC; on PHerc. 0125 it would train on dot rows | 2–3 % as a finder |
| rebuild cross-resolution distillation | 10–40 L4-h | already inside every released model | 3 % |
| face asymmetry as a self-supervised signal | 5–10 L4-h | solved from fibre anatomy without ink | 2 % |
| synthetic ink pasted on target papyrus | 10–20 L4-h | others measured AUC 0.6–0.77 against 0.85 for real text; L6 says there is no additive signature to paste | 2 % |

**Verdict.** Not run. Every public label is already inside the released models, so we have no data advantage, and every cheap adaptation recipe moved known-text AUC by 0.00–0.05 for those who tried.

**How this could be wrong or miss ink.**
- The study predicts that training will not help; it did not test it. The fine-tune with held-out positives would cost about 4 L4-hours and was not run.
- The intensity audit (do the unread scrolls' CT histograms fall inside Studio's training range?) costs CPU only and was not run.
- The remaining scrolls are those on which many teams with these models saw nothing: a selected, harder set. Arguments from the won scrolls may not carry over.

**Not done.** All of the above.

**Where the code is.** None.
