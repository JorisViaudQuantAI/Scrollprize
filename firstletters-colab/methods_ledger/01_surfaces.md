# 01 — Surfaces

A detector can only read a surface that lies on one sheet of papyrus, at the right depth, over a few square centimetres.
Most of our uncertainty is here, not in the detectors.
Voxel sizes: 8.64 µm (PHerc. 0175A, 0268, 0483B, 0490A, 0800, 1218, 0343, 0306B) or 9.36 µm (the others).
"m7" is the organisers' published surface prediction for each volume; it is a model output, not ground truth.

---

<a id="s1"></a>
## S1 — Patch-grown windows and the on-sheet gate

**Question.** Can a headless patch grower give single-sheet surface of at least 4 cm² on unread scrolls at low cost, and how do we decide that a window is on one sheet before spending ink time?

**What was done.**
- Tool: the VC3D AppImage, `vc_grow_seg_from_seed` ("GrowPatch") on the m7 prediction and its normal grids, then `vc_render_tifxyz`.
- Render: 41 layers, one native voxel per layer along the mesh normal, mesh at layer 20, pixel = native voxel, uint8 read from the published masked volume without rescaling, no `--flip-normals`.
- Seeds are snapped to the m7 sheet before growing.
- Three generations of windows:

| set | windows | generations | chart area each | how seeds were chosen |
|---|---|---|---|---|
| first window and small scouting patches | one grown window (PHerc. 0175A) and small own patches on 5 scrolls | 80 for the grown window | 7.45 cm²; patches 1.2–3.9 cm² | scan-quality map targets, by hand |
| nine "pre-protocol" windows (PHerc. 0175A, 0358, 0813; three each) | 9 | 70 | 5.6 cm² (8.64 µm), 6.6 cm² (9.36 µm) | surface-prediction coverage only |
| tranche 1 (10 scrolls) | 39 | 100 | 11.7 cm² (8.64 µm), 13.7 cm² (9.36 µm) | scored on anatomy only, rule below |

- Tranche seed rule, fixed before any ink. Candidates every 192 voxels at 2–3 heights per scroll, in the clearest bands of a public scan-quality map and away from public meshes.
  Per candidate, from the m7 prediction (level 1, 64-slice slab) and CT (level 2): planarity and bend of the structure tensor, |n_z|, median spacing of m7 bands along the normal and its coefficient of variation, band continuity over 25 probe lines, CT contrast along the normal.
  Score S = planarity − bend + CT contrast + continuity − spacing variation + clipped spacing, each as a robust z.
  Filters: radial fraction 0.25–0.90, spacing ≥ 10 voxels, bend ≤ 25°, ≥ 3 mm from the scan edge, ≥ 25 mm from our earlier seeds.
  Strata: height × radial band (0.25–0.55 and 0.55–0.90); best S per cell; picks ≥ 30 mm apart; a cell with no candidate is relaxed and recorded.
  Six extra seeds were taken later with 20 mm exclusion and separation.
- Number of generations, from a pilot on one PHerc. 0358 seed with a rule fixed first (largest value that grows in ≤ 900 s with ≥ 1.2× the on-sheet area of the next smaller value):

| generations | grow time | chart cm² | on-sheet cm² (fraction) | clean cm² |
|---|---|---|---|---|
| 70 | 306 s | 6.68 | 3.39 (0.52) | 0.91 |
| 100 | 653 s | 13.74 | 7.24 (0.53) | 1.20 |
| 130 | 1,042 s | 23.34 | 10.97 (0.47) | 1.39 |

- **On-sheet test (the "m7 pixel test").** The m7 volume is rendered on the same mesh with the same call.
  A pixel is on-sheet when an m7 band (value > 127) lies within ±3 voxels of the mesh.
  A pixel is "clean" when, in addition, no other band lies within ±9 voxels.
  A 1 mm block is on-sheet when ≥ 60 % of its valid pixels are on-sheet.
  On-sheet fraction = on-sheet block area / valid area.
  **Core** = the largest 4-connected region of on-sheet blocks.
- **Gate.** First written as "on-sheet fraction ≥ 0.5". Changed to "core ≥ 4 cm²" after the first CT checks and before any window of the tranche was inked.
  A secondary arm inked windows that fail the core gate but have ≥ 5 cm² on-sheet in total; they are reported separately.
  After the first five secondary windows the arm was restricted by an eye check of the mid layer before ink.
- Each window also gets a 3 × 3 grid of CT cross-sections (3 mm wide, ±48 voxels along the normal) with the mesh line and ±8-voxel ticks, and an en-face CT image.

**Controls and nulls.**
- The awarded PHerc. 0343 winding through the same check scores on-sheet 0.73 and clean 0.55. A mesh that carries legible text is therefore not 100 % on-sheet by this test.
- The seed features separate two crush-zone seeds (planarity 0.63–0.66, bend 32–35°) from two clean ones (0.92–0.97, 7–13°) among four earlier PHerc. 0358 seeds.
- There is no null for the gate itself (no window known to be off-sheet with ink underneath).

**Numbers.**

| quantity | value |
|---|---|
| tranche windows with a verdict | 39 on 10 scrolls |
| all 39: chart / valid / on-sheet | 505 / 485 / 241 cm² (private final figure) |
| same, as published in `collab/claude-011` | 38 windows, 492 / 471 / 235 cm² (interim, one window missing) |
| inked and reviewed | 22 windows on 9 scrolls: 288 chart / 280 valid / 151 on-sheet / 98 cm² in cores |
| not inked or judged not one sheet | 17 windows, `uninformative` |
| on-sheet fraction of every window | 0.36–0.73; none ≥ 0.8 |
| strata (38 windows, interim) | below 0.5: 20 windows, 9 inked; 0.5 to below 0.8: 18 windows, 16 inked |
| a re-grown mesh from a seed on a flagged feature | coincides with the old mesh: median normal distance 0.005 and 0.010 voxels |
| same seed re-grown on another machine | chart identical (13.744 cm²); on-sheet 0.708 against 0.725 |

Per-scroll rows are in [07_negative_coverage.md](07_negative_coverage.md).

**Verdict.** Worked in part. At 100 generations a window is on one predicted sheet over about half its area, and growth beyond about 70 generations adds mostly oblique margin.
The tranche did not reach its target of 250–300 cm² of on-sheet surface.

**How this could be wrong or miss ink.**
- The gate trusts m7. Where m7 is wrong (crushed zones, merged layers) both the seed score and the on-sheet test are wrong in the same direction.
- The pixel test overstates usable area in crushed windows: several "on-sheet" windows are a star of crossed folds in the mid layer, and the ink maps follow the radial streaks.
- One row-like feature (F2) sat in a window that failed the core gate. A hard cutoff would have discarded it.
- Seeds scored on anatomy avoid crushed and compressed zones. If writing survives mainly there, the tranche never looked.
- The mesh normal sign is not controlled by the grower, so both orders must be run on every window; we did.
- A re-grown mesh converges to the same surface. "We re-grew it and the feature changed" is a re-tessellation test, not an independent surface.
- One global depth offset per window (brightest layer of the window-mean profile) centres the detector window. Local sheet relief of ±10 voxels is not followed.
- Windows of 6–14 cm² are small against a column of text (about 6 cm wide on Scroll 1): a window can cut a line anywhere.

**Not done.** Growth with fewer generations and more seeds per band. A CT term in the seed score (voids, wood-grain) over a 2–3 cm radius. Any surface deliberately offset from the grown one. Windows on PHerc. 0257 and 0268 (dropped for low scan quality) and on PHerc. 0846A (left to others at the time).

**Where the code is.** `code/onsheet_gate_m7.py` (pixel classes, block rule, core, gate). The headless grow-and-render scripts are in `firstletters-colab/colab_pipeline/`.

---

<a id="s2"></a>
## S2 — Published atlas meshes, inked as published

**Question.** A public atlas holds spiral-fit windings of seven eligible scrolls with ready-made renders. Can they be screened as they are?

**What was done.**
- Meshes: the public repository `pscamillo/vesuvius-eligible-meshes`. Renders: the Hugging Face dataset `rodriguescarson/eligible-scroll-atlas-renders`.
  Each render is a (31, H, W) uint8 stack made with `vc_render_tifxyz --flip-normals`; plane 15 is the mesh.
- Population used: the 250 windings that have an area value and a published render.

| scroll | windings | chart cm² |
|---|---|---|
| PHerc. 0813 | 75 | 437.0 |
| PHerc. 0800 | 95 | 501.9 |
| PHerc. 0125 (the 62 older windings) | 62 | 389.4 |
| PHerc. 0257 | 6 | 22.7 |
| PHerc. 0268 | 6 | 19.6 |
| PHerc. 0826 | 3 | 14.1 |
| PHerc. 0358 | 3 | 11.1 |
| total | 250 | 1,395.8 |

- Per winding: download, check (31 layers, ≥ 20,000 valid pixels, valid pixels against the manifest), ink, post-process, delete the stack.
- The meshes were **not** moved (no snap, no re-centring). See S7 and S9 for why.
- **Reading-face rule.** The rule of the PHerc. 0306B winners' repository (`reading_face`, from the winding geometry and the umbilicus) was computed for all 250 windings, with the official umbilicus (5 scrolls) and with the ring centroid (all 7).
- The newer set of about 278 PHerc. 0125 windings added to the atlas later is not in the render manifest and is not part of this population (a teammate's lane).

**Controls and nulls.**
- Renderer identity: our renderer and the published render of one PHerc. 0358 winding agree layer by layer in the same order (r = 0.998 at layer 15, 0.983 at layers 5 and 25; 0.155 with the order reversed), with zero lateral offset. Checked on that one winding only.
- Face rule on the positives: the rule labels the PHerc. 0343 windings `reverse` and the PHerc. 0306B claim mesh `forward`. In our flipped-normal renders the letters read in the opposite order in both cases. So the rule's label is the order without `--flip-normals`.

**Numbers.**
- All 250 windings get the label `reverse` from the rule, with the two centre definitions agreeing on 238 of 238 where both exist.
- The majority share behind the label is ≥ 0.9 for 244 of 250 windings. The six lower ones are short PHerc. 0813 windings (0.51–0.88 against the official umbilicus, 0.94–1.0 against the ring centroid).
- Depth placement of the atlas, from one winding: raw per-block CT peak within ±2 voxels of the mesh for 33 % of blocks; the atlas's own documentation says 38–46 %.
  Offset field p1–p99 −7.0…+5.1 voxels; 25 % of the area needs a correction above 4 voxels.
- Snap offsets on three PHerc. 0813 windings: median 4.0, 3.3 and 1.6 voxels; 28, 34 and 60 % of vertices already within 2 voxels.

**Verdict.** Worked with stated limits: 1,294.6 cm² of valid render were screened and 14.8 cm² judged `uninformative` (PHerc. 0268).

**How this could be wrong or miss ink.**
- reader-v3 was run in **one order only** on these windings, the order given by the rule. The rule was checked on two positives from another scan campaign and never on known text of these seven scrolls.
- The surfaces are several voxels off the sheet over most of their area. Our tolerance figures (D8) come from positives that are on the sheet, with a smooth injected error.
- A third of the area of PHerc. 0813 and 0800 is not sheet-parallel by our own score (L2).
- The atlas samples each scroll: selected z tranches and selected windings. The fraction of each scroll's papyrus that it covers was not measured.
- The same atlas held 90 PHerc. 0211 windings that had been screened negative with older tools before that scroll was won.

**Not done.** The other order with reader-v3. The production path on the PHerc. 0211 atlas windings as a hindsight test. A paired raw-against-snapped ink comparison at common sites (prepared, cancelled).

**Added 2026-11-23 (hindsight test, after PHerc. 0211 was won).**
- The atlas publishes every 20th winding of each fitted z-window (w020, w040, w060, w080, w100), about 5 % of the windings in those windows, and leaves some z-windows out.
- The awarded PHerc. 0211 letters lie on a sheet between two atlas windings. No vertex of the 90 atlas meshes of that scroll comes closer than 46 voxels to the winner's published surface. This screen could not have found them.
- On the winner's published mesh, the unchanged production path (reader-v3 in the rule order, the line finder, the montage review) ranks the letter row first of 28 places, also when the surface is displaced by 3–9 voxels. So the miss is one of sampling, not of detection.
- Read every atlas row of this ledger as "windings w020 … w100 of each fitted window show nothing"; it says nothing about the other 19 windings in 20. Areas in cm² overstate the coverage of a scroll region by about 20 times.
- The other order with reader-v3 has now been run on a sample of 10 windings (49.9 cm²): nothing.

**Where the code is.** Not included; the face rule is the winners' public code, and the per-winding loop is plain I/O.

---

<a id="s3"></a>
## S3 — Spiral fit with a shortened schedule (PHerc. 0358): negative

**Question.** Does the organisers' spiral fitter reach the sheet on a 15 GB GPU when the 30,000-step schedule is shortened?

**What was done.**
- Fitter: villa `fit_spiral.py` at commit `0fb38c45`, with the two small patches the winners' recipes apply (outer shell optional, stop-at-step) and a third of ours (stop after a wall-time limit and resume).
- Overrides as in the winners' recipes: patches disabled, shell weights 0, `dense_spacing_mode: phase`.
- Inputs for band z 9200–10700: published tracks (1,895,471 tracks, 86.1 M points in the band), Lasagna normals (group 2), a surface distance field built from m7 (level 1, threshold 128, CT-masked), official umbilicus. The fitter computed 3.89 M exact crossings itself.
- Run A: winding sense `CW`, horizon 10,000 steps, stopped at 1,000, 2,900 and 4,800.
- Run B: sense `ACW` (S5), band z 9450–10450, a complete 7,000-step schedule with the track distance-transform stage starting at one third, as in the 30,000-step recipe.
- Schedule facts read from the pinned code: the learning rate decays exponentially to 0.3× over the horizon; the track distance-transform loss switches on at step 10,000 of 30,000. Run A never reached that stage.
- Check: `code/spiral_oncheck.py`, 150–200 random valid vertices per winding, profiles along the normal over ±16 voxels through CT (level 1) and m7 (level 0).

**Controls and nulls.**
- Control: the same vertices shifted along the normal by a uniform random offset in [−8, +8] voxels.
- Calibration: the published, snapped PHerc. 0343 winding w047 scores 0.60 on a sheet against 0.28 for its control (40 vertices).
- Sense null: `CW` and `ACW` probes of 1,000 steps each.

**Numbers.**

| run A, 94 windings inside the scroll, 4,800 steps | raw fit | after snap | control |
|---|---|---|---|
| vertex on an m7 sheet | 0.23 | 0.478 (p10 0.42, p90 0.55, max 0.61) | 0.25 |
| median distance to the nearest m7 sheet (voxels) | 4.2 | 1.9 | 3.8 |
| CT peak within 2 voxels | 0.42 | 0.46 | 0.42 |
| CT z-score at the surface | +0.01 | +0.12 | +0.02 |

- Snapped fraction on the same 40 windings at 1,000 / 2,900 / 4,800 steps: 0.43 / 0.46 / 0.46.
- Run A output: 120 windings, 1,047 cm²; 94 windings (667 cm²) inside the scroll.
- `CW` against `ACW` at 1,000 steps: loss 931 against 933 at step 800; satisfied tracks 11.6 % both; snap rejection 0.2426 against 0.2417.
- Run B: loss flat from about step 5,000; satisfied tracks 11.7 % → 14.1 %. Fraction of vertices within 2 voxels of an m7 sheet centre before snapping: 0.286, against 0.250–0.251 for run A, whose same-vertex control equalled the fit.
- A pilot of eight snapped windings of run A (38.8 cm², snapped fraction 0.51–0.61) and two raw ones was rendered. On the mesh layer every panel shows the surface cutting obliquely through neighbouring sheets, with a clean fibre grid on a third to a half of the area. No ink was run; verdict `uninformative`.

**Verdict.** Negative for this scroll at these step counts. It does not show that the method fails on PHerc. 0358: the full 30,000 steps were never run there.

**How this could be wrong or miss ink.**
- Run B lost its final meshes, checkpoint, same-vertex control and CT panels when the virtual machine stopped. Its 0.286 comes from the snap script's own report, not from our check.
- The sense is not established on this scroll (S5). With the wrong sense the windings are mirrored spirals.
- The core of this scroll is a crushed void with folded sheets; the band may simply be a bad one.

**Not done.** The 30,000-step schedule on PHerc. 0358 (about 9 T4-hours). Any ink on these windings.

**Where the code is.** `code/spiral_oncheck.py`.

---

<a id="s4"></a>
## S4 — Spiral fit with the full 30,000-step schedule (PHerc. 0175A, 1203, 0846A)

**Question.** With the winners' full schedule on a 23 GB GPU, does the raw fit leave the chance level, and on how many windings?

**What was done.** Same fitter, patches and overrides as S3, horizon 30,000, track distance-transform stage from step 10,000, run in resumable chunks of 26–35 minutes.

| | PHerc. 0175A | PHerc. 1203 | PHerc. 0846A |
|---|---|---|---|
| band (z, level-0 slices) | 6900–8400 | 9000–10400 | 2450–3750 |
| why this band | most regular wraps by eye on 7 slices; matches the scan-quality map | most regular part inside the 2.4 µm band | clearest concentric spiral inside the 2.4 µm band |
| tracks | published store; 2,012,154 tracks, 145.5 M points in the band | **our own** (S6); 1,240,635 tracks, 53.7 M points | **our own** (S6); 1,491,530 tracks, 80.2 M points |
| crossings (computed by the fitter) | 7.10 M | 2.49 M | 4.09 M |
| normals | synthesised from m7 with the 0306B recipe's script (no Lasagna normals exist) | Lasagna, group 2 | synthesised from m7 |
| umbilicus | **our own**: centre placed by eye on 16 slices, smoothed (σ 75 slices); unreviewed | official, smoothed the same way | **our own**: 11 slices, by eye, smoothed |
| winding sense | `ACW`, an assumption taken from a comment in the 0306B recipe | `CW`, from track chains, 19 against 5 | `CW`, from track chains, 300 against 43 |
| trainable parameters | 57.1 M | 54.1 M | 51.1 M |
| peak GPU memory | 19.8 GB | 9.5–9.8 GB | 12.4–12.6 GB |
| speed on an L4 | 0.5–0.6 s per step | about 0.55 s per step | about 3,500 steps per 33-minute loop, as on PHerc. 1203 |
| GPU time | about 8 L4-hours with gates | 5.3 L4-hours | running |

- The own umbilicus of PHerc. 0175A lies inside the innermost 1–3 wraps on all 16 slices, but about 80–160 voxels off the visual centre on three of them, along the long axis of the flattened core.
- **Gate, fixed before any ink.** For every winding inside the scroll (CT found on ≥ 90 % of its profiles), 200 random vertices; fraction on an m7 sheet at the vertex, for the raw fit, against the same vertices shifted at random by up to ±8 voxels.
  Windings with a raw fraction above 0.40 go to ink.

**Controls and nulls.** The random-shift control at each gate. The awarded PHerc. 0343 winding on the same measure: 0.60 against 0.28. A CT z-score at the surface, which does not use m7.

**Numbers.** "on m7" = m7 > 127 at the vertex. "≤ 2" = nearest m7 sheet centre within 2 voxels. Each cell is fit / control.

| scroll, step | windings | raw on m7 | raw ≤ 2 | snapped on m7 | CT z raw | CT z snapped |
|---|---|---|---|---|---|---|
| 0175A, 1,737 | 119 | 0.202 / 0.194 | 0.272 / 0.266 | 0.651 / 0.218 | +0.010 / +0.013 | +0.167 / +0.034 |
| 0175A, 7,592 | 118 | 0.197 / 0.204 | 0.264 / 0.274 | 0.645 / 0.219 | +0.020 / +0.016 | +0.188 / +0.049 |
| 0175A, 13,738 | 118 | 0.269 / 0.201 | 0.348 / 0.271 | 0.653 / 0.224 | +0.042 / +0.020 | +0.189 / +0.051 |
| 0175A, 30,000 | 119 | **0.321 / 0.207** | 0.401 / 0.275 | 0.668 / 0.218 | +0.069 / +0.033 | +0.197 / +0.041 |
| 1203, 2,725 | 62–63 | 0.224 / 0.236 | 0.272 / 0.281 | 0.602 / 0.248 | +0.017 / +0.011 | +0.116 / +0.019 |
| 1203, 6,320 | 66 | 0.232 / 0.234 | 0.276 / 0.280 | 0.589 / 0.251 | +0.001 / −0.000 | +0.130 / +0.004 |
| 1203, 13,383 | 69 | 0.318 / 0.235 | 0.368 / 0.276 | not recorded | +0.025 / +0.007 | not recorded |
| 1203, 30,000 | 65 | **0.344 / 0.237** | 0.401 / 0.282 | 0.579 / 0.244 | +0.039 / +0.010 | +0.121 / +0.026 |
| 0846A, 6,269 | not recorded | 0.192 / 0.188 | 0.244 / 0.244 | 0.588 / 0.223 | +0.004 / +0.008 | +0.106 / not recorded |
| 0846A, 13,051 | 106 | 0.247 / 0.188 | 0.312 / 0.245 | not recorded | +0.016 / +0.008 | not recorded |
| 0846A, 30,000 | — | not measured yet | | | | |

| at 30,000 steps | PHerc. 0175A | PHerc. 1203 |
|---|---|---|
| CT peak within 2 voxels, raw | 0.401 / 0.385 | 0.409 / 0.404 |
| CT peak within 2 voxels, snapped | 0.441 / 0.383 | 0.449 / 0.413 |
| raw on m7 by winding block, innermost to outermost | 0.23, 0.29, 0.36, 0.41 (blocks of 30) | 0.367, 0.329, 0.341, 0.322 (blocks of 20; controls 0.197, 0.254, 0.259, 0.237) |
| windings above 0.40 raw | 53 (520.1 cm² nominal) | 17 (64.4 cm² nominal) |
| windings above 0.50 raw | 23 | not recorded |
| best winding | 0.685 | 0.66 |
| windings inside the scroll / exported | 119 / 120, 1,006 cm² | 65 / 120 |

- Satisfied tracks on PHerc. 0175A: 22.4 % at step 1,737, 17.6 % at 7,592, 21.6 % at 10,781, 24.6 % at 13,738. The fraction fell while the loss fell, then rose after the track distance-transform stage began.
- Between steps 1,737 and 7,592 the mean stayed at chance but the per-winding spread widened (p10 / p50 / p90 0.06 / 0.16 / 0.38 against 0.15 / 0.21 / 0.26 for the control): whole windings lock on or between sheets.
- On PHerc. 1203 the fitter exports 120 windings but the scroll has only about 65 turns in the band; the outer ones run outside the papyrus.
- A published note gives "the best reach 0.68" for PHerc. 0175A; the table value is 0.685 and one private table rounds it to 0.69. All three are the same number.

**Verdict.** Worked in part. The raw fit leaves chance only after the track distance-transform stage, so the step count mattered.
About half the windings of PHerc. 0175A and a quarter of those of PHerc. 1203 follow a sheet over much of their length; the rest run between sheets.

**How this could be wrong or miss ink.**
- The control shift is a fixed ±8 voxels. Sheets are 5–8 voxels apart in compressed zones, so the control level (0.19–0.25) depends on sheet density, and a fit locked half a pitch off scores below it.
- The threshold 0.40 has no derivation. 66 windings of PHerc. 0175A and 48 of PHerc. 1203 inside the scroll were not inked because of it.
- The m7-independent evidence is thin: CT peak within 2 voxels improves by 0.02 raw and 0.06 snapped; even the awarded winding separates only 0.60 from 0.40.
- Sheet identity along a winding was never verified. A winding can switch to the neighbouring sheet and still score as "on a sheet".
- PHerc. 0175A: the sense is assumed, the umbilicus is ours and visibly off-centre on some slices, and the normals are synthetic. Any of the three could be wrong without the gate showing it.
- PHerc. 1203 and 0846A use tracks made by us (S6).
- The windings are not isometric (S8).

**Not done.** The sense test on PHerc. 0175A. A second band on any scroll. The 66 lower windings of PHerc. 0175A. A fit of the other sense at full schedule for comparison.

**Where the code is.** `code/spiral_oncheck.py`.

---

<a id="s5"></a>
## S5 — Winding sense from chains of tracks

**2026-11-23 follow-up, Q12.** The historical results below are preserved, but the
original diagnostic was found to depend on point-storage and crossing-row order.
A real public PHerc0175A replay reproduced the defect; corrected-v2 passed four
frozen storage-invariance checks. An independent tapered-spiral counterexample
also shows that mirrored drift signs can agree on the wrong physical sense when
the chain changes z. The old sign counts below need replay before use as sense
evidence. The current official catalog derives ACW for PHerc0175A, which supports
its existing setting independently of this chain test. See
[the executed audit](../winding_audit/RESULTS.md) for code, measurements and limits.

**Question.** Viewed from low z with x right and y down, does the radius grow clockwise (`CW` in the fitter) or anticlockwise (`ACW`)?

**What was done.**
- Convention read from the pinned code: radius growing with image angle θ = atan2(y − cy, x − cx) is `CW`; `ACW` mirrors x.
- No single transverse track spans a full turn (longest 355° on PHerc. 0358), so tracks are chained: start on a transverse track near a chosen z, walk towards increasing θ, and at the front end hop along a crossing longitudinal track to another transverse track (|Δz| ≤ 16, |Δθ| < 0.05 rad, |Δr| < 20 voxels), until 2π is accumulated.
- dr = radius after one full turn minus the start radius. Only chains that end within one plausible pitch (5 < |dr| < 60 voxels) are counted.
- The same walk is run on the x-mirrored geometry. A real sense signal must flip sign there; a drift bias of the walk does not.

**Controls and nulls.** The mirrored run. On PHerc. 0358 the raw sign counts did not flip under mirroring (the walk has a negative-drift bias), but the pitch-band chains did.

**Numbers.**

| scroll | full turns (plain / mirrored) | pitch-band chains, plain: dr > 0 / dr < 0 | mirrored: dr > 0 / dr < 0 | combined | decision |
|---|---|---|---|---|---|
| PHerc. 0358 | 128 / 95 | 10 / 20 | 13 / 5 | 33 `ACW` against 15 `CW` | `ACW`, moderate support |
| PHerc. 1203 | 51 / 23 | 14 / 2 | 3 / 5 | 19 `CW` against 5 `ACW` | `CW`, moderate support |
| PHerc. 0846A | 474 / 382 | 140 / 25 | 18 / 160 | 300 `CW` against 43 `ACW` | `CW`, strong support |
| PHerc. 0175A | not run | | | | `ACW` assumed |

- Normal sign that follows: `CW` fits export normals pointing away from the umbilicus (outward fraction 0.99), `ACW` fits towards it (0.004–0.02).
- Tracing one predicted sheet round a full turn on a slice of PHerc. 0358 never closed (≤ 1 % of walks reached 180°).

**Verdict.** Inconclusive on PHerc. 0358, moderate on PHerc. 1203, strong on PHerc. 0846A, untested on PHerc. 0175A.

**How this could be wrong or miss ink.** Chains share tracks, so the counts are not independent and no p-value is quoted. On a flattened core almost no chain closes (14 of 24,000 starts in the first PHerc. 1203 run). The pitch band 5–60 voxels was chosen by us.

**Not done.** The test on PHerc. 0175A, although its tracks and crossings are published. A fit of each sense at full schedule.

**Where the code is.** `code/sense_chain.py`.

---

<a id="s6"></a>
## S6 — Making our own surface tracks

**Question.** PHerc. 1203 has no published tracks, and the published store of PHerc. 0846A starts at z 4500, above the regular part of its fine band. Can the pinned extractor reproduce the organisers' tracks?

**What was done.**
- Extractor: `extract_surface_tracks.py` of the pinned villa commit, CPU only.
- Parameters copied from the `.extract.json` published beside two existing stores (identical in both): m7 level 0 at threshold 0.2, downsample 2, horizontal chunk depth 2 and stride 8, vertical chunk thickness 4 and stride 16, dust threshold 400, maximum area 640,000, minimum path 10 vertices, path mode `interjoint`.
- The module's own defaults differ (downsample 4, coarser strides) and were overridden.
- Differences we could not remove: our vertical slabs span only our band, not the whole scroll; our slab origin comes from 20 sampled slices of the band; we run one worker per process; the organisers used a later revision of the script, with the same per-slab algorithm.

**Controls and nulls.** Validation on PHerc. 0175A, z 7604–7804, our extraction with the published slab origin against the published store:

| | ours | published |
|---|---|---|
| horizontal tracks in the 25 ribbons | 80,747 | 80,747 |
| horizontal points | 12,942,692 | 12,942,692 |
| points per track, p10 / p50 / p90 / p99 | 11 / 47 / 413 / 1,606 | identical |
| arc length, p10 / p50 / p90 (voxels) | 25.3 / 109.5 / 983.7 | identical |
| vertical tracks touching the central 100 slices | 221,074 | 175,510 |
| vertical points inside the central 100 slices | 4,540,656 | 4,359,575 |
| per-slab vertical point counts, correlation | 0.991 and 0.988 | |
| vertical points per track, p50 / p90 | 25 / 70 | 33 / 411 |

**Numbers.** PHerc. 1203: 1,396,662 tracks, 62,803,105 points for z 8936–10464, 25 minutes. PHerc. 0846A: 1,690,832 tracks, 93,224,175 points for z 2386–3814, 23 minutes.
A published note says the PHerc. 1203 extraction ran on 11 workers; the private log says 8 workers for PHerc. 1203 and 11 for the validation and for PHerc. 0846A. The private log is right.

**Verdict.** Worked. Horizontal tracks are identical; vertical tracks differ as expected from the band restriction (+4 % points, more and shorter tracks).

**How this could be wrong or miss ink.** Shorter vertical tracks may weaken the fit; we have no fit of one band with both stores to measure it. Tracks inherit every error of m7.

**Not done.** A fit of the PHerc. 0175A band with our tracks instead of the published ones.

**Where the code is.** `code/cmp_tracks.py` (the comparison). The extractor is the organisers' code.

---

<a id="s7"></a>
## S7 — Snapping a surface to the predicted sheet

**Question.** The three winners snapped each winding onto the m7 sheet before inking. Should we?

**What was done.** `snap_meshes.py` from the PHerc. 0343 winner's public repository, default settings, m7 level 0, CPU. Applied to our spiral fits, to three atlas windings, and to a few early scouting patches.

**Controls and nulls.** The random-shift control of S3/S4. The CT z-score, which does not use m7. A teammate's audit of the snap on a PHerc. 0813 winding.

**Numbers.**
- PHerc. 0358 fit: on a sheet 0.23 raw → 0.48 snapped (control 0.25); rejected fraction 0.21; median |offset| 2.5 voxels, p90 6.7.
- PHerc. 0175A fit: 0.321 → 0.668 (control 0.218); CT z +0.069 → +0.197.
- PHerc. 1203 fit: 0.344 → 0.579 (control 0.244); CT z +0.039 → +0.121.
- Atlas windings (S2): median snap offset 1.6–4.0 voxels.
- Teammate's audit (`collab/codex-008`): the snap moved the surface to a **different** CT peak in 9 of 23 profiles.
- Early scouting patches: re-scoring 11.9 cm² of snapped variants changed the best triage score by ≤ 0.4 and never moved a render across a decision boundary.
- Cost: 25 s for 3 windings, 117 s for 120 windings on 6 processes.

**Verdict.** Inconclusive. The snap moves the surface onto the prediction, which is what it aims at, so the m7 column is partly circular. The independent CT measure improves modestly.
Decision taken: the atlas was inked unsnapped at two depth shifts; the spiral fits were inked snapped.

**How this could be wrong or miss ink.**
- A snap that jumps to the neighbouring sheet splits a line of text between two sheets at the jump.
- The snapped PHerc. 0175A windings score higher on m7 (0.668) than the awarded PHerc. 0343 winding (0.60) while showing visibly more cut-through. The measure does not rank surface quality.
- The atlas decision rests on reader-v3 tolerating a smooth injected offset on one positive (D8); real offsets are not smooth.

**Not done.** Ink on the same winding raw and snapped (planned twice, cancelled twice for GPU time). A check of sheet identity before and after the snap.

**Where the code is.** `code/spiral_oncheck.py` for the measure; the snap script is the winner's public code.

---

<a id="s8"></a>
## S8 — Fitted windings are not isometric

**Question.** A spiral-fit winding is a grid with rows along z and columns at equal steps in angle. Is a grid cell 20 × 20 voxels, as the renderer assumes?

**What was done.** Measured the 3-D distance between neighbouring vertices along rows. Wrote a resampler: one common column map (median over rows of the cumulative arc length) and one common row map applied to the whole grid, target spacing 20 voxels.

**Controls and nulls.** A per-row variant, which removes the residual scale error, was tried and rejected because it shears the grid.

**Numbers.**
- Spacing along a row, 5th–95th percentile: 11–60 voxels on PHerc. 1203; 12–41 on PHerc. 0175A (measured by the coordinator on six snapped windings across the fit: column spacing 5th/50th/95th percentile 11.5–15.1 / 17.6–21.4 / 35.9–44.5 voxels, row spacing 17–18 / 22–23 / 28–37; the median is near 20 because the column count grows with the winding, the spread is local).
- A renderer that assumes 20 voxels per cell stretches or squeezes the image along the lines by 0.6–2×, up to 3× on PHerc. 1203.
- Areas computed as vertex count × cell size are wrong by the same factor: one PHerc. 1203 winding is 1.99 cm² nominal and 2.80 cm² true.
- After the common-map resampling: median spacing 20.0 voxels, residual local scale within a row 0.5–1.7× (5th–95th percentile).

**Verdict.** They are not isometric. The 2.4 µm reads of PHerc. 1203 use the resampling. The 9 µm screen of PHerc. 0175A (520 cm²) was rendered **without** it.

**How this could be wrong or miss ink.**
- Letters stretched or squeezed by up to 2× change shape and scale for the detectors. reader-v3 tolerates ±20 % isotropic scale error on PHerc. 0841 (D1); anisotropic errors of this size were not tested.
- The awarded PHerc. 0343 windings read through the uncorrected path, but their own spacing distribution was not measured; they may simply be more regular.
- The "valid = chart" areas quoted for PHerc. 0175A are nominal.
- A residual local scale of 0.5–1.7× remains after correction.

**Not done.** Re-rendering PHerc. 0175A isometrically. Measuring the spacing of the PHerc. 0343 windings. A stretch test on a positive.

**Added 2026-11-23 (stretch of the windings that were read, and two stretch tests by another agent).**
- All 53 PHerc. 0175A windings that were read, snapped meshes, spacing between neighbouring vertices divided by the 20 voxels the renderer assumed; median over windings of each winding's percentiles, with the range over windings in brackets.

| direction | 5th | 50th | 95th | share of cells off by more than 1.5× |
|---|---|---|---|---|
| along the turn | 0.61 (0.55–0.76) | 1.04 (0.87–1.07) | 2.12 (1.60–2.24) | 0.26 (0.17–0.38) |
| along the scroll axis | 0.88 (0.85–0.90) | 1.15 (1.09–1.17) | 1.82 (1.42–1.99) | 0.14 (0.03–0.17) |

- A ratio above 1 means the papyrus was compressed in the render by that factor, below 1 that it was enlarged. Pooled over the fit, half of the cells along the turn are off by more than 1.25×.
- True area over nominal area: 1.29 (1.03–1.44). The 520 cm² quoted for this screen are about 670 cm² of surface.
- Two tests on known ink now exist (`studio_scale_stress/RESULTS.md`, `collab/chatgpt-adaptive-046.md`). Studio on PHerc. 0841: 2× compression costs 6 AP points along one image axis and 14 along the other, with recall at 0.5 falling from 0.68 to 0.58 and 0.42. A-hn on PHerc. 0306B: 2× enlargement along the line drops recall at 0.5 from 0.33 to 0.07; 0.6× raised it, with more flags on untraced papyrus.
- Reading: on about a quarter of the PHerc. 0175A surface the detectors worked at a scale where they lose a substantial part of known ink. That screen is a weak negative on that quarter, independent of the on-sheet question.

**Where the code is.** `code/isometric_resample.py`.

---

<a id="s9"></a>
## S9 — Re-centring a render from its own depth profile: failed, dropped

**Question.** The atlas renders sit several voxels off the sheet. Can the offset be estimated from the 31-layer stack itself and removed without the mesh or m7?

**What was done.**
- Estimator: 8 × 8 pixel block means of the depth profile; lateral Gaussian 24 px; depth Gaussian 1.2 layers; strongest local maximum within ±7 layers of the expected sheet layer, sub-voxel; reject blocks with low prominence or more than 3 voxels from the 7 × 7 median; fill by normalised convolution; Gaussian 0.5 mm; clip at ±7; two passes (estimate, re-centre, estimate the residual).
- Test: from the correctly placed 41-layer render of each positive, cut a 31-layer stack with a known smooth offset d(x, y) = 1 + 5·sin·cos with a 12 mm period (p1–p99 −3.9…+5.9 voxels), then re-centre it from the stack alone.

**Controls and nulls.** Gate fixed before the numbers were read: ≥ 90 % of the area within ±1 voxel of the injected offset. The correct render re-centred the same way, to see whether the correction harms a good render.

**Numbers.**

| positive | bias, CT peak against mesh (voxels) | error against truth, median / p90 | within ±1 | within ±2 |
|---|---|---|---|---|
| PHerc. 0343 windings | −1.56 | 2.07 / 5.2 | 24.5 % | 48 % |
| PHerc. 0306B crop | +1.33 | 1.81 / 4.9 | 30.5 % | 54 % |

- On three PHerc. 0813 atlas windings the estimated field correlates weakly with the snap offsets (r 0.16, 0.23, 0.10); agreement within 2 voxels 30, 28, 43 %, about the same as for the wrong sign (25, 25, 40 %).
- Ink is the decisive test (reader-v3, reading order, PHerc. 0343 letter boxes):

| 31-layer stack | AUC at shift −2 / 0 / +2 | line finder, best chain |
|---|---|---|
| correct | 0.682 / 0.684 / 0.675 | 10.91 |
| displaced by the known field, inked as is | 0.679 / 0.685 / 0.681 | 11.10 |
| displaced, then re-centred | 0.659 / 0.669 / 0.691 | 9.95 |
| correct, then re-centred | 0.651 / 0.666 / 0.687 | 9.94 |

**Verdict.** Negative. The estimator re-finds its own answer but the CT intensity peak is not where the winners' snapped mesh is, and the bias has opposite signs on the two positives. Dropped.
Production choice that follows: ink the published stacks unmoved at shifts −3 and +3 layers, keep both maps and their mean, take no maximum.

**How this could be wrong or miss ink.**
- "A smooth displacement of a few voxels costs reader-v3 nothing" is one positive, one detector, one synthetic field. Real misplacement includes jumps between sheets.
- The mean of two shifted maps dilutes a response present at one shift only. Ranking used the mean.
- A teammate's separate estimator (`firstletters-colab/depth_recentering/`) was not compared with ours on the same stacks.

**Not done.** The same test with Studio. A test with a discontinuous injected offset. Ranking on each shift separately.

**Where the code is.** `code/depth_recentre_test.py`.
