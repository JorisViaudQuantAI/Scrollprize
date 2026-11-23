# Methods ledger: everything this lane tried on the Vesuvius Challenge, with numbers, verdicts and weak points

Status: 2026-11-23. Author: the Claude lane of this repository (`firstletters-colab/`).
No letters have been found by this lane on any unread scroll. Nothing in this folder is a reading or a claim.

## Purpose

This ledger lists every approach the lane has tried, in a form that another agent or a person can check.
It is written for a reader who was not in our sessions.
It exists so that others can re-test pieces, propose other ideas, and above all find flaws that could have made us miss ink.
The file that matters most is [09_open_questions.md](09_open_questions.md): it lists where we think our methods can be wrong or blind, each as a test a reviewer can run.

## How to read it

- Each family of approaches has one file. Each entry in a file has the same eight headings:
  **Question · What was done · Controls and nulls · Numbers · Verdict · How this could be wrong or miss ink · Not done · Where the code is.**
- Entry ids are stable (`S` surfaces, `D` detectors, `C` controls, `R` review, `F` false leads, `L` label-free and physics,
  `H` fine resolution, `T` title lane, `N` coverage, `K` compute, `Q` open questions). Refer to them in `collab/` notes.
- Numbers come from our private lane reports and from the notes already published in `collab/claude-001…017.md`.
  Where a private report and a published note differ, the entry says which is right and why.
  Where a number was not measured, the entry says "not measured".
- `code/` holds 15 short scripts, sanitised, for checking our arithmetic. They are not a packaged tool. See [code/README.md](code/README.md).

### Words used with a fixed meaning

| word | meaning here |
|---|---|
| unread scroll | one of the scrolls still eligible for the First Letters prize; nothing on it has been read |
| positive, control | a public surface with known text, used to check a pipeline; never an unread scroll |
| layer, plane | one image of a rendered stack; one layer = one native voxel along the mesh normal unless stated |
| shift | displacement of the detector's depth window, in layers (9 µm lane) or in fine voxels (2.4 µm lane) |
| order, face | the direction in which the stack of layers is fed to a detector; reversing it reads the sheet "from the other side" |
| reading order | the order in which a detector shows the known text of a positive |
| chart area | area of the mesh grid (vertex count × cell size); for spiral windings this is nominal, see S8 |
| valid area | rendered pixels that contain CT data |
| on-sheet area | valid area that passes an on-sheet test; two different tests are used (S1 for own windows, L2 for windings); both are upper bounds |
| hot share | share of pixels with a per-map z-score above 2 |
| blob, cloud | a detector response without stroke shape: round, 1–3 mm (blob) or diffuse, 5–20 mm (cloud) |

### Verdict words

For unread material we use only three verdicts, as agreed in `collab/claude-004.md`:
`uninformative` (the input cannot support a conclusion), `screened — no qualifying reading identified` (under this pipeline), and `candidate under private review`.
For methods the index uses: **worked**, **negative**, **inconclusive**, **not run**.

## Sanitisation statement

This repository is public. The ledger was written under these rules, and a reviewer who finds a breach should say so at once.

1. No personal data: no names of the repository owner, no e-mail addresses, no account names of ours.
2. No tokens, no private storage identifiers or links, no identifiers of compute sessions or virtual machines, no local absolute paths.
3. No location of anything that was ever flagged as possibly ink on an unread scroll, even though every such place is now closed.
   Flagged places are described only as "one place on scroll X": no mesh name, winding id, seed, coordinates, angular position or narrow z-range.
   Surface sets are described at the level already published in `collab/` (scroll, number of windows or windings, z-band of a whole fit, areas).
4. PHerc. Paris 4 (Scroll 1) is a read scroll. The place of its ghost text is given in general terms only, without pixel coordinates.
5. No images of unread material. This folder contains no images at all.
6. Public third-party resources (model repositories, the atlas of published meshes, the organisers' open-data bucket) are named by their public names, because a method cannot be checked without them.
7. Winding ids appear only for public controls: the awarded PHerc. 0343 windings (w047, w048), the public PHerc. 1447 windings used as a pitch control, and the published multi-turn segment of PHerc. Paris 4.

The scripts in `code/` were rewritten where needed so that they contain no path, no crop of unread material and no target name.

## Index

Areas are in cm². AUC values on PHerc. 0343 and PHerc. 0306B are box-diluted (see C2); they are not comparable with the per-pixel AUCs on PHerc. 0841.

### Surfaces — [01_surfaces.md](01_surfaces.md)

| id | question | data | control(s) | headline number(s) | verdict | entry |
|---|---|---|---|---|---|---|
| S1 | Do patch-grown windows give single-sheet surface cheaply, and how is "on one sheet" tested? | 39 own windows on 10 unread scrolls, plus 9 earlier ones | awarded PHerc. 0343 winding through the same test: 0.73 | of 280 valid cm² screened, 151 on a predicted sheet and 98 in connected cores | worked in part (half the area) | [S1](01_surfaces.md#s1) |
| S2 | Can the published atlas of spiral-fit windings be inked as published? | 250 windings, 7 scrolls, 1,396 chart cm² | renderer identity on one mesh; two public positives for the face rule | about a third of blocks within 2 voxels of the CT peak; all 250 get one face label | worked with stated limits | [S2](01_surfaces.md#s2) |
| S3 | Does a spiral fit with a shortened schedule reach the sheet? | PHerc. 0358, 1,000–7,000 steps | same vertices shifted at random by ±8 voxels | raw 0.23 on a sheet against 0.25 control; 0.286 against about 0.25 after a complete 7,000-step schedule | negative | [S3](01_surfaces.md#s3) |
| S4 | Does the winners' 30,000-step schedule reach the sheet on a T4/L4 budget? | PHerc. 0175A, 1203, 0846A (running) | random-shift control at four step counts | raw 0.321 against 0.207 (0175A); 0.344 against 0.237 (1203); 53 and 17 windings above 0.40 | worked in part | [S4](01_surfaces.md#s4) |
| S5 | Which way does the spiral wind? | track chains on 0358, 1203, 0846A | the same walk on the mirrored geometry | 33:15, 19:5, 300:43 | inconclusive to strong, by scroll; not tested on 0175A | [S5](01_surfaces.md#s5) |
| S6 | Can we make surface tracks for scrolls that have none? | PHerc. 1203, 0846A | published track store of PHerc. 0175A, 200 slices | horizontal tracks identical (80,747 tracks, 12,942,692 points) | worked | [S6](01_surfaces.md#s6) |
| S7 | Should surfaces be snapped to the predicted sheet before inking? | 0358, 0813, 0175A, 1203 | random-shift control; CT z-score, which does not use the prediction | 0.668 against 0.218 on a sheet after the snap, CT z +0.197 against +0.041 | inconclusive (partly circular) | [S7](01_surfaces.md#s7) |
| S8 | Are fitted windings isometric? | 0175A and 1203 fits | nominal 20-voxel spacing | 11–60 voxels between neighbours; one winding 1.99 nominal, 2.80 true cm² | negative (they are not) | [S8](01_surfaces.md#s8) |
| S9 | Can a render be re-centred on the sheet from its own depth profile? | 0343 and 0306B positives with an injected offset | gate fixed first: 90 % of area within ±1 voxel | 24.5 % and 30.5 % | negative, dropped | [S9](01_surfaces.md#s9) |

### Detectors applied zero-shot — [02_detectors.md](02_detectors.md)

| id | question | data | control(s) | headline number(s) | verdict | entry |
|---|---|---|---|---|---|---|
| D1 | Does reader-v3 run correctly here and read known text? | 0841, 0343, 0306B, Scroll 1 | published maps (pixel parity); reversed order; off-sheet | parity within 1/255; AUC 0.863 forward, 0.54–0.57 reversed | worked | [D1](02_detectors.md#d1) |
| D2 | Is `ink_9um` useful beside reader-v3? | 0841 | same | AUC 0.72–0.75; adding it lowers reader-v3 from 0.863 to 0.840 | negative as a screen | [D2](02_detectors.md#d2) |
| D3 | Can Hecate be used on a small GPU? | 0841, Scroll 1 | fp32 output | fp16 5.7× faster than bf16 on a T4, output within 1/255 | worked; used for follow-ups only | [D3](02_detectors.md#d3) |
| D4 | Does the Studio detector run on a T4/L4, and which planes does it need? | 0841, 0343, 0306B | other plane order; depth shifts; tile origin | 0.916–0.924 on 0841 (training donor); 0.745 on 0343 boxes; tile origin changes stride-96 maps (r 0.888) | worked; shapes only at stride 48 | [D4](02_detectors.md#d4) |
| D5 | What do A-hn and TRACE add? | published 0306B maps; teammates' runs | per-row evaluation | no model shows the second 0306B line as a line | not run by us | [D5](02_detectors.md#d5) |
| D6 | Do the 2.4 µm models run on our renders? | 0841 at 2.4 µm | published map; reversed order | r 0.929 with the published map; AUC 0.975 / 0.930 | worked (path check only) | [D6](02_detectors.md#d6) |
| D7 | Which layer order reads, for each renderer? | three positives, four render paths | wrong order | table of conventions | worked | [D7](02_detectors.md#d7) |
| D8 | How much depth error does each detector tolerate? | 0841, 0343, 0306B | — | reader-v3 letters visible −6…+6 layers; Studio 0.92 → 0.81 at ±6 | worked, on few positives | [D8](02_detectors.md#d8) |
| D9 | Do combinations of detectors help? | 0841 | — | best of 305 combinations 0.891 (post hoc) | inconclusive | [D9](02_detectors.md#d9) |

### Controls and calibration — [03_controls.md](03_controls.md)

| id | question | data | control(s) | headline number(s) | verdict | entry |
|---|---|---|---|---|---|---|
| C1 | PHerc. 0841 w00 | 2.6 cm² at 9.37 µm; 1.04 cm² labelled at 2.4 µm | answer key is a model output | see D1–D6 | usable for input-path checks | [C1](03_controls.md#c1) |
| C2 | PHerc. 0343 awarded windings | 8.09 cm², 18 letter boxes | wrong order still 0.60 | reader-v3 0.684, Studio 0.745 | usable; exposure not settled | [C2](03_controls.md#c2) |
| C3 | PHerc. 0306B claim crop | 7.66 cm², two published lines | rows were mislabelled once, corrected | line 1 AUC 0.65–0.71; line 2 0.54–0.63 | usable; the least exposed | [C3](03_controls.md#c3) |
| C4 | Scroll 1 final columns | 2.4 µm surface volume | every detector trained on Scroll 1 | 0.90–0.95 | input path only | [C4](03_controls.md#c4) |
| C5 | PHerc. 172 | 14.6 cm² at 7.91 µm | key is a model output | label-free 0.64 | "ink visible in CT" case only | [C5](03_controls.md#c5) |
| C6 | PHerc. 1447 | three public windings, 69 cm² | fewer than 5 lines of text | — | sanity control only | [C6](03_controls.md#c6) |
| C7 | Which detector saw which scroll in training? | public model cards and configs | — | exposure matrix | — | [C7](03_controls.md#c7) |

### Review procedure and false leads — [04_review_and_false_leads.md](04_review_and_false_leads.md)

| id | question | data | control(s) | headline number(s) | verdict | entry |
|---|---|---|---|---|---|---|
| R1 | Does a row score rank text windows? | 0841 | 210 control windows | dense text ≥ 3.70, controls ≤ 3.07, sparse text 1.3–2.0 | worked for dense rows only | [R1](04_review_and_false_leads.md#r1) |
| R2 | Does the 0306B line finder transfer to our maps? | 0343, 0306B | wrong face | 8.9–10.9 on text, 6.0–6.7 wrong face, atlas maximum 8.60 | worked for ordering | [R2](04_review_and_false_leads.md#r2) |
| R3 | Does the `line_review` helper rank known lines first? | 0343, 0306B | — | top proposals on the known rows | worked for ordering | [R3](04_review_and_false_leads.md#r3) |
| R4 | What is our bar for "candidate", and how is the eye review done? | — | positive rendered the same way beside each view | four conditions | may be too strict (see Q) | [R4](04_review_and_false_leads.md#r4) |
| F1 | Rows of dots in Studio at stride 48 | one place on PHerc. 0125 | 12 random windings, 54 cm² | 0.11 rows per cm² in one order; strokeness 0.6–1.3 against 1.88 | closed, not a reading | [F1](04_review_and_false_leads.md#f1) |
| F2 | Rows of thin connected marks | one place on PHerc. 0846B | three other clean cores | row contrast 2.3; not in Hecate or `ink_9um` | closed, not a reading | [F2](04_review_and_false_leads.md#f2) |
| F3 | Isolated strokes on compressed layers, cracks, void edges | places on PHerc. 0813, 0191, 0175A | re-grown surface; CT sections; neighbours | re-grown mesh within 0.01 voxel (not independent) | closed, not readings | [F3](04_review_and_false_leads.md#f3) |
| F4 | Clouds in both orders or across turns | several scrolls | other order; adjacent turn | — | structure, not ink | [F4](04_review_and_false_leads.md#f4) |
| F5 | Ghost text from the neighbouring wrap | PHerc. Paris 4 | 19 shifted or mirrored nulls | r 0.86 against 0.25 ± 0.04 | closed: letter-for-letter match | [F5](04_review_and_false_leads.md#f5) |
| F6 | Face asymmetry of the response | PHerc. 0125 | Hecate (no asymmetry) | 2–3× in Studio and reader-v3, 2.2× in A-hn | origin undecided | [F6](04_review_and_false_leads.md#f6) |
| F7 | Edge rows, fibre bars, tile grids, a fine-scan cloud | several | — | — | artefact classes | [F7](04_review_and_false_leads.md#f7) |

### Label-free and physics ideas — [05_label_free_and_physics.md](05_label_free_and_physics.md)

| id | question | data | control(s) | headline number(s) | verdict | entry |
|---|---|---|---|---|---|---|
| L1 | Does any simple CT feature at the sheet surface separate ink from blank? | 0841, 0343, 0306B, PHerc. 172 | depth-shifted, other face, rolled labels | best 0.53–0.56 on the three; 0.64 on PHerc. 172 | negative | [L1](05_label_free_and_physics.md#l1) |
| L2 | Can a render alone say whether it is parallel to the sheet? | 250 atlas windings | 59 windings judged by the atlas author | AUC 0.83 | worked as an upper bound | [L2](05_label_free_and_physics.md#l2) |
| L3 | Does a tiny classifier transfer between controls? | same four | spatial-shift and face-swap nulls | 0.51–0.58 without geometry | negative | [L3](05_label_free_and_physics.md#l3) |
| L4 | Does the PHerc. 172 feature find anything on the atlas? | 250 windings, 1,309 valid cm² | same statistic on controls | ranks a 0841 window above the PHerc. 172 text row | uninformative | [L4](05_label_free_and_physics.md#l4) |
| L5 | Can line pitch be detected below legibility by folding? | Scroll 1, 1447, 0841 | wrong order, rotated, raw CT, blank | z 4.8 (4.2 corrected) where the eye fails; bar was 5 on two scrolls | NO-GO | [L5](05_label_free_and_physics.md#l5) |
| L6 | How big is the ink imprint in CT grey levels? | Scroll 1 at 2.4 µm | blank-papyrus noise | within ±3 grey; noise 21–33 grey | negative (≲ 0.3σ per pixel) | [L6](05_label_free_and_physics.md#l6) |
| L7 | Does ink add column mass? | Scroll 1 at 2.4 µm | rolled labels; window outside the sheet | AUC 0.516 against 0.462; +1.6 % | negative | [L7](05_label_free_and_physics.md#l7) |
| L8 | Can the 113/77 keV pair show lead? | scan metadata of 1203, 0846A | — | our two studies disagree: ×1.7 or 1/25 | not run; disagreement open | [L8](05_label_free_and_physics.md#l8) |
| L9 | Is the 0.22 m / 1.2 m pair a phase measurement? | scan metadata | — | no | ruled out on paper | [L9](05_label_free_and_physics.md#l9) |
| L10 | Can super-resolution or deconvolution recover fine detail? | edge of PHerc. 1203 in both scans | — | transfer 0.31 at 40 µm, about 1e-4 at 20 µm | ruled out on paper | [L10](05_label_free_and_physics.md#l10) |
| L11 | Should we train or adapt a detector ourselves? | public record | — | priors 2–10 % | not run | [L11](05_label_free_and_physics.md#l11) |

### Fine-resolution lane — [06_fine_resolution.md](06_fine_resolution.md)

| id | question | data | control(s) | headline number(s) | verdict | entry |
|---|---|---|---|---|---|---|
| H1 | Where is the 2.4 µm band in the 9.36 µm frame? | 1203, 0846A | leave-one-out landmarks | about 1.2 and 1.1 coarse voxels | worked near the centre | [H1](06_fine_resolution.md#h1) |
| H2 | Can a 9 µm mesh be read at 2.4 µm? | 0841 w00 | organisers' refined surface volume | 0.975 against 0.971 | worked (path only) | [H2](06_fine_resolution.md#h2) |
| H3 | How much surface error does the 2.4 µm read tolerate? | 0841 | — | 4 coarse voxels cost a third of the recall | measured | [H3](06_fine_resolution.md#h3) |
| H4 | Stride 128 or 192? | 0841; 1203 | — | 2.2× faster, AUC −0.003; blocky on blank papyrus | screening only | [H4](06_fine_resolution.md#h4) |
| H5 | Is there ink on PHerc. 1203 inside the fine band? | 39 public-mesh blocks; 17 fitted windings (1 complete) | 0841 text share 0.13–0.30 | 0.00–0.09 everywhere | screened, in progress | [H5](06_fine_resolution.md#h5) |

### Coverage, cost, open questions

| file | content |
|---|---|
| [07_negative_coverage.md](07_negative_coverage.md) | N: per scroll and surface set, areas, detectors, orders, verdict words; T1–T3: the PHerc. Paris 4 title lane |
| [08_compute_and_cost.md](08_compute_and_cost.md) | K: throughput per detector, fit timings, memory, data volumes |
| [09_open_questions.md](09_open_questions.md) | Q1–Q51: suspected weak points, each as a testable challenge |

## What a reviewer could do first

1. Read [09_open_questions.md](09_open_questions.md) and pick an item whose test fits your hardware.
2. Check the arithmetic of the criterion you doubt most with the matching script in `code/`.
3. Answer in `collab/` with the entry id, what you ran, and the number you got.
