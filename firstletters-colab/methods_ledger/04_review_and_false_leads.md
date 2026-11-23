# 04 — Review procedure and catalogue of false leads

Part 1 (R1–R4) is how maps are ranked and looked at. Part 2 (F1–F7) is every false lead, with its mechanism and the test that closed it.
All places on unread scrolls are closed. Their locations are not given, here or anywhere public.
The prize needs ten legible letters within 4 cm². Nothing we saw came near that.

---

<a id="r1"></a>
## R1 — Triage row score

**Question.** Can an automatic score rank windows by how text-like their ink map is?

**What was done.**
- Windows of 14 mm, step 1.5 mm. Components per window: row periodicity at a 4–7 mm line pitch along either fibre direction (`period`, `rowmod`, measured pitch); stroke statistics (`stroke`, `orient`, `cover`); stability (Dice of thresholded maps across depth shifts and across models); layer-order asymmetry.
- Score = z-scores of log period, rowmod, stroke, orient and cover against control windows, clipped, divided by √5.
- Calibration on PHerc. 0841 w00, reader-v3 at shift 0: text windows split by answer-key ink cover; controls = reversed order (30 windows), off-sheet ±20 layers (120), noise stack (60).
- Rule used for the first scans: score ≥ 3.1 and three gates (asymmetry ≥ +0.5, depth Dice ≥ 0.7, Dice with `ink_9um` ≥ 0.3) → label "candidate"; 2.0–3.1 → "look".
- Later changes: the asymmetry gate was dropped from the label and kept as a recorded number, because reversal is not a veto (D7). A teammate's patch made the row pitch physical across gaps. The cutoffs are no longer quoted as decisions.

**Controls and nulls.** The 210 control windows above.

**Numbers.**

| window class | n | score p10 / p50 / p90 / max |
|---|---|---|
| text, top quartile of ink cover | 8 | 3.75 / 4.10 / 4.21 / 4.29 |
| text, middle | 14 | 2.02 / 2.76 / 3.59 / 3.86 |
| text, bottom quartile | 8 | 1.28 / 1.52 / 2.03 / 2.05 |
| reversed order | 30 | −0.44 / 0.14 / 0.89 / 1.29 |
| off-sheet | 120 | −0.34 / 1.13 / 2.38 / 3.07 |
| noise | 60 | −2.99 / −2.07 / −1.59 / −1.15 |

- First scan of 18 renders on 6 unread scrolls: one window passed all gates, marginally (3.46), and shows blobs by eye; its deep read with four models failed the cross-model and order checks.
- Nine pre-protocol windows: best 3.01, failing the asymmetry gate.
- On Studio maps the score puts more windows in the "look" band, because Studio maps are denser; it was calibrated for reader-v3.

**Verdict.** Worked for dense rows of text only. Sparse text scores 1.3–2.0, inside the control range.

**How this could be wrong or miss ink.**
- A single line, or isolated letters, is invisible to this score. A First Letters claim can be one line.
- The only compute decision ever tied to it: below 2.0 a window got no second model and no depth sweep in the first scans.
- Windows overlap heavily and come from one segment; the effective sample is small.
- The line-pitch prior of 4–7 mm comes from read scrolls.

**Not done.** Calibration on a second scroll.

**Where the code is.** `firstletters-colab/controls/score_text.py` (already public).

---

<a id="r2"></a>
## R2 — The PHerc. 0306B winners' line finder and glyph model on our maps

**Question.** Does the single-line finder that ranked the winning line first in its own project keep its calibration on our maps?

**What was done.** `line_find.py` and the 2.6 MB glyph model from the winners' public repository, as released, on our z-scored maps at the repository's sampling. Used for **ordering only**.

**Controls and nulls.** The two positives in the wrong order. The repository's own calibration: known text lines 8.2–11.2; reviewed null lanes at most 6.8–7.3.

**Numbers.** Best chain per map.

| map | score |
|---|---|
| PHerc. 0343, reader-v3, reading order / other order | 10.91 / 5.95 |
| PHerc. 0306B, reader-v3, reading order / other order | 8.93 / 6.73 |
| atlas PHerc. 0813 / 0800 / 0125, maximum over all windings | 7.60 / 7.91 / 8.60 |
| atlas small scrolls, maximum | 7.22 |
| PHerc. 0175A fitted windings, sheet-parallel tiles only, maximum | 5.67 |
| PHerc. 0343 on the same sheet-parallel mask, reading / other order | 7.3–8.1 / 1.5–1.6 |
| same known letters in Studio at stride 96 / stride 48 | 3.82 / 9.48 |

- The two atlas scores inside the text range (8.60 and 8.50, both on PHerc. 0125) are round blobs beside a streak and round blobs at a mesh edge.
- In the own-window tranche a geometric variant without the glyph model was used: positive 3.95, windows up to 5.2. The scores overlap, so they only order places.

**Verdict.** Worked for ordering. Never used to dismiss a surface.

**How this could be wrong or miss ink.** Its calibration was made on the winners' own fused maps. Masking tiles cuts chains and lowers every score (10.9 → 8.1 on the same positive). On Studio maps at stride 96 it is unstable.

**Not done.** Recalibration of the glyph threshold on our maps.

**Where the code is.** The winners' public repository.

---

<a id="r3"></a>
## R3 — The `line_review` helper

**Question.** Does a teammate's single-line proposal tool rank known lines first, and does it find places the other finder misses?

**What was done.** `firstletters-colab/line_review/review.py`, defaults frozen, on reader-v3 maps of the positives and of the first ten PHerc. 0813 atlas windings. Later on every window of the tranche. For the rest of PHerc. 0813 and 0800 only a degraded run was possible, on the JPEG montages, because the float maps were lost with a virtual machine.

**Controls and nulls.** The positives.

**Numbers.** PHerc. 0343: the first two proposals lie on the known row (5.57, 5.22). PHerc. 0306B: the first lies on line 2 (4.55), proposals 2–4 on line 1. First ten atlas windings: top scores 0.88–4.27; the two places it ranked that the other finder had not are speckle and fibre streaks at 20 px/mm. Montage run: positive 5.77, atlas maximum 5.15; all top 20 places are blob strings, clouds or crack chains. Tranche: positive 5.34, windows up to 5.4.

**Verdict.** Worked for ordering.

**How this could be wrong or miss ink.** The montage run is 2.7× coarser than intended and its scores are not comparable with the others.

**Not done.** A proper run on full-resolution maps of PHerc. 0813 and 0800 (needs a re-ink, about 2 L4-hours).

**Where the code is.** `firstletters-colab/line_review/` (teammate's tool, already public).

---

<a id="r4"></a>
## R4 — Eye review and the bar for "candidate"

**Question.** Who looks at what, at which resolution, and what would make us call a place a candidate?

**What was done.**
- Every inked winding or window was opened at least once as a montage (CT and ink overlay), in every order that was computed.
- Top-ranked places of each ranker were opened at 20 px/mm (10 px/mm for PHerc. Paris 4 tiles, 104 px/mm for its top windows), each beside the PHerc. 0343 positive rendered and inked the same way.
- The reviewer is the agent itself, looking at images. No papyrologist has looked at any of our maps.
- A place that shows stroke shapes goes through a fixed sequence: the other detector family; the other order; the CT layer at the mesh; Studio at stride 48; depth shifts; a shifted tile origin; CT cross-sections from the full volume; the neighbourhood (does a row continue, do neighbouring turns show anything); and a base rate with a criterion written down before counting.
- **Working bar for "candidate"** (as published in `collab/claude-013`): connected strokes, in two detector families at the same place, on a surface shown to be on one sheet there, repeating on the neighbouring turn.

**Controls and nulls.** The positive beside every view. Random crops as a base rate in the later lanes (5 random windings per scroll, fixed seed).

**Numbers.** Atlas pass on PHerc. 0813 and 0800: 170 windings opened as batch montages, about 40 at 20 px/mm. PHerc. 0175A: all 53 windings as half-winding panels at 20 px/mm in both orders. Places taken through the full sequence: one on PHerc. 0125, one on PHerc. 0846B, two on PHerc. 0813, one on PHerc. 0191, one on PHerc. 0175A, one on PHerc. Paris 4.

**Verdict.** This is a judgement procedure, not a measurement. It has passed every positive it was shown.

**How this could be wrong or miss ink.**
- A full-resolution review for sparse strokes of **every** view was never done. This gap was stated in `collab/claude-004` and still applies to every negative.
- The review is not blind: the positive is labelled as such, and the reviewer knows which surfaces are unread.
- "Two detector families" are not independent (shared labels and ancestry), and the reverse error is possible too: real ink that only one family draws would be rejected.
- "Repeating on the neighbouring turn" cannot be tested where no surface exists on that turn, and a First Letters find can be a single line.
- Several closing arguments use letter size (2–4 mm on the positives). Letter height on the unread scrolls is unknown.

**Not done.** A human reader's look at the two panels that were set aside for one (F2). A blinded pass with positives mixed in.

**Where the code is.** None; `code/row_criterion.py`, `code/row_pair_criterion.py` and `code/strokeness.py` are the written criteria.

---

<a id="f1"></a>
## F1 — Rows of dots in Studio at stride 48 (one place on PHerc. 0125)

**Question.** On one public surface of PHerc. 0125, Studio at stride 48 and reader-v3 put about ten separate marks at the same positions along one straight row, in one order only, stable over depth shifts. Is it a line of writing?

**What was done.** Before looking at new data we wrote a prediction (writing implies further rows at a constant pitch of plausibly 4–8 mm, same order only) and a criterion, then ran four tests.
- **Row criterion** (`code/row_criterion.py`). A row event = at least 6 separate marks, each 0.6–4 mm in largest extent, centres on one straight line within ±0.7 mm, consecutive gaps ≤ 6 mm and median gap 2–5 mm, in one order and with fewer than 3 marks of the other order on that line. Mark = connected component of z > 2.5 after a 0.3 mm opening, on Studio stride-48 maps at shift 0.
  The first version asked for marks of 1.5–4 mm and found nothing on the place itself; the lower bound was changed to 0.6 mm on that place, before any control was counted.
- **Base rate.** 12 older atlas windings of PHerc. 0125 with a sheet-parallel fraction ≥ 0.85, drawn at random with a fixed seed (54.0 cm²), Studio stride 48, both orders.
- **Extension.** Three patch-grown windows of 13.7 cm² seeded on the row (on-sheet fractions 0.62–0.78), inked with reader-v3 and Studio in both orders.
- **Cross-sections** from the full volume through the row and beside it, and the angle between the mesh normal and the sheet normal (structure tensor).
- **Strokeness sweep** (`code/strokeness.py`). Strokeness = mean over the six largest components (z > 2, 0.15 mm opening, ≥ 0.15 mm²) in the row band of skeleton length divided by equivalent diameter. Studio at every shift from −7 to +7 in both orders, reader-v3 at −6…+6, Hecate at three shifts, on the place, on two control rows, on a row of an extension window, and on the first letter row of PHerc. 0343.

**Controls and nulls.** The 12 random windings. The blank parts of the two positives. Real PHerc. 0343 letters in the wrong order and at the wrong depth.

**Numbers.**

| set | area | rows in the reading order | rows in the other order |
|---|---|---|---|
| 12 random windings | 54.0 cm² | 6, on 5 windings | 2, on 1 winding |
| the flagged winding | — | 1 (the place) | 0 |
| three extension windows | 40.0 cm² | 0 + 1 + 2 | 0 + 1 + 1, plus 7 oblique in one window |
| PHerc. 0306B and 0343, blank parts | about 12 cm² | 0 | 1 (8 dots, pitch 2.5 mm) |
| PHerc. 0343, inside the letter boxes | — | 1 raw event | 9 raw events |
| PHerc. 0306B line 1 | — | 0 | — |

- Base rate 0.11 rows per cm² in the reading order, 0.04 in the other. The flagged winding's single row is what its area predicts.
- Row spacings where two rows can be compared: 3.1, 4.85 and about 5 mm. No common pitch.
- On the extension surface the row itself is absent: hot share within ±1 mm of the row line 0.00–0.04 in all seven maps, against 0.05–0.25 elsewhere in the strips. That surface passes 12–14 voxels from the public mesh there; layers are 5–8 voxels apart.
- The public tranches above and below are on other wraps (24–114 voxels away), so the atlas itself cannot show neighbouring lines.
- The mesh is parallel to the sheet within about 15° in a band about 5 mm wide around the row and leaves the sheet 3 mm above and below it.

| target | Studio, reading order: hot share / strokeness | Studio, other order | maximum strokeness, any detector, order, shift |
|---|---|---|---|
| PHerc. 0343 letters | 0.14 / 1.88 at shift 0; 0.06–0.08 / 0.59–0.67 at ±8 | 0.08 / 1.15 (range 0.73–1.54) | 1.88 |
| the place | 0.07–0.13 / 0.63–1.18 over 15 shifts; no peak | 0.01–0.06 / 0.22–1.00 | 1.35 |
| two control rows | 0.06–0.11 / 0.89–1.24 | 0.04–0.07 / 0.48–1.01 | 1.28 |
| a row on an extension window | 0.06–0.16 / 0.70–1.29 | 0.02–0.04 / 0.49–0.94 | 1.46 |

**Verdict.** Closed: `screened — no qualifying reading identified`. It is one instance of a pattern that occurs about once per 9 cm² on this scroll, it does not repeat at a pitch, no depth or order turns its dots into strokes, and the same dot-row form appears on blank papyrus of the positives.

**How this could be wrong or miss ink.**
- The base rate rests on 6 events. One observed against 0.8 expected cannot distinguish "at the base rate" from "twice the base rate".
- The criterion was changed once after seeing the target. It detects dot rows, which real letters produce only in the wrong order or at the wrong depth; it is not a detector of text.
- "Absent on our own surface" may mean "different layer": 12–14 voxels is about two layer spacings.
- The base rate itself may be the signal. If PHerc. 0125 carries faint writing everywhere, the 12 random windings are not blank controls (F6).
- A perfectly round component has a skeleton of one pixel, so strokeness near 0.6–1 for "dots" depends on their irregularity. Two evaluation choices were made after a first run (threshold 2.5 → 2; the control's band redefined).
- No human reader saw the stride-48 strip.

**Not done.** Letter-template matching. The other public detector families at stride 48 beyond Hecate. A surface built on the adjacent layer.

**Where the code is.** `code/row_criterion.py`, `code/strokeness.py`.

---

<a id="f2"></a>
## F2 — Rows of thin connected marks (one place on PHerc. 0846B)

**Question.** In one own window, Studio at stride 48 joins the stride-96 blobs into two or three rows of thin, connected, wavy marks in one order. Lines of writing?

**What was done.**
- Prediction written first: lines of writing run within about 20° of the circumferential direction; rows within 20° of the scroll axis cannot be text.
- Row lines were fitted to the hot pixels and mapped through the mesh to 3-D.
- A row-pair criterion was fixed before any base-rate map (`code/row_pair_criterion.py`): at least two parallel rows, each ≥ 10 mm of elements 1–3 mm high with gaps ≤ 1.5 mm, row pitch 3–8 mm, in one order and not the other, present at two of three depth shifts.
- Studio at five depth shifts and with the tile origin moved by 24 px; Hecate and `ink_9um` (two seeds) in both orders; element sizes measured against PHerc. 0343 through the same path.

**Controls and nulls.** Stride-48 Studio maps of three other clean cores (PHerc. 1203, 0175A, 0358). The other order. The moved tile origin.

**Numbers.**
- Rows: 15.4, 14.3 and 6.3 mm long; 77–83° from the scroll axis, 12–17° from circumferential; pitch 4.1 and 5.8 mm; rms about the fitted line 0.46–0.83 mm.
- Row contrast (rows against the band between them): Studio 1.89 / 2.29 / 2.44 / 2.44 / 1.94 at shifts −2 / 0 / +2 / +4 / +6; 2.31 with the moved origin; other order 1.49.
- Hecate 1.37 against 1.10 in the other order, with no visible row; `ink_9um` 1.10 and 1.05. reader-v3 has 3–4 blobs on one row.
- Elements: height 1.0–1.2 mm (median) against 2.8 mm for the positive's letters; stroke width 0.42–0.58 mm against 0.77 mm.
- The criterion does **not** fire on the place (longest connected run 9.0 mm, bar 10 mm). On the three other cores it fires once, on dense mottle; nearest miss 9.2 mm, also mottle.
- CT there: compressed, merged layers, with the mesh crossing layers obliquely about 1.5 mm away.

**Verdict.** Closed: `screened — no qualifying reading identified`. It remains the most row-like response we have seen, and it is essentially specific to Studio in one order.

**How this could be wrong or miss ink.**
- The geometry does not exclude text, the rows are stable in depth, and they are not a tiling artefact. We closed it on size, on the lack of a second family, and on not being able to name a letter.
- The automatic criterion separated nothing, so there is **no rate to quote**; "unlike the other cores" is a statement by eye on three windows (about 40 cm²).
- Element height 1.0–1.2 mm equals Studio's 128 px window at this voxel size. Whether the marks scale with the window or with the papyrus was not tested.
- The window was in the secondary arm (it failed the core gate), so the surface is poor.
- The panels were flagged for a human look that has not happened.

**Not done.** A surface offset by 2–4 voxels along the normal. Studio at another input scale. The 2.4 µm lane does not help: this scroll has no fine rescan.

**Where the code is.** `code/row_pair_criterion.py`.

---

<a id="f3"></a>
## F3 — Isolated stroke shapes on compressed layers, cracks and void edges

**Question.** Several windows show one letter-sized shape of two to four strokes, stable in depth and tile origin, sometimes in two detector families. What are they?

**What was done.** Each was taken through the R4 sequence.

| place | what the detectors show | what closed it |
|---|---|---|
| PHerc. 0813, first | a bar with a stem and a hook, Studio only | CT sections: the mesh runs through a dark gap between two sheets; the shape sits on a void edge |
| PHerc. 0813, second | a 5 mm cluster of three or four strokes in Studio; same under four tile origins (r 0.993–0.997) and across ±2 layers (r 0.83–0.97) | reader-v3 reproduces one diagonal stroke only; CT shows compressed layers 7–10 voxels apart with the mesh stepping across them within 1 mm; on two re-grown meshes the diagonal reappears and the rest breaks into blobs; an 8 mm extension has no letter-sized neighbours |
| PHerc. 0191 | a T-shaped mark of about 4 mm in reader-v3 and Studio, same order, stable over ±2 layers and a 48 px origin shift | compressed layers within ±8 voxels; mesh crossing layers 1–2 mm to one side; no second mark in a 12 × 12 mm crop |
| PHerc. 0175A | a hook of about 5 × 4 mm and a 6 mm stroke in reader-v3, Studio and faintly Hecate, one order, stable over −4…+4 layers and a 24 px origin shift | CT: wavy compressed layers 5–8 voxels apart with lens-shaped voids 10–20 voxels thick; the hook outlines a void; adjacent turns at the same angular position show nothing; a randomly drawn crop shows the same kind of two-detector agreement |
| five single strokes on four scrolls | one bar or arc each, mostly in one detector | each alone; most lie on a crack visible in the CT; not followed further |

**Controls and nulls.** Re-grown meshes; neighbouring turns; random crops (PHerc. 0175A).

**Numbers.** Re-grown meshes coincide with the original within 0.005–0.010 voxels (median normal distance). Detector windows span ±8 voxels, so neighbouring layers 5–10 voxels apart are inside the input.

**Verdict.** Closed, none a reading. Lesson: stability in tile origin and depth does not separate these from text; a second detector and a CT cross-section did, together with the absence of neighbours.

**How this could be wrong or miss ink.**
- In compressed zones the detector input contains adjacent wraps. A stroke there could be real ink **of the adjacent sheet**, seen through a mesh that crosses layers. Then its neighbours lie on that sheet's surface, which we never rendered.
- The re-grown meshes were not independent surfaces. No surface was built on the adjacent layer for any of these places.
- "No neighbours" was judged on crops of 8–12 mm on the same mesh.
- The random-crop comparison on PHerc. 0175A is one crop.

**Not done.** For each place: an offset surface following the adjacent layer, inked in both orders.

**Where the code is.** None specific.

---

<a id="f4"></a>
## F4 — Clouds in both orders, and clouds that persist across turns

**Question.** Diffuse patches of elevated response, 5 mm to 2 cm across, appear on almost every surface. Ink?

**What was done.** Compared the two orders at the same place; compared adjacent turns of a spiral fit at the same angular position; compared detectors; looked at the CT layer.

**Controls and nulls.** The other order and the adjacent turn.

**Numbers.** None beyond counts by eye. Typical cases: a saturated cloud of about 12 × 10 mm at all depth shifts on a PHerc. 0813 window; clouds of 1–2 cm at the same place in both orders on several PHerc. 0175A windings; one cloud that continues across two adjacent PHerc. 0175A turns; a teammate's clouds on touching PHerc. 1203 wraps in all four passes of two detector families.

**Verdict.** Structure, not ink: a response that is not face-specific, or that continues across turns at the same angular position, follows material or geometry (compressed bands, fibre-grid patches, radial cracks).

**How this could be wrong or miss ink.** A cloud can hide strokes: maps are z-scored per surface, so a large cloud raises the standard deviation and lowers every other pixel's z. We did not re-normalise cloud-bearing surfaces with the cloud masked out.

**Not done.** Masked re-normalisation. Any quantitative cloud detector.

**Where the code is.** None.

---

<a id="f5"></a>
## F5 — Ghost text from the neighbouring wrap (PHerc. Paris 4, a read scroll)

**Question.** Under the short final column of Scroll 1, within about 1 cm of the column's left edge, body-text detectors show rows of clear letters when the reading window is moved outward. A title, or the neighbouring wrap?

**What was done.**
- A first mark (a wavy stroke about 1 cm long, roughly 3.5 line pitches below the last line) was examined with a depth sweep: own renders, reader-v3 window centred from −240 to +250 µm in 29 µm steps, both orders.
- The rows that appear at +58 and +125 µm were then tested for letter identity (`code/ghostmatch_geom.py`, `code/ghostmatch_compare.py`):
  each point of the strip was moved 125 µm along the outward normal and matched to the nearest point on the **previous turn of the same published segment**;
  that turn's own ink map (reader-v3 at its own depth, and separately the organisers' published 2-D map) was warped into the strip's coordinates;
  the two maps were compared by normalised cross-correlation after a Gaussian of 0.15 mm.
- Nulls: the source map shifted by ±1 and ±2 line pitches, by ±3 to ±10 mm along the lines, mirrored, and mirrored with ±1 pitch: 19 in all.

**Controls and nulls.** The 19 nulls. The final column's own last lines through the same sweep. A fold-knee survey of the blank area for similar components.

**Numbers.**
- The outer wrap lies 146 µm outward of the mesh (median; p10 100, p90 241). The matched points are 31 µm from the shifted points (median).
- The map between the two turns is a translation of 25.4 mm along the winding with a Jacobian close to identity and positive determinant in 98 % of cells: no mirror.

| outer-wrap map | real r | nulls, mean ± sd (maximum) | per row, six rows |
|---|---|---|---|
| reader-v3 at its own depth | 0.862 | 0.249 ± 0.039 (0.315) | 0.79–0.92 |
| organisers' published 2-D map | 0.742 | 0.226 ± 0.039 (0.297) | 0.65–0.83 |

- Correlation with the outer wrap rises with the outward shift: 0.37 at 0, 0.71 at +58 µm, 0.86 at +125 µm.
- Depth sweep, share of pixels above 0.5 in a box minus the crop average: real text of the final column peaks at 0 and is gone from +106 µm; the first mark peaks at +19…+48 µm; the next letter-like form down the edge peaks at +192 µm.
- A vertical series of such forms along the column edge sits on the **previous column's** line grid (offsets 0.1–0.8 mm), not on the final column's (0.7–2.8 mm).
- The near-surface part of the first mark follows a dark crease seen in two independent scans (2.4 µm and 1.129 µm).
- Blank-area survey: fold knees cover 5.6 % of the area and wrap contacts 22 %; two components pass the size test in eight tiles and neither is specific to the reading order.

**Verdict.** Closed: the rows are the previous column's text on the next wrap, letter for letter. Not title evidence.
General lesson: published stacks of ±130 µm reach the next wrap at fold contacts, and detectors read through.

**How this could be wrong or miss ink.**
- The letter names in our notes are shape descriptions; they were not checked against the edition's text.
- The test covered a strip about 2 cm wide. Further into the column only speckle was seen, at five window depths.
- On an **unread** scroll the same mechanism is not a false lead in the same sense: letters from the neighbouring wrap are still writing in that scroll. There the caution is about which sheet a letter belongs to, not about whether it is ink.

**Not done.** The two lowest tiles of the blank stretch; the outermost wraps; a repeat of the inner part on the other autosegmentation pass. See T1–T3 in [07_negative_coverage.md](07_negative_coverage.md).

**Where the code is.** `code/ghostmatch_geom.py`, `code/ghostmatch_compare.py`.

---

<a id="f6"></a>
## F6 — Face asymmetry on PHerc. 0125, seen by three detector families

**Question.** On PHerc. 0125 the response in one order is 2–3 times that in the other, everywhere we looked. Faint writing everywhere, or a property of the papyrus?

**What was done.** Collected the hot shares of every map made for F1; added a teammate's A-hn result; looked for the pattern a fibre artefact would predict (rows along the other fibre family in the other order); compared with blank parts of the positives.

**Controls and nulls.** Hecate on the same crops. Blank papyrus of PHerc. 0306B and 0343.

**Numbers.**
- Studio and reader-v3: reading-order hot share about 2–3 times the other order on all three targets of the strokeness sweep (for example 0.07–0.13 against 0.01–0.06).
- A-hn, measured by a teammate on a different winding: 12.3 % against 5.6 % of pixels above 0.5 (2.2×).
- Hecate: 0.03–0.07 in both directions; no asymmetry.
- All row events on this scroll run along the winding (within 20°); axial rows: none in either order on the public windings.
- Blank papyrus of the positives is asymmetric too, with **opposite signs**: PHerc. 0306B background is hotter in the non-reading order (0.07–0.09 against 0.02–0.04), PHerc. 0343 far-field is hotter in the reading order (0.04 against 0.01–0.02).

**Verdict.** Origin undecided. Nothing positive supports writing; nothing excludes a faint, face-specific signal that these detectors cannot resolve into strokes.

**How this could be wrong or miss ink.**
- This is our largest single open item. Three families trained on related labels agree on the asymmetry; that is not three independent observations.
- Which order is the reading face on PHerc. 0125 was inferred from the umbilicus rule (S2), never shown on known text of this scroll. Since blank-papyrus asymmetry has no fixed sign, the hotter order need not be the written face.
- The older windings of this scroll were screened with reader-v3 in the "reading" order only. The other order was run with Studio on 16 windings and in the F1 work-up.
- Real letters at ±6–8 layers, or in the wrong order, look like these dot rows (D8).

**Not done.** A positive on a scroll scanned in the same session as PHerc. 0125. An oblique-surface test on a positive. The 2.4 µm lane cannot help: this scroll has no fine rescan.

**Where the code is.** `code/strokeness.py`.

---

<a id="f7"></a>
## F7 — Other artefact classes met in review

**Question.** Which recurring response types should a reviewer expect and discount?

**What was done.** Collected from every lane's review notes.

| class | what it looks like | mechanism, as far as tested |
|---|---|---|
| rows along a mesh edge | a row of round marks 1–2 mm inside the mesh boundary, in both reader-v3 and Studio | tiles at the boundary see zero padding; CT shows nothing at the row |
| fibre-parallel bars | runs of short bars 4–6 mm long, a few mm apart, all along one fibre family | fibre streaks; no horizontals or curves |
| crack-following chains | chains of marks along a dark diagonal line of the CT | crack or fold edge |
| radial streaks, "wood-grain" | streaks following concentric or radial bands | the surface cuts obliquely through wraps |
| tile grid at stride 96 (Studio) | blocky 96 px patches, broken chains | no blending between tiles |
| squares at stride 192 (2.4 µm models) | square patches of about 0.46 mm on blank papyrus | per-tile responses that the wider stride does not blend |
| L-shaped marks outside the papyrus | bright corners where tiles are partly empty | also present in published maps |
| a cloud with a one-order excess at 2.4 µm | amorphous, 3 × 5 mm, forward-to-reverse ratio 4.1 on one public PHerc. 1203 block | not reproduced from the 9.36 µm volume, where it is present in both orders; later seen again by Hecate 2.4 µm, still amorphous |
| response on a smooth bright region at a mesh edge | Hecate 2.4 µm, mostly in the reversed order | edge of a mask or fill |
| letter-like strokes at ±6 layers on a written scroll | strokes in other places than at shift 0 | plausibly the neighbouring wraps (PHerc. 0841) |

**Controls and nulls.** As listed per class.

**Numbers.** On the 2.4 µm control, known text gives a forward-to-reverse ratio of ink pixels of 8.5; blank-like blocks 0.9–2.2.

**Verdict.** Artefact classes; each is recognisable from the CT layer or from the other order.

**How this could be wrong or miss ink.** Writing along a mesh edge, along a crack, or on a cut-through surface would be discounted with its class. Edges and cracks are not rare: a radial crack crosses every winding of the PHerc. 0175A fit.

**Not done.** A mask-aware detector run for edge rows (re-rendering with real data beyond the mesh edge).

**Where the code is.** None.
