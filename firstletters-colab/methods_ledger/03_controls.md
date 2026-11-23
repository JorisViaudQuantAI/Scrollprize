# 03 — Controls and calibration

A control is a public surface with known text. We use controls for two different things, and they must not be confused:
an **input-path check** (are the layers, order, depth and scale right?) and a **sensitivity check** (would this detector see ink on a scroll it has not seen?).
Only a control that the detector did not train on can serve the second purpose. We have very few of those.

---

<a id="c1"></a>
## C1 — PHerc. 0841, segment w00

**Question.** Is our inference path right, and what do the 9 µm detectors show on a native 9.37 µm scan of a scroll with known text?

**What was done.**
- Data: the organisers' published surface volume of segment w00 at 9.366 µm / 113 keV, its mesh, the 2.4 µm rescan of the same segment, and the published ink map of the canonical 2.4 µm model on it.
- 9 µm answer key: that published 2.4 µm map resampled to the 9.366 µm grid; "ink" = map ≥ 128. Region: 1664 × 1792 px = 2.6 cm² of text; AUC on every third pixel.
- We also render the segment ourselves (65 layers with our renderer; 41 layers with `vc_render_tifxyz`), because the unread surfaces go through our renderers.
- 2.4 µm control: published ink labels on the fine surface volume, 1.04 cm² supervised in three blocks, 22 % ink, scored at 9.6 µm (H2).

**Controls and nulls.** Reversed order; off-sheet depth (±20 layers); a pure-noise stack; published maps of the same models for parity.

**Numbers.** See D1–D6. Registration between the 2.4 µm and 9.37 µm renders of the segment is within about 6–10 px at 9.37 µm.
Our own render correlates only 0.88–0.91 with the organisers' surface volume, although it is aligned at 0 px.

**Verdict.** Usable for input-path checks of every detector, and as a sensitivity check for reader-v3 and `ink_9um` only.

**How this could be wrong or miss ink.**
- The answer key is a model output from the finer scan, not a transcription. A detector that copies that model's habits scores well.
- PHerc. 0841 is a training donor of Studio and sits in Hecate's published training bucket. The canonical 2.4 µm model's published labels come from its own family.
- It is one region of one scroll. Its ink is evidently readable at 9 µm; the unread scrolls were left unread by many teams with the same models.
- Off-sheet windows of this scroll reach a row score of 3.07, probably because neighbouring wraps carry real text. Our "blank" baseline is therefore not blank.

**Not done.** A second region of the scroll. A labelled control on PHerc. 0814 at 2.4 µm (its only public segment with all inputs has no labels and an illegible published map).

**Where the code is.** `firstletters-colab/controls/` (already public).

---

<a id="c2"></a>
## C2 — PHerc. 0343, the awarded windings (w047 and w048, joined)

**Question.** Does a search pipeline, end to end, show the letters of a First Letters win on a spiral winding of a scroll of the unread type?

**What was done.**
- Data: the winner's public repository holds the snapped windings and 18 letter boxes. Volume at 8.64 µm.
- We render the joined mesh with the renderer of whichever lane is being checked (41 layers; 2560 × 6720 px, 8.09 cm² valid) and run that lane's exact jobs on it before the unread surfaces.
- **Letter-box AUC**: positives = valid pixels inside any box; negatives = valid pixels ≥ 1.5 mm from every box; per-map z-score; rank AUC. Also "hot share" (z > 2) inside the boxes and in the far field.

**Controls and nulls.** The other order. The lane's rankers must put the letters first.

**Numbers.**

| detector, setting | AUC boxes | hot in boxes / far | line finder, best chain |
|---|---|---|---|
| reader-v3, reading order | 0.684 | 0.31 / 0.03 | 10.91 |
| reader-v3, other order | 0.601 | 0.12 / 0.02 | 5.95 |
| Studio, reading order, stride 48 | 0.745 | 0.34 / 0.04 | 9.48 |
| Studio, reading order, stride 96, shifts −4…+4 | 0.711–0.744 | 0.28–0.33 / 0.04 | 3.82–7.75 |
| Studio, other order | 0.640–0.716 | 0.11–0.13 / 0.01–0.02 | 2.51–5.41 |
| on-sheet tests of this surface | m7 pixel test 0.73; layer coherence 0.85 in one render and 0.73 in another | | |

- Through the PHerc. 0175A jobs, restricted to sheet-parallel tiles (which keep 77 % of the letter-box area): line finder 8.14 (reader-v3) and 7.26 (Studio) in the reading order, 1.52 and 1.61 in the other.
- The letters are 3–4 mm high in the maps and measured 2.8 mm (median element height) through the stride-48 Studio path.

**Verdict.** Usable, and the best control we have for spiral windings. Every lane's positive check passed on it.

**How this could be wrong or miss ink.**
- Boxes are diluted positives: most pixels in a letter box are blank. An AUC of 0.68–0.75 here means clearly visible letters. Do not compare it with 0.86–0.92 on PHerc. 0841.
- The far field may contain unboxed letters (a second partial row is visible), which lowers the AUC.
- Exposure is not settled: the Studio release lists label rows named PHerc0343P among Reader v2/v3 training inputs. We did not establish whether that object is this scroll, or whether the rows overlap these windings.
- It is a win: the best-case surface, snapped and checked by hand. Our surfaces are worse.
- The same winding scores 0.848 and 0.728 on our layer-coherence score in two renders made by two lanes with nominally the same settings. The difference is unexplained.

**Not done.** The winner's **raw** (unsnapped) fit through our path. A stretched or tilted version of the positive (S8, D8).

**Where the code is.** `code/letterbox_auc.py`.

---

<a id="c3"></a>
## C3 — PHerc. 0306B, the claim crop, with the corrected row labels

**Question.** Same as C2 on a second win, and: how do detectors behave on a line that is nearly empty?

**What was done.**
- Data: the winners' public claim mesh, the claim box (38 × 9.6 mm) and two published row centres. Volume at 8.64 µm. Our 41-layer render of the crop: 5200 × 2000 px, 7.66 cm² valid, 98.6 % valid.
- Per row: a band of ±1.5 mm over the claim-box width. Background: the box width at ≥ 3 mm from both rows. Hot share (z > 2), band-against-background AUC, best line-finder chain centred within ±1.5 mm of the row, best `line_review` proposal.
- Line 2 was also split into its first 1.5 cm and the rest.
- The winners' own published maps (TRACE v3B, A-hn, their product, their reader maps, a decoy) were cropped to our frame and scored the same way.

**The correction.** The claim image is the render flipped top to bottom. Our first table (`collab/claude-008`) used the unflipped conversion, so the two rows were mirrored and 0.75 mm off.
With the release's own conversion, checked against row-profile peaks in the winners' maps:
- the row our maps show clearly is **line 1**, the line that was read;
- the row at background level is **line 2**, where the published transcription has one uncertain letter.
So what `collab/claude-008` called a blind spot on a line of letters is a line with almost no read text. The correction is in `collab/claude-010`.

**Controls and nulls.** The other order. The winners' decoy model (labels moved by 3.5 × 3 mm).

**Numbers.** Reading order unless stated; ranges are over depth shifts −4…+4.

| map | background hot | line 1 hot / AUC | line 2 hot / AUC | line 2, first 1.5 cm | line 2, rest |
|---|---|---|---|---|---|
| reader-v3 | 0.03–0.04 | 0.23–0.30 / 0.64–0.66 | 0.04–0.06 / 0.54–0.58 | 0.10–0.13 / 0.62–0.73 | 0.01–0.03 / 0.51–0.53 |
| Studio (GPU, fp16) | 0.02–0.04 | 0.24–0.33 / 0.65–0.71 | 0.08–0.10 / 0.59–0.63 | 0.15–0.19 / 0.62–0.70 | 0.05–0.07 / 0.57–0.60 |
| Studio, other order | 0.07–0.09 | 0.02–0.07 / 0.42–0.55 | 0.06–0.08 / 0.50–0.56 | 0.00–0.03 | 0.07–0.11 |
| winners' TRACE v3B | 0.02 | 0.21 / 0.72 | 0.05 / 0.67 | 0.12 / 0.73 | 0.03 / 0.65 |
| winners' A-hn (mean of 3) | 0.04 | 0.26 / 0.66 | 0.12 / 0.63 | 0.22 / 0.75 | 0.08 / 0.58 |
| winners' decoy model | 0.01 | 0.15 / 0.69 | 0.09 / 0.65 | 0.25 / 0.82 | 0.02 / 0.59 |

- Line finder on line 1, reading order: 8.9 (reader-v3) and 10.4 (Studio, stride 48) at shift 0; 5.9–10.4 over all shifts from −6 to +6.
- Line finder on line 2: at most 8.81, and that chain is a tilted join of the left-end marks with marks above the line, not a row along line 2.
- Claim-box AUC (the whole box, mostly blank): reader-v3 0.572, Studio 0.601.
- On-sheet tests of this surface: layer coherence 0.95.

**Verdict.** Usable; the least training-exposed positive we have (the Studio release does not name this scroll among its training categories).
No public detector we can run or score makes line 2 stand out as a line of letters; its first centimetre and a half is visible.

**How this could be wrong or miss ink.**
- A line as faint as line 2 would be missed by every screen in this ledger, with any of these models.
- The decoy model's high score on the left end of line 2 shows that part to be a strong but not text-specific response.
- We mislabelled the rows once. A reviewer should re-derive the row positions from the release rather than trust ours.
- One crop, one line of read text.

**Not done.** The winners' full window set through our path.

**Where the code is.** `code/letterbox_auc.py`; the per-row evaluation script is not included (it is the same band arithmetic).

---

<a id="c4"></a>
## C4 — PHerc. Paris 4 (Scroll 1), the final columns

**Question.** Is the input path right on the published 2.4 µm surface volumes of Scroll 1, pooled to 9.6 µm?

**What was done.**
- Data: the published surface volume of the multi-turn segment `…-w010-027` (two autosegmentation passes). It keeps all 109 layers of 2.4 µm at every lateral level; layer 54 is the mesh.
- Pooling: level 2 laterally (9.6 µm) and means of 4 layers in depth give 27 planes at 9.6 µm; plane 13 is the mesh. This is the representation reader-v3 and `ink_9um` were trained on.
- Control crop: 14.7 mm square on the final text column (two lines). Blank crop: 1 cm to the right of the column.
- Answer keys: the published 3-D ink volume projected on the mesh, and the published 2-D map; positive ≥ 128, negative < 64.

**Controls and nulls.** Other order; depth shifts; a coarser level (19.2 µm fed as 9.6 µm).

**Numbers.**

| detector, order | AUC against 3-D ink | AUC against 2-D map | share > 0.5 on control / blank |
|---|---|---|---|
| reader-v3 forward | 0.90 | 0.94–0.95 | 17–18 % / 1 % |
| reader-v3 reverse | 0.56–0.66 | 0.57–0.67 | < 1 % / < 1 % |
| Studio ascending | 0.86–0.87 | 0.92–0.94 | 18–23 % / 3.5 % |
| Studio descending | 0.52–0.56 | 0.49–0.55 | < 1 % / < 1 % |
| Hecate forward (fp16) | 0.92 | 0.95 | 11 % / 0.07 % |
| reader-v3 forward at 19.2 µm | 0.81 | 0.89 | |

- Pooled input against the raw 109-layer input: r 0.994, mean |Δp| 0.007.
- Layout measured on the published map: line pitch 6.75 mm (final column 6.9 mm, another column 6.5 mm), letters about 2.8 mm tall, columns 61.4 mm wide, intercolumnium 17.3 mm. The published edition gives 62 mm, 17 mm and 6.3–6.7 mm per line.

**Verdict.** Input path only. Every public detector learned from Scroll 1 body text, and the final column itself lies in training regions (17 % of its line area inside one supervision mask; 27 % within 50 µm of a legacy label row for Studio).

**How this could be wrong or miss ink.** These AUCs are optimistic and say nothing about an unusual ink. Studio was taught that a strip of about 1.6 × 3.3 cm next to the final column is background.

**Not done.** A detector with no Scroll 1 exposure; none exists publicly.

**Where the code is.** Not included.

---

<a id="c5"></a>
## C5 — PHerc. 172 (Scroll 5)

**Question.** What does a label-free feature do on a scroll whose ink is known to be denser in CT?

**What was done.** A 14.6 cm² crop (5,376 × 4,352 px) of a published 33-layer surface volume at 7.91 µm, with the published TimeSformer prediction as the key (≥ 128 = ink, 14.5 % of pixels). Used only in L1–L4.

**Controls and nulls.** Depth-shifted, other-face and rolled-label nulls (L1).

**Numbers.** Maximum projection over ±6 layers at stroke scale: AUC 0.640 (nulls 0.55 / 0.54 / 0.51). Ink pixels are about 5 grey levels brighter than non-ink at every depth from −10 to +3 voxels.

**Verdict.** The "ink visible in CT" case, and only that. No letter is visible by eye in its raw layers at 40 px/mm either.

**How this could be wrong or miss ink.** The key is a neural prediction, so 0.64 is agreement with that model. No 9 µm detector was run on it by us.

**Not done.** reader-v3 or Studio on this crop.

**Where the code is.** Not included.

---

<a id="c6"></a>
## C6 — PHerc. 1447

**Question.** Is there a second scroll with enough known lines to test a line-pitch statistic?

**What was done.** The public set for windings w058, w060 and w062 (69 cm² together, 8.64 µm), with human labels and a base-model map, stacked in absolute z.

**Controls and nulls.** The human labels themselves through the same statistic.

**Numbers.** w058 has two short lines (about 10 letters), w060 four short rows. Label pitch 6.06 mm. See L5.

**Verdict.** Sanity control only: fewer than five lines of known text, not a periodic column.

**How this could be wrong or miss ink.** A-hn used w058 in training (`collab/codex-010`).

**Not done.** reader-v3 or Studio on these windings.

**Where the code is.** `code/linefold_stat.py`.

---

<a id="c7"></a>
## C7 — Which detector saw which scroll in training

**Question.** For each control and each detector, is the control held out?

**What was done.** Read the public model cards, training configurations and data plans of each release. For Scroll 1 we also measured distances: the labelled 3-D points of each public label source were put in a KD-tree and the distance from every mesh point of the tested segment to the nearest one was computed ("overlap" = within 50 µm).

**Controls and nulls.** None; this is a provenance table.

**Numbers.** "yes" = training data; "no" = stated or listed as not used; "?" = not settled.

| detector | PHerc. 0841 w00 | PHerc. 0343 | PHerc. 0306B | Scroll 1 | PHerc. 1447 | the 19 eligible scrolls |
|---|---|---|---|---|---|---|
| reader-v3 | no (author's statement) | ? (rows named PHerc0343P in its label list) | not listed | yes (eight label rows) | no (author's statement) | not listed |
| `ink_9um` | no | not listed | not listed | yes | not listed | not listed |
| Hecate 9.6 µm | in its published training bucket | ? | not listed | indirectly (distilled from Scroll 1 teachers) | ? | ? for the two with a fine rescan |
| Studio | **yes** (one of three donors) | ? (same PHerc0343P rows) | not named in its training categories | yes | not listed | not listed; PHerc. 0211 is its named hold-out |
| canonical 2.4 µm, Hecate 2.4 µm | treated as yes (its own published maps and model-assisted labels) | — | — | labels not disclosed | — | ? |
| A-hn | no | ? | fine-tuned on this scroll with an exclusion zone | ? | yes (w058) | not listed |

Scroll 1 distances: the blank end stretch (agraphon) has no 2-D supervision mask within 50 µm and no 3-D scribble within 147 µm for any detector, except one legacy Studio row that covers a strip beside the final column as background.

**Verdict.** No control is clean for every detector. PHerc. 0306B is the cleanest for Studio and reader-v3; PHerc. 0841 is clean for reader-v3 and `ink_9um` only; nothing is clean at 2.4 µm.

**How this could be wrong or miss ink.** We read cards and configuration files, not the training data. "Not listed" is weaker than "held out". The PHerc0343P question is open.

**Not done.** A spatial non-overlap check for PHerc. 0343. Any contact with the model authors.

**Where the code is.** Not included.

---

## What each control can and cannot show

| control | can show | cannot show |
|---|---|---|
| PHerc. 0841 w00, 9.37 µm | layer order, depth window, tiling, precision; that reader-v3 reads an unseen scroll with clear ink | Studio's or Hecate's sensitivity (exposed); anything about faint ink |
| PHerc. 0841 w00, 2.4 µm | that a 9 µm mesh mapped through our registration can be read at 2.4 µm | whether the 2.4 µm models see ink on a scroll they did not train on |
| PHerc. 0343 windings | that a whole lane, from render to ranker, shows the letters of a win on a spiral winding | how the lane behaves on an unsnapped, stretched or cut-through surface |
| PHerc. 0306B crop | the same on a second win; behaviour on a nearly empty line | sensitivity to a faint but fully written line |
| Scroll 1 final columns | the input path for published surface volumes; layout numbers | anything about sensitivity (all detectors exposed) |
| PHerc. 172 | that a label-free feature registers CT-dense ink statistically | readability; behaviour of the 9 µm detectors |
| PHerc. 1447 | a weak second check for pitch statistics | periodic-column behaviour |
| reversed order | that a response is face-specific | that it is ink: blank papyrus is also asymmetric between orders |
| off-sheet depth, noise stack | the floor of the row score | a blank-papyrus baseline: off-sheet windows of a written scroll can contain text |
