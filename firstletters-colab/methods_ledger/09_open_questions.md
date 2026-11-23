# 09 — Open questions and suspected weak points

<a id="q"></a>
This is the part of the ledger we most want attacked.
Each item states a weak point, the evidence that makes us doubt ourselves, and a test a reviewer can run, with a rough cost where we have a measured rate.
Entry ids in brackets point to the method concerned. Costs use the rates of [08_compute_and_cost.md](08_compute_and_cost.md).
Where a test needs the location of a closed place, the repository owner holds it privately; do not ask for it in public notes.

**If you can do only a few, do these first:** Q7 (hindsight test on a won scroll), Q1 (the other order on the atlas), Q33 (what misplaced text looks like), Q21 (review recall with inserted positives), Q10 (stretched renders), Q15 (adjacent-layer surfaces), Q34 (would our bar have passed a real win), Q22 (Studio's intensity range), Q12 (winding sense of PHerc. 0175A), Q42 (the lead calculation).

## A. Reading face and layer order

**Q1 — The atlas was read in one order only.** [S2, D7]
- Weak point: reader-v3 ran on 1,295 cm² of atlas in the single order given by a geometric face rule. The rule was checked on two positives from the 8.64 µm campaign and never on known text of the seven atlas scrolls.
- Evidence of doubt: the two positives read in opposite orders in the same renderer; the wrong order scores 0.50–0.60, so a wrong label would have hidden text.
- Test: ink a random 10 % of the atlas windings, stratified by scroll, in the other order with the same settings. Compare hot shares, line-finder maxima and montages. About 130 cm² × 2 passes at 500 cm²/h: half a GPU-hour.

**Q2 — Six windings with an ambiguous face label.** [S2]
- Weak point: six short PHerc. 0813 windings have a majority share of only 0.51–0.88 for their label against the official umbilicus.
- Test: recompute the rule per vertex and report where on those windings it flips; ink them in both orders.

**Q3 — Which order is the written face of PHerc. 0125?** [F6]
- Weak point: we call the hotter order the reading order because the rule says so. Blank papyrus of the positives is hotter in the non-reading order on PHerc. 0306B and in the reading order on PHerc. 0343, so the sign of an asymmetry identifies nothing.
- Test: the public atlas also holds windings of PHerc. 0211, a read scroll from the same 9.36 µm campaign. Apply the rule there and check it against that scroll's published text; measure the sign of the blank-papyrus asymmetry there.

**Q4 — Fitted PHerc. 1203 windings are read in one order, chosen from geometry.** [H5]
- Weak point: the bulk run reads the order inferred from the normal direction. The first winding showed text in neither order, and its upper half had more response in the other order.
- Test: read the other order on all 16 windings (98 cm², two models, stride 192: about 2.7 L4-hours), or on a random third.

**Q5 — "In one order and not the other" is used as if the other order were a null.** [D7, F1, F2, R4]
- Weak point: both written criteria and the working bar require order-specificity. Real letters stay partly visible in the wrong order (box AUC 0.60–0.72), and blank papyrus differs between orders by a factor of up to 3.
- Test: on the blank parts of both positives, tile the maps and report the distribution of the hot-share ratio between orders; state what ratio a criterion may treat as "specific".

## B. Surfaces

**Q6 — Depth tolerance was measured with a smooth displacement, on one positive, with one detector.** [S9, D8]
- Weak point: the decision to ink the atlas unsnapped at ±3 layers rests on reader-v3 losing nothing on PHerc. 0343 displaced by a smooth field of −3.9…+5.9 voxels. Real misplacement has jumps between sheets and larger offsets.
- Test: (a) run the PHerc. 0343 winner's **raw, unsnapped** fit of the same windings through the atlas path; (b) inject a piecewise-constant offset with jumps of ±6–10 voxels; (c) repeat both with Studio.

**Q7 — Would our atlas pass have found a scroll that was later won?** [S2]
- Weak point: the atlas holds 90 windings of PHerc. 0211 that were screened negative with older tools before that scroll was won. We never ran our production path on them.
- Test: run the exact atlas path (reader-v3, rule order, −3/+3, line finder, montage review) on those windings and report whether the awarded letters are found. First check that the awarded region is covered by an atlas winding at all; we did not inspect the winner's mesh. About 540 cm²: two to three GPU-hours.
- Reading: if the letters are not found, every atlas negative in [07](07_negative_coverage.md) is much weaker than it looks.

**Q8 — Ranking used the mean of two shifted maps.** [S9]
- Weak point: a response present at one shift only is halved in the mean before the rankers see it.
- Test: on the positives displaced by +6 and −6 layers, compare the line-finder score of the mean map with that of each single map.

**Q9 — Sheet identity along a winding was never checked, and our on-sheet measure saturates.** [S4, S7]
- Weak point: snapped PHerc. 0175A windings score 0.668 on a predicted sheet, above the awarded PHerc. 0343 winding (0.60), while looking worse. A snap that jumps sheets still scores as on-sheet. A teammate found 9 jumps in 23 profiles.
- Test: along each row of a winding, count the places where the surface crosses a dark gap in the CT between consecutive vertices; report crossings per centimetre for our windings and for the awarded one.

**Q10 — The PHerc. 0175A screen (520 cm²) used renders stretched or squeezed by 0.6–2× along the lines.** [S8]
- Weak point: detector tolerance to anisotropic scale error of this size is unknown; ±20 % isotropic is all we measured.
- Test: (a) measure the vertex-spacing distribution of the PHerc. 0343 windings, to learn how much stretch the positive itself carried; (b) resample the positive's render by 0.5× and 2× along the lines and re-ink; (c) re-render the eight top-ranked PHerc. 0175A windings isometrically with `code/isometric_resample.py` and re-ink (about 100 cm²).

**Q11 — The spiral gate has an arbitrary threshold and a crude control.** [S4]
- Weak point: windings below 0.40 raw on-sheet were not inked (66 on PHerc. 0175A, 48 on PHerc. 1203). The control shifts every vertex by a fixed ±8 voxels although sheets are 5–8 voxels apart in places. The script's own documentation describes a different control from the one it runs (see the header of `code/spiral_oncheck.py`).
- Test: ink ten randomly chosen windings below the gate and compare their maps and CT with ten above it. Recompute the gate with a control scaled to the local sheet spacing.

**Q12 — Three inputs of the PHerc. 0175A fit are unverified.** [S4, S5]
- Weak point: the winding sense is an assumption copied from another team's comment; the umbilicus is ours and sits 80–160 voxels off the visual centre on three slices; the normals are synthetic.
- Test: run `code/sense_chain.py` on the published tracks and crossings of this scroll (about one CPU-hour). If it says `CW`, the fit and the 520 cm² screen rest on a mirrored spiral.
- Follow-up (2026-11-23): [Q12 audit](../winding_audit/RESULTS.md) fixes a reproduced
  storage-order defect and establishes that a chain's raw drift sign cannot by
  itself decide whether the fit is mirrored. The full official catalog currently
  derives **ACW**, matching the existing setting. The frozen public graph replay
  validates representation invariance, not anatomical winding sense; center,
  same-sheet continuity and axial closure remain separate questions.

**Q13 — Fits on our own tracks.** [S6]
- Weak point: our vertical tracks are shorter than the organisers'. PHerc. 1203, fitted on our tracks, passed 17 of 65 windings; PHerc. 0175A, on published tracks, 53 of 119. We cannot separate scroll from tracks.
- Test: fit the PHerc. 0175A band once more with tracks made by our extractor and compare the gate.

**Q14 — The on-sheet test for own windows trusts the surface prediction.** [S1]
- Weak point: where the prediction is wrong the seed score, the gate and the reported on-sheet area are wrong together. We saw it overstate crushed windows.
- Test: label the en-face CT of 20 windows by hand in 1 mm blocks and report agreement with `code/onsheet_gate_m7.py`.

**Q15 — No closed place was ever tested on an adjacent-layer surface.** [F3, S1]
- Weak point: four isolated stroke shapes were closed as "compressed layers under the mesh, no neighbours". In such zones the detector input includes the adjacent sheet. If a stroke is ink of that sheet, its neighbours lie on a surface we never rendered. Our re-grown meshes coincided with the original within 0.01 voxel, so they tested nothing.
- Test: for each place, build a surface offset along the normal onto the neighbouring CT layer (7–10 voxels), render and ink it in both orders over at least 2 × 2 cm, and look for a row.

**Q16 — "Absent on our own surface" may mean "different layer".** [F1]
- Weak point: one closing argument for the PHerc. 0125 place is that a patch-grown surface 12–14 voxels away does not show the row. Layers there are 5–8 voxels apart.
- Test: shift the public mesh along its normal in steps of 2 voxels over ±16, re-render each as a surface (not as a window shift) and track the row.

**Q17 — The layer-coherence score gives 0.848 and 0.728 for the same positive.** [L2, C2]
- Weak point: two lanes rendered the same PHerc. 0343 mesh with nominally the same settings and got different sheet-parallel fractions. We use this score to restrict rankers and to quote on-sheet area.
- Test: render the mesh both ways, diff the tile-class maps, and find the cause (interpolation mode, edge tiles, mid-layer index).

**Q18 — The layer-coherence score is an upper bound validated on 59 labels of someone else.** [L2]
- Weak point: structureless tiles count as parallel; depth offset is invisible to it.
- Test: on windings with an m7 render, compare tile classes with the m7 pixel test of S1.

**Q19 — One depth offset per window.** [S1, D8]
- Weak point: own windows are inked at the window's brightest layer ±1, while sheets wander by ±10 voxels locally.
- Test: on the positives, replace the mesh by its 5 mm smoothed version and ink at one global offset.

**Q20 — Registration of the fine band is checked only near the centre.** [H1, H3]
- Weak point: all landmarks lie within about 11 mm of the centre; fitted windings extend further; four coarse voxels of error cost a third of the recall at 2.4 µm.
- Test: measure the residual by local cross-correlation at 20–30 points at 10–20 mm radius; compare with the teammate's annular registration.

## C. Detectors and sensitivity

**Q21 — We have no sensitivity limit and no measured review recall.** [R4, all N rows]
- Weak point: the coverage table says where nothing was found, not how faint an ink could have been found. The review was never tested blind.
- Test: (a) cut stack patches of known text from the positives and insert them, at full and at reduced contrast, into render stacks of unread surfaces at places unknown to the reviewer; run the full pipeline including ranking and montage review; report the recall. (b) Report the contrast at which recall falls to 50 %, per detector.
- Caveat from L6: there is no additive ink signature, so reduced-contrast insertion is only a rough yardstick; others measured AUC 0.6–0.77 for pasted text.

**Q22 — Studio reads absolute intensities; the unread scrolls' histograms were never compared with its training range.** [D4, L11]
- Weak point: Studio clips at 200 and divides by 255, with training augmentation of gain 0.8–1.2 and bias ±15. Volumes are 8-bit with per-scan export windows and two different beam filter sets.
- Test: for each unread scroll, the 1st, 50th and 99th percentiles of the 16-plane on-sheet stack, against the same for PHerc. 0841, 0343 and 0306B. CPU only.

**Q23 — Studio's small curved marks have the size of its window.** [F2, D4]
- Weak point: element height 1.0–1.2 mm equals 128 px at these voxel sizes. They may be per-window responses joined by blending.
- Test: resample the input by 0.8× and 1.25× and re-run at stride 48. If the marks keep their size in pixels they are an artefact; if they keep their size in millimetres they follow the papyrus.

**Q24 — Studio saw mostly what reader-v3 ranked.** [D4, N]
- Weak point: outside the small scrolls, Studio ran on the top of reader-v3's ranking plus five random windings on one scroll.
- Test: Studio at stride 48, both orders, on a random 10 % of the atlas area chosen without reference to any map.

**Q25 — Most Studio negatives are at stride 96.** [D4, F2]
- Weak point: the one row structure we worked up appeared only at stride 48; at stride 96 the same known letters score 3.8 instead of 9.5 on the line finder.
- Test: stride 48, both orders, on the cores of the 22 inked own windows (98 cm²: about 2.5 L4-hours).

**Q26 — Hecate's silence was used as evidence.** [D3, F1, F2]
- Weak point: Hecate barely shows the PHerc. 0343 letters (hot share 0.05–0.07, strokeness 1.1–1.35).
- Test: Hecate on PHerc. 0306B line 1 and on PHerc. 0343 at stride and depth settings matched to the work-ups; state its recall on known letters before using its absence again.

**Q27 — Agreement between detector families is not independent evidence, in either direction.** [R4, F6]
- Weak point: Studio, reader-v3 and A-hn share label ancestry. We ask for two families to accept, and we count three families agreeing on the PHerc. 0125 asymmetry.
- Test: a provenance table of label sources per detector, and the correlation of their maps on blank papyrus of the positives.

**Q28 — The 2.4 µm models have no held-out control.** [D6, H2]
- Weak point: every 2.4 µm number we have is on training-family data. Their negatives on PHerc. 1203 carry no measured sensitivity.
- Test: find any labelled 2.4 µm region outside both models' training (the authors would know), or run a label-free check at 2.4 µm (Q44).

**Q29 — The bulk fine read uses stride 192.** [H4]
- Weak point: on blank papyrus stride 192 gives square patches; its effect on weak text is unmeasured.
- Test: on the control, lower the text contrast by blending with blank stacks and compare stride 128 with 192.

**Q30 — Is PHerc. 0343 a held-out control?** [C2, C7]
- Weak point: label rows named PHerc0343P appear in the Reader and Studio training lists.
- Test: establish what that object is, and whether its rows overlap the awarded windings in space.

**Q31 — Per-surface z-scores let a cloud hide strokes.** [F4, R2]
- Weak point: "hot" means z > 2 within one map; a large cloud inflates the standard deviation.
- Test: on cloud-bearing windings, mask the cloud, re-normalise, re-rank.

**Q32 — All detectors come from a few scrolls' labels, and the remaining scrolls are the hard ones.** [L11]
- Weak point: we assume the released models transfer; the evidence is two box-diluted positives.
- Test: the fine-tune with held-out positives described in L11 (about 4 L4-hours), judged by criteria fixed beforehand.

**Q33 — What does misplaced text look like?** [D8, F1, F6]
- Weak point: real letters at ±6–8 layers, or in the wrong order, become rows of dots with strokeness below 1. That is the form of the PHerc. 0125 response, which occurs there at 0.11 rows per cm². We never measured what an **oblique** or wrong-layer surface does to known text.
- Test: render the PHerc. 0343 positive on surfaces tilted by 5°, 10° and 20° and displaced by 6–12 voxels; run `code/row_criterion.py` and `code/strokeness.py`. If dot rows appear at a similar rate, the right response on PHerc. 0125 is a better surface, not a closed file.

## D. Criteria and review

**Q34 — Would our bar have passed a real win at first sighting?** [R4]
- Weak point: the bar asks for connected strokes, two families, a surface shown to be on one sheet, and repetition on the neighbouring turn. The PHerc. 0306B claim is one line.
- Test: apply the bar, as written, to the two public wins as they appear in our own maps, and to their neighbouring turns. Report which condition each would have failed.

**Q35 — We rejected marks for being small.** [F1, F2]
- Weak point: 1.0–1.2 mm elements were set against 2.8 mm letters of one positive. Letter heights measured on the two wins span about 1.2–3.8 mm in our notes, and heights on the unread scrolls are unknown.
- Test: measure element height on PHerc. 0306B line 1 through the same stride-48 path; collect letter heights for Herculaneum hands from the literature.

**Q36 — The dot-row base rate.** [F1]
- Weak point: the criterion was changed once after seeing the target; the rate rests on 6 events; and the "controls" are windings of the same scroll, which may carry the same signal.
- Test: count the same criterion on blank windings of a read scroll from the same campaign (Q3), and on PHerc. 0813.

**Q37 — The reviewer is an AI agent looking at montages.** [R4]
- Weak point: no papyrologist has seen any map; full resolution was used for top-ranked places only; the review was not blind.
- Test: Q21(a); and a human reader's look at the two closed row-like places.

**Q38 — A score once decided where compute went.** [R1]
- Weak point: in the first scans a window below 2.0 on a score that cannot see single lines got no second model and no depth sweep.
- Test: re-run those windows (about 60 cm²) with Studio at stride 48.

**Q39 — Rules changed after first contact with data.** [S1, F1, L1]
- The cases: the window gate (fraction ≥ 0.5 → core ≥ 4 cm², before any ink); the row criterion (mark size, on the target, before controls); the strokeness threshold and the control band (after a first run); the sheet-centre rule of the label-free features (after a first pass on one control).
- Test: re-run each analysis under the original rule and report what changes.

**Q40 — The line-pitch NO-GO may be a lack of controls.** [L5]
- Weak point: the statistic reached z 4.8 where the eye fails, against a bar of 5, and had no real second column to be tested on.
- Test: an exploratory pass over existing maps of two unread scrolls, with the false-alarm rate measured on the same maps by the rotated and the wrong-order versions.

**Q41 — A line as faint as the second PHerc. 0306B line is invisible to every screen here.** [C3]
- Weak point: no detector we can run shows that line beyond its first 1.5 cm.
- Test: none cheap. State it with every negative.

## E. Physics

**Q42 — Our two lead calculations disagree by a factor of about 25.** [L8]
- Test: run `code/pb_kedge_spectra.py`; vary the source and filter assumptions; then check the prediction on bright mineral grains present in both scans of PHerc. 1203 (their coarse-to-fine brightness ratio distinguishes the two models).

**Q43 — "No lead in the controls" was carried over to the unread scrolls.** [L8, L4]
- Weak point: lead at published levels would add 11–55 % column mass, visible without any model. We never ran a screen built for that, and the one label-free pass we ran was judged uninformative for a different reason.
- Test: inject strokes of +10 % and +30 % column mass into atlas stacks and check whether a simple maximum projection flags them; if yes, run that over the atlas and the fine bands.

**Q44 — The one label-free feature that worked at 2.4 µm was never run on an unread scroll.** [L1]
- Weak point: a depth top-hat gave AUC 0.73–0.76 on Scroll 1 at 2.4 µm; PHerc. 1203 and 0846A have 2.4 µm bands.
- Test: the same top-hat on the 39 blocks, with the PHerc. 0841 fine control alongside.

**Q45 — Column mass was killed on one scroll at 2.4 µm.** [L7]
- Test: the same measurement on PHerc. 0841 at 9.37 µm and on PHerc. 172.

## F. Coverage and records

**Q46 — We do not know what fraction of any scroll we screened.** [N]
- Weak point: bands and seeds were chosen for regular, well-separated wraps. If text survives mainly in compressed zones, our surfaces avoid it by design.
- Test: per scroll, estimate total sheet area from the surface prediction and report the screened fraction by radius and height.

**Q47 — A sweep in depth was never used as a screen.** [D1, F5]
- Weak point: on a written scroll, reader-v3 at ±6 layers shows strokes from neighbouring wraps; on Scroll 1 detectors read the next wrap 146 µm away. A wide sweep over a misplaced surface might therefore find text the mesh misses. We used ±3 layers on the atlas.
- Test: displace the PHerc. 0343 positive by 15 voxels, sweep ±20 layers, and see whether the letters reappear at the right offset; if so, sweep a sample of atlas windings.

**Q48 — Some negatives can no longer be re-reviewed.** [N, K6]
- The cases: full-resolution maps of the PHerc. 0813 and 0800 atlas (montages and half-resolution maps only); the Studio subset of PHerc. 0800 (lost before review); the complete-schedule fit of PHerc. 0358 (no meshes).
- Test: re-ink a random 10 % and compare with the stored montages.

**Q49 — The PHerc. 0175A spacing figure rests on six windings.** [S8]
- Weak point: the 12–41 voxel spread quoted for PHerc. 0175A was measured on six snapped windings only (see S8), not on all 53 that were read.
- Test: compute the row and column spacing percentiles for every winding of both fits and the local scale map; report what share of the read area was rendered outside 0.8–1.25 of true scale.

**Q50 — Things we decided not to run.** [several]
- The A-hn replay on the published PHerc. 0306B window set; ink on the same winding raw and snapped; Studio over the PHerc. 0813 and 0800 atlas; the 66 lower PHerc. 0175A windings and their unsnapped versions; a second fine-resolution control; the 30,000-step fit of PHerc. 0358; stage 2 of the line-pitch statistic; the intensity audit; any training.
- Test: pick one and run it; each is described in its entry under "Not done".

**Q51 — PHerc. Paris 4.** [T1–T3]
- Weak point: no detector is independent of that scroll; the lowest 4 cm of the sheet, the two lowest tiles, the outermost wraps and 30 % of the inner blank stretch are not covered at reliable sensitivity; the core was examined only through published predictions.
- Test: the outermost wraps in both orders; the other autosegmentation pass on the inner part.
