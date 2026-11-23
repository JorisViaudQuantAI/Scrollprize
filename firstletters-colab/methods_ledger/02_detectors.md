# 02 — Detectors applied zero-shot

We trained nothing. Every detector below is a released checkpoint, run as released, on our renders or on published ones.
"Zero-shot" means the detector has not seen the scroll it is applied to; C7 lists the exceptions among the controls.

| short name | public source | input | normalisation | license note |
|---|---|---|---|---|
| reader-v3 | Hugging Face `domenicor046/reader-v3` | 17 layers × 384 × 384 px, native 8.64–9.36 µm | per patch: clip to p1–p99, (x − median) / (1.4826 · MAD) | MIT |
| `ink_9um` | Hugging Face `scrollprize/ink_9um`, seeds 42 and 43 | 17 layers × 128 × 128 px | as reader-v3 (same architecture) | MIT |
| Hecate 9.6 µm | Hugging Face `scrollprize/hecate` | 16 layers × 64 × 64 px, resampled to exactly 9.6 µm | uint8 / 255 | MIT |
| Hecate 2.4 µm | same repository | 64 layers × 256 × 256 px at 2.4 µm | uint8 / 200 | MIT |
| Studio | release of the PHerc. 0211 winner (`studio-softloss-…-ema`), 223 M parameters | 16 planes × 128 × 128 px, native voxels | clip(x, 0, 200) / 255, absolute | weights CC BY-NC 4.0 |
| canonical 2.4 µm | Hugging Face `scrollprize/ink_canonical_2um` | 62 layers (23…85 of a 109-layer render) × 256 × 256 px | clip 0–200, / 200 | MIT |
| A-hn, TRACE v3B | releases of the PHerc. 0306B winners | — | — | not run by us (D5) |

All checkpoints loaded with `weights_only=True` or safetensors, except the canonical 2.4 µm Lightning checkpoint, which needs full unpickling.
The Studio weights were checked by SHA-256 against the release constant in every job.

---

<a id="d1"></a>
## D1 — reader-v3

**Question.** Does our inference code reproduce published reader-v3 maps, does the model read known text of a scroll it was not trained on, and how sensitive is it to order, depth and pixel size?

**What was done.**
- Our inference script (17-layer window centred on the mesh layer, 50 % overlap, Hann blending, fp16 on GPU, fp32 on CPU) was run on the published 28-layer surface volume of PHerc. 0841 w00 and compared with the organisers' published reader-v3 map of the same segment.
- AUC against the published 2.4 µm ink map of that segment, resampled to the 9.366 µm grid, "ink" = map ≥ 128, on a 2.6 cm² text region, every third pixel.
- Depth sweep −8…+8 layers in steps of 2, both orders; an off-sheet run at ±20 layers; a pure-noise stack with the render's per-layer mean and standard deviation.
- Pixel-size emulation: the native scan resampled so that one model pixel equals 7.2…11.5 µm.

**Controls and nulls.** Reversed layer order. Off-sheet depth. Noise stack. A comparison shifted by 8 px (r falls to 0.92, so the parity is not an artefact of smooth images).

**Numbers.**
- Parity with the published map, whole segment (20 Mpx): maximum difference 1 of 255, mean 0.010–0.016, r 0.999996.
- PHerc. 0841, forward order, by depth shift (layers):

| −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 | reversed (best shift) | off-sheet ±20 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.686 | 0.755 | 0.813 | 0.837 | **0.863** | 0.857 | 0.840 | 0.772 | 0.692 | 0.568 | 0.457 / 0.463 |

- On the organisers' own surface volume the same model gives 0.865 (reversed 0.574); on our render 0.863 (our render correlates only 0.88–0.91 with theirs).
- Pixel size, AUC: 7.2 µm 0.857; 7.91 µm 0.852; 8.64 µm 0.862; 9.0 µm 0.872; 9.366 µm (as is) 0.882; 9.6 µm 0.871; 10.4 µm 0.890; 11.5 µm 0.857 (on a 1.55 cm² sub-region, hence 0.882 at native size).
- Native scan against the 2.4 µm scan pooled to 9.6 µm: 0.882 against 0.895.
- Unexposed positives: PHerc. 0343 letter boxes 0.684 in the reading order, 0.601 in the other order; PHerc. 0306B claim box 0.572 and 0.503.
- PHerc. Paris 4 final column (training region): 0.90 / 0.94–0.95 against two answer keys; reversed 0.56–0.67.
- Tile-origin sensitivity on the PHerc. 0306B crop (array as is against padded by 96 px): r 0.985, mean |Δp| 0.012, maximum 0.20.

**Verdict.** Worked. This is our main screening detector: fast, tolerant, and it draws letter-shaped strokes on PHerc. 0841.

**How this could be wrong or miss ink.**
- The answer key on PHerc. 0841 is itself a model output from the finer scan.
- At ±6 layers on PHerc. 0841 the model still draws letter-like strokes **in other places**, plausibly text of neighbouring wraps. A deep sweep can import letters from another sheet.
- It does not show the second published line of PHerc. 0306B beyond its first 1.5 cm (C3). A line that faint on an unread scroll is invisible to our screen.
- Per-patch normalisation makes it immune to brightness shifts, and equally unable to use absolute brightness.
- Its training list includes rows named for PHerc. 0343P; whether that overlaps our PHerc. 0343 control is not settled (C7).

**Not done.** Test-time augmentation in the bulk screens. A labelled control on a real 8.64 µm scan other than the box-diluted positives.

**Where the code is.** `firstletters-colab/ink/` (inference script, already public). `code/letterbox_auc.py` for the box AUC.

---

<a id="d2"></a>
## D2 — `ink_9um`

**Question.** Do the organisers' official 9 µm checkpoints add anything beside reader-v3?

**What was done.** Same code path as D1 (the architecture is identical). Parity against the published map; AUC on PHerc. 0841; used as a "stability companion" in the first scan (Dice between thresholded maps).

**Controls and nulls.** As D1.

**Numbers.**
- Parity: maximum difference 2 of 255, r 0.99998.
- PHerc. 0841 AUC: 0.72–0.75 single checkpoint; 0.752 for two seeds averaged; reversed 0.54–0.56.
- Mean of reader-v3 and `ink_9um`: 0.840, below reader-v3 alone (0.863).
- Native scan against pooled fine scan: 0.744 against 0.841 (the largest gap of the three models).
- On the place of F2 it shows flat mottle in both orders.

**Verdict.** Negative as a screen on scrolls it has not seen: it reads texture. Dropped after the first scan, kept for follow-ups.

**How this could be wrong or miss ink.** The early "cross-model" gate (Dice with `ink_9um` ≥ 0.3) required agreement with a detector that fails on PHerc. 0841. That gate affected only automatic labels, never which previews were opened.

**Not done.** Other training steps of the checkpoint series.

**Where the code is.** As D1.

---

<a id="d3"></a>
## D3 — Hecate (9.6 µm)

**Question.** Hecate was part of the winning ensembles. Can it be run at useful speed on a small GPU, and does it agree with reader-v3?

**What was done.** Wrapper around the official `hecate.py`; inputs resampled to 9.6 µm. Speed measured under three autocast settings. Later a `--precision fp16` patch for the official script, measured on a 27 × 999 × 999 crop of PHerc. 0841.

**Controls and nulls.** fp32 output as the reference for fp16 and bf16.

**Numbers.**

| setting, Colab T4 | wrapper throughput | patched official script, wall time per fresh process | difference from fp32 |
|---|---|---|---|
| bf16 autocast | 12.2 cm²/h | 228.6 s | maximum 11–13 of 255, r 0.9999 |
| fp32 | 17.9 cm²/h | 123.6 s | — |
| fp16 autocast | 69.9 cm²/h | 41.2 s | maximum 1 of 255, r 0.99999 |

- PHerc. 0841 AUC by shift: −4 0.782; −2 0.814; 0 0.846; +2 0.787; +4 0.737; reversed 0.552.
- L4, fp16: 134 cm²/h per pass.
- fp16 produced NaN in one 64 px tile of a PHerc. Paris 4 control (0.1 % of the map); non-finite values are now treated as invalid.
- On PHerc. 0343 letters Hecate is weak: hot share 0.05–0.07 in the row band, strokeness 1.11–1.35 (F1).
- On PHerc. 0125 Hecate does not show the face asymmetry that Studio, reader-v3 and A-hn show (F6).

**Verdict.** Worked technically. Used only for deep reads and follow-ups, never in a bulk screen, because it stays about 5–15× slower than reader-v3 (T4: 41–70 against 600–730 cm²/h; L4: 134 against 480–790).

**How this could be wrong or miss ink.**
- "Hecate does not reproduce it" was one of our reasons for closing F2 and part of F1. Hecate barely shows the PHerc. 0343 letters, so its silence is weak evidence.
- PHerc. 0841 w00 sits in Hecate's published training bucket; its 0.846 there is not a held-out figure.
- Hecate is sharper in depth than reader-v3 (0.846 → 0.737 at +4), so a surface a few voxels off costs it more.

**Not done.** Hecate over any atlas or spiral surface. A check of our wrapper against the official script on identical input beyond the crop above.

**Where the code is.** `firstletters-colab/hecate_fp16/` (already public).

---

<a id="d4"></a>
## D4 — Studio (the PHerc. 0211 winner's detector)

**Question.** Does the released Studio model run on a 15 GB or 23 GB GPU, which 16 planes must it be given, and how stable are its maps?

**What was done.**
- Model and `predict` copied from the release; our wrapper gives bit-identical output to the release function on CPU in fp32 (maximum difference 0.0).
- Input: `clip(x, 0, 200) / 255`, 16 planes, 128 px windows. Stride 96 without blending is the release's own tiling; stride 48 with Hann blending is ours.
- Plane convention found on PHerc. 0841 with two renders of the same mesh that have opposite normal signs (D7).
- Later runs use the packaged tool `firstletters-colab/studio_t4/`, which writes float maps, validity masks and receipts.
- Tile-origin test on the PHerc. 0306B crop: stride 96 at origins (0, 0) and (48, 48). The tool has no origin option, so the stack was padded by 48 zero pixels on top and left and the map cropped back.

**Controls and nulls.** The other plane order. Depth shifts. fp32 as the reference for fp16 and bf16. Two tile origins at the same precision.

**Numbers.**

| measurement | value |
|---|---|
| T4, batch 1, fp32 | 340 ms per window, 11.35 GB allocated |
| T4, batch 1 / batch 16, fp16 | 132 / 105 ms per window, 1.17 / 5.7 GB |
| L4, batch 16, fp16 | 83 ms per window, 7.6 GB; fp32 at batch 4: 96 ms, 15.7 GB |
| fp16 against fp32 | maximum \|Δp\| 0.0019–0.0031, same AUC to four digits |
| bf16 against fp32 | maximum \|Δp\| 0.035 (L4); 0.013 on another crop |
| PHerc. 0841 (a **training donor**), reading order | 0.916 at stride 96; 0.924 at stride 48 |
| same, shifted by ±1 / ±2 / ±4 / ±6 planes | 0.91 / 0.90 / 0.86 / 0.81 |
| same, other order | 0.568 |
| PHerc. 0343 letter boxes, reading order, shifts −4…+4 | 0.711–0.745 (0.745 at stride 48); other order 0.640–0.716 |
| PHerc. 0306B line 1, reading order | 0.653–0.706; other order 0.42–0.55 |
| tile origin (0, 0) against (48, 48), stride 96 | r 0.888, mean \|Δp\| 0.058, maximum 0.70 |
| line-finder score of the same known letters, stride 96 against stride 48 | 3.82 against 9.48 (0343); 8.26 against 10.40 (0306B) |

- The origin (0, 0) run reproduced an earlier GPU map exactly, so the tool is deterministic.
- An earlier comparison in `collab/claude-011` (r 0.980 and 0.964) mixed a tile-grid change with a precision change and stays labelled as confounded.

**Verdict.** Worked. Studio draws known letters as sharper strokes than reader-v3 and scores higher on both unexposed positives.
Rule adopted: stride 96 answers "is there a row here"; shapes and line-finder scores are judged on stride-48 maps only.

**How this could be wrong or miss ink.**
- Studio uses absolute intensities. We never compared the CT histograms of the unread scrolls with those of its training donors; a scroll outside that range could be read badly without any sign.
- Most Studio screening was done at stride 96, where mark shapes depend on the tile grid. One row-like structure (F2) appeared only at stride 48.
- Studio was run on ranked subsets chosen from reader-v3 maps, so places only Studio would see were seldom looked at with Studio.
- Studio produces fine curved marks on clean woven papyrus. The size of those marks (1.0–1.2 mm) equals its 128 px window at these voxel sizes; whether they are per-window responses joined by blending was not tested.
- Its depth tolerance was measured on a training donor only.

**Not done.** Studio at stride 48 over whole windings. Studio over the PHerc. 0813 and 0800 atlas (one order at stride 96 would be about 4.5 T4-hours for 880 cm²). A test at a changed input scale.

**Where the code is.** `firstletters-colab/studio_t4/` (already public, with tests).

---

<a id="d5"></a>
## D5 — A-hn and TRACE (not run by us)

**Question.** Would the PHerc. 0306B winners' own models change any of our conclusions?

**What was done.**
- We did not run A-hn or TRACE. A replay of the three A-hn finalists on the published window set was requested by a teammate and never run by us; teammates ran it (`collab/codex-010`, `collab/gemini-001`).
- We evaluated the 0306B team's **published maps** on the claim surface with our per-row metrics (C3), cropped and pixel-aligned to our frame.
- A teammate ran A-hn beside Studio on PHerc. 0125 and PHerc. 1203 surfaces; the numbers are in `collab/gemini-005` and `-006`.

**Controls and nulls.** The team's published decoy model (labels moved by 3.5 × 3 mm).

**Numbers.** On PHerc. 0306B line 2 (the line read as one uncertain letter): A-hn mean AUC 0.628, TRACE v3B 0.673, their product 0.657, against 0.54–0.58 for reader-v3 and 0.54–0.63 for Studio.
The decoy model makes the left end of line 2 stand out most of all (AUC 0.82 there).
On PHerc. 0125 a teammate measured about 2.2× more A-hn response in one order than in the other, with no strokes.

**Verdict.** Not run by us. From the published maps, no model shows line 2 as a line of letters.

**How this could be wrong or miss ink.** A-hn has never been run on the 884 cm² of PHerc. 0813 and 0800 atlas that we screened with reader-v3 only.

**Not done.** Everything above that says "not run".

**Where the code is.** None of ours.

---

<a id="d6"></a>
## D6 — The 2.4 µm models (canonical model and Hecate 2.4 µm)

**Question.** Do the organisers' fine-resolution models run correctly on stacks we render from the 2.4 µm rescans?

**What was done.** A minimal re-implementation of the organisers' inference for the canonical model (layers 23…85 of a 109-layer render, clip 0–200, tile 256, stride 128, Hann blending, fp16), and the official Hecate script for the 2.4 µm checkpoint. Details of the production pipeline are in [06_fine_resolution.md](06_fine_resolution.md).

**Controls and nulls.** The organisers' published map of the same model on PHerc. 0841. Reversed layer order. Window shifted by ±8 fine layers.

**Numbers.**
- Our output against the published map on a 0.54 cm² text region: r 0.929, mean absolute difference 13.8 of 255, fraction above 128 0.245 against 0.233.
- Reversed order on that text: fraction above 128 falls from 0.245 to 0.029 (8.5× fewer ink pixels).
- Against published ink labels (1.04 cm² supervised): canonical 0.975, Hecate 2.4 µm 0.930; other order 0.51–0.58.

**Verdict.** Worked as a path check. PHerc. 0841 is training-family data for both models, so these numbers say nothing about a scroll they have not seen.

**How this could be wrong or miss ink.** There is no public 2.4 µm control with labels that these models did not train on. Their sensitivity on PHerc. 1203 or 0846A is unmeasured.

**Not done.** The older PHerc. 1667 iteration models and the 3-D ink models at 2.4 µm.

**Where the code is.** Not included (thin wrappers around the organisers' code).

---

<a id="d7"></a>
## D7 — Layer order and plane conventions

**Question.** For each way of rendering, which order of layers shows known text?

**What was done.** The same PHerc. 0841 mesh was rendered with `vc_render_tifxyz` (41 layers, no `--flip-normals`) and with our own renderer (65 layers, normals negated). The two are the same pixels with the layer axis reversed (layer j of one = layer 52 − j of the other, slope −1.000, per-layer r 0.95–0.97). Each detector was run in both orders on each. The result was then checked on PHerc. 0343, PHerc. 0306B and PHerc. Paris 4.

**Controls and nulls.** The other order in every case.

**Numbers.** Orders that read:

| render | reader-v3 | Studio 16 planes | checked on |
|---|---|---|---|
| `vc_render_tifxyz`, 41 layers, mesh at 20, **no** flip | reverse | descending 28 → 13 | 0841 (0.916 against 0.568), 0343 |
| flipped normals, 41 layers, mesh at 20 (our renderer; the spiral lanes) | forward on 0343, reverse on 0306B | ascending 12 → 27 on 0343, descending 28 → 13 on 0306B | 0343, 0306B |
| published 31-layer atlas stacks (flipped normals, mesh at 15) | forward | ascending 7 → 22 | by rule for all 250 windings; checked through 0343 |
| published surface volumes of PHerc. Paris 4 (109 layers at 2.4 µm, pooled to 27 planes) | forward | ascending | final column |
| 2.4 µm renders, 109 layers, flipped normals | "fwd" for both 2.4 µm models | — | 0841 |

- The two positives read in **opposite** orders in the same renderer, because their meshes have opposite normal signs relative to the written face.
- So the order is a property of mesh plus renderer, not of the renderer alone. On a mesh of unknown orientation both orders must be run.

**Verdict.** Worked. The convention is unambiguous on a mesh of known orientation (wrong order 0.50–0.61).

**How this could be wrong or miss ink.**
- For spiral windings we infer the face from geometry (normal towards or away from the umbilicus), not from known text on that scroll.
- The wrong order is not a clean null: PHerc. 0343 letters remain weakly visible in it (box AUC 0.60 for reader-v3, 0.64–0.72 for Studio).
- On blank papyrus the two orders are not symmetric either: PHerc. 0306B background hot share is 0.02–0.04 in the reading order and 0.07–0.09 in the other; PHerc. 0343 far-field is 0.04 in the reading order and 0.01–0.02 in the other. The sign differs between the two positives.

**Not done.** A positive on a 9.36 µm scroll other than PHerc. 0841.

**Where the code is.** Conventions are documented in `firstletters-colab/studio_t4/README.md`.

---

<a id="d8"></a>
## D8 — Depth-shift tolerance

**Question.** How far can the surface be from the sheet before a detector stops showing known text?

**What was done.** Depth sweeps on the positives, moving the detector window along the normal.

**Controls and nulls.** None beyond the sweep itself; all positives are on the sheet.

**Numbers.**

| detector, positive | at 0 | ±2 | ±4 | ±6 | ±8 |
|---|---|---|---|---|---|
| reader-v3, 0841 (AUC) | 0.863 | 0.837 / 0.857 | 0.813 / 0.840 | 0.755 / 0.772 | 0.686 / 0.692 |
| Hecate 9.6, 0841 (AUC) | 0.846 | 0.814 / 0.787 | 0.782 / 0.737 | — | — |
| Studio, 0841 (AUC, training donor) | 0.916 | 0.901 / 0.896 | 0.867 / 0.860 | 0.808 / 0.806 | — |
| Studio, 0343 boxes (AUC) | 0.742 | 0.744 / 0.734 | 0.732 / 0.711 | — | — |
| reader-v3, 0306B line 1 (hot share; background 0.02–0.04) | 0.28 | 0.30 / 0.26 | 0.30 / 0.22 | 0.30 / 0.14 | — |
| Studio, 0343 letters, strokeness (F1) | 1.88 | 1.83 / 1.56 | 1.48 / 1.17 | — | 0.67 / 0.59 |

- By eye, reader-v3 keeps the letters of both unexposed positives from −6 to +6 layers, weaker at the ends.
- A smooth injected displacement of −3.9…+5.9 voxels changed reader-v3's box AUC on PHerc. 0343 by 0.001 (S9).
- Production settings that follow: atlas, shifts −3 and +3; own windows, brightest layer −1 and +1 for reader-v3 and the brightest layer for Studio; spiral windings, −3 and +3.

**Verdict.** Measured, on few positives. reader-v3 is broad; Hecate is the sharpest; Studio is in between.

**How this could be wrong or miss ink.**
- "Two shifts of ±3 cover offsets of −9…+9" extrapolates the ±6 figure; at the edge of that range the hot share on PHerc. 0306B is already halved.
- At ±6–8 layers real letters turn into rows of dots in Studio (strokeness below 1). A surface that far off would show dots, which is exactly the form of F1.
- All of this is for surfaces parallel to the sheet. Tolerance to an **oblique** surface was never measured.

**Not done.** A tilt test. A sweep with the surface, not the window, displaced.

**Where the code is.** `code/letterbox_auc.py`, `code/strokeness.py`.

---

<a id="d9"></a>
## D9 — Combinations of detectors

**Question.** Does averaging or multiplying detector maps help?

**What was done.** On PHerc. 0841, 305 combinations were scored: model sets × (mean, geometric mean, mean of z-scores) × (single depth, mean over −2/0/+2).

**Controls and nulls.** Reversed order for the combined maps.

**Numbers.** Best: reader-v3 + Hecate, mean, depth-averaged, 0.891 (reader-v3 alone 0.863; depth-averaged alone 0.874). All four models with the mean of z-scores: 0.875. reader-v3 + `ink_9um`: 0.840.

**Verdict.** Inconclusive. The 0.891 is the best of 305 chosen on the same region against another model's map, so it is a model-agreement figure with post-hoc selection.
The combination was frozen as "reader-v3 + Hecate, mean, depth-averaged over −2/0/+2" before any use elsewhere, and in practice was used only in two early deep reads.

**How this could be wrong or miss ink.** We never used a product rule (as the 0306B winners did) or a reader-v3 × Studio combination on unread surface, so any gain from combining is untested in our screens.

**Not done.** reader-v3 × Studio as a fixed deep-read rule, measured on PHerc. 0343 and 0306B.

**Where the code is.** `firstletters-colab/controls/analyze_controls.py` (already public).
