# 07 — Negative coverage

What was screened, with which detectors and orders, and with which verdict word.
A row here means "under this pipeline, on this surface, nothing qualified". It does not mean "this papyrus is blank".
The rows agree with `collab/claude-012`, `-013` and `-017`; where an earlier note gave an interim figure, the difference is stated.

## How areas are defined

| column | definition |
|---|---|
| chart | mesh grid area (vertex count × cell size). For spiral windings this is nominal and can be off by 0.6–2× locally (S8). |
| valid | rendered pixels with CT data |
| on-sheet | for own windows: valid area in 1 mm blocks that pass the m7 pixel test (S1). For windings: valid area × `f_parallel` of the layer-coherence score (L2), or the area of its parallel tiles. Both are upper bounds, and they are different tests. |
| core | own windows only: largest connected on-sheet region (S1) |

Unique-sheet area is unknown: we never checked whether two surfaces lie on the same sheet. No depth-correct area exists either: "on-sheet" by L2 says nothing about depth offset.

<a id="n"></a>
## N — Coverage table (unread scrolls)

Areas in cm². "rv3" = reader-v3. "−3/+3" = two depth shifts, both maps and their mean kept.

| scroll | surface set | chart | valid | on-sheet | detectors, orders | verdict |
|---|---|---|---|---|---|---|
| PHerc. 0813 | 75 atlas windings | 437.0 | 407.3 | 316.9 | rv3, one order (by the face rule), −3/+3; Studio, one order, stride 96, on 5 windings (28.8 valid) | screened — no qualifying reading identified |
| PHerc. 0800 | 95 atlas windings | 501.9 | 476.8 | 367.6 | rv3, one order, −3/+3; Studio on 3 windings inked, results lost before review | screened — no qualifying reading identified (rv3 only) |
| PHerc. 0125 | 62 older atlas windings | 389.4 | 363.8 | 314.8 | rv3, one order, −3/+3; Studio both orders, stride 96, on 16 windings; Studio stride 48 both orders on 12 random windings (54.0) and the work-up of F1 | screened — no qualifying reading identified |
| PHerc. 0257 | 6 atlas windings | 22.7 | 22.1 | 19.2 | rv3 one order; Studio both orders, stride 96 | screened — no qualifying reading identified |
| PHerc. 0358 | 3 atlas windings | 11.1 | 10.8 | 9.3 | same | screened — no qualifying reading identified |
| PHerc. 0826 | 3 atlas windings | 14.1 | 13.8 | 12.0 | same | screened — no qualifying reading identified |
| PHerc. 0268 | 6 atlas windings | 19.6 | 14.8 | 4.5 | rv3 one order | uninformative (cut-through) |
| PHerc. 0175A | 53 snapped windings of our spiral fit, band z 6900–8400 (those above 0.40 raw on-sheet) | 520.1 | 520.1 | 399.4 | rv3 both orders, −3/+3, all windings; Studio stride 48 both orders on 17 crops of 30 mm (49.8); Hecate on one place | screened — no qualifying reading identified |
| PHerc. 0175A | 4 own windows, 3 screened | 35.1 | 34.4 | 20.7 | rv3 both orders, two shifts; Studio both orders, stride 96 | screened — no qualifying reading identified |
| PHerc. 0191 | 5 own windows, 4 screened | 55.0 | 54.4 | 26.3 | same | screened — no qualifying reading identified |
| PHerc. 0358 | 4 own windows, 4 screened | 55.0 | 53.3 | 29.7 | same | screened — no qualifying reading identified |
| PHerc. 0483B | 4 own windows, 2 screened | 23.4 | 22.3 | 11.9 | same | screened — no qualifying reading identified |
| PHerc. 0490A | 3 own windows, 1 screened | 11.7 | 11.3 | 5.6 | same | screened — no qualifying reading identified |
| PHerc. 0846B | 4 own windows, 1 screened | 13.7 | 13.6 | 5.7 | same, plus Studio stride 48 at five shifts, Hecate and `ink_9um` on one place (F2) | screened — no qualifying reading identified |
| PHerc. 1203 | 4 own windows, 2 screened | 27.5 | 26.8 | 16.7 | same; Studio stride 48 on the best core | screened — no qualifying reading identified |
| PHerc. 1218 | 4 own windows, 1 screened | 11.7 | 11.6 | 6.1 | same | screened — no qualifying reading identified |
| PHerc. 1545 | 5 own windows, 4 screened | 55.0 | 52.9 | 28.6 | same | screened — no qualifying reading identified |
| PHerc. 0826 | 2 own windows | 27.5 | 26.2 | 10.9 | same | uninformative (not one sheet) |
| PHerc. 0175A, 0358, 0813 | nine earlier own windows, three per scroll | 56 | not recorded | not measured | rv3 both orders × 3 shifts; Studio both orders × 3 shifts | eight screened — no qualifying reading identified; one uninformative (not one sheet) |
| PHerc. 1203 | three public `segments/raw` meshes at 9.36 µm, and a fourth small one | — | 36.7 + 2.8 | not measured (about a third is off-sheet periphery by eye) | rv3 and `ink_9um`, both orders, 3–5 shifts; deep read of the top window with four models; Studio both orders on two of the four | screened — no qualifying reading identified |
| PHerc. 0800, 0175A, 0125, 0490A, 0826 | early official meshes and small own patches | — | 22.9 | not measured | rv3 and `ink_9um`, both orders, 5 shifts | screened — no qualifying reading identified |
| PHerc. 1203 | public `segments/raw` meshes inside the 2.4 µm band, 39 blocks | — | 15.75 | 22–24 of 39 blocks on a clean fibre grid by eye | canonical 2.4 µm and Hecate 2.4 µm, both orders, native 2.4 µm | screened — no qualifying reading identified; 15–17 crumpled blocks uninformative |
| PHerc. 1203 | first of 17 fitted windings, at 2.4 µm | 1.99 nominal | 2.80 | not measured | both 2.4 µm models, both orders | screened — no qualifying reading identified |
| PHerc. 1203 | 16 more fitted windings, at 2.4 µm | — | 98.3 | — | both 2.4 µm models, one order, stride 192 | in progress |
| PHerc. 0358 | 8 snapped windings of the shortened spiral fit | 38.8 | — | — | none (CT showed cut-through) | uninformative |
| all 7 atlas scrolls | 250 atlas windings, label-free feature (L4) | 1,395.8 | 1,309.4 | — | one CT feature | uninformative for every scroll; **not coverage** |

Notes on the table.
- **Added 2026-11-23.** The atlas rows cover one winding in 20 of each fitted z-window (S2, hindsight test on PHerc. 0211). They are a sparse sample of those scrolls, not coverage of the regions they span.
- Own-window rows give the areas of the **screened** windows only; 39 windows in all, 22 screened (288 chart, 280 valid, 151 on-sheet, 98 in cores), 17 uninformative. These rows replace the interim per-scroll counts of `collab/claude-011`.
- The on-sheet figures for PHerc. 0813 and 0800 are 407.3 × 0.778 and 476.8 × 0.771; `collab/claude-010` gives them as "about 78 %" and "77 %".
- The nine earlier windows are 3 × 5.60 + 6 × 6.58 = 56.3 cm² of chart; one private report says "about 54".
- The 39 blocks: `collab/claude-017` says 22 on-sheet and 17 crumpled; the private list gives 23 and 16 at the verdict and 24 and 15 at the first triage. The difference is how two part-crumpled blocks are counted.
- Sums, for orientation only (surfaces may overlap; areas of fitted windings are nominal): atlas valid area with a "screened" verdict 1,294.6 cm², of which about 1,040 cm² sheet-parallel by L2; own windows 280 cm² valid, 151 cm² on a predicted sheet; PHerc. 0175A fit 520 cm² nominal, 399 cm² in parallel tiles.
- Boxes of the early negative regions, with volume ids, models, depths and faces, are in `firstletters-colab/negative_results/negatives.csv`.

## What each row leaves out

| limit | applies to |
|---|---|
| one layer order only with reader-v3 | all atlas rows |
| surfaces not snapped, several voxels off the sheet over most of the area | all atlas rows |
| Studio on a ranked subset only, or not at all | PHerc. 0813, 0800, 0125 atlas; 0175A fit (crops) |
| Studio at stride 96 only | own windows except four cores; small-scroll atlas rows |
| renders stretched or squeezed along the lines by 0.6–2× | PHerc. 0175A fit |
| on-sheet over about half the area | own windows; 0175A fit by m7 (0.41–0.69 raw) |
| full-resolution review of top-ranked places only | every row |
| reading order inferred from geometry | PHerc. 1203 fitted windings |

## Per eligible scroll: what this lane has and has not touched

| scroll | this lane | not touched by this lane |
|---|---|---|
| PHerc. 0125 | 62 older atlas windings; one small patch | the roughly 278 newer atlas windings (a teammate's lane) |
| PHerc. 0175A | spiral fit of one band (53 of 119 windings); 8 own windows (4 + 3 + 1); small patches | the rest of the height; the 66 lower windings |
| PHerc. 0175B | nothing | everything |
| PHerc. 0191 | 5 own windows | the public spiral-fit windings (a teammate's lane) |
| PHerc. 0257 | 6 atlas windings | the rest |
| PHerc. 0268 | 6 atlas windings, uninformative | effectively everything |
| PHerc. 0358 | 3 atlas windings; 7 own windows; a failed spiral fit | the rest |
| PHerc. 0483A | nothing | everything |
| PHerc. 0483B | 4 own windows | the rest |
| PHerc. 0490A | 3 own windows; one small patch | the rest |
| PHerc. 0490B | nothing | everything |
| PHerc. 0800 | 95 atlas windings; 6 small official meshes | the other order; Studio |
| PHerc. 0813 | 75 atlas windings; 3 own windows | the other order; Studio beyond 5 windings |
| PHerc. 0826 | 3 atlas windings; 2 own windows (uninformative); one small patch | the public spiral-fit bands (a teammate's lane) |
| PHerc. 0846A | nothing yet; a spiral fit of one band is running | everything |
| PHerc. 0846B | 4 own windows, 1 screened | the rest |
| PHerc. 1203 | 4 own windows; 3 public meshes at 9 µm; the fine band at 2.4 µm (39 blocks, 17 windings in progress) | most of the scroll outside the band; 48 fitted windings below the gate |
| PHerc. 1218 | 4 own windows, 1 screened | the rest |
| PHerc. 1545 | 5 own windows, 4 screened | the rest; it has no published tracks |

The fraction of each scroll's papyrus that these surfaces represent was **not measured** for any scroll.

---

# PHerc. Paris 4 title lane (a read scroll)

The title prize asks for the title of Scroll 1. The lane is closed, negative. The ghost-text result is F5.

<a id="t1"></a>
## T1 — Published predictions on the blank end stretch, and the unmeshed core

**Question.** Is a title visible in what the organisers already published: the 2-D and 3-D ink predictions on the blank stretch after the final column, and the two 3-D ink volumes inside the innermost meshed wrap?

**What was done.** Viewed the published 2-D maps (2.4 µm and 1.129 µm models) and the 3-D ink maximum projection on the blank stretch. For the core: polar maximum projections of both published 3-D ink volumes around the umbilicus, in five radial shells from the axis to 0.15 mm inside the innermost meshed wrap, over the whole height at 9.6 µm; plus a row-periodicity screen.

**Controls and nulls.** A projection ±96 µm around a meshed turn with a full text column (positive), and around a turn of the blank stretch (negative), drawn the same way.

**Numbers.** Control-valid height 105 mm for one model and 114 mm for the other. Row periodicity (median): text control 0.36 and 0.43; blank control 0.19 and 0.16; core shells 0.13–0.19. Papyrus left unmeshed in the core: about 2 cm (1.7–2.8 cm), 1.5–2.5 windings, from skeleton lengths at seven heights.

**Verdict.** Negative. No ink is detectable in the unmeshed core by either published 3-D model, and the blank stretch shows only speckle in the published maps.

**How this could be wrong or miss ink.** Both models were trained on this scroll's body text. Each core shell overlays several sheets. The periodicity screen is weak (the text control drops to blank level over part of the height). The top 2 mm are unreliable.

**Not done.** Any new inference in the core. The 1.129 µm volume there.

**Where the code is.** Not included.

<a id="t2"></a>
## T2 — Does the mesh of the blank stretch stay on one sheet?

**Question.** A title could be hidden by a mesh that switches sheets exactly there.

**What was done.** Four checks on the published two-pass segment: the organisers' winding annotations against the mesh; pass 1 against pass 2 (point-to-plane distance); self-pitch (distance to the neighbouring turn of the same mesh); CT profile along the normal. A combined flag map.

**Controls and nulls.** The same measures on the text columns.

**Numbers.** At the one annotated height, twelve winding annotations lie 6–15 µm from the mesh with a consistent turn count. Passes agree within 30 µm on 95 % of the text columns and 86 % of the blank stretch. Flagged area: 6 % of a text column, 9 % of the final column, 12 % of the outer blank stretch (57.8 cm²), 30 % of the inner blank stretch (73.7 cm²).

**Verdict.** The outer 5.6 cm is meshed as reliably as the text it follows. The inner 7.3 cm is less reliable in patches.

**How this could be wrong or miss ink.** Only one height is annotated. Each wrap is a bundle of 2–4 thin layers there, so a jump between bundles cannot be decided by eye at 9.6 µm.

**Not done.** A repeat of the detectors on the other pass over the inner part (cut for budget).

**Where the code is.** Not included.

<a id="t3"></a>
## T3 — Detectors over the blank stretch and under the final column

**Question.** Do detectors with a different training mix, the other layer order, other depths or another script size show anything where the published body-text maps do not?

**What was done.**
- Control first (C4). Then reader-v3 (both orders, shifts −2 and +2), Studio (both orders, stride 96) and Hecate (both orders, on the 12 top-ranked 1 cm windows) over the final-column box and the whole blank stretch of the second pass: 10 of 12 tiles. The two not run are the lowest tiles of the first and last tile columns.
- Under the short final column (about 29 cm²): reader-v3 with the window at 0, ±58 and ±125 µm, both orders.
- Input shrunk by 1.5 and 2 for a larger script, with the final column as the in-run control.
- reader-v3 at 19.2 µm on CPU over the same area, both orders, as an independent low-resolution pass.

**Controls and nulls.** The final column in every run. The other order. A survey of fold knees and wrap contacts for false positives.

**Numbers.** Blank tiles in the reading order: best 1 cm windows at z 1.8–2.2 against all blank windows; Hecate share above 0.5 ≤ 0.85 % in the reading order and ≤ 3 % in the other. Shrunk input: control still 13–14 % above 0.5; the boxes 0.2–0.8 %. The other order is busier than the reading order in cracked tiles (1.5–4.5 % above 0.5) and follows vertical cracks.

**Verdict.** Negative: no aligned group of marks at letter spacing anywhere in the blank stretch, down to where the published models stop reading their own text. The only structured responses under the final column are ghosts of the previous column (F5).

**How this could be wrong or miss ink.**
- No public detector is independent of Scroll 1. A title in an ink these models cannot see at all is not excluded.
- Studio was trained with a strip beside the final column labelled as background.
- The lowest roughly 4 cm of the sheet is a detection gap, not a blank: the published models fail their own text control there.
- 30 % of the inner blank stretch is flagged by T2.

**Not done.** The two lowest tiles. The outermost wraps (an initial title). The unmeshed core with new inference. The other pass on the inner part.

**Where the code is.** `code/ghostmatch_geom.py`, `code/ghostmatch_compare.py` for F5; the rest is not included.
