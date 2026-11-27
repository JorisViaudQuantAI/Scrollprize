# Calibration panel: awarded surfaces through our ink-detection jobs

Five surfaces from awarded scrolls (PHerc. 0846B, 0211, 0306B, 0343), each put through the same render and the same
four detector settings, in both orders. Use it to see what real text looks like through each setting, and to check a
detector path before you trust a negative result from it. All maps are at 20 px/mm.

## Items
| item | scroll | surface | volume (level 0) |
|---|---|---|---|
| p846 | PHerc0846B | the awarded segment `PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750`, full, 19.7 cm² | 20250804142305-9.362um-1.2m-113keV-masked |
| p211win | PHerc0211 | the awarded PHerc0211 mesh (tifxyz, 86 × 300) | 20250821151803-9.362um-1.2m-113keV-masked |
| p211fw033 | PHerc0211 | our own spiral-fit winding w033, flattened with villa flatboi (50 it, symmetric Dirichlet), crop of grid columns 68–228 | same |
| p306 | PHerc0306B | claim surface, `--grid-crop 115 70 235 75` | 20250521133212-8.640um-1.2m-116keV-masked |
| p343 | PHerc0343 | awarded winding, letter crop, `--grid-crop 40 20 240 76` | 20250521140437-8.640um-1.2m-116keV-masked |

## Render (identical for every item)
`render_tifxyz.py --layers 41 --step 1 --surface-interp bicubic --scale 1`. Sampling is trilinear. The mesh is at layer 20.
Normals are N = dP/du × dP/dv, negated (`flip_normals`, the renderer's default, like `vc_render_tifxyz --flip-normals`).
`--voxel-um` is 9.362 or 8.64, depending on the volume.

## Detectors
| name | what it is |
|---|---|
| `rv3old_{fwd,rev}` | **Our earlier reader-v3 path**: `infer_ink.py` on the 41-layer stack, surface layer 20, the 26-plane input window centred there, run at z-shifts −3 and +3 and averaged. Resampling is `auto` (9.362 → 9.6 µm; 8.64 → 9.6 µm). The checkpoint is reader-v3 `main` (sha256 ea405417…, identical to revision 39f626e5). Per-patch robust normalisation, overlap 0.5, Hann blend. |
| `villa_{fwd,rev}` | **The PHerc0846B winner's recipe** run on our render: villa `vesuvius.ink_detection.inference.infer` @ b29f310 in a Python 3.14 venv built from the winner's requirements.txt. Input is the central 17 layers (12–28) of our 41-layer stack. 384 px patches, stride 192, Hann blend, no resampling. Checkpoint reader-v3 @ 39f626e5. |
| `s96_{asc,desc}` | Studio packaged CLI (priyankaranke/vesuvius-studio-multiscroll-cpt @ bf66f753, weights `studio-softloss-20261023-ema`, sha256 96b32628…), fp16, `--mesh-index 20 --shift 0 --stride 96`, no blending. |
| `s48_{asc,desc}` | Same, `--stride 48`, Hann blend. |

`fwd` / `asc` are the reading order for p846, p211win and p343. `rev` / `desc` are the reading order for p211fw033 and
p306. In every case, the reading order is the order in which the text shows; the other order never shows text.

## Measures (tables/panel.tsv, tables/panel.json)
- `share_p_gt_0.5`: the share of pixels with p > 0.5, on the sheet-parallel tile mask. For rv3old it is the mean over the two shifts.
- `line_finder_max`: the best chain score of the PHerc0306B line finder (`at_atlas.py post`, glyph gA). It is computed
  on the z-scored map, on the same mask. `None` means no chain was found.
- `hot_z2`: the share of pixels with z > 2.
- `rows_legible_20px_per_mm`: a judgement by eye on the 20 px/mm maps (images/panel_*.jpg). The bottom-right panels of each image are the other order.

## Result on the PHerc0846B segment: our earlier path against the winner's published raw TIFF
Pearson r is computed on 4×-binned maps (tables/t1_0846B_correlations_raw.txt, `code/cmp.py`):
| map | r |
|---|---|
| winner's script re-run | 1.000 |
| our `infer_ink.py` on the winner's own 17-layer surface volume (z0, no resample) | 1.000 |
| our render + our infer (z0, no resample) | 0.935 |
| winner's recipe (villa) on our render | 0.931 |
| our earlier path (z ±3 averaged, resample 9.6) | 0.910 |
| any reverse map | 0.22–0.33 |
| Studio s48 asc / desc | 0.685 / 0.241 |

images/t1_0846B_winner_vs_ourpath_vs_s48asc.jpg shows three maps: the winner's map, our earlier path, and Studio s48 asc.
The same letters are legible in all three, and Studio s48 asc shows the most.

## Lessons
1. Our earlier reader-v3 path is equivalent to the winner's. The inference code gives identical output on identical
   input. The ±3 shift averaging and the 9.6 µm resample cost at most 0.03 in r, and nothing that can be seen.
2. Text shows in **one order only**, in every setting, on every item. Studio s48 gives the most legible letters.
   Studio s96 is usable, but the tile grid is visible.
3. **On dense text, the reading-order Studio map is saturated.** On p846, s48 asc has share p>0.5 = 0.32, and the
   z-scored "hot" share is only 0.002. The line finder returns None on s48 asc, even though the letters there are the
   clearest of the whole panel. Rankers built on z-scores or line chains can therefore miss dense, clearly written
   text. Look at the maps by eye.
4. A high share of p>0.5 is not text either. Saturated amorphous clouds also reach 0.2–0.3. What separates text from
   clouds is shape: connected strokes forming letters, in rows at a regular pitch.

## PHerc0846B: our earlier "C846" place (history)
Two days before the 0846B award, we read four GrowPatch windows of our own on this scroll. The windows were seeded at
z 9300 and 12200. In one of them, k0, Studio s48 showed rows of thin marks in one order only: 1.0–1.2 mm high, at a row
pitch of 4.1 and 5.8 mm. We closed this as a false lead.
What we know now:
- The k0 mesh is **not** on the winner's surface. The nearest distance is 874 voxels and the median 1863.
- k0 lies deep inside the flattened core, next to a cavity, and part of it bridges the cavity.
- Its grid quality was poor: |shear| ≤ 8° on only 30 % of cells (the winner's segment: 86 %). Roughness was greater
  than 20° on 37 % of cells (winner: 7 %).
- A fresh GrowPatch regrowth at the same place is equally poor: 42 % of cells with |shear| ≤ 8°, and 31 % with
  roughness greater than 20°.
- The winner's text lies on the outermost wraps at the top of the flattened roll, a few layers below the outer surface.
(The re-read of C846 and of two blind patches through this panel's jobs is reported in the follow-up.)

## Files
- `code/hb_read.sh`: the detector and post functions.
- `code/panel_env.sh`: the environment, cache and the `si_*` functions.
- `code/cmp.py`: the correlation script.
- `jobs/*.sh`: the exact jobs. They ran on one Colab L4 through our job bridge: `$INPUTS` and `$OUT` are bridge
  variables, and jobs with `# slot: cpu` are CPU jobs.
- The scripts `render_tifxyz.py`, `infer_ink.py`, `onsheet_score.py`, `si_tools.py`, `studio_t4.py`, `at_atlas.py` and
  `vw/fl/scripts/line_find.py` come from our internal package `wl_ink_v3.tgz`. They are not included here.
- Timings on the L4, Studio s48: about 0.6 min per cm² per order. The villa recipe: about 6 s per cm² per direction.
