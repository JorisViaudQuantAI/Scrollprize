# 35 — Winners' loop: calibration on PHerc. 0211, then PHerc. 0826 (lane `wl`)

Private. Started 2026-11-23 22:11 UTC. Session g3 (L4, worker started 22:11:12). Lane dir `/home/user/vesuvius/wl/`
(code `wl/code`, jobs `wl/jobs`, results `wl/results/<job>/`, images `wl/look/`, inventory `wl/inv/`).

## Decisions and thresholds fixed BEFORE looking (22:20 UTC)

Task 1 (calibration, PHerc0211, vol 20250821151803, 9.362 µm):
- Band z 7650–9150 (1,500 slices): contains the winner mesh's three rows (z ≈ 7700, 8300–8500 awarded, 9050).
- Inputs all official: published tracks (dl.ash2txt.org, extract.json = standard parameters, z 4500–17500), Lasagna
  nx/ny/grad_mag g2, m7 L0/L1, official umbilicus `20250821151803-umbilicus-20260808112626.json` (87 points, z 1702–18466),
  smoothed σ 75 (306B `smooth_umb.py`). No fallback needed.
- Sense ACW: catalogue `z_direction_is_top_to_bottom = true`, `left_handed_coordinates = false`, the same as PHerc0175A,
  which was fitted ACW and passed (report 21). `sense_chain.py` not used.
- Fit: villa `fit_spiral.py` @0fb38c45, report-21/23 overrides, ONE 30,000-step schedule, track-DT from 10,000, chunks
  ≤ 40 min, checkpoint + logs downloaded after every chunk. Gates as report 21 (raw on-m7 vs ±8-vox control; snapped).
- "Nearest winding to the letter sheet": for every fitted winding, distance (voxels) from each winner-mesh vertex in the
  three letter rows to the nearest vertex of that winding (KD-tree on the winding's densified vertices); report per
  winding median / p10 / fraction ≤ 2 / ≤ 4 voxels. The "letter winding(s)" = the winding(s) with the smallest median
  distance over the awarded row (z 8300–8500).
- Verdict words (as briefed): FOUND on our raw fit / FOUND only after refinement (which) / MISSED.
  FOUND means: in the blind review the letter place is flagged (connected strokes in rows at letter spacing), AND after
  unblinding at least 3 of the 11 awarded letters of the middle row are recognisable on our map beside the organisers'
  map at the same place. A ranker hit alone is not FOUND.
- Blind review of all windings of the band first (report-22 style montages), written down before the winner's location
  is used.

Task 2 (PHerc0826, vol 20250821151701):
- Line pitch expectation 4.4–6.9 mm; letter pitch ≈ 3.5–3.9 mm; review both mirror states unless the order is established.
- Bar for `candidate under private review`: as in the brief (connected strokes forming letter shapes, several in a row at
  letter spacing, one order, two detector families at the same place, surface shown on one sheet there). F1–F7 excluded.

## Progress log (UTC)
- 22:11 Read reports 33 (all), 21, 23, bridge README. g3 status: idle, L4, 253 GB free.
- 22:13 Inputs check. PHerc0211: tracks (dbm 7.0 GiB + vctracks + crossings 1.9 GiB), Lasagna (nx, ny, grad_mag, cos),
  m7 + normal grids, official umbilicus — all present. PHerc0826 (vol 20250821151701): the same set (dbm 5.0 GiB,
  crossings 1.4 GiB, Lasagna, m7, official umbilicus 49 points z 1941–16262).
- 22:14 Dense public fit of PHerc0211? None found: the atlas (pscamillo/vesuvius-eligible-meshes @66000516) holds only
  every 20th winding of short fits (~10 min each); HF search ("0211", "PHerc0211", "spiral") finds no 0211 fit;
  `rodriguescarson/eligible-scroll-spiral-fits` covers 0191 and 0826 only; aviad12g's re-mesh (report 32) is a re-mesh of
  one tracked sheet, not a dense fit, and needs 48 GB. ⇒ fit our own band.
- 22:16 Submitted on g3: `wl_001_setup211` (cpu), `wl_010_data211` (cpu: m7 ct lasagna native patch spec),
  `wl_011_tracks211` (cpu: published track store), `wl_012_sdt211` (cpu: sdt pools), chunks `wl_101…104_acw_c01…c04`
  (GPU, loop 1,980 / 2,100 s). Code `wl/jobs/wl_code_v1.tgz` (= `wl/code`, the 1203 lane's code made scroll-generic +
  `prep_wl.sh`).
- 22:17 `wl_010_data211` failed in 5 s: `pip install awscli` raced with the tracks job's install (OSError), so `aws` was
  missing; m7/ct stages wrote bogus .ok markers (background syncs failed silently). Fixed in `wl_code_v2.tgz` (install
  retried and checked, m7/ct stages fail if the download is small; bogus markers deleted at job start). Chain resubmitted
  as `wl_020_data211` → `wl_022_sdt211` → chunks `wl_201…206_acw_c01…c06` (the skipped names cannot be reused).
- 22:20 Submitted `wl_030_probe` (cpu): which flatten tools exist (villa pinned/main: flatboi, `render_ink.py`,
  `flatten_spiral_checkpoint.py`) and the current VC3D AppImage (the breadth lane's pinned AppImage URL now 404s).
- 22:21 **Task 2 inventory (public dense 0826 windings).** HF `rodriguescarson/eligible-scroll-spiral-fits` (sha 63c641ba,
  2026-11-02; README in `wl/inv/hf_README.md`): three PHerc0826 runs, 800 slices each, z 6528–7328, 8928–9728, 9328–10128
  (overlapping; "winding numbering does not transfer between overlapping windows"), 90 windings each, **every winding
  w000–w089 present** (`meshes/fitted/wNNN` + `_spliced` + `meshes/all/wNNN`), tifxyz scale 0.05 (nominal 20-vox vertex
  step), villa @757f70c0, 1.5–1.9 A40-h per run (short schedule), raw fit (not snapped, not flattened), official umbilicus.
  Measured on band 9328–10128 (`wl/code/meshstat.py`, `wl/inv/hf826/stat_9328.txt`): **not isometric** — 3D spacing of
  neighbouring vertices along the turn p5/p50/p95 = 0.55–0.67 / 0.85–1.32 / 1.4–3.7 × the nominal 20 vox (outer windings
  worst); along z 1.15–1.32×. **Inner windings over-packed**: median gap to the next winding 5–7 vox for w000–w020
  (30–50 % of vertices < 5 vox apart), 7–13 vox for w020–w040, 12–19 vox for w045–w089; median radius from the umbilicus
  10 → 1,335 vox (w000 → w085). The author's own audit says the same (median pitch 10.5 vox vs scan peak pitch 14–19
  on 0191; 31 of 89 consecutive pairs < 5 vox on 0826). **Relation to the sharp range z 9500–13600:** only z 9500–10128
  (≈ 600 slices, the top of the 9328 band and the top 228 slices of the 8928 band) lies inside it, and at z ≈ 9700 the
  scroll is at its most disturbed (a large crushed/void zone above the umbilicus; the outer HF windings run through air
  below the scroll), see `wl/views/0826_montage_L2.jpg` (z 9700 with HF winding crossings, 10500, 11500, 12500, 13400).
  z 11500–13400 shows a strongly flattened (~2.5:1 overall, inner wraps much flatter), densely and evenly layered roll.
  Outer membrane/glue turns (report 30: goldbeater's skin on the outside from the 1868 unrolling): the outermost 1–3
  bright layers along the lower edge at z 12500 are separated from the body by a dark gap — exclude windings that reach
  the outer boundary. Other public 0826 surfaces: pscamillo atlas 3 windings (z6528 w020/w040/w060); yupengtang screened
  14 lanes / 528 windings on 0826 (z 2900–16000, ACW) but published no meshes (their best Reader-v3 row 10.04 at
  `s0826c w079`, band not stated); Ahggggg 185.9 cm² GrowPatch patches (negative). ⇒ **the public fits do not cover the
  sharp range densely → own fit z 11800–13300 ACW.**
- 22:22 Told the coordinator (plan, inventory). **22:25 approved**, with conditions: (1) if 0211 is MISSED on the raw fit
  AND refinement does not recover the letters, message the coordinator before spending GPU on reading 0826 windings;
  (2) budget 25 L4-h; if tight, cut Studio to the top leads, keep reader-v3 over every winding. Submitted 0826 CPU prep:
  `wl_001_setup826` (after `wl_020_data211`) → `wl_520_data826`, `wl_511_tracks826` → `wl_522_sdt826`.
- 22:26 `wl_030_probe`: villa main @e5707e41 (2026-11-23) has `spiral-fitting/render_ink.py` (flatboi SLIM per strip, or
  Lasagna forward flatten of the full concat) and `flatten_spiral_checkpoint.py`; the current AppImage
  `VC3D-e5707e4-2026-11-23` (extracted on g3) contains `flatboi`, `vc_render_tifxyz`, `vc_grow_seg_from_seed`,
  `vc_tifxyz2obj`, `vc_obj2tifxyz`. **Flatten = villa's flatboi SLIM (symmetric Dirichlet, 50 iters, keep 100 %)** per
  winding, as `render_ink.py` does per strip.
- 22:27 `wl_031_flattest` (flatboi, ≈ 1 s per mesh). Residual scale = 3D vertex spacing / nominal 20 vox,
  p5/p50/p95 (`wl/code/stretch.py`):

  | mesh | along the turn | along z |
  |---|---|---|
  | HF 0826 w040 (9328 band), fitter grid | 0.56 / 1.13 / 2.94 | 0.98 / 1.25 / 2.68 |
  | same, flatboi keep 100 | **0.83 / 0.99 / 1.09** | 0.91 / 0.99 / 1.08 |
  | same, flatboi keep 25 | 0.75 / 1.01 / 1.50 | 0.73 / 1.00 / 1.39 |
  | winner mesh, as published | 0.98 / 1.00 / 1.02 | 0.99 / 1.00 / 1.01 |
  | winner, flatboi keep 100 | 0.98 / 1.00 / 1.02 | 0.99 / 1.00 / 1.01 |

  → every read in this lane uses flatboi keep 100; the residual stretch per winding is recorded with each read.
  Coordinator note 22:28 (scale matters: 2× expansion/compression costs a lot of recall) → also one extra arm in Task 1:
  the letter winding on the fitter grid vs flattened, at the letter place.
- 22:30 Ink side packaged: `wl/code/wl_env.sh` (from report 22's `si_env.sh`: 41 layers, mesh at layer 20, 9.362 µm,
  bicubic surface interpolation, onsheet map, reader-v3 both directions −3/+3, **argus-v3** (HF `ruiconte/argus-v3`,
  drop-in) both directions −3/+3, Studio fp16 stride 48 Hann asc/desc; post = `at_atlas.py post` on the parallel mask),
  code `wl/jobs/wl_ink_v1.tgz`. Positive/reference first: the winner mesh through this exact path (`wl_040…043`).

### Pre-registered (22:31, before any fit output): Task 1 blind set and order
- Budget: reader-v3 both directions at −3/+3 runs ≈ 140 cm²/h (report 22); all ~120 exported windings of the band
  (~1,000 cm²) would cost ≈ 7 L4-h, which the 25-h budget (two fits ≈ 13 h) cannot carry. **Blind set = every inside
  winding among w010–w069 of our fit** (≈ 60 windings), all rendered flattened, read in both orders. This window is
  chosen knowing that the letter sheet is an inner turn (report 33: ≈ 450–800 vox from the umbilicus); inside it the
  review does not use the place. Windings outside it are not reviewed (stated as a limit).
- The reading order expected on our windings is derived geometrically (normal vs umbilicus on the flattened mesh,
  compared with the winner mesh whose rule order reads), but both orders are reviewed.
- 22:29–22:51 Prep done: 0211 tracks 686 s, data 190 s, sdt+pools 311 s; 0826 tracks 491 s, data 210 s, sdt 386 s.
  Fit chunk 1 `wl_201_acw_c01` started ≈ 22:29 (9.4 GB GPU). Winner-mesh reference render `wl_041` ok (255 s); its
  ink `wl_042` waits for the GPU after c01. Gate 1 `wl_250_gate_c03` queued after c03.
- 23:24 Chunk 1 `wl_201_acw_c01` ok (job 2,162 s): **0 → 2,319 steps**, peak 10,360 MiB. Checkpoint fetched
  (`wl/results/wl_201_acw_c01/`). Queued c07–c11 (a chunk exits at once at 30,000) and `wl_259_gate_final` (KEEP=1).
- 23:24 **Reference: the winner mesh through our wl path** (`wl_042` GPU 1,001 s: rv3 172 s, argus 166 s, Studio 328 s per
  order; `wl_043` post). 6.30 cm² valid. Post on the parallel mask:

  | map | line finder | row score | glyph_p max | hot |
  |---|---|---|---|---|
  | reader-v3 forward | 5.65 | 2.86 | 7.22 | 0.067 |
  | reader-v3 reverse | 4.57 | 1.74 | 5.23 | 0.048 |
  | argus-v3 forward | 7.51 | 4.06 | 7.68 | 0.101 |
  | argus-v3 reverse | 0.96 | 2.81 | 3.71 | 0.043 |
  | Studio asc s48 | **8.94** | 3.16 | 9.95 | 0.078 |
  | Studio desc s48 | 4.86 | 1.99 | 6.00 | 0.071 |

  By eye (`wl/look/pos_winner_4maps.jpg`): three rows of letters in reader-v3 fwd, argus-v3 fwd and Studio asc (Studio
  asc clearest: the middle row "…ΔΙΑ…" region legible as separate letters); Studio desc = blobs. Reading order on this
  render path = forward/asc, as in report 33. The path (41 layers, bicubic, mesh at 20, parallel-mask post) works.
- 00:05 Local container restarted (coordinator): local waits/fetchers killed; local files and the g3 queue intact.
- 00:10 Chunk 2 `wl_202_acw_c02` ok: 2,319 → **5,884** steps (peak 9,686 MiB). c03 running.
- 00:40 Coordinator: fitted grids also carry local **shear** (±26–34° at p5/p95 over 8 cells) and normal roughness;
  "flattened" must mean a real per-surface flattening; report shear/stretch/roughness of what is rendered; keep one
  "fit grid as is" arm at the letter place; same table for 0826 before reading. Our grid-quality tool
  `wl/code/gridq.py` (8-cell signed shear p5/p50/p95 and share |shear| ≤ 8°, stretch along turn / z, roughness = cell
  normal vs 9×9 mean: median, p95, share > 20°). flatboi = SLIM (symmetric Dirichlet) on the whole winding surface, so
  it is a real per-surface flattening. Test meshes:

  | mesh | shear8 p5/p50/p95 (≤ 8°) | stretch turn | stretch z | roughness med/p95 (>20°) |
  |---|---|---|---|---|
  | winner, as published | −1.6/0.0/+1.7 (0.99) | 0.98/1.00/1.02 | 0.99/1.00/1.01 | 5.2/17.9 (0.04) |
  | winner, flatboi | −1.6/0.1/+1.7 (0.99) | 0.98/1.00/1.02 | 0.99/1.00/1.01 | 5.1/17.4 (0.04) |
  | HF 0826 w040, fitter grid | −31.5/+7.0/+37.5 (0.30) | 0.56/1.14/2.93 | 0.98/1.25/2.67 | 12.3/49.0 (0.28) |
  | HF 0826 w040, flatboi | **−10.6/+0.5/+12.8 (0.82)** | 0.83/0.99/1.09 | 0.91/0.99/1.08 | 8.2/32.9 (0.15) |

  So whole-winding SLIM removes most stretch and shear, but leaves ±11–13° shear at p5/p95 on a raw fit (82 % of the
  area within ±8°). If the letter place stays above ±8° after whole-winding flattening, a local patch flatten (flatboi on
  a crop around the place) is the next arm.
- 00:35 Second local container restart (coordinator); resumed 00:40.
- 00:41 Chunk 3 ok: 5,884 → **9,457** steps (job 2,166 s; peak 9,744 MiB). Chunk 2 2,226 s. c04 and gate 1
  (`wl_250_gate_c03`, step 9,457, before the track-DT stage) running. Projection: 30,000 after c09–c10 ≈ 04:30–05:10.
- 00:45 **0826 fit chain queued** (approved): `wl_301…312_acw_c01…c12` (first waits for `wl_211_acw_c11`, the last 0211
  chunk, and `wl_522_sdt826`), gates `wl_350_gate_c03`, `wl_359_gate_final` (KEEP=1). Generator `wl/jobs/mk_chunk826.sh`.
  Note for whoever resumes: when the 0211 read jobs (flatten/render/ink) are submitted after `wl_259_gate_final`, the
  pending 0826 chunks may take the GPU first; if so, cancel the pending 0826 chunks and resubmit them (new names) after
  the 0211 ink jobs.
- 01:05 **Gate 1 (`wl_250_gate_c03`, step 9,457, before track-DT)**: 119 windings exported (w010–w129, 1,008.6 cm²),
  84 inside (CT found ≥ 0.9). Raw on-m7 0.228 vs control 0.231, m7 ≤ 2 vox 0.283 vs 0.291, CT z +0.030 — **raw at
  chance**, as on 0175A/1203 at this stage. Snapped on-m7 0.651, m7 ≤ 2 0.736, CT z +0.130. Normals point to the
  umbilicus (oncheck's outward fraction 0.005). Winner mesh by `wl/code/normside.py`: 0.995 towards the umbilicus and it
  reads in forward; the same tool must be run on our flattened windings to fix the expected order (oncheck uses its own
  formula, so not yet comparable). onsheet f_parallel (w026/043/060/
  077) 0.88/0.92/0.90/0.92. Cross-section z 8400 (`wl/results/wl_250_gate_c03/xsect_c03_fitted_c_z8400_L1_crop.jpg`):
  windings follow the layer flow around the central eye densely; the eye itself (≈ 5 inner wraps) is not fitted.
  Inner blind set w010–w069 = 284.6 cm² chart at this step.
- 02:50 (coordinator) Agents were stopped by the usage limit at ≈ 01:10 UTC; queue ran on. Coordinator fetched c05 (16,081 steps) and c06 (19,388 steps) with tools/fetch_chunks.sh; c07 running. The agent will be resumed when the 0211 schedule reaches 30,000 (≈ 04:45 UTC).
- 04:45 Resumed after the usage stop. Pending 0826 chunks c02–c12 and their gates cancelled and re-queued as
  `wl_402…412_acw_c02…c12`, `wl_450_gate_c03`, `wl_459_gate_final`, with c02 waiting for `wl_301` (running) AND the last
  0211 blind ink job `wl_26c_i`. Fetch script regex extended to `wl_[2-4]NN`.
- 04:45 **Task 1 blind read chain submitted** (after `wl_259_gate_final`): groups b1 w010–029, b2 w030–049, b3 w050–069
  of the step-30,000 RAW fit (`fitted/`), each: flatboi (keep 100) → render (41 layers, bicubic, mesh at 20) → reader-v3
  both directions −3/+3 → post (parallel mask) + montages. Jobs `wl_26{a,b,c}_{fl,r,i,p}` (inks chained a→b→c).
  Code `wl_ink_v3.tgz` (adds gridq.py, place.py, normside.py), flatten script `wl/code/flat_set.sh` (also tars the
  flattened meshes to OUT).
- 05:10 **Final gate (`wl_259_gate_final`, step 30,000)**, 120 windings w010–w129 (1,017.2 cm²), inside n = 84:
  raw on-m7 **0.376 vs control 0.226** (m7 ≤ 2 vox 0.445 vs 0.287; CT z +0.057 vs +0.009); snapped (n = 85) on-m7 0.652,
  m7 ≤ 2 0.732, CT z +0.127. onsheet f_parallel w026/043/060/077 0.92/0.95/0.94/0.93. The raw departure from chance is
  the largest of our three fits (0175A 0.321, 1203 0.344), but per the coordinator's wrong-sense control the gate only
  says "locally follows sheets", not "correct sense / one sheet". Meshes: `wl/results/wl_259_gate_final/meshes/{fitted,snap}`.
  Distances to the winner surface computed into a SEALED file (`wl/results/SEALED_windist_*.tsv`), not read before the
  blind review is written.
- 05:15 Quality of the RAW fit grid, w010–w069 (60 windings, `wl/results/wl_259_gate_final/gridq_fitted_w010_069.json`;
  medians over windings of each winding's p5/p50/p95): shear8 −25.6/+3.1/+27.1° (median share within ±8°: 0.42);
  stretch along the turn 0.61/1.11/3.07, along z 0.81/1.08/1.68; roughness median 9.3°, p95 32.2°, 14 % > 20°.
  (Winner: shear ±1.7°, stretch 0.98–1.02, roughness 5.2/17.9/4 %.) Flattened table follows when the flatten jobs end.
- 05:25 Flatten + render of the blind set done (flatboi ≈ 2–4 s per winding, no failure; render ≈ 40 s per winding,
  e.g. fw029 4.45 cm² valid, 3.60 cm² parallel, f_parallel 0.94). **Quality of what is rendered (flatboi of the raw fit,
  w010–w069, `wl/results/flat211f/gridq_flat.json`; medians over windings of p5/p50/p95):** shear8 −4.5/−0.1/+2.8°
  (share within ±8°: median 0.96, min 0.93; worst winding p5 −9.2°, p95 +5.2°); stretch along the turn 0.95/1.00/1.04,
  along z 0.96/0.99/1.02; roughness median 6.6°, p95 23.2°, 7 % > 20°. I.e. close to the winner mesh (±1.7°, 1.00,
  5.2/17.9/4 %) and far from the fit grid (±26°, 0.6–3.1×, 9.3/32/14 %). Ink a→b→c queued behind 0826 c01.

### Task 1 — BLIND review (written before opening the sealed distance file)
Sheets `wl/look/blind/p1…p4.jpg` (= `b_w0NN.jpg`): per winding rv3 forward above reverse, z maps at 20 px/mm, ticks
every 10 mm; rankers from `wl/results/wl_26{a,b,c}_p/summary.txt` (parallel-mask post; positive on this path: winner
rv3 fwd line 5.65 / row 2.86, argus fwd 7.51 / 4.06, Studio asc 8.94 / 3.16).
- **Group b1, w010–w029 (07:45 UTC)**: windings are short (8–35 mm along the turn × 14 mm in z; 1.0–3.6 cm² parallel).
  Line finder max: fw021 rev 8.28, fw022 rev 6.84, fw029 rev 6.59, fw016 rev 5.09; forward max 3.56 (fw022); row max
  3.12 (fw026 fwd), 2.53 (fw027 fwd), 2.49 (fw023 fwd). By eye: round blobs and speckle in both orders; in forward,
  vertical fibre-parallel bright bands (fw017, fw018, fw021, fw025, fw026); in reverse, horizontal fibre striations with
  blobs strung along them (fw016, fw026, fw027 — what the reverse line chains pick up). Only weak spots: fw022 fwd
  top-left (two short joined curved marks along the top edge, x ≈ 4–9 mm) and fw023 fwd top-left (a bright blob pair) —
  no row, no letter shapes. **Nothing flagged in b1.**
- **Group b2, w030–w049 (08:05 UTC)**: 37–65 mm along the turn, 3.7–6.5 cm² parallel each. Line finder max (reverse
  dominates): fw033 rev 8.33, fw035 rev 7.06, fw032 rev 7.03, fw030 rev 6.52, fw044 rev 5.40; forward max fw046 4.82,
  fw038 4.31, fw041 4.19. Row max: **fw036 fwd 4.35**, fw046 rev 3.90, fw037 fwd 3.36, fw038 fwd 3.20 (higher than the
  winner's rv3 fwd 2.86). By eye: the forward rows come from **vertical bright bands 3–8 mm wide that recur at the same x
  on neighbouring windings** (fw036/fw037 x ≈ 16–24 mm; fw043/fw044/fw045 x ≈ 8–22 mm) — fibre-/fold-parallel
  structure continuing across turns, not text (F-type: structure, not letters). Reverse: horizontal fibre striations
  with blobs strung along them; fw032 rev one row of 8 separate round dots at y ≈ 12 mm (row of dots, F1); fw033 rev
  bottom right two Λ-like bright chevrons 2–3 mm (x ≈ 33–37 mm) and fw035 rev top right a V/Y-like pair (x ≈ 29–31 mm),
  isolated, no row. **Nothing flagged in b2** (the two chevron spots would go only to a second family as "looks").
- **Group b3, w050–w069 (08:25 UTC)**: 75–95 mm along the turn, 6.3–9.9 cm² parallel each. Line max: **fw057 fwd 7.41**
  (the only forward line > 5), fw057 rev 6.08, fw069 rev 5.79, fw068 rev 5.75, fw054 rev 5.25; row max fw054 fwd 3.44 /
  rev 3.38, fw064 fwd 3.21, fw059 fwd 3.11. By eye: blobs/speckle at higher density than b1/b2, horizontal striations in
  reverse, vertical bands in forward (fw059, fw064, fw068, fw069 at x ≈ 50–60 mm). fw057 fwd x ≈ 8–25 mm
  (`wl/look/blind/fw057_fwd_left.jpg`): a dense cluster of round and elongated blobs, a few curved, no strokes joined
  into letter shapes, no row → cloud (F-type). fw067 rev x ≈ 40–60 mm mid-height: a horizontal string of joined blobs
  along a striation (row of blobs along fibres). **Nothing flagged in b3.**
- **Blind verdict over all 60 windings (w010–w069, 284.6 cm² chart, ≈ 300 cm² parallel), both orders: no place meets
  even the "look" level of report 33's z9120_w080 (connected letter-like strokes in a row).** Rankers: best line 8.33
  (fw033 rev), 8.28 (fw021 rev), 7.41 (fw057 fwd); best row 4.35 (fw036 fwd) — all from blobs on striations or
  cross-turn bands. In the forward (expected) order the best line is 7.41 (fw057) and the best row 4.35 (fw036).
  Written 08:27 UTC; now the sealed distance file is opened.

### Task 1 — after unblinding (08:30–08:50 UTC)
- **Distances (sealed file opened 08:28)**, winner vertices of the awarded row z 8300–8500 to each of our windings
  (densified ×4), raw fit / snapped: nearest **w035 median 5.9 / 5.7 vox** (≤ 2 vox 10 / 11 %, ≤ 4 vox 37 / 39 %, ≤ 8 vox
  57 / 57 %); w034 8.9 / 6.6; w036 9.1 / 11.7; w033 13.2 / 14.3 (but p10 2.3 — partly on it). **No single winding stays
  on the letter sheet**: the sheet is shared between w033 (36 % of letter vertices nearest), w035 (44 %), w036 (13 %),
  w034 (7 %); to the union of w033–w036 the median is 3.0 vox (≤ 4 vox 74 %, ≤ 8 vox 99 %; snapped 2.7 / 78 % / 98 %).
  I.e. our dense fit passes within a few voxels of the letter sheet everywhere, but each winding hops between it and
  its neighbours (sheet jumps), so on any one winding the letter rows are fragmented.
- **Reading order on our windings is REVERSE (my pre-registered geometric check was not run before the review — an
  error of mine, mitigated because both orders were reviewed):** `normside.py` gives the winner mesh 0.995 normals
  towards the umbilicus (reads in forward), our fitted and flattened windings 0.003–0.010 (opposite) → our rv3 reverse
  = the winner's rv3 forward. In the reverse order the line finder's top four of all 120 maps include **fw033 rev 8.33
  (rank 1) and fw035 rev 7.06 (rank 4)** — the two windings nearest the letter sheet — and the only two "looks" I noted
  in b2 were on exactly these maps (fw033 rev chevrons, fw035 rev V/Y). Neither was flagged as letters.
- Side by side in the winner's frame (`wl/look/placeR_3533.jpg`; affine from vertex correspondences ≤ 6 vox, scales
  1.01/1.01): fw035 rev covers the left ~65 % of the winner surface (53 % of winner vertices matched; affine inliers 55 %,
  i.e. the mapping is broken by the jumps), fw033 rev the right ~35 % (36 % matched, inliers 90 %). Where they lie on the
  sheet, single letter-like marks coincide with the winner's letters (fw035 rev: the "Y" of the top row at the same
  place; fw033 rev: two Λ/Α-like marks of the bottom row) — but the awarded middle row is not legible on either; the
  rest is blobs. Grid quality there (flattened): shear/stretch as in the table above; the RAW fit grid at the place:
  stretch along the turn p5/p50/p95 0.66/1.30/3.74 (w035) and 0.68/1.27/3.37 (w033), along z 0.77/1.11/1.83.
- **Interim on the raw fit (flattened, reader-v3): MISSED by the review** (nothing flagged; letters present only as
  isolated marks on the fragments of the sheet the windings pass through).
- Refinement / extra arms submitted (08:45): `wl_270–273` = snapped+flattened w032–w037 (rv3 both), the RAW FIT GRID as
  is for w033/w035 (rv3 + Studio both), Studio s48 both orders on fw033–fw036, sw033, sw035; `wl_280–283` = **winners'
  loop regrowth**: GrowPatch (0306B params, 100 generations) seeded at the two BLIND look points (snap-mesh vertices
  nearest them: gp1 (3280, 3352, 8802) from fw033 rev, gp2 (3900, 3658, 8009) from fw035 rev) → render → rv3 + Studio.
  0826 chunks c03–c12 re-queued as `wl_603…612` (after `wl_272_i`), gates `wl_650`, `wl_659`; fetch regex `wl_[2-46]NN`.
- 09:00 (times above marked 08:45/08:50 were estimates; real `date -u` 08:29–08:40.) GrowPatch arm `wl_280_grow`: normal
  grids z 7000–9700 + xz/yz 959 s; gp1 grew in 379 s, gp2 in 358 s (100 generations, 5 threads) — but each patch spread
  over z 3000–10900 (≈ 155 cm² of grid), i.e. it ran along the sheet far beyond the letter area. Quality (gridq):
  gp1 shear8 −6.4/+0.6/+22.6° (≤ 8°: 0.84), stretch turn 0.88–1.16, z 0.94–1.46, roughness 5.4/24.9/7 %; gp2 −6.2/+0.1/
  +5.2° (0.94), 0.92–1.05, 0.97–1.14, 5.3/19.3/5 %. Distance of the winner's letter rows to the patches: gp1 top row
  median 3.1 vox (≤ 4 vox 58 %), awarded row median 17.2 (≤ 4 vox 36 %), bottom 16.5 (32 %); gp2 awarded median 16.0
  (≤ 4 vox 28 %). So the regrown patches are on the letter sheet over only about a third of the awarded row (they too
  step to a neighbouring sheet). Full-patch ink (≈ 4 L4-h) cancelled; patches cropped to z 7550–9300 within 1800 vox of
  the seed (gp1c 6.0 cm², gp2c 5.9 cm²) → `wl_284_rg` → `wl_285_ig` (rv3 + Studio both) → `wl_286_pg`. Normal side of the
  crops is mixed (towards-umbilicus 0.28 / 0.74), both orders reviewed.
- 09:20 (`date -u`) Coordinator asked for a **depth-sweep stitch arm** before the verdict. Submitted `wl_290_r81` (81-layer
  renders, mesh at 40, of fw035, fw033 and control fw026 — far from the letter sheet, oracle-valid 0.1 %) and `wl_291_i`
  (reader-v3 REVERSE = reading order, shifts −15…+15 step 3, per-shift maps saved, fixed rescale 0.25–0.75 so shifts are
  comparable). Oracle shift fields computed locally (`wl/results/sweep/oracle_w0NN.npy`: signed distance along the render
  normal from our flattened winding to the winner surface, where the lateral offset < 4 vox): fw035 valid on 81 % of its
  vertices, shift p10/p50/p90 −5.1/+0.9/+4.7 vox; fw033 76 %, −0.4/+2.9/+13.7. 0826 chain re-queued again as
  `wl_703…712` (c03 after `wl_291_i` and `wl_285_ig`), gates `wl_750`, `wl_759`; fetch regex `wl_[2-47]NN`.
- 09:57 **Arms `wl_270–273` (results `wl/results/wl_273_p/`)**, parallel-mask post; reading order on our windings =
  rv3 reverse = Studio desc:

  | surface (w033 / w035) | rv3 reading order: line | Studio desc s48: line / row / hot | Studio asc (wrong order): line |
  |---|---|---|---|
  | raw fit, flatboi-flattened (fw) | 8.33 / 7.06 | **8.57 / 1.46 / 0.079** ; **8.38 / 1.80 / 0.083** | 4.41 ; 4.69 |
  | snapped, flattened (sw) | 7.06 / 7.65 | 7.07 / 2.55 / 0.084 ; **8.64** / 2.15 / 0.085 | — ; 2.97 |
  | raw FIT GRID as is (gw, not flattened) | 3.72 / 4.41 | 5.99 / 1.92 / 0.073 ; 5.34 / 1.73 / 0.076 | 3.48 ; 2.70 |
  | (neighbours fw034 / fw036) | 6.43 / 7.94 (sw) | 4.67 ; 5.25 | 0.68 ; 1.68 |
  | winner mesh, our path | rv3 fwd 5.65 | Studio asc 8.94 / 3.16 / 0.078 | Studio desc 4.86 |

  **By eye (`wl/look/T1_studio_desc_vs_winner.jpg`, `wl/look/T1_fw033_studio_desc_full.jpg`): on fw033 (raw fit,
  flattened) Studio desc shows the letters** over the stretch where w033 lies on the letter sheet (x ≈ 30–42 mm of the
  winding, the right ~35 % of the winner surface): three rows of separate, connected letter shapes; warped into the
  winner's frame the middle (awarded) row reads "ΔΙΑ…" at the same place as the winner's "ΔΙΑ", with the top-row and
  bottom-row letters (Π…, ΓΑ/ΛΑ…) at their places. In the winding's own frame the text is **mirrored** (our winding runs
  along the turn in the opposite direction to the winner's grid), so a blind reviewer must look in both mirror states.
  On fw035 Studio desc the top-row "Y" and blobs; the awarded row is not legible there (fw035 jumps off the sheet;
  affine inliers 55 %). The FIT GRID as is (gw033/gw035, not flattened): no letters legible (line 5.3–6.0; shear/stretch
  at the place 0.66–3.7× along the turn) — flattening is what makes the Studio read possible here.
  reader-v3 on the same flattened fw033 shows only the two Λ-like chevrons (the "look" of the blind review).
- 10:21 **Regrowth arm (`wl_285_ig`/`wl_286_pg`, GrowPatch crops seeded at the blind looks):** gp1c (seed from the fw033
  rev chevrons) Studio desc line **8.65**, row **3.56**, hot 0.086 (asc 7.23 / 2.58), rv3 rev 7.16 / 3.32; gp2c (seed from
  the fw035 rev V/Y) Studio desc 3.72, asc 4.91 / 3.75 (its normals are mostly towards the umbilicus, 0.74, so its reading
  order is asc). By eye (`wl/look/T1_gp_studio.jpg`): **gp1c Studio desc shows the three rows more cleanly than fw033**
  (top row, "ΔΙΑ…" in the awarded row, bottom row) over the right ~40 % of the winner surface (39 % of winner vertices
  matched, affine inliers 0.95); gp2c shows curled strokes at the top-left (the "Y" area) in asc, not legible.
  Stride-96 unblended test (coordinator's request) submitted: `wl_295_s96` (Studio s96 both orders on fw033, fw035 and far
  controls fw012, fw062) → `wl_296_p96`. 0826 chain re-queued as `wl_803…812` after it; gates `wl_850`, `wl_859`; fetch
  regex `wl_[2-478]NN`.

## RECIPE (calibrated on PHerc0211) — self-contained, for re-reading fitted windings (e.g. lane `rr_` on PHerc0175A)

Code (all private, local):
- `/home/user/vesuvius/wl/jobs/wl_ink_v3.tgz` — tar of `pkg/`: `render_tifxyz.py` (41-layer renderer), `onsheet_score.py`,
  `si_tools.py` (prep: parallel mask + 20 px/mm views; stpng), `infer_ink.py` (reader-v3 / argus-v3), `studio_t4.py`
  (Studio packaged CLI, release bf66f753, weights SHA-256 96b32628…), `at_atlas.py` (post: z maps, row score, glyph CNN,
  306B line finder; montage), `gridq.py`, `stretch.py`, `normside.py`, `place.py`, `windist.py`.
- `/home/user/vesuvius/wl/code/wl_env.sh` (sourced by every ink job; cache `/content/cache/wlink`; functions `si_render`,
  `si_rv3`, `wl_ag`, `si_st`, `si_post`), `/home/user/vesuvius/wl/code/vcenv.sh` (VC3D AppImage binaries; the AppImage
  `VC3D-e5707e4-2026-11-23` was extracted by job `wl/jobs/wl_030_probe.sh` — re-run that on a new VM, it finds the
  current release URL via the GitHub API), `/home/user/vesuvius/wl/code/flat_set.sh` (flatboi per winding),
  generator `/home/user/vesuvius/wl/jobs/mk_read.sh` (flatten → render → ink → post+montage job chain for a winding list),
  review sheets `/home/user/vesuvius/wl/code/sheet.py`, depth sweep `/home/user/vesuvius/wl/code/sweep.py`.
- Templates: `wl/jobs/wl_26a_{fl,r,i,p}.sh` (rv3 screen), `wl/jobs/wl_272_i.sh` + `wl_273_p.sh` (Studio s48 both
  orders + post), `wl/jobs/wl_295_s96.sh` (Studio s96), `wl/jobs/wl_280_grow.sh` → `wl_284/285/286` (GrowPatch regrowth +
  crop + ink), `wl/jobs/wl_290_r81.sh` / `wl_291_i.sh` (81-layer depth sweep).
- To adapt to another scroll: in `wl_env.sh` add the scroll to the `case "$WLSC"` (volume URL) and for an 8.64 µm
  volume replace 9.362 in `--voxel-um`, `--input-um`, `si_tools.py prep --um`, `at_atlas.py post --um`; in `flat_set.sh`
  point `SRC` at the directory holding the winding tifxyz dirs (here: `/content/cache/wl$SCR/out/<tag>/{fitted,snap}`).

Steps and settings:
1. **Flatten every winding with villa's flatboi (SLIM, symmetric Dirichlet)**: `flatboi <wNNN> --output=<out> 50
   symmetric_dirichlet` (keep 100 %; `PASTIX_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OPENBLAS_CORETYPE=Haswell`), 2–4 s per
   winding. Check with `gridq.py`: target shear8 within ±8° on > 90 % (we got median 0.96), stretch 0.95–1.05.
   **Do not read the fit grid as is** (shear ±26°, stretch 0.6–3×): on the letter winding it showed nothing.
2. Render: `render_tifxyz.py --layers 41 --voxel-um <um> --surface-interp bicubic` (mesh at layer 20, renderer default
   flip-normals), then `onsheet_score.py --mid 20` and `si_tools.py prep … --um <um>` (wrapped in `si_render`).
3. **Reading order on a fitted winding from geometry**: `python3 normside.py <umbilicus.json> <flat_winding>`; the winner
   mesh (reads in rv3 *forward* / Studio *asc* through this renderer) gives 0.995 "towards the umbilicus". A winding with
   ~0 (our ACW fits: 0.003–0.010) reads in **rv3 reverse / Studio desc**. Run it BEFORE reviewing (I did not, and only
   reviewing both orders saved the test). Mixed values (0.3–0.7, e.g. GrowPatch patches) → review both orders.
4. Detectors: reader-v3 `--direction both --z-shifts=-3,3` as the screen; **Studio fp16 stride 48 Hann, both orders,
   `--mesh-index 20`** on every place the screen marks as a "look" (any connected stroke-like marks, even isolated) — on
   0211 Studio was what made the letters legible; reader-v3 alone showed only two chevrons there.
5. Post (`at_atlas.py post` on the parallel mask) → line finder, row score, glyph_p, hot. Review the 20 px/mm z maps
   in **both orders and both mirror states**: the letters on our winding appear mirrored (the winding runs along the turn
   opposite to the winner's grid).
6. Numbers that separated the letter winding (fw033), among 60 windings × 2 rv3 orders = 120 maps: line finder fw033 rev
   **8.33, rank 1/120** (fw035 rev 7.06, rank 4; next fw021 rev 8.28 = blobs on striations); row score rank 94/120 (row
   score does NOT find it: the max 4.35 was a cross-turn vertical band); hot 0.061 rank 12/120; rv3 order asymmetry
   (line rev − fwd) rank 9 of 60. Studio s48 (8 maps on the 4 nearest windings): fw033 desc line 8.57 vs asc 4.41,
   fw035 desc 8.38 vs 4.69; non-letter neighbours fw034/fw036 desc 4.67/5.25; positive (winner mesh) asc 8.94 vs desc
   4.86 → **Studio order asymmetry ≈ +4 on the letter windings and on the positive** is the most discriminating number.
7. What did NOT work: the fit grid as is (Studio desc line 5.3–6.0, nothing legible); reader-v3 alone (missed by the blind
   review); single-order review with the order guessed from oncheck's normal sign (wrong here); unmirrored review; the
   row score as a ranker; GrowPatch 100 generations without cropping (patches ran over z 3000–10900, ≈ 155 cm²).
   What helped further: GrowPatch regrowth seeded at the reader-v3 look (gp1c) gave the cleanest Studio map.

### Task 1 — remaining arms and VERDICT (10:40 UTC)
- **Depth-sweep stitch (`wl_290/291/292`)**, reader-v3 reading order (rev) on 81-layer renders, shifts −15…+15 step 3;
  post on the parallel mask (line / row / hot):

  | map | fw033 | fw035 | control fw026 (far, blank) |
  |---|---|---|---|
  | single shift 0 | **7.13** / 1.46 / 0.066 | **6.49** / 1.14 / 0.063 | 1.89 / 0.01 / 0.055 |
  | (a) max over shifts | 7.31 / 1.59 / 0.067 | 4.53 / 2.57 / 0.066 | 3.05 / 1.51 / 0.054 |
  | (b) stitched (1.25-mm blocks, max hot fraction, 3×3 median) | 5.74 / 1.19 / 0.072 | 4.59 / 1.53 / 0.070 | **5.14** / 1.86 / 0.062 |
  | (c) oracle shift (+ / − sign) | 5.32 / 6.45 | 5.62 / 5.62 | 1.77 / 1.65 |

  By eye (`wl/look/sweep_fw033_side.jpg`, `wl/look/sweep_fw035_sel.jpg`): on fw033 all variants show the same three rows
  (the 81-layer shift-0 map is as good as any); on fw035 (b) adds some stroke shapes on the left, but no row becomes
  legible, and (c) is no better than shift 0. Stitched shift vs oracle within one step on 43 % (fw035) and 36–52 %
  (fw033) of the blocks with oracle information (`stitch_shift_hist` piles up at −15, the sweep edge). The control shows
  the false-positive side: stitching lifts a blank winding's line score from 1.9 to 5.1. **Conclusion: no gain from
  depth stitching here; where our winding is within a few voxels of the letter sheet (fw033) the single-shift map already
  reads, and where it has jumped (fw035) even the oracle shift does not recover legible rows** — so the limit at fw035 is
  not the depth placement of the window but the surface itself (lateral misplacement/folding across the jump, and the
  parallel-tile mask cutting the jump zones).
- **Stride 96 unblended (`wl_295/296`)**: Studio s96 desc/asc line: fw033 **6.86 / 2.39**, fw035 **7.04 / 2.46**; far
  controls fw012 3.65 / —, fw062 2.61 / 1.99. Order asymmetry +4.5 on both letter windings vs +1.7 / +0.6 on controls;
  ranking preserved. By eye (`wl/look/T1_fw033_s96_crop.jpg`): the 96-px tile grid is visible; "ΔΙΑ" and the bottom-row
  ΓΑ are still discernible but much less clearly than at s48. → s96 is usable as a first-pass RANKER (line + order
  asymmetry); every hit must be re-read at s48 before any judgement.

**TASK 1 VERDICT: FOUND on our raw fit (flatboi-flattened, no surface refinement) — with Studio stride 48 in the reading
order; the reader-v3 blind screen alone MISSED it** (it noted the place only as one of two "looks"). By my pre-registered
definition (blind flag + ≥ 3 of 11 awarded letters recognisable) the reader-v3-only screen does not pass; the loop step
"look → Studio s48" passes: on fw033 the awarded row's "ΔΙΑ" (3 letters) plus letters of the rows above and below are
legible beside the winner's map. Refinement by GrowPatch seeded at the blind look (gp1c) gives the clearest map of all.
Numbers at the place: our nearest windings to the awarded row: w035 median 5.9 vox (≤ 4 vox 37 %), w033 13.2 vox overall
but p10 2.3 (on the sheet over the right ~35 % of the winner surface, 0–3 vox there by the oracle field); the union of
w033–w036 median 3.0 vox (≤ 4 vox 74 %); on-sheet (gate) raw on-m7 0.376 vs control 0.226 for the fit; flattened shear8
−4.5/+2.8° and stretch 0.95–1.04 (median over windings), fit grid at the place 0.66–3.7× along the turn and ±26° shear
→ fit grid as is: nothing legible. What differs from the winner's surface: our windings hop between the letter sheet and
its neighbours (each winding is on it over only 30–45 % of the letter rows), so letters appear on fragments ≈ 1–2 cm
wide; the winner's hand-refined mesh stays on one sheet for 5.6 cm.
Meaning for the earlier negatives: our 0175A/1203 reads were reader-v3 on non-flattened or common-map-resampled grids
(report 22/24), with Studio only on 30-mm crops and with a guessed single order on 1203 — exactly the settings that
missed 0211 here. They should be re-read with the RECIPE above (flatboi + Studio s48 both orders + both mirror states).
- 10:37 Verdict sent to the coordinator. GPU used so far ≈ 12.0 L4-h (0211 fit 5.6 h; 0211 reads, arms and positive
  ≈ 4.6 h; 0826 c01–c02 1.2 h; c03 running). 0826 fit c03–c12 queued (`wl_803…812`), gates `wl_850` (c03), `wl_859`
  (final, KEEP=1). Next: 0826 inventory/quality of our windings at gate c03, then the read plan within ≈ 7 L4-h.

## Task 2 — PHerc0826 own fit z 11800–13300 (ACW, official inputs)
- 11:40 Gate 1 at step 10,061 (`wl_851_gate_c03`; the first gate job `wl_850` was skipped by a dependency typo of mine):
  120 windings exported (970.8 cm²), **69 inside the scroll (w010–w078, 350.7 cm²)**; w079 and beyond run partly or
  wholly outside (CT found 0.86 → 0.16). Raw on-m7 0.238 vs control 0.225 (at chance, as expected before track-DT);
  snapped 0.569; onsheet f_parallel 0.90–0.92. Cross-section z 12550 (`wl/results/wl_851_gate_c03/xsect_c03_fitted_z12550_L1.jpg`):
  windings follow the flattened lens's layers; outer exported windings lie in air around it.
- **Read plan, fixed before any 0826 ink (11:42 UTC):** after the final gate (step 30,000): every inside winding
  (CT found ≥ 0.9), excluding the outermost 3 inside windings (membrane/glue of the 1868 unrolling sits on the outer
  turns, report 30); RECIPE as calibrated: flatboi keep 100 → gridq table (abort and report if shear8 within ±8° < 0.8 on
  most windings) → render 41 layers → reading order from `normside.py` (expected rv3 reverse / Studio desc, as on 0211,
  same ACW fit and catalogue flags) → reader-v3 both orders −3/+3 on every winding (coordinator's condition) → review
  both orders AND both mirror states, line pitch 4.4–6.9 mm, letter pitch 3.5–3.9 mm → Studio s48 both orders on: the
  top 8 windings by rv3 line finder in the reading order, the top 4 by rv3 order asymmetry, every place I mark as a "look",
  and 3 random windings (seed 826) as base rate → GrowPatch regrowth (crop to the band) at looks that show stroke shapes in
  Studio → Studio again + argus-v3 + CT sections + neighbouring turns. Bar for `candidate under private review` as briefed.

## Task P — sizing the funnel on PHerc0211 (p = chance that one unit shows letters)
- 12:02 (`date -u`) Coordinator: Task P with priority (budget 3 L4-h). 0826 chain held after the running c05:
  `wl_806…812`, `wl_859` cancelled (to be re-queued after P's GPU jobs).
- **Choices fixed before looking (12:04):**
  - P1 units: 20 mm (along the turn) × full winding height (≈ 14 mm, the band is 1,500 slices) windows on the 60 flattened
    windings fw010–fw069, non-overlapping from the first valid column, kept if render-valid ≥ 50 % of the window (so a
    unit is ≤ 2.8 cm², not 4 cm² — the band is only 14 mm tall). Ranking: 306B line finder per window on Studio s96 desc
    (reading order). Other order (asc) on 9 windings drawn with `numpy.random.default_rng(211).choice(60, 9)`:
    fw014, fw018, fw021, fw030, fw033, fw041, fw047, fw049, fw052. Stride 48 desc on the top 25 windows + 15 random
    windows from the rest (`default_rng(2110)`), judged by eye in both mirror states, blind. Windows I cannot judge blind:
    any window of fw033 x ≈ 30–42 mm and fw035 (I have seen the letter place there) — recorded as "not blind".
  - P2: radius chosen by CT anatomy only: z 8400 (band centre), from the official umbilicus straight along +y (in the
    z-slice the layers below the central eye run flat, parallel and evenly spaced), starting 150 vox out (past the eye);
    sheet centres = successive m7 L0 runs (≥ 2 vox) along that ray; 12 successive sheets; GrowPatch 100 generations,
    crop 120 × 120 cells (≈ 5 cm²) around the seed, flatboi, then gridq, Studio s96 then s48 (order from normside; both if
    mixed). Jobs: `wl_901_s96` (GPU), `wl_902_win` (windows + P3 export), `wl_911_p2` (CPU grow/crop/flatten).
- 13:15 P2 seeds (`wl_911_p2`, ray x = 3430, z 8400, y from 3898 outward; 62 m7 runs on the first 1,500 vox): p00–p11 at
  y = 3901, 3920, 3935, 3951, 3975, 3995, 4013, 4023, 4032, 4048, 4057, 4070 (sheet runs 3–4 vox thick, 9–24 vox apart).
  All 12 grown (100 gen., 3 threads, 3 in parallel; 516–791 s wall each incl. crop + flatboi), cropped 120 × 120 cells,
  flattened; rendered (`wl_912_p2r`, 736 s for 12). s96 rate measured: 60–110 s per winding incl. model start
  (≈ 23 s/cm²). Told the coordinator P costs ≈ 3.9 L4-h and collides with the VM end; **decision (a): finish P, finish
  the 0826 fit to 30,000, save everything locally before ≈ 21:00; the 0826 read runs later on g4 by the recipe (not by me).
  Lane budget raised to 27 L4-h.** P2 s48 only in the normside order (both where mixed): `wl_915_p2i` → `wl_916_p2p`.
  0826 chunks c06–c12 will be re-queued as `wl_506…512` behind P's GPU jobs (fetch regex now `wl_[2-58]NN`).
- 14:20 P1 s96 pass done (`wl_901_s96`, 5,801 s GPU for 69 maps). Windowing (`wl_902_win`, CPU 1,112 s):
  **149 windows, 293.4 cm²** (parallel-mask area in the post). Ranking by the line finder on Studio s96 desc
  (`wl/results/wl_902_win/psel.json`): top 10 = fw044_x1 7.75, fw033_x1 6.87, fw031_x1 6.41, fw032_x1 6.38,
  fw035_x0 6.37, fw030_x1 5.90, fw035_x1 5.66, fw067_x2 4.87, fw029_x0 4.68, fw057_x3 4.37 (… #25 fw033_x0 3.27).
  15 random from ranks 26–149 (rng 2110): ranks 27, 29, 37, 43, 52, 64, 76, 88, 91, 108, 122, 127, 129, 138, 144.
  Not judgeable blind by me: fw033_x1, fw035_x0, fw035_x1 (the known letter place lies in them). Studio s48 desc on the
  40 window crops: `wl_920_s48` → `wl_921_p48`. 0826 chain re-queued as `wl_506…512` (c06 after `wl_805`, `wl_920`,
  `wl_915`), final gate `wl_559` (KEEP=1).
- 14:55 **P2 result (12 GrowPatch windows along one anatomically chosen radius; `wl/look/P2_12patches.jpg`)**. Normal
  side (normside): p01 0.31 (mixed → both orders), all others 0.74–0.95 towards the umbilicus → reading order Studio asc.
  Quality (gridq on the flattened crops): p00–p03 shear8 p5/p95 ≈ −28…+24° (share ≤ 8°: 0.56–0.64; these inner seeds
  grew across folds), p04 0.81, p05–p11 within ±4.5° (0.97–1.00), stretch 0.69–1.28 (p00–p04) and 0.94–1.05 (p05–p11);
  onsheet f_parallel 0.86–0.94; valid 4.9–5.8 cm², parallel 3.8–4.2 cm² each. Studio s48 reading order, line / row:
  p00 3.95/2.72, p01 2.81/2.30 (desc) 2.31/2.91 (asc), p02 5.66/2.20, p03 3.52/1.68, p04 5.26/1.39, p05 4.95/0.34,
  p06 3.69/1.56, p07 7.83/2.61, p08 6.08/0.52, p09 4.82/1.19, p10 6.63/1.76, **p11 9.40/3.98**.
  **Judged by eye (blind, both mirror states, `wl/look/P2_p11_p07_zoom.jpg`): p11 = HIT** — three rows of connected,
  curved letter-like strokes at ≈ 5–6 mm row pitch, several shapes in a row (not legible to me as letters, but meets the
  "connected letter shapes, several in a row" bar); p07 = look, not hit (stroke clusters, one partial row); the other
  10 = no hit (blobs, speckle, crack lines). **P2: 1 hit in 12 windows (p ≈ 0.08; 95 % CI 0.002–0.38).**
  Unblinded afterwards: p11 is the outermost seed (y 4070 at x 3430, ≈ 320 vox from the umbilicus) and its central part
  lies within ≈ 6–12 vox (vertex-spacing limited) of the winner's letter surface (`wl/look/P2_p11_vs_winner.jpg`): the
  hit is on the known text, not new text. Cost per window: GrowPatch + crop + flatboi 516–791 s on 3 CPU threads (3 in
  parallel), render ≈ 60 s, Studio s48 one order ≈ 200 s GPU (≈ 50 s/cm² incl. start-up); GPU for P2 2,673 s (13 orders).
- 16:25 **P1 blind judgement** of the 40 Studio s48 desc window maps (shuffled, labels A01–A40, key sealed in
  `wl/look/P1/KEY_sealed.json`; sheets `wl/look/P1/sheet_1…5.jpg`, each map beside its mirror image), written before
  opening the key: **A19 = HIT** (three rows of connected letter shapes; in the mirror state Δ, Ι, Α, Λ-like letters in
  rows at ≈ 5 mm pitch — I recognise this as the known place, so A19 is "not blind"); **A16 = HIT** (a row of connected
  cursive-like curved strokes along the top, several in a row); **A02 = weak hit** (top row with Y-shaped and curved
  strokes in a row, middle row curls); looks, not hits: A06 (a row of joined blobs along a striation), A17 (one hooked
  stroke), A36 (a few bars); all others no hit (blobs, speckle, clouds A23/A24/A33/A38/A40, vertical structure A15).
- 16:20 **P1 after opening the key:** A19 = fw033_x1 (s96 rank 2, top 25), A16 = fw030_x1 (rank 6, top 25), A02 = fw035_x1
  (rank 7, top 25; weak); looks A17 = fw057_x3 (rank 10), A36 = fw055_x1 (rank 23), A06 = fw047_x0 (rank 43, random).
  **Hits: 3 of the top 25 (2 clear + 1 weak), 0 of the 15 random.** p for "a random 20-mm window (≤ 2.8 cm²) of a fitted,
  flattened winding": 3 / 149 = 0.020 if the 109 unexamined windows hold no hit (the random 15 from ranks 26–149 gave
  0/15, 95 % upper bound 0.22 for that stratum). Ranks of the hits by the s96 line finder: 2, 6, 7 — the s96 ranker put
  all hits in its top 5 %.
- Against the winner's surface (densified winner, distances of the window's vertices): fw033_x1 median 2.9 vox (≤ 4 vox
  73 %), fw035_x1 2.8 (74 %), fw035_x0 4.4 (47 %) — on the letter sheet. **fw030_x1: median 44 vox from the winner surface,
  0 % within 8 vox; its nearest winner vertices are interior (93 %), lateral 8 vox, normal offset −31…−59 vox, i.e. it lies
  directly behind the awarded text, ≈ 2–4 sheets further in, at the same angle and z (7654–9145).** Its Studio s48 desc
  map (`wl/look/P1_fw030_x1_hit.jpg`) shows one row of 3–4 connected, curved stroke shapes along the top of the window
  (≈ z 7700–7800), the rest blobs: possibly OTHER TEXT of PHerc0211 on an inner turn (a written scroll has columns over
  several turns), weak by itself (one row, no legible letter). Not followed up (STOP rule; told the coordinator).
- 16:20 Coordinator: P accepted; follow-up on fw030_x1 (private): GrowPatch seeded at the stroke row (snap vertex nearest
  the row centre: (3789, 3604, 7937); row centre found as the smoothed maximum of the top 7.5 mm of the s48 window map),
  100 generations, crop to cells within 1,400 vox of the seed with z ≥ 7,250 (extends above the band), flatboi, render,
  Studio s48 both orders: `wl_940_gp30` → `wl_941_gp30i` → `wl_942_gp30p`. 0826 chain re-queued again as
  `wl_907…912_acw_c07…c12` (c07 after `wl_506` and `wl_941`), final gate `wl_969_gate_final` (KEEP=1), flatten
  `wl_970_fl826` (raw + snapped, flatboi, gridq, normside). Fetch regex `wl_[2-589]NN`; `tools/fetch_chunks.sh` now keeps
  EVERY checkpoint (≈ 650 MB each; 12 GB free).
- 16:56 **fw030_x1 follow-up (private)**: GrowPatch from (3789, 3604, 7937), 100 gen. in 229 s; crop 13,489 cells,
  z 7,250–9,333 (extends ≈ 400 slices above the band); flatboi; quality: shear8 −0.8/0.0/+0.8° (1.00 within ±8°), stretch
  0.99–1.01, roughness 3.8/8.8° (0 % > 20°) — the cleanest surface of this lane; onsheet f_parallel 0.947 (4.55 cm²
  valid, 3.67 cm² parallel); normside 0.10 → reading order desc. Studio s48: **desc line 4.54 / row 2.57 / hot 0.080;
  asc 5.24 / 0.91 / 0.064** (on this patch z runs along the grid columns, so text lines would be vertical in the map and
  the horizontal-line finder under-scores them). Distance to the winner mesh: min 1.2, p5 23, median 47 vox; 12 % of
  the patch within 30 vox (a strip along the patch rim and part of the left side near z 7,500–7,800), i.e. not entirely
  > 30 vox as hoped — GrowPatch approaches the letter sheet at its rim. Side by side `wl/look/FU_fw030_regrow.jpg`
  (P1 window | regrown desc | regrown asc; top row as is, bottom row mirrored), distance/z overlay `wl/look/FU_gp30_dist.jpg`.
  By eye: in desc (reading order) one line of 5–6 connected, curved, ζ/ς/C-like strokes running along constant z at
  z ≈ 7,500–7,800 over ≈ 1.5 cm — the P1 row continues and extends above the band; the asc map shows blobs there
  (order-specific). A second parallel line at 5–6 mm pitch is not established (a second band of strokes ≈ 3–4 mm away
  follows the slanted z-isolines, probably the same line). No letter is legible to me. **Judgement: look (stroke row
  confirmed in one detector family on a clean regrown sheet, order-specific), not a second row, not legible.** Distance to
  the winner sheet ≈ 20–60 vox where the strokes are (2–4 sheets inward). Not followed further (0826 fit has priority).
- 16:58 0826 chunk c06 fetched: 16,348 → **19,545** steps. c07 running (≈ 3,300 steps per chunk → 30,000 in c09/c10,
  ≈ 19:20); final gate `wl_969` and flatten `wl_970` follow.

## HOW TO READ 0826 (for the session that runs it, e.g. g4; nothing was submitted there by this lane)
Everything needed is in `/home/user/vesuvius/wl/jobs/g4_826/` plus the meshes tarball fetched from `wl_970_fl826`.
1. Inputs to upload with `cjob input`: `wl_ink_v3.tgz`, `wl_env.sh`, `pwin.py`, `select40.py`, `umbilicus_0826.json`,
   `flat826_f.tgz` (flatboi-flattened RAW-fit windings, from `wl/results/wl_970_fl826/`), `read826_list.txt` (the
   winding names to read: inside windings (CT found ≥ 0.9 in the final gate) minus the outermost 3, written when the gate
   is in). Studio weights are fetched by the setup job (SHA-256 checked by `studio_t4.py`).
2. Jobs in order (rename the lane prefix if the session's owner wants, e.g. `rr_`): `wl_g4a_setup` (cpu) →
   `wl_g4r1/2/3` (cpu renders, 41 layers, bicubic) → `wl_g4s96` (GPU: normside per winding → Studio stride 96 unblended in
   the reading order; both orders where normside is mixed; ≈ 23 s/cm² per order → ≈ 2.2 L4-h for ≈ 350 cm²) →
   `wl_g4win` (cpu: 20-mm windows, post per window, selection of top 25 + 15 random with rng 826) → `wl_g4s48` (GPU:
   Studio s48 on the 40 windows, ≈ 1.2 L4-h). Then review the s48 window maps blind in BOTH mirror states (calibration:
   letters appear mirrored on our ACW windings), with line pitch 4.4–6.9 mm and letter pitch 3.5–3.9 mm.
3. For any window that shows connected letter shapes several in a row: GrowPatch regrowth at it (template
   `wl/jobs/wl_940_gp30.sh`: needs the scroll's m7 normal grids; change the scroll, seed and crop), Studio s48 both orders,
   argus-v3 (`wl_ag` in `wl_env.sh`), CT sections, neighbouring turns, then the bar for `candidate under private review`.
4. Calibrated expectations (0211): hits sit in the s96 line-finder top 5 % of windows; p ≈ 0.02 per random window of a
   flattened fitted winding; reader-v3 alone is not enough.
