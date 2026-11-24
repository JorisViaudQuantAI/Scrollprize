# 23 — Spiral fit on PHerc1203 inside the 2.4 µm band (lane `sg`)

Private. Started 2026-11-23 07:23 UTC. Outcome (first hand-back 07:26): PHerc1203 has no public tracks. **Go-ahead 07:26 from the coordinator: make our own
tracks (pinned `extract_surface_tracks.py`, published parameters), validate them on 0175A, then fit as briefed.**
**All PHerc1203 tracks in this lane are OUR OWN, not the organisers'.**

## Progress log (UTC)
- 07:23 Read report 21 (all, incl. `spiral_full/results/sf_209` final gate: raw on-m7 0.321 vs control 0.207,
  snapped 0.668), report 15 (all), report 09 (all), bridge README; listed `spiral_full/code` and `jobs`.
- 07:24 Input check for PHerc1203:
  - `https://dl.ash2txt.org/datasets/spiral_datasets/` index (today): PHerc0009B, 0125, 0139, 0175A, 0175B, 0191,
    0211, 0257, 0268, 0306B, 0332, 0343, 0343P, 0358, 0483A, 0483B, 0490A, 0490B, 0500P2, 0800, 0813, 0814, 0826,
    0841, 0846A, Paris4. **No PHerc1203** (`.../spiral_datasets/PHerc1203/` → HTTP 404). Same as on 22 Nov (report 21 §1).
  - Public bucket `PHerc1203/` (anonymous ListObjectsV2): `photos/`, `segments/raw/`, `volumes/` (9.362 µm
    20250820131727 + vcd3d copy; 2.403 µm 20260319130212 + vcd3d copy), `representations/predictions/surfaces/`
    (m7 L0 on 20250820131727 + normal grids; m7 L2 on the fine volume + normal grids),
    `representations/predictions/lasagna/20250820131727-lasagna-20260419180421/`, and
    `representations/umbilicus/20250820131727-umbilicus-20260924035533.json` (official). **No tracks, no crossings.**
  - Other dl.ash2txt.org datasets (`fiber-skeletons`, `seg-derived-recto-surfaces`, `grand-prize-banner-region`):
    nothing for 1203 tracks. Hugging Face search "1203": nothing Vesuvius-related. Report 21 already found no 1203
    fits/tracks in the 306B repo or `rodriguescarson/eligible-scroll-spiral-fits`.
  - So of the four inputs: m7 yes, Lasagna yes, official umbilicus yes, **tracks no**.
- 07:25 Per the brief ("If PHerc1203 lacks tracks, stop and tell me at once — do not switch scroll") I stopped.
  Band selection (transverse slices) and sense were not done, since they only matter if a fit can run.

## What would make it runnable (for the coordinator; not started)
- The pinned commit itself contains the track generator: `volume-cartographer/scripts/spiral/extract_surface_tracks.py`
  @0fb38c45 (CPU: cc3d + kimimaro skeletons of binarised m7 on horizontal ribbons every 16 slices and vertical
  slabs every 64 vox, downsample 4, `write_native_packed_store = True` → `.dbm` + `.dbm.vctracks` that the fitter
  reads). Current villa main also has a GPU backend (Brook/cuCIM, needs compute capability ≥ 8.0; the L4 is 8.9).
- This would be our own tracks, made with parameters that may differ from the organisers' published stores
  (to check: compare against `metadata.json` of a published store, e.g. 0175A `surface_m7_L0_th0.2`). It is a
  new pipeline step, not in the winners' recipe as we used it, so it needs a go-ahead.
- Rough cost (unmeasured): m7 L0 band download for 1,500 slices of 1203 (6844² slices) ≈ 5–10 GB; CPU kimimaro
  extraction on 12 cores probably 1–3 h for 1,500 slices (pinned code, no timing published); then the report-21 fit
  (≈ 8 L4-h incl. gates). The 1203 umbilicus is official and Lasagna exists, so no other fallback would be needed.

## Go-ahead (07:26): own tracks, validation, band, fit — log
- 07:27 `g3` status: fresh worker started 07:22:17, L4 idle, 253 GB free. Submitted `sg_001_setup` (villa @0fb38c45, uv
  env Python 3.14); ok in 105 s.
- 07:27 Published track parameters: both published stores carry an `.extract.json` (0175A
  `PHerc0175A_20250521115057_surface_m7_L0_th0.2.extract.json`, and 0846A from the same BM18 campaign/day as 1203,
  `PHerc0846A_20250728152254_...extract.json`); copies in `/home/user/vesuvius/spiral_1203/meta/`. Identical parameters
  in both: m7 L0, threshold 0.2 (the published m7 `th0.2` zarr, binarised `> 0`), `downsample_factor 2`,
  `z_chunk_depth_h 2`, `z_chunk_stride_h 8`, `yx_chunk_thickness_v 4`, `yx_stride_v 16`, `dust_threshold 400`,
  `max_area_threshold 640000`, `min_path_vertices 10`, `path_mode interjoint`, kimimaro `parallel 3` / chunk 250.
  The pinned module's own defaults differ (downsample 4, h depth 4 / stride 16, v stride 64) and are overridden.
  Differences I cannot match exactly / chose: (1) the published stores were made in one pass over z 4500–end with
  vertical slabs spanning the whole scroll height; ours span only our band (8936–10464), so vertical tracks end at
  the band ends and CCs larger than the band can be split differently near the ends; (2) slab x/y origin: published
  from the yx bounds of the whole scroll, ours from 20 sampled slices of our band (pinned `find_yx_range` logic);
  (3) parallelisation: kimimaro `parallel 1` inside 8–11 separate worker processes (same algorithm; output should not
  depend on it); (4) the published `.extract.json` has fields (`skeleton_parallel*`, `yx_bounds`,
  `effective_z_bounds`) that the pinned script does not have, i.e. the organisers used a later revision of the
  script; the per-slab algorithm (cc3d 6-conn CC at full res, max-pool, dust [min,max], TEASAR scale 1 const 2,
  interjoint paths ≥ 10 vertices) is the same in the pinned file and in villa main. Store format = pinned
  `tracks.write_packed_track_store` (the fitter's own reader).
  Implementation: `spiral_1203/code/xtracks.py` (pack the binarised m7 band bit-packed into a memmap, N worker
  processes each running the pinned per-slab functions on every N-th key into its own DBM, merge, packed store).
- 07:29 Band choice: 8 level-2 slices z 8000/8500/9000/9500/10000/10500/11000/11700 (`views/1203_montage_L2.jpg`) and
  900×900 L2 crops around the official umbilicus (`views/1203_centre_montage_L2.jpg`). No slice shows a clean
  concentric spiral: the scroll is strongly flattened (lens-shaped core), with S-folds and sheared wedges.
  z 8000–8500: swirls and voids near the centre; z 10500–11700: a chaotic triangular cavity around the centre. The
  most regular, countable layering is z ≈ 9000–10000 (flattened concentric sheets above and below a narrow lens
  around the umbilicus). **Band: z 9000–10400 (1,400 slices)**, 1,000 voxels above the usable fine-band floor (8000)
  and 1,365 below its ceiling (11765). Track extraction z 8936–10464 (±64).
- 07:30 Submitted `sg_002_val0175` (cpu): 0175A m7 L0 z 7604–7804, our extraction with the published slab origin
  (y 384, x 0), comparison with the published store in the same sub-band (`code/cmp_tracks.py`).
- 07:32 Submitted `sg_010_data` (m7 L0 z-chunks 45–55 + L1, CT L4 + L1, Lasagna nx/ny/grad_mag g2 — 1203 has a
  published Lasagna, so the 0358 recipe applies, not the synthetic normals of 0175A —, native modules, fitter
  patches incl. loop-time patch, spec with the official umbilicus pre-smoothed by 306B `smooth_umb.py` σ 75),
  then `sg_011_tracks` (our extraction, 8 workers) and `sg_012_sdt` (surf-SDT + pools) in parallel.
- 07:36 `sg_010_data` ok (190 s): m7 L0 z-chunks 45–55 + L1 1.9 GB, CT L4 + L1 5.3 GB, Lasagna g2 band 0.93 GB,
  native modules, patches (shell-optional, STOP_AT, LOOP_SECONDS). Official umbilicus (131 points, z 1744–18688,
  step ≈ 144) smoothed σ 75: raw vs smoothed median 5.7 vox, max 46 vox.
- 07:46 `sg_012_sdt` ok (626 s): surf-SDT (m7 L1 thr 128, CT-masked) 4.2 GB, pools sdt/normals/grad_mag 47 s.
- 07:37 `sg_011_tracks`: band bit-packed (1528 × 6844 × 856 bytes, 67 s); yx range of the m7 in the band
  y 1152–5567, x 768–6143 → 191 ribbons + 276 zx slabs + 336 zy slabs = 803 keys on 8 workers. First rate (sharing
  the CPU with SDT and the validation job): ≈ 21 s per key per worker, ≈ 2,800 tracks per key → ≈ 35–45 min total.
  (The pinned CPU extractor is therefore well inside the 3-hour limit; the GPU version is not needed.)
- 07:46 `sg_002_val0175`: the extraction started only at ≈ 07:46 because my job script waited for the background
  download of the published 0175A store (7.9 GB) before extracting (script error, cost ≈ 12 min, no harm).
- 08:01 `sg_011_tracks` ok (1,529 s incl. 67 s packing): **our PHerc1203 store: 1,396,662 tracks, 62,803,105 points**
  for z 8936–10464 (DBM 0.79 GB + packed store). Parameters as listed above, slab origin y 1152 / x 768
  (`results/sg_011/*extract_params.json`, `tracks_metadata.json`). For scale: the published 0175A store gave the
  0175A fit band (1,500 slices, larger cross-section) 2.01 M tracks / 145.5 M points; the published 0358 store gave
  1.90 M / 86.1 M for its 1,500-slice band.
- 07:46–08:00 `sg_002_val0175`: our extraction of 0175A z 7604–7804 ran (11 workers, 1,171 keys, 547 s), but the
  merge failed (output directory not created — bug in my `xtracks.py merge`, fixed; on 1203 the directory existed so
  the 1203 store is fine). Merge + comparison rerun as `sg_003_valcmp` (08:05).
- 08:02 `sg_013_sense` (chain test of report 15 §7.2 on our tracks; crossings sidecar by the pinned
  `build_track_crossings.py`: 3.50 M exact crossing events, 902,849 of 1.40 M tracks with partners, 13 s): 4,000
  starts per level at z 9300/9700/10100, RMAX 1500, hop |Δr| < 20, |Δz| ≤ 16, plain and x-mirrored. Almost no
  chain completes a turn (14 of 24,000; max broken accumulation 250–276°), so the core geometry (flattened lens)
  defeats the method. Of the 14: plain geometry z 9700: 11 turns, dr > 0 in 8 (6 within the pitch band 5–60 vox,
  0 negative in band) → CW; mirrored: 3 turns, all dr < 0 (2 in band) → also CW. Far too few to decide; larger
  sample (`sg_014_sense2`, 7 levels, 30,000 starts) submitted. The sidecar was then moved out of the tracks folder
  so that the fitter computes crossings itself, as on 0175A.
- 08:13 **Validation on 0175A (`sg_003_valcmp`), z 7604–7804, ours vs published store:**

  | | ours | published |
  |---|---|---|
  | horizontal tracks in the 25 ribbons | 80,747 | 80,747 |
  | horizontal points | 12,942,692 | 12,942,692 |
  | points/track p10 / p50 / p90 / p99 | 11 / 47 / 413 / 1,606 | identical |
  | arclength p10 / p50 / p90 (vox) | 25.3 / 109.5 / 983.7 | identical |
  | vertical tracks touching z 7654–7754 | 221,074 | 175,510 |
  | vertical points inside z 7654–7754 | 4,540,656 | 4,359,575 (ours +4 %) |
  | per-slab vertical point counts, correlation | zx 0.991, zy 0.988 | |
  | vertical points/track p50 / p90 (whole track) | 25 / 70 | 33 / 411 |

  Horizontal tracks are **identical** (same counts, points and distributions), so parameters, binarisation and the
  per-slab algorithm match the organisers' run exactly. Vertical tracks differ only as expected from the band
  restriction: our zx/zy slabs are 200 slices tall instead of the whole scroll, so tracks end at the band and CC
  splits/size limits differ near the ends (+4 % points, more and shorter tracks). Overlay at ribbon z 7700
  (`results/sg_003/overlay_z7700.png`, left published, right ours, 1,800² crop at y 4600 x 3600; full slice
  `overlay_full_z7700_ds4.png`): the same skeleton lines everywhere. **Validation passed.** On 1203 the band is
  1,528 slices, so the band-end effect touches a smaller fraction of the vertical tracks.
- 08:14 **Sense, larger sample (`sg_014_sense2`, 7 levels z 9100–10300):** chains completing a full turn are rare
  (plain 51, mirrored 23). Pitch-band chains (5 < |dr| < 60 vox): plain geometry 14 with dr > 0 (CW) vs 2 with dr < 0;
  mirrored geometry 5 negative (= CW in the original) vs 3 positive. Combined **19 CW vs 5 ACW**, mostly at z 9700/9900
  (the method has a negative-drift bias, report 15, and the plain-geometry excess is positive, against that bias).
  Chains share tracks, so this is moderate evidence, comparable in strength to what decided 0358 (33 vs 15).
  **Decision: CW** (from the data; moderate support). Priors from other scrolls conflict (0826/343 CW, 306B's 0175A
  and our 0358 ACW), so no campaign prior was used.
- 08:15 Fit chunks submitted (CW, horizon 30,000, track-DT from 10,000, loop 1,980 s ⇒ job ≈ 38–40 min). The first
  submission failed in 5 s (`run_fit.sh` looked for 0175A's `prep/ALL.ok`; fixed to check this lane's stage
  markers) and its chain c02–c08 was skipped; resubmitted as `sg_101_cwb_c01` … `sg_108_cwb_c08`; c01 running 08:16.
  Fetch script (one pass): `spiral_1203/tools/fetch_chunks.sh`.
- 08:52 **Chunk 1 `sg_101_cwb_c01` ok** (job 2,161 s, loop 1,980 s): **0 → 2,725 steps.** Band z 9000–10400: 1,240,635
  of our tracks / 53.7 M points in the ROI, 2.49 M exact crossings (native in-memory, computed by the fitter),
  54.1 M trainable parameters; resident pools normals (Lasagna) 0.58 GiB + SDT 2.64 GiB; **nvidia-smi peak 9,650 MiB
  of 23,034** (0175A: 19.8 GB), so no band shrink. Loss 6,333 (0) → 318 (1,600) → 280 (2,600); satisfied_tracks
  12.9 %, satisfied_track_points 52.9 %. Checkpoint 649 MB downloaded (`results/sg_101_cwb_c01/`).
  Projection ≈ 2,700–3,000 steps per chunk ⇒ 30,000 after ≈ 10–11 chunks (≈ 6.5 L4-h). Queued c09–c11 (a chunk exits
  at once if 30,000 is reached), pipeline test gate `sg_200_gate_c01` (cpu, now), gates `sg_201_gate_c03`
  (≈ 8,500 steps, the "≈ 7,500" point), `sg_202_gate_c05` (≈ 14,500, the "≈ 15,000" point) and `sg_209_gate_final`
  (after c11, last tag, mesh tarballs, umbilicus, README).
- 09:20 Chunk 2 ok (job 2,046 s): 2,725 → **6,320** (3,595 steps, ≈ 0.55 s/step), peak 9,697 MiB. With ≈ 3,600 steps per
  chunk, c03 ends ≈ 9,900 and c04 ≈ 13,500, so I moved the gates: **gate 1 on c02 (step 6,320)** `sg_203_gate_c02`
  and **gate 2 on c04 (≈ 13,500)** `sg_204_gate_c04`; cancelled `sg_201_gate_c03` / `sg_202_gate_c05` before they ran.
- 09:25 Pipeline-test gate on step 2,725 (`sg_200_gate_c01`, cpu, 1,236 s): 120 windings w010–w129, 998 cm² in all,
  but only **62–63 windings inside the scroll** (CT found on ≥ 90 % of profiles; w070 0.91, w088 0.83, w100 0.64,
  w112 0.23): the scroll has fewer turns in this band than the exported range, the outer ~50 windings run partly or
  wholly outside it (`results/sg_200/xsect_c01_fitted_z9700_L1.jpg`). Median 4.8 cm² per inside winding.

  | step 2,725 (inside, n = 62/63) | raw fit | control | snapped | control |
  |---|---|---|---|---|
  | on m7 (`m7on`) mean | 0.224 | 0.236 | 0.602 | 0.248 |
  | nearest m7 ≤ 2 vox | 0.272 | 0.281 | 0.672 | 0.288 |
  | CT peak ≤ 2 vox | 0.404 | 0.405 | 0.442 | 0.403 |
  | CT z-score at the surface | +0.017 | +0.011 | +0.116 | +0.019 |

  Raw at chance (expected this early; 0175A was too). CW normals point away from the umbilicus (outward 0.99), as
  the CW convention gives. onsheet f_parallel of 4 renders (w022/w034/w047/w059): 0.86 / 0.87 / 0.86 / 0.88.
- 09:50 **Gate 1 (`sg_203_gate_c02`, step 6,320)**, inside windings n = 66 (raw) / 67 (snapped):

  | step 6,320 | raw fit | control (±8 vox) | snapped | control |
  |---|---|---|---|---|
  | on m7 (`m7on`) mean / median | 0.232 / 0.235 | 0.234 / 0.235 | 0.589 / 0.560 | 0.251 |
  | nearest m7 ≤ 2 vox mean | 0.276 | 0.280 | 0.662 | 0.292 |
  | CT peak ≤ 2 vox | 0.397 | 0.399 | 0.452 | 0.402 |
  | CT z-score at the surface | +0.001 | −0.000 | +0.130 | +0.004 |

  **Raw fit at chance**, as on 0175A at 7,592 (before the track-DT stage). Spread of per-winding raw on-m7 widened
  (p10/p50/p90 0.14/0.24/0.34 vs control 0.17/0.24/0.29; 2 windings > 0.40). By block (raw on-m7 mean): w010–029
  0.18, w030–049 0.25, w050–069 0.27, w070–089 0.23. onsheet f_parallel w023/w036/w049/w062 0.95/0.93/0.89/0.85.
  Mesh tarballs for this step kept on the VM results (`sg_203_gate_c02`), not downloaded.
- 10:55–13:45 This agent was paused by an account usage limit (≈ 10:55); the queued chain kept running on g3 and the
  coordinator fetched every chunk with `tools/fetch_chunks.sh`. Chunk end points: c03 9,906 · c04 13,383 · c05 ≈ 16,800 ·
  c06 ≈ 19,800 · c07 22,825 · c08 26,246 · c09 29,670 · **c10 30,000 (schedule complete)**; c11 exited at once.
  nvidia-smi peak 9.5–9.8 GB in every chunk. Final checkpoint local: `results/sg_110_cwb_c10/checkpoint_c10.ckpt`.
  **GPU used by the fit: 19,022 s ≈ 5.3 L4-h** (c01–c10 job times; no other GPU job in this lane).
- 11:00 **Gate 2 (`sg_204_gate_c04`, step 13,383)**, inside n = 69: raw on-m7 **0.318 vs control 0.235**, m7 ≤ 2 vox
  0.368 vs 0.276, CT peak ≤ 2 vox 0.412 vs 0.395, CT z-score +0.025 vs +0.007 — first clear departure from chance,
  after the track-DT stage switched on at 10,000 (as on 0175A at 13,738: 0.269 vs 0.201). onsheet f_parallel
  w023/w037/w051/w065 0.88/0.85/0.85/0.79.
- 13:46 **Final gate (`sg_209_gate_final`, step 30,000), raw fit, inside windings n = 65 of 120** (CT found ≥ 90 %):
  raw on-m7 **0.344 vs control 0.237** (p10/p50/p90 0.20/0.34/0.52); m7 ≤ 2 vox 0.401 vs 0.282; CT peak ≤ 2 vox 0.409
  vs 0.404; CT z-score +0.039 vs +0.010. By block (raw / control): w010–029 0.367/0.197, w030–049 0.329/0.254,
  w050–069 0.341/0.259, w070–089 0.322/0.237 (n 20/20/19/6). **17 inside windings > 0.40 raw, 64.4 cm²**, ordered
  by raw score: w018, w025, w026, w024, w015, w021, w029, w044, w016, w027, w046, w033, w051, w032, w063, w071, w037.
  Five more windings exceed 0.40 but lie partly outside the scroll (CT found < 0.9: w075, w080, w089, w094, w105);
  not counted.
- 14:02 Final gate, snapped (343 `snap_meshes.py`, m7 L0), inside n = 66: on-m7 **0.579 vs control 0.244**, m7 ≤ 2 vox
  0.658 vs 0.288, CT peak ≤ 2 vox 0.449 vs 0.413, CT z-score +0.121 vs +0.026 (0175A snapped: 0.668 / +0.197).
  onsheet f_parallel (41-layer 9.362 µm renders, mesh at layer 20, raw fitted) w023/w036/w049/w062 0.87/0.89/0.91/0.87.
  Mid layer of w049 by eye (`results/sg_209/render_w049_layer_mid.png`): papyrus fibre texture over most of the crop,
  clean on the right third (with two long cracks); dark off-sheet blobs scattered over the left two thirds, i.e. the
  raw surface follows the layer but leaves the sheet locally (w049 raw 0.30, not a passing winding).
- **Passing windings (raw > 0.40, inside):** table `results/sg_209/passing_windings.tsv` (raw, control, snapped,
  area, area in the usable fine band z 8000–11765, z range). Every passing winding spans z 9000–10400, so **in-band
  area = full area = 64.4 cm²** (1.65–7.92 cm² each). Snapped on-m7 of these 17: 0.58–0.82.
- **Normal sign:** our fitted (and snapped) windings have n = cross(dP/drow, dP/dcol), P = (z, y, x), pointing away from
  the umbilicus (outward 0.997; 0.96–0.97 against per-row curvature centres). The PHerc0841 control mesh of report 24
  (`hf/ctrl/20260220213127-on-20250821151531-9.366um.tifxyz`, on which `fwd` is the reading order) gives the same
  formula pointing outward too (0.993 against per-row curvature centres; 0841 has no umbilicus, so this side rests on
  curvature only). Same convention, and the reader's renderer flips both alike ⇒ **`fwd` is expected to be the
  reading order on our windings**; to be confirmed on any known-text check.
- Meshes: `/home/user/vesuvius/spiral_1203/meshes_c10/{fitted,snap}/wNNN` (tarballs `results/sg_209/meshes_*_c10.tgz`,
  README `results/sg_209/README_meshes.txt`, umbilicus official + smoothed, extract parameters).
- 14:05 Sent to the coordinator (`main`): mesh directory and the 17 passing windings in raw-score order.
  The 9 µm reader-v3 pass of the original step 4 was skipped on the coordinator's instruction (≥ 10 windings pass;
  the 2.4 µm read supersedes it). No ink was run in this lane.
