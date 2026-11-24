# 21 — Full-schedule spiral fit (lane `sf`)

Private. Started 2026-11-22 21:21 UTC. Outcome: **stopped at step 1, before any GPU use.** Neither candidate has
the full input set that the brief requires, so per the brief I did not run the fit. GPU used: 0 h. No `sf_` jobs were submitted.

## Progress log (UTC)
- 21:21 Read report 15 (all), 13, 18 (1545/0175A rows), the bridge README. `cjob status` OK: worker started 12:33,
  cq/br jobs running/pending on `main`. Nothing from the old `sp_` lane is on this VM.
- 21:22 Input check on the public bucket (S3 ListObjectsV2, anonymous), dl.ash2txt.org `spiral_datasets/`, and the
  community umbilicus set. Results in §1.
- 21:24 Looked at 7 transverse slices of PHerc0175A at level 2 (34.6 µm/px, HTTP range reads of single planes from the
  uncompressed zarr). See §2.
- 21:26 Decided to stop and hand back (§3).

## 1. Inputs per candidate (checked 22 Nov, 21:30)

The fit needs these inputs (report 15 §2): (a) tracks (`.vctracks` packed store at dl.ash2txt.org), (b) Lasagna
nx/ny/grad_mag, (c) m7 surface prediction, (d) umbilicus, which must be official or from the community set
`AlexeyDrobkovStrikesBack/herculaneum-umbilici`.

| Scroll | tracks | Lasagna normals | m7 | umbilicus (official / community) | usable? |
|---|---|---|---|---|---|
| PHerc1545 (vol 20250821151648, 9.362 µm) | **no**: `PHerc1545/` does not exist under `dl.ash2txt.org/datasets/spiral_datasets/` (index lists 0009B…0846A, ParisIV; no 1545, 1203, 1218 or 0846B) | yes (`lasagna/20250821151648-lasagna-20260419180421/`) | yes (+ normal grids) | official: none; community: **yes** (1545 is one of its ten) | **no: tracks missing** |
| PHerc0175A (vol 20250521115057, 8.64 µm) | yes: 11,098,366 tracks, 706.8 M points, coordinates 7.9 GiB, plus a published `.dbm.crossings.npz` | **no** (`representations/predictions/` has only `surfaces/`) | yes (+ normal grids) | official: **none** (`representations/umbilicus/` absent); community: **none** (its ten are 0191, 0257, 0268, 0358, 0800, 0813, 1203, 1218, 1447, 1545) | **no: normals and umbilicus missing** |

Possible extra candidates (not on the excluded lists): PHerc0483B and 0490A. Both have tracks but no Lasagna, no
umbilicus (official or community), and lower SQM (0.30/0.38 and 0.16/0.29). So there is no better third candidate.
Every eligible scroll that has all four inputs (0125, 0191, 0257, 0268, 0358, 0800, 0813, 0826) is on one of the
excluded lists.

Note: I could not list the community repository through the GitHub API (this session has no access and `add_repo` was
refused). I read its README through raw.githubusercontent.com (`/home/user/vesuvius/spiral_full/umb_dl/README.md`),
which lists the ten scrolls it covers and states this explicitly.

## 2. Winding regularity of PHerc0175A (by eye, level 2)
Montage `/home/user/vesuvius/spiral_full/views/0175A_montage_L2.jpg`, with z 2500, 4800, 6900, 7700 (top row) and
8400, 9500, 11000 (bottom row). Each slice is also saved separately as `views/0175A_z*_L2.png`, and there is a
centre crop at z 7700: `views/0175A_z7700_L2_crop.png`.
- z 6900–9500 shows **concentric, countable wraps around a clear, compact centre** (a small bright inclusion marks it).
  The scroll is flattened (oval) with S-folds and wedges at the upper outer edge. There is no crush star and no
  central void like 0358's. This is the most regular core of any scroll we have looked at for this lane.
- z 2500 and z 11000 are the broken ends (torn, partly missing).
- So the best band would be about **z 6900–8400** (1,500 slices). It matches the SQM best band (z 6912–8448) and lies
  between our clean windows 0175A k0/x0 (z 4800) and k1 (z 9500).

Sense: the published crossings file exists for 0175A. However, 306B's `erwin_data.sh` notes that on "an anticlockwise
fit (PHerc0175A)" the cache "failed to remap". So the 306B team ran 0175A **ACW** (their sense choice; we do not know
how they decided it). I did not run the §7.2 chain test.

## 3. Decision
The brief says: "If no candidate has the inputs, say so and stop — do not improvise a different method." PHerc1545
has no tracks, and generating tracks would be a new pipeline. PHerc0175A has neither normals nor an umbilicus from an
allowed source. **I stopped.** No GPU was used and the VM was not touched.

## 4. What would make 0175A runnable (for the coordinator to decide)
Both missing inputs have **documented fallbacks in the 306B winner's own recipe**. The fit method itself would not change:
1. Normals: `winners/pherc0306b-first-letters/discovery/x_recipe_erwin/tools/make_normals.py`. It builds a structure
   tensor from m7 L1 and writes Lasagna-format nx/ny group 2. It takes minutes on the L4. 306B used this for 0306B
   itself (no Lasagna there), with pools `sdt,normals` and no grad_mag.
2. Umbilicus: 306B's `build_umb.py` needs the VC3D binary `vc_gen_umbilicus`, which means a C++ build of villa on the
   VM. That is a new risk and probably 20–40 min. The cheaper alternative is to place the centre by hand on about
   10–15 level-2 slices of the band (the centre is visible, see the crop) and smooth it with 306B's `smooth_umb.py`
   (σ 75 slices). Either way the umbilicus would be our own, not a reviewed one.
3. Sense: ACW, as 306B used for this scroll (an assumption), or the §7.2 chain test on the published crossings
   (CPU, about 1 h).
4. Rough timing if approved soon: data prep on the VM (tracks 8 GB, m7, CT, SDT, synthetic normals) ≈ 45–60 min. The
   1,500-slice band 6900–8400 is less certain; s/step on the L4 is unmeasured (T4: 0.89–1.07 s/step), so 30,000 steps
   take ≈ 4–8 h. If the fit starts by about 01:00 UTC, the 08:00 deadline holds for a 1,000–1,500-slice band.
   To get the s/step number early, use a 1,000-slice band (z 7200–8200) if the go-ahead comes later.

## 5. Go-ahead (coordinator, ≈21:27): PHerc0175A z 6900–8400 with 306B fallbacks — log
- 21:28 Published fitted 0175A windings or checkpoint? **None found**: 306B repo (only 0306B lanes/meshes; the 0175A ACW fit
  is mentioned only in a comment of `erwin_data.sh`), HF `Prodigle/*` (0306B data and A-hn weights only), HF
  `rodriguescarson/eligible-scroll-spiral-fits` (0191, 0826 only), bucket `PHerc0175A/` (volumes, m7, photos only). → refit.
  306B's lane registry gives the sense source as "registry (upstream catalog/volume properties)"; no such property is in
  the public volume metadata, so ACW for 0175A stays an assumption taken from 306B.
- 21:29 Submitted `sf_001_setup` (cpu: villa @0fb38c45, `uv sync --python 3.14`) and `sf_010_data` (cpu:
  `code/prep_sf.sh`: whole tracks store 8.8 GB + placeholder DBM, m7 L0 band, m7 L1 and CT L1 over z 6100–9200, CT L4,
  native vc modules, fitter patches incl. the loop-time patch, surf-SDT with m7 L1 thr 128, CT-masked, 5 workers).
  Code tarball `jobs/sf_code_v1.tgz` = `/home/user/vesuvius/spiral_full/code/`.
- 21:30 Umbilicus (**our own, unreviewed**): centre of the innermost wrap placed by eye on 16 level-2 slices
  z 6500–8800 (every 150, last 200) on gridded 700×700 crops (`views/umb/grp*.jpg`), raw points
  `umb/umb_PHerc0175A_raw.json` (L0 x 4120–4600, y 5280–5500), smoothed with 306B `smooth_umb.py` (σ 75 slices,
  step 25; raw vs smoothed median 11.7 vox, max 75 vox) → `umb/umbilicus_PHerc0175A.json` (93 points, z 6500–8800).
  Overlay check `views/umb_overlay.jpg` (400×400 L2 crops centred on the smoothed axis): the marker lies inside the
  innermost 1–3 wraps on all 16 slices; on z 7100, 7550 and 8000 it sits ~20–40 L2 px (~80–160 L0 vox) off the
  visual centre along the long axis of the flattened core (the inner wraps are ellipses ~10:1 there). Accepted.
- 21:32 Submitted `sf_011_normals` (gpu: 306B `make_normals.py` on m7 L1 for z 6644–8656 → Lasagna-format nx/ny group 2;
  pools `sdt,normals` (no grad_mag; `dense_spacing_mode: phase` does not need it); spec for CW/ACW with the umbilicus)
  and chunks `sf_101_acw_c01` → `sf_102_acw_c02` (ACW, horizon 30,000, `loss_start_track_dt` 10,000, loop ≤ 26 min,
  job ≤ 45 min; every chunk copies `checkpoint_cKK.ckpt`, loss lines, fit-log tail, nvidia-smi and RAM traces to OUT).
  **Deviations from report 15's 0358 recipe:** normals synthesised from m7 (306B), not Lasagna; own umbilicus.
  **Crossings:** the published `.dbm.crossings.npz` is not used: 306B record that on their ACW 0175A fit the cache
  "failed to remap (walk crossing partner is out of range)" and the fallback rejects mirrored coordinates; the
  fitter computes exact crossings itself (~1 min on 0358).
  **Sense probe:** not run. Report 15 showed CW and ACW losses identical within noise at 1,000 steps, and a 300-step probe
  costs the full per-run start-up (~5–8 min resampling) for no discriminating metric; ACW is an assumption from 306B.
- 21:58 `sf_010_data` ok (1,621 s): tracks 8.8 GB, m7/CT bands, native modules, patches, surf-SDT 676 s → 8.7 GB
  (working z 6836–8464, 687 tiles, 69 % CT-empty chunks skipped). VM disk after: 129 GB free.
- 22:23 `sf_011_normals` ok (396 s incl. waiting for the GPU only after another lane's job): make_normals 38,531 group-2
  bricks (2.4 GiB, 28 s on the L4); at z ≈ 7650 32 % of the slice box carries a normal (≈ the scroll's cross-section),
  median horizontal component |(nx,ny)| 0.97 (normals lie in the slice plane, as expected for sheets wrapped around a
  vertical axis); pools (sdt, normals) 50 s. Chunk 1 `sf_101_acw_c01` started ≈22:23.
- 22:54 Chunk 1 `sf_101_acw_c01` ok (1,802 s job, loop 1,560 s): **0 → 1,737 steps**. Band: 2,012,154 tracks /
  145.5 M points in z 6900–8400, 7.10 M exact crossings (native in-memory, 34 s), 57.1 M trainable parameters;
  GPU pools normals 1.79 GiB + SDT 6.41 GiB; nvidia-smi peak 19,816 MiB of 23,034; host RAM used ≈ 6.6 GB.
  **Speed on the L4: 0.58–0.60 s/step** (9 steps per 5.2 s), after an 8.5-min stall at step 0 (native track
  resampling, recomputed every run) and ~4 min load + export per job. Loss 9,981 (0) → 2,874 (200) → 644 (800) →
  541 (1,000) → 390 (1,600); track_radius dominates; satisfied_tracks 22.4 %, satisfied_track_points 56.9 % at 1,737.
  Checkpoint 686 MB, downloaded in 97 s (`results/sf_101/checkpoint_c01.ckpt`).
  Decision: chunks 3+ use loop 2,100 s (job ≈ 39 min, ≈ 2,500 steps/chunk) to cut the per-chunk overhead share
  (chunk 2 was already queued with 1,560 s). Projection: step 30,000 after ≈ 13 chunks, ≈ 06:30 UTC if the GPU queue
  stays ours; GPU ≈ 8.2 L4-h. No band shrink needed. Queued c03–c08 and gate job `sf_201_gate_c03` (≈ 6,000 steps).
- 23:22 Chunk 2 `sf_102_acw_c02` ok (1,748 s). Chunk 3 started 23:22 (loop 2,100 s).
- 23:20 Pipeline test of the gate on the step-1,737 meshes (`sf_200_gate_c01`, CPU, 1,347 s; not one of the three gate
  points): 120 windings w010–w129 exported, 119 inside the scroll (CT found on ≥ 90 % of profiles), median 7.7 cm² per
  winding. Same measure as report 15 §3 (200 random vertices per winding, profiles along the normal, control = same
  vertices shifted U(−8, 8) vox):

  | step 1,737 (inside, n = 119) | raw fit | control | snapped (343 snap_meshes) | control |
  |---|---|---|---|---|
  | on m7 at the vertex (`m7on`) | 0.202 | 0.194 | 0.651 | 0.218 |
  | nearest m7 sheet ≤ 2 vox (`m7_le2`) | 0.272 | 0.266 | 0.748 | 0.288 |
  | CT peak ≤ 2 vox | 0.381 | — | 0.435 | 0.386 |
  | CT z-score at the surface | +0.010 | +0.013 | +0.167 | +0.034 |

  → raw fit at chance at 1,737 steps (as expected this early; on 0358 it was also at chance). The snapped surface is on
  m7 by construction (partly circular); the m7-independent CT z-score rises to +0.17 (0358 snapped: +0.12).
  Normals point to the umbilicus (outward fraction 0.004), as for ACW on 0358.
  Renders (41 layers, 8.64 µm, mesh layer 20, 1,600 × 1,820 px crop at mid-θ) of w034/w058/w082/w106: onsheet_score
  f_parallel 0.77 / 0.79 / 0.88 / 0.90. Mid layer of w082 by eye: papyrus fibre texture over much of the crop,
  interrupted by many dark gap blobs where the surface leaves the sheet — consistent with a surface that follows the
  layer direction but not one sheet. Render pipeline: remote L0 fetch ≈ 1 min per crop.
- 23:22 / 23:59 Chunk 2: 1,737 → 4,232 (2,495 steps, loop 1,560 s; the step-0 stall is shorter on resume). Chunk 3:
  4,232 → **7,592** (3,360 steps in a 2,100-s loop, ≈ 0.5 s/step incl. stall; job 2,228 s; peak 19,556 MiB). Loss 390
  (1,600) → 262 (4,000) → 213 (7,400). satisfied_tracks 22.4 % → 19.7 % → 17.6 %, satisfied_track_points 56.9 → 54.2 →
  51.2 % (both fall while the loss falls — the same pattern as on 0358). Projection: 30,000 after ≈ 8 more chunks,
  ≈ 05:00 UTC. Queued c09–c12. First gate `sf_201_gate_c03` running on the step-7,592 meshes (the "~6,000" point).
- 00:22 **Gate 1 (`sf_201_gate_c03`, step 7,592)** — inside-scroll windings (CT found ≥ 90 %), 200 vertices each:

  | step 7,592 | raw fit | control (±8 vox) | snapped | control |
  |---|---|---|---|---|
  | on m7 at the vertex (`m7on`), mean / median | 0.197 / 0.163 | 0.204 / 0.205 | 0.645 / 0.655 | 0.219 |
  | nearest m7 sheet ≤ 2 vox (`m7_le2`), mean / median | 0.264 / 0.218 | 0.274 / 0.279 | 0.747 | 0.292 |
  | CT peak ≤ 2 vox | 0.376 | 0.379 | 0.431 | 0.384 |
  | CT z-score at the surface | +0.020 | +0.016 | +0.188 | +0.049 |

  **Raw fit at chance** (n = 118). One change from step 1,737: the per-winding spread of the raw on-m7 fraction widened
  (p10/p50/p90 0.06 / 0.16 / 0.38 vs control 0.15 / 0.21 / 0.26; 8 windings > 0.4, e.g. w040 0.41, w056 0.45, w068 0.40):
  whole windings are now coherently on or coherently between sheets, i.e. the windings lock to the layer
  structure with a winding-dependent phase, rather than scattering at random. The mean stays at chance.
  onsheet_score f_parallel of the 4 renders 0.81 / 0.86 / 0.87 / 0.90 (w035 / w059 / w082 / w106).
  Cross-section crop z 7650 (`results/sf_201/xsect_c03_fitted_c_z7650_L1_crop.jpg`): windings follow the layer flow
  through folds, but many lines run through the dark gaps between sheets. Next gate after chunk 5 (`sf_203_gate_c05`, ≈ 14,000).
- 01:00 Chunk 4 ok (2,283 s): 7,592 → 10,781. Track-DT switched on at 10,000 (loss 203 → 292 at 10,600; track_radius 173,
  track_dt 91). satisfied_tracks 21.6 %, satisfied_track_points 57.3 % at 10,781 (up again from 17.6 / 51.2 %).
- 01:42 Chunk 5 ok (job 2,235 s wall fit): 10,781 → 13,738. satisfied_tracks 24.6 %, satisfied_track_points 61.4 %;
  loss 281 (track_radius 174, track_dt 78). Checkpoints c04 and c05 downloaded (only the newest kept locally).
- 01:42 **Gate 2 (`sf_203_gate_c05`, step 13,738)**, inside windings n = 118:

  | step 13,738 | raw fit | control | snapped | control |
  |---|---|---|---|---|
  | on m7 (`m7on`) mean / median | **0.269 / 0.280** | 0.201 / 0.205 | 0.653 | 0.224 |
  | nearest m7 ≤ 2 vox (`m7_le2`) mean / median | **0.348 / 0.380** | 0.271 / 0.273 | 0.749 | 0.297 |
  | CT peak ≤ 2 vox | 0.392 | 0.379 | 0.437 | 0.390 |
  | CT z-score at the surface | +0.042 | +0.020 | +0.189 | +0.051 |

  **First departure from chance** after the track-DT stage switched on: raw on-m7 +0.07 above control, 25 of 118
  windings > 0.4 (p10/p50/p90 0.06 / 0.28 / 0.47). It is concentrated in the outer windings: mean raw on-m7 by
  winding block w010–039 0.20, w040–069 0.22, w070–099 0.30, w100–129 0.35 (step 7,592: 0.17 / 0.20 / 0.20 / 0.21).
  Still below the brief's ≥ 0.40 bar. onsheet f_parallel 0.76 / 0.83 / 0.86 / 0.80.
  GPU note: at 01:42 another lane's GPU job (`br_z1_c846`) holds the GPU (fair share; this lane has used the most
  minutes), so chunk 6 waits. Queued final gate `sf_209_gate_final` (after c12; uses the last completed chunk's tag).
- 02:22 Chunk 6 running (started ≈ 02:10 after another lane's GPU job). Queue: c07–c12 (each exits immediately once 30,000 is
  reached), then `sf_209_gate_final` (CPU: gate on the last tag, mesh tarballs, umbilicus, README). Projected 30,000 ≈ 05:40.
- 02:25 **Run handed back at this point (forced hand-back)**, with the fit still running on the VM. Steps 30,000 and
  the final gate are not yet measured. Ink has not been started. Local fetcher script: `tools/fetch_chunks.sh` (downloads
  each finished chunk's checkpoint and logs, keeps only the newest checkpoint).
  GPU used so far ≈ 3.5 L4-h (c01 0.50, c02 0.49, c03 0.62, c04 0.63, c05 0.64, normals 0.11, c06 running); projected
  total ≈ 7.5 L4-h.
