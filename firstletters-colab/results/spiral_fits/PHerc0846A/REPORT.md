# 29 — Spiral fit on PHerc0846A inside its 2.4 µm band (lane `sg`, second fit)

Private. Started 2026-11-23 13:47 UTC on session g3 after the PHerc1203 fit (report 23). Working dir
`/home/user/vesuvius/spiral_0846A/` (code/, jobs/, results/, views/, umb/, tools/fetch_chunks.sh = one-pass fetcher).

## Progress log (UTC)
- 13:47 **Fine band verified** from our corrected registration
  (`submission/registration/transforms/PHerc0846A/20260319102732-to-20250728152254-r1/fit/transform.json`, similarity,
  LOO rms 1.1 coarse vox): fine z 0 maps to coarse z 1839–1892 and fine z 15136 to 5724–5776 depending on x/y, so the
  band fully covered at every x/y is coarse z 1892–5724; with 60 vox margin **z 1952–5664** (matches the reader's
  1950–5665). Report 09's earlier transform is not used.
- 13:48 Inputs on the bucket: m7 L0 (+ normal grids) yes; **no Lasagna** and **no official umbilicus**
  (`representations/` has only `predictions/surfaces/`; the community umbilicus set does not cover 0846A) ⇒ 0175A
  fallbacks: synthetic normals from m7 (306B `make_normals.py`) and our own umbilicus.
- 13:48 Band choice: level-2 slices z 2100–5550 (`views/0846A_montage_L2.jpg`, plus 9 more slices 2350–3850).
  z 2100 is the scroll's broken lower end; **z 2350–3850 shows the clearest concentric spiral of any scroll we have
  looked at** (oval wraps around a visible centre with a bright inclusion); from z ≈ 4100 up the cross-section becomes
  folded/crumpled (S-folds, sheared lobes, 4600–5550 chaotic). **Band z 2450–3750 (1,300 slices)**, ≥ 498 slices
  inside the usable band at both ends.
- 13:49 **Own umbilicus (unreviewed, clearly ours):** centre of the innermost wrap placed by eye on 11 level-2 slices
  z 2350–3850 (`views/umb_grid0.jpg`, `umb_grid1.jpg`), raw `umb/umb_PHerc0846A_raw.json` (L0 x 3360–3720,
  y 3780–3940), smoothed with 306B `smooth_umb.py` σ 75 → `umb/umbilicus_PHerc0846A.json` (raw vs smoothed median
  15 vox, max 59). Overlay `views/umb_overlay.jpg`: inside the innermost wrap on all 6 checked slices.
- 13:50 Submitted `sg_301_846_data` (published tracks + crossings, m7 L0/L1, CT L4/L1, spec, surf-SDT),
  `sg_302_846_sense`, `sg_303_846_normals` (GPU).
- 14:04 `sg_301` ok (906 s; tracks 388 s, SDT 427 s, 5.9 GB). **Problem: the published 0846A track store starts at
  z 4500** (its `.extract.json`: `z_min 4500`, `effective_z_bounds [4500, 14019]`), so it has **no tracks in our band**
  (the sense job found 0 transverse tracks at z 2550–3550). Inside the fine band the published tracks cover only
  z 4500–5664, which is the crumpled part. Decision: keep the regular band and make **our own tracks** for z 2386–3814
  with exactly the pipeline validated on 0175A in report 23 (pinned extractor, published parameters; horizontal tracks
  identical to the published ones, vertical r = 0.99). `sg_304_846_tracks` (cpu) also re-points the spec to our store
  and repeats the sense chain test on it. (The coordinator's note said to use the published tracks; they do not exist
  for this z range. Alternative would be a band at 4500–5600 on published tracks, which is the chaotic region.)
- 14:09 `sg_303_846_normals` ok (295 s GPU): synthetic normals from m7 L1 for z 2194–4006 (at z ≈ 3100, 33.7 % of the
  slice box carries a normal, median |(nx,ny)| 0.963), pools sdt + normals.
- 14:30 `sg_304_846_tracks` ok (1,397 s): **our 0846A store: 1,690,832 tracks, 93,224,175 points** for z 2386–3814
  (911 keys, 11 workers, 1,340 s incl. packing), crossings sidecar 5.63 M events (then moved aside).
- **Sense (chain test on our tracks, own umbilicus, 30,000 starts at 6 levels z 2550–3550):** here the spiral is regular
  enough that many chains close a turn (474 plain, 382 mirrored). Pitch-band chains (5 < |dr| < 60 vox): plain geometry
  140 dr > 0 vs 25 dr < 0; mirrored 160 dr < 0 vs 18 dr > 0; combined **300 consistent with CW vs 43 with ACW**,
  consistent at every level (plain-geometry positive excess against the method's negative-drift bias, and it flips
  under mirroring). **Decision: CW, strong support from the data.**
- 14:39 Submitted `sg_401_cw_c01` … `sg_411_cw_c11` (CW, horizon 30,000, track-DT 10,000, loop 1,980 s), gates
  `sg_503_846_gate_c02`, `sg_504_846_gate_c04`, `sg_509_846_gate_final` (after c11). c01 running 14:38.
  One-pass fetcher: `spiral_0846A/tools/fetch_chunks.sh`.
- 15:15 **Chunk 1 ok: 0 → 2,701 steps**; band z 2450–3750: 1,491,530 of our tracks / 80.2 M points, 4.09 M exact
  crossings (fitter, in memory), 51.1 M parameters, pools normals 1.15 GiB + SDT 3.89 GiB, **nvidia-smi peak 12.6 GB**
  of 23; satisfied_tracks 17.8 % / points 57.0 % (1203 at the same point: 12.9 / 52.9 %). Checkpoint downloaded.
- 16:02 Chunk 2: 2,701 → 6,269; 16:43 chunk 3: → 9,753 (peak 12.4 GB; ≈ 3,500 steps per 38-min chunk, as on 1203).
- 16:40 **Gate 1 (`sg_503_846_gate_c02`, step 6,269)**, inside windings: raw on-m7 0.192 vs control 0.188, m7 ≤ 2 vox
  0.244 vs 0.244, CT peak ≤ 2 vox 0.414 vs 0.413, CT z-score +0.004 vs +0.008 → **raw at chance** (as for 1203 and
  0175A before the track-DT stage). Snapped: on-m7 0.588 vs 0.223, CT z-score +0.106. onsheet f_parallel
  w033/w054/w075/w096 0.77/0.80/0.89/0.92.
- 17:25 Chunk 4: → 13,051 (peak 12.6 GB). **Gate 2 (`sg_504_846_gate_c04`, step 13,051)**, inside n = 106: raw on-m7
  **0.247 vs control 0.188** (p10/p50/p90 0.09/0.25/0.41; 11 windings > 0.40), m7 ≤ 2 vox 0.312 vs 0.245, CT peak ≤ 2
  vox 0.418 vs 0.413, CT z-score +0.016 vs +0.008 → departure from chance after track-DT (+0.06; 1203 at this point
  +0.08, 0175A +0.07). onsheet f_parallel w033/w054/w075/w096 0.73/0.74/0.84/0.96.
