# Small tools and measured results for First Letters work on one free GPU

Author: JorisViaudQuantAI · Licence: MIT (see `LICENSE`; vendored third-party code keeps its own MIT licence)

This repository bundles five small contributions that came out of running the First Letters workflow on
real scroll data with one Google Colab T4 GPU. Each section says what the piece is, how to reproduce it,
what evidence backs it, and what it does **not** show. Numbers marked *measured* come from logs of runs on
real data; anything else is labelled.

| # | Contribution | Folder | Status |
|---|---|---|---|
| 1 | Fine↔coarse registration of the 2.4 µm rescans of PHerc. 1203 and PHerc. 0846A to their First-Letters 9.36 µm volumes, in the bucket's `transforms/` format | `registration/` | measured, CPU only |
| 2 | Headless First Letters window on a free Colab T4: VC3D AppImage GrowPatch + render + reader-v3/ink_9um ensemble, timings, pitfalls, and a Drive-fed job worker | `colab_pipeline/`, `ink/` | one window measured end to end; packaged scripts parametrised afterwards (see limits) |
| 3 | Model comparison on a known-text control (PHerc. 0841): agreement (AUC) with the organisers' 2.4 µm ink map by depth offset, combinations, reversed-layer control, and a text-likeness triage score with its calibration | `controls/` | measured on one region; combination and thresholds chosen on that same region |
| 4 | Negative results with coordinates (12 regions searched by us on 4 scrolls + Scroll 1: 10 negatives, 2 unresolved or uninformative, one location withheld; plus 3 PHerc. 1203 windows screened by Codex) and their meshes | `negative_results/` | measured / by eye, labelled per row |
| 5 | `--precision fp16` for the official `hecate.py`: 3.0× faster than its fp32 default end to end on a T4 (patched official script, one small crop); 3.9× model throughput in our wrapper | `hecate_fp16/` | measured on a T4, both ways, one crop of one scroll |

AI assistance: the code, the runs and this text were produced with AI coding agents, directed by the repository
owner. See "How this was made" at the end.

---

## 1. Registration of the 2.4 µm rescans (PHerc. 1203, PHerc. 0846A)

**What.** Both scrolls have a 2.403 µm, 77 keV rescan of a 3.6 cm tall band (BM18, scanned a day after the
9.362 µm scan), but the open-data bucket has **no `transforms/` folder for either sample** (checked
2026-11-21: `PHerc1203/transforms/` and `PHerc0846A/transforms/` list no keys). Without a transform, a mesh or a
finding on one scan cannot be moved to the other. This folder provides a 7-parameter similarity transform
(scale, rotation, translation) for each pair, in the same file layout and schema as the organisers'
transforms (e.g. `PHerc0841/transforms/20260319124803-to-20250821151531-r1/`):

```
registration/transforms/PHerc1203/20260319130212-to-20250820131727-r1/fit/transform.json, provenance.json
registration/transforms/PHerc0846A/20260319102732-to-20250728152254-r1/fit/transform.json, provenance.json
```

`transformation_matrix` maps moving (2.4 µm) level-0 voxel xyz to fixed (9.36 µm) level-0 voxel xyz, as in
`foundation/volume-registration/transform_schema.json` of villa (required keys and shapes checked).

| pair | landmarks | model | in-sample RMS | leave-one-out RMS (max) | scale | rotation |
|---|---|---|---|---|---|---|
| PHerc1203 `20260319130212 → 20250820131727` | 5 (NCC 0.88–0.96) | similarity | 0.67 coarse vox | **1.23 coarse vox ≈ 4.8 fine vox ≈ 11.5 µm** (1.65) | 0.256633 | 0.09° |
| PHerc0846A `20260319102732 → 20250728152254` | 7 (NCC 0.66–0.89) | similarity | 0.72 coarse vox | **1.14 coarse vox ≈ 4.4 fine vox ≈ 10.6 µm** (1.89) | 0.256627 | 0.31° |

For comparison (same landmarks): a scale-plus-translation model without rotation gives a leave-one-out RMS of
2.35 (1203) and **7.8** (0846A) coarse voxels; a 12-parameter affine gives 2.36 and 1.41 (it overfits 5–7
points). The 0.31° tilt of PHerc0846A matters: ignoring it puts the bottom of the band 13–14 coarse voxels
(≈130 µm, several sheets) off. Figure `registration/figures/PHerc0846A_checker_z2276_new_vs_old.jpg`
(9.36 µm slice | checkerboard with the 2.4 µm slice mapped by the new transform | by the no-rotation one):
sheets run continuously across the tiles with the rotation, and break with the old transform.

**Same-mesh check (PHerc1203, earlier no-rotation transform, which differs from the new one by ≤ 2 coarse
voxels over the test mesh):** the same mesh rendered from both volumes shows the same papyrus, fibres and
cracks (`figures/p2_coarse9um_vs_fineL2_same_mesh.jpg`, r = 0.84 between the 9.36 µm render and the 2.4 µm
render pooled to 9.6 µm; `figures/raw031_coarse9um_vs_fineL2_same_mesh.jpg`, a public 9 µm mesh mapped to
2.4 µm, r = 0.71).

**Reproduce (CPU, a few minutes per landmark, reads ~MBs by HTTP range requests; run from `firstletters-colab/`):**

```bash
pip install numpy opencv-python requests tifffile
# one landmark: region of fine level-2 slice 900 centred at (4000, 3311), search coarse z 8850..8878
python registration/reg_point.py --fine PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr \
  --coarse PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr --fine-um 2.403 --coarse-um 9.362 \
  --prior 13246.5 13246.5 3380 3418 --z2 900 --cx2 4000 --cy2 3311 --half 300 --zc 8850 8878 2
# -> ncc=0.957 zc=8862 ... fine_L0(16000,13244,z3600) -> coarse(4086.0,3420.0,8862)
python registration/fit_landmarks.py registration/landmarks.json --sample PHerc1203 --out registration/transforms/PHerc1203
# move a tifxyz mesh: fine level-2 mesh -> 9.36 um frame, or (--inverse) a 9.36 um mesh -> 2.4 um frame
python registration/apply_transform_tifxyz.py <transform.json> <mesh_dir> <out_dir> --level 2
python registration/apply_transform_tifxyz.py <transform.json> <mesh_dir> <out_dir> --inverse --zrange 8000 11765
python registration/test_apply_transform_tifxyz.py   # self-test on a 4-vertex fixture (no download)
```

`apply_transform_tifxyz.py` keeps a vertex only if x, y and z are all finite and non-negative, writes every
other vertex as −1, computes the output `bbox` over valid vertices only, and sets `volume` / `volume_id` in
`meta.json` to the destination volume (from `transform.json` or the pair's `provenance.json`, or
`--dst-volume-id`); the source values are kept under `source_frame`.

**Limits.**
- Below the organisers' publishing bar. Their `volreg` records (villa `foundation/volume-registration/pipeline`)
  use hundreds of dense-correlation landmarks, a held-out patch review and a thin-plate-spline bend, and require a
  median hold-out error of ≤ 2 fine voxels. Ours is ~4–5 fine voxels RMS from 5–7 slice matches. That is enough
  to move a mesh onto the same sheet (sheets are ~10 coarse voxels apart) and to compare the two scans, but not
  for voxel-exact work. The files can serve as a `--start` transform for `volreg`.
- Landmarks lie within ~8 mm in x/y of the centre of the 2.4 µm field of view (not an umbilicus) and span the
  band's height; for PHerc. 1203 that is 3.4–11.2 mm from the official umbilicus
  (`PHerc1203/representations/umbilicus/20250820131727-umbilicus-20260924035533.json`); PHerc. 0846A has no
  published umbilicus. Accuracy farther out is extrapolated, not measured.
- Each landmark is a 2-D slice match with coarse-z steps of 2; sub-voxel conventions (pixel centres between
  levels) are not modelled (< 0.5 coarse voxel).
- No deformable part: if the scroll changed shape between the two scans, it is not captured.

## 2. A First Letters window, headless, on a free Colab T4

**What.** The sequence normal grids → `vc_grow_seg_from_seed` (GrowPatch) → `vc_render_tifxyz` (41 layers) →
reader-v3 + ink_9um (seeds 42, 43) at five depth shifts, both layer orders → mean-of-z-scores ensemble →
previews with a 1 cm bar, run without a GUI from the VC3D Linux AppImage on a Colab T4, plus a small worker that
turns a Colab tab into a job runner fed from a Google Drive folder.

**Measured on a T4 (PHerc0175A, seed (5423, 5806, 6960), 7.45 cm² patch, 2026-11-20/21):**

| step | time | notes |
|---|---|---|
| normal grids (seed z ± 3400) | 21 min (1280 s) | 9.4 GB, reusable for further seeds of the same scroll |
| GrowPatch, 80 generations, 2 threads | 8 min (489 s) | CPU |
| render, 41 layers, CT streamed from S3 | 4 min (250 s) | 3240 × 3240 px, 517 MB; 2.3 GB chunk cache |
| reader-v3, 5 shifts × 2 faces | 7 min (425 s) | ~640 cm²/h per pass |
| ink_9um s42 + s43, same | 16 min (937 s) | |
| previews | ~1 min | |
| **total** | **≈ 1 h for the first window**; ≈ 35 min for another seed of the same scroll (derived: the same steps without the grid download, not separately run) | peak VRAM 2.85 GB, disk ≈ 13 GB |

**Pitfalls hit and fixed** (each is commented in `colab_pipeline/jobs/env.sh`):
1. The AppImage needs GL/X/font libraries absent from Colab even when headless → `apt-get install` list.
2. No FUSE on Colab → `--appimage-extract` once, then `squashfs-root/AppRun <tool>`.
3. VC3D tools cache streamed chunks; point the cache at the large disk via `VC3D.ini` and keep 5 GiB free.
4. `python -m venv` fails on Colab (no `ensurepip`) → `pip install --target` and `PYTHONPATH`.
5. `zarr` 3 vs the v2 stores used by the helpers → pin `zarr<3`, `numcodecs<0.16`, which then need `deprecated`, `wrapt`.
6. Hecate inside a T4 job: with bf16 autocast it ran at ~12 cm²/h per pass and a 90-minute job timed out with
   no output → dropped from the per-window recipe; see section 5 for the fp16 fix.
7. `vc_render_tifxyz` now takes the volume URL directly with `--volume` (the older `--remote-url` used in some
   public recipes is gone).
8. The GrowPatch surface sat ~2 voxels off the brightest papyrus layer → always sweep depth (−3…+1 here).
9. Free Colab sessions are reclaimed: the worker writes `DONE.txt` per job and skips finished jobs, so a restart
   resumes; jobs cache under `/content/cache` and must not share cache paths when two copies may run.

**Reproduce.** On a Colab GPU runtime, in one `%%bash` cell or a terminal (each `!` line of a notebook is a
separate shell, so exported variables would be lost):

```bash
# this package is on the branch claude/firstletters-colab (not on main); add `git -C /content/repo checkout <commit>` to pin
git clone -b claude/firstletters-colab https://github.com/JorisViaudQuantAI/Scrollprize.git /content/repo
P=/content/repo/firstletters-colab
export INPUTS=/content/inputs OUT=/content/out/PHerc0175A_5423_5806_6960
mkdir -p $INPUTS && cp $P/colab_pipeline/fl_helpers.py $INPUTS/ && cp -r $P/ink $INPUTS/ink
# defaults reproduce the PHerc0175A window; export SCROLL UM SEED_X SEED_Y SEED_Z VOL_URL M7_URL NG_KEY for another,
# and use one OUT per seed
bash $P/colab_pipeline/jobs/01_grow.sh
MESH=$(cat $OUT/mesh_path.txt) bash $P/colab_pipeline/jobs/02_render_ink.sh
```

Pass the mesh explicitly as above. Without `MESH`, `02_render_ink.sh` falls back to a hand-off file under
`/content/cache/grow/` named after scroll, seed and generations, so it only finds the grow made with the same
`SCROLL`, `SEED_X/Y/Z` and `GENS`.

The step-by-step notebook `colab_pipeline/first_letters_colab.ipynb` (parameters, seed snapping, a positive
control on the PHerc. 343 winning mesh, Hecate and villa-CLI options, depth-shuffle control) documents the same
route for a person at the keyboard.

**Drive job worker** (`colab_pipeline/gpu_worker/`): `worker_v002.py` polls a Drive folder `jobs/`, runs each
`*.sh` (header lines `# timeout:`, `# slot: cpu`, `# after: <job>`), uploads `$OUT` and a rolling `log.txt` to
`results/<job>/`, writes `DONE.txt`, never re-runs a finished job, runs up to two jobs (one GPU), can be
cancelled per job by a `CANCEL_<job>` file, and updates itself when a higher `worker_vNNN.py` appears.
`python make_notebook_v2.py --root-id … --jobs-id … --results-id … --inputs-id …` writes the one-cell bootstrap
notebook for your own Drive folders (the notebook's user instructions are in French; the code is not). Tests:
`python test_worker_v2.py` (22 tests, CPU, local folder instead of Drive; pass).

**What already existed.** The PHerc. 343 winner rendered with the AppImage from a script; the 306B team ran
GrowPatch from scripts on rented GPUs; the PHerc. 0826 write-up documented the spiral-fit route on RunPod;
the VC3D tutorial documents `APPIMAGE_EXTRACT_AND_RUN` for containers. Older Colab notebooks exist for ink
training (2024). I found no published Colab recipe for GrowPatch + render + current ink models with measured
times; that is the new part, together with the pitfall list and the worker.

**Limits.**
- One window on one scroll was measured. The packaged `jobs/*.sh` are parametrised copies of the jobs that ran
  (`rcp_012`, `rcp_025`): only variables, paths and the mesh hand-off changed. They pass `bash -n`. They were
  re-run once as shipped, from a fresh clone at `37011ec`, on one new PHerc0175A window (8-core Colab VM with a T4:
  `01_grow.sh` 1103 s, `02_render_ink.sh` 1323 s; details in `collab/claude-003.md`). The fallback hand-off without
  `MESH` and the 2-core free-tier timings were not re-checked in that run.
- The notebook was written before the runs; its helper and parameter cells were executed on CPU, its GPU cells
  were not executed as a notebook (the same steps ran as worker jobs). Cells are marked `UNTESTED` where so.
- Colab compute-unit prices and free-tier limits change; the ≈ 1 h per window is for a T4.

## 3. Known-text control on PHerc. 0841 w00: models, depth, combinations, and a triage score

**What.** On a 65-layer render of PHerc. 0841 w00 (9.366 µm, known text) we scored four public 9 µm models
against the organisers' 2.4 µm `new_canon_autoresearch_recipe` map resampled to the render grid ("ink" =
key ≥ 128), inside a 2.6 cm² ROI, at depth shifts along the render normal and with the layer order reversed.
The reference is itself a model output, so every AUC below measures **agreement with another model**, not
accuracy against human-labelled ink.
Raw numbers: `controls/evidence/controls_summary.json`; panel `controls/fl_002_controls_panel_0841.jpg`.

| model | z −4 | −2 | **0** | +2 | +4 | reversed |
|---|---|---|---|---|---|---|
| reader-v3 | 0.813 | 0.837 | **0.863** | 0.857 | 0.840 | 0.538 |
| Hecate 9.6 µm (fp16) | 0.782 | 0.814 | **0.846** | 0.787 | 0.737 | 0.552 |
| ink_9um s42 75k | 0.716 | 0.747 | 0.733 | 0.745 | 0.719 | 0.539 |
| ink_9um s43 75k | 0.709 | 0.735 | 0.722 | 0.715 | 0.710 | 0.555 |

Combinations: 305 were scored on this same ROI (model subset × rule mean / geometric mean / mean of z-scores
× single depth or average of three neighbouring shifts × centre shift), and the best was picked afterwards:
**reader-v3 + Hecate, mean, depth-averaged over −2/0/+2: AUC 0.891**. Because it was selected post hoc on the
region it is scored on, 0.891 is an optimistic in-sample model-agreement figure, not an expected performance.
This combination is frozen as stated (reader-v3 + Hecate, mean, depth-averaged over −2/0/+2) and is to be used
unchanged on any other scroll or region; it has not yet been tested elsewhere. Same ROI, for reference:
reader-v3 alone 0.863 (0.874 with the same depth averaging); adding ink_9um to reader-v3 lowers agreement
(0.840 with the mean rule at z 0; 0.864 for the best rule with depth averaging, versus 0.874 for reader-v3
alone); the four-model mean of z-scores used by the PHerc. 343 winner gives 0.875. Off-sheet (z ±20): 0.46. At z ±6
reader-v3 draws letter-like strokes in *other places* — probably text of neighbouring wraps, a warning for deep
depth sweeps.

**Triage score** (`controls/score_text.py`, calibration `controls/calibration.json`): 14 mm windows scored for
row periodicity at a 4–7 mm line pitch along the fibre directions, letter-sized stroke statistics, depth and
cross-model stability, and forward/reversed asymmetry. Calibrated on 210 control windows (reversed layers,
off-sheet z ±20, a noise stack with the render's per-layer statistics):

| window class (reader-v3, z 0) | n | score p10 / p50 / p90 / max |
|---|---|---|
| dense text (top quartile of ink cover) | 8 | 3.75 / 4.10 / 4.21 / 4.29 |
| sparse text (bottom quartile) | 8 | 1.28 / 1.52 / 2.03 / 2.05 |
| reversed layers | 30 | −0.44 / 0.14 / 0.89 / 1.29 |
| off-sheet | 120 | −0.34 / 1.13 / 2.38 / 3.07 |
| noise | 60 | −2.99 / −2.07 / −1.59 / −1.15 |

Reversed layer order is a diagnostic, not proof. As Codex notes (`collab/codex-002.md`), which order shows
ink depends on the model and on the render orientation, so reversing the layers can suppress genuine ink as well
as artefacts. Our scans run both orders, but the asymmetry gate above (and the forward/reverse ratios in
section 4) assume that the forward order is the right one for the render.

Every dense-text window scores ≥ 3.70 and every control window ≤ 3.07. Measured line pitch on real text:
5.7 mm. Decision rule used for the scan in section 4: ≥ 3.1 plus gates (asymmetry ≥ +0.5, depth stability
≥ 0.7, cross-model stability ≥ 0.3) = candidate; 2.0–3.1 = look by eye; below = nothing. These thresholds
were set on the same segment's windows (8 per text class) and have not been validated on another scroll.

**Blind spot (stated plainly): sparse text scores 1.3–2.0, inside the control range. The score finds dense
rows of text, not isolated letters.** A "nothing" from it does not exclude a few letters.

**Reproduce:** `python controls/analyze_controls.py --render <render dir> --maps <model maps…> --canon <answer key> --out <dir>`
(see `--help` for `--offset`, `--textroi` and the window size); the maps come from `ink/infer_ink.py` with
`--z-shifts` and `--direction both`.

**Limits.** One region of one scroll. The answer key is itself a model output (2.4 µm). The render is ours,
not the official surface volume (r ≈ 0.9 between them), and reader-v3's AUC here (0.863) is slightly below the
0.865–0.882 we measured on the official surface volume (that run's numbers are not in `evidence/`). The calibration windows overlap and come from one
segment, so the effective sample size is small. Off-sheet "controls" may contain real ink from other wraps
(0841 is written throughout), so they are conservative negatives.

## 4. Negative results with coordinates

`negative_results/negatives.csv` lists every region we searched (and three screened by Codex, marked in `source`), with scroll, volume id, mesh or seed, bounding
box in level-0 voxels of the listed volume, area, models, depths, faces, verdict, caveats and source run.
`negative_results/meshes.tar.xz` (1.4 MB) holds our own meshes (tifxyz) for rows N1–N5 and N7.

| id | scroll | where (level-0 voxels) | cm² | models | verdict |
|---|---|---|---|---|---|
| N1 | PHerc0175A (8.64 µm) | GrowPatch seed (5423,5806,6960); x 3764–5965, y 4678–6412, z 5350–8499 | 7.45 | reader-v3, ink_9um ×2, 5 depths, both faces | no ink (blobs, no rows/strokes) |
| N2 | PHerc0175A | x 3177–5224, y 5075–6423, z 5131–7433 (overlaps N1) | 3.91 | reader-v3, ink_9um, 5 depths, both | score 1.90 = nothing |
| N3 | PHerc0175A | x 3378–4177, y 5406–6655, z 2565–5030 | 2.18 | same | score 1.44 = nothing |
| N4 | PHerc0125 (9.362 µm) | x 2819–5128, y 3428–4024, z 9266–11575 | 3.49 | same | 3.27 but fails gates; not text |
| N5 | PHerc0826 | x 2136–4075, y 4352–5923, z 8219–9391 | 1.16 | same | nothing (weak: torn sheets) |
| N6 | PHerc1203 | public `auto_grown_20251005221856743` crop; x 2203–4188, y 3513–4124, z 5785–8585 | 4.69 | same | 3.78 but fails gates; blobs |
| N7 | PHerc1203 **2.4 µm** rescan | own patches; (9.36 µm frame) x 2300–3900, y 3800–4600, z 8100–10130 | 2.8 | ink_canonical_2um, both orders, ±8 layers; reader-v3 at 9.36 µm | no ink (orientation-symmetric blobs) |
| N8 | PHerc1203 2.4 µm | public mesh `…230830031` block; x 1815–2738, y 4887–5086, z 8503–9626 | 0.56 | ink_canonical_2um | no ink |
| N9 | PHerc1203 2.4 µm | one block of a public mesh; coordinates withheld until the 9.36 µm cross-check is done | 0.56 | ink_canonical_2um | **unresolved** 3×5 mm cloud (fwd/rev 4.1); not letter-like |
| N10 | PHerc1203 2.4 µm | two blocks on crumpled surface | 1.12 | ink_canonical_2um | uninformative |
| N11 | Scroll 1 (PHerc. Paris. 4), 2.4 µm | agraphon of `20260623141924-w010-027`, canvas x 116160–170160 | ~140 | published predictions only | no line structure (confirms the organisers' statement) |
| N12 | Scroll 1, 2.4 µm | unmeshed core, z 25500/29000–72900, inside the innermost meshed wrap | ~2 cm wide | published 3D ink volumes | no ink; core holds only ~2 cm of papyrus |
| N13 | PHerc1203 (9.362 µm) | public mesh `…230830031`, Codex window A; x 1858–4412, y 4729–5087, z 7504–9265 | — | Reader v3 (both layer orders, overlapping 384 px tiles), Hecate on selected regions; 25 depths −12…+12 | screened, no ten-letter sequence identified (Codex) |
| N14 | PHerc1203 | same mesh, window B; x 1752–4620, y 4441–5069, z 5864–7710 | — | same | screened, no ten-letter sequence identified (Codex) |
| N15 | PHerc1203 | same mesh, window C; x 1519–3345, y 4500–4907, z 6030–7780 | — | same | screened, no ten-letter sequence identified (Codex) |

Rows N13–N15 are by Codex (`collab/codex-002.md`; surface rectangles in the CSV). Their boxes round the surface
bounds outward, so they enclose curved sheets and include unscreened tissue off the mesh; B and C overlap. Our
own 9.36 µm scan lane on PHerc. 1203 also covers the public mesh `…230830031`, so these windows will serve as an
independent replication once our verdicts are in.

Controls used to read these: PHerc. 0841 known text processed the same way (section 3); for the 2.4 µm rows,
the same code path reproduces the organisers' published 2.4 µm map of PHerc. 0841 (r = 0.929) and gives
8.5× more ink pixels forward than reversed on text, versus 0.9–2.2× on the PHerc. 1203 blocks.

**Limits.** Small areas (≈ 28 cm² of our own searching on eligible scrolls, rows N1–N10, counting the N1/N2
overlap and the unresolved N9/N10), mostly near scroll centres; a
negative with these models says nothing about stronger future models; rows N2–N6 rely on the triage score,
which cannot see isolated letters (section 3); N9 and N10 are not negatives. Public negatives by others, for
context: PHerc. 0826 z 10000–11000 w010–w065 (spiral fit, ink_9um), PHerc. 0211 same window (later won
elsewhere), PHerc. 0846A ≈ 107 cm² of GrowPatch meshes with Hecate.

## 5. `--precision fp16` for the official Hecate script

**What.** The official `hecate.py` (huggingface.co/scrollprize/hecate, unchanged since 2026-09-15; villa has no
Hecate code) offers `fp32` (default) and `bf16`. GPUs before Ampere, including the T4 of free Colab, have no
fast bf16 path. `hecate_fp16/0001-hecate-add-fp16-precision.patch` adds an `fp16` choice (3 hunks; default
unchanged).

Two measurements, on a Colab T4, kept separate because they measure different things.

**(a) Throughput in our wrapper** (earlier job; PHerc. 0841 w00 text tile, 0.92 cm², batch 32, same model
weights and autocast call as `hecate.py`, run through `ink/infer_ink.py --model hecate --hecate-amp …`; cm²/h per
pass as reported by the wrapper; `cudnn.benchmark` was on, which `hecate.py` does not set):

| precision | cm²/h per pass | vs fp32 output (uint8) |
|---|---|---|
| bf16 autocast | 12.2 | max diff 13, mean 0.205, r 0.9999 |
| fp32 (official default) | 17.9 | — |
| **fp16 autocast** | **69.9** | **max diff 1, mean 0.024, r 0.999992** |

**(b) The patched official script, end to end** (2026-11-21, `hecate_fp16/measure_t4.sh` as committed at
`fca6b04`; Colab Tesla T4, 15 GB, torch 2.14.1+cu130; official `hecate.py` + the fp16 patch; one 27 × 999 × 999
crop of the public PHerc. 841 w00 surface volume resampled to 9.6 µm by `prepare_crop.py`; batch size 32; wall
seconds per fresh process, model load included). Input crop, recomputed here on CPU with the same script and
defaults (key and `crop_yx` [1408, 2048, 1024], 64 chunks read, none absent): sha256 of the 28 × 1024 × 1024
source crop `c1599abb9802cff0d3cb70c79103c2c3484c8d5235df45c9fe3732920202a037`, of the resampled 27 × 999 × 999
array `49c0d7a4475dc5dfc3343e3a6399cf43b85ed26b5f132725e2ac5f3b607758df` (scipy 1.18.1; the resampled bytes may
differ with another scipy version). The T4 run itself recorded no hash, so its input is identified by key, crop
and shape only; `prepare_crop.py` and `measure_t4.sh` now print the hash on every run, and `prepare_crop.py`
stops on any HTTP status other than 200/404 instead of filling the chunk with zeros:

| precision | seconds | vs fp32 output (uint8 levels) |
|---|---|---|
| **fp16** | **41.2** | **max 1, mean 0.047, r 0.99999** |
| fp32 (script default) | 123.6 | — |
| bf16 | 228.6 | max 11, mean 0.321, r 0.999893 |

So fp16 is 3.0× the official default end to end in (b) and 3.9× in model throughput in (a); bf16 is slower than
fp32 on a T4 in both. In (b) fp16 ran first, with a cold file cache, so its time is if anything pessimistic. On
the same known-text control, Hecate run in fp16 gave AUC 0.846 (section 3). Also checked on CPU: the patched
script's default (`fp32`) output is byte-identical to the unpatched script on a real PHerc. 0841 crop, and
`--precision fp16` on CPU is refused with a clear message. `bash hecate_fp16/measure_t4.sh [WORKDIR]` repeats (b)
in a few minutes (downloads the checkpoint and the crop, runs fp16 / bf16 / fp32, prints times and differences).

**Limits.** One crop of one scroll, one batch size, one run per precision (no repeats, so no spread). fp16 has
a smaller range than bf16; no overflow was seen here, but it was not tested on other scrolls or at other batch
sizes. On Ampere or newer GPUs bf16 remains the right choice.

---

## Layout

```
registration/      reg_point.py, fit_landmarks.py, apply_transform_tifxyz.py (+ test_), landmarks.json, transforms/, figures/
colab_pipeline/    jobs/ (env.sh, 01_grow.sh, 02_render_ink.sh), first_letters_colab.ipynb, fl_helpers.py, gpu_worker/
ink/               infer_ink.py (reader-v3 / ink_9um / Hecate, depth shifts, both faces), render_tifxyz.py, vendor/, third_party/
controls/          score_text.py, calibration.json, analyze_controls.py, evidence/, panel figure
negative_results/  negatives.csv, meshes.tar.xz, figures/
hecate_fp16/       patch, patched hecate.py, prepare_crop.py, measure_t4.sh
```

Python: numpy, scipy, opencv-python, scikit-image, tifffile, requests, torch (GPU parts), `zarr<3` and
`numcodecs<0.16` where noted. All scripts print usage with `--help`.

## How this was made

The code, the runs and this text were produced with AI coding agents, directed by the repository owner. All
numbers above come from runs on public scroll data.
