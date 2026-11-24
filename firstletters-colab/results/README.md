# Results of the Claude lane, released for reuse (24 November 2026)

The owner asked for all results to be published so that every agent can reuse them. This folder holds the working
reports, code, job scripts, tables, images, maps and surfaces behind `collab/claude-013` … `claude-028`.

The `REPORT.md` files are the lanes' working logs, unedited. They were written as private notes: they carry UTC
times, job names, failures and dead ends. Absolute paths in them (`/home/user/vesuvius/...`) are those of the
working machine; the table below says where the same material sits here.

## Contents

| folder | what | start with |
|---|---|---|
| `pherc0211_calibration/` | Our own spiral fit of PHerc. 0211 (awarded scroll), read blind, then compared with the winner's mesh. The recipe that shows the awarded letters on our own surface. The measurement of the hit rate per search unit. | `REPORT.md`, section "RECIPE (calibrated on PHerc0211)" and section "Task P" |
| `pherc0211_calibration/code/`, `jobs/` | Everything that ran: fit chunks, gates, flatboi wrapper (`flat_set.sh`), renderer, Studio and reader jobs (`wl_env.sh`, `mk_read.sh`), rankers, `normside.py` (reading order from the normal side), `gridq.py` (shear, stretch, roughness), `windist.py`, `place.py` | `code/wl_env.sh` |
| `pherc0211_calibration/meshes/` | `fitted_w010_w129_step30000.tgz`: all 120 windings of the fit as it comes. `flatboi_w010_w069.tgz`: the 60 inner windings after flatboi, as rendered. `tifxyz`, voxel coordinates of volume `20250821151803` | — |
| `pherc0211_calibration/maps_export211/` | Benchmark maps of the band: Studio stride 96 for 60 windings in the reading order and 9 in the other order, stride 48 for 40 windows, validity and parallel masks, `windows.json`, `README.json` with the recipe and explicit slice lists | `README.json` |
| `pherc0211_calibration/look/` | Review images: our windings beside the winner's map, the regrown patches, the blind review sheets of Task P (`P1/`), the follow-up of the third hit (`FU_*`) | `T1_studio_desc_vs_winner.jpg` |
| `pherc0211_calibration/tables/` | Gate summaries, quality tables before and after flatboi, distances of every winding to the winner's rows | — |
| `wrong_sense_control_0175A/` | PHerc. 0175A refitted in the wrong winding sense: the on-sheet gate does not tell the senses apart; the fitter's losses do, slightly | `REPORT.md` |
| `spiral_fits/PHerc0175A/` | Recipe, code, job scripts, gates, our umbilicus and the 120 fitted windings of the 30,000-step fit (band z 6900–8400) | `REPORT.md` |
| `spiral_fits/PHerc1203/` | Same for PHerc. 1203 (band z 9000–10400), including how its tracks were made, since none are published | `REPORT.md` |
| `spiral_fits/PHerc0846A/` | The fit that was lost at step 16,243 with its session: report with the gate at step 13,051, and job scripts. No meshes | `REPORT.md` |
| (to come) `spiral_fits/PHerc0826/` | The fit at z 11800–13300 is still running; its recipe is in `pherc0211_calibration/REPORT.md` and `jobs/mk_chunk826*.sh` | — |
| `hindsight_0211/` | Why the atlas screen could not find PHerc. 0211: the atlas holds one winding in 20 | `REPORT.md` |

Elsewhere in the repository: `pherc0211_fit_windings/` (four windings with per-vertex distances to the winner's
mesh), `industrial_funnel/` (review of RFC-001, S0 priors per z-band for all scrolls), `methods_ledger/` (every
method tried, with numbers, and the review scripts), `onsheet_score/`, `studio_t4/`, `hecate_fp16/`.

## How to reuse the recipe on another scroll

1. Fit a band (`spiral_fits/*/jobs/*_c01.sh` and `run_chunk.sh`: villa `fit_spiral.py` at `0fb38c45`, one
   30,000-step schedule in chunks of about 35 minutes; 5.6–8 L4-hours).
2. Flatten every winding with flatboi (`flat_set.sh`), and check the quality table (`gridq.py`): shear within
   about ±5°, stretch 0.95–1.05.
3. Render 41 layers, then Studio stride 96 unblended as a ranker; reading order from `normside.py`.
4. Open every map in both mirror states; Studio stride 48 on the top sixth by the line finder plus a random draw.
5. Regrow a local surface at each place (`jobs/wl_940_gp30.sh` is the template) and read it at stride 48.
6. Run one flattened PHerc. 0211 winding through the same jobs: it must give about 8.6 in the reading order.

## What is not here, and why

- **Ink-prediction maps and review notes on unread, eligible scrolls** (PHerc. 0175A, 1203, 0125, 0846B and the
  others). The First Letters rules forbid making a find public before the organisers announce it. A map can hold
  something we failed to recognise, so the maps stay private. Their aggregate results are in `methods_ledger/`
  and in the collab notes, and will be updated when the current re-read closes.
- **The exact places of the false leads that were worked up and closed** on unread scrolls, for the same reason.
- **Fit checkpoints** (650 MB each) and **track stores** (several GB): too large for git. The reports give the
  commands and parameters that regenerate them.
- Account identifiers, tokens and anything personal: none of the files here contains any.

## One caution on PHerc. 0211

Task P flagged three windows as showing letter shapes. Two are on the winner's sheet. The third is not: it lies a
few sheets further in, and a regrown surface there shows a short run of connected strokes in the reading order
only (`look/FU_fw030_regrow.jpg`). It is weak and no letter is legible. Treat it as an unverified observation on
an already awarded scroll, not as a reading.
