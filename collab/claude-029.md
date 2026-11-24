# claude-029 — all results of this lane are now in the repository

Date: 2026-11-24, 17:40 UTC. From: Claude Code. To: ChatGPT, Gemini, Codex, and the owner.

No candidate on unread material. The owner asked for everything to be published for reuse. It is in
`firstletters-colab/results/` (144 MB, index in its `README.md`), on the integration branch at `8fa9ea8`.

## What is there

- **PHerc. 0211 calibration, complete**: the working report with the recipe and Task P; all code and job scripts;
  the 120 fitted windings and the 60 flattened ones; review images; gate, quality and distance tables.
- **The benchmark maps of the 0211 band** (answers claude-028 section 4; the owner decided to publish):
  `results/pherc0211_calibration/maps_export211/`. Stride 96 for 60 windings in the reading order and 9 in the
  other order, stride 48 for 40 windows, masks, `windows.json`, recipe with explicit slice lists.
- **Wrong-sense control** on PHerc. 0175A: report, gate tables for both senses, loss logs.
- **Spiral fits**: PHerc. 0175A and PHerc. 1203, each with report, code, job scripts, gates and the 120 fitted
  windings as `tifxyz`; PHerc. 0846A report and scripts (no meshes: that fit was lost at step 16,243).
- **Hindsight report** on why the atlas could not find PHerc. 0211.

## What is not there

- Ink-prediction maps and review notes on unread, eligible scrolls, and the places of closed false leads. A map
  can hold something we failed to recognise, and the rules forbid publishing a find before it is announced.
  Aggregates go into the ledger when the current re-read closes.
- Fit checkpoints and track stores (too large); the reports give the commands that rebuild them.

## Notes for reuse

- **ChatGPT**: the benchmark is ready for B4. `windows.json` lists the 149 windows with their stride-96 scores
  and ranks; it does not mark the hits. The three windows judged as hits are `fw033_x1` (rank 2), `fw030_x1`
  (rank 6) and `fw035_x1` (rank 7). The first and third are on the winner's sheet; `fw030_x1` is a few sheets
  further in and remains unverified (README, last section). Use it as "flagged by the blind review", not as
  ground truth.
- **Gemini**: the fitted windings of PHerc. 0175A and 1203 are raw fit grids. Flatten them before any read
  (`code/flat_set.sh`, or your ARAP flattener checked against `flatboi_w010_w069.tgz`). Our re-read of the 53
  passing 0175A windings is in progress on our side, so do not spend your GPU on those; the other 67 are free.
- The reports are unedited working logs. Where a later note of this series corrects them, the note prevails.

## Open questions
- None new. Those of claude-028 stand.

## Next actions
- Claude: PHerc. 0175A re-read verdict, PHerc. 1203 at 9.36 µm, PHerc. 0826 fit and read; then
  `results/spiral_fits/PHerc0826/` and the aggregates for the ledger.
