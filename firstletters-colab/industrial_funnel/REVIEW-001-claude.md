# Review of RFC-001 (industrial search funnel) — Claude lane

2026-11-24. Reviewer: Claude Code. Requested format: blocking issue / evidence / proposed patch / acceptance
test / owner. Private figures come from our own runs; no unread location appears here.

**Position.** The RFC is right about what must never happen (a "negative" that hides an unsupported chart, a
permanent rejection, an uncalibrated probability). Its risk register is the best list we have. But as written it
optimises the cheap part of the search and leaves the expensive part and the main unknown outside the funnel. I
propose keeping its contracts and survey code, dropping the coordinator service, and putting two stages in front
of it. Section 3 is the amended funnel.

## 1. Measured costs the design has to fit

One L4, fp16. Area in cm² of chart.

| step | cost | source |
|---|---|---|
| dense spiral fit of one band of 1,500 slices (120 windings, about 1,000 cm²) | 5.6–8 L4-hours | 0211, 0175A, 1203 fits |
| flatboi | 2–4 s per winding, CPU | 0211 calibration |
| render, 41 layers | about 40 s per winding, CPU | same |
| Studio stride 48, one order | 52 s per cm² | same |
| Studio stride 96 unblended, one order | about 13 s per cm² | a quarter of the above |
| GrowPatch window of 11–14 cm² | 11 min growth + 5 min render, CPU; normal grids 23–32 min once per scroll | own-window tranche |
| share of a fitted winding that is on a sheet | 0.32–0.38 raw | three fits |
| share of a GrowPatch window that is on one sheet | about half | 39 windows |

- A scroll is about ten such bands. Fit plus one stride-96 order for all 19 scrolls is about 1,800 L4-hours.
  The owner's remaining budget is about 140 L4-hours. **The funnel has to choose about 1 band in 15.**
- Per band, with a deep read (stride 48, both orders) of the top 12 % of the area: fit about 6 hours, coarse read
  3.6, deep read 3.5. Halving the deep read, which is what adaptive 071 tests, saves about 13 % of a band. Not
  fitting a band saves all of it.

## 2. Blocking issues

**B1. The funnel starts after the expensive step.**
- Evidence: table above. Also `claude-020`: the atlas screen of 1,295 cm² missed PHerc. 0211 because the letter
  sheet was not among its surfaces, not because of ranking.
- Patch: add a stage that needs no surface (S0 below) and a stage that samples sheets cheaply before any fit
  (S1). Ranking of detector maps becomes S3.
- Acceptance test: on PHerc. 0211, S0 and S1 run blind must put a unit on the letter rows within the per-scroll
  budget, or the design is rejected.
- Owner: Claude.

**B2. The design has no sizing parameter.**
- The number of units a scroll needs depends on p, the chance that one unit on a scroll with detectable ink shows
  letters under the recipe. Nobody has measured p. Text covers most of a written sheet, so p is set by where ink
  is detectable and where the surface holds its sheet, not by layout.
- Evidence that p may be small: in our 0211 band, 60 flattened windings (285 cm²) hold one legible stretch on
  one winding.
- Patch: measure p on PHerc. 0211 before sizing anything: the calibrated recipe over all 60 flattened windings,
  and a blind line of GrowPatch windows through the letter height. Then units per scroll =
  ln(0.05) / ln(1 − p) for a 95 % chance of one hit.
- Acceptance test: p reported with its count (hits / units) for both unit types; if p < 0.03 a per-scroll
  survey is not affordable and the funnel must rank scrolls and zones by priors only.
- Owner: Claude (running today).

**B3. The search unit "physical sheet hypothesis" cannot be computed today.**
- Evidence: adaptive 056–069 fail their own identity gates; our fitted windings step between sheets
  (`claude-025`); our on-sheet gate cannot even tell the winding sense (`claude-024`).
- Patch: account in 3-D footprint cells. A read covers the cells of a coarse grid (64 voxels) that its rendered
  surface crosses, with the normal direction. Two reads are duplicates when they share cells within 3 voxels
  along the normal. Denominator per scroll: cells crossed by the m7 sheet prediction. Sheet identity stays a
  research lane, not a prerequisite.
- Acceptance test: the winner's 0211 mesh and our four published windings must come out as overlapping where
  they are within 3 voxels and distinct elsewhere.
- Owner: Codex or ChatGPT.

**B4. The ranking benchmark uses maps of winning-quality surfaces.**
- Evidence: 071 ranks tiles of the released 0306B product and a 0343 display image. On fit-quality surfaces
  the picture differs: the line finder ranks the two letter windings 1st and 4th of 120 maps; the row score
  ranks the letter map 94th; order asymmetry is not specific (a crack zone on PHerc. 1203 gave 0.22 against
  0.03 in two families, with no stroke).
- Patch: benchmark on `pherc0211_fit_windings/` and on stride-96 maps of all 120 maps of that band, which we
  will export with masks. Drop the row score. Treat asymmetry as a ranker, never as evidence.
- Acceptance test: at the budget fraction proposed for deployment, both letter windings survive.
- Owner: ChatGPT, with our maps.

**B5. The promotion gates cannot be measured.**
- "At least 95 % survival, no worse than 5 points per stratum" needs dozens of independent positives. There are
  three scrolls.
- Patch: say so, and build test units from what exists: the three winner meshes, each displaced, sheared and
  re-fitted (our 0211 fit is the first re-fit). Report survival per scroll, with n.
- Owner: all.

**B6. The coordinator service solves failures we do not have.**
- What actually cost us work in three days: account usage limits (3 times), local container restarts (3), a
  session death that lost a fit (1), no GPU available (1), a job chain blocked by its own dependency (1). None
  is a lease or a race.
- Patch: keep fingerprints and receipts as flat files written by each job; keep `survey.py`; queue jobs in
  chains that run without the agent. No SQLite service, no leases, until two producers share one queue.
- Owner: ChatGPT.

**B7. The 70 / 20 / 10 split has no basis before a first positive on a scroll.**
- Until a scroll shows anything, all work on it is exploration. Allocation is between scrolls and zones.
- Patch: allocate by S0 priors with a floor per scroll, and a stop rule per scroll (section 3).

**B8. Admissibility rules that the calibration made mandatory are only in the risk register.**
- Patch: make them preconditions of any read: per-surface flattening with its quality table; reading order from
  the normal side, checked on the positive; review in both mirror states; a positive in the same run and
  configuration.

## 3. Amended funnel

| stage | what | unit | cost | output |
|---|---|---|---|---|
| S0 | priors, no surface: per z-band sharpness, merged-layer share, void share from down-sampled CT; opened-piece evidence of lines; scan campaign; existing surfaces with their quality table; earlier reads marked valid or void | scroll × z-band | CPU minutes per scroll | ranked bands, all 19 scrolls |
| S1 | cheap sheets: a line of GrowPatch windows seeded along one radius on successive clean sheets, in the two best bands; flatten; quality table; 3-D footprint | window | CPU only | about 10 windows per scroll |
| S2 | coarse read: Studio stride 96, reading order by the normal rule, the other order on a 15 % sample, on-sheet tiles only, positive in the batch | window | about 1.5 GPU-minutes | maps and masks |
| S3 | rank within scroll: line finder, hot share, order asymmetry; every map opened in both mirror states | — | minutes | top 15 % + noted places + 5 % random |
| S4 | deep read: Studio stride 48 both orders, second family, regrow locally, CT sections, neighbours | place | 10–20 GPU-minutes | private verdict |
| S5 | dense fit of a band, flatboi, S2–S4 on every winding | band | 9–11 L4-hours | only where S2–S4 or S0 justify it |

- Budget for S0–S4 on all 19 scrolls: about 25 session-hours, most of it CPU. One S5 band costs as much as
  S1–S4 on seven scrolls.
- Stop rule per scroll, to be fixed once p is known: after n admissible units with nothing, the scroll drops
  to the floor allocation. It is never closed. It reopens on a new detector, scan or surface method.
- Savings that are measured and missing from the RFC: one order by rule (2×); stride 96 first (4×); no
  reader-v3 in the first pass (it missed 0211); inference on on-sheet tiles only (1.5–2×); public fits only
  after flatboi and the quality table.
- What this funnel does not solve: surfaces in crushed zones. S1 and S5 both lose the sheet there.

## 4. What I accept as written

The verdict vocabulary and "no permanent rejection"; four separate coverage denominators; the adapter
metadata (pixel centre, slice order as an explicit list, masks with every map); the stratified survey with no
substitution; private handling of anything on unread material.

## 5. Offered by this lane

- p on PHerc. 0211, both unit types.
- Stride-96 and stride-48 maps with masks for the whole 0211 band, as the common benchmark.
- The S0 table for the 19 scrolls from our volume statistics.
