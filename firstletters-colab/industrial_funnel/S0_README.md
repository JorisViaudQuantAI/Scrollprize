# S0 — priors per z-band, all scrolls, no surface needed

File: `S0_priors_by_zband.csv`, 194 rows: 19 eligible scrolls and 7 read scrolls as controls, 8 height levels
each (z as a fraction of the scroll's height: 0.08, 0.20, … 0.92). It is the first stage of the amended funnel
in `REVIEW-001-claude.md`. It contains anatomy statistics only, nothing about ink or about any reading.

## How it was made

- Source: the listed ~9 µm volumes of the open-data bucket, read at full resolution.
- At each height level, up to 4 slabs at two radii (0.4 and 0.75 of the scroll radius) and two angles.
- Per slab, from the CT alone:

| column | meaning |
|---|---|
| `edge_rise_um_q75` | width of the air-to-papyrus edge, 75th percentile, in µm; smaller is sharper |
| `merged_frac` | share of the radial profile where neighbouring sheets are not separated by air |
| `coherence` | local orientation coherence of the sheets, 0 to 1 |
| `air_frac` | share of air |
| `clean_slabs` / `slabs` | slabs inside the range of slabs where ink has been read: rise ≤ 20.0 µm, merged ≤ 0.164, coherence ≥ 0.52 |

- The thresholds are the 10th–90th percentile range of 50 slabs taken where ink was read on five scrolls
  (PHerc. 0211, 0306B, 0343, 0841, 1447).

## What it is good for, and what not

| scroll | status | clean slabs |
|---|---|---|
| PHerc. 0826 | eligible | 17 / 27 |
| PHerc. 0813 | eligible | 16 / 28 |
| PHerc. 0175A | eligible | 15 / 28 |
| PHerc. 0358 | eligible | 11 / 27 |
| PHerc. 1545 | eligible | 10 / 29 |
| PHerc. 1218 | eligible | 9 / 29 |
| PHerc. 0483B | eligible | 8 / 28 |
| PHerc. 0483A | eligible | 7 / 29 |
| PHerc. 0125 | eligible | 7 / 31 |
| PHerc. 1203 | eligible | 6 / 27 |
| PHerc. 0800 | eligible | 6 / 30 |
| PHerc. 0191 | eligible | 5 / 28 |
| PHerc. 0490A | eligible | 4 / 22 |
| PHerc. 0257 | eligible | 4 / 27 |
| PHerc. 0846B | eligible | 4 / 30 |
| PHerc. 0175B | eligible | 4 / 32 |
| PHerc. 0846A | eligible | 3 / 27 |
| PHerc. 0268 | eligible | 0 / 26 |
| PHerc. 0490B | eligible | 0 / 24 |
| PHerc. 0306B | read, awarded | 11 / 28 |
| PHerc. 0343 | read, awarded | 8 / 28 |
| PHerc. 0211 | read, awarded | 7 / 28 |

- **Within a scroll** it orders the bands: use it to choose where to seed windows and which band to fit. Where
  letters were found on the awarded scrolls, the papyrus was sharper and less merged than the scroll's average.
- **Between scrolls** it has no demonstrated value. The three awarded scrolls have 25–39 % clean slabs and
  would rank in the middle of this list. A low count does not say that a scroll cannot be read.
- The sample is sparse: at most 4 slabs per level. A level with 0 of 4 can still hold clean sheets.
- PHerc. 0268 and 0490B have no clean slab at all under this criterion; they belong in the "geometry first"
  queue, not at the head of a read queue.
- To be added by whoever holds the data: opened-piece evidence of lines per scroll, scan campaign, existing
  surfaces with their quality table, and earlier reads marked valid or void.
