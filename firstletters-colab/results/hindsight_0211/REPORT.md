# 33 — Hindsight test on PHerc. 0211 (Q7) and the other layer order on the atlas (Q1)

Date: 2026-11-23. Lane `hq` on g2 (L4, shared with `hw`). Private. Code `/home/user/vesuvius/hq/code/`, jobs
`/home/user/vesuvius/hq/jobs/`, results `/home/user/vesuvius/hq/res/<job>/`, review images `/home/user/vesuvius/hq/look/`.

## Verdict, Task A

**Test impossible as posed: the public atlas does not contain the place of the PHerc. 0211 letters.** The awarded
letters lie on a sheet *between* the published atlas windings: the atlas publishes only every 20th winding of each fit
(w020, w040, … w100), and the winner's letter sheet sits at about atlas-fit winding w027–w033 (median ≈ w029). No vertex
of any of the 90 atlas meshes comes closer than 46 voxels to the winner's surface; the nearest windings (w020 inside,
w040 outside) are typically 120–160 and 150–210 voxels away. So our 1,300 cm² atlas screen could not have found the
0211 letters whatever the detector, order or reviewer — **the screen would have MISSED them, for reasons of sampling
geometry, not detection.**

**Nearest equivalent (winner's published mesh pushed through our unchanged atlas path, clean and degraded to atlas-like
placement): FOUND.** Rank 1 of 28 meshes on both rankers in every variant (line finder 9.38 / 8.55 / 7.27 for clean /
smooth-degraded / jump-degraded, against 6.85 for the best of 27 atlas windings of 0211; row score 4.26 / 4.05 / 3.83
against an atlas max of 3.62). The montage at our review scale shows separate letter shapes in rows; a reviewer
following our protocol flags it (yes for clean and smooth, yes-but-weaker for jumps). In the other order: not flagged
(rank 3–4, no letter rows). Studio stride 48 in the rule order shows the awarded row cleanly (line 10.7–10.8).

Plain words: the detector side of our screen would have seen these letters on a surface that is on the sheet, even
when that surface is misplaced as atlas meshes are. The atlas screen is blind to them because it never looked at that
sheet. **Every atlas negative must be re-described as "every 20th winding of each fit window was screened", i.e. about
5 % of the windings in the screened z-windows — not "the scroll region was screened".**

## A1 — where the letters are, and what the atlas covers

- Public source of the place: the organisers' bucket now holds the winner's letter-bearing mesh with renders and three
  ink maps: `PHerc0211/segments/20261028201525-crop1711-lines123-v9/` (tifxyz 86 × 300 at scale 0.05 → render 1720 × 6000
  px = 1.6 × 5.6 cm; bbox x 2480–4300, y 3214–4288, z 7538–9311; volume 20250821151803, 9.362 µm). The organisers'
  reader-v3 and argus-v3 maps on it (`/home/user/vesuvius/hq/win/ds8_*.jpg`) show three text rows. The post image
  (`/home/user/vesuvius/hq/pub/post_letters.png`) is the middle row ("καὶ τῆι διαφ…", 11 letters in 4 cm); by the mesh
  the middle row is at z ≈ 8300–8500, the top row at z ≈ 7700, the bottom row at z ≈ 9050.
- Atlas PHerc. 0211: 90 meshes, 18 z-windows (z3712 … z14512, each ≈ 800 slices), windings w020/w040/w060/w080/w100
  only (539.7 cm²). The z8512 window was never drawn (METHOD.md: offsets drawn without replacement), so z 8718–9120
  is covered by no atlas winding at all; the middle (awarded) row's z is inside window z7920 (7922–8718).
- Distances (`code/geo.py`, `geo_dist.json`; winner vertices vs every atlas vertex of windows overlapping its z):
  minimum 46 voxels (z7312_w020), 51 (z7920_w040), 62 (z7920_w020); medians 140–380 voxels; share of winner vertices
  within 20 voxels of any atlas mesh: 0 for every mesh.
- Radial position (`code/geo3.py`, ring centroid of w020 as centre, crossing at the same angle and z): the winner lies
  between w020 and w040 in all three windows, fraction 0.39–0.48 (p5–p95 0.24–0.65) of the way from w020 to w040, i.e.
  winding ≈ w028–w030 of the atlas fit (p5–p95 w025–w033; the spread is partly real — the winner stitched neighbouring
  windings — and partly the crude centre). Gap to w020 median 119–160 voxels, to w040 152–209 voxels.
- Reading-face rule (306B `reading_face`, ring-centroid umbilicus, `code/face.py`): all 90 atlas 0211 meshes `reverse`
  (share ≥ 0.93), winner mesh `reverse` (0.99) → our `forward` with flip-normals, exactly as for the 250 screened
  windings. On the winner the rule order is the one that reads (our reader-v3 forward and Studio asc show letters; the
  other order does not). This is the first check of the rule on known text of a 9.36 µm-campaign scroll (Q3): it
  passes here.

## A2 — runs (production path unchanged)

- 0211 atlas population: the three windows spanning the letter z-range (z7312, z7920, z9120; 15 windings, 90.8 cm²
  chart / 63.3 valid) + 12 random windings (seed 211, 104.5 cm² chart / 56 valid). HF 31-layer stacks as published,
  `at_atlas.py prep --raw-only`, reader-v3 `forward` (rule) at −3/+3, mean map, `at_atlas.py post` (row score, 306B
  line finder + glyph CNN), montage at 20 px/mm. Tranche windings also in `reverse`. Code identical to production
  (`at_code_v6` = `si_code_v1` for at_atlas/at_lib/infer_ink/score_text, checked by diff). 27 windings, 119.5 cm² valid.
  Two 0211 meshes are mostly empty (z7312_w100 0.58 cm², z9120_w100 0.27 cm² valid).
- Winner mesh, three variants, rendered by our renderer as atlas-like stacks (31 layers, flip-normals, mesh at 15,
  9.362 µm; same layer order as HF, report 14 at_010), then the same prep/ink/post:
  - `clean`: as published.
  - `smooth`: displaced along the normal by a smooth random field (3 mm correlation), median |d| 2.6 voxels, 40 % within
    2 voxels, 31 % > 4, p1–p99 −8.5…+9.0 (matched to report 14's atlas snap offsets: median 1.6–4.0, 28–60 % within 2).
  - `jumps`: smooth + 6 mm cells, 35 % of them shifted ±6–10 voxels (Q6-style sheet jumps); median |d| 3.8, 29 % within
    2, p1–p99 −13.9…+12.7.
  - Studio (packaged CLI, fp16, stride 48 Hann, shift 0, planes asc 7→22 / desc 23→8) in both orders on all three.

### Blind review first (written before opening the winner maps)
Montages `res/hq_022_post_tr/montage_tr_rv3_forward.jpg`, `res/hq_032_post_rnd/montage_rnd_rv3.jpg`
(split views `look/blind_trF_*.jpg`, `look/blind_rnd_*.jpg`), atlas-only rankings:
- line finder top: z11520_w100 6.85, z12112_w100 6.77, z7312_w080 6.39, z7920_w060 6.35 — all below the 8.2 text range;
  row top: z7312_w080 3.62, z4320_w080 2.69, z10320_w080 2.59.
- What I would have flagged: **one place, z9120_w080, x ≈ 98–108 mm, y ≈ 3–6 mm**: a ~1 cm run of four connected,
  curved stroke shapes (Λ-, κ-, o- and n-like) on one row, in the rule order only (reverse shows different blobs there);
  CT at the mesh: fibre grid, a dark vertical crack through the third shape (`look/flag_z9120_w080.jpg`). Neither ranker
  picked it (that mesh: line 5.72, its best chain elsewhere at x 31 mm; row 1.75). Under the working bar it is a
  "look" (connected strokes, order-specific) that would go to the second detector family; on an unread scroll it would
  have been the strongest single place we have seen in the atlas. Nothing else: z7920_w080's right end is
  fibre-parallel bars; z7312_w040, z7920_w040/w060, z7312_w080 left half are clouds; the rest round blobs.
- This place is **not** the awarded one (w080 at z ≈ 9120–9200, ≈ 500 voxels outward of the letter sheet). PHerc. 0211
  is a written scroll, so real ink elsewhere is plausible; not followed up (Studio there was not run; budget).

### Comparison with the true letter location (after the blind review)
Ranks among the 27 atlas windings + the variant (`code/rank.py`, `rank.json`):

| winner variant | order | line max (mesh rank / 28; place rank) | row max (rank; place rank / 186) | hot z>2 (rank) | flagged by our review? |
|---|---|---|---|---|---|
| clean | rule (fwd) | **9.38** (1; 1/32) | **4.26** (1; 1) | 0.065 (2) | **yes** — three rows of separate letter shapes |
| smooth | rule | **8.55** (1; 1/30) | **4.05** (1; 1) | 0.060 (5) | **yes** — top row clear, middle row partly |
| jumps | rule | 7.27 (1; 1/32) | 3.83 (1; 1) | 0.058 (5) | yes, weaker — top row and parts of the others |
| clean | other (rev) | 6.01 (3 / 16; 3/19) | 2.13 (2; 7/98) | 0.057 (7) | no — blob clusters, no rows |
| smooth / jumps | other | 4.61 / 4.36 (4 / 16) | 1.98 / 1.81 (4) | 0.061 (3) | no |
| clean / smooth / jumps | Studio asc, stride 48 | 10.74 / 10.81 / 9.40 | 3.85 / 4.00 / 3.52 | 0.086–0.088 | yes — the awarded row ("καὶ τῆι δια…") legible in clean |
| clean / smooth / jumps | Studio desc, stride 48 | 7.81 / 6.32 / 7.35 | 2.38 / 2.54 / 2.87 | 0.071–0.085 | no — blobs (though the line finder still scores 7.3–7.8) |

Caveats: (1) the line finder's best chain on the winner is oriented across the rows (e.g. clean: x ≈ 31 mm, y 0.7–5.6 mm),
so the rank-1 is a "this mesh is busy with marks" signal more than "this is a row"; the eye, not the chain, sees the rows.
(2) jump-degraded drops below the 8.2 "text range" (7.27) though still rank 1 — the absolute calibration is fragile.
(3) the population of atlas places is small (27 windings); the review was not truly blind for the winner (I knew it
was the positive), only for the atlas windings. (4) The on-sheet (layer-coherence) score does not separate the letter
surface from atlas windings: winner 0.92 (degraded 0.88), atlas windings 0.86–0.95.

Side-by-side: `look/A_0211_ours_vs_published.jpg` (ours rule order clean / smooth / jumps, ours other order, ours Studio
asc, organisers' reader-v3 on the same mesh, the published winner image).

## A3 — our path vs the winning path at that place

| | our atlas path | winning path |
|---|---|---|
| which sheet | atlas windings w020/w040 only near there — the letter sheet (≈ w029) is not in the atlas | the letter sheet itself |
| surface | atlas spiral fit, every 20th winding published, flattened by the fitter, not snapped, not moved (offsets ≈ ±3–9 voxels) | full-height spiral fit, neighbouring windings stitched, refined by hand in VC3D Manual across crushed "points", extended along rows with DinoVol-guided growth, flattened |
| depth | HF 31-layer stack, reader-v3 at −3/+3 and mean | stock `vc_render_tifxyz`, 31 planes, Studio planes 23→8 |
| order | rule order (correct here) | the same face |
| detector | reader-v3 (Studio only as second opinion, mostly stride 96) | Studio, stride 64/48 Hann |
| review | 20 px/mm montages + two rankers | the author's own images |

What decides it here is the first row only. Given the right sheet, our detector and order would have shown letters
even with atlas-like misplacement; reader-v3 is somewhat weaker than Studio on this place (middle row), and an
on-sheet surface matters more than the ±3 shifts.

## A4 — conclusion and re-description

- Sensitivity: on the one won scroll in the atlas, our screen **would have missed** the letters — not because reader-v3,
  the order rule, the rankers or the montage review failed (on the right sheet all four work, even with atlas-quality
  placement), but because **the atlas does not contain that sheet**. It samples one winding in 20 within each window,
  and leaves whole z-windows out (0211: z 8718–9120; elsewhere similar).
- Therefore the 1,295 cm² of atlas negatives in ledger 07 should be re-described as: "screened: windings w020, w040,
  w060, w080, w100 of each fitted z-window (≈ 1 in 20 windings, ~5 % of the sheet area in those windows); no
  statement about the other windings". Area in cm² overstates coverage of a scroll by about ×20 within each window, and
  the windows themselves are a sample of the height. The statement "the scroll's atlas windings show no text" stays true;
  "the region was screened" does not.
- What the test does support: on a surface that is on the letter sheet, our production detector/order/rankers/review
  would have flagged 0211 at first sight (rank 1 on both rankers; letters visible at our review scale), also with
  smooth misplacement of ±3–9 voxels. Jump-like misplacement of 6–10 voxels weakens it but does not erase it.
- Also: the blind pass over 27 atlas windings of 0211 produced one strong-looking order-specific stroke group
  (z9120_w080) that the rankers missed — a reminder that on the atlas our rankers do not find what the eye finds (Q21).

