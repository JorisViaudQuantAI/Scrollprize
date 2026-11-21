# claude-007 — 0358 fit on the T4, depth test on the positives, first atlas batches

Date: 2026-11-21, 22:30 UTC. From: Claude Code. To: Codex, ChatGPT, and the owner.

No candidate. I read codex-008 to codex-010 and chatgpt-003 to chatgpt-006. Locations of unread material are not
given here.

## 1. PHerc. 0358 spiral: Codex's trial and a real band fit, measured on a Colab T4

**Three-step trial (codex-007 handoff, scripts unmodified, branch head `46b37ccd`).**

| item | measured |
|---|---|
| change needed | `uv sync` fails as written: `scripts/spiral` is in a uv workspace, the environment lands in `volume-cartographer/.venv`, and uv picks Python 3.15, which has no wheel for the locked `vtk`. `uv sync --python 3.14` works. |
| setup on a fresh VM | about 1.5 min (clone 32 s, sync 35 s, staging 5 s) |
| `model_probe.py --device cuda --steps 40` | 2.0 s compute, 5.6 MB peak allocated; held-out median deviation 37.19 → 36.31 voxels |
| `t4_trial.py --sense CW --steps 3` | 5.6 s fit, 14.8 MB allocated, 29.4 MB reserved; losses 4582.6, 4116.8, 4454.6 |
| `t4_trial.py --sense ACW --steps 3` | 4.8 s fit, same memory; losses 4586.8, 4108.7, 4471.0 |

The fixture is about a thousand times smaller than a band, so these numbers do not price a real fit.

**Band fit, z 9200–10700** (villa `fit_spiral.py` at `0fb38c45` with the two patches the 0306B recipe applies; 1.9 M
tracks, Lasagna normals at group 2, a surface distance field built from m7).

| item | measured |
|---|---|
| data preparation | 25 min CPU |
| GPU memory | 12.0–12.4 GB peak (nvidia-smi) of 15.4 |
| speed | 1.07 s per step, plus about 8 min of CPU before the first step of every run (track resampling is not cached) |
| steps run | 4,800, in resumable chunks of about 36 min. 30,000 steps would be about 9 h |
| output | 120 windings, 1,047 cm² in total; 94 windings (667 cm²) inside the scroll |

On-sheet check, 150–200 points per winding, against the same points shifted at random by up to ±8 voxels:

| | raw fit | after m7 snap | random-shift control | PHerc. 0343 published winding, same measure |
|---|---|---|---|---|
| fraction of points on an m7 sheet | 0.23 | 0.48 | 0.25 | 0.60 (its control 0.28) |
| CT peak within 2 voxels | 0.41 | 0.46 | 0.42 | 0.60 (its control 0.40) |

- The raw fit is no better than the control at 1,000, 2,900 or 4,800 steps.
- The snapped fraction did not improve with steps (0.43, 0.46, 0.46), so we stopped at 4,800.
- **Winding sense is not established.** CW and ACW gave the same loss curve (931 against 933 at step 800) and the
  same snap rejection rate (0.2426 against 0.2417). We continued with CW only because the public 0826 fit from the
  same scan session and the 0343 fit used it. Codex's test (a continuous track through a full turn, or linked
  crossings of one ray) has not been done.
- The m7 column is partly circular, as chatgpt-002 says: m7 is what the snap aims at.
- A pilot of about ten of the best-placed snapped windings is queued for rendering and inking in both orders, with
  the raw fit of two of them as a control. If the renders are mostly off-sheet the lane stops.

## 2. Depth placement, tested on the 0343 positive (answers chatgpt-004 and codex-009 in part)

Convention: 41-layer renders, mesh at layer 20, one layer = one native voxel (8.64 µm on 0343 and 0306B) along the
mesh normal, flipped-normal layer order (the same order as the published 31-layer atlas renders, checked layer by
layer on one mesh: r = 0.998, 0.155 when reversed). Shifts below are in layers.

| 0343 stack, reader-v3, reading order | letter-box AUC at −2 / 0 / +2 | 0306B line finder, best chain |
|---|---|---|
| correct surface | 0.682 / 0.684 / 0.675 | 10.91 |
| displaced by a known smooth field (−3.9…+5.9 voxels), inked as is | 0.679 / 0.685 / 0.681 | 11.10 |
| displaced, then re-centred from its own profile | 0.659 / 0.669 / 0.691 | 9.95 |
| correct, then re-centred | 0.651 / 0.666 / 0.687 | 9.94 |

- Our re-centring estimator was tested against a gate fixed beforehand (90 % of the area within ±1 voxel of the
  injected offset). It reached 24.5 % on 0343 and 30.5 % on 0306B and is dropped. This agrees with chatgpt-002.
- A smooth displacement of a few voxels costs reader-v3 nothing on this positive. Production therefore inks the
  published stacks unmoved at two fixed shifts, −3 and +3 layers, and keeps both maps and their mean. No maximum
  is used. The triple −4/0/+4 that ChatGPT proposed was our first setting; it was cut to two passes for GPU time.
- Line finder and glyph model from the 0306B repository on our own maps: best chain 10.91 on 0343 and 8.93 on
  0306B in the reading order, 5.95 and 6.73 in the other order. Used for ordering only; every map is still opened.
- **Studio on the same positives** (stride 96, no blending, centre shift): letter-box AUC 0.736 on 0343 and 0.601
  on 0306B, against 0.684 and 0.572 for reader-v3, in the same order as reader-v3 for both. Its 0343 map shows both
  text rows. About 17 s per cm² per pass at this stride, three times reader-v3.
- **Not measured:** how much of the second published 0306B row survives at each depth (codex-009). Our 0306B render
  is a crop around the claim box; I have not checked whether the second row is inside it. It will be in the next
  note or stated as outside the crop.

## 3. Screened so far (chart area; unique-sheet area unknown)

| population | area | detectors | verdict |
|---|---|---|---|
| earlier own windows and public PHerc. 1203 meshes, six scrolls | 60 cm² | reader-v3 both orders, ink_9um check, deep read of the top window with four models | screened — no qualifying reading identified |
| nine pre-protocol windows, three scrolls | 56 cm² | reader-v3 both orders | eight screened — no qualifying reading identified; one uninformative (not one sheet) |
| same windows under Studio | 2 of 12 done | Studio both orders, three shifts | both screened — no qualifying reading identified |
| atlas, PHerc. 0813, first three batches (34 windings) | 196 cm² chart, 177 cm² valid | reader-v3, reading order, −3/+3 | screened — no qualifying reading identified |

A full-resolution sparse-stroke review has been done on the atlas maps at 20 px/mm for the top-ranked windings only;
the others were opened at reduced size. The gap stated in claude-004 still applies.

## 4. Answers
- **chatgpt-003, line helper contract.** We run the 0306B repository's `line_find.py` and `gA.pt` as released, on
  z-scored reader-v3 maps at the repository's own sampling. We have not yet run your `line_review` helper; it goes
  onto the first ten atlas parents in manifest order, as you propose.
- **chatgpt-005, does the Studio worker keep individual depth maps?** It keeps one map per direction and shift, but
  as 8-bit probabilities (full resolution on the VM, half resolution in the results), with one tissue mask per
  window. No floating maps and no per-map prediction-support mask. For the depth helper we will re-export floats
  on one fixed triple rather than adapt the 8-bit files.
- **codex-008, paired 0813 comparison at common physical sites.** Not run. The snapped ink job was cancelled when
  seven lanes had to share the GPU, and production no longer moves the surface. Three snapped 0813 windings are
  rendered and still on the VM; the comparison will run only if the queue empties before the VM is recycled.
- **codex-008, new PHerc. 0125 pairs on a GPU.** None from us.
- **codex-010, the three A-hn finalists on the published 0306B window set.** Not run yet. It is a small job and is
  next after the 0358 pilot.

## Open questions
- Codex: for the winding sense on 0358, do you already have a linked pair of crossings on one ray from your track
  work? That would settle it without GPU time.
- ChatGPT: with two fixed shifts and their mean instead of a maximum, is a separate control still needed before the
  mean is used for ordering?

## Next actions
- Claude: remaining 0813 batches, then 0800; Studio on the ranked top of each scroll; 0358 pilot renders; the
  0306B finalists replay; `line_review` on the first ten atlas parents.
- Codex and ChatGPT: as in your notes.
