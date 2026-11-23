# 06 — Fine-resolution lane (2.4 µm)

Two eligible scrolls, PHerc. 1203 and PHerc. 0846A, have a second scan: 2.403 µm voxels, labelled 77 keV, 0.22 m propagation, covering the full cross-section over a band 3.64 cm tall.
The prize lists the 9.36 µm volumes. Whether a place found at 2.4 µm may be shown at 9 µm for a claim is a question for the organisers that we have not asked.
Shifts in this file are in fine voxels (4 fine voxels ≈ 1 coarse voxel).

---

<a id="h1"></a>
## H1 — Registration of the fine rescans to the 9.36 µm frame

**Question.** The bucket publishes no transform for these two scrolls. Where is the fine band in the coarse frame, and how well?

**What was done.** Silhouette matching of z-slices for a first guess; then normalised cross-correlation of CT texture between fine level-2 slice regions and coarse level-0 slices at hand-picked landmarks; a similarity transform fitted to the landmarks, written in the bucket's own `transforms/` layout. Published in `firstletters-colab/registration/`.

**Controls and nulls.** Leave-one-out error of the landmarks. The same mesh rendered from both volumes (same papyrus, fibres and cracks). On PHerc. 0841, our mapping of the coarse mesh against the organisers' own mapped mesh.

**Numbers.**

| | PHerc. 1203 | PHerc. 0846A |
|---|---|---|
| leave-one-out error | about 1.2 coarse voxels | about 1.1 coarse voxels |
| landmarks | five, 3.4–11.2 mm from the official umbilicus | within about 8 mm of the centre of the fine field of view; no umbilicus exists |
| error at five check points | ≤ 3.2 coarse voxels | not measured |
| fine band in the coarse frame (z) | 7945–11818; used 8000–11765 | 1892–5724 fully covered; used 1950–5665 |
| same mesh, coarse render against fine render pooled to 9.6 µm | r 0.84 on one patch, 0.71 on a whole mesh crop | not measured |

- PHerc. 0841 check: our inverse mapping agrees with the organisers' mapped mesh to 0.7 / 1.6 / 2.9 fine voxels (median, x / y / z), 99th percentile ≤ 7.

**Verdict.** Worked near the centre of the field of view. A first version (pure scale, no rotation) was replaced by the similarity fit; older notes that quote z 7939–11824 and 1865–5751 use the first version.

**How this could be wrong or miss ink.** No landmark lies beyond about 11 mm from the centre. Outer windings may be off by more than the check points show, and H3 says that 4 coarse voxels of surface error cost a third of the recall. A teammate has a separate annular registration; the two were not compared.

**Not done.** A blind check at 10–20 mm radius. The organisers' own registration pipeline.

**Where the code is.** `firstletters-colab/registration/` (already public).

---

<a id="h2"></a>
## H2 — Reading a 9 µm mesh at 2.4 µm: the pipeline and its control

**Question.** A spiral winding or a public mesh lives in the 9.36 µm frame, with a vertex every 20 coarse voxels. Can it be read by the 2.4 µm models after mapping, as well as a surface made at 2.4 µm?

**What was done.**
- Map the mesh into the fine frame with the registration; drop everything outside the band.
- Cut it into blocks of 42 × 42 vertices (about 3200 × 3200 px, 0.56–0.59 cm²); keep a block when ≥ 40 % of its cells are in band and valid.
- Render 109 layers along the flipped normal, one fine voxel per layer and per pixel, layer 54 = mesh. The surface shape is only as fine as the 9 µm mesh; the image is sampled at 2.4 µm.
- Run the canonical 2.4 µm model (layers 23…85) and Hecate 2.4 µm, each in both layer orders: tile 256, stride 128, Hann blending, fp16, batch 4.
- Score at 9.6 µm (maps and labels area-averaged × 4).
- Control: PHerc. 0841 w00, which publishes everything needed: the coarse mesh, the organisers' fine-to-coarse transform, the fine volume, a surface volume made from a refined fine mesh, ink labels on it, and the canonical model's map.
  The control runs coarse mesh → transform → our render → both models, exactly as an unread winding will. The published surface volume through the same inference is the "perfect mesh" reference.

**Controls and nulls.** Reversed layer order. The organisers' refined-mesh surface volume. Canvas alignment between our render and theirs (normalised cross-correlation of mid-layer composites 0.97–0.98).

**Numbers.** Three blocks, 1.04 cm² supervised, 22 % ink.

| source and pass | AUC | average precision | ink recall at 5 % background | r with the published map |
|---|---|---|---|---|
| coarse mesh, canonical, reading order | 0.975 | 0.928 | 0.874 | 0.929 |
| coarse mesh, canonical, other order | 0.506 | 0.240 | 0.055 | 0.064 |
| coarse mesh, Hecate, reading order | 0.930 | 0.837 | 0.747 | 0.807 |
| coarse mesh, Hecate, other order | 0.580 | 0.273 | 0.077 | 0.107 |
| published surface volume, canonical, reading order | 0.971 | 0.915 | 0.835 | 0.959 |
| published surface volume, Hecate, reading order | 0.955 | 0.861 | 0.753 | 0.823 |

- The coarse mesh lies 2.0 fine voxels (median) from the organisers' refined mesh along the normal (p90 5.7, p99 13).
- On control text the share of pixels above 128 is 0.13–0.30 in the reading order and 0.03–0.08 in the other.

**Verdict.** Worked as a path check. A coarse mesh costs the canonical model nothing measurable and Hecate a little.

**How this could be wrong or miss ink.**
- Both models were trained on data of this kind and the labels come from the canonical model's family. This control cannot say whether either model sees ink on PHerc. 1203.
- There is no second control: the only other candidate (PHerc. 0814) has no public labels and an illegible published map.
- The control mesh is a good one, 2 fine voxels from the refined surface. Spiral windings are worse.

**Not done.** One control block (0.11 cm² supervised) was skipped for budget. A conversion check: are lines chosen on the 2.4 µm map also readable at native 9.37 µm with the 9 µm detectors?

**Where the code is.** Not included (planning, rendering and stitching code around the organisers' models); `code/isometric_resample.py` is the one piece with arithmetic of our own.

---

<a id="h3"></a>
## H3 — Surface-error tolerance at 2.4 µm

**Question.** How far can the mapped surface be from the sheet before the 2.4 µm models lose the text?

**What was done.** On the two most-labelled control blocks (0.96 cm² supervised), the model window was moved along the normal by s fine voxels, reading order.

**Controls and nulls.** s = 0.

**Numbers.** AUC / ink recall at 5 % background.

| s, fine voxels (coarse voxels) | −16 (−4) | −8 (−2) | −4 (−1) | 0 | +4 (+1) | +8 (+2) | +16 (+4) |
|---|---|---|---|---|---|---|---|
| canonical | 0.870 / 0.54 | 0.934 / 0.77 | 0.949 / 0.80 | 0.974 / 0.87 | 0.962 / 0.83 | 0.966 / 0.83 | 0.947 / 0.72 |
| Hecate 2.4 µm | 0.864 / 0.51 | 0.891 / 0.61 | 0.904 / 0.67 | 0.931 / 0.75 | 0.896 / 0.68 | 0.909 / 0.65 | 0.878 / 0.51 |

- Letters stay legible by eye at every shift up to ±16 fine voxels in both models; they get thinner and patchier.
- An earlier run with the window shifted by ±8 layers changed the mean response on text from 0.330 to 0.328 and 0.338.

**Verdict.** Measured. One to two coarse voxels of error cost 1–4 AUC points; four cost a third of the recall. Rule adopted: read at s = 0; add ±8 only where the fit is known to be more than 2 coarse voxels off; take no maximum over shifts.

**How this could be wrong or miss ink.** Measured on a training-family control. The raw spiral fit is on the sheet at only a third of its vertices; we read the snapped windings, whose residual error we know only against m7 (S7).

**Not done.** Depth shifts on any unread block ("nothing flagged that would need them").

**Where the code is.** Not included.

---

<a id="h4"></a>
## H4 — Tile stride 128 or 192

**Question.** The true area of the fitted windings is larger than nominal (S8). Can a wider stride keep the read inside the GPU budget?

**What was done.** The control was re-read with stride 192. The bulk run of 16 windings was then started at stride 192, reading order only for both models, with the other order on two windings.

**Controls and nulls.** Stride 128 on the same control.

**Numbers.** Canonical: AUC 0.972 against 0.975, recall 0.854 against 0.874. Hecate: 0.924 against 0.930, 0.732 against 0.747. Throughput about 74 against 34 cm² per L4-hour per pass (2.2×).
On blank-like PHerc. 1203 papyrus the stride-192 maps are blocky: square patches of about 0.46 mm with sharp edges, in both models and orders. At stride 128 the same blobs are round.

**Verdict.** Stride 192 for screening only. Any place that is not plain blobs or squares is to be re-rendered and re-read at stride 128 before it is judged.

**How this could be wrong or miss ink.** The control is dominated by strong text, so its metrics hide what the wider stride does to weak responses. The bulk run reads one order on 14 of 16 windings.

**Not done.** Stride 128 on the bulk windings.

**Where the code is.** Not included.

---

<a id="h5"></a>
## H5 — Results on PHerc. 1203 so far

**Question.** Is there ink-like structure on PHerc. 1203 inside the fine band?

**What was done.** Three surface sets, in order.
1. Two small own patches near the core (2.8 cm² of valid papyrus), canonical model only, both orders. Early work, on a T4.
2. The public `segments/raw` meshes inside the band: the eight meshes with the most area in band (22.3 cm² of mesh lies in band over all 22 public meshes), cut into 39 blocks. Twelve blocks were read early with the canonical model only; later all 39 were read under the corrected registration, 27 of them with both models in both orders and the 12 old ones again with Hecate in both orders.
   Surface quality of each block was judged from the 2.4 µm CT composite **before** any ink map was seen.
3. The 17 windings of our spiral fit that passed the gate (S4), snapped and resampled isometrically. One winding is complete with four passes at stride 128; the other 16 are running at stride 192 (H4).

**Controls and nulls.** The PHerc. 0841 control text shares (0.13–0.30 in the reading order, 0.03–0.08 in the other). The other order on every block of sets 1 and 2.

**Numbers.**

| set | valid area | passes | share of pixels above 128 | what the maps show |
|---|---|---|---|---|
| two own patches | 2.8 cm² | canonical, both orders | 0.02–0.28 forward, 0.01–0.20 reversed | saturated amorphous blobs 0.5–2 mm, as frequent and as strong in the other order (correlation between orders 0.04–0.17); stable over ±8 layers (r 0.84–0.86) |
| 39 public-mesh blocks | 15.75 cm² | both models, both orders | canonical 0.000–0.091, Hecate 0.001–0.092 in the reading order | isolated round blobs 0.3–2 mm, as frequent in the other order, at different places in the two models; no strokes, no rows |
| first fitted winding | 2.80 cm² | both models, both orders, stride 128 | canonical 0.026 / 0.041; Hecate 0.024 / 0.015 (reading / other order) | round blobs 0.5–1.5 mm; more in the reading order in the lower half, more in the other order in the upper half; no strokes |
| 16 fitted windings | 98.3 cm² true area | both models, reading order, stride 192 | not complete | first two: round or square blobs, on one of them more in the other order |

- Of the 39 blocks, 22–24 are on a clean fibre grid and 15–17 are crumpled (mesh crossing layers), depending on how part-crumpled blocks are counted. `collab/claude-017` gives 22 and 17; the private list gives 23 and 16 at the verdict.
- The largest one-order structure is an amorphous cloud 5–8 mm long across the boundary of two blocks, in Hecate's reading order only (ratio between orders 5.1 and 3.2). It is the cloud of F7; it has no stroke shapes.
- Forward-to-reverse ratios of ink pixels scatter on both sides of 1 (0.06–5.1), against 8.5 on control text.

**Verdict.** `screened — no qualifying reading identified` for the 39 blocks and the first winding; crumpled blocks `uninformative`. The bulk of the fitted windings is in progress.

**How this could be wrong or miss ink.**
- Reading order on the fitted windings rests on geometry, not on known text: in our renderer the layer index increases towards the umbilicus at 98 % of vertices and lies on the concave side of the sheet for 67–74 % of samples, as on the control mesh (75 %). The first winding showed text in neither order, so it could not confirm this.
- If that inference is wrong, the bulk run (reading order only) is blind: the other order scores 0.51–0.58 on the control.
- The public meshes reach only the lower half of the band (none above z 10150).
- 17 windings passed a gate at 0.40; 48 windings inside the scroll were not read.
- The 2.4 µm models' sensitivity on this scroll is unmeasured (H2).
- Residual local scale of 0.5–1.7× remains in the resampled windings (S8).

**Not done.** PHerc. 0846A at 2.4 µm (its fit is running). A 9.36 µm reader-v3 pass on the fitted PHerc. 1203 windings (skipped by decision: the fine read supersedes it). A label-free top-hat at 2.4 µm (L1). The question to the organisers.

**Where the code is.** Not included.
