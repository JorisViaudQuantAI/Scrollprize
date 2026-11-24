# B7 — exact matched-step wrong-sense geometry replay

Source: Claude-029 release, commit 1770a7b9f223a016167fda440f1867848373119d. This is a geometry-only follow-up to the existing PHerc0175A wrong-sense control, not a new fit or ink inference. Eight source files are pinned by Git blob SHA1 in the script and receipt.

## Matched optimizer logs

Join ACW and CW by exact logged step, not by chunk or approximate stopping time. All 30 common steps 7800–13600 are present. Separate 12 steps before track_dt is active (through step 10000) and 18 after.

| Metric | Correct ACW mean | Wrong CW mean | CW higher (steps) |
|---|---:|---:|---:|
| Total loss | 253.38 | 277.11 | 30/30 |
| track_radius | 173.970 | 193.563 | 30/30 |
| dense_normals | 2.44 | 2.78 | 30/30 |
| dense_spacing_density | 12.323 | 12.260 | 2/30 |
| track_dt, after activation only | 84.117 | 89.606 | 18/18 |

track_radius relative excess is 11.26% pooled, 11.98% before track_dt and 10.77% after. The smallest matched difference is +13.5. This strengthens the descriptive observation that the loss difference is not merely a consequence of activation of track_dt or comparing mismatched endpoints. These are serially correlated steps in one seed pair, not 30 independent experiments. No calibrated threshold or p-value.

## Wrong-sense gate by radial block

Reuse actual per-winding fitted and snapped TSVs from both CW checkpoints. Inside means fit_ct_found >= 0.9, as in the published summary. Also report the 118 windings common to all four inside sets.

| CW checkpoint | Fitted excess M7 occupancy | Snapped excess M7 occupancy | Snapped positive deltas |
|---|---:|---:|---:|
| 7781 | 0.00017 | 0.38154 | 120/120 |
| 14030 | 0.06195 | 0.39908 | 120/120 |

On the 118 common windings the raw excess changes from 0.00055 to 0.06195; snap excess from 0.38475 to 0.40242. At the later fitted checkpoint the four prespecified 30-winding radial block mean excesses are 0.01679, 0.02417, 0.07467, 0.12917 (two excluded by the inside gate in the first block). 63/118 individual wrong-sense windings show excess >=0.05. Already at the early checkpoint every snapped wrong-sense winding shows excess >=0.05. The snap gate can therefore produce excellent local scores despite the wrong global sense. It cannot establish sheet identity or continuity. No right-sense matched per-winding gate table was published at the intermediate steps; do not treat a late ACW table as a matched control.

## Reproduction and next experiment

From repository root:
```sh
node firstletters-colab/geometry_review/wrong_sense_replay/replay.mjs firstletters-colab/results/wrong_sense_control_0175A/gates
```

Six parser/census/replay checks executed on connector-returned source files. No GitHub CI, local clone, or new GPU jobs. Node CLI fresh replay requested.

Before any automated sense selection: use a second catalog-labelled manuscript, equal frozen step schedule for CW and ACW, >=3 paired seeds per manuscript where resources permit, identical staged inputs verified by hash, log all losses and score track_radius at fixed common steps; use within-pair direction and held-out manuscript-level accuracy. Keep global orientation separate from local sheet following and evaluate seam continuity with independent labelled anchors or winding-crossing evidence. This is a proposed validation, not a promoted gate or proven early-stop savings.
