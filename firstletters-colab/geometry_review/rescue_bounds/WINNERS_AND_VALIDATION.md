# B10 — 26 Nov 2026: winners' methods and validation routing

## Operational change

Use the dated [machine-readable registry](prize_registry_20261126.json), not the old 19-window inventory, for First Letters routing. Official [rules](https://scrollprize.org/prizes) list 17 exact volume IDs and five remaining $50k awards. 0211, 0343, 0306B, 0846A/B, 1447 are not in that list. 0211 and 1447 remain Grand Prize eligible. Do not stop other agents' Grand Prize work. 846A/B share one original manuscript; 306A/B also share a parent. Treat paired halves as a group in splits, not independent replications. Keep unpublished candidate material private. Snapshot must be refreshed before dispatch; do not infer eligibility from catalog icons.

## Method comparison: what to reproduce first

| Public positive | Winning recipe (published, not independently rerun here) | Next experiment |
|---|---|---|
| 0846B, 24 Nov | A good surface and off-the-shelf Reader v3 | Reproduce native winning mesh/render + checkpoint; deliberately perturb anchor/normal/scale separately; fixed physical support. |
| 0846A, 24 Nov | AFV v2 fibre-constrained geometry + manual correction; z-score fusion of tuned Reader v3, tuned v8-in and lightweight HercUeNet | Freeze verified crossing subsets and held-out fibre residuals; compare TPS correction vs unchanged mesh; separately compare calibration-matched single models, product, mean and z-score. |
| 0211, 15 Nov | Studio 3D residual U-Net / 16 planes; donor similarity via self-supervised 3D DINO, high-to-native resolution label transfer, CPT replay and soft loss; full-height spiral fit, stitching, local VC3D and DinoVol growth | Reuse existing Q33/B4/B8 outputs, test geometry correction before threshold tuning; compare donor selection to actual leave-parent-out transfer. |
| 0306B, 2 Nov announcement | Spiral screening → M7 GrowPatch → local snap; TRACE v3B multiplied by A-hn v8-in fine-tune with pseudo-labels from other regions + background penalty | Reuse Codex public replay; compare recall at matched FPR and faint-stroke preservation; target training disqualifies the winner as a cross-scroll holdout. |
| 0343, 8 Oct | Average of two ink_9um seeds, Hecate and dense_native (139 donor), no 343 training according to announcement | Compare fixed physical footprint under anchor/scale perturbations, do not re-run unchanged public replay. |

### 846A fibre-guided correction: concrete algorithm

Read [fiber-sheet](https://github.com/jeff-j-chen/fiber-sheet), [HercUeNet](https://github.com/jeff-j-chen/HercUeNet/), [v8-in fine-tuning fork](https://github.com/jeff-j-chen/nader-fork). The first repository documents an AFV-based tracer and dynamic-programming path refinement. Two crossing fibre families constrain integer layer/ply assignments; trim unsupported paths and reject hops. Project samples to reference UV, compensate ply separation, estimate signed normal offsets via a smoothing thin-plate spline, then invalidate extrapolation outside supported fibre hull. The author reports 38 horizontal and 78 vertical fibres with 97% good crossings in the full set. These crossing checks are not independent anatomical ground truth: a group can jump sheets together or cut across a bend; in-plane hops can also survive crossing consistency. The final sheet involved manual corrections.

Connection to Adaptive 069: the older sparse-anchor TPS improved within-2-voxel coverage over an affine fit but failed some local gates. Verified fibre constraints offer a stronger, spatially distributed anchor set, not a guarantee. Compare no correction, existing M7 snap, robust affine and fibre TPS at the same physical locations; hold out entire fibres or contiguous crossings, never random neighbouring vertices. Monitor sheet jumps, invalid footprint, physical error, distortion, and independent ink evidence separately. Do not choose the best correction by reading the positive control and then report it as independent.

The winning custom detector uses fewer depth slices (8) and the author reports much lower model size than Reader v3. Gradient cosine similarity was used to choose useful donor data. Target-trained 846A models reportedly converge quickly; measure transfer rather than assume this is universal. To test gradient selection prospectively, use only labeled development parents to form target gradients, rank donor batches by cosine gradient similarity, compare uniform and DINO-embedding selection under equal updates/compute; target validation labels must never drive selection. On unseen eligible scrolls, target gradients from true labels are unavailable until legitimately acquired, so start with unsupervised DINO or a separately defined pseudo-label development procedure.

### 846B and model ensemble baseline

[Winning reproduction](https://github.com/brkzak/Pherc0846B_submission); [Reader v3](https://github.com/DomRusso2/reader-v3). Test the unmodified winner first. Geometry improvement is a separate factor from detector improvement. B9 bounds show uncalibrated OR can force background detections even on a donor control; do not equate z-score ensemble of distinct checkpoints with the OR of renders. Compare per-model thresholds, mean, product and z-score with normalization and calibration fixed from development controls; use common masks, report disagreements and per-scroll errors. A product can suppress genuine weak detections as well as false positives.

### October technique intake

[Winners list](https://scrollprize.org/winners) identifies HercUNet/HercuViT (virtual unwrapping), automated fibres, scalar winding-field (integer level surfaces), fibre-guided ScrollFiesta UV flattening and ink-sdk model battery. These are distinct from HercUeNet (846A ink model). Priority: integrate fibre correction and standardized native model recipes first, then evaluate winding-field versus spiral on the same reserved cuts and identical mask/area denominator. No quantitative superiority claimed from prize amounts. Pin upstream commits before running; review licenses and do not publish new candidate positions.

## Validation design and leakage

Use four evidence types: (1) native registered independent optical labels where available (Frag4 needs identity registration/radiometry/order verification and is development-exposed); (2) independently annotated ink and confirmed background on public positive control surfaces, with unknown regions excluded, not marked blank; (3) published predicted maps for reproducibility/geometry only; (4) papyrologist-verified strokes/letters for positive readability, not dense ground truth. Awarded controls are a development/regression set until held-out regions and model-specific provenance are established. 841/814/9B are Studio donors. 846A/B one parent, group 343/343P conservatively pending provenance check. No train/calibration/test overlap in native XYZ, include spatial guard band >= actual receptive field and check all checkpoint ancestry. Leave whole parent manuscripts out for transfer; report per-parent results and denominator, avoid IID pixel intervals. Freeze recipes, masks, thresholds, budgets and failure rules before inference. Record actual CT bytes, render/model time, review minutes, unknown support and failed attempts; no automatic blank inference.

## Coordination requests

- Claude: register the public 846A/B winner meshes and verified fibres; close 0211/846A First Letters queues, retain 0211 Grand Prize option. Compare fibre TPS with raw/snap and holdout crossings. Do not duplicate 0175A re-read already in progress.
- Gemini: pin Reader v3, TRACE, v8-in, Studio, HercUeNet recipes/checkpoint lineage; native float32 same-footprint map exports and measured cost; run B9 pixelwise replay, then distinct-model fusion on public positives only.
- Codex: verify registry exact IDs, parent grouping, source hashes, B9 independent replay and guard-band leakage; audit fibre path/layer/coverage against reserved geometry, not just M7 occupancy.
- Owner: choose budget and objective (First Letters vs Grand Prize). No merge, production dispatch, training budget, unread inference or external submission authorized here.

## Primary sources

- [846A/B official announcement, 24 Nov](https://scrollprize.substack.com/p/both-halves-of-pherc-846-give-up)
- [0211 official announcement, 15 Nov](https://scrollprize.substack.com/p/first-letters-emerge-from-pherc-211); [Studio code](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt); [VC3D fork](https://github.com/priyankaranke/vc3d-manual)
- [306B official announcement, 2 Nov](https://scrollprize.substack.com/p/writing-seen-in-pherc-306b-for-the); [reproduction](https://github.com/SirProdigle/pherc0306b-first-letters); [data](https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters); [A-hn model](https://huggingface.co/Prodigle/pherc0306b-ink-ahn)
- [343 official announcement, 8 Oct](https://scrollprize.substack.com/p/50k-first-letters-prize-awarded-for)
- [Official pipeline/open problems](https://scrollprize.org/from_ash_to_text), [prizes](https://scrollprize.org/prizes), [winners](https://scrollprize.org/winners).

This is a source review and concrete experimental protocol, not an independent reproduction of the new prize-winning pipelines. External binaries and weights have not been fetched or pinned yet.
