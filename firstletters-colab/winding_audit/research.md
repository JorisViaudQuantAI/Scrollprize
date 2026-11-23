# Q12 research note: geometric invariants before winding inference

This note distinguishes a numerical correctness test from an anatomical claim. It introduces no new ink detector, no candidate location, and no unpublished reading.

## 1. Selected cross-disciplinary method: metamorphic testing

**Primary source.** Raunak and Olsen, *Metamorphic Testing on the Continuum of Verification and Validation of Simulation Models*, MET 2021, pp. 47–52, DOI [10.1109/MET52542.2021.00015](https://doi.org/10.1109/MET52542.2021.00015); [NIST publication record](https://csrc.nist.gov/pubs/conference/2021/06/02/metamorphic-testing-on-the-continuum-of-verificati/final). The paper describes relations between executions that can expose faults even when a trustworthy answer for an individual simulation is unavailable. This is relevant because the anatomical winding sense is unresolved for several eligible volumes.

**Our application, not a claim from that paper.** A polyline's point-storage order is arbitrary. Reverse any subset of stored tracks, remap every crossing index by `new_index = track_length - 1 - old_index`, and keep the same physical points and crossing graph. A winding-sense estimator should emit the same geometric answer. Compare completion count, selected physical paths, radial change and verdict under original, all-reversed and seeded randomly reversed storage. These are paired executions of the same data, not independent observations.

Other valid representation checks are a translation of both geometry and the umbilicus, and an integer 90-degree rotation of both. A *physical reflection* is different: it changes the orientation of the plane. For a known Archimedean spiral, it must exchange CW and ACW. Exact per-chain sign reversal on arbitrary real graphs is not automatic if the greedy algorithm starts from a different physical end after reflection; compare explicitly paired paths, or use a symmetric generated fixture with known answer. Do not turn a failed unpaired reflection comparison into a proof of anatomical direction.

**Why this differs from existing work.** Ledger S5 already reports a mirror diagnostic and restricts its conclusions to chains with 5 < |delta r| < 60 voxels. It does not test reversal of the *stored point order while preserving the same geometry*. That invariant isolates an implementation error without relying on an uncertain physical truth or on a post hoc pitch filter.

**Limit.** Passing these relations proves only the tested implementation properties. It does not prove that a track is on the correct sheet, that a crossing is anatomical, that a chosen umbilicus is correct, or that selected chains estimate the population sense without selection bias.

## 2. Geometric confounding: one angular turn is not sufficient

**Primary geometry context.** Henderson, *Virtually Unrolling the Herculaneum Papyri by Diffeomorphic Spiral Fitting*, WACV 2026, pp. 6401–6411, [CVF record](https://openaccess.thecvf.com/content/WACV2026/html/Henderson_Virtually_Unrolling_the_Herculaneum_Papyri_by_Diffeomorphic_Spiral_Fitting_WACV_2026_paper.html), [arXiv:2512.04927](https://arxiv.org/abs/2512.04927). The method fits a deformed globally continuous sheet to local evidence; this motivates evaluating connectivity and geometry separately from an ink model's response.

**Our derivation.** If a local radial representation is `r(theta, z)`, a chain whose height changes obeys

`dr = (partial r / partial theta) dtheta + (partial r / partial z) dz`.

After accumulating one angular turn, the observed end-to-start radial change therefore mixes winding progression with axial deformation whenever the end height differs from the start. A z-dependent umbilicus subtracts translation of the center, but does not remove axial changes in the sheet shape. Report the final height difference, maximum height excursion, all accepted hop gaps and angular progress. Do not label every chain that reaches 2*pi as an anatomically closed-turn observation. A strict same-height endpoint is a useful additional control, subject to its own selection/availability limitation. Even that does not prove same-sheet continuity across crossings.

**Practical implementation implication.** Keep the primary fix limited to the deterministic sign/order defect and fail on malformed indices, nonfinite geometry and non-forward progress. Treat pitch ranges and anatomical confidence as separate reporting or validation questions. Any new jump penalty or center adjustment would change the estimator and require a separate experiment.

## 3. Current official convention and the PHerc. 175A decision

The official Villa code at revision `e5707e413e61fc3e67af6b98ea5bd713fe0110d1`, [`spiral-fitting/surface_orientation.py`](https://github.com/ScrollPrize/villa/blob/e5707e413e61fc3e67af6b98ea5bd713fe0110d1/spiral-fitting/surface_orientation.py), defines `ACW` when `z_direction_is_top_to_bottom != left_handed_coordinates`, otherwise `CW`. Both inputs must actually be known booleans for this derivation to be used.

The official full [metadata.json](https://vesuvius-challenge-open-data.s3.amazonaws.com/metadata.json), fetched in this session with ETag `f6f6ca1b1c8fca9e9480a2844f537c38`, says PHerc. 0175A volume `20250521115057` has `z_direction_is_top_to_bottom=true` and `left_handed_coordinates=false`: the catalogue convention is therefore **ACW**. It supports the existing fit setting. It is not an independent anatomical measurement. The compact receipt records exact response hashes, headers, derivation and eight selected scrolls; the complete compressed response is retained locally for audit.

Useful positive-control convention references are PHerc. 0343 **CW**, PHerc. 0306B **ACW**, and PHerc. 0211 **ACW**, all extracted from the same catalogue snapshot. PHerc. 0125, 1203 and 0846A remain underivable there because their z direction is null.

[Villa issue #1909](https://github.com/ScrollPrize/villa/issues/1909) independently documents why a fitted spiral or a single static crop should not be treated as a winding oracle. The issue's tracks-only fits could not distinguish both senses on two tested scrolls; a comment points to a different CT-snapped phase-constrained fit that did discriminate on PHerc. 0211. These are different experiments. Its maintainer response says the unresolved directions require text orientation or a manual cross-section reading. Consequently a fixed arithmetic defect does **not** establish that PHerc. 0175A was fitted in the wrong sense.

## 4. Exact public data source for the bounded experiment

Official source directory:

`https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0175A/20250521115057/tracks/`

The directory lists `PHerc0175A_20250521115057_surface_m7_L0_th0.2.dbm.vctracks/`, a 2.4 GiB `.dbm.crossings.npz`, an 8.2 GiB `.dbm`, and a small `.extract.json`. The packed store lists contiguous `coordinates.i32`, `offsets.i64`, `z_bounds.i32`, `family_codes.i8`, `source_ids.u64`, `arclengths.f64`, `tortuosities.f64`, `header.bin`, and `metadata.json`. No umbilicus is listed at this volume root. If a center must be estimated from these tracks for a representation-invariance experiment, document that it is a source-derived experimental assumption; do not call it a validated physical umbilicus.

The crossing archive can be inspected via ZIP central-directory range reads before choosing byte ranges. Villa's [`grow_track_graph.py`](https://github.com/ScrollPrize/villa/blob/e5707e413e61fc3e67af6b98ea5bd713fe0110d1/spiral-fitting/grow_track_graph.py) provides a `StoredNpz` implementation for locally mapping uncompressed NPY members and uses independent four-cycle witnesses and bridge pruning for a separate sheet-growth task. That code is a useful format reference; its graph acceptance rules should not be silently mixed into this winding-sign correction.

