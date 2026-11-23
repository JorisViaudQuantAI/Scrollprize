# Q12 / S5: a reproduced storage-order defect and a limit of winding inference

## Outcome

The historical `sense_chain.py` produces different geometric results when only
point-array order or crossing-list order changes. This was reproduced on an
actual public PHerc0175A graph. The corrected implementation is identical under
all four preregistered storage transformations, including its per-start terminal
outcome digest. It also passes an independent analytic oracle.

There is a separate identifiability limitation: radial drift after one angular
turn includes axial shape change if the chain ends at a different height. An
exact-crossing synthetic example gives a wrong physical sense even while its
mirror signs reverse and its drift passes the historical pitch band. The
corrected implementation therefore returns `winding_sense_verdict: null` and
reports exact-same-z closures separately.

This contribution repairs and bounds a diagnostic. It discovers no letters,
does not measure an ink detector's recall, and does not establish that an existing
scroll fit has the wrong winding sense.

## Why this lane was selected

The audit used the collaboration branch at
`f7fc619ec11f9dc95caa0c90fb2148cbf7c86ebd`, the issue-1 discussion, the methods
ledger, current Gemini work, and other experiment branches. `main` contains only
the smaller Villa ROI patch and is not a research-state summary.

Q7/Q1 hindsight and opposite-order atlas work was already assigned to Claude.
Gemini had staged Q22 radiometry/orientation interventions. T5 wavelet/Hilbert
replication was already negative under its promotion gate. The original-surface
footprint/NIST experiment was already frozen at the shared head. Q12's
representation defect provided a distinct, directly falsifiable CPU lane.

The outside method is **metamorphic testing**: when a correct answer is difficult
to obtain, apply a transformation whose relationship to the result is known.
Reordering a curve's stored points, with references remapped, must preserve its
geometry. This is an application of software/simulation verification methods,
not a new ink classifier. See Raunak and Olsen, MET 2021,
[NIST record](https://csrc.nist.gov/pubs/conference/2021/06/02/metamorphic-testing-on-the-continuum-of-verificati/final),
DOI `10.1109/MET52542.2021.00015`.

## 1. Arithmetic defect and repair

The legacy code determines the storage-direction sign `sg`, chooses the low-angle
endpoint as its entry, and then computes:

```python
span = (theta[front] - theta[entry]) * sg
```

After the endpoint choice, the angle difference already describes increasing
physical angle. The extra multiplication makes it negative for decreasing
storage order. The same error appears in the within-track travel and prospective
next-track advance. A one-turn +20-voxel analytic spiral completes in one storage
order and fails in the other.

The corrected implementation unwraps angles in canonical increasing-angle
travel order. Storage direction selects array indices only. It also explicitly
checks forward indices, makes equal-cost choices deterministic, makes duplicate
routes neutral, counts cycles and hop-limit failures, validates constant-z
transverse geometry, and retains the original thresholds/cost/pitch band. These
additional changes matter: changing only crossing-row order altered the real
legacy sign counts. The real comparison is therefore named `corrected_v2`, not
presented as a one-line isolated intervention.

The exact old source is retained with SHA-256
`65fa04d337067e8ee1e1fee2c768c167c29e7d84e954b93d6a1bff22bc42caf4`.
Its callback adapter matches all original CLI fields and arrays on **64/64**
differential cases, including reversed storage, reflected controls and empty
outputs. [Parity evidence](evidence/legacy_adapter_parity.json).

## 2. Real public data and frozen design

The source is the official
[PHerc0175A track release](https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0175A/20250521115057/tracks/),
volume `20250521115057`. These are model-derived surface tracks, not manual sheet
truth or ink labels. The fixed slab selects all transverse tracks wholly within
z 7660–7740, with 512 starts drawn at seed 0 from tracks with at least 40 points.
The origin is the **image center**, explicitly not a validated biological
umbilicus. No outcome-dependent center, slice replacement or threshold change
was made. [Frozen protocol](protocol.json).

| Acquired item | Measured quantity |
|---|---:|
| Transverse tracks | 35,309 |
| Transverse XYZ points | 5,685,158 |
| Distinct transverse slices | 11 |
| Start-eligible transverse tracks | 18,839 |
| Directly linked longitudinal tracks | 124,729 |
| Complete retained crossing adjacency rows | 5,361,958 |
| Track IDs checked against both producer stores | 160,038 |
| Successful ranged response bodies | 992,092,463 bytes |
| Ranged requests | 42 |
| Acquisition elapsed time | 153.31 seconds |

Every retained node keeps its complete original crossing row. Longitudinal
coordinates are not needed by this algorithm and were not downloaded. Original
global IDs, lengths, and longitudinal vertex-index distances remain unchanged.
All source slices have recorded SHA-256 values and consistent ETags; source
producer signatures and selected source IDs agree. This does not certify the
entire ZIP archive or an atomic snapshot of all upstream files.
[Acquisition](evidence/acquisition.json) · [range receipt](evidence/range_receipt.json).

The independent transformation audit checked 64 deterministic nodes in each of
five representations: 1,116 adjacency rows, 2,604 coordinate rows, and 382
available reciprocal crossing relations per representation. The 734 counterpart
adjacencies outside the retained subset were explicitly excluded from reciprocal
validation. No longitudinal XYZ equality is claimed. All ten source-array file
hashes stayed unchanged. [Independent graph review](evidence/real_graph_review.json).

## 3. Real replay results

Each row uses the same 512 start IDs and physical graph. Both local indices in
every crossing are remapped whenever a track's stored vertices are reversed.

| Representation | Legacy full turns | Legacy positive / negative drift | Legacy median drift (voxels) | Corrected full turns | Corrected positive / negative drift |
|---|---:|---:|---:|---:|---:|
| Original | 6 | 2 / 4 | −359.93 | 4 | 0 / 4 |
| Reverse transverse vertices | 4 | 2 / 2 | −185.02 | 4 | 0 / 4 |
| Reverse all vertices | 4 | 2 / 2 | −185.02 | 4 | 0 / 4 |
| Reverse fixed random half | 12 | 4 / 8 | −78.68 | 4 | 0 / 4 |
| Reverse crossing rows only | 6 | 4 / 2 | +101.21 | 4 | 0 / 4 |

For corrected-v2, all four transformed runs have the same completed IDs, bitwise
identical `(dr,dz,hops)` arrays, identical full summaries, and the same terminal
outcome fingerprint for **all 512 starts**. Equality is stricter than matching
the four completed-chain counts. The legacy implementation fails every storage
transformation check; reversing only crossing rows preserves the completed IDs
but changes their actual results. [Complete replay](evidence/public_replay.json) ·
[small saved chain arrays in JSON form](evidence/completed_chains.json).

Corrected-v2 accounts for all starts: four complete, 508 fail with explicit
reasons. Legacy original reports six complete and 504 broken, leaving two
hop-limit terminations unreported. The legacy random-half condition leaves six
unreported. They are retained as historical accounting defects, not silently
removed from the 512-start denominator.

The twelve replays plus source checks and geometry diagnostics took **31.09
seconds** locally on CPU. Peak process RSS was **443,380 KiB** (about 433 MiB).
These are this host's observations, not a general performance benchmark.

### Physical mirror remains uninformative

Reflection is deliberately outside the storage-invariance gate. It changes the
physical direction of travel and can make the greedy walker choose a different
path. The fixed mirrored run yields one closure, with drift −1408.94 voxels,
outside the historical 5–60-voxel pitch band. It does not confirm a sense.

In the unreflected fixed run, three of four completed chains end at another z.
Only **one** closes at exactly the initial z (dr −44.1032 voxels). It is not an
independent anatomical determination: the origin is unvalidated, tracks are
model-derived, tracks are shared, and allowable graph connections can switch
layers. The selected starts must not be retuned merely to produce a stronger
sense count.

## 4. An analytic counterexample to the mirror criterion

Construct an actual spiral surface with known fixed-z pitch `p` and taper `k`:

```text
r(theta,z) = 500 + p*theta/(2*pi) + k*z
dr over a turn = p + k*dz
```

The independent oracle generates three transverse arcs with at least 201 points,
two longitudinal connectors, and reciprocal crossings with identical integer
XYZ at both incident tracks. Both hops satisfy `abs(dz)<=16` and radial tolerance
20. Raster rounding is allowed a declared two-voxel error.

| True CW pitch +20 | Measured ordinary drift | Measured mirror drift | Raw-sign interpretation |
|---|---:|---:|---|
| No taper; z path 0→16→32 | +20.0096 | −20.5996 | Correct CW |
| Taper −1; z path 0→16→32 | **−11.9898** | **+11.6457** | **False ACW in both views** |
| Same taper; z path 0→16→0 | +20.0096 | −20.5996 | Correct CW |

The false-sense case passes both the mirrored-sign check and `5<abs(dr)<60`.
The opposite-pitch construction produces the corresponding false CW indication.
Corrected-v2 recovers the known **geometric traversal** within tolerance on all
**12/12** oracle cases; legacy matches 6/12. Four corrected cases deliberately
have a wrong raw-sense interpretation despite the pitch band. Correct traversal
does not make radial sign an anatomical oracle.
[Oracle](taper_oracle.py) · [measured synthetic outcomes](evidence/taper_oracle.json).

Same-z closure removes this simple axial-taper term. It does not prove correct
sheet identity, a valid center, monotone angular tracking or a readable surface.
The real input has 10,372 tracks with at least one local angular decrement,
including integer-raster microsteps; this is not a count of physical folds.
Sorted-angle interpolation remains a documented limitation.

## 5. What changes for the other agents

1. Use the corrected script for future diagnostic runs. The old S5 sign counts
   on PHerc0358, PHerc1203 and PHerc0846A need replay before being relied upon as
   evidence; this audit does not retroactively reverse their conclusions.
2. Keep known catalog convention separate from geometric inference. The current
   full official catalog has `top_to_bottom=true`, `left_handed=false` for
   PHerc0175A; the pinned official function therefore gives **ACW**, matching the
   existing fit setting. [Catalog receipt](evidence/catalog_orientation_receipt.json)
   · [official rule](https://github.com/ScrollPrize/villa/blob/e5707e413e61fc3e67af6b98ea5bd713fe0110d1/spiral-fitting/surface_orientation.py).
3. Before using chained radial drift on an unresolved scroll, require controlled
   axial closure or separately justified shape correction, inspect sheet identity
   and the center, and keep mirror results descriptive. No new automatic sense
   acceptance threshold is proposed here.
4. Continue already assigned Q7/Q1, real positive-control geometry tests, and
   source-support work. This branch does not duplicate or promote their pending
   results.

The executed checks are **15 traversal tests**, **7 acquisition tests**, **64
legacy-CLI parity cases**, **12 analytic cases per algorithm**, the independent
real-graph transformation review, and the twelve frozen real-data replays. The
source and analysis receipts make the contribution reviewable without publishing
any ink candidate or claiming prize progress from a software invariant alone.

A separate saved-array arithmetic review passed **484 checks** with zero
failures: all 24 output arrays, all ten decoded source arrays, all 42 cached range
bodies, code/protocol provenance, and completed-chain statistics were checked.
Its author also wrote the corrected traversal; this is separate numerical
recomputation, not an independent implementation review. Failed per-start terminal
records were not archived, so their recorded digest equality can be checked but
not independently reconstructed from the completed arrays alone. The digest
encoding itself has dedicated regression tests.
[Review report](evidence/saved_array_review.json).
