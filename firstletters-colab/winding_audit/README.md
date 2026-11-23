# Q12: storage-invariant winding diagnostics

This lane tests the arithmetic in `methods_ledger/code/sense_chain.py`. It is
separate from the claimed ink, radiometry, rendering, and atlas-screening lanes.

Two issues motivate it. The original walker chooses the low-angle endpoint,
then multiplies its increasing angular displacement by the original storage
direction a second time. Reversing a track's array can therefore change the
answer without moving any point. Independently, a chain that finishes at a
different height measures radial drift from both winding and axial shape.
For `r(theta,z) = R + b theta + k z`, a turn gives
`delta r = 2 pi b + k delta z`. A mirror-sign check does not remove that term.

The real-data experiment is frozen in [protocol.json](protocol.json). It uses
the public PHerc0175A track store, a fixed slice band, and the image center as an
explicitly unvalidated origin. It tests storage invariance, not anatomy. The
current official catalog independently derives `ACW` for this volume; a code
defect is not evidence that the existing PHerc0175A fit is mirrored.

## Executed result

**The original diagnostic is representation-dependent on real public data. The
corrected version passes all four frozen storage-invariance checks.** With the
same physical graph and 512 seeded starts:

| Representation | Legacy full turns | Legacy positive / negative drift | Corrected full turns | Corrected positive / negative drift |
|---|---:|---:|---:|---:|
| Original | 6 | 2 / 4 | 4 | 0 / 4 |
| Reverse transverse point storage | 4 | 2 / 2 | 4 | 0 / 4 |
| Reverse all point storage | 4 | 2 / 2 | 4 | 0 / 4 |
| Reverse a fixed random half of tracks | 12 | 4 / 8 | 4 | 0 / 4 |
| Reverse crossing-row order only | 6 | 4 / 2 | 4 | 0 / 4 |

Corrected equality includes every summary field, the completed-start IDs, exact
saved float64 result arrays, and the digest of every start's terminal outcome.
This is correctness of representation handling. More or fewer completed chains
does not measure anatomical quality. The physical mirror remains inconclusive.

See [RESULTS.md](RESULTS.md) for the analysis and limits,
[public_replay.json](evidence/public_replay.json) for measured results, and
[research.md](research.md) for primary sources and the cross-disciplinary method.

## Reproduction

Python 3.12 and NumPy are sufficient. The executed replay used Python 3.12.14 and
NumPy 2.3.5. Raw source arrays and run outputs belong outside the git checkout.

```bash
python -m pip install numpy==2.3.5

python firstletters-colab/methods_ledger/code/test_sense_chain.py -v
python firstletters-colab/winding_audit/test_acquire.py -v

python firstletters-colab/winding_audit/check_legacy_adapter.py \
  --oracle firstletters-colab/winding_audit/taper_oracle.py \
  --output /tmp/q12-legacy-parity.json

python firstletters-colab/winding_audit/taper_oracle.py \
  --algorithm legacy=firstletters-colab/winding_audit/legacy_snapshot.py \
  --algorithm corrected_v2=firstletters-colab/methods_ledger/code/sense_chain.py \
  --output /tmp/q12-taper-oracle.json

python firstletters-colab/winding_audit/acquire.py \
  --output-dir /tmp/q12-public-source

python firstletters-colab/winding_audit/review_graph.py \
  --source /tmp/q12-public-source \
  --runner firstletters-colab/winding_audit/run.py \
  --output /tmp/q12-graph-review.json

python firstletters-colab/winding_audit/run.py \
  --source-dir /tmp/q12-public-source \
  --output-dir /tmp/q12-public-replay

python firstletters-colab/winding_audit/review_saved_arrays.py \
  --repo . --source /tmp/q12-public-source --run /tmp/q12-public-replay \
  --output /tmp/q12-saved-array-review.json
```

Use fresh paths for every run. The source acquisition reads about **992 MB** of
successful range-response bodies, not the entire 8.48 GB coordinate file or
2.62 GB crossing archive. It validates response ranges, lengths, ETags, NumPy
layouts and source-ID alignment. It verifies the selected byte ranges, not the
full archive CRC or an unpublished atomic snapshot. Mutable upstream objects
may change: compare the acquired hashes with [the recorded receipt](evidence/acquisition.json).

The archived diagnostic is in `legacy_snapshot.py`; do not use it for new
scientific conclusions. `legacy_adapter.py` preserves that exact numerical loop
for comparison, with 64 archived-CLI parity cases. The corrected production
diagnostic stays at its existing `methods_ledger/code/sense_chain.py` path and
keeps the positional CLI. Its callback API supports sparse acquired graphs.

## Freeze and review trail

- [Protocol freeze](https://github.com/JorisViaudQuantAI/Scrollprize/commit/edaa645742d27bc6b5ea670a216aa6882e85a8b0): before analysis of public coordinate data.
- [Acquisition implementation](https://github.com/JorisViaudQuantAI/Scrollprize/commit/2656adb1f47b2c9e2bf4f6fce201c7c6916ee49c): version used for the real download. Later guards reject impossible range totals and fractional row indices; they do not change valid-source selection.
- [Corrected code and paired-runner freeze](https://github.com/JorisViaudQuantAI/Scrollprize/commit/4923329a46f987b3083de6f75a740afc908d3287): before the twelve real graph replays.
- [Independent real-graph transformation review](evidence/real_graph_review.json): no traversal or tuning.
- [Independent analytic oracle](evidence/taper_oracle.json) and [baseline parity](evidence/legacy_adapter_parity.json).
- [Saved-array arithmetic review](evidence/saved_array_review.json): 484 checks,
  zero failures. The traversal author performed this separate recomputation;
  the implementation and graph-transform reviews were done by another agent.

No raw volumes, unread ink maps, or candidate readings are part of this lane.
Public source data retain their original terms; the new code follows this
repository's MIT license.
