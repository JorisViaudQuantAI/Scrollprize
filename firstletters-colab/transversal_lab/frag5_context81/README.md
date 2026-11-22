# Frag5: fixed-center context enlargement

**My inference / purpose.** [ChatGPT 016](../../../collab/chatgpt-016.md) found
weak small-neighborhood correspondence and weak scale-persistent variation at
one of four public control sites. This follow-up asks whether a larger
anatomical context helps. It keeps the old centers and affine unchanged.
Record 20 and record 25 are deliberately chosen as a previously correlated and
uncorrelated contrast: this is **adaptive development evidence**.

## Frozen question and support

**From the protocol.** An 81³ scanner-grid cube at nominal 7.91 µm has 640.71 µm
voxel support and 632.8 µm between extreme sample centers. Fine point sampling
and the original 125-midpoint nominal voxel-footprint approximation are both
retained. Native coarse support is 83³ to retain all 27 original ±1-grid-cell
shifts. This is a scanner-grid context; no oriented detector tensor,
photographic reference, ink face or single physical sheet is certified.

**From the protocol.** The nested 17³, 33³, 65³ and 81³ correlations use the same
center. All 432 rows are reported: two sites × two fine arms × four context
sizes × 27 shifts. Each reports Pearson and Spearman, with zero shift primary.
There is no fitted translation. A separate fixed 65³ comparison reports
zero/one/four binomial passes on identical support, with variance retention.
The 81³ outer shell excluding the central 33³ cube is reported separately:
high surrounding correlation cannot validate the original central patch.

**From the protocol.** Before interpreting new metrics, both new central 33³
fine representations must reproduce the old arrays within 1e-8 uint16 counts,
and the entire old 35³ native neighborhood must reproduce exactly. The runner
preserves both overlap results and stops interpretation if either fails.
The original failed legacy surface-render test remains failed.

## Resource admission

**From metadata-only planning.** The selected pair needs 45 fine and 4 coarse
conservative box-support chunks. Six are already cached; 43 are new. All old
14 chunks plus the 43 new ones total 57 unique objects, below the 64-chunk ceiling.
Reserving at most 4,259,840 compressed bytes per new object gives 183,173,120
new bytes. Adding 57,064,081 previous raw bytes gives 240,237,201, below 256 MiB.
These are admission reservations; actual compressed sizes are unknown until
HEAD and actual transfer is reported after execution.

**From the protocol.** Before any new source body, the runner snapshots HEAD
for all 49 target objects and both Zarr metadata objects. Cached objects must
retain their recorded ETag and length. Oversized objects or an excess total
stop admission. Every subsequent body is checked with If-Match and exact
length, through the unchanged verified acquisition code. Both Zarr metadata
hashes must match before raw chunk GETs. It uses at most four I/O workers,
four decoder threads after I/O completes, and 360 process CPU seconds.
No retries or replacement regions are automatic.

The shared [stage1 code](../frag5_native_pair/frag5_native_pair_stage1.py)
and [local finite-filter code](../frag5_local_structure/analyze_local_structure.py)
are imported only after their pinned hashes pass. The 49-chunk support is
recomputed from the original geometry before execution.

## Reproduction

This is another bounded resumer. It needs the canonical original acquisition
receipts, the six selected old source objects at their recorded cache paths,
the two old anatomy NPZ arrays, and the published ChatGPT 016 result. All are
individually hash-pinned in [protocol.json](protocol.json). Recreating these
inputs from public sources is a separate staging step; a fresh timed receipt
is not substituted for a canonical historical one.

```bash
python test_context81.py --out generated_checks.json

python run_context81.py \
  --protocol protocol.json \
  --stage1-dir ../frag5_native_pair \
  --local-structure-dir ../frag5_local_structure \
  --stage1-cache /external/frag5-stage1-run-v1 \
  --native-grid-cache /external/frag5-native-grid-run-v1 \
  --prior-context-result ../frag5_local_structure/evidence/result.json \
  --check-only
```

For one execution after the separate public freeze, replace `--check-only`
with `--out /external/frag5-context81-run-v1 --freeze-reference FULL_COMMIT_SHA`.
Choose a fresh output directory. Dependencies are Python, NumPy, SciPy and
numcodecs, inherited from the original verified sampler.

The generated checks cover inverse-affine point sampling and all 125 midpoint
samples against an analytic linear field, nested-support invariance, native
halo bounds, shell size and overlap-gate failure injection. They use no CT.

## Limits

**My inference.** Enlarging support can recover correlation by including a
nearby interface even when the central patch remains uninformative. This
can support a context-aware anatomical check; it cannot establish central
ink sensitivity, a photographic label transfer or a calibrated registration
error. Four smoothing passes do not reproduce the native scanner's PSF.
The pair is selected after prior results and does not estimate prevalence
across the fragment or unread scrolls. No labels, photographs, detector or
language model are accessed by this experiment.

## Sources

- [Unchanged original geometry and source metadata](../frag5_native_pair/frag5_native_pair_stage1_protocol.json).
- [Native-grid acquisition and arrays](../frag5_native_grid/README.md).
- [Local-structure result](../frag5_local_structure/evidence/result.json).
- [Fine official source](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes/20231121133215/meta.json)
  and [coarse official source](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes/20231130111236/meta.json).
