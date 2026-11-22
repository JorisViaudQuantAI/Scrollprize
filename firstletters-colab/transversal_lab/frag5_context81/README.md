# Frag5: fixed-center context enlargement

**My inference / purpose.** [ChatGPT 016](../../../collab/chatgpt-016.md) found
weak small-neighborhood correspondence and weak scale-persistent variation at
one of four public control sites. This follow-up asks whether a larger
anatomical context helps. It keeps the old centers and affine unchanged.
Record 20 and record 25 are deliberately chosen as a previously correlated and
uncorrelated contrast: this is **adaptive development evidence**.

## Completed result

**From a source / measured.** Both old-overlap gates passed with exactly zero
error. At record 25, zero-shift footprint Pearson increased from 0.0267 on
33³ support to 0.3762 on 81³ support, while Spearman increased only from
0.0062 to 0.0836. The predefined outer shell gave 0.3897/0.0890. Four fixed
smoothing passes on 65³ support gave Pearson 0.8687 but Spearman 0.1556.
The original center remained unchanged and unresolved. Record 20 retained
appreciable correspondence across the declared supports. No transform was
fitted or applied, and no ink accuracy was measured.

- [Full result](evidence/result.json) and [interpretation](../../../collab/chatgpt-017.md).
- [All nested contexts and shifts](evidence/nested_contexts_all_shifts.csv),
  [fixed smoothing comparisons](evidence/fixed65_smoothing_zero_shift.csv),
  [outer shell](evidence/outer_shell_zero_shift.csv) and [structure records](evidence/fixed65_structure.csv).
- [Context figure](evidence/context_dependence.svg) and [Pearson/rank figure](evidence/smoothing_and_rank.svg).
- [Overlap gates](evidence/overlap_gates.json), [source ledger](evidence/acquisition.json)
  and [artifact manifest](evidence/artifact_manifest.json).
- [Independent result review](evidence/independent_result_review.json)
  and [all 30 fixed raw-source checks](evidence/fixed_raw_sample_checks.json).

**From a source / measured.** The run used 41.4822 process CPU seconds and
151.0277 elapsed seconds, acquiring 175,385,570 new raw bytes plus 801
metadata bytes. Cumulative raw bytes are 232,449,651 and cumulative unique
chunks are 57. All 51 HEADs and 45 new bodies passed. Independent verification
took 10.2543 CPU seconds and reproduced all 896 correlations to within
1.9984×10⁻¹⁵. It also checked raw interpolation/integration at 30 fixed
positions, not every output voxel. See the review for exact tolerances and
coverage. No new source requests were made by the checker.

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
These are admission reservations. Actual compressed sizes were unknown until
HEAD in this run; the measured transfer totals are reported above.

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

Independent review of the completed local run:

```bash
python review_context81.py \
  --protocol protocol.json --runner run_context81.py \
  --stage1-protocol ../frag5_native_pair/frag5_native_pair_stage1_protocol.json \
  --prior-context-result ../frag5_local_structure/evidence/result.json \
  --stage1-cache /external/frag5-stage1-run-v1 \
  --native-grid-cache /external/frag5-native-grid-run-v1 \
  --run-dir /external/frag5-context81-run-v1 \
  --review-dir /external/frag5-context81-review-v1
```

This review needs the recorded new source bodies, both saved context81 NPZ
arrays, the six retained old source bodies and the two old anatomy arrays,
along with their canonical ledgers. It reconstructs every metric independently
and manually checks point/125-midpoint integration at the fixed 30 positions.
It does not redownload source data or repeat the complete integration.

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
