# Frag5 adaptive follow-up: native source fidelity and raw scanner-grid anatomy

This is a **separate adaptive experiment**, designed after the failed surface-rendering source test. It preserves that earlier `neither` result. It asks a narrower question: do 32 fixed native fine-TIFF voxel values equal the corresponding already acquired Zarr values, and, conditional on exact equality, how do fixed fine and coarse scanner-grid neighborhoods compare under the published affine?

The runner is `frag5_native_grid_followup.py`; the full prospective specification is `frag5_native_grid_followup_protocol.json`. It imports and hash-checks the original stage1 code and protocol. New pixel access requires a separate published follow-up freeze. The code is a bounded resumer using an existing, verified cache. It is not a generic fresh replay of either experiment.

## Why this is a new question

**From the prior execution record.** The [original Frag5 result](../frag5_native_pair/evidence/result.json) rejected both declared surface-profile orders at all four fixed geometry locations. Its anatomy stage did not run. Its [independent review](../frag5_native_pair/evidence/independent_result_review.json) verified the cached values and interpolation calculations. This follow-up changes neither that result nor its criterion.

**From a source.** The official [native fine-volume directory](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes/20231121133215/) contains 7,010 files named `0000.tif` through `7009.tif` and [volume metadata](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes/20231121133215/meta.json). The metadata identifies the 4,420 × 1,400 × 7,010 fine scan. The [pinned Volume Cartographer source](https://github.com/educelab/volume-cartographer/tree/7640ba6c8e41f39a7aef63e9780cd13c296b2d15) reads native TIFF slices, whereas our previous numerical comparison used the [released fine Zarr representation](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes_zarr/70keV_3.24um_.zarr/0/.zarray). Equality between these two representations at the chosen points had not been measured.

**From the metadata-only preflight.** The eight required native files are little-endian, one-channel, unsigned 16-bit TIFFs with uncompressed raster data in one strip at byte 368. They contain no orientation, predictor or fill-order tag; the frozen test verifies the exact already inspected header bytes. The prospective comparison needs only two adjacent X samples per row: 16 exact four-byte ranges, **64 native intensity bytes in total**. The directory and header preflight acquired 1,054,299 bytes of metadata and no intensities. The preflight receipts and header hashes are preserved.

**My inference.** A local exact-integer check can help distinguish a representation-conversion problem from a remaining surface-rendering discrepancy. Passing at 32 positions does not authenticate an entire volume, a physical face, a surface-layer phase or a historical rendering command. The old PPM points serve here only as previously selected geometric locations.

## Frozen sequence

1. **Verify the preserved inputs.** Check the five canonical stage1 JSON receipts separately from the six cached fine-chunk objects. Each old chunk is bound to its public URL, compressed byte count and SHA256. Decode those existing chunks using the original guarded decoder. Their data are already acquired fine data, not a fresh held-out sample.
2. **Verify current source identity before mixing data.** Re-HEAD the original complete 14-object raw chunk union and require the same ETag and length as the original prospective snapshot. Check the eight native TIFF identities against the new metadata preflight. Verify both raw Zarr metadata hashes and every native header range before new intensity access.
3. **Compare 32 integers.** At each of the same four PPM points, enumerate exactly the eight Cartesian floor/ceil scanner-XYZ corners. Read only those corresponding native TIFF integers and compare them with the old cached fine-Zarr integers. All **32 of 32 must be exactly equal**. Any discrepancy retains a failed source-fidelity result and stops before new fine/coarse chunk payloads. No location, phase, normal or alignment alternatives are tried.
4. **Report a secondary adaptive diagnostic regardless of that gate.** Interpolate each set of eight native TIFF corner values at its original PPM point with the unchanged stage1 trilinear helper. Report the rounded and unrounded discrepancy against the already observed surface TIFF value at index 32. All four comparisons are reported. They do not affect the integer-equality gate, choose a phase, fit a correction or authorize anatomy independently.
5. **Only after exact integer equality, run the original raw-grid comparison.** Acquire the original remaining two fine and six coarse chunks. Retain the original four rounded coarse centers, 33³ grids, published affine, point-sampled fine arm, 125-midpoint nominal-coarse-footprint arm, all 27 native coarse shifts in `{-1,0,+1}³`, zero-shift primary comparison, and Pearson/Spearman intensity and 31³-interior Laplacian metrics. No shifted registration is selected or fitted. No legacy surface-TIFF intensity enters this anatomy calculation.

**Protocol limitations.** Fine data previously acquired during stage1 are reused after observing that failure. The native TIFF values and raw coarse outcomes are new at this follow-up freeze; this is an adaptive engineering investigation, not an untouched independent evaluation. The four locations share one physical fragment. Correlation does not establish same-sheet identity or ink signal. Nominal coarse-footprint integration approximates a voxel support, not a scanner's full resolution kernel or noise process. Raw uint16 windows differ between acquisitions; see the official [EduceLab information sheet](https://dl.ash2txt.org/other/EduceLab%20Herculaneum%20Scroll%20Data%20(2023)%20Info%20Sheet%20v3.pdf). No photographs, ink labels, models, physical-face claims or automatic label-stage approval are included.

## Resource accounting

**From planning and the preserved receipt.** The original six fine chunks cost 24,571,323 compressed bytes. The remaining fixed eight raw objects account for 32,492,758 bytes in the original HEAD snapshot. A completed follow-up would therefore have **57,064,081 cumulative raw chunk bytes**, with the same 14 unique decoded chunks. The combined raw chunk limit remains 256 MiB and the decoded limit remains 64 chunks.

The separate 2 MiB TIFF/metadata budget conservatively charges all of the following before this run: 11,591 bytes from the original stage1 execution, 1,054,299 from this native listing/header preflight, 4,862 from the earlier PPM/surface-header preflight, and 3,419 from the earlier fine-Zarr metadata lookup. That totals 1,074,171 prior bytes. New planned bodies are 64 native intensity bytes, 1,744 header bytes and 801 Zarr metadata bytes. Each response is charged prospectively before reading, including failed responses.

The cumulative CPU budget is one hour, with a conservative 60-second reservation for the prior 5.605-second CT execution and related small preflights. The follow-up receives at most 3,540 CPU seconds. Network concurrency and decoding use the original four-thread rules. Any integrity, transport, source-bound or resource failure stops execution without retries or replacement.

## Minimal inputs and reproducibility

The resumer needs:

- The unchanged stage1 runner and protocol from `../frag5_native_pair/`.
- The **five original canonical JSON receipts** from `../frag5_native_pair/evidence/`: `result.json`, `identity_result.json`, `acquisition.json`, `acquisition_heads_freeze.json` and `start.json`. Place these at the root of an external cache directory using those names. Their hashes verify the historical record. A newly executed stage1 produces different runtime-timed JSON hashes; it is not expected to match these canonical receipts.
- The **six raw fine-chunk objects** listed in this follow-up protocol, at their recorded relative `objects/…bin` paths under that cache directory. Their individual public URLs, sizes and SHA256 values are pinned separately. No other old TIFF pixel payloads or full prior object directory are required.

The team's existing original run directory already satisfies that arrangement. Other users can assemble an equivalent external cache from the canonical receipts and six exact source objects. Downloading those objects afresh is a separate cache-staging cost; the executed bounded resumer counts the actual already acquired six chunks in its cumulative resource ledger. No full fresh reproduction has been claimed or run.

Hash and geometry validation only, without decoding CT or requesting network data:

```bash
python frag5_native_grid_followup.py \
  --protocol frag5_native_grid_followup_protocol.json \
  --stage1-dir ../frag5_native_pair \
  --prior-run /external/verified-frag5-cache \
  --check-only
```

After publication and the execution handoff, run once into a fresh separate output tree:

```bash
python frag5_native_grid_followup.py \
  --protocol frag5_native_grid_followup_protocol.json \
  --stage1-dir ../frag5_native_pair \
  --prior-run /external/verified-frag5-cache \
  --out /external/frag5-native-grid-run-v1 \
  --freeze-reference FULL_40_CHARACTER_PUBLISHED_COMMIT \
  --run
```

The dependency requirements are inherited from stage1: Python, NumPy, SciPy and numcodecs. Tests additionally require pytest and the `FRAG5_ORIGINAL_DIR` environment variable pointing to the original stage1 directory. The three small generated/geometry checks verify the fixed 32-corner support and complete HEAD union, exact-equality gate behavior, and the secondary trilinear diagnostic on an analytic field. They do not use real CT intensities.

## Open questions

- **ChatGPT:** Are the two raw fine representations exactly equal at the 32 fixed corners? What does the secondary center calculation show about the remaining legacy-render discrepancy?
- **Codex / Claude Code:** If the raw-grid stage completes, do its full zero-shift and offset-stress results justify a separately designed anatomical or photographic-label validation?
- **Owner:** This experiment supplies no letters or prize submission. Any later label experiment, candidate handling or submission remains a separate decision.

## Next actions

- **ChatGPT:** Finish independent review, publish the exact follow-up freeze through the owning agent, execute once and retain every outcome. Keep the original failed surface experiment unchanged.
- **Codex / Claude Code:** Continue existing work; treat this as local source and anatomy evidence without using it to tune a detector or choose a favorable transform.
- **Owner:** Review a later substantive contribution or submission when evidence supports one. No organiser contact is performed by this experiment.
