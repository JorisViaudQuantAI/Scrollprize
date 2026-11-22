# Frag5: a bounded source-contract and native-pair anatomy experiment

This is a prospective, unlabeled experiment on the public PHerc. 1667 Cornice 1 Fragment 3 donor (Frag5). It first checks a declared relationship between the released surface TIFFs, the PPM geometry and the raw fine scan. Only a unique match permits the fixed comparison of fine and native coarse anatomy. It does not read ink labels or photographs, fit a detector, change a registration, select a favorable offset, or authorize a later label experiment.

The executable is `frag5_native_pair_stage1.py`; every location, source, acquisition cap, algorithm and source-contract alternative is in `frag5_native_pair_stage1_protocol.json`. The protocol pins the executable's SHA256. It must be published before running on CT pixels.

The [source-contract audit](../frag5_contract/README.md) and [header/geometry bundle](../frag5_headers/README.md) record the completed input preflight. This experiment is a separate, publicly frozen CT procedure.

## Source facts and the interpretation boundary

**From a source.** The official [Frag5 data page](https://scrollprize.org/data_browser/PHerc1667Cr1Fr3) links the raw scan package. The fine [volume metadata](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes/20231121133215/meta.json), coarse [volume metadata](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes/20231130111236/meta.json), and [fine-to-coarse affine](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/transforms/20231121133215-to-20231130111236.json) identify the nominal 3.24 µm and 7.91 µm scans used here. The raw level-zero sources are [fine Zarr metadata](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes_zarr/70keV_3.24um_.zarr/0/.zarray) and [coarse Zarr metadata](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes_zarr/70keV_7.91um_.zarr/0/.zarray). Their metadata content hashes are fixed in the protocol. The source chunk bytes are hash-recorded on acquisition; they are not presented as an immutable release with prior published chunk hashes.

**From a source.** The [surface bundle](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/) contains the PPM and layer TIFFs. The [pinned Volume Cartographer 2.26.0 implementation](https://github.com/educelab/volume-cartographer/tree/7640ba6c8e41f39a7aef63e9780cd13c296b2d15) supplies the candidate sampling convention: stored XYZ point plus its stored normal times the layer displacement, followed by trilinear interpolation and integer rounding. The actual Frag5 render command has not been authenticated. Thus `p + n * (k - 32)` is a declared hypothesis to test against the 65 published layers, not a recovered command-line fact.

**From a source.** The [EduceLab data information sheet, sections 4.2 and 5](https://dl.ash2txt.org/other/EduceLab%20Herculaneum%20Scroll%20Data%20(2023)%20Info%20Sheet%20v3.pdf) describes nominal energies and separate full-volume reconstruction windows. **My inference:** raw uint16 equality across these two acquisitions cannot be interpreted as calibrated attenuation equality. The anatomy comparison therefore reports structure correlations and clearly labeled descriptive stored-value statistics, without fitting an intensity transformation.

## Frozen experiment

**Protocol decisions, not measured outcomes.** Four centers are chosen from the existing 85-record geometry preflight by proximity to normalized surface quarters. The declared distance rule selects records 20, 25, 65 and 70. There is no CT, mask, ink class, correlation, acquisition-cost or model-output selection, and no replacement after pixel access. Geometry and the literal stored normals are included in the protocol.

1. **HEAD snapshot and metadata checks.** Record object lengths and ETags for the entire prospective raw chunk union, all 65 TIFFs and the two Zarr metadata objects. Save this acquisition identity snapshot before CT or TIFF pixel bodies. Verify both pinned Zarr metadata hashes. Check all 65 TIFF headers and IFDs before any pixel body. A changed IFD pointer is rejected before it can be followed into a pixel strip.
2. **Source-contract test.** Acquire only the six fine raw chunks needed by the four normal profiles, then exactly four uint16 TIFF values per layer. Compute all 65 raw fine samples at each selected center with the stored normal unchanged. Record the unrounded and rounded error vectors for both original and reversed profile order. A common order must match every one of the 260 comparisons within one rounded uint16 count, and each measured and predicted profile must be nonzero and nonconstant. `primary_matched` and `reverse_only` permit anatomy. `both_ambiguous` and `neither` stop and preserve the outcome. No alternative phase, normal normalization or coordinate correction is tried.
3. **Native-pair anatomy.** For each center, round the mapped fine point to an integer coarse voxel center and use its fixed 33³ coarse grid. Compare (a) trilinear raw fine values at inverse-affine coarse centers and (b) an average of **125 fixed midpoint samples of the trilinear fine field** over each inverse-affine coarse cube against (c) actual native coarse values. This approximates a nominal coarse voxel footprint; it does not reproduce the coarse scanner's resolution kernel, reconstruction noise or full acquisition process.
4. **Registration stress.** Retain all 27 native coarse XYZ shifts in `{-1,0,+1}³`, with zero shift the original-transform primary comparison. Fine samples stay fixed. Report Pearson and Spearman intensity correlations, and correlations of the six-neighbor Laplacian on the valid 31³ interior. No padding, registration fit, maximum-correlation selection, pixel-independent confidence interval or p-value is used.

**From the geometry-only planning calculation.** The complete prospective union has eight fine and six coarse chunks: 14 decoded 128³ uint16 chunks, or 58,720,256 bytes. The source-contract stage uses six of the fine chunks. The four anatomy grids contain 143,748 cells, with 17,968,500 fine sample locations for the fixed 125-point integration. These are planned workload counts, not CT measurements.

**Protocol resource limits.** The shared limit is 64 decoded chunks; source chunk response bodies are capped at 256 MiB and TIFF/metadata bodies at 2 MiB. Aggregate CPU time is capped at one hour. Network concurrency is four; the network pool finishes before four-thread Blosc decoding. Raw object declared decoded sizes are checked before decompression. Complete expected response bytes are charged before reads, so a failed or truncated response is conservatively included in the transfer budget. Missing data, changed metadata, HTTP/range mismatch and cap violations stop execution without replacement, filling, retries or tuning.

**My inference / limits.** A source-contract pass at four points supports that particular sampling relationship. It does not establish the physical face, authenticate the historical full rendering command, or validate every PPM pixel. Correlation of two 3D anatomy neighborhoods can be informative while still leaving repeated-layer identity uncertain. The four neighborhoods belong to one physical fragment, and the original affine was fitted using sparse published landmarks. No automatic gate authorizes a photographic-label experiment. Such a stage requires a separate prospective decision and protocol.

## Commands

Run these commands from this directory, with outputs in a fresh external directory. The runtime requires Python, NumPy, SciPy and numcodecs. Tests additionally require pytest. The existing team runtime exposes the analysis dependencies through `/tmp/vesuvius-ct-deps`.

Local metadata/geometry validation, without network or pixel access:

```bash
PYTHONPATH=/tmp/vesuvius-ct-deps python frag5_native_pair_stage1.py \
  --plan frag5_native_pair_stage1_protocol.json --check-plan
```

For a reproduction after the published freeze, run once using the full commit reference and a fresh output directory:

```bash
PYTHONPATH=/tmp/vesuvius-ct-deps python frag5_native_pair_stage1.py \
  --plan frag5_native_pair_stage1_protocol.json \
  --run \
  --freeze-reference FULL_40_CHARACTER_PUBLISHED_COMMIT \
  --out /tmp/frag5-stage1-verification
```

The executable refuses an existing output directory. Preserve failed attempts. The output contains the start receipt and environment versions, pre-pixel HEAD snapshot, all TIFF header checks, acquisition ledger and object hashes, complete source-profile comparisons, all retained anatomy shifts, compact numerical arrays, resource accounting, and a result or failure receipt. No full image or model checkpoint is needed.

## Generated-input review

`test_frag5_native_pair_stage1.py` checks Cartesian chunk boundaries, XYZ interpolation and complete support, inverse-affine midpoint behavior on a linear field, interior Laplacian behavior, safe rejection of an HTTP 200 range response, reproducible fixed geometry, the unique-order gate, and rejection of a changed TIFF IFD pointer before following it. `generated_test_receipt.json` distinguishes executed generated tests from later bookkeeping edits. The [independent review receipt](independent_review.json) records a direct eight-corner interpolation cross-check. The published generated-test command substitutes generic dependency paths for private workspace paths; runner, protocol and test bytes are unchanged. These are engineering checks; no test asserts real source identity, real registration accuracy or ink quality.

## Open questions

- **ChatGPT:** Does either declared order match the raw fine volume at all four frozen geometry points? How much anatomy disagreement persists at the published affine across all fixed shifts?
- **Codex / Claude Code:** If this first stage is informative, what separate anatomical evidence would support same-sheet identity before a photographic ink comparison?
- **Owner:** The experiment produces no letters, candidate or prize claim, and sends nothing to the organisers. A later label-stage or submission decision remains separate.

## Next actions

- **ChatGPT:** Complete independent review, publish the exact protocol and runner through the owning agent, and execute once only after the published freeze handoff. Preserve every gate result.
- **Codex / Claude Code:** Continue existing work; use any later result to assess rendering and registration assumptions, without retuning a model on these four locations.
- **Owner:** Review any later substantive contribution or prize submission; no action is required for this bounded internal test.
