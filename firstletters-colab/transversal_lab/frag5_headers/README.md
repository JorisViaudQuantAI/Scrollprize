# Frag5 header and sparse-geometry preflight

This directory packages the completed, bounded preflight described in
[FRAG5_HEADER_GEOMETRY_REPORT.md](FRAG5_HEADER_GEOMETRY_REPORT.md).
The executed reader, local plan, summary script, metadata, receipts and results
are copied unchanged. **Packaging did not rerun acquisition or scientific
calculations.** The acquired binary `ranges/` cache is deliberately not included.

The original report SHA-256 remains
`ddcb0c5a2e55e7db6ff18041a8299bb3ea349e0568ae59b48f6ec02bd65d1631`.
Its statement that files were outside git pending review describes the time
of the completed preflight; this directory is the subsequent public package.
The report's artifact table and the copied receipts retain their original
hash relationships.

## What was established

The completed run acquired 4,862 response-body bytes in 102 exact HTTP 206
ranges: three TIFF headers/IFDs, three PNG IHDRs, the PPM header and 85 sparse
PPM geometry records. It decoded no CT intensities, photographic pixels,
ink-label pixels or model outputs. See the full report for findings and limits.

The geometry-selection rule was saved locally before acquisition; the exact
index list was saved before the 85 deliberate geometry-record requests. This
was **not a public preregistration of CT or model outcomes**. Incidental binary
bytes fetched while locating the PPM header terminator are explicitly accounted
for in the report and were not decoded by the header stage.

The report's proposed twelve-pixel TIFF identity recipe is preserved as a
historical proposal. A separately published CT acquisition protocol supersedes
that recipe for subsequent work; use the current
[laboratory README](../README.md) to find that protocol. This package does not
authorize or implement the later CT experiment, establish local registration
accuracy, or settle the label palette and detector radiometry contracts.

## Read the recorded evidence

- [header_results.json](header_results.json): parsed TIFF/PNG/PPM headers.
- [geometry_sample_coordinates.json](geometry_sample_coordinates.json): fixed
  surface-grid sample and selection rule.
- [geometry_results.json](geometry_results.json): all 85 geometry records and
  both byte-order interpretations.
- [range_receipt.json](range_receipt.json): exact canonical URLs, ranges,
  response lengths, statuses, ETags and acquired-range hashes.
- [preflight_summary.json](preflight_summary.json): recorded compatibility and
  conditional source-bound checks, with their assumptions.
- [source_metadata_receipt.json](source_metadata_receipt.json): canonical URLs
  and hashes for the unchanged small files in `source_metadata/`.

Range SHA-256 values authenticate only the bytes acquired at those offsets.
They are **not whole-file hashes**. ETag/length consistency checks constrain
within-run source changes; server timestamps and version strings do not prove
rendering lineage. A new run has its own acquisition timestamps and receipt
hash. The raw source URLs are mutable.

## Run a fresh header/geometry reproduction outside the repository

The reader resolves all output paths relative to **its own script file**.
Do not execute it inside this directory: doing so would overwrite recorded
JSON files. Copy only the execution inputs to a newly created external directory
as below. Do not copy the published `range_receipt.json` into that fresh run:
it refers to the intentionally omitted original binary range cache.

From the repository root, using Python 3 with its standard library:

```bash
frag5_replay_dir="$(mktemp -d -t frag5-headers.XXXXXX)"
cp firstletters-colab/transversal_lab/frag5_headers/header_geometry_preflight.py \
   firstletters-colab/transversal_lab/frag5_headers/preflight_plan.json \
   firstletters-colab/transversal_lab/frag5_headers/summarize_geometry.py \
   firstletters-colab/transversal_lab/frag5_headers/source_metadata_receipt.json \
   "$frag5_replay_dir/"
cp -R firstletters-colab/transversal_lab/frag5_headers/source_metadata \
   "$frag5_replay_dir/"

python "$frag5_replay_dir/header_geometry_preflight.py" headers
python "$frag5_replay_dir/header_geometry_preflight.py" geometry
python "$frag5_replay_dir/summarize_geometry.py"
```

The first two commands access only the declared public headers and sparse PPM
geometry, under the reader's shared 16 MiB response-body cap. They require
exact HTTP 206/Content-Range/Content-Length and stable per-object ETags, and
stop on unexpected responses. They do not fall back to full-file downloads.
The summary command uses the newly recorded geometry plus the copied source
metadata; it performs no network or image access.

This reproduction is a technical repeat of the same public fragment and
geometry sample. It is not a new physical donor, independent registration
validation or ink-detection experiment.
