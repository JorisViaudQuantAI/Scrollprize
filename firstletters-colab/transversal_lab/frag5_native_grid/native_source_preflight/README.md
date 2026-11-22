# Native fine-TIFF metadata preflight

These are unchanged metadata-only records for the eight native fine TIFF
slices needed by the separately frozen 32-corner diagnostic. The filenames
were discovered in the public volume listing; all selected header ranges
preceded new pixel access. This preflight acquired 1,054,299 response-body
bytes: 1,052,555 directory-listing bytes and 1,744 header/IFD bytes. It decoded
no native TIFF intensity.

- [Listing receipt](listing_receipt.json): canonical directory URL and hash.
- [All eight TIFF headers](native_tiff_headers.json): parsed storage contract.
- [Range ledger](header_range_receipt.json): exact bounded metadata acquisitions.

Each selected file is little-endian, unsigned 16-bit, single-channel,
uncompressed, 4,420 by 1,400 pixels, with one strip at byte 368 and 12,376,000
pixel bytes. Orientation is absent; the selected layout uses the default
orientation. These facts are from the recorded headers, not inferred from
filename similarity. The adaptive follow-up pins exact header-range hashes
and ETags before requesting 32 voxels in 16 adjacent-pixel ranges (64 bytes).

The full directory HTML and original binary header ranges remain external;
this bundle records their acquisition rather than including the source bodies.
Range hashes authenticate only those ranges, not whole TIFFs. A successful
32-voxel comparison would support local source-grid consistency only; the
failed legacy surface-render experiment remains failed.
