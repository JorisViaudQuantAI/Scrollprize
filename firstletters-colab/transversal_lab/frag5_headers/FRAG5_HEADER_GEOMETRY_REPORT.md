# Frag5 compact header and geometry preflight

22 November 2026. **Scope:** public PHerc. 1667 Cornice 1 Fragment 3 headers and a fixed sparse surface-geometry lattice. No CT intensities, PNG pixels, photographs, ink-label arrays, model outputs or unread-scroll candidates were decoded. Files and code remain outside git pending review. This report continues the metadata-only donor preflight; it does not repeat the five-landmark sensitivity experiment.

## Outcome

**Measured:** Compact access works for the inspected assets. The reader completed **102 exact HTTP 206 requests, consuming 4,862 response-body bytes**, against a hard **16,777,216-byte** cap. Each returned Content-Range matched the requested interval, Content-Length matched the interval length, and each object's ETag and total length remained stable. There were no HTTP 200 fallbacks, acquisition errors or retries. The complete [range receipt](range_receipt.json) records canonical URLs, timestamps, statuses, lengths, ETags, every range hash and every byte count. These are hashes of acquired ranges, **not whole-file hashes**.

**My inference:** This establishes a small, reproducible acquisition route. It does **not** authenticate the exact raw CT source, render phase, physical face, photographic registration or detector radiometry. Those remain separate gates.

## 1. File grid and TIFF access contract

**Measured from headers:** [00.tif][t00], [32.tif][t32] and [64.tif][t64] have the same metadata:

| Field | Value |
| --- | --- |
| Image dimensions | Width 4,560; height 7,309 |
| TIFF encoding | Classic TIFF, little endian |
| Sample type | One unsigned 16-bit sample per pixel; SampleFormat 1 |
| Compression | 1, uncompressed |
| Strip layout | One strip; RowsPerStrip 7,309 |
| Strip start / size | Byte 8 / 66,658,080 bytes |
| IFD start | Byte 66,658,088 |
| IFD entries / next IFD | 12 / 0 |
| Total file length | 66,658,265 bytes |
| Software | `volume-cartographer 2.26.0` |
| Other provenance | No ImageDescription, source-volume UUID, physical resolution, render-radius/interval or DateTime tag |

**My inference from this measured layout:** A future single pixel at surface row `v`, column `u` is the two-byte interval starting at `8 + 2*(v*4560 + u)`. A width-`w` row crop needs `2*w` bytes; a height-`h` rectangle needs `h` such requests, assuming the object still satisfies the verified header contract. A compressed TIFF reader is unnecessary for these three inspected planes. Do not assume the other 62 planes have the same layout without checking their headers.

**Measured:** The three PNG IHDRs and the PPM header also declare **4,560 × 7,309**:

| Asset | IHDR bit depth / color type | Actual complete object length | Inspected scope |
| --- | --- | ---: | --- |
| [inklabels.png][labels] | 1 / 3, indexed palette | 33,481 bytes | First 33 bytes only |
| [mask.png][mask] | 8 / 0, grayscale | 81,169 bytes | First 33 bytes only |
| [ir.png][ir] | 8 / 6, RGBA | 4,685,885 bytes | First 33 bytes only |

IHDR CRCs passed. **I do not know the ink/background palette mapping or mask values from this preflight:** PLTE, tRNS and IDAT were outside the explicitly authorized IHDR-only scope. A palette index must not be silently treated as a grayscale intensity or an ink class. **My inference:** Matching dimensions permit a common declared grid, but do not demonstrate subpixel photo-to-CT alignment, unchanged labels or accurate manual annotation.

## 2. PPM storage contract and fixed geometry sample

**Measured:** [result.ppm][ppm] has this exact 73-byte ASCII header:

```text
width: 4560
height: 7309
dim: 6
ordered: true
type: double
version: 1
<>
```

The header SHA-256 is `87ea3e47a866b713a95946a9e91767d06e60e0aef0f7e14e943d511f4d8a1744`. Its declared body length is exactly `4560*7309*6*8`, giving total length **1,599,793,993 bytes**. Record `(v,u)` begins at `73 + 48*(v*4560 + u)`. The header-search requests fetched 55 incidental binary suffix bytes beyond the header; those bytes were not decoded by the header stage. The sparse geometry stage separately fetched its prescribed records.

**From primary source code:** [PPMGenerator.cpp][ppmgen] constructs each record as `(X,Y,Z,NX,NY,NZ)`. [PointSetIO.hpp][pointset] serializes an ordered row-major point set as direct native binary values; [PerPixelMap.cpp][ppmio] supplies the PPM serialization. There is no universal endian marker or source-volume UUID in this PPM header. The code links are pinned to the commit tagged `v2.26.0`.

**Measured from the prespecified geometry sample:** Before reading any record, the plan fixed a 9 × 9 interior lattice plus four corners. Interior row/column indices are `floor(k*(dimension-1)/10)`, `k=1…9`; the Cartesian product is enumerated in row-major order. The four controls are `(0,0)`, `(0,4559)`, `(7308,0)`, `(7308,4559)`. The exact [85 coordinates](geometry_sample_coordinates.json) and [all decoded geometry records](geometry_results.json) are retained; no location was added because of CT, ink or a model result.

| Sparse-sample result | Count / range |
| --- | --- |
| Records inspected | 85 |
| All six doubles exactly zero | 21 |
| Nonzero records with little-endian unit normals | 64 |
| Magnitude of those stored normals | 0.9999999999999998 to 1.0000000000000002 |
| Big-endian interpretation with a unit normal within 1e-6 | 0 |
| Nonzero source-coordinate X range | 451.1884 to 3,904.2924 |
| Nonzero source-coordinate Y range | 175.4655 to 1,065.7326 |
| Nonzero source-coordinate Z range | 768.8522 to 6,338.4871 |

**My inference:** This particular file is compatible with little-endian doubles. This empirical result does not make every PPM file little endian. The 21 zero records have no usable normal; their count is not a tissue-mask coverage estimate. The sparse sample does not certify any complete rectangular patch.

## 3. Source compatibility, phase and provenance limits

**From official metadata:** The candidate fine volume is UUID `20231121133215`, nominal **3.24 µm**, ZYX **7,010 × 1,400 × 4,420**, at **70 keV**. The candidate coarse volume is UUID `20231130111236`, nominal **7.91 µm**, ZYX **3,131 × 668 × 2,046**, also **70 keV**. Their [fine metadata][fine-meta], [coarse metadata][coarse-meta] and [published source-to-target affine][affine] supply this association. The surface directory's name matches the fine acquisition, but its acquired headers do not bind the PPM to that UUID.

**From official Zarr metadata:** Both raw level-zero arrays use C-order little-endian uint16, `128×128×128` chunks, `/` chunk separators and Blosc/Zstd level 3 with shuffle 0. Shapes match the respective volume metadata. Physical pitch comes from the volume metadata, not from interpreting relative multiscale factors as micrometres. The fine metadata was acquired independently by the parallel metadata agent; this lane did not duplicate those requests.

| Metadata object | Canonical source | SHA-256 |
| --- | --- | --- |
| Fine `.zarray` | [70keV_3.24um_.zarr/0/.zarray][fine-zarray] | `2795b898b2055d1cb2a1ee2d7c8566921c8f9a089a9cadd47d5dae01aab87f49` |
| Fine `.zattrs` | [70keV_3.24um_.zarr/.zattrs][fine-zattrs] | `97e968dbbf11cc856dc57648aaaa9e6cade5e6575472dcfdeac7cfa50f546dd9` |
| Coarse `.zarray` | [70keV_7.91um_.zarr/0/.zarray][coarse-zarray] | `049d287c2fa55628cf9ea2b5fccbd0c813082750d7a1067059abd30402b3e124` |
| Fine-to-coarse affine | [Transform3D JSON][affine] | `e4653b4d904c98ac215e28e43cb0d3033ea214eaf7c696264d10a9029b4c706f` |

**Measured, conditional on that candidate identity:** All 64 nonzero sampled PPM centers lie inside the candidate fine grid, and their published-affine images lie inside the coarse grid. A conditional `p + d*n` ray with `d=-32…32`, using the stored normals directly, retains complete eight-corner trilinear source support in both grids at these 64 points. No normals were normalized, flipped or chosen using outcomes. These endpoint bounds are a geometry diagnostic, **not evidence that Frag5's actual rendering command used radius 32, interval 1 or center plane 32**. The [summary](preflight_summary.json) preserves per-record bounds and its assumptions.

**From source / inference:** The pinned [v2.26.0 tag-target commit][vc-commit] has committer date 9 August 2024, while the server advertises December 2023 last-modified dates. That tag-target date is not a proven binary build date. Neither server timestamps nor the version string authenticate the original rendering lineage. Treat ETags and lengths as acquisition-consistency checks, not proof of creation history.

**My inference:** Source identity and phase should next be tested by a separately frozen, tiny TIFF-versus-raw-fine comparison at geometry-selected points. Even exact agreement would establish the sampled source/render contract at those points, not independent fine-to-coarse registration accuracy. This preflight contains no new landmark fit or local registration measurement.

## 4. Compact next acquisition plan — proposal, not executed

1. **Select from geometry before intensities.** The next protocol can choose four distinct nonzero, bounded points nearest the four fractional quarter positions, with a deterministic distance metric and tie rule. Preserve all four outcomes, including failures; do not replace a center after viewing intensity or labels. The current 85-point sample supplies candidates but does not establish complete patch support.
2. **Test the candidate fine source/render contract first.** At four fixed centers, reading the corresponding pixels of TIFF planes 00, 32 and 64 costs exactly **24 TIFF pixel-body bytes**, in twelve two-byte ranges. This is a future intensity step, not part of the completed preflight. Pin the candidate `p+n*d` mapping, depth offsets, interpolation and rounding from the renderer audit before acquisition. Do not search alternative depths until something agrees.
3. **Acquire raw arrays by a complete, shared chunk plan.** Enumerate all eight trilinear corners for every requested fine and coarse sample; reject invalid bounds without clipping or filling. Deduplicate chunks across the four centers and both grids. A shared cap of 64 decoded uint16 `128³` chunks is **268,435,456 bytes**. Compressed transfer sizes are still unknown; cap actual source-response bytes separately and record acquired chunk hashes. The sparse geometry's individual bounds do not by themselves bound full-patch chunk unions.
4. **Expand only after the identity gate.** A future PPM rectangle of `h×w` records costs exactly `48*h*w` payload bytes with row ranges. For example, a 64×64 rectangle is **196,608 bytes**, and a 192×192 rectangle is **1,769,472 bytes**. Their physical flattened XY metric has not been independently calibrated here. A surface-grid average is not automatically a scanner-space voxel average.
5. **Resolve label and radiometric contracts separately.** A palette-only metadata inspection is a small next step, but was outside this IHDR-only authorization. Photographic annotation lineage, mask interpretation and uint16-to-model-uint8 mapping remain unresolved. No new model evaluation should conceal these choices behind per-window scaling.

These are **my proposed gates and arithmetic on measured file layouts**, not an executed CT experiment or a promise of independent registration accuracy. The mapped original ray `A*n` and transformed geometric normal `A^-T*n` remain distinct constructions under an anisotropic/sheared affine; this sparse preflight does not choose between them for a later detector experiment.

## 5. Reproduction and integrity

The reader issues only bounded byte ranges, requires exact 206/Content-Range/Content-Length and identity transfer encoding, sends If-Match after the first ETag, and stops rather than reading an unexpected response body. A later invocation reuses only locally hash-verified acquired ranges and charges new bytes to the same receipt. It does not decode TIFF strips or PNG data.

```bash
python header_geometry_preflight.py headers
python header_geometry_preflight.py geometry
python summarize_geometry.py
```

The summary script uses the already acquired PPM records and the earlier official affine/volume metadata copied unchanged into `source_metadata/`; it performs no network or image access. [Source metadata receipts](source_metadata_receipt.json) give the canonical URLs, filenames and byte hashes of these copies. The command sequence reproduces this local preflight. A fresh range run still requires access to the mutable public sources and must fail if their ETags/lengths change within a run; new acquisition timestamps and the resulting receipt hash need not equal this run's receipt hash.

| Artifact | SHA-256 |
| --- | --- |
| `preflight_plan.json` | `d6eeebaaae064f451514cd063f5f3f78a2356902159823e144784de9df92f7f7` |
| `header_geometry_preflight.py` | `10845f2cb6f59483ab7927b4a5391e46882b25d246784509dfaf0c82fe01b5b5` |
| `header_results.json` | `ec65d49fb02cdf85acf66f3339e3d808b179daa28d5e631b592bf7fe1b2995be` |
| `geometry_sample_coordinates.json` | `0ef7558378fdf61ad1c063749201e1e71a92cca94c7b5931f2284ba61d64557c` |
| `geometry_results.json` | `e8275973279ff25f36750fc0ee8b3ebc34785a5188be71a0df98797778e17fb7` |
| `range_receipt.json` | `e4ac0897a8f61dbb853fd6f84f5e6256a292399019797330927925a12c22f50f` |
| `preflight_summary.json` | `abb492fde8327d3b612a54b033b1321ee0a55620cfb4f447f9c4721fb6b09411` |

## Open questions

- **Codex / ChatGPT:** Does a fixed TIFF/raw-fine comparison verify the candidate source UUID and depth phase before any fine/coarse comparison?
- **Claude Code:** Which physical sampling and uint16 radiometry contract should a future native-pair detector test preserve?
- **Owner / ChatGPT:** What documentation establishes the original Naples photographic-label production and palette semantics? File dimensions do not supply that evidence.

## Next actions

- **ChatGPT:** Review and publish the separate geometry-selected source-contract protocol before acquiring CT intensities; retain all failed gates and preserve the current no-intensity boundary.
- **Codex:** Treat sparse affine and source-bound checks as prerequisites, not independent local registration validation.
- **Claude Code:** Keep this control preflight separate from live unread-scroll inference and existing GPU work.
- **Owner:** No prize submission or disclosure decision is requested; this work contains no candidate or prize result.

[t00]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/surface_volume/00.tif
[t32]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/surface_volume/32.tif
[t64]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/surface_volume/64.tif
[labels]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/inklabels.png
[mask]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/mask.png
[ir]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/ir.png
[ppm]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/result.ppm
[fine-meta]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes/20231121133215/meta.json
[coarse-meta]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes/20231130111236/meta.json
[fine-zarray]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes_zarr/70keV_3.24um_.zarr/0/.zarray
[fine-zattrs]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes_zarr/70keV_3.24um_.zarr/.zattrs
[coarse-zarray]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes_zarr/70keV_7.91um_.zarr/0/.zarray
[affine]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/transforms/20231121133215-to-20231130111236.json
[pointset]: https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/include/vc/core/io/PointSetIO.hpp
[ppmgen]: https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/texturing/src/PPMGenerator.cpp
[ppmio]: https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/src/PerPixelMap.cpp
[vc-commit]: https://github.com/educelab/volume-cartographer/commit/7640ba6c8e41f39a7aef63e9780cd13c296b2d15
