# Frag5: second audit of the Volume Cartographer sampling contract

**Scope:** static source inspection only. No experimental pixels, fitted radius, phase, normal, intensity scale, registration, or additional source-volume acquisition. The relevant source is Volume Cartographer revision `7640ba6c8e41f39a7aef63e9780cd13c296b2d15` (v2.26.0). This audits that source implementation; it does not authenticate the historical executable or command used to create Frag5's surface TIFFs.

**Finding:** no additional numerical operation was found that repairs the failed stage-1 hypothesis. The code confirms the tested default-chain arithmetic. It also exposes specific unresolved invocation and source-representation contracts.

## Confirmed source behavior

| Component | From a source: exact behavior |
|---|---|
| PPM serialization | `PointSetIO<cv::Vec6d>` reads/writes binary scalar storage directly, with no endian conversion. It reads one full width of six-component records per row. `OrderedPointSet(y,x)` indexes `y*width+x`. The last three components are read as the normal by `LayerTexture`. [PointSetIO](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/include/vc/core/io/PointSetIO.hpp), [OrderedPointSet](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/include/vc/core/types/OrderedPointSet.hpp), [LayerTexture](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/texturing/src/LayerTexture.cpp). |
| Masks and ordering | A PPM mask controls whether a location is rendered. It does not change its coordinates or intensity. `LayerTexture` sorts processing by source Z for access order but writes each result to its original surface `(y,x)`. [PerPixelMap](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/src/PerPixelMap.cpp), [LayerTexture](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/texturing/src/LayerTexture.cpp). |
| Parameters and ray | Radius/interval/direction setters assign their supplied values directly. `LineGenerator` uses the first supplied axis without renormalizing it, then calls `Volume::interpolateAt(p + offset*n)`. Bidirectional offsets start at `-abs(radius)` and increment by `interval`. [NeighborhoodGenerator](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/include/vc/core/neighborhood/NeighborhoodGenerator.hpp), [LineGenerator](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/src/LineGenerator.cpp). |
| Integer grid and interpolation | `Volume` reads `getSliceData(z).at<uint16_t>(y,x)`. Trilinear interpolation uses the eight integer neighbors, double arithmetic and `cvRound`, then returns uint16. The inspected path adds no half-voxel translation, axis swap, normal flip, or window normalization. [Volume](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/src/Volume.cpp). |
| Native source representation | The actual `Volume` loader uses native TIFF slices through `cv::imread(path,-1)`, not Zarr. Slice filenames are zero-based integers padded to `len(str(number_of_slices))`, followed by `.tif`. With 7,010 slices this means four digits. A direct TIFF byte reader must separately validate orientation, channels, photometric interpretation and storage layout. [Volume](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/src/Volume.cpp). |
| Surface TIFF output | `LayerTexture` produces uint16 single-channel images. The TIFF branch passes their values to `WriteTIFF`; that writer copies rows and sets uint16/min-is-black metadata without intensity rescaling for this case. Numbering follows the returned layer order. [ImageIO](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/src/ImageIO.cpp), [ImageIO sequence](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/include/vc/core/io/ImageIO.hpp), [TIFFIO](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/src/TIFFIO.cpp). |

## Concrete counterexamples to an overconfident rendering contract

1. **My inference from the source:** 65 layers do not uniquely establish a radius of 32 or `layer32=p`. For bidirectional sampling at interval 1, the count is `floor(2*radius)+1`. Every radius in `[32,32.5)` yields 65 samples, while the offset of sample 32 is `32-radius`. This is a mathematical counterexample; no radius was estimated or tested against the data. [LineGenerator](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/src/LineGenerator.cpp).

2. **My inference from the source:** a directory listing of 65 TIFFs does not itself establish a single 65-plane invocation. `LayerFromPPM` accepts an existing output directory, and `WriteImageSequence` writes its generated indices without deleting older extra planes. This makes stale higher-numbered files a code-permitted counterfactual, not an allegation about the actual dataset. [LayerFromPPM](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/apps/src/LayerFromPPM.cpp), [ImageIO sequence](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/include/vc/core/io/ImageIO.hpp).

3. **From a source:** the CLI can select a different volume, apply an explicit transform or inverse transform, and use explicit radius/interval/direction options. The fallback radius is material thickness divided by twice the selected volume's voxel size. An applied transform also normalizes the transformed normal by default. These choices are not encoded in the basic PPM header. Their use in the historical Frag5 invocation remains unknown. [LayerFromPPM](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/apps/src/LayerFromPPM.cpp), [Transforms](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/src/Transforms.cpp).

## Consequences for the separate adaptive experiment

**My inference:** comparing the 32 integer native-TIFF and cached-Zarr corner values at the same four fixed PPM points tests an unverified interface directly. It does not require assuming where legacy surface plane 32 lies. Exact equality would establish equality of those 32 values, supporting local source-grid compatibility; it would not authenticate the entire volume, all historical inputs, the surface map, or a photograph/label association.

**My inference:** a separately published, explicitly adaptive scanner-grid anatomy experiment is defensible after that limited check. Its question is the similarity of the named raw fine/coarse neighborhoods under the fixed published affine, using the already-fixed four selectors, 33-cubed grids, 125-point approximate footprint and all 27 shifts. Legacy surface-TIFF reproduction is not a logical prerequisite for that narrower question. The PPM locations are only prior geometric selectors. Coarse-grid/affine binding remains a metadata-supported interpretation under examination, and correlations alone cannot prove physical sheet identity.

The original stage-1 result remains `neither`. There is no silent relaxation of that gate, fitted correction, whole-volume identity claim, photograph/label correspondence claim, or automatic permission for a label stage.

## Source acquisition record

Most sources were already cached from the first audit. Four missing helper files were fetched at the same immutable revision to close the setter/storage/writer path: 20,512 bytes of source code in total, each verified against the cached Git tree's blob SHA1 and recorded with SHA256. Their receipt is `source_supplement/acquisition.json`. No experimental data were acquired during this reaudit. Upstream source files remain outside the public repository; the report contains descriptions and links, not copied implementation code.

## Open questions

- What exact executable, options, native source version and output-directory state produced the legacy surface TIFFs? I do not know.
- Are the named native TIFF and raw-Zarr samples equal at the fixed 32 integer corners? The newly proposed test has not run as part of this audit.
- Does the named coarse raw grid share the intended native volume coordinate convention? The planned raw-grid anatomy experiment can assess descriptive agreement, not uniquely identify the cause of disagreement.

## Next actions

- **ChatGPT implementation agent:** complete the bounded native TIFF metadata preflight and draft the separate fixed 32-voxel protocol.
- **ChatGPT reviewer:** review the new protocol and acquisition guards before any new pixels are read.
- **Owner / root coordinator:** preserve the failed original stage, publish the distinct adaptive protocol, and keep its conclusions limited to source representation and unlabeled scanner-grid anatomy.
