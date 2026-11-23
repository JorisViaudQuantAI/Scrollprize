# Cross-scan displacement on the public PHerc. 1203 mesh

This test applies the published leave-height-out ridge correction to real surface normals. The mesh is the official current `auto_grown_20251005230830031`, with a raster of 232 rows by 268 columns. The study follows its assignment to the fixed scan `20250820131727-9.362um-1.2m-113keV-masked.zarr`. The moving scan is `20260319130212-2.403um-0.2m-77keV-masked.zarr`. The mesh metadata includes a geometric area of 16.0994 cm². The tests below sample individual neighborhoods within it. [Official mesh metadata](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/segments/raw/auto_grown_20251005230830031/meta.json) · [Published mesh-to-scan assignment](https://github.com/JorisViaudQuantAI/Scrollprize/blob/a3495c910f71584a53fc39fe84ae15c241e8cb75/collab/codex-002.md) · [Registration source](https://github.com/JorisViaudQuantAI/Scrollprize/tree/f96d8cdd175de616754412b3c20fdede46d7064a/registration-checks/annular)

## Selection and independent image planes

We used finite central UV differences of XYZ in fixed L0 voxel coordinates to compute a unit normal. The largest absolute coordinate sets a reproducible positive sign. Positive or negative normal values do not identify a recto or verso. Each of the four adjacent vertices had to be positive, finite, and within 28 fixed voxels. The geometry had to leave a 64-voxel margin against the bounds of both arrays. The public coarse-to-fine registration uses the inverse affine, followed by division by four to index the fine scan at level 2. Thus the effective intensity sampling pitches are 9.362 µm and 9.612 µm. The correction is reported in the fixed frame.

At stride eight in UV, 189 of 986 interior grid positions passed the geometry and volume checks. Using ±160 fixed voxels around the three median heights of the frozen source coordinates gave 30, 0 and 0 positions in A, B and C. The first site at each height was closest to its center. Subsequent sites maximized the minimum XYZ distance from previously chosen sites, with at least 100 voxels separation and a four-site cap. When the stride missed an entire band, we applied the same selector to every interior vertex. Of 61,180 interior vertices, 12,151 passed the geometry and bounds checks, including 1,901, 17 and zero in A, B and C. This fallback was specified after seeing geometry counts and before reading the B intensities.

The displacement model excludes **all** residual measurements from the test site's height. It centers XYZ on the training set, scales it by 1,000 and fits slopes by ridge regression with unit penalty and a free intercept. The fixed-frame correction maps nominal fine anatomy into fixed coordinates. For a mesh already defined in the fixed scan, correcting its fine sampling coordinates requires subtracting that displacement then applying the inverse affine. One extra substitution accounts for the near-linear spatial variation.

For each selected vertex, we compared a 48 × 48 central region on each of two fixed-frame XY planes, at Z offsets −13 and +13. Neither plane enters any fit. Before and after use identical masked pixels with material in all three images. We eroded that joint mask by two pixels and required at least 1,152 samples per plane. Raw NCC and the difference-of-Gaussians NCC (σ 1 and 4) describe local agreement. The two planes of a site share anatomy. The tabulated pixels are correlated and cannot be treated as independent repeats.

## Results

Six vertices were tested, four in A and two in B. All 12 held-out planes met the material criterion, with 27,648 retained plane pixels in total. At A the model used 12 training sites from other heights. At B it used 11. No vertex of the mesh passed the geometric test at C.

| On 12 unseen axial planes | Median nominal | Median corrected | Median of paired gains | Improved planes |
| --- | ---: | ---: | ---: | ---: |
| Raw intensity NCC | 0.82354 | 0.87973 | +0.03336 | 11/12 |
| Difference-of-Gaussians NCC | 0.86307 | 0.90517 | +0.03574 | 11/12 |

Averaging each vertex's two planes gives improvements at 5/6 vertices for both metrics. One plane deteriorated at UV (105,217), with band-pass NCC dropping by 0.03958. The absolute normal correction had median **1.780 µm** (range 0.253–2.475). The tangent magnitude had median **7.913 µm** (5.422–10.367). All six paired normal-profile correlations increased, with median gain 0.05219. Five of six fixed profiles had a central peak and a distinct neighbor, with spacings of 5.0, 6.0, 6.0, 12.5 and 14.5 fixed voxels. None of those five normal corrections reached half the observed spacing. Three sites also had a unique corresponding moving peak both ways. Their offsets from the fixed peak were 1.0→1.0, 0.0→0.5 and 4.0→3.5 voxels. The remaining profiles did not yield that stricter comparison.

The 17 eligible full-grid points near the boundary of B permit two separated sites. The six vertices come from one mesh dominated by a similar local normal orientation. Neither the median displacement nor the count of improved sites characterizes every orientation, height or sheet.

## Scope of the surface check

Along the fixed surface normal we also sampled both scans at half-voxel intervals in a ±24-voxel window. We counted fixed bright peaks after Gaussian smoothing (σ 0.6 fixed voxels), with prominence 12/255 and peak separation of at least four voxels. A central peak had to lie within three voxels. Its nearest distinct maximum gives a local spacing proxy. We tested whether the normal correction reached half that separation. An additional diagnostic measures the nearest moving peak within six voxels when it is unique, and evaluates the paired 1D profile NCC. These CT peaks may describe fibers, partial-volume effects or different interfaces. This experiment does not establish the anatomical identity of adjacent sheets or detect ink.

## Reproduction

From the repository root, install NumPy, SciPy, requests, tifffile and imagecodecs, then run `OPENBLAS_NUM_THREADS=1 python registration-checks/annular/mesh-transfer/check.py`. The script reads the canonical public XYZ files and limited public S3 voxel ranges. It emits coordinates, normals, two correlation pairs per site, geometry filters, exact source hashes and peak diagnostics to `mesh-transfer-result.json`. [Runner](check.py) · [Per-site evidence](result.json) · [Completed run](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/44903894816)
