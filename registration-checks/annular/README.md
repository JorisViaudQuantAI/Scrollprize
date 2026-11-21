# A held-out CT registration check at 10–20 mm from the scroll axis

This study estimates the local XYZ residual of the published fine-to-coarse transformations of PHerc. 1203 and PHerc. 0846A. It uses X-ray attenuation and sheet geometry from the two public scans of each scroll. The 48 positions and the measurement protocol were frozen before reading cross-scan similarity in those regions. [Candidate transforms at the exact source commit](https://github.com/JorisViaudQuantAI/Scrollprize/tree/fca6b046ecc97c56227bb13e664db21b7a83df60/firstletters-colab/registration/transforms)

## Run the study

Node.js 20 or later and a writable cache directory are sufficient. Run this command from the repository root. HTTPS requests go directly to the public S3 bucket.

```bash
node registration-checks/annular/run.cjs "$HOME/.cache/scrollprize/annular-ct"
```

The first run plans 411 byte ranges covering 207,699,968 logical bytes. Store the cache on a persistent disk. Each completed range has an atomic receipt and a SHA-256 digest. An interrupted download can resume before a complete per-run manifest exists. Concurrent single-scroll runs use separate metadata and manifest files. A missing object counts as the declared Zarr zero fill only after a public bucket listing confirms its absence. The output identifies transport failures separately. Rerunning reuses verified ranges and completed sites. Append `--offline` for a cache-only replay or `--sample PHerc1203` to run one scroll. The single-scroll output files include the sample name.

The cache contains `residuals.tsv`, `summary.json`, and one JSON file per planned site under `sites/`. Each site file contains the plane-level support, the training NCC, and the held-out correlations at nominal, corrected and six displaced positions. The TSV gives signed residuals in fixed voxels and their Euclidean norms in voxels and micrometres. A positive X residual means the fine anatomy agrees best with material farther along positive X in the coarse scan. A fixed voxel is 9.362 µm.

## Positions and anatomical reference

Each scroll contributes 24 predetermined centers. At fine level-2 axial indices 1120, 2240 and 3200, place four points at radius 12 mm and four at 18 mm. At zero-based stratum k and ring j, the angular origin is 15 + 30k + 45j degrees from positive image X, increasing toward positive Y. The four points are 90 degrees apart. Coordinates are rounded to the nearest fine level-2 pixel. All positions stay in the denominator, including absent material, insufficient anatomical support and failed downloads.

PHerc. 1203 uses linear interpolation of the organizers' coarse-volume umbilicus. For PHerc. 0846A, the catalog supplies sheet-normal grids. The secondary utility `derive-centers.cjs` estimates the winding center from local sheet orientations on three fine sections per stratum. It takes their coordinate-wise median. A second estimate uses the nearest coarse normal-grid section. The estimates differ by 1.08, 0.06 and 4.03 mm. All 24 points meet the specified radii under the primary estimate. Under both estimates, 22 points fall between 10 and 20 mm. The other two have second-axis radii of 8.55 and 21.85 mm. The output reports the primary group and that predefined sensitivity subset separately. [Official PHerc. 1203 umbilicus](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/representations/umbilicus/20250820131727-umbilicus-20260924035533.json) · [VC winding-center estimator](https://github.com/ScrollPrize/villa/blob/c8f3ba45304f0a317a49a155929e30d294f201eb/volume-cartographer/core/src/normalgridtools.cpp)

The secondary utility can regenerate the anatomy record into a separate directory. It saves the public normal-grid source and its digest and produces `rings.json`. The main run uses the original frozen positions in `frozen-sites.json`. The preserved pre-acquisition protocol has SHA-256 `5a00c3394b73a8942fea6f73410aa17fd84b725412a2358d8322953d31ac83fd`. The frozen site document has SHA-256 `c5966d83214ea8ebdb59bab6af98ff168d84c493c2fc4c114fca61cd3c56ce87`.

## Residual and validation

The moving fine scan is sampled on the coarse grid by inverse transformation and trilinear interpolation. The coordinate convention follows the published matrix and uses moving level-0 XYZ equal to four times the fine level-2 indices. Each axial image is 80 × 80 coarse pixels. The interior 64 × 64 pixels supply the common support. A valid pixel must have intensity greater than 3 in both nominal images and at the six coarse-axis displacements of ±8 voxels. At least half the interior must satisfy this condition on two training planes and one held-out plane. A site is recorded as outside when every nominal plane has under 1% material. Other sites failing the support requirement are recorded as insufficient.

The estimator maximizes the weighted mean of within-plane Pearson correlations after subtracting Gaussian blurs of standard deviations 1 and 4 pixels. The three training planes have Z offsets −8, 0 and +8. Training uses every second interior sample along both X and Y. The shift search evaluates the even grid from −8 to +8 and the full integer grid from −4 to +4 in each axis. It refines within the same bounds at steps of 1, 0.5 and 0.25 voxel. Boundary maxima are flagged. The reservation of sites preceded this search. The reported fitted offsets quantify translations within this search range.

The held-out sections at offsets −13 and +13 evaluate the selected offset on every valid interior pixel. Raw and band-pass correlations are both supplied, together with six nominal-position controls displaced by ±12 voxels along each axis. Medians and linearly interpolated 90th percentiles describe material-qualified sites, with the count of each exclusion shown. Adjacent planes share anatomy. The independent sampling unit for interpretation is the site.
