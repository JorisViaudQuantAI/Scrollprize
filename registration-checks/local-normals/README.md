# Normal and tangential components of the public CT registration residuals

This experiment estimates the unsigned direction perpendicular to local attenuation bands in the 48 frozen anatomical regions. It projects the measured registration displacement and the prediction from the other two heights onto this direction. Projection units are micrometres on the fixed scan. The fixed scan has 9.362 µm isotropic voxels. [Published registration study and height-transfer model](https://github.com/JorisViaudQuantAI/Scrollprize/tree/f96d8cdd175de616754412b3c20fdede46d7064a/registration-checks/annular)

Run the tests and analysis from the repository root with Node.js 20 or later. A previous annular cache can be reused. Append `--offline` to the second command for a cache-only replay.

```bash
node registration-checks/local-normals/synthetic.cjs
node registration-checks/local-normals/run.cjs "$HOME/.cache/scrollprize/annular-ct"
```

The run produces `local-normals/summary.json` and `local-normals/sites.json` in the cache. The summary includes every planned anatomical region, all quality exclusions, and empirical medians and 90th percentiles for eligible regions. The second file preserves the individual tensor spectra and checks under the previously published region IDs. Public S3 data are read by the existing annular range reader. [Frozen anatomical positions](../annular/frozen-sites.json)

## Direction from the two acquisitions

At each location with an existing measured displacement, the analysis samples a 27 × 27 × 27 neighborhood on the fixed grid, covering about 0.25 mm. The fine CT is sampled on that grid using the previously published volume transformation at the nominal center. Each volume is smoothed in three dimensions with a one-voxel Gaussian. The structure tensor is the weighted mean of the outer products of centered intensity gradients within the central 17 × 17 × 17 voxels. Gaussian weights have a standard deviation of four voxels. Its principal eigenvector estimates the local attenuation-gradient axis. Both scans must have at least 70% material above gray level 3, gradient RMS of at least one gray level per voxel, and relative separation of their two largest eigenvalues of at least 0.45.

The two scan axes must agree within an unsigned angle of 20°. The fixed scan supplies six additional windows centered four voxels on either side along X, Y, or Z. Their Gaussian standard deviation is two voxels and radius is three voxels. At least four must meet the material and gradient requirements and a relative eigenvalue separation of 0.35. The 90th percentile of their angular deviations from the central axis must be at most 25°. Two locally supported axes differing by at least 35° raise a separate competing-directions flag.

## Repeated material ridges

Five parallel intensity profiles run along the fixed scan's central axis. They extend 18 voxels each side of the center. The four off-center rays are 2.5 voxels away along perpendicular directions. After one-voxel Gaussian smoothing, two peaks must each stand above both nearby troughs by at least ten gray levels or one fifth of the profile’s 10th–90th percentile range, whichever is larger. Their separation must be 4–18 voxels with an intervening valley. If at least three rays reproduce both peak positions to within two voxels, the site has repeated material ridges. The output separately counts weak or conflicting directions and such repeated ridges.

Repeated ridges may arise from different papyrus sheets or internal lamination. A single unresolved profile provides no sheet count. Parallel neighboring sheets can share a reliable axis, while a fold can give divergent local axes. This experiment uses CT neighborhoods. Identifying a particular sheet, its exposed side or the normal of a target mesh requires tracing that mesh through the region.

## Displacement components

For an unsigned unit axis n and a fixed-coordinate displacement v, the normal magnitude is `abs(dot(v,n))` and the tangential magnitude is `sqrt(dot(v,v)-dot(v,n)^2)`. Both are multiplied by 9.362 µm per voxel. The three vectors reported per reliable site are the measured local displacement, its leave-one-height-out linear prediction, and their vector difference. The normal of the difference is computed after subtracting vectors. Its sign would depend on an arbitrary choice of axis orientation. The summary also reports the reliable subset without a supported pair of ridges, and the PHerc. 0846A sites lying within the annulus under both published center estimates.

The orientation and ridge criteria were fixed before reading the study CT for this decomposition. The source text in `protocol-frozen.txt` has SHA-256 `8215f90324cbd65417fe031011866336cc693593b17a05923d035172a06953f4`. Synthetic fixtures check a known oblique plane, two parallel ridges, constant material, competing orientations and the vector projection. The reused annular sites and displacement estimates define the population for this follow-up.
