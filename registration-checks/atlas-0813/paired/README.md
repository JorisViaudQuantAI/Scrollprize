# Physically paired CT profiles on PHerc. 0813

The 24 rows in [sites.csv](sites.csv) correspond to the same public winding and ordered 24 sites as the November 21 geometry audit. They can be joined to paired renderings by `id`, provided both rendering runs record their surface and pose. The small [summary.json](summary.json) contains the aggregate checks. The full [geometry.json](geometry.json) contains the relative XYZ, normals, projected triangles and all retained CT peak positions.

## What moved

| Measurement | Sites | Median | Maximum |
|---|---:|---:|---:|
| Separation of old and snapped mesh centers at identical UV | 24 | 45.19 µm | 117.25 µm |
| Angle between original and snapped finite-difference normals | 24 | 4.58° | 19.72° |
| Physical separation caused solely by rotating the normal at depth 12 voxels | 24 | 8.98 µm | 38.48 µm |
| Distance from original center to closest point on the snapped surface | 24 | 45.08 µm | 115.60 µm |
| Perpendicular miss of the original CT reference from the snapped same-UV ray | 23 | 2.97 µm | 43.13 µm |
| Euclidean distance from the fixed reference to the closest newly sampled CT crest on that ray | 23 | 3.29 µm | 55.03 µm |
| Perpendicular miss of the original CT reference from the ray through the projected old center | 23 | 4.69 µm | 48.79 µm |

The earlier, frozen test keeps the same original profile throughout. Its distance to the initial unique crest changes from 4.00 to 6.13 voxels in 23 cases. Allowing selection of a different crest yields 4.00 to 2.62 voxels across all 24. The nearest crest changes at 9 of 23 uniquely matched sites. ID 15 has two initially equidistant crests, and therefore has no single fixed biological proxy.

The new sensitivity test re-samples native CT in each physical pose. All 96 five-ray profiles returned material and at least one peak. Medians, in native voxels, describe each pose’s distance to its own nearest CT maximum.

| Center and orientation | Median nearest distance | Profiles with multiple peaks | Exact nearest ties | Runner-up within 0.5 voxel, including ties |
|---|---:|---:|---:|---:|
| Old center, original axes | 4.0 | 20/24 | 1/24 | 3/24 |
| Same physical XYZ, snapped axes | 4.0 | 19/24 | 2/24 | 2/24 |
| Snapped center at same UV, snapped axes | 2.5 | 20/24 | 1/24 | 1/24 |
| Old center projected onto snapped triangles, local snapped axes | 2.0 | 20/24 | 0/24 | 1/24 |

At ID 22 the newly sampled same-UV snapped profile has an exact tie. Profiles and m7 come from the same physical acquisition, `20250821151723`. The public masked and VCD3D volumes encode that one acquisition. No independently acquired scan or histological trace identifies which peak belongs to which sheet. The geometric comparison supports keeping raw and snapped renderings side by side.

## Reconstruct the same physical point

1. Load public winding `z11296_w020` at the pinned atlas commit. Each row gives zero-based `uv_row,uv_col` on the stored TIFF grid. Measure the original XYZ `P` there using the three coordinate TIFFs. Their native voxel pitch is 9.362 µm. The grid metadata gives `scale=[0.05,0.05]`. A stored-grid step is 20 full-resolution chart pixels. [Public input](https://github.com/pscamillo/vesuvius-eligible-meshes/tree/66000516c97e043b696ce874f145e29638109c91/meshes/PHerc0813/z11296_w020)
2. The raw reference `A` is the point `P + fixed_landmark_raw_offset_vox × Nraw` on the central ray. That offset is the unique nearest crest in the averaged five-ray profile. It gives a reproducible physical target even when the center of a new profile moves. A neighbouring interface or a fibre can produce the crest. `null` marks initial ambiguity. All XYZ vectors in this packet are expressed relative to the public original center, in native X/Y/Z order.
3. The snapped center at the same UV is `S = P + same_uv_delta_xyz_vox`. To place an overlay at an unchanged physical XYZ, keep `P` and use `rays.same_xyz_new_frame` for the rotated axes. To find the point on the snapped surface nearest `P`, use `rays.projected_raw.center_minus_raw_xyz_vox` and `projection_of_raw_onto_snap`. To project the reference `A` itself, use `projection_of_fixed_landmark_onto_snap`. Each projection gives the public triangle UVs, barycentric weights and fractional chart position. Shared-edge face ties are separated from distinct physical positions. For the projection through `P`, all 24 sites have a unique closest 3D position.
4. At a render center `C` with normal `N`, the same landmark has signed depth `dot(A−C,N)` and transverse miss `norm((A−C)−dot(A−C,N)N)`. The point on the ray at that depth still misses the original reference by the reported transverse amount. A flipped rendering normal reverses its signed depth. The two in-plane tangent directions are the chart-column derivative projected orthogonally to the normal and its cross-product companion. In the projected frame the normal is interpolated barycentrically and the tangent comes from that triangle. The five rays use the center and ±2-voxel shifts along each tangent. Consequently the fixed-center CT comparison includes the changed in-plane frame as well as normal tilt. The analytic fixed-center ray-tip distances at 6, 12 and 24 voxels isolate normal tilt.
5. `nearest_peak_to_center.indices` refers to indices in that same frame’s ordered `peak_offsets_vox`. `margin` is the runner-up minus nearest absolute distance. It is undefined with fewer than two peaks. `nearest_crest_to_landmark_3d` compares the *positions* of newly measured crests on the current central ray to `A`. Comparing peak-list indices across different poses has no intrinsic sheet interpretation.

The original selection used valid atlas geometry, a central initial site and farthest-point sampling in 3D before reading CT or m7. The minimum physical separation is 245.43 voxels, or 2.30 mm. The original aggregate CT results were public before this extension. Even IDs are reserved for a subsequent no-tuning rendering comparison, and odd IDs are available for testing the adapter. All 24 remain in the geometry table. [Original sampling protocol](../protocol.txt) · [Extension frozen before measurements](https://github.com/JorisViaudQuantAI/Scrollprize/commit/6898f232664f4a2da63aedfc23787c213149451e)

## Reproduction and verification

Run the original CPU reproduction in the parent directory, then run the two commands below from this directory. The first recomputes the new measurements from both full meshes and the official native CT. The verifier recalculates the geometry identities, ambiguity conventions, correspondence distances and earlier paired summaries directly from the table, with no CT download. It also runs against the checked-in files on a clean checkout.

```sh
python ../check.py
python measure.py
python verify.py
```

The recorded CPU job used 46,560 trilinear samples across 51 public byte ranges, totalling 12,500,992 bytes for the four new frames. It verified the three snapped TIFF hashes and the earlier paired endpoints against the existing audit. The independent coordinate check reproduced 24 sites, 96 frames, 23 unique original references and nine nearest-crest switches. [Pinned execution](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/44920691758) · [Original reproduction and software versions](../README.md) · [Official scan](https://scrollprize.org/data_browser/PHerc0813)
