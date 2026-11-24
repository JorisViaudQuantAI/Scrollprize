# Four neighbouring windings of our own spiral fit of PHerc. 0211, with distances to the winner's surface

PHerc. 0211 is a scroll whose First Letters prize has been awarded; the winner's mesh is public. Everything here is
derived from public data of that scroll. Nothing here concerns unread material.

Purpose: a real test field for "which sheet is this vertex on". In our fit, the awarded row of letters is shared by
these four windings, each of which steps between the letter sheet and a neighbouring sheet (`collab/claude-025.md`).

## Provenance

- Scroll and volume: PHerc. 0211, `PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr` in the
  open-data bucket. All coordinates are voxel indices (x, y, z) of that volume at full resolution.
- Fit: villa `fit_spiral.py` at commit `0fb38c45`, band z 7650–9150, anticlockwise, one 30,000-step schedule
  (track-DT stage from step 10,000), official tracks, Lasagna normals and umbilicus. No use of the winner's mesh.
  Gate at the end: 0.376 of vertices on an m7 sheet against 0.226 for a ±8-voxel shift control.
- Winner's surface used for the distances: the public mesh
  `PHerc0211/segments/20261028201525-crop1711-lines123-v9/` (our local `tifxyz` copy, 86 × 300 vertices).

## Files

Each winding is a `tifxyz` folder: `x.tif`, `y.tif`, `z.tif` (float32 grids, one vertex per pixel, invalid where
z ≤ 0) and `meta.json`. Rows run along the scroll axis, columns along the turn.

| folder | what |
|---|---|
| `fitted/w033` … `w036` | the fit as it comes; nominal vertex step 20 voxels, in fact sheared and stretched (`claude-023`) |
| `snap/w033` … `w036` | the same vertices moved along their normal to the nearest m7 sheet (regularised; settings in `meta.json`) |
| `flat/w033` … `w036` | the fitted winding flattened per surface with villa flatboi and re-gridded at 20 voxels; this is what was rendered |

Labels:

- `*/wNNN/nearest_dist_to_winner_vox.npy`: per vertex, the distance in voxels to the nearest point of the winner's
  mesh (winner mesh up-sampled 4× by bilinear interpolation; NaN on invalid vertices). Unsigned. Vertices of our
  winding that lie outside the winner's patch laterally get large values: threshold on distance, do not read a
  large value as "wrong sheet".
- `flat/w033/` and `flat/w035/signed_dist_to_winner_along_normal.npy`: signed distance along our render normal
  from the flattened winding to the winner's surface, where the lateral offset is under 4 voxels; NaN elsewhere.
- `nearest_dist_summary.json`: per set and winding, the share of vertices within 2, 4 and 8 voxels and the median.
- `dist_summary_fitted_w030_w039.tsv`, `dist_summary_snap_w030_w039.tsv`: the other direction. For the vertices
  of the winner's three rows (top, awarded, bottom), distance to each of our windings w030–w039: median, 10th
  percentile, share within 2, 4 and 8 voxels.

## What is known about this field

- Share of the awarded row nearest to each winding: w035 44 %, w033 36 %, w036 13 %, w034 7 %. To the union of the
  four: median 3.0 voxels, 74 % within 4.
- Ink (Studio, stride 48, reading order) on `flat/w033` shows the three rows of letters over the stretch where
  that winding is on the letter sheet, about 35 % of the winner's surface. On `flat/w035`: one letter and blobs.
- Normals of these windings point away from the umbilicus; the winner's mesh has them towards it.
- The distances are to the winner's mesh, which is itself a hand-refined surface, not an independent ground truth
  for sheet identity. A vertex within 2 voxels of it is on the letter sheet for practical purposes; a vertex 6–10
  voxels away is most likely on the neighbouring sheet.

## Licence

Derived from Vesuvius Challenge open data; the data licence of the challenge applies.
