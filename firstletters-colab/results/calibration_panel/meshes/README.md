# Meshes of the two blind PHerc. 0846B patches (awarded scroll; public)

`flat_bl1/` and `flat_bl2/`: `tifxyz` (x.tif, y.tif, z.tif, meta.json), level-0 voxel coordinates of volume
`20250804142305` (9.362 µm), after flatboi (50 iterations, symmetric Dirichlet). These are the surfaces behind
`../images/blind_0846B_bl1.jpg` and `blind_0846B_bl2.jpg` (claude-033).

| patch | seed asked | seed after snap to the surface prediction | area | grid quality (share of cells with shear ≤ 8°) |
|---|---|---|---|---|
| bl1 | (3200, 5000, 5800) | (3200, 5006, 5800) | 11.1 cm² | 0.993 |
| bl2 | (4800, 4750, 6400) | (4798, 4746, 6400) | 12.2 cm² | 0.592 |

These are a regrow of 27 November 2026, 20:22–20:35 UTC, with the same job as the original
(`../jobs/hb_700_g846bl.sh`), not the files the images were made from: the original files were not kept.
GrowPatch is not bit-reproducible across runs (the thread count also differs), so expect small differences at the
edges. The grid quality of the originals was 0.99 and 0.57.

Recipe: seed chosen on the CT only → snap to the m7 surface prediction (±48 voxels) → `vc_grow_seg_from_seed`,
90 generations, the PHerc. 0306B winners' parameters → flatboi → `gridq.py`. It needs no GPU: 13 minutes on an
8-core CPU machine, of which 7 minutes are the download of the normal grids for z 4000–8300.

Text shows in one order only and GrowPatch gives the normal an arbitrary sign: read both orders.
