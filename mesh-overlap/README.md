# Exclude shared papyrus geometry from training patches

Two segments can contain the same 3D vertices even when their UV origins or dimensions differ. `surface_patch_mask.py` identifies those vertices and returns patches whose entire input footprint avoids them. Use it when planning train and evaluation splits across registered papyrus surfaces.

```bash
python -m pip install numpy scipy tifffile imagecodecs
python surface_patch_mask.py target.tifxyz reference.tifxyz \
  --frame 20260102150214 --patch 16 16 --stride 8 8 --output samples.npz
```

Pass multiple reference directories to exclude their union. `--frame` declares the common CT coordinate frame. If a file carries a different `volume_id`, the program reports an error. Meshes from different scans need registration first. Outputs use row, column indexing on the target TIFXYZ grid. The patch and stride are in native mesh grid cells. Convert your model's full input footprint, including context, into these units.

The default radius of zero compares exact stored XYZ coordinates. A positive radius, in CT voxels, also catches nearby vertices. Independent resampling or deformation can defeat a vertex-distance test. A generous radius can flag genuinely different sheets.

## Outputs

`origins_yx` contains the upper-left corner of every retained patch. `valid`, `excluded` and `eligible_vertices` explain the selection. With `--labels labels.npz`, supply binary `known` and `ink` arrays on the exact same UV grid as the target. The program also writes a conservative supervision mask for samplers that ignore `origins_yx`. It removes a label wherever an arbitrarily aligned patch of the declared size could contain both that label and a copied vertex. Blank annotations retain their distinction from unknown labels.

## Public measurements

These examples use 16 × 16 grid nodes at stride 8. Each surface pair uses the same scanner frame. Both tests compare against a filter that considers only the central node of each patch.

| Target and references | Radius (voxels) | Complete patches | Pass center check | Of those, touch shared vertices elsewhere | Pass whole-patch check |
| --- | ---: | ---: | ---: | ---: | ---: |
| PHerc. 139 w046 against w045 | 0 | 1,553 | 308 | 66 | 242 |
| PHerc. 139 w046 against adjacent w047 | 0 | 1,553 | 1,553 | 0 | 1,553 |
| PHerc. 343 composite against both parents | 0 | 337 | 22 | 19 | 3 |
| PHerc. 343 w047 against adjacent w048 | 0 | 265 | 265 | 0 | 265 |
| PHerc. 343 w047 against adjacent w048 | 2 | 265 | 265 | 31 | 234 |

The PHerc. 139 duplicate was reported by nerln. Villa already has a pairwise overlap diagnostic. This tool supplies training-patch origins and masks for downstream samplers. No physical independence can be inferred solely from retained vertices. [Reported PHerc. 139 duplicate](https://github.com/ScrollPrize/villa/issues/1547) · [Villa pairwise diagnostic](https://github.com/ScrollPrize/villa/pull/1578) · [Cross-surface training discussion](https://github.com/LimeGS/ink-sdk/issues/17)

## Reproduce

Run from this folder. The replay downloads 28 small mesh files. It checks all input hashes before running 12 comparisons. An existing cache can be replayed with the additional `--offline` flag.

```bash
python -m unittest -v test_surface_patch_mask.py
python reproduce_public.py --work /tmp/vesuvius-patch-replay
```

The fixtures manifest pins the geometry bytes. The two original sources provide their checksums. The PHerc. 343 meshes are credited to Erwin Nieuwlaar and the Vesuvius Challenge under CC BY-NC 4.0. The PHerc. 139 fixture is credited to nerln and the Vesuvius Challenge. This repository contains the tool, measurements and input references. [PHerc. 343 source](https://github.com/Nieuwlaar/pherc343-first-letters/tree/e8370f1f88a211166056df62ea6fcfffe4a1063e) · [PHerc. 139 fixture](https://github.com/nerln/vesuvius-ladder/tree/30ba654750de0671119cb7edbaf86a367caf2c82)
