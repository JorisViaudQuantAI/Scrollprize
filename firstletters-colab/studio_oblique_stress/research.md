# Q33: native CT sampling and obliquity

Source and asset audit, 24 November 2026. This note motivates and constrains the experiment; it contains no detector result. The executable protocol specifies the perturbation actually tested.

## Why sample the native volume

The [official Vesuvius FAQ](https://scrollprize.org/faq#does-segmenting-and-flattening-need-to-happen-before-ink-detection) explains that sampling directly from the original 3D volume avoids resolution lost when constructing a differently oriented surface volume. This motivates using the original CT when testing geometry: a second resampling of the already rendered 16-plane tensor would combine an orientation change with additional interpolation and limited depth support. It would not reproduce the same observation as a single sampling operation on the native CT.

An applicable primary medical-imaging reference is P. Thévenaz, T. Blu and M. Unser, [“Interpolation Revisited”](https://bigwww.epfl.ch/publications/thevenaz0002.html), *IEEE Transactions on Medical Imaging* 19(7), 739–758, 2000, [DOI 10.1109/42.875199](https://doi.org/10.1109/42.875199). It analyzes approximation and interpolation artifacts under image transformations. Its relevance here is experimental control: keep the kernel, number of interpolation passes, quantization and normalization fixed across arms. It does **not** predict the direction or size of Studio's response to obliquity. This experiment does not compare interpolation algorithms.

These sources support the sampling design. Quantitative statements about the released checkpoint must come from the frozen local experiment.

## Public positive control and exact source contract

Use the same author holdout dataset, `841-expanded-layer-03-approved-r000001`, and the same 53 labeled central cores as the preceding Studio scale experiment. Its labels distinguish unknown, background and ink; unknown pixels remain excluded. It is an author holdout within a training scroll, not an independent new-scroll test or an optical ground truth.

The data pin is [`paranke/vesuvius-814-841-9b-studio-training-data@ad919d288ede1c4daf92725288bb6fe29c558591`](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/tree/ad919d288ede1c4daf92725288bb6fe29c558591). Author source revision: [`bf66f753176792a966bca17822629fb8821f20c8`](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/tree/bf66f753176792a966bca17822629fb8821f20c8).

- [Pinned selection record](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/ad919d288ede1c4daf92725288bb6fe29c558591/donor-studio-resunet-scratch-20260926/prepared/841-expanded-layer-03-approved-r000001/selection.json): SHA-256 `1d52f2feb76c53b408612ff5b0b3c5cca6a7a8ffeba27e60c79047a77b9fe83b`, 17,271 bytes.
- [Pinned completed-render record](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/ad919d288ede1c4daf92725288bb6fe29c558591/donor-studio-resunet-scratch-20260926/prepared/841-expanded-layer-03-approved-r000001/render.json): SHA-256 `bc2bdfc1280ea8fb837815454c10d5f2125b4c2f45d0ee313bf85a4ac330329e`, 1,433 bytes.
- [Public backup member index](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/ad919d288ede1c4daf92725288bb6fe29c558591/backup/file-index.json): SHA-256 `e81e3d6971e3bd4a52f8872e7d931c8112c503ded0b3ca1fda25e01c22bca40b`.

The source mesh is in a 9.612 µm geometry grid. The author recipe's affine matrix already maps those stored mesh coordinates into native level-0 coordinates; its linear part includes the factor four from that geometry grid to the high-resolution source grid. The separate `transform.json` is retained only as provenance. Applying it again would be incorrect.

The native CT is:

```text
https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0841/volumes/20250821151531-9.366um-1.2m-113keV-masked.zarr/0/
```

Fresh metadata confirms shape `[19400,7703,7703]` in ZYX order, uint8, uncompressed C order, chunks `[128,128,128]`, slash-separated keys. Its `0/.zarray` SHA-256 is `85082292fa204f738d1bc452ab78fdf31a0ca12e9cb192e01b3b5e0e1a0c3a0f`. The normal sign is +1, native depth offsets are `[-8,…,7] × 9.366 µm`, and runtime depth reversal is false. Label pixel centers use `(index+0.5) × 9.36 µm` in the original UV chart. These are separate XY/depth pitches and should retain their recorded values.

The author's released CT tiles remain the baseline reference: matching geometry metadata or array shape alone does not establish correct numerical resampling. Verify reconstructed zero-tilt inputs against those fixed tiles before detector comparisons.

## Bounded geometry acquisition

`assets.acquire_geometry(cache_dir)` returns `(mesh_dir, recipe_dict, provenance_dict)` without obtaining any CT or checkpoint. It verifies the index and source records, then recovers the four mesh files plus the transform receipt from the first **8,388,608 bytes** of the pinned `backup/remote-artifacts.tar.gz`. The complete archive is 1,376,603,369 bytes; its full download is unnecessary.

The actual prefix request returned exact HTTP 206 and `Content-Range: bytes 0-8388607/1376603369`. Its SHA-256 is `b5d6f3cd285e5e168fccddc546659a2db6ef7d1be1f8c8fe53954bf36e454c69`. Only five explicitly named regular members are read, all checked against their pinned byte counts and SHA-256 values. They total 215,700 bytes. No archive-wide extraction is used; no claim of verifying the untransferred whole archive is made. Verified cached members can be reused. Returned receipts exclude author-specific absolute filesystem paths.

## Define the perturbation before scoring

Two geometrically different tests are possible and must not share an ambiguous “tilt” label:

1. **Perturb sampling direction while preserving the surface.** For original surface point `S(u,v)`, unit normal `n`, and a unit tangent `t` orthogonal to `n`, sample `S(u,v) + d[cos(a)n + sin(a)t]`. Every zero-depth anchor and original label coordinate remains fixed. This measures sensitivity to the direction of the depth rays. At 20 degrees and depth eight native voxels, maximum lateral displacement is about 2.74 native voxels.
2. **Rotate the whole patch around its center.** Both the in-plane sampling locations and depth direction change. In the ideal flat-plane case, an original surface point at tangent coordinate `u` appears at rotated coordinates `q=u cos(a)` and `d=-u sin(a)`. Scores require a corresponding mapping back to the original physical points. Curved surfaces require an explicit geometric correspondence; the flat formula is insufficient. A 128-pixel patch at 20 degrees requires roughly 29.2 native voxels of normal context at its most distant corners, so an existing 16-plane stack is insufficient.

Those numerical bounds are direct geometric calculations from the recorded pitches, not empirical detector findings. Neither operator by itself demonstrates tolerance to arbitrary wrong-sheet sampling, tears, or unknown global registration errors. The selected operator should remain fixed across all signs and magnitudes, with one unchanged set of known ink/background pixels and fully observed CT support. Preserve quantization order and report any baseline mismatch before interpreting a score difference.
