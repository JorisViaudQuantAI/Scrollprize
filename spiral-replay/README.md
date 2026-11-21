# PHerc. 0358 spiral geometry replay

This experiment uses public anatomy from the 9.362 µm scan `20250821151737`. The requested band is `9200 <= z < 10700` in native voxels, a height of 14.043 mm. The replay samples whole transverse M7 sections around the manually annotated umbilicus. No ink detector is involved.

## CPU replay

Use Python 3.11 or newer with NumPy, SciPy, PyTorch and einops. The source manifest contains six axial sections selected before fitting. The default repeats the three that were available in the original CPU test. It transfers 10,423,240 bytes of coordinate and offset ranges, about 10 MiB. The `six` profile requests all six and has separate training and evaluation sections.

```sh
cd spiral-replay
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 FIT_SPIRAL_NUM_THREADS=1
python stage.py --profile replay
python arc_probe.py
python model_probe.py --device cpu --steps 40 --allow-minimal-cpu-imports
python t4_trial.py
```

The stage checks each retrieved Villa file against its Git blob and each replay section against the recorded coordinate and offset hashes. Existing original files or indexed byte ranges can be reused via `stage.py --index-cache DIRECTORY`. Both probes run on held-out *axial sections*. Each section is sampled in eight angular sectors, with up to 32 polylines per sector and 24 vertices per polyline. Adjacent sections may include the same physical sheet.

The first probe runs upstream angle/radius and umbilicus code and fits a two-parameter affine deformation. The second calls the actual `SpiralAndTransform` class on a coarse Cartesian lattice with the upstream stationary RK4 integrator. It updates only the velocity field. If Pyro or torchdiffeq is absent, the CPU option supplies PyTorch's native distribution transform classes and guards the unused time-varying ODE entry point with an exception. The normal CUDA environment uses both actual dependencies.

The CPU test prepared a standard DBM of 233 training arcs and 48,005 vertices from one section. It reserved 483 arcs from two other sections. After 40 coarse field updates, the median per-arc mean absolute deviation of shifted radius across the reserved arcs changed from 37.19 to 36.40 canonical voxels, with P90 from 90.62 to 89.58. The two-parameter affine trial gave 37.65 to 37.62 and P90 from 91.05 to 95.93. Deviations are measured about each track's own median shifted radius. The objective gives no absolute winding labels and permits local compression. It cannot establish a continuous sheet, assess ink or determine the scroll's handedness. The 40-step single-thread geometry process peaked at roughly 395 MiB host RSS on CPU PyTorch 2.9.1.

## CUDA handoff

The pinned Villa revision places the full fitter at `volume-cartographer/scripts/spiral/fit_spiral.py`. Its project declares Python 3.14 or newer and CUDA-capable PyTorch. In an existing compatible GPU environment, verify that PyTorch can launch an operation on the T4 before starting the fixture. We measured no T4 memory or timing.

```sh
git clone --filter=blob:none https://github.com/ScrollPrize/villa.git villa
git -C villa checkout 0fb38c45b488e192566ea83ae36369cb6db92652
VILLA_SPIRAL="$(pwd)/villa/volume-cartographer/scripts/spiral"
cd "$VILLA_SPIRAL"
uv sync
SPIRAL_PY="$VILLA_SPIRAL/.venv/bin/python"
cd - >/dev/null
"$SPIRAL_PY" -c 'import torch; a=torch.ones(1,device="cuda"); print(torch.cuda.get_device_name(),a.item())'
"$SPIRAL_PY" stage.py --upstream "$VILLA_SPIRAL" --profile replay
"$SPIRAL_PY" model_probe.py --device cuda --steps 40
"$SPIRAL_PY" t4_trial.py --villa "$VILLA_SPIRAL" --sense CW --steps 3 --run-cuda
```

The last command prepares the small DBM on the host, starts the official `FitContext`, runs three optimizer steps, saves `checkpoint_smoke.ckpt`, and writes measured allocated/reserved VRAM, elapsed seconds and losses in `t4-output/measure-CW.json`. It uses a 192-voxel flow lattice, one RK4 integration step, 32 tracks × 24 points per step and no crossings on these transverse-only examples. `CW` is a trial setting. The public umbilicus describes the axis and carries no outward-sense annotation. Pass `--sense ACW` for a separate trial. Each invocation starts a new trial; copy any existing measurement for the same sense before repeating.

The pinned headless CLI fills `outer_shell` with the conventional path even when the directory is absent. Its loader attempts to open a shell whenever both the tracks and shell paths are truthy, even if `loss_weight_shell_outer=0`. The fixture preparation reproduces this path-resolution condition on CPU. The CUDA wrapper supplies an explicit empty `SpiralInputPaths.outer_shell` to the official library entry points. The public three-step driver has been verified only through input preparation on CPU. CUDA training, checkpoint creation and VRAM reporting by this driver await a GPU run.

## Inputs for a full band fit

- **Tracks.** The packed M7 directory contains 12,256,170 whole-volume polylines. Of these, 828,563 horizontal and 1,066,908 vertical tracks lie entirely inside the band. For a production fit use both families and matching exact crossing identities. The packed coordinate file is 5,637,581,208 bytes, the source DBM is 5,977,288,704 bytes and the published crossings file is 1,496,163,972 bytes. Rebuilding a reduced DBM changes its source keys, so its companion crossings must be recomputed. The original fitter checks DBM signatures that include file name, size and modification time. [Public tracks and extraction manifest](https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0358/20250821151737/tracks/)
- **Umbilicus.** The official annotation has 76 manual control points in full-resolution XYZ coordinates. The upstream loader sorts by z and returns interpolated y,x; use scale 1.0. [Official umbilicus](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0358/representations/umbilicus/20250821151737-umbilicus-20260922235631.json)
- **Normals.** The published Lasagna `nx`, `ny` and `grad_mag` Zarr stores begin at group `2`, 4 native voxels per stored voxel. Group `2` has a ZYX shape of 3686 × 1946 × 1946 with 32³ uint8 chunks. The official fitter expects locally packed resident pools. With these three Zarr stores already staged, run `python pack_resident_pools.py /path/to/lasagna-inputs --what normals,grad_mag --normal-group 2 --io-threads 1` in the Villa directory, then add `--normal-root /path/to/lasagna-inputs` to the fixture command. This enables the gradient-magnitude spacing mode with normal supervision. The default phase mode expects an additional surface signed-distance field. We verified the public metadata and training predicates. Full-volume normal staging and the dense CUDA losses remain untested here. [Official Lasagna catalog](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0358/representations/predictions/lasagna/20250821151737-lasagna-20260419180421/PHerc0358.lasagna.json)
- **Full fit and physical validation.** A fit using the complete registered stores, three-dimensional crossings, normal fields and an independently determined outward sense must be validated against native CT before exporting, snapping or unwrapping any continuous winding. The 233-track fixture measures software feasibility with low data volume. [Pinned Villa fitter and host instructions](https://github.com/ScrollPrize/villa/tree/0fb38c45b488e192566ea83ae36369cb6db92652/volume-cartographer/scripts/spiral)
