# PHerc. 0813 atlas winding and native CT

This single-CPU replay compares meshed positions before and after an m7 surface snap on one public atlas winding. It reads the original CT for an anatomical check. The official volume inventory lists the masked Zarr and VCD3D encodings of acquisition `20250821151723`, at 9.362 µm and 113 keV. Both share an acquisition ID. The public transforms prefix is empty. The inter-scan calibration for PHerc. 1203 and PHerc. 0846A therefore has no matching scan pair or calibration points here. [Scan catalog](https://scrollprize.org/data_browser/PHerc0813) · [Public S3 volume inventory](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/?list-type=2&prefix=PHerc0813%2Fvolumes%2F&delimiter=%2F)

The input is the lexicographically first winding in Paulo Camillo's PHerc. 0813 atlas, `z11296_w020`. Its 5,110 valid vertices were published before this test. We run Erwin Nieuwlaar's original PHerc. 343 snapping script, unchanged, using the native m7 volume of the same acquisition. The defaults search ±15 voxels along each mesh normal, reject 5 × 5 median jumps above 6 voxels and smooth the displacement field with sigma 1 grid cell. The code and mesh files are verified against pinned SHA-256 digests. Both source repositories are MIT-licensed. [Input winding](https://github.com/pscamillo/vesuvius-eligible-meshes/tree/66000516c97e043b696ce874f145e29638109c91/meshes/PHerc0813/z11296_w020) · [Upstream snap](https://github.com/Nieuwlaar/pherc343-first-letters/blob/e8370f1f88a211166056df62ea6fcfffe4a1063e/mesh/snap_meshes.py)

The test selects 24 disjoint neighborhoods by mesh geometry before reading m7 or CT intensity. The primary measure is each vertex's distance to the same native CT crest nearest its original position. A secondary measure permits a different nearest crest after snapping. It records how often that changes the answer. A CT intensity maximum can reflect fibers, a neighbouring interface or the target sheet. The m7 prediction also derives from this CT. These measurements assess local placement and do not establish the identity of a sheet or the presence of ink. [Frozen protocol](protocol.txt)

From the repository root on Linux, use Python 3.12 and one CPU worker.

```bash
python -m pip install numpy==2.1.3 scipy==1.15.3 requests==2.32.5 tifffile==2025.5.10 zarr==2.18.7 numcodecs==0.15.1
python registration-checks/atlas-0813/check.py
```

The script reads only the m7 chunks intersecting the winding's normal rays and the native CT byte ranges containing the 24 evaluation neighborhoods. It places the full reconstructed mesh in `registration-checks/atlas-0813/snapped/z11296_w020/` and writes aggregate metrics to `result.json` in the same study directory. Source receipts and per-site diagnostics are available in that local workspace.
