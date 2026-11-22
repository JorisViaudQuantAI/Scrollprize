# Producer-verified component handoff to three-depth review

This small CPU adapter connects an existing **completed** `infer_ink.py --export-components` run to `depth_screening/combine.py` **without rerunning any detector**. It generates the three validity masks on the original source grid, a directly consumable `inputs.json` and an integrity `handoff.json`. It does not convert clipped PNGs into probabilities, select an ink depth, rank a candidate, or infer a scan's geometry.

**Status (22 November 2026):** Code and 12 synthetic contract/regression tests were executed locally; all passed (NumPy 2.3.5, CPython). No actual acquisition/context mask with verified surface geometry has yet been received for the earlier tiny public Hecate smoke, so no valid real-scroll depth combination was run. That is an explicit blocked validation gate, not a positive result.

## Required external producer data

Supply a boolean `acquisition_context.npy` **on the exact export output grid**. Each `true` must mean every required source CT sample and spatial model input context for this rendered pixel was genuinely acquired/available under the declared producer and sampling convention, not just that one CT value is nonzero. Keep sheet/tissue validity separate if needed; intersect it before supplying this mask. Neither nonzero CT, an outline, nor `prediction_support.npy` by itself is adequate.

Supply a JSON declaration, filled from the **actual rendering/CT producer records**, not from a filename or guessed normal. The fields below show structure only: repeated digits and example descriptions are placeholders and are **not** valid evidence of real geometry.

```json
{
  "schema": "scrollprize.producer-context-support.v1",
  "support_kind": "acquisition-and-model-context",
  "producer": "actual renderer and acquisition context pipeline revision",
  "definition": "exact per-pixel input-plane and spatial-context coverage algorithm",
  "mask_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "mesh_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "normal_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "normal_convention": "actual native-XYZ normal construction and positive direction",
  "plane_order": "actual ordered source depth planes and normal orientation",
  "source_content_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "frame": {
    "acquisition": "actual scan ID",
    "axes": "XYZ", "units": "native_voxels", "level": 0,
    "voxel_um_xyz": [8.64, 8.64, 8.64],
    "origin_native_xyz": [0, 0, 0]
  },
  "shift_to_normal_sign": 1
}
```

`mesh_sha256`, `normal_sha256`, `source_content_sha256`, `normal_convention`, `plane_order`, and `frame` must **exactly match** the geometry declarations already retained in *each* component. The operator must establish these when starting inference; the adapter never overwrites a null declaration. `shift_to_normal_sign` (+1 or −1) is the **producer-established** mapping from increasing model-plane shift to the physical positive normal for the selected direction. The adapter does not guess it; the direction is selected explicitly. A copied declaration or all-true support mask can be syntactically valid but scientifically false: audits must trace their real producer source.

## Commands

From repository root, after a complete inference component receipt and source mask exist, using a **new private directory outside Git**:

```bash
python firstletters-colab/component_export/prepare_handoff.py \
  --export-dir /private/finished-inference/components \
  --acquisition-context-mask /private/producer/acquisition_context.npy \
  --producer-declaration /private/producer/context.json \
  --model-index 0 --direction forward \
  --output-dir /private/finished-inference/depth-handoff

python firstletters-colab/depth_screening/combine.py \
  --manifest /private/finished-inference/depth-handoff/inputs.json \
  --output-dir /private/finished-inference/depth-comparison
```

The inference run must contain **exactly three distinct physical depth offsets including zero** for the selected checkpoint and direction. This is the depth helper's contract, not a requirement to change Claude's existing **two-shift** production plan. If a production run has only −3/+3, keep both individual float maps and their declared mean separately; do not forge a zero-depth third component or pass the mean as if it were independent.

The adapter hashes **every completed component payload and its metadata** against the run receipt, checks the selected checkpoint identity and shared resampling/tiling parameters, rejects incomplete source interpolation, unresolved XY padding, unknown physical pitch or mismatched source geometry, and intersects each `complete_depth_support.npy` with the external acquisition/context mask. It verifies source bytes again before writing, and again before the final handoff receipt. `inputs.json` refers to the existing float32 score files (not copies); downstream reads and hashes those files again. Masks are newly saved `.npy` boolean files. A run cannot overwrite its output directory.

`handoff.json` is written last, with input hashes, source depth indices, physical sign, per-depth/common support counts and saved-output hashes. If it is missing, the handoff is incomplete. The external support declaration and mesh/normal identifiers are **assertions**, not independently authenticated geometry; the code checks structure and byte identity. Human review must still inspect the three originals, median, central and winning-depth map before interpreting any maximum. Existing model-score thresholds are **not** calibrated to a three-depth maximum.

## Validation and constraints

```bash
python -m unittest discover -s firstletters-colab/component_export -p 'test_prepare_handoff.py' -v
```

The 12 synthetic tests cover successful three-depth handoff, mask intersections, physical sign, source-hash mismatches, missing geometry, incomplete run and depth, bad XY context, wrong mask kind, output non-overwrite, Git-worktree exclusion and tampered component content. None is a detector-accuracy test or proof that real source geometry is right. The existing public Hecate smoke deliberately has unknown mesh, normal, frame and external context support and must continue to fail this gate until a true producer supplies them. A three-depth **public-positive** real run, with ground-truth geometry/masks, is still required for end-to-end evidence and measured inference-to-review usefulness.
