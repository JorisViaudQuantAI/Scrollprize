# Portable metric correction: public 306B geometry control

**Status before acquisition:** implementation and protocol, with no result yet.
Results will be added in separate files after the immutable source/protocol
commit. Nothing here claims a new reading or better ink detection.

**From a source:** [Claude 017](../../collab/claude-017.md) reports that equal-angle
winding charts have substantial local scale variation, while independent
per-row arc-length correction can introduce shear. The published PHerc. 306B
submission includes a real pre-flattening surface construction, an ARAP
correction, and its released output: [pinned source directory](https://github.com/SirProdigle/pherc0306b-first-letters/tree/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/code/surface).

**My inference:** this provides a small, executable geometry control for a
correction Claude can use before spending GPU time on another winding. We use
the existing community optimizer, and add portable input handling, explicit
geometry checks and an inverse sampler with original-triangle provenance.

**From the latest team source:** [Claude 018](../../collab/claude-018.md) and
the [methods ledger](../methods_ledger/09_open_questions.md) name this issue as
Q10, with coverage-wide distortion measurement as Q49. This control addresses
the correction and measurement part of those questions; it does not measure
detector tolerance or every unread winding. Claude is already running Q7 and
Q1, so this experiment does not duplicate those jobs.

## What the experiment compares

| Arm | Geometry | Inverse sampling |
| --- | --- | --- |
| A | Reconstructed published touch-up, before flattening | Original carrier nodes |
| B | Published 60-iteration ARAP settings | Published `LinearNDInterpolator` |
| C | Exactly the same saved UV solution as B | Barycentric interpolation in original source triangles |

**From a source:** upstream normalizes XYZ by 20 voxels, uses 8.64 µm source
voxels, weights unusually stretched triangles by 0.02, and aligns the corrected
chart by rotation and translation without scale. Source, fixed public regions,
input hashes and limits are recorded in [source_manifest.json](source_manifest.json)
and [protocol.json](protocol.json). The pinned
[reflatten.py](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/code/surface/reflatten.py)
and [build_surface.sh](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/code/surface/build_surface.sh)
define the exact calculation.

**Design choice:** reproduce B against the decoded published XYZ first: equal
support and maximum absolute coordinate error at most **0.01 source voxel**.
Failure stops the experiment before C. Passing that check does not establish
the papyrus surface's physical accuracy.

**Design choices, not statistical confidence bounds:** C must retain at least
99% of B's support overall and 99.5% in the published claim crop. On common
four-corner cells in the publisher's right-hand group, it must halve the median
absolute vertical stretch error, leave at most 5% of cells outside ±5% in either
axis, and have 95th-percentile absolute shear at most 5°. Full source-triangle
metrics retain all difficult or reversed faces. No gate changes after seeing
the result.

The published 82 × 246-node claim-region crop includes 2 mm of context; it is
not the exact qualifying claim rectangle. The metric diagnostics use the
publisher's three-corner formula on decoded float32 arrays; they are not a
direct replay of its pre-serialization JSON summaries.

The control is **retrospective development**: the published defect and summary
before/after measurements were read before choosing this protocol. It is not
an independent test of a learned method. Fixed target-grid cells before and
after are not exact material correspondences; the full original-triangle
population provides a complementary measurement without changing triangles.

## Original-triangle sampler and limits

`topology.py` never constructs a new triangulation. Each accepted integer
target query has a source triangle ID, barycentric weights, and reconstructed
XYZ. Negative/degenerate UV coverage invalidates a query. Incompatible multiple
coverage invalidates a query even if two surfaces happen to have the same XYZ.
Shared original edges/vertices are accepted only with consistent coordinates.
Every source component and rejected face remains counted.

**My inference from the algorithms:** constructing a new Delaunay triangulation
can connect vertices across a missing piece of source support. An explicit
7 × 7 vertex removal around the published fixed vertex tests this on the real
public geometry. This is a **designed missing-support test**, not observed
manuscript damage and not evidence that the released 306B surface is wrong.

**Limits:** original-triangle provenance at integer samples does not prove
global injectivity between samples. A regular-grid renderer can still bridge
a subpixel slit or seam between two accepted nodes. Consumers must validate
their rendering cells/edges against source support and must honor invalid
regions. The XYZ files alone do not enforce this contract. Do not hand the
output to a renderer that silently fills missing support.

The optimizer's fold weighting does not prevent UV flips. Their occurrence and
sample loss are measured, not repaired or discarded from the metrics. Single
connected patches without isolated vertices are required; there is no implicit
largest-component selection. Physical face and surface correctness remain
unestablished. Generic input handling treats any `-1` coordinate as invalid,
slightly stricter than upstream's `x == -1` policy; the accepted replay must
still match upstream's result.

## Run the frozen control

Dependencies: Python 3, NumPy, SciPy, Pillow. No PyTorch, GPU, tifffile, OpenCV,
CT download or ink-model execution is required. The script requests four CPU
threads, permits 900 seconds and a 2 GiB peak-RSS budget checked at stage
boundaries, and bounds the problem at 100,000 vertices and 200,000 triangles.
The wall alarm and RSS stage checks are soft process limits, not OS isolation.

The 17 pinned assets total **14,205,351 bytes**, with a 16,000,000-byte cap,
20-request cap, no retries and no redirects. Actual transfers are recorded.
Caches and outputs must be outside Git, with existing parent directories and
fresh target names. The script verifies that every implementation/protocol file
matches the supplied immutable commit before source acquisition.

```bash
python firstletters-colab/metric_flatten/experiment.py \
  --freeze-commit FULL_PROTOCOL_COMMIT_SHA \
  --cache /tmp/metric-flatten-source \
  --out /tmp/metric-flatten-control
```

To repeat from verified cached inputs without a network request:

```bash
python firstletters-colab/metric_flatten/experiment.py \
  --freeze-commit FULL_PROTOCOL_COMMIT_SHA \
  --cache /tmp/metric-flatten-source \
  --out /tmp/metric-flatten-replay \
  --reuse-cache
```

Inspect `result.json`, including every gate. It records consumed source/code
hashes, numerical differences, all support losses, runtime and artifacts. A
failed run remains a failed run. Output float32 rounding is reported separately
from the 1e-6-voxel float64 barycentric reconstruction tolerance.

Generated geometry edge cases can be checked before the control:

```bash
python -m unittest discover -s firstletters-colab/metric_flatten -p 'test_*.py'
```

## Handoff to Claude Code or Gemini

If all geometry gates pass, the next authorized step is a **single bounded
comparison on already-published known text** with identical CT input, depth,
model and display policy. Compare A and C, honor source support, and record
both reading quality and physical distortion. The geometry result alone cannot
justify an unread-scroll survey or claim increased detection accuracy.

For a small, already-cropped connected winding patch, the portable command is:

```bash
python firstletters-colab/metric_flatten/flatten.py \
  --mesh /tmp/input-patch-tifxyz \
  --out /tmp/corrected-patch-tifxyz \
  --upstream-cache /tmp/metric-flatten-source \
  --spacing-voxels 20 \
  --voxel-um SOURCE_VOXEL_SIZE
```

Use the actual input scale and source voxel size. They are verified against
metadata when declared, allowing only float32 serialization error. The generic
CLI uses a central rectangular anchor; the frozen published control instead
uses the exact published anchor and fixed vertex. The CLI has a 600-second
wall alarm and performs no downloads. It creates geometry for controlled
review; it does not automatically pass the experiment's acceptance gates.

Outputs are `x.tif`, `y.tif`, `z.tif`, `mask.tif`, `meta.json`, and
`provenance.npz`, with input/code hashes, original XYZ/triangles, optimized UV,
source triangle IDs, barycentric weights and rejection bits. Invalid XYZ is
`[-1,-1,-1]`. Keep any unread-scroll outputs and their locations private.

## Rules, attribution and claims

**From a source:** the [First Letters rules](https://scrollprize.org/prizes)
require meshes with a low-distortion isometric flattening and an ink image
programmatically derived from the original CT; training-data overlap with the
predicted region is prohibited. **My inference:** this geometry-only public
control adds no training overlap, but does not resolve exposure of any future
ink model or satisfy the reading requirement.

The source code is MIT-licensed; its notice is preserved in
[LICENSE.upstream](LICENSE.upstream). The publisher identifies the data as
CC BY-NC 4.0 in the [source repository](https://github.com/SirProdigle/pherc0306b-first-letters/tree/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc).
Raw arrays and original metadata remain outside this repository. Source
adaptation omits historical local paths and plotting, preserves numerical
arithmetic and float32 rounding, and verifies every byte before parsing.

## Open questions

- **Claude Code:** can the current renderer enforce original-support cells and
  edges, as well as invalid-node masks, on one published control?
- **Gemini:** does a matched public-text comparison show a reading benefit after
  geometric correction, with the same CT and model settings?
- **Codex:** can the triangle-provenance contract be preserved through the
  renderer's interpolation and normal estimation?

## Next actions

- **ChatGPT:** freeze these files, run the declared control, publish the measured
  result and preserve every failed gate.
- **Claude Code:** use the resulting geometry decision to choose one public
  rendering comparison; no additional broad survey is requested here.
- **Owner:** decide whether any later validated result merits an organiser
  submission. No submission or prize qualification is claimed by this work.
