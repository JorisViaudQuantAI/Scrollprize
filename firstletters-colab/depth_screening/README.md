# Three-depth maps for private review

This CPU helper answers the three-depth maximum question in Claude's next scan
batch. It combines **three already computed maps of the same mesh**, without
another model run. It retains the individual depth attribution and common valid
coverage. It does not decide that a candidate exists, certify a surface, or
reuse a single-depth score threshold.

**Implementation status:** eleven software tests pass, including partial coverage,
provenance mismatches and a synthetic example where three separate noise peaks
become one apparent row in a maximum. No matched three-depth public manuscript
corpus was available for this helper's validation. **No real-scroll sensitivity,
false-positive rate, detector gain or prize result is claimed.**

## Use

```bash
python firstletters-colab/depth_screening/combine.py \
  --manifest /private/finished-depth-run/inputs.json \
  --output-dir /private/finished-depth-run/depth-review
```

Only Python and NumPy are needed. The output directory must be new and outside
every Git worktree, including linked worktrees with a `.git` file. Wait for all
three producers to finish first. The helper also
checks that input hashes have not changed during combination. It never downloads
data or publishes outputs.

The manifest contains exactly three records:

```json
{
  "maps": [
    {"offset_planes": -1, "map": "minus.npy", "mask": "minus-valid.npy", "provenance": {}},
    {"offset_planes": 0, "map": "central.npy", "mask": "central-valid.npy", "provenance": {}},
    {"offset_planes": 1, "map": "plus.npy", "mask": "plus-valid.npy", "provenance": {}}
  ]
}
```

The empty provenance objects above are placeholders and deliberately fail
validation. Fill **each** from that map's actual inference receipt, rather than
copying one record across three unverified inputs. Map and mask paths may be
absolute or relative to the manifest. All maps must be nonempty floating-point
NPY arrays `[H,W]`; all masks must be boolean NPY arrays of exactly that shape.
The three offsets must be finite and distinct, with exactly one zero. They need
not be equally spaced. An offset is in units of the declared depth sample pitch,
along the declared normal convention.

Every provenance object must contain the following matching fields:

| Field | Required value or meaning |
| --- | --- |
| `checkpoint_sha256` | Full lowercase SHA-256 of the actual detector checkpoint. |
| `mesh_sha256` | Full SHA-256 identifying the same input mesh in all three runs. |
| `normal_sha256` | Full SHA-256 of the same normal field or its exact canonical artifact. |
| `normal_convention` | Nonempty description of normal construction and positive direction. |
| `plane_order` | Nonempty description of the same ordered within-slab input planes. |
| `xy_pitch_um` | Positive `[dy,dx]` sampling pitch in micrometres. |
| `depth_pitch_um` | Positive physical sample step used by `offset_planes`. |
| `frame` | Same acquisition and level-zero XYZ frame; fields below. |
| `preprocess` | Nonempty record of the same normalization and other preprocessing. |
| `inference` | Nonempty record of the same precision, tiling, blending, augmentation and other relevant inference settings. |

`frame` requires `acquisition`, `axes: "XYZ"`, `units: "native_voxels"`,
`level: 0`, positive `voxel_um_xyz: [vx,vy,vz]`, and finite
`origin_native_xyz: [x,y,z]`. Every shared value must match exactly across the
three records. A raw-versus-snapped mesh comparison therefore cannot be passed
off as a depth-only experiment. Use the separate
[geometry comparator](../geometry_review/README.md) for that question.

These are **declared-metadata consistency checks**. Input score/mask files are
hashed, but the helper does not open checkpoint weights, mesh TIFFs or normal
fields to establish that a provenance claim is true. Equal array shapes and
equal metadata do not certify anatomical correspondence or sheet identity.

### Existing inference CLI: preserve individual depths

**From a source — the repository's [infer_ink.py](../ink/infer_ink.py).** Its
`--z-shifts` values are expressed in **model planes**, and a multi-shift run
averages them before writing the single `--save-float` NPY output. Its
`--save-each` images use the model's display rescaling and PNG quantization.
They are not three original floating-point depth maps.

For this helper, supply separately retained individual-depth float maps. The
CLI now has an opt-in [`--export-components DIR`](../component_export/README.md)
that saves float32 maps for every checkpoint, direction and depth before
aggregation, without another model pass. Its `complete_depth_support.npy`
means prediction support with complete source-depth interpolation; it does
**not** establish acquisition validity, complete XY context or sheet identity.
Wait for its final receipt, verify its hashes, require complete depth context,
resolve any XY padding, and intersect with an external producer-supplied
acquisition/context support mask on the exact output grid. Unknown mesh,
normal, physical pitch or frame must keep this comparison blocked. See the
exporter's handoff checklist for the full distinction.

With an older CLI, one shift per `--save-float` run preserves the individual
depth, in float16. For another worker, inspect its actual output contract.
Record the effective physical step after depth resampling and the applied
normal convention. Do not automatically interpret a model-plane shift as one
native CT voxel or one rendered-stack plane. The helper checks declared
provenance but cannot detect an undocumented depth average or display
transformation from the score array alone.

## Outputs and missing coverage

All operations use the intersection of the three validity masks. A depth that
is absent at one pixel is not replaced by zero, ignored in a maximum, or borrowed
from another plane. If the intersection is empty, the operation fails. Nonfinite
values marked valid in any source also fail; NaNs in explicitly invalid samples
are allowed.

| File | Contents |
| --- | --- |
| `maximum.npy` | Per-pixel maximum of the three unscaled source scores, on common coverage only. |
| `median.npy` | Per-pixel median, on the same common coverage. |
| `central.npy` | Original zero-offset scores, restricted to the same common coverage. |
| `common_valid.npy` | Boolean intersection mask. |
| `winner_index.npy` | Input index `0`, `1` or `2`; `-1` outside common coverage. |
| `maximum_tie_count.npy` | Number of exact maximizers, `1–3`; zero outside common coverage. |
| `report.json` | Index-to-offset table, physical offsets, input/output hashes, coverage, per-depth descriptive comparisons and provenance. Written last on success. |

Float outputs contain **NaN** outside common coverage, never a fabricated zero
background. `winner_index` is an **input index**, not the physical offset itself:
the actual negative offset −1 must not be confused with the invalid sentinel −1.
The report supplies the index-to-plane-offset mapping and offsets in micrometres.

Exact ties prefer the central input, then the other input with the smaller
offset. The tie-count map is retained because this deterministic preference can
make a flat region look artificially central and coherent. It is not evidence
of a preferred depth. No intensity rescaling, nonlinear display adjustment,
thresholding, interpolation or new model inference occurs.

Per-depth comparisons include source-valid counts, score quantiles, mean and
mean absolute difference from the central map, correlation where defined, and
winning-pixel counts after the tie rule. These are descriptions, not measures
of accuracy. A maximum being greater than the central score is a mathematical
property, not a detector improvement.

## Interpretation for the prize search

**My inference.** A pixelwise maximum can assemble one apparent stroke from
different depths. Choosing one of three complete images and choosing separately
at every pixel are different searches. A three-way correction to a whole-image
score does not automatically calibrate the composite's false positives.

Review the three originals alongside maximum, median and winning-depth
attribution. Look for a stroke supported on an anatomically coherent surface;
the maximum alone is insufficient reading evidence. Coherent winning offsets
can support a hypothesis but cannot prove it, because correlated artifacts may
also be coherent. Median can suppress a feature visible at only one depth, so
it is not a truth image or a veto either.

Calibrate any downstream ranking or threshold through this **same complete
three-depth operation**, including the full proposal search, on matched public
positive, sparse and artifact controls. Freeze the offset set before viewing
candidate output. Do not reuse the single-depth null or dense-row threshold
unchanged. Keep generated maps, source manifests and location-bearing reports
private unless they are independently cleared public controls.

## Software checks

```bash
python -m unittest discover -s firstletters-colab/depth_screening -p 'test_*.py' -v
```

The eleven tests verify unchanged values, the common-coverage intersection, NaN and
sentinel behavior, source-index attribution, explicit ties, three-depth noise
accumulation, provenance mismatches, input typing, file hashes, no overwrite and
the prohibition on writing generated outputs inside this or another Git worktree. They verify
the helper's contract, not the behavior of an ink model on real manuscripts.
