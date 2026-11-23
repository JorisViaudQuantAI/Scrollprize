# Geometric normal transport through a complete point map

This standalone NumPy helper computes **target geometric sheet normals**. It uses
generated geometry only in its tests and does not load CT, meshes, registration
parameters, annotations or model outputs. It does not change another agent's code.

## Source finding and the integration question

**From a source:** at Gemini revision
`8f5a55d030497f834c0a8e2737444c06a4c97d44`, the
[point-mapping function](https://github.com/JorisViaudQuantAI/Scrollprize/blob/8f5a55d030497f834c0a8e2737444c06a4c97d44/gemini_lane/paris4_dual_energy.py)
implements

\[
F(x)=Ax+b+d_{TPS}(x).
\]

It adds the TPS displacement evaluated at the input point to the affine output.
This is not generally the same as \(A(x+d_{TPS}(x))+b\).
The [surface sampler](https://github.com/JorisViaudQuantAI/Scrollprize/blob/8f5a55d030497f834c0a8e2737444c06a4c97d44/gemini_lane/paris4_crust_1129.py)
computes `e_n_tgt = normalize(A n)` with `apply_affine_vectors`, then uses that
direction to sample its slab. It does not differentiate the nonlinear residual.
Source URLs, SHA256 pins, function locations and qualifications are retained in
`source_audit.json`.

**Derived geometry:** two different contracts must be distinguished before any
integration:

| Intended quantity | Correct local expression |
|---|---|
| Normal perpendicular to the transformed sheet | \(J_F(x)^{-T}n\), then normalise |
| Transport of an existing sampling ray or tangent | \(J_F(x)v\); normalise only if a unit direction is wanted |

For an oriented surface with tangent vectors \(u,v\),

\[
(Ju)\times(Jv)=\det(J)J^{-T}(u\times v).
\]

Thus the inverse-transpose formula is the cofactor normal after normalisation
when the determinant is positive. The implementation solves the transpose linear
system rather than forming an explicit inverse. It never silently changes faces.

For a positive similarity without a spatially varying nonlinear residual,
`normalize(A n)` is directionally correct. General anisotropic scale or shear
requires the inverse transpose for a geometric normal; a nonlinear map requires
the derivative of the complete map. **We have not measured the actual affine's
departure from a similarity, any actual TPS derivative, or a manuscript angular
error.** Correctly differentiating a noisy fitted map does not establish that the
map represents the true physical deformation. This package demonstrates geometry
correctness, not an actual scroll failure, a reading or an ink gain.

## API and units

```python
from normal_transport import Settings, transport_normals

result = transport_normals(
    complete_physical_point_map,
    points_um,                  # float64 (N,3), physical Cartesian micrometres
    source_physical_normals,     # (N,3), dimensionless; normalisation is internal
    settings=Settings(step_um=0.01),
)
target_sheet_normals = result.normals
```

`point_map` must be deterministic, pointwise and side-effect free. It receives
float64 `(B,3)` batches and returns matching float64-or-better arrays. Both frames
must have consistently handed, orthonormal Cartesian axes in micrometres.
Differentiating the supplied full callable avoids assuming a composition order.

For a voxel-index map with potentially anisotropic pitches, the caller must wrap
both scales explicitly:

```python
def complete_physical_point_map(x_um):
    source_indices = x_um / source_pitch_um_xyz
    target_indices = complete_voxel_point_map(source_indices)
    return target_indices * target_pitch_um_xyz
```

Construct the source normal from surface tangents in the **physical** source
frame. A normal calculated in anisotropic index coordinates cannot simply be used
as a physical normal or converted as though it were a tangent vector. The helper
cannot infer units, coordinate handedness or physical registration quality from
numeric arrays.

## Checks and limits

Each point is evaluated at its centre and at ± each coordinate axis with steps
`h`, `h/2`, `h/4`: **19 mapped-point evaluations per point**. The finest central
Jacobian is retained only if:

- Actual positive and negative floating-point coordinate increments match their
  nominal steps within the declared relative tolerance, default `1e-8`.
- Jacobian step differences are small and decrease unless already within the
  requested tolerance. One-sided derivative gaps also decrease, rejecting a
  generated cusp whose central differences alone falsely appear stable.
- All three estimates have finite values, positive determinant, minimum singular
  value at least `1e-8`, and condition number at most `1e6` by default.
- The two finest transported normals differ by at most `0.001` degrees by default.

Jacobian tolerances default to absolute `1e-9` plus relative `1e-6` in Frobenius
norm. These are explicit numerical policies, not measured registration errors or
guaranteed bounds on the true derivative. No finite set of samples can exclude
all unresolved oscillations or folds between samples. Negative determinants are
rejected conservatively, including deliberate reflections; consistent physical
coordinate handedness is a prerequisite.

The helper rejects the entire request if any point fails. It does not return a
mask that could silently remove inconvenient locations. Defaults cap the request
at 4,096 points, use batches of 256, and impose a cooperative ten-second deadline.
The timer is checked around each map call and final calculation; it cannot preempt
an unresponsive callback or bound the callback's own memory allocation.

## Generated verification

The tests compare returned normals with **cross products of independently
transformed surface tangents**, plus analytic Jacobians for a smooth nonlinear
map. They cover rigid/uniform scaling, anisotropic scaling, shear, actual versus
alternative displacement composition, anisotropic voxel-unit wrappers, a resolved
versus under-resolved difference step, a cusp, amplified angular error, a singular
map, reversed orientation, a fold in one batch member, conditioning and nonfinite
data. They also retain the independent reviewer's regression: tiny but nonzero
steps at a large coordinate must not falsely rotate a normal under the identity
map.

Run the generated tests and compact angular-error demonstration with one thread:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python firstletters-colab/normal_transport/generated_review.py \
  --out /tmp/scrollprize-normal-transport-v1/result.json
```

The receipt runner refuses to write inside Git or overwrite an existing receipt.
It records code hashes, NumPy/Python versions, test results, angular errors and
resource use, and requires completion within ten seconds and 256 MiB peak RSS.
It performs no network requests, CT reads or model calls.

A generated shear example transforms a plane with source normal `(0,0,1)` by
`A = [[2,.5,1],[0,1,0],[0,0,.5]]`. Its correct target normal remains `(0,0,1)`;
the old vector expression is proportional to `(1,0,.5)`, an analytically derived
angular error of **arctan(2) = 63.4349488 degrees**. This deliberately constructed
example establishes why the two contracts differ. It says nothing about the
magnitude of an error on any real registration.

## Next integration step

First state whether a downstream sampler needs a target sheet normal or a
transported source ray, and establish its source/target physical coordinate
conventions. Only then can a separate, bounded geometry diagnostic inspect a
cached public transform. There is no automatic integration into Gemini's pipeline
and no automatic acquisition or inference step in this package.
