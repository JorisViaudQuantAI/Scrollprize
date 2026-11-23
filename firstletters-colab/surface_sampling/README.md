# Original-surface footprint rendering — CPU experiment

**Proposed intervention, not an established ink improvement:** sample the
original surface triangles at every output query, require coverage of each
whole UV pixel, then integrate its mapped footprint with fixed midpoint
quadrature. This changes the renderer itself. There is no learned model,
image enhancement, depth averaging or filled geometry.

The experiment is frozen before accessing new NIST CT pixels. Results belong
in a separate `RESULTS.md`; the protocol and this description remain unchanged.

## Why this is a different experiment

**From repository sources:** [ChatGPT 030](../../collab/chatgpt-030.md) found a
source-support failure in a PHerc. 306B mesh replay. Its 99% retention gate
failed. [ChatGPT 031](../../collab/chatgpt-031.md) independently replicated
Gemini's depth-profile features; their promotion gates also failed.

**From Gemini's current source:** the
[80a2afb branch revision](https://github.com/JorisViaudQuantAI/Scrollprize/tree/80a2afb/gemini_lane)
already contains cross-scroll depth features, a cloud-masked TFCE comparison,
seismic-coordinate preflight and a prepared GPU radiometry/orientation lane.
The present experiment repeats none of those numerical interventions.

**From the literature:** [EWA Volume Splatting](https://www.merl.com/publications/TR2001-27)
combines reconstruction and low-pass filtering for volume rendering.
[Thin-Volume Visualization on Curved Domains](https://doi.org/10.1111/cgf.14296)
uses a guiding triangle mesh to inspect curved structures through transformed
volume views. **My inference:** rendering theory gives us a useful intervention
upstream of ink inference, where geometric distortion changes the physical
region represented by one output pixel. **Speculation:** reducing sampling
artefacts might help a frozen ink model. That speculation is not tested here.

This implementation is **uniform UV-area midpoint quadrature through a
piecewise affine source surface**. It is not Gaussian EWA and does not weight
samples by physical-area Jacobians. Face normals are unsmoothed and may jump
across source edges. Surface winding does not establish the ink-bearing face.

## Contracts that the implementation enforces

1. XYZ, unit normals and derivatives come from the original source triangles.
   There is no new Delaunay triangulation or interpolation through missing
   output nodes. Shared boundaries choose the smallest original face ID.
2. Each whole UV pixel is clipped against candidate source triangles. Full
   covered area is required; pairwise overlaps and inconsistent source
   boundaries are rejected. Negative or degenerate faces poison the pixel.
   A lower bound on union area subtracts all pair-intersection areas, so many
   individually tiny overlaps cannot compensate a larger uncovered hole.
   This uses float64 with explicit tolerances, not exact arithmetic.
3. Finite quadrature points alone are insufficient: a narrow slit can lie
   between all sixteen points of a 4×4 rule. The separate continuous-coverage
   check rejects that case before rendering.
4. Every quadrature point at **every requested depth** must have all eight CT
   interpolation vertices. No coordinate clamping, partial-pixel
   renormalization or missing-data filling is allowed. Rejected pixels are
   zero with an explicit mask; zero does not mean blank papyrus.
5. Depths remain separate and in the requested order. Float64 accumulation
   produces float32 output once. CT units are not automatically rescaled.

These checks do not establish anatomical sheet identity, normal-ray
injectivity or continuous CT support between quadrature points. A physically
wrong but topologically valid mesh can still pass.

## Two distinct controls

### Previously examined PHerc. 306B geometry

**From the cached, published replay:** the source has 77,185 vertices and
153,224 original triangles. The experiment verifies its fixed input hashes,
integer-grid XYZ/face/support parity and two fixed 32×32 UV patches. This is a
retrospective engineering test, with no new CT or ink claim. The old failed
retention gate remains failed.

Input arrays are reproducible with the pinned
[metric-flatten experiment](../metric_flatten/README.md). The selected public
winning source is
[PHerc. 306B at feabff6c54f95bd8e52a90e0a7a76881ee67d4cc](https://github.com/SirProdigle/pherc0306b-first-letters/tree/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc).

### New NIST modern physical phantom

**From [NIST mds2-4227](https://doi.org/10.18434/mds2-4227), its pinned README
and the [associated paper](https://doi.org/10.1371/journal.pone.0353485):** this
is a modern carbonized-papyrus experiment with deliberately lead-containing
ink. The released reconstruction has nominal 130 µm sampling. It provides a
physical volume for rendering tests, not an ancient carbon-ink control.

The source reader selects exactly released slices `[246,374)` from the middle
of the archive, without inspecting their pixels. It retrieves individual ZIP
members through exact HTTP ranges and verifies the pinned directory, ETag,
local headers, CRC32 and decoded float32 data. The whole ZIP's declared SHA-256
is retained but **cannot be verified by partial acquisition**.

The actual experiment uses two charts fixed by volume dimensions alone:
an affine oblique chart and a curved chart, each at four UV phases. These are
**sampling fixtures, not recovered papyrus surfaces**. Each case has 64×64
pixels and depths −2, 0 and +2 native voxels. The arms are one central sample,
4×4 midpoints and 8×8 midpoints. A fixed central 8×8-pixel subset also uses a
composite 16×16 numerical reference. No optical labels or text masks are used.
This composite uses the same renderer and averages four half-cell float32
outputs; it is not an independent implementation or a once-rounded 256-sample
sum. Tiny roundoff-level differences must not be interpreted as efficacy.

**Predeclared numerical gate:** preserve at least 99% common fixture support;
reduce RMSE against the 8×8 rule by at least half in every one of the eight
cases; and show ordered 1→4→8 convergence against the composite 16×16 rule on
all eight fixed central subsets. This tests numerical integration only. The
dense rules are not continuous ground truth, and smoother images alone do
not show improved ink detection.

The run is bounded to one CPU thread, 1,200 seconds, 2 GiB peak RSS checked at
stage boundaries, 96 MiB charged response bodies and 300 charged requests.
The acquisition itself uses 259 instrumented requests with no retries or
redirects. Historical metadata preflight did not instrument every redirect
hop; the manifest reserves an allowance and states that limitation rather
than claiming an exact all-session wire count. GPU forwards and training
updates are zero.

## Run and reuse

Generated tests use no manuscript or NIST pixels:

```bash
python -m unittest discover -s firstletters-colab/surface_sampling -p 'test_*.py'
```

Run the frozen experiment only after the public commit exists:

```bash
python firstletters-colab/surface_sampling/experiment.py \
  --freeze-commit FULL_PUBLIC_COMMIT_SHA \
  --geometry-cache /tmp/previous-public-geometry-replay \
  --source-cache /tmp/new-nist-source-cache \
  --out /tmp/new-surface-render-result
```

The source cache and output directory must be new, outside Git, and under an
existing parent. A failed acquisition is retained with its receipt. There is
no automatic retry, replacement field or threshold adjustment.

The standalone `cli.py --help` exposes the same renderer for a local float
volume and an NPZ containing `uv`, `xyz`, `triangles`. It writes float32 TIFF
layers, a validity mask and a reproducibility report outside Git. An explicit,
separately validated radiometric conversion is required before feeding these
raw-unit TIFFs to a model expecting another numeric encoding. This CLI is an
opt-in component; the existing production renderer is unchanged.

## Attribution and publication scope

**From the dataset-linked [NIST terms](https://www.nist.gov/open/license):**
applicable NIST data may be modified and distributed with acknowledgement and
a notice of changes; the source is supplied as is. We do not assert blanket
worldwide public-domain status for every NIST-hosted dataset. The manifest
records the actual linked terms and the source README's lack of additional
restrictions. Any derived figure must acknowledge NIST, DOI
10.18434/mds2-4227, and state: "Modified 2026-11-23: selected 128 released
slices and resampled generated numerical surfaces; not an anatomical
unwrapping or an ink annotation."

Source metadata contains contact fields. Keep raw metadata, acquired arrays,
maps and receipts outside Git until a public-safe subset is reviewed. No
unread candidate location or image belongs in this directory. No submission
or communication with organisers is part of this experiment.
