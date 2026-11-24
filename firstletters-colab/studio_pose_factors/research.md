# Separating anchor placement and ray direction in a surface sampler

Research and independent design note, 2026-11-24. This document motivates the
experiment and its interpretation; it contains no newly measured model result.

## 1. Why the previous rigid tilt did not identify a single cause

The previous native-CT experiment rotated both the sampled surface and its
normal rays. Its losses could reflect misplaced anchors, changed ray direction,
different depth placement of local structures, a different surrounding
context, and resampling. A detector probability cannot distinguish those
mechanisms on its own.

The official Vesuvius technical account defines a surface volume as CT samples
around a mesh along its local normals. It also explains that ordinary ink
labels are two-dimensional and do not identify the ink's precise depth in CT.
These are separate uncertainties: numerical alignment of a reference surface
point does not establish the true three-dimensional location of the material
associated with its label.

Source: [Vesuvius Open Problems](https://scrollprize.org/from_ash_to_text),
sections *Ink recovery* and *Fragment-trained ink models*. The
[official Markdown source](https://github.com/ScrollPrize/villa/blob/main/scrollprize.org/docs/37_2026_open_problems.md)
was also inspected because the website sometimes returns an empty extracted
page. This is primary project documentation, not an external explanation of
the present result.

## 2. A complete two-factor design on renderer components

Let S(q) be the original surface, n(q) its signed unit normal, C the original
patch-centre reference point, and R one fixed rotation constructed from that
original surface. In ideal arithmetic, define

```text
F_ab(q,d) = C + R^a (S(q)-C) + d R^b n(q),    a,b in {0,1}.
```

| Cell | Anchor positions | Sampling directions |
|---|---|---|
| identity / 00 | Original | Original |
| normal_only / 01 | Original | Rotated |
| anchor_only / 10 | Rotated | Original |
| rigid / 11 | Rotated | Rotated |

The name `normal_only` is an implementation label. In the two hybrid cells,
the sampling directions generally are **not geometric normals of the anchor
surface**. The scientific factors are therefore *anchor placement* and *ray
direction*. These are controlled interventions in the sampling algorithm;
they are not four equally plausible anatomical surfaces or a new CT acquisition.

The NIST engineering handbook describes a full factorial design as observing
every factor setting with every setting of the other factors. Keeping all
four cells avoids assuming the two renderer components act independently.
Sources: [full factorial designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri333.htm)
and [models with main and interaction effects](https://www.itl.nist.gov/div898/handbook/pri/section4/pri43.htm).
We use this arrangement for paired descriptive contrasts, not an ANOVA with
independent pixels or randomized anatomical subjects.

### Preserve the earlier floating-point contract

The source tube is already represented as FP32 coordinates N32 constructed by
the pinned author sampler. The hybrids preserve that tube's discrete depth
offsets through

```text
C + R^a (S32-C) + R^b (N32-S32).
```

Identity and rigid cells must retain the historical construction exactly so
that the previous model outputs are legitimately reusable. Algebraically
equivalent rearrangements can round differently. Reuse requires matching
input bytes, model identity and stored depth order, not merely a close AP.

The reference surface is at depth index 8 in the asymmetric offsets
[-8,...,+7]. It must not be silently moved to the stack midpoint 7.5.

## 3. Independent planar oracles

For a plane S(u,v)=(u,v,0), normal +Z, centre C=0, rotation about +X, and the
original reference point X=(u0,v0,0), the exact inverse is:

| Cell | Readback u',v' | Depth d' |
|---|---|---|
| Identity | u0, v0 | 0 |
| Rays only | u0, v0 | 0 |
| Anchors only | u0, v0/cos(theta) | -v0*tan(theta) |
| Rigid | u0, v0*cos(theta) | -v0*sin(theta) |

The author's `cross(drow,dcol)` convention gives -Z for this chart ordering.
That reverses depth signs, not XY coordinates. For a +Y rotation, substitute
u for v and reverse the indicated depth sign. Physical chart spacing must
be included before expressing depth in native CT voxels; an unequal chart
metric is an explicit test fixture.

The secant in the anchor-only case and cosine in the rigid case are different
geometric effects. Reusing a rigid readback for the anchor-only cell would be
wrong even on a perfectly flat sheet.

The reference point is an exact root at d=0 for rays only, including on a
curved mesh. This justifies an exact identity readback rather than a numerical
search. It does not remove the need to check local conditioning: at tangency,
the ray/surface tube has a singular Jacobian even when a reference point lies
on it.

The general Jacobian has columns

```text
A*S_u + d*B*n_u,    A*S_v + d*B*n_v,    B*n,
A = R^a, B = R^b.
```

The coordinate-level factorial interaction is identically zero in ideal
arithmetic:

```text
F_11 - F_10 - F_01 + F_00 = 0.
```

An observed interaction in CT values or detector scores can still be large:
the sampled CT field, interpolation, network, output readback, threshold and
AP functional are not additive operations. It is not a newly discovered
interaction term in this simple coordinate equation.

`test_factor_geometry.py` checks these oracles, compatibility at large native
coordinate origins, a cylinder inverse derived independently in its circular
cross-section, missing geometry, and rejection of tangent hybrid rays.

## 4. Labels, inversion and information loss

Suppose actual ink is at X=S(q)+h*n(q), with unknown h. On the planar example,
ray-only rotation keeps the h=0 reference fixed, but maps a nonzero h to
depth h/cos(theta) and a shifted column proportional to h*tan(theta). Keeping
reference anchors fixed therefore does not guarantee identical material in
each original labelled column. This follows analytically from the sampler;
we do not have three-dimensional ink labels to remove that uncertainty.

The TorchIO primary paper and official documentation support tracing spatial
transformations and returning model predictions to the original coordinate
space. They also distinguish geometric inverses from lossless data recovery:
affine resampling and cropping can lose information. Our implementation uses
the explicit surface tube rather than importing TorchIO's transform classes.

Sources:

- Perez-Garcia, Sparks and Ourselin, *TorchIO: A Python library for efficient
  loading, preprocessing, augmentation and patch-based sampling of medical
  images in deep learning*, Computer Methods and Programs in Biomedicine 208,
  106236 (2021), [primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8542803/),
  [DOI](https://doi.org/10.1016/j.cmpb.2021.106236).
- [Official transforms documentation](https://docs.torchio.org/latest/transforms/),
  sections *Reproducibility*, *Invertibility* and *Interpolation*.

The same CT acquisition is used throughout. Its scanner point-spread function
and acquisition artifacts remain fixed. Turning sampling coordinates is not
a simulation of rotating the object during a new scan.

## 5. Comparisons and masks

The primary comparison is one complete 2x2 block for a declared axis and
angle. All four cells in a block share a geometry-valid label mask, fixed
before the new model outputs are inspected. Each block must retain at least
95% of its known labels globally and 90% in every original core. No core is
substituted on the basis of support or score.

Masks may differ between angle blocks. Consequently, a change in AP between
angles is not automatically a comparison on an identical population. Report
the block-specific identity and denominators, and interpret the paired
within-block contrasts. The secondary intersection across all 25 distinct
conditions is descriptive and provides a common population if desired; it
does not replace the declared primary analysis.

For a metric m on the common block population, retain:

```text
anchor change with original rays:     m10 - m00
anchor change with rotated rays:      m11 - m01
ray change with original anchors:     m01 - m00
ray change with rotated anchors:      m11 - m10
interaction contrast:                m11 - m10 - m01 + m00
```

These are renderer contrasts conditional on the same source, model and
reference labels. They are not a percentage causal attribution of the prior
failure. Publish mixed-core and pure-control results alongside pooled metrics:
AP is non-additive across these groups, and a lower fixed-threshold FPR may
coexist with a severe recall loss.

## 6. Reused outputs require a fresh support audit

The earlier Q33 audit certified its own global scoring mask. A new block mask
can include pixels that were excluded from that older mask, even though the
older probability archive contains values there. A previous PASS is not a
certificate for the larger support.

For this execution, the complete audit should recompute the reused rigid
readback from the hash-verified previous 128-square output arrays and previous
mapping, then compare on the new masks. The portable public-archive audit can
check finite stored probabilities, mappings/labels represented by the archive,
metrics and contrasts; it cannot independently certify the original network
forward or native CT acquisition. The two scopes must remain explicit.
