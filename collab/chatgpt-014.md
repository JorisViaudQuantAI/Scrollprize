# ChatGPT 014 — a real Frag5 source-contract failure, preserved before coarse acquisition

22 November 2026. Continuing [ChatGPT 013](chatgpt-013.md),
[ChatGPT 012](chatgpt-012.md), [Claude 010](claude-010.md) and
[Codex 010](codex-010.md). This work uses the public, already-opened
PHerc. 1667 Cornice 1 Fragment 3 control (Frag5). No unread-scroll candidate,
letter reading or prize outcome is reported. Claude Code retains its live
GPU work; Codex retains its native geometry work.

## 1. Executed result

**From a source / measured.** The first actual CT stage completed and rejected
both declared source-order hypotheses. At all four fixed points, the ordinary
trilinear profile from the proposed raw fine source failed to reproduce the
released surface TIFFs within the frozen tolerance of one uint16 count.
The pipeline stopped before reading coarse-scan pixels or computing anatomy.
[Complete result](../firstletters-colab/transversal_lab/frag5_native_pair/evidence/result.json),
[all 260 profile samples](../firstletters-colab/transversal_lab/frag5_native_pair/evidence/source_profiles.csv).

| Fixed public geometry record | Primary maximum error | Primary median error | Reversed maximum error | Primary comparisons within one count |
| --- | ---: | ---: | ---: | ---: |
| 20 | 3,235 | 125 | 17,791 | 3 / 65 |
| 25 | 3,321 | 160 | 9,573 | 0 / 65 |
| 65 | 6,576 | 223 | 40,114 | 3 / 65 |
| 70 | 3,964 | 638 | 32,017 | 1 / 65 |

**From a source / measured.** Errors are stored uint16 intensity counts.
Every profile was nonconstant and all 65 samples in each were nonzero;
this was not an empty-profile failure. Only 7 of 260 primary comparisons
were within one count, including two exact matches. Center-plane errors
were 1,530, 236, 0 and 904 counts. The
[full profile plot](../firstletters-colab/transversal_lab/frag5_native_pair/evidence/source_identity.svg)
retains both declared orders without alignment or rescaling.

**From a source / independent verification.** A second implementation reread
all cached TIFF values and recomputed the 260 fine samples directly from
the six cached raw chunks. Its manual eight-corner interpolation agreed
with the runner to at most 1.46×10⁻¹¹; every rounded error vector and the
`neither` classification reproduced exactly. All 398 acquired response
objects passed hash/length verification. The
[independent result review](../firstletters-colab/transversal_lab/frag5_native_pair/evidence/independent_result_review.json)
used no additional acquisition, labels, model or fitted correction.
Its planned anatomy-metric branch remains unexercised because no anatomy
comparison ran.

## 2. What was fixed before looking at CT

**From a source / execution records.** The scientific protocol, executable,
generated checks, independent code review and source preflight were published
at [4ea8eed](https://github.com/JorisViaudQuantAI/Scrollprize/commit/4ea8eedb90d5757272cef59144c94d0b82cbb5e0)
before this run accessed any CT intensities or TIFF pixels. The exact
[protocol](../firstletters-colab/transversal_lab/frag5_native_pair/frag5_native_pair_stage1_protocol.json)
pins the [runner](../firstletters-colab/transversal_lab/frag5_native_pair/frag5_native_pair_stage1.py).
Its four points were chosen from 85 existing PPM geometry samples by a
fixed nearest-quarter rule; no intensity, label, model score or acquisition
cost selected them. The independent reviewer reproduced the same four
records from the existing geometric sample.

**From a source / protocol.** For each fixed point `p` and literal stored
normal `n`, the primary candidate is `p + (k - 32)*n`, for `k = 0…64`.
The reversed same profile was also declared before pixels. A common order
had to match all 260 TIFF values within one rounded uint16 count, with all
four measured/predicted profiles nonzero and nonconstant. A unique primary
or reverse match permitted anatomy; both-order ambiguity and neither-order
failure stopped it. There was no alternative phase, center correction,
normal normalization, intensity fit or replacement location. The
[independent review](../firstletters-colab/transversal_lab/frag5_native_pair/independent_review.json)
caught and corrected an initially permissive both-order gate before freezing.

**From a source / protocol.** The anatomy plan uses four 33³ coarse grids
centered on rounded affine images of the source points. Fine CT is represented
both by trilinear samples at inverse-affine coarse centers and by a fixed
5³ midpoint quadrature over each inverse-affine coarse voxel. All 27 native
coarse XYZ offsets in `{-1,0,+1}³` are retained; zero shift is primary.
Intensity and valid-interior six-neighbor Laplacian correlations use Pearson
and Spearman measures. No registration, blur or detector is fitted.
The 125-sample average approximates a nominal voxel footprint of the
interpolated fine field, not the coarse scanner's actual transfer function.

**My inference / physical scale.** At the stated coarse pitch of 7.91 µm,
one stress-test step is nominally 7.91 µm and a 33-voxel cell footprint spans
about 0.261 mm per axis. These small neighborhoods test source/anatomy
compatibility. They are not letter-sized detection contexts, independent
physical donors, or a validation of every sheet in the fragment. The affine
is the exact published transform; the nominal pitch ratio is not substituted
for its linear matrix. [Official coarse metadata](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes/20231130111236/meta.json),
[published affine](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/transforms/20231121133215-to-20231130111236.json).

## 3. Source and implementation findings that change our assumptions

**From a source / correction to ChatGPT 013.** Its chunk helper initially
counted the all-lower and all-upper trilinear corners, missing mixed corners
when multiple axes cross chunk boundaries. The previous note's statement
that it counted every corner was incorrect. A generated two-axis crossing
required four chunks where the old code counted two; a three-axis crossing
required eight where it counted two. Both regressions failed before the fix;
all seven tests passed after it. The corrected helper enumerates the full
Cartesian corner set and rejects budgets below that true count.
[Fix](https://github.com/JorisViaudQuantAI/Scrollprize/commit/3b99c0fe1d3625e0f9788633ef66cc7229be7758),
[reproduced regression receipt](../firstletters-colab/transversal_lab/evidence/native_pair_chunk_guard_regression.json).
The affine-fit and leave-one-out functions were unchanged, so their previous
measurements were not repeated or revised.

**From a source.** The [header/geometry preflight](../firstletters-colab/transversal_lab/frag5_headers/README.md)
read 4,862 response-body bytes in 102 exact HTTP 206 ranges, including 85
sparse PPM geometry records. It decoded no CT, photograph or ink-label
pixels. Sixty-four records were nonzero and compatible with little-endian
unit normals; 21 were all zero. Bounds compatibility did not prove the
source volume. The local geometric selection preceded those geometry
reads, but this early header preflight was not a public CT-outcome
preregistration. The separate stage-1 publication supplies that boundary.

**From a source.** The [pinned code audit](../firstletters-colab/transversal_lab/frag5_contract/FRAG5_SOURCE_CONTRACT.md)
found that the VC 2.26.0 candidate sampler uses stored normals unchanged,
integer voxel centers, trilinear interpolation and integer rounding. Its
PPM header does not store source UUID or byte order. Sixty-five filenames
alone do not prove a radius of 32 or a step of one; the inspected utility's
current Frag5-config default would instead produce 47 planes. The actual
historical Frag5 render invocation remains unknown. A numerical match at
four rays therefore establishes local consistency with a declared recipe,
not authentication of the whole rendering history or physical ink face.

**From a source / additional static counterexamples.** The exact
[pinned line sampler](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/src/LineGenerator.cpp)
uses `floor(2*r/interval)+1`. At unit interval, every radius `32 <= r < 32.5`
produces 65 layers; its index-32 displacement is `32-r`, so that index need
not be exactly at the PPM point. The
[layer utility](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/apps/src/LayerFromPPM.cpp)
and [sequence writer](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/include/vc/core/io/ImageIO.hpp)
also allow an existing output directory without clearing stale planes.
**My inference:** these are reasons the file count cannot authenticate a
single centered render. They do not establish what happened to Frag5, and
no radius, phase or mixed-generation explanation was fitted to our data.

**From a source / mathematical distinction.** Literal historical VC PPM
transformation uses normalized `A*n`; a mapped source ray uses `A*n` with
its source step, while a geometric surface normal under an affine uses
`A^{-T}*n` in the appropriate coordinate metric. These are different
operations. The earlier small sampled angular difference between them
cannot certify face or same-sheet identity. The exact source-code references
and the ray/normal distinction are retained in the contract audit.

**From a source.** Fine and coarse reconstruction windows differ. Equal
stored uint16 values therefore do not imply equal calibrated attenuation.
**My inference / mathematics.** Pearson correlation is invariant to a positive affine display conversion,
and Spearman correlation to a strictly increasing conversion, but clipping,
noise, reconstruction differences and geometry can still change their
measured values. The source audit records the actual window metadata and
its limits. No per-patch intensity transformation was fitted here.

**I do not know** the exact transformation that produced Frag5's official
`volumes_standardized` uint8 arrays or whether those particular arrays
formed Hecate's training representation. The
[bounded source-only follow-up](../firstletters-colab/transversal_lab/frag5_contract/standardization/FRAG5_STANDARDIZATION_HANDOFF.md)
inspected 89,658 new source-response bytes and found no file-specific
production recipe. A stored-data conversion, a loader's optional division
by its dtype maximum, and Hecate's model-input division by 200 or 255 are
separate operations. Neither `1/257` nor per-window min/max is established
as the historical conversion. Reopen this documentation search only with
a concrete new source.

**I do not know** the complete annotation lineage of the exact current
Naples `inklabels.png`: whether every stroke was photograph-derived,
whether a model assisted it, or the exact IR-to-CT warp and its residuals.
The source audit distinguishes the original paper's Paris-fragment process
from these Naples assets. No label or IR pixel was decoded in this run.

## 4. Interpretation and next scientific decision

**My inference.** This is a successful bounded acquisition and a negative
result for one exact source/rendering hypothesis. It provides no ink-model
negative, no proof of absent ink, and no test of the fine-to-coarse affine.
The mismatch is far larger than integer rounding and was independently
reproduced, but its cause remains unresolved. Source representation,
historical rendering/PPM provenance and sampling convention remain distinct
hypotheses. We did not search phase, radius, interpolation variant or intensity
scaling after seeing the profiles, and did not relax the one-count gate.

**My inference / next test.** Separate the raw-data representation question
from the historical surface-render question. The smallest next probe is
exact equality between native fine-volume TIFF values and the cached raw
Zarr at the eight integer corners around each of the four fixed PPM centers:
32 voxels across eight source slices. A separate bounded header preflight
and public protocol must precede those new pixel reads. This is an adaptive
follow-up motivated by the preserved failure, not an untouched confirmatory
rendering test. Its result could eliminate one source-conversion explanation;
it cannot authenticate the legacy surface recipe on its own.

**My inference.** A source-render order match is not proof of the order a
particular model expects, and neither establishes the physical ink-facing
side. Keep the full composition of PPM normal, renderer ordering, tensor
ordering and model preprocessing in the inference manifest. The severe
Hecate orientation sensitivity in ChatGPT 012 is not resolved by authenticating
one source renderer.

**My inference / constructive next option.** An unknown historical uint8
recipe does not forbid an explicitly defined new conversion. It prevents
calling that conversion a faithful reproduction of the released training
pipeline. A useful next test can freeze its own intensity mapping and
physical sampling, calibrate them on separate public controls, and evaluate
once on separated tissue. It must report the unknown inherited-model
exposure and exact label lineage. Neither the best score among conversion
variants nor the best local alignment on these four observed points can
become evidence of generalization.

## 5. Reproduction, resource accounting and prize relevance

**From a source / measured.** The run used six decoded fine chunks,
**24,571,323 source bytes + 11,591 TIFF/metadata bytes**, **5.605 process
CPU seconds** and **496.309 elapsed seconds**. All 81 HEAD checks and
398 body responses succeeded, including the pinned Zarr metadata and
all 65 TIFF header/IFD checks. The 260 TIFF pixel samples used 520 bytes.
No coarse chunk, label, IR photograph or checkpoint was fetched. The
original 14-chunk/56-MiB decoded plan was a prospective upper workload,
not the observed acquisition. The independent result review added 0.265
CPU seconds and zero network requests.

**From a source / receipts.** The final result SHA-256 is
`e473ad97f5afa8126e6d2f6e57ebc1080f906980cc3b7a46ed0c7381d8f3fe36`;
the independent review is
`4287d5a91ad87fc5d7271aa3da8ab8249924eeebb674a5ed0193b4770b57ade6`.
The [reproduction guide](../firstletters-colab/transversal_lab/frag5_native_pair/README.md)
links the original frozen inputs, complete ledger, all samples, numerical-only
plotter and independent local-cache checker. Hashes describe the bytes
actually acquired; the upstream source objects remain mutable. The ledger
records request order, while the frozen code's completed-future barriers
establish that all headers completed before pixel reads; no per-response
completion timestamps were recorded.

**My inference.** The reusable deliverable is now a bounded raw-data
acquisition/rendering path with explicit source, depth, byte and interpolation
contracts, plus a measured raw fine-CT source mismatch. Its value to the team is in
removing concrete ambiguity before detector experiments. It is not, by itself,
a new ink result or a reason for another standalone prize submission. Any
later learned correction needs a fixed preprocessing contract, a prospectively
separated evaluation region and useful ink evidence. These four observed
locations are now development evidence; do not relabel them untouched tests
for a method chosen from this note.

## Open questions

- **Codex:** Which source-independent local anatomical evidence can establish
  same-sheet correspondence beyond these small neighborhoods, including
  normal and tangential error? Preserve the published affine and every
  registered sensitivity until a separately documented amendment exists.
- **Claude Code:** Is there a pinned data-production or training manifest that
  binds the exact raw/standardized Frag5 assets to Hecate's input conversion?
  The model's division by 255 alone does not answer this.
- **ChatGPT:** What is the smallest additional independent anatomical control
  needed before a native-versus-fine-derived ink comparison is interpretable?
  Do not use unknown photographic-label lineage as independent ground truth.
- **Owner:** No candidate, qualifying image or prize award is reported. No
  intervention is requested; any eventual submission or discovery disclosure
  remains your decision.

## Next actions

- **ChatGPT:** Preserve the frozen failure and all profile errors. Complete the
  separate 32-voxel native-source check after its prospective publication. Keep
  the raw scans and original arrays outside git with reproducible hashes.
- **Codex:** Reuse the corrected all-corner support accounting and separate
  source compatibility from sheet-identity validation. Continue the already
  owned native-geometry work.
- **Claude Code:** Keep the existing GPU queue and working released-model
  baseline. This stage supplies no justification for a new adaptation sweep
  or for tuning depth order against ink labels.
- **Owner:** Continue reviewing the team's concrete evidence. Nothing here
  submits to the organisers, posts externally or merges to `main`.
