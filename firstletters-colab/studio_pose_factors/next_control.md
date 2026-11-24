# Next control: native CT and optical ink labels from Frag4

Prepared on 2026-11-24. This is a source and feasibility investigation, not a
new model experiment or an amendment to the active pose-factor protocol.
No model forward or CT chunk acquisition was performed for this investigation.
The accompanying [source receipt](evidence/next_control_source_receipt.json)
records the small metadata and HTTP range checks already made.

## Recommendation and scope

**Frag4 / PHercParis1Fr39 is a concrete optical control external to author841.**
Its public assets include an optical ink mask, a mapping from that mask's
surface frame to native CT coordinates and normals, and a chunked native CT.

**Frag4 is excluded from gradient updates according to the published ancestry
and training manifests of the final Studio checkpoint, but it was exposed
during model development and comes from the same parent manuscript as the
trained Frag3. It is not an independent validation set.**

The supported next use is a mechanistic replication of the anchor/direction
factor decomposition on another physical fragment with optical labels.
Absence from published training rows is a provenance finding, not an
independent audit of every historical action taken by the model author.
No conclusion about robustness or ink performance on this control is made here.

## Why the existing controls do not answer the same question

| Control | What is available | Limitation for a native optical factor experiment |
| --- | --- | --- |
| Gemini T17, PHerc0211 | Native geometry and CT; public readable text; baseline and perturbed detector maps | The principal reference is a canonical predicted map plus visual review, not an optical pixel ink mask. |
| Adaptive051, optical Frag3/Frag4 | Infrared-derived ink labels and a range-readable patchpack | The executed experiment used already flattened, clipped uint8 patches with 24 fixed depth planes. That patchpack alone does not supply the native geometry needed to rotate anchors and sampling directions. |
| Native Frag3 / PHercParis1Fr34 | Optical mask, native `result.ppm`, and 54 keV CT in Zarr | Frag3 is explicitly included in the Studio CPT training mixture. |
| Native Frag4 / PHercParis1Fr39 | Optical mask, native PPM, and 54 keV CT in Zarr | Best candidate among the inspected controls, with the development exposure and shared-parent limitations stated above. |

Repository evidence:
[Gemini007 / T17](https://github.com/JorisViaudQuantAI/Scrollprize/blob/74c55996176970025a41992bce22bdf893289910/collab/gemini-007.md);
[Adaptive051](https://github.com/JorisViaudQuantAI/Scrollprize/blob/16001d8b0bb63f7d708c5746e7e0281a8c9f7f07/collab/chatgpt-adaptive-051.md);
[Adaptive051 frozen protocol](https://github.com/JorisViaudQuantAI/Scrollprize/blob/16001d8b0bb63f7d708c5746e7e0281a8c9f7f07/firstletters-colab/adaptive_surface_recovery/optical_scale_transfer/protocol.md).

Adaptive051's public patchpack revision is
`YoussefMoNader/ink-8um-v8-patchpack@d67b1c1548ef93df9e00e804870f89127817f4c5`.
Its [dataset card](https://huggingface.co/datasets/YoussefMoNader/ink-8um-v8-patchpack/blob/d67b1c1548ef93df9e00e804870f89127817f4c5/README.md)
states that the fragment labels derive from infrared photographs. Its Frag4
canvas is 4010 × 2483 in H×W, whereas the native registered optical assets below
are 9791 × 6064. These frames must not be silently interchanged.
The official [ink-label documentation](https://scrollprize.org/data/datasets/ink-labels-README.md)
describes aligning infrared photographs to fragment surface volumes and tracing
the ink manually.

## Native Frag4 assets and checks

All paths in this section are relative to the
[official Frag4 volume package](https://dl.ash2txt.org/fragments/Frag4/PHercParis1Fr39.volpkg/).

| Purpose | Exact relative path | Observed properties |
| --- | --- | --- |
| Optical ink label | `working/54keV_exposed_surface/PHercParis1Fr39_54keV_inklabels.png` | PNG, W×H = 6064×9791; 98,366 bytes; header range returned HTTP 206. |
| Valid surface mask | `working/54keV_exposed_surface/PHercParis1Fr39_54keV_mask.png` | Same W×H; 54,426 bytes; header range returned HTTP 206. |
| Native surface mapping | `working/54keV_exposed_surface/extras/PHercParis1Fr39_54keV.ppm` | W×H = 6064×9791, six doubles per pixel, ordered, version 1; 73-byte header; total 2,849,886,025 bytes. |
| Native CT, full-resolution level | `volumes_zarr/54keV_3.24um_.zarr/0/` | Zarr v2; ZYX shape [9231,1968,5808]; 128³ chunks; little-endian uint16; Blosc/Zstd. |
| CT identity and pitch | `volumes/20230215185642/meta.json` | UUID 20230215185642; 54 keV; 3.24 µm isotropic voxels; dimensions match Zarr. |
| Optical reference images | `working/reference/` | Listing contains the 1000 nm infrared TIFF and photographic views. Images were not downloaded. |
| Alignment material | `working/54keV_exposed_surface/extras/PHercParis1Fr39_54keV_alignment.psd` and `PHercParis1Fr39_54keV_ir.png` in that same directory | Present in the official listing. Contents were not downloaded or visually verified. |

Direct metadata and directory links:
[registered optical assets](https://dl.ash2txt.org/fragments/Frag4/PHercParis1Fr39.volpkg/working/54keV_exposed_surface/),
[native mapping and alignment assets](https://dl.ash2txt.org/fragments/Frag4/PHercParis1Fr39.volpkg/working/54keV_exposed_surface/extras/),
[Zarr metadata](https://dl.ash2txt.org/fragments/Frag4/PHercParis1Fr39.volpkg/volumes_zarr/54keV_3.24um_.zarr/0/.zarray),
[native scan metadata](https://dl.ash2txt.org/fragments/Frag4/PHercParis1Fr39.volpkg/volumes/20230215185642/meta.json),
[reference photographs](https://dl.ash2txt.org/fragments/Frag4/PHercParis1Fr39.volpkg/working/reference/).

The native PPM's header was read with `Range: bytes=0-1023`. A separate
48-byte interior point read returned HTTP 206 and yielded native XYZ values
inside the CT extent and an oblique normal. This establishes partial-read
feasibility and basic numerical consistency; it does not establish pixel-exact
render reproduction. No full PPM, TIFF stack, CT chunk, PSD, or model output was
downloaded. Full asset SHA-256 values were not computed and are not claimed.

The [official PPM documentation](https://scrollprize.org/tutorial3) defines
`ppm[y,x,0:3]` as the native 3D position and `ppm[y,x,3:6]` as the normal.
The optical mask and native PPM have matching dimensions. A future registration
check must still verify that the native mapping reproduces the supplied render
in the correct label frame.

## Two PPMs: native geometry versus an already flattened volume

The two similarly named Frag4 files are materially different:

| PPM | Coordinate system | Use for the proposed experiment |
| --- | --- | --- |
| `extras/PHercParis1Fr39_54keV.ppm` | Native CT XYZ plus an oblique surface normal | This is the candidate native anchor/direction mapping. |
| `extras/PHercParis1Fr39_54keV_surface_volume.ppm` | Flattened surface volume; the tested interior value is exactly (x,y,32,0,0,1) | Do not use it as a native CT mapping. |

The second PPM is version 1.0 with a 75-byte header and a total size of
2,849,886,027 bytes. Its central 48-byte range was checked independently.
The adjacent
[`PHercParis1Fr39_54keV_surface_volume.json`](https://dl.ash2txt.org/fragments/Frag4/PHercParis1Fr39.volpkg/working/54keV_exposed_surface/extras/PHercParis1Fr39_54keV_surface_volume.json)
references this flattened PPM and the optical labels because it describes work
inside the already flattened stack. Copying its PPM choice into a native CT
renderer would change the experiment.

The same distinction was independently observed on Frag3:
`working/54keV_exposed_surface/result.ppm` is native, while
`working/54keV_exposed_surface/surface_volume.ppm` is planar.
Those checks do not substitute for the Frag4 checks.

## Complete published checkpoint ancestry

Audit source:
`priyankaranke/vesuvius-studio-multiscroll-cpt@bf66f753176792a966bca17822629fb8821f20c8`.
The checkpoint under discussion is
`studio-softloss-20261023-ema.safetensors`, SHA-256
`96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90`.
The checkpoint was not downloaded or rehashed as part of this preparatory task.

| Stage | Published initialization and data | Implication for Frag4 | Exact author source paths |
| --- | --- | --- | --- |
| Studio parent | Random initialization; `pretrained_checkpoint: null`; 814/841/9B sparse donor data; epoch 9 | No fragment input is declared for the parent initialization/training. | `donor-studio-resunet-scratch-20260926/config.json`; root `README.md`; `reproduction.py` |
| Multi-scroll CPT | Starts from Studio epoch-9 SHA-256 `ffffbf4fcb2f1eb91940a6899366cd0d22b2ea03987270d07dc2a205dbd91145`; 69 Expanded9 rows, then 47 Reader rows added | Both current and reference mixtures include fragment1,2,3,5,6 and omit fragment4. README explicitly attributes the exclusion to ambiguous depth order. | `studio-multiscroll-cpt-20261023/config.json`; `reference/config.json`; `mixture.json`; `reference/mixture.json`; `README.md` |
| Final softloss | Starts from CPT EMA; same mixture, replay, model, and augmentation; 3,000 further updates | Code loads the same CPT mixture; no new Frag4 training row is declared. | `studio-softloss-20261023/train_softloss.py`; `studio-softloss-20261023/README.md`; root `reproduction.py` |
| Prior development exposure | The author tested forward and reverse depth order on `fragment4-exposed` and used that evidence to exclude it | Frag4 is not an untouched development holdout. | `studio-multiscroll-cpt-20261023/reference/orientation-expanded9.json`; CPT `README.md` |

Source links:
[Studio scratch configuration](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/donor-studio-resunet-scratch-20260926/config.json);
[CPT configuration](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/config.json);
[CPT mixture](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/mixture.json);
[CPT reference mixture](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/reference/mixture.json);
[CPT dataset explanation](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/README.md);
[orientation check](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/reference/orientation-expanded9.json);
[softloss loader](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-softloss-20261023/train_softloss.py);
[softloss explanation](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-softloss-20261023/README.md);
[full reproduction chain](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/reproduction.py).

The standalone Expanded9 dataset itself contains a Frag4 representation.
That fact alone does not prove that the released Studio checkpoint trained on
it: the relevant CPT loader mixture excludes it. Conversely, an exclusion from
CPT alone would not establish independence without checking the parent
initialization and the subsequent softloss stage, as done above.

Frag3 is PHercParis1Fr34; Frag4 is PHercParis1Fr39. Their official volume-package
names and the author's orientation records identify the shared PHercParis1
parent. Therefore a Frag4 experiment cannot claim a wholly held-out parent
manuscript, even though its physical fragment and optical annotation differ
from author841.

## Four checks required before any future model inference

1. **Close the native registration chain.** Verify XYZ-to-ZYX ordering, PPM
   header parsing and endianness, normal length/sign, and mask validity. Render
   a small identity region from the 54 keV native CT and compare it with the
   corresponding supplied surface-volume region. Confirm the optical labels
   refer to that same region. The header and single-point checks above are not
   a replacement for this test.

2. **Freeze physical pitch and the resampling contract.** The native CT pitch
   is 3.24 µm; the active author841 experiment uses a different physical scale.
   Specify the target surface sampling pitch, CT depth step, context footprint,
   interpolation and label resampling explicitly. Do not apply a 9.36 µm
   offset as though it were one native voxel. Do not substitute Adaptive051's
   resized patchpack frame for the 6064×9791 native optical frame.

3. **Verify the intensity conversion and normalization.** The native CT is
   uint16, while the existing Studio worker accepts a prepared uint8 stack.
   Establish and freeze the documented conversion, clipping and normalization
   needed by the selected rendering/training convention. Inspect saturation
   and identity agreement. Adaptive051's clipped patchpack bytes are not
   evidence that the same conversion is valid for native Frag4.

4. **Resolve the depth convention without performance-driven selection.**
   Frag4 was excluded by the model author because the forward/reverse check was
   ambiguous. Use the public geometry and renderer convention to establish
   anchor depth, normal sign and the 16-plane order. If ambiguity remains,
   predeclare both conventions and report both rather than selecting the one
   with the better optical AP. Freeze the small control selection, all factor
   arms and common-support rules before running new model predictions.

Only after these checks should a separate protocol define the new factor
experiment. Read the PPM rows and native CT chunks needed by the union of the
four arms' footprints; full PPM/TIFF downloads are unnecessary. The 3.24 µm
fragment scan, exposed-surface anatomy, and optical annotation regime differ
from author841 and must remain explicit limitations.

## Open questions

- Can a bounded identity render match the published surface stack after a
  documented intensity conversion and physical-scale conversion?
- Can the Frag4 normal/depth convention be determined without choosing it from
  detector performance?
- What small, frozen set of mixed and background regions retains sufficient
  common support for every factor arm at the chosen physical scale?

## Next actions

Complete the four acquisition/render checks above, publish their receipts and
the resulting fixed protocol, then decide whether the available control
supports a new mechanistic experiment. Until then, retain the status
**candidate optical control; excluded from gradients according to published
provenance, exposed during development, shared parent manuscript with Frag3;
not independent validation**.

