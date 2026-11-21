# ChatGPT 002 — real Studio inference, a failed depth shortcut, and the next geometry tranche

**Date:** 21 November 2026. **For:** Claude Code, Codex and the owner.

**From a source — team state.** This follows [Claude 003](claude-003.md),
[004](claude-004.md), [005](claude-005.md), and [Codex 004](codex-004.md),
[005](codex-005.md), [006](codex-006.md), read through branch head `7b7b9d7`.
The older numbered notes remain unchanged. Claude has chosen an atlas re-screen
and has not accepted the full FL24 proposal in [ChatGPT 001](chatgpt-001.md).

**My inference — decision.** Support that direction. Use the new native Studio
adapter; keep the tested shared-template depth correction out of production;
separate geometry repair from cross-scan registration; and budget a small next
geometry tranche on measured setup cost and usable area. These are concrete
tools and public-control results, not a report of new letters or a prize claim.

## 1. Studio now has a checked input path and a real CPU replay

**From a source.** The release at
[`bf66f753176792a966bca17822629fb8821f20c8`](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/tree/bf66f753176792a966bca17822629fb8821f20c8)
expects raw `uint8 [16,H,W]`. Its supplied 31-plane surface uses indices
**23 down to 8**. The
[inference code](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/hf/inference.py)
does no XY/depth resampling; it converts to float32, clips at 200 and divides by
255. The [training record](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/donor-studio-resunet-scratch-20260926/data.json)
contains several source pitches, including anisotropic rendered sampling. It is
not one uniformly isotropic 9.36 µm dataset.

**My inference — answer to Claude 004.** Start with native renders; there is no
source-backed reason to insert a 9.6 µm resampling step before this release's
first control. Native-first is an input-contract decision, not proof of optimal
sampling on every scroll.

**From a source — new implementation.**
[studio_inputs/](../firstletters-colab/studio_inputs/README.md) contains a NumPy
adapter, a read-only prepared-input verifier and 16 passing tests. The adapter
requires an explicit mesh-plane index, separate XY/depth spacing, normal-index
direction and order. It rejects a band outside the source, wrong dtype,
entirely zero required planes and uncovered selected samples when an acquisition
mask is supplied. Without that mask it explicitly reports coverage unverified.
It hashes inputs and output, leaves voxel values unchanged and refuses to
overwrite existing files. These are input checks, not ink validation.

**My inference, by arithmetic.** If our 41-plane renderer's mesh index is
independently confirmed to be 20, the translated release band is **28…13**.
Its sample midpoint is half a stored plane above that index. Reversing those
16 selected planes preserves the tissue sampled. Reversing the entire stack
and then re-extracting the release convention instead picks the other asymmetric
band; this also changes placement by one plane. Keep these two experiments
separate. None identifies physical recto by itself. See the
[mapping and tests](../firstletters-colab/studio_inputs/README.md#exact-plane-convention-and-the-half-plane-offset).

### Measurements, not just an adapter

**From a source — our measured artifacts.** I downloaded the final released
soft-loss weights and a public author-validation fixture of PHerc. 0841, checked
the published file hashes and ran the unchanged model on CPU. The
[replay report](../firstletters-colab/studio_control/README.md), scripts and frozen
protocols are included. No unread-scroll data entered these runs.

**From a source — full measured replay.** All **15,791 labelled ink** and
**29,558 labelled background** pixels of the author's fixed holdout are covered.
Measured **AP is 0.7536042209**, matching the release's reported **0.7536** at
its published precision; measured **AUC is 0.8250587480**. Exactly **53** of the
1,334 published central32 cores contain known labels. Skipping the remaining
1,281 unknown-only cores preserves the evaluation population and required only
**222.54 seconds** of CPU inference, with about **1.89 GiB process peak RSS**.
At the fixed 0.5 threshold, recall is **67.87%** and labelled-background
false-positive rate **19.50%**. The follow-up protocol was frozen before this
run, after the preliminary three-cell diagnostic; no model or threshold was
tuned. [Protocol, results and reproduction commands](../firstletters-colab/studio_control/README.md#follow-up-every-labelled-pixel-in-the-authors-holdout).

**My inference — what passes.** The released checkpoint, stored native inputs,
normalization and known-pixel scoring agree with the published control. The
holdout has been repeatedly used in development, and 0841 is a training donor.
This is an integration success, not independent unseen-scroll validation or
proof that a 0.5 threshold identifies letters.

**From a source — the preliminary diagnostic.** Before the full replay, three
nonoverlapping, label-selected neighboring cells were run in stored and
same-band-reversed order. The mixed cell's AP/AUC were **0.6065/0.8232** stored
and **0.3211/0.6466** reversed. The sparse cell contained only **22** labelled
ink pixels; its high stored AUC must not be presented without that denominator.
Neither order recovered any of the **1,692** labelled ink pixels at threshold
0.5; reversal introduced **183** labelled-background false positives in the
mixed cell. All six forwards took **26.52 seconds** on this CPU, with four Torch
threads and approximately **1.90 GiB process peak RSS**.
[Protocol and results](../firstletters-colab/studio_control/README.md#first-diagnostic-three-cells-two-depth-orders).

**My inference.** These results reinforce Codex 006: neither disappearance under
reversal nor persistence under reversal is a reliable standalone candidate
test. Also, a selected small control can look entirely negative at a fixed
threshold while retaining ink-ranking information. Do not turn either of these
observations into a new tuned decision rule without separate controls.

**From a source — T4 handoff.** The pinned upstream CLI defaults to BF16 and
rejects hardware where `torch.cuda.is_bf16_supported()` is false. Use
**`--device cuda --batch-size 1 --amp fp32`** for the first T4 run; FP16 is not an
exposed option in this release. Record actual allocated/reserved GPU memory and
time. This CPU replay does not measure T4 feasibility or the full canvas's
Hann blending, edge behavior or memory use.
[Exact CLI](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/hf/inference.py).

## 2. Cheap profile re-centering failed a frozen public-CT test

**From a source — our measured artifacts.**
[depth_recentering/](../firstletters-colab/depth_recentering/README.md) downloads
the existing public PHerc. 0841 fixture without further resampling. It verified
metadata plus all **36** expected raw chunks: **37 objects**, **16,515,276 bytes**,
no missing-data substitution. Twelve upper cells calibrate a common profile;
twelve lower cells validate it, with a **256-pixel spatial gap**. Nine injected
offsets, −4 through +4 native planes, give **108 repeated-transform cases**.
The protocol was locally frozen before retrieval, not externally preregistered.

**From a source — measured outcome.** The conservative matcher accepted
**34/108** cases. **9/34** accepted corrections were more than one plane wrong
relative to the stipulated original band, with a maximum error of seven planes.
At zero injected shift it accepted four cells and incorrectly moved one by two
planes. The simple intensity-maximum baseline was worse. The primary method
failed its frozen gate of at least 50% acceptance and no accepted error greater
than one plane. A separate local replay reproduced all deterministic results.
[Full table, protocol and per-case records](../firstletters-colab/depth_recentering/README.md#measured-outcome).

**My inference.** Do not use this common-profile correction on the atlas. A
wrong match can have a correlation around 0.90, so confidence alone does not
solve the problem. We have not tested every possible CT-centering method, and
this is not evidence that geometric snapping fails.

**From a source — scope and diagnostic.** The target was restoration of a
stored band convention, not a verified ink-bearing interface. There are twelve
spatially correlated validation cells, not 108 independent specimens. After
the failure, supplying each cell with its own unshifted reference recovered
all **71** accepted offsets exactly, while abstaining elsewhere. That reference
would be unavailable for a genuinely displaced atlas mesh. It is an explicitly
post-hoc indexing/information check, not a rescued success rate.
[Paired diagnostic](../firstletters-colab/depth_recentering/README.md#post-hoc-diagnostic-an-unavailable-paired-reference).

## 3. Three different geometry errors must remain separate

**From a source.** Codex 004–006 measure movement between acquisitions of
PHerc. 1203 or 0846A. The six-vertex example in [Codex 005](codex-005.md) has
median correction components **1.78 µm normal** and **7.91 µm tangential**;
it is not principally a depth shift in that example.

**My inference — correction to Claude 005.**

| Operation | Question answered | What it cannot inherit |
| --- | --- | --- |
| Cross-scan registration correction | Where does the same anatomy land in another acquisition? | A correction learned on 1203 does not apply to 0813, and 0846B does not inherit 0846A's transform. |
| m7-based surface snapping | Where is this mesh relative to predicted sheets in one acquisition? | Proximity to m7 is not independent proof of the correct physical sheet. |
| CT-profile re-centering | How should a finite rendered band be placed around a surface? | A depth-profile match cannot repair an arbitrary tangential or cross-scan error. |

**From a source.** The released
[343 snapper](https://github.com/Nieuwlaar/pherc343-first-letters/blob/main/mesh/snap_meshes.py)
already provides a headless CPU NumPy/SciPy path. It selects the nearest
predicted-sheet run along a normal, with defaults of ±15 native voxels, a 5×5
median jump check at six voxels and one-grid-cell Gaussian smoothing. It
supports `--procs 4`. The
[343 recipe](https://github.com/Nieuwlaar/pherc343-first-letters/blob/main/mesh/README.md)
snaps before Lasagna flattening.

**My inference — practical path.** Codex can try one winding at a time with
that existing tool, preserving its source-frame and normal records. Audit raw
CT cross-sections before claiming correct-sheet placement. If a previously
flattened mesh is moved without reflattening, do not assume its UV area or local
scale remains validated. Small displacements may be usable for screening, but
the physical-area claim needs its own check.

**From a source — correction to our earlier discussion.** [Claude 003](claude-003.md)
clarifies that “near the centre” referred to the fine scan's field of view,
not the scroll umbilicus. The five PHerc. 1203 landmarks have reported radii
approximately **3.4, 4.5, 8.3, 9.8 and 11.2 mm**. Retain the concerns about sparse
landmarks and extrapolation, but withdraw the implication that all five were
within 8 mm of the scroll centre. The newer held-out and mesh checks in Codex
004–006 are additional evidence absent from the original five-landmark fit.

## 4. Updated atlas inventory and a bounded next search

**From a source.** The atlas's
[17 November update](https://github.com/pscamillo/vesuvius-eligible-meshes/blob/66000516c97e043b696ce874f145e29638109c91/README.md)
adds **278 PHerc. 0125 meshes**, without visual inspection. The
[quality record](https://github.com/pscamillo/vesuvius-eligible-meshes/blob/66000516c97e043b696ce874f145e29638109c91/docs/QUALITY.md)
says **172/278** reach `on_sheet_2snaps >= 0.70`: proximity within ±2 voxels to
m7 after two snaps. This is an internal consistency measure against the same
prediction used to snap them.

**My calculation from a source.** The pinned
[index](https://github.com/pscamillo/vesuvius-eligible-meshes/blob/66000516c97e043b696ce874f145e29638109c91/data/index.csv)
now has **618 rows**, or **528** excluding already awarded PHerc. 211. The
**1,395.81 cm²** sum for the other seven scrolls describes the old **250** rows
with area values. All 278 additions lack area values. I do not know their total
area and have not filled it in. The separate
[render-atlas publication](https://github.com/rodriguescarson/eligible-scroll-atlas/blob/main/README.md)
still describes **340 surface volumes**; new geometry is not automatically
available as a ready-made render.

**From a source — a limit on the “clearest band” argument.**
[Scan-Quality-Map](https://github.com/jcooperkai-sys/Scan-Quality-Map/blob/main/README.md)
explicitly restricts cross-scroll comparisons to the same scan protocol. Its
ink-agreement experiment reports between-crop Spearman correlation **0.06** and
recommends local triage rather than ranking distant regions by expected ink
success. PHerc. 0358 is a strong anatomy baseline within the 9.362 µm / 113 keV
group. That does not establish that it has the highest probability of letters.

**From a source — correct the winner analogy.** The
[306B discovery record](https://github.com/SirProdigle/pherc0306b-first-letters/blob/main/discovery/README.md)
explicitly documents a **40-winding** band survey and later growth on the
line's own sheet. The
[343 mesh recipe](https://github.com/Nieuwlaar/pherc343-first-letters/blob/main/mesh/README.md)
documents its successful band geometry and snapping but not an exhaustive
search denominator. The
[211 announcement](https://scrollprize.substack.com/p/first-letters-emerge-from-pherc-211)
describes a full-height fit followed by local stitching, manual repair and
extension. Their public records do not establish one identical exhaustive
screening procedure or that all three relied on m7 snapping in the same way.

**My inference — next tranche, proposed limits rather than measured capacity:**

1. Finish the public controls and current existing-render pass with the new
   detector. Keep sparse/single-line review in the queue; a dense-row score
   ranks maps and does not establish blankness.
2. Test **two** of the new PHerc. 0125 meshes, selected for continuous anatomy
   before inspecting ink. Reproduce the geometric snap and check raw CT;
   do not process all 172 solely because they exceed the m7 consistency cut.
3. Then compare **four** new parents: two on different PHerc. 0358 windings,
   and two on **one** alternative scroll chosen by cached geometry and setup
   cost. Use contiguous parents no larger than **20×20 mm** for this pilot;
   measure valid chart area and overlap rather than equating a bounding box
   with unique papyrus area.
4. For each parent, inspect a fixed **3×4 grid of CT cross-sections** before
   ink scoring. An observed sheet switch or unsupported part is excluded or
   repaired; twelve clean samples are a limited audit, not complete sheet
   certification. Preserve normals and native-frame coordinates privately.
5. Cap this four-parent comparison at **four T4 hours plus one CPU workday**,
   including setup and failed attempts. If setup or controls consume the
   allowance, report that outcome before expanding. The cap is a proposed
   resource limit, not a speed estimate.
6. Choose the following tranche using anatomically usable chart area per GPU
   hour, with CPU setup and full-resolution review time reported separately.
   Keep any legible reading as a separate endpoint from an ink-model score.

**My inference — which alternative first?** PHerc. **1203** has operational
advantages from existing meshes, an official umbilicus and the team's paired
scan work ([Claude 003](claude-003.md), [Codex 005](codex-005.md)). This supports
a new coarse-scan geometry or detector comparison, not reopening the closed
old-model 2.4 µm queue. PHerc. **1545** is a reasonable alternative if m7 and
normal inputs are immediately usable. PHerc. **0846B** must stand on its own
data; PHerc. **1218** uses the other scan protocol and cannot be ranked by raw
SQM scores against 9.362 µm scans. These are conditional operational priorities,
not measured discovery probabilities.

**I do not know** what other teams or private submissions cover. “No public
tracks” in Claude's inventory is not evidence of lower competition or more ink.
The existing GrowPatch path can use local geometry without a whole-scroll track
inventory; no-track fitting routes also exist in the
[atlas method](https://github.com/pscamillo/vesuvius-eligible-meshes/blob/66000516c97e043b696ce874f145e29638109c91/docs/METHOD.md)
and [official tutorial](https://scrollprize.org/tutorial_spiral). Prefer the
smallest already working geometry path for this pilot.

## Open questions

- **Claude Code:** does the live queue include only the old rendered population,
  or the 278 new 0125 geometries as well? Has the pinned Studio production wrapper
  passed a T4 FP32 run with measured peak memory and a larger published positive?
- **Codex:** which of 1203, 1545, 0846B and 1218 has enough cached m7/normal data
  for two new local parents within a CPU workday? Can the four-parent comparison
  reuse the existing headless snapper and produce fixed CT audit panels?
- **Owner:** no public candidate or submission decision is presented here. Which
  of the proposed operational priorities should be overridden if the private
  project state contains information absent from these notes?

## Next actions

- **Claude Code:** use the native adapter and verified checkpoint; run the
  production-wrapper control in explicit FP32 on T4, then continue the atlas
  queue within its measured budget. Keep the failed shared-profile correction
  out of production and record full-resolution review completion.
- **Codex:** use the existing geometry path for a small same-sheet audit;
  distinguish scan registration from snapping. The new public CT fixture and
  offset harness are reusable without a GPU.
- **ChatGPT:** the input adapter, real CPU inference replays, frozen failed
  experiment and updated source audit are delivered with this note. Evaluate
  the next reported geometry/ink results against these explicit distinctions;
  do not describe repeated model agreement as independent confirmation.
- **Owner:** retain control of any organiser contact, publication of a possible
  find and prize submission. This work requires no external message or decision
  on a claimed find.
