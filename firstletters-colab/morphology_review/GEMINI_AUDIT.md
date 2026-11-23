# Source-only audit of Gemini's Paris 4 roughness control

Status: completed source-only review; no change to the published method, receipt, or failed gate. No CT payload, retained prediction array, glyph output, or new model computation was opened for this review.

## Decision

**[My inference] Do not reverse the score and reopen the unread-region scan.** The existing positive-direction gate remains failed. The receipt does contain sizeable negative-direction *regional* associations, so “these descriptors do not separate carbon ink from blank papyrus” is broader than the experiment establishes. The defensible conclusion is: **none of the positively oriented, tested descriptors passed the specified pooled text-region versus blank-region gate; ink-specific discrimination was not established.** A separate, within-sheet, independently labelled test is scientifically reasonable, but its label-to-surface provenance and geometry must be established before another CT acquisition.

## Sources and audit scope

**[From a source]** I inspected the following byte-exact files at Gemini lane revision `8f5a55d030497f834c0a8e2737444c06a4c97d44`: [roughness implementation][crust], [shared geometry and statistics implementation][shared], [generated tests][tests], and [published control receipt][receipt]. Their URLs, byte sizes and SHA256 hashes are retained in [the source receipt](gemini_source_receipt.json); a compact, coordinate-free [derived audit](gemini_receipt_audit.json) is provided separately. The original source files are linked, not republished here. [Gemini004, section 3][note] is the reporting claim under review. No original feature arrays were available to, or used by, this audit; reported measurements below are the published receipt's measurements, not an independent feature rerun.

## What the scores measure

**[From a source]** The acquisition is nominally 1.129 µm at 78 keV, but the scored surface slab is resampled onto nominal 1.5 µm UV spacing and 2 µm normal steps. Each control slab has shape 49 × 65 × 97. Peak recentering searches within ±36 µm, with a tile anchor, a ±10 µm local corridor, normal/lateral smoothing and interpolated recentering. The surface intensity is the mean over |n′| ≤4 µm; “bulk” averages available samples at 10≤|n′|≤22 µm on both sides. The reference used for registration is the 4.8 µm level of another 78 keV acquisition. [Sources: implementation and receipt][crust]

**[My inference]** The nominal UV centre span is 96 × 144 µm per tile, far smaller than a complete letter. The point-to-point physical spacing also depends on the surface parametrisation; the code interpolates UV indices and constructs local normals, rather than explicitly equalising the local surface metric. These are sub-letter regional texture measurements. They cannot by themselves establish recovery of a stroke or a letter.

**[From a source]** The feature definitions are as follows. Intensity is reconstructed from uint8 by the source window's affine scale; the inspected files do not establish an absolute calibrated attenuation unit. [Sources: `uint8_to_linear_attenuation`, `compute_crust_crackle_maps`, `build_composite_crust_z`][shared]

| Feature | Actual operation | Units/convention |
|---|---|---|
| `surface_mu_recentered` | Surface-band mean intensity | Reconstructed intensity scale |
| `tophat_recentered` | Surface mean minus two-sided bulk mean | Same intensity scale |
| `hf_roughness_local` | Local RMS of a **2D discrete intensity Laplacian**, Gaussian σ=12 µm | Unscaled discrete-grid intensity curvature; no division by UV pitch squared |
| `hf_roughness_excess` | Surface local Laplacian RMS minus bulk local Laplacian RMS | Same grid-dependent scale |
| `crackle_dog_rms` | RMS of a surface-intensity difference of Gaussians, σ=2 and 8 µm, locally averaged with σ=12 µm | Intensity scale |
| `crackle_isotropic` | Square root of the structure tensor's smaller eigenvalue after σ=2 µm smoothing; derivatives use UV pitch | Intensity per nominal µm |
| `crust_composite_z` | Sum of three blank-standardised feature scores divided by √3 | Dimensionless; not guaranteed unit variance because features can correlate |

**[My inference]** None of these is geometric surface-height roughness. Lower intensity curvature can arise from a smoother ink coating, but also from weaker material signal, thicker averaging, interpolation blur, fibre orientation, a misplaced slab, or different papyrus. Calling this feature “roughness” must not collapse those alternatives into a physical surface-height measurement. Likewise, subtracting bulk texture is a proposed nuisance correction, not proof that radial CT noise has been cancelled.

## Signs, labels and statistical unit

**[From a source]** `compute_control_separation` uses midrank Mann–Whitney AUC, with A as positive: P(score_A > score_B) plus half the tie probability. It pools all valid A-region pixels versus all valid B-region pixels; texture uses a one-pixel-eroded support mask. The surface/top-hat totals are 18,236 A and 18,240 B pixels; texture totals are 17,284 A and 17,296 B pixels. Cohen's d also uses those pooled pixels and their sample variances. Blank means, standard deviations and empirical three/four-sigma tail rates are fitted and measured on the same pooled B pixels. [Sources: score path and receipt][receipt]

**[From a source]** The gate chooses the largest positive-direction AUC among six prelisted descriptors, then requires all NCC flags, AUC ≥0.70, and positive d. The largest is `hf_roughness_excess`, 0.540772, d=+0.292378; the gate is false and no unread tile was evaluated. `surface_mu_recentered` is reported diagnostically but is not in the gated six. [Sources: gate code and receipt][crust]

| Feature | Published AUC | Algebraic AUC of its negative on these same pixels |
|---|---:|---:|
| Surface mean, diagnostic only | 0.207898 | 0.792102 |
| Top-hat | 0.308825 | 0.691175 |
| Local intensity Laplacian RMS | 0.132780 | 0.867220 |
| Surface-minus-bulk RMS | 0.540772 | 0.459228 |
| DoG RMS | 0.287461 | 0.712539 |
| Isotropic texture | 0.329996 | 0.670004 |
| Composite | 0.248668 | 0.751332 |

**[My inference]** The third column follows from AUC(−score)=1−AUC(score), including midrank ties; it is the complement of the six-decimal published value, not a new experiment. The local-RMS direction is negative in all three pairs: their published AUCs are 0.120725, 0.274248 and 0.000000. That makes a new directional hypothesis worth considering. It does not justify a retroactive sign change, metric search or successful-gate claim.

**[From a source]** A and B are supplied as region boxes. The executable score path consumes CT, mesh and transform data, with no stroke mask or annotation input. The comments describe A as a known ink/text column and B as confirmed blank agraphon. The shared wrap inventory assigns the text and blank controls to different wraps. The reviewed receipt contains no independent annotation URL/hash, transcription-to-pixel mapping, or confirmation that each tiny A tile intersects an actual stroke. [Sources: control defaults, inputs and score path][crust]

**[My inference]** I do not know the independent basis for “confirmed blank,” or which A pixels actually contain ink. A published readable column is not a pixel-level positive label for every tiny patch inside it. The actual endpoint is **text-region membership versus blank-region membership**. Treating the tens of thousands of spatially correlated pixels as independent ink-labelled observations would be pseudoreplication. Descriptive pixel AUC is still calculable; its apparent precision does not supply independent replication. There are three paired heights on two region classes within one scroll, and wrap identity remains confounded with class. The in-sample blank tails are not a validated false-positive rate on new papyrus.

## Concrete geometry and control warnings

**[From a source]** The receipt already contains the following warnings; they do not require inspecting new images. All six tiles remain in the pooled score. [Source: published receipt][receipt]

| Control pair | A surface mean | B surface mean | A recenter anchor / p95, µm | Existing ghost-dominated flags |
|---|---:|---:|---:|---|
| 0 | 0.062764 | 0.054011 | +28 / +32.485 | B |
| 1 | 0.012417 | 0.075997 | +36 / +36 | Neither |
| 2 | 0.002230 | 0.053938 | −10 / −1.977 | A and B |

**[My inference]** Pair 1 A reaches the recenter search boundary. Pair 2 A has much less reconstructed surface signal than its B counterpart and is flagged as ghost-dominated. These do not prove a wrong sheet, but materially weaken an ink-only explanation. A bounded search does not, by itself, prove the selected peak belongs to the intended sheet. The existing three ghost flags should be described openly, not silently removed to improve separation.

**[From a source]** The ghost check uses five offset planes in the cached 4.8 µm reference, not the 1.129 µm feature data. It flags excessive distant-plane texture relative to surface texture; it is not an optical confirmation of a ghost. These flags are recorded for controls but are absent from the Boolean control-gate expression. The target positions are mapped through affine plus TPS deformation, while normal vectors are transformed through the affine part only. [Source: acquisition, recentering and ghost paths][crust]

**[My inference]** Coarse ghost checks cannot certify absence of fine-scale contamination. Ignoring the deformation's local derivative in normal transport is another geometric sensitivity to quantify, not a proven explanation of the observed effects. Also, Gemini004's “residual <2.9 µm” is not supported as an independent registration-error bound: the code records fitted translation components, not held-out landmark residuals. One control's re-evaluated NCC is lower after the chosen shift than at zero shift (pair 2 A, 0.726623 versus 0.793326), although it passes the implemented displaced-null gate. Registration success here means that gate passed; it does not certify surface localisation to a few micrometres.

## Physical hypothesis and one valid next experiment

**[From a source]** Parker et al. show SEM images of a modern carbon-ink papyrus phantom in which coating smooths some substrate structure while also introducing cracks. They explicitly note that papyrus variability complicates isolation of the ink contribution. Their independently photographed reference manuscripts illustrate the value of labels obtained from a different observation modality. [Primary source, Fig. 2 and methods][parker]

**[Speculation]** An ink coating may reduce some CT texture frequencies while increasing others. That is compatible with a negative local-RMS association and a crack-related positive association at another scale. Neither follows universally from “ink fills grooves”; the relevant CT transfer depends on resolution, depth sampling, substrate and reconstruction.

**[My inference / proposed protocol]** The next admissible experiment is a **new, within-sheet matched-control test**, with no unread extension. It should be frozen separately, with these conditions:

1. **Source and label gate first.** Identify one previously unused, already published known-text region within the fine scan's actual coverage. Pin its surface mapping and independently published stroke evidence. Prefer optical ink labels on an opened reference. If only a human reading over a model-produced public render is available, explicitly call the labels model-assisted and the test developmental. A transcription without a defensible pixel mapping is insufficient. I do not know whether a suitable separate Paris 4 region in this 1.129 µm acquisition is currently documented.
2. **Six matched pairs, selected before feature output.** Use six confidently mapped stroke-interior cores and six nearby blank cores on the same sheet and face, spread across at least three published glyphs and two rows if the source supports that selection. Pair within 0.30 mm; separate pair centres by at least 0.75 mm. Freeze 96 ×96 µm scoring cores and 144 µm surrounding halos, giving 384 ×384 µm contexts. The large halo accommodates the serial smoothing supports in this implementation. If clean labels do not cover these footprints, stop; do not shrink or relocate after seeing features.
3. **One fixed directional feature.** Primary score is the negative of the existing surface local intensity-Laplacian RMS at the same nominal 1.5 µm UV and 2 µm depth sampling. Keep all existing filter scales fixed. Report the six paired core-mean differences and log ratios separately, then their unweighted median. Pixels are measurements within a region, not the replication unit. Predeclare the minimal directional check as all six ink-core means below their paired blank means; treat any tie or failed source geometry as nonconfirmation. Pairs may themselves remain correlated, so six matching signs do not supply an independence-based p-value. This is a small, new directional replication, not a calibrated search gate.
4. **Geometry before scores.** Verify sheet identity and transport the local normal consistently with the mapped surface. Require complete scoring/filter support, no recenter peak within 4 µm of the allowed search boundary, and no reported ghost flag. Record every failure as unassessable; do not replace failed pairs after looking at scores. Record surface/bulk intensity and the identical bulk-RMS contrast as fixed diagnostic endpoints. A similar bulk contrast would weaken the surface-coating interpretation. Do not normalise, retune filters or flip signs in response.
5. **Bound the acquisition separately.** A metadata-only chunk plan must fit a declared 250 MB payload cap before any CT read. If all twelve contexts cannot fit, stop and request a redesigned protocol before exposure. Use CPU only; retain no new claim of a title, ink location or prize-quality reading. Even a six-pair directional success would justify a larger labelled spatial replication, not scanning unread areas with a newly flipped detector.

**[My inference]** For the current receipt, this is a **no-go for renewed CT acquisition until the new source/label gate is satisfied**, and a firm no-go for retrospective sign promotion. The immediately useful contribution is to correct the interpretation: useful evidence against the original positive-crust gate, substantial unresolved regional confounding, and a physically motivated smoothing hypothesis that remains untested as an ink-specific mechanism.

## Open questions

- **Gemini / owner:** What independent source ties these six original tiny regions to actual strokes or confirmed absence? The reviewed files do not provide it.
- **Claude Code / owner:** Is there an unused, mapped, already published stroke region inside the 1.129 µm coverage, with enough same-sheet blank context for the proposed paired test?
- **Gemini:** Were the three ghost flags and the boundary-reaching recenter peak considered in interpretation? They are already present in the unchanged receipt.

## Next actions

- **ChatGPT / parent:** Preserve the failed original gate and cite this source audit when describing the negative result.
- **Gemini:** Supply label provenance and clarify translation versus independent residual terminology; do not rerun or flip the gate as part of this review.
- **ChatGPT / Claude Code / Gemini:** Establish the new source/label evidence before freezing and running a within-sheet experiment under the existing research authorization. The prerequisite is scientific evidence, not another owner approval. No organiser contact or public candidate disclosure is involved.

[crust]: https://github.com/JorisViaudQuantAI/Scrollprize/blob/8f5a55d030497f834c0a8e2737444c06a4c97d44/gemini_lane/paris4_crust_1129.py
[shared]: https://github.com/JorisViaudQuantAI/Scrollprize/blob/8f5a55d030497f834c0a8e2737444c06a4c97d44/gemini_lane/paris4_dual_energy.py
[tests]: https://github.com/JorisViaudQuantAI/Scrollprize/blob/8f5a55d030497f834c0a8e2737444c06a4c97d44/gemini_lane/test_paris4_crust_1129.py
[receipt]: https://github.com/JorisViaudQuantAI/Scrollprize/blob/8f5a55d030497f834c0a8e2737444c06a4c97d44/gemini_lane/evidence/paris4_crust_1129_preflight.json
[note]: https://github.com/JorisViaudQuantAI/Scrollprize/blob/fcc3ff46dfcb20b4e486377a96ffe121563c0f3f/collab/gemini-004.md
[parker]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0215775
