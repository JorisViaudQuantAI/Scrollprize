# Transversal research: experiments that can change the next decision

22 November 2026. ChatGPT, with independent learning, imaging and experimental-design reviews.

**Claim convention.** **From a source** identifies observations in linked primary material or this team's measured receipts. **My inference** identifies proposed methods and interpretations. **Speculation** identifies mechanisms with no demonstrated benefit here. Every budget and promotion threshold below is a proposed stopping rule, not a measured runtime, expected gain, statistical guarantee or prize requirement. I do not know the probability that any approach will recover qualifying text.

Measured CPU outcomes are in [ChatGPT 010](../../collab/chatgpt-010.md); exact recipes and receipts are linked from the [reproduction guide](README.md). The optical pilot and the published-306B map pilot are separate measurements. Neither is new CT ink detection on an unread scroll.

## What changes the research priority

**From a source.** The [306B winning method, section 8](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/docs/METHOD.md) reports that frozen DinoVol features were at chance for ink, deconvolution invented strokes, and several adaptation variants did not improve its difficult line. Those are results of that team's experiments, not impossibility theorems. The [Hecate release](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/README.md) already uses fine-to-coarse distillation, native coarse inputs, and blur/contrast augmentation.

**My inference.** Another generic self-supervision proposal or another round of blur augmentation adds little without identifying a specific failure. Ask whether information is lost at acquisition, surface placement, feature extraction, depth aggregation, or review ranking. A cheap diagnostic that distinguishes these is worth more than an attractive new model name.

**From a source.** [Claude 009](../../collab/claude-009.md) already owns the weak-row Studio comparison and scalar CT-feature screen. [Claude 008](../../collab/claude-008.md) reports an artifact that survived several prediction checks but failed anatomical sheet-identity inspection. These packages complement those lanes.

### Priority order — my inference

| Priority | Package | First bounded experiment | What would justify more work? |
|---|---|---|---|
| 1 | Probe Hecate before depth aggregation | At most 256 donor contexts; 30 CPU minutes or one T4 hour | Better locked donor ranking at a fixed background operating point |
| 2 | Compare a virtual coarse scan with a real coarse scan | Six public 0841 patches; 1 GB fetch / two CPU hours | A reproducible native-versus-synthetic mismatch that an explicit corruption model explains |
| 3 | Separate CT depth-profile nuisance from residual signal | Twelve donor blocks; one CPU hour on cached profiles | Residual or removed components contribute beyond Claude's fixed raw features |
| 4 | Preserve multiple review routes | One 20-object public-control review batch | Weak-row recovery without displacing strong evidence or exceeding a fixed workload |
| 5 | Condition existing models on anatomy | Cached maps from at least twelve parents / three donors | Better held-out-donor utility than the best fixed expert |
| 6 | Propagate geometry uncertainty | Three coherent offsets on public controls | Better weak-text sensitivity without accepting known sheet-jump artifacts |
| 7 | Source-anchored adaptation | One readable-donor emulation; 300 updates / one T4 hour | Transfer gain with source performance preserved |
| 8 | Controlled phantom data | Availability/provenance check first | Usable independent writing/CT correspondence, not just illustrative images |
| 9 | Energy-dependent Paris 4 residual | One public 4 by 4 mm calibration patch | Residual survives native coverage, registration, blur and misregistration controls |
| 10 | Transplant measured CT residuals for training | Only after a donor CT signal passes; one T4 hour | Gain on real held-out CT against two sham augmentation arms |
| 11 | Independent projection reconstructions | Establish raw-data availability first | Measured noise independence and retained weak text |

## 1. Probe an ink specialist before it combines depth

**From a source.** Hecate exposes volumetric logits and decoder features before its learned depth aggregation. Its coarse checkpoint requires 16 by 64 by 64 input samples at 9.6 micrometres in all axes. The [pinned implementation](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/hecate.py) provides the relevant feature and 3D-output interfaces. The [public Studio donor package](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data) distinguishes unknown, background and ink annotations; its native tiles have different physical sampling across donors.

**Hypothesis — speculation.** Useful weak-ink information may survive in the representation but be suppressed by the learned readout or depth combination. This tests an identified boundary inside an ink-trained model. It is different from expecting generic frozen CT features to discover ink automatically.

**Experiment — my inference.** Use PHerc. 0841 and 0009B for eight training parents and four calibration parents; reserve four parents on PHerc. 0814 for the locked test. Select them before new scores. Each parent is at least 256 by 256 working pixels; take a fixed four-by-four grid of 64-pixel contexts and score only each central 32-pixel square. This makes 256 contexts and deliberately incomplete scored coverage. Render exact 9.6-micrometre XYZ support; relabeling an 8.64 or 9.36-micrometre array is invalid. Stop if geometry, labels and support cannot be established within 1 GiB of downloads.

The radiometric preflight must also establish the released uint8 CT representation. **From a source:** the pinned Hecate card specifies uint8 input and division by 255 for the coarse model; its feature API expects already normalized tensors. **My inference:** document any donor conversion and divide by 255 exactly once, with no per-image min-max scaling. Stop if the mapping from native data is unknown. Correct voxel spacing cannot repair an undocumented intensity transformation.

Freeze the backbone, including normalization running statistics. Compare three readouts: original 2D output with scalar donor calibration; a single global nonnegative sum-to-one weighting of the sixteen depth logits; and L2 logistic regression on the final decoder features averaged over the fixed central four planes. Assert actual tensor dimensions from the pinned code. Do not select a depth independently at every pixel. For the feature head, use training-only StandardScaler, C=1 and seed 20261122, with at most 50,000 explicitly labeled pixels per class balanced across parents. Unknown labels stay unknown.

Report every test parent, AP, AUROC, and recall at a threshold selected once to admit at most 1% of labeled calibration background. A provisional resource gate is AP improvement on at least three of four test parents, macro AP gain at least 0.02, held-out labeled-background false positives at most 2%, and no strong-control deterioration at the same review budget. These thresholds are choices, not significance claims. The later published 306B rows can be a transfer diagnostic after freezing; their boxes are not ink-pixel truth.

**Cost / failure / expected gain — my inference.** Cap extraction at thirty CPU minutes or one T4 hour and the readout at ten CPU minutes. Gain is unknown. False positives can come from neighboring-sheet ink, donor-label bias, or linear separation of fibers from selectively annotated backgrounds. New-head holdout is not necessarily backbone holdout: establish the frozen model's training lineage. A negative probe rejects this readout, not the possibility of ink in the CT. **Owner: Claude Code**, with Codex validating the physical support.

## 2. Build a measured virtual scanner

**From a source.** The [0841 catalogue](https://scrollprize.org/data_browser/PHerc0841) lists 2.403-micrometre CT at 77 keV and 9.366-micrometre CT at 113 keV. Its fine ink renders are model predictions. The [official open-problems account](https://scrollprize.org/2026_open_problems) describes local acquisition degradation, including compressed-fiber haze; nominal voxel size alone does not describe the imaging conditions.

**Hypothesis — speculation.** A model trained on downsampled fine scans may see the wrong kind of coarse image. Real coarse CT can differ through blur direction, phase effects, correlated texture and intensity response. The useful contribution is measuring this difference, then testing whether matching it helps.

**Experiment — my inference.** Preselect six disjoint 2 by 2 mm patches on published 0841 surfaces: three calibration and three untouched test patches, including weak text and fiber/crack-rich material. Compare native fine CT, fine CT blurred and sampled onto the coarse lattice, and actual coarse CT. Codex must verify native nonfill support and the same physical interface. Hold out five anatomical landmarks; use less than one coarse voxel registration error as an initial rejection gate, and report normal and tangential errors separately. This gate is proposed and does not certify ink-pixel alignment.

Fit only a small anisotropic blur, affine intensity and correlated-noise family from calibration anatomy. Freeze its parameters before test patches. Reuse Claude's fixed CT features or a frozen detector; do not add a simultaneous feature search. If an adaptation fit follows, compare unchanged inputs, generic blur/contrast and the measured corruption with identical labels, training samples and update budgets.

If native fine fails, stop coarse adaptation. If fine succeeds while synthetic and native coarse both fail, this representation has not demonstrated recoverable coarse information; the result does not prove that the original CT contains none. If synthetic coarse succeeds and actual coarse fails beyond the registration/noise envelope, the simulator is inadequate. Require a proposed learned correction to improve actual held-out coarse data over generic corruption. Teacher-made masks make this an agreement/mechanism experiment, not independent truth.

**Cost / failure / expected gain — my inference.** Fetch at most 1 GB; two CPU hours for the diagnostic, at most one T4 hour for a justified head update. Gain unknown. Blur matching can erase faint signal or make cracks resemble teacher positives. This bridge remains worthwhile despite the mixed optical pilot: optical height and native CT are different measurements. **Owners: Codex** for geometry and **Claude Code** for frozen inference.

## 3. Treat the depth profile like a short spectrum

**My inference — mechanism.** At each surface point, keep the whole CT profile x(z). A small surface displacement changes a smooth reference profile approximately as b(z-delta) = b(z) - delta b'(z). Thus displacement can look like a material anomaly. Fit a donor reference basis B containing a constant, b and its derivative; compare original profiles, residuals x minus B times its least-squares coefficients, and the removed coefficients themselves. Check basis rank and conditioning. Learn the basis only from separate calibration tissue; applying that frozen linear operator is inference.

**From a source.** [Robust PCA](https://arxiv.org/abs/0912.3599) needs assumptions that identify low-rank and sparse components. It does not establish that fibers are low rank and ink is sparse. That shortcut is especially unsafe when cracks are sparse too.

**Experiment — my inference.** Reuse twelve independent public donor blocks: four to define the nuisance basis, four for any readout/calibration, four for testing. Use the same labels, support and depth orientation for all arms. Compare Claude's fixed scalar feature set with that set plus residual energy, residual shape and the three removed coefficients. Use a small fixed logistic readout and the same held-out operating-point gate as package 1. One hour on cached profiles; zero extra backbone passes.

**Main failure / expected gain — my inference.** Ink microrelief may itself resemble b'. Projecting it away could remove the desired signal. Therefore always retain the removed component and test it separately. A residual that looks cleaner but loses the weak control is a failure. This may disentangle geometry from material contrast; the gain is unknown. The normalized-map CPU pilot tests a related ranking idea, not this raw-CT mechanism. **Owners: Claude Code**, with **ChatGPT** interpreting the ablation.

## 4. Borrow experimental allocation from active search

**From a source — our measurements.** In [the 306B support-amendment pilot](evidence/nuisance_v2_summary.json), local normalization advanced the weak row but harmed strong-row top-five coverage and exhausted the search limit. In [the optical pilot](evidence/profilometer_results.json), scale matching helped the top-one-percent tail while AP had opposite directions in the two regions.

**My inference.** These results suggest preserving complementary routes into the review queue. They do not establish a better fused ranker. Define the operating objective first: finding a small useful review set differs from producing a good full ink map.

**New, unrun experiment — my inference.** On new public parent blocks, evaluate a fixed twenty-object batch: ten from the established route, at most five from a locally normalized or raw-profile route, and five systematic samples across predeclared geometry/quality strata. Deduplicate objects by physical footprint before counting slots; fill exhausted queues using a fixed deterministic rule. Keep raw evidence next to every transformed view. This is an explicit exploration allocation, not uncertainty sampling by whichever model is most confident.

Compare at the same twenty-object budget with the existing ranking and systematic sampling alone. Report each weak/strong control, readable strokes under blinded review if available, matched-artifact burden, unsupported area and total search trials. A method that only improves a global AUC cannot earn more slots. Do not deploy this revised policy on unread material because of the reused 306B example alone. An admissible follow-up needs a new physical public control with adequate context.

**Cost / failure / expected gain — my inference.** Thirty CPU minutes on cached maps plus a small fixed human review; benefit unknown. Exploration can waste slots and related models can share confident artifacts. This is a way to test whether complementary evidence helps under a fixed budget, not a calibrated bandit policy. **Owners: ChatGPT and Claude Code.**

## 5. Choose model weights from anatomy, not confidence

**Hypothesis — speculation.** Reader, Hecate and Studio may be useful under different measurable imaging conditions. A small anatomy-based gate might transfer better than taking their most confident response.

**Experiment — my inference.** Require at least twelve independently selected parents across three readable donors and use leave-one-donor-out development. Reuse cached model maps. Fit a tiny gate on at most six non-ink covariates: actual voxel pitch, sheet-normal contrast, adjacent-sheet separation, surface-fit residual, fiber anisotropy and valid-support fraction. It supplies one nonnegative sum-to-one weight vector per parent, not letter-shaped pixelwise weights. Fit expert calibration and gating inside the training folds. Compare the best fixed expert and an equal-weight mixture.

**Cost / failure / expected gain — my inference.** Thirty CPU minutes after maps exist; gain unknown. The gate may memorize donor identity or favor fiber-responsive experts. Do not fit it if the independent donor diversity is unavailable. Stop unless it improves an untouched donor at the frozen background operating point. Votes from models with shared training are not independent confirmation. **Owner: Claude Code; ChatGPT** can implement the small gate when the donor manifest exists.

## 6. Marginalize uncertain geometry instead of selecting its best ink

**Hypothesis — speculation.** A family of plausible surface locations may be more useful than one uncertain surface. Optimizing the surface to maximize apparent ink can manufacture its own confirmation.

**Experiment — my inference.** On public donor controls, use three coherent offsets of the same anatomically established sheet: -2, 0 and +2 native voxels. Fix weights from anatomy alone, or use one-third weights if no justified anatomy likelihood is available. Combine them as a fixed expectation and retain the disagreement map. Do not take a per-pixel maximum or independently move each vertex toward the strongest prediction. Include the team's known public artifacts and real sheet-interface checks.

Compare weak-text retention and artifact burden against the fixed central surface. The purpose is uncertainty propagation, not reviving the rejected density-peak recentering method. Existing -3/+3 averaged maps cannot stand in for the three individual conditions. If an eventual joint geometry/ink optimizer is tried, rerun the entire optimization under every null intervention; testing a frozen final surface after selecting it for ink understates false positives.

**Cost / failure / expected gain — my inference.** One CPU hour with existing separate maps, or three bounded donor inference passes. Gain unknown. Averaging can erase real faint ink; a consistent wrong sheet can survive all offsets. It therefore cannot replace Codex's sheet-identity check. **Owner: Codex**, with **Claude Code** supplying individually preserved predictions.

## 7. Conservative adaptation on separate tissue

**Hypothesis — speculation.** A small change in feature scale/bias may correct appearance mismatch without requiring full model retraining. It cannot create an absent measurement.

**Experiment — my inference.** First emulate an unseen scroll using a readable donor. Take 64 anatomy-selected unlabeled contexts in region A and lock a physically separate labeled region B. Freeze convolutions and normalization running statistics. If the chosen architecture supports it, update only the affine scale/bias of the first two encoder normalization layers, with scale changes bounded to ten percent and shifts to 0.1 source-feature standard deviations. Use supervised source examples and target consistency in a fixed two-to-one sampling ratio, with a penalty toward original parameters. No entropy-minimization objective. Fix 300 steps, batch one, AdamW at 0.0001, one seed and a one-hour T4 cap; stop after a ten-minute feasibility failure.

**Failure / expected gain — my inference.** Consistency can reward blank maps and stable cracks. Stop if source AP drops by more than 0.01 or if consistency improves without benefit on locked B. New tuning converts B into development data. Using A from an eventual target also makes its physical support training exposure, including rescans and receptive-field halos; prefer donors for the first experiment. Gain unknown. **Owner: Claude Code**, after the probe and native-scan diagnosis.

## 8. Use real physical phantoms before trusting synthetic letters

**From a source.** [Seiler et al., September 2026](https://doi.org/10.1371/journal.pone.0353485) read deliberately leaded writing in a modern carbonized-papyrus model using laboratory CT and tested XRF detection. They declare [a NIST dataset](https://doi.org/10.18434/mds2-4227). This is not ancient-scroll recovery. The 30 kV tube acquisition is different from the challenge's high-energy scans.

**My inference.** A real written-and-carbonized phantom can test geometry, reconstruction artifacts and known writing without using a model's output as the answer key. It is potentially a better falsification source than computer-drawn Greek letters pasted onto papyrus.

**Availability gate — from a source / my inference.** [DataCite metadata](https://api.datacite.org/dois/10.18434/mds2-4227) describes the deposit, but NIST returned service errors during this session. File contents, aligned ink masks and raw projection availability were not verified. First check the archive and provenance; do not promise a CT benchmark before doing so. If usable, choose one writing/control pair before model outputs, reproduce the original view, and test one fixed detector and one artifact intervention within a one-GB/two-CPU-hour cap.

**Failure / expected gain — my inference.** High-contrast leaded modern ink may be an easy and irrelevant task for ancient carbon ink. It cannot establish title chemistry or unread-scroll sensitivity. Handheld XRF selection requires physical access we do not have. Gain unknown; this is a later donor-data opportunity. **Owner: ChatGPT** for the availability/format check; **Codex** only if usable geometry is present.

## 9. Energy residuals for Paris 4: a high-risk physical hypothesis

**From a source.** Lead was associated with ink in two ancient fragments in [Brun et al.](https://doi.org/10.1073/pnas.1519958113). [NIST](https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z82.html) lists its K edge at 88.0045 keV. [Paris 4's catalogue](https://scrollprize.org/data_browser/PHercParis4) includes 2.4-micrometre scans at 78 and 137 keV. None establishes lead in the Paris 4 title.

**Hypothesis — speculation.** A material-sensitive residual between energies might reveal something a carbon-ink-trained detector misses. Uncalibrated subtraction of two reconstructions is not K-edge tomography: spectra, phase effects, blur and registration can dominate it.

**Experiment — my inference.** First find one public 4 by 4 mm body-text calibration patch with demonstrated native common coverage. Fit registration from anatomy and withhold five landmarks, using less than one native voxel as a proposed initial gate. Match blur and intensity on tissue without using ink predictions. Test residual specificity against published text and matched cracks/fibers; repeat with deliberate plus/minus one-voxel misregistrations. Stop if the desired residual is no better than those artifacts. One paired patch and one CPU day maximum before any title application.

**Failure / expected gain — my inference.** Misregistration naturally makes paired bright/dark edges. Coarse-pyramid nonfill is not native coverage, and UV-node overlap is not physical-area overlap. I do not know whether usable energy-pair coverage reaches a title or whether its ink differs chemically. **Owners: Codex** for support/registration, **Claude Code** for an already-published body control.

## 10. Ambitious: transplant measured 3D residuals across backgrounds

**Hypothesis — speculation.** A detector may learn donor-specific fibers rather than ink. Training with ink-associated residuals on different papyrus backgrounds could weaken that shortcut.

**Experiment — my inference.** Only after a real donor CT signal has passed, estimate a smooth background on training donors and retain a thin 3D residual under independent ink labels. Place it at a consistent sheet interface in different explicitly annotated background. Compare four fixed arms: genuine residual, equal-energy non-ink residual, phase-scrambled residual and no augmentation. Fit the same small head with equal data/update budgets and evaluate untouched real CT. This is training augmentation; synthetic images never become evidence or a submission render.

**Cost / failure / expected gain — my inference.** At most two CPU days to make the residual definition explicit and one T4 hour for the controlled fit. Gain unknown; feasibility uncertain. The same ancient papyrus with and without ink is unavailable, so the residual may be a fiber, subtraction artifact or seam. Kill the idea unless the genuine arm beats both sham arms and no augmentation on real held-out tissue. A readable synthetic letter is not a successful endpoint. **Owner: ChatGPT** for the method, **Claude Code** only after prior gates pass.

## 11. Ambitious: two independent reconstructions instead of adjacent slices

**From a source.** [Noise2Self](https://proceedings.mlr.press/v97/batson19a.html) relies on an appropriate independence structure in the measurement noise. Adjacent reconstructed slices are not asserted to satisfy that assumption by the paper.

**Hypothesis — speculation.** If raw projections are public, disjoint angular subsets could supply two measurements of the same small donor volume. They may support a denoising diagnostic with stronger physical grounding than treating neighboring surface layers as repeat observations.

**Experiment — my inference.** Establish projection/calibration availability first. Reconstruct a small public labeled donor from interleaved angular halves; match transfer functions and measure background noise cross-correlation. Compare a fixed denoiser with the full-data reconstruction at the same total exposure, using separate training/test tissue. Angular undersampling introduces its own artifacts, so split projections do not automatically guarantee independent noise. Stop if only visual smoothness improves or weak text disappears. I do not know whether the required raw data are available; this is not a current GPU task. **Owner: ChatGPT** for data discovery, **Codex** for any later reconstruction feasibility check.

## Shared validation and contest contract

**From a source.** [First Letters rules](https://scrollprize.org/prizes) prohibit training/prediction overlap and require held-out validation. A prohibition on higher-resolution data from the submitted volume appears in the Grand Prize section; the Paris 4 title section expressly allows its published scans including 2.4 micrometres. I do not know whether organisers intend the Grand Prize clause to govern every First Letters adaptation scenario.

**My inference.** Track physical source footprints for base models, pseudo-labels, fitted preprocessing, gates and readouts. Donor-only new training is not proof that an inherited checkpoint is unexposed. Exclude the same tissue in every rescan, all shifted depth slabs, augmentation/input halos and registration uncertainty. A provisional five-mm parent separation must be enlarged when those supports require it; it is not a universal safe distance. Keep learning off intended prediction tissue. No external clarification or submission is sent by this work.

**My inference — minimum utility test.** AUC at least 0.60 is insufficient. For example, give 100 positives score 1, 600 negatives score 0 and 400 artifacts score 2: AUC is 0.60, yet the first 400 objects are artifacts. Also, mean, variance, maximum and absolute adjacent-depth variation are unchanged by depth reversal. Such a reversal tests order sensitivity, not absence of ink or the other physical face. Even an opposite interface needs matched controls and is not automatically blank.

Use spatially separated parents, weak/sparse positives, matched artifacts, explicit original support, and a frozen review budget. Preserve conditional and full-source coverage separately. Count all transforms, angles, depths, thresholds, failed fits and truncated searches. A null intervention is not a population false-positive estimate. Any test used to choose a new method becomes development data.

**My inference — prize value.** These ideas become a stronger contribution when they improve an actual donor render or ink result and plug into the team's existing pipeline. The immediate handoff is the first three packages. The more speculative packages remain recorded with stopping rules, rather than consuming the whole compute budget at once.

## Open questions

- **Claude Code:** Can the new head distinguish information present before depth aggregation from information absent in the representation, with a verified donor geometry/label manifest?
- **Codex:** Which public fine/coarse donor patches pass native support and local anatomical alignment without additional manual sheet tracing?
- **ChatGPT:** Does the NIST archive contain a usable CT/writing pair or raw projections when its service is available?
- **Owner:** Organiser interpretations, prize submission and disclosure remain your decisions; none is needed for the bounded public-donor diagnostics above.

## Next actions

- **Claude Code:** Finish the already-owned weak-row and CT-feature runs. Then take package 1 if the support/lineage preflight passes; stop at its cap.
- **Codex:** Supply the package-2 three-calibration/three-test native-support manifest and normal/tangential registration errors before any paired inference.
- **ChatGPT:** Review their measured outcomes, keep the failed and mixed CPU pilots visible, and implement only the next method whose data and control gates are satisfied.
- **Owner:** Review a concrete qualifying result if one emerges. This note neither submits an entry nor claims a prize.
