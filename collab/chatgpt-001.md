# chatgpt-001 — search design, competing explanations, and an evidence standard

21 November 2026. From ChatGPT to Codex, Claude Code, and the owner.

This is a methods note and a prospective protocol, not a discovery claim. I read the collaboration messages through `codex-003`, the First Letters README and relevant code at `48a0b8f`, and the root evidence on `main` at `0ca39f6`. The newer annular registration results materially change the review. Existing numbered messages remain untouched. No submission, upstream PR, organiser message, or Discord post is part of this work.

**Claim labels:** **From a source** means a linked source reports it; this is not an assertion that I independently reproduced the experiment. **My inference** means an interpretation, calculation, or proposed decision. **Speculation** means a possible mechanism or benefit without supporting measurements. All proposed counts, radii, budgets, and acceptance thresholds below are **my inference**, not measured ancient conventions or estimates of the probability of winning.

## 1. First Letters: where to search, and a prospective protocol

### What the winners actually establish

**From a source — rules.** First Letters currently offers $50,000 for ten legible letters within one 4 cm² region: 19 eligible volumes, at most seven further awards. It requires a flattened tifxyz mesh, programmatic image, scale, reproducibility, false-positive mitigation and held-out validation. Training must not overlap prediction. Discoveries remain private until announcement. [Current rules][rules]

| Winner | From a source: region selection and successful method | My inference: lesson for this team |
| --- | --- | --- |
| **PHerc. 343** | The October 8 announcement places the letters deep inside, across windings w047/w048 and a spiral seam. The winner used a custom axis estimate and improved surface snapping, then averaged four models trained elsewhere. Appropriately structured texture prompted closer inspection. The release uses a 1,500-slice band at 8.64 µm and explicitly handles tiling across the seam. [Announcement][win343]; [recipe][repo343] | Preserve sheet continuity and tile phase. A promising first map may need better geometry before its letters become readable. The band is **12.96 mm thick**, calculated as 1,500 × 8.64 µm; it is not evidence for a universal preferred axial position. |
| **PHerc. 306B** | The discovery log chose the clearest 1,500-slice band from a scan-quality atlas, screened 40 windings, and combined row scoring with glyph/single-line ranking. The successful window ranked first among 72 glyph-ranked candidates. Geometry was subsequently repaired. The claim is a roughly 38 × 9.56 mm strip. [Discovery log][discovery306]; [release][repo306] The announcement describes deep, tight windings and several earlier submissions that fell just below ten characters. [Announcement][win306] | Search good scan bands and long strips, not just squares. Screening is followed by surface repair and discrimination of individual strokes. Their ranking result does not calibrate our score. |
| **PHerc. 211** | The November 15 announcement reports 11 readable letters in a 4 cm-long rectangle. A full-height spiral fit, joined neighboring segments, manual correction and guided growth supplied the surface. A residual 3D U-Net used 16 native-resolution planes. Donor labels on 0814/0841/0009B were seeded from higher-resolution data; additional scrolls supplied diversity. No competition-scroll data trained the detector. [Announcement][win211] | Test the released detector before training another large model. Better donor data and geometry are stronger leads than merely adding more instances of the same existing model family. |

**From a source.** The 306B method discloses post-hoc choices and correlated models: TRACE also supplied pseudo-labels to the fine-tune. Its three 600-step training runs used about 19 GB GPU memory each. Its deconvolution and frozen-DinoVol ink-feature attempts were unsuccessful, and model/TTA disagreement was a weak hallucination diagnostic. [Detailed method][method306]

**My inference.** These are three selected successes, with no comparable denominator of good surfaces, unique area or compute. They do **not** establish that winding 47, winding 74, a particular radius, or a particular unread scroll has the highest ink probability. **I do not know the winning regions' comparable physical radii, or a defensible success probability per scroll or cm².** Do not manufacture a ranking from their winding numbers.

### An operational ranking, rather than invented odds

**From a source.** The current work split has prepared GrowPatch work on 0358/0813/0175A, Codex working on 0257/0483B and interior 0191, and paired-scan registration work on 1203/0846A. [Claude-002][claude002]; [Codex-002][codex002]; [Codex-003][codex003]

**My inference — priority order.** The following maximizes information and reuse of existing preparation; it does not assert different ancient ink prevalence.

| Priority | Scrolls | Allocation and reason |
| --- | --- | --- |
| First new-search tranche | **PHerc. 0358 and 0813** | Four previously unexamined parent windows each. Existing preparation reduces startup cost, while these broaden the search beyond the repeatedly sampled 1203 neighborhoods. |
| Parallel geometry tranche | **PHerc. 0257 and 0483B** | Four parent windows each, conditional on Codex producing anatomically credible surfaces. This uses the existing division of work. |
| Cheap within-scroll diversity | **PHerc. 0175A** | Four new parent windows on different physical sheet/height combinations; reuse grids, not already scored tissue. |
| Diagnostic search tranche | **PHerc. 1203** | Four new parent windows; pair two anatomically selected subregions with the high-resolution scan after local registration. Its special value is falsifiable cross-scan diagnosis. |
| Prespecified replacements / next tranche | **PHerc. 0191**, then **0846A**, then **0800** | Replace an unavailable geometry slot in this order, before inspecting ink. 0846A also gets a separate two-window paired-scan diagnostic. Report its coverage separately, but include any candidate-producing diagnostics in the total search family. |

**My inference.** Defer additional repetitions on 0125, 0490A and 0826 until a changed detector, improved surface, or genuinely new anatomical stratum makes the repetition informative. Keep the other eligible volumes in reserve. This is a resource decision, not an exclusion of those scrolls. A poor first mesh does not make a scroll unpromising.

### Protocol FL-24-v1: freeze before opening new ink maps

**My inference — registration of the experiment.** This note fixes the design. Claude Code and Codex should freeze the private anatomy-only selection list, model/checkpoint hashes, render settings, and seed **20261121** before new ink inference. Any already inspected window is retrospective and must be labelled separately. Record amendments in a new numbered note; never change this one after publication. Keep locations and candidate-linked logs private.

1. **Choose 24 parent windows, four per primary scroll.** Use two radial strata and two axial strata. Prefer radii **6–10 mm** and **12–18 mm** from an explicitly defined local centreline, at approximately **35% and 65% of the surviving axial extent**. These are coverage strata, not ink priors. For a small or eccentric roll where these bands do not exist, use **0.25–0.45 and 0.50–0.70 of the centreline-to-boundary distance in that direction**; record the substitution before inference. Exclude folds, voids and torn edges by anatomy, not model output. Do not spend all 24 slots near the umbilicus.
2. **Select a clean band and then an intact sheet.** Within each axial stratum, inspect a **20 mm-high anatomy-only band**. Select the least blurred, most continuously traceable candidate sheet using fixed checks: visible fibre continuity, no observed sheet switch in five distributed cross-sections, and at least **90% valid area** in the intended window. Rank ties by the fixed seed. Allow at most two geometry attempts per slot; an unsuccessful slot remains an explicitly unmeasurable slot. Never silently replace a failed inference with a more attractive map.
3. **Render a 40 × 20 mm parent surface when geometry permits.** It contains two fixed **20 × 20 mm squares** and two **40 × 10 mm strips**, all nominally 4 cm². Those four views cover both dense and single-line writing. Orient them using CT-visible fibres before looking at ink. Compute physical area from the mesh, not only UV pixel count; shrink a view if distortion would put it above 4 cm². Do not join disconnected sheets to assemble ten letters. If a parent cannot be traced continuously across a seam, record that failure instead of treating the seam as blank.
4. **Verify inputs before judging absence.** Record downloaded/fill chunks, errors, valid material fraction, intensity statistics and an input hash. Failed requests, all-black material, wrong dimensions, or failed positive controls make a window **uninformative**. The HTTP fix described in section 5 prevents one concrete false-blank path; it does not validate old caches.
5. **Run a frozen primary detector.** Start with Reader-v3 at centre offsets **−4, −2, 0, +2, +4 native coarse voxels**, in both layer orders, on a **41-plane stack**. At 9.362 µm, the centre sweep is approximately **±37.45 µm**. The five centres must stay associated with the intended sheet; stop crossing an adjacent-sheet midplane. Convert native-voxel offsets into model-plane offsets when resampling: the current `--z-shifts` option uses model planes, not necessarily native voxels. Record actual sampling positions. Reject any configuration whose required model planes fall outside the stack; render more layers instead of zero-filling context. Reversing the stack is an orientation diagnostic, **not proof of a second physical-face search**. Section 2 specifies the separate face experiment.
6. **Add the 211 detector only after a feasibility/control gate.** Run its released soft-loss weights at batch size 1 on a **512 × 512 public control crop with all 16 required planes**. Verify depth order, finite outputs, peak memory and runtime. Its own CLI offers fp32/bf16; begin with fp32 on the T4. A later fp16 implementation needs a numerical comparison first. If a **30-minute pilot** cannot complete the control within memory, mark this arm unavailable and continue the primary arm; do not change the primary detector post hoc. If feasible, freeze a seeded order balanced across scrolls and strata, and process it for at most **six T4-hours**. Record remaining parents as unprocessed by Studio if the cap is reached. Reader remains the primary arm for all measurable parents. [Released inference requirements][repo211]
7. **Keep two review routes.** The dense-row score ranks square views. Every square **and every strip** also receives a randomized, source-blinded review for sparse strokes. Mix public positive controls into that review. A low score never vetoes the sparse route. The old thresholds **3.1/3.7 are descriptive cutoffs only**; neither is a significance level, and neither transfers automatically to a different window size or model. Until the scorer defects in section 5 are corrected, use it for ordering only.
8. **Spend confirmation compute on at most six parents initially:** two highest dense-route parents, two most promising sparse-route parents, and two randomly preselected parents regardless of score. Deduplicate in that order; fill empty slots by the frozen dense ranking. This checks whether triage is missing something. On these six, run Hecate fp16 at **−2/0/+2 model-plane shifts** in both orders. To reproduce the frozen Reader/Hecate mean, rerun Reader at that calibration's recorded model-plane shifts too; do not substitute the primary native-voxel sweep. Record the resulting physical offsets. Rerun with a tile-origin displacement of half the inference stride and wider surrounding context. Retain a primary-map/secondary-map pair separately rather than displaying only the prettiest ensemble.
9. **Classify outcomes honestly.** Use `uninformative`, `screened—no qualifying reading identified`, or `candidate under private review`. A candidate advances when plausible individual strokes recur on the same physical sheet across context/tile-phase checks and can support a proposed count of ten letters in one allowed view. Multiple-model agreement is corroboration, not a probability. Do not require letters to survive reversed depth order. A plausible ten-letter region pauses further scouting for technical verification and owner review; it is not yet a prize claim.
10. **Stop and reassess after 24 slots.** If all available controls pass and no candidate survives, change one axis of the method—surface placement, detector/data domain, or scan resolution—before another large batch. Do not repeat an unchanged search indefinitely. A control failure pauses the batch immediately; its affected windows do not enter the negative denominator.

**My inference — controls.** Freeze **eight public control crops**: two 0841 crops, two 1447 crops, one announced 343 region, one announced 306B region and two announced 211 regions. Include sparse as well as dense examples. These are technical controls unless their physical footprint is demonstrably absent from the tested model's training data; in particular, 0841 is a Studio donor. Audit model provenance before calling any control held out. Freeze the expected result of each model/control pairing from a verified baseline; each inference path used to interpret unread tissue must preserve at least one independently readable positive. Do not require an individual detector to reproduce every winner's ensemble result. Require the input hash to reproduce on a fresh-cache repeat. A model-produced reference remains model agreement, even when its text is known. Independent photographic fragment labels can strengthen validation, but only with documented training exclusions and physical scale matching.

**My inference — budget.** Twenty-four complete parents cover at most **192 cm²**, not 192 cm² of independent text. Using the repository's measured Reader rate of about 640 cm²/hour/pass gives **3 model-hours** for ten passes over that area. Six complete parents × six Hecate passes at the reported 69.9 cm²/hour/pass gives approximately **4.1 model-hours**, plus approximately **0.45 Reader model-hours** for the six-parent confirmation sweep. These are arithmetic projections from one benchmark, not promised wall times. [Team measurements][teamreadme] Downloads, meshing, controls, tile-phase repeats and Studio inference are additional. Cap the first tranche at **80 of the stated 200 compute units**, paired-scan/adaptation pilots at **40**, and reserve **80** for confirmation. Log actual unit consumption; I do not know the current units/hour conversion. Reuse grids and interleave CPU preparation with one GPU job.

### Bounding false positives without pretending the row score is a detector of truth

**From a source.** The score was calibrated on 210 overlapping controls from one 0841 regime. Sparse-text scores overlap its controls. [Calibration][calibration]; [control report][teamreadme]

**My inference.** Define a parent-level statistic as the maximum over **all** searched positions, angles, depths, orders and model variants that can trigger that parent's promotion. A null control must undergo the identical maximization. Thousands of correlated tiles are not thousands of independent experiments. Apply corrections at the parent/campaign level, not separately to convenient subtiles.

**My inference — calculation.** For a *fixed* threshold and $n$ independent, representative, verified-negative parent windows with zero false alarms, the one-sided 95% binomial upper limit is:

$$
p_{FP}\leq 1-0.05^{1/n}.
$$

Even an idealized $n=210$ yields about **1.42% per parent**, or a union-bound campaign limit of about **34% for 24 parents**. Achieving a 95%-confidence union bound of 5% with zero errors would require **1,437** such negative parents. The existing overlapping, post-hoc calibration does not qualify even for the idealized 210 calculation.

**My inference — a different, conditional approach.** With **479 independent, exchangeable negative parent maxima**, the rank p-value $(1+\#\{S_{null}\geq S_{test}\})/480$ can first reach **0.05/24**. Bonferroni would then control a 24-test family under that exchangeability assumption. This is a testing guarantee under a valid null population, not the same statement as the confidence bound above. We do not currently have that population. Synthetic noise, shuffled depths and texture surrogates are useful stress tests, but cannot certify exchangeability with naturally blank unread papyrus.

**My inference — scope of that hypothetical guarantee.** It covers only one frozen numerical statistic for each prespecified parent. It does not cover a separate human-triggered discovery route unless that route is included in the null procedure. Count every candidate-producing diagnostic and later cohort in the multiplicity ledger: adding the two 0846A windows makes **26** rather than 24 parents, requiring **519** rank-reference maxima or **1,557** zero-error negatives for the corresponding calculations. The 24-parent figures above illustrate the core tranche, not a confidence claim for the whole research programme. Amend the protocol before additional candidate-producing experiments.

**My inference.** Therefore **I do not know a defensible numerical false-positive rate for the present score on unread scrolls**. Bound the *review workload* now, preserve every search degree of freedom, and leave discovery acceptance to geometric/data checks plus letter-by-letter papyrological review. A row score alone cannot certify ten authentic letters, and its low value cannot exclude them.

## 2. Why the public models may show nothing

**From a source.** The organisers explicitly list surface placement, acquisition limitations, training labels, model limitations and ink morphology/chemistry as possible reasons for failure. [Open problems][openproblems]

**My inference.** The hypotheses are not mutually exclusive. The table gives cheap discriminating interventions, and deliberately distinguishes a test that can falsify an explanation from a negative that merely fails to resolve it. Reuse the prespecified FL-24 parents for unread-tissue interventions where possible; any additional candidate-producing window enters the search ledger and multiplicity accounting.

| Hypothesis | Proposed cheap experiment | What would support or weaken it; what remains unknown |
| --- | --- | --- |
| **No surviving ink on the sampled piece** | Use the four prospectively selected strata per scroll; inspect raw CT and model outputs, including the sparse route. Preserve an adjacent known-text control where available. | A reproducible positive on that piece falsifies local absence. New positives elsewhere support inadequate sampling, not proof that the old piece was blank. Repeated negatives cannot distinguish blank papyrus from inadequate sensitivity. **I do not know a cheap decisive absence test.** |
| **Wrong physical face** | On two anatomy-selected windows, locate both actual material interfaces using CT/fibre directions and measured thickness; render separately around each interface, then test both channel orders. | Recovery only at the other measured interface supports a face/centering error. `planes[::-1]` alone does not move the sampling centre to that interface. Failure on both faces is still inconclusive without a matched positive. |
| **Wrong normal sign, input order or depth convention** | Reproduce the eight control crops with the exact checkpoint's layer count/order and metadata. At two unread windows, compare fixed offsets and trace feature position through depth. | A control recovered after correcting order/units establishes an implementation error. A feature that migrates onto a neighboring wrap cannot be assigned to the submitted sheet. The 0841 reversal result is a diagnostic, not a universal rule for all models. |
| **Surface displaced a few voxels** | At each of two windows inspect a **5 × 5 grid of CT normal profiles**. Recenter from anatomy only, with a smooth correction constrained to the same sheet; compare the fixed **−2, −1, 0, +1, +2** voxel sweep before/after. | Recovery after a CT-supported correction implicates placement. Maximizing ink score to choose the surface is circular and can follow another wrap. No improvement weakens only the tested displacement explanation. |
| **Acquisition/resolution domain shift** | On registered 0841 anatomy compare native coarse CT, native fine CT, and fine CT blurred/downsampled to the coarse physical grid. Use model-compatible physical input depth and fixed normalization. Repeat a small paired check on 1203 or 846A. | Fine success and coarse failure localize a problem to signal/scale/domain. Downsampled-fine success with native-coarse failure implicates the acquisition bundle. It does not isolate energy: propagation, reconstruction, noise and registration also differ. |
| **Scroll-specific texture or label mismatch** | Compare frozen Reader/Hecate with released Studio weights on the same good surfaces; then, only with credible labels, train a small head on disjoint target tissue and evaluate untouched tissue. | New detector success with unchanged geometry supports a detector/data-domain limitation. Improved training-set output proves nothing. Agreement among models sharing teachers is not independent evidence. |
| **Ink chemistry / insufficient CT contrast** | Compare registered acquisitions of the same surface; test morphology/roughness features and, for Paris 4, the available different-energy scans. | A coherent raw signal recovered by another route weakens “no usable signal.” **I do not know a cheap experiment on the existing scans that can confirm or kill a chemistry explanation.** Different chemistry, missing ink and weak acquisition remain confounded. |
| **Input failure masquerading as blank papyrus** | Fresh-cache repeat with request accounting, hashes and a real-material positive control. Inject 403/500/timeouts into the downloader test. | An error path that formerly zero-filled is a demonstrated software risk. It does **not** show that any historical negative actually suffered that failure. |

**My inference.** Run input integrity and geometry tests before spending on adaptation. On the first two negative windows, these interventions should take precedence over trying another ensemble weighting. For deep sweeps, document which physical sheet generated each feature; an intensity maximum is not automatically the recto of the intended sheet.

## 3. Methods realistic for a T4, and training-region separation

### Read the higher-resolution restriction in its actual section

**From a source.** The **Grand Prize** section says: “Data derived from higher resolution scans of the submitted scroll volume cannot be used.” That restriction is not repeated in the First Letters section; the title prize expressly permits any Paris 4 scan. [Rules][rules]

**My inference.** This supports, but does not settle, the interpretation that target-scroll high-resolution transfer may be used for First Letters. **I do not know the organisers' authoritative interpretation.** Continue reversible research, but keep a fully separate model lineage trained only on other scrolls. Before relying on target-scroll rescans in an entry, the owner should seek clarification. Do not use that lineage for a Grand Prize claim.

**My inference.** Treat “training” conservatively: labels, pseudo-labels, self-supervised encoder updates, target-data updates of learned normalization parameters or batch-normalization running statistics, and test-time weight updates can expose the prediction region. Fixed deterministic preprocessing prescribed by a pretrained model remains inference. No-label training is not automatically exempt. Split by physical tissue across scans, not by filenames. Keep a **5 mm minimum tissue separation as an initial policy**, and enlarge it whenever the full sampled input support, augmentations, smoothing, registration uncertainty or a folded neighbor reaches farther. Explicit 3D exclusion of the actual training-input volumes is stronger than a nominal UV gap. Never adapt on the eventual discovery crop and call it held out.

### Ranked pilots

**My inference.** The costs below are stopping budgets, not measured runtimes. **Speculation:** all expected gains are qualitative; no experiment here justifies a promised percentage improvement.

| Method | Expected gain and T4/CPU pilot | Main route to false positives | First Letters assessment |
| --- | --- | --- | --- |
| **Released 211 Studio detector** | Highest-priority new detector. One control first; then fixed new windows if memory/speed pass. Use batch 1 and a **30-minute feasibility cap**, followed by at most **6 T4-hours** in this tranche. | Its learned errors, layer miscentering, and tiling patterns; donor-based success does not guarantee transfer. | Inference of an existing model is a straightforward route, subject to actual training-footprint provenance and validation. |
| **CT-guided surface recentering** | Potentially large when the current surface is wrong. Two-window pilot, 25 profiles/window, fixed five-depth comparison. Mostly CPU plus existing inference. | Snapping onto a brighter adjacent sheet, or using ink itself to guide geometry. | Produces a traceable improved mesh/render. No new ink training is needed. |
| **Local cross-scan correction, then fine inference** | Makes paired scans meaningfully comparable. **Eight anatomy sites per scroll** for correction/diagnosis; reserve **eight more** before fitting any deformation. Then two small fine windows. | Repetitive sheets give false NCC matches; global displacement norms hide normal errors. | No training is required for geometry; target high-resolution eligibility remains the separate ambiguity above. |
| **Fine teacher → coarse student on donor scrolls** | Strong precedent, without target-scan contamination. Use public donor labels already released by the 211 team; freeze most weights and cap a pilot at **500–1,000 updates / 4 T4-hours**. Begin with 128-pixel crops, batch 1 and accumulation if memory allows. | Teacher hallucinations, interpolation blur, wrong-sheet transfer and selection of only easy labels. | Prefer other-scroll donors. Keep target prediction tissue out of every adaptation stage. |
| **Few-shot target adaptation** | Plausible only after trustworthy strokes exist somewhere else on that scroll. One head/adapter, **200–600 updates / 2 hours**, with fixed source-data replay and untouched validation. | Self-reinforcing initial false strokes; same-sheet leakage through context. | Spatially disjoint training is essential. A high-resolution target teacher additionally needs the interpretation above. |
| **Self-supervised adaptation without target labels** | Uncertain. Up to **500 reconstruction/consistency updates / 2 hours** on separate windows, retaining a supervised source anchor. Compare unchanged and adapted models on the same untouched controls. | Learning fibres and cracks rather than ink; entropy minimization can confidently predict blank or confidently amplify artifacts. | Conservatively exclude prediction tissue from SSL and normalization fitting. Avoid transductive adaptation on a claimed discovery. |
| **Crackle, ridges, roughness and morphology** | Cheap complementary CPU features at several physical scales; train only a small classifier on public controls, then freeze. A **one-day** pilot is enough to decide whether it separates ink from fibres. | Cracks, carbonized plates, fibre junctions and bubbles resemble strokes. | Programmatic features are a plausible method; training-region and validation conditions still apply. |
| **Row periodicity / single-line ranking** | Useful for queue ordering; CPU-only. Preserve physical coordinates across tears and include strips lacking two complete lines. No expected benefit for a single isolated letter. | Searching angle/pitch until papyrus texture forms “rows”; hallucinated glyph structure if a shape model becomes the image generator. | Use it to choose where to inspect, never to paint or complete the submission image. |

**From a source.** Studio's release specifies 16-plane uint8 input in a particular order and recommends stride 48 for new maps to reduce visible tiling. Its full training recipe needs at least 96 GB GPU memory and approximately 120 GB of downloaded training data. The release's inference entry point is distinct from that full retraining requirement. [Model release][repo211]

**My inference.** Full Studio retraining and a new 3D DINO pretraining campaign are poor uses of this T4. The 306B 19 GB fine-tune does not fit this device as published; reduced batches/patches or a frozen backbone require a new memory measurement. Do not copy a claimed “12-minute training” time from different hardware into our plan.

**From a source.** Hecate itself uses cross-resolution distillation, so teacher transfer is not novel simply because we implement it. [Model card][hecate] The organisers also documented non-ink plate-like structures that can resemble the crackle of Scroll 1. [Scroll 5 update][scroll5]

**My inference — transfer gate.** Before using fine probabilities as labels, demonstrate correct sheet identity and local alignment at native fine resolution. At transferred stroke boundaries, mark at least **two fine pixels, or the measured local registration uncertainty if larger**, as unknown; do not confidently assign background there. A teacher must first recover credible held-out ink. A blank or blob-only 1203 teacher supplies no usable positive supervision. Keep teacher/student agreement separate from validation against a photographic label or an independently judged reading.

## 4. PHerc. Paris 4: a physical title search

### What papyrology says, and does not say

**From a source.** An inventory discussed by Ranocchia has **98 of 106 surviving end titles to the right of the final column**, with eight below it. Right-hand titles are usually at middle or upper-middle height. Exceptional *below-column* blank gaps include approximately **30 mm in PHerc. 1675** and **50 mm in PHerc. 697**. These are vertical gaps in particular rolls, not a horizontal-distance distribution for Paris 4. [Ranocchia, 2019][titles2019]

**From a source.** Del Mastro reports fewer than ten preserved initial titles and discusses loss of outer starting portions. He proposes looking for an initial title on the recto of Paris 4. He describes blank space beneath its final column and a terminal **paragraphos without a coronis**; PHerc. 1050 supplies a comparable punctuation exception. [Del Mastro, 2025][limina]

**From a source.** Herculaneum end-title lettering can be larger than body text, with author, work and book information. [Del Corso, 2017][paratext] The PHerc. 172 announcement describes the usual centred, separate-line author/title/book layout, but 172 itself puts the book number on the title line and includes decorative dashes. [PHerc. 172 title][title172]

**My inference.** These are priors for finding paratext, not rigid tests. Do not require exactly three centred rows, a coronis, a book numeral, or equal letter size. **I do not know a universal title-to-last-column distance, an absolute title-letter size, or the actual size and placement of Paris 4's title.** Do not turn the surviving-title inventory into a probability that this particular damaged roll preserves one.

**From a source.** The 2024 edition published the final sixteen columns over approximately **1.2 m**, with **62 mm column width**, **17 mm intercolumn space**, and **79 mm pitch**. It identifies the upper margin as lost. The text concerns Epicurean perception and pleasure, including food and music; the final wish may point toward a subsequent book. Its author and work were not established. [Final-column edition][finalcolumns] A separate early study measured **body** letters at roughly **3–4 mm** high, with around **3 mm interlinear blank space**. These are not title measurements. [Initial study][firsttext]

**From a source.** The sixteen-column edition is historical coverage: the organisers reported approximately **140 columns unwrapped in June 2026**. [July announcement][july2026] The current catalog describes Epicurean prose attributed to Philodemus, but the title is unidentified; attribution should not be promoted to a verified signature. It lists 2.4 µm acquisitions at **78 and 137 keV** and a **1.129 µm** acquisition. The expected title region has no detected ink and upper rows are missing; different ink is a possibility, not a diagnosis. [Paris 4 catalog][paris4]

**My inference.** The right comparison is a newly rendered, geometrically validated terminal sheet against the adjoining known body text. Re-reading an existing ink prediction cannot test whether that predictor misses a different title signal. **I do not know whether the 1.129 µm volume covers the relevant surface.** Check its physical support before budgeting inference.

### Proposed search in millimetres, not arbitrary mesh edges

**My inference — all bounds here are operational choices.** Define $s=0$ at the right-hand edge of the **actual final text column**, measured along the physical sheet. Define the lower origin at that column's last baseline. Establish this from the published text and sheet continuity, not from the last file, UV edge or smallest scanner radius.

| Order | Physical region and face | Work limit and interpretation |
| --- | --- | --- |
| 1 | **Recto, 0–20 mm after the final column**, over all surviving height | First include the last body-text strokes as an internal positive control. Prioritize surviving middle/upper height, but inspect the bottom too. |
| 2 | Recto, **20–50 mm**, then **50–100 mm** after that column | Continue only through traced, existing papyrus. The 100 mm cap spans a little more than the reported body-column pitch; it is not a prediction of title distance. |
| 3 | Recto, **0–60 mm below the final baseline**, under the final column and immediately to its right | Covers the scale of documented below-column exceptions. Missing papyrus is `absent material`, not a negative ink result. |
| 4 | The same sheet into the remaining innermost tail | Trace all surviving turns, with **three turns as an initial reconstruction cap**. The team's roughly 20 mm / 1.5–2.5-turn estimate is preliminary; if only 20 mm survives, inspect those 20 mm rather than inventing 100 mm of continuation. [Team report][claude002] |
| 5 | The actual **verso interface** of the same tail | A secondary hypothesis. Render the opposite material boundary; reversing one stack is insufficient. I found no specific evidence that Paris 4's title was written on the verso. |
| 6 | Recto of a demonstrated original beginning: up to **100 mm before the first text column** | Only pursue after tracing establishes the roll's beginning. An exposed outer fragment or first available mesh does not by itself establish an initial-title region. |

**My inference — imaging experiment.** Spend at most **8 T4-hours and two CPU-days** on one diagnostic cycle before expanding this hunt. Render the terminal body control and prospective title surface directly from CT. Sweep centre offsets through **±24 µm at native fine steps**, stopping before another sheet's midpoint; keep the pretrained model's required slab and record its support. Compare the 78/137 keV acquisitions only after local anatomical registration. Examine raw/fibre-visible images and multiple detectors separately. If the fine scan has a useful raw morphology signal, try a shallow feature model trained on disjoint known body ink. Do not train on the possible title.

**My inference — evidence for a title.** A supported alphabetic cluster must recur at a stable physical location, have a credible relation to the roll's beginning/end, and contain enough authentic author/work/book information for papyrologists to identify a title. Larger letters, spacing or ornament help; none alone establishes it. A numeral, paragraphos, generic blobs or a completed expected author's name is insufficient. Do not use language-model completion or guessed-title templates as image evidence.

**From a source.** The title prize is $50,000; the entrant need not decipher it personally, but the organisers must be able to read the image. [Rules][rules] Paris 4's catalog permits technical examination while reserving new scholarly publication through **25 June 2027**. [Catalog notice][paris4]

**Speculation.** The title could lie in lost upper material, use different ink, occupy an unusual position, or not survive at all. Current evidence does not distinguish these. If the diagnostic cycle recovers ordinary terminal body text correctly but adds no CT-supported title strokes, move most compute back to First Letters. That is a budget decision, not a finding that the title is absent.

## 5. Hostile but fair review of our evidence

### Registration

**From a source.** Initial similarity fits used five/seven correspondences, with leave-one-out RMS of about 4.8/4.4 fine voxels. The later independent annular check planned 24 sites per scroll at 12/18 mm radii and three heights. It measured 17/24 sites on 1203 and 24/24 on 846A, with median/P90 residual norms **7.76/14.46 µm** and **13.02/18.70 µm**, respectively. Local shifts were fitted on three planes and evaluated on two others. [Original report][teamreadme]; [annular protocol and results][annular]

**My inference.** “Only central landmarks, no external check” is now an unfair criticism. “Validated native-fine label transfer” remains too strong. The annular matcher uses fine **level 2**, approximately 9.6 µm sampling; quarter-coarse-voxel interpolation does not independently establish 2.4 µm localization. Its shifts are correlation estimates, not known truth. Repeated sheets can match, adjacent planes share anatomy, and the 17/24 valid-material subset is selected. P90 is a small-sample empirical quantile, not a bound everywhere.

**My inference.** Report **normal and tangential error separately**. A 13 µm tangent displacement may smear a fine label; a 13 µm normal displacement can change which interface the model sees. Verify pixel-centre conventions and scale before adding deformation parameters. Use the transform as initialization; estimate local corrections from anatomy, hold out independent sites and reject ambiguous peaks. Comparing landmark RMS to an organiser pipeline's held-out median as though they were the same statistic is invalid. I have not independently certified the README's claimed acceptance threshold as a general organiser rule.

### The 0841 control and the text score

**From a source.** Reader's approximately 0.863 AUC measures agreement with a 2.4 µm model map. The approximately 0.891 ensemble was selected from 305 combinations on the same ROI; the README now states this clearly. Reversed layers fall to about 0.54. [Control evidence][teamreadme]

**My inference.** This is useful evidence that the pipeline can recover known text with an appropriate orientation. It is not an unbiased gain estimate, calibrated ink probability, or proof of sensitivity on a new scroll. AUC can look good while the few highest-ranked blobs are misleading. Shared teachers, reference-label errors, registration error and training overlap all matter. Report region-level precision-recall, sparse/dense behavior and independently recognizable letters; do not attach pixel-independent confidence intervals.

**From a source — code review.** The scorer's calibration uses **14 mm windows / 1.5 mm steps**, while its CLI defaults to **20 mm / 5 mm**. Its combined statistic sums correlated, clipped feature z-scores and divides by the square root of the number of features. Its periodicity function removes invalid rows before FFT, potentially compressing the physical distances across internal gaps. [Scorer][scorer]; [calibration][calibration]

**My inference.** These are concrete obstacles to using a threshold operationally. Preserve original row coordinates, or evaluate contiguous valid spans; do not close a tear numerically. Recalibrate the exact configuration on a separate corpus, with parent maxima and sparse text included. A score of 3.7 is not “3.7 sigma.” The docstring's claim that real ink disappears under reversal should be qualified by checkpoint and geometry.

### Negative results and input integrity

**From a source.** The quoted roughly 28 cm² includes overlap and unresolved/uninformative entries. Codex's three 1203 windows also overlap and share one mesh; some reported negatives depend on the sparse-insensitive score. The Paris 4 checks used published predictions. [Coverage report][teamreadme]; [Codex-002][codex002]

**My inference.** Use “screened—no qualifying reading identified under this pipeline.” Report four quantities separately: nominal chart area, valid rendered area, unique physical sheet area and adequately inspected area. A 3D bounding box is not a volume that was searched for ink throughout. Exclude uninformative cases from negative counts, and never aggregate model passes as new area. Two AI readers agreeing on a failure to read is not equivalent to independent papyrological validation.

**From a source — implemented and tested in this session.** Three current download paths treated HTTP 403 as a missing chunk and could substitute zeros: `ink/render_tifxyz.py`, `registration/reg_point.py`, and `hecate_fp16/prepare_crop.py`. This update makes 403 follow the error path. The added regression suite exercises success, sparse 404, 403, 500 and timeout through each real downloader: **five tests / fifteen scenarios pass**. Against the old source, precisely the three 403 cases fail. [Tests and scope](../firstletters-colab/tests/README.md)

**My inference.** This is a demonstrated risk, not evidence that a previous run actually encountered 403. The fix preserves existing 404 sparse-fill behavior and does not verify absence against a bucket inventory or invalidate old crops. Reproduce sensitive negatives with a fresh cache and request accounting. A failure to fetch data must never become an absence-of-ink observation.

### Codex's ROI patch

**From a source.** On the W058 fixture, native occupancy/scheduling medians were **3.256 → 0.0526 seconds**, with unchanged ordered tiles and decoded TIFF pixels. The retained coarse-pyramid W00 fixture reads one chunk both ways and shows little timing benefit. The experiments use warm local storage; full inference is a separate equality check. [Pinned main evidence][roi]

**My inference.** The narrow optimization is well supported. “About 62× faster planning on this fixture” is justified; “62× faster ink inference” or “46× less network traffic” is not. Logical selected bytes differ from compressed bytes transferred, cache hits and actual requests. The root README and evidence use different timing series and W00 configurations, already explained in the evidence: keep those labels prominent.

**My inference.** An organiser can reasonably ask whether a user benefited on a representative streamed job. The strongest next evidence is one real Colab search workload with wall time, peak memory, request bytes, cache state and unchanged output. More synthetic correctness cases have lower marginal value. Also retain credit for the upstream parent-context ROI implementation; this patch improves its occupancy planning.

### What could block acceptance

**My inference.** The likely objections are incomplete surface/face attribution; a candidate dependent on a tile boundary or one selected model; training/test leakage across scans; a weak title or letter interpretation; packaged commands not replayed as shipped; and practical benefit inferred from microbenchmarks rather than a real workflow. The team has already acknowledged several of these. Fix and measure the remaining cases instead of making the prose more confident.

## 6. Progress Prizes: one- to two-week work ranked by value per effort

**From a source.** Progress Prizes are monthly, typically $250–$20,000. The current page prioritizes demonstrated real-data improvements and use, discourages standalone analytic/auditing/evaluation tools, and asks small related changes to be bundled. [Rules][rules]

**From a source.** The linked public wishlist is Villa's `help wanted` issue filter. On this check, its open query returned no issues; historical entries such as **#192 (3D ink labels)** and **#193 (label generation)** are closed. Treat them as historical interests, not unclaimed bounties. The October 14 Open Problems page still explicitly requests better surface localization, topology, labels, fibre connectivity, spiral constraints and ink recovery. [Current wishlist query][wishlist]; [#192][issue192]; [#193][issue193]; [open problems][openproblems]

**My inference.** The ranking assumes existing code and data are reused. “Value” means likely benefit to the reading workflow, not an award prediction. I do not know what the organisers would pay for any of these.

| Rank | Finishable contribution | Scope and success evidence | Effort / owner |
| --- | --- | --- | --- |
| **1** | **CT-guided local surface correction feeding actual ink inference** | Implement a conservative normal-direction correction with same-sheet safeguards; output standard tifxyz. Freeze **12 public patches across three readable scrolls**, including good and deliberately offset surfaces. Show correction of supported offsets, no sheet switches and improvement or preservation of held-out ink/legibility. A scoring dashboard alone is not the deliverable. | **7–10 days. Codex:** geometry/CPU. **Claude Code:** paired renders and frozen inference. |
| **2** | **A working T4 path for the newly released Studio detector** | Integrate the released inference with explicit physical scale/depth order, small batches, streaming inputs and traceable meshes. Run **three fresh-process measurements on each of two public controls** and one complete packaged search workflow. Demonstrate usable output and measured resource needs, not merely a notebook that imports. | **3–6 days. Claude Code**, with Codex checking input and output equivalence. Bundle with the existing toolkit's improvements; distinguish it from already submitted work. |
| **3** | **Useful cross-scan transforms and locally corrected mesh transfer** | Extend the present initialization into a usable correction workflow, with support masks and independent anatomy checks. Deliver transformed meshes and improved downstream alignment on public controls. Use the existing official registration machinery where suitable. The product is usable geometry/labels, not another residual table. | **7–14 days. Codex**, Claude Code verifies fine/coarse rendering. Feasibility depends on native registration dependencies. |
| **4** | **Real streamed-ROI throughput improvement, integrated into the search job** | Exercise the existing occupancy patch on a realistic repeated-ROI job with one loaded model; measure actual wall time, memory, requests and output equality. Implement a further bottleneck fix only if profiling identifies one. | **3–5 days. Codex + Claude Code.** A supplement to the submitted ROI work, not a duplicate new entry. |
| **5** | **Small, aligned donor-label improvement plus a released model ablation** | Start from existing 814/841/9B donor data; fix a bounded set of geometric/label errors, train a lightweight adaptation, and show benefit on untouched scrolls. Release exact inputs/checkpoints if suitable. Kill the project if labels cannot be independently trusted. | **7–14 days. Claude Code**, with owner review of the small label set. Conditional value; the lack of annotation staff is a real constraint. |
| **6** | **A narrow conservative fibre-tracing improvement** | One specific failure mode on public annotated fibres, producing better geometry in the existing tools. Require an actual connectivity improvement, not an attractive embedding visualization. | **10–14 days if an existing implementation can be extended. Codex.** Lower certainty than ranks 1–4. |

**My inference.** Avoid a new universal ink model from scratch, whole-scroll autonomous segmentation, a Greek-lookingness leaderboard, another generic audit tool, or redoing already proposed normalization fixes. The HTTP fix in this update is valuable hygiene and belongs with the toolkit; by itself it is not evidence of prize-level progress. The best one-week bet is to make one complete, trustworthy GPU workflow work better on real papyrus, then let the owner decide how to present the measured contribution.

### Sources

Source links are public. Mutable pages were checked on 21 November 2026; repository claims above use the named snapshots where possible. No candidate coordinates, images, private job identifiers, credentials or new transcriptions are included.

[rules]: https://scrollprize.org/prizes
[openproblems]: https://scrollprize.org/2026_open_problems
[win343]: https://scrollprize.substack.com/p/50k-first-letters-prize-awarded-for
[repo343]: https://github.com/Nieuwlaar/pherc343-first-letters
[win306]: https://scrollprize.substack.com/p/writing-seen-in-pherc-306b-for-the
[repo306]: https://github.com/SirProdigle/pherc0306b-first-letters
[discovery306]: https://github.com/SirProdigle/pherc0306b-first-letters/blob/main/discovery/README.md
[method306]: https://github.com/SirProdigle/pherc0306b-first-letters/blob/main/docs/METHOD.md
[win211]: https://scrollprize.substack.com/p/first-letters-emerge-from-pherc-211
[repo211]: https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt
[hecate]: https://huggingface.co/scrollprize/hecate/blob/main/README.md
[scroll5]: https://scrollprize.substack.com/p/exciting-news-from-scroll-5
[teamreadme]: https://github.com/JorisViaudQuantAI/Scrollprize/blob/48a0b8fa4bf4b47a939dda6b791402c9fcd2acec/firstletters-colab/README.md
[claude002]: https://github.com/JorisViaudQuantAI/Scrollprize/blob/48a0b8fa4bf4b47a939dda6b791402c9fcd2acec/collab/claude-002.md
[codex002]: https://github.com/JorisViaudQuantAI/Scrollprize/blob/48a0b8fa4bf4b47a939dda6b791402c9fcd2acec/collab/codex-002.md
[codex003]: https://github.com/JorisViaudQuantAI/Scrollprize/blob/48a0b8fa4bf4b47a939dda6b791402c9fcd2acec/collab/codex-003.md
[annular]: https://github.com/JorisViaudQuantAI/Scrollprize/tree/17dd73fb9e9b898ff70e4e08ba15eb7d212c0d1e/registration-checks/annular
[calibration]: https://github.com/JorisViaudQuantAI/Scrollprize/blob/48a0b8fa4bf4b47a939dda6b791402c9fcd2acec/firstletters-colab/controls/calibration.json
[scorer]: https://github.com/JorisViaudQuantAI/Scrollprize/blob/48a0b8fa4bf4b47a939dda6b791402c9fcd2acec/firstletters-colab/controls/score_text.py
[roi]: https://github.com/JorisViaudQuantAI/Scrollprize/tree/0ca39f67ad224826346b18763ab5b008bf976a32/evidence
[titles2019]: https://www.academia.edu/38498087/A_New_End_Title_in_the_Herculaneum_Papyri_and_the_First_Case_of_a_Preserved_subscriptio_in_One_of_the_Books_Assigned_to_Philodemus_Systematic_Arrangement_of_the_Philosophers_PHerc_327_Mnemosyne_72_2019_437_458
[limina]: https://edizioni.libreriauniversitaria.it/wp-content/uploads/2025/11/Limina_2025_I.pdf
[paratext]: https://www.academia.edu/45518459/Text_and_Paratext_in_Early_Greek_Bookrolls_Some_Reflections_on_Extant_Papyrological_and_Literary_Evidence
[title172]: https://scrollprize.substack.com/p/60000-first-title-prize-awarded
[finalcolumns]: https://www.academia.edu/125007883/THE_FINAL_COLUMNS_OF_PHERC_PARIS_4_REVEALED_THROUGH_VIRTUAL_UNWRAPPING
[firsttext]: https://www.academia.edu/116563322/F_Nicolardi_S_Parsons_et_alii_REVEALING_TEXT_FROM_A_STILL_ROLLED_HERCULANEUM_PAPYRUS_SCROLL_PHERC_PARIS_4_aus_Zeitschrift_f%C3%BCr_Papyrologie_und_Epigraphik_229_2024_1_13
[july2026]: https://scrollprize.substack.com/p/a-new-1m-grand-prize-for-2027
[paris4]: https://scrollprize.org/data_browser/PHercParis4
[wishlist]: https://api.github.com/repos/ScrollPrize/villa/issues?state=open&labels=help%20wanted&per_page=100
[issue192]: https://github.com/ScrollPrize/villa/issues/192
[issue193]: https://github.com/ScrollPrize/villa/issues/193

## Open questions

- **Claude Code:** Which prospective windows have genuinely not had their ink maps inspected, and can the released Studio model finish the small public-control pilot on the T4? These answers determine execution, not a retrospective rewrite of this protocol.
- **Codex:** Can the annular residuals be decomposed into target-sheet normal/tangent components and used to correct real transferred meshes? Which sites remain ambiguous when checked at native fine resolution?
- **Claude Code and Codex:** Can we establish training-footprint provenance for the public controls and record unique physical surface coverage? Where provenance is missing, retain the weaker label rather than assuming independence.
- **Owner:** If target-scroll high-resolution supervision becomes necessary, will you request an authoritative First Letters interpretation? No such request has been sent. The team can meanwhile use the other-scroll donor route.
- **Owner:** A prize cannot be guaranteed by continued computation. The present evidence supports no ten-letter or title claim. A private, technically reproducible candidate and organiser adjudication are still necessary.

## Next actions

1. **Claude Code:** Re-run a packaged public control from a fresh cache with the HTTP fix, exact input hashes and the frozen configuration. Then run the bounded Studio feasibility test. Record failures as failures, not negative ink findings.
2. **Codex:** Supply CT-validated meshes for the allocated 0257/0483B slots; turn the registration measurements into local mesh corrections. Use a public known-text workload for the next ROI timing demonstration.
3. **Claude Code and Codex:** Freeze the private FL-24 selection list and configuration, correct the scorer's physical-coordinate handling before recalibration, and execute the 24-slot tranche with sparse review of every view. Keep candidate details off this repository.
4. **Claude Code:** Limit Paris 4 work to the single diagnostic cycle above before deciding whether more title compute has evidential value.
5. **Owner:** Review the resulting evidence and choose whether, when and how to submit or amend a prize entry. No automatic submission, upstream PR, public candidate disclosure or external message is authorized by this note.
