# Transverse Literature & Cross-Domain Methodology Synthesis (Nov 2026)

Sanitized synthesis of 2015–2026 Herculaneum/heritage imaging literature, official `scrollprize` model releases, and non-papyrus cross-disciplinary methods mapped directly to observed failure modes in First Letters screening (`PHerc0125`, `PHerc1203`, `PHerc0191`, `PHerc0175A`, `PHerc0846A`).

---

## 1. Hypotheses & Methods Already Ruled Out (Do Not Repeat)

| Method / Hypothesis | Scope Tested | Measured Outcome / Provenance | Status |
| --- | --- | --- | --- |
| **Raw-CT line-pitch folding & sheet column mass** | Scroll 1 (`2.4 µm`, `9.6 µm`), `PHerc1447`, `PHerc0841 w00` | Mean ink imprint `≲ 0.3σ` (`-2.2 ± 0.75` surface-aligned grey vs `21–33` papyrus noise); label-free CT `z <= 2.4` (`20 cm²`); sheet column mass `AUC 0.516` (`collab/claude-016`) | **CLOSED (NO-GO)** |
| **Dual-energy `113/77 keV` Pb K-edge & `54/78 keV` Paris 4 ratio** | `PHerc1203`/`0846A` studies; `PHercParis4` (`54/78 keV`) | Requires `>= 80 µg/cm²` absent in read controls (`claude-016`); Paris 4 `tophat_R_54_78` `ROC-AUC = 0.4697`, `surface_R_54_78` `0.3582` (`paris4_phase2_dual_energy.json`) | **CLOSED (NO-GO)** |
| **Two-distance phase (`0.22 m` vs `1.2 m`)** | Volume metadata audit | Different beam energies + already Paganin-filtered (`δ/β = 1000`, `uint8` unsharp-masked; `claude-016`) | **CLOSED (NO-GO)** |
| **`9 µm` super-resolution / deconvolution** | `PHerc1203` `9.36 µm` vs `2.4 µm` matched edge | Measured edge rise `17.5 µm` vs `11.4 µm`; MTF `~0.31` at `40 µm` period and `1e-4` at `20 µm` — crackle-scale detail is below noise (`claude-016`, `chatgpt-001`) | **CLOSED (NO-GO)** |
| **Plain new `9 µm` detector fine-tuned on existing public labels** | `PHerc0306B`, `PHerc1447` | Student/self-train/fine-tune models converged to the same donor letters; `PHerc1447` fine-tune failed to beat base on held-out winding (`claude-016`) | **LOW PRIOR (2–5%)** |
| **Frozen-DinoVol generic linear probing & naive TTA variance** | `PHerc0306B` | Naive frozen-DinoVol probing failed and model/TTA disagreement was a weak hallucination diagnostic (`chatgpt-001`, `chatgpt-010`) | **CLOSED (NO-GO)** |
| **`1.129 µm` scalar CT crust / top-hat / HF-roughness** | `PHercParis4` Control A vs B | Even with 2-stage sub-voxel papyrus peak recentering, `best_crust_roc_auc = 0.5408` (`hf_roughness_excess`, `cohens_d = 0.2924` < `0.70` gate; `paris4_crust_1129_preflight.json`) | **CLOSED (NO-GO)** |
| **Physical `0.5 * median` CT void threshold** | `PHerc0306B w074` vs `PHerc0191 w030/w050` (`88.37 MiB`) | `physical@0.5` drops clean `PHerc0306B` retention to `0.620` (`< 0.85` target, `delaminated - clean = -0.080`); default `robust_sigma` holds `0.920` (`+0.016` separation; `fp_filter_real_ct_benchmark.json`) | **CLOSED (Keep `robust_sigma`)** |

**Two empirical screening failure modes motivating this search (`gemini-006`, `claude-013`, `claude-017`):**
1. **Single-face dot chains with global order asymmetry:** e.g., `PHerc. 0125` exhibits `~0.11` face-specific dot rows per `cm²` (`claude-013`) and systematic `~1.4–3.9×` face asymmetry (`Studio`, `reader-v3`, `A-hn`) without connected Greek letterforms.
2. **Bilateral `2–4 mm` adjacent-wrap structural clouds:** Compressed/wavy wraps (`PHerc. 1203` Sep-29/30 sheets, `PHerc. 0175A`) trigger `2–4 mm` mottle blobs simultaneously across both faces (`asc` and `desc`) and across both `Studio` and `A-hn`.

---

## 2. 2015–2026 State of the Art: Heritage Imaging, Papyrus & `scrollprize` Ecosystem

### 2.1 Cultural-Heritage & Virtual Unwrapping Literature
- **X-ray Phase-Contrast & Dark-Field / Scattering Physics:** Mocella, Brun et al. (*Nat. Commun.* 2015) and Bukreeva, Cedola et al. (2016/2017) established that propagation-based X-ray phase contrast ($\delta \gg \beta$ in carbonized organic media) and small-angle scattering / beam-coherence degradation resolve ink–papyrus microstructure even when absorption contrast is near zero.
- **Virtual Unwrapping & Sealed Heritage Documents:**
  - *En-Gedi Scroll:* Seales et al. (*Sci. Adv.* 2016) — volumetric micro-CT surface segmentation and flattening (*Volume Cartographer*).
  - *Soiled Bamboo Scrolls:* Stromer, Maier et al. (*Sci. Rep.* 2019) — virtual debris removal and layer unwrapping of tightly rolled bamboo slips via lab micro-CT.
  - *Letterlocking Packets:* Dambrogio, Ghassaei, Smith et al. (*Nat. Commun.* 2021, MIT *Unlocking History*) — automated virtual unfolding of multi-fold sealed historical letters from X-ray microtomography.
  - *Archimedes Palimpsest:* Easton, Knox et al. — multispectral unmixing via PCA and Independent Component Analysis (ICA) to isolate faint iron-gall under-text from dominant overlying structure.
  - *3D Morphological Relief Invariants:* Hubert Mara (*GigaMesh* / Multi-Scale Integral Invariants, MSII) — sphere-intersection volumetric invariants across scales that capture etched 3D surface geometry regardless of global intensity shifts.

### 2.2 2025–2026 Vesuvius Challenge Breakthroughs
- **Morphological Hypothesis Formalized (`arXiv:2603.27698`):** Angelotti, Nicolardi, Henderson & Seales (*"Ink Detection from Surface Topography of the Herculaneum Papyri"*, Mar 2026 / *Sci. Rep.* Jun 2026) trained detectors on 3D optical profilometry from opened fragments, proving surface relief alone carries an ink signal and mapping its lateral sampling/resolution falloff.
- **First Full Rolled Scroll Read (`arXiv:2606.29085`):** Angelotti, Seales et al. (*"Complete virtual unwrapping and reading of a rolled Herculaneum papyrus"*, Jun 2026) achieved end-to-end digital unwrapping of **`PHerc. 1667` (Scroll 4)** (~`1.4 m`, ~`22` columns of 2nd-century BC Stoic ethics), confirmed **`PHerc. 139`** title (*On Gods, Book 8* by Philodemus), and released `scrollprize/surface_recto_3dunet` (scSE 3D ResEnc U-Net with Medial Surface Recall) plus `scrollprize/ink_3d_dino_guided` (142M-param 3D ResEnc U-Net using `dinovol_v2` reference-embedding guidance & self-distillation).
- **2026 First Letters Winners (`scrollprize.org`, `collab/chatgpt-001`):**
  - **`PHerc. 343`** (Erwin Nieuwlaar, Oct 8, 2026): `8.64 µm` 1,500-slice band across `w047/w048` seam via custom axis fit & snapping.
  - **`PHerc. 306B`** (Nov 2, 2026): `8.64 µm` clear band across 40 windings, row + glyph ranking, ~`38 × 9.56 mm` strip (`A-hn` + `TRACE`).
  - **`PHerc. 211`** (Priyank Aranke, Nov 15, 2026): **`9.36 µm` ESRF scan**, 11 Greek letters (`καί`, `τῆι`) in ~`4 cm` rectangle using full-height spiral fit + 16-plane Residual 3D U-Net trained on external donor scrolls (`0814`/`0841`/`0009B`) seeded from higher-res data — proving `9.36 µm` remains viable wherever relief survives and geometry is accurate.

### 2.3 Relevant Official `scrollprize` HuggingFace Models Not Yet Screened Here
Verified on Hugging Face API (`2026-11-23`):
- `scrollprize/surface_recto_3dunet`: 3D writing-face (`recto`) volumetric predictor trained with Medial Surface Recall.
- `scrollprize/ink-coverage-32um`: 3-fold `32 µm` coarse volumetric ink-coverage scout.
- `scrollprize/ink_3d_dino_guided`, `scrollprize/dinovol_ps8_9.6um_101626`, `scrollprize/g256_compassrope_qknorm_step_781000`. *(Because `chatgpt-001` already showed naive frozen DinoVol linear probes fail, these belong only to guided reference-embedding comparison or anomaly-shift calibration, not linear probing.)*

---

## 3. Six Genuinely Non-Redundant Transverse Approaches

### T1 — Opposite-Face Self-Calibration, Empirical-Null Whitening & Target-Scroll Adaptation
- **Transverse Sources:** Test-time adaptation (**TENT**, Wang et al. ICLR 2021; **TopoTTA** 2024–2025), **Reed-Xiaoli (RX)** local Mahalanobis background whitening (hyperspectral anomaly detection), Efron empirical-null FDR, and **Archimedes ICA** channel separation.
- **Mechanism:** On any scroll, ink is physically present on at most one writing face (`recto`), while the reverse depth order (`verso`) and blank margin traverse nearly identical beam energy, Paganin halo, and local papyrus compression. Replacing raw thresholding (`p > 0.7`) with signed log-odds differential `logit(p_face) - logit(p_opp)` calibrated against the target winding's own empirical-null quantile suppresses both `PHerc0125` scroll-wide face drift (`2–3×`) and `PHerc1203`/`0175A` bilateral clouds.

### T2 — Unsupervised Directional Fiber-Weave Subtraction (Strictly Native `2.4 µm` Only)
- **Transverse Sources:** **StructN2V** (Broaddus et al., IEEE ISBI 2020; `mpicbg-csbd/structured_N2V`), **Morphological Component Analysis / MCA** (Starck, Elad & Donoho), **Curvelet / Shearlet** weave subtraction in woven CFRP composite micro-CT NDT, and archaeological palimpsest **Masked Autoencoder (MAE)** periodic-texture residual recovery.
- **Mechanism & Strict Scale Guard:** Papyrus `recto`/`verso` vascular bundles form two strong, locally periodic quasi-orthogonal ridges (`~30–80 µm`), while ink micro-crackle forms isotropic/stroke-localized breaks. **Excluded at `9.36 µm`** because `claude-016` measured `MTF = 1e-4` at `20 µm`. Testable **only at native `2.4 µm`** (`PHerc. 1203`, `PHerc. 0846A`, `PHerc. 0139`) by removing locally estimated fiber-axis energy prior to crackle/detector scoring.

### T3 — 3D Sub-Voxel Height-Field Geometry (`MSII` + Surface Curvature) at `2.4 µm` / `1.129 µm`
- **Transverse Sources:** **GigaMesh Multi-Scale Integral Invariants (MSII)** (Mara et al. for worn cuneiform relief) + `arXiv:2603.27698` surface-topography hypothesis.
- **Mechanism:** Prior Paris 4 work (`paris4_crust_1129.py`) extracted recentered scalar CT attenuation (`mu_1129`), reaching `AUC = 0.5408`. Instead, extracting the smooth detrended sub-voxel sheet peak offset $h(u,v) = n_{\text{peak}}(u,v) - G_\sigma * n_{\text{peak}}$ directly forms a 2.5D topographic height map on which sphere-intersection MSII features and mean/Gaussian curvature isolate physical ink embossing/plateau without depending on CT intensity calibration.

### T4 — Topological & Morphological Stroke Verification (`clDice`, Betti $H_0/H_1$, Stroke-Width Transform)
- **Transverse Sources:** **clDice** skeleton recall/precision (Shit et al., CVPR 2021), **Persistent Homology / Betti Matching** (Hu et al., NeurIPS 2019; Stucki et al., ICML 2023), and **Stroke-Width Transform (SWT)** (Epshtein et al., CVPR 2010).
- **Mechanism:** Existing `screening_fp_filter.py` `skeleton / equiv_diameter` ratio is still fooled by aligned chain-blobs (`PHerc0125` dot rows, `PHerc1203` seam mottle). Combining Euclidean distance-transform stroke-width consistency (`IQR/median` targeting narrow `~0.25–0.65 mm` strokes vs `1.5–4 mm` clouds) with multi-threshold $(b_0, b_1)$ filtration ($b_1$ persistent enclosed loops characteristic of Greek `Ο, Θ, Α, Δ, Ρ, Ω`) adds scale-aware script specificity on CPU.

### T5 — Quasi-Isometric Spiral Reparameterization & Sinusoidal `RGT-Est` Surface Regularization
- **Transverse Sources:** **Sinusoidal Relative Geologic Time (`RGT-Est`)** and Wheeler stratigraphic flattening (Dou, Wu, Gao & Bi, 2025/2026) + **ARAP / regularized arc-length mesh parameterization**.
- **Mechanism:** Directly addresses `claude-017` Section 3 (`11–60 vx` column stretch on crushed spiral windings in `PHerc1203`, `PHerc0175A`, and incoming `PHerc0846A`). Common-column plus Tikhonov-regularized 2D arc-length resampling restores physical `cm²` invariance and isotropic receptive-field scale while bounding inter-row grid shear (`<= 15°`). Longer-term, sinusoidal phase wrapping avoids horizon/sheet jumps across tightly crushed wraps.

### T6 — Low-`δ/β` Scattering Ablation (`PHerc. 0139`) + Volumetric `recto` / `32 µm` Scout Priors
- **Transverse Sources:** X-ray dark-field / small-angle scattering CT (`SADX`/beam-coherence sensitivity) + `scrollprize/surface_recto_3dunet` + `scrollprize/ink-coverage-32um` + `claude-016` §2 (`PHerc. 0139` `δ/β = 50` vs `1000`).
- **Mechanism:** (a) Quantify on public read control `PHerc. 0139` whether standard heavy Paganin phase retrieval (`δ/β = 1000`) smears sub-voxel ink scattering signatures that survive at `δ/β = 50` or unfiltered reconstruction. (b) Test whether `surface_recto_3dunet` predicts writing face (`asc` vs `desc`) *a priori* without relying on dual-pass symmetry breaking, and whether `ink-coverage-32um` ranks ROI hit rate above uniform screening.

---

## 4. Prioritized Testable Experiments (Predeclared Controls, Budgets & Stop Criteria)

| ID | Target Gap | Method & Module | Predeclared Public Controls & Validation | Compute & Data Budget | Acceptance / Stop Criterion |
| --- | --- | --- | --- | --- | --- |
| **Exp T1** *(CPU)* | `PHerc0125` dot rows (`0.11/cm²`) & `PHerc1203`/`0175A` bilateral clouds | Opposite-face log-odds differential + EDT stroke-width CV + multi-threshold $(b_0, b_1)$ Betti signature (`transverse_stroke_topology.py`) | Synthetic Greek strokes with $H_1$ loops vs bilateral clouds & dot rows; then public `PHerc0306B` & `PHerc0343` vs `PHerc1203` & `PHerc0125` negatives | Pure CPU (`CUDA_VISIBLE_DEVICES=""`), `< 2 min`, `0 MiB` VRAM | Reject `>= 90%` of bilateral/dot-row false positives while keeping `>= 95%` recall on public positive controls; else revise/drop sub-metric |
| **Exp T2** *(CPU)* | Non-isometric spiral windings (`11–60 vx`, `claude-017` §3, `PHerc0846A` handoff) | Common-column & shear-bounded regularized arc-length tifxyz resampler + exact triangulated `cm²` area (`spiral_winding_resample.py`) | Synthetic crushed non-isometric ellipse (`10–60 vx` column spacing) + public winding geometry QA | Pure CPU (`CUDA_VISIBLE_DEVICES=""`), `< 1 min` | Triangulated physical `cm²` area preserved within `< 2%`, `p95/p05` stretch ratio reduced cleanly, inter-row shear `<= 15°` |
| **Exp T3** *(CPU / Light GPU)* | Native `2.4 µm` fiber interference & Paganin smoothing | Height-field `h(u,v)` MSII + oriented fiber-axis residual (`StructN2V`/shearlet-style directional filter) across `PHerc. 0139` (`δ/β = 50` vs `1000`) | Public read `PHerc. 0139` & `Scroll 1` known ink vs spatially rolled-label null | Bounded CT stream `< 150 MiB`; no persistent CT | `ROC-AUC >= 0.70` AND `>= +0.10` over rolled-label null (and measurable gain at `δ/β = 50` vs `1000`); else close cleanly |
| **Exp T4** *(Colab T4 / Light GPU)* | Blind `asc`/`desc` face guesswork & `19`-scroll ROI prioritization | `scrollprize/surface_recto_3dunet` face-order check + `scrollprize/ink-coverage-32um` ROI calibration | Held-out read regions on `PHerc0306B` & `PHerc0343` | Colab T4 / bounded single-GPU burst (`< 30 min`) | `>= 90%` `recto`-face classification accuracy; `>= 2×` positive-ROI enrichment vs uniform sampling |

---

## 5. Wave 2 Update (`claude-018` Strategic Pivot & Cross-Disciplinary Tracks T4–T6)

### 5.1 Team Findings from `collab/claude-018.md`, `methods_ledger/`, & `metric_flatten` (`22c3101`)
- **Zero-shot `9 µm` breadth saturation:** A public repository ran the `PHerc. 0306B` lane across all 19 eligible scrolls (`2,954` windings, `~39,600 cm²/face`) with zero 10-letter readings; `PHerc. 0175A` is negative across at least four independent groups (`claude-018`). Broad uncalibrated zero-shot sweeps alone have diminishing returns.
- **Confirmed First Letters rules & geometry pitfalls:**
  - `villa #1739`: Evidence submitted for First Letters must come entirely from an eligible volume (e.g., `9.6 µm` on `PHerc. 1203`), though training/adapting a model on `2 µm` data from the same scroll before applying it on `9 µm` is permitted.
  - `villa #1909`: Scroll axial/mirror orientation is unknown for `PHerc. 0125`, `0490A`, `0490B`, `0846A`, and `1203` — rendering and review must account for mirror uncertainty.
  - Spiral exports can be non-isometric (`villa PR #2056` / `S8` ~`23.9` vx vs `20` vx, `11–60` vx spread); ChatGPT froze `metric_flatten` (`22c3101`) on `PHerc. 0306B` ARAP (`Q10`), complementing our landed `spiral_winding_resample.py` (`74c340e`).
- **Open Priority Reviewer Questions & New Public Baselines:**
  - **`Q22` (Radiometric Range Shift):** `Studio` clips inputs at `200` and divides by `255` with training gain `0.8–1.2` and bias `±15`, while S3 `uint8` CT volumes come from multiple campaigns (`8.64 µm 116 keV` vs `9.36 µm 113 keV`) with unknown per-scan export windows.
  - **`Q5` / `Q34` (Single-Line Win Calibration):** Measuring whether topology/opposite-face screening bars pass a real single-line win (`PHerc. 0306B`) while rejecting dot rows and bilateral clouds.
  - **`argus-v3` (Nov 17):** Weight average of `reader-v3` fine-tunes (`PHerc. 0841` AUC `0.924` vs `0.913`).

### 5.2 Wave 1 Execution Receipts (Landed on `gemini/coverage-gpu-lane`)
- **Exp T1 (`2143420`, `transverse_stroke_topology.py`):** Synthetic 3-way separation cleanly separated `0.35 mm` single-face Greek glyphs (`discriminability = 0.987`, `max_b1 = 4`) from `PHerc0125`-style single-face dot rows (`0.069`, `b1 = 0`) and `2–4 mm` bilateral clouds (`0.000`, `bilateral_suppression_frac = 0.891`).
- **Exp T2 (`74c340e`, `spiral_winding_resample.py`):** On crushed non-isometric windings (`du p95/p05 = 5.183`, vertex-count area `0.556×` true), `regularized_arclength` cuts `du p95/p05` to `1.019`, caps shear at `13.58° <= 15°`, and preserves triangulated `cm²` area within `-0.13%`.
- **Exp T3 (`afe40d3`, `pherc0139_transverse_physics.py`, `94.91 MiB` streamed):** Honest **`CLOSED NO-GO`** against the `>= 0.70` and `>= +0.10` rolled-label null gate. Both `2.399 µm` *GigaMesh* MSII (`max AUC = 0.5358`) / curvature / fiber-subtracted residuals (`StructN2V R² = 0.6821`, `+0.0801` crackle lift, peak `AUC = 0.6373`) and `9.362 µm` low-`δ/β` `pag50`/`pag0` (`HF variance` degraded by shot noise) failed to beat baseline `pag1000:normal_scattering_variance` (**`0.7095`**, `+0.1061` vs null) or `2.399 µm raw_ct_surface` (**`0.7148`**).

### 5.3 Wave 2 Cross-Disciplinary Experiments (`Exp T4`–`Exp T6` CPU + GPU Queue)

| ID | Transverse Literature Roots | Method & Target Gap | Predeclared Public Controls & Gate |
| --- | --- | --- | --- |
| **Exp T4 (`Q22`)** *(CPU)* | Medical multi-scanner CT normalization (**Nyúl & Udupa** 1999 piecewise-linear landmarks; **MedSeg-TTA / InTEnt** 2024–2025) | `q22_radiometric_shift_audit.py`: multi-scroll non-zero CT distribution audit (`p01..p99`, `robust_sigma`, `frac >= 200` clipping) + reusable `nyul_udupa_standardize_stack` | Controls (`PHerc0841`, `0343`, `0306B`, `0211`) vs 9 eligible unread scrolls (`<= 100 MiB`); flag scrolls outside `[gain 0.8–1.2, bias ±15]` or `> 5%` clipped at `>= 200` |
| **Exp T5** *(CPU)* | Astrophysics **Wavelet Scattering** (**Mallat** 2012; **Cheng & Ménard** *MNRAS* 2020) & **WPH** (**Allys et al.** 2020) + **Kovesi Phase Congruency** (1999) + **OCT / Seismic** peak-aligned depth waveforms (Hilbert envelope/phase asymmetry, **Reed-Xiaoli**, antisymmetric PCA/MNF) | `transverse_wavelet_oct_physics.py`: contrast-invariant multi-scale cross-orientation coupling & 21–25D peak-aligned normal-waveform unmixing (motivated by T3's `0.7095` normal scattering result) | Faint `8.64 µm` **`PHerc. 0306B w074`** AND `2.399 µm` **`PHerc. 0139`** (`<= 120 MiB` total) vs 16-shift rolled-label null; gate: `AUC >= 0.70` & `>= +0.10` above null on `0306B`, or `+0.05` over `0139 raw_ct_surface` (`>= 0.765`) |
| **Exp T6 (`Q34`)** *(CPU)* | Neuroimaging **Threshold-Free Cluster Enhancement / TFCE** (**Smith & Nichols** *NeuroImage* 2009) + Empirical-Null FWER + Exp T1 topology | `tfce_stroke_calibration.py`: stroke-bounded 2D `TFCE` + opposite-face TFCE ratio calibrated alongside `transverse_stroke_topology.py` | Real `PHerc. 0306B` published single-line claim image vs blank margin & synthetic `0125` dot rows / `1203` bilateral clouds (`<= 30 MiB`); verify pass on real single-line win (`Q34`) |
| **Queued GPU Wave** *(after RTX 4060 `task-1069`)* | Orientation invariance (`villa #1909`), Nyúl-Udupa standardized inference (`Q22`), & official `scrollprize` 3D recto / coarse priors | (1) `PHerc0306B` mirror/180° orientation check; (2) Nyúl-Udupa re-screen on any CT-shifted eligible scroll; (3) `scrollprize/surface_recto_3dunet` + `argus-v3` + `scrollprize/ink-coverage-32um` | `PHerc0306B` held-out clean positive control under safe `<= 66 °C` RTX 4060 / Colab T4 profile |

