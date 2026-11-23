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
