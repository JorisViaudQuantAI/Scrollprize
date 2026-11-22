# PHerc. Paris 4 (Scroll 1) Title Prize ($50,000) — Unexplored Zones & Different-Ink Streaming Protocol

Prepared from Gemini Neon read-only S3/transform audit (`2026-11-22`) and coordinated with [`collab/claude-011.md`](../../collab/claude-011.md) §4 (Claude finished ~29 cm² below the final column as negative and continues the right-of-column Hecate/Studio pass; Gemini owns the **different-ink dual-energy CT lane**, **outer wraps `w130–w135`**, and the **`20260701` extended height**).

---

## 1. Corrected Public Volume & Energy Inventory

* **DLS 54 keV (`7.910 µm`):** `20230205180739` (`[14258, 7888, 8096]`, covers `z_78 ≈ 29,543–76,047`) + `20230206171837` (part 2, lower z, no public S3 transform). *(Note: DLS is 54 keV, not 53 keV.)*
* **ESRF 74 keV & 110 keV (`45.532 µm`):** `20260310170716` (`74 keV`, transform `r2` to DLS) and `20260310173927` (`110 keV`, full scroll overview).
* **ESRF 78 keV (`2.400 µm`):** `20260411134726` (`[75784, 32693, 32693]`, full scroll; `window_f32 = [-0.04, 0.22]`).
* **ESRF 137 keV (`2.400 µm`, ROI1):** `20260323153942` (`15.9 × 20.2 × 20.2 mm`, `z_78 ≈ 42,128–48,769`, missing umbilicus; covers 3.2% of `20260623` / 2.0% of `20260701` `w010-027`; `window_f32 = [0.01, 0.16]`).
* **ESRF 78 keV (`1.129 µm`):** `20260608103018` (`[59969, 36006, 32354]`, `67.7 × 40.7 × 36.5 mm`, `z_78 = 28,080–56,356`, includes umbilicus; `r2` non-rigid bend transform median residual 2.28 µm).
* **All 6 carbon-ink CNN families** (`Studio`, `A-hn`, `v8-in`, `reader-v3`, `ink_9um`, `Hecate`) had Scroll 1 labels/teachers in training.

---

## 2. Unexplored Physical Zones Outside Carbon-Ink & `20260623` Coverage

1. **Agraphon recto under dual-energy / `1.129 µm` L0 CT contrast:**
   - `20260701183124-w010-027` (`U_frac >= 0.683`): **68.9%** overlap with **54 keV DLS** (`96.8%` of `20260623`), **45.8%** with **1.129 µm** (`71.3%` of `20260623`), **2.0%** (`z_78 = 42,128–48,769`, `U_frac = 0.703–0.982`, 15.9 mm strip + adjacent last column!) with **137 keV**.
2. **Extended axial height in `20260701183124-w010-027` (`[109, 69760, 170000]`):**
   - Extends **38.4 mm lower** (`z_78 = 11,953–27,964`) and **9.6 mm higher** (`68,063–72,051`) than `20260623` (checked in N11). Below `z ≈ 25,500`, N12's 3D models failed their control. Must verify sheet preservation against top-row physical loss first.
3. **Outermost wraps `w130–w135` & outer flakes:**
   - Present in `20260926044226-spiral-output-generation-19` (`w10..w135`), outside the `w010–w129` 2.4 µm segment series; zero 2.4 µm 2D-ink map or dual-energy inspection (`Del Mastro` initial-title recto hypothesis).
4. **Innermost unmeshed wraps `w001–w009` (`1.129 µm` FOV)** & **true verso interface** (~100–200 µm opposite material boundary).

---

## 3. Low-Storage (< 180 MB Streaming, CPU-Only) Different-Ink Protocol

### Physics & Mandatory Mitigations
- Carbon ink ($Z \approx 6$) matches carbonized papyrus in X-ray linear attenuation; high-$Z$ mineral/metallic inks (Fe/Cu/Ca, or **Pb $K = 88.0\text{ keV}$ / Hg $K = 83.1\text{ keV}$**, sitting **between 78 keV and 137 keV**!) exhibit steep photoelectric contrast $\propto Z^4 / E^3$ or K-edge jumps.
- **Global affine drift (`codex/paris4-energy-coverage`):** `137 keV -> DLS` `p95 = 8.6 µm`, `78 keV -> DLS` `p95 = 42.9 µm` (max `50.3 µm`). **Never ratio directly on global affines!**
- **Mitigations:** (1) Convert `uint8 -> linear attenuation` via each scan's `metadata.json` `window_f32`; (2) estimate a **local 3D NCC lock (`Δu, Δv, Δn`)** per `2 mm` sub-tile and gate against 6 displaced nulls (`±20 µm`); (3) PSF-match `78 keV` (`2.4 µm`) to `7.91 µm` before comparing with `54 keV` DLS; (4) reject features peaking `90–260 µm` outward or aligned with the neighbouring wrap's `6.75 mm` line grid (`claude-011` §4).

### 4-Step Streaming Plan (< 180 MB Total Network I/O)
1. **Step 0 — Radiometric & NCC Preflight (~15 MB):** 3 paired `10 × 10 mm` tiles on `20260701183124-w010-027` (`U_frac ≈ 0.60–0.67` last legible column Control A; `U_frac ≈ 0.72–0.78` confirmed blank intercolumnium/agraphon Control B; `z_78 = 15k–25k` sheet-integrity check). Verify local NCC beats all 6 nulls and measure blank noise floor $\sigma_{\text{blank}}$ before any test map.
2. **Step 1 — `78 vs 137 keV` Dual-Energy Strip (~55 MB):** `z_78 = 42,128–48,769` (~15.9 mm spanning both last column & `11 cm` of agraphon). Fit linear papyrus background $\mu_{137} \approx \alpha \mu_{78} + \beta$ strictly on Control B blank papyrus, evaluate residual $R_{78/137}$ + surface-vs-bulk depth top-hat (`|n| <= 8 µm` vs `16..36 µm`), and report exact blank-papyrus FPR at $3\sigma / 4\sigma$.
3. **Step 2 — `54 vs 78 keV` Full-Agraphon Screen (~50 MB):** Stream `78 keV` L2 (`9.6 µm`) + `54 keV` DLS L0 (`7.91 µm`) over `z_78 = 29.5k–72.0k` (~130 cm²). Fit $R_{54/78}$ on Control B blank papyrus ($\sim 3\times$ photoelectric boost at 54 keV for high-$Z$ elements).
4. **Step 3 — Native `1.129 µm` L0 Crust & Outer-Wrap `w130–w135` Spot Check (~60 MB):** Check top 2–3 ROIs inside `z_78 = 28.1k–56.4k` at `1.129 µm` L0 (using `r2` bend transform) + `w130–w135` recto continuity & layout at `4.8 µm` / `7.91 µm`.

**Hard stopping rule:** Escalate only if a feature beats Control B's $99.9\%$ tail on `|n| < 12 µm` (not `90–260 µm` ghost) with `0.25–0.6 mm` stroke coherence across `>= 15 mm`.
