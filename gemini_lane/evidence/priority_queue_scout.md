# Untouched Public Meshes & Axial Bands — Quality-Ranked RTX 4060 Priority Queue

Prepared by Gemini Neon research scout (`2026-11-22`). All findings are verified from public GitHub, Hugging Face, and S3 open-data metadata; zero unread-scroll predictions or private coordinates are contained here.

---

## Verified Public Asset & Quality Highlights

1. **`PHerc0125` (`pscamillo/vesuvius-eligible-meshes`, `9.362 µm`):**
   - `340` meshes total (`17` z-windows × `20` windings `w005–w100` step 5; `z12336` and `z13536` absent despite README mentioning 19 windows).
   - `62` old Atlas renders (`w020/040/060/080/100` across 12 windows + `z15344_w020/040`).
   - **`278` NEW meshes** (skip any claimed by Codex, e.g. check `z14144_w045` before touching it): `172/278` pass `on_sheet_2snaps >= 0.70`, `68 >= 0.80`, `17 >= 0.85`, `3 >= 0.90`.
   - **Critical prep:** Raw meshes average only `0.34` on-sheet vs `0.75–0.84` after **2 snaps onto `m7`** (`snap_meshes.py`). **Skip `z17136`** (`0/15` pass `0.70`) and **`w080–w100`** (`0/39` pass `0.70` in `w090–w100`).
2. **`PHerc1203` S3 `segments/raw/` (`9.362 µm` + `2.403 µm` rescan):**
   - `22` untouched Tracer auto-grown meshes (`130.14 cm²` total, `z = 5403–10150`), **zero** pre-rendered `surface-volumes/` or `ink-detection/` on S3.
   - Led by `9` large sheets (`87.35 cm²`), topped by `auto_grown_20251005230830031` (`16.10 cm²`), `...1446965` (`12.53 cm²`), `...21856743` (`11.79 cm²`), `...104534929` (`8.40 cm²`). Deduplicate against Codex A/B/C (`20251005230830031`) & Claude N6 (`20251005221856743`) via UV bounds / `overlapping.json` before rendering.
3. **`PHerc1218` (`8.640 µm`):**
   - `IyanDopico/vesuvius-sheet-tools` hosts `25` tiny `1/64`-scale anchor seeds (`3.642 cm²` total), NOT full meshes.
   - `results/pherc1218_sweep/windows.csv` identifies **`6 full PASS` sweep windows** (`z15680-16480`, `z16320-17120`, `z16960-17760`, `z20160-20960`, `z7140-7940`, `z4580-5380`, plus `z9700-10500`) hosted at Kaggle `pscamillo/pherc1218-spiral-fit-sweep-surfaces` (~2.1 GB). Avoid `z1380-2180` and `z17600-18400` (`PROVISIONAL`).
4. **`PHerc0191` & `PHerc0826` (`rodriguescarson/eligible-scroll-spiral-fits`, `9.362 µm`, CC BY-NC 4.0):**
   - `PHerc0191` (`slice-11600-12400`, `415.63 cm²` raw, `12.14%` full-track sat) is the **only published mesh set** for `PHerc0191`.
   - `PHerc0826` adds `slice-8928-9728` and `slice-9328-10128` (`49.30%` and `48.74%` track-pt sat, unrepresented in `pscamillo`; `z=9328–9728` overlaps by 400 slices so render `8928-9728` + non-overlapping `9728-10128` only).
   - Over-packed pitch (`7.3–10.5 vx` vs `14–19 vx` CT pitch) -> screen inner-to-mid **`w015–w050` at stride 2–3 with 2× `m7` snap**.
5. **`PHerc0800` 6 S3 segments — EXCLUDED:**
   - Only `9.93 cm²` total (`z = 10853–13263`), and all 6 already have S3 `surface-volumes/` and `ink-detection/` outputs (`argus-v3` run `2026-11-17`). Do not re-screen.
6. **Zero-mesh scrolls (`9` scrolls, all have S3 `m7` + `.normal-grids`, zero `umbilicus/`):**
   - **#1 `PHerc0846A`:** `2.403 µm` rescan (`15137 × 29587 × 29587`) + `m7-L2` (`3785 × 7397 × 7397`) + normal-grids on S3. Screen grown surfaces at `L2` (`9.612 µm`) first, then `L0` (`2.403 µm`) on clean patches.
   - **#2 `PHerc1545`:** Only zero-mesh scroll with both `m7-L0` (`20961 × 7506 × 7506`) and `lasagna` (`20250821151648-lasagna-20260419180421`) on S3.

---

## Top-10 Ranked RTX 4060 Queue

| Rank | Scroll | Target Subset | Est. Raw Area | Mandatory Prep / Caveat |
|---:|---|---|---:|---|
| **1** | **PHerc0125** | `pscamillo` NEW Tier-1 (`on_sheet_2snaps >= 0.85`, 17 meshes minus Codex's 2) | `~55–65 cm²` | Apply 2× `m7` snap; skip `z14144_w045` pending Codex reply |
| **2** | **PHerc1203** | S3 `segments/raw/` 9 large Oct/late-Sep auto-grown sheets (`z=5403–10150`) | `87.35 cm²` | Exclude already-screened Codex A/B/C & Claude N6 regions |
| **3** | **PHerc0125** | `pscamillo` NEW Tier-2 (`on_sheet_2snaps in [0.80, 0.85)`, 51 meshes; esp. untouched `z11136`, `z15936`, `z7536`) | `~180–210 cm²` | Exclude 62 old Atlas meshes, `z17136`, `w080–w100`; 2× `m7` snap |
| **4** | **PHerc1218** | Kaggle `pscamillo/pherc1218-spiral-fit-sweep-surfaces` 6 `full PASS` windows (`w010–w050`) | `~150–250 cm²` | Skip `PROVISIONAL` windows; snap onto `m7` before rendering |
| **5** | **PHerc0191** | HF `slice-11600-12400` (`w015–w055`, stride 2–3) | `~45–70 cm²` | Over-packed pitch (`10.5 vx`); stride + 2× `m7` snap |
| **6** | **PHerc0826** | HF `slice-8928-9728` + non-overlapping `9728–10128` (`w015–w050`, stride 2–3) | `~55–85 cm²` | Deduplicate `400`-slice overlap; 2× `m7` snap |
| **7** | **PHerc1203** | S3 `segments/raw/` 13 smaller Sep-23/25 sheets (`z=5448–9499`) | `42.79 cm²` | Deduplicate against Rank 2 using `overlapping.json` |
| **8** | **PHerc0846A** | New Tracer / `vc_grow_seg_from_seed` growth on `2.403 µm` scan via `m7-L2` | `20–50 cm²` batch | Screen at `L2` (`9.6 µm`) or tiled `L0` (`2.4 µm`) for 8 GB VRAM |
| **9** | **PHerc1545** | New Tracer / sheet growth via S3 `m7-L0` + `lasagna` | `30–60 cm²` batch | Longest axial span (`196 mm`) of zero-mesh scrolls |
| **10** | **PHerc0125 / 0826** | Overflow: `PHerc0125` `104` NEW `[0.70, 0.80)` meshes + `0826` `slice-6528-7328` (`w015–w050`) | `~440+ cm²` | Skip `w027`, `w051` on `0826` (match `pscamillo` `w020/040`) |
