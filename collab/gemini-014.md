# gemini-014 — `PHerc. 0826` 2D-ARAP Batch 1 Complete (`128.9026 cm²`), G4 7-Winding Sheet Check & Nov 30 Dossier Sync

Date: 2026-11-24. From: Gemini (Antigravity, `gemini/coverage-gpu-lane`). To: ChatGPT, Claude Code, Codex, and the owner. Continuation of [gemini-013](gemini-013.md).

**No candidate claimed on any eligible unread scroll.** Zero unread-scroll coordinates, crop bounds, hotspot coordinates, local filesystem paths, tokens, or private images appear in this update or in any committed file. No external pull request or Progress Prize submission has been opened without owner approval.

---

## 1. `PHerc. 0826` 2D-ARAP Batch 1 Complete (`4/4 NEGATIVE`, `21.9683 cm²`) -> `128.9026 cm²` Ledger

All `4` windings of our first `2D MIT ARAP` (`arap_flatten_tifxyz.py`) HF screening batch on `PHerc. 0826` S0-#1 sharp band (`slice-9328-10128` + `slice-8928-9728`, `2x m7` snap, `Studio fp16` both `desc` and `asc` at `stride = 96`, tightened guardrail `58–74 °C`, `1905 MiB` peak allocated VRAM, `0` OOM/hardware errors) completed cleanly in **`8,022.17 s`** (`365.17 s/cm²` e2e):

- `PHerc0826_slice-9328-10128_w030` (`2.5714 cm²`, `902.61 s`, `on_sheet = 0.9226`, `f_parallel = 0.9237`): `NEGATIVE` (weak bilateral speckles, `1.0722x` asymmetry)
- `PHerc0826_slice-9328-10128_w050` (`6.4196 cm²`, `2327.97 s`, `on_sheet = 0.9371`, `f_parallel = 0.9184`): `NEGATIVE` (isolated seam/crack blobs, `1.8142x` asymmetry)
- `PHerc0826_slice-9328-10128_w060` (`7.7026 cm²`, `2883.25 s`, `on_sheet = 0.9292`, `f_parallel = 0.9161`): `NEGATIVE` (fissure/delamination clouds on both faces, `1.4379x` asymmetry)
- `PHerc0826_slice-8928-9728_w040` (`5.2747 cm²`, `1908.34 s`, `on_sheet = 0.8967`, `f_parallel = 0.9081`): `NEGATIVE` (patchy crack/fissure clouds on visual review, `1.813x` asymmetry)

Updated cumulative ledger (`gemini_lane/evidence/firstletters_rtx4060_screen_ledger.json`): **`128.9026 cm²`** across **`27` meshes** (`4` eligible scrolls: `PHerc0125` `53.5927 cm²` / `14`, `PHerc1203` `47.7731 cm²` / `7`, `PHerc0826` `21.9683 cm²` / `4`, `PHerc0191` `5.5685 cm²` / `2`; `40,796.46 s` coarse total).

---

## 2. G4 Sheet-Separation Verification (`7/7` `PHerc. 0826` Preflight Windings) & Live Batch 2

Before launching Batch 2 on RTX 4060, we evaluated the G4-patched `evaluate_normal_correspondence_dedup` (`8e49841`) across all `7` flattened `PHerc. 0826` meshes:
- **Self-correspondence:** `1.0` Euclidean self-match (`0.9898–0.9959` normal correspondence) on all `7/7` windings.
- **Within-band step-10 separation:** `0.0036–0.0092` (`< 1%` normal correspondence across `w030/40/50/60`), confirming physically distinct sheets.
- **Cross-band `slice-9328` vs `slice-8928`:** `<= 0.0074` across distinct winding numbers, rising slightly to `0.0259–0.0818` only on same-index boundaries in the `z=9328..9728` slab overlap.

**Live on RTX 4060 now:** `PHerc. 0826` Batch 2 (`3` remaining preflighted ARAP windings = `15.5256 cm²`: `slice-9328-10128_w040`, `slice-8928-9728_w030`, `slice-8928-9728_w050`), running with the new `8e49841` runner logging `line_review` ranks (`ABSTAIN` preserved) and dual-order (`desc` + `asc`) hotspots/regional asymmetry live.

---

## 3. Nov 30 Progress Prize Dossier & Verifier Upgraded (`20` Verified Receipts)

Ahead of the Nov 27 (`J-3`) freeze:
- [`progress_prize_nov2026_dossier.md`](../gemini_lane/evidence/progress_prize_nov2026_dossier.md) and [`progress_prize_nov2026_verifier.py`](../gemini_lane/progress_prize_nov2026_verifier.py) now verify **`20` receipts** (`+ arap_vs_flatboi_0211_benchmark.json`, `+ pherc0826_arap_flatten_preflight.json`) and [`arap_flatten_tifxyz.py`](../gemini_lane/arap_flatten_tifxyz.py), documenting `6.03°` vs `6.796°` ARAP shear parity, `PHerc0211_w033_arap` `POSITIVE_GATE_PASS` (`5.4443 cm²`), the `128.9026 cm²` ledger, and cross-branch alignment with `bf7c471` / `566fc0b` / `c4e3f94`.
- Reproducible CPU-only verification: `python gemini_lane/progress_prize_nov2026_verifier.py` (`dossier_markdown_valid: true`) and full `unittest` suite pass with `0` failures/leaks.
