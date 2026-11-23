# PHerc1203 Public S3 `segments/raw/` Deduplication & Execution Manifest

Verified from public S3 open-data metadata/TIFF inspection and repository coordination logs ([collab/codex-002.md](../../collab/codex-002.md), [collab/claude-003.md](../../collab/claude-003.md), [collab/claude-006.md](../../collab/claude-006.md), [collab/claude-011.md](../../collab/claude-011.md), [firstletters-colab/negative_results/negatives.csv](../../firstletters-colab/negative_results/negatives.csv)). Contains zero unread predictions or local machine paths.

---

## 1. Exact S3 Zarr v2 Root URLs (no trailing `/0`)

- **CT (`9.362 µm`, Zarr v2):**
  `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr`
  - `0/.zarray`: `zarr_format=2`, `shape=[18977, 6844, 6844]`, `chunks=[128, 128, 128]`, `dtype="|u1"`, `dimension_separator="/"`, `compressor=null`, `filters=null`, `fill_value=0`.
  - **Do not use** `...-masked-fast-q1-c64-v0.3.2.vcd3d.zarr`: that asset is Zarr v3 (`zarr.json`) and is rejected by `RemoteZarrV2`.
- **`m7` Surface Prediction (`9.362 µm` L0 frame, Zarr v2):**
  `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/representations/predictions/surfaces/20250820131727-surface-20260413222639-surface-m7-L0-th0.2.zarr`
  - `0/.zarray`: `zarr_format=2`, `shape=[18977, 6844, 6844]`, `chunks=[192, 192, 192]`, `dtype="|u1"`, `dimension_separator="/"`, `compressor={"id": "blosc", "cname": "zstd", "clevel": 1, "shuffle": 1, "blocksize": 0}`, `filters=null`, `fill_value=0`.
  - Requires `numcodecs` for `blosc`/`zstd` decompression (`numcodecs 0.17.1` verified in the runtime virtualenv). Note `--snap-passes 0` skips `m7` reads when running without `numcodecs`.

---

## 2. Cohort Reconciliation & Large S3 `segments/raw/` Meshes

- **Total public S3 meshes (`22`):** `130.16 cm²` (`meta.json` `area_cm2`):
  - **9 newer late-Sep / Oct-05 meshes** (`20250929–20251005`, `versions/`, no `overlapping.json`): **`80.76 cm²`** (`8` sheets `> 6 cm²` = `77.02 cm²` + `auto_grown_20251005231446963` = `3.74 cm²`).
  - **13 older Sep-23 / Sep-25 meshes** (`overlapping.json` present, no `versions/`): **`49.40 cm²`** (`11` Sep-23 sheets × `3.89 cm²` = `42.79 cm²` + `2` Sep-25 sheets = `6.61 cm²`).
  - *Reconciliation note:* [priority_queue_scout.md](./priority_queue_scout.md) listed `87.35 cm²` for the 9 large sheets by subtracting only the `11` Sep-23 sheets (`42.79 cm²`) from `130.15 cm²`, folding the `2` Sep-25 sheets' `6.61 cm²` into that figure.

Base prefix: `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/segments/raw/`

| # | Mesh ID | Grid (`H×W`) / Valid | `area_vx2` | `area_cm2` | `z`-range | `scale` | `overlapping.json` | Overlap vs Codex A/B/C & Claude N6/N8/N10 |
|---:|---|---|---:|---:|---|---|---|---|
| 1 | `auto_grown_20251005230830031` | `232×268` / `42,292` | `18,368,442.2` | `16.10` | `5819.1–10150.4` | `[0.05000000074505806, ...]` | None | Partial Codex A/B/C (`55.4%` UV, `59.9%` 3D same-sheet) + Claude N8 (`r=84:125, c=43:84`) + Claude `reader-v3` pass; `0%` N6 |
| 2 | `auto_grown_20251005231446965` | `260×260` / `33,024` | `14,297,374.9` | `12.53` | `5404.7–9443.5` | `[0.05000000074505806, ...]` | None | `0.0%` Codex A/B/C, `0.0%` N6 (Claude N10 `r=188:229, c=103:144` uninformative crumpled scrap) |
| 3 | `auto_grown_20251005221856743` | `253×226` / `30,858` | `13,454,751.8` | `11.79` | `5402.7–9077.3` | `[0.05000000074505806, ...]` | None | `0.0%` Codex A/B/C; Partial Claude N6 (`43.8%` UV, `r=30:175, c=62:158`) |
| 4 | `auto_grown_20250930104534929` | `216×216` / `22,684` | `9,587,355.5` | `8.40` | `5818.5–8678.9` | `[0.05000000074505806, ...]` | None | `0.0%` Codex A/B/C, `0.0%` N6, `0%` peer overlap |
| 5 | `auto_grown_20251005230118636` | `164×213` / `21,322` | `8,684,870.1` | `7.61` | `5406.3–8480.0` | `[0.05000000074505806, ...]` | None | `0.0%` Codex A/B/C, `0.0%` N6, `0%` peer overlap |
| 6 | `auto_grown_20250929222256117` | `210×174` / `20,596` | `8,522,916.7` | `7.47` | `5519.7–8270.0` | `[0.05000000074505806, ...]` | None | `0.0%` Codex A/B/C, `0.0%` N6; adjacent wrap to `#7` (`3.5%` pinched contact `<=4 vx`) |
| 7 | `auto_grown_20250930000321811` | `185×206` / `19,673` | `7,888,049.1` | `6.91` | `5601.9–8270.9` | `[0.05000000074505806, ...]` | None | `0.0%` Codex A/B/C, `0.0%` N6; adjacent wrap to `#6` (`3.6%` pinched contact `<=4 vx`) |
| 8 | `auto_grown_20251005230830030` | `165×136` / `16,750` | `7,066,945.4` | `6.19` | `6115.4–8888.8` | `[0.05000000074505806, ...]` | None | Partial Codex A/B/C (`14.6%` `<=4 vx` concentrated in `c >= 68`: `24.1%` vs `5.5%` in `0:68`), `0%` N6 |
| 9a | `auto_grown_20251005231446963` | `148×148` / `10,512` | `4,267,778.3` | `3.74` | `6214.2–8569.5` | `[0.05000000074505806, ...]` | None | `0.0%` Codex A/B/C, `0.0%` N6; adjacent wrap to `#2` (`6.9%` pinched `<=4 vx`) |
| 9b | `auto_grown_20250923163217042` | `152×152` / `11,100` | `4,438,400.0` | `3.89` | `5929.7–7966.1` | `[0.05000000074505806, ...]` | `15` pruned UUIDs (`0` on S3) | `0.0%` Codex A/B/C, `0.0%` N6, `0%` top-9 overlap |

---

## 3. Non-Overlapping Screening Targets

### Tier-1 Clean Full Meshes (`46.66 cm²`, `0%` overlap with Codex A/B/C and Claude N6)
1. `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/segments/raw/auto_grown_20251005231446965` (`12.53 cm²`)
2. `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/segments/raw/auto_grown_20250930104534929` (`8.40 cm²`)
3. `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/segments/raw/auto_grown_20251005230118636` (`7.61 cm²`)
4. `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/segments/raw/auto_grown_20250929222256117` (`7.47 cm²`)
5. `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/segments/raw/auto_grown_20250930000321811` (`6.91 cm²`)
6. `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/segments/raw/auto_grown_20251005231446963` (`3.74 cm²`)

*(Optional 7th clean Sep-23 sheet: `auto_grown_20250923163217042`, `3.89 cm²`.)*

### Tier-2 Partially Overlapping Meshes (run each `--uv-crop` in its own invocation)
- `auto_grown_20251005221856743` (disjoint from Claude N6 `r=30:175, c=62:158`):
  - Right strip (`~2.18 cm²`): `--uv-crop 0:253,158:226`
  - Left strip (`~1.85 cm²`): `--uv-crop 0:253,0:62`
  - Bottom-mid strip: `--uv-crop 175:253,62:158`
- `auto_grown_20251005230830030` (low-overlap left half, `5.5%` pinched contact, `~3.02 cm²`): `--uv-crop 0:165,0:68`
- `auto_grown_20251005230830031` (lowest priority, `55.4%` Codex A/B/C overlap; unscreened strips: `187:232,0:268`, `0:46,0:159`, `0:187,0:43`).

---

## 4. `meta.json` and TIFF Sentinel Verification

Across all **22** `PHerc1203/segments/raw/` meshes:
- `scale`: `[0.05000000074505806, 0.05000000074505806]` (`float32(0.05)`), handled cleanly by `render_surface_stack` with exact integer output dimensions.
- TIFF headers/sentinels: all `x.tif`, `y.tif`, `z.tif` are 2D `float32`; zero `NaN`/`Inf`, zero `(0,0,0)` sentinels, zero partial-channel negatives; every invalid vertex uses `-1.0` across all 3 channels (`mask.tif` absent / HTTP 404, cleanly ignored by `fetch_mesh_dir`).
