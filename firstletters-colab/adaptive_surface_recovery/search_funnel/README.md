# Budgeted First Letters search funnel

Status: routing core and public-control replay, not a deployed all-scroll search or a calibrated ink-probability system. The input is an exact versioned scroll inventory plus private chart manifests. The output is a **priority queue**, never an automatic blank verdict.

## Pipeline and agent ownership

| Stage | Reused component | Next action |
| --- | --- | --- |
| 0 — inventory | Gemini `coverage_inventory.py` / pinned coverage matrix | register every eligible scroll, record unmeshed areas and missing data |
| 1 — geometry | Claude spiral fitting + per-winding flatboi; Gemini streaming + RGT/patch growers; exact support and shear diagnostics | repair or register chart, never label a geometry failure blank |
| 2 — matched positive | Claude 026 and Gemini T17 calibration; Codex physical frame checks | match model weights, flattening, physical scale, stack convention, stride and both orders |
| 3 — coarse | Gemini `screen_scroll_rtx4060.py` Studio both orders stride 96, or same-recipe Claude rendering | retain original native maps, masks, timing, provenance outside Git |
| 4 — rank | `router.rank_tiles`, actual `line_review.rank_map`, Gemini `component_strokeness` | budgeted tile priority, morphology/line/signal and exploration slots; no hard reject |
| 5 — deep | Studio stride 48 + A-hn both orders, optional independent local regrowth/flattening | expand selected 10 mm tiles into 20 mm review windows, verify original-frame overlap; keep offsets/orders separate |
| 6 — review | Claude/Codex independent CT landmark and adjacent-sheet tests; papyrology | preserve false positives and ambiguous identities, keep private evidence; no automated reading or prize claim |

Every generated chart has a measured source-support mask, physical frame, immutable geometry/render/recipe fingerprints, same-configuration positive receipt and measured cost. The router trusts manifest assertions, so the operator/producer must verify those assertions; it is not an anatomical certificate. Unknown support or failed control sends the chart to repair, not rejection. Only identical registered geometry/render/recipe in the same frame is deduplicated. Distinct carrier IDs that may trace the same sheet need independent physical correspondence first. Unique papyrus coverage cannot be inferred from chart counts or repeated crops.

## Local usage

Run `python router.py --manifest /path/outside/repo/private-charts.json --out-private /path/outside/repo/next-actions.json`. The manifest is a JSON array, one object per private chart. Fields: `scroll`, `chart_id`, `frame_id`, `geometry_sha256`, `source_support` (verified or unknown), `geometry_state` (verified_local or repair), `positive_gate` (same_configuration_pass or other), `coarse_orders` (array of actual completed orders), `render_sha256`, `recipe_sha256`, `cost_seconds`. Record actual evidence, not asserted success. Paths, chart identifiers and review maps stay outside Git. This command makes a plan only, not GPU invocations.

The GPU operator uses the existing pinned `gemini_lane/screen_scroll_rtx4060.py` interface with their authorized compute and private input. Geometry must already be flattened/validated. Example options for coarse pass: `--detectors studio --studio-stride 96 --num-slices 31 --snap-passes 0` plus actual scroll, mesh, CT frame URL, voxel pitch and private output directory. Snap is an explicit separate recipe requiring its own control, not a universally safe default. For deep follow-up: `--detectors studio,ahn --studio-stride 48` with corresponding frozen geometry/crop. No command is executed by the router. Missing map export or provenance requires a private producer adapter before automatic ranking. The GPU script's summary alone is not equivalent to full native maps.

## Allocation and validation

Within each registered chart, tile cost is one padded 10 mm tile; H = high-score quantile, L = single-line proposal overlap, S = component strokeness, R = fixed-seed exploration, F = cyclic portfolio. Scores are not comparable across detectors or scrolls without calibration. Do not spend all budget on a single scroll; keep explicit geometry-generation and exploration queues and measured per-stage cost. The 25/50/75% replay budgets measure only downstream tile allocation on existing maps. They do not include acquisition, geometry, coarse inference, expansion to 20mm, or human reading time. Real savings require measuring those terms, plus a blinded same-configuration coarse-to-fine recall test on multiple known-positive surfaces and sampling low-priority charts. No permanent exclusion based on this replay.

See [protocol.md](protocol.md) for frozen evaluation and limitations. Public evidence contains only already-awarded control maps' aggregate ranking and the aggregate 19-scroll inventory. Never commit unread tile locations, private charts, identifiers or images. 
