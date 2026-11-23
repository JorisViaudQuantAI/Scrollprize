# Adaptive-surface lane — cross-swarm integration index

Snapshot 2026-11-23 21:20 UTC. This file is a navigation and coordination record, not an experimental result. Check HEADs and issue #1 again before launching work. Branch snapshots below are references, not instructions to merge. Do not disclose unread locations, private data or candidate images. The owner retains main merges, upstream submissions and prize claims.

## Read before work

- [Shared Claude branch](https://github.com/JorisViaudQuantAI/Scrollprize/tree/claude/firstletters-colab), [methods ledger](https://github.com/JorisViaudQuantAI/Scrollprize/blob/claude/firstletters-colab/firstletters-colab/methods_ledger/09_open_questions.md), [Claude 019](https://github.com/JorisViaudQuantAI/Scrollprize/blob/claude/firstletters-colab/collab/claude-019.md): Q7/Q1 computed, verdicts pending; PHerc.1203/0846A fine-band reading and fit, wrong-sense control assigned to Claude. Do not repeat. Q10 Villa flatten output QA is assigned there.
- [Winding order audit, immutable revision 94155b2](https://github.com/JorisViaudQuantAI/Scrollprize/tree/94155b2966259b0c461540f90a2b12b748f328c6/firstletters-colab/winding_audit): corrected Q12 storage bug; accepted by Claude; real winding order still open. Do not repeat legacy correction or take over queued wrong-sense fit. The shared branch already contains it. The winding lane is distinct from this adaptive-surface lane.
- [Surface-footprint renderer, freeze f7fc619](https://github.com/JorisViaudQuantAI/Scrollprize/tree/f7fc619ec11f9dc95caa0c90fb2148cbf7c86ebd/firstletters-colab/surface_sampling): original-triangle whole-pixel clipping and 1/4/8-point quadrature, separate NIST phantom protocol. No result located as of this snapshot. Reuse this API; don't reimplement its sampler or repeat the phantom test.
- [Gemini d48e4e9](https://github.com/JorisViaudQuantAI/Scrollprize/tree/d48e4e9abac1e6922be599e44028f60120c6b6d8/gemini_lane): RGT multi-sheet, regularized arclength resampling, source-supported streaming CT, FP16, TFCE/robust MAD and radiometry tests. [Progress dossier](https://github.com/JorisViaudQuantAI/Scrollprize/blob/d48e4e9abac1e6922be599e44028f60120c6b6d8/gemini_lane/evidence/progress_prize_nov2026_dossier.md). Do not duplicate T2–T10 or dossier submission. Use pinned reader and its explicit masks. Treat simulated clouds and proxy labels as development controls, not independent ink ground truth.
- [Codex 0813 paired geometry](https://github.com/JorisViaudQuantAI/Scrollprize/tree/1ed75191bbef1c0fa7834c1249f4b0bb8c0eaeaf/registration-checks/atlas-0813), [306B CPU reference](https://github.com/JorisViaudQuantAI/Scrollprize/tree/59f56be9d6d1feed9fa749dce072b697f79c5770), [R50 Hann comparison](https://github.com/JorisViaudQuantAI/Scrollprize/tree/43a5365692be70f37258dbb3c6e29fef08f340b1). Use as baselines; no repetition of published model parity or geometric registration checks.

## This lane's reusable work

Each row links a complete handoff with pinned inputs, protocol, script and numerical results. Negative results remain immutable. Note the name collision: `collab/chatgpt-032.md` in the shared branch is the **winding Q12** report; `collab/chatgpt-032.md` in this branch is the **anchored recovery** report. Always cite branch or commit, not bare number. New notes from this lane use `collab/chatgpt-adaptive-NNN.md`, beginning 039, to avoid collisions.

| Handoff (this branch) | Experiment | Decision |
| --- | --- | --- |
| [032](../../collab/chatgpt-032.md) | Anchored depth tracking and paired ray guard | FAIL |
| [033](../../collab/chatgpt-033.md) | Adaptive depth hypotheses | FAIL |
| [034](../../collab/chatgpt-034.md) | Review window coverage | FAIL |
| [035](../../collab/chatgpt-035.md) | High-Z digital injection | FAIL |
| [036](../../collab/chatgpt-036.md) | Fixed-boundary single chart | FAIL |
| [037](../../collab/chatgpt-037.md) | Two-chart original-face atlas | PASS geometry, not ink |
| [038](../../collab/chatgpt-038.md) | Native face preselection and actual five-plane CT parity | PASS geometry/runtime, not ink |
| [Adaptive 039](../../collab/chatgpt-adaptive-039.md) | Cross-swarm atlas / footprint renderer / Gemini CT reader integration | PASS on two native 31-plane public ROIs; no ink claim |

Artifact: [public source atlas from run 45265373300](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45265373300/artifacts/14862856229); SHA256 of source_atlas.npz `43c12969b32c42b978e9d1cbb3bd70fb645a998100062d889d66228a130fd8d0`. Artifacts expire; regenerate with 037's pinned protocol and check numeric/source identities. Do not commit large raw CT or model maps. For native point samples use [FaceROIIndex](indexed_native_renderer/run.py); for whole pixel support and quadrature use shared branch's OriginalSurface / surface_sampling. Both preserve chart assignment; the two charts must never be composited in UV coordinates.

## Next shared interface

**Completed in [Adaptive 039](../../collab/chatgpt-adaptive-039.md):** [ChartBridge](shared_renderer_bridge/run.py) links the original-face atlas to the existing shared whole-pixel footprint renderer and Gemini's source-supported reader. The [receipt](shared_renderer_bridge/result.json) records zero CT value differences across 46,624 actual values and exact original-face IDs, with 32 unsupported partial-footprint pixels correctly removed on the fallback ROI. All eight gates pass. Next: coordinate one bounded, annotated public-control ink-inference comparison with the existing shared renderer and GPU lanes; no parallel reimplementation or unread deployment.

At the end of each substantive experiment: freeze protocol/code before result, publish complete numerical receipt and commands on the dedicated branch, communicate in issue #1, and name the consuming agent, expected output, advance/stop criterion and limitations. No automatic task or schedule.
