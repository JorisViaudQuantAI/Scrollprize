# Native 002 — A02 compatibility and published PR

2026-11-24. Follow-up to immutable [Native 001](chatgpt-native-001.md). Initial contribution published as [draft PR 7](https://github.com/JorisViaudQuantAI/Scrollprize/pull/7), targeting industrial RFC PR 6. Initial published head `cf5a620c9b419b61c3e6df0827327b0143067404` has exactly the locally reviewed Git tree `b9cc7a0cf16fd1415d2e75f41b2fd5dd52171daa`.

The initial [push checks](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45396301968) and [PR checks](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45396414210) both succeed: 98 tests and the public numerical evidence verification. These links identify actual completed runs, not planned CI.

During publication the upstream industrial branch added A02's experimental single-order allocation and paired opposite-order survey. Upstream revision `e1da697f9f56891dec215f56f83c8f106341a08e` integrates without conflicts. The actual combined local suite passes **116/116 tests**, no failures, errors or skips: the 98 previously published checks plus eighteen upstream order-audit regressions. [Compatibility receipt](../firstletters-colab/industrial_funnel/native_tests_a02.json), [integration explanation](../firstletters-colab/industrial_funnel/NATIVE_INTEGRATION.md). Preserve the original 98-test receipt at its original revision; its source hashes intentionally describe the earlier control/file implementations.

This verifies software compatibility. The new experimental order policy does not retrospectively supply missing population, assignment, semantic-control, or worker evidence for our historical one-order maps. All native imports remain computational observations with no production or scientific promotion. The original visual findings, four exact reverse pairs, 13 maps / 1,091 historical forwards, and failed 071 cutoff are unchanged. This follow-up performs no CT download or model inference.

## Open questions

- Native producer exports, fit-quality recall/cost validation and independent physical correspondence remain the same scientific prerequisites.
- A02's order rule is producer-asserted; its actual sensitivity requires the frozen paired adjudication. Software survey correctness is not normal-sign or legibility calibration.

## Next actions

Review PR 7 alongside the current RFC. Reuse its native array validator in the actual producer export while keeping A02 experimental-only and retaining complete attempt, support, control and resource evidence.
