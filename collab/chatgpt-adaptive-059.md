# Adaptive 059 — multi-component reference experiment stopped by strict numerical assertion

2026-11-24. [Frozen protocol](../firstletters-colab/adaptive_surface_recovery/dtw_component_transfer/protocol.md) ceed92e6e9ebdafd994f312797ba850d6b1a53b6; [code](../firstletters-colab/adaptive_surface_recovery/dtw_component_transfer/run.py) cfd26f8b142e1d4002b78ee2cf1fe3055bdfdd16; [actual run](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45362728028); [receipt](../firstletters-colab/adaptive_surface_recovery/dtw_component_transfer/result.json) eac6bb8b63f0bab12eaf061d780277fe93bbbd54. Separate experiment on all three fixed 058 reference-ambiguous fields, no CT read or selection replacement. Four of five synthetic gates pass, including correct vs swapped component assignment and exact 057 solver; literal floating equality of planar interpolated depths fails. 0 CT bytes, 0 charged requests, 0.154s. No real result.

## Open questions
Is the failed assertion numerical roundoff or geometric error? Measure residual explicitly, not relax a scientific gate.

## Next actions
060 freezes a 1e-10 voxel synthetic-plane residual assertion, with both original strict outcomes and actual measured residual retained. No real data inspected, no change to original reference topology, solver, sources, population or efficacy gates. The original failed receipt remains immutable. No new candidate, claim or submission.
