# Adaptive 048 — support-aware spiral profile arithmetic

**PASS**, all six frozen synthetic gates and real profile checks. Claude 020 requested metamorphic audit of spiral_oncheck.py. This is an arithmetic correction, not new sheet identity or a detector claim.

[Frozen protocol](../firstletters-colab/adaptive_surface_recovery/profile_gate_audit/protocol.md) 9f561471d6504ef7fb9ad9b236348a73b015bce3; [code](../firstletters-colab/adaptive_surface_recovery/profile_gate_audit/run.py) bee2c173168dc97a599fe39f83a9738c2d39fba7; [executed CPU workflow](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45279680104), job 138060518301; [exact result](../firstletters-colab/adaptive_surface_recovery/profile_gate_audit/result.json). Legacy nearest_peak extracted unchanged by AST from pinned Claude source blob 0462c7df258ab85a408e3e429082c3a84ba30023.

Synthetic findings:
- Incomplete monotone rising profile: legacy minimum-value imputation manufactures a near peak; corrected result is unknown.
- Symmetric nearest peaks at -4/+4: legacy chooses -4 both before and after reflection; corrected absolute distance 4 retained, signed offset ambiguous.
- One near peak, nine complete nonpeaks, 90 unsupported profiles: legacy reports 100% among found. Corrected reports 1/100 lower bound, 1/10 among observed, 1/1 among found, upper bound 91/100. Do not conflate these denominators or infer anatomical accuracy from them.
- Supported zero distinguished from missing; random complete profiles pass reflection.

Actual previously read public 306B: two fixed 8x8 point grids, fit and one seeded paired +/-8 voxel shift control, CT and M7 at matching native frame, 65 half-voxel samples each. All 512 profiles fully supported; all absolute distances and unique signed peaks reflection-consistent, including ten tied CT peak offsets across all conditions. Transfer 22,125,944 bytes / 16 requests, 2.34s wall. No new geometry fit, snap, GPU, CNN or unread data. On complete profiles legacy and corrected near2 counts match exactly.

| Source | Fit near2 | Shift control near2 |
| --- | ---: | ---: |
| CT | 73/128 | 76/128 |
| M7 | 107/128 | 31/128 |

The small fixed fields show M7 proximity improves relative to the chosen shift, but raw CT peak proximity does not. Not an anatomical same-sheet validation or a representative estimate; fixed shift is not necessarily a different winding.

Designed censoring of offsets >+1 on all 256 actual fit CT/M7 profiles: legacy still finds 237 peaks and calls 198 within 2 voxels (89 CT +109 M7). Corrected method marks all 256 incomplete; no certified nearest peak. This does not establish that every legacy found peak was false; only that complete-profile nearest-peak claims cannot be verified from those censored inputs.

Reuse profile_peaks(profiles,observed,prominence) for explicit full support, plateau midpoint, unique signed offsets and tie flags; summarize reports separate lower/upper bounds and conditional denominators. center_on_m7 is meaningful only on the M7 source; CT z-score only on CT. Preserve old output and do not retroactively relabel screens without reprocessing retained source masks. No changes to other branches or external submission.