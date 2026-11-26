# 846B exact reproduction intake (source review, not executed)

[Winner repository](https://github.com/brkzak/Pherc0846B_submission), README and infer_letters.py (public). Freeze full Git SHAs and weight/input hashes before execution.

- Public tifxyz 316×316, scale 0.05, reported surface area 19.69 cm², exported grown segment; full rendered output 6320×6320.
- Native CT 20250804142305, 9.362 µm, level 0, 113 keV masked. Normals oriented toward the volume-centre umbilicus. Trilinear stack of 17 layers spaced one voxel, centred at layer 8.
- Reader v3 weight revision 39f626e5 (author's abbreviated revision); Villa b29f310 (author's abbreviated revision); tile 384, stride 192, Hann blend, forward direction.
- Reference raw TIFF is uint8 (probability ×255, observed range 0–231); display uses percentile 1–99.9 windowing. Neither display nor raw uint8 TIFF is native float32 nor independent ink/background labels. Verify raw agreement, preserve off-segment mask and compute native float32 sidecar before calibration.
- Use exact author commands from README for baseline only; then freeze matched same-physical-footprint anchor-only, normal-only and rigid perturbations. Count attempted tiles, failed support, CT bytes and elapsed render/inference separately. No new reading or threshold promotion. Compare actual reference output before GPU-heavy perturbations.

The [343 announcement](https://scrollprize.substack.com/p/50k-first-letters-prize-awarded-for) also describes adapting spiral fitting to 8.64 µm, estimating the umbilicus, refining snap and stitching neighbouring windings at the angular seam. This reinforces the need for seam tests and correct physical scale, independent of the four-model mean.

## Advisory routing check

`node route.mjs --selftest` or `node route.mjs PHerc0483B 20251124083638 first_letters 2026-11-26` from this directory. 68 cases passed in the JS connector runtime; standalone Node replay requested. Only exact ID+volume matches on the dated snapshot are eligible for review; a stale verification date fails closed. Grand Prize path explicitly needs volume verification and is not dispatched; controls are provenance-review only. Caller must actually recheck the official rules, not supply a fictitious verification date. This helper is not wired into the industrial worker and never authorizes dispatch.
