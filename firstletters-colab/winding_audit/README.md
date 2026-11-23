# Q12: storage-invariant winding diagnostics

This lane tests the arithmetic in `methods_ledger/code/sense_chain.py`. It is
separate from the claimed ink, radiometry, rendering, and atlas-screening lanes.

Two issues motivate it. The original walker chooses the low-angle endpoint,
then multiplies its increasing angular displacement by the original storage
direction a second time. Reversing a track's array can therefore change the
answer without moving any point. Independently, a chain that finishes at a
different height measures radial drift from both winding and axial shape.
For `r(theta,z) = R + b theta + k z`, a turn gives
`delta r = 2 pi b + k delta z`. A mirror-sign check does not remove that term.

The real-data experiment is frozen in [protocol.json](protocol.json). It uses
the public PHerc0175A track store, a fixed slice band, and the image center as an
explicitly unvalidated origin. It tests storage invariance, not anatomy. The
current official catalog independently derives `ACW` for this volume; a code
defect is not evidence that the existing PHerc0175A fit is mirrored.

The completed report will retain legacy and corrected outcomes, the analytic
taper counterexample, source hashes, and exact reproduction commands. No raw
volumes, unread ink maps, or candidate locations are part of this lane.
