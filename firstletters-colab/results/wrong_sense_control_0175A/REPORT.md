# 36 — Wrong-sense control on PHerc0175A (lane `ws`, session g2)

Private. Started 2026-11-23 23:55 UTC. Working dir `/home/user/vesuvius/wrongsense_0175A/` (jobs/, results/,
tools/fetch_chunks.sh = one-pass fetcher for g2).

## Question and design
Can our on-sheet gate (fraction of 200 random vertices per winding on an m7 sheet, against the same vertices shifted
by U(−8, 8) voxels) tell a right-sense fit from a wrong-sense one? PHerc0175A is the known case (catalogue:
z_direction_is_top_to_bottom = true, left_handed = false; fitted ACW in report 21). Refit **identically except the
sense = CW**: band z 6900–8400, same public inputs re-staged on g2 with the same scripts (`spiral_full/jobs/sf_*`,
code tarball `sf_code_v2.tgz` = v1 + gate files), same synthetic normals recipe, same own umbilicus
(`spiral_full/umb/umbilicus_PHerc0175A.json`), villa @0fb38c45 with the same patches, horizon 30,000, track-DT from
10,000, same chunk loop lengths (1,560, 1,560, 2,100, 2,100, 2,100 s) so the gates fall at about the same steps
(ACW: 7,592 after c03, 13,738 after c05), same gate script (`gate.sh`, 200 vertices, seed 0, ±8 vox control).
Stop after c05 (≈ 13,700). Note: re-staging on another VM recomputes the SDT and synthetic normals; they should be
bit-identical up to float nondeterminism, not verified.

## Decision rule (written 23:58 UTC, before any gate)
Reference (right sense, ACW, report 21, inside windings): raw on-m7 minus control = −0.007 at 7,592 and **+0.068** at
13,738 (0.269 vs 0.201); 0.321 vs 0.207 (+0.114) at 30,000.
Let Δ = mean raw `m7on` − mean control `m7on` over inside windings (CT found ≥ 0.9) of the CW fit at ≈ 13,700.
- **GATE DISCRIMINATES SENSE** if Δ(CW) ≤ +0.03 (i.e. within 0.03 of its control) **and** Δ(ACW) − Δ(CW) ≥ 0.04.
- **GATE DOES NOT DISCRIMINATE** if Δ(CW) ≥ +0.05 (a wrong-sense fit departs from chance about as the right one did).
- **INCONCLUSIVE** otherwise (+0.03 < Δ(CW) < +0.05), or if the run does not reach ≈ 13,000 steps.
Secondary diagnostics recorded at both gates for both senses (not part of the rule): loss terms (total, track_radius,
track_dt, dense_normals, dense_spacing_*), satisfied_tracks / satisfied_track_points, per-block raw m7on (w010–039,
040–069, 070–099, 100–129) and their spread, CT z-score at the surface, snapped on-m7 and snap rejection.

## Progress log (UTC)
- 23:57 Submitted on g2 (idle, worker since 06:22; dies ≈ 06:20 24 Nov): `ws_001_setup` → `ws_010_data` →
  `ws_011_normals` (gpu) → `ws_101_cw_c01` … `ws_105_cw_c05` (gpu, chain), gates `ws_201_gate_c03`, `ws_203_gate_c05`
  (cpu). Projected: data ≈ 00:30, normals ≈ 00:37, chunks ≈ 00:37–03:55, last gate ≈ 04:20 (GPU ≈ 3.4 L4-h).
- 00:05 Local container restarted (background wait killed; g2 queue unaffected). 00:08 resumed: `ws_001_setup` ok (95 s), `ws_010_data` running.
- 00:35 Second local restart. 00:40 checked `ws_010_data`: not stuck, slower bucket reads on g2 than on the original VM (tracks ≈ 14 min, m7 56 s, CT L4+L1 1,013 s); native + patches done 00:30, surf-SDT running since 00:30 (≈ 11 min on the original run). New projection: chunks ≈ 00:50–03:55, gate c05 ≈ 04:20 (before the 05:30 hard stop).
- 02:45 (coordinator) Agents were stopped by the usage limit at ≈ 01:10 UTC; queue ran on. Coordinator fetched c02 (4,348 steps) and c03 (7,781 steps) with tools/fetch_chunks.sh; gate c03 and chunk c04 running at 02:40. The coordinator will read the gates and apply the decision rule above.
- 03:39 (coordinator) fetched c04 (11,007 steps) and gate c03 (step 7,781): raw m7on 0.199 vs control 0.199 (Δ 0.000; right sense was −0.007 at 7,592). Both senses at chance at this stage, as expected; the rule applies at ≈ 13,700.
- 04:32 (coordinator) fetched c05 (14,030 steps) and `ws_203_gate_c05`.

## Result and verdict (coordinator, 04:40 UTC 24 Nov)

**GATE DOES NOT DISCRIMINATE SENSE.**

| quantity, inside windings | right sense (ACW), step 13,738 | wrong sense (CW), step 14,030 |
|---|---|---|
| raw on-m7, fit / control | 0.269 / 0.201 (Δ +0.068) | 0.265 / 0.203 (Δ **+0.062**) |
| raw within 2 voxels of m7, fit / control | 0.348 / 0.271 | 0.344 / 0.271 |
| step ≈ 7,600–7,800: raw on-m7, fit / control | 0.197 / 0.204 | 0.199 / 0.199 |
| snapped on-m7, fit / control | (0.668 / 0.218 at 30,000) | 0.614 / 0.215 |
| CT z-score at the surface, raw fit / control | (+0.069 / +0.033 at 30,000) | +0.040 / +0.013 |
| total loss (last two logged steps) | 276.8, 281.3 | 309.0, 306.1 |
| track_radius | 171.0, 173.7 | 190.7, 191.9 |
| track_dt | 78.7, 78.0 | 83.6, 84.5 |
| dense_normals | 2.5, 2.4 | 2.8, 2.7 |
| satisfied tracks / track points | 24.6 % / 61.4 % | 23.7 % / 59.9 % |
| same at ≈ 10,800: loss, track_radius | 292, 171–173 | 316–318, 193 |

Rule applied as written: Δ(CW) = +0.062 ≥ +0.05 → the gate does not discriminate. A wrong-sense fit leaves chance
exactly as the right-sense fit did.

Reading.
- The on-sheet gate measures whether a winding follows a sheet locally. A fit follows tracks and normals locally in
  either sense; a wrong-sense spiral only has to change sheet once per turn somewhere, which 200 random vertices per
  winding do not see. So the gates of PHerc1203 (CW, 0.344 vs 0.237) and PHerc0846A (CW, 0.247 vs 0.188) say the
  windings follow sheets locally, and say nothing about the sense.
- For First Letters (letters within 4 cm²) local sheet-following is what matters; the global sense matters for
  linking turns and for the reading order by geometry (normal towards/away from the umbilicus is unaffected).
  So the sense question does not block reading fitted windings, and the 0846A fit may resume in its current sense.
- The fitter's own losses are the only quantity that separates the senses here: total loss +10 %, track_radius +11 %
  (171–174 against 191–193) at matched steps, at both 10,800 and 13,800; satisfied track points 61.4 % against 59.9 %.
  One scroll, one seed pair: suggestive, not established. A sense test for a scroll with no catalogue value = fit both
  senses to ≈ 11,000 steps and compare track_radius.
- Not done: the wrong-sense fit to 30,000 steps (whether the gap in on-sheet fraction opens later); per-block table.
GPU: ws lane 411 L4-minutes on g2 in the fair-share counter (includes CPU prep time accounting); g2 stopped after this.
