# Adaptive surface recovery 043 — corrected native origin, actual A-hn inference

**Status: completed; scientific gate FAIL (6/7).** No prize/candidate claim. Inputs and labels are published, previously read PHerc0306B material; the six trace crops use the author's human trace as a proxy, not independent ink ground truth. Two author-blank crops are imperfect negatives. The eight frozen 040 tiles were retained exactly. No threshold was changed after seeing results.

- Protocol commit: 96a9a98de30c657bbe2aed9b8862654c745d6226.
- Runner commit: 20c3f54c3cc2f7835f7ed22da6d1138a8d96ce25.
- Workflow: [45273917577](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45273917577), completed successfully.
- Exact [machine-readable result](../firstletters-colab/adaptive_surface_recovery/corrected_detector/result.json): 7ec2c87a8b3fcdc8ea8820dce1131d322466be85.
- Eight new FP32 CPU A-hn forwards only; 040 baseline B and P/Q predictions reused. Wall 124.21 s, peak RSS 2.377 GB, native CT 96,469,231 bytes over 47 requests, below 160 MiB cap. All prior B input hashes and trace counts verified exactly.
- H = half-pixel corrected point renderer, with the same geometry, orientations, depth and checkpoint. Credit to Claude's existing native +0.5 convention and the shared Claude/Gemini components.

| Metric | B (released render) | P (uncorrected atlas) | Q (quadrature) | H (corrected atlas) |
|---|---:|---:|---:|---:|
| Trace AP | .786649889 | .753612617 | .752253562 | .777267237 |
| Trace AUC | .764847791 | .708930619 | .706753320 | .745012245 |
| Trace recall @.5 | .332890071 | .446275311 | .437175898 | .365563889 |
| Distant untraced flagged @.5 | .194707254 | .207814097 | .221094449 | .204036656 |
| Author blank flagged @.5 | 0 | .094238281 | .071289063 | .032714844 |
| CT MAE to B | 0 | 5.337686451 | — | 2.090418869 |

H gains 0.02365462 AP over P, decreases CT MAE 60.8366% and reduces blank activations 65.28% relative to P. H clears the AP improvement, noninferiority, recall and CT reduction gates. **Blank gate fails**: .032714844 exceeds B+.03 by .002714844 (67/2048 macro-equivalent pixels). One blank tile has 67/1024 flagged and the other zero. This is not a successful detector release; no unread screening promotion. Do not relax the fixed threshold or remove the failed tile. Do not count the same trace samples as independent validation or a new reading.

The CT correction itself passed iteration 042 and remains a useful tested coordinate convention. A separate fresh public control is needed to assess transfer; the original atlas footprint/face identity and support advantages remain distinct from detector efficacy.
