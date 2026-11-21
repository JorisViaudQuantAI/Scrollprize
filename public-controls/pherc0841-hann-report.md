# PHerc. 841 tile blending and grid stability

Across two disjoint mixed fields, the ratio of abrupt changes at tile boundaries to nearby baseline changes fell from **28.27–35.41 with uniform blending to 0.92–1.16 with Hann blending**. The phase-to-phase Dice at threshold 0.5 increased in both fields. Ink discrimination depended on the field and grid origin. On the 512-pixel cores, ROC AUC increased in one of four field–phase comparisons and decreased in the other three. [Public protocol fixed before inference](https://github.com/JorisViaudQuantAI/Scrollprize/blob/f70ea2d3f8f24ea8e28e9af5b446ce7e09cdadf9/public-controls/pherc0841-hann-protocol.md) · [Complete numerical results](pherc0841-hann-result.json)

## Public fields and model calls

The label-only rule selected one field with 88,414 ink pixels out of 262,144 trusted pixels (33.7%) and another with 169,799 out of 262,144 (64.8%). Both met the 90% coverage tier and in fact had complete supervision and validation masks across the evaluated area. The 640 × 640 source rectangles were disjoint and excluded the seven earlier public control rectangles. Each evaluated 512 × 512 field spans 1.23 mm on PHerc. 841 w00. [Field coordinates and input hashes](pherc0841-hann-result.json)

The grids differ by 64 pixels (0.154 mm) on each axis. Each field used 16 ascending-order R50 tile predictions for A and nine for B. Uniform and Hann received the exact same tile predictions in each phase, for 50 native forward passes in total. Replaying the old implementation reproduced all four default uniform maps and their tiling metadata exactly. The job also passed four tests for the command, irregular border anchoring, whole tiles and seam coordinates. [Historical implementation](https://github.com/JorisViaudQuantAI/Scrollprize/blob/329e9552c35d06a52720c8fbb4077005bef29bdc/public-controls/r50_stack.py) · [Completed CPU run](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/44919843418)

## Grid stability on the common 512 × 512 area

The seam statistic uses the absolute change across actual tile-entry or tile-exit edges. Its baseline uses the same trusted scanlines 16 pixels on either side. A ratio of one means equal mean changes. Phase Dice measures agreement between the two binary prediction maps. Probability change is the absolute difference between the two grids at the same pixel.

| Field | Blend | Seam ratio A / B | Phase Dice at 0.5 | Phase Dice at 0.8 | Mean probability change | P95 change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Uniform | 35.41 / 28.49 | 0.866 | 0.812 | 0.0757 | 0.2422 |
| 1 | Hann | 0.92 / 0.99 | 0.885 | 0.934 | 0.0552 | 0.2130 |
| 2 | Uniform | 28.27 / 30.39 | 0.877 | 0.933 | 0.0702 | 0.2687 |
| 2 | Hann | 1.16 / 0.96 | 0.919 | 0.942 | 0.0491 | 0.1840 |

## Comparison with the ink annotations

Counts and rates below use threshold 0.5 and the 512 × 512 common area. Pairs show phase A followed by phase B.

| Field | Blend | ROC AUC A / B | True positives A / B | False positives A / B | Recall A / B | False-positive rate A / B | Label Dice A / B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Uniform | 0.8268 / 0.8155 | 66,991 / 68,020 | 38,462 / 40,157 | 0.758 / 0.769 | 0.221 / 0.231 | 0.691 / 0.692 |
| 1 | Hann | 0.8384 / 0.8123 | 68,869 / 66,464 | 47,725 / 40,623 | 0.779 / 0.752 | 0.275 / 0.234 | 0.672 / 0.680 |
| 2 | Uniform | 0.8938 / 0.8065 | 114,870 / 105,487 | 2,576 / 2,604 | 0.677 / 0.621 | 0.028 / 0.028 | 0.800 / 0.759 |
| 2 | Hann | 0.8735 / 0.8050 | 112,486 / 103,558 | 1,812 / 2,261 | 0.662 / 0.610 | 0.020 / 0.024 | 0.792 / 0.751 |

Field 1 phase A gained recall with 9,263 extra false positives. Field 2 phase A removed 764 false positives while recall fell from 0.677 to 0.662. At 0.8, field 2 had zero false positives in either phase for both blends. Field 1 produced 8,550 / 7,036 false positives with uniform and 8,375 / 7,483 with Hann at that threshold. The JSON includes every confusion count and both thresholds. [Numerical results](pherc0841-hann-result.json)

## Center with four tile contributions in both phases

The central 256 × 256 pixels of each field have the same tile count in both grids. In this region, Hann roughly halved the mean phase-to-phase probability change in both fields. At threshold 0.8 the second field's phase Dice changed slightly downwards. In the first field, phase A's false-positive rate at 0.5 increased from 0.252 to 0.509.

| Field | Blend | Seam ratio A / B | Phase Dice at 0.5 | Phase Dice at 0.8 | Mean probability change | ROC AUC A / B | False-positive rate A / B at 0.5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Uniform | 61.44 / 24.47 | 0.914 | 0.551 | 0.0905 | 0.9143 / 0.9783 | 0.252 / 0.432 |
| 1 | Hann | 0.78 / 0.97 | 0.958 | 0.880 | 0.0442 | 0.9232 / 0.9650 | 0.509 / 0.388 |
| 2 | Uniform | 39.06 / 38.98 | 0.865 | 0.769 | 0.0850 | 0.8891 / 0.8171 | 0.037 / 0.170 |
| 2 | Hann | 1.07 / 1.01 | 0.944 | 0.768 | 0.0434 | 0.8490 / 0.8298 | 0.004 / 0.131 |

Hann is available as an optional comparison when assessing tile-grid dependence. These two fields come from one public surface. The measurements support a reduction in measured scoring discontinuities at native tile boundaries. They also show why different blends can give different readings at the same threshold. [Usage](r50-usage.md)
