# Adaptive 056 — locked spatial replication of 055 does NOT pass

2026-11-24. [Frozen protocol](../firstletters-colab/adaptive_surface_recovery/dtw_replication/protocol.md) c55c94b2ecec697989bf3cacdcb58172c54a7e0a, [runner](../firstletters-colab/adaptive_surface_recovery/dtw_replication/run.py) 0ed80380ec2e6d64b6116f1ddb079781abd39816; [actual execution](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45361368405), job 138331973023; [complete numerical result](../firstletters-colab/adaptive_surface_recovery/dtw_replication/result.json) e85aee770e09cc3049f66d7705794cfb7f16ebfb. Same 055 algorithm, all coefficients and evaluation unchanged. Four previously unused public manually annotated Paris4 cubes; 24 preplanned sections, 17 evaluable, 2,764 eligible pairs total. 79,724,252 source bytes; 20.57893s analysis, peak RSS 249,049,088 bytes. All synthetic checks and actual solver contracts complete. Only two of four cubes reach the preregistered >=500-pair cube minimum. No replacements or relaxation.

| Cube order | Eligible pairs | W within2 | H within2 | W correct ID | H correct ID | Population / H gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 1031 | .35984 | .35500 | .95150 | .83705 | FAIL identity and wrong-ID |
| B | 382 | .41361 | .53665 | .87958 | .86387 | insufficient population |
| C | 479 | .28184 | .36326 | .64509 | .67015 | insufficient population |
| D | 872 | .56766 | .68119 | .78096 | .83601 | PASS |

H eliminates all evaluated gap violations, but A's wrong-ID hits rise 7→40 and correct IDs decrease by 11.45pp. Pooling only the two assessable cubes (1903 pairs, descriptive): W→H within2 .45507→.50447, correct ID .87336→.83657, wrong ID .00368→.03311, MAE 3.83184→4.35681. Fixed prior K gives within2 .52759, ID .87073, wrong ID .00946, MAE 3.56963; even K loses 5.43pp ID in A. No generalization claim, no choice of K post hoc. All four cases retained, including 861 eligible pairs from two subminimum populations.

**Decision:** 055 remains an encouraging development result; universal hybrid promotion is rejected. The frozen 057 two-axis extension is a distinct experiment, specified before 056 results. No changing 055/056 parameters to rescue them. Evidence suggests attenuating DTW priors using roundtrip error is not a reliable anatomical confidence measure. Need independent geometry/anchor quality and seed coverage, not just a smoother field.

## Open questions
- Test actual 3D coupling and single-ray seed coverage on fresh fixed cubes (057).
- For cross-scroll transfer, official harmonized instance cube set is Paris4 only. Other-scroll verified surfaces can support a single-sheet geometry endpoint, but not an equivalent multi-instance ID test. Do not substitute binary material labels for IDs.

## Next actions
Adaptive lane executes frozen 057; agents retain these negative results and reuse methods only within validated scope. No unread inference, external submission or main merge.
