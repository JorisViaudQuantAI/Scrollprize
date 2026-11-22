# Executed four-pass repaired ResNet probe

**From the [executed result](evidence/result.json):** all four fixed CPU FP32 passes completed. Model inference took 4.947657 seconds; runner wall time including input preparation, acquisition, loading and plots was 40.989570 seconds. Peak process RSS was 943,564 KiB, about 0.90 GiB. The run acquired 342,876,204 application-body bytes through six requests, with **zero CT requests**. It used the published [freeze a0690c8](https://github.com/JorisViaudQuantAI/Scrollprize/commit/a0690c8c7641f4fe592cb26c2d5fff04221adba6).

## Outcomes, retaining both orders

**From the [result](evidence/result.json):** each core has 1,024 known categorical centres. Ink prevalence is 62.99% on PHerc. 0841 and 74.80% on PHerc. 0814. High AP must be interpreted alongside that prevalence. Recall uses each donor/order's descriptive known-background operating point; actual selected background counts are 18/379 and 12/258, respectively, below 5% because of integer counts.

| Control / order | AP | ROC AUC | Ink recall at descriptive 5% background point |
| --- | ---: | ---: | ---: |
| PHerc. 0841 / source | 0.695127 | 0.525692 | 0.133333 |
| PHerc. 0841 / reverse | 0.736008 | 0.540558 | 0.249612 |
| PHerc. 0814 / source | 0.899537 | 0.721583 | 0.381201 |
| PHerc. 0814 / reverse | 0.918204 | 0.787560 | 0.370757 |

**From root's fixed-scale visual review:** all four complete 1.0112 mm core maps are close to white on the 0–1 scale. The maps do not provide legible strokes in these small fields. They have not been stretched separately to create an apparent reading.

**My inference:** this probe does not justify using the repaired model as a validated detector or veto on the team's unread-scroll search. PHerc. 0841 discrimination is weak in both orders; PHerc. 0814 has some ranking signal, but reversing depth does not improve its fixed-background recall. The result does not refute the author's native Scroll 1 validation: these inputs are fine-derived, use a fixed recentered physical slab, and come from different donors.

## Verification and scope

**From the [independent check](evidence/independent_review.json):** exact pinned assets, retained fine tensors and original labels match. An independent Fraction-based integration in a different axis order reproduces both complete model inputs exactly. Original categorical centres, all four normalized input hashes, complete finite maps and 48 scalar metrics agree; maximum scalar discrepancy is 1.11e-16. The check made no new forwards or requests. Unchanged evaluation buffers are reported and guarded by the frozen runner, not independently remeasured by this checker.

The model has 85,651,636 parameters. The larger float32-value count in the checkpoint also includes normalization buffers; it is not a loading mismatch. All weights loaded strictly, no training occurred, and the separate Hecate gate was not changed.

**My inference:** the cores and score grids differ from the Hecate experiment. The tables must not be read as a direct model ranking against its whole-field metrics. This is a small architecture-transfer probe with model-assisted labels, unresolved inherited ancestry, and no target-acquisition or promotion rule. No unread scroll was inspected.

The original [README](README.md), [protocol](protocol.json), [runner](run.py) and tests remain byte-identical to the public freeze so its command can be replayed. See [ChatGPT 021](../../collab/chatgpt-021.md) for the combined team handoff.
