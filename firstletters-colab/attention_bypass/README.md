# Retrospective Hecate attention-bypass diagnostic

**Decision: do not promote an unrestricted maximum through depth from this coarse-model panel.** It reduced mean matched-background recall in both ink-bearing donors and both depth orders. This result does not reject every possible surface-constrained projection or test Hecate's fine-resolution checkpoint.

This is a **retrospective analysis of previously observed development outputs**. An exact local analysis protocol was saved before the new calculation, but the outputs already existed and had informed earlier work. This portable package was prepared **after the outcome**. Neither the package nor the original calculation is a publicly preregistered independent holdout.

## What was tested

The old Hecate experiment saved sixteen coarse 3D logit planes and the released learned-attention 2D output for each central 32 × 32 core. The fixed alternative takes the maximum over **all sixteen** logits at every pixel, then applies sigmoid and the same float32 probability clipping. The two arms have identical known-label support. No model was run, trained, or modified.

The comparison retains all **48 physical contexts in both stored depth orders**: 96 context/order records from PHerc. 0009B, 0841 and 0814. All cached features and labels are hash checked. The core is only 0.3072 mm across. The annotations are model assisted, inherited training exposure is unknown, and there is no field-scale reading endpoint.

For each arm and original context, its threshold is computed from known background **after the complete depth maximum**. With `k = floor(0.05 * n_background)`, the threshold is descending background score `k`, and selection uses strict `score > threshold`. Thus the empirical background budget includes the whole depth search; ties cannot exceed the budget. This descriptive operating point is not a transferred target threshold or a statistical false-positive guarantee.

AP/AUC are undefined for pure-class contexts. Every background-only, ink-only and unsupported/unknown context remains in [evidence.json](evidence.json); none is replaced according to its result.

## Measured result

These are equally weighted means over all mixed contexts in each donor. Recall is measured at each context's own descriptive 5% background operating point. AP means average precision.

| Donor / order | Mixed contexts | Released recall | Depth-max recall | Released AP | Depth-max AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| PHerc. 0841 / source | 3 | 0.14824 | 0.09816 | 0.71873 | 0.71088 |
| PHerc. 0841 / reverse | 3 | 0.11439 | 0.10498 | 0.69648 | 0.68551 |
| PHerc. 0814 / source | 2 | 0.05920 | 0.04749 | 0.36557 | 0.42175 |
| PHerc. 0814 / reverse | 2 | 0.99226 | 0.98984 | 0.99116 | 0.99173 |

The prespecified retrospective screen required both ink-bearing donors, in reverse order, to gain at least 0.05 absolute macro recall and reduce the missed-ink fraction by at least 20%, with AP nonworse in at least 75% of mixed contexts. Neither passed. PHerc. 0814 also had a baseline ceiling above 0.95 recall. Its small AP increase does not establish useful recovery.

The original coverage limitations remain:

| Donor | Mixed | Background only | Ink only | No known supported labels |
| --- | ---: | ---: | ---: | ---: |
| PHerc. 0009B | 0 | 5 | 0 | 11 |
| PHerc. 0841 | 3 | 12 | 1 | 0 |
| PHerc. 0814 | 2 | 2 | 1 | 11 |

A secondary threshold-transfer check also failed: a threshold calibrated to 4.989% of PHerc. 0009B's background selected 16.405% of PHerc. 0814 background under depth-max/reverse, with only 57.207% ink recall. The released projection selected 17.162% with 97.555% recall. Lower background selection alone therefore did not indicate recovery. All orders, donors, per-context counts and transferred rates are retained.

The optional argmax-depth consistency diagnostic had insufficient newly recovered ink-neighbor edges in either reverse-order donor. It does not establish physical ink depth. No current native-resolution field scores or metrics were read, and no gates from that separate experiment changed.

## Replay from existing caches

Dependencies: Python, NumPy, SciPy, scikit-learn and Pillow. Supply the **exact original** cached feature and label directories; this script does not acquire data or regenerate predictions.

```bash
python firstletters-colab/attention_bypass/replay.py \
  --old-run-root ../old-hecate-run \
  --old-role-root ../old-hecate-role-run \
  --output ../attention-bypass-replay.json
```

Each supplied root contains `features/` and `source-cache/`. Label files use the original URL-hash cache keys. `--manifest` can point to a relocated copy of the same hash-pinned donor manifest. The output must be a fresh path outside Git. The script compares every retained result and public-control feature hash against this package and emits `exact_replay_match`.

Measured original analysis body: **0.585 seconds**, with **zero new forwards and zero network requests**. Runtime does not include library-import startup. The original code and protocol SHA256 values are recorded separately from the after-outcome portable wrapper in both JSON files. Replay writes its own code/protocol hashes and timing; those are not represented as the original executed hashes.

## Evidence and interpretation

- [protocol.json](protocol.json) records the fixed analysis, source hashes, original execution identities, and explicit after-outcome replay status.
- [evidence.json](evidence.json) contains all 96 context/order outcomes, donor summaries, secondary checks and limits.
- [replay.py](replay.py) repeats only the cached-array calculation.
- [ChatGPT 012](../../collab/chatgpt-012.md) documents the earlier feature experiment, its role amendment, label coverage and exposure limits.
- The [pinned Hecate release](https://huggingface.co/scrollprize/hecate/tree/9cb86e500e944b11a06a7020403cde5dffb5bcb2) documents the supervised 3D output and learned depth collapse. Its 3D predictions may respond to fibres or neighboring sheets.

**Inference:** the tested shortcut raises background scores along with ink scores. This panel does not support spending new coarse-model forwards on unrestricted whole-depth maxima. It says nothing decisive about a physically justified, independently validated surface-constrained projection. No unread scroll was accessed and no reading or prize is claimed.
