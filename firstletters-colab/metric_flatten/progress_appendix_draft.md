# Draft toolkit appendix for the owner's review

**Status:** draft only. No submission or external announcement has been made.
This text describes the completed geometry experiment, including its failed
adoption gate. It does not claim a new reading or ink-detection improvement.

## Contribution

We provide a small CPU package that reproduces the published PHerc. 306B
metric correction, measures its physical distortion, and records exactly which
source triangles support its inverse samples. The ARAP algorithm and its
original successful correction are credited to the
[published 306B submission](https://github.com/SirProdigle/pherc0306b-first-letters/tree/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/code/surface).
Our additions are portable reconstruction, explicit geometry contracts,
source-triangle provenance, adversarial support checks and reproducible
failure evidence.

## Demonstrated result

**From the [frozen experiment and receipt](RESULTS.md):** the baseline exactly
matches the released float32 XYZ and support. In the fixed published text
group, median absolute vertical stretch error falls from 13.6008% to 0.2971%.
The guarded inverse sampler preserves that correction, but retains only
97.3905% of whole-output baseline support, failing the declared 99% gate.
All 20,172 nodes in the padded published claim crop remain valid.

**From the [support analysis](results/support_analysis.json):** every one of
45,189 common samples that uses an original source triangle agrees exactly
between inverse samplers. Large discrepancies are concentrated in changed
source connections. In a deliberately removed 7 × 7-vertex square, the
Delaunay baseline fills 36 of 36 unsupported interior queries; the proposed
sampler fills none. The perturbation is designed missing support, not observed
manuscript damage.

The complete cached experiment takes 49.532 seconds and 362,446,848 bytes peak
RSS, with no CT acquisition or CNN work. Code, source hashes, gates, receipts
and a replay command are public in this repository.

## Why it may help

**My inference:** geometry checks of this kind can prevent spending inference
time on images whose metric or source connections differ materially from the
intended sheet. The package supports reviewable geometry decisions; it does
not establish correct anatomical surfaces or improved reading sensitivity.

## Limitation the submission must retain

The whole-output adoption gate failed. Integer-sample provenance also does not
certify the support of a renderer's cells or edges between those samples.
The package is a diagnostic and experimental exporter, not an unattended
replacement. A renderer integration that honors source support is the next
implementation task. No effectiveness or novelty claim beyond the measured
scope should be added without further evidence.

## Open questions

- **Owner:** does this belong as an evidence appendix to the existing toolkit
  effort, after the renderer support contract is completed?
- **Codex / Claude Code:** which rendering path can expose that contract with
  the smallest implementation change?

## Next actions

- **ChatGPT:** preserve reproducible evidence and the failed gate.
- **Codex / Claude Code:** evaluate renderer support propagation on public
  controls.
- **Owner:** decide whether to use or submit this draft. Nothing is submitted
  automatically.
