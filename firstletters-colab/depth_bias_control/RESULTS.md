# Removing the depth-coordinate bias did not visibly improve stroke continuity

Date: 2026-11-23. **My inference from the complete fixed-display comparison:**
setting the archived depth-coordinate scalar to zero produced no definite
improvement in the published control's stroke continuity. Source order remains
reference-consistent and reverse order remains incomplete. This is a retained
negative result for one intervention on one development control from a scroll
with documented teacher exposure. Exact per-letter training membership is not
established by the inspected sources.

## The intervention was fixed before the parameter or images were seen

**From a source:** the [four-file public freeze](https://github.com/JorisViaudQuantAI/Scrollprize/commit/5b48d79fde5a41f8239b8548f1609d4349de70ff)
preceded inspection of the actual archived scalar, logits, attention or glyph
outputs. The [protocol](protocol.json) permits exactly one intervention:
replace `depth_coordinate_scale` by zero, retaining every raw attention value,
3D logit, support mask, physical depth, precision and Hann weight. It does not
permit a scalar sweep, attention-temperature search, depth maximum or a newly
chosen preferred order.

**From a source — [completed run](evidence/result.json):** the recorded scalar
was **0.09391672164201736**, so the intervention was not a parameter no-op.
Before any counterfactual, all 450 original patch probabilities and both full
Q8 contexts reproduced **exactly**. The run then replayed all 450 zero-bias
projections. Total: **900 projections, zero CNN forwards, checkpoint loads,
network requests, new CT requests or training steps**. Runtime was
**34.643500 s**, peak RSS **1,005,480 KiB**. All consumed source artifacts
remained unchanged.

## Measured changes on the complete fixed cores

**From a source — [run aggregate](evidence/result.json):** both cores contain
3,686,400 acquired, nonempty CT columns. Nonempty CT is not an ink or tissue
label. No pixel was removed from the comparison.

| Descriptive quantity | Source order | Reverse order |
| --- | ---: | ---: |
| Mean signed probability change | −0.0000196655 | +0.0000377138 |
| Mean absolute probability change | 0.0000713276 | 0.0001745878 |
| Maximum absolute probability change | 0.00971015 | 0.01779085 |
| Changed Q8 pixels | 67,437 (1.82935%) | 155,580 (4.22038%) |
| Upward crossings of strict P > 0.5 | 87 | 117 |
| Downward crossings of strict P > 0.5 | 192 | 159 |

The **0.5** boundary was predeclared as descriptive. It does not classify
physical ink/background, and the crossing counts are not accuracy estimates.
No label-dependent cutoff or efficacy gate was added after the result.

**My inference from visual inspection:** baseline and zero-bias images remain
structurally very similar within each order under the same fixed logit display.
The zero-bias reverse image does not visibly restore the missing continuous
bridge. The original Q8 source image already contains the expected complete
structure, so the intervention has not supplied a missing known letter.

**From a source — [secondary structural review](evidence/secondary_visual_review.md):**
a fresh AI reviewer received four unchanged complete views with neutral labels,
without the expected character or method labels. It grouped A/C and B/D as
visually similar and resolved no definite added connection within either pair.
The fixed mapping was A = source/zero, C = source/baseline; B = reverse/baseline,
D = reverse/zero. The reviewer described tonal structures and fine dark traces,
and explicitly did not decide which structures were ink. This supports only
the narrow observation of structural similarity, not an independent reading,
ground truth or validation of the expected glyph.

## Attention describes the calculation, not the physical surface

**From a source — [run aggregate](evidence/result.json):** each arm summarizes
921,600 supported internal positions across 225 overlapping tiles. Physical
positions repeat. The median effective-plane counts and weighted mean depths
were:

| Order | Quantity | Original | Zero bias |
| --- | --- | ---: | ---: |
| Source | Effective plane count, exp(entropy) | 5.03467 | 5.03721 |
| Reverse | Effective plane count, exp(entropy) | 10.26005 | 10.30285 |
| Source | Mean depth relative to source plane 54, µm | +10.21283 | +10.15195 |
| Reverse | Mean depth relative to source plane 54, µm | −32.89028 | −32.18434 |

These are medians of per-position summaries, not the attention distribution
of an independently labeled ink population. Plane 54 is a stack reference,
not a measured papyrus surface. Source features span physical planes 23–84;
reverse features span 85–24. The archive was not reversed again.

**My inference:** on this control, the scalar term is not the main explanation
for the strong difference between the two order outputs. Removing it leaves
the learned attention and 3D logits unchanged, so this experiment does not
isolate those mechanisms, rule out all depth errors or demonstrate correct
physical-sheet selection. The more diffuse reverse attention is an
association, not a causal explanation of the missing stroke.

## Verification and decision

**From a source:** an independent cached-array review is recorded in
[evidence/independent_review.json](evidence/independent_review.json), with its
separate [verifier](verify_result.py). It executed the pinned released
projection body with archived head outputs, without learned model operations
or imports of the producer's analysis helpers. All **450 baseline patch
comparisons** were bitwise exact before the **450 zero-bias replays**. All
**24 saved exports**, including **eight decoded PNGs**, agreed exactly; the
maximum difference across **84 checked summary scalars** was **0**. It
verified **46 file hashes**, took **36.105354 s** and used **955,376 KiB** peak
RSS, with no CNN, checkpoint load or network request. Producer timings are
reported from the original receipt, not independently system-traced.

The original run receipt and visual-review records remain separately
hash-bound; they are not rewritten to replace their initial pending-review
fields. Numerical reconstruction does not independently validate the CNN
features, physical sheet or visual reading.

**My inference / decision:** retain the released scalar for the team's current
inference. This result gives no reason to spend GPU time repeating its removal
on unread targets. A different mechanism would require its own declared
question and validation; this failed intervention is not permission for an
unbounded search of projections. All earlier failed method gates remain
failed. No new reading, title, independent generalization or award is claimed.
