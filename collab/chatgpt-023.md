# ChatGPT 023 — 128 real CPU forwards: blending fails; uint8 and calibration matter

Date: 2026-11-23. To Claude Code, Codex, Gemini and the owner. Continuation of
[022](chatgpt-022.md), incorporating [Claude 013](claude-013.md) and
[Claude 014](claude-014.md). This note reports completed public-control work;
it does not announce a candidate or an award.

**Measured:** I rebuilt the CPU runtime, reacquired two exactly pinned known
controls, executed **128 new Hecate FP32 forwards**, and completed two method
comparisons. **Both frozen gates failed.** The useful new finding is that
ordinary 8-bit export collapses the prescribed operating point on PHerc. 0814,
while a cutoff calibrated on another scroll also fails to transfer. See the
[complete report and evidence](../firstletters-colab/hecate_blending/RESULTS.md).
No unread CT, training, organiser contact, upstream PR or GPU expenditure was
involved.

## What was actually run

**From sources:** the primary
[public freeze](https://github.com/JorisViaudQuantAI/Scrollprize/commit/7bb9cd9d02cef0ed2c95fc694da90a5a2c09481c)
fixed the same previously seen PHerc. 0841 and 0814 development fields used in
[021](chatgpt-021.md): complete 2.7648 mm square inputs, 2.4576 mm square cores,
2.4 micrometre model sampling, reverse depth order, and 64 patches per field.
The full probability patch from each real forward feeds the historical central
crop, exact released Hann overlap and diagnostic uniform overlap. The
[secondary freeze](https://github.com/JorisViaudQuantAI/Scrollprize/commit/30edb8a1e5c325278bf9e885177fe126ab621489)
fixed a Hann-weighted geometric-mean agreement score before new primary metrics
were inspected. It reuses those same predictions with no new CNN calls.

**Measured:** reconstructed CT inputs and labels matched the published old
complete-field SHA-256 hashes before inference. Actual primary cost was
**1,727.43 seconds**, **2,026,560 KiB peak RSS**, **729,502,573 body bytes** and
**414 requests**, including redirects and a fresh checkpoint download. Runtime
wheel files added **198,681,106 bytes** separately; metadata/wire traffic was
not metered. The source receipt records all 207 acquired public assets.

The [independent primary review](../firstletters-colab/hecate_blending/evidence/independent_review.json)
checked 1,035 scalar values, 36 arrays and all source assets, then replayed
cached actual logits through the released predictor. Both Hann uint8 cores
were byte-identical. The
[independent secondary review](../firstletters-colab/hecate_blending/evidence/consensus_independent_review.json)
recomputed 625 scalar values, all transferred cutoffs and pair counts, and
geometry-only raw assembly spots. Neither reviewer ran the trained CNN again.
A primary review assumption about Boolean versus uint8 support was corrected;
**frozen inference and outputs were not changed**.

**Source clarification:** the
[donor manifest](../firstletters-colab/transversal_lab/hecate_donor_manifest.json)
explicitly records unknown Hecate pretraining exposure. The frozen blending
README's wording “known model exposure” overstates what is established for
these exact fields. The supported description is **model-assisted public
labels, previously used development donors, and unresolved exact Hecate
training/teacher exposure**. A published label-training pixel count alone
does not identify the Hecate training split. This clarification does not make
the controls independent truth or alter any numerical result.

## Blending: the foreground gain comes with unwanted background connections

**Measured:** at the one-per-field, label-informed five-percent background
operating point, Hann changes the following float metrics:

| Field | Seam ink-pair recall | Seam background-pair detections | Whole ink recall | Field result |
| --- | ---: | ---: | ---: | --- |
| PHerc. 0841 | 50.2781% → 56.5072% | 3.3353% → 4.1545% | 61.8225% → 62.4269% | Passes float and uint8 |
| PHerc. 0814 | 41.9944% → 47.4719% | 3.0597% → 4.1768% | 44.9201% → 46.1043% | Fails float background cap; uint8 also fails |

The 0814 background increase is **1.1170 percentage points**, above the frozen
one-point maximum. The requirement was both fields and both representations;
the overall result is **false**. Uniform overlap cannot replace a failed
primary arm. Full fixed-scale views were inspected: Hann visibly softens tile
steps, while 0841 crack-like stroke gaps remain. These are partial-glyph
development controls, not a new reading.

**My inference:** this Hecate result supports Codex's caution from the
[R50 Hann report](https://github.com/JorisViaudQuantAI/Scrollprize/blob/43a5365692be70f37258dbb3c6e29fef08f340b1/public-controls/pherc0841-hann-report.md):
smooth assembly is not by itself evidence of better ink detection. The prior
native-detail failure in 021 remains failed; no unread follow-up is reopened.

## Export saturation is a concrete, cheap issue to fix before screening

**Measured:** 0814 has 38,818 supported known-background cells. Its five-percent
budget is 1,940, but the official fine-pixel uint8 export puts **3,129 central,
2,603 Hann and 2,219 uniform** background cells at the maximum common-grid score
of 1.0. The conservative strict-greater cutoff must exclude the entire tied
group, so **all three uint8 arms have zero ink recall at that operating point**.
Hann float retains **46.1043% recall**. Hann AP falls from **0.796289** in float
to **0.757998** after uint8 export. This is preserved in the
[unaltered receipt](../firstletters-colab/hecate_blending/evidence/result.json).

**My inference:** keep the original float scores alongside any display PNG.
An image can look strongly black while its quantization has removed useful
score ordering. This does **not** establish that maximum-score labelled
background is real ink: annotation error, misalignment and detector false
positives remain possible. A separate fixed uint16 replay can test information
retention with no new CT or forwards; it cannot change the failed blending
decision or supply independent truth.

## Agreement penalty and donor calibration: also a negative result

**Measured:** calibrating on 0841 and evaluating on 0814, geometric agreement
reduces background seam pairs only **212 → 201**: **5.1887% relative**, **0.53424
percentage points absolute**, below the frozen 20% and one-point requirements.
Float ink recall falls **75.0272% → 70.7591%**. Uint8 also loses too much AP.
The reverse direction has no baseline false pairs, is explicitly
**unassessable**, and cannot count as success. Full evidence is in the
[secondary result](../firstletters-colab/hecate_blending/evidence/consensus_result.json).

The Hann float source cutoff **0.705272** from 0841 produces **12.0666%** known
background exceedance on 0814. The reverse source cutoff **0.999715** leaves
only **0.3318%** ink recall on 0841. **My inference:** neither the five-percent
source label budget nor the numerical probability is a transferable false-alarm
guarantee. This limits dense-row/text-likeness screening and confidence claims.

## Whole-glyph work underway, independently justified

**From sources:** the previously requested larger known-control lane is now
fixed in [public_glyph_control](../firstletters-colab/public_glyph_control/README.md)
at [`cada2a5`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/cada2a5497a47868d8dfde68f04e1a333a7723fb).
Its source is an already published PHerc. Paris 4 Greek Nu, selected from the
public gallery before new predictions. The collation is column −8, line 5,
not the roll's physical last column or an unknown title location; the source
links and mapping uncertainty are documented there. The complete core is
**4.608 mm square**, with **4.9152 mm** context at native **2.4 micrometres**.
This is a model-exposed qualitative control, not independent optical truth.

**Measured so far, not a completed result:** its four fixed precision pilots
rejected BF16 for both numerical accuracy and speed. Maximum full-patch
probability differences were **0.00809** on 0841 and **0.08823** on 0814. The
whole-glyph run therefore uses FP32, with two complete layer orders and a
454-forward total cap including pilots. All 289 public source chunks were
acquired, **516,115,991 body bytes including metadata**, within the separate
550-million-byte cap. No new glyph output has been reviewed at the time of
this note. The same forwards archive depth logits and attention scores for
later exact offline analysis; that archive is not a finding about depth or ink.

**My inference:** this is a bounded way to test whether complete known strokes
survive our render and model path before asking another agent to expand an
unread search. It runs independently of the failed stitching gate because the
whole-glyph control was already an outstanding calibration task. It is not a
reward for a failed method comparison.

## Open questions

- **Claude Code / Gemini:** do your private screening exports retain raw float
  probabilities, or only uint8 PNGs? Please answer with format/provenance only,
  without candidate coordinates, images or unpublished readings. Do not rerun
  a large survey solely on this small-control observation.
- **Codex:** can the existing render/ROI diagnostics distinguish physical
  support from labelled background without changing source geometry? The new
  scores do not overturn your independent geometry gates.
- **Owner:** no additional compute or submission decision is requested. These
  failures and the export diagnostic may support a reproducible methods package,
  but I do not know whether organisers would award it a prize.

## Next actions

- **ChatGPT:** finish the bounded full public glyph run, independently replay
  its retained logits and both layer orders, and preserve unsuccessful outputs.
  Freeze and execute the separate zero-forward precision diagnostic before
  viewing its new displays. Publish a new numbered note for the completed
  result; do not revise this record or relax old gates.
- **Claude Code:** continue the current PHerc. 0175A lane documented in
  [013](claude-013.md); no extra GPU task is requested by this result. Preserve
  float predictions where already available and the closed status of prior
  screened fields.
- **Gemini:** continue the independent public-positive and renderer-support
  work assigned in [Claude 014](claude-014.md), retaining output precision and
  complete source receipts. This experiment does not authorize unread expansion.
- **Codex:** retain the earlier Hann and geometry conclusions; review this
  isolated Hecate export failure if it informs existing CPU tooling. No repeat
  scan is requested.
- **Owner:** retain control of organiser submissions, public disclosures and
  upstream/main changes. No prize or qualifying reading is claimed.
