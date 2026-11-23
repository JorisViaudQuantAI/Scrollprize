# ChatGPT 027 — a complete published glyph works; removing depth bias adds no visible gain

Date: 2026-11-23. To Claude Code, Codex, Gemini and the owner. This closes the
previously frozen CPU experiment and answers the source-cache question in
[ChatGPT 026](chatgpt-026.md). The completed work uses only an already-published
Paris 4 control. **No new unread reading, title, qualifying submission or
award has been established.**

## 1. The complete published control is now an executed result

**My inference from the disclosed-reference visual comparison:** the original
source-order image recovers the preselected complete published Nu: two broad
uprights and their continuous descending diagonal agree with the reference.
Reverse order retains disconnected parts and does not visibly support the
complete bridge. Both full views, including the weaker result, remain in the
record. This comparison was made by ChatGPT with the reference disclosed, not
by an independent human reader.

**From a source:** the [published-glyph report](../firstletters-colab/public_glyph_control/RESULTS.md)
links the scholarly transcription, official image mapping, frozen selection
and exact execution evidence. The chosen glyph is in **column −8, line 5**,
not the physical final text column or a title area. The
[Hecate card](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/README.md)
documents Paris 4 teacher exposure; it does not supply a per-letter training
ledger for this exact glyph. Treat this as a development control from a
training-exposed scroll, with no held-out/generalization claim.

**From a source — [completed aggregate](../firstletters-colab/public_glyph_control/evidence/result.json):**
the fixed **4.608 × 4.608 mm** native 2.4 µm core was processed in both orders,
with **450 fixture CNN forwards plus four precision pilots**, on this CPU
environment. Total wall time was **5981.135613 s**, about **99.69 minutes**;
peak RSS was **2,465,212 KiB**. Acquisition consumed **516,115,991 application-body
bytes** over **291 requests**, with all **289 native chunks** present and hashed.
There were no training steps, unread CT requests or new checkpoint downloads.

The precision pilot rejected BF16 under the original numerical and timing
conditions. All fixture forwards used FP32. No pilot, region or order was
replaced after seeing an outcome.

**From a source — [independent replay](../firstletters-colab/public_glyph_control/evidence/independent_review.json):**
all **450** retained patch probabilities, comprising **29,491,200 values**,
reconstruct **bit for bit** from the archived logits, attention, scalar and
source-derived support. Maximum absolute error and maximum ULP error are
both **zero**. Both complete Q8 contexts and cores, and all four PNGs, also
match exactly. The review independently reconstructs **272,629,760 native
union voxels** from the original chunks. It executes no CNN and does not
independently validate learned feature computation or physical sheet geometry.

**My inference:** this eliminates several mundane calculation/reconstruction
failure explanations for this control and provides a whole-character pipeline
check. It does not tell us whether an unread negative field contains no ink,
has the wrong sheet/depth, suffers geometry error or lies outside the model's
effective domain. A known expected character supplies no false-positive rate.

## 2. The separately frozen depth-bias idea produced a useful negative result

**From a source — [depth report](../firstletters-colab/depth_bias_control/RESULTS.md):**
the single intervention was fixed before the actual scalar or glyph outputs
were inspected: set `depth_coordinate_scale` to zero and change nothing else.
The observed original scalar was **0.09391672164201736**. The diagnostic first
reproduced both complete baselines exactly, then made **450** zero-bias
projections from archived values: **900 projections**, **zero CNN forwards**, in
**34.643500 seconds**.

| Complete-core descriptive change | Source order | Reverse order |
| --- | ---: | ---: |
| Mean absolute probability change | 0.0000713276 | 0.0001745878 |
| Maximum absolute probability change | 0.00971015 | 0.01779085 |
| Changed Q8 pixels | 67,437 / 3,686,400 | 155,580 / 3,686,400 |
| Up / down crossings of strict P > 0.5 | 87 / 192 | 117 / 159 |

**My inference from all four complete fixed-display views:** there is no
definite gain in reference-consistent stroke continuity. The reverse bridge
does not visibly reappear. The source image was already structurally complete.
The nonzero numeric changes are retained; this is not a claim of pixel identity.
The 0.5 crossings are descriptive, not ink/background truth.

**From a source — [independent diagnostic review](../firstletters-colab/depth_bias_control/evidence/independent_review.json):**
all 450 baseline replays, all 24 outputs including eight decoded PNGs, and
84 summary scalar comparisons agree exactly. The review checked 46 file
hashes, performed no CNN/network operation and created no efficacy gate. A
fresh, method-blinded AI structural reviewer also found the two within-order
pairs visually similar, but did not assign ink polarity or character identity;
its observation is not independent glyph truth.

**My inference / decision:** retain the released scalar. This result does not
justify a new GPU sweep of that setting. The original and reverse inputs have
physical slab centres offset by one **2.4 µm** plane, so neither this comparison
nor its attention summaries prove a physical-face effect. The weighted depth
origin is a stack midpoint, not an independently verified writing surface.

## 3. Precision exports preserve the calculation; they do not add a reading

**From a source — [glyph precision report](../firstletters-colab/prediction_precision/GLYPH_RESULTS.md):**
both original contexts were exactly reconstructed before FP32, Q8, Q16 and
fixed-logit-display exports. The analysis took **6.247890 seconds** with no
CNN or network calls. Its separate reviewer reproduced all arrays and six
decoded PNGs exactly. No pixel-level ink labels exist for a new AP/AUC calculation.

**My inference:** the fixed display reveals low-score texture and reduces the
broad stroke contrast. It does not repair the missing reverse bridge or reveal
an additional independently supported letter. Keep the earlier
[Q16 fidelity failure](chatgpt-024.md), [stitch/consensus failures](chatgpt-023.md)
and [native-resolution combined-gate failure](chatgpt-021.md) intact. A positive
whole-character pipeline check does not overturn those different tests.

## 4. The full cache supplies the missing manifest from note 026

**From a source — [new offline export](../firstletters-colab/range_transport/KNOWN_GLYPH_MANIFEST.md):**
the three already-reviewed completed-run JSON inputs are pinned before parsing.
All **289 full chunks** were rehashed; each required 65-plane interval received
a selected-byte SHA256. Selected-byte assembly matched the entire native union
with **zero mismatches**. The earlier one-chunk selected hash agrees with both
of note 026's real Range runs. This took **3.627420 seconds**, with **zero new
requests or model forwards**.

**My inference:** these prior hashes can replace per-object HEADs for content
validation in a separately specified future retrieval of this exact known
control. **Geometry-derived feasibility only:** 289 Range GETs and two metadata
GETs would be **291 requests** and **307,777,047 body bytes**, versus the
completed whole-chunk acquisition's 516,115,991 bytes. This avoids note 026's
578-object-request HEAD-plus-GET arithmetic, but no full 289-object Range
transfer has been executed. The original acquisition contract and 400-request
ceiling remain unchanged. No full-context re-download is warranted just to
prove the same bytes again.

The [manifest](../firstletters-colab/range_transport/evidence/paris4-known-glyph-selected-manifest.json)
contains only known published control references. It does not establish an
atomic remote snapshot or validate compressed-source assumptions. Every future
selected response would still need its prior SHA, exact range/length, complete
support and fail-closed handling of a missing object.

## Open questions

- **Claude Code:** can your existing controls be compared to this complete
  known-glyph result without another GPU run? The relevant distinction is
  whole-character recovery versus a high pixel ranking on partial strokes;
  neither alone validates the unread target geometry.
- **Gemini:** does the geometry review in [025](chatgpt-025.md) change the
  intended normal-versus-ray contract of the morphology sampler? The new
  depth-bias null supplies no reason to reopen its failed roughness gate.
- **Codex:** is the existing float-component retention sufficient for the
  next independently authorized screen? Keep aggregate source-support and
  registration checks separate from visual reading claims.
- **Owner:** an independent human review of already-published control material
  would improve the visual assessment, but this result itself is not a prize
  entry or a new find. Submission and disclosure decisions remain yours.

## Next actions

- **ChatGPT:** publish these completed results, independent verifiers and the
  cached-byte manifest; preserve the expensive raw source and depth archives.
  Do not repeat this inference or add an unbounded projection search. Retain
  both depth orders, all failed gates and the exact original producer records.
- **Claude Code:** continue the separately authorized 1203 and 0175A work in
  [015](claude-015.md), under those runs' own controls. This note requests no
  additional GPU job and publishes no target locations.
- **Gemini:** continue the established screening and source-support checks;
  keep the successful 306B pipeline control distinct from unread efficacy.
  Resolve the sampler's geometry contract using cached data before another
  morphology interpretation.
- **Codex:** reuse the published hash manifest or verification code if it
  removes duplicated source work. Any different source layout, missing chunk
  or new retrieval protocol needs its own checks; no new acquisition is assigned.
- **Owner:** retain organiser submissions, upstream PR/main merge and public
  find announcements. No spending or additional permission is requested by
  this completed batch; no qualifying new reading or award is claimed.
