# Native-003 — actual G4 follow-up, backend-independent A03 admission, exact audit-total bound

Date: 2026-11-24. Continuation of [native-002](chatgpt-native-002.md), on
`chatgpt/industrial-native-evidence` / PR #7. Owner-requested continuation;
no main merge, GPU dispatch, prize submission or new letter claim.

## Latest integration state

Claude `331daddad65e1f0801bd498cbfe639154e8ad028` has already merged native PR7
head `9ac322f` through `fa42c87`; PR7 itself remains draft/open. This update
integrates the newer RFC **`4c4433ef058078c4a59af38326d759d7036bb1eb`**, including
075's thirty A03 tests and evidence-bound status. Adaptive head observed:
`70c3efdbf540e3e11786c11fa693c25a8cfaa7be`. Gemini head observed:
`e310d6e188b42d196662bc75ffec61f36335ae24`.

[Gemini-010](https://github.com/JorisViaudQuantAI/Scrollprize/blob/e310d6e188b42d196662bc75ffec61f36335ae24/collab/gemini-010.md)
now owns the ARAP flattener, seven 0826 preflight targets and the four additional
0125 screenings. We did not duplicate these experiments. Its new 0125 candidate
remains pending; its renders are not included in that commit or linked for an
independent visual reread. The full native 0211 coarse/fine benchmark remains
unavailable in the reviewed updates.

## 1. G4: a concrete support fix, and measured limits of inference reuse

[Protocol, report, maps and replay](../firstletters-colab/industrial_funnel/geometry_dedup_followup/README.md).
CPU execution on the twelve actual public PHerc0211 fitted/snap/flat w033–w036
grids, with all 36 directed non-self pairs and 12 self-pairs. Final run: 70.40 s,
52 geometry files/2,771,097 bytes; no CT or model inference.

- Original bilinear target sampling omitted usable boundary vertices:
  **2,935/164,591 self-comparison vertex occurrences rejected**.
- Preserving original usable target vertices restores **100% on all 12 real
  self-pairs**. The [patch](../firstletters-colab/industrial_funnel/geometry_dedup_followup/gemini_preserve_vertices.patch)
  applies cleanly to the pinned real Gemini adapter and was verified there,
  not only in our reconstruction. It is ready for the Gemini owner to integrate.
- Of 39,560 accepted non-self directed correspondences, 18,427 (46.58%) have
  more than 3 vox maximum separation between sign-aligned sampling rays over ±8 vox;
  279 exceed 6 vox. These are occurrences, not physical area or missed letters.
- Reference-near source occurrences total 13,588: 11,833 target-near,
  1,585 intermediate,166 unknown, only 4 target-far. All 4 far distances remain
  inside ±8 vox. No large text-loss or out-of-window-reference claim.
- Exact masks and dense/normal nearest indices match upstream for 48/48 pairs.
  Four hundred thirty-two numeric arrays, all pair aggregates and four
  near-to-far trace records are published. All three atlases were inspected.

The correction is not universally monotone for non-self pairs: one flat
w035→w036 point switches to a nearer Euclidean target whose normal component
fails the threshold. This is reproduced in the real patched producer.

**Integration decision:** geometry can propose overlaps. Exact inference reuse
needs actual input/context/recipe identity. Approximate skipping still needs
native-map recall evidence. Retain pre-snap lineage for physical accounting.

## 2. A03: revalidate after producer reads

[Code, probes and migration](../firstletters-colab/industrial_funnel/PRODUCER_INTEROP.md).
We reproduced a composition flaw using the actual Gemini backend: a legacy-spec
COARSE with only asc and a self-declared audit flag unlocked ordinary RANK when
A03 trusted that reader, while the shared reader blocked the same bytes.

`file_admission.validate_file_envelope` now rechecks canonical identity and
receipt semantics after `files.read`, using the existing CAS and validator.
The incompatible COARSE/RANK is blocked with either backend; incomplete costs
halt admission. No dispatch is implemented or authorized.

Seven synthetic producer probes preserve the exact old observations. Twelve
new regressions cover compatibility/rejection; one independent regression
covers valid legacy and all three A02 arms after PARTIAL→COMPLETE attempts via
both canonical and permissive readers. This work is distinct from 075's tests
of the canonical reader and preserves 075's `evidence_verified` scope.

The Gemini adapter and leak guard are byte-identical at new head e310d6e.
One additional connection gap is recorded: ARAP emits `mit_arap_2d`, currently
rejected by the adapter's parameterization allowlist. No blind allowlist
expansion or claimed validation of ARAP ink sensitivity was made.

## 3. Optional exact bound on the stratified total

[Proof, implementation and results](../firstletters-colab/industrial_funnel/followup_a03/README.md).
For a complete zero-event stratified audit, maximize the zero-event probability
over allocations of a fixed total using decreasing marginal hypergeometric
factors and exact fractions. Independent exhaustive verification: 5,511 plans.

Synthetic 2×(20,3) upper total at 95% improves 26→14; 6×(20,3) improves 90→17 compared
with the current sum of Bonferroni bounds. The existing bound is valid and also
serves simultaneous per-stratum inference. The new comparator gives the total
only. No actual recall or GPU saving was measured and no production gate uses
this result. Nine statistical tests pass.

## Verification and handoff

**168/168 combined tests pass**, zero failures/errors/skips, on the integrated
RFC 4c4433e tree: 146 upstream/native tests plus 22 added here. See
[receipt](../firstletters-colab/industrial_funnel/native_tests_followup.json).
Offline geometry numeric verification and the original 53-file public visual
evidence verification also pass. CI now recalculates geometry evidence without
network, CT or models; the full geometric replay is separately reproducible.

[French report](../firstletters-colab/industrial_funnel/FOLLOWUP-003.md).
Remaining evidence request is unchanged: actual paired native coarse/fine maps,
matched controls, attempted-unit denominator and independent reviews. Our
original weak O-like candidate remains unconfirmed. These contributions prevent
unsupported pruning and improve audit accounting; they do not announce a new
discovery or a deployed industrial search.
