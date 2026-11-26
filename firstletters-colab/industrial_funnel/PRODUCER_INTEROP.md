# Producer interoperability: canonical receipts and independent admission

## Result

The shared A03 file admission verifier now revalidates every producer envelope
against the shared receipt contract after the producer's `read` method returns.
The check covers the envelope schema, canonical spec, envelope digest, job and
attempt identity, output and input CAS bytes, resources, and assigned orders.
`file_receipts.validate` and the existing A02 order-plan validator remain the
authorities for those rules. No receipt is converted or rewritten.

This closes a reproduced integration error. With a synthetic, correctly linked
REGISTER / CONTROL / GEOMETRY / COARSE / RANK batch, the unmodified Gemini reader
accepted a single-order receipt under a legacy standard spec. Before this fix,
A03 returned `COARSE=COMPLETE` and `RANK=READY_FOR_OPERATOR` when given that reader.
The same files were blocked when read through the shared `FileReceipts` class.
After the fix, A03 blocks the incompatible receipt and its child for either
reader, reports `halted=true`, and preserves `costs_complete=false` because it
cannot treat an invalid receipt as complete accounting.

Both the old and new verifier return `dispatch_authorized=false`. The defect was
in the reported batch admission state; this experiment did not launch a job,
spend GPU time, or demonstrate that anyone had actually used this combination in
production.

## Pinned producer and evidence

The replay uses the public Gemini producer at commit
[`da94312727c7e95f7f7fdc2d7481d502265434c5`](https://github.com/JorisViaudQuantAI/Scrollprize/commit/da94312727c7e95f7f7fdc2d7481d502265434c5):

| Source | SHA256 |
| --- | --- |
| `gemini_lane/industrial_funnel_adapter.py` | `7a43443989eaaa08d0c9bf1f8f9b98fac95bf768fbaa48ccebd3772589c57bec` |
| Its `leak_guard.py` dependency | `01b8142e63be80be1462913f9c397f08083685b0b63f03a41df3362dba48193c` |
| Shared A03 before this fix, from RFC commit `52f90c8713d28d98be64262499deb67f10d062a6` | `c1cd9eaae83d59158d52974d74f5484607210a988f47a17a291b49053abb3163` |

The producer module and dependency are checked before import. The replay records
the current shared verifier's source hash as well. The compact
[`producer_interop_result.json`](producer_interop_result.json) contains observed
outputs and the archived before-fix comparison. All test payloads, unit IDs,
accounting values, controls, and NumPy arrays in that experiment are synthetic.
No full native map or unread-region coordinates were uploaded.

## Reproduced contract differences

| Case | Observation on the pinned producer | Shared interpretation |
| --- | --- | --- |
| Paired receipt schema | A legacy paired receipt is accepted by both validators. Adding `order_mode="paired_both_orders"`, as the Gemini bridge does by default, is accepted by Gemini and rejected by the shared receipt validator. | Same envelope schema name does not establish compatibility when receipt field sets differ. |
| Frozen experimental spec | The shared A02 spec includes `order_policy="single_order_plus_audit"` and a CAS-bound plan. Gemini requires exactly the eight legacy spec fields and rejects this spec. | The plan and arm-specific recipe/calibration are part of immutable job identity. They cannot be reconstructed from a later receipt flag. |
| Audit selection | Twenty synthetic single-order units are each accepted with a declared audit rate of 0.15 and `audit_selected=false`; zero paired units exist. | A per-receipt rate declaration cannot verify a 15% population draw. This demonstrates missing cohort binding, not a violation by any observed real campaign. |
| Arrays versus metadata | A real float32 NPY fixture contains one supported NaN and has a correct file hash. `native_evidence.validate_native` rejects it with `NONFINITE_SCORE`; Gemini's bridge stores the opaque bytes and returns `outcome=COMPLETE` plus its negative-claim flag. | Byte integrity and a metadata consistency check do not establish native array validity. The negative-claim flag is a producer declaration, not scientific authorization. |
| Detector family pairing | `studio_desc` plus `ahn_s0_asc` produces `orders=["asc","desc"]` and is accepted by the bridge. | Those two labels do not establish a pair from the same model family, input depth selection, checkpoint or recipe. The fixture does not assess either detector's performance. |
| Resume versus admission | One COMPLETE attempt followed by one FAILED attempt still gives `completed=true` in Gemini resume. A03 rejects the conflicting attempt history. | Resume is local evidence availability. The shared `FileReceipts.resume` also remains availability-only; it deliberately reports that admission is still required. |
| Backend-dependent A03 interpretation | Before the fix, the permissive reader let a legacy-spec single-order COARSE result advance its RANK child. After the fix, the shared validator rejects it even when the same producer reader is used. | Admission rules must be independent of a storage reader's supported schema version. |

The native-array contradiction does **not** mean the shared receipt validator
checks image semantics. It also treats artifact bytes as opaque by design.
`native_evidence` must be run separately on the six actual native arrays and
their manifest. A receipt-compatible payload can still contain invalid arrays,
and a valid native bundle does not establish same-sheet anatomy, calibrated
recall, text absence, or a discovery.

Gemini's bridge also fills the metadata fields
`padded_pixels_excluded_from_area` and `missing_chunk_fill_excluded` with `true`
from summary-level inputs. The replay records these declarations without
treating them as measurements. Independently, its Studio 31-plane convention
uses ascending planes 7 through 22 and descending planes 23 through 8. Those
lists are shifted by one plane, rather than an exact reversal of the same band.
The existing shared `assess_pair` validator is the appropriate place to check
exact array reversal when both native bundles actually exist.

## Smallest integration path

1. Preserve existing producer envelopes and their hashes as historical evidence.
   Do not silently remove `order_mode`, replace a spec, or create a new job hash
   and describe the result as the original receipt.
2. For new authorized work, construct the spec with the shared `validate_spec`
   and, when applicable, freeze the A02 plan before observing model outputs.
   Use the plan's assigned orders, recipe and calibration. The shared receipt
   contains `orders`; producer-specific diagnostics can remain separate CAS
   artifacts. Their field names do not change the receipt contract.
3. Reuse compatible storage primitives without copying admission rules.
   Gemini `CrossPlatformCAS.verify` checks the actual bytes and returns a `Path`,
   matching the shared store interface. The new A03 guard calls shared validation
   over that same store; it does not instantiate a POSIX-only writer on Windows.
4. Validate native input, probability and mask arrays independently. Pair only
   compatible model/recipe/depth bundles, with explicit uncertainty for missing
   native evidence. A same-family pair still needs actual reversal verification.
5. Inspect the pinned batch with A03, using a trusted inspection time and current
   control revocations. Local `resume.completed` and a geometry-only flag cannot
   substitute for batch or scientific admission.

No producer-specific script, G1 scheduler, G4 worker, cross-platform writer or
audit sampling algorithm was reimplemented here. The only common-code change is
`validate_file_envelope` and its invocation immediately after `files.read` in
`file_admission.py`. The diagnostic inspector delegates to that same guard.

## Read-only inspection

Run from the repository root, using an existing private receipt and CAS folder:

```bash
python firstletters-colab/industrial_funnel/producer_interop.py inspect \
  --receipt PRIVATE_RECEIPT.json \
  --cas PRIVATE_CAS_DIRECTORY
```

Exit status is 0 for receipt compatibility and 2 for incompatible or unreadable
input. The command creates no receipt, CAS object, directory, order plan or batch.
It always leaves `production_complete=false`,
`scientific_admission="NOT_ESTABLISHED"`, and `dispatch_authorized=false`.
The CLI checks envelope self-consistency; the API additionally accepts an
independently supplied expected spec and attempt. The report explicitly says
which identity check was requested.

Inspection reports are private diagnostics: exception details may reference
inputs supplied by the operator. The committed replay report uses synthetic
fixtures only. Treating the inspector's compatible status as an execution or
negative-text decision would exceed its scope.

## Replay the pinned producer

The repository must contain the pinned Gemini commit. The replay needs NumPy and
SciPy because the existing producer imports them; it does not import a GPU model
or make network calls. Extract the two source files into a temporary folder:

```bash
interop_source=$(mktemp -d)
git archive da94312727c7e95f7f7fdc2d7481d502265434c5 \
  gemini_lane/industrial_funnel_adapter.py gemini_lane/leak_guard.py \
  | tar -x -C "$interop_source"
python firstletters-colab/industrial_funnel/producer_interop.py replay \
  --producer-path "$interop_source/gemini_lane/industrial_funnel_adapter.py" \
  --producer-sha256 7a43443989eaaa08d0c9bf1f8f9b98fac95bf768fbaa48ccebd3772589c57bec
```

This opt-in replay writes synthetic receipts, CAS payloads and native fixtures in
temporary directories, then removes those directories. It prints the measured
interoperability outcomes. It uses the existing analytic native fixture from
`test_native_evidence.py`, so the shared test files must be available. The
archived before-fix observation in the committed JSON is provenance, not a claim
that the current fixed code still accepts the old path.

## Verification and limits

The following command passed **113 tests**, including the **12 new offline
interop tests**, on Linux:

```bash
cd firstletters-colab/industrial_funnel
python -m unittest test_producer_interop test_file_receipts test_order_audit \
  test_native_evidence test_control_plane
```

The new regressions cover the real producer's incompatible receipt shape,
preservation of compatible paired behavior, unchanged input files, missing CAS
bytes, envelope/hash/job mismatches, frozen A02 identity, conflicting attempts,
and rejection of an unpinned producer before its code executes. The separate
pinned-producer replay exercised all seven cases above.

A replay of Gemini's own 14 tests with its committed validation JSON present
passed 13 tests in this Linux environment. The remaining test flagged ordinary
`cross-agent` wording because the generic account name `agent` is not in that
producer's generic-account allowlist. This is an environment-dependent false
positive; no identifying filesystem path was found in that public JSON. The
producer's privacy guard was not modified here. Windows behavior and the
producer's historical 48-test Windows claim were not independently revalidated.

The fixture store obeys the shared `verify(SHA256) -> Path` integrity contract.
An intentionally dishonest store or arbitrary hostile code is outside this
patch's model. Scientific calibration, reviewer reliability, GPU metering,
cross-batch spending and dispatch authorization remain outside this diagnostic
and admission correction.

## Latest producer head: `e310d6e`

The subsequent head `e310d6e188b42d196662bc75ffec61f36335ae24` leaves the adapter
and `leak_guard.py` byte-identical to the pins above, so it does not resolve the
observed interoperability differences. Its five changed files add a CPU MIT
ARAP flattener, four tests and aggregate PHerc0826 preflight evidence, update the
coordination note, and increase GPU thermal-polling frequency. No new native
coarse/fine probability arrays or file receipts are included in that commit;
the coordination note describes additional executions and an ongoing GPU
positive control, which are distinct from committed native evidence.

One new metadata-contract mismatch is confirmed: the flattener writes
`parameterization_method="mit_arap_2d"`, but the unchanged adapter rejects that
name. Using the pinned producer extracted above (identical at the new head), the
exact probe with the published positive-control QA values is:

```bash
PYTHONPATH="$interop_source/gemini_lane" python -c "from industrial_funnel_adapter import validate_geometry_contract; validate_geometry_contract({'parameterization_method':'mit_arap_2d','shear_abs_p95_deg':7.371,'pct_area_gt_20deg':0.151,'area_inflation_true_over_nominal':0.9893})"
```

It raises `ValueError: Unknown parameterization_method: 'mit_arap_2d'`.
`latest_head_compatibility` in the result JSON records the head, source hashes,
probe input/output, and complete changed-file list. This is a compatibility
finding only; it does not assess ARAP image quality, ink sensitivity or scientific
admission. No method was silently relabeled or added to the producer's allowlist.
