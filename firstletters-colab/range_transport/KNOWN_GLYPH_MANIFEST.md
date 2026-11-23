# Selected-byte identities for the completed published-glyph control

Date: 2026-11-23. This answers the complete-manifest question in
[ChatGPT 026](../../collab/chatgpt-026.md) using already retained public control
data. **No network request, detector inference or new CT acquisition was made.**
The original glyph run and the two earlier Range tests remain unchanged.

## Completed offline result

**From a source — [export receipt](evidence/paris4-known-glyph-manifest-review.json):**
[export_known_glyph_manifest.py](export_known_glyph_manifest.py) verified the
exact already-reviewed result, source-receipt and acquisition-plan hashes before
parsing their chunk identities. It checked all **289 full native chunks**,
their **516,112,384 bytes**, and both metadata objects. Each selected contiguous
65 × 128 × 128 interval received its own SHA256. Reconstruction using the
selected bytes matched all **272,629,760** retained native-union bytes exactly,
with complete coverage and **zero mismatches**.

The selected hash of the previously tested single chunk agrees with **both**
published Range receipts, and its full-object hash agrees too. The result
therefore connects those one-chunk tests to the completed whole-glyph source
cache. It does not turn them into measured 289-object Range retrieval.

The export took **3.627420 s**, with **578,548 KiB** peak RSS. Generated checks
accepted a valid fixture and rejected missing, truncated and same-size corrupt
objects. The source was separately reviewed before execution, including the
correction that pins the three completed-run JSON inputs first. The export
receipt reports the exporter's own measured validation; it is **not** a second
independent implementation of the exporter.

The [manifest](evidence/paris4-known-glyph-selected-manifest.json) SHA256 is
`7c56ebb5623b7313439efd35b4fce147d9c838840ffa22aebdd596f4d171843d`.
It contains only already-published control-source references, full and selected
hashes, byte intervals and assembly geometry. It contains no private cache
paths, credentials or unread locations.

## What this enables, and what has not been run

**My inference:** a future reader can verify selected response content against
these **prior** per-chunk hashes. For this exact known control, content-identity
validation can remove the need for a separate HEAD solely to acquire a current
ETag. The manifest or its immutable repository revision must itself be pinned
before requesting bytes. Every response must be HTTP 206 with the exact
Content-Range, total object length, selected-body length and identity encoding,
and its selected SHA256 must match before assembly. Missing or mismatched
objects must abort; no absent source may become synthetic zero CT.

**Geometry-derived projection, not measured transfer:** 289 selected GETs plus
two metadata GETs would require **291 requests** and **307,777,047
application-body bytes**, within the earlier numerical 400-request ceiling.
This is **208,338,944 fewer body bytes** than the completed whole-chunk
acquisition. No full-context Range reader or transfer was executed here, and
the original whole-chunk acquisition protocol has not been changed. A future
execution requires its own declared and frozen retrieval contract.

**Limit:** selected hashes identify those intervals only. They do not establish
an atomic remote scan snapshot, authenticate other bytes in the objects, or
transfer this uncompressed-layout assumption to compressed data. No measured
speedup, better ink detector or prize result is asserted.

## Reproduce the offline derivation

Use the completed native source cache and a fresh output directory. No model
weights, GPU or network connection are needed. The script deliberately rejects
an existing destination artifact instead of replacing prior evidence.

```bash
python firstletters-colab/range_transport/export_known_glyph_manifest.py \
  --run-dir /absolute/path/to/completed-public-glyph-control \
  --output-dir /absolute/path/to/new-manifest-directory
```

The separate `--self-test` mode runs only the small generated valid/missing/
truncated/corrupt-object checks. These are source-integrity checks, not
manuscript or detector validation.
