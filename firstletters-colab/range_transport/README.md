# A bounded HTTP Range check on an existing public control

**Status at protocol freeze:** code and offline tests are ready; no live transport
outcome has been observed for this experiment. The root agent must publish the
protocol and runner before the first live object request. The runner enforces a
byte-identical Git freeze of those two files.

## Measured result: exact values, smaller Range bodies, higher observed latency

**From a source — measured locally after the
[public freeze](https://github.com/JorisViaudQuantAI/Scrollprize/commit/e2b20a4bf3864669e30ee1db894eae4ea39de941):**
the [executed receipt](evidence/result.json) passes all three fixed boxes with
zero voxel mismatches against both the fresh whole GET and the independently
acquired old cache. The whole GET followed one redirect; the five subsequent
Range responses were all 206, with the same strong ETag and the exact requested
bounds and lengths. There were seven counted HTTP requests in total.

| Observation | Whole-object reference | Five Range requests |
| --- | ---: | ---: |
| Application body bytes returned | 1,064,960 | 32,808 |
| Elapsed time | 10.923828 s, including redirect | 19.515980 s |
| Source bytes matched | Entire pinned chunk | Every selected byte |

**Calculated from the measured receipt:** the Range bodies are 3.080679% of the
unique full object, a 32.46-fold reduction for these particular slices. The
verification experiment intentionally also downloads the complete reference, so
its actual completed-read total is **1,097,768 bytes**, not 32,808. This verifier
is not a production downloader that already avoids the initial whole GET.
Peak process RSS was 31,208 KiB.

**My inference:** Range transport can recover the selected values exactly on this
observed public HF path, but this serial five-request sequence was slower than
the one complete-object retrieval. The fixed whole-first order and one observation
do not establish a general speed ratio. An operational reader needs a request
budget and an interval-coalescing policy as well as a byte budget. The next
source-specific experiment must verify its version acquisition and transport
contract without claiming that this HF result proves S3/Paris 4 behavior.

The [independent receipt review](evidence/independent_review.json) recomputes the
Range plans, byte totals, body hashes and reconstructed-box hashes from the old
cache. It issues no new request. It checks the retained observations; it is not a
second independent wire capture. The frozen runner and protocol remain unchanged.

## Question and fixed source

**My inference:** the native whole-chunk budget failure described in
[ChatGPT 019](../../collab/chatgpt-019.md) motivates checking whether partial byte
requests can preserve exact voxel values while reducing representation bytes.
This experiment answers that transport question for one already acquired public
Hugging Face surface object. A success would not establish Range behavior on the
Paris 4 S3 endpoint, a full physical sampling footprint, or an ink result.

**From a source:** the prior public-donor acquisition receipt and cached metadata
identify an uncompressed, unfiltered, C-order uint8 chunk of shape 65 × 128 × 128,
containing 1,064,960 bytes. The metadata and full cached object were checked locally
against their recorded SHA-256 values. The selected object is the first
lexicographically sorted non-metadata PHerc. 0841 chunk in that old receipt; no ink
score or pixel preview selected it. The relevant source is also documented in the
[existing donor manifest](../transversal_lab/hecate_donor_manifest.json).

- [Selected public object](https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/841/auto_grown_20260220144552896/auto_grown_20260220144552896.zarr/0/0.17.75)
- [Array metadata](https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/841/auto_grown_20260220144552896/auto_grown_20260220144552896.zarr/0/.zarray)
- Full object SHA-256: 94a67b09ecb71d4fbfe05cfba14bdd8156748cc0eade2e3a74321db2bb71911b.
- Old acquisition-receipt SHA-256: 8fc138c17ba26453355575d6d1721437ce5f9b9fd705ce37cfb3758c6bb2eb26.
- Exact source pins, selected boxes, cached slice hashes, request order and limits:
  [frozen_protocol.json](frozen_protocol.json).

This is a transport control on an already published data product. The chunk was
acquired in a prior task; the current design fetches no new region or labels and
does not inspect or predict letters.

## Three fixed boxes, six logical requests

**Proposed procedure:** retrieve the complete object once, require its bytes and
hash to match the independent old cache, then request the following boxes in
fixed order. Bounds are half-open chunk-local ZYX coordinates, not manuscript
candidate coordinates.

| Case | Lower ZYX | Upper ZYX | Range requests | Requested bytes |
| --- | --- | --- | ---: | ---: |
| Partial row | (0,16,12) | (1,17,28) | 1 | 16 |
| Three partial rows | (5,7,120) | (6,10,128) | 3 | 24 |
| Two complete planes | (31,0,0) | (33,128,128) | 1 | 32,768 |

These are five Range requests after one whole-object GET. Adjacent selected bytes
are coalesced, including row and depth boundaries in the two-plane case. All
returned voxels must equal both the fresh full GET and the old cached reference,
byte for byte. Slice hashes were computed only after fixing selection and boxes.

The requested Range bodies total 32,808 bytes, versus 1,064,960 bytes for the one
unique whole chunk needed to supply all three boxes. The full GET is an experimental
reference cost and is reported separately; it is not falsely multiplied by three.
The fixed whole-first order and single observation do not provide a fair
cold-cache throughput benchmark.

## Transport contract

**From a source:** HTTP specifies conditional requests, byte ranges, partial
responses and their representation bounds in
[RFC 9110](https://www.rfc-editor.org/rfc/rfc9110.html).
[The Zarr v2 specification](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html)
defines the chunk shape, compressor, filters, dtype and C/F byte order used to
interpret the uncompressed object.

**Implementation choice, stricter than universal HTTP compatibility:**

1. Use HTTPS, no retries, no automatic whole-object fallback. Count each redirect
   as a request; allow at most three redirects per logical request and 24 requests
   total.
2. Request identity encoding. A full response must be 200 with the exact object
   Content-Length and a strong ETag. Its complete bytes must match the prior pin.
3. Keep the resolved endpoint in process memory. Subsequent Range requests send
   If-Match with that strong ETag. Require 206, the exact inclusive Content-Range
   including total object length, the exact Content-Length, the same strong ETag,
   and absent or identity content/transfer encodings.
4. Reject weak or missing ETags, duplicate critical headers, exposed HTTP body
   lengths that differ from the expected length, version changes, absent/wrong
   ranges and transformed encodings. Python's HTTP parser bounds the exposed body
   by Content-Length; bytes sent beyond that framing are not detected by this
   check.
5. Reject 403, 404, 412 and other unexpected statuses. Never turn a forbidden,
   missing or failed object into a filled array. If a Range request returns 200,
   stop before reading its body.
6. Read at most 5 MiB of application response-body bytes in the complete run,
   including a possible overlength probe. Do not retain fetched pixel arrays in
   the repository.

The byte account records three different quantities: requested representation
length, server-declared response length, and application-read body bytes returned
by completed read calls. **It is not a hard wire-byte cap.** Socket buffering,
TLS, HTTP headers, bytes beyond the declared HTTP body framing, and unread
redirect or error bodies are not measured. If a read raises, bytes consumed
internally before the exception may not appear in the body count; that run fails
and produces no successful savings result. A server may already have transmitted
bytes before a bad status or range header causes the client to close the response.

An ETag is a version validator, not necessarily a content hash. This check also
uses independently acquired complete SHA-256 and exact byte comparisons. It
observes correct conditional-request headers and matching responses; it does not
prove behavior under a future concurrent object mutation.

## Execute only after publishing the freeze

Use the task's already retained prior acquisition receipt and source cache.
The cache keys are SHA-256 hashes of canonical public URLs. The runner verifies
the old receipt, repeats the lexical selection, and validates cached metadata and
bytes before networking.

    python firstletters-colab/range_transport/run_transport.py \
      --freeze-reference PUBLISHED_FULL_COMMIT \
      --prior-receipt /path/to/prior/source_acquisition.json \
      --cache-dir /path/to/prior/source-cache \
      --output-dir /tmp/scrollprize-public-range-v1

The output directory must be new and outside every Git worktree. A failed run
leaves its result receipt and must not be overwritten or silently replaced with
another object. Detailed output includes the safe canonical public source and
aggregate request fields; redirected signed URL queries are never recorded.
No credentials or account sessions are used.

The result records every request status, declared length and bytes read, exact
slice matches, timing, peak RSS, protocol and runner hashes, and the freeze commit.
On failure it records the stopping reason. Body/encoding/status checks happen
before accepting a result. A failed length/hash comparison is not a negative ink
finding.

## Offline validation

    python -m unittest discover -s firstletters-colab/range_transport -p 'test_*.py' -v

**Measured locally before live transport:** 17 tests pass, using generated bytes
and fake HTTP responses. They cover C-order row/plane offsets, invalid boxes,
ignored ranges, exact range bounds and lengths, compression/transfer encodings,
strong/weak/missing/changing ETags, forbidden and missing sources, body/request
caps, counted HTTPS redirects, signed-query omission, actual If-Match/Range
request headers, whole-object mutation, and the complete six-request algorithm.
An offline test using Python's actual HTTP response parser also demonstrates that
trailing bytes beyond Content-Length are outside the exposed-body check.
The separately checked old cached reference also matches its pinned hash.

## Limits that remain after a possible pass

This is one object and one observed HF transport path. It does not demonstrate
Range support or efficiency on S3, compressed chunks, sharded Zarr, absent objects,
or all chunks in a volume. Header acceptance and exact values do not establish
same-sheet geometry, original acquisition support, a correct layer order,
scientific contrast, or legible ink.

The next scientific I/O plan must still freeze complete paired physical supports,
registration/PSF halos and depth sampling, then count the actual intervals and
requests. Sparse bytes can reduce payload while increasing latency; one successful
small range does not make a large multi-range surface run practical.
