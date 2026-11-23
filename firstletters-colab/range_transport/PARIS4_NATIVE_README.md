# Two direct-S3 native Paris 4 transport controls (published CT only)

**Measured, 2026-11-23:** Two small public GitHub Actions CPU experiments establish that the native, uncompressed 2.4 µm Paris 4 L0 source serves a strict conditional HTTP 206 byte range, preserving selected CT bytes exactly. This is an engineering prerequisite for a future sparse native reader, not an ink result, title search, or evidence about the separate compressed m7 renderer.

## Pre-outcome freezes and real results

The selected public chunk has native shape 109 × 128 × 128, 1,785,856 bytes, C-order uint8. Only the already-frozen Hecate public-glyph fixture z union [22,87) was requested: inclusive bytes **360,448 through 1,425,407**, or **1,064,960 bytes**. This chunk is from an already-published known-text Paris 4 control; no output-dependent source selection occurred.

| Experiment and provenance | HTTP operations | Measured application-body bytes | Outcome |
| --- | --- | ---: | --- |
| [Whole reference and conditional Range](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45148129539), [pre-HTTP freeze](https://github.com/JorisViaudQuantAI/Scrollprize/commit/989730831a6773e97ac3b996bbeda8009488bd91) | metadata 200, complete object 200, conditional 206 | 2,851,024 (208 + 1,785,856 + 1,064,960) | **PASS**, zero mismatched selected bytes |
| [HEAD plus sparse GET](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45148553606), [second pre-HTTP freeze](https://github.com/JorisViaudQuantAI/Scrollprize/commit/886be9fc8a13a90714143afeb3139efdcb8cea29) | HEAD 200 and conditional GET 206; zero whole GETs | **1,064,960** | **PASS**, selected SHA-256 matches the earlier whole-reference slice |

The second run used **40.367% fewer application-body bytes than downloading one complete chunk**. Both runs enforced strong ETags, exact Content-Range and Content-Length, identity encoding, and no redirects, retries or missing-data fill. The code has **7 and 8 deterministic offline checks** respectively. The receipts are [reference result](evidence/paris4-native-public-v1.json) and [sparse result](evidence/paris4-native-sparse-public-v1.json), including GitHub run provenance, numerical cross-checks and success status; no CT arrays are committed.

**Source identity limit:** the first run compares its partial reply with a complete GET and verifies pinned metadata; the second run checks the partial payload hash against that first reference. This proves identical selected content across these two transactions, not permanent immutability or correctness for arbitrary new chunk objects. Future first-contact chunks need a trusted source identity or exact selected hash independently established.

## Full-field budget arithmetic, not a completed 289-chunk experiment

The already-frozen known-glyph context intersects **17 × 17 = 289** full-depth L0 chunks, all needing planes 22 through 86 inclusive. If the same single-range strategy worked on all, projected chunk response bodies would be **307,773,440 B (293.52 MiB)** versus **516,112,384 B (492.20 MiB)** with full GETs; arithmetic saving **208,338,944 B** (40.367%). These are geometry-derived estimates, **not measured full-context transfers**. Metadata and HTTP/TLS framing are excluded.

**Request count is currently blocking under the original 400-call cap:** running HEAD and conditional GET for every chunk would require **578 object requests** plus metadata. The measured workflow therefore cannot simply replace the current complete-glyph reader under that cap. To get to 289 conditional GETs, a separately verified per-object ETag/version manifest or a known-hash integrity policy would be necessary. No such 289-object manifest was established. The separately running whole-glyph task has its own unchanged retrieval protocol; do not restart or alter it based on this projection.

There is no measured speedup, complete field reconstruction, sheet coverage, ink improvement, or Paris 4 title evidence here. Timing is a single fixed-order observation. All requests used a public CPU runner, with no model forward, unread data, paid compute or submission.

## Offline use

From repository root:

- python firstletters-colab/range_transport/paris4_native_range.py --self-test
- python firstletters-colab/range_transport/paris4_native_sparse.py --self-test

Both scripts intentionally pin one specific published chunk rather than implement a generalized reader. The GitHub Actions workflows require a separately committed freeze SHA, enforce unchanged code and protocol before network access, and preserve success/failure receipts. A repeat on the same object is not a second scroll-level control.
