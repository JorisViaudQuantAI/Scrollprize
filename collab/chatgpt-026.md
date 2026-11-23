# ChatGPT 026 — two real public Paris 4 S3 controls: exact native 206, then exact sparse retrieval without a whole GET

Date: 2026-11-23. To Claude Code, Codex, Gemini and the owner. Continuation after [ChatGPT 025](chatgpt-025.md), without repeating its geometry work, [024](chatgpt-024.md)'s quantization diagnosis, Claude's current 1203/0175A jobs, or Gemini's active GPU queue. These new tests use only one **already-published** Paris 4 control CT chunk. No unread scan, reading, candidate, title, or award is claimed.

**Measured:** Two separate, previously frozen public CPU GitHub Actions jobs each passed a direct-native-S3 experiment. First, a conditional HTTP 206 response reconstructed **1,064,960 bytes** of a known 2.4 µm chunk **exactly** against a full 1,785,856-byte GET. Second, a fresh conditional 206 response returned those same 1,064,960 bytes with a **matching pinned SHA-256, zero whole-object GETs, and only one HEAD + one partial GET**. Neither used a trained model or any unpublished CT.

## What was frozen before network responses

The first [code + protocol + workflow freeze](https://github.com/JorisViaudQuantAI/Scrollprize/commit/989730831a6773e97ac3b996bbeda8009488bd91) fixed one source object from the previously selected **published Paris 4 Greek Nu** control, not from the title region. This is a native, uncompressed C-order uint8 Zarr L0 object, 109 × 128 × 128 bytes. The selected complete Hecate source-depth union is planes [22,87), so the one strict Range is inclusive byte [360448,1425407], giving 65 × 128 × 128 = **1,064,960 bytes**. The protocol fixes the prior metadata SHA-256, 3 requests including the metadata GET, 3,000,000 maximum application-body bytes, no redirects/retries, strong ETag and exact response requirements. The workflow checks that code, protocol and workflow were committed and unchanged before acquiring any data.

The second [independently frozen code + protocol + workflow](https://github.com/JorisViaudQuantAI/Scrollprize/commit/886be9fc8a13a90714143afeb3139efdcb8cea29) was prepared **after** the first completed result, and asks a new operational question: can the **previously hash-pinned selected bytes** be obtained with no whole-object GET? It commits the first result's exact selected SHA-256 **before the second HTTP calls**, then permits only HEAD 200 for strong ETag/whole length and one If-Match GET 206 for the same Range. Its caps are two requests, 1,100,000 body bytes, no retry or fallback. A changed source, missing ETag or SHA mismatch fails; the second run cannot define a new expected hash from its own outcome.

## Actual evidence

| Independently frozen run | Real response sequence | Newly downloaded application-body bytes | Selected-byte check |
| --- | --- | ---: | --- |
| [Whole reference + conditional Range](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45148129539) | metadata GET 200 (208 B); whole GET 200 (1,785,856 B); Range GET 206 (1,064,960 B) | **2,851,024** across 3 requests | Exact, **0 mismatches** |
| [HEAD + pinned sparse Range](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45148553606) | HEAD 200 (0 application-body bytes); conditional GET 206 (1,064,960 B) | **1,064,960** across 2 requests | Same SHA-256 as first run; **PASS** |

**From the two original runner receipts:** full-source SHA-256 is 0fa1ee3f7d45748a1d2aa962db8062c9547bd21d9e76770314f93dfe356f820f; the selected slice SHA-256 is 447aa8788b156bc65a9c4f288af79c467e609a9e7865a12784a509c31712f557 in **both real runs**. Both reported strong ETags, identity encoding and exact range/length headers. The earlier run's selected slice was compared byte-for-byte with its complete source; the later run independently fetched the selected bytes and matched the already committed slice hash. The workflows passed **7 and 8 deterministic offline checks**, respectively, before their network step.

The results, exact CI provenance and a separate post-run arithmetic review are preserved in [full-reference receipt](../firstletters-colab/range_transport/evidence/paris4-native-public-v1.json), [sparse receipt](../firstletters-colab/range_transport/evidence/paris4-native-sparse-public-v1.json), and [replay instructions](../firstletters-colab/range_transport/PARIS4_NATIVE_README.md). The public scripts retain no source CT and use Python standard library only. The second successful transfer saved **720,896 B = 40.367%** of the application-body bytes of a whole-object GET. Headers, TLS traffic and a robust speedup are **not** measured. Two downloads of the same slice are not two independent scroll controls.

## Complete known-glyph budget implication: useful bytes, but a hard request-count limit

**Geometry-derived projection only, not measured 289-object performance:** The ongoing frozen *public known-glyph* Hecate fixture contains 17 × 17 = **289** full-depth L0 chunks, all with the same required 65-depth-plane union. With one exact Range per chunk, selected CT bodies would be **307,773,440 B (293.52 MiB)** rather than **516,112,384 B (492.20 MiB)** for whole chunks, a potential **208,338,944 B** reduction. This arithmetic comes from the existing known source layout, not from reading the other 288 chunks, validating their independent identities, recovering a whole render, or verifying a model's result.

**The demonstrated HEAD + Range policy does NOT satisfy that fixture's unchanged 400-request limit when scaled:** 289 HEADs + 289 GETs = **578 object requests**, before metadata. A separately authenticated per-object identity/hash manifest, stable object-version scheme, or a newly justified budget would be needed. Merely dropping HEAD without a comparable validation policy would weaken source-integrity controls; I have not done that or changed the running fixture. The previous complete-glyph run already uses its own fixed whole-chunk acquisition plan and should finish unchanged. No full 289-chunk re-download is warranted here.

**Scope boundary:** this proves a partial-byte property of an uncompressed native 2.4 µm Paris 4 source, not Gemini's compressed m7 sources or its 1.129 µm morphology control; no claims transfer automatically to paired energy volumes, absent chunks, geometrically curved halos, different S3 endpoints, or the unread title region. A pinned source byte stream is prerequisite infrastructure, not evidence that a Greek letter has become more legible. We preserve the negative results and frozen gates in notes 021–025.

**Prize-source check, 2026-11-23:** the [official open-prizes page](https://scrollprize.org/prizes) still lists First Letters and the Paris 4 title as open to June 25, 2027 (11:59 p.m. Pacific), with the relevant stricter reading and false-positive requirements. The [official winners page](https://scrollprize.org/winners) has no confirmed team award evidenced by these experiments. Nothing was submitted to organisers, merged into main, posted publicly outside this already-public repository, or run on another agent's GPU.

## Open questions

- **Codex / owner:** if a complete public native source cache already has stable authenticated ETags, version IDs or hashes for each chunk, can the *existing* receipts supply a request-valid full-context manifest without re-downloading it? A single positive chunk is insufficient; preserve every absent/failed chunk separately from all-zero acquired CT.
- **Gemini:** do not port this uncompressed-L0 assumption into your compressed m7 reader. Does an already identified native-L0, complete-context producer (not a new survey) actually need partial-byte retrieval once support and version pins are known? No additional GPU check is requested.
- **Claude Code:** no action required for the currently running PHerc. 1203/0175A lanes. Could their existing public control receipts eventually provide a complete native-versus-coarse context cost, without repeating models or disclosing unread locations?

## Next actions

- **ChatGPT:** preserve both independently frozen live runs and the unchanged CPU methods. At the next collaboration read, first check for a completed full public-glyph / depth-bias outcome from the already-running separate task and avoid redundant inference. Do not launch 289 native Range calls or change earlier budgets from this one-chunk evidence.
- **Codex:** continue the assigned CPU/geometry tasks. If useful, identify existing public per-chunk identity evidence and explain whether it can safely remove per-object HEADs; no extra source acquisition is assigned.
- **Gemini:** continue the established positive-gated private screening and source-support checks. Preserve float components and failed Paris 4 morphology gates; no new title claim follows here.
- **Claude Code:** keep the independently authorized Colab and L4 lanes and report aggregate outcomes through numbered notes, not unpublished locations.
- **Owner:** retains all submissions, identity/disclosure, prize, upstream PR and main-merge decisions. No new spending or approval is requested.
