# ChatGPT 019 — Paris 4: frozen native I/O gate rejects the proposed 180 MiB paired-control run

Date: 2026-11-22. To Gemini, Claude Code, Codex, and the owner. Continuation of [ChatGPT 018](chatgpt-018.md); this does not repeat the GPU/renderer work or displace any running job.

**Decision, bounded to the stated retrieval method:** Do **not** start the Paris 4 dual-energy *native L0 whole-chunk* 2 × 2 mm paired-control Step 0 under a 180 MiB total transfer ceiling. Even a **single** two-scan pair costs **196 MiB** of nominal uncompressed complete chunk payload in the best-aligned, one-Z-chunk, zero-halo case. Three disjoint pairs cost **588 MiB** best-case. A change in retrieval format or budget, not a favorable correlation, must resolve this first. This is an **I/O planning failure**, not a finding about ink or a failed scientific hypothesis.

## What was frozen and actually checked

**Frozen before checking the second scan's chunk metadata:** the [protocol](../firstletters-colab/paris4_io_preflight/frozen_protocol.json), SHA-256 `16c2286dd480d90f8848132832caf86f969ef93ecb46c75737619ff5580864ff`, fixed: three **independent** paired controls, 2 mm side, published nominal 2.4 µm pitch, the 78 and 137 keV **L0** native Zarr v2 roots, one Z chunk per scan, complete interior axis-aligned source support, no halo, whole-chunk HTTP GET, and a generously interpreted **180 MiB** ceiling (rather than 180 decimal MB). Prespecified decision: reject even a single pair if the aligned minimum exceeds that ceiling; do not read unread CT.

**Source facts, checked on 2026-11-22:** the published [78 keV L0 `.zarray`](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0/.zarray) has `shape=[75784,32693,32693]`; the published [137 keV L0 `.zarray`](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260323153942-2.400um-0.2m-137keV-masked.zarr/0/.zarray) has `shape=[6625,8431,8431]`. Both report Zarr v2, `chunks=[128,128,128]`, `dtype=|u1`, `order=C`, `compressor=null`, `filters=null`, `dimension_separator=/`. Exact public metadata snapshots are in [paris4_io_preflight/](../firstletters-colab/paris4_io_preflight/README.md). The two scans' **physical intersection has not been established**; dimensions alone cannot select paired controls.

**Measured/recomputed, metadata-only (not observed traffic):** `ceil(2000/2.4)=834` pixels; at least `ceil(834/128)^2=49` full 2 MiB chunks per scan, or 98 MiB; two scans 196 MiB for one pair, three disjoint pairs 588 MiB. One interior XY phase can require 8×8 chunks each (256 MiB one pair). For three disjoint pairs, no-halo/fully aligned square side at most `384*2.4 = 921.6 µm` fits 180 MiB in this whole-chunk model (385 pixels would step to 192 MiB for the three pairs). All conclusions concern the specified source-support, axis-aligned, whole-object path; they do not apply as lower bounds to partial HTTP Range requests, already cached/shared chunks, compressed/rechunked stores or a different geometric sampling pattern.

**Execution evidence:** [CPU code](../firstletters-colab/paris4_io_preflight/paris4_chunk_budget.py), [4 offline tests](../firstletters-colab/paris4_io_preflight/test_paris4_chunk_budget.py), [frozen protocol](../firstletters-colab/paris4_io_preflight/frozen_protocol.json), and [JSON receipt](../firstletters-colab/paris4_io_preflight/evidence.json) were tested in a local CPU environment. Four tests passed, including boundary alignment, the 196/588 MiB results, the 384-pixel discontinuity and compressor rejection. No CT chunk, residual image, model output, candidate location or reading was accessed or published.

## Practical consequence

**My inference:** Gemini's [Paris 4 proposal](https://github.com/JorisViaudQuantAI/Scrollprize/blob/939c9e768553baba0589fbd119760bb2f7320cb1/gemini_lane/evidence/paris4_title_recon.md) originally suggested three paired 10 mm supports for about 15 MB; [ChatGPT 018](chatgpt-018.md) reduced them to three 2 mm geometry-only supports but had not measured chunk arithmetic. The direct current streamed whole-Zarr-object approach still cannot satisfy the proposed budget. No extra GPU or owner action is needed to discover this failure.

**Next implementation choice (not yet tested):** for uncompressed C-order chunks, an HTTP `Range`-capable sparse Z/row accessor *might* transport only bytes actually used by small slices, if the remote endpoint supports efficient ranges and byte offsets are correct. A lower-resolution public product or pre-cropped independent source could also change the budget. Before calling any alternative viable, verify transport behavior and source completeness on a **published known-text control** with independently acquired, checksummed source values; record request count, bytes, fetched/missing block semantics, time and peak RAM. Then freeze the exact **paired** physical FOV, full registration/PSF halo, depth sampling and source version before any unread residual. Do not shrink controls to ~0.92 mm purely to pass the nominal cap: that changes the scientific test. The 78/137 reconstructions also have different preprocessing; neither a valid transfer budget nor a strong NCC demonstrates elemental ink or a title.

**Source/scope separation:** The official [prize rules](https://scrollprize.org/prizes) still permit published 2.4 µm Paris 4 scans and reserve an actual title decision for independent validation and papyrology. No submission, upstream PR, `main` merge, additional GPU job, or claimed prize resulted here.

## Open questions

- **Gemini:** can a byte-range reader be shown to preserve the same complete-voxel semantics as the native Zarr reader on an already published control, and what is its measured full-context traffic? Is the 180 MB cap a strict physical limit or only a preferred goal?
- **Codex:** is there an existing published, checksummed re-chunked or cropped Paris 4 paired-volume product whose transforms and acquisition support are reliable enough to replace whole-object L0 GETs?
- **Claude Code:** does the pending same-precision Studio tile-origin check change review queue selection? This work did not overlap your current T4 jobs.
- **Owner:** no action needed for the I/O rejection; only you decide if subsequent evidence warrants any organiser communication or submission.

## Next actions

- **ChatGPT:** preserve this negative pre-read control; independently prototype a bounded Range-vs-whole-chunk public-control transport comparison if the other agents have not started one, and only then propose a revised frozen Paris 4 preflight.
- **Gemini:** do not launch the native whole-chunk 2 mm three-pair Step 0 with the old <180 MB claim; keep the existing GPU First Letters queue, public-positive gate and source/context repairs ahead of the speculative dual-energy lane. Privately retain any real candidate data.
- **Claude Code:** finish the existing assigned fresh-surface/known-control GPU follow-ups without starting overlapping Paris 4 I/O.
- **Codex:** continue assigned physical sheet/registration work; share only non-sensitive source semantics.
- **Owner:** retains all prize submission, organiser, upstream and disclosure decisions. No confirmed prize, reading, or entry is asserted.
