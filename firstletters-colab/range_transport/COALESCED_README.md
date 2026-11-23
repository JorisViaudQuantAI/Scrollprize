# Coalesced HTTP ranges — executed on a previously published PHerc. 0841 control

This is a **real CPU-only transport comparison**, not an ink detector, a generated-data-only test, a Hecate inference, or a test of unread papyrus. It extends the earlier exact-value [one-row / one-plane experiment](README.md) without changing that experiment's frozen protocol.

## Fixed before new transport

[coalesced_protocol.json](coalesced_protocol.json) and [coalesced_control.py](coalesced_control.py) were frozen together in [commit 6bf5791](https://github.com/JorisViaudQuantAI/Scrollprize/commit/6bf5791c49d317d450a10e695376902817099196), before any HTTP request in this experiment. The source is the same known public, SHA-256-pinned, uncompressed C-order uint8 Zarr chunk used in the preceding control. The selected interior 16×64×64 box matches a coarse Hecate input **in array shape only**; it is not a mesh-following render, model input, known ink mask, or proof of true 3D scientific support.

For every row of this box, compute exact byte offsets in the source object. Merge sorted, non-overlapping byte spans when the omitted gap is no larger than a fixed policy threshold. Reconstruct only the requested row bytes and compare them with the independently hash-checked whole object. The thresholds and theoretical request/body counts were frozen before the live run:

| Policy | Maximum merged gap | HTTP partial requests | Expected partial-body bytes |
| --- | ---: | ---: | ---: |
| Literal rows, arithmetic only; **not executed** | 0 B | 1,024 | 65,536 B |
| Within-plane | 64 B | 16 | 130,048 B |
| Across-planes | 8,256 B | 1 | 253,888 B |

Each live range must have HTTP 206, exact Content-Range and length, strong ETag equal to the pinned full object's response, and conditional If-Match. No retry, ignored-Range fallback, replacement chunk or missing-as-zero semantics are accepted. The fixed live order is full-object reference, within-plane, across-planes; the budget is 1,500,000 application-body bytes and 24 requests including redirects.

## Actual measured result

The first CI attempt [failed **before transport**](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45102791692): shallow checkout could not retrieve the earlier freeze commit. The workflow was amended to fetch full history; the experiment code and protocol were **not changed**. The complete [second CI run](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/45102879762) succeeded with the frozen code, and its exact [compact receipt](evidence/coalesced-public-v1.json) was published afterward.

| Observed operation | Body bytes actually read | Time on the CI runner | Values |
| --- | ---: | ---: | --- |
| Whole source GET | 1,064,960 | 0.909454 s | Original SHA-256 matched |
| Within-plane, 16 ranges | 130,048 | 2.957942 s | All 65,536 ROI voxels match |
| Across-planes, 1 range | 253,888 | 0.314420 s | All 65,536 ROI voxels match |

The complete run read **1,448,896 application-body bytes** and used **19 HTTP requests**: one redirect (302), one whole-object (200) and seventeen partials (206). Both partial strategies yield exactly the same ROI SHA-256; neither has a mismatched byte. Peak process RSS was 51,652 KiB on Linux. Three newly added deterministic tests also passed on this runner and on a local CPU, exercising byte accounting, reconstruction and failure cases.

**Scope:** reducing requests required reading more irrelevant bytes. In this single ordering the one-range policy was faster than the 16-range policy, and both partial policies avoided most of the complete object's body; no robust performance distribution, repeatability, S3 support or generalized bandwidth speedup is established. Timing is for GitHub Actions accessing this one HF CDN object, not the previous local environment. HTTP/TLS header and wire bytes are not measured. The complete object was downloaded **only to verify** exact values; a production sparse reader need not request that reference. This result does not repair the native Paris 4 full-context budget in [ChatGPT 019](../../collab/chatgpt-019.md), which concerns different objects and sampling.

## Reproduction

Use a full-history checkout and Python 3.11 with NumPy installed:

```sh
python -m unittest discover -s firstletters-colab/range_transport -p test_coalesced_control.py -v
python firstletters-colab/range_transport/coalesced_control.py \
  --freeze-commit 6bf5791c49d317d450a10e695376902817099196 \
  --output /tmp/scrollprize-coalesced-replay.json
```

The CLI refuses to overwrite a prior receipt or write raw CT into Git. It prints a compact JSON record and writes it outside the worktree. Keep model arrays and any unread data private. To adapt this planner for an actual rendered surface, first certify native source version/encoding, complete depth and spatial halo, missing-chunk and acquisition support semantics, and a source-specific request/byte budget.
