# Parker audit: final bounded transport tranche

## State before this amendment

The scientific audit remains frozen at `bcd67e3005132b74e295964cea99f4a23d63608c`; its four scientific files are unchanged. The canonical-route resumption was published at `410f233f624efbc3689bd3e33253ff662e7cc0bc`.

**Observed:** Attempt 002 reached 135 of 151 verified source members, then stopped on three 30-second network timeouts. It produced no new HTTP 403 response. It also produced no CT pixel arrays, raw profiles, descriptor scores or scientific result. Exactly 16 originally declared members remain, totaling 11,105,233 range bytes. Earlier failure, download and start receipts are preserved and pinned by SHA256.

## Final transport policy

`parker_transport_final.py` imports the exact original scientific runner and retains every original source member, hash, CRC, sample, depth, descriptor, label shift, metric orientation and interpretation. It changes only transport:

1. Obtain an ordinary redirect from the same original public OSF HEAD endpoint; verify HTTP 200, the frozen ETag and archive length.
2. Reuse that unchanged URL only while its local age is less than 20 seconds. Refresh through the same public endpoint under a lock when older. Never alter a signature, token, URL parameter or access control.
3. Use a 45-second timeout and at most four download workers. Each still-missing member gets at most two further attempts. A failed first attempt forces a fresh ordinary redirect before the second attempt.
4. Verify all reused members against their previous SHA256, in addition to the original CRC/header checks. Keep every receipt sanitized: no signed URL, request identifier or credential.
5. Carry forward the 105,902,334-byte cumulative source reservation and 13,000,000-byte conservative preflight metadata reservation from attempt002. The original 300 MB source and 350 MB total archive limits remain cumulative. Successful bodies are counted separately from conservative reservations.
6. Retain one numerical thread and the absolute 2 CPU-hour process cap. Existing profile or result outputs still block execution. An exclusive final-tranche start receipt prevents another execution in the same cache.

This is the last allowed transport tranche in this session. If it fails, preserve the incomplete acquisition and do not retry again. No scientific outcome existed before this amendment, and no analysis setting is chosen using an outcome.

## Reproduction

Publish this amendment and wrapper before the final source requests. Preserve both previous attempts' three receipts in the external cache. Run with the published 40-character final transport commit:

```bash
python firstletters-colab/transversal_lab/parker_transport_final.py \
  --execute \
  --cache-dir /tmp/vesuvius-phantom-preflight \
  --freeze-reference PUBLISHED_FINAL_TRANSPORT_COMMIT
```

The current cache needs 16 members. A fresh cache may need all 151 plus the geometry/label packet, although 135 member hashes have historical verification. Copy the committed prior receipts into that cache first. This is reproduction of the same case, not an independent letter or scroll. The default command prints the plan without network; `--self-test` uses generated arrays only.
