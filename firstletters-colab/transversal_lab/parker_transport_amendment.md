# Parker audit: transport-only amendment before the first scientific result

## What happened

The four original scientific audit files were published at commit `bcd67e3005132b74e295964cea99f4a23d63608c`. Attempt 001 verified 53 of 151 source members, then stopped during acquisition after HTTP403 responses. Its frozen retry allowance was exhausted. Pending downloads were cancelled. No CT pixel arrays, raw-profile output, descriptor scores, or scientific result file were produced.

The preserved local records are `parker_attempt001_failure.json`, `parker_attempt001_download_receipts.json`, and `parker_profile_start_001.json`; their hashes are in the amendment JSON. The original scientific files are unchanged.

## Evidence and inference

**Observed:** The ordinary public download redirect advertises a 60-second lifetime. Two authorized metadata-only tests requested just bytes0–15 of the ZIP through the original published URL, https://osf.io/download/pvmu3/ . They returned HTTP206 with the original archive size, matching frozen ETag and ZIP header. The tests were separated by more than 60 seconds. No CT member was downloaded in those diagnostics.

**My inference:** A reused ordinary signed redirect expired during the longer download batch. This fits the late simultaneous 403 responses. The failed run did not save signed URLs or response bodies, so its exact server-side rejection reason was not established.

## Amendment

`parker_transport_resume.py` imports the exact original runner by SHA256. It routes every bounded request through the same original public canonical OSF URL and follows the service's ordinary redirects. It does not change signatures, extend token lifetimes, use credentials, bypass access controls, substitute a source object, or alter the archive ETag.

The wrapper retains all 151 declared members, every byte range/CRC/size, the PPM and label hashes, source/header validation, every sample/depth/descriptor/label shift, signed AUC orientation, one-case interpretation, numerical thread limit and absolute 2 CPU-hour cap. The only retry change is stricter: each still-missing member receives at most one new attempt, with no new retry. Members that previously exhausted two attempts can therefore receive at most one additional attempt under this amendment.

All cached members are checked against their previous SHA256 as well as the unchanged ZIP CRC/header checks. Attempt 001 failure/start/download receipts are preserved and verified. Attempt 002 produces its own start receipt and records both the scientific freeze commit and transport-amendment commit. Existing scientific result or raw-profile outputs still cause an abort.

The original 300,000,000-byte source cap and 350,000,000-byte total archive cap apply cumulatively. The resumed source reservation begins with 44,711,621 bytes allocated in attempt001 plus 446,773 bytes from the earlier source-header preflight. A conservative 13,000,000-byte metadata reservation covers the preflight geometry packet, ZIP metadata and tiny diagnostic requests. These are upper-bound budget reservations, not claims that all reserved bytes were received. Successful attempt001+attempt002 HTTP range bodies are counted separately; preflight bytes are reserved in the cumulative caps.

## Gate and command

Publish this wrapper and amendment before resuming source acquisition. Then run, using the published 40-character amendment commit:

```bash
python firstletters-colab/transversal_lab/parker_transport_resume.py \
  --execute \
  --cache-dir /tmp/vesuvius-phantom-preflight \
  --freeze-reference PUBLISHED_AMENDMENT_COMMIT
```

`--self-test` uses generated arrays only. The default invocation validates hashes and prints the plan without network or CT access. The amendment does not add a scientific hypothesis or change an outcome-dependent setting; no scientific outcome existed when it was written.
