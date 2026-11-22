# Prospective Parker 2019 published-letter depth audit

Status: prospective; source-pixel execution is gated on publication of this protocol. Geometry and supplied-label coverage inspected; no CT pixel array, feature, model score, or image-based choice inspected. One TIFF header inspected. This is one published lunate sigma, not independent glyph or scroll transfer.

## Provenance and scope

- Public dataset DOI: https://doi.org/10.17605/OSF.IO/ZDKN4 ; archive https://osf.io/download/pvmu3/ . Data license CC BY 4.0 from the OSF node license relationship. Attribution: Parker et al. (2019), *From invisibility to readability: Recovering the ink of Herculaneum*, https://doi.org/10.1371/journal.pone.0215775 .
- Approximate 1.3 micrometre voxel pitch and 53 keV come from S1 Appendix: https://doi.org/10.1371/journal.pone.0215775.s001 . Treat physical distances as approximate.
- Published PPM, binary ink label, mask, and reference PNG are already acquired with member CRC checks and SHA256 receipts. Do not use the RGB reference to select sample locations or repair PPM.
- The paper describes photograph-to-texture registration using non-ink features and manual binarization. Shared dimensions do not independently verify that registration. We retain the supplied alignment and test fixed perturbations.
- Publish this file and its JSON protocol before source-pixel inspection. No copied GPL source. Independently implement ZIP parsing, geometry checks, and elementary statistics.

## Fixed sample and geometry

Use the complete 1440×2530 surface as one case. Sample surface coordinates x=8+16k and y=8+16j. Use only supplied mask==255, finite PPM coordinates, and complete depth support. Do not clip coordinates.

The verified TIFF header gives source slice dimensions y=1440,x=286, unsigned 16-bit, big-endian. There are 2530 consecutively named slice members. PPM records are little-endian float64 xyz,nxyz, with source y=surface x and source z=surface y. All supplied normals are exactly (-1,0,0); report this as the supplied fixed extraction axis, not a measured local surface normal. PPM source x is an integer-valued height map with some invalid values. Reject any sample without x in [16,269], y in [0,1439], z in [0,2529]. Normal offsets are d=-16,...,16 and sampled source coordinates are (x-d,y,z). No interpolation is needed at these integer points.

The prespecified eligible lattice contains 11,693 points: 4,678 supplied positive labels and 7,015 negatives. Coverage is a preflight count only. Among mask==255 pixels, 1,057/3,016,367 centers are outside the source volume (0.03504%); 21,827 fail full 33-depth support. These failures are excluded by geometry before CT access.

## Acquisition and resource cap

Fetch only slice IDs recorded in the exact source-slice list in parker_profile_protocol.json, 107,664,592 compressed bytes across 151 members. Source download cap 300,000,000 bytes; total archive-member download cap 350,000,000 bytes, including retries. Limit download concurrency to 4; at most one retry per failed member. HTTP206 and exact Content-Range are mandatory. Pin ETag, verify each member CRC32, record local SHA256. Do not claim the entire upstream ZIP SHA256 was locally verified. Abort if source dimensions/dtype differ from the checked header, source changes, required members fail, or cap is exceeded. CT memory/CPU is small; stop after 2 CPU hours. The process enforces this as an OS CPU-time limit, with expected analysis far smaller.

## Frozen calculations

For every eligible point, extract the complete 33-depth uint16 profile. Retain these transparent descriptors, without training or tuning: center intensity I(0); mean of all 33 samples; front-minus-back contrast mean(I(d), d=1,...,8) minus mean(I(d), d=-8,...,-1). Positive depth follows the supplied (-1,0,0) axis. Also retain all 33 single-depth intensities; report all depths in fixed order, never just the best.

For each descriptor and depth, report ink and background distributions, raw signed mean difference, pooled-standard-deviation contrast, and signed AUC with ties worth one half. Include signed AUC of PPM x as a geometry-only nuisance comparator. Do not flip AUC after inspecting its value. Report how many samples/rows have both label classes. Per-row signed AUC macro averages are descriptive sensitivity checks, not independent samples.

Registration controls: evaluate supplied labels shifted along surface x and y by ±16, ±64, ±256 pixels (12 fixed shifts). For each shift compare native and shifted labels on exactly the same geometry-eligible intersection of source and shifted surface support. No zero fill, no wraparound, no selection of the best alignment. Include every shift, coverage, prevalence, and native comparison.

## Interpretation and stopping rule

This is an input-integrity and mechanism audit. It has no classifier pass/fail threshold, no claimed false-positive rate, and no independent-scroll inference. Spatial translations are stress tests, not exchangeable randomizations, so do not report them as p-values. Pixels and sampled rows are not independent specimens. Native associations weaker than geometry/shifted associations challenge a simple intensity or depth-profile explanation; native associations stronger than all controls motivate an independently labeled second donor, not a discovery claim. A negative result applies only to these fixed raw descriptors on this one published letter.

Report all declared outputs, elapsed CPU/wall time, data hashes, invalid fractions, and limitations. Stop without adding filters, selecting regions, changing mask rules, or tuning depth. Any follow-up requires a new prospective protocol and an independent donor where possible.

## Reproduction and generated-array checks

Run `python parker_profile_audit.py --self-test` for small synthetic array checks of AUC orientation/ties, negative-x depth indexing, and shared shift support. No data access occurs. Run without `--execute` to inspect the frozen acquisition plan. After publication and authorization, pass `--execute --cache-dir /an/external/data/cache --freeze-reference PUBLISHED_COMMIT` (the reference must be the 40-character published commit); raw public donor arrays remain outside git. The manifest predeclares every exact ZIP member, byte range, size and CRC. Only previously downloaded members have a preflight SHA256; all new downloads record a SHA256 receipt. No claim is made that the entire archive hash was locally verified.

The runner sets numerical library thread limits to one before importing NumPy. It records Python, NumPy and Pillow versions, declared/observed dtypes, code/protocol hashes, and the supplied published freeze reference in start and result receipts. Existing result or raw-profile outputs cause an abort; prepared metadata/member caches may be reused. It validates the supplied label image is the original binary encoding, without grayscale thresholding. On a fatal retrieval failure it cancels pending downloads. Failure receipts contain sanitized exception types/stages, never exception text or redirected URLs. These are execution safeguards and do not change the frozen descriptors or samples.
