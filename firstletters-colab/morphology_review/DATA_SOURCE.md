# Public optical profilometry data: available, with unresolved scale and validity issues

Date: 2026-11-23. This is a source-availability review, not an experimental result.

**From a source:** the official [ScrollPrize profilometer dataset](https://huggingface.co/datasets/scrollprize/profilometer/tree/a806bead2f3b9100c20e19de37814fb285cfeefd)
is a public release accompanying Angelotti, Nicolardi, Henderson and Seales,
[“Ink detection from surface topography of the Herculaneum papyri”](https://pmc.ncbi.nlm.nih.gov/articles/PMC13547395/).
Its [pinned dataset card](https://huggingface.co/datasets/scrollprize/profilometer/blob/a806bead2f3b9100c20e19de37814fb285cfeefd/README.md)
describes 14 co-registered regions encompassing 16 letters from PHerc. 248,
PHerc. 250 and PHerc. 500P2. Each sample has a calibrated height TXT export,
normalized height TIFF, brightfield TIFF and ink-label PNG. The stated licence
for data, annotations, folds and configurations is **CC BY-NC 4.0**.

## Availability and bounded acquisition cost

**Measured metadata:** revision
`a806bead2f3b9100c20e19de37814fb285cfeefd` was public and ungated at inspection.
Its [recursive file tree](https://huggingface.co/api/datasets/scrollprize/profilometer/tree/a806bead2f3b9100c20e19de37814fb285cfeefd?recursive=true&limit=1000)
listed 74 files totalling 4,460,520,403 bytes. Summing advertised file sizes gives:

| Proposed input group | Files | Advertised bytes |
| --- | ---: | ---: |
| All normalized height TIFFs | 14 | 642,189,298 |
| All ink-label PNGs | 14 | 1,464,771 |
| **Normalized heights plus labels** | **28** | **643,654,069** |
| All calibrated height TXT exports, separately | 14 | 2,853,542,350 |

**Measured metadata:** HEAD requests for all four files of the already-published
PHerc. 250 κ sample returned HTTP 200 and content lengths matching the tree.
The [metadata preflight](metadata_preflight.json) records these lengths and the
repository's advertised LFS SHA-256 values. **No image, label or height-data body
was downloaded.** The advertised data hashes have therefore not been independently
verified against file bodies. Only repository metadata and text configuration
files were read.

**From the inspected tree and metadata:** no separate raw-validity mask,
standalone training-code file or model checkpoint was present. Machine-readable
descriptors are available: [manifest.csv](https://huggingface.co/datasets/scrollprize/profilometer/resolve/a806bead2f3b9100c20e19de37814fb285cfeefd/manifest.csv),
[sampling.json](https://huggingface.co/datasets/scrollprize/profilometer/resolve/a806bead2f3b9100c20e19de37814fb285cfeefd/configs/sampling.json),
[experiment.json](https://huggingface.co/datasets/scrollprize/profilometer/resolve/a806bead2f3b9100c20e19de37814fb285cfeefd/configs/experiment.json)
and [official splits](https://huggingface.co/datasets/scrollprize/profilometer/resolve/a806bead2f3b9100c20e19de37814fb285cfeefd/folds/splits_final.json).
The inspected manifest, sampling and experiment descriptors provide no per-sample
raw-validity mask or measured missing-data fraction. This is a statement about
the inspected revision, not a claim that no additional release exists anywhere.

## Lateral sampling remains unresolved

**From a source:** the [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13547395/)
reports 0.34 µm lateral sampling. The pinned release's card, manifest and sampling
configuration instead state **0.68793625 µm/pixel in both axes**. The card's
**3 September 2026 notice** explicitly acknowledges this inconsistency and says
it is being investigated. No resolution of that discrepancy appears in the
inspected card. The release value must not be presented as a confirmed correction
to the paper.

**My inference:** an initial feature experiment could report distances and kernel
sizes in pixels. It could not establish a physical resolution requirement or
match a CT voxel pitch while this discrepancy remains unresolved.

## Why normalized heights alone leave a specific confound

**From the [provided preprocessing configuration](https://huggingface.co/datasets/scrollprize/profilometer/resolve/a806bead2f3b9100c20e19de37814fb285cfeefd/configs/experiment.json):**
nonfinite heights are inpainted with **OpenCV Telea, radius 3 pixels**. This uses
a temporary uint8 image scaled between the 0.5th and 99.5th percentiles, then maps
back to the height range and restores the original finite measurements. The
released height output is normalized per sample to [0, 1] and converted to
uint16 [0, 65535]. Raw import also documents removing empty/trailing nonfinite
rows or columns and matching the label shape at the top-left, with bottom/right
cropping or nonfinite padding when necessary. The [card](https://huggingface.co/datasets/scrollprize/profilometer/blob/a806bead2f3b9100c20e19de37814fb285cfeefd/README.md)
identifies `nan` in the calibrated TXT exports as missing reconstruction.

**My inference:** an inverse-local-roughness feature could respond to smooth
inpainted holes. If missing reconstruction is associated with the ink labels,
that could produce a positive association without establishing a physical
ink-roughness mechanism. Neither the actual missingness fraction nor its label
association was measured in this review. A validity mask cannot safely be
recovered by declaring dark, flat or zero-valued TIFF regions invalid: that
would make validity depend on the proposed feature itself.

**Decision:** defer the 28-file normalized-only experiment. Its approximately
644 MB payload would answer an association question on normalized, inpainted
inputs while leaving this specific mechanism unresolved. This is not evidence
that the dataset or the paper's result is invalid.

## Three-stage future route

**My proposed method, not an executed experiment:**

1. **Derive validity from the actual calibrated TXT files.** Preserve the
   original finite/nonfinite map before inpainting, normalization or feature
   computation; account explicitly for the documented shape matching. Freeze
   the feature and support rule before inspecting feature/label association.
   Compute a local feature only where its entire required neighbourhood has
   original finite measurements, and report both eligible and excluded counts
   against the full label population. Report missingness by label as a separate
   diagnostic; do not silently replace missing regions with smooth values.
2. **Retain pixel units until the pitch discrepancy is resolved.** Preserve the
   source's physical metadata as reported, without choosing whichever pitch
   makes a result favourable. A normalized input test must remain explicitly
   an association on that representation; it is not calibrated physical
   roughness or validation of inpainting.
3. **Fit and evaluate on separate complete samples or papyri using the official
   splits.** The release provides five sample-level cross-validation splits
   and three leave-one-papyrus-out splits. Freeze any scale, sign, fitting and
   threshold choices using training samples only; report held-out complete
   samples and papyri, rather than treating adjacent pixels as independent
   replications. An optical result would make **no CT-transfer claim** and would
   not change any existing ink-detection or unread-region promotion gate.

