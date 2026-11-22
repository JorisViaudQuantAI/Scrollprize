# Second photographic CT donor: metadata preflight

Date: 2026-11-22. Scope: public primary documentation, directory listings, JSON metadata, source code, and HTTP HEAD only. No CT chunks, TIFF pixels, PNG labels, photographs, or model outputs were downloaded or inspected. This is a feasibility report, not a frozen experiment or an acquisition authorization.

## Recommendation

**My inference:** Use **PHerc. Paris 1 Fragment 34 (Frag3)** as the first additional physical donor. Its ink labels have the strongest verified photographic provenance. For a genuinely native-coarse follow-up, prioritize **PHerc. 1667 Cornice 1 Fragment 3 (Frag5)**; **PHerc. 51 Cornice 4 Fragment 8 (Frag6)** is a third physical donor. Both Naples fragments have official fine-scan IR/label/surface bundles, actual 7.91 µm raw CT Zarrs, and published fine-to-coarse affine transforms. Coarse label transfer still requires validation; it is not already established by the existence of those affines.

**From a source:** Paris 1 fragments 34 and 39 are from the same roll; Paris 2 fragments 47 and 143 are from another roll. The paper describes manual tracing of photographic ink, with papyrologist supervision, and deformable photographic-to-CT alignment using manually matched features. It explicitly acknowledges alignment and tracing error. Its surface construction samples 65 planes at one native voxel per step around the per-pixel surface and normal. These are photographic 2D ink labels, not measured 3D ink depths. [Original EduceLab-Scrolls paper, v4, §§2.1, A.5–A.9](https://arxiv.org/html/2304.02084v4).

**My inference:** Parker's Paris 2 Fr96 and Frag1/Frag2 must share a Paris 2 parent grouping. Frag3 and Frag4 share a Paris 1 grouping. Neither separate fragments, separate scans nor many patches create independent parent replicates. The official [Frag5 page](https://scrollprize.org/data_browser/PHerc1667Cr1Fr3) identifies its parent as PHerc. 1667, the same parent as Scroll 4; group all such material together.

## Three options

| Option | Verified public assets | Physical geometry and acquisition | Remaining blocker |
|---|---|---|---|
| **A. Paris 1 Fr34 / Frag3** | Official IR photograph, binary ink-label file, tissue mask, PPM, 65 surface TIFFs. Paper directly documents this fragment's photographic labels. [Official page](https://scrollprize.org/data_browser/PHercParis1Fr34); [surface bundle](https://dl.ash2txt.org/fragments/Frag3/PHercParis1Fr34.volpkg/working/54keV_exposed_surface/). | Surface metadata: ZYX **65×7606×5249**, nominal **3.24 µm**, **54 keV**, native range 0–65535. Raw level-zero Zarr: ZYX **6656×1440×6312**, uint16, 128³ chunks. | Compact TIFF range reader or PPM-crop/raw-Zarr renderer must be verified. Radiometric conversion to Hecate's uint8 must be specified before scoring. This is fine-derived 9.6 µm input, not native coarse. |
| **B. PHerc. 1667 Cr1 Fr3 / Frag5** | Official fine-scan bundle contains `ir.png`, `inklabels.png`, `inklabels.psd`, `mask.png`, PPM and TIFFs 00–64. Official raw coarse scan and affine + five paired landmarks are present. [Official page](https://scrollprize.org/data_browser/PHerc1667Cr1Fr3); [fine surface bundle](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/). | Actual coarse scan: **70 keV, 7.91 µm**, ZYX **3131×668×2046**, uint16, 128³ chunks. Fine scan: **70 keV, 3.24 µm**, ZYX **7010×1400×4420**. | Fine surface `meta.json` returns 404; exact surface XY dimensions/PNG alignment and depth phase need header/provenance verification. Original annotator and tracing history for this Naples label are not established by the Paris-fragment paper. Coarse transfer requires independent geometric checks. |
| **C. PHerc. 51 Cr4 Fr8 / Frag6** | Same official fine bundle structure, including IR, PNG/PSD labels, mask, PPM, TIFFs 00–64. Coarse affine + five paired landmarks exist. [Official page](https://scrollprize.org/data_browser/PHerc51Cr4Fr8); [fine surface bundle](https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing/). | Actual coarse scan: **53 keV, 7.91 µm**, ZYX **3683×1068×2724**, uint16, 128³ chunks. Fine source: **53 keV, 3.24 µm**, ZYX **8855×2260×6300**. | Same label-lineage, TIFF-header, phase and coarse-transfer checks as B. Published landmark fit is less tight than B; that alone does not establish which surface transfer is more accurate. |

All asset and numeric entries in this table are **from a source**, except the prioritization and blocker assessments, which are **my inference**. Native-volume metadata: [A](https://dl.ash2txt.org/fragments/Frag3/PHercParis1Fr34.volpkg/volumes/20230215142309/meta.json), [B fine](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes/20231121133215/meta.json), [B coarse](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes/20231130111236/meta.json), [C fine](https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/volumes/20231121152933/meta.json), [C coarse](https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/volumes/20231130112027/meta.json). [A surface metadata](https://dl.ash2txt.org/fragments/Frag3/PHercParis1Fr34.volpkg/working/54keV_exposed_surface/surface_volume/meta.json).

**From a source:** An existing independent author's [pinned downloader](https://github.com/kartoun/vesuvius-fragment-ink-depth/blob/fe8628624f21879b21659f8d29f979fa38f303f1/download_fragments.py) uses exactly these official fine-surface bundles, and their [published benchmark](https://github.com/kartoun/vesuvius-fragment-ink-depth/tree/fe8628624f21879b21659f8d29f979fa38f303f1) describes all six fragments as IR-grounded. That supports the Naples interpretation but does not replace original label-production records. **I do not know** the exact Naples tracing/approval history or the existing Hecate backbone's exposure to any of these fragments.

## Native-coarse mapping is available, but not certified

**From a source:** Official transforms name their source and target volume UUIDs:

- Frag5: [`20231121133215 → 20231130111236`](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/transforms/20231121133215-to-20231130111236.json), with [five point pairs](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/transforms/20231121133215-to-20231130111236-points.txt).
- Frag6: [`20231121152933 → 20231130112027`](https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/transforms/20231121152933-to-20231130112027.json), with [five point pairs](https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/transforms/20231121152933-to-20231130112027-points.txt).

**My inference from the paired coordinates and volume dimensions:** The transformation contract is source fine-scan XYZ voxel indices to target coarse-scan XYZ voxel indices, not micrometres and not NumPy ZYX order. This interpretation reproduces the published pairs and is consistent with the source/target grid extents; the original transform reader should still be checked before rendering.

**My inference / arithmetic on metadata only:** Applying each published 4×4 affine to its published XYZ pairs gives target-voxel Euclidean RMS residuals **0.2682** (Frag5) and **0.8841** (Frag6), maximum **0.3823** and **1.0612**. These are in-sample residuals from five pairs, not held-out registration accuracy or error bounds. Linear singular values are approximately 0.4134–0.4152 and 0.4127–0.4156. Do not substitute the nominal voxel-size ratio for the published affine.

**My inference:** A defensible coarse-profile route is label-grid pixel → fine-scan PPM point → published affine → coarse raw CT. Two distinct sampling constructions must be kept explicit. For an affine with linear part A, the transformed **geometric surface normal** is proportional to A^{-T} n, with coordinate units and the physical metric handled correctly and then normalized; use that direction when re-rendering along the transformed surface normal. Preserving correspondence along the **original sampled ray** instead maps its displacement by A: A(p + d n) = Ap + d A n (the translation is added to both mapped points). Thus a previously sampled fine ray maps along A n. Under anisotropic scaling or shear, A n and A^{-T} n generally differ; neither is a universal substitute for the other. The protocol must specify whether it preserves the original ray or re-renders along the target surface normal, and convert its depth parameter into physical units accordingly. Do not blindly reuse source normals. Freeze geometric exclusions and physical depth support before any outcome. Validate the PPM's source-volume identity and axis convention first. Then verify registration using unlabeled geometric landmarks or texture, outside future scored areas where feasible. Do not optimize this transfer against held-out ink-label agreement.

**My inference:** A direct Hecate replication adds two unresolved implementation decisions: preserving the flattened XY metric under this mapping, and converting released uint16 intensities to the released model's uint8 convention. Arbitrary per-test-window min/max scaling would make it a different experiment. A native coarse depth-profile audit can avoid the learned-model radiometry problem while those details are resolved.

## Compact routes and estimated bytes

**From a source:** These exact level-zero raw Zarr endpoints exist:

- A: [`54keV_3.24um_.zarr/0/.zarray`](https://dl.ash2txt.org/fragments/Frag3/PHercParis1Fr34.volpkg/volumes_zarr/54keV_3.24um_.zarr/0/.zarray)
- B: [`70keV_7.91um_.zarr/0/.zarray`](https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes_zarr/70keV_7.91um_.zarr/0/.zarray)
- C: [`53keV_7.91um_.zarr/0/.zarray`](https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/volumes_zarr/53keV_7.91um_.zarr/0/.zarray)

All three declare little-endian uint16, C order, 128³ chunks, `/` chunk separator, Blosc/Zstd level 3. Their multiscale `.zattrs` scales are relative voxel factors, not micrometre declarations; physical pitch comes from the volume metadata above. **My inference / arithmetic:** One decoded chunk is **4,194,304 bytes**. Compressed acquisition size is unknown until chunk HEADs; cap both actual transfer bytes and decoded working set. A budget of 64 decoded chunks is 256 MiB, not a promise that any particular surface selection fits that budget.

**From a source:** HTTP HEAD advertises `Accept-Ranges: bytes` for surface TIFFs and PPMs. It does not prove safe compact TIFF decoding; strip/tile headers and actual 206 responses still need checking. The files are mutable URLs: record ETag/length and acquired-range hashes, not an invented preexisting payload hash.

| Asset | Frag3 | Frag5 | Frag6 |
|---|---:|---:|---:|
| One surface TIFF, `00.tif`, exact HEAD bytes | 79,847,973 | 66,658,265 | 109,865,915 |
| Ink-label PNG + mask PNG, exact HEAD bytes | 104,390 | 114,650 | 178,685 |
| PPM, exact HEAD bytes | 1,916,346,985 | 1,599,793,993 | 2,636,777,593 |
| Estimated 65 full TIFFs, assuming each equals `00.tif` | 5.190 GB | 4.333 GB | 7.141 GB |

The first three rows are **from sources**: the linked surface directories and HEAD receipts. The last row is **my arithmetic estimate**, not a complete download manifest. Do not download whole PPMs or all TIFF planes for a small control. Seek verified byte-range surface crops or byte-range PPM records plus the needed raw Zarr chunks.

**My inference:** At nominal 3.24 µm pitch, 65 planes span enough depth to cover a centered 16×9.6 µm slab (153.6 µm of cell support), unlike the 16-native-plane Studio exports. This establishes possible support, not the correct surface phase or face. No depth sign should be selected using test labels.

## Novelty and scientific limits

**From a source:** The [existing fragment benchmark](https://github.com/kartoun/vesuvius-fragment-ink-depth/tree/fe8628624f21879b21659f8d29f979fa38f303f1) already studies fixed-depth profiles, surface-aligned profiles, and flat versus surface-band targets, and reports leave-one-fragment-out evaluation. **My inference:** Repeating generic profile separability is unlikely to be a distinctive contribution. The sharper opportunity is a reproducible, byte-bounded **native-coarse versus fine-scan photographic-label comparison**, grouped by physical roll, with fixed models/readouts and explicit transfer uncertainty. Its fragment-level split cannot automatically support an unseen-parent claim because Paris 1 and Paris 2 each contribute two fragments.

**My inference:** Label-independent photographic origin improves the ink-identity endpoint. It does not prove accurate ink depth, eliminate manual registration error, establish intact-scroll generalization, or make Hecate's historical training exposure disappear.

## Metadata receipts and hashes

All metadata and HEAD receipts are in `/tmp/vesuvius-second-donor/`; no image arrays are present. Key byte hashes:

| Metadata | SHA-256 |
|---|---|
| Frag3 surface `meta.json` | `4ee01953fb2e683703fbd0c2f976f39f80560a36a4070a05ab5d536ba8543559` |
| Frag3 raw level-zero `.zarray` | `7ae31aafc8799780600cd388189733bf9345e0d8323ba3c055e35e3732b96aaa` |
| Frag5 coarse level-zero `.zarray` | `049d287c2fa55628cf9ea2b5fccbd0c813082750d7a1067059abd30402b3e124` |
| Frag6 coarse level-zero `.zarray` | `ce0573dcc5218a54ba66ae37907c1705d938d7aa43c8b38603aaf6a9b16bd6cd` |
| Frag5 fine→coarse affine | `e4653b4d904c98ac215e28e43cb0d3033ea214eaf7c696264d10a9029b4c706f` |
| Frag6 fine→coarse affine | `75e85c30effe9782a93390bbeb1a990c8c160b38e9fe4090a2120ba28c53c049` |
| Pinned author downloader | `9213f09fc6916ddddccc78428773b657faa84fee6710328701f46a3fab65c666` |

These hashes pin metadata bytes already inspected. They do not authenticate unacquired CT chunks, label pixels, or photographs.

## Open questions

- Exact Naples annotation-production provenance, and whether original Photoshop material records it.
- TIFF compression/strip layout; PNG dimensions and label encoding; PPM header/shape and source-volume identity.
- Independent local error of published five-landmark coarse transforms.
- Hecate's training exposure and exact uint16→uint8 convention for this historical data.

## Next actions

1. **ChatGPT / Codex:** If this follow-up is selected, perform a new metadata/header-only preflight for Frag5 first, with no label pixels or CT arrays. Verify compact acquisition and a conservative actual-byte cap.
2. **Claude Code:** Review the existing fragment benchmark and avoid duplicating its generic depth classifier; propose the exact native-coarse/fine comparison only after geometry and radiometry are settled.
3. **Owner:** Keep prize submission decisions separate from this control experiment. There is no ink discovery or prize claim in this report.
