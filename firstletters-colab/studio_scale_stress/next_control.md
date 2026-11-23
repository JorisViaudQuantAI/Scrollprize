# Next small Studio scale control: PHerc0306B provenance

Read-only source audit, 23 November 2026. No model inference or CT chunks were downloaded for this note. Small public metadata, source code and three human tracing PNGs were retrieved. This is an asset/protocol handoff, not an experimental result.

## Decision

**PHerc0306B is the better next candidate for a documented training-unseen scroll control of the released Studio soft-loss checkpoint. It has a publicly accessible native CT render and human tracings. The tracings are weak positive evidence, not a complete binary ink/background ground truth.** Its published winner image, bounding boxes and organizer ink maps must not be converted into a gold mask.

PHerc0343 is a useful second control but cannot presently be described as independent new-scroll transfer: Studio explicitly trained on five PHerc0343P rows. The official catalog calls 0343 a scroll and 0343P a fragment, with different volume IDs. The sources inspected did not resolve their physical manuscript relationship. A different sample suffix or volume ID does not establish a different original scroll.

| Candidate | Studio exposure established by published training chain | Available annotation | Defensible interpretation |
|---|---|---|---|
| PHerc0306B winner surface | No 0306B or 0306A in the documented input groups/rows; scratch base uses 0814/0841/0009B | Papyrologist tracings, uncertain character readings, model-derived letter dimensions; no exhaustive background mask found | Documented training-unseen scroll; weak human-tracing transfer control after registration |
| PHerc0343 winner surface | PHerc0343P exposure explicit; relationship to PHerc0343 unresolved | Letter boxes/rows and model-derived images; no dense human mask in the inspected winner tree | Exposure status unresolved; do not label as independent new-scroll accuracy |
| PHerc0343P exposed fragment | Explicit row `reader-0343P-20250511003658-tifxyz` in training | Public IR-reference assets are associated with this same segment | Training-exposed control, not new-scroll transfer |
| November organizer maps on 0343/0306B | Model-specific ancestry would need separate checking | Three `ink-detection` model predictions each | Agreement/diagnostic outputs, not human labels |

### Model ancestry sources

The audited Studio source revision is [`bf66f753176792a966bca17822629fb8821f20c8`](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/tree/bf66f753176792a966bca17822629fb8821f20c8). The checkpoint is `studio-softloss-20261023-ema.safetensors`, SHA-256 `96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90`.

- [Root README](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/README.md): random-initialized base on PHerc0814, PHerc0841 and PHerc0009B; later Expanded9 and Reader-label corpora; no downloaded pretrained network in the documented chain.
- [Exact continued-pretraining mixture](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/mixture.json): 47 additional Reader rows include five PHerc0343P rows. Saved body SHA-256 `1a03d65842209f287c49d47ef8451e1423e879f93afd8299058d774f5f04d68f`.
- [Final-stage record](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-softloss-20261023/README.md): same data, replay and mixture as the parent. Loss and continued optimization change.

This establishes the **published input ancestry**, not a forensic proof of everything an author or pseudo-label teacher ever saw. PHerc0306B labels were also produced after inspecting other networks' predictions, which limits label independence even if Studio's own training inputs omit this scroll.

## Exact native PHerc0306B assets

Winner source revision: [`feabff6c54f95bd8e52a90e0a7a76881ee67d4cc`](https://github.com/SirProdigle/pherc0306b-first-letters/tree/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc). Dataset revision: [`f435998b55269762942a979052044309b29650ac`](https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/tree/f435998b55269762942a979052044309b29650ac).

The author's [data manifest](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/docs/MODELS_AND_DATA.md) identifies the original claim CT, and the [method](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/docs/METHOD.md) specifies 41 planes, native 8.64 µm, layer 20 at the surface, one voxel between planes. These assets refer to the published winning claim, not an unread candidate.

```text
https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/f435998b55269762942a979052044309b29650ac/results/claim_surface_render_41layers.zarr/
```

Append `.zattrs`, `.zgroup`, `0/.zarray`, or `0/0/{chunk_y}/{chunk_x}`. Fresh GETs succeeded for the metadata. The level-0 array is `[41,4900,7780]`, uint8, C order, chunks `[41,128,128]`, slash-separated, Blosc LZ4 level 3 with shuffle 1. The `.zattrs` body SHA-256 is `1ac7b58bb87ef23019083960444adbec791b0cd7254b771da245c6dcc50077dd`; `0/.zarray` SHA-256 is `6e9dfd265acd7a74f56d3ec82a82fd7de1715ec473cd966c5d3d141c3cd94d51`.

**Small useful acquisition:** retain a 272×272 native context around each 128×128 Studio input to support the 0.5 scale arm, using the same explicit anti-aliasing and common-pixel evaluation convention as Q10. A 272 window can overlap **four** 128 chunks per axis at an arbitrary origin, so budget up to 16 chunks. A fixed origin whose remainder is at most 112 in each axis can use nine. Do not quote nine as the universal bound.

For a concrete availability check, the HF tree lists rows 16–18, columns 23–25 of this original published claim raster: nine chunks of 671,760 bytes, total **6,045,840 bytes**, with public LFS SHA-256 values. The example is a storage/provenance check, not a chosen evaluation sample. Chunk `0/0/17/24` has SHA-256 `8c0ec318c8ac8fbbaa687016e78892af82dec1d2b2708ab7f753194b8a1d66e7`. A general 16-chunk budget is about 10.75 MB at this observed size. No full 1.6 GB render is needed. Decode with a compatible Blosc/Zarr-v2 reader; verify the expected decoded shape and every retrieved chunk hash.

These public metadata records were inspected directly:

- [Original render metadata](https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/f435998b55269762942a979052044309b29650ac/results/claim_surface_render_41layers.zarr/.zattrs)
- [Original array metadata](https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/f435998b55269762942a979052044309b29650ac/results/claim_surface_render_41layers.zarr/0/.zarray)
- [Example row inventory with chunk checksums](https://huggingface.co/api/datasets/Prodigle/pherc0306b-first-letters/tree/f435998b55269762942a979052044309b29650ac/results/claim_surface_render_41layers.zarr/0/0/17?recursive=false&limit=1000)

## Human tracing semantics and registration

The three exact source files are:

| Raw tracing | Bytes | SHA-256 |
|---|---:|---|
| [16 October conservative](https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/supporting/reading/papyrologist_label_2026-10-16_conservative.png) | 74,483 | `3d7216f7bcef6372ea7fbb81e62a7efa6de1beed336adb5ec5805abd475050bf` |
| [16 October fuller/tentative](https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/supporting/reading/papyrologist_label_2026-10-16_fuller_tentative.png) | 79,533 | `907825940adea79e93cc950d45a4bcf7a8bdb6c2e9387713c9bd5b9ad0ad84d6` |
| [23 October tracing](https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/supporting/reading/papyrologist_tracing_2026-10-23.png) | 68,845 | `0fed7652601dd5a557e860722e69bb912d49f6a7a5311ea6641b114cc69682ef` |

The conservative image was visually inspected: white, fairly thick human-traced letter shapes on black. The October-16 files are 1999×923 RGBA; October-23 is 5278×2437 grayscale. **Black means untraced here, not independently certified blank papyrus.** A character tracing is not an exact stroke-boundary measurement of the buried original.

The pinned [`letter_table_label3.py`](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/code/letter_table_label3.py) supplies this registration for the October-16 conservative tracing:

1. Threshold the maximum RGB channel at 127, then nearest-neighbor resize to the papyrologist pack size **5278×2437**. This is restoration to the documented pack, not evidence that the original strokes were annotated at native CT resolution.
2. Pack `(x,y)` maps to the winner **main image** at `(x+1713,y+1062)`. The author reports this offset was checked against the ink maps with correlation 0.992, best within ±3 pixels. That is an author-reported check, not an independent registration validation performed here.
3. The [version-7 image sidecar](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/PHerc0306B_20250521133212_w074own_touchup_V5Rc.json) records main-image map offset `(220,195)` and map size 6945×4051. The map is a resampled context and then flipped vertically into reading orientation. Use the exact pinned [`claimlib_v2.py::sample_raster`](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/code/claimlib_v2.py) and [`make_claim_image_v2.py`](https://github.com/SirProdigle/pherc0306b-first-letters/blob/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/code/make_claim_image_v2.py) conventions. It is not an integer crop of the CT. The continuous carrier-grid origin is fractional, so a simple integer paste plus reflection loses registration precision.

Critically, `letter_table_label3.py` does more than read a human mask: it widens that tracing by 0.15 mm and **intersects it with the winner model image at half tone or darker**, then derives bounding boxes. Consequently `letter_dimensions.json` and masks regenerated from its measurement procedure incorporate model predictions. They are unsuitable as independent dense truth.

The method also explicitly discloses that human tracings influenced surface checks, selection of earlier model combinations, and interpretation; they are not blind annotations from an independent physical modality. The later letter-grid edges were adjusted using gaps in model ink. Keep raw human masks separate from those derivative products.

## November organizer release: useful inputs, different raster

The official [full metadata catalog](https://vesuvius-challenge-open-data.s3.amazonaws.com/metadata.json) snapshot has Last-Modified **19 November 2026 15:24:05 GMT**; the new surface metadata has Last-Modified **18 November 2026**. The catalog is gzip encoded despite its `.json` name. Its downloaded compressed-body SHA-256 is `585cb00d4d1226ccd9650d18e86c64bd27718f19323f39f81480bd214c22d3d1`, decoded-body SHA-256 `31eef2b3c3623e6c3117ea671e5036e5aec6ec1cbab56c13fdb2a3c08fffd608`.

Exact new surface roots:

```text
https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0306B/segments/20261028110559-w074own_touchup_V5Rc/surface-volumes/8.64um-1.2m-116keV-volume-20250521133212.zarr/

https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0343/segments/20261007213123-concat_w047-w048_R5B2_z9500-11000/surface-volumes/8.64um-1.2m-116keV-volume-20250521140437.zarr/
```

Append `.zattrs` and `0/.zarray`. Both are native 8.64 µm, 31 planes, uint8, uncompressed C-order chunks `[31,128,128]`; each full chunk is 507,904 bytes. A general 272×272 request is at most 16 chunks, **8,126,464 bytes**. Official 0306B shape is `[31,4540,7360]`, versus original winner `[41,4900,7780]`. Official 0343 shape is `[31,2660,6780]`, versus original imported mesh properties 2560×6720. **These are different rasterizations.** Normalized/oriented mesh metadata does not supply a complete pixel correspondence to the original winner frame; directly reusing old letter coordinates or resizing to match the new shape is not justified.

The catalog associates each new segment with three maps, identifiable by model ID:

- `20261117205746-argus-v3-tile384-stride192`
- `20260814010243-ink9um-hybrid3d2d-s43-20k-tile128-stride64`
- `20261019111505-reader-v3-tile384-stride192`

For example, the exact 0306B Argus file is:

```text
https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0306B/segments/20261028110559-w074own_touchup_V5Rc/ink-detection/PHerc0306B-20261028110559-8.64um-1.2m-116keV-volume-20250521133212-20261117205746-argus-v3-tile384-stride192.tif
```

These are `ink-detection` predictions with explicit `parameters.model_id`, not organizer hand annotations. No model-map TIFF was downloaded for this audit. The catalog's 0306B import records winner commit `c29d6868ed0d`, whereas the checked final winner package above is `feabff6c…`; retain that distinction as well as the different canvas dimensions. For 0343 the catalog records winner revision `e8370f1f88a2` and the [winner's letter boxes](https://github.com/Nieuwlaar/pherc343-first-letters/blob/e8370f1f88a2/inputs/concat_w047-w048_R5B2_z9500-11000_letter_boxes.json), which likewise are not dense ink masks.

## Small defensible next experiment

Freeze the **original 0306B render + October-16 conservative tracing + exact version-7 registration**, then independently verify alignment before any scale comparison. Use a deterministic native tile lattice and an annotation-only rule to select trace-intersecting windows; include all qualifying windows under a prespecified small budget, not the ones favored by Studio. Acquire sufficient native context and retain one fixed depth order and normal convention; the 41-plane surface center alone does not certify Studio's expected order. Do not pick a depth order or scale by whichever makes this traced line score best.

Compare the already frozen scale arms on identical physical trace pixels. Report paired trace coverage/response and untraced-area response separately, with any threshold fixed from the prior donor control. Such a result measures agreement with human tracing and response stability on a documented training-unseen scroll. **It does not support dense AP, specificity, or false-positive claims unless an independent background annotation is added.** The winner's “blank R” is itself documented as holding faint marks and having different CT contrast; it is a diagnostic region, not an unquestionable negative gold mask.

If both positive and negative labels are required, first obtain a separate expert-reviewed sparse annotation set, preserve uncertain pixels as unknown, freeze it before seeing scale outputs, and retain the existing traces as a secondary sensitivity analysis. Do not bootstrap pixels as independent specimens; this asset contains one selected textual region on one scroll.

## Local receipts accompanying this note

The JSON receipts in `evidence/next_control/` preserve retrieved URLs, exact byte counts/hashes, response headers where available, and the extracted official catalog records: `next_control_source_receipt.json`, `next_control_winner_receipt.json`, `next_control_render_receipt.json`, `next_control_human_receipt.json`, `next_control_mapping_receipt.json`, and `next_control_chunk_tree_receipt.json`. Public URLs above make the note usable without the local cache. The initially fetched top-level recursive HF tree stopped at 1000 entries and is incomplete; absence from that listing was not used as evidence. The targeted original-render row inventories provide the chunk checksums used here.
