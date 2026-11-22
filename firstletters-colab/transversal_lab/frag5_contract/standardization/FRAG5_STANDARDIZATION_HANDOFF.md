# Frag5 standardized uint8 provenance: bounded source-only handoff

22 November 2026. This is independent of the frozen raw-uint16 native-pair experiment. No CT chunks, TIFF intensities, PNG/IR pixels, model outputs or fitted conversions were accessed. Eleven new primary-source/document responses totaled **89,658 response-body bytes**, below the **16 MiB** direct-source cap. Their canonical URLs, byte counts and hashes are in `source_receipt.json`. Previously acquired public metadata and Hecate source code were read locally.

## Answer

**I do not know the exact published operation that produced Frag5's `volumes_standardized` uint8 arrays.** This bounded search did not find a dataset-specific conversion command, pinned production script, parameter file or execution receipt. **Division by 257 is not established.** The evidence below defines the uncertainty; it does not prove that no such record exists elsewhere.

**I also did not find an explicit link from the pinned public Hecate release to these Frag5 standardized assets.** Its runtime specifies the model's input contract, not the provenance of this uint8 product.

## What the inspected primary sources establish

| Source | Verified information | What remains unestablished |
| --- | --- | --- |
| Official Frag5 standardized [fine group][fine-root] and [coarse group][coarse-root], plus their previously inspected [fine array][fine-array], [coarse array][coarse-array] and [attributes][fine-attrs] | Both root listings contain levels 0–5, `.zattrs` and `.zgroup`. The level-zero arrays declare uint8 with the expected fine/coarse shapes. Attributes list relative pyramid scales. | No conversion recipe, processing parameters or source-to-output provenance is declared in the inspected metadata. Root-directory absence of a sidecar is not an exhaustive search of all possible documentation. |
| Official [open-data README at `f4d8ec2`][open-data] | Its general processing description includes windowing 16-bit intensities and exporting 8-bit OME-Zarr. | It does not supply Frag5-specific windows, rounding/clipping rules or a production record for these legacy `volumes_standardized` paths. A general pipeline description cannot certify their exact transformation. |
| Official archived [Vesuvius loader at `c12730f`][volume-loader] and [README][vesuvius-readme] | The optional `normalize=True` access path divides returned values by the maximum of the stored dtype. The default is false. | This is an access-time floating-point normalization option. It does not explain how a stored uint16 volume became the published uint8 volume. |
| Hecate [model card][hecate-card] and [runtime][hecate-code], both pinned to `9cb86e5` | Inputs must already be unnormalized uint8 ZYX surface volumes at the declared physical sampling. The 2.4 µm model divides by 200; the 9.6 µm model divides by 255, without input clipping. | Neither file identifies Frag5 or `volumes_standardized`, nor supplies a uint16-to-uint8 converter or a dataset-specific production receipt. |

All table claims are **from the linked primary sources or the recorded metadata inspection**. The conclusions about what those records can establish are **my inference**.

**From the Hecate model card:** The release describes fine-derived inputs, later native-coarse inputs aligned to fine renders, and general use of the Scroll Prize ink dataset/open-data collection. That description does not identify the preprocessing of these two Frag5 arrays. The pinned release tree contains the runtime, model card, checkpoints, requirements, license and illustration assets; no separate dataset-preprocessing script is listed there. [Pinned release tree][hecate-tree]. No checkpoint or illustration was downloaded for this search.

## Consequence for our work

**My inference:** Keep the raw-uint16 protocol unchanged. The existence of a matching-shape uint8 array is insufficient to substitute it into a causal fine-versus-native-coarse comparison or to describe it as Hecate's verified training representation. A reproducible uint8 input path needs both a declared transformation and a traceable association with the exact source and output assets.

The missing contract includes the input source identity, intensity window/mapping, clipping and rounding, treatment of background/fill values, and the parameters and code revision actually used. A generic conversion helper, a dtype-normalization option, the model's division by 255, or visual agreement would not establish that production history.

No conversion was fitted from pixels, and no method was selected from downstream detector scores. This search stops at the present evidence gap rather than expanding indefinitely.

## Reproducibility pointers and scope

| Newly inspected source | SHA-256 |
| --- | --- |
| Pinned Hecate README | `17d7f029ffcff421dbf767c89e0e36a91f141fdfb5a647cc254e4c88cffabe9e` |
| Pinned Hecate file-tree response | `65277ea968e918f3fbaaa1014225026bdf63aa72494b04b70ae38a4b6eef43be` |
| Pinned open-data README | `eed764cf45888768a1ebd68852c6b786b1f37babad87a91d23dcb5e12a5202c5` |
| Pinned archived Vesuvius loader | `dc703d160e549a363f1ad1442feb28062ba1077dc32ca327893d63c5dffeb2d1` |
| Fine standardized root listing | `ba3ed6058140e98d6ae26bb28b27fcdf96244481f3bb821c04cb390b661563f4` |
| Coarse standardized root listing | `dd48b4dc53c55872ec6d5af3dfb7b881397747cb2bff0c733e070609fd11708a` |

The locally reused Hecate runtime SHA-256 is `c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59`, matching the earlier pinned contract audit. Searches also inspected the cached official Villa file tree at `c8f3ba45304f0a317a49a155929e30d294f201eb` and the current official open-data/archived Vesuvius repository trees. This was a targeted documentation/code search, not exhaustive analysis of every repository history or public discussion. No user, maintainer or organiser was contacted.

## Open questions

- **ChatGPT / Codex:** Is there a public processing manifest or command that binds the exact Frag5 raw and standardized assets to a named conversion and revision?
- **Claude Code:** Does an independently documented training/preprocessing manifest identify the same standardized assets for Hecate, beyond the model card's general dataset links?

## Next actions

- **ChatGPT:** Preserve the current raw-uint16 source/anatomy trial and interpret it on its own fixed terms. Reopen this provenance question only when a concrete new primary source is available.
- **Codex / Claude Code:** Do not label `1/257`, per-window scaling or the archived loader's `normalize=True` option as the verified Frag5 standardization recipe.
- **Owner:** No action, contact or submission is requested. This handoff reports an unresolved preprocessing contract, not a detector result or prize claim.

[fine-root]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes_standardized/70keV_3.24um_.zarr/
[coarse-root]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes_standardized/70keV_7.91um_.zarr/
[fine-array]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes_standardized/70keV_3.24um_.zarr/0/.zarray
[coarse-array]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes_standardized/70keV_7.91um_.zarr/0/.zarray
[fine-attrs]: https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes_standardized/70keV_3.24um_.zarr/.zattrs
[open-data]: https://github.com/ScrollPrize/open-data/blob/f4d8ec2b895b10cf8535bd6504375b552ff5484a/README.md
[volume-loader]: https://github.com/ScrollPrize/vesuvius/blob/c12730f17c014bed0237fd00e1bd4a6a1c749ca5/src/vesuvius/volume.py
[vesuvius-readme]: https://github.com/ScrollPrize/vesuvius/blob/c12730f17c014bed0237fd00e1bd4a6a1c749ca5/README.md
[hecate-card]: https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/README.md
[hecate-code]: https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/hecate.py
[hecate-tree]: https://huggingface.co/scrollprize/hecate/tree/9cb86e500e944b11a06a7020403cde5dffb5bcb2
