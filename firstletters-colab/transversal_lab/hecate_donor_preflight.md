# Hecate donor preflight: three physical surfaces, 48 fixed contexts

22 November 2026. Metadata and code review by the donor-preflight agent. No CT arrays or detector outputs were inspected. PHerc. 0814 test labels remain undecoded. Calibration labels were accessed only as a pooled known/unknown support mask, under the root agent's instruction.

## Decision

**My inference.** A bounded Hecate representation probe is executable using three released high-resolution surface volumes, physically resampled to the declared 9.6 µm target lattice. The roles are PHerc. 0009B training, PHerc. 0841 calibration, and PHerc. 0814 locked test. Sixteen fixed contexts per surface and two fixed depth orders require at most 96 forwards. This is a three-parent, one-test-donor mechanism experiment. It is not the originally proposed 12- or 16-independent-parent experiment and is not a native-coarse acquisition test.

The exact public-safe manifest is `hecate_donor_manifest.json`, SHA256 `6cad423f3ac34f02d47b20fe3ca6959b98b7adc32007d49873e124f5090fab23`. The root agent must publish the protocol and manifest before CT staging or feature extraction. The executable deterministic selector is `select_contexts.py`, SHA256 `efc58333ee9809a8efdd518c5630d38f3c98708fa1f7d2521c42cbeb51e24ed3`.

## Sources and exact identities

**From a source.** The Studio release is pinned to dataset revision [`a63c3796ebd13e1c70282433cdeb5c34aaa45c07`](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/tree/a63c3796ebd13e1c70282433cdeb5c34aaa45c07). Primary preparation code is pinned to [`bf66f753176792a966bca17822629fb8821f20c8`](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/tree/bf66f753176792a966bca17822629fb8821f20c8). Source-specific metadata, rather than the dataset card's broad description of native tiles, determines each input's actual acquisition.

**From a source.** The following full-canvas label counts, shapes, crop transforms and acquisition descriptions are in the three linked `label.json` files. The manifest includes all source URLs, source metadata hashes, live Zarr metadata hashes, categorical-image hashes, geometry hashes, and exact context origins.

| Role | Physical released surface | Source nominal XYZ pitch | Source shape Z,Y,X | Label shape Y,X at 9.36 µm | Saved ink / background |
|---|---|---:|---|---|---:|
| Train | PHerc. 0009B `auto_grown_20250919055754487_inp_hr_2um` | 2.401 µm | 65, 25984, 21248 | 6665, 5450 | 1,197,568 / 3,321,378 |
| Calibration | PHerc. 0841 `auto_grown_20260220144552896` | 2.403 µm | 65, 16763, 18712 | 2661, 2136 | 369,155 / 749,921 |
| Test | PHerc. 0814 `814_46527_2um_try2` | 2.403 µm | 65, 10240, 13160 | 2628, 3378 | 288,657 / 650,405 |

Sources: [0009B metadata](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/a63c3796ebd13e1c70282433cdeb5c34aaa45c07/donor-continued-training-20260910/labels/source-9b-approved-r000001/label.json), [0841 metadata](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/a63c3796ebd13e1c70282433cdeb5c34aaa45c07/donor-continued-training-20260910/labels/source-841-grow-a-approved-r000001/label.json), [0814 metadata](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/a63c3796ebd13e1c70282433cdeb5c34aaa45c07/donor-continued-training-20260910/labels/source-814-approved-r000001/label.json).

**From a source.** All three categorical `refined_state.png` hashes and geometry-mask hashes exactly match `sourceSignature` in the [frozen Studio label snapshot](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/a63c3796ebd13e1c70282433cdeb5c34aaa45c07/donor-studio-resunet-scratch-20260926/label-snapshot.json). This binds the small PNG files to the documented snapshot without downloading the much larger label NPY files. The categorical encoding is **0 unknown, 1 background, 2 ink**. Unknown labels are excluded from fitting and scoring.

## Geometry and intensity contract

**From a source.** The publisher's legacy recipe takes source planes 0:64, averages 4×4×4, then linearly resamples to 9.36 µm. Its explicit 16 source-depth centers are symmetric about source index **31.5**. The 65-plane geometric midpoint is 32; silently substituting that phase would change the documented alignment. The label-grid mapping is `crop_origin + (label_index+0.5)*9.36/source_pitch - 0.5`. The 0841 crop is `(x0,y0,x1,y1)=(5888,1792,14208,12160)`; 0814 and 0009B use their full source canvases. Sources: the three metadata files above.

**My inference / prescribed experiment.** Use exact separable voxel-volume averages at 9.6 µm, with depth centers `31.5 + (k-7.5)*9.6/source_pitch` for k=0…15. The complete depth boxes fit inside the 65 released source voxel cells; no reflection, extrapolation or fabricated depth padding is needed. Use the same physical XY origin as the published labels and nearest-neighbor categorical transfer. Round the averaged CT to uint8 once, then divide by 255 exactly once for Hecate. Keep both ascending source order and its reversal fixed; neither reversal nor orientation stability proves ink absence.

**From a source.** Live `.zarray` metadata confirms uint8 and chunks of 65×128×128 for all three sources. 0841 is uncompressed; 0814 and 0009B use Blosc/Zstd with shuffle. The fine surface `.zattrs` axes are ZYX and slice step one. Their multiscale transforms are unitless; physical pitch comes from the publisher's source-specific label metadata rather than treating scale 1 as one micrometre. These arrays are high-resolution 77 keV sources, not the later native coarse acquisitions. Sources: [0841 array metadata](https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/841/auto_grown_20260220144552896/auto_grown_20260220144552896.zarr/0/.zarray), [0814 array metadata](https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/814/814_46527_2um_try2/814_46527_2um_try2.zarr/0/.zarray), [0009B array metadata](https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/0009b/auto_grown_20250919055754487_inp_hr_2um/auto_grown_20250919055754487_inp_hr_2um.zarr/0/.zarray), and the linked source-specific metadata.

**My inference — limit.** Preserve the released uint8 representation without min-max scaling or additional clipping at 200. I do not know the complete upstream reconstruction-to-uint8 intensity-window history from these metadata alone. We can reproduce the released representation; we cannot call it raw attenuation or claim radiometric identity with native coarse CT.

## Selection, support and leakage

**From a source.** The [published sampler](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/donor-continued-training-20260910/prepare.py) selects a 256-pixel parent containing ink and then retains its four eligible 128-pixel children, including potentially empty children. The released 0814 and 0009B coordinate lists are therefore historically label-enriched. This was not inferred by looking at the held-out label raster.

**My inference / prescribed experiment.** Select 16 contexts from each published coordinate list by deterministic farthest-point sampling, beginning nearest the coordinate centroid. Snap their physical centers to the global 9.6 µm grid. For 0841, which has no admitted native-training coordinate list, consider a 64-pixel grid and retain only complete geometry footprints with at least 128 known central pixels; pool ink and background while selecting. Then apply the same farthest-point rule. No CT score, model output or held-out label class is used. The resulting 48 input rectangles and 48 central scored rectangles do not overlap within their respective surfaces.

**My inference / prescribed experiment.** Before inference, reject a whole predefined context if any geometry-mask cell intersecting its complete physical XY input rectangle is invalid. Apply the same decision to both depth orders, report the rejected ID, and do not replace it. Require all needed source chunk objects, matching metadata and complete source bounds; a missing object, decode error or source-bound violation hard-fails the run. Save each acquired chunk's hash and actual bytes. The saved geometry mask is an approved UV-domain mask, not a new proof of native acquisition support or correct sheet identity.

**From a source / my inference.** The Studio [frozen data record](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/a63c3796ebd13e1c70282433cdeb5c34aaa45c07/donor-studio-resunet-scratch-20260926/data.json) excluded released 0841 surfaces because native physical registration to its own fixed 0841 holdout was unverified. This proposed experiment keeps each entire donor in one role and never uses that within-0841 native split. The exclusion does not establish that the released same-canvas label/CT mapping is wrong. Source 0009B's raw-native registration is also explicitly unresolved, so this probe uses its released fine slab and does not transfer its labels to raw native CT.

**My inference — limit.** There are three physical surfaces and one held-out donor. Forty-eight contexts do not create 48 independent specimens. Report conditional metrics on the released annotation support, not atlas prevalence or a universal false-positive rate. Historical label enrichment, model-assisted annotations, selective backgrounds and unknown Hecate backbone exposure remain limitations. The probe can test a readout bottleneck; it cannot establish unseen-scroll generalization or prize qualification.

## Costs and executable handoff

**From a source / computed from metadata.** The frozen origins require 429 unique fine-CT chunks: 144 for 0009B, 144 for 0841 and 141 for 0814. Their total uncompressed payload is **456,867,840 bytes**. All three categorical PNGs plus all three geometry PNGs total **235,037 bytes**. The roughly 522.6 MB Hecate checkpoint, these CT chunks and small metadata fit within a 1 GiB transfer cap, with compressed-source bytes checked at runtime. Cache every chunk across contexts and both depth orders. Whole prepared CT tiles are unnecessary.

**My inference.** This is the largest honest immediately executable test under the current preflight and cap. The prepared native 16-plane tiles span only 15 intervals: 140.43 µm at 9.362, 140.49 µm at 9.366, or 129.60 µm at 8.64, all shorter than the 144 µm interval span of 16 samples at 9.6. A broader native-parent experiment needs a new render from saved physical geometry or wider source slabs; relabeling the 16-plane tile pitch is invalid. The [backup documentation](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/a63c3796ebd13e1c70282433cdeb5c34aaa45c07/backup/README.md) distinguishes these reconstruction routes.

## Reproducing the metadata-only context selection

**From a source / executable record.** `select_hecate_contexts.py` is the unchanged historical selector, SHA256 `efc58333ee9809a8efdd518c5630d38f3c98708fa1f7d2521c42cbeb51e24ed3`. It resolves its four inputs relative to its own directory. Copy it into an external working directory before replaying; do not stage donor labels in the public repository. The replay decodes the calibration support mask only and never downloads CT or the PHerc. 0814 test labels.

The four exact pinned inputs are:

| Local relative path | Bytes | SHA256 | Canonical source |
|---|---:|---|---|
| `coordinates_814.npy` | 3224 | `02d59000572cf4291e937945ca2e2f3e2886f8a03245cfb4eb9812aad786a5cc` | [Published 0814 coordinates](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/a63c3796ebd13e1c70282433cdeb5c34aaa45c07/donor-studio-resunet-scratch-20260926/prepared/source-814-approved-r000001/coordinates.npy) |
| `coordinates_9b.npy` | 12624 | `fdfa5b3635a45070c50ad10d48ca44416f950d098b6486dac5415fe03e8f6755` | [Published 0009B coordinates](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/a63c3796ebd13e1c70282433cdeb5c34aaa45c07/donor-studio-resunet-scratch-20260926/prepared/source-9b-approved-r000001/coordinates.npy) |
| `calibration_support/refined_state.png` | 30940 | `54b254d887625050e63596e8016ade073f085d38555935ddfc8c8298acdfe81b` | [0841 categorical state](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/a63c3796ebd13e1c70282433cdeb5c34aaa45c07/donor-continued-training-20260910/labels/source-841-grow-a-approved-r000001/refined_state.png) |
| `calibration_support/geometry-valid.png` | 11860 | `b0389a95ea9f9b131591ae08520bfbaccf023615d9c90b95b9e91dadb5a75602` | [0841 geometry mask](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/a63c3796ebd13e1c70282433cdeb5c34aaa45c07/donor-continued-training-20260910/labels/source-841-grow-a-approved-r000001/geometry-valid.png) |

Run from the repository root in an environment with NumPy and Pillow:

```bash
HECATE_SELECTION_WORK=/tmp/vesuvius-hecate-selection-replay
mkdir -p "$HECATE_SELECTION_WORK/calibration_support"
cp firstletters-colab/transversal_lab/select_hecate_contexts.py "$HECATE_SELECTION_WORK/select_contexts.py"
python - "$HECATE_SELECTION_WORK" <<'PY'
from pathlib import Path
from urllib.request import urlopen
import hashlib
import sys

work = Path(sys.argv[1])
base = "https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/a63c3796ebd13e1c70282433cdeb5c34aaa45c07/"
assert hashlib.sha256((work / "select_contexts.py").read_bytes()).hexdigest() == "efc58333ee9809a8efdd518c5630d38f3c98708fa1f7d2521c42cbeb51e24ed3"
inputs = [
    ("coordinates_814.npy", "donor-studio-resunet-scratch-20260926/prepared/source-814-approved-r000001/coordinates.npy", 3224, "02d59000572cf4291e937945ca2e2f3e2886f8a03245cfb4eb9812aad786a5cc"),
    ("coordinates_9b.npy", "donor-studio-resunet-scratch-20260926/prepared/source-9b-approved-r000001/coordinates.npy", 12624, "fdfa5b3635a45070c50ad10d48ca44416f950d098b6486dac5415fe03e8f6755"),
    ("calibration_support/refined_state.png", "donor-continued-training-20260910/labels/source-841-grow-a-approved-r000001/refined_state.png", 30940, "54b254d887625050e63596e8016ade073f085d38555935ddfc8c8298acdfe81b"),
    ("calibration_support/geometry-valid.png", "donor-continued-training-20260910/labels/source-841-grow-a-approved-r000001/geometry-valid.png", 11860, "b0389a95ea9f9b131591ae08520bfbaccf023615d9c90b95b9e91dadb5a75602"),
]
for relative, remote, size, expected in inputs:
    with urlopen(base + remote, timeout=45) as response:
        data = response.read(size + 1)
    assert len(data) == size
    assert hashlib.sha256(data).hexdigest() == expected
    destination = work / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
PY
python "$HECATE_SELECTION_WORK/select_contexts.py"
```

**My inference / replay expectation.** The resulting `selected_contexts.json` should reproduce the manifest's 16 origins per donor, zero input/core overlap, and the declared farthest-point sequence. This replay is for provenance, not a mechanism to replace contexts after model outputs exist.

**From a source / limitation.** The dataset metadata and labels above use immutable Hugging Face revision URLs and verified file hashes. The three fine-CT surface sources use public **mutable bucket URLs**. Their `.zarray` metadata hashes are frozen in the manifest, but the manifest does **not** contain previously verified CT chunk hashes. The runner verifies metadata before access and records every chunk's SHA256 and byte count at acquisition. Preserve that receipt for later reproduction; a later server change cannot be ruled out merely by comparing the manifest hash.

## Open questions

- **Hecate agent / Claude Code:** Can the three readouts be evaluated on enough explicitly labeled pixels after the fixed support rejection, without changing contexts or depth order after seeing outputs?
- **Codex:** Can a later native-coarse replication obtain wider slabs and independent sheet-support evidence without rebuilding the large geometry archive?
- **Root / owner:** Hecate pretraining exposure and label-teacher dependence limit claims; which external validation would be required before promoting any positive result?

## Next actions

- **Root:** Review and publicly freeze the manifest with the Hecate runner and complete statistical protocol. Do not present this metadata preflight as an ink result.
- **Hecate agent:** Stage only the fixed source chunks after that freeze, verify all hashes, and leave 0814 label decoding until readouts are frozen. Use exact geometry-mask rejection and no fallback sampling.
- **Claude Code:** Use the L4 only if CPU extraction exceeds the bounded engineering budget; the same frozen sources, coordinates and scoring rules must apply.
- **Owner:** Keep any organiser submission decision separate; nothing has been submitted or announced by this work.
