# Reproduce the Villa flat ROI measurements

This benchmark checks the ROI patch against two public papyrus surfaces. It compares the ordered model tiles, times the occupancy and tile-selection steps in fresh processes, and runs the full Villa inference command on both checkouts. The last step compares every decoded TIFF pixel and the geometry recorded in the ROI sidecar.

## Results

Each median below uses five fresh CPU processes per version and a warm local filesystem cache. Each output covers 128 × 128 pixels. MB means 1,000,000 bytes.

| Public surface | Occupancy level | Logical CT selected, before → after | Chunk positions | Native planning time | Peak process memory | Changed output pixels |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PHerc. 1447 W058 | 0 | 775.45 → 16.78 MB | 2,304 → 49 | 3.256 → 0.0526 s | 1,967 → 412 MB | 0 / 16,384 |
| PHerc. 841 W00 | 3 | 258,048 → 129,472 bytes | 1 → 1 | 1.252 → 1.112 ms | 371 → 371 MB | 0 / 16,384 |

W058 keeps four parent-grid tiles in the same order. The first series on this same fixture measured 2.328 → 0.0354 seconds. W00 keeps one tile and both reads fall within one existing coarse chunk. Its before/after timings overlap. The W00 example in the project’s main README uses only level 0. The fixture here also retains the published coarse level 3, which Villa prefers.

The timing includes mask construction and tile selection. Model inference was run separately to verify the TIFFs. The logical byte count measures the selection in the native data type. The chunk count describes positions intersected on the chunk grid. Memory is peak RSS for the entire process, including libraries. This experiment measures the cost of investigating known regions and output equivalence. The five individual times, memory readings, environment and output checks are in [results.json](results.json).

## Reproduce

Start in an activated Villa flat-ink-inference environment. The fixture builder also uses NumPy, Zarr-Python 3, numcodecs, tifffile and imagecodecs. Native mode needs Villa's compiled `vc_reader`, PyTorch and the model dependencies from the pinned source. The CPU measurements used Python 3.12 running Villa from its source path. The current monorepo installer has its own Python-version and system-library requirements. [Villa developer setup](https://github.com/ScrollPrize/villa/blob/c8f3ba45304f0a317a49a155929e30d294f201eb/CONTRIBUTING.md#developer-installation) · [Native reader setup](https://github.com/ScrollPrize/villa/blob/c8f3ba45304f0a317a49a155929e30d294f201eb/vc_reader/examples/README.md#run)

Prepare two checkouts once from an empty directory using Bash.

```bash
git clone https://github.com/JorisViaudQuantAI/Scrollprize.git scrollprize-roi
git clone --filter=blob:none https://github.com/ScrollPrize/villa.git villa
base=c8f3ba45304f0a317a49a155929e30d294f201eb
git -C villa checkout --detach "$base"
git -C villa worktree add --detach "$PWD/villa-baseline" "$base"
git -C villa apply --check "$PWD/scrollprize-roi/flat-roi.patch"
git -C villa apply "$PWD/scrollprize-roi/flat-roi.patch"
```

Download the smaller fixture, verify all input checksums and make the native comparison with one command.

```bash
python scrollprize-roi/evidence/reproduce.py w00 --villa villa --baseline villa-baseline --work run-w00 --cache scroll-data-cache --native
```

Repeat with `w058` and `--work run-w058` for the single-level test. Each run writes a local Zarr plus `*-selection.json`, `*-native-timing.json` and `*-native-tiff.json`. Omitting `--native` runs the tile-schedule comparison without the checkpoint or the compiled reader. Set `--runs 1` for a quick end-to-end check. The fixture and verified downloads can be reused between runs. W058 needs about 2 GB of free storage while repacking, plus room for the checkouts. The unpatched process reaches about 2 GB RSS on that case.

The two fixture tests run offline. The three Villa regression modules exercise flat inference, ROI selection and sidecar metadata.

```bash
python -m unittest discover -s scrollprize-roi/evidence -p test_evidence.py
PYTHONPATH="$PWD/villa/vesuvius/src${PYTHONPATH:+:$PYTHONPATH}" python -m pytest \
  villa/vesuvius/tests/ink_detection/test_{roi_inference,roi_output_metadata,flat_inference}.py
```

## Inputs and attribution

- **PHerc. 1447 W058.** Youssef Nader and the Vesuvius Challenge published the 8.64 µm surfaces. The fixture downloads 21 TIFF planes numbered 02–22, 469,684,344 bytes total, from a pinned Hugging Face revision. It preserves all 4,512 × 8,184 pixels per plane and packages them as one Zarr level with 21 × 128 × 128 chunks and Zstandard compression with bitshuffle. The ROI is rows 3,072–3,199 and columns 1,920–2,047. The dataset card lists CC BY-NC 4.0. [W058 source](https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces/tree/7e4d918712a8d8642a6fdeee45b80217c1e98317/w058) · [License](https://creativecommons.org/licenses/by-nc/4.0/)
- **PHerc. 841 W00.** The Vesuvius Challenge published surface `20260220213127-w00` from the 9.366 µm scan. The fixture downloads 42 hash-verified objects, 18,350,486 bytes total, for original level-0 rows and columns 2,688–3,455 and the corresponding published level-3 rows and columns 336–431. Within the local 768 × 768 crop, the ROI covers rows and columns 32–159. The catalog lists CC BY-NC 4.0. [PHerc. 841 catalog](https://scrollprize.org/data_browser/PHerc0841) · [Data licenses](https://scrollprize.org/data#licenses)
- **Reader v3.** Native mode downloads Domenico Russo’s MIT-licensed checkpoint, 138,379,659 bytes. It uses a 384-pixel tile with 50% overlap and Hann blending. W058 uses reverse orientation, W00 forward. [Pinned model](https://huggingface.co/domenicor046/reader-v3/tree/aee1d64bced4eb35546348f3d276b9e2a0b99c46)

The repository distributes scripts under its [MIT license](../LICENSE). The [source manifest](sources.json) records the upstream URLs and SHA-256 hashes. The runner downloads scan regions and weights directly from their original publishers.
