# R50 inference on local surface volumes

The input is a NumPy `.npy` array of unsigned 8-bit CT values in depth, row, column order. Supply exactly 62 registered depth layers and at least 256 real pixels on both spatial axes. The published PHerc. 841 control uses 2.403 µm per voxel. A 256-pixel tile spans 0.615 mm at that spacing. [Model card](https://huggingface.co/scrollprize/PHerc.1667-iteration-0)

The CPU environment used for verification has Python 3.12, PyTorch 2.8.0, Transformers 4.57.6, NumPy 2.2.6, Safetensors 0.6.2 and Requests 2.x. The following commands populate and verify the public weight cache once, then score a local stack offline.

```sh
python public-controls/r50_stack.py weights --cache ./r50-cache
python public-controls/r50_stack.py predict stack.npy scores.npz --cache ./r50-cache --order asc --stride 128
```

The `desc` order reverses the input depth axis. The `permutation` order applies a fixed 62-element permutation. To use a specific depth control, supply its JSON list of input indices, with each integer from 0 through 61 appearing once. Output layer `k` reads input layer `indices[k]`.

```sh
python public-controls/r50_stack.py predict stack.npy reordered.npz --cache ./r50-cache --order permutation --permutation-file order.json
```

Each model window contains 256 × 256 observed pixels. The window at the far edge is anchored to the last valid starting pixel. The default `--blend uniform` takes the same arithmetic mean of overlapping probabilities after bilinear upsampling from 64 × 64. With `--blend hann`, each tile receives a separable edge taper using `np.hanning(258)[1:-1]` along both axes. The program normalizes by the total weight at each pixel. The weights are positive at image boundaries, and tiles with no overlap produce the same probabilities in both modes. The manifest identifies the selected blend. It writes the float32 probability image and JSON provenance together by atomic replacement. The manifest records the input file and array hashes, model revision, weight hash, depth indices, tile starts, coverage and probability-array hash. The terminal also prints the SHA-256 of the complete output file.

```python
import json
import numpy as np

with np.load("scores.npz", allow_pickle=False) as result:
    probability = result["probability"]
    metadata = json.loads(result["manifest"].item())
```

For an explicit Hann run:

```sh
python public-controls/r50_stack.py predict stack.npy scores-hann.npz --cache ./r50-cache --order asc --stride 128 --blend hann
```

The public PHerc. 841 evaluation compares both blends using identical tile outputs, two grid origins and a common physical field. [Frozen evaluation protocol](pherc0841-hann-protocol.md)

The branch workflow compares a 256 × 256 public tile pixel-for-pixel with the frozen control at commit `8aad6c48becbc3760e96aacd793c8d9164ef7c84`. It checks every output pixel on a 272 × 278 field against four explicitly placed real tiles. The tests also cover all depth orders and interruption of a staged output write. [Public dataset](https://scrollprize.org/data_browser/PHerc0841)
