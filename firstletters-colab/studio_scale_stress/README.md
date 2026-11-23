# Q10: Studio anisotropic sampling stress on an annotated public control

This lane measures the released Studio model's response to 0.5× and 2× image scale along either axis. It uses the author's hash-pinned PHerc0841 prepared holdout, existing release adapter, and exactly the same physical evaluation pixels in every condition. It does not change or compare flatteners, run the shared atlas renderer, inspect an unread surface, or repeat the orientation/radiometry tests.

## Prospective experiment

Five input conditions are identity, x×0.5, x×2, y×0.5, and y×2. A scale below one compresses the physical pattern on the 128×128 model grid and increases its physical field of view. A scale above one enlarges it and reduces the field of view. Source context is reconstructed from the actual prepared 512×512 tiles, with an explicit availability mask. The compressed axis receives a Gaussian anti-alias filter (sigma0.5 native pixel, truncate4), followed by bilinear interpolation. Depth order and samples are unchanged.

The fixed native central32 labels are never resampled. Each probability map is interpolated back onto those physical pixels. Unknown labels remain excluded. The selected source cores do not overlap, but their input contexts do and all are in one compact annotated region. No independent53-sample or unseen-scroll accuracy claim is appropriate.

The main measurement is pooled average precision (AP) change relative to identity. A loss greater than0.05 is a prospectively specified engineering harm flag. ROC AUC, recall/background false-positive rate at threshold0.5, and all mixed-core AP changes are also reported. Identity must reproduce the prior author's AP0.7536 within0.02 before stress inference proceeds, provided all53 original cores survive the common-support check.

A second prospective hypothesis is that an unweighted mean of all five aligned probabilities can help. Its advancement gate requires AP gain≥0.02, median mixed-core AP change≥0, background false-positive-rate increase≤0.02, and no more than10% of mixed cores losing more than0.10 AP. Passing would justify another independent-control experiment, not deployment. A baseline-probability round trip through each score grid separately measures output-interpolation effects without a new neural forward; it is not a correction to the neural result.

`evidence/protocol.json` is generated before any checkpoint is loaded, with all53 control identities, all support checks, data/code hashes, prepared-array hashes, per-input hashes, and gates. Both protocol and executable code are committed before neural inference. Large public inputs and outputs stay outside git; summary receipts are committed afterward.

## Reproduce

Required packages are Python3, numpy, scipy, torch, and safetensors. The receipt records exact executed versions. A CPU implementation is provided; the frozen run uses two worker processes, four Torch threads each, batch1 FP32. Approximately1.02GB of public source/checkpoint downloads and about300MB of prepared arrays are required; inference cost must be measured on the actual machine.

```bash
git clone https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt.git studio-upstream
git -C studio-upstream checkout bf66f753176792a966bca17822629fb8821f20c8
python -m unittest discover -s firstletters-colab/studio_scale_stress -p 'test_*.py'
python firstletters-colab/studio_scale_stress/run.py prepare \
  --source-repo studio-upstream \
  --checkpoint cache/studio-softloss-20261023-ema.safetensors \
  --cache cache/author841 --prepared run-prepared --output run-results \
  --workers 2 --threads 4
python firstletters-colab/studio_scale_stress/run.py infer \
  --source-repo studio-upstream \
  --checkpoint cache/studio-softloss-20261023-ema.safetensors \
  --cache cache/author841 --prepared run-prepared --output run-results
```

`prepare` downloads missing files and verifies byte counts and SHA256. It fails on reused output directories or unsupported source context; it never pads missing source. `infer` verifies the code, inputs, checkpoint and upstream model before execution. The source mask certifies presence in the prepared-tile manifest, not an independent audit of the original raw scan inside each author-prepared tile. Results preserve each condition separately; no label-based best-scale selection is performed.

## Relation to the shared work

Base: shared Claude commit `cff6d9a03ca0a50f685c9a9e9e83cd6b52ad7f2b`. At selection time D4 still listed a changed-input-scale test as not done, and Q10(b) explicitly proposed0.5×/2× anisotropic re-inking. Adaptive037–039 own atlas geometry/whole-footprint integration and its next annotated-model comparison. Gemini's current lane owns flips, rotations and radiometric interventions. This experiment supplies a detector sensitivity measurement those geometric checks cannot provide.

The author's checkpoint was trained with independent XY scales around0.9–1.1 and other spatial augmentations. The stress levels here are deliberately outside that scale range. Scale changes interpolation, information content and the physical receptive field together; results cannot isolate architectural scale equivariance. Image x/y are not asserted to be text-line coordinates.

## Primary sources

- [Existing pinned public-control adapter](../studio_control/run_public_control.py) and [all53-core replay](../studio_control/run_author_holdout.py).
- [Released source at the fixed revision](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/tree/bf66f753176792a966bca17822629fb8821f20c8).
- [Public prepared dataset at the fixed revision](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/tree/ad919d288ede1c4daf92725288bb6fe29c558591).
- [Zhang, ICML2019: Making Convolutional Networks Shift-Invariant Again](https://proceedings.mlr.press/v97/zhang19a.html). Motivates attention to anti-aliasing; no claim that this paper validates our model or restores information lost by resampling.
- [Wang et al., Neurocomputing2019: Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation](https://pmc.ncbi.nlm.nih.gov/articles/PMC6783308/). Motivates mapping transformed predictions into common image coordinates before combining them. Our five deterministic stresses are not a calibrated uncertainty distribution.
