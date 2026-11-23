# Research rationale and limits

## Why this experiment fills a gap

Q10 records that PHerc0175A renders had substantial anisotropic metric error. Earlier mesh work measured that error and tested geometry correction, while D4 still listed changed-scale Studio inference as not done. A direct detector experiment can establish whether geometric differences of that magnitude matter for known labelled ink. It cannot identify ink in the prior screened area or retrospectively determine the correct metric there.

The source/control recipe is reused from `studio_control` without new fitting. The model architecture, checkpoint, labels, depth order, normalization and core population remain pinned. Only the XY sampling transformation and its physical context change. All five conditions retain a fixed32×32 core in native physical coordinates. There are53 disjoint cores but they belong to one compact annotation area, with overlapping128-pixel input contexts.

## Cross-disciplinary methods used

**Sampling theory and CNN robustness.** [Azulay and Weiss, JMLR2019](https://jmlr.org/papers/v20/19-519.html) experimentally show that modest rescalings/translations can change CNN predictions, and that training augmentation does not guarantee invariance. [Zhang, ICML2019](https://proceedings.mlr.press/v97/zhang19a.html) examines anti-aliasing in CNN downsampling. These papers motivate measuring the behavior and controlling interpolation. They do not validate a papyrus detector or prove that a Gaussian input filter makes it scale invariant. The fixed Gaussian sigma0.5/truncate4 plus bilinear recipe is an explicit engineering sampler, not an ideal band-limited reconstruction.

**Test-time augmentation from medical image segmentation.** [Wang et al., Neurocomputing2019](https://arxiv.org/abs/1807.07356) studies transformed predictions and their combination in common image coordinates. This motivates the frozen uniform mean of five inverse-mapped probability fields. The four extreme scale stresses here are not draws from a calibrated acquisition prior, and their dispersion must not be called calibrated uncertainty. No best condition is chosen per pixel or after looking at labels.

**Paired interventions.** The experiment changes one image axis at a time, preserves the original categorical labels, excludes unknown pixels in every arm and reports every arm against a fresh identity run. A separate identity-probability round trip isolates the numerical score-grid remapping effect. It is not subtracted from model losses because input interpolation, information loss, receptive field and nonlinear network response interact.

## What the pinned release actually trained on

The [final soft-loss README](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-softloss-20261023/README.md) specifies that augmentation/model/window remain inherited from its parent and reports PHerc0841 AP0.7536. The [training configuration](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/donor-studio-resunet-scratch-20260926/config.json), [loader](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/donor-studio-resunet-scratch-20260926/sparse_data.py) and [augmentation implementation](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/pherc1447-architecture-study-20260906/scroll_augment.py) specify independent XY scale draws around0.9–1.1, alongside rotations, flips, shear and appearance changes. Thus0.5×/2× probes extrapolation outside that explicit scale range.

The upstream loader warps raw CT before clipping. We likewise preserve float interpolation results and then clip to0–200 and divide by255 once. The released [architecture](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/blob/bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/hf/model.py) uses GroupNorm. Changing the physical context of a fixed128 window can therefore change statistics as well as apparent stroke width. The experiment measures the complete sampling/context path, not purely internal scale equivariance.

## Source geometry and independence

The [prepared dataset](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/tree/ad919d288ede1c4daf92725288bb6fe29c558591) stores16 native prepared planes and uint8 CT, with labels0 unknown,1 background and2 ink. Its [render receipt](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/resolve/ad919d288ede1c4daf92725288bb6fe29c558591/donor-studio-resunet-scratch-20260926/prepared/841-expanded-layer-03-approved-r000001/render.json) attributes XY pitch9.36µm and depth pitch9.366µm. Those pitches explain physical field of view but do not establish the orientation of writing lines in this array.

This is a donor/control used repeatedly in release selection and repository diagnostics. Reproducing its published AP checks integration. Perturbing it measures local sensitivity. Even a passing ensemble gate would need a new held-out control, a second scroll and the production tiling/renderer path before deployment. No label population confidence interval is inferred from correlated pixels or53 neighbouring windows.
