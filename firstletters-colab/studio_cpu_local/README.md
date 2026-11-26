# B13 — completed local CPU experiments (26 Nov 2026)

Owner supplied all seven public PHerc0841 artifacts requested after B12. SHA256 of every input verified against the released Q10/Q33 protocol and archive records. All 53 labeled central32 regions match labels.npy and coordinates; all common identity pixels across the three prediction archives are bitwise equal, as are the corresponding rigid-rotation pixels on common masks. 28 prepared 512x512x16 uint8 tiles; 178423 known labeled pixels in the canvas (111310 background, 67113 ink); 45349 scored central-core pixels (15791 ink,29558 background). Unknown remains excluded. This is one exposed donor, not independent papyrus validation. Studio checkpoint not supplied; no new Studio forward. This work extends B9/#12 and is a new local supervised CPU baseline, not a proposal alone.

## B9 real pixelwise follow-up

Original score_fusion.py exact blob f7212ef5430c718799c8f72926e3932d91ca18a0, 176 held-core folds executed locally (wall 1.35s). [Counts](fusion_counts.tsv) guard=0 means leave-one-core-out, product not tested there. Baseline geometry correct: max and zmean improve recall 0/8 blocks; mean 1/8. Rigid-degraded geometry with reference-derived hybrids: max recall improves 8/8 (+0.71 to +18.42 pp), but FPR often rises. Raw OR adds 0–199 ink detections and 57–1237 background detections across the identity cases; 352 literal core/class bound checks confirm B9; [all 16 actual union counts and bounds](or_bounds.tsv). For y−20 rigid OR, +2975 TP, +2077 FP exactly.

Further spatial guard protocol 1e3fa83a. 352 more held-core fits, origin Chebyshev distance >=128 or >=192 prepared-chart pixels. All 32 perturbation checks of held/excluded scores leave calibration thresholds unchanged. At guard192, max on identity loses 3.29–14.12 recall pp, wins 0/8; product wins 1/8 with lower FPR. On rigid max gains 1.69–21.57 recall pp and FPR changes −0.87 to +5.39 pp, just 1/8 achieves higher recall without higher FPR. Product does so in 2/8, not universal. Calibration caps *training* background FP, not held-out FPR. No chosen fusion rule or deployment; oracle hybrids require reference geometry. Spatial guard is nominal chart, not native-XYZ separation. Run wall 2.44s (1.87+0.56 CPU), internal loop 0.586s. All blocks and methods retained.

## B13 real trained CT baselines

Protocol frozen 2a1704e4 before training, code byte-identical to executed script. Reconstruct 517x554 rectangle with verified 9x9 feature support. 31 profile features (16 intensities +15 depth differences), or 111 context features (3/9 local mean, SD, raw-minus-local mean). Twelve folds: three row-origin tertile or three column-origin tertile groups at guards 128/192; each train pixel has Chebyshev distance >guard from every test core; independently checked literal rectangle distances. All 36 model fits complete (logistic, HGB profile, HGB context). No training with unknown labels, no whole-panel train, no threshold tuning; same donor and exposed historical Studio comparator. 32 literal CT feature checks and independent tie-aware AP check for every fold/aggregate.

Primary macro fold AP at guard192:

| Method | Row partition | Column partition |
|---|---:|---:|
| Historical Studio | .729942 | .793755 |
| Logistic profile | .353902 | .318921 |
| HGB profile | .366272 | .310460 |
| HGB context | .361148 | .313409 |
| Fixed mean Studio + context | .611859 | .606782 |
| Fixed product Studio * context | .472254 | .528752 |

Both primary gates fail. No classical model or fixed fusion beats Studio on any of the 12 folds by AP. Studio pooled AP .753604, TP10718/15791 and FP5765/29558. Context learner at row192 TP1235 and FP2733: quiet scores not good detection. Product row192 TP346 and FP156. Differences partly reflect spatial prevalence/domain shifts and limited training support (1888 training ink in smallest fold); do not extrapolate to unseen scrolls. [All 12 fold metrics plus every pooled and macro aggregate](ct_compact.tsv): each method cell is AP,AUC,TP,FP except explicitly marked macro rows (AP,AUC,recall,FPR). Recall/FPR computed from exact counts otherwise. No GPU or new unread candidates.

## Reproduction

Use Python3.13.5, numpy2.3.5, scipy1.17.0, sklearn1.8.0, threadpoolctl. Put the seven owner-supplied public files in one directory named as PINS in train_ct.py. Exact upstream prepared source revision ad919d288ede1c4daf92725288bb6fe29c558591 on HF dataset paranke/vesuvius-814-841-9b-studio-training-data; Q10/Q33 NPZ in existing repository evidence.

```sh
cd firstletters-colab
python geometry_review/rescue_bounds/score_fusion.py /path/input/pose_factors.npz > /tmp/b9_pixelwise.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python studio_cpu_local/train_ct.py --input /path/input --output /tmp/b13 --threads 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python studio_cpu_local/guarded_fusion.py --input /path/input --output /tmp/b13/guarded_fusion.json --b9-script geometry_review/rescue_bounds/score_fusion.py
```

Actual training process elapsed 22.03s; user+sys 71.39s, peak RSS 526916 KiB. Internal run 20.037s. All code/results tables in this branch were verified by comparing local and GitHub git blob SHAs: train_ct.py 400e147637a2cf5f538abe3ab7bd2be8e8ad7719; guarded_fusion.py b367a966041b3d7dca0e8ec7b4e58ef52aed7808; ct_compact.tsv 0ded4fa3f54f05d066352b407eae47d69b4e7546; fusion_counts.tsv 04507ee7d65c32b7a0909b5ee1ecfafdf88a62b4; or_bounds.tsv ed3ba16e8ee91192c7ea04cd920f6401bc1dcf38. Full local output supplied to owner as ZIP rather than large binary in git; result.json SHA256 0d72cb9619ddad852c298865d357e7e8d666260b29ceba69d0bb2bbf36d27bdc; predictions.npz 8009af40f7bf7d30553c453bdef7d04aa4745fd9c2b01963163c9f7e273669f4; b9_pixelwise.json 201f09695f3e8000d46f91a12dc653658fabd025f68aa8d1bd8e40055a11faf8; guarded_fusion.json d04a61b956f1e61f572c80717f47c8ee36913192228e43606d6670320aedabaf. Timing fields vary. No merge, automated skip or generalization claim.

## Handoff

Gemini: do not repeat B9 pixelwise and spatial guard; prioritize truly distinct models at matched physical support. Claude: geometry repair rather than brute fusion of bad renders. Codex: independent verify input SHA pins, train/test guard and AP. B13 weak classical learner is a negative control only, not a new deployment candidate. Next actual Studio inference needs the missing hash-pinned checkpoint (893443724 bytes).
