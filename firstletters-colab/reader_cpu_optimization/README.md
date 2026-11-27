# B15–B17 — Reader v3 CPU preprocessing, false-positive and depth-alignment experiments
26 Nov 2026. Continuation of B13/B14. Local CPU only; no Reader v3 weights downloaded, no fresh network forward, no new unread CT or mesh. Source arrays already supplied by owner and hash pinned.

## B15: exact integer robust MAD specialization
[Reader v3 public config](https://github.com/DomRusso2/reader-v3/blob/main/train_config.json) uses 384 spatial patches, 17 slices and robust_mad 1/99. [Villa b29f310957c7b29e7d332f9876e0e487819cfaa3](https://github.com/ScrollPrize/villa/tree/b29f310957c7b29e7d332f9876e0e487819cfaa3) flat inference calls normalize_robust. For byte-valued patches compute histogram, exact order statistics and float32 clipped median/MAD; normalize via a 256-entry LUT. Initial synthetic checks exposed scalar NumPy percentile rounding differences (max observed absolute mismatch 9.54e-7); corrected using scalar dtype-matched virtual rank/interpolation, then reran. NumPy 2.3.5, Python 3.13.5. Fallback to original for degenerate MAD, non-integer data, signed-zero or unsupported dtype. Guarded candidate keeps the reader's float32 input contract; direct uint8 benchmark is an optional alternative, not installed.

| Stage | Reference ms | Direct uint8 ms | Guarded float32 ms |
|---|---:|---:|---:|
| normalization | 77.510 | 11.089 | 13.228 |
| construction + normalization | 79.946 | 12.257 | 14.519 |

168 real padded crops (28 CT tiles x 3 positions x 2 depth directions), 336 bitwise comparisons, 1556 synthetic checks, 18 degenerate fallbacks, 1008 timings, 59.24s local wall. Exact float32 bit patterns versus a faithful unmasked/default upstream operation sequence, not a full Villa/model run. The existing 16-depth Studio CT is padded to 17 for benchmark only. Median ratios 6.99x and 6.52x direct; 5.86x and 5.51x guarded. These are preprocessing-only measurements, not full inference speedups or memory measurements. Real Villa integration needs paired end-to-end CLI test and exact decoded outputs before promotion. The installer is syntax checked, not executed on a local full Villa checkout; verifies both pinned source blob hashes, enables only NumPy 2.3.5 and 17x384x384, otherwise original.

## B16: stroke postprocessing and calibration shift
181 guarded folds attempted, 169 admitted, 2185 method/arm/fold evaluations, 5 synthetic checks. Real archived Studio scores (not Reader), common 3x3 valid support; calibrate thresholds targeting 80% positive recall on spatially separated identity cores. All filters fail prospective gate (FPR reduction >=2 points, recall loss <=2 points). Scale mixed subset: line support changes 1293/2875 to 1290/2875 TP, 505/2525 to 489/2525 FP; 0.104pp recall loss and 0.634pp FPR improvement. Mean pose mixed line delta -0.028pp recall/-0.114pp FPR. Identity raw held mixed recall 44.97%, FPR20.00%; same identity-calibrated threshold under x-half gives recall44.59%, FPR38.26%, x-double recall18.23%, FPR0.119%. A threshold that appears to reject false positives may suppress most true ink. Twelve scale folds abstain; report only admitted support and no universal threshold. Same exposed donor, chart guards not verified native disjointness.

## B17: relative depth registration, not actual anatomical correction
28 actual 128x128x16 CT crops; 363888 interior, 318319 informative reference pixels per stress arm. 2100 per-tile evaluation rows, 4 synthetic checks, 9.595s. Controlled integer axial shifts and a radiometric perturbation (gain1.15, bias8, noise SD3). ZNCC vs SSD, seven candidates, window sizes 1/5/15. Gate best correlation >=.6, margin >=.1. Deliberately unrelated XY crop as decoy.
| Window | Noisy ramp exact/accepted (ZNCC) | Coverage among informative | Decoy accepted / informative |
|---|---:|---:|---:|
| 1 |259518/259575|81.55%|91255/318319|
| 5 |269592/269741|84.74%|21663/318319|
| 15 |242095/242319|76.12%|290/318319|

5x5 SSD gets 255605/318319 exact, 80.30%. 15x15 gated ZNCC gets 99.91% exact among admitted, decoy acceptance 0.091%, but abstains more and risks smoothing a depth discontinuity: step boundary accepted 18926/39287 vs 28936 with 5x5. No native CT acquisition, no actual anatomical sheet identity, no mesh repair or new ink recall. Easy synthetic perturbation from same reference, decoys are nonoverlapping chart crops only. Use only as a candidate relative alignment check before physical seam/anchor tests; no promotion.

## Reader v3 transfer audit: author results, not new independent replay
Pinned author config and published metric blobs in summary.json. Eight training group names: 0139,1667,Paris4,0814,0009B,0500P2,0343P,0172. The config omits 0841/1447; full checkpoint ancestry and parent equivalence still need independent verification. 0139 and 0009B are training groups, not independent cross-manuscript validation. On five published 0841/1447 regions, v3 vs v2 false-ink rate at 80% recall macro 30.4% vs 35.4%; one region (0841_ag1) worsens 19.9% ->32.3%, w060 still51.9%. No universal calibration. Compare native aligned masks, label provenance, per-parent metrics, worst-group FPR/recall, open-stroke continuity and blank controls at both energies. No target label tuning in held-out groups.

## Reproduce
Use original ct-tiles.npy, scale_stress.npz, pose_factors.npz, coordinates.npy, labels.npy, original names and expected SHA256 in scripts. From this directory:
```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench.py /path/to/data results.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python stroke_guard.py /path/to/data stroke_results.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python registration.py /path/to/data registration_results.json
python install_villa.py /path/to/pinned/villa # dry run only
```
All five published script Git blob identities verified against local bytes. Compact summary.json checked in; full per-row local results ZIP provided to owner, SHA256 b8cbd50d54c986783ed57218cf3ca524d72ef5d80e2b0bcfdf1c3dbee9665197. Wall fields vary across reproductions.

Checkpoint alternative: Reader v3 public release is 138 MB (vs Studio 893 MB), [download](https://huggingface.co/domenicor046/reader-v3/resolve/39f626e5/reader-v3.pth). Local DNS and attempted binary fetch failed, not downloaded. Original Studio can be split losslessly into smaller parts or accessed by a network-enabled runner; no quantized checkpoint prepared or claimed. Prioritize full CLI A/B on awarded 846B or existing W058 and W00 public fixtures, then native paired alternate surfaces, then leave-parent-out Reader FP calibration. No GPU budget or production dispatch, main merge, or prize claim.
