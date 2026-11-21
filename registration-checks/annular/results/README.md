# Measured residual displacement on two scrolls

We checked the published affine transforms on 48 anatomical sites chosen before reading the local cross-scan agreement. The corrections below are translations of the fixed CT, estimated on three planes per site and tested on two held-out planes. The distances describe material at the sampled sites. [Protocol and frozen coordinates](../)

| Scroll | Sites with material / planned | Median / P90 in fixed 9.362 µm voxels | Median / P90 in native fine 2.403 µm voxels | Median / P90 in µm | Held-out band-pass correlation, median before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| PHerc. 1203 | 17 / 24 | 0.83 / 1.54 | 3.23 / 6.02 | 7.76 / 14.46 | 0.875 → 0.946 |
| PHerc. 0846A | 24 / 24 | 1.39 / 2.00 | 5.42 / 7.78 | 13.02 / 18.70 | 0.790 → 0.951 |

Seven PHerc. 1203 sites fell outside the material mask. On PHerc. 0846A the 22 points within the 10–20 mm band for both centerline estimates give 13.55 µm median and 18.91 µm P90. The quantiles are over site-level Euclidean displacements. At all 41 measurable sites the corrected band-pass correlation exceeded every ±12-voxel displacement control on held-out planes. The change relative to the nominal transform was positive at 40 sites and zero at one site. For raw intensity correlation it was positive at 39 sites, zero at one and −0.00065 at one. No fit reached a search boundary.

The first direct S3 run under Node 24.22.1 read all 411 planned ranges, including 65 confirmed implicit zero chunks. An offline replay after the cache update produced an identical site table. The SHA-256 of `residuals.tsv` is `624c459a6a6e1148802c8307b17e2514d6dcee77a0b76db801811e6e61a8f480`. The table contains every planned site. [Full site table](residuals.tsv) · [Machine-readable summary](summary.json)

From the repository root, run the following commands to download the public inputs and repeat the calculation from the cached bytes. On a host with a required HTTP proxy, Node 24 supports adding `--use-env-proxy` before the program name on the first command.

```bash
node registration-checks/annular/run.cjs ./annular-cache
node registration-checks/annular/run.cjs ./annular-cache --offline
```
