# Transferring a correction to an unseen height

Can the geometry at two other heights predict the local CT displacement on a third? We used the 41 material-qualified sites in the published residual table. Each test removes every site at the height under evaluation from training. Three folds leave out A, B and C in turn. The fitted displacements come from CT anatomy. [Source measurements](residuals.tsv)

| Scroll | Measured sites | Original median / P90 error, µm | Constant median shift | Linear correction | Sites improved by the linear correction |
| --- | ---: | ---: | ---: | ---: | ---: |
| PHerc. 1203 | 17 | 7.76 / 14.46 | 9.93 / 11.97 | 4.25 / 7.18 | 15 / 17 |
| PHerc. 0846A | 24 | 13.02 / 18.70 | 12.38 / 19.88 | 5.29 / 9.76 | 21 / 24 |

The outcome is the Euclidean distance to the measured XYZ displacement of a withheld site. We used the fixed scan's 9.362 µm pitch. Each scroll has its own models. The constant model uses the componentwise median. The linear model fits a translation in three coordinates by ridge regression. It centers the fixed coordinates on the training sample, divides them by 1,000 fixed voxels, and applies a unit ridge penalty to the three slopes. The intercept is unpenalized. These values were chosen for this exploratory follow-up. The study samples three heights in the scanned band.

Running the first command repeats the table calculation from the pinned site data. With a cache directory the second command also applies each prediction directly to the original scans. It measures raw and band-pass correlations on the two reserved image planes at every site, and writes `height-transfer.json` to the cache. An existing annular cache can be reused with `--offline`.

```bash
node registration-checks/annular/height-transfer.cjs
node registration-checks/annular/height-transfer.cjs ./annular-cache
```
