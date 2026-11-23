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

## Direct image check at the withheld heights

A standard GitHub runner downloaded the public voxels from S3. For each site it applied the translation predicted from the two other heights, then evaluated both original scans on the two reserved axial planes. A second run using only the cache produced byte-identical output. [Execution log](https://github.com/JorisViaudQuantAI/Scrollprize/actions/runs/44902739781) · [All 41 paired site results](height-transfer.json)

| Scroll | Uncorrected → corrected median band-pass correlation | Sites improved | Uncorrected → corrected median raw intensity correlation | Sites improved |
| --- | ---: | ---: | ---: | ---: |
| PHerc. 1203 | 0.875 → 0.929 | 16 / 17 | 0.856 → 0.913 | 16 / 17 |
| PHerc. 0846A | 0.790 → 0.932 | 18 / 24 | 0.782 → 0.916 | 19 / 24 |

All remaining paired correlations decreased. The median of the paired gains in band-pass correlation was +0.069 on 1203 and +0.139 on 0846A. The SHA-256 of the archived site data is `638948570f691c9c78389d3537ac3415a5ce0be93034aedec1564e06f6dccd6e`. These are measurements of tissue alignment in held-out images. Implementations that transfer a particle or mesh vertex still need to check which sheet and side the vertex lands on.
