# HTTP missing-chunk regression

The renderer, registration slice reader and Hecate timing-crop downloader must
raise on HTTP 403, HTTP 500 or an exhausted transport failure. Those responses do
not establish that a sparse Zarr chunk is absent. Silently substituting zeroes
could make a failed download look like a negative ink result.

Run from the repository root with the existing CPU dependencies installed
(`numpy`, `scipy`, `opencv-python-headless`, `scikit-image`, `tifffile`, `requests`):

```bash
python -m unittest discover -s firstletters-colab/tests -v
```

Five tests exercise each of the three real download paths: successful voxels,
sparse HTTP 404 zero fill, HTTP 403, HTTP 500, and timeout. Responses are mocked;
no network, scan data or GPU is needed. Existing retry behavior is preserved.

This narrow fix preserves the existing HTTP 404 sparse-fill convention. It does
not prove that a returned 404 represents a missing public object, validate every
possible HTTP response or purge cached crops produced by earlier code. Use a
fresh cache for a reproducibility run when prior download integrity is unknown.

The text-score tests use synthetic rows separated by 5.5 mm, a 2.8 mm tear, a flat control and a contiguous support mask. They also check the calibration warning from the CLI.
