"""Regression: an HTTP failure must not become a blank CT crop.

Run from the repository root:
    python -m unittest discover -s firstletters-colab/tests -v

The tests run the real downloader paths against in-memory HTTP responses.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
META = {
    "zarr_format": 2, "shape": [1, 2, 2], "chunks": [1, 2, 2],
    "dtype": "|u1", "compressor": None, "filters": None,
    "fill_value": 0, "order": "C", "dimension_separator": "/",
}
VALUES = np.array([[[11, 27], [43, 89]]], dtype=np.uint8)


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RENDER = load_module("http_test_render", "ink/render_tifxyz.py")
REG = load_module("http_test_registration", "registration/reg_point.py")
CROP = load_module("http_test_crop", "hecate_fp16/prepare_crop.py")


class FakeSession:
    def __init__(self, chunk_status=200, error=None):
        self.chunk_status = chunk_status
        self.error = error
        self.chunk_requests = 0

    def get(self, url, **kwargs):
        response = requests.Response()
        response.url = url
        if url.endswith(".zarray"):
            response.status_code = 200
            response._content = json.dumps(META).encode()
        else:
            self.chunk_requests += 1
            if self.error is not None:
                raise self.error
            response.status_code = self.chunk_status
            response._content = VALUES.tobytes() if self.chunk_status == 200 else b"HTTP error"
        return response


def run_downloader(name, session):
    if name == "render":
        with mock.patch.object(RENDER.requests, "Session", return_value=session), \
                mock.patch.object(RENDER.time, "sleep"):
            reader = RENDER.RemoteZarrArray("https://example.invalid/ct.zarr", workers=1)
            try:
                data, origin = reader.read_box((0, 0, 0), (1, 2, 2))
                assert origin == (0, 0, 0)
                return data
            finally:
                reader.pool.shutdown(wait=True)
    if name == "registration":
        with mock.patch.object(REG, "sess", return_value=session):
            return REG.zslice("https://example.invalid", "ct.zarr", 0,
                              0, 0, 2, 0, 2, workers=1)[None]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "crop.npy"
        argv = ["prepare_crop.py", str(output), "--size", "2", "--y0", "0", "--x0", "0",
                "--src-um", "9.6", "--dst-um", "9.6"]
        with mock.patch.object(CROP.requests, "Session", return_value=session), \
                mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            try:
                CROP.main()
            except Exception:
                # An aborted download must not leave a seemingly valid blank crop.
                assert not output.exists()
                raise
        return np.load(output)


class MissingChunkTests(unittest.TestCase):
    downloaders = ("render", "registration", "crop")

    def test_success_preserves_real_voxels(self):
        for name in self.downloaders:
            with self.subTest(downloader=name):
                np.testing.assert_array_equal(run_downloader(name, FakeSession()), VALUES)

    def test_sparse_404_retains_zero_fill(self):
        for name in self.downloaders:
            with self.subTest(downloader=name):
                session = FakeSession(404)
                np.testing.assert_array_equal(run_downloader(name, session), np.zeros_like(VALUES))
                self.assertEqual(session.chunk_requests, 1)

    def test_forbidden_403_raises_instead_of_zero_fill(self):
        for name in self.downloaders:
            with self.subTest(downloader=name):
                with self.assertRaises(requests.HTTPError) as error:
                    run_downloader(name, FakeSession(403))
                self.assertEqual(error.exception.response.status_code, 403)

    def test_server_500_raises_instead_of_zero_fill(self):
        for name in self.downloaders:
            with self.subTest(downloader=name):
                with self.assertRaises(requests.HTTPError) as error:
                    run_downloader(name, FakeSession(500))
                self.assertEqual(error.exception.response.status_code, 500)

    def test_timeout_raises_instead_of_zero_fill(self):
        for name in self.downloaders:
            with self.subTest(downloader=name):
                with self.assertRaises(requests.Timeout):
                    run_downloader(name, FakeSession(error=requests.Timeout("fixture timeout")))


if __name__ == "__main__":
    unittest.main()
