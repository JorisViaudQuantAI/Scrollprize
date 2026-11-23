"""The 19 offline checks used before freezing the bounded NIST reader.

Run from the repository root:
    python -B -m unittest discover -s firstletters-colab/surface_sampling \
        -p test_nist_source.py -v

Only generated TIFF/Deflate bytes and simulated HTTP responses are used.
Opening a real urllib transport is blocked for every check. No source pixels,
private metadata cache, network service, or acquisition freeze is required.
"""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np
from PIL import Image


_MODULE_PATH = Path(__file__).with_name("nist_source.py")
_SPEC = importlib.util.spec_from_file_location("nist_source_offline_checks", _MODULE_PATH)
source = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(source)


def _tiff_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="TIFF", compression="raw")
    return buffer.getvalue()


class _Response:
    def __init__(self, body, url, etag, content_range="bytes 0-3/293236353"):
        self.stream = io.BytesIO(body)
        self.url = url
        self.status = 206
        self.headers = {"Content-Range": content_range, "ETag": etag}

    def geturl(self):
        return self.url

    def read(self, length):
        return self.stream.read(length)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()


class _FakeOpener:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def open(self, request, timeout):
        self.calls += 1
        return self.response


class NistSourceOfflineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.manifest_digest = source._load_manifest(
            _MODULE_PATH.with_name("nist_manifest.json")
        )

    def setUp(self):
        self.network_block = patch.object(
            source.urllib.request.OpenerDirector,
            "open",
            side_effect=AssertionError("Real HTTP is forbidden in offline checks"),
        )
        self.network_block.start()
        self.addCleanup(self.network_block.stop)
        self.temporary = tempfile.TemporaryDirectory(prefix="nist-source-offline-check-")
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name)
        self.array = np.arange(20, dtype=np.float32).reshape(4, 5) - 10
        self.raw = _tiff_bytes(self.array)

    def _deflate_fixture(self):
        encoder = zlib.compressobj(wbits=-15)
        compressed = encoder.compress(self.raw) + encoder.flush()
        entry = {"decoded_file_bytes": len(self.raw),
                 "crc32": zlib.crc32(self.raw) & 0xFFFFFFFF}
        return compressed, entry

    def _transfer(self, body=b"abcd", content_range="bytes 0-3/293236353"):
        baseline = self.manifest["accounting"]["budget_baseline"]
        receipt = {
            "transfer": {
                "new_requests": 0, "new_body_bytes": 0,
                "charged_requests": baseline["charged_requests"],
                "charged_body_bytes": baseline["body_bytes"],
            },
            "requests": [],
        }
        archive = self.manifest["archive"]
        transfer = source._Transfer(self.manifest, self.cache, receipt)
        transfer.opener = _FakeOpener(_Response(
            body, archive["canonical_url"], archive["observed_etag"], content_range
        ))
        return transfer, receipt

    def _fetch_four(self, transfer):
        archive = self.manifest["archive"]
        return transfer.fetch(archive["canonical_url"], 4, start=0,
                              etag=archive["observed_etag"])

    def test_float32_tiff_round_trip(self):
        np.testing.assert_array_equal(source._decode_tiff(self.raw), self.array)

    def test_nonfinite_tiff_rejected(self):
        bad = self.array.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            source._decode_tiff(_tiff_bytes(bad))

    def test_integer_tiff_rejected(self):
        with self.assertRaises(ValueError):
            source._decode_tiff(_tiff_bytes(self.array.astype(np.uint16)))

    def test_common_shape_checked(self):
        with self.assertRaises(ValueError):
            source._decode_tiff(self.raw, (5, 4))

    def test_bounded_deflate_and_crc(self):
        compressed, entry = self._deflate_fixture()
        self.assertEqual(source._inflate(compressed, entry), self.raw)

    def test_crc_mismatch_rejected(self):
        compressed, entry = self._deflate_fixture()
        with self.assertRaises(ValueError):
            source._inflate(compressed, dict(entry, crc32=0))

    def test_deflate_trailing_data_rejected(self):
        compressed, entry = self._deflate_fixture()
        with self.assertRaises(ValueError):
            source._inflate(compressed + b"extra", entry)

    def test_deflate_decoded_size_bounded(self):
        compressed, entry = self._deflate_fixture()
        with self.assertRaises(ValueError):
            source._inflate(compressed, dict(entry, decoded_file_bytes=20))

    def test_descriptor_style_local_header(self):
        entry = self.manifest["members"][0]
        header = struct.pack("<4s5H3I2H", b"PK\x03\x04", 20, 8, 8,
                             0, 0, 0, 0, 0, 55, 32)
        self.assertEqual(source._local_header(header, entry), (55, 32))

    def test_local_compression_method_mismatch_rejected(self):
        entry = self.manifest["members"][0]
        header = struct.pack("<4s5H3I2H", b"PK\x03\x04", 20, 8, 0,
                             0, 0, 0, 0, 0, 55, 32)
        with self.assertRaises(ValueError):
            source._local_header(header, entry)

    def test_zip_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            source._safe_member_name("recon_binned_9x_down_sampling/../escape")

    def test_missing_freeze_hash_rejected_before_file_access(self):
        with self.assertRaises(ValueError):
            source._freeze_digest(self.cache / "absent-freeze", None, self.manifest_digest)

    def test_exact_range_body_and_ledger(self):
        transfer, receipt = self._transfer()
        self.assertEqual(self._fetch_four(transfer), b"abcd")
        self.assertEqual(receipt["transfer"]["new_requests"], 1)
        self.assertEqual(receipt["transfer"]["new_body_bytes"], 4)

    def test_content_range_total_mismatch_rejected_before_body(self):
        transfer, receipt = self._transfer(content_range="bytes 0-3/999")
        with self.assertRaises(ValueError):
            self._fetch_four(transfer)
        self.assertEqual(receipt["transfer"]["new_body_bytes"], 0)

    def test_overlong_http_body_rejected_and_counted(self):
        transfer, receipt = self._transfer(body=b"abcde")
        with self.assertRaises(ValueError):
            self._fetch_four(transfer)
        self.assertEqual(receipt["transfer"]["new_body_bytes"], 5)

    def test_request_cap_blocks_transport(self):
        transfer, receipt = self._transfer()
        receipt["transfer"]["charged_requests"] = 300
        with self.assertRaises(RuntimeError):
            self._fetch_four(transfer)
        self.assertEqual(transfer.opener.calls, 0)

    def test_body_cap_blocks_transport(self):
        transfer, receipt = self._transfer()
        receipt["transfer"]["charged_body_bytes"] = 96 * 1024 * 1024 - 4
        with self.assertRaises(RuntimeError):
            self._fetch_four(transfer)
        self.assertEqual(transfer.opener.calls, 0)

    def test_cache_inside_git_rejected(self):
        checkout = self.cache / "synthetic-checkout"
        checkout.mkdir()
        (checkout / ".git").mkdir()
        with self.assertRaises(ValueError):
            source._new_private_cache(checkout / "forbidden-cache")

    def test_existing_cache_rejected(self):
        with self.assertRaises(FileExistsError):
            source._new_private_cache(self.cache)


if __name__ == "__main__":
    unittest.main()
