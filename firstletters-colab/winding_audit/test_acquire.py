"""Offline acquisition-contract checks using actual NumPy ZIP bytes."""
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(os.environ.get("SCROLL_ACQUIRE_PATH", Path(__file__).with_name("acquire.py")))
spec = importlib.util.spec_from_file_location("winding_acquire_under_test", SOURCE)
acquire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquire)


class Response(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status, self.headers = status, headers


class Server:
    def __init__(self, body, mode="ok"):
        self.body, self.mode, self.calls = body, mode, []

    def __call__(self, request, timeout):
        self.calls.append(dict(request.header_items()))
        lo, hi = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        payload = self.body[lo:hi + 1]
        headers = {"Content-Range": f"bytes {lo}-{hi}/{len(self.body)}", "ETag": '"v1"'}
        status = 206
        if self.mode == "whole": status = 200
        if self.mode == "wrong_range": headers["Content-Range"] = f"bytes {lo+1}-{hi+1}/{len(self.body)}"
        if self.mode == "impossible_total": headers["Content-Range"] = f"bytes {lo}-{hi}/{hi}"
        if self.mode == "short": payload = payload[:-1]
        if self.mode == "long": payload += b"x"
        if self.mode == "encoded": headers["Content-Encoding"] = "gzip"
        if self.mode == "missing_etag": headers.pop("ETag")
        if self.mode == "changed_etag" and len(self.calls) > 1: headers["ETag"] = '"v2"'
        return Response(payload, status, headers)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.serial = 0

    def ranges(self, **overrides):
        self.serial += 1
        limits = dict(maximum_source_body_bytes=1_000_000, maximum_requests=100,
                      maximum_acquisition_seconds=60)
        limits.update(overrides)
        return acquire.Ranges(Path(self.temp.name) / str(self.serial), limits)

    def test_stored_numpy_zip_scalar_metadata_and_array_offsets(self):
        array = np.arange(30, dtype=">i4").reshape(10, 3)
        stream = io.BytesIO()
        np.savez(stream, metadata=np.array(json.dumps({"db_signature": "known"})), coordinates=array)
        data = stream.getvalue()
        server, ranges = Server(data), self.ranges()
        with patch.object(acquire.urllib.request, "urlopen", server):
            members = acquire.zip_arrays(ranges, "https://fixture/archive.npz", len(data))
            self.assertEqual(json.loads(str(members["metadata"].all()[0])), {"db_signature": "known"})
            np.testing.assert_array_equal(members["coordinates"].all(), array)
            np.testing.assert_array_equal(members["coordinates"].rows([9, 0, 4, 4], gap=0), array[[9, 0, 4, 4]])
            before = len(server.calls)
            self.assertEqual(members["coordinates"].block(3, 3).shape, (0, 3))
            self.assertEqual(members["coordinates"].rows([]).shape, (0, 3))
            self.assertEqual(len(server.calls), before)
        self.assertTrue(all(row["status"] == "complete" for row in ranges.rows))

    def test_compressed_zip_is_rejected(self):
        stream = io.BytesIO()
        np.savez_compressed(stream, values=np.arange(100))
        with patch.object(acquire.urllib.request, "urlopen", Server(stream.getvalue())):
            with self.assertRaisesRegex(ValueError, "ZIP_STORED"):
                acquire.zip_arrays(self.ranges(), "https://fixture/compressed.npz", len(stream.getvalue()))

    def test_adjacency_keeps_empty_boundary_rows_and_omits_unselected_nodes(self):
        values = np.arange(15, dtype="<i8").reshape(3, 5)
        server, ranges = Server(values.tobytes()), self.ranges()
        arrays = {name: acquire.Array(ranges, "https://fixture/raw", i*40, "<i8", (5,))
                  for i, name in enumerate(("partners", "self_local", "partner_local"))}
        offsets = np.array([0, 0, 2, 2, 3, 3, 5, 5])
        ids = np.array([0, 1, 2, 4, 5, 6])  # Node 3's adjacency is deliberately excluded.
        with patch.object(acquire.urllib.request, "urlopen", server):
            for gap in (0, 32):
                rows = acquire.read_adjacency(arrays, ids, offsets, gap)
                self.assertEqual(set(rows), set(ids))
                for node in ids:
                    for k in range(3):
                        np.testing.assert_array_equal(rows[node][k], values[k, offsets[node]:offsets[node+1]])
        self.assertEqual(acquire.coalesce([(5, 7), (0, 2), (2, 4), (4, 4)], 0), [[0, 4], [5, 7]])
        self.assertEqual(acquire.coalesce([(5, 7), (0, 2), (2, 4)], 1), [[0, 7]])

    def test_invalid_range_responses_fail_and_never_complete_receipt(self):
        for mode in ("whole", "wrong_range", "impossible_total", "short", "long", "encoded", "missing_etag"):
            with self.subTest(mode=mode):
                ranges = self.ranges()
                with patch.object(acquire.urllib.request, "urlopen", Server(b"0123456789", mode)):
                    with self.assertRaises(ValueError): ranges.get("https://fixture/raw", 1, 4)
                self.assertEqual(ranges.rows[0]["status"], "failed")

    def test_etag_change_and_conditional_request(self):
        server, ranges = Server(b"0123456789", "changed_etag"), self.ranges()
        with patch.object(acquire.urllib.request, "urlopen", server):
            self.assertEqual(ranges.get("https://fixture/raw", 0, 2), b"01")
            with self.assertRaisesRegex(ValueError, "changed"): ranges.get("https://fixture/raw", 2, 4)
        self.assertEqual(server.calls[1].get("If-match"), '"v1"')

    def test_budgets_block_before_network(self):
        for limits in (dict(maximum_source_body_bytes=3), dict(maximum_requests=1)):
            ranges, server = self.ranges(**limits), Server(b"0123456789")
            with patch.object(acquire.urllib.request, "urlopen", server):
                ranges.get("https://fixture/raw", 0, 2)
                with self.assertRaises(RuntimeError): ranges.get("https://fixture/raw", 2, 4)
            self.assertEqual(len(server.calls), 1)
        ranges, server = self.ranges(), Server(b"0123456789")
        with patch.object(acquire.urllib.request, "urlopen", server), \
                patch.object(acquire.time, "monotonic", return_value=ranges.started + 61):
            with self.assertRaisesRegex(RuntimeError, "time budget"):
                ranges.get("https://fixture/raw", 0, 2)
        self.assertEqual(len(server.calls), 0)

    def test_invalid_array_indices_are_not_silently_truncated(self):
        array = acquire.Array(self.ranges(), "https://fixture/raw", 0, "<i8", (5,))
        with patch.object(acquire.urllib.request, "urlopen") as network:
            for indices in ([0.5], [-1], [5]):
                with self.subTest(indices=indices), self.assertRaises(ValueError): array.rows(indices)
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
