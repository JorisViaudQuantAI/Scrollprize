import copy
from email.message import Message
import http.client
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("public_range_transport", HERE / "run_transport.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def headers(length, *, span=None, total=None, etag='"fixed-v1"', **extra):
    result = Message()
    result["Content-Length"] = str(length)
    result["ETag"] = etag
    if span is not None:
        result["Content-Range"] = f"bytes {span[0]}-{span[1]}/{total}"
    for key, value in extra.items():
        result[key] = value
    return result


class Response:
    def __init__(self, body=b"", status=200, fields=None):
        self.body, self.status = io.BytesIO(body), status
        self.headers = fields if fields is not None else headers(len(body))
        self.read_calls, self.closed = 0, False

    def read(self, n):
        self.read_calls += 1
        return self.body.read(n)

    def close(self):
        self.closed = True


class QueueOpener:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def open(self, request, timeout):
        self.requests.append(request)
        return self.responses.pop(0)


class TransportTests(unittest.TestCase):
    def test_fixed_boxes_reconstruct_c_order_and_cross_depth(self):
        shape = (65, 128, 128)
        a = (np.arange(np.prod(shape)) % 251).astype(np.uint8).reshape(shape)
        raw = a.tobytes()
        boxes = ([0, 16, 12, 1, 17, 28], [5, 7, 120, 6, 10, 128], [31, 0, 0, 33, 128, 128])
        self.assertEqual([len(t.plan_box(b, shape)) for b in boxes], [1, 3, 1])
        for box in boxes:
            spans = t.plan_box(box, shape)
            actual = b"".join(raw[lo:hi+1] for lo, hi in spans)
            z, y, x, Z, Y, X = box
            self.assertEqual(actual, a[z:Z, y:Y, x:X].tobytes())

    def test_bad_boxes_fail(self):
        for b in ([0, 0, 0, 66, 1, 1], [-1, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 0], [False, 0, 0, 1, 1, 1]):
            with self.assertRaises(ValueError):
                t.plan_box(b, (65, 128, 128))

    def test_exact_range_headers(self):
        length, etag = t.validate_headers(206, headers(3, span=(2, 4), total=10), object_size=10, byte_range=(2, 4), etag='"fixed-v1"')
        self.assertEqual((length, etag), (3, '"fixed-v1"'))

    def test_ignored_range_fails_before_body(self):
        response = Response(b"0123456789", 200)
        client = t.RangeClient(opener=QueueOpener([response]))
        with self.assertRaisesRegex(t.TransportError, "unexpected_http_status:200"):
            client.get("https://example.org/chunk", object_size=10, byte_range=(2, 4), etag='"fixed-v1"')
        self.assertEqual(response.read_calls, 0)
        self.assertEqual(client.body_bytes_read, 0)
        self.assertTrue(response.closed)

    def test_missing_or_forbidden_is_never_fill(self):
        for status in (403, 404, 412, 500):
            response = Response(b"error", status)
            client = t.RangeClient(opener=QueueOpener([response]))
            with self.assertRaisesRegex(t.TransportError, "unexpected_http_status"):
                client.get("https://example.org/chunk", object_size=10)
            self.assertEqual(response.read_calls, 0)
            self.assertEqual(len(client.requests), 1)

    def test_range_bounds_and_total_must_match(self):
        for span, total in (((1, 3), 10), ((2, 4), 11), ((2, 3), 10)):
            fields = headers(3, span=span, total=total)
            with self.assertRaisesRegex(t.TransportError, "content_range_mismatch"):
                t.validate_headers(206, fields, object_size=10, byte_range=(2, 4), etag='"fixed-v1"')
        fields = headers(3)
        with self.assertRaisesRegex(t.TransportError, "content_range_mismatch"):
            t.validate_headers(206, fields, object_size=10, byte_range=(2, 4))

    def test_encoding_and_transfer_transforms_rejected(self):
        for key, value in (("Content-Encoding", "gzip"), ("Transfer-Encoding", "chunked")):
            fields = headers(3, span=(2, 4), total=10, **{key: value})
            with self.assertRaisesRegex(t.TransportError, "encoding_not_identity"):
                t.validate_headers(206, fields, object_size=10, byte_range=(2, 4))

    def test_weak_missing_or_changed_etag_rejected(self):
        for value in ('W/"fixed-v1"', "", "unquoted"):
            with self.assertRaisesRegex(t.TransportError, "missing_or_weak_etag"):
                t.validate_headers(200, headers(10, etag=value), object_size=10)
        fields = headers(10); del fields["ETag"]
        with self.assertRaisesRegex(t.TransportError, "missing_or_weak_etag"):
            t.validate_headers(200, fields, object_size=10)
        with self.assertRaisesRegex(t.TransportError, "etag_changed"):
            t.validate_headers(206, headers(3, span=(2, 4), total=10, etag='"changed"'),
                               object_size=10, byte_range=(2, 4), etag='"fixed-v1"')

    def test_duplicate_or_wrong_length_rejected_before_body(self):
        fields = headers(10); fields["Content-Length"] = "10"
        with self.assertRaisesRegex(t.TransportError, "duplicate_response_header"):
            t.validate_headers(200, fields, object_size=10)
        with self.assertRaisesRegex(t.TransportError, "content_length_mismatch"):
            t.validate_headers(200, headers(9), object_size=10)
        fields = headers(10); del fields["Content-Length"]
        with self.assertRaisesRegex(t.TransportError, "missing_or_invalid"):
            t.validate_headers(200, fields, object_size=10)

    def test_exposed_short_and_long_bodies_counted_and_rejected(self):
        # This fake exposes all bytes. Real HTTP framing can hide trailing bytes;
        # the separate HTTPResponse regression below preserves that limitation.
        for body in (b"012", b"01234567890"):
            response = Response(body, 200, headers(10))
            client = t.RangeClient(opener=QueueOpener([response]))
            with self.assertRaisesRegex(t.TransportError, "response_body_length_mismatch"):
                client.get("https://example.org/chunk", object_size=10)
            self.assertEqual(client.body_bytes_read, len(body))

    def test_real_http_framing_hides_trailing_wire_bytes(self):
        # No socket is opened: feed a real HTTPResponse parser generated bytes.
        class FakeSocket:
            def makefile(self, mode):
                return io.BytesIO(
                    b'HTTP/1.1 200 OK\r\nContent-Length: 10\r\n'
                    b'ETag: "fixed-v1"\r\n\r\n0123456789X'
                )

        response = http.client.HTTPResponse(FakeSocket())
        response.begin()
        client = t.RangeClient(opener=QueueOpener([response]))
        body, _, _ = client.get("https://example.org/chunk", object_size=10)
        self.assertEqual(body, b"0123456789")
        self.assertEqual(client.body_bytes_read, 10)

    def test_body_and_request_caps(self):
        response = Response(b"0123456789")
        client = t.RangeClient(body_cap=10, opener=QueueOpener([response]))
        with self.assertRaisesRegex(t.TransportError, "body_cap"):
            client.get("https://example.org/chunk", object_size=10)
        self.assertEqual(response.read_calls, 0)
        client = t.RangeClient(request_cap=0, opener=QueueOpener([]))
        with self.assertRaisesRegex(t.TransportError, "request_cap"):
            client.get("https://example.org/chunk", object_size=10)

    def test_redirects_counted_without_recording_signed_query(self):
        redirect_headers = Message(); redirect_headers["Location"] = "https://cdn.example.org/object?secret=DO_NOT_RECORD"
        responses = [Response(status=302, fields=redirect_headers), Response(b"0123456789")]
        client = t.RangeClient(opener=QueueOpener(responses))
        body, etag, final = client.get("https://example.org/chunk", object_size=10)
        self.assertEqual(body, b"0123456789")
        self.assertIn("secret=", final)  # Remains in process memory only.
        self.assertNotIn("DO_NOT_RECORD", json.dumps(client.requests))
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(client.body_bytes_read, 10)

    def test_https_only_and_redirect_limit(self):
        for url in ("http://example.org/a", "https://person:secret@example.org/a"):
            with self.assertRaisesRegex(t.TransportError, "nonpublic_https"):
                t.RangeClient(opener=QueueOpener([])).get(url, object_size=10)
        fields = Message(); fields["Location"] = "https://example.org/again"
        client = t.RangeClient(opener=QueueOpener([Response(status=302, fields=fields) for _ in range(4)]))
        with self.assertRaisesRegex(t.TransportError, "redirect_cap"):
            client.get("https://example.org/chunk", object_size=10)
        self.assertEqual(len(client.requests), 4)

    def test_if_match_and_identity_are_actually_sent(self):
        opener = QueueOpener([Response(b"234", 206, headers(3, span=(2, 4), total=10))])
        client = t.RangeClient(opener=opener)
        client.get("https://example.org/chunk", object_size=10, byte_range=(2, 4), etag='"fixed-v1"')
        sent = {k.lower(): v for k, v in opener.requests[0].header_items()}
        self.assertEqual(sent["range"], "bytes=2-4")
        self.assertEqual(sent["if-match"], '"fixed-v1"')
        self.assertEqual(sent["accept-encoding"], "identity")

    def test_generated_full_runner_six_requests(self):
        shape = [65, 128, 128]
        a = (np.arange(np.prod(shape)) % 251).astype(np.uint8).reshape(shape)
        raw = a.tobytes()
        protocol = json.loads((HERE / "frozen_protocol.json").read_text())
        protocol["source"].update(url="https://example.org/chunk", bytes=len(raw), sha256=t.digest(raw))
        responses = [Response(raw)]
        for case in protocol["cases"]:
            z, y, x, Z, Y, X = case["box_zyx"]
            case["expected_sha256"] = t.digest(a[z:Z, y:Y, x:X].tobytes())
            for lo, hi in t.plan_box(case["box_zyx"], shape):
                responses.append(Response(raw[lo:hi+1], 206, headers(hi-lo+1, span=(lo, hi), total=len(raw))))
        client = t.RangeClient(opener=QueueOpener(responses))
        result = t.run(protocol, raw, client)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(len(client.requests), 6)
        self.assertEqual(client.body_bytes_read, len(raw)+32808)
        self.assertFalse(result["s3_or_paris4_support_established"])

    def test_changed_whole_stops_before_ranges(self):
        protocol = json.loads((HERE / "frozen_protocol.json").read_text())
        raw = bytes(protocol["source"]["bytes"])
        protocol["source"]["sha256"] = t.digest(raw)
        for case in protocol["cases"]:
            case["expected_sha256"] = t.digest(bytes(case["requested_bytes"]))
        client = t.RangeClient(opener=QueueOpener([Response(b"\1" + raw[1:])]))
        with self.assertRaisesRegex(t.TransportError, "whole_object_no_longer"):
            t.run(protocol, raw, client)
        self.assertEqual(len(client.requests), 1)


if __name__ == "__main__":
    unittest.main()
