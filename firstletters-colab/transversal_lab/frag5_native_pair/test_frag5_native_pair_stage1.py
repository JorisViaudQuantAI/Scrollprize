"""Generated-input checks only: no remote requests, CT or photograph pixels."""
import importlib.util
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("stage1", HERE / "frag5_native_pair_stage1.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_cartesian_chunk_corners_and_bounds():
    assert len(M.box_chunks([127,127,127], [128,128,128], [256,256,256])) == 8
    assert len(M.box_chunks([127,127,10], [128,128,10], [256,256,256])) == 4
    assert M.box_chunks([255,255,255], [255,255,255], [256,256,256]) == [(1,1,1)]
    with pytest.raises(ValueError):
        M.box_chunks([255,255,255], [256,255,255], [256,256,256])


def test_trilinear_xyz_axis_weights_and_exact_terminal_integer():
    z,y,x = np.indices((9,8,7))
    values = 1000 + 2*x + 7*y + 13*z
    region = M.Region(values, [100,200,300])
    local = np.array([[.2,.7,.9], [6,7,8], [0,0,0], [2.5,3.25,4.75]])
    observed = region.sample(local + [100,200,300])
    np.testing.assert_allclose(observed, 1000+local @ np.array([2,7,13]), rtol=0, atol=1e-11)
    with pytest.raises(ValueError, match="support"):
        region.sample([[106.0001,207,308]])


def test_inverse_affine_midpoint_average_preserves_linear_field():
    z,y,x = np.indices((40,40,40))
    region = M.Region(1000 + 2*x + 7*y + 13*z, [0,0,0])
    A = np.array([[.41,.002,0],[.001,.42,0],[0,.001,.415]])
    t = np.array([2,3,4])
    q = np.array([[8.,9.,10.],[9,11,12]])
    inv = np.linalg.inv(A)
    expected = region.sample((q-t) @ inv.T)
    u = (np.arange(5)+.5)/5-.5
    actual = sum(region.sample((q+offset-t) @ inv.T) for offset in itertools.product(u, repeat=3))/125
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_laplacian_uses_only_valid_interior_and_constant_metrics_unassessable():
    z,y,x = np.indices((7,7,7))
    np.testing.assert_array_equal(M.laplacian(x*x + y*y + z*z), np.full((5,5,5),6))
    assert M.correlation(np.zeros(10), np.arange(10))["status"] == "constant_unassessable"
    assert M.correlation(np.arange(20), np.arange(20)[::-1])["spearman"] == pytest.approx(-1)


def test_unexpected_200_range_body_is_not_read(tmp_path):
    (tmp_path/"objects").mkdir()
    reader=M.Acquisition(tmp_path,{"cpu_seconds":3600,"tiff_metadata_body_bytes":2097152},M.time.process_time())
    url="https://dl.ash2txt.org/generated-only.tif"
    reader.heads[url]={"length":100,"etag":"fixed"}
    class Response:
        status=200
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def geturl(self): return url
        def read(self,*args): raise AssertionError("Unexpected200bodymustnotberead")
    with patch.object(M,"urlopen",return_value=Response()):
        with pytest.raises(ValueError,match="status"):
            reader.fetch(url,"tiff_metadata",(8,9))
    assert reader.receipts[0]["body_bytes_read"] == 0
    assert reader.receipts[0]["status"] == "failed"


def test_frozen_geometry_reproduces_and_rejects_plan_tampering():
    plan=json.loads((HERE/"frag5_native_pair_stage1_protocol.json").read_text())
    result=M.check_plan(plan)
    assert result["decoded_chunks"] == 14
    assert len(result["fine_chunk_keys_zyx"]) == 8
    assert len(result["coarse_chunk_keys_zyx"]) == 6
    assert len(result["identity_fine_chunk_keys_zyx"]) == 6
    plan["support_plan"]["fine_chunk_keys_zyx"].pop()
    with pytest.raises(ValueError,match="reproduce"):
        M.check_plan(plan)


def test_ambiguous_identity_and_zero_profiles_never_allow_anatomy():
    def row(primary, reverse, informative=True):
        return {"nonconstant_nonzero":informative,"primary_max_abs_error_uint16":primary,
                "reverse_max_abs_error_uint16":reverse}
    assert M.classify_identity([row(0,0)]*4)["classification"] == "both_ambiguous"
    assert not M.classify_identity([row(0,0)]*4)["anatomy_allowed"]
    assert M.classify_identity([row(0,100)]*4)["anatomy_allowed"]
    assert M.classify_identity([row(100,0)]*4)["classification"] == "reverse_only"
    assert not M.classify_identity([row(0,100,False)]*4)["anatomy_allowed"]
    assert not M.classify_identity([row(0,100)]*3+[row(100,0)])["anatomy_allowed"]


def test_changed_ifd_pointer_rejected_before_following_it():
    class Reader:
        calls=[]
        def fetch(self,url,category,byte_range):
            self.calls.append(byte_range)
            return b"II*\x00"+M.struct.pack("<I",8)
    reader=Reader()
    with pytest.raises(ValueError,match="pointer"):
        M.check_tiff_header(reader,{"surface_base_url":"https://dl.ash2txt.org/generated"},0)
    assert reader.calls == [(0,7)]
