"""Small generated/geometry-only checks. Never acquire or decode real CT."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("followup", HERE / "frag5_native_grid_followup.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def original_path():
    # Test runner supplies the published stage1 directory; no private path is embedded.
    import os
    return Path(os.environ["FRAG5_ORIGINAL_DIR"])


def load():
    protocol = json.loads((HERE / "frag5_native_grid_followup_protocol.json").read_bytes())
    original, plan = M.load_original(original_path(), protocol)
    return protocol, original, plan


def test_fixed_corner_and_full_union_contract():
    protocol, original, plan = load()
    M.check_protocol(protocol, original, plan)
    assert len(protocol["corner_rows"]) == 16
    assert sum(b-a+1 for r in protocol["corner_rows"] for a,b in [r["byte_range"]]) == 64
    assert len({tuple(x) for row in protocol["corner_rows"] for x in row["xyz"]}) == 32
    protocol["all_source_expected_heads"].pop(next(iter(protocol["all_source_expected_heads"])))
    with pytest.raises(ValueError, match="fourteen"):
        M.check_protocol(protocol, original, plan)


def test_exact_32_gate_rejects_one_difference_or_missing_value():
    rows = [{"native_uint16":1000+i, "zarr_uint16":1000+i} for i in range(32)]
    assert M.all_corners_equal(rows)
    assert not M.all_corners_equal(rows[:-1])
    rows[17]["native_uint16"] += 1
    assert not M.all_corners_equal(rows)
    assert not M.all_corners_equal([])


def test_native_center_diagnostic_analytic_field_and_gate_independence():
    protocol, original, plan = load()
    field = lambda p: 1000 + np.asarray(p) @ np.array([2,3,5])
    rows = []
    for r in protocol["corner_rows"]:
        for xyz in r["xyz"]:
            value = int(field(xyz))
            rows.append({"center_id":r["center_id"], "xyz":xyz, "native_uint16":value, "zarr_uint16":value})
    old = []
    errors = [1530,236,0,904]
    for p, error in zip(plan["centers"], errors):
        expected = float(field(p["fine_xyz"]))
        old.append({"id":p["id"], "tiff_values":[int(np.rint(expected))+error]*65,
                    "fine_trilinear_values":[expected]*65, "primary_absolute_errors":[error]*65})
    report = M.native_center_diagnostic(original, plan, rows, {"profiles":old})
    assert report["affects_gate"] is False
    assert [r["native_vs_previous_surface_rounded_absolute_error"] for r in report["centers"]] == errors
    for result, p in zip(report["centers"], plan["centers"]):
        assert result["native_corner_trilinear"] == pytest.approx(field(p["fine_xyz"]), abs=1e-10)
    for row in rows:
        row["native_uint16"] += 100
    assert not M.all_corners_equal(rows)
    failed_gate_report = M.native_center_diagnostic(original, plan, rows, {"profiles":old})
    assert len(failed_gate_report["centers"]) == 4
    assert failed_gate_report["affects_gate"] is False
