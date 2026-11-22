#!/usr/bin/env python3
"""Generated local Zarr -> patched renderer -> Studio CPU-context preflight.

No network, torch, weights, scroll images or model forwards. Detailed temporary
arrays stay outside Git. The final result contains generated-fixture counts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_store(root):
    """Uncompressed, explicit uint8 Zarr v2 chunks with no implicit fill."""
    array_dir = root / "0"
    array_dir.mkdir(parents=True)
    (root / ".zgroup").write_text('{"zarr_format": 2}\n')
    shape, chunks = (48, 256, 256), (16, 64, 64)
    (array_dir / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "shape": shape, "chunks": chunks, "dtype": "|u1",
        "compressor": None, "filters": None, "fill_value": 0,
        "order": "C", "dimension_separator": "/",
    }) + "\n")
    z, y, x = np.indices(shape, dtype=np.int16)
    volume = (40 + (3 * z + y // 8 + x // 8) % 180).astype(np.uint8)
    for iz in range(3):
        for iy in range(4):
            for ix in range(4):
                path = array_dir / str(iz) / str(iy) / str(ix)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(volume[iz * 16:(iz + 1) * 16,
                                        iy * 64:(iy + 1) * 64,
                                        ix * 64:(ix + 1) * 64].tobytes())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--renderer", type=Path, required=True,
                   help="Patched gemini_lane/stream_snap_render.py")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    root = a.output_dir.resolve()
    if any(os.path.lexists(parent / ".git") for parent in (root, *root.parents)):
        raise ValueError("Generated raw outputs must stay outside every Git worktree")
    if root.exists():
        raise FileExistsError("Use a fresh output directory")
    root.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("integration_renderer", a.renderer)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    y, x = np.indices((256, 256), dtype=np.float32)
    renderer.write_tifxyz(root / "mesh", x, y, np.full_like(x, 24), meta={"scale": 1.0})
    write_store(root / "present")
    shutil.copytree(root / "present", root / "missing_context")
    # This missing chunk is outside the evaluated central pixels but within a
    # positively contributing tile's full input context.
    (root / "missing_context/0/1/0/0").unlink()
    preflight = HERE.parent / "producer_support/prepare_studio.py"
    cases = {}
    for case in ("present", "missing_context"):
        out = root / (case + "_render")
        out.mkdir()
        command = [sys.executable, str(a.renderer), "--mesh-dir", str(root / "mesh"),
                   "--ct-zarr", str(root / case), "--snap-passes", "0",
                   "--num-slices", "31", "--uv-stride", "1", "--workers", "1",
                   "--cache-mb", "2", "--out-npy", str(out / "stack.npy"),
                   "--out-support-npz", str(out / "support.npz"),
                   "--out-receipt", str(out / "render.json")]
        subprocess.run(command, check=True, capture_output=True, text=True)
        subprocess.run([sys.executable, str(preflight), "--input", str(out / "stack.npy"),
                        "--support-npz", str(out / "support.npz"), "--mesh-index", "15",
                        "--order", "desc", "--stride", "48",
                        "--output-dir", str(out / "context")],
                       check=True, capture_output=True, text=True)
        record = json.loads((out / "context/preflight.json").read_text())
        cases[case] = {
            "counts": record["counts"], "handoff_ready": record["handoff_ready"],
            "model_forward_calls": record["plan"]["model_forward_calls"],
            "array_support_semantics": "stored source availability, not physical acquisition",
        }
        support = np.load(out / "context/source_context_support.npy", allow_pickle=False)
        expected = np.zeros((256, 256), dtype=bool)
        expected[96:160, 96:160] = True
        if case == "missing_context":
            expected[96:144, 96:144] = False
        np.testing.assert_array_equal(support, expected)
        # The central patch remains fully sampled; the loss is only from the
        # missing external context of one positively contributing tile.
        with np.load(out / "support.npz", allow_pickle=False) as masks:
            assert masks["source_available_lhw"][:, 96:160, 96:160].all()
        assert record["handoff_ready"] is False
    assert cases["present"]["counts"]["context_supported_pixels"] == 4096
    assert cases["missing_context"]["counts"]["context_supported_pixels"] == 1792
    result = {
        "schema": "scrollprize.synthetic-stream-studio-integration.v1", "status": "pass",
        "generated_data_only": True, "network_requests": 0, "model_forward_calls": 0,
        "renderer_sha256": sha(a.renderer), "integration_code_sha256": sha(__file__),
        "preflight_code_sha256": sha(preflight),
        "support_code_sha256": sha(HERE.parent / "producer_support/support.py"),
        "studio_code_sha256": sha(HERE.parent / "studio_t4/studio_t4.py"),
        "fixtures": {"source_shape_zyx": [48, 256, 256], "chunks_zyx": [16, 64, 64],
                     "stack_shape_lhw": [31, 256, 256], "mesh_plane_z": 24,
                     "removed_chunk_zyx": [1, 0, 0], "center_box_yxyx": [96, 96, 160, 160]},
        "cases": cases,
        "finding": "A fully sampled central patch can lose model-context support because one contributing tile sees a missing chunk outside that patch.",
        "limits": "No actual inference, real scroll, physical acquisition, correct face, unique area, or ink improvement is certified.",
    }
    (root / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
