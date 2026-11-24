"""Receipt/cache adapter around the existing Gemini supported Zarr transport.

Preserves the author's FP32-coordinate, uint8-source trilinear arithmetic.
This public-control adapter persists hash-checked chunks for local replay; it
does not change Gemini's production zero-disk reader or its support semantics.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading

import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gemini_lane"))
from stream_snap_render import RemoteZarrV2, _positive_weight_stencil_support


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_surface(source_repo, mesh_dir, recipe):
    path = Path(source_repo)/"donor-labels-20260906/expansion/volume_sampling.py"
    spec = importlib.util.spec_from_file_location("q33_author_surface", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    surface = mod.Surface(mesh_dir, geometry_um=9.612, ct_um=9.366,
                          matrix=recipe["matrix"], normal_sign=1,
                          geometry_shape=[3781,7618,7618])
    return surface


class SupportedNative(RemoteZarrV2):
    def __init__(self, url, cache, *, byte_budget=512*1024*1024):
        self.body_cache = Path(cache)
        self.body_cache.mkdir(parents=True, exist_ok=True)
        source_file = self.body_cache / "source_url.txt"
        source_url = url.rstrip("/") + "/"
        if source_file.exists() and source_file.read_text().strip() != source_url:
            raise ValueError("Native cache belongs to another source URL")
        source_file.write_text(source_url + "\n")
        self.receipt_lock = threading.Lock()
        self.new_http_bytes = 0
        self.byte_budget = int(byte_budget)
        manifest = self.body_cache/"object_receipts.json"
        self.object_receipts = json.loads(manifest.read_text()) if manifest.exists() else {}
        self.requested = set()
        super().__init__(url, level=None, cache_mb=512, workers=6, timeout=60, max_retries=3)
        if self.dtype != np.dtype("uint8") or self.fill_value != 0:
            raise ValueError("Unexpected native CT dtype/fill")

    def _get_bytes(self, key, *, allow_404=True):
        parts = key.split("/")
        if key.startswith("/") or ".." in parts:
            raise ValueError("Unsafe source object key")
        file = self.body_cache.joinpath(*parts)
        with self.receipt_lock:
            old = self.object_receipts.get(key)
        if file.exists():
            raw = file.read_bytes()
            if old is None or digest(raw) != old["sha256"] or len(raw) != old["bytes"]:
                raise ValueError("Cached source body changed")
            with self.receipt_lock:
                self.requested.add(key)
            return raw
        raw = super()._get_bytes(key, allow_404=allow_404)
        if raw is None:
            with self.receipt_lock:
                self.requested.add(key)
                self.object_receipts[key] = dict(status="missing", bytes=0, sha256=None)
            return None
        entry = dict(status="present", bytes=len(raw), sha256=digest(raw))
        with self.receipt_lock:
            self.new_http_bytes += len(raw)
            if self.new_http_bytes > self.byte_budget:
                raise ValueError("Native HTTP byte budget exceeded")
            if old is not None and old != entry:
                raise ValueError("Previously received source object changed")
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(raw)
            self.object_receipts[key] = entry
            self.requested.add(key)
            (self.body_cache/"object_receipts.json").write_text(json.dumps(self.object_receipts, indent=2)+"\n")
        return raw

    def sample_exact(self, coords_xyz):
        """Return author's FP32 trilinear samples, requiring every needed corner."""
        shape = np.asarray(coords_xyz).shape[:-1]
        points = np.asarray(coords_xyz, dtype=np.float32).reshape(-1,3)[:,::-1]
        if (not np.isfinite(points).all() or np.any(points < 0)
                or np.any(points > np.asarray(self.shape)-1)):
            raise ValueError("Sample outside native volume")
        lo = np.floor(points.min(axis=0)).astype(int)
        hi = np.ceil(points.max(axis=0)).astype(int)+1
        if np.prod(hi-lo) > 32_000_000:
            raise ValueError("A patch's native bounding box exceeds the frozen budget")
        block, origin, available = self.read_box(lo, hi, return_support=True)
        local = (points-np.asarray(origin,dtype=np.float32)).T
        support = _positive_weight_stencil_support(available, local)
        if not support.all():
            raise ValueError(f"Native CT stencil missing for {int((~support).sum())} samples")
        values = map_coordinates(block, local, output=np.float32, order=1, mode="nearest", prefilter=False)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite sampled CT")
        return values.reshape(shape)

    def sample(self, coords_xyz):
        return self.sample_exact(coords_xyz)

    def receipt(self):
        return dict(url=self.root+"/", shape=list(self.shape), chunks=list(self.chunks),
                    dtype=str(self.dtype), new_http_bytes=self.new_http_bytes,
                    requested_objects={k:self.object_receipts[k] for k in sorted(self.requested)},
                    requested_body_bytes=sum(self.object_receipts[k]["bytes"] for k in self.requested),
                    byte_budget_semantics="Post-response stop threshold; concurrent in-flight objects can exceed it; not an exact transport byte cap",
                    source_support="Every positive-weight corner comes from a present decoded object; stored masked zeros remain valid values, not proof of original acquisition",
                    consistency="Full received-body SHA256, no atomic whole-volume snapshot guarantee")
