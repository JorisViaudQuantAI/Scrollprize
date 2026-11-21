#!/usr/bin/env python3
"""infer_ink.py - CPU-friendly ink detection on rendered Herculaneum segments.

Runs the official Vesuvius Challenge cross-scroll ink models on a *surface
volume* (a stack of layers resampled around the papyrus surface) and writes an
ink-probability image.

Supported models (auto-detected from the checkpoint)
  * scrollprize/ink_9um  "hybrid_3d2d"  (default; 17 x 128 x 128 patches,
    ~9.4-9.6 um isotropic voxels, robust median/MAD normalisation per patch)
  * scrollprize/hecate   "hecate_9.6um" (16 x 64 x 64 patches, 9.6 um isotropic,
    intensity / 255) - needs hecate.py from the model repo, see README.md

Supported inputs
  * <dir>/layers/NN.tif   classic layout (00.tif = first layer along the surface
                          normal, middle layer = the traced surface); optional
                          <dir>/mask.png and <dir>/render_meta.json | meta.json
  * a directory that directly contains NN.tif / NN.png / NN.jpg
  * a Zarr array or OME-Zarr group, local path or http(s) URL (Z, Y, X order)
  * a .npy file or a multi-page TIFF holding a (Z, Y, X) stack

Only the layers and the XY region that are needed are read, so remote
surface volumes can be sampled with --roi without downloading them.

See README.md next to this file for the exact conventions (layer count and
ordering, voxel size, resampling of 7.91 / 8.64 / 9.362 um data).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"

LAYER_RE = re.compile(r"^(\d+)\.(tif|tiff|png|jpg|jpeg)$", re.IGNORECASE)


def log(*args):
    print(*args, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Input sources
# --------------------------------------------------------------------------
class Source:
    """A (Z, Y, X) uint8 surface volume that can be read region by region."""

    depth: int
    height: int
    width: int
    um_xy: float | None = None
    um_z: float | None = None
    description: str = ""

    def read(self, z_indices, y0, y1, x0, x1) -> np.ndarray:  # (len(z), y1-y0, x1-x0) uint8
        raise NotImplementedError


def _to_uint8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.uint16:
        return (a >> 8).astype(np.uint8)
    if np.issubdtype(a.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(a)) <= 1.0 else 1.0
        return np.clip(a * scale, 0, 255).astype(np.uint8)
    raise TypeError(f"unsupported layer dtype {a.dtype}")


class LayersSource(Source):
    """Numbered single-layer images, sorted by their integer file stem."""

    def __init__(self, layers_dir: Path, cache_bytes: int = 1_200_000_000):
        import tifffile

        self._tifffile = tifffile
        files = []
        for p in layers_dir.iterdir():
            m = LAYER_RE.match(p.name)
            if m:
                files.append((int(m.group(1)), p))
        if not files:
            raise FileNotFoundError(f"no numbered layer images (NN.tif) in {layers_dir}")
        files.sort()
        self.files = [p for _, p in files]
        numbers = [n for n, _ in files]
        if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            raise ValueError(f"layer numbering in {layers_dir} has gaps: {numbers[:5]}...{numbers[-5:]}")
        first = self._load(0)
        if first.ndim != 2:
            raise ValueError(f"layer images must be single-channel 2D, got shape {first.shape}")
        self.depth, (self.height, self.width) = len(self.files), first.shape
        self._cache: dict[int, np.ndarray] = {}
        self._cache_bytes = cache_bytes
        self.description = f"layers dir {layers_dir} ({self.depth} layers, {first.dtype})"

    def _load(self, z: int) -> np.ndarray:
        p = self.files[z]
        if p.suffix.lower() in (".tif", ".tiff"):
            try:
                return self._tifffile.memmap(p, mode="r")
            except Exception:
                return self._tifffile.imread(p)
        import cv2

        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise IOError(f"cannot read {p}")
        return a

    def _layer(self, z: int) -> np.ndarray:
        if z in self._cache:
            return self._cache[z]
        a = self._load(z)
        if not isinstance(a, np.memmap):
            if (len(self._cache) + 1) * a.nbytes > self._cache_bytes:
                self._cache.clear()
            self._cache[z] = a
        return a

    def read(self, z_indices, y0, y1, x0, x1):
        out = np.empty((len(z_indices), y1 - y0, x1 - x0), np.uint8)
        for i, z in enumerate(z_indices):
            out[i] = _to_uint8(np.asarray(self._layer(int(z))[y0:y1, x0:x1]))
        return out


class ArraySource(Source):
    """Anything indexable as array[z, y, x] (zarr array, numpy memmap)."""

    def __init__(self, array, description=""):
        if len(array.shape) != 3:
            raise ValueError(f"expected a 3D (Z, Y, X) array, got shape {tuple(array.shape)}")
        if int(np.argmin(array.shape)) != 0:
            raise ValueError(
                f"expected Z (the short axis) first, got shape {tuple(array.shape)}; "
                "transpose the stack to (Z, Y, X)"
            )
        self.array = array
        self.depth, self.height, self.width = (int(v) for v in array.shape)
        self.description = description

    def read(self, z_indices, y0, y1, x0, x1):
        z_indices = [int(z) for z in z_indices]
        z_lo, z_hi = min(z_indices), max(z_indices) + 1
        block = np.asarray(self.array[z_lo:z_hi, y0:y1, x0:x1])
        return _to_uint8(block[[z - z_lo for z in z_indices]])


class BinnedSource(Source):
    """Integer XY mean-binning of another source (e.g. 2.4 um layers -> 9.6 um with factor 4)."""

    def __init__(self, base: Source, factor: int):
        self.base, self.f = base, int(factor)
        self.depth, self.height, self.width = base.depth, base.height // self.f, base.width // self.f
        self.um_xy = None if base.um_xy is None else base.um_xy * self.f
        self.um_z = base.um_z
        self.description = f"{base.description}; XY mean-binned x{self.f}"

    def read(self, z_indices, y0, y1, x0, x1):
        f = self.f
        out = np.empty((len(z_indices), y1 - y0, x1 - x0), np.uint8)
        step = max(1, 96 * 1024 * 1024 // max(1, len(z_indices) * (x1 - x0) * f * f))  # ~96 MB of source per strip
        for r in range(y0, y1, step):
            r1 = min(y1, r + step)
            block = self.base.read(z_indices, r * f, r1 * f, x0 * f, x1 * f)
            block = block.reshape(len(z_indices), r1 - r, f, x1 - x0, f).mean(axis=(2, 4), dtype=np.float32)
            out[:, r - y0:r1 - y0] = np.rint(block).astype(np.uint8)
        return out


def open_zarr_source(path: str, level: str) -> ArraySource:
    import zarr

    is_url = "://" in path
    if is_url:
        from zarr.storage import FSStore

        # Only "key missing" may be turned into fill_value; network errors must raise.
        store = FSStore(path.rstrip("/"), mode="r", exceptions=(KeyError, FileNotFoundError))
        root = zarr.open(store, mode="r")
    else:
        root = zarr.open(path, mode="r")
    um_xy = um_z = None
    if hasattr(root, "shape"):
        array, used = root, "<array>"
    else:
        if level not in root:
            raise KeyError(f"level {level!r} not in {path}; available: {sorted(root.array_keys())}")
        array, used = root[level], level
        try:
            for ds in root.attrs["multiscales"][0]["datasets"]:
                if str(ds["path"]) == str(level):
                    for tr in ds["coordinateTransformations"]:
                        if tr["type"] == "scale":
                            um_z, um_y, um_x = (float(v) for v in tr["scale"][-3:])
                            um_xy = 0.5 * (um_y + um_x)
        except Exception:
            pass
    src = ArraySource(array, f"zarr {path} level {used} shape {tuple(array.shape)} dtype {array.dtype}")
    src.um_xy, src.um_z = um_xy, um_z
    return src


def open_source(path: str, level: str) -> tuple[Source, Path | None]:
    """Return the source and (for directories) the folder holding mask/meta files."""
    if "://" in path or path.rstrip("/").endswith(".zarr"):
        return open_zarr_source(path, level), None
    p = Path(path)
    if p.is_dir():
        if (p / ".zarray").exists() or (p / ".zgroup").exists():
            return open_zarr_source(path, level), None
        layers_dir = p / "layers" if (p / "layers").is_dir() else p
        seg_dir = p if layers_dir != p else p.parent if p.name == "layers" else p
        src = LayersSource(layers_dir)
        for name in ("render_meta.json", "meta.json"):
            meta_path = seg_dir / name
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                xy = meta.get("pixel_size_um") or meta.get("voxel_size_um")
                z = meta.get("layer_spacing_um")
                if z is None and meta.get("voxel_size_um") is not None:
                    z = float(meta["voxel_size_um"]) * float(meta.get("step_voxels", 1.0))
                src.um_xy = float(xy) if xy else None
                src.um_z = float(z) if z else src.um_xy
                src.description += f"; voxel size from {name}"
                break
        return src, seg_dir
    if p.suffix == ".npy":
        return ArraySource(np.load(p, mmap_mode="r"), f"npy {p}"), None
    if p.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        return ArraySource(tifffile.imread(p), f"tiff stack {p}"), None
    raise FileNotFoundError(f"do not know how to open {path}")


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class Ink9umModel:
    """scrollprize/ink_9um hybrid 3D-stem / 2D U-Net (villa vesuvius.ink_detection)."""

    name = "ink_9um"
    nominal_um = 9.6  # trained on 9.596 um (pooled 2.4 um) and native 9.362 um renders
    native_range_um = (8.5, 10.5)  # run without resampling inside this range
    display_rescale = (0.25, 0.75)  # BCE label smoothing 0.5 -> background ~0.25, ink ~0.75

    def __init__(self, checkpoint: dict, path: str, device: str = "cpu", amp: bool = False):
        import torch

        self.device, self.amp = device, amp
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        from vesuvius.ink_detection.config import InkConfig
        from vesuvius.ink_detection.models.model import make_model

        cfg = checkpoint["config"]
        if cfg.get("mode") != "flat":
            raise ValueError(f"{path}: only flat (surface-volume) checkpoints are supported, got mode={cfg.get('mode')}")
        norm = cfg["image_normalization"]
        if norm["mode"] == "robust_mad":
            if (norm.get("percentile_lower", 1.0), norm.get("percentile_upper", 99.0)) != (1.0, 99.0):
                raise ValueError("robust_mad normalisation is only supported with percentiles 1/99")
            self.normalization = "robust_mad"
        elif norm["mode"] == "divide" and float(norm.get("divisor", 255.0)) == 255.0:
            self.normalization = "divide_255"
        else:
            raise ValueError(f"unsupported image_normalization {norm}")
        self.depth, self.patch, patch_w = (int(v) for v in cfg["crop_size"])
        assert self.patch == patch_w
        with contextlib.redirect_stdout(io.StringIO()):  # the builder prints its whole config
            model = make_model(InkConfig.from_mapping(cfg))
        state = checkpoint.get("ema_model") or checkpoint["model"]
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        self.model = model.eval().to(device)
        self.torch = torch
        self.info = {
            "model": self.name,
            "device": device,
            "amp_fp16": bool(amp),
            "model_type": cfg.get("model_type"),
            "checkpoint": str(path),
            "step": int(checkpoint.get("step", -1)),
            "seed": cfg.get("seed"),
            "patch_zyx": [self.depth, self.patch, self.patch],
            "normalization": self.normalization,
        }

    def normalize(self, patch_zyx: np.ndarray) -> np.ndarray:
        """Per-patch normalisation, identical to vc_reader Normalize('robust', 1, 99)."""
        x = patch_zyx.astype(np.float32, copy=True)
        if self.normalization == "divide_255":
            return x / 255.0
        flat = x.reshape(-1).astype(np.float64)
        deviation = float(flat.std())
        low, high = np.percentile(flat, [1.0, 99.0])
        np.clip(flat, low, high, out=flat)
        center = float(np.median(flat))
        scale = 1.4826 * float(np.median(np.abs(flat - center)))
        if not (np.isfinite(scale) and scale >= 1e-6):
            if np.isfinite(deviation) and abs(deviation) >= 1e-6:
                scale = abs(deviation)
            elif np.isfinite(high - low) and abs(high - low) / 2 >= 1e-6:
                scale = abs(high - low) / 2
            else:
                scale = 1.0
        return ((flat - center) / scale).astype(np.float32).reshape(x.shape)

    def predict(self, batch_bzyx: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(batch_bzyx).unsqueeze(1).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            out = self.model(x)
        logits = out["ink"] if isinstance(out, dict) else out
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        prob = logits.float().sigmoid()
        if prob.shape[-2:] != x.shape[-2:]:
            prob = torch.nn.functional.interpolate(prob, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return prob[:, 0].cpu().numpy()


class HecateModel:
    """scrollprize/hecate (9.6 um checkpoint). Uses hecate.py shipped in the model repo."""

    name = "hecate"
    display_rescale = (0.0, 1.0)

    def __init__(self, path: str, hecate_py: str | None, device: str = "cpu", amp: bool = False, amp_dtype: str = "auto"):
        import importlib.util

        import torch

        self.device, self.amp = device, amp
        # bf16 autocast is only fast on sm_80+; on a T4 (sm_75) it is emulated and ~30x slower -> fp16 there.
        if amp_dtype == "auto":
            amp_dtype = "fp16"
            if device.startswith("cuda") and torch.cuda.get_device_capability(torch.device(device))[0] >= 8:
                amp_dtype = "bf16"
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

        candidates = [hecate_py] if hecate_py else []
        candidates += [str(Path(path).with_name("hecate.py")), str(HERE / "third_party" / "hecate" / "hecate.py")]
        module_path = next((c for c in candidates if c and Path(c).exists()), None)
        if module_path is None:
            raise FileNotFoundError("hecate.py not found; pass --hecate-py (download it from huggingface.co/scrollprize/hecate)")
        spec = importlib.util.spec_from_file_location("hecate_runtime", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["hecate_runtime"] = module
        spec.loader.exec_module(module)
        self.model = module.load_model(path, device).eval()
        self.depth, self.patch, patch_w = (int(v) for v in self.model.patch_size)
        self.divisor = float(self.model.divisor)
        spacing = float(self.model.sampling_um)
        self.nominal_um = spacing
        self.native_range_um = (spacing * 0.99, spacing * 1.01)  # the card requires the exact spacing
        self.torch = torch
        self.info = {
            "model": self.name,
            "device": device,
            "amp": (str(self.amp_dtype).replace("torch.", "") if amp else "off"),
            "checkpoint": str(path),
            "patch_zyx": [self.depth, self.patch, self.patch],
            "normalization": f"divide_{self.divisor:g}",
            "spacing_um": spacing,
        }

    def normalize(self, patch_zyx):
        return patch_zyx.astype(np.float32) / self.divisor

    def predict(self, batch_bzyx):
        torch = self.torch
        x = torch.from_numpy(batch_bzyx).unsqueeze(1).to(self.device)  # B,1,Z,Y,X already divided
        support = (x != 0).any(2, keepdim=True)  # B,1,1,Y,X: columns that contain CT data
        valid = support.expand(-1, 1, self.depth, -1, -1)
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp):  # card: bf16; fp16 on pre-Ampere GPUs
            logits = self.model(x, valid)
        prob = logits.float().sigmoid()[:, 0]
        return (prob * support[:, 0, 0]).cpu().numpy()  # empty CT columns give zero, as hecate.py


def load_model(path: str, args):
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device.startswith("cuda") and not args.no_amp
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    kind = args.model
    if kind == "auto":
        kind = "hecate" if "hecate" in Path(path).name.lower() else "ink9um"
    if kind == "hecate":
        return HecateModel(path, args.hecate_py, device, amp, args.hecate_amp)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not args.allow_unsafe_pickle:
            raise RuntimeError(
                f"{path} cannot be loaded with torch.load(weights_only=True): {exc}\n"
                "Only if the file comes from a source you trust, re-run with --allow-unsafe-pickle."
            ) from exc
        log("WARNING: falling back to full unpickling (--allow-unsafe-pickle)")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return Ink9umModel(checkpoint, path, device, amp)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def sliding_positions(length: int, patch: int, stride: int) -> list[int]:
    """Same schedule as villa: regular stride plus a final boundary-aligned tile."""
    if length <= patch:
        return [0]
    positions = list(range(0, length - patch + 1, max(1, stride)))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


def blend_window(patch: int, mode: str) -> np.ndarray:
    if mode == "constant":
        return np.ones((patch, patch), np.float32)
    n = np.arange(patch, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (patch - 1))  # symmetric Hann, as torch.hann_window(periodic=False)
    w = np.outer(hann, hann)
    w /= max(w.max(), 1e-12)
    return np.clip(w, 0.001, None).astype(np.float32)


def z_weights(depth_in: int, depth_out: int, center: float, ratio: float, shift: float) -> np.ndarray:
    """(depth_out, depth_in) matrix mapping source layers to model planes.

    ratio = model plane spacing / source layer spacing (in source layers).
    Output plane k sits at  center + (k - (depth_out-1)/2 + shift) * ratio.
    ratio <= 1.5: linear interpolation; otherwise box average over `ratio` layers
    (ratio == 4 reproduces the official 4-plane mean pooling of 2.4 um renders).
    Planes that fall outside the source stay zero.
    """
    w = np.zeros((depth_out, depth_in), np.float64)
    for k in range(depth_out):
        pos = center + (k - (depth_out - 1) / 2.0 + shift) * ratio
        if ratio <= 1.5:
            lo = math.floor(pos)
            frac = pos - lo
            for idx, wt in ((lo, 1.0 - frac), (lo + 1, frac)):
                if wt > 1e-9 and 0 <= idx < depth_in:
                    w[k, idx] += wt
        else:
            a, b = pos - ratio / 2.0, pos + ratio / 2.0
            for idx in range(max(0, math.floor(a + 0.5)), min(depth_in, math.ceil(b + 0.5))):
                overlap = min(b, idx + 0.5) - max(a, idx - 0.5)
                if overlap > 1e-9:
                    w[k, idx] += overlap / ratio
    return w


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def sweep(source: Source, model, args, roi, mask, voxel) -> tuple[dict, dict]:
    """Run one model over the ROI.

    Returns ({(direction, shift): probability on the *input* ROI grid}, info).
    """
    import cv2

    D, H, W = source.depth, source.height, source.width
    y0, x0, y1, x1 = roi
    um_xy, um_z, voxel_known = voxel
    P, DZ = model.patch, model.depth

    # ---- resampling decision
    lo, hi = model.native_range_um
    in_range = lo <= um_xy <= hi and lo <= um_z <= hi
    resample = (not in_range) if args.resample == "auto" else args.resample == "always"
    if not resample and not in_range:
        log(f"WARNING: voxel size xy={um_xy} z={um_z} um is outside {model.name}'s range {lo:g}-{hi:g} um but --resample never was given")
    target_um = args.target_um or model.nominal_um
    s_xy = um_xy / target_um if resample else 1.0  # model px per source px
    z_ratio = target_um / um_z if resample else 1.0  # source layers per model plane
    if resample and not (0.45 <= s_xy <= 2.2):
        raise ValueError(
            f"XY scale {s_xy:.3f} (source {um_xy} um -> {target_um} um) is outside 0.45-2.2; "
            "use a coarser pyramid level (--level) or pre-bin the layers"
        )

    # ---- depth windows
    if args.surface_layer is not None:
        center = float(args.surface_layer)
    else:
        center = float(D // 2) if not resample else (D - 1) / 2.0
    if not resample and DZ % 2 == 0 and float(center).is_integer():
        center -= 0.5  # even window: layers c-DZ/2 .. c+DZ/2-1, as the official centre crops
    shifts = [float(v) for v in str(args.z_shifts).split(",")]
    directions = ["forward", "reverse"] if args.direction == "both" else [args.direction]
    zmats = {sh: z_weights(D, DZ, center, z_ratio, sh) for sh in shifts}
    z_log = {}
    for sh, zm in zmats.items():
        used = np.nonzero(zm.sum(0) > 0)[0]
        empty = int((zm.sum(1) < 1e-6).sum())
        if not used.size:
            raise ValueError(f"depth window with shift {sh:+g} does not intersect the {D}-layer stack")
        z_log[f"{sh:+g}"] = [int(used.min()), int(used.max())]
        log(
            f"  z window (shift {sh:+g}): source layers {used.min()}..{used.max()} -> {DZ} model planes"
            + (f"  WARNING: {empty} model planes fall outside the stack and are zero-filled" if empty else "")
        )
    z_needed = sorted(set(int(i) for zm in zmats.values() for i in np.nonzero(zm.sum(0) > 0)[0]))

    # ---- model-grid geometry
    if resample:
        Hm, Wm = int(round(H * s_xy)), int(round(W * s_xy))
        my0, mx0 = int(math.floor(y0 * s_xy)), int(math.floor(x0 * s_xy))
        my1, mx1 = min(Hm, int(math.ceil(y1 * s_xy))), min(Wm, int(math.ceil(x1 * s_xy)))
    else:
        Hm, Wm, my0, mx0, my1, mx1 = H, W, y0, x0, y1, x1
    out_h, out_w = my1 - my0, mx1 - mx0
    stride = args.stride or max(1, int(round(P * (1.0 - args.overlap))))
    blend = args.blend if args.blend != "auto" else ("constant" if stride >= P else "hann")
    window = blend_window(P, blend)
    # tile grid anchored on the full image, so an ROI run equals the crop of a full run
    ys = [y for y in sliding_positions(Hm, P, stride) if y < my1 and y + P > my0]
    xs = [x for x in sliding_positions(Wm, P, stride) if x < mx1 and x + P > mx0]
    mask_m = None
    if mask is not None:
        mask_m = cv2.resize(mask, (Wm, Hm), interpolation=cv2.INTER_NEAREST) if resample else mask
        mask_m = (mask_m > 0).astype(np.float32)
    log(
        f"  {model.name}: resample={'%g um (xy scale %.4f, %.3f layers/plane)' % (target_um, s_xy, z_ratio) if resample else 'no'}; "
        f"model grid {Hm} x {Wm}, region {out_h} x {out_w}, patch {DZ}x{P}x{P}, stride {stride}, blend {blend}, "
        f"{len(ys) * len(xs)} tiles x {len(shifts) * len(directions)} pass(es)"
    )

    bands, cur = [], []
    for y in ys:
        if cur and (y + P - cur[0]) > args.band_rows:
            bands.append(cur)
            cur = []
        cur.append(y)
    if cur:
        bands.append(cur)

    keys = [(d, sh) for d in directions for sh in shifts]
    sums = {k: [np.zeros((out_h, out_w), np.float32), np.zeros((out_h, out_w), np.float32)] for k in keys}
    data_seen = np.zeros((out_h, out_w), bool)
    stats = {"tiles_run": 0, "tiles_skipped_empty": 0, "tiles_skipped_mask": 0, "forward_seconds": 0.0, "read_seconds": 0.0}
    total_tiles = len(ys) * len(xs) * len(keys)
    done = 0
    t_start = last_log = time.time()

    for band in bands:
        by0, by1 = band[0], min(Hm, band[-1] + P)
        bx0, bx1 = xs[0], min(Wm, xs[-1] + P)
        if resample:
            sy0 = max(0, int(math.floor((by0 + 0.5) / s_xy - 0.5)) - 3)
            sy1 = min(H, int(math.ceil((by1 - 0.5) / s_xy - 0.5)) + 4)
            sx0 = max(0, int(math.floor((bx0 + 0.5) / s_xy - 0.5)) - 3)
            sx1 = min(W, int(math.ceil((bx1 - 0.5) / s_xy - 0.5)) + 4)
        else:
            sy0, sy1, sx0, sx1 = by0, by1, bx0, bx1
        t_read = time.time()
        raw = source.read(z_needed, sy0, sy1, sx0, sx1)  # (len(z_needed), h, w) uint8
        stats["read_seconds"] += time.time() - t_read

        for si, sh in enumerate(shifts):
            sub = zmats[sh][:, z_needed].astype(np.float32)
            if bool(np.all((sub == 0) | (sub == 1))):  # pure layer selection: stay in uint8
                planes = np.zeros((DZ,) + raw.shape[1:], np.uint8)
                for k in range(DZ):
                    idx = np.nonzero(sub[k])[0]
                    if idx.size:
                        planes[k] = raw[idx[0]]
            else:
                planes = np.tensordot(sub, raw.astype(np.float32), axes=(1, 0))
            if resample:
                bh, bw = by1 - by0, bx1 - bx0
                # model pixel (mx, my) samples source ((mx + 0.5) / s - 0.5, (my + 0.5) / s - 0.5)
                M = np.array(
                    [
                        [1.0 / s_xy, 0.0, (bx0 + 0.5) / s_xy - 0.5 - sx0],
                        [0.0, 1.0 / s_xy, (by0 + 0.5) / s_xy - 0.5 - sy0],
                    ]
                )
                box = int(round(1.0 / s_xy)) if s_xy < 0.75 else 1  # anti-alias strong reductions
                warped = np.empty((DZ, bh, bw), np.float32)
                for k in range(DZ):
                    plane = planes[k].astype(np.float32)
                    if box > 1:
                        plane = cv2.blur(plane, (box, box))
                    warped[k] = cv2.warpAffine(
                        plane, M, (bw, bh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
                    )
                planes = warped

            for direction in directions:
                stack = planes[::-1] if direction == "reverse" else planes
                psum, wsum = sums[(direction, sh)]
                batch, metas = [], []

                def flush():
                    nonlocal done
                    if not batch:
                        return
                    tf = time.time()
                    x = np.stack(batch)
                    prob = model.predict(x)
                    if args.tta:
                        prob = prob + model.predict(np.ascontiguousarray(x[:, :, ::-1]))[:, ::-1]
                        prob = prob + model.predict(np.ascontiguousarray(x[:, :, :, ::-1]))[:, :, ::-1]
                        prob = prob + model.predict(np.ascontiguousarray(x[:, :, ::-1, ::-1]))[:, ::-1, ::-1]
                        prob = prob / 4.0
                    stats["forward_seconds"] += time.time() - tf
                    for p, (ty, tx, vh, vw) in zip(prob, metas):
                        wgt = window[:vh, :vw]
                        if mask_m is not None:
                            wgt = wgt * mask_m[ty:ty + vh, tx:tx + vw]
                        cy0, cy1 = max(ty, my0), min(ty + vh, my1)
                        cx0, cx1 = max(tx, mx0), min(tx + vw, mx1)
                        src = (slice(cy0 - ty, cy1 - ty), slice(cx0 - tx, cx1 - tx))
                        dst = (slice(cy0 - my0, cy1 - my0), slice(cx0 - mx0, cx1 - mx0))
                        psum[dst] += p[:vh, :vw][src] * wgt[src]
                        wsum[dst] += wgt[src]
                    stats["tiles_run"] += len(batch)
                    done += len(batch)
                    batch.clear()
                    metas.clear()

                for ty in band:
                    vh = min(P, Hm - ty)
                    for tx in xs:
                        vw = min(P, Wm - tx)
                        if mask_m is not None and not mask_m[ty:ty + vh, tx:tx + vw].any():
                            stats["tiles_skipped_mask"] += 1
                            done += 1
                            continue
                        patch = np.zeros((DZ, P, P), stack.dtype)
                        patch[:, :vh, :vw] = stack[:, ty - by0:ty - by0 + vh, tx - bx0:tx - bx0 + vw]
                        if not patch.any():  # raw-empty tiles are skipped, as in the official runtime
                            stats["tiles_skipped_empty"] += 1
                            done += 1
                            continue
                        if si == 0 and direction == directions[0]:
                            cy0, cy1 = max(ty, my0), min(ty + vh, my1)
                            cx0, cx1 = max(tx, mx0), min(tx + vw, mx1)
                            data_seen[cy0 - my0:cy1 - my0, cx0 - mx0:cx1 - mx0] |= (
                                patch[:, cy0 - ty:cy1 - ty, cx0 - tx:cx1 - tx].max(0) > 0
                            )
                        batch.append(model.normalize(patch))
                        metas.append((ty, tx, vh, vw))
                        if len(batch) >= args.batch_size:
                            flush()
                            if time.time() - last_log > 30:
                                el = time.time() - t_start
                                log(f"    {done}/{total_tiles} tiles, {el:.0f}s elapsed, eta {el / max(done, 1) * (total_tiles - done):.0f}s")
                                last_log = time.time()
                flush()
            del planes
        del raw

    elapsed = time.time() - t_start
    probs = {}
    for key, (psum, wsum) in sums.items():
        prob = np.divide(psum, wsum, out=np.zeros_like(psum), where=wsum > 1e-6)
        np.clip(prob, 0.0, 1.0, out=prob)
        if resample:
            prob = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        probs[key] = prob
    px_um = target_um if resample else um_xy
    area_cm2 = float(data_seen.sum()) * (px_um * 1e-4) ** 2
    passes = len(keys)
    info = dict(model.info)
    info.update(
        {
            "resampled": resample,
            "model_pixel_um": px_um,
            "xy_scale": s_xy,
            "source_layers_per_model_plane": z_ratio,
            "surface_layer": center,
            "source_layers_used_by_shift": z_log,
            "stride": stride,
            "blend": blend,
            "passes": passes,
            "seconds": round(elapsed, 2),
            "seconds_forward": round(stats["forward_seconds"], 2),
            "seconds_reading": round(stats["read_seconds"], 2),
            "tiles_run": stats["tiles_run"],
            "tiles_skipped_empty": stats["tiles_skipped_empty"],
            "tiles_skipped_mask": stats["tiles_skipped_mask"],
            "area_with_data_cm2": round(area_cm2, 4),
            "cm2_per_hour_per_pass": round(area_cm2 * passes / max(elapsed, 1e-9) * 3600.0, 2),
        }
    )
    log(
        f"  {model.name}: {stats['tiles_run']} tiles in {elapsed:.1f}s ({stats['forward_seconds']:.1f}s network, "
        f"{stats['read_seconds']:.1f}s reading); {area_cm2:.2f} cm2 with data -> {info['cm2_per_hour_per_pass']} cm2/h per pass"
    )
    return probs, info


def combine(maps: list[np.ndarray], how: str) -> np.ndarray:
    stack = np.stack(maps).astype(np.float32)
    if how == "mean":
        return stack.mean(0)
    if how == "product":
        return stack.prod(0)
    if how == "geomean":
        return np.exp(np.log(np.clip(stack, 1e-6, 1.0)).mean(0))
    if how == "min":
        return stack.min(0)
    raise ValueError(how)


def run(args) -> dict:
    import cv2
    import torch

    cv2.setNumThreads(max(1, args.threads))
    torch.set_num_threads(max(1, args.threads))
    torch.set_grad_enabled(False)

    source, seg_dir = open_source(args.input, args.level)
    if args.input_um:
        source.um_xy = float(args.input_um)
        source.um_z = float(args.input_um_z or args.input_um)
    elif args.input_um_z:
        source.um_z = float(args.input_um_z)
    if args.xy_bin > 1:
        source = BinnedSource(source, args.xy_bin)  # --roi, --mask and the output then live on the binned grid
    log(f"input: {source.description}")
    D, H, W = source.depth, source.height, source.width

    um_xy = source.um_xy
    um_z = source.um_z or um_xy
    voxel_known = um_xy is not None
    if args.roi:
        y0, x0, y1, x1 = args.roi
        if not (0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W):
            raise ValueError(f"--roi Y0 X0 Y1 X1 = {args.roi} must lie inside the {H} x {W} input")
    else:
        y0, x0, y1, x1 = 0, 0, H, W

    mask_path, mask = args.mask, None
    if mask_path is None and seg_dir is not None and not args.no_mask:
        for cand in ("mask.png", "mask.tif", f"{seg_dir.name}_mask.png"):
            if (seg_dir / cand).exists():
                mask_path = str(seg_dir / cand)
                break
    if mask_path and not args.no_mask:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (H, W):
            raise ValueError(f"mask {mask_path} must be a readable image of shape {(H, W)}, got {None if mask is None else mask.shape}")
        log(f"mask: {mask_path} ({100 * float((mask > 0).mean()):.1f}% foreground)")
    else:
        mask_path = None

    t_all = time.time()
    per_model, infos = [], []
    for ckpt in args.checkpoint:
        t0 = time.time()
        model = load_model(ckpt, args)
        log(f"loaded {Path(ckpt).name} ({model.name}) in {time.time() - t0:.1f}s")
        if um_xy is None:
            log(f"WARNING: voxel size unknown (no --input-um, no metadata): ASSUMING the model's native {model.nominal_um} um. "
                "Pass --input-um for 7.91 / 8.64 um data.")
            voxel = (model.nominal_um, model.nominal_um, False)
        else:
            voxel = (um_xy, um_z, True)
        probs, info = sweep(source, model, args, (y0, x0, y1, x1), mask, voxel)
        per_model.append((model, probs))
        infos.append(info)
        del model

    directions = ["forward", "reverse"] if args.direction == "both" else [args.direction]
    shifts = [float(v) for v in str(args.z_shifts).split(",")]
    single = len(per_model) == 1 and len(shifts) == 1
    families = {m.name for m, _ in per_model}
    raw_combine = args.combine == "mean" and len(families) == 1  # official-style probability mean
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = {}
    for direction in directions:
        path = out_path if len(directions) == 1 else out_path.with_name(f"{out_path.stem}_{direction}{out_path.suffix}")
        files = []
        lo_d, hi_d = per_model[0][0].display_rescale
        if single or raw_combine:
            prob = combine([probs[(direction, sh)] for _, probs in per_model for sh in shifts], "mean")
            save_image(path, (prob * 255.0).astype(np.uint8))  # truncation, as the official flat inference
            files.append(str(path))
            if (lo_d, hi_d) != (0.0, 1.0) and not args.no_display:
                dpath = path.with_name(f"{path.stem}_display.png")
                save_image(dpath, (np.clip((prob - lo_d) / (hi_d - lo_d), 0, 1) * 255.0 + 0.5).astype(np.uint8))
                files.append(str(dpath))
            value_note = "uint8 = floor(255 * p)"
        else:
            # heterogeneous models or a non-mean rule: bring every map to a common 0..1 scale first
            scaled = []
            for model, probs in per_model:
                lo_m, hi_m = model.display_rescale
                per_shift = [np.clip((probs[(direction, sh)] - lo_m) / (hi_m - lo_m), 0, 1) for sh in shifts]
                scaled.append(combine(per_shift, "mean"))  # depth ensemble inside a model is always a mean
            prob = combine(scaled, args.combine)
            save_image(path, (prob * 255.0 + 0.5).astype(np.uint8))
            files.append(str(path))
            value_note = f"uint8 = round(255 * {args.combine} over models of clip((p - lo) / (hi - lo))) [per-model display scale]"
        if args.save_each and not single:
            for (model, probs), ckpt in zip(per_model, args.checkpoint):
                lo_m, hi_m = model.display_rescale
                tag = re.sub(r"[^A-Za-z0-9]+", "-", "-".join(Path(ckpt).parts[-2:]).rsplit(".", 1)[0])
                for sh in shifts:
                    epath = path.with_name(f"{path.stem}__{tag}_z{sh:+g}.png")
                    q = np.clip((probs[(direction, sh)] - lo_m) / (hi_m - lo_m), 0, 1)
                    save_image(epath, (q * 255.0 + 0.5).astype(np.uint8))
                    files.append(str(epath))
        if args.save_float:
            fpath = path.with_name(f"{path.stem}_prob.npy")
            np.save(fpath, prob.astype(np.float16))
            files.append(str(fpath))
        written[direction] = files
        log(f"wrote {files[0]} (+{len(files) - 1} more)  mean {float(prob.mean()):.3f}, p99 {float(np.percentile(prob, 99)):.3f}")

    elapsed = time.time() - t_all
    area = infos[0]["area_with_data_cm2"]
    meta = {
        "input": args.input,
        "source": source.description,
        "source_shape_zyx": [D, H, W],
        "roi_y0_x0_y1_x1": [y0, x0, y1, x1],
        "voxel_um_xy": um_xy,
        "voxel_um_z": um_z,
        "voxel_size_known": voxel_known,
        "directions": directions,
        "z_shifts_model_planes": shifts,
        "combine": "mean of raw probabilities" if (single or raw_combine) else args.combine + " of per-model rescaled probabilities",
        "output_value": value_note,
        "display_rescale_lo_hi": list(per_model[0][0].display_rescale),
        "mask": mask_path,
        "tta": bool(args.tta),
        "threads": args.threads,
        "batch_size": args.batch_size,
        "models": infos,
        "outputs": written,
        "seconds_total_including_model_load": round(elapsed, 2),
        "area_with_data_cm2": area,
        "cm2_per_hour_this_configuration": round(area / max(elapsed, 1e-9) * 3600.0, 2) if area else None,
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=1))
    log(f"done in {elapsed:.1f}s; {area:.2f} cm2 -> {meta['cm2_per_hour_this_configuration']} cm2/h for this configuration; metadata {meta_path}")
    return meta


def save_image(path: Path, image_u8: np.ndarray) -> None:
    import cv2

    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        tifffile.imwrite(path, image_u8, compression="zlib", tile=(256, 256) if min(image_u8.shape) >= 256 else None)
    else:
        if not cv2.imwrite(str(path), image_u8):
            raise IOError(f"could not write {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="segment dir (with layers/NN.tif), layers dir, zarr path/URL, .npy or TIFF stack")
    p.add_argument("-c", "--checkpoint", nargs="+", required=True, help="one or more checkpoints (probabilities are averaged)")
    p.add_argument("-o", "--output", required=True, help="output image (.png or .tif); a .json sidecar is written next to it")
    p.add_argument("--model", choices=("auto", "ink9um", "hecate"), default="auto")
    p.add_argument("--hecate-py", help="path to hecate.py from huggingface.co/scrollprize/hecate (only for --model hecate)")
    p.add_argument("--level", default="0", help="pyramid level for OME-Zarr inputs (default 0)")
    p.add_argument("--roi", type=int, nargs=4, metavar=("Y0", "X0", "Y1", "X1"), help="half-open pixel region of the input to predict")
    p.add_argument("--mask", help="mask image on the input grid (default: <dir>/mask.png when present)")
    p.add_argument("--no-mask", action="store_true", help="ignore any mask")
    p.add_argument("--input-um", type=float, help="in-plane pixel size of the input in micrometres (default: from metadata)")
    p.add_argument("--input-um-z", type=float, help="spacing between layers in micrometres (default: same as --input-um / metadata)")
    p.add_argument("--resample", choices=("auto", "always", "never"), default="auto",
                   help="resample to the model's voxel size; auto = only when the input is outside the model's native range")
    p.add_argument("--target-um", type=float, help="voxel size to resample to (default: the model's nominal size, 9.6)")
    p.add_argument("--xy-bin", type=int, default=1,
                   help="mean-bin the input by this integer factor in XY first (e.g. 4 for 2.4 um layers; --input-um stays the "
                        "ORIGINAL pixel size, --roi/--mask/output refer to the binned grid)")
    p.add_argument("--surface-layer", type=float, help="index of the layer on the traced surface (default: the middle layer)")
    p.add_argument("--z-shifts", default="0", help="comma-separated shifts of the depth window in model planes; several values are averaged (e.g. -2,0,2)")
    p.add_argument("--direction", choices=("forward", "reverse", "both"), default="forward",
                   help="depth orientation; use 'both' when the side of the sheet that carries ink is unknown")
    p.add_argument("--overlap", type=float, default=0.5, help="tile overlap fraction (default 0.5)")
    p.add_argument("--stride", type=int, help="explicit tile stride in model pixels (overrides --overlap)")
    p.add_argument("--blend", choices=("auto", "hann", "constant"), default="auto")
    p.add_argument("--combine", choices=("mean", "product", "geomean", "min"), default="mean",
                   help="how to combine several checkpoints (mean = official-style probability average; product/geomean/min "
                        "operate on per-model rescaled probabilities and suppress detections only one model makes)")
    p.add_argument("--save-each", action="store_true", help="also write one rescaled PNG per checkpoint and per z shift (depth sweep)")
    p.add_argument("--tta", action="store_true", help="average over the 4 XY mirror variants (4x slower)")
    p.add_argument("--device", default="auto", help="auto (cuda when available), cpu, cuda, cuda:0 ...")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU (fp16 for ink_9um, bf16 for Hecate)")
    p.add_argument("--hecate-amp", choices=("auto", "bf16", "fp16"), default="auto",
                   help="Hecate autocast dtype on CUDA (auto: bf16 on sm_80+, fp16 otherwise, e.g. T4)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--band-rows", type=int, default=1536, help="rows of the model grid held in memory at once")
    p.add_argument("--save-float", action="store_true", help="also save the float16 probability as .npy")
    p.add_argument("--no-display", action="store_true", help="do not write the contrast-rescaled *_display.png")
    p.add_argument("--allow-unsafe-pickle", action="store_true", help="allow torch.load(weights_only=False) for trusted checkpoints only")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
