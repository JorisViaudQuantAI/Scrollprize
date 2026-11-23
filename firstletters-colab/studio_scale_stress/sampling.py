"""Sampling contract for the five frozen Q10 Studio scale-stress arms.

Pixel centres use integer native coordinates. A native 128-square patch at
``(y, x)`` has centre ``(y + 63.5, x + 63.5)``. A scale below one compresses
features and therefore reads a larger native context. The 16 depth samples
are neither reordered nor interpolated. Returned inputs retain raw CT units;
the inference runner owns clipping and normalization.

Only the predeclared scales 0.5, 1 and 2 are accepted. At 0.5, the compressed
axis receives a Gaussian with sigma=(1/scale-1)/2=0.5 and truncate=4 (radius2),
followed by bilinear interpolation. Gaussian convolution is *valid*: no
padding, reflection, clamping or extrapolation contributes to any sample.
For these scales the union of all positive-weight stencils is a rectangle.
Tile presence certifies this rectangle's support, not the raw acquisition
history inside an author's prepared tile.
"""
import operator

import numpy as np


class NativeMosaic:
    """Native Y,X,depth values with an explicit tile-presence mask."""

    def __init__(self, ct_yxd, available_yx):
        values = np.asarray(ct_yxd)
        available = np.asarray(available_yx)
        if values.ndim != 3 or values.shape[2] != 16 or values.dtype != np.uint8:
            raise ValueError("Native CT must be uint8 [height, width, 16]")
        if available.shape != values.shape[:2] or available.dtype != np.bool_:
            raise ValueError("Availability must be a boolean native spatial mask")
        self.ct_yxd = values
        self.available_yx = available


def build_mosaic(ct_tiles, tile_origins):
    """Stitch nonoverlapping uint8 N,H,W,16 tiles at integer Y,X origins.

    Unprovided cells carry a false mask; their initialized zeros never become
    valid observations. A provided all-zero tile remains available.
    """
    tiles = np.asarray(ct_tiles)
    origins = np.asarray(tile_origins)
    if (tiles.ndim != 4 or tiles.shape[-1] != 16 or tiles.dtype != np.uint8
            or min(tiles.shape[:3]) < 1):
        raise ValueError("Tiles must be a nonempty uint8 [N, H, W, 16] array")
    if (origins.shape != (len(tiles), 2)
            or not np.issubdtype(origins.dtype, np.integer)
            or np.any(origins < 0)):
        raise ValueError("Origins must be nonnegative integer [N, 2] Y,X coordinates")
    th, tw = tiles.shape[1:3]
    height = int(origins[:, 0].max()) + th
    width = int(origins[:, 1].max()) + tw
    values = np.zeros((height, width, 16), dtype=np.uint8)
    available = np.zeros((height, width), dtype=bool)
    for tile, (yy, xx) in zip(tiles, origins):
        y, x = int(yy), int(xx)
        region = np.s_[y:y + th, x:x + tw]
        if available[region].any():
            raise ValueError("Overlapping tiles are ambiguous")
        values[region] = tile
        available[region] = True
    return NativeMosaic(values, available)


def _scales(sx, sy):
    result = []
    for value in (sx, sy):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("A scale must be 0.5, 1 or 2")
        value = float(value)
        if value not in (0.5, 1.0, 2.0):
            raise ValueError("A scale must be 0.5, 1 or 2")
        result.append(value)
    return tuple(result)


def _origin(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Patch origins must be integer native coordinates")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError("Patch origins must be integer native coordinates") from exc


def _input_geometry(y, x, sx, sy):
    sx, sy = _scales(sx, sy)
    y, x = _origin(y), _origin(x)
    qy = y + 63.5 + (np.arange(128, dtype=np.float64) - 63.5) / sy
    qx = x + 63.5 + (np.arange(128, dtype=np.float64) - 63.5) / sx
    sigma_y, sigma_x = max((1 / sy - 1) / 2, 0), max((1 / sx - 1) / 2, 0)
    ry, rx = int(4 * sigma_y + 0.5), int(4 * sigma_x + 0.5)
    # ceil, not floor+1: integer samples have no positive-weight upper neighbour.
    y0, y1 = int(np.floor(qy[0])) - ry, int(np.ceil(qy[-1])) + ry + 1
    x0, x1 = int(np.floor(qx[0])) - rx, int(np.ceil(qx[-1])) + rx + 1
    return qy, qx, (sigma_y, sigma_x), (ry, rx), (y0, y1, x0, x1)


def context_support(mosaic, y, x, *, sx=1.0, sy=1.0):
    """Return JSON-safe support metadata without reading image intensities.

    ``source_bounds_yx`` is [[y_min,y_max_exclusive], [x_min,x_max_exclusive]].
    Required pixels include every Gaussian and positive-weight bilinear source
    sample. Pixels outside the mosaic count as missing, as do missing tiles.
    """
    _, _, sigma, radius, bounds = _input_geometry(y, x, sx, sy)
    y0, y1, x0, x1 = bounds
    height, width = mosaic.available_yx.shape
    required = (y1 - y0) * (x1 - x0)
    ay0, ay1, ax0, ax1 = max(0, y0), min(height, y1), max(0, x0), min(width, x1)
    present = (int(np.count_nonzero(mosaic.available_yx[ay0:ay1, ax0:ax1]))
               if ay1 > ay0 and ax1 > ax0 else 0)
    return dict(complete=bool(present == required),
                source_bounds_yx=[[y0, y1], [x0, x1]],
                required_source_pixels=int(required),
                missing_source_pixels=int(required - present),
                aa_sigma_yx=[float(v) for v in sigma],
                aa_radius_yx=[int(v) for v in radius])


def _valid_gaussian(values, *, axis, sigma, radius):
    """Separable Gaussian accumulation using only measured source positions."""
    if radius == 0:
        return values
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    shape = list(values.shape)
    shape[axis] -= 2 * radius
    result = np.zeros(shape, dtype=np.float64)
    for k, weight in enumerate(weights):
        window = [slice(None)] * values.ndim
        window[axis] = slice(k, k + shape[axis])
        result += weight * values[tuple(window)]
    return result


def _bilinear(values, qy, qx):
    """Evaluate a Cartesian query grid without touching zero-weight neighbours."""
    y0, x0 = np.floor(qy).astype(np.int64), np.floor(qx).astype(np.int64)
    fy, fx = qy - y0, qx - x0
    y1, x1 = y0 + (fy > 0), x0 + (fx > 0)
    if (y0.min() < 0 or x0.min() < 0
            or y1.max() >= values.shape[0] or x1.max() >= values.shape[1]):
        raise ValueError("Interpolation stencil extends outside supported values")
    tail = (1,) * (values.ndim - 2)
    wy, wx = fy.reshape((-1, 1) + tail), fx.reshape((1, -1) + tail)
    upper = ((1 - wx) * values[y0[:, None], x0[None, :]]
             + wx * values[y0[:, None], x1[None, :]])
    lower = ((1 - wx) * values[y1[:, None], x0[None, :]]
             + wx * values[y1[:, None], x1[None, :]])
    return (1 - wy) * upper + wy * lower


def sample_input(mosaic, y, x, *, sx=1.0, sy=1.0):
    """Return contiguous FP32 [16,128,128] in unchanged raw CT units.

    The baseline is an exact native crop, followed only by uint8-to-FP32
    conversion. Other arms accumulate Gaussian/bilinear operations in float64
    then round once to FP32. Incomplete measured support raises ValueError.
    """
    qy, qx, sigma, radius, bounds = _input_geometry(y, x, sx, sy)
    support = context_support(mosaic, y, x, sx=sx, sy=sy)
    if not support["complete"]:
        raise ValueError(f"Incomplete native context: {support['missing_source_pixels']} missing source pixels")
    y0, y1, x0, x1 = bounds
    native = mosaic.ct_yxd[y0:y1, x0:x1]
    if sx == 1 and sy == 1:
        return np.ascontiguousarray(native.transpose(2, 0, 1), dtype=np.float32)
    values = native.astype(np.float64)
    for axis in (0, 1):
        values = _valid_gaussian(values, axis=axis, sigma=sigma[axis], radius=radius[axis])
    sampled = _bilinear(values, qy - (y0 + radius[0]), qx - (x0 + radius[1]))
    return np.ascontiguousarray(sampled.transpose(2, 0, 1), dtype=np.float32)


def map_prediction(probability_hw, *, sx=1.0, sy=1.0, available_hw=None):
    """Read probabilities at the same physical 32-square core for every arm.

    Physical core index j maps to model coordinate 63.5 + scale*(j-15.5).
    This is bilinear interpolation of probabilities, with no thresholding,
    label resampling or output Gaussian. The optional mask certifies all
    positive-weight model-grid stencil positions. Invalid values elsewhere
    in a partially supported probability field are not used.
    """
    sx, sy = _scales(sx, sy)
    probability = np.asarray(probability_hw)
    if probability.shape != (128, 128) or not np.issubdtype(probability.dtype, np.number):
        raise ValueError("Probability map must be numeric [128, 128]")
    if np.iscomplexobj(probability):
        raise ValueError("Probability map must be real")
    qy = 63.5 + sy * (np.arange(32, dtype=np.float64) - 15.5)
    qx = 63.5 + sx * (np.arange(32, dtype=np.float64) - 15.5)
    y0, y1 = int(np.floor(qy[0])), int(np.ceil(qy[-1])) + 1
    x0, x1 = int(np.floor(qx[0])), int(np.ceil(qx[-1])) + 1
    if y0 < 0 or x0 < 0 or y1 > 128 or x1 > 128:
        raise ValueError("Physical core falls outside the model prediction")
    if available_hw is not None:
        available = np.asarray(available_hw)
        if available.shape != (128, 128) or available.dtype != np.bool_:
            raise ValueError("Prediction availability must be boolean [128, 128]")
        if not available[y0:y1, x0:x1].all():
            raise ValueError("Incomplete probability interpolation support")
    used = probability[y0:y1, x0:x1]
    if not np.isfinite(used).all() or np.any(used < 0) or np.any(used > 1):
        raise ValueError("Used prediction stencil must contain finite probabilities in [0,1]")
    if sx == 1 and sy == 1:
        return np.ascontiguousarray(used, dtype=np.float32)
    result = _bilinear(used.astype(np.float64), qy - y0, qx - x0)
    return np.ascontiguousarray(result, dtype=np.float32)
