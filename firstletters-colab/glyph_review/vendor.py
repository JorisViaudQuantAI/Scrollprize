"""Exact isolated MIT source excerpts for the public gA CPU adapter.

Source: SirProdigle/pherc0306b-first-letters,
commit feabff6c54f95bd8e52a90e0a7a76881ee67d4cc.
See LICENSE-UPSTREAM for copyright and permission notice.
Only imports and the UM constant are supplied locally; no upstream module is imported.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UM = 50.0


class GlyphNet(nn.Module):
    def __init__(self, cin=2, w=24):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.b1, self.b2, self.b3, self.b4 = blk(cin, w), blk(w, 2 * w), blk(2 * w, 4 * w), blk(4 * w, 8 * w)
        self.head = nn.Sequential(nn.Linear(16 * w, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, a, m):
        x = torch.cat([a, m], 1)
        x = F.max_pool2d(self.b1(x), 2)      # 64
        x = F.max_pool2d(self.b2(x), 2)      # 32
        x = F.max_pool2d(self.b3(x), 2)      # 16
        x = self.b4(x)                       # 16x16 cells of 0.4 mm
        x = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
        return self.head(x).squeeze(1)

def resample(a, um, interp=cv2.INTER_AREA):
    f = um / UM
    h, w = a.shape
    return cv2.resize(a.astype(np.float32), (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=interp)

def zscore(a, m):
    v = a[m]
    if v.size < 100:
        return np.zeros_like(a)
    return ((a - v.mean()) / (v.std() + 1e-6)).astype(np.float32)

def a_from_z(z, um, valid, down=3):
    """Overlay representation used in review galleries: z (native res, 0 off-valid) -> 1/down res (AREA) ->
    a = clip((z-0.9)/1.8, 0, 1) -> resampled to UM. Returns (a50, mask50)."""
    z = np.where(valid, z, 0).astype(np.float32)
    h, w = z.shape
    z3 = cv2.resize(z, (w // down, h // down), interpolation=cv2.INTER_AREA)
    v3 = cv2.resize(valid.astype(np.float32), (w // down, h // down), interpolation=cv2.INTER_AREA)
    return a_from_a3(np.clip((z3 - 0.9) / 1.8, 0, 1) * (v3 > 0.5), v3, um * down)

def a_from_a3(a3, v3, um3):
    a = resample(a3.astype(np.float32), um3)
    m = resample(v3.astype(np.float32), um3) > 0.6
    a[~m] = 0
    return a.astype(np.float32), m

def _tta(net, a, m, tta):
    s = net(a, m)
    if tta:      # D4 subset: rot90 x3 + flip
        for k in (1, 2, 3):
            s = s + net(torch.rot90(a, k, (2, 3)), torch.rot90(m, k, (2, 3)))
        s = s + net(a.flip(3), m.flip(3))
        s = s / 5
    return s
