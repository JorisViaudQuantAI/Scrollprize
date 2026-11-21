'use strict';
const fs = require('node:fs');
const path = require('node:path');
const {
  toMoving,
  volumes
} = require('./volume.cjs');

// Each raw byte range covers complete native XY chunk planes.
function fineAccessor(jobs, cacheDir) {
  const cache = new Map();
  for (const j of jobs) {
    const k = j.x + 4096 * j.y + 16777216 * j.z,
      buf = fs.readFileSync(path.join(cacheDir, 'ranges', j.file));
    if (buf.length !== j.len) throw Error('Incomplete ' + j.file);
    cache.set(k, {
      buffer: buf,
      z0: j.z0,
      z1: j.z1
    });
  }
  return function(x, y, z) {
    const c = cache.get((x >> 7) + 4096 * (y >> 7) + 16777216 * (z >> 7));
    if (!c || z < c.z0 || z > c.z1) throw Error('Fine access outside acquired planes ' + [x, y, z]);
    return c.buffer[(z - c.z0) * 16384 + (y & 127) * 128 + (x & 127)];
  };
}

function coarseVolume(jobs, ctr, cacheDir) {
  const q = 120,
    zN = 51,
    [lx, ly, lz] = [ctr[0] - 60, ctr[1] - 60, ctr[2] - 25],
    cub = Array.from({
      length: zN
    }, () => new Uint8Array(q * q));
  for (const j of jobs) {
    const buf = fs.readFileSync(path.join(cacheDir, 'ranges', j.file));
    if (buf.length !== j.len) throw Error('Incomplete ' + j.file);
    const xa = Math.max(lx, j.x * 128),
      xb = Math.min(lx + q, (j.x + 1) * 128),
      ya = Math.max(ly, j.y * 128),
      yb = Math.min(ly + q, (j.y + 1) * 128);
    for (let z = j.z0; z <= j.z1; z++)
      for (let y = ya; y < yb; y++) {
        const from = (z - j.z0) * 16384 + (y - j.y * 128) * 128 + xa - j.x * 128;
        cub[z - lz].set(buf.subarray(from, from + xb - xa), (y - ly) * q + xa - lx);
      }
  }
  return cub;
}

function trilinear(g, x, y, z) {
  const X = Math.floor(x),
    Y = Math.floor(y),
    Z = Math.floor(z),
    a = x - X,
    b = y - Y,
    c = z - Z;
  let t0 = (g(X, Y, Z) * (1 - a) + g(X + 1, Y, Z) * a) * (1 - b) + (g(X, Y + 1, Z) * (1 - a) + g(X + 1,
    Y + 1, Z) * a) * b;
  let t1 = (g(X, Y, Z + 1) * (1 - a) + g(X + 1, Y, Z + 1) * a) * (1 - b) + (g(X, Y + 1, Z + 1) * (1 -
    a) + g(X + 1, Y + 1, Z + 1) * a) * b;
  return (1 - c) * t0 + c * t1;
}
// Use separable Gaussian kernels of radius four sigma and a reflected border.
const kernels = new Map();

function gaussianBlur(a, n, s) {
  const id = n + '-' + s;
  if (!kernels.has(id)) {
    const rad = 4 * s,
      w = [],
      m = new Uint16Array(n * (2 * rad + 1));
    let sum = 0;
    for (let j = -rad; j <= rad; j++) {
      w.push(Math.exp(-.5 * (j / s) ** 2));
      sum += w.at(-1);
    }
    for (let j = 0; j < w.length; j++) w[j] /= sum;
    for (let i = 0; i < n; i++)
      for (let k = -rad; k <= rad; k++) {
        const x = i + k;
        m[i * w.length + k + rad] = x < 0 ? -x - 1 : x >= n ? 2 * n - x - 1 : x;
      }
    kernels.set(id, [w, m]);
  }
  const [w, m] = kernels.get(id), L = w.length, out = new Float32Array(n * n), tmp = new Float32Array(n *
    n);
  for (let y = 0; y < n; y++) {
    const row = y * n;
    for (let x = 0; x < n; x++) {
      let v = 0,
        at = x * L;
      for (let k = 0; k < L; k++) v += w[k] * a[row + m[at + k]];
      tmp[x * n + y] = v;
    }
  }
  for (let x = 0; x < n; x++) {
    const col = x * n;
    for (let y = 0; y < n; y++) {
      let v = 0,
        at = y * L;
      for (let k = 0; k < L; k++) v += w[k] * tmp[col + m[at + k]];
      out[y * n + x] = v;
    }
  }
  return out;
}

function dog(a, n) {
  const lo = gaussianBlur(a, n, 1),
    hi = gaussianBlur(a, n, 4);
  for (let i = 0; i < lo.length; i++) lo[i] -= hi[i];
  return lo;
}

function centeredPixels(arr, sel) {
  const out = Float64Array.from(sel.map(v => arr[v[0]]));
  let mean = 0;
  for (const val of out) mean += val;
  mean /= out.length;
  let sq = 0;
  for (let j = 0; j < out.length; j++) {
    out[j] -= mean;
    sq += out[j] ** 2;
  }
  return {
    vals: out,
    sq
  };
}
// Cache the centered moving values. Every shift uses exactly the same selected pixels.
function referencePlane(raw, D, selection) {
  return {
    idx: Int16Array.from(selection.map(v => v[1])),
    raw: centeredPixels(raw, selection),
    dog: centeredPixels(D, selection)
  };
}

function scoreSite(st, jobs, cacheDir) {
  const N = 80,
    Q = 120,
    fixedRaw = coarseVolume(jobs.filter(j => j.t === 'coarse'), st.fixed, cacheDir),
    movingVoxel = fineAccessor(jobs.filter(j => j.t === 'fine'), cacheDir),
    m = volumes[st.sample].matrix;
  const fixedBandpass = new Array(51);
  const fixedPlane = (k, mode) => mode === 'raw' ? fixedRaw[k] : (fixedBandpass[k] ??= dog(fixedRaw[k],
    Q));
  // Three planes fit the residual. The two planes at +/-13 never enter the fit.
  const all = [],
    innerArea = 64 * 64,
    fixedCenter = st.fixed;
  for (const dz of [-13, -8, 0, 8, 13]) {
    const r = new Float32Array(N * N),
      mask = [];
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++) {
        const p = toMoving(m, [fixedCenter[0] - 40 + x, fixedCenter[1] - 40 + y, fixedCenter[2] + dz]);
        r[y * N + x] = trilinear(movingVoxel, p[0] / 4, p[1] / 4, p[2] / 4);
      }
    const c0 = fixedRaw[25 + dz];
    let nom = 0;
    for (let y = 8; y < 72; y++)
      for (let x = 8; x < 72; x++) {
        const k = y * N + x,
          j = (y + 20) * Q + x + 20;
        if (r[k] > 3 && c0[j] > 3) {
          nom++;
          if (c0[j - 8] > 3 && c0[j + 8] > 3 && c0[j - 8 * Q] > 3 && c0[j + 8 * Q] > 3 && fixedRaw[25 +
              dz - 8][j] > 3 && fixedRaw[25 + dz + 8][j] > 3) mask.push([k, j]);
        }
      }
    const plane = {
      dz,
      n_nom: nom,
      n_mask: mask.length
    };
    if (mask.length >= .5 * innerArea) {
      const D = dog(r, N);
      plane.full = referencePlane(r, D, mask);
      plane.fit = referencePlane(r, D, mask.filter(([k]) => !(k % 2) && !((k / N | 0) % 2)));
    }
    all.push(plane);
  }
  const train = all.filter(p => p.full && Math.abs(p.dz) !== 13),
    hold = all.filter(p => p.full && Math.abs(p.dz) === 13),
    res = {
      ...st,
      plane_support: all.map(p => [p.dz, p.n_nom, p.n_mask]),
      train_planes: train.length,
      heldout_planes: hold.length
    };
  if (train.length < 2 || hold.length < 1) {
    res.status = all.every(p => p.n_nom < .01 * innerArea) ? 'outside' : 'insufficient_anatomy';
    return res;
  }

  function correlationAt(delta, planes, mode, subset) {
    const [dx, dy, dz] = delta, X = Math.floor(dx), Y = Math.floor(dy), Z = Math.floor(dz), fx = dx - X,
      fy = dy - Y, fz = dz - Z, integerOffset = !fx && !fy && !fz, off = X + Q * Y;
    const w = [(1 - fx) * (1 - fy) * (1 - fz), fx * (1 - fy) * (1 - fz), (1 - fx) * fy * (1 - fz), fx *
      fy * (1 - fz), (1 - fx) * (1 - fy) * fz, fx * (1 - fy) * fz, (1 - fx) * fy * fz, fx * fy * fz
    ];
    let wt = 0,
      nr = 0;
    for (const p of planes) {
      const a = p[subset],
        r = a[mode],
        idx = a.idx,
        L = idx.length,
        t = 25 + p.dz + Z,
        c = fixedPlane(t, mode),
        e = integerOffset ? c : fixedPlane(t + 1, mode);
      let s = 0,
        sq = 0,
        co = 0;
      if (integerOffset) {
        for (let j = 0; j < L; j++) {
          const b = c[idx[j] + off];
          s += b;
          sq += b * b;
          co += r.vals[j] * b;
        }
      } else
        for (let j = 0; j < L; j++) {
          const i = idx[j] + off,
            b = w[0] * c[i] + w[1] * c[i + 1] + w[2] * c[i + Q] + w[3] * c[i + Q + 1] + w[4] * e[i] + w[
              5] * e[i + 1] + w[6] * e[i + Q] + w[7] * e[i + Q + 1];
          s += b;
          sq += b * b;
          co += r.vals[j] * b;
        }
      const den = Math.sqrt(r.sq * Math.max(0, sq - s * s / L));
      if (den > 1e-9) {
        wt += co / den * L;
        nr += L;
      }
    }
    return nr ? wt / nr : null;
  }
  // Search the broad even grid plus the complete central integer grid, then refine locally.
  const seen = new Map();
  let best = null;

  function run(v) {
    const k = v.join(',');
    if (seen.has(k)) return;
    const a = correlationAt(v, train, 'dog', 'fit');
    if (a === null) return;
    seen.set(k, [v, a]);
    if (!best || a > best[1] + 1e-12 || Math.abs(a - best[1]) <= 1e-12 && Math.hypot(...v) < Math.hypot(
        ...best[0])) best = [v, a];
  }
  for (const [lim, step] of [
      [8, 2],
      [4, 1]
    ])
    for (let z = -lim; z <= lim; z += step)
      for (let y = -lim; y <= lim; y += step)
        for (let x = -lim; x <= lim; x += step) run([x, y, z]);
  if (!best) {
    res.status = 'no_CT_contrast';
    return res;
  }
  for (const step of [1, .5, .25]) {
    const center = best[0];
    for (const x of [-step, 0, step])
      for (const y of [-step, 0, step])
        for (const z of [-step, 0, step]) {
          const v = [center[0] + x, center[1] + y, center[2] + z];
          if (v.every(c => Math.abs(c) <= 8)) run(v);
        }
  }
  res.status = 'measured';
  res.delta_xyz = best[0];
  res.norm_vox = Math.hypot(...best[0]);
  res.boundary = best[0].some(v => Math.abs(v) >= 8);
  res.train_ncc = best[1];
  const distantCompetitors = Array.from(seen.values()).filter(([v]) => Math.hypot(...v.map((x, i) => x -
    best[0][i])) >= 2);
  res.margin_to_fit_2vox = best[1] - Math.max(...distantCompetitors.map(x => x[1]));
  // Evaluate on all common pixels of held-out planes for both raw intensity and band-pass contrast.
  res.heldout = {};
  for (const mode of ['raw', 'dog']) {
    const zero = correlationAt([0, 0, 0], hold, mode, 'full'),
      corrected = correlationAt(best[0], hold, mode, 'full'),
      co = [];
    for (let axis = 0; axis < 3; axis++)
      for (const sign of [-1, 1]) {
        const v = [0, 0, 0];
        v[axis] = sign * 12;
        co.push(correlationAt(v, hold, mode, 'full'));
      }
    res.heldout[mode] = {
      nominal: zero,
      corrected,
      controls12: co
    };
  }
  return res;
}

module.exports = {
  scoreSite
};
