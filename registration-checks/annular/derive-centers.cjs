'use strict';
const fs = require('node:fs'),
  path = require('node:path'),
  crypto = require('node:crypto');
const d = path.resolve(process.argv[2] || '.'),
  base = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com';
const cfg = require('./volumes.json'),
  zs = [1120, 2240, 3200];
async function pull(key, file) {
  const dst = path.join(d, file);
  if (fs.existsSync(dst)) return fs.readFileSync(dst);
  for (let i = 0; i < 3; i++) {
    const r = await fetch(base + '/' + key, {
      signal: AbortSignal.timeout(20000)
    });
    if (r.status === 200) {
      const b = Buffer.from(await r.arrayBuffer());
      fs.writeFileSync(dst, b);
      return b;
    }
    if (i === 2) throw Error(r.status + ' ' + key);
  }
}

function at(m, v) {
  return m.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2] + r[3]);
}

function xyFromFixedAtFineZ(m, q, z) {
  const [a, b, c, t] = m[0], [e, f, g, u] = m[1], X = q[0] - c * z - t, Y = q[1] - g * z - u, det = a *
    f - b * e;
  return [(f * X - b * Y) / det, (a * Y - e * X) / det];
}

function along(umb, z) {
  let r = umb.findIndex(p => p.z >= z);
  if (r <= 0) throw Error('outside umbilicus');
  const a = umb[r - 1],
    b = umb[r],
    f = (z - a.z) / (b.z - a.z);
  return [a.x * (1 - f) + b.x * f, a.y * (1 - f) + b.y * f];
}

function officialAxis(points, m, z) {
  let v = [13244, 13244, 4 * z];
  for (let k = 0; k < 8; k++) {
    const qz = at(m, v)[2],
      q = along(points, qz);
    const xy = xyFromFixedAtFineZ(m, q, v[2]);
    v = [...xy, v[2]];
  }
  return v.slice(0, 2).map(t => t / 4);
}
// Deterministic angular alignment of the official VC normal-grid sheet segments.
// Endianness and delta decoding follow VC GridStore v3 and LineSegList.
function normalAxis(bin) {
  if (bin.readUInt32BE(0) !== 0x56434753 || bin.readUInt32BE(4) !== 3) throw Error(
    'Expected VC GridStore v3');
  const w = bin.readUInt32BE(16),
    h = bin.readUInt32BE(20),
    n = bin.readUInt32BE(32),
    end = bin.readUInt32BE(44);
  let off = bin.readUInt32BE(40);
  const seg = [];
  for (let i = 0; i < n; i++) {
    let x = bin.readUInt32BE(off),
      y = bin.readUInt32BE(off + 4),
      len = bin.readUInt32BE(off + 8);
    off += 12;
    if (len % 2) throw Error('Invalid polyline');
    for (let j = 0; j < len; j += 2) {
      const dx = bin.readInt8(off + j),
        dy = bin.readInt8(off + j + 1),
        mag = Math.hypot(dx, dy);
      if (mag) seg.push([x + dx / 2, y + dy / 2, -dy / mag, dx / mag]);
      x += dx;
      y += dy;
    }
    off += len;
  }
  if (off !== end || !seg.length) throw Error('Invalid grid end ' + off + ' ' + end);
  const nuse = Math.min(10000, seg.length),
    pts = Array.from({
      length: nuse
    }, (_, i) => seg[Math.floor((i + .5) * seg.length / nuse)]);

  function score(x, y, weighted) {
    let v = 0;
    for (const s of pts) {
      const dx = s[0] - x,
        dy = s[1] - y,
        d2 = dx * dx + dy * dy;
      if (d2 <= 1e-12) continue;
      const dot = dx * s[2] + dy * s[3];
      v += dot * dot / d2 / (weighted ? Math.max(100, Math.sqrt(d2)) : 1);
    }
    return v;
  }
  let bx = 0,
    by = 0,
    max = -Infinity;
  for (let iy = 0; iy < 32; iy++)
    for (let ix = 0; ix < 32; ix++) {
      const x = (ix + .5) * w / 32,
        y = (iy + .5) * h / 32,
        t = score(x, y, false);
      if (t > max) {
        max = t;
        bx = x;
        by = y;
      }
    }
  let best = score(bx, by, true);
  for (let step = 1024; step >= 1; step /= 2) {
    for (let repeat = 0; repeat < 40; repeat++) {
      let nx = bx,
        ny = by,
        nv = best;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const x = bx + dx * step,
            y = by + dy * step;
          if (x < 0 || x >= w || y < 0 || y >= h) continue;
          const a = score(x, y, true);
          if (a > nv) {
            nv = a;
            nx = x;
            ny = y;
          }
        }
      if (nv <= best) break;
      bx = nx;
      by = ny;
      best = nv;
    }
  }
  return {
    xy: [bx, by],
    grid_wh: [w, h],
    nsegments: seg.length,
    sampled: nuse
  };
}
// PHerc0846A has no official centerline in the open catalog. Estimate its center independently on both scans.
async function unofficial(z, type) {
  const isF = type === 'fine',
    name = isF ? '20260319102732-surface-20260413222639-surface-m7-L2-th0.2' :
    '20250728152254-surface-20260413222639-surface-m7-L0-th0.2',
    key = 'PHerc0846A/representations/predictions/surfaces/' + name + '.normal-grids/xy/' + String(z)
    .padStart(6, '0') + '.grid';
  const f = '846-' + type + '-' + z + '.grid',
    b = await pull(key, f);
  return {
    ...normalAxis(b),
    key,
    sha256: crypto.createHash('sha256').update(b).digest('hex'),
    z
  };
}
async function main() {
  fs.mkdirSync(d, {
    recursive: true
  });
  const umbkey = 'PHerc1203/representations/umbilicus/20250820131727-umbilicus-20260924035533.json',
    u = JSON.parse((await pull(umbkey, 'umb1203.json')).toString()),
    out = {
      centers: {
        PHerc1203: [],
        PHerc0846A: []
      },
      sites: []
    };
  const med = v => v.slice().sort((a, b) => a - b)[Math.floor(v.length / 2)];
  for (let k = 0; k < zs.length; k++) {
    const z = zs[k];
    out.centers.PHerc1203.push({
      fine_l2_z: z,
      xy: officialAxis(u.control_points, cfg.PHerc1203.matrix, z),
      official: umbkey
    });
    const fine = await Promise.all([-8, 0, 8].map(zz => unofficial(z + zz, 'fine'))),
      xy = [0, 1].map(i => med(fine.map(g => g.xy[i])));
    const pred = at(cfg.PHerc0846A.matrix, [4 * xy[0], 4 * xy[1], 4 * z]),
      cz = Math.round(pred[2]),
      coarse = await unofficial(cz, 'coarse');
    const cross_distance_mm = Math.hypot(coarse.xy[0] - pred[0], coarse.xy[1] - pred[1]) * .009362;
    out.centers.PHerc0846A.push({
      fine_l2_z: z,
      xy,
      fine,
      coarse,
      coarse_based_fine_xy: xyFromFixedAtFineZ(cfg.PHerc0846A.matrix, coarse.xy, 4 * z).map(v =>
        v / 4),
      cross_distance_mm
    });
  }
  for (const [s, records] of Object.entries(out.centers))
    for (const [k, record] of records.entries())
      for (let j = 0; j < 2; j++)
        for (let t = 0; t < 4; t++) {
          const rad = [12, 18][j],
            angle = 15 + 30 * k + 45 * j + 90 * t,
            th = angle * Math.PI / 180,
            x = Math.round(record.xy[0] + rad / .009612 * Math.cos(th)),
            y = Math.round(record.xy[1] + rad / .009612 * Math.sin(th)),
            moving = [4 * x, 4 * y, 4 * zs[k]],
            fixed = at(cfg[s].matrix, moving).map(v => Math.round(v));
          out.sites.push({
            sample: s,
            id: ['ABC' [k], j + 1, t + 1].join(''),
            stratum: k,
            radius_mm: rad,
            realized_radius_mm: .009612 * Math.hypot(x - record.xy[0], y - record.xy[1]),
            angle_deg: angle % 360,
            moving,
            fixed
          });
        }
  fs.writeFileSync(path.join(d, 'rings.json'), JSON.stringify(out, null, 2) + '\n');
  console.log(JSON.stringify(Object.fromEntries(Object.entries(out.centers).map(([s, a]) => [s, a.map(
    x => [x.fine_l2_z, x.xy, x.cross_distance_mm])]))));
}
main().catch(e => {
  console.error(e);
  process.exitCode = 1;
});
