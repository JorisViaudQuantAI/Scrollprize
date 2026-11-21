'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const volumes = require('./volumes.json');
const bucket = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com';

function sha256(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

// Solve the frozen affine map fixed_xyz = matrix * moving_xyz1.
function toMoving(matrix, fixed) {
  const [a, b, c] = matrix;
  const [x, y, z] = fixed.map((value, axis) => value - matrix[axis][3]);
  const det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[
    1] - b[1] * c[0]);
  return [
    ((b[1] * c[2] - b[2] * c[1]) * x + (a[2] * c[1] - a[1] * c[2]) * y + (a[1] * b[2] - a[2] * b[1]) *
      z) / det,
    ((b[2] * c[0] - b[0] * c[2]) * x + (a[0] * c[2] - a[2] * c[0]) * y + (a[2] * b[0] - a[0] * b[2]) *
      z) / det,
    ((b[0] * c[1] - b[1] * c[0]) * x + (a[1] * c[0] - a[0] * c[1]) * y + (a[0] * b[1] - a[1] * b[0]) *
      z) / det
  ];
}

function rangesForSites(sites, metadata) {
  const ranges = [];

  function appendBox(sample, type, id, low, high) {
    const {
      url,
      meta
    } = metadata[sample][type];
    const xyzShape = meta.shape.slice().reverse();
    if (low.some((value, axis) => value < 0 || high[axis] >= xyzShape[axis])) {
      throw Error('Requested box outside source volume: ' + sample + '/' + id);
    }
    const chunk = 128;
    for (let z = Math.floor(low[2] / chunk); z <= Math.floor(high[2] / chunk); z++) {
      for (let y = Math.floor(low[1] / chunk); y <= Math.floor(high[1] / chunk); y++) {
        for (let x = Math.floor(low[0] / chunk); x <= Math.floor(high[0] / chunk); x++) {
          const z0 = Math.max(low[2], z * chunk),
            z1 = Math.min(high[2], (z + 1) * chunk - 1);
          ranges.push({
            s: sample,
            t: type,
            id,
            x,
            y,
            z,
            z0,
            z1,
            start: (z0 - z * chunk) * chunk * chunk,
            len: (z1 - z0 + 1) * chunk * chunk,
            url: url + '/' + [z, y, x].join(meta.dimension_separator ?? '.'),
            file: [sample, type, id, x, y, z, z0, z1].join('_') + '.bin'
          });
        }
      }
    }
  }
  for (const site of sites) {
    const {
      sample,
      id,
      fixed
    } = site;
    const [x, y, z] = fixed;
    appendBox(sample, 'coarse', id, [x - 60, y - 60, z - 25], [x + 59, y + 59, z + 25]);
    const corners = [];
    for (const dx of [-40, 39])
      for (const dy of [-40, 39])
        for (const dz of [-13, 13]) {
          corners.push(toMoving(volumes[sample].matrix, [x + dx, y + dy, z + dz]).map(v => v / 4));
        }
    const low = [0, 1, 2].map(axis => Math.floor(Math.min(...corners.map(p => p[axis]))));
    const high = [0, 1, 2].map(axis => 1 + Math.floor(Math.max(...corners.map(p => p[axis]))));
    appendBox(sample, 'fine', id, low, high);
  }
  return ranges;
}

function saveAtomic(file, bytes) {
  const tmp = file + '.' + process.pid + '.part';
  fs.writeFileSync(tmp, bytes);
  fs.renameSync(tmp, file);
}
async function request(url, options = {}) {
  let error;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const r = await fetch(url, {
        ...options,
        signal: AbortSignal.timeout(30000)
      });
      if (r.status !== 429 && r.status < 500) return r;
      await r.arrayBuffer();
      error = Error('HTTP ' + r.status + ' ' + url);
    } catch (e) {
      error = e;
    }
    if (attempt < 2) await new Promise(resolve => setTimeout(resolve, 300 * (attempt + 1)));
  }
  throw error;
}

async function acquire(sites, cache, offline) {
  const metaFile = path.join(cache, 'metadata.json');
  const checkFile = path.join(cache, 'range-checksums.json');
  const previous = fs.existsSync(checkFile) ? JSON.parse(fs.readFileSync(checkFile)) : [];
  const indexed = new Map(previous.filter(r => r.sha256).map(r => [r.file, r]));
  const metadata = offline ? JSON.parse(fs.readFileSync(metaFile)) : {};
  const selectedSamples = [...new Set(sites.map(s => s.sample))];
  for (const sample of selectedSamples) {
    metadata[sample] ??= {};
    for (const [type, level] of [
        ['fine', 2],
        ['coarse', 0]
      ]) {
      const url = bucket + '/' + sample + '/volumes/' + volumes[sample][type] + '/' + level;
      if (!offline) {
        const reply = await request(url + '/.zarray');
        if (reply.status !== 200) throw Error('Zarr metadata ' + reply.status + ' ' + url);
        metadata[sample][type] = {
          url,
          meta: await reply.json()
        };
      }
      const entry = metadata[sample][type];
      const m = entry?.meta;
      if (entry?.url !== url || m?.dtype !== '|u1' || m.order !== 'C' || m.compressor !== null ||
        m.fill_value !== 0 || String(m.chunks) !== '128,128,128') {
        throw Error('Unsupported source layout ' + url);
      }
    }
  }
  if (!offline) saveAtomic(metaFile, JSON.stringify(metadata, null, 2) + '\n');
  const ranges = rangesForSites(sites, metadata);
  const folder = path.join(cache, 'ranges');
  fs.mkdirSync(folder, {
    recursive: true
  });
  let cursor = 0;
  const results = [];
  async function worker() {
    while (cursor < ranges.length) {
      const item = ranges[cursor++],
        file = path.join(folder, item.file);
      try {
        const prior = indexed.get(item.file);
        if (prior?.url === item.url && prior.start === item.start && prior.len === item.len && fs
          .existsSync(file) &&
          fs.statSync(file).size === item.len && sha256(fs.readFileSync(file)) === prior.sha256) {
          results.push(prior);
          continue;
        }
        if (offline) throw Error('Cache unavailable or digest mismatch');
        const end = item.start + item.len - 1;
        const reply = await request(item.url, {
          headers: {
            Range: 'bytes=' + item.start + '-' + end
          }
        });
        let bytes, status = String(reply.status);
        if (reply.status === 403 || reply.status === 404) {
          await reply.arrayBuffer();
          const key = item.url.slice(bucket.length + 1);
          const listing = await request(bucket + '/?list-type=2&max-keys=1&prefix=' +
            encodeURIComponent(key));
          const xml = await listing.text();
          if (listing.status !== 200 || !xml.includes('<ListBucketResult') || xml.includes('<Key>' +
              key + '</Key>')) {
            throw Error('Object absence could not be confirmed');
          }
          bytes = Buffer.alloc(item.len);
          status = 'absent_Zarr_zero_fill';
        } else {
          bytes = Buffer.from(await reply.arrayBuffer());
          if (reply.status === 200 && bytes.length === 128 ** 3) bytes = bytes.subarray(item.start,
            end + 1);
          else if (reply.status !== 206 || !reply.headers.get('content-range')?.startsWith('bytes ' +
              item.start + '-' + end + '/')) {
            throw Error('Invalid Range response ' + reply.status);
          }
        }
        if (bytes.length !== item.len) throw Error('Incomplete byte range');
        saveAtomic(file, bytes);
        results.push({
          file: item.file,
          url: item.url,
          start: item.start,
          len: item.len,
          status,
          sha256: sha256(bytes)
        });
      } catch (error) {
        results.push({
          file: item.file,
          url: item.url,
          start: item.start,
          len: item.len,
          error: String(error)
        });
      }
    }
  }
  await Promise.all(Array.from({
    length: 8
  }, worker));
  results.sort((a, b) => a.file.localeCompare(b.file));
  const updated = new Map(previous.map(r => [r.file, r]));
  for (const r of results) updated.set(r.file, r);
  saveAtomic(checkFile, JSON.stringify([...updated.values()].sort((a, b) => a.file.localeCompare(b
    .file)), null, 2) + '\n');
  return {
    ranges,
    results,
    failed: new Set(results.filter(r => r.error).map(r => r.file))
  };
}
module.exports = {
  volumes,
  sha256,
  toMoving,
  saveAtomic,
  acquire,
  rangesForSites
};
