#!/usr/bin/env node

'use strict';
const fs = require('node:fs');
const path = require('node:path');
const {
  acquire,
  sha256,
  saveAtomic,
  toMoving,
  volumes
} = require('../annular/volume.cjs');
const {
  fineAccessor,
  coarseVolume,
  trilinear
} = require('../annular/measure.cjs');
const {
  leaveOneHeightOut
} = require('../annular/height-transfer.cjs');
const {
  analyzeAxes,
  project,
  percentile
} = require('./tensor.cjs');

function collectCube(size, sample) {
  const out = new Float32Array(size ** 3),
    half = (size - 1) / 2;
  for (let z = 0; z < size; z++)
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++)
        out[(z * size + y) * size + x] = sample(x - half, y - half, z - half);
  return out;
}

function siteAxes(site, ranges, cache) {
  const fixed = coarseVolume(ranges.filter(r => r.t === 'coarse'), site.fixed, cache);
  const moving = fineAccessor(ranges.filter(r => r.t === 'fine'), cache);
  const matrix = volumes[site.sample].matrix;
  const readFixed = (dx, dy, dz) => fixed[25 + dz][(60 + dy) * 120 + 60 + dx];
  const readMoving = (dx, dy, dz) => {
    const fine = toMoving(matrix, [site.fixed[0] + dx, site.fixed[1] + dy, site.fixed[2] + dz]);
    return trilinear(moving, fine[0] / 4, fine[1] / 4, fine[2] / 4);
  };
  return analyzeAxes(collectCube(27, readFixed), collectCube(27, readMoving), collectCube(43,
  readFixed));
}

function numberRows(tsv) {
  const rows = tsv.trimEnd().split(/\r?\n/),
    header = rows.shift().split('\t');
  return rows.map(line => Object.fromEntries(line.split('\t').map((value, i) => [header[i], value])));
}

function distribution(values) {
  return {
    n: values.length,
    median: percentile(values, .5),
    p90: percentile(values, .9)
  };
}

function count(values) {
  const ans = {};
  for (const value of values) ans[value] = (ans[value] || 0) + 1;
  return ans;
}

function components(rows) {
  const result = {};
  for (const method of ['measured_offset', 'other_height_linear_prediction',
    'after_linear_prediction']) {
    result[method] = {};
    for (const key of ['normal_um', 'tangent_um', 'total_um'])
      result[method][key] = distribution(rows.map(r => r.projections[method][key]));
  }
  return result;
}

function summarize(sites) {
  const measured = sites.filter(r => r.annular_status === 'measured'),
    evaluated = measured.filter(r => r.quality);
  const accepted = evaluated.filter(r => r.quality.reliable);
  const knownAngles = evaluated.map(r => r.quality.scan_angle_degrees).filter(v => v !== null);
  const withoutPair = accepted.filter(r => r.quality.ridges.status === 'no_supported_pair');
  const competing = evaluated.filter(r => r.quality.competing_local_directions === true);
  return {
    planned: sites.length,
    annular_status: count(sites.map(r => r.annular_status)),
    orientation_status: count(sites.map(r => r.orientation_status)),
    measured_sites_with_ct: evaluated.length,
    reliable_orientation: accepted.length,
    rejection_reason_counts_nonexclusive: count(evaluated.filter(r => !r.quality.reliable).flatMap(r => r
      .quality.issues)),
    competing_local_directions: competing.length,
    ridge_flags_all_evaluated: count(evaluated.map(r => r.quality.ridges.status)),
    ridge_flags_reliable: count(accepted.map(r => r.quality.ridges.status)),
    cross_scan_axis_angle_degrees: distribution(knownAngles),
    reliable_site_components: components(accepted),
    reliable_without_detected_pair: {
      n: withoutPair.length,
      components: components(withoutPair)
    }
  };
}
async function main() {
  const args = process.argv.slice(2),
    offline = args.includes('--offline'),
    pos = args.filter(a => a !== '--offline');
  if (pos.length !== 1 || pos[0].startsWith('--')) throw Error(
    'Usage: node run.cjs CACHE_DIRECTORY [--offline]');
  const cache = path.resolve(pos[0]);
  fs.mkdirSync(cache, {
    recursive: true
  });
  const root = path.join(__dirname, '..', 'annular');
  const siteBytes = fs.readFileSync(path.join(root, 'frozen-sites.json'));
  const tsvBytes = fs.readFileSync(path.join(root, 'results', 'residuals.tsv'));
  if (sha256(siteBytes) !== 'c5966d83214ea8ebdb59bab6af98ff168d84c493c2fc4c114fca61cd3c56ce87')
  throw Error('Unexpected anatomical sites');
  if (sha256(tsvBytes) !== '624c459a6a6e1148802c8307b17e2514d6dcee77a0b76db801811e6e61a8f480')
  throw Error('Unexpected residuals');
  const sites = JSON.parse(siteBytes).sites,
    tsv = tsvBytes.toString('utf8');
  const rows = new Map(numberRows(tsv).map(r => [r.sample + ':' + r.site, r]));
  const transfer = leaveOneHeightOut(tsv);
  const preds = new Map(Object.entries(transfer).flatMap(([sample, result]) => result.details.map(
    r => [sample + ':' + r.site, r.predictions.linear])));
  let acquired;
  for (let pass = 0; pass < (offline ? 1 : 3); pass++) {
    acquired = await acquire(sites, cache, offline);
    if (!acquired.failed.size) break;
  }
  const output = [];
  for (const site of sites) {
    const publicId = site.sample + ':' + site.id,
      row = rows.get(publicId);
    if (!row) throw Error('Missing published site ' + publicId);
    const record = {
      sample: site.sample,
      site: site.id,
      annular_status: row.status,
      orientation_status: 'not_measured'
    };
    if (row.radius_second_axis_mm) record.second_axis_radius_mm = Number(row.radius_second_axis_mm);
    if (row.status === 'measured') {
      const ranges = acquired.ranges.filter(x => x.s === site.sample && x.id === site.id);
      if (ranges.some(r => acquired.failed.has(r.file))) record.orientation_status =
        'transport_incomplete';
      else {
        const quality = siteAxes(site, ranges, cache);
        record.quality = quality;
        record.orientation_status = quality.reliable ? 'reliable' : 'orientation_unreliable';
        if (quality.reliable) {
          const d = ['dx_vox', 'dy_vox', 'dz_vox'].map(k => Number(row[k])),
            pred = preds.get(publicId);
          const error = d.map((v, i) => v - pred[i]);
          record.projections = {
            measured_offset: project(d, quality.coarse.axis),
            other_height_linear_prediction: project(pred, quality.coarse.axis),
            after_linear_prediction: project(error, quality.coarse.axis)
          };
          for (const val of Object.values(record.projections)) {
            if (Math.abs(Math.hypot(val.normal_um, val.tangent_um) - val.total_um) > 1e-8) throw Error(
              'Projection mismatch ' + publicId);
          }
        }
      }
    }
    output.push(record);
  }
  const result = {
    version: 1,
    base_study: 'f96d8cdd175de616754412b3c20fdede46d7064a',
    protocol_sha256: sha256(fs.readFileSync(path.join(__dirname, 'protocol-frozen.txt'))),
    raw_ranges: acquired.ranges.length,
    transport_failures: acquired.failed.size,
    groups: {}
  };
  for (const sample of [...new Set(sites.map(r => r.sample))]) {
    const subset = output.filter(r => r.sample === sample);
    result.groups[sample] = summarize(subset);
    if (sample === 'PHerc0846A') result.groups[sample].dual_axis_annulus = summarize(subset.filter(r =>
      r.second_axis_radius_mm >= 10 && r.second_axis_radius_mm <= 20));
  }
  const dst = path.join(cache, 'local-normals');
  fs.mkdirSync(dst, {
    recursive: true
  });
  saveAtomic(path.join(dst, 'sites.json'), JSON.stringify(output, null, 2) + '\n');
  saveAtomic(path.join(dst, 'summary.json'), JSON.stringify(result, null, 2) + '\n');
  console.log('NORMALS_SUMMARY ' + JSON.stringify(result));
  console.log('NORMALS_SITE_QA ' + JSON.stringify(output.map(r => ({
    sample: r.sample,
    site: r.site,
    status: r.orientation_status,
    flags: r.quality?.issues,
    ridges: r.quality?.ridges.status,
    competing: r.quality?.competing_local_directions
  }))));
  if (acquired.failed.size) process.exitCode = 2;
}
if (require.main === module) main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
module.exports = {
  siteAxes,
  summarize
};
