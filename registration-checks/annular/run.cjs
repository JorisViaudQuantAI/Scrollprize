#!/usr/bin/env node

'use strict';
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const {
  acquire,
  sha256,
  saveAtomic
} = require('./volume.cjs');
const {
  scoreSite
} = require('./measure.cjs');

const FROZEN_SHA = 'c5966d83214ea8ebdb59bab6af98ff168d84c493c2fc4c114fca61cd3c56ce87';
const VOXEL_UM = 9.362;

function percentile(values, probability) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  return sorted[lower] + (sorted[Math.ceil(position)] - sorted[lower]) * (position - lower);
}

function summarize(sites) {
  const measured = sites.filter(site => site.status === 'measured');
  const norms = measured.map(site => site.norm_vox);
  const counts = {};
  for (const site of sites) counts[site.status] = (counts[site.status] || 0) + 1;
  const stats = values => ({
    median: percentile(values, 0.5),
    p90: percentile(values, 0.9)
  });
  return {
    planned: sites.length,
    status_counts: counts,
    boundary_optima: measured.filter(site => site.boundary).length,
    heldout_improved_or_equal: measured.filter(site =>
      site.heldout.dog.nominal !== null && site.heldout.dog.corrected !== null &&
      site.heldout.dog.corrected >= site.heldout.dog.nominal).length,
    residual_norm_voxels: stats(norms),
    residual_norm_um: stats(norms.map(value => value * VOXEL_UM)),
    median_signed_xyz_voxels: [0, 1, 2].map(axis => percentile(measured.map(site => site.delta_xyz[
      axis]), 0.5)),
    p90_absolute_xyz_voxels: [0, 1, 2].map(axis => percentile(measured.map(site => Math.abs(site
      .delta_xyz[axis])), 0.9)),
    heldout_dog_nominal: stats(measured.map(site => site.heldout.dog.nominal).filter(v => v !== null)),
    heldout_dog_corrected: stats(measured.map(site => site.heldout.dog.corrected).filter(v => v !==
      null))
  };
}

function writeTable(sites, output) {
  const columns = ['sample', 'site', 'fixed_x', 'fixed_y', 'fixed_z', 'fine_x', 'fine_y', 'fine_z',
    'radius_fine_axis_mm', 'radius_second_axis_mm', 'status', 'fit_planes', 'heldout_planes',
    'dx_vox', 'dy_vox', 'dz_vox', 'norm_vox', 'norm_um', 'search_boundary', 'fit_dog',
    'heldout_dog_nominal', 'heldout_dog_corrected', 'heldout_raw_nominal', 'heldout_raw_corrected'
  ];
  const rows = sites.map(site => {
    const delta = site.delta_xyz || [null, null, null];
    return [site.sample, site.id, ...site.fixed, ...site.moving, site.realized_radius_mm,
      site.second_axis_radius_mm, site.status, site.train_planes, site.heldout_planes, ...delta,
      site.norm_vox, site.norm_vox == null ? null : site.norm_vox * VOXEL_UM, site.boundary,
      site.train_ncc, site.heldout?.dog.nominal, site.heldout?.dog.corrected,
      site.heldout?.raw.nominal, site.heldout?.raw.corrected
    ].map(value =>
      value == null ? '' : typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(
      6) : value
    ).join('\t');
  });
  saveAtomic(output, [columns.join('\t'), ...rows].join('\n') + '\n');
}

async function main() {
  const args = process.argv.slice(2);
  if (args.includes('--help')) {
    console.log('node run.cjs CACHE_DIRECTORY [--offline] [--sample PHerc1203|PHerc0846A]');
    return;
  }
  const sampleIndex = args.indexOf('--sample');
  const sample = sampleIndex < 0 ? null : args[sampleIndex + 1];
  if (sampleIndex >= 0) {
    if (!['PHerc1203', 'PHerc0846A'].includes(sample)) throw Error('Unknown or missing --sample');
    args.splice(sampleIndex, 2);
  }
  const offline = args.includes('--offline');
  const positional = args.filter(arg => arg !== '--offline');
  if (positional.length !== 1 || positional[0].startsWith('--')) throw Error(
    'A cache directory is required. See --help.');
  const cache = path.resolve(positional[0]);
  fs.mkdirSync(cache, {
    recursive: true
  });
  const source = fs.readFileSync(path.join(__dirname, 'frozen-sites.json'));
  if (sha256(source) !== FROZEN_SHA) throw Error('Frozen site list has changed');
  const study = JSON.parse(source);
  const selected = study.sites.filter(site => !sample || site.sample === sample);
  const dataset = await acquire(selected, cache, offline);
  const sources = new Map(dataset.results.map(item => [item.file, item]));
  const programDigest = sha256(Buffer.concat(['run.cjs', 'volume.cjs', 'volumes.json', 'measure.cjs']
    .map(name => fs.readFileSync(path.join(__dirname, name)))));
  const results = [];
  const siteDirectory = path.join(cache, 'sites');
  fs.mkdirSync(siteDirectory, {
    recursive: true
  });
  for (const site of selected) {
    const ranges = dataset.ranges.filter(range => range.s === site.sample && range.id === site.id);
    const missing = ranges.filter(range => dataset.failed.has(range.file)).map(range => range.file);
    const digest = sha256([FROZEN_SHA, programDigest, ...ranges.map(range => sources.get(range.file)
      ?.sha256 || 'missing')].join(' '));
    const file = path.join(siteDirectory, site.sample + '-' + site.id + '.json');
    let result;
    if (!missing.length && fs.existsSync(file)) {
      const saved = JSON.parse(fs.readFileSync(file));
      if (saved.digest === digest) result = saved;
    }
    if (!result) {
      result = missing.length ? {
        ...site,
        status: 'transport_incomplete',
        missing
      } : scoreSite(site, ranges, cache);
      const otherCenter = study.centers[site.sample][site.stratum].coarse_based_fine_xy;
      result.second_axis_radius_mm = otherCenter ? 0.009612 * Math.hypot(
        site.moving[0] / 4 - otherCenter[0], site.moving[1] / 4 - otherCenter[1]) : null;
      result.digest = digest;
      saveAtomic(file, JSON.stringify(result, null, 2) + '\n');
    }
    results.push(result);
    console.log(site.sample, site.id, result.status, result.delta_xyz?.join(',') || '');
  }
  const groups = {};
  for (const name of [...new Set(selected.map(site => site.sample))]) {
    const subset = results.filter(site => site.sample === name);
    groups[name] = summarize(subset);
    if (name === 'PHerc0846A') {
      groups[name].within_10_20_mm_under_both_axis_estimates = summarize(subset.filter(site =>
        site.second_axis_radius_mm >= 10 && site.second_axis_radius_mm <= 20));
    }
  }
  const tag = sample ? '-' + sample : '';
  const report = {
    frozen_sha256: FROZEN_SHA,
    software_sha256: programDigest,
    node: process.version,
    platform: os.platform(),
    downloaded_ranges: dataset.ranges.length,
    confirmed_missing_zarr_objects: dataset.results.filter(item =>
      item.status === 'absent_Zarr_zero_fill').length,
    transport_failures: dataset.failed.size,
    groups
  };
  writeTable(results, path.join(cache, 'residuals' + tag + '.tsv'));
  saveAtomic(path.join(cache, 'summary' + tag + '.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
  if (dataset.failed.size) process.exitCode = 2;
}
main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
