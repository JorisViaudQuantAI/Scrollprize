#!/usr/bin/env node
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { acquire, sha256, saveAtomic } = require('./volume.cjs');
const { scoreSite } = require('./measure.cjs');

function leaveOneHeightOut(tsv) {
  const lines = tsv.trimEnd().split(/\r?\n/);
  const names = lines.shift().split("\t");
  const rows = lines.map(line => Object.fromEntries(
    line.split("\t").map((value, index) => [names[index], value])
  )).filter(row => row.status === "measured");
  const axes = ["x", "y", "z"];
  const coord = row => axes.map(axis => Number(row["fixed_" + axis]));
  const shift = row => axes.map(axis => Number(row["d" + axis + "_vox"]));
  const avg = a => a.reduce((sum, x) => sum + x, 0) / a.length;
  function quantile(a, fraction) {
    const b = [...a].sort((x, y) => x - y);
    const at = (b.length - 1) * fraction, low = Math.floor(at);
    return b[low] * (1 - (at - low)) + b[Math.ceil(at)] * (at - low);
  }
  function solve(matrix, vector) {
    const a = matrix.map((row, i) => [...row, vector[i]]);
    for (let col = 0; col < 3; col++) {
      let best = col;
      for (let i = col + 1; i < 3; i++) if (Math.abs(a[i][col]) > Math.abs(a[best][col])) best = i;
      [a[best], a[col]] = [a[col], a[best]];
      const pivot = a[col][col];
      if (Math.abs(pivot) < 1e-10) throw Error("Singular regression");
      for (let j = col; j < 4; j++) a[col][j] /= pivot;
      for (let i = 0; i < 3; i++) {
        if (i === col) continue;
        const scale = a[i][col];
        for (let j = col; j < 4; j++) a[i][j] -= scale * a[col][j];
      }
    }
    return a.map(row => row[3]);
  }
  function train(examples) {
    const xyz = examples.map(coord), target = examples.map(shift);
    const median = axes.map((_, j) => quantile(target.map(r => r[j]), 0.5));
    const origin = axes.map((_, j) => avg(xyz.map(r => r[j])));
    const offset = axes.map((_, j) => avg(target.map(r => r[j])));
    const design = xyz.map(row => row.map((x, j) => (x - origin[j]) / 1000));
    const gram = axes.map((_, i) => axes.map((_, j) => (i === j ? 1 : 0) + design.reduce((s, x) => s + x[i] * x[j], 0)));
    const weights = axes.map((_, k) => solve(gram, axes.map((_, j) => design.reduce((s, x, i) => s + x[j] * (target[i][k] - offset[k]), 0))));
    return row => {
      const x = coord(row).map((v, j) => (v - origin[j]) / 1000);
      const linear = offset.map((value, k) => value + weights[k].reduce((s, w, j) => s + x[j] * w, 0));
      return {original: [0, 0, 0], median, linear};
    };
  }
  const result = {};
  for (const sample of [...new Set(rows.map(r => r.sample))]) {
    const items = rows.filter(r => r.sample === sample);
    const details = [];
    for (const height of ["A", "B", "C"]) {
      const training = items.filter(r => r.site[0] !== height);
      const testing = items.filter(r => r.site[0] === height);
      if (!testing.length || !training.length) throw Error("Missing height");
      const predict = train(training);
      for (const r of testing) {
        const actual = shift(r), predictions = predict(r), errors = {};
        for (const [method, p] of Object.entries(predictions)) errors[method] = Math.hypot(...p.map((x, j) => actual[j] - x));
        details.push({site:r.site, train_n:training.length, predictions, errors});
      }
    }
    const scores = {};
    for (const method of ["original", "median", "linear"]) {
      const errors = details.map(r => r.errors[method]);
      scores[method] = {median_vox:quantile(errors,.5), p90_vox:quantile(errors,.9),
        better_than_original:details.filter(r => r.errors[method] < r.errors.original - 1e-9).length,
        equal_to_original:details.filter(r => Math.abs(r.errors[method] - r.errors.original) <= 1e-9).length,
        by_height_median_vox:Object.fromEntries(["A","B","C"].map(h => [h,quantile(details.filter(r=>r.site[0]===h).map(r=>r.errors[method]),.5)]))};
    }
    result[sample] = {sites:details.length,scores,details};
  }
  return result;
}

async function main() {
  const input = fs.readFileSync(path.join(__dirname,'results','residuals.tsv'));
  if (sha256(input) !== '624c459a6a6e1148802c8307b17e2514d6dcee77a0b76db801811e6e61a8f480') throw Error('Unexpected residual table');
  const result = leaveOneHeightOut(input.toString('utf8'));
  const args = process.argv.slice(2), offline = args.includes('--offline');
  const positional = args.filter(arg => arg !== '--offline');
  if (positional.length > 1 || (offline && !positional.length)) throw Error('Use node height-transfer.cjs [CACHE [--offline]]');
  if (!positional.length) {
    console.log(JSON.stringify(Object.fromEntries(Object.entries(result).map(([name, r]) => [name, r.scores])), null, 2));
    return;
  }
  const cache = path.resolve(positional[0]);
  fs.mkdirSync(cache, {recursive:true});
  const files = fs.readFileSync(path.join(__dirname,'frozen-sites.json'));
  if (sha256(files) !== 'c5966d83214ea8ebdb59bab6af98ff168d84c493c2fc4c114fca61cd3c56ce87') throw Error('Unexpected sites');
  const sites = JSON.parse(files).sites;
  const dataset = await acquire(sites, cache, offline);
  if (dataset.failed.size) throw Error(dataset.failed.size + ' CT ranges unavailable');
  const intensities = [];
  for (const site of sites) {
    const row = result[site.sample].details.find(r => r.site === site.id);
    if (!row) continue;
    const ranges = dataset.ranges.filter(r => r.s === site.sample && r.id === site.id);
    const scored = scoreSite(site, ranges, cache, row.predictions);
    if (scored.status !== 'measured') throw Error('Changed support at ' + site.sample + ' ' + site.id);
    for (const measures of Object.values(scored.transferred)) {
      if (!Number.isFinite(measures.dog) || !Number.isFinite(measures.raw)) throw Error('Undefined correlation');
    }
    intensities.push({sample:site.sample, site:site.id, planes:scored.heldout_planes, ncc:scored.transferred});
  }
  function median(values) {
    const a = [...values].sort((v,w) => v-w);
    const i = (a.length - 1)/2;
    return (a[Math.floor(i)] + a[Math.ceil(i)])/2;
  }
  const summary = {};
  for (const [name, curve] of Object.entries(result)) {
    const measured = intensities.filter(r => r.sample === name);
    const correlations = {};
    for (const mode of ['raw','dog']) {
      correlations[mode] = {};
      for (const method of ['original','median','linear']) {
        const gain = measured.map(r => r.ncc[method][mode] - r.ncc.original[mode]);
        correlations[mode][method] = {median_ncc:median(measured.map(r => r.ncc[method][mode])),
          paired_median_gain:median(gain), strictly_improved:gain.filter(g => g>1e-10).length,
          equal:gain.filter(g => Math.abs(g)<=1e-10).length};
      }
    }
    summary[name] = {measured:measured.length, errors:curve.scores, correlations};
  }
  saveAtomic(path.join(cache,'height-transfer.json'), JSON.stringify({summary,sites:intensities},null,2)+'\n');
  console.log(JSON.stringify(summary,null,2));
}
if (require.main === module) main().catch(error => {console.error(error);process.exitCode=1;});
module.exports = {leaveOneHeightOut};
