#!/usr/bin/env node
'use strict';

const volumes = require('./volumes.json');
const {fixedXYZ, fitDisplacements, measuredRows, readResidualTable, RESIDUALS_SHA256} =
  require('./height-transfer.cjs');

const PITCH_UM = 9.362;
const USAGE = 'Usage: node registration-checks/annular/predict.cjs PHerc1203|PHerc0846A X Y Z\n' +
  'X Y Z are fixed (9.362 µm) level-0 voxel coordinates, in that order.';

function difference(a, b) {
  return a.map((value, axis) => value - b[axis]);
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]];
}

function insideHull(point, sites) {
  // A 3D convex hull has a supporting face through at least three of its sites.
  // Both frozen scroll samples span 3D. A point on any face counts as inside.
  for (let i = 0; i < sites.length - 2; i++) {
    const origin = sites[i];
    const relative = sites.map(site => difference(site, origin));
    const query = difference(point, origin);
    for (let j = i + 1; j < sites.length - 1; j++) {
      for (let k = j + 1; k < sites.length; k++) {
        const normal = cross(relative[j], relative[k]);
        const tolerance = 1e-7 * Math.hypot(...normal);  // fixed voxels × face area
        const side = dot(normal, query);
        if (side > tolerance && relative.every(v => dot(normal, v) <= tolerance)) return false;
        if (side < -tolerance && relative.every(v => dot(normal, v) >= -tolerance)) return false;
      }
    }
  }
  return true;
}

function predictAt(sample, xyz, rows) {
  if (!Object.hasOwn(volumes, sample)) throw Error('Unknown scroll. ' + USAGE);
  if (xyz.length !== 3 || !xyz.every(v => Number.isFinite(v) && v >= 0 && v <= Number.MAX_SAFE_INTEGER)) {
    throw Error('Expected three finite, nonnegative fixed voxel coordinates. ' + USAGE);
  }
  const sites = rows.filter(row => row.sample === sample);
  if (!sites.length) throw Error('No measured calibration sites for ' + sample);
  const positions = sites.map(fixedXYZ);
  const shift = fitDisplacements(sites)(xyz).linear;
  const distances = positions.map(p => Math.hypot(...difference(xyz, p)));
  const nearest = distances.indexOf(Math.min(...distances));
  return {
    sample,
    fixed_volume: volumes[sample].coarse,
    moving_volume: volumes[sample].fine,
    fixed_xyz_vox: xyz,
    fixed_voxel_um: PITCH_UM,
    correction_xyz_fixed_vox: shift,
    correction_xyz_um: shift.map(value => value * PITCH_UM),
    direction: 'Add to moving anatomy after the nominal fine-to-fixed affine.',
    calibration: {
      sites: sites.length,
      hull: insideHull(xyz, positions) ? 'inside_or_on' : 'outside',
      nearest_site: sites[nearest].site,
      nearest_distance_mm: distances[nearest] * PITCH_UM / 1000,
      residuals_sha256: RESIDUALS_SHA256,
    },
  };
}

function main(args) {
  if (args.length === 1 && ['-h', '--help'].includes(args[0])) {
    console.log(USAGE);
    return;
  }
  if (args.length !== 4) throw Error(USAGE);
  const xyz = args.slice(1).map(value => value.trim() ? Number(value) : NaN);
  const result = predictAt(args[0], xyz, measuredRows(readResidualTable()));
  console.log(JSON.stringify(result, null, 2));
}

if (require.main === module) {
  try {
    main(process.argv.slice(2));
  } catch (err) {
    console.error(err.message);
    process.exitCode = 2;
  }
}

module.exports = {predictAt};
