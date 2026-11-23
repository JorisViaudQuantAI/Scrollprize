'use strict';
const assert = require('node:assert/strict');
const {
  analyzeAxes,
  project,
  axialAngleDegrees
} = require('./tensor.cjs');

function sampleCube(size, fun) {
  const cube = new Float32Array(size ** 3),
    h = (size - 1) / 2;
  for (let z = 0; z < size; z++)
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) cube[(z * size + y) * size + x] = fun(x - h, y - h, z - h);
  return cube;
}
const n = [.36, .48, .80],
  t = (x, y, z) => n[0] * x + n[1] * y + n[2] * z;
const plane = (x, y, z) => 120 + 65 * Math.tanh(t(x, y, z) / 1.5);
const twoRidges = (x, y, z) => {
  const u = t(x, y, z);
  return 70 + 115 * (Math.exp(-.5 * ((u - 5) / 1.4) ** 2) + Math.exp(-.5 * ((u + 5) / 1.4) ** 2));
};
const xplane = x => 120 + 65 * Math.tanh(x / 1.5),
  yplane = (_, y) => 120 + 65 * Math.tanh(y / 1.5);
const run = (coarse, fine = coarse) => analyzeAxes(sampleCube(27, coarse), sampleCube(27, fine),
  sampleCube(43, coarse));
const single = run(plane);
assert.ok(single.reliable);
assert.ok(axialAngleDegrees(single.coarse.axis, n) < 2);
assert.equal(single.ridges.status, 'no_supported_pair');
const repeated = run(twoRidges);
assert.ok(repeated.reliable);
assert.equal(repeated.ridges.status, 'repeated_material_ridges');
assert.equal(repeated.ridges.profiles_agreeing, 5);
assert.ok(run(() => 100).issues.includes('coarse_weak_gradients'));
assert.ok(run(xplane, yplane).issues.includes('acquisition_direction_disagreement'));
const crossing = run((x, y) => 100 + 35 * Math.sin(.7 * x) + 35 * Math.sin(.7 * y));
assert.ok(!crossing.reliable);
assert.ok(crossing.issues.includes('coarse_ambiguous_eigenvalues'));
const projected = project([3, 4, 0], [1, 0, 0]),
  reversed = project([3, 4, 0], [-1, 0, 0]);
assert.deepEqual(projected, reversed);
assert.ok(Math.abs(projected.normal_um - 3 * 9.362) < 1e-9);
assert.ok(Math.abs(projected.tangent_um - 4 * 9.362) < 1e-9);
console.log(
  'Synthetic plane, ridges, flat material, scan disagreement, competing textures and vector decomposition passed.'
  );
