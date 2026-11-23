'use strict';

const SIZE = 27;
const CENTER = 13;
const FIXED_VOXEL_UM = 9.362;
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[
0]];
const unit = vector => vector.map(v => v / Math.hypot(...vector));

function percentile(values, p) {
  if (!values.length) return null;
  const ordered = [...values].sort((a, b) => a - b);
  const at = (ordered.length - 1) * p,
    lo = Math.floor(at);
  return ordered[lo] + (ordered[Math.ceil(at)] - ordered[lo]) * (at - lo);
}

function axialAngleDegrees(first, second) {
  return Math.acos(Math.min(1, Math.abs(dot(first, second)))) * 180 / Math.PI;
}

function gaussianKernel() {
  const weights = Array.from({
    length: 7
  }, (_, i) => Math.exp(-((i - 3) ** 2) / 2));
  const sum = weights.reduce((a, b) => a + b, 0);
  return weights.map(weight => weight / sum);
}

function gradientField(intensity) {
  if (intensity.length !== SIZE ** 3) throw Error('Expected a 27-cubed CT neighborhood');
  const weights = gaussianKernel();
  let smooth = intensity;
  for (const step of [1, SIZE, SIZE * SIZE]) {
    const dest = new Float32Array(smooth.length);
    for (let i = 0; i < smooth.length; i++) {
      const position = Math.floor(i / step) % SIZE;
      for (let d = -3; d <= 3; d++) {
        const q = Math.max(0, Math.min(SIZE - 1, position + d));
        dest[i] += weights[d + 3] * smooth[i + (q - position) * step];
      }
    }
    smooth = dest;
  }
  const grad = Array.from({
    length: 3
  }, () => new Float32Array(smooth.length));
  for (let z = 1; z < SIZE - 1; z++)
    for (let y = 1; y < SIZE - 1; y++)
      for (let x = 1; x < SIZE - 1; x++) {
        const at = (z * SIZE + y) * SIZE + x;
        grad[0][at] = (smooth[at + 1] - smooth[at - 1]) / 2;
        grad[1][at] = (smooth[at + SIZE] - smooth[at - SIZE]) / 2;
        grad[2][at] = (smooth[at + SIZE * SIZE] - smooth[at - SIZE * SIZE]) / 2;
      }
  return {
    grad,
    intensity
  };
}

function eigensystem(symmetric) {
  const a = symmetric.map(row => [...row]);
  const axes = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ];
  for (let iter = 0; iter < 32; iter++) {
    let p = 0,
      q = 1;
    for (const [i, j] of [
        [0, 2],
        [1, 2]
      ])
      if (Math.abs(a[i][j]) > Math.abs(a[p][q])) {
        p = i;
        q = j;
      }
    const scale = Math.abs(a[0][0]) + Math.abs(a[1][1]) + Math.abs(a[2][2]) + 1;
    if (Math.abs(a[p][q]) < 1e-12 * scale) break;
    const theta = .5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
    const c = Math.cos(theta),
      s = Math.sin(theta);
    const app = a[p][p],
      aqq = a[q][q],
      apq = a[p][q];
    for (let k = 0; k < 3; k++)
      if (k !== p && k !== q) {
        const kp = a[k][p],
          kq = a[k][q];
        a[k][p] = a[p][k] = c * kp - s * kq;
        a[k][q] = a[q][k] = s * kp + c * kq;
      }
    a[p][p] = c * c * app - 2 * c * s * apq + s * s * aqq;
    a[q][q] = s * s * app + 2 * c * s * apq + c * c * aqq;
    a[p][q] = a[q][p] = 0;
    for (const row of axes) {
      const vp = row[p],
        vq = row[q];
      row[p] = c * vp - s * vq;
      row[q] = s * vp + c * vq;
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[j][j] - a[i][i]);
  const eig = order.map(i => Math.max(0, a[i][i]));
  let axis = unit(axes.map(r => r[order[0]]));
  const pivot = [0, 1, 2].sort((i, j) => Math.abs(axis[j]) - Math.abs(axis[i]))[0];
  if (axis[pivot] < 0) axis = axis.map(v => -v);
  return {
    values: eig,
    axis
  };
}

function tensor(field, center, radius, sigma, minGap) {
  const j = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];
  let wsum = 0,
    count = 0,
    material = 0;
  for (let dz = -radius; dz <= radius; dz++)
    for (let dy = -radius; dy <= radius; dy++)
      for (let dx = -radius; dx <= radius; dx++) {
        const index = ((CENTER + center[2] + dz) * SIZE + CENTER + center[1] + dy) * SIZE + CENTER +
          center[0] + dx;
        const w = Math.exp(-(dx * dx + dy * dy + dz * dz) / (2 * sigma * sigma));
        wsum += w;
        count++;
        material += field.intensity[index] > 3;
        const g = field.grad.map(a => a[index]);
        for (let i = 0; i < 3; i++)
          for (let k = i; k < 3; k++) j[i][k] += w * g[i] * g[k];
      }
  for (let i = 0; i < 3; i++)
    for (let k = i; k < 3; k++) j[k][i] = j[i][k] = j[i][k] / wsum;
  const {
    values,
    axis
  } = eigensystem(j);
  const coverage = material / count,
    rms = Math.sqrt(values.reduce((a, b) => a + b, 0));
  const separation = values[0] > 0 ? (values[0] - values[1]) / values[0] : 0;
  const issues = [];
  if (coverage < .70) issues.push('insufficient_material');
  if (rms < 1) issues.push('weak_gradients');
  if (separation < minGap) issues.push('ambiguous_eigenvalues');
  return {
    axis,
    values,
    coverage,
    rms,
    separation,
    issues
  };
}

function interpolateCube(cube, size, x, y, z) {
  const ix = Math.floor(x),
    iy = Math.floor(y),
    iz = Math.floor(z);
  const fx = x - ix,
    fy = y - iy,
    fz = z - iz,
    at = (iz * size + iy) * size + ix;
  let value = 0;
  for (let dz = 0; dz <= 1; dz++)
    for (let dy = 0; dy <= 1; dy++)
      for (let dx = 0; dx <= 1; dx++)
        value += cube[at + dx + dy * size + dz * size * size] * (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) *
        (dz ? fz : 1 - fz);
  return value;
}

function ridgePairs(profile) {
  const weights = gaussianKernel();
  const signal = profile.map((_, i) => weights.reduce((sum, w, k) => sum + w * profile[Math.max(0, Math
    .min(profile.length - 1, i + k - 3))], 0));
  const prominence = Math.max(10, .2 * (percentile(signal, .9) - percentile(signal, .1)));
  const peaks = [];
  for (let i = 3; i < signal.length - 3; i++) {
    if (signal[i] < signal[i - 1] || signal[i] <= signal[i + 1]) continue;
    const left = Math.min(...signal.slice(Math.max(0, i - 6), i)),
      right = Math.min(...signal.slice(i + 1, i + 7));
    if (signal[i] - Math.max(left, right) >= prominence) peaks.push(i);
  }
  const pairs = [];
  for (let i = 0; i < peaks.length; i++)
    for (let j = i + 1; j < peaks.length; j++) {
      const a = peaks[i],
        b = peaks[j];
      if (b - a < 4 || b - a > 18) continue;
      if (Math.min(signal[a], signal[b]) - Math.min(...signal.slice(a, b + 1)) >= prominence) pairs.push(
        [a - 18, b - 18]);
    }
  return pairs;
}

function repeatedRidges(coarseCube, axis) {
  const basis = [0, 0, 0];
  basis[[0, 1, 2].sort((a, b) => Math.abs(axis[a]) - Math.abs(axis[b]))[0]] = 1;
  const t1 = unit(cross(axis, basis)),
    t2 = cross(axis, t1);
  const paths = [
    [0, 0],
    [2.5, 0],
    [-2.5, 0],
    [0, 2.5],
    [0, -2.5]
  ].map(([u, v]) => {
    const profile = Array.from({
      length: 37
    }, (_, k) => {
      const xyz = axis.map((n, i) => 21 + (k - 18) * n + u * t1[i] + v * t2[i]);
      return interpolateCube(coarseCube, 43, ...xyz);
    });
    return ridgePairs(profile);
  });
  let votes = 0;
  for (const [a, b] of paths.flat()) {
    const support = paths.filter(pairs => pairs.some(([x, y]) => Math.abs(x - a) <= 2 && Math.abs(y -
      b) <= 2)).length;
    votes = Math.max(votes, support);
  }
  return {
    status: votes >= 3 ? 'repeated_material_ridges' : 'no_supported_pair',
    profiles_agreeing: votes
  };
}

function analyzeAxes(coarseInner, fineInner, coarseOuter) {
  const main = [0, 0, 0],
    coarseField = gradientField(coarseInner);
  const coarse = tensor(coarseField, main, 8, 4, .45);
  const fine = tensor(gradientField(fineInner), main, 8, 4, .45);
  const locals = [];
  for (let axis = 0; axis < 3; axis++)
    for (const step of [-4, 4]) {
      const c = [0, 0, 0];
      c[axis] = step;
      locals.push(tensor(coarseField, c, 3, 2, .35));
    }
  const supported = locals.filter(x => !x.issues.length);
  const localP90 = supported.length ? percentile(supported.map(x => axialAngleDegrees(coarse.axis, x
    .axis)), .90) : null;
  const pairAngles = [];
  for (let i = 0; i < supported.length; i++)
    for (let j = i + 1; j < supported.length; j++) pairAngles.push(axialAngleDegrees(supported[i].axis,
      supported[j].axis));
  const scanAngle = (!coarse.issues.length && !fine.issues.length) ? axialAngleDegrees(coarse.axis, fine
    .axis) : null;
  const issues = [...coarse.issues.map(x => 'coarse_' + x), ...fine.issues.map(x => 'fine_' + x)];
  if (supported.length < 4) issues.push('insufficient_supported_subwindows');
  if (localP90 !== null && localP90 > 25) issues.push('spatial_direction_disagreement');
  if (scanAngle !== null && scanAngle > 20) issues.push('acquisition_direction_disagreement');
  const ridges = coarse.issues.length ? {
    status: 'indeterminate_axis',
    profiles_agreeing: null
  } : repeatedRidges(coarseOuter, coarse.axis);
  return {
    reliable: issues.length === 0,
    issues,
    coarse,
    fine,
    scan_angle_degrees: scanAngle,
    n_supported_subwindows: supported.length,
    subwindow_p90_degrees: localP90,
    max_subwindow_pair_degrees: pairAngles.length ? Math.max(...pairAngles) : null,
    competing_local_directions: pairAngles.length ? Math.max(...pairAngles) >= 35 : null,
    ridges
  };
}

function project(vector, axis) {
  const norm2 = dot(vector, vector),
    normal = Math.abs(dot(vector, axis));
  const tangent = Math.sqrt(Math.max(0, norm2 - normal * normal));
  return {
    normal_um: normal * FIXED_VOXEL_UM,
    tangent_um: tangent * FIXED_VOXEL_UM,
    total_um: Math.sqrt(norm2) * FIXED_VOXEL_UM
  };
}
module.exports = {
  analyzeAxes,
  project,
  percentile,
  eigensystem,
  axialAngleDegrees
};
