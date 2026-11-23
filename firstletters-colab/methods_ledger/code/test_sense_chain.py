"""Regression tests for storage-invariant traversal, using analytic geometry.

No manuscript coordinates or model outputs are used. The sparse fixture exposes
transverse coordinates only, as a range-fetching real-data client can do. Exact
reciprocal crossings are rebuilt when any point array is reversed. Drift on a
known Archimedean spiral provides the oracle; comparison with the same algorithm
on reordered input alone would not establish the physical sign convention.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest

import numpy as np

from sense_chain import trace_winding_sense


class Tracks:
    def __init__(self):
        self.points, self.family, self.crossings = {}, {}, {}

    def add(self, number, points, family=0):
        self.points[number] = np.asarray(points, dtype=float)
        self.family[number] = family
        self.crossings[number] = []

    def connect(self, a, ia, b, ib, connector, count=9):
        pts = np.linspace(self.points[a][ia], self.points[b][ib], count)
        self.add(connector, pts, family=1)
        self.crossings[a].append((connector, ia, 0))
        self.crossings[b].append((connector, ib, count - 1))
        self.crossings[connector] += [(a, 0, ia), (b, count - 1, ib)]

    def reordered(self, reverse=(), shuffle_seed=None, duplicate=False):
        out = copy.deepcopy(self)
        reverse = set(reverse)
        for t in reverse:
            out.points[t] = out.points[t][::-1].copy()
        rng = np.random.default_rng(shuffle_seed)
        for t, records in self.crossings.items():
            remapped = [
                (p, len(self.points[t]) - 1 - i if t in reverse else i,
                 len(self.points[p]) - 1 - j if p in reverse else j)
                for p, i, j in records
            ]
            if duplicate:
                remapped += list(remapped)
            if shuffle_seed is not None:
                rng.shuffle(remapped)
            out.crossings[t] = remapped
        return out

    def run(self, **kwargs):
        def coordinates_for(t):
            if self.family[t] != 0:
                raise AssertionError("longitudinal coordinates must not be loaded")
            return self.points[t]

        def crossings_for(t):
            rows = np.asarray(self.crossings[t], dtype=np.int64).reshape(-1, 3)
            return tuple(rows[:, k] for k in range(3))

        options = dict(z0=0, dz=40, zstep=16, rmax=2000)
        options.update(kwargs)
        return trace_winding_sense(
            [t for t in self.points if self.family[t] == 0],
            coordinates_for, crossings_for, self.family.__getitem__,
            lambda t: len(self.points[t]), [{"z": 0, "y": 0, "x": 0}], **options,
        )

    def save(self, root):
        ids = sorted(self.points)
        if ids != list(range(len(ids))):
            raise ValueError("CLI fixture requires contiguous IDs")
        arrays = [np.rint(self.points[t]).astype("<i4") for t in ids]
        np.r_[0, np.cumsum([len(a) for a in arrays])].astype("<i8").tofile(root / "offsets.i64")
        np.concatenate(arrays).tofile(root / "coordinates.i32")
        np.asarray([[a[:, 0].min(), a[:, 0].max()] for a in arrays], dtype="<i4").tofile(root / "z_bounds.i32")
        np.asarray([self.family[t] for t in ids], dtype="i1").tofile(root / "family_codes.i8")
        rows = np.asarray([row for t in ids for row in self.crossings[t]], dtype=np.int64).reshape(-1, 3)
        np.savez(root / "crossings.npz", offsets=np.r_[0, np.cumsum([len(self.crossings[t]) for t in ids])],
                 partners=rows[:, 0], self_local=rows[:, 1], partner_local=rows[:, 2])
        (root / "umbilicus.json").write_text(json.dumps({"control_points": [{"z": 0, "y": 0, "x": 0}]}))


def spiral(start=0, end=2 * np.pi + 0.2, pitch=20, z=0, taper=0, n=721, radial_offset=0):
    theta = np.linspace(start, end, n)
    radius = 1000 + pitch * theta / (2 * np.pi) + taper * z + radial_offset
    return np.column_stack([np.full(n, z), radius * np.sin(theta), radius * np.cos(theta)])


def one_turn(pitch=20, start=0):
    fixture = Tracks()
    fixture.add(0, spiral(start, start + 2 * np.pi + 0.2, pitch=pitch))
    return fixture


def linked(pitch=20, zs=(0, 8, 16), taper=0, offsets=(0, 0, 0)):
    fixture = Tracks()
    fixture.add(0, spiral(-.2, 2.4, pitch, zs[0], taper, n=261, radial_offset=offsets[0]))
    fixture.add(1, spiral(2.0, 4.7, pitch, zs[1], taper, n=271, radial_offset=offsets[1]))
    fixture.add(2, spiral(4.3, 6.6, pitch, zs[2], taper, n=231, radial_offset=offsets[2]))
    fixture.connect(0, 240, 1, 20, 3)
    fixture.connect(1, 250, 2, 20, 4)
    return fixture


class WindingSenseTests(unittest.TestCase):
    def test_analytic_cw_and_acw_signs(self):
        for pitch in (-20, 20):
            for mirror in (-1, 1):
                with self.subTest(pitch=pitch, mirror=mirror):
                    summary, rows = one_turn(pitch).run(mirror=mirror)
                    self.assertEqual(summary["chains_full_turn"], 1)
                    np.testing.assert_allclose(rows[0], [pitch * mirror, 0, 0], atol=1e-10)
                    self.assertEqual(summary["exact_same_z"]["chains_full_turn"], 1)
                    self.assertIsNone(summary["winding_sense_verdict"])

    def test_whole_track_reversal_and_atan2_cut(self):
        for start in (0, np.pi - .03, -np.pi + .03, 2.7):
            for pitch in (-20, 20):
                for mirror in (-1, 1):
                    with self.subTest(start=start, pitch=pitch, mirror=mirror):
                        fixture = one_turn(pitch, start)
                        base, rows = fixture.run(mirror=mirror)
                        changed, other = fixture.reordered([0]).run(mirror=mirror)
                        self.assertEqual(base, changed)
                        np.testing.assert_array_equal(rows, other)
                        np.testing.assert_allclose(rows[0, 0], pitch * mirror, atol=1e-10)

    def test_exact_reciprocal_multitrack_chain(self):
        fixture = linked()
        for t, records in fixture.crossings.items():
            for partner, local, other in records:
                np.testing.assert_array_equal(fixture.points[t][local], fixture.points[partner][other])
                self.assertIn((t, other, local), fixture.crossings[partner])
        base, rows = fixture.run()
        self.assertEqual(base["completed_start_ids"], [0])
        np.testing.assert_allclose(rows, [[20, 16, 2]], atol=1e-10)
        self.assertEqual(base["exact_same_z"]["chains_full_turn"], 0)
        mirrored, other = fixture.run(mirror=-1)
        self.assertEqual(mirrored["completed_start_ids"], [2])
        np.testing.assert_allclose(other, [[-20, -16, 2]], atol=1e-10)

    def test_randomized_track_and_connector_storage_reversals(self):
        fixture = linked()
        rng = np.random.default_rng(1742)
        for mirror in (-1, 1):
            baseline, rows = fixture.run(mirror=mirror)
            for seed in range(30):
                reverse = [t for t in fixture.points if rng.integers(2)]
                changed, actual = fixture.reordered(reverse, shuffle_seed=seed).run(mirror=mirror)
                self.assertEqual(changed, baseline)
                np.testing.assert_array_equal(actual, rows)

    def test_duplicate_crossings_do_not_change_results(self):
        fixture = linked()
        baseline, rows = fixture.run()
        changed, actual = fixture.reordered([0, 2, 3, 4], shuffle_seed=3, duplicate=True).run()
        self.assertEqual(changed, baseline)
        np.testing.assert_array_equal(actual, rows)

    def test_equal_cost_branches_have_stable_choice(self):
        fixture = linked()
        alternate = fixture.points[1].copy()
        radius = np.hypot(alternate[:, 1], alternate[:, 2])
        extra = np.clip((np.arange(len(alternate)) - 20) / 100, 0, 1) * 4
        alternate[:, 1:] *= ((radius + extra) / radius)[:, None]
        fixture.add(5, alternate)
        ending = fixture.points[2].copy()
        radius = np.hypot(ending[:, 1], ending[:, 2])
        ending[:, 1:] *= ((radius + 4) / radius)[:, None]
        fixture.add(8, ending)
        fixture.connect(0, 240, 5, 20, 6)
        fixture.connect(5, 250, 8, 20, 7)
        baseline, rows = fixture.run()
        self.assertEqual(baseline["completed_start_ids"], [0])
        np.testing.assert_allclose(rows[0, 0], 20, atol=1e-10)
        # Make the alternative distinguishable: without the lower-ID branch
        # the exact same first crossing must instead produce drift +24.
        alternate_only = copy.deepcopy(fixture)
        alternate_only.crossings[0] = [r for r in alternate_only.crossings[0] if r[0] != 3]
        alternate_only.crossings[3] = [r for r in alternate_only.crossings[3] if r[0] != 0]
        np.testing.assert_allclose(alternate_only.run()[1][0, 0], 24, atol=1e-10)
        for seed in range(12):
            changed, actual = fixture.reordered([0, 1, 3, 5, 7, 8], shuffle_seed=seed, duplicate=True).run()
            self.assertEqual(changed, baseline)
            np.testing.assert_array_equal(actual, rows)

    def test_axial_taper_confound_is_not_hidden_by_mirroring(self):
        fixture = linked(pitch=20, zs=(0, 16, 32), taper=-1)
        ordinary, rows = fixture.run(rtol=20)
        mirrored, other = fixture.run(rtol=20, mirror=-1)
        np.testing.assert_allclose(rows, [[-12, 32, 2]], atol=1e-10)
        np.testing.assert_allclose(other, [[12, -32, 2]], atol=1e-10)
        self.assertEqual(ordinary["n_abs_dr_5_60_neg"], 1)
        self.assertEqual(mirrored["n_abs_dr_5_60_pos"], 1)
        self.assertEqual(ordinary["exact_same_z"]["chains_full_turn"], 0)
        self.assertIsNone(ordinary["winding_sense_verdict"])

    def test_exact_same_z_return_removes_analytic_taper_term(self):
        fixture = linked(pitch=20, zs=(0, 16, 0), taper=-1)
        summary, rows = fixture.run(rtol=20)
        np.testing.assert_allclose(rows, [[20, 0, 2]], atol=1e-10)
        self.assertEqual(summary["exact_same_z"]["n_abs_dr_5_60_pos"], 1)
        self.assertEqual(summary["nonzero_dz_closures"], 0)

    def test_hop_filters_keep_their_units_and_roles(self):
        fixture = linked()
        self.assertEqual(fixture.run(jmax=8, zstep=8, front=.9)[0]["chains_full_turn"], 1)
        for parameters in (dict(jmax=7), dict(zstep=7), dict(front=.95), dict(rmax=999)):
            with self.subTest(parameters=parameters):
                self.assertEqual(fixture.run(**parameters)[0]["chains_full_turn"], 0)
        shifted = linked(offsets=(0, 4, 4))
        self.assertEqual(shifted.run(rtol=3.9)[0]["chains_full_turn"], 0)
        np.testing.assert_allclose(shifted.run(rtol=4.1)[1][0, 0], 24, atol=1e-10)

    def test_hop_limit_and_no_continuation_are_counted(self):
        summary, rows = linked().run(max_hops=1)
        self.assertEqual(rows.shape, (0, 3))
        self.assertEqual(summary["chains_started"], summary["broken"])
        self.assertGreater(summary["broken_reasons"]["hop_limit"], 0)

    def test_empty_or_zero_start_selection(self):
        summary, rows = Tracks().run()
        self.assertEqual(rows.shape, (0, 3))
        self.assertEqual(summary["chains_started"], 0)
        self.assertIsNone(summary["median_dr"])
        self.assertEqual(one_turn().run(nstart=0)[1].shape, (0, 3))

    def test_outcome_digest_encoding_includes_completions_and_failures(self):
        prefix = b"sense-chain-terminal-outcomes-v1\0"
        summary, rows = one_turn().run()
        encoded = prefix + b"C" + struct.pack("<qddq", 0, rows[0, 0], 0.0, 0)
        self.assertEqual(summary["per_start_outcome_sha256"], hashlib.sha256(encoded).hexdigest())
        incomplete = Tracks()
        incomplete.add(0, spiral(end=np.pi))
        summary, _ = incomplete.run()
        reason = b"no_continuation"
        encoded = prefix + b"B" + struct.pack("<qqdI", 0, 0, 0.0, len(reason)) + reason
        self.assertEqual(summary["per_start_outcome_sha256"], hashlib.sha256(encoded).hexdigest())
        self.assertEqual(Tracks().run()[0]["per_start_outcome_sha256"], hashlib.sha256(prefix).hexdigest())

    def test_outcome_digest_detects_a_per_start_swap_hidden_by_aggregates(self):
        fixture = one_turn(pitch=20)
        fixture.add(1, spiral(pitch=-20))
        changed = copy.deepcopy(fixture)
        changed.points[0], changed.points[1] = changed.points[1], changed.points[0]
        before, _ = fixture.run()
        after, _ = changed.run()
        before_hash = before.pop("per_start_outcome_sha256")
        after_hash = after.pop("per_start_outcome_sha256")
        self.assertEqual(before, after)
        self.assertNotEqual(before_hash, after_hash)

    def test_invalid_transverse_geometry_and_indices_fail(self):
        fixtures = []
        variable_z = linked()
        variable_z.points[0][1, 0] = 1
        fixtures.append(variable_z)
        nonfinite = linked()
        nonfinite.points[0][1, 1] = np.nan
        fixtures.append(nonfinite)
        bad_local = linked()
        bad_local.crossings[0][0] = (3, 99999, 0)
        fixtures.append(bad_local)
        negative_partner = linked()
        negative_partner.crossings[0][0] = (-1, 240, 0)
        fixtures.append(negative_partner)
        for fixture in fixtures:
            with self.subTest(fixture=fixtures.index(fixture)), self.assertRaises(ValueError):
                fixture.run()

    def test_cli_preserves_positional_interface_and_result_shape(self):
        fixture = one_turn()
        with tempfile.TemporaryDirectory(prefix="sense-chain-test-") as name:
            root = Path(name)
            fixture.reordered([0]).save(root)
            process = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("sense_chain.py")), str(root),
                 str(root / "crossings.npz"), str(root / "umbilicus.json"), str(root / "out.npy"),
                 "0", "40", "3000", "6", ".3", "200", "16", "2000", "-1"],
                check=True, capture_output=True, text=True,
            )
            summary = json.loads(process.stdout)
            rows = np.load(root / "out.npy")
            self.assertEqual(summary["chains_full_turn"], 1)
            self.assertEqual(rows.shape, (1, 3))
            self.assertAlmostEqual(rows[0, 0], -20, delta=.6)


if __name__ == "__main__":
    unittest.main()
