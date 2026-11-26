#!/usr/bin/env python3
"""Recalculate the compact published geometry results offline; NumPy only.

This verifies recorded numeric consistency. A full geometric replay additionally
requires prepare.py/run.py and their hash-pinned public sources. Neither check
measures ink recall or authorizes pruning.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=HERE):
    result = json.loads((root / "result.json").read_bytes())
    assert result["protocol_sha256"] == sha(root / "PROTOCOL.md")
    assert result["script_sha256"] == sha(root / "run.py")
    assert result["maps_sha256"] == sha(root / "numeric_maps.npz")
    assert result["scientific_pruning_admission"] == "NOT_ESTABLISHED"
    assert result["new_ink_inferences"] == result["new_discovered_letters"] == 0
    assert all(result["synthetic"]["checks"].values())
    expected = {(s, a, b) for s in ("fitted", "snap", "flat")
                for a in ("w033", "w034", "w035", "w036")
                for b in ("w033", "w034", "w035", "w036")}
    assert {(r["state"], r["source"], r["target"]) for r in result["pairs"]} == expected
    assert len(result["pairs"]) == 48
    totals = {s: {k: 0 for k in ("accepted", "separation_gt3", "separation_gt6", "source_near",
                                 "near_to_far", "near_to_unknown", "near_to_intermediate", "near_to_near")}
              for s in ("fitted", "snap", "flat")}
    self_total = self_rejected = 0
    with np.load(root / "numeric_maps.npz", allow_pickle=False) as maps:
        assert len(maps.files) == 48 * 9
        for row in result["pairs"]:
            stem = f"{row['state']}_{row['source']}_{row['target']}"
            def get(suffix):
                return maps[stem + "_" + suffix]
            ids, match, fixed = get("source_ids"), get("match"), get("preserved_match")
            ca, cb = get("source_reference"), get("target_reference")
            depths_a, depths_b = get("source_reference_depth"), get("target_reference_depth")
            grid, separation = get("classification"), get("ray_separation")
            n = row["n_source"]
            assert ids.shape == match.shape == fixed.shape == ca.shape == cb.shape == depths_a.shape == depths_b.shape == (n,)
            assert match.dtype == fixed.dtype == np.bool_
            assert ca.dtype == cb.dtype == grid.dtype == np.uint8
            assert ids.dtype == np.int32 and len(np.unique(ids)) == n
            assert (ids >= 0).all() and (ids < grid.size).all()
            assert int(match.sum()) == row["n_accepted"]
            assert float(match.mean()) == row["accepted_fraction"]
            assert int(fixed.sum()) == row["preserved_n_accepted"]
            assert int((fixed & ~match).sum()) == row["preserved_added"]
            assert int((match & ~fixed).sum()) == row["preserved_removed"]
            assert ((ca <= 6) & (cb <= 6)).all()
            assert (ca[~match] == 0).all() and (cb[~match] == 0).all()
            assert np.isnan(depths_a[~match]).all() and np.isnan(depths_b[~match]).all()
            assert separation.shape == grid.shape and separation.dtype == np.float32
            assert np.isfinite(separation.ravel()[ids[match]]).all()
            assert int(np.isfinite(separation).sum()) == int(match.sum())
            table = np.zeros((7, 7), np.int64)
            np.add.at(table, (ca[match], cb[match]), 1)
            assert table.tolist() == row["reference_confusion_7x7"]
            near = match & (ca == 4)
            for code, name in ((6, "far"), (5, "intermediate"), (4, "near")):
                assert int((near & (cb == code)).sum()) == row["source_near_target_" + name]
            unknown = near & np.isin(cb, [0, 1, 2, 3])
            assert int(unknown.sum()) == row["source_near_target_unknown"]
            assert int(near.sum()) == row["source_near"]
            expected_grid = np.zeros(grid.shape, np.uint8)
            expected_grid.ravel()[ids[match]] = 1
            expected_grid.ravel()[ids[near]] = 2
            expected_grid.ravel()[ids[near & (cb == 6)]] = 3
            expected_grid.ravel()[ids[unknown]] = 4
            expected_grid.ravel()[ids[near & (cb == 5)]] = 5
            assert np.array_equal(grid, expected_grid)
            for threshold in (1, 3, 6):
                count = int((separation.ravel()[ids[match]] > threshold).sum())
                assert count == row[f"accepted_ray_separation_gt{threshold}"]
            # A finite geometric reference classification implies a finite depth.
            for codes, depths in ((ca, depths_a), (cb, depths_b)):
                for code in (4, 5, 6):
                    assert np.isfinite(depths[codes == code]).all()
                assert (np.abs(depths[codes == 4]) <= 3).all()
                assert (np.abs(depths[codes == 6]) >= 6).all()
            assert len(row["all_near_to_far_examples"]) == row["source_near_target_far"]
            if row["self_pair"]:
                assert row["source"] == row["target"] and fixed.all()
                self_total += n
                self_rejected += int((~match).sum())
            else:
                total = totals[row["state"]]
                for dest, source in (("accepted", "n_accepted"), ("separation_gt3", "accepted_ray_separation_gt3"),
                                     ("separation_gt6", "accepted_ray_separation_gt6"), ("source_near", "source_near"),
                                     ("near_to_far", "source_near_target_far"), ("near_to_unknown", "source_near_target_unknown"),
                                     ("near_to_intermediate", "source_near_target_intermediate"), ("near_to_near", "source_near_target_near")):
                    total[dest] += row[source]
    return {"passed": True, "pairs_verified": 48, "self_vertices": self_total,
            "self_rejected_before_correction": self_rejected, "state_totals_nonself": totals,
            "new_model_forwards": 0, "scientific_pruning_admission": "NOT_ESTABLISHED",
            "scope": "Offline numeric consistency, not a new full geometry replay or ink validation"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
