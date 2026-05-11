"""Tests for save/load of per-rep stats + aggregation."""

import numpy as np
import pytest

from validation._lib.io import (
    save_rep_stats,
    load_rep_stats,
    aggregate_track,
)


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "rep_000" / "stats.npz"
    save_rep_stats(
        path,
        pi__A=np.array([1.0, 2.0, 3.0]),
        dxy__A_B=np.array([4.0, 5.0, 6.0]),
        timing_seconds=12.34,
    )
    loaded = load_rep_stats(path)
    np.testing.assert_array_equal(loaded["pi__A"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(loaded["dxy__A_B"], [4.0, 5.0, 6.0])
    assert float(loaded["timing_seconds"]) == pytest.approx(12.34)


def test_aggregate_three_reps(tmp_path):
    track_dir = tmp_path / "track_test"
    for r in range(3):
        path = track_dir / f"rep_{r:03d}" / "stats.npz"
        save_rep_stats(
            path,
            pi__A=np.array([float(r), float(r) + 1.0]),
        )
    agg = aggregate_track(track_dir)
    np.testing.assert_array_equal(agg["pi__A"], [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])


def test_aggregate_skips_missing(tmp_path):
    track_dir = tmp_path / "track_partial"
    save_rep_stats(track_dir / "rep_000" / "stats.npz", pi__A=np.array([1.0]))
    save_rep_stats(track_dir / "rep_002" / "stats.npz", pi__A=np.array([3.0]))
    agg = aggregate_track(track_dir)
    np.testing.assert_array_equal(agg["pi__A"], [[1.0], [3.0]])
    assert agg["__rep_indices__"].tolist() == [0, 2]
