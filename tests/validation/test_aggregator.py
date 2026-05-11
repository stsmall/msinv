"""Tests for the equivalence aggregator."""
import numpy as np
import pytest

from validation._lib.aggregator import track_equivalence_table
from validation._lib.io import save_rep_stats


def _seed_rep_dir(track_dir, n_reps, stat_value_fn):
    """Helper: write a fake track directory with n_reps stats.npz files.

    stat_value_fn(rep) -> dict of stat-name -> np.ndarray.
    """
    for rep in range(n_reps):
        save_rep_stats(
            track_dir / f"rep_{rep:03d}" / "stats.npz",
            **stat_value_fn(rep),
        )


def test_aggregator_identical_dists_all_equivalent(tmp_path):
    """Same generator on both sides → every stat equivalent."""
    rng = np.random.default_rng(0)
    dir_a = tmp_path / "engine_a"
    dir_b = tmp_path / "engine_b"
    # Use the same RNG seed shape so distributions are identical-ish.
    _seed_rep_dir(dir_a, 30, lambda r: {
        "pi__F_S": rng.normal(0.001, 1e-4, size=20),
    })
    rng = np.random.default_rng(0)
    _seed_rep_dir(dir_b, 30, lambda r: {
        "pi__F_S": rng.normal(0.001, 1e-4, size=20),
    })
    table = track_equivalence_table(dir_a, dir_b)
    assert "pi__F_S" in table
    assert table["pi__F_S"]["verdict"] == "equivalent"


def test_aggregator_shifted_dists_not_equivalent(tmp_path):
    """Mean shift of >1 SD on the engine_b side → not_equivalent."""
    rng = np.random.default_rng(1)
    dir_a = tmp_path / "engine_a"
    dir_b = tmp_path / "engine_b"
    _seed_rep_dir(dir_a, 50, lambda r: {
        "pi__F_S": rng.normal(0.0, 1.0, size=40),
    })
    rng = np.random.default_rng(2)
    _seed_rep_dir(dir_b, 50, lambda r: {
        "pi__F_S": rng.normal(2.0, 1.0, size=40),  # 2 SD shift
    })
    table = track_equivalence_table(dir_a, dir_b)
    assert table["pi__F_S"]["verdict"] == "not_equivalent"


def test_aggregator_returns_ks_and_d(tmp_path):
    rng = np.random.default_rng(3)
    dir_a = tmp_path / "engine_a"
    dir_b = tmp_path / "engine_b"
    _seed_rep_dir(dir_a, 20, lambda r: {
        "pi__F_S": rng.normal(0.0, 1.0, size=10),
    })
    _seed_rep_dir(dir_b, 20, lambda r: {
        "pi__F_S": rng.normal(0.0, 1.0, size=10),
    })
    table = track_equivalence_table(dir_a, dir_b)
    row = table["pi__F_S"]
    assert "ks_stat" in row
    assert "ks_p" in row
    assert "cohens_d" in row
    assert row["verdict"] in ("equivalent", "not_equivalent", "investigate")
