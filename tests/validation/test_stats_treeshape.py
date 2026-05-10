"""Tests for tree-shape distributions (TMRCA, total branch, Colless)."""
import numpy as np
import msprime
import pytest

from validation._lib.stats import tree_shape_stats


@pytest.fixture
def small_ts():
    ts = msprime.sim_ancestry(
        samples=10, population_size=1000,
        sequence_length=10_000, recombination_rate=1e-7,
        random_seed=11, ploidy=1)
    return ts


def test_tree_shape_returns_three_dists(small_ts):
    out = tree_shape_stats(small_ts, n_samples=50)
    assert "tmrca" in out
    assert "total_branch" in out
    assert "colless" in out
    assert out["tmrca"].shape == (50,)
    assert out["total_branch"].shape == (50,)
    assert out["colless"].shape == (50,)


def test_tmrca_positive(small_ts):
    out = tree_shape_stats(small_ts, n_samples=20)
    assert (out["tmrca"] > 0).all()


def test_total_branch_positive(small_ts):
    out = tree_shape_stats(small_ts, n_samples=20)
    assert (out["total_branch"] > 0).all()


def test_colless_in_range(small_ts):
    """Colless index for n leaves is in [0, (n-1)*(n-2)/2]."""
    out = tree_shape_stats(small_ts, n_samples=20)
    n = small_ts.num_samples
    upper = (n - 1) * (n - 2) // 2
    assert (out["colless"] >= 0).all()
    assert (out["colless"] <= upper).all()
