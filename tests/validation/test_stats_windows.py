"""Tests for window-level stats: pi, dxy, Fst, Tajima's D, SFS."""

import numpy as np
import msprime
import pytest

from validation._lib.stats import window_stats, sfs


@pytest.fixture
def two_pop_ts():
    """Small msprime ts with 2 populations for stat testing."""
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=1000)
    demography.add_population(name="B", initial_size=1000)
    demography.set_migration_rate(source="A", dest="B", rate=1e-4)
    demography.set_migration_rate(source="B", dest="A", rate=1e-4)
    ts = msprime.sim_ancestry(
        samples={"A": 10, "B": 10},
        demography=demography,
        sequence_length=100_000,
        recombination_rate=1e-7,
        random_seed=42,
        ploidy=1,
    )
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=43)
    return ts


def test_window_stats_shape(two_pop_ts):
    ts = two_pop_ts
    pop_a = list(ts.samples(population=0))
    pop_b = list(ts.samples(population=1))
    out = window_stats(ts, sample_sets={"A": pop_a, "B": pop_b}, n_windows=40)
    assert out["pi"]["A"].shape == (40,)
    assert out["pi"]["B"].shape == (40,)
    assert out["dxy"]["A_B"].shape == (40,)
    assert out["fst"]["A_B"].shape == (40,)
    assert out["tajimas_d"]["A"].shape == (40,)


def test_window_stats_pi_positive(two_pop_ts):
    ts = two_pop_ts
    pop_a = list(ts.samples(population=0))
    out = window_stats(ts, sample_sets={"A": pop_a}, n_windows=40)
    assert (out["pi"]["A"] >= 0).all()


def test_window_stats_fst_in_range(two_pop_ts):
    ts = two_pop_ts
    pop_a = list(ts.samples(population=0))
    pop_b = list(ts.samples(population=1))
    out = window_stats(ts, sample_sets={"A": pop_a, "B": pop_b}, n_windows=40)
    fst = out["fst"]["A_B"]
    valid = ~np.isnan(fst)
    assert ((fst[valid] >= -0.01) & (fst[valid] <= 1.01)).all()


def test_sfs_shape(two_pop_ts):
    ts = two_pop_ts
    pop_a = list(ts.samples(population=0))
    s = sfs(ts, sample_set=pop_a, folded=True)
    n = len(pop_a)
    assert s.shape == (n // 2 + 1,)
    assert (s >= 0).all()


def test_sfs_unfolded_shape(two_pop_ts):
    ts = two_pop_ts
    pop_a = list(ts.samples(population=0))
    s = sfs(ts, sample_set=pop_a, folded=False)
    n = len(pop_a)
    assert s.shape == (n + 1,)
