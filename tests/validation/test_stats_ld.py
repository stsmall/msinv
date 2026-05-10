"""Tests for LD r²-decay binning."""
import numpy as np
import msprime
import pytest

from validation._lib.stats import ld_decay


@pytest.fixture
def ld_ts():
    ts = msprime.sim_ancestry(
        samples=20, population_size=1000,
        sequence_length=100_000, recombination_rate=1e-7,
        random_seed=21, ploidy=1)
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=22)
    return ts


def test_ld_decay_shape(ld_ts):
    bins = np.logspace(2, 5, 11)  # 10 bins from 100 to 1e5 bp
    out = ld_decay(ld_ts, distance_bins=bins, max_pairs=2000, seed=0)
    assert out["bin_edges"].shape == (11,)
    assert out["mean_r2"].shape == (10,)
    assert out["count"].shape == (10,)


def test_ld_decay_values_in_unit(ld_ts):
    bins = np.logspace(2, 5, 11)
    out = ld_decay(ld_ts, distance_bins=bins, max_pairs=2000, seed=0)
    valid = ~np.isnan(out["mean_r2"])
    assert ((out["mean_r2"][valid] >= 0) & (out["mean_r2"][valid] <= 1)).all()


def test_ld_decay_decreases_with_distance(ld_ts):
    """At a moderate recomb rate, r² should generally decrease with distance.

    Use larger sample for better signal-to-noise; this is a soft check
    (not strictly monotonic at small N, but mean of bins 0-1 > mean of 8-9).
    """
    bins = np.logspace(2, 5, 11)
    out = ld_decay(ld_ts, distance_bins=bins, max_pairs=5000, seed=0)
    near = np.nanmean(out["mean_r2"][:2])
    far = np.nanmean(out["mean_r2"][-2:])
    assert near > far
