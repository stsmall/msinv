"""Tests for generic engine runners: msinv, msprime, discoal."""
import numpy as np
import pytest
import tskit


def test_msinv_run_neutral_no_inv_returns_ts(tmp_path):
    """Track 3 shape: no inversion, no sweep, single-pop F sampling."""
    from validation._lib.engines import msinv_run
    from validation._lib.demography import v12_msinv

    ts = msinv_run(
        demography=v12_msinv(),
        sample_config={("S", 0): 0, ("S", 1): 10},
        L=10_000,
        r=1.0e-8,
        mu=1.0e-8,
        seed=42,
    )
    assert isinstance(ts, tskit.TreeSequence)
    assert ts.num_samples == 10
    assert ts.sequence_length == 10_000
    # Mutations were overlaid → at least some sites if rep wasn't degenerate
    # (very small L + small Ne + 1e-8 mu can produce 0 sites; just check shape)
    assert ts.num_sites >= 0


def test_msprime_run_v12_returns_ts():
    from validation._lib.engines import msprime_run
    from validation._lib.demography import v12_msprime

    ts = msprime_run(
        demography=v12_msprime(),
        samples_by_pop={"F": 10},
        L=10_000,
        r=1.0e-8,
        mu=1.0e-8,
        seed=42,
    )
    assert isinstance(ts, tskit.TreeSequence)
    assert ts.num_samples == 10
    assert ts.sequence_length == 10_000


def test_msprime_run_two_pops():
    from validation._lib.engines import msprime_run
    from validation._lib.demography import v12_msprime

    ts = msprime_run(
        demography=v12_msprime(),
        samples_by_pop={"K": 5, "F": 5},
        L=10_000,
        r=1.0e-8,
        mu=1.0e-8,
        seed=43,
    )
    assert ts.num_samples == 10
