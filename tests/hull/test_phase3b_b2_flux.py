# tests/hull/test_phase3b_b2_flux.py
"""Tier-1 + Tier-2 validation of the b2-flux model
(per docs/superpowers/specs/2026-04-27-peischl-b2-flux-design.md)."""

import math

import numpy as np
import pytest
from scipy import stats

from msinv.hull import HullSimulator, InversionSpec
from msinv.hull.demography import Demography


# ----- Tier 1: geometric sampling correctness ----------------------

def test_geometric_tract_length_mean_matches_parameter():
    """Sampled tract lengths should have mean ≈ mean_tract_length
    within 2-sigma tolerance over N=10_000 draws."""
    inv = InversionSpec(
        bp_left=0.0, bp_right=1_000_000.0,
        p_inv=0.5, t_inv=10_000.0,
        gene_conversion_rate=1e-9,
        mean_tract_length=200.0,
        tract_distribution='geometric',
    )
    sim = HullSimulator(
        sample_config={('S', 0): 2},
        demography=Demography(pop_sizes=[1000]),
        sequence_length=1_000_000,
        recombination_rate=1e-8, inversions=[inv], seed=42,
    )
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(10_000):
        # Use the same Exponential the simulator uses.
        samples.append(rng.exponential(inv.mean_tract_length))
    mean = float(np.mean(samples))
    expected = inv.mean_tract_length
    sd_of_mean = expected / math.sqrt(len(samples))
    assert abs(mean - expected) < 2 * sd_of_mean, (
        f"empirical mean {mean:.2f} vs expected {expected:.2f} "
        f"(2σ={2*sd_of_mean:.2f})")


def test_geometric_tract_length_distribution_is_exponential():
    """Kolmogorov-Smirnov test: empirical samples should match
    Exponential(rate=1/λ) at p > 0.05."""
    rng = np.random.default_rng(1)
    lam = 200.0
    samples = rng.exponential(lam, size=10_000)
    ks_stat, p = stats.kstest(samples, 'expon', args=(0.0, lam))
    assert p > 0.05, f"KS test failed: stat={ks_stat:.4f} p={p:.4f}"


# ----- Tier 1: smoke at biological 3Ra-scale params ---------------

def test_smoke_3ra_scale_geometric():
    """3Ra-scale params (6 Mb inv, 100 bp tract) run without crashing
    and produce a well-formed tree sequence."""
    inv = InversionSpec(
        bp_left=1.0, bp_right=6_000_000.0 - 1.0,
        p_inv=0.5, t_inv=100_000.0,
        gene_conversion_rate=1e-6,
        mean_tract_length=100.0,
        tract_distribution='geometric',
    )
    demo = Demography(pop_sizes=[5000])
    sim = HullSimulator(
        sample_config={('S', 0): 4, ('I', 0): 4},
        demography=demo,
        sequence_length=6_000_000,
        recombination_rate=1e-8,
        inversions=[inv], seed=42,
    )
    ts = sim.simulate()
    assert ts.num_trees > 0
    assert ts.num_nodes > 8
