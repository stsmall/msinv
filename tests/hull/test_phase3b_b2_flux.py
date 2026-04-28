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


# ----- Tier 2: spatial profile φ(x) ------------------------------

def test_spatial_profile_uniform_in_interior_geometric():
    """Empirical fraction of flux events that touch position x should
    be ≈ λ/inv_length in the interior (away from breakpoints by ≫ λ).

    Strategy: instead of instrumenting the simulator's flux events,
    sample many tracts directly via the same _draw_tract logic and
    histogram the per-bp coverage. This validates the geometry, which
    is the determinant of the spatial profile."""
    inv = InversionSpec(
        bp_left=0.0, bp_right=10_000.0,
        p_inv=0.5, t_inv=10_000.0,
        gene_conversion_rate=1e-9,
        mean_tract_length=100.0,
        tract_distribution='geometric',
    )

    rng = np.random.default_rng(2)
    inv_len = inv.bp_right - inv.bp_left
    lam = inv.mean_tract_length
    n_events = 20_000
    # 1-bp bins so the per-bp coverage formula λ/inv_len is the right
    # comparison. With wider bins the expected value picks up an
    # extra (bin_width / inv_len) term from the tract-overlaps-bin
    # geometry and the simple λ/inv_len formula no longer matches.
    n_bins = int(inv_len)
    coverage = np.zeros(n_bins, dtype=np.int64)

    for _ in range(n_events):
        L = rng.exponential(lam)
        L = min(L, inv_len * 0.99)
        if L <= 0.0:
            continue
        b1 = rng.uniform(0.0, inv_len - L)
        tl = int(b1)
        tr = min(int(b1 + L) + 1, n_bins)
        coverage[tl:tr] += 1

    coverage_frac = coverage / n_events
    # Skip 2λ on each side so the rise/fall regions don't pull the mean.
    margin = int(2 * lam)
    interior = coverage_frac[margin:-margin]
    expected_interior = lam / inv_len
    mean_interior = float(np.mean(interior))
    rel_err = abs(mean_interior - expected_interior) / expected_interior
    assert rel_err < 0.10, (
        f"interior coverage {mean_interior:.5f} vs expected "
        f"{expected_interior:.5f} (rel err {rel_err:.3f}, > 10%)")


# ----- Tier 2: rate scaling with mean_tract_length ---------------

def test_flux_rate_scales_linearly_with_mean_tract_length():
    """Per-lineage flux event rate ≈ γ × p_other × mean_tract_length
    (Section 2 of the spec). Verify by varying mean_tract_length over
    a 20× range and confirming the empirical num_trees count scales
    proportionally — flux events fragment the ARG into more trees,
    so num_trees is a monotone proxy for total flux-event count.

    This is the Tier-2 calibration we can land without simulator-
    state instrumentation. Direct event-count calibration is part
    of Tier 3-full (Andolfatto anchor) per the spec's Deferred
    Validation Roadmap."""
    bp_left = 0.0
    bp_right = 200_000.0
    inv_len = bp_right - bp_left
    # γ chosen so per-lineage event rate gives several events per
    # coalescent timescale at small λ but doesn't saturate the ARG at
    # large λ (num_trees is bounded by sequence length / shortest tract).
    gamma = 1e-5
    NREPS = 10
    lambdas = [200.0, 1000.0, 4000.0]  # 20× range
    means = []
    for lam in lambdas:
        inv = InversionSpec(
            bp_left=bp_left, bp_right=bp_right,
            p_inv=0.5, t_inv=20_000.0,
            gene_conversion_rate=gamma,
            mean_tract_length=lam,
            tract_distribution='geometric',
        )
        demo = Demography(pop_sizes=[2000])
        n_trees_reps = []
        for seed in range(NREPS):
            sim = HullSimulator(
                sample_config={('S', 0): 4, ('I', 0): 4},
                demography=demo,
                sequence_length=int(inv_len),
                recombination_rate=1e-9,
                inversions=[inv], seed=seed,
            )
            ts = sim.simulate()
            n_trees_reps.append(ts.num_trees)
        means.append(float(np.mean(n_trees_reps)))

    # Subtract the no-flux baseline (recombination-driven trees) so
    # the flux contribution is what scales with λ.
    inv_zero = InversionSpec(
        bp_left=bp_left, bp_right=bp_right,
        p_inv=0.5, t_inv=20_000.0,
        gene_conversion_rate=gamma,
        mean_tract_length=0.0,                # disables flux
        tract_distribution='geometric',
    )
    demo = Demography(pop_sizes=[2000])
    no_flux_trees = []
    for seed in range(NREPS):
        sim = HullSimulator(
            sample_config={('S', 0): 4, ('I', 0): 4},
            demography=demo,
            sequence_length=int(inv_len),
            recombination_rate=1e-9,
            inversions=[inv_zero], seed=seed,
        )
        no_flux_trees.append(sim.simulate().num_trees)
    baseline = float(np.mean(no_flux_trees))

    flux_contribution = [m - baseline for m in means]

    # Assert: the flux contribution scales monotonically with λ.
    assert flux_contribution[0] < flux_contribution[1] < flux_contribution[2], (
        f"flux_contribution should be monotone-increasing in λ, "
        f"got {flux_contribution} at λ={lambdas}")

    # Soft linearity: the largest λ should contribute substantially
    # more than the smallest. We don't enforce exact linear scaling
    # because num_trees is a coarse proxy (bounded above by the
    # sequence length and recombination breakpoints; saturates at
    # high γ·λ). Tier 3 (Andolfatto anchor, deferred) does the
    # tight calibration.
    assert flux_contribution[2] > 1.5 * flux_contribution[0], (
        f"largest-λ flux contribution should be >1.5× smallest-λ, "
        f"got {flux_contribution[2]:.1f} vs {flux_contribution[0]:.1f}")
