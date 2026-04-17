"""Summary-stat cross-validation of hull vs msprime.

Extends test_msprime_validation.py with:

  * Explicit segregating-sites comparison (not just pi — which is pi
    per site; segregating_sites counts raw events).
  * Multi-rho coverage including rho ≥ 500 where the 2026-04 bitmap
    and sweepline optimisations live — ensures those paths reproduce
    msprime's expected site frequency, not just run faster.
  * Two-population dxy at moderate rho.
  * SFS shape comparison via normalised L2 distance across the first
    k non-zero bins.

Tolerances reflect the coalescent variance at 100 replicates; tighten
with NREPS where tight bounds are needed.
"""

import numpy as np
import pytest
import msprime

from msinv import HullSimulator, Demography


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _msp_haploid(n, Ne, L, r, seed, pops=None, demo=None):
    if demo is None:
        return msprime.sim_ancestry(
            samples=[msprime.SampleSet(n, ploidy=1)],
            population_size=Ne, sequence_length=L,
            recombination_rate=r, random_seed=seed)
    sample_sets = [
        msprime.SampleSet(cnt, population=p, ploidy=1)
        for p, cnt in pops.items()
    ]
    return msprime.sim_ancestry(
        samples=sample_sets, demography=demo,
        sequence_length=L, recombination_rate=r,
        random_seed=seed)


def _mutate(ts, mu, seed):
    return msprime.sim_mutations(ts, rate=mu, random_seed=seed,
                                  discrete_genome=False)


def _afs_l2(a, b):
    """Normalised L2 distance between two SFS vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    k = min(len(a), len(b))
    a = a[:k] / max(a.sum(), 1.0)
    b = b[:k] / max(b.sum(), 1.0)
    return float(np.sqrt(((a - b) ** 2).sum()))


# ---------------------------------------------------------------
# Panmictic seg sites — low and high rho
# ---------------------------------------------------------------

@pytest.mark.parametrize("rho,tol", [(20.0, 0.15), (200.0, 0.15), (500.0, 0.20)])
def test_panmictic_segsites_vs_msprime(rho, tol):
    """Mean segregating sites should match msprime within tol at
    each rho point. rho=500 exercises the bitmap iter_pairs + sweepline
    GC paths added on feature/rho-optimization."""
    Ne = 1_000
    L = 100_000.0
    r = rho / (4.0 * Ne * L)
    mu = 1e-8
    n = 10
    nreps = 100 if rho < 500 else 60

    hull_s = []
    msp_s = []
    for seed in range(nreps):
        ts = HullSimulator(
            n_std=n, n_inv=0, population_size=Ne,
            sequence_length=L, recombination_rate=r,
            seed=seed).simulate()
        hull_s.append(_mutate(ts, mu, seed + 1_000).num_sites)

        ts2 = _msp_haploid(n, Ne, L, r, seed + 2_000)
        msp_s.append(_mutate(ts2, mu, seed + 3_000).num_sites)

    ratio = np.mean(hull_s) / np.mean(msp_s)
    assert 1.0 - tol < ratio < 1.0 + tol, (
        f"rho={rho}: hull/msprime seg sites ratio = {ratio:.3f} "
        f"(hull={np.mean(hull_s):.1f}, msp={np.mean(msp_s):.1f})")


# ---------------------------------------------------------------
# Panmictic SFS shape
# ---------------------------------------------------------------

def test_panmictic_sfs_shape_matches_msprime():
    """Hull and msprime SFS shapes should be close under the neutral
    coalescent. Uses normalised L2 distance across the first n-1
    non-fixed frequency bins; 0.15 tolerates 100-rep noise."""
    Ne = 1_000
    L = 50_000.0
    r = 1e-8
    mu = 1e-8
    n = 10
    nreps = 100

    hull_sfs = np.zeros(n + 1)
    msp_sfs = np.zeros(n + 1)
    for seed in range(nreps):
        ts = HullSimulator(
            n_std=n, n_inv=0, population_size=Ne,
            sequence_length=L, recombination_rate=r,
            seed=seed).simulate()
        mts = _mutate(ts, mu, seed + 1_000)
        hull_sfs += mts.allele_frequency_spectrum(
            polarised=True, span_normalise=False)

        ts2 = _msp_haploid(n, Ne, L, r, seed + 2_000)
        mts2 = _mutate(ts2, mu, seed + 3_000)
        msp_sfs += mts2.allele_frequency_spectrum(
            polarised=True, span_normalise=False)

    # Compare bins 1..n-1 (drop invariant/fixed).
    d = _afs_l2(hull_sfs[1:-1], msp_sfs[1:-1])
    assert d < 0.15, (
        f"SFS L2 distance {d:.3f} exceeds 0.15 "
        f"(hull={hull_sfs}, msp={msp_sfs})")


# ---------------------------------------------------------------
# Two-pop seg sites + dxy
# ---------------------------------------------------------------

def test_two_pop_split_segsites_and_dxy():
    """Two-pop split: match segregating sites per-pop AND dxy to
    msprime. Ties the optimization paths to a structured-demography
    use case."""
    Ne = 2_000
    L = 50_000.0
    r = 1e-8
    mu = 1e-8
    t_split = 5_000.0
    n_per = 6
    nreps = 120

    hull_s0 = []; hull_s1 = []; hull_dxy = []
    msp_s0 = []; msp_s1 = []; msp_dxy = []
    for seed in range(nreps):
        demo_h = Demography(pop_sizes=[Ne, Ne])
        demo_h.add_event(('ej', t_split, 1, 0))
        ts = HullSimulator(
            sample_config={(None, 0): n_per, (None, 1): n_per},
            demography=demo_h, sequence_length=L,
            recombination_rate=r, seed=seed).simulate()
        mts = _mutate(ts, mu, seed + 1_000)
        g0 = list(range(n_per))
        g1 = list(range(n_per, 2 * n_per))
        hull_s0.append(float(mts.segregating_sites(g0)))
        hull_s1.append(float(mts.segregating_sites(g1)))
        hull_dxy.append(float(mts.divergence([g0, g1])))

        demo_m = msprime.Demography()
        demo_m.add_population(initial_size=Ne)
        demo_m.add_population(initial_size=Ne)
        demo_m.add_mass_migration(
            time=t_split, source=1, dest=0, proportion=1.0)
        ts2 = _msp_haploid(None, Ne, L, r, seed + 2_000,
                            pops={0: n_per, 1: n_per}, demo=demo_m)
        mts2 = _mutate(ts2, mu, seed + 3_000)
        msp_s0.append(float(mts2.segregating_sites(g0)))
        msp_s1.append(float(mts2.segregating_sites(g1)))
        msp_dxy.append(float(mts2.divergence([g0, g1])))

    for lbl, h, m in [("seg_sites_pop0", hull_s0, msp_s0),
                      ("seg_sites_pop1", hull_s1, msp_s1),
                      ("dxy", hull_dxy, msp_dxy)]:
        ratio = np.mean(h) / np.mean(m)
        assert 0.85 < ratio < 1.15, (
            f"{lbl}: hull/msprime ratio = {ratio:.3f} "
            f"(hull={np.mean(h):.3g}, msp={np.mean(m):.3g})")


# ---------------------------------------------------------------
# High-rho panmictic dxy across two sampled groups
# ---------------------------------------------------------------

def test_panmictic_high_rho_segsites_tight():
    """At rho=500, single-pop seg sites is the most sensitive check
    for a bitmap/iter_pairs regression (structured path runs there
    too). Tighter tolerance than the parametrised test."""
    Ne = 1_000
    L = 100_000.0
    rho = 500.0
    r = rho / (4.0 * Ne * L)
    mu = 1e-8
    n = 20
    nreps = 60

    hull_s = []; msp_s = []
    for seed in range(nreps):
        ts = HullSimulator(
            n_std=n, n_inv=0, population_size=Ne,
            sequence_length=L, recombination_rate=r,
            seed=seed).simulate()
        hull_s.append(_mutate(ts, mu, seed + 1_000).num_sites)

        ts2 = _msp_haploid(n, Ne, L, r, seed + 2_000)
        msp_s.append(_mutate(ts2, mu, seed + 3_000).num_sites)

    hull_m = np.mean(hull_s); msp_m = np.mean(msp_s)
    ratio = hull_m / msp_m
    # Ratio-of-means SE: sqrt((sd_h/m_h)^2 + (sd_m/m_m)^2) / sqrt(n)
    se = np.sqrt(
        (np.std(hull_s) / hull_m) ** 2
        + (np.std(msp_s) / msp_m) ** 2
    ) / np.sqrt(nreps)
    # 3-sigma bound.
    assert abs(ratio - 1.0) < 3.0 * se + 0.05, (
        f"rho=500 seg sites: ratio={ratio:.3f}, 3*SE={3*se:.3f} "
        f"(hull={hull_m:.1f}, msp={msp_m:.1f})")
