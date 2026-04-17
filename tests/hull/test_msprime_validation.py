"""Cross-validation of hull simulator against msprime ground truth.

Uses haploid samples (ploidy=1) in both msprime and hull for
consistency. Tolerances are 20% for means over 50-100 replicates.
"""

import numpy as np
import pytest
import msprime

from msinv import HullSimulator, InversionSpec, Demography


NREPS = 500


def _msp_ancestry(n, Ne, L, r, seed, pops=None, demo=None):
    """Helper: msprime sim_ancestry with haploid samples."""
    if demo is None:
        return msprime.sim_ancestry(
            samples=[msprime.SampleSet(n, ploidy=1)],
            population_size=Ne, sequence_length=L,
            recombination_rate=r, random_seed=seed)
    else:
        sample_sets = []
        for pop, count in pops.items():
            sample_sets.append(
                msprime.SampleSet(count, population=pop, ploidy=1))
        return msprime.sim_ancestry(
            samples=sample_sets, demography=demo,
            sequence_length=L, recombination_rate=r,
            random_seed=seed)


# ---------------------------------------------------------------
# Panmictic
# ---------------------------------------------------------------

def test_panmictic_diversity_matches_msprime():
    """Mean pi should match msprime within 20%."""
    Ne = 5_000; mu = 1e-8; L = 50_000; r = 1e-8; n = 10

    hull_pi = []; msp_pi = []
    for seed in range(NREPS):
        ts = HullSimulator(n_std=n, n_inv=0, population_size=Ne,
            sequence_length=L, recombination_rate=r, seed=seed).simulate()
        mts = msprime.sim_mutations(ts, rate=mu, random_seed=seed+1000,
                                     discrete_genome=False)
        hull_pi.append(float(mts.diversity()))

        ts2 = _msp_ancestry(n, Ne, L, r, seed+2000)
        mts2 = msprime.sim_mutations(ts2, rate=mu, random_seed=seed+3000,
                                      discrete_genome=False)
        msp_pi.append(float(mts2.diversity()))

    ratio = np.mean(hull_pi) / np.mean(msp_pi)
    assert 0.95 < ratio < 1.05, (
        f"Hull/msprime pi ratio = {ratio:.3f}")


def test_panmictic_tmrca_matches_msprime():
    """Mean T_MRCA for n=2 should match msprime within 20%."""
    Ne = 1_000; L = 1_000.0; r = 1e-9  # rho > 0 required

    hull_t = []; msp_t = []
    for seed in range(NREPS):
        ts = HullSimulator(n_std=2, n_inv=0, population_size=Ne,
            sequence_length=L, recombination_rate=r,
            seed=seed).simulate()
        hull_t.append(ts.first().time(ts.first().root))

        ts2 = _msp_ancestry(2, Ne, L, r, seed+5000)
        msp_t.append(ts2.first().time(ts2.first().root))

    ratio = np.mean(hull_t) / np.mean(msp_t)
    assert 0.95 < ratio < 1.05, (
        f"Hull/msprime T_MRCA ratio = {ratio:.3f} "
        f"(hull={np.mean(hull_t):.0f}, msp={np.mean(msp_t):.0f})")


def test_panmictic_tree_count_matches_msprime():
    """Mean tree count should match msprime within 25%.

    Hull uses *r* as the per-lineage recombination rate (diploid
    convention).  msprime with ``ploidy=1`` halves the effective rate,
    so we use the default ``ploidy=2`` here for a like-for-like
    comparison of breakpoint counts.
    """
    Ne = 5_000; L = 50_000; r = 1e-8; n = 6

    hull_t = []; msp_t = []
    for seed in range(NREPS):
        ts = HullSimulator(n_std=n, n_inv=0, population_size=Ne,
            sequence_length=L, recombination_rate=r, seed=seed).simulate()
        hull_t.append(ts.num_trees)

        # ploidy=2 (default) so msprime uses the same effective rate as hull
        ts2 = msprime.sim_ancestry(n, population_size=Ne,
            sequence_length=L, recombination_rate=r,
            random_seed=seed+4000)
        msp_t.append(ts2.num_trees)

    ratio = np.mean(hull_t) / np.mean(msp_t)
    assert 0.95 < ratio < 1.05, (
        f"Hull/msprime tree ratio = {ratio:.3f} "
        f"(hull={np.mean(hull_t):.1f}, msp={np.mean(msp_t):.1f})")


# ---------------------------------------------------------------
# Two-pop split
# ---------------------------------------------------------------

def test_two_pop_split_dxy_matches_msprime():
    """Two-pop split dxy should match msprime within 25%."""
    Ne = 2_000; mu = 1e-8; L = 50_000; r = 1e-8; t_split = 5_000
    n_per = 3; NREPS_2P = 300

    hull_dxy = []; msp_dxy = []
    for seed in range(NREPS_2P):
        # Hull
        demo_h = Demography(pop_sizes=[Ne, Ne])
        demo_h.add_event(('ej', t_split, 1, 0))
        ts = HullSimulator(
            sample_config={(None, 0): n_per, (None, 1): n_per},
            demography=demo_h, sequence_length=L,
            recombination_rate=r, seed=seed).simulate()
        mts = msprime.sim_mutations(ts, rate=mu, random_seed=seed+1000,
                                     discrete_genome=False)
        hull_dxy.append(float(mts.divergence(
            [list(range(n_per)), list(range(n_per, 2*n_per))])))

        # msprime (haploid) — use mass_migration to match hull's ej
        demo_m = msprime.Demography()
        demo_m.add_population(initial_size=Ne)
        demo_m.add_population(initial_size=Ne)
        demo_m.add_mass_migration(time=t_split, source=1, dest=0,
                                   proportion=1.0)
        ts2 = _msp_ancestry(None, Ne, L, r, seed+2000,
                             pops={0: n_per, 1: n_per}, demo=demo_m)
        mts2 = msprime.sim_mutations(ts2, rate=mu, random_seed=seed+3000,
                                      discrete_genome=False)
        msp_dxy.append(float(mts2.divergence(
            [list(range(n_per)), list(range(n_per, 2*n_per))])))

    ratio = np.mean(hull_dxy) / np.mean(msp_dxy)
    assert 0.94 < ratio < 1.06, (
        f"Hull/msprime dxy ratio = {ratio:.3f} "
        f"(hull={np.mean(hull_dxy):.2e}, msp={np.mean(msp_dxy):.2e})")


# ---------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------

def test_inversion_cross_class_dxy_elevated():
    """Cross-class dxy inside inversion should be > 2x same-class."""
    Ne = 5_000; mu = 1e-8; L = 100_000; r = 1e-8
    bp_l, bp_r = 30_000, 70_000; NREPS_INV = 30

    dxy_cross = []; dxy_same = []
    for seed in range(NREPS_INV):
        ts = HullSimulator(
            n_std=5, n_inv=5, population_size=Ne,
            sequence_length=L, recombination_rate=r,
            inversions=[InversionSpec(bp_left=bp_l, bp_right=bp_r,
                                       p_inv=0.5, t_inv=200_000)],
            seed=seed).simulate()
        mts = msprime.sim_mutations(ts, rate=mu, random_seed=seed+1000,
                                     discrete_genome=False)
        G = mts.genotype_matrix().T
        pos = np.array([v.site.position for v in mts.variants()])
        S = list(range(5)); I = list(range(5, 10))
        mask = (pos >= bp_l) & (pos < bp_r)
        if mask.sum() == 0: continue
        dc = sum((G[a,mask]!=G[b,mask]).sum() for a in S for b in I) / (25*mask.sum())
        ds = sum((G[a,mask]!=G[b,mask]).sum() for a in S for b in S if b>a) / (10*mask.sum())
        dxy_cross.append(dc); dxy_same.append(ds)

    assert np.mean(dxy_cross) > 2 * np.mean(dxy_same), (
        f"Cross ({np.mean(dxy_cross):.2e}) should be > 2x same ({np.mean(dxy_same):.2e})")
