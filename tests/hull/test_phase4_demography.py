"""Phase-4 tests for the hull simulator: multi-pop + demography.

Verifies:
  - Sample_config dispatches samples to correct populations.
  - Cross-pop coalescence forbidden when migration=0 and no ej event.
  - With ej event at time t_split, cross-pop T_MRCA >= t_split.
  - With migration > 0, lineages move between pops at expected rates.
  - Population size scaling: per-pop coal rate scales with 1/Ne_pop.
  - Inversion + multi-pop combined (Kir/Fol style).
"""

import numpy as np
import pytest
import tskit

from msinv.hull import HullSimulator
from msinv.hull.demography import Demography


# ---------------------------------------------------------------------------
# Demography unit tests
# ---------------------------------------------------------------------------

def test_demography_basic():
    d = Demography([10000, 5000])
    assert d.n_pops == 2
    assert d.size_at(0, 0.0) == 10000
    assert d.size_at(1, 100.0) == 5000  # no growth, no events


def test_demography_growth_size():
    d = Demography([10000])
    d.growth_rates[0] = 0.001  # backwards exp shrink
    assert d.size_at(0, 0.0) == 10000
    # N(t) = N0 * exp(-g*t) going backward
    assert abs(d.size_at(0, 1000.0) - 10000 * np.exp(-1.0)) < 1.0


def test_demography_ej_event_moves_lineages():
    """ej event at time t moves all lineages of src pop to dst pop."""
    from msinv.hull.lineage import Lineage, reset_uids
    from msinv.hull.segment import Segment
    reset_uids()
    s1 = Segment(0, 1, 0); s2 = Segment(0, 1, 1)
    a = Lineage(s1, s1, branch_class='S', population=0)
    b = Lineage(s2, s2, branch_class='S', population=1)
    active = [a, b]
    d = Demography([10000, 5000])
    d.add_event(('ej', 1000.0, 1, 0))   # pop 1 → pop 0
    d.apply_event_at(1000.0, active)
    assert a.population == 0
    assert b.population == 0   # moved
    # Migration to/from pop 1 zeroed
    assert d.migration_matrix[1][0] == 0.0
    assert d.migration_matrix[0][1] == 0.0


# ---------------------------------------------------------------------------
# Two-pop simulation: cross-pop barrier respected without migration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_no_migration_cross_pop_mrca_at_least_t_split(seed):
    """With ej at t_split and no migration, cross-pop T_MRCA must be >= t_split."""
    n_each = 4
    t_split = 5000.0
    demo = Demography([10000, 10000])
    demo.add_event(('ej', t_split, 1, 0))  # pop 1 merges into pop 0

    sim = HullSimulator(
        sample_config={(None, 0): n_each, (None, 1): n_each},
        demography=demo,
        sequence_length=100.0,
        seed=seed,
    )
    ts = sim.simulate()
    samples = list(ts.samples())
    pop0 = samples[:n_each]
    pop1 = samples[n_each:]
    for tree in ts.trees():
        for a in pop0:
            for b in pop1:
                assert tree.time(tree.mrca(a, b)) >= t_split - 1e-6, (
                    f"Cross-pop T_MRCA below t_split: "
                    f"{tree.time(tree.mrca(a, b))} < {t_split}")


# ---------------------------------------------------------------------------
# Migration: with M > 0, cross-pop coalescence below t_split is allowed
# ---------------------------------------------------------------------------

def test_migration_allows_cross_pop_mrca_below_t_split():
    """With M > 0, some cross-pop MRCAs should be below t_split."""
    n_each = 5
    t_split = 50_000.0   # very deep
    demo = Demography(
        [10000, 10000],
        migration_matrix=[[0.0, 1e-3], [1e-3, 0.0]],
    )
    demo.add_event(('ej', t_split, 1, 0))

    below_t_split = 0
    total = 0
    for seed in range(10):
        sim = HullSimulator(
            sample_config={(None, 0): n_each, (None, 1): n_each},
            demography=demo,
            sequence_length=10.0,
            seed=seed,
        )
        ts = sim.simulate()
        samples = list(ts.samples())
        pop0 = samples[:n_each]
        pop1 = samples[n_each:]
        for tree in ts.trees():
            for a in pop0:
                for b in pop1:
                    total += 1
                    if tree.time(tree.mrca(a, b)) < t_split - 1e-6:
                        below_t_split += 1
    assert below_t_split > 0, (
        "With M > 0, some cross-pop MRCAs should occur before t_split.")


# ---------------------------------------------------------------------------
# Per-pop size scaling: smaller pop → faster coal
# ---------------------------------------------------------------------------

def test_per_pop_size_scales_coal_rate():
    """A pop with size Ne/10 should have ~10× higher within-pop coal rate."""
    n_each = 5
    big_Ne = 10000
    small_Ne = 1000
    demo = Demography([big_Ne, small_Ne])
    # No migration, very deep ej so within-pop dominates
    demo.add_event(('ej', 1e9, 1, 0))

    big_mrca = []
    small_mrca = []
    for seed in range(20):
        sim = HullSimulator(
            sample_config={(None, 0): n_each, (None, 1): n_each},
            demography=demo,
            sequence_length=1.0,
            seed=seed,
        )
        ts = sim.simulate()
        samples = list(ts.samples())
        pop0 = samples[:n_each]
        pop1 = samples[n_each:]
        for tree in ts.trees():
            big_mrca.append(tree.time(tree.mrca(*pop0)))
            small_mrca.append(tree.time(tree.mrca(*pop1)))

    ratio = np.mean(big_mrca) / np.mean(small_mrca)
    assert 5.0 < ratio < 20.0, (
        f"Expected ~10× ratio (big/small Ne), got {ratio:.2f}")


# ---------------------------------------------------------------------------
# Inversion + multi-pop (Kir/Fol style mini)
# ---------------------------------------------------------------------------

def test_inversion_with_two_pops():
    """Kir/Fol-style mini: 2 pops, ej at t_split, inversion shared."""
    Ne = 10000
    t_split = 14_000.0   # 14k gen
    t_inv = 80_000.0     # 80k gen, very old
    p_inv_anc = 0.5

    demo = Demography([Ne, Ne])
    demo.add_event(('ej', t_split, 1, 0))   # Fol → Kir merge

    # 5 K-S, 3 Fol-S, 3 Fol-I
    sample_config = {
        ('S', 0): 5,
        ('S', 1): 3,
        ('I', 1): 3,
    }
    sim = HullSimulator(
        sample_config=sample_config,
        demography=demo,
        sequence_length=100.0,
        p_inv=p_inv_anc, t_inv=t_inv,
        bp_left=0.0, bp_right=100.0,
        gene_conversion_rate=0.0,
        seed=42,
    )
    ts = sim.simulate()
    samples = list(ts.samples())
    K_S = samples[:5]
    Fol_S = samples[5:8]
    Fol_I = samples[8:11]
    # Inside the inv (whole sequence is inv here):
    # K-S vs Fol-S: cross-pop, must wait for t_split (5000+ gen).
    # K-S vs Fol-I: cross-class, must wait for t_inv (80000 gen).
    # Fol-S vs Fol-I: cross-class within pop, must wait for t_inv.
    for tree in ts.trees():
        for k in K_S:
            for fs in Fol_S:
                t = tree.time(tree.mrca(k, fs))
                assert t >= t_split - 1e-6
            for fi in Fol_I:
                t = tree.time(tree.mrca(k, fi))
                assert t >= t_inv - 1e-6
        for fs in Fol_S:
            for fi in Fol_I:
                t = tree.time(tree.mrca(fs, fi))
                assert t >= t_inv - 1e-6
