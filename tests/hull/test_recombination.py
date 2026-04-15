"""Tests for recombination in the hull simulator.

Verifies:
  - Recombination produces multiple trees (more than without recomb).
  - Recombination + inversion still respects the class barrier.
  - TreeSequence is structurally valid (single-rooted trees).

Lineage GC removes non-overlapping fragments after recombination
events, keeping the O(n^2) pair enumeration tractable up to rho ~ 40.
Higher rho (> 100) needs the Rust backend or a Fenwick-tree approach.
"""

import pytest

from msinv.hull import HullSimulator, InversionSpec


def test_panmictic_recomb_produces_multiple_trees():
    """With recombination, expect more than 1 tree."""
    # rho = 4 * 5000 * 1e-8 * 50000 = 10
    sim = HullSimulator(
        n_std=5, n_inv=0,
        population_size=5000,
        sequence_length=50_000.0,
        recombination_rate=1e-8,
        seed=7,
    )
    ts = sim.simulate()
    assert ts.num_trees > 1, (
        f"Expected multiple trees with recombination, got {ts.num_trees}")
    for tree in ts.trees():
        assert tree.num_roots == 1


def test_rho_zero_is_rejected():
    """rho=0 is forbidden globally — must raise ValueError."""
    with pytest.raises(ValueError, match="recombination_rate must be > 0"):
        HullSimulator(
            n_std=4, n_inv=0,
            population_size=500,
            sequence_length=200.0,
            recombination_rate=0.0,
            seed=42,
        )


def test_recomb_with_inversion_valid_trees():
    """Recombination + inversion still produces valid single-rooted trees."""
    # rho = 4 * 10000 * 1e-8 * 100000 = 40
    sim = HullSimulator(
        n_std=5, n_inv=5,
        population_size=10_000,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        bp_left=30_000.0, bp_right=70_000.0,
        p_inv=0.5, t_inv=100_000.0,
        seed=42,
    )
    ts = sim.simulate()
    assert ts.num_trees > 1
    for tree in ts.trees():
        assert tree.num_roots == 1, (
            f"Tree at {tree.interval} has {tree.num_roots} roots")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_recomb_class_barrier_preserved(seed):
    """Cross-class TMRCA inside the inversion should still be >= t_inv
    even with recombination in the collinear flanks."""
    t_inv = 100_000.0
    sim = HullSimulator(
        n_std=3, n_inv=3,
        population_size=10_000,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        bp_left=30_000.0, bp_right=70_000.0,
        p_inv=0.5, t_inv=t_inv,
        gene_conversion_rate=1e-12,  # negligible flux (gamma>0 enforced)
        seed=seed,
    )
    ts = sim.simulate()
    S = list(range(3))
    I = list(range(3, 6))
    tree = ts.at(50_000.0)
    for s in S:
        for i in I:
            mrca_time = tree.time(tree.mrca(s, i))
            assert mrca_time >= t_inv - 1.0, (
                f"Cross-class TMRCA at pos 50 = {mrca_time:.1f}, "
                f"expected >= {t_inv}")


def test_higher_recomb_more_nodes():
    """Higher recombination rate yields more coalescence nodes."""
    params = dict(
        n_std=5, n_inv=0,
        population_size=5000,
        sequence_length=50_000.0,
        seed=7,
    )
    ts_lo = HullSimulator(**params, recombination_rate=1e-9).simulate()
    ts_hi = HullSimulator(**params, recombination_rate=1e-7).simulate()
    assert ts_hi.num_nodes > ts_lo.num_nodes, (
        f"high rho: {ts_hi.num_nodes} nodes vs low rho: {ts_lo.num_nodes}")


def test_recomb_with_multi_inv():
    """Recombination with multiple non-overlapping inversions."""
    sim = HullSimulator(
        n_std=3, n_inv=3,
        population_size=10_000,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[
            InversionSpec(bp_left=15_000.0, bp_right=45_000.0,
                          p_inv=0.5, t_inv=100_000.0),
            InversionSpec(bp_left=55_000.0, bp_right=85_000.0,
                          p_inv=0.3, t_inv=150_000.0),
        ],
        seed=42,
    )
    ts = sim.simulate()
    assert ts.num_trees >= 1
    for tree in ts.trees():
        assert tree.num_roots == 1
