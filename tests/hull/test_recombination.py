"""Tests for recombination in the hull simulator.

Verifies:
  - Recombination produces multiple trees (more than without recomb).
  - Recombination + inversion still respects the class barrier.
  - TreeSequence is structurally valid (single-rooted trees).

NOTE: rho values are kept very low (rho < 5) because the Python hull
does not yet have lineage GC — non-overlapping fragments accumulate
and the O(n^2) pair enumeration hangs at higher rho. Once GC is added
these tests can use realistic rho values.
"""

import pytest

from msinv.hull import HullSimulator, InversionSpec


def test_panmictic_recomb_produces_multiple_trees():
    """With recombination, expect more than 1 tree."""
    # rho = 4 * 200 * 1e-3 * 50 = 40 — but only n=2 so manageable
    sim = HullSimulator(
        n_std=2, n_inv=0,
        population_size=200,
        sequence_length=50.0,
        recombination_rate=1e-3,
        seed=7,
    )
    ts = sim.simulate()
    assert ts.num_trees > 1, (
        f"Expected multiple trees with recombination, got {ts.num_trees}")
    for tree in ts.trees():
        assert tree.num_roots == 1


def test_no_recomb_gives_single_tree():
    """Baseline: without recombination, exactly 1 tree."""
    sim = HullSimulator(
        n_std=4, n_inv=0,
        population_size=500,
        sequence_length=200.0,
        recombination_rate=0.0,
        seed=42,
    )
    ts = sim.simulate()
    assert ts.num_trees == 1


def test_recomb_with_inversion_valid_trees():
    """Recombination + inversion still produces valid single-rooted trees."""
    # rho = 4 * 200 * 5e-4 * 100 = 40 but n=2+2=4 total
    sim = HullSimulator(
        n_std=2, n_inv=2,
        population_size=200,
        sequence_length=100.0,
        recombination_rate=5e-4,
        bp_left=30.0, bp_right=70.0,
        p_inv=0.5, t_inv=5000.0,
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
    t_inv = 5000.0
    sim = HullSimulator(
        n_std=2, n_inv=2,
        population_size=200,
        sequence_length=100.0,
        recombination_rate=1e-4,
        bp_left=30.0, bp_right=70.0,
        p_inv=0.5, t_inv=t_inv,
        gene_conversion_rate=0.0,
        seed=seed,
    )
    ts = sim.simulate()
    S = list(range(2))
    I = list(range(2, 4))
    tree = ts.at(50.0)
    for s in S:
        for i in I:
            mrca_time = tree.time(tree.mrca(s, i))
            assert mrca_time >= t_inv - 1.0, (
                f"Cross-class TMRCA at pos 50 = {mrca_time:.1f}, "
                f"expected >= {t_inv}")


def test_recomb_more_nodes_than_no_recomb():
    """With recombination, more nodes from additional coalescences."""
    params = dict(
        n_std=2, n_inv=0,
        population_size=200,
        sequence_length=50.0,
        seed=7,
    )
    ts_no = HullSimulator(**params, recombination_rate=0.0).simulate()
    ts_yes = HullSimulator(**params, recombination_rate=1e-3).simulate()
    assert ts_yes.num_nodes > ts_no.num_nodes, (
        f"recomb: {ts_yes.num_nodes} nodes vs no-recomb: {ts_no.num_nodes}")


def test_recomb_with_multi_inv():
    """Recombination with multiple non-overlapping inversions."""
    sim = HullSimulator(
        n_std=2, n_inv=2,
        population_size=200,
        sequence_length=200.0,
        recombination_rate=1e-4,
        inversions=[
            InversionSpec(bp_left=20.0, bp_right=80.0,
                          p_inv=0.5, t_inv=5000.0),
            InversionSpec(bp_left=120.0, bp_right=180.0,
                          p_inv=0.3, t_inv=8000.0),
        ],
        seed=42,
    )
    ts = sim.simulate()
    assert ts.num_trees >= 1
    for tree in ts.trees():
        assert tree.num_roots == 1
