"""Phase-5c.2 tests: nested / overlapping inversions.

A position inside multiple inversions carries a ``frozenset`` of class
tags, one per containing inversion. Compatibility for coalescence
requires the segments' frozensets to be EQUAL.

Verifies:
  - Sample initialization correctly assigns frozenset class to nested
    positions.
  - Two samples sharing all karyotypes can coalesce inside the
    overlap; samples differing at any inversion can NOT coalesce
    until the relevant t_inv lifts.
  - Each inversion's class barrier independently lifts the relevant
    tag from frozensets.
"""

import numpy as np
import pytest

from msinv.hull import HullSimulator, InversionSpec
from msinv.hull.tables import TableBuilder
from msinv.hull.segment import make_initial_segments


# ---------------------------------------------------------------------------
# Initial-segment construction with nested invs
# ---------------------------------------------------------------------------

def _classes_of_lineage(lin):
    out = []
    seg = lin.head
    while seg is not None:
        out.append((seg.left, seg.right, seg.branch_class))
        seg = seg.next
    return out


def test_nested_inv_segments_carry_frozenset_class():
    """Inversion B nested inside inversion A: positions in B carry a
    frozenset {'S0', 'S1'} (or 'I0'/'I1'). Positions in A only get
    a single tag like 'S0'."""
    inv_outer = InversionSpec(bp_left=10.0, bp_right=90.0,
                              p_inv=0.5, t_inv=1000.0)
    inv_inner = InversionSpec(bp_left=30.0, bp_right=70.0,
                              p_inv=0.5, t_inv=2000.0)
    sim = HullSimulator(
        sample_config={(('S', 'I'), 0): 1},
        population_size=1000, sequence_length=100.0,
        inversions=[inv_outer, inv_inner], seed=1)
    tables = TableBuilder(sequence_length=100.0)
    active = sim._initial_lineages(tables)
    classes = _classes_of_lineage(active[0])
    # Expect: [0,10) P, [10,30) S0, [30,70) {S0, I1}, [70,90) S0, [90,100) P
    assert classes[0] == (0.0, 10.0, 'P')
    assert classes[1] == (10.0, 30.0, 'S0')
    assert classes[2] == (30.0, 70.0, frozenset({'S0', 'I1'}))
    assert classes[3] == (70.0, 90.0, 'S0')
    assert classes[4] == (90.0, 100.0, 'P')


def test_overlapping_invs_segments_carry_frozenset():
    """Two non-nested overlapping inversions: A=[0,50), B=[40,80).
    The overlap region [40,50) should carry both inversions' tags."""
    a = InversionSpec(bp_left=0.0, bp_right=50.0, p_inv=0.5, t_inv=1000.0)
    b = InversionSpec(bp_left=40.0, bp_right=80.0, p_inv=0.5, t_inv=1000.0)
    sim = HullSimulator(
        sample_config={(('S', 'I'), 0): 1},
        population_size=1000, sequence_length=100.0,
        inversions=[a, b], seed=1)
    tables = TableBuilder(sequence_length=100.0)
    active = sim._initial_lineages(tables)
    classes = _classes_of_lineage(active[0])
    assert classes[0] == (0.0, 40.0, 'S0')
    assert classes[1] == (40.0, 50.0, frozenset({'S0', 'I1'}))
    assert classes[2] == (50.0, 80.0, 'I1')
    assert classes[3] == (80.0, 100.0, 'P')


# ---------------------------------------------------------------------------
# Class barrier with nested invs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_nested_invs_each_barrier_independent(seed):
    """Outer t_inv = 1000, inner t_inv = 5000. Inside the inner inv,
    samples differing at the inner inv must wait for the inner t_inv
    to coalesce (regardless of outer karyotype agreement)."""
    Ne = 1000
    L = 10000.0
    inv_outer = InversionSpec(bp_left=0.0, bp_right=10000.0,
                              p_inv=0.5, t_inv=1000.0)
    inv_inner = InversionSpec(bp_left=3000.0, bp_right=7000.0,
                              p_inv=0.5, t_inv=5000.0)

    # 4 samples: SS (S in outer + S in inner), SI (S in outer, I in inner)
    sim = HullSimulator(
        sample_config={
            (('S', 'S'), 0): 2,
            (('S', 'I'), 0): 2,
        },
        population_size=Ne, sequence_length=L,
        inversions=[inv_outer, inv_inner],
        recombination_rate=1e-8, seed=seed)
    ts = sim.simulate()
    samples = list(ts.samples())
    SS = samples[:2]; SI = samples[2:]

    # Inside the inner inv (3000,7000), SS samples carry frozenset({S0, S1})
    # and SI samples carry frozenset({S0, I1}). They differ at inv 1
    # → can't coalesce until inv 1's t_inv = 5000.
    inner_violations = 0
    for tree in ts.trees():
        l, r = tree.interval.left, tree.interval.right
        if l < inv_inner.bp_left or r > inv_inner.bp_right:
            continue
        for s in SS:
            for i in SI:
                tmrca = tree.time(tree.mrca(s, i))
                if tmrca < inv_inner.t_inv - 1e-6:
                    inner_violations += 1
    assert inner_violations == 0


def test_nested_outer_only_barrier_when_inner_karyotypes_match():
    """When samples agree at the inner inv (both 'SS'), only the
    outer t_inv constrains them. Inside the outer-only region (not
    overlapping the inner), normal structured-coal applies."""
    Ne = 1000
    L = 10000.0
    inv_outer = InversionSpec(bp_left=0.0, bp_right=10000.0,
                              p_inv=0.5, t_inv=10_000.0)  # very deep
    inv_inner = InversionSpec(bp_left=4000.0, bp_right=6000.0,
                              p_inv=0.5, t_inv=1000.0)

    # All samples 'SS' (S in both invs)
    sim = HullSimulator(
        sample_config={(('S', 'S'), 0): 5},
        population_size=Ne, sequence_length=L,
        inversions=[inv_outer, inv_inner],
        recombination_rate=1e-8, seed=42)
    ts = sim.simulate()
    samples = list(ts.samples())
    # All samples can coalesce in the outer-only region (not in the
    # inner inv) at the structured rate (~2·Ne·p_std = 1000), well
    # before the outer t_inv.
    outer_only_mrcas = []
    for tree in ts.trees():
        l, r = tree.interval.left, tree.interval.right
        # Tree fully outside the inner inv but inside the outer
        if r <= inv_inner.bp_left or l >= inv_inner.bp_right:
            outer_only_mrcas.append(tree.time(tree.mrca(*samples)))
    if outer_only_mrcas:
        median_t = np.median(outer_only_mrcas)
        assert median_t < inv_outer.t_inv / 2, (
            f"With matching inner karyotypes, outer-only positions "
            f"should follow normal structured-coal (~2·Ne·p_std), "
            f"got median T_MRCA = {median_t:.0f} vs outer t_inv "
            f"= {inv_outer.t_inv}.")


# ---------------------------------------------------------------------------
# Tree sequence well-formedness
# ---------------------------------------------------------------------------

def test_treeseq_valid_with_nested_invs():
    inv_outer = InversionSpec(bp_left=0.0, bp_right=10000.0,
                              p_inv=0.5, t_inv=2000.0)
    inv_inner = InversionSpec(bp_left=3000.0, bp_right=7000.0,
                              p_inv=0.5, t_inv=3000.0)
    sim = HullSimulator(
        sample_config={(('S', 'S'), 0): 2,
                        (('I', 'I'), 0): 2,
                        (('S', 'I'), 0): 2},
        population_size=1000, sequence_length=10000.0,
        inversions=[inv_outer, inv_inner],
        recombination_rate=1e-8, seed=42)
    ts = sim.simulate()
    tables = ts.dump_tables()
    tables.sort()
    ts2 = tables.tree_sequence()
    assert ts2.num_samples == 6
    for tree in ts2.trees():
        assert tree.num_roots == 1
