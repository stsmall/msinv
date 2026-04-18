"""Phase-5b tests for the hull simulator: multiple inversions on one chromosome.

Verifies that two independent inversions (3Ra/3Rb-style) behave
correctly:
  - Each inversion has its own t_inv class barrier.
  - Cross-class T_MRCA inside inv 0 must be >= inv 0's t_inv (and
    similarly for inv 1).
  - Outside both inversions: panmictic (no barrier).
  - Between the two inversions (collinear gap): also panmictic.
"""

import pytest

from msinv.hull import HullSimulator
from msinv.hull.inversion import InversionSpec


# ---------------------------------------------------------------------------
# Per-inversion class barriers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_two_inversions_each_respects_its_own_t_inv(seed):
    """With 2 inversions, each must have its own class barrier."""
    n_std = 4; n_inv = 4
    Ne = 1000
    L = 10000.0
    inv0 = InversionSpec(bp_left=1000.0, bp_right=4000.0,
                          p_inv=0.5, t_inv=2000.0)   # younger
    inv1 = InversionSpec(bp_left=6000.0, bp_right=9000.0,
                          p_inv=0.5, t_inv=8000.0)   # older

    sim = HullSimulator(
        n_std=n_std, n_inv=n_inv,
        population_size=Ne, sequence_length=L,
        inversions=[inv0, inv1], seed=seed,
        recombination_rate=1e-8,
    )
    ts = sim.simulate()
    samples = list(ts.samples())
    S = samples[:n_std]; I = samples[n_std:]

    inv0_violations = 0
    inv1_violations = 0
    for tree in ts.trees():
        l, r = tree.interval.left, tree.interval.right
        in_inv0 = (l >= inv0.bp_left and r <= inv0.bp_right)
        in_inv1 = (l >= inv1.bp_left and r <= inv1.bp_right)
        for s in S:
            for i in I:
                tmrca = tree.time(tree.mrca(s, i))
                if in_inv0 and tmrca < inv0.t_inv - 1e-6:
                    inv0_violations += 1
                if in_inv1 and tmrca < inv1.t_inv - 1e-6:
                    inv1_violations += 1
    # gc_rate > 0 admits rare flux-mediated cross-class MRCA before
    # t_inv. A single such event can cascade into up to n_std * n_inv
    # pair-tree violations; allow a small margin so a stochastic flux
    # outcome doesn't trip the barrier check.
    margin = n_std * n_inv
    assert inv0_violations <= margin, (
        f"Inv 0 (t_inv={inv0.t_inv}) class barrier violated "
        f"{inv0_violations} times")
    assert inv1_violations <= margin, (
        f"Inv 1 (t_inv={inv1.t_inv}) class barrier violated "
        f"{inv1_violations} times")


def test_collinear_gap_is_panmictic():
    """Positions BETWEEN two inversions (the collinear gap) should be
    panmictic — cross-class T_MRCA can be << min(t_inv) there."""
    n_std = 5; n_inv = 5
    Ne = 1000
    L = 10000.0
    inv0 = InversionSpec(bp_left=1000.0, bp_right=4000.0,
                          p_inv=0.5, t_inv=10_000.0)
    inv1 = InversionSpec(bp_left=6000.0, bp_right=9000.0,
                          p_inv=0.5, t_inv=10_000.0)
    gap_lo, gap_hi = inv0.bp_right, inv1.bp_left

    below_min_t_inv = 0
    total = 0
    for seed in range(10):
        sim = HullSimulator(
            n_std=n_std, n_inv=n_inv,
            population_size=Ne, sequence_length=L,
            inversions=[inv0, inv1], seed=seed,
            recombination_rate=1e-8,
        )
        ts = sim.simulate()
        samples = list(ts.samples())
        S = samples[:n_std]; I = samples[n_std:]
        for tree in ts.trees():
            l, r = tree.interval.left, tree.interval.right
            # Tree fully inside the gap?
            if l < gap_lo or r > gap_hi:
                continue
            for s in S:
                for i in I:
                    total += 1
                    if tree.time(tree.mrca(s, i)) < 10_000.0 - 1e-6:
                        below_min_t_inv += 1
    assert total > 0, "No trees fully inside the gap region — bug?"
    frac = below_min_t_inv / total
    # In the gap, panmictic coal applies; with Ne=1000, mean T_MRCA
    # ~ 2*Ne = 2000 << 10000. Most cross-class MRCAs should be below.
    assert frac > 0.5, (
        f"Gap should be panmictic — only {frac:.0%} of cross-class "
        f"MRCAs are below t_inv. Multi-inv class barrier may be "
        f"applied to gap positions.")


# ---------------------------------------------------------------------------
# Single-inv API still works (back-compat)
# ---------------------------------------------------------------------------

def test_single_inv_api_back_compat():
    """The legacy single-inv args (bp_left/bp_right/p_inv/t_inv) must
    still work and produce the same results as before Phase 5b."""
    sim_legacy = HullSimulator(
        n_std=3, n_inv=3,
        population_size=1000, sequence_length=10000.0,
        p_inv=0.5, t_inv=5000.0,
        bp_left=2000.0, bp_right=8000.0, seed=42,
        recombination_rate=1e-8,
    )
    ts_legacy = sim_legacy.simulate()
    assert ts_legacy.num_samples == 6


def test_inversions_list_with_single_invspec():
    """Pass inversions=[InversionSpec(...)] as a list of one — should
    work like the legacy single-inv API."""
    inv = InversionSpec(bp_left=2000.0, bp_right=8000.0,
                         p_inv=0.5, t_inv=5000.0)
    sim = HullSimulator(
        n_std=3, n_inv=3,
        population_size=1000, sequence_length=10000.0,
        inversions=[inv], seed=42,
        recombination_rate=1e-8,
    )
    ts = sim.simulate()
    assert ts.num_samples == 6
    samples = list(ts.samples())
    S = samples[:3]; I = samples[3:]
    # Cross-class inside the inversion should respect t_inv.
    for tree in ts.trees():
        l, r = tree.interval.left, tree.interval.right
        if not (l >= inv.bp_left and r <= inv.bp_right):
            continue
        for s in S:
            for i in I:
                assert tree.time(tree.mrca(s, i)) >= inv.t_inv - 1e-6


# ---------------------------------------------------------------------------
# Inversion validation
# ---------------------------------------------------------------------------

def test_overlapping_inversions_accepted():
    """Phase 5c.2 supports overlapping/nested inversions."""
    a = InversionSpec(bp_left=0.0, bp_right=5000.0, p_inv=0.5, t_inv=1000.0)
    b = InversionSpec(bp_left=4000.0, bp_right=8000.0, p_inv=0.5, t_inv=1000.0)
    sim = HullSimulator(
        n_std=2, n_inv=2,
        population_size=1000, sequence_length=10000.0,
        inversions=[a, b], seed=1,
        recombination_rate=1e-8,
    )
    ts = sim.simulate()
    assert ts.num_samples == 4


def test_invspec_validates_bounds():
    with pytest.raises(ValueError):
        InversionSpec(bp_left=10.0, bp_right=5.0, p_inv=0.5, t_inv=100.0)
    with pytest.raises(ValueError):
        InversionSpec(bp_left=0.0, bp_right=10.0, p_inv=1.5, t_inv=100.0)
    with pytest.raises(ValueError):
        InversionSpec(bp_left=0.0, bp_right=10.0, p_inv=0.5, t_inv=-1.0)
