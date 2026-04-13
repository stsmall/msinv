"""Phase-6 tests for the hull simulator: forced-coalescence sweep events.

A Sweep at (x_sel, t_event, target_class) forces all qualifying
lineages (carrying material at x_sel of the target class) to
coalesce into a single ancestor at t_event. Verifies:

  - Without sweep: T_MRCA at x_sel follows the structured-coalescent
    expectation (e.g. ~2·Ne for panmictic).
  - With sweep at t_event: T_MRCA at x_sel is exactly t_event for all
    target-class samples (sweep MRCA).
  - Far from x_sel: T_MRCA unaffected (since we have no recombination
    propagating the sweep window — that would be Phase 7+ work).
  - Sweep targeting one class doesn't force-merge other-class samples.
"""

import numpy as np
import pytest

from msinv.hull import HullSimulator, InversionSpec
from msinv.hull.sweep import Sweep


# ---------------------------------------------------------------------------
# Sweep event basic behaviour
# ---------------------------------------------------------------------------

def test_sweep_forces_coalescence_at_x_sel():
    """All target-class samples should have T_MRCA = t_event at x_sel."""
    Ne = 10_000
    L = 100.0
    t_sweep = 200.0  # very recent — without sweep T_MRCA would be ~2·Ne
    sweep = Sweep(x_sel=50.0, t_event=t_sweep, target_class='P')

    sim = HullSimulator(
        samples=10,
        population_size=Ne,
        sequence_length=L,
        sweeps=[sweep],
        seed=42,
    )
    ts = sim.simulate()
    # Find the tree containing x_sel and check its MRCA time.
    for tree in ts.trees():
        if tree.interval.left <= 50.0 < tree.interval.right:
            samples = list(ts.samples())
            tmrca = tree.time(tree.mrca(*samples))
            # All samples should coalesce by t_sweep (the sweep MRCA).
            # Allow a tiny epsilon for the sweep window.
            assert tmrca <= t_sweep + 1.0, (
                f"After sweep at t={t_sweep}, T_MRCA at x_sel should "
                f"be ~{t_sweep}, got {tmrca}.")
            break


def test_no_sweep_gives_normal_coalescent_tmrca():
    """Without a sweep, T_MRCA follows the standard panmictic
    expectation (~2·Ne for n samples)."""
    Ne = 10_000
    n = 10
    sim = HullSimulator(
        samples=n,
        population_size=Ne,
        sequence_length=100.0,
        seed=42,
    )
    ts = sim.simulate()
    samples = list(ts.samples())
    tmrca = ts.first().time(ts.first().mrca(*samples))
    # E[T_MRCA] = 2·Ne·(1 - 1/n) = 18000 for n=10
    expected = 2 * Ne * (1 - 1 / n)
    # Allow generous tolerance — single rep.
    assert 0.3 * expected < tmrca < 3.0 * expected


# ---------------------------------------------------------------------------
# Sweep targeting a karyotype class inside an inversion
# ---------------------------------------------------------------------------

def test_sweep_on_S_class_inside_inversion():
    """Sweep targeting S samples at x_sel inside an inversion: only
    S samples force-coalesce, I samples remain at T_MRCA >= t_inv."""
    Ne = 10_000
    L = 100.0
    inv = InversionSpec(bp_left=20.0, bp_right=80.0,
                         p_inv=0.5, t_inv=20_000.0)
    t_sweep = 500.0
    sweep = Sweep(x_sel=50.0, t_event=t_sweep, target_class='S')

    sim = HullSimulator(
        n_std=5, n_inv=5,
        population_size=Ne, sequence_length=L,
        p_inv=0.5, t_inv=20_000.0,
        bp_left=inv.bp_left, bp_right=inv.bp_right,
        sweeps=[sweep],
        seed=42,
    )
    ts = sim.simulate()
    samples = list(ts.samples())
    S = samples[:5]; I = samples[5:]
    for tree in ts.trees():
        if tree.interval.left <= 50.0 < tree.interval.right:
            # All S samples should coalesce by t_sweep.
            ss_mrca = tree.time(tree.mrca(*S))
            assert ss_mrca <= t_sweep + 1.0, (
                f"S samples should have T_MRCA ~{t_sweep}, got {ss_mrca}")
            # I samples among themselves: NOT touched by the S-targeted
            # sweep, so I-I T_MRCA should be > t_sweep (free coal in
            # the I sub-pop).
            ii_mrca = tree.time(tree.mrca(*I))
            assert ii_mrca > t_sweep, (
                f"I-I T_MRCA should be unaffected by S-targeted sweep, "
                f"got {ii_mrca}.")
            # Cross-class T_MRCA still >= t_inv (class barrier).
            for s in S:
                for i in I:
                    si_mrca = tree.time(tree.mrca(s, i))
                    assert si_mrca >= 20_000.0 - 1e-6
            break


# ---------------------------------------------------------------------------
# Sweep DOES NOT affect distant positions when no recombination
# ---------------------------------------------------------------------------

def test_sweep_does_not_affect_distant_positions():
    """With no recombination implemented yet, the sweep only affects
    the position x_sel ± sweep_window. Positions far from x_sel
    should have normal-looking T_MRCA distributions."""
    Ne = 10_000
    L = 1000.0
    t_sweep = 200.0
    sweep = Sweep(x_sel=500.0, t_event=t_sweep,
                  target_class='P', sweep_window=5.0)

    sim = HullSimulator(
        samples=10,
        population_size=Ne, sequence_length=L,
        sweeps=[sweep], seed=42,
    )
    ts = sim.simulate()
    # Tree at x=10 (far from x_sel=500) should NOT be the sweep MRCA.
    far_tree = next(tree for tree in ts.trees()
                    if tree.interval.left <= 10.0 < tree.interval.right)
    samples = list(ts.samples())
    far_tmrca = far_tree.time(far_tree.mrca(*samples))
    assert far_tmrca > t_sweep * 5, (
        f"Distant position T_MRCA = {far_tmrca}; expected » sweep "
        f"t_event = {t_sweep} (sweep should not affect distant trees "
        f"without recombination).")


# ---------------------------------------------------------------------------
# Sweep validation
# ---------------------------------------------------------------------------

def test_sweep_dataclass_construction():
    s = Sweep(x_sel=50.0, t_event=100.0, target_class='S')
    assert s.x_sel == 50.0
    assert s.t_event == 100.0
    assert s.target_class == 'S'
    assert s.population is None
    assert s.sweep_window == 0.0


def test_sweep_with_no_target_lineages_is_noop():
    """If no lineages have material at x_sel of the target class, the
    sweep should be a no-op, not crash."""
    Ne = 1000
    sim = HullSimulator(
        samples=5,
        population_size=Ne, sequence_length=100.0,
        sweeps=[Sweep(x_sel=50.0, t_event=10.0, target_class='I')],
        # All samples are 'P' (no inversion); no 'I' class anywhere.
        seed=42,
    )
    ts = sim.simulate()
    assert ts.num_samples == 5  # ran to completion without crashing
