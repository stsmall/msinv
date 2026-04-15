"""Phase-6 tests for the hull simulator: forced-coalescence sweep events.

A Sweep at (x_sel, t_event, target_class) forces all qualifying
lineages (carrying material at x_sel of the target class) to
coalesce into a single ancestor at t_event. Verifies:

  - Without sweep: T_MRCA at x_sel follows the structured-coalescent
    expectation (e.g. ~2·Ne for panmictic).
  - With sweep at t_event: T_MRCA at x_sel is exactly t_event for all
    target-class samples (sweep MRCA).
  - Far from x_sel: T_MRCA unaffected by the sweep.
  - Sweep targeting one class doesn't force-merge other-class samples.
"""

import pytest

from msinv.hull import HullSimulator, InversionSpec
from msinv.hull.sweep import Sweep


# ---------------------------------------------------------------------------
# Sweep event basic behaviour
# ---------------------------------------------------------------------------

def test_sweep_forces_coalescence_at_x_sel():
    """All target-class samples should have T_MRCA = t_event at x_sel."""
    Ne = 10_000
    L = 100_000.0
    t_sweep = 200.0  # very recent — without sweep T_MRCA would be ~2·Ne
    sweep = Sweep(x_sel=50_000.0, t_event=t_sweep, target_class='P')

    sim = HullSimulator(
        samples=10,
        population_size=Ne,
        sequence_length=L,
        recombination_rate=1e-8,
        sweeps=[sweep],
        seed=42,
    )
    ts = sim.simulate()
    tree = ts.at(50_000.0)
    samples = list(ts.samples())
    tmrca = tree.time(tree.mrca(*samples))
    assert tmrca <= t_sweep + 1.0, (
        f"After sweep at t={t_sweep}, T_MRCA at x_sel should "
        f"be ~{t_sweep}, got {tmrca}.")


def test_no_sweep_gives_normal_coalescent_tmrca():
    """Without a sweep, T_MRCA follows the standard panmictic
    expectation (~2·Ne for n samples)."""
    Ne = 10_000
    n = 10
    sim = HullSimulator(
        samples=n,
        population_size=Ne,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
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
    L = 100_000.0
    t_inv = 20_000.0
    t_sweep = 500.0
    sweep = Sweep(x_sel=50_000.0, t_event=t_sweep, target_class='S')

    sim = HullSimulator(
        n_std=5, n_inv=5,
        population_size=Ne, sequence_length=L,
        p_inv=0.5, t_inv=t_inv,
        bp_left=20_000.0, bp_right=80_000.0,
        recombination_rate=1e-8,
        sweeps=[sweep],
        seed=42,
    )
    ts = sim.simulate()
    samples = list(ts.samples())
    S = samples[:5]; I = samples[5:]
    tree = ts.at(50_000.0)
    # All S samples should coalesce by t_sweep.
    ss_mrca = tree.time(tree.mrca(*S))
    assert ss_mrca <= t_sweep + 1.0, (
        f"S samples should have T_MRCA ~{t_sweep}, got {ss_mrca}")
    # I samples among themselves: NOT touched by the S-targeted
    # sweep, so I-I T_MRCA should be > t_sweep.
    ii_mrca = tree.time(tree.mrca(*I))
    assert ii_mrca > t_sweep, (
        f"I-I T_MRCA should be unaffected by S-targeted sweep, "
        f"got {ii_mrca}.")
    # Cross-class T_MRCA still >= t_inv (class barrier).
    for s in S:
        for i in I:
            si_mrca = tree.time(tree.mrca(s, i))
            assert si_mrca >= t_inv - 1e-6


# ---------------------------------------------------------------------------
# Sweep does not affect distant positions
# ---------------------------------------------------------------------------

def test_sweep_does_not_affect_distant_positions():
    """With recombination, the sweep affects a region around x_sel but
    positions far away should have T_MRCA >> t_sweep."""
    Ne = 10_000
    L = 100_000.0
    t_sweep = 200.0
    sweep = Sweep(x_sel=50_000.0, t_event=t_sweep,
                  target_class='P', sweep_window=500.0)

    sim = HullSimulator(
        samples=10,
        population_size=Ne, sequence_length=L,
        recombination_rate=1e-8,
        sweeps=[sweep], seed=42,
    )
    ts = sim.simulate()
    # Tree at x=1000 (far from x_sel=50000) should NOT be the sweep MRCA.
    far_tree = ts.at(1000.0)
    samples = list(ts.samples())
    far_tmrca = far_tree.time(far_tree.mrca(*samples))
    assert far_tmrca > t_sweep * 5, (
        f"Distant position T_MRCA = {far_tmrca}; expected >> sweep "
        f"t_event = {t_sweep}.")


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
    assert s.selection_coefficient == 0.0


def test_sweep_with_no_target_lineages_is_noop():
    """If no lineages have material at x_sel of the target class, the
    sweep should be a no-op, not crash."""
    Ne = 1000
    sim = HullSimulator(
        samples=5,
        population_size=Ne, sequence_length=100_000.0,
        recombination_rate=1e-8,
        sweeps=[Sweep(x_sel=50_000.0, t_event=10.0, target_class='I')],
        # All samples are 'P' (no inversion); no 'I' class anywhere.
        seed=42,
    )
    ts = sim.simulate()
    assert ts.num_samples == 5  # ran to completion without crashing


# ---------------------------------------------------------------------------
# Multi-inversion sweep: target_class='S0' vs 'S1' (Bug #4)
# ---------------------------------------------------------------------------

def test_multi_inv_sweep_S0_only_hits_inv0():
    """In a two-inversion setup, a sweep targeting 'S0' should only
    force-coalesce S0 lineages (inv 0's standard class), not S1."""
    Ne = 10_000
    L = 100_000.0
    t_inv = 20_000.0
    t_sweep = 500.0
    # x_sel inside inv 0 (20-50 kb), outside inv 1 (60-90 kb)
    sweep = Sweep(x_sel=35_000.0, t_event=t_sweep, target_class='S0')

    sim = HullSimulator(
        n_std=5, n_inv=5,
        population_size=Ne, sequence_length=L,
        inversions=[
            InversionSpec(bp_left=20_000.0, bp_right=50_000.0,
                          p_inv=0.5, t_inv=t_inv),
            InversionSpec(bp_left=60_000.0, bp_right=90_000.0,
                          p_inv=0.5, t_inv=t_inv),
        ],
        recombination_rate=1e-8,
        sweeps=[sweep],
        seed=42,
    )
    ts = sim.simulate()
    samples = list(ts.samples())
    S = samples[:5]; I = samples[5:]

    tree = ts.at(35_000.0)
    # S lineages at inv 0 are tagged 'S0'. Sweep should coalesce them.
    ss_mrca = tree.time(tree.mrca(*S))
    assert ss_mrca <= t_sweep + 1.0, (
        f"S0 samples at x_sel should coalesce by sweep at t={t_sweep}, "
        f"got T_MRCA={ss_mrca}")

    # I lineages at inv 0 are tagged 'I0' — NOT targeted by 'S0' sweep.
    # I-I T_MRCA should be > t_sweep (not swept).
    ii_mrca = tree.time(tree.mrca(*I))
    assert ii_mrca > t_sweep, (
        f"I0 lineages should NOT be swept by 'S0' target, "
        f"got I-I T_MRCA={ii_mrca}")


def test_multi_inv_sweep_S1_only_hits_inv1():
    """Sweep targeting 'S1' should only coalesce lineages that are
    Standard at inv 1, regardless of their class at inv 0."""
    Ne = 10_000
    L = 100_000.0
    t_inv = 20_000.0
    t_sweep = 500.0
    # x_sel inside inv 1 (60-90 kb)
    sweep = Sweep(x_sel=75_000.0, t_event=t_sweep, target_class='S1')

    sim = HullSimulator(
        n_std=5, n_inv=5,
        population_size=Ne, sequence_length=L,
        inversions=[
            InversionSpec(bp_left=20_000.0, bp_right=50_000.0,
                          p_inv=0.5, t_inv=t_inv),
            InversionSpec(bp_left=60_000.0, bp_right=90_000.0,
                          p_inv=0.5, t_inv=t_inv),
        ],
        recombination_rate=1e-8,
        sweeps=[sweep],
        seed=42,
    )
    ts = sim.simulate()
    samples = list(ts.samples())
    S = samples[:5]; I = samples[5:]

    tree = ts.at(75_000.0)
    # S samples are S0-S1 at both inversions. At inv 1, they're 'S1'.
    # The sweep targets 'S1', so S samples should coalesce.
    ss_mrca = tree.time(tree.mrca(*S))
    assert ss_mrca <= t_sweep + 1.0, (
        f"S1 samples at inv 1 should coalesce by sweep at t={t_sweep}, "
        f"got T_MRCA={ss_mrca}")

    # I samples are I0-I1 at both inversions. At inv 1, they're 'I1'.
    # NOT targeted by 'S1' sweep.
    ii_mrca = tree.time(tree.mrca(*I))
    assert ii_mrca > t_sweep, (
        f"I1 lineages should NOT be swept by 'S1' target, "
        f"got I-I T_MRCA={ii_mrca}")


def test_multi_inv_sweep_bare_S_matches_both_S0_S1():
    """In a multi-inversion setup, target_class='S' (bare) matches
    both 'S0' and 'S1' via fuzzy matching. This is intentional for
    convenience but users should be aware it's ambiguous."""
    Ne = 10_000
    L = 100_000.0
    t_inv = 20_000.0
    t_sweep = 500.0
    # x_sel inside inv 0
    sweep = Sweep(x_sel=35_000.0, t_event=t_sweep, target_class='S')

    sim = HullSimulator(
        n_std=5, n_inv=5,
        population_size=Ne, sequence_length=L,
        inversions=[
            InversionSpec(bp_left=20_000.0, bp_right=50_000.0,
                          p_inv=0.5, t_inv=t_inv),
            InversionSpec(bp_left=60_000.0, bp_right=90_000.0,
                          p_inv=0.5, t_inv=t_inv),
        ],
        recombination_rate=1e-8,
        sweeps=[sweep],
        seed=42,
    )
    ts = sim.simulate()
    samples = list(ts.samples())
    S = samples[:5]

    tree = ts.at(35_000.0)
    # Bare 'S' should match 'S0' at this position (inside inv 0).
    ss_mrca = tree.time(tree.mrca(*S))
    assert ss_mrca <= t_sweep + 1.0, (
        f"Bare 'S' should fuzzy-match 'S0' and sweep S lineages, "
        f"got T_MRCA={ss_mrca}")


# ---------------------------------------------------------------------------
# Sweep + non-overlapping lineages in apply_coalescence (Bug #7)
# ---------------------------------------------------------------------------

def test_sweep_hitchhiking_produces_valid_ts_at_moderate_rho():
    """Hitchhiking sweep at moderate rho: the tree sequence should be
    valid and T_MRCA at x_sel should reflect the sweep time."""
    Ne = 10_000
    L = 100_000.0
    t_sweep = 200.0
    s_coef = 0.01
    sweep = Sweep(x_sel=50_000.0, t_event=t_sweep,
                  target_class='P', selection_coefficient=s_coef)

    for seed in range(5):
        sim = HullSimulator(
            samples=10,
            population_size=Ne,
            sequence_length=L,
            recombination_rate=1e-8,  # rho=40
            sweeps=[sweep],
            seed=seed,
        )
        ts = sim.simulate()
        # ts should be internally consistent
        assert ts.num_samples == 10
        # T_MRCA at x_sel should be near t_sweep
        tree = ts.at(50_000.0)
        samples = list(ts.samples())
        tmrca = tree.time(tree.mrca(*samples))
        assert tmrca <= t_sweep + 1.0, (
            f"Hitchhiking sweep: T_MRCA at x_sel should be ~{t_sweep}, "
            f"got {tmrca} (seed={seed})")


@pytest.mark.parametrize("seed", range(5))
def test_sweep_window_mode_no_disjoint_corruption(seed):
    """Window-mode sweep: verify that apply_coalescence produces a
    valid tree at x_sel even when lineages have been fragmented by
    recombination."""
    Ne = 10_000
    L = 100_000.0
    t_sweep = 200.0
    sweep = Sweep(x_sel=50_000.0, t_event=t_sweep,
                  target_class='P', sweep_window=5_000.0)

    sim = HullSimulator(
        samples=10,
        population_size=Ne,
        sequence_length=L,
        recombination_rate=1e-8,  # rho=40
        sweeps=[sweep],
        seed=seed,
    )
    ts = sim.simulate()
    assert ts.num_samples == 10

    tree = ts.at(50_000.0)
    samples = list(ts.samples())
    tmrca = tree.time(tree.mrca(*samples))
    assert tmrca <= t_sweep + 1.0, (
        f"Window sweep: T_MRCA at x_sel should be ~{t_sweep}, "
        f"got {tmrca} (seed={seed})")

    # Far from x_sel should NOT be affected
    far_tree = ts.at(5_000.0)
    far_tmrca = far_tree.time(far_tree.mrca(*samples))
    assert far_tmrca > t_sweep * 2, (
        f"Far position should not be swept, got T_MRCA={far_tmrca}")


def test_sweep_hitchhiking_inside_inversion_with_recombination():
    """Hitchhiking sweep on S class inside an inversion at moderate rho.

    This exercises the sweep path where recombination fragments lineages
    before the sweep fires. The merged lineage may accumulate disjoint
    segments — verify the tree sequence is still valid and the sweep
    correctly hits x_sel."""
    Ne = 10_000
    L = 100_000.0
    t_inv = 20_000.0
    t_sweep = 500.0
    sweep = Sweep(x_sel=50_000.0, t_event=t_sweep,
                  target_class='S', selection_coefficient=0.01)

    for seed in range(5):
        sim = HullSimulator(
            n_std=5, n_inv=5,
            population_size=Ne,
            sequence_length=L,
            p_inv=0.5, t_inv=t_inv,
            bp_left=20_000.0, bp_right=80_000.0,
            gene_conversion_rate=1e-15,  # negligible (gamma>0 enforced)
            recombination_rate=1e-8,
            sweeps=[sweep],
            seed=seed,
        )
        ts = sim.simulate()
        assert ts.num_samples == 10

        samples = list(ts.samples())
        S = samples[:5]; I = samples[5:]
        tree = ts.at(50_000.0)

        # S samples at x_sel should coalesce at t_sweep
        ss_mrca = tree.time(tree.mrca(*S))
        assert ss_mrca <= t_sweep + 1.0, (
            f"S-class hitchhiking sweep: T_MRCA={ss_mrca}, "
            f"expected ~{t_sweep} (seed={seed})")

        # Cross-class barrier should still hold
        for s in S:
            for i in I:
                si_mrca = tree.time(tree.mrca(s, i))
                assert si_mrca >= t_inv - 1e-6, (
                    f"Cross-class barrier violated: "
                    f"S-I T_MRCA={si_mrca} < t_inv={t_inv}")
