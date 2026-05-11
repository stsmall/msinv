"""Phase-2 tests for the hull simulator: karyotype class barrier.

Verifies that with structured initialisation (n_std S samples + n_inv I
samples + finite t_inv), the resulting tree-sequence respects the class
barrier:

  T_MRCA(any S sample, any I sample) >= t_inv

This is the central correctness property that the SMC simulator
struggles to maintain under repeated prune-reattach events.
"""

import numpy as np
import pytest
import tskit

from msinv.hull import HullSimulator


# ---------------------------------------------------------------------------
# Phase 1 backwards compatibility
# ---------------------------------------------------------------------------


def test_panmictic_via_samples_arg_still_works():
    sim = HullSimulator(
        samples=8,
        population_size=1.0,
        sequence_length=10000.0,
        recombination_rate=1e-8,
        seed=1,
    )
    ts = sim.simulate()
    assert ts.num_samples == 8
    assert ts.num_trees >= 1


# ---------------------------------------------------------------------------
# Class barrier property
# ---------------------------------------------------------------------------


def _cross_class_mrca_times(ts, n_std):
    """For each (S sample, I sample) pair, return the tree T_MRCA in
    each marginal tree of ts."""
    samples = list(ts.samples())
    s_samples = samples[:n_std]
    i_samples = samples[n_std:]
    times = []
    for tree in ts.trees():
        for s in s_samples:
            for i in i_samples:
                times.append(tree.time(tree.mrca(s, i)))
    return np.array(times)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_cross_class_mrca_at_least_t_inv(seed):
    """SS-II MRCAs must be at or above t_inv."""
    n_std = 5
    n_inv = 5
    Ne = 10_000
    t_inv = 4.0 * 2 * Ne  # 4 Ne generations (deep but not infinite)
    p_inv = 0.5

    sim = HullSimulator(
        n_std=n_std,
        n_inv=n_inv,
        population_size=Ne,
        sequence_length=10000.0,
        recombination_rate=1e-8,
        p_inv=p_inv,
        t_inv=t_inv,
        seed=seed,
    )
    ts = sim.simulate()
    times = _cross_class_mrca_times(ts, n_std)
    # Every S-I MRCA must be >= t_inv (allow tiny FP slop).
    assert (times >= t_inv - 1e-6).all(), (
        f"Class barrier violated: min cross-class T_MRCA = {times.min():.2f} "
        f"vs t_inv = {t_inv}"
    )


def test_within_class_mrca_can_be_below_t_inv():
    """Within-S and within-I T_MRCA distributions should be much
    shallower than t_inv (otherwise we're not really running the
    structured coalescent — we'd just be waiting for t_inv)."""
    n_std = 10
    n_inv = 10
    Ne = 10_000
    t_inv = 4.0 * 2 * Ne
    p_inv = 0.5

    within_S = []
    within_I = []
    for seed in range(20):
        sim = HullSimulator(
            n_std=n_std,
            n_inv=n_inv,
            population_size=Ne,
            sequence_length=10000.0,
            recombination_rate=1e-8,
            p_inv=p_inv,
            t_inv=t_inv,
            seed=seed,
        )
        ts = sim.simulate()
        samples = list(ts.samples())
        S = samples[:n_std]
        I = samples[n_std:]
        for tree in ts.trees():
            # MRCA of all S samples
            within_S.append(tree.time(tree.mrca(*S)))
            within_I.append(tree.time(tree.mrca(*I)))
    within_S = np.array(within_S)
    within_I = np.array(within_I)
    # Most within-class MRCAs should be well below t_inv.
    assert np.median(within_S) < t_inv, (
        f"within-S median {np.median(within_S):.0f} >= t_inv {t_inv}"
    )
    assert np.median(within_I) < t_inv


# ---------------------------------------------------------------------------
# Structured rate scaling sanity
# ---------------------------------------------------------------------------


def test_rare_class_coalesces_faster_than_panmictic():
    """The I sub-population (when p_inv is small) has effective size
    Ne * p_inv → coalesces faster than a panmictic pop of the same n."""
    n_inv = 5
    Ne = 10_000
    p_inv = 0.1  # I is the rare class
    t_inv = 10.0 * 2 * Ne  # deep enough that class barrier doesn't constrain I-I coal

    rare_class_mrca = []
    panmictic_mrca = []
    for seed in range(40):
        sim = HullSimulator(
            n_std=1,
            n_inv=n_inv,  # 1 S so class barrier is mostly irrelevant
            population_size=Ne,
            sequence_length=10000.0,
            recombination_rate=1e-8,
            p_inv=p_inv,
            t_inv=t_inv,
            seed=seed,
        )
        ts = sim.simulate()
        samples = list(ts.samples())
        I = samples[1:]  # all I
        for tree in ts.trees():
            rare_class_mrca.append(tree.time(tree.mrca(*I)))

        sim_pan = HullSimulator(
            samples=n_inv,
            population_size=Ne,
            sequence_length=10000.0,
            recombination_rate=1e-8,
            seed=seed,
        )
        ts_pan = sim_pan.simulate()
        for tree in ts_pan.trees():
            panmictic_mrca.append(tree.time(tree.mrca(*list(ts_pan.samples()))))

    # Rare class with effective Ne = Ne * p_inv = Ne * 0.1 should
    # coalesce ~10x faster on average than panmictic.
    ratio = np.mean(panmictic_mrca) / np.mean(rare_class_mrca)
    assert 5.0 < ratio < 20.0, f"Expected rate ratio ~10x, got {ratio:.2f}"


# ---------------------------------------------------------------------------
# Tree sequence well-formedness
# ---------------------------------------------------------------------------


def test_treeseq_is_valid_with_class_barrier():
    sim = HullSimulator(
        n_std=4,
        n_inv=4,
        population_size=1000,
        sequence_length=10000.0,
        recombination_rate=1e-8,
        p_inv=0.5,
        t_inv=5000.0,
        seed=42,
    )
    ts = sim.simulate()
    # tskit's `dump_tables`/load roundtrip would catch any structural issues.
    tables = ts.dump_tables()
    tables.sort()
    ts2 = tables.tree_sequence()
    assert ts2.num_samples == 8
    assert ts2.first().num_roots == 1
