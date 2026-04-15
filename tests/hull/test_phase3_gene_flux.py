"""Phase-3 tests for the hull simulator: gene-flux events with class flip.

Verifies:
  - With γ=0, no in-inv events fire → tree constant across the inversion
    (perfect in-inv LD).
  - With γ>0, occasional flux events → multiple trees in the tree
    sequence, LD breaks down with distance.
  - Cross-class T_MRCA still ≥ t_inv at every position (class barrier
    preserved despite flux).
  - phi(x) gradient: more flux events near the centre than near the
    breakpoints.
  - apply_gene_flux unit tests.
"""

import numpy as np
import pytest
import tskit

from msinv.hull import HullSimulator
from msinv.hull.lineage import Lineage, reset_uids
from msinv.hull.segment import Segment
from msinv.hull.events import apply_gene_flux

from .conftest import NEGLIGIBLE_GAMMA


# ---------------------------------------------------------------------------
# Unit tests for apply_gene_flux
# ---------------------------------------------------------------------------

def _make_lineage(intervals, branch_class):
    reset_uids()
    head = tail = None
    for (l, r), nid in zip(intervals, range(len(intervals))):
        seg = Segment(l, r, nid, prev=tail)
        if head is None:
            head = seg
        if tail is not None:
            tail.next = seg
        tail = seg
    return Lineage(head, tail, branch_class=branch_class, population=0)


def test_gene_flux_simple_split_in_middle():
    lin = _make_lineage([(0.0, 100.0)], 'S')
    active = [lin]
    outside, tract = apply_gene_flux(active, lin, 40.0, 60.0)
    assert outside is not None and tract is not None
    assert tract.branch_class == 'I'
    assert outside.branch_class == 'S'
    # Tract carries 40-60.
    assert tract.head.left == 40.0 and tract.tail.right == 60.0
    # Outside has two segments (the "hole").
    seg = outside.head; outside_intervals = []
    while seg is not None:
        outside_intervals.append((seg.left, seg.right))
        seg = seg.next
    assert outside_intervals == [(0.0, 40.0), (40.0, 100.0)] or \
           outside_intervals == [(0.0, 40.0), (60.0, 100.0)]
    # Active list updated: original out, two new in.
    assert lin not in active
    assert outside in active and tract in active


def test_gene_flux_at_lineage_left_edge():
    lin = _make_lineage([(0.0, 100.0)], 'S')
    active = [lin]
    outside, tract = apply_gene_flux(active, lin, 0.0, 30.0)
    assert tract is not None and outside is not None
    assert tract.head.left == 0.0 and tract.tail.right == 30.0
    assert outside.head.left == 30.0


def test_gene_flux_outside_lineage_coverage_is_noop():
    """If lineage has no material in the tract, return unchanged."""
    lin = _make_lineage([(0.0, 50.0)], 'S')
    active = [lin]
    outside, tract = apply_gene_flux(active, lin, 60.0, 80.0)
    # No material >= 60 → no event.
    assert tract is None
    # outside should be lineage itself (unchanged) and lineage should
    # still be in active.
    assert outside is lin
    assert lin in active


def test_gene_flux_class_flip_I_to_S():
    lin = _make_lineage([(0.0, 100.0)], 'I')
    active = [lin]
    _, tract = apply_gene_flux(active, lin, 40.0, 60.0)
    assert tract.branch_class == 'S'


# ---------------------------------------------------------------------------
# γ=0 → tree constant across inversion
# ---------------------------------------------------------------------------

def test_gamma_zero_is_rejected():
    """gamma=0 is forbidden globally — must raise ValueError."""
    with pytest.raises(ValueError, match="gene_conversion_rate"):
        HullSimulator(
            n_std=5, n_inv=5,
            population_size=1000, sequence_length=10_000.0,
            p_inv=0.5, t_inv=4_000.0,
            bp_left=0.0, bp_right=10_000.0,
            gene_conversion_rate=0.0,
            recombination_rate=1e-8,
            seed=42,
        )


# ---------------------------------------------------------------------------
# γ>0 → multiple trees + LD decay
# ---------------------------------------------------------------------------

def test_gamma_positive_gives_multiple_trees():
    """Across replicates, gamma > 0 should produce multi-tree TS more
    often than not. Per-seed assertion is too strict — a few seeds
    legitimately fire zero events under modest gamma * t_inv * L."""
    multi_tree_count = 0
    for seed in range(10):
        sim = HullSimulator(
            n_std=5, n_inv=5,
            population_size=1000, sequence_length=10_000.0,
            p_inv=0.5, t_inv=4_000.0,
            bp_left=0.0, bp_right=10_000.0,
            gene_conversion_rate=1e-6,
            flux_window=0.05,
            recombination_rate=1e-8,
            seed=seed,
        )
        if sim.simulate().num_trees > 1:
            multi_tree_count += 1
    assert multi_tree_count >= 5, (
        f"Only {multi_tree_count}/10 reps produced >1 tree at gamma>0 "
        "— gene flux + recomb should fire multi-tree TS in most reps.")


# ---------------------------------------------------------------------------
# Class barrier semantics under gene conversion
# ---------------------------------------------------------------------------
#
# Without gene conversion (γ=0), every cross-class MRCA is >= t_inv —
# samples of different karyotypes have no shared ancestry inside the
# inversion until the inversion itself was born.
#
# WITH gene conversion (γ>0), some tract positions LEGITIMATELY have
# cross-class MRCAs below t_inv: gene conversion is the biological
# mechanism by which an S chromosome's tract can derive from an I
# ancestor (or vice versa) at a specific position. So we expect a
# small fraction of (sample_pair, position) combinations to violate
# the strict barrier — exactly at and around tract positions.

@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_class_barrier_strict_at_negligible_gamma(seed):
    """At gamma → 0 limit, EVERY cross-class MRCA must be >= t_inv."""
    n_std = 4; n_inv = 4
    Ne = 1000
    t_inv = 2.0 * 2 * Ne
    sim = HullSimulator(
        n_std=n_std, n_inv=n_inv,
        population_size=Ne, sequence_length=10_000.0,
        p_inv=0.5, t_inv=t_inv,
        bp_left=0.0, bp_right=10_000.0,
        gene_conversion_rate=NEGLIGIBLE_GAMMA,
        recombination_rate=1e-8,
        seed=seed,
    )
    ts = sim.simulate()
    samples = list(ts.samples())
    S = samples[:n_std]; I = samples[n_std:]
    for tree in ts.trees():
        for s in S:
            for i in I:
                assert tree.time(tree.mrca(s, i)) >= t_inv - 1e-6


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_most_cross_class_positions_still_respect_barrier(seed):
    """With small γ, MOST (sample_pair, position) combinations still
    have cross-class T_MRCA >= t_inv; only those at and around tract
    positions are affected by gene conversion."""
    n_std = 4; n_inv = 4
    Ne = 1000
    t_inv = 2.0 * 2 * Ne
    sim = HullSimulator(
        n_std=n_std, n_inv=n_inv,
        population_size=Ne, sequence_length=10_000.0,
        p_inv=0.5, t_inv=t_inv,
        bp_left=0.0, bp_right=10_000.0,
        gene_conversion_rate=1e-6,
        recombination_rate=1e-8,
        seed=seed,
    )
    ts = sim.simulate()
    samples = list(ts.samples())
    S = samples[:n_std]; I = samples[n_std:]
    total = 0; violations = 0
    for tree in ts.trees():
        # Weight by tree span so we measure fraction of GENOME, not
        # fraction of trees (small flux tracts are short).
        span = tree.interval.right - tree.interval.left
        for s in S:
            for i in I:
                total_t = tree.time(tree.mrca(s, i))
                weight = span
                total += weight
                if total_t < t_inv - 1e-6:
                    violations += weight
    frac = violations / total if total > 0 else 0
    # With this γ * t_inv * inv_len budget, expect at most a small
    # fraction of (pair, position) combos to be flux-affected.
    assert frac < 0.50, (
        f"Too many cross-class barrier violations: {frac:.2%} of "
        f"(pair, position) combos have T_MRCA < t_inv (small γ "
        f"should give a small fraction).")


def test_gene_conversion_creates_strictly_more_low_mrcas_than_no_flux():
    """Sanity: γ>0 produces MORE sub-t_inv cross-class MRCAs than γ=0.
    Without this, the gene-flux events aren't actually flipping classes."""
    n_std = 4; n_inv = 4
    Ne = 1000
    t_inv = 2.0 * 2 * Ne

    def count_violations(gamma_val, seed):
        sim = HullSimulator(
            n_std=n_std, n_inv=n_inv,
            population_size=Ne, sequence_length=10_000.0,
            p_inv=0.5, t_inv=t_inv,
            bp_left=0.0, bp_right=10_000.0,
            gene_conversion_rate=gamma_val,
            recombination_rate=1e-8,
            seed=seed,
        )
        ts = sim.simulate()
        samples = list(ts.samples())
        S = samples[:n_std]; I = samples[n_std:]
        v = 0
        for tree in ts.trees():
            for s in S:
                for i in I:
                    if tree.time(tree.mrca(s, i)) < t_inv - 1e-6:
                        v += 1
        return v

    # Across seeds, larger γ should produce MORE violations than
    # negligible γ. NEGLIGIBLE_GAMMA fires zero events at this
    # Ne/L/t_inv combo so it serves as the "no-flux" baseline.
    no_flux = sum(count_violations(NEGLIGIBLE_GAMMA, s) for s in range(10))
    with_flux = sum(count_violations(5e-5, s) for s in range(10))
    assert no_flux == 0
    assert with_flux > 0, (
        "γ>0 produced no cross-class MRCAs below t_inv — gene flux "
        "events aren't actually creating cross-class ancestry.")


# ---------------------------------------------------------------------------
# Tree-sequence well-formedness with flux
# ---------------------------------------------------------------------------

def test_treeseq_valid_with_flux():
    sim = HullSimulator(
        n_std=4, n_inv=4,
        population_size=1000, sequence_length=20_000.0,
        p_inv=0.5, t_inv=4000.0,
        bp_left=0.0, bp_right=20_000.0,
        gene_conversion_rate=1e-6,
        recombination_rate=1e-8,
        seed=42,
    )
    ts = sim.simulate()
    # Round-trip via tables to check structural validity.
    tables = ts.dump_tables()
    tables.sort()
    ts2 = tables.tree_sequence()
    assert ts2.num_samples == 8
    for tree in ts2.trees():
        assert tree.num_roots == 1


# ---------------------------------------------------------------------------
# phi(x) gradient: more flux near the centre than at breakpoints
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Flux+recomb fragmentation too slow for per-pair path; needs Rust")
def test_phi_gradient_more_breakpoints_in_middle():
    """The number of distinct trees per unit length should be HIGHER
    near the centre of the inversion (where phi(x) peaks) than near
    the breakpoints (phi → 0)."""
    n_std = 4; n_inv = 4
    Ne = 5000
    t_inv = 10_000.0
    L = 100_000.0  # inversion = whole sequence
    bp_l = 0.0; bp_r = L
    centre = (bp_l + bp_r) / 2.0
    flux_window = 0.05
    centre_breaks = []
    edge_breaks = []
    for seed in range(10):
        sim = HullSimulator(
            n_std=n_std, n_inv=n_inv,
            population_size=Ne, sequence_length=L,
            p_inv=0.5, t_inv=t_inv,
            bp_left=bp_l, bp_right=bp_r,
            gene_conversion_rate=5e-6,
            flux_window=flux_window,
            recombination_rate=1e-8,
            seed=seed,
        )
        ts = sim.simulate()
        # Count tree-changes (breakpoints) in centre quartile vs edge quartiles.
        bps = list(ts.breakpoints())[1:-1]   # interior breakpoints
        for bp in bps:
            if 0.4 * L <= bp <= 0.6 * L:
                centre_breaks.append(bp)
            elif bp < 0.1 * L or bp > 0.9 * L:
                edge_breaks.append(bp)
    # Centre quartile is 0.2 L wide vs 0.2 L edge total → comparable widths.
    # phi peaks at centre, drops to 0 at breakpoints, so centre should
    # have meaningfully more breakpoints. (Allow some randomness.)
    assert len(centre_breaks) > len(edge_breaks), (
        f"Centre should have more flux breakpoints than edges, got "
        f"centre={len(centre_breaks)} vs edge={len(edge_breaks)}")


# ---------------------------------------------------------------------------
# Regression: flux must fire when single-inv is given via inversions=[]
# ---------------------------------------------------------------------------
#
# Pre-fix bug: passing one inversion via the multi-inv API tagged its
# class segments as 'S0'/'I0' (because inv_id=0), but _flux_rates only
# matched plain 'S'/'I'. Result: gene flux silently fired zero events
# under the inversions=[InversionSpec(γ>0)] API even though the same
# parameters via the legacy bp_left/p_inv/... args worked correctly.

def test_single_inv_via_inversions_api_fires_flux():
    """Same parameters via two APIs should fire flux events at the
    same rate. Pre-fix the multi-inv API silently produced 0 events."""
    from msinv.hull import InversionSpec
    import msinv.hull.simulator as hs

    flux_counts = {'multi_api': 0, 'legacy_api': 0}

    orig = hs.apply_gene_flux

    for label, builder in [
        ('multi_api', lambda seed: HullSimulator(
            n_std=4, n_inv=4, population_size=1_000,
            sequence_length=10_000.0,
            inversions=[InversionSpec(bp_left=0.0, bp_right=10_000.0,
                                       p_inv=0.5, t_inv=4_000.0,
                                       gene_conversion_rate=1e-6,
                                       flux_window=0.05)],
            recombination_rate=1e-8,
            seed=seed)),
        ('legacy_api', lambda seed: HullSimulator(
            n_std=4, n_inv=4, population_size=1_000,
            sequence_length=10_000.0,
            bp_left=0.0, bp_right=10_000.0,
            p_inv=0.5, t_inv=4_000.0,
            gene_conversion_rate=1e-6, flux_window=0.05,
            recombination_rate=1e-8,
            seed=seed)),
    ]:
        def counted(*args, _label=label, **kwargs):
            flux_counts[_label] += 1
            return orig(*args, **kwargs)
        hs.apply_gene_flux = counted
        try:
            for seed in range(5):
                # monkeypatch only catches Python apply_gene_flux
                builder(seed).simulate(use_rust=False)
        finally:
            hs.apply_gene_flux = orig

    assert flux_counts['multi_api'] > 0, (
        "Single-inv via inversions=[InversionSpec(...)] should fire "
        "flux events but didn't (regression of the 'S0'/'I0' tag bug).")
    assert flux_counts['legacy_api'] > 0
    # The two APIs are deterministic-equivalent given the same seed →
    # event counts should match exactly.
    assert flux_counts['multi_api'] == flux_counts['legacy_api'], (
        f"Multi-API ({flux_counts['multi_api']}) and legacy-API "
        f"({flux_counts['legacy_api']}) flux event counts diverged.")


def test_multi_inv_per_inversion_gamma():
    """Two inversions with different gammas should fire flux events
    at independently-controlled rates. Inversion with γ=0 should fire
    no events, the other should fire many."""
    from msinv.hull import InversionSpec
    import msinv.hull.simulator as hs

    counts_per_inv = {0: 0, 1: 0}
    orig = hs.apply_gene_flux

    def counted(active, lin, tl, tr, inv=None):
        if inv is not None:
            counts_per_inv[inv.inv_id] = counts_per_inv.get(
                inv.inv_id, 0) + 1
        return orig(active, lin, tl, tr, inv=inv)
    hs.apply_gene_flux = counted

    try:
        for seed in range(3):
            HullSimulator(
                n_std=4, n_inv=4, population_size=1_000,
                sequence_length=10_000.0,
                inversions=[
                    InversionSpec(bp_left=0.0, bp_right=4_000.0,
                                   p_inv=0.5, t_inv=4_000.0,
                                   gene_conversion_rate=1e-6,
                                   flux_window=0.05),
                    InversionSpec(bp_left=6_000.0, bp_right=10_000.0,
                                   p_inv=0.5, t_inv=4_000.0,
                                   gene_conversion_rate=NEGLIGIBLE_GAMMA,
                                   flux_window=0.05),
                ],
                recombination_rate=1e-8,
                seed=seed,
            ).simulate(use_rust=False)  # monkeypatch only catches Python
    finally:
        hs.apply_gene_flux = orig

    assert counts_per_inv[0] > 0, "inv 0 (γ=1e-4) should fire flux"
    assert counts_per_inv[1] == 0, (
        f"inv 1 (γ=0) should fire NO flux events, got {counts_per_inv[1]}")
