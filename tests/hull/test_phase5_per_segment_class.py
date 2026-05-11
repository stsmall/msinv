"""Phase-5 tests for the hull simulator: per-segment class.

Verifies that with the inversion as a sub-region of the chromosome
(not the whole sequence), the class barrier applies INSIDE the
inversion bounds but NOT to outside-inv positions:

  - dxy_SI INSIDE the inversion: same as before (cross-class can only
    coalesce after t_inv).
  - dxy_SI OUTSIDE the inversion: panmictic (no class barrier; can
    coalesce at any time).

This was the limitation flagged by the SMC-vs-hull comparison: under
phase 4, the class barrier was applied globally to the whole lineage,
giving wrongly-elevated outside-inv dxy_SI.
"""

import numpy as np
import pytest

from msinv.hull import HullSimulator

from .conftest import NEGLIGIBLE_GAMMA


# ---------------------------------------------------------------------------
# Initial-segment construction
# ---------------------------------------------------------------------------


def test_sample_segments_split_at_inv_bounds():
    """A sample with bp_left=3000, bp_right=7000 in L=10000 should have 3
    initial segments: outside-left ('P'), inside-inv ('S' or 'I'),
    outside-right ('P')."""
    sim = HullSimulator(
        n_std=2,
        n_inv=2,
        population_size=1000,
        sequence_length=10000.0,
        p_inv=0.5,
        t_inv=10_000.0,
        bp_left=3000.0,
        bp_right=7000.0,
        recombination_rate=1e-8,
        seed=42,
    )
    from msinv.hull.tables import TableBuilder

    tables = TableBuilder(sequence_length=10000.0, num_populations=1)
    active = sim._initial_lineages(tables)
    # First lineage is class S
    lin = active[0]
    classes = []
    seg = lin.head
    while seg is not None:
        classes.append((seg.left, seg.right, seg.branch_class))
        seg = seg.next
    assert classes == [
        (0.0, 3000.0, "P"),
        (3000.0, 7000.0, "S"),
        (7000.0, 10000.0, "P"),
    ]


# ---------------------------------------------------------------------------
# Cross-class T_MRCA depends on POSITION
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_inside_class_barrier_outside_panmictic(seed):
    """Cross-class T_MRCA should be >= t_inv ONLY inside the inversion;
    outside positions can coalesce panmictically."""
    n_std = 4
    n_inv = 4
    Ne = 1000
    t_inv = 4.0 * 2 * Ne  # 8000 gen
    L = 10000.0
    bp_left = 3000.0
    bp_right = 7000.0

    sim = HullSimulator(
        n_std=n_std,
        n_inv=n_inv,
        population_size=Ne,
        sequence_length=L,
        p_inv=0.5,
        t_inv=t_inv,
        bp_left=bp_left,
        bp_right=bp_right,
        recombination_rate=1e-8,
        seed=seed,
    )
    ts = sim.simulate()
    samples = list(ts.samples())
    S = samples[:n_std]
    I = samples[n_std:]

    # Walk every tree; check positional class-barrier semantics.
    inside_violations = 0
    outside_below_t_inv = 0
    outside_total = 0
    for tree in ts.trees():
        l, r = tree.interval.left, tree.interval.right
        in_w = max(0.0, min(r, bp_right) - max(l, bp_left))
        out_w = (r - l) - in_w
        for s in S:
            for i in I:
                t_mrca = tree.time(tree.mrca(s, i))
                # Inside-inv positions in this interval must have T_MRCA
                # >= t_inv. Outside-inv positions are unconstrained.
                if in_w > 0 and t_mrca < t_inv - 1e-6:
                    inside_violations += 1
                if out_w > 0:
                    outside_total += 1
                    if t_mrca < t_inv - 1e-6:
                        outside_below_t_inv += 1
    # gc_rate > 0 allows rare flux-mediated cross-class MRCA before
    # t_inv (a lineage gets its class flipped by gene conversion and
    # then coalesces across classes). A single such event can cascade
    # into up to n_std * n_inv = 16 pair-tree violations. Allow a small
    # margin so stochastic flux outcomes don't trip the barrier check.
    assert inside_violations <= 20, (
        f"inside-inv class barrier violated {inside_violations} times "
        f"— far more than a single flux-mediated MRCA would produce"
    )
    # Outside-inv: most cross-class MRCAs should be MUCH younger than t_inv
    # (panmictic coal). At Ne=1000 and t_inv=8000, we expect mean T_MRCA
    # ~ 2*Ne = 2000 << 8000.
    assert outside_below_t_inv > 0, (
        "Outside-inv cross-class MRCAs should be able to occur before "
        "t_inv (panmictic). Got 0 — class barrier is being incorrectly "
        "applied to outside-inv positions."
    )
    frac = outside_below_t_inv / max(outside_total, 1)
    assert frac > 0.5, (
        f"Outside-inv: only {frac:.0%} of cross-class MRCAs are below "
        f"t_inv; expected most (panmictic). Class barrier may still be "
        f"applied globally."
    )


# ---------------------------------------------------------------------------
# Within-class T_MRCA panmictic outside, structured inside
# ---------------------------------------------------------------------------


def test_within_class_outside_panmictic():
    """For S-S samples: inside the inversion they're in the S sub-pop
    (effective Ne·p_std), outside they're in the full pop. So outside
    T_MRCA should be ~2× inside T_MRCA when p_std=0.5."""
    n_std = 5
    n_inv = 5
    Ne = 1000
    t_inv = 8_000.0  # 4*Ne; deep enough that class barrier doesn't constrain S-S
    L = 10000.0
    bp_left = 2500.0
    bp_right = 7500.0

    inside_T = []
    outside_T = []
    for seed in range(15):
        sim = HullSimulator(
            n_std=n_std,
            n_inv=n_inv,
            population_size=Ne,
            sequence_length=L,
            p_inv=0.5,
            t_inv=t_inv,
            bp_left=bp_left,
            bp_right=bp_right,
            recombination_rate=1e-8,
            seed=seed,
        )
        ts = sim.simulate()
        S = list(ts.samples())[:n_std]
        for tree in ts.trees():
            l, r = tree.interval.left, tree.interval.right
            in_w = max(0.0, min(r, bp_right) - max(l, bp_left))
            out_w = (r - l) - in_w
            tmrca = tree.time(tree.mrca(*S))
            if in_w > 0:
                inside_T.append((tmrca, in_w))
            if out_w > 0:
                outside_T.append((tmrca, out_w))

    def _wmean(pairs):
        ts_, ws = zip(*pairs)
        return float(np.average(ts_, weights=ws))

    in_mean = _wmean(inside_T)
    out_mean = _wmean(outside_T)
    ratio = in_mean / out_mean
    # Expectation: inside ~ 2·Ne·p_std = 1000, outside ~ 2·Ne = 2000
    # → inside/outside ratio ~ 0.5. Allow generous tolerance.
    assert 0.3 < ratio < 0.8, (
        f"Inside/outside within-S T_MRCA ratio = {ratio:.2f}, expected ~0.5 (= p_std)."
    )


# ---------------------------------------------------------------------------
# Tree-sequence well-formedness with per-segment class
# ---------------------------------------------------------------------------


def test_treeseq_valid_with_inv_subregion():
    sim = HullSimulator(
        n_std=4,
        n_inv=4,
        population_size=1000,
        sequence_length=20000.0,
        p_inv=0.5,
        t_inv=8000.0,
        bp_left=5000.0,
        bp_right=15000.0,
        recombination_rate=1e-8,
        gene_conversion_rate=NEGLIGIBLE_GAMMA,
        seed=42,
    )
    ts = sim.simulate()
    # Round-trip via tables.
    tables = ts.dump_tables()
    tables.sort()
    ts2 = tables.tree_sequence()
    assert ts2.num_samples == 8
    for tree in ts2.trees():
        # Should still have a single root in every tree (every position
        # is fully ancestrally resolved).
        assert tree.num_roots == 1
