"""Phase-5c.1 tests: independent karyotype per inversion.

A sample's karyotype is now per-inversion. With ``inversions=[A, B]``
on the chromosome and a sample with karyotype ``('S', 'I')``, the
sample is S at inv A and I at inv B independently. Inv A's class
barrier acts on the S/S vs S/I cross-class pair regardless of inv B
status, and vice versa.

Verifies:
  - Sample initialization assigns per-inv classes correctly.
  - Linked karyotype (single 'S' or 'I') still works (back-compat).
  - Multi-char string shorthand ('SI', 'IS') works.
  - Tuple-of-karyotypes works.
  - Per-inv class barrier still respected when karyotypes are
    independent (inv 0's barrier doesn't depend on inv 1).
"""

import numpy as np
import pytest

from msinv.hull import HullSimulator, InversionSpec
from msinv.hull.tables import TableBuilder


# ---------------------------------------------------------------------------
# Sample-init: per-inv class assignments
# ---------------------------------------------------------------------------

def _classes_of_lineage(lin):
    out = []
    seg = lin.head
    while seg is not None:
        out.append((seg.left, seg.right, seg.branch_class))
        seg = seg.next
    return out


def test_linked_karyotype_uses_single_S_for_both_invs():
    """Linked karyotype: sample 'S' assigns S<inv_id> at every inv."""
    inv0 = InversionSpec(bp_left=10.0, bp_right=30.0,
                          p_inv=0.5, t_inv=1000.0)
    inv1 = InversionSpec(bp_left=50.0, bp_right=80.0,
                          p_inv=0.5, t_inv=1000.0)
    sim = HullSimulator(
        sample_config={('S', 0): 1},
        population_size=1000, sequence_length=100.0,
        inversions=[inv0, inv1], seed=1)
    tables = TableBuilder(sequence_length=100.0)
    active = sim._initial_lineages(tables)
    classes = _classes_of_lineage(active[0])
    assert classes == [
        (0.0, 10.0, 'P'),
        (10.0, 30.0, 'S0'),
        (30.0, 50.0, 'P'),
        (50.0, 80.0, 'S1'),
        (80.0, 100.0, 'P'),
    ]


def test_independent_karyotype_tuple_assigns_per_inv():
    """Independent karyotype ('S', 'I') → S0 at inv 0 and I1 at inv 1."""
    inv0 = InversionSpec(bp_left=10.0, bp_right=30.0,
                          p_inv=0.5, t_inv=1000.0)
    inv1 = InversionSpec(bp_left=50.0, bp_right=80.0,
                          p_inv=0.5, t_inv=1000.0)
    sim = HullSimulator(
        sample_config={(('S', 'I'), 0): 1},
        population_size=1000, sequence_length=100.0,
        inversions=[inv0, inv1], seed=1)
    tables = TableBuilder(sequence_length=100.0)
    active = sim._initial_lineages(tables)
    classes = _classes_of_lineage(active[0])
    assert classes[1] == (10.0, 30.0, 'S0')
    assert classes[3] == (50.0, 80.0, 'I1')


def test_string_shorthand_for_two_invs():
    """The 2-char string 'SI' → equivalent to ('S', 'I') tuple."""
    inv0 = InversionSpec(bp_left=10.0, bp_right=30.0,
                          p_inv=0.5, t_inv=1000.0)
    inv1 = InversionSpec(bp_left=50.0, bp_right=80.0,
                          p_inv=0.5, t_inv=1000.0)
    sim = HullSimulator(
        sample_config={('SI', 0): 1},
        population_size=1000, sequence_length=100.0,
        inversions=[inv0, inv1], seed=1)
    tables = TableBuilder(sequence_length=100.0)
    active = sim._initial_lineages(tables)
    classes = _classes_of_lineage(active[0])
    assert classes[1] == (10.0, 30.0, 'S0')
    assert classes[3] == (50.0, 80.0, 'I1')


def test_None_at_one_inv_uses_panmictic_there():
    """Karyotype tuple ('S', None) → S at inv 0, panmictic at inv 1."""
    inv0 = InversionSpec(bp_left=10.0, bp_right=30.0,
                          p_inv=0.5, t_inv=1000.0)
    inv1 = InversionSpec(bp_left=50.0, bp_right=80.0,
                          p_inv=0.5, t_inv=1000.0)
    sim = HullSimulator(
        sample_config={(('S', None), 0): 1},
        population_size=1000, sequence_length=100.0,
        inversions=[inv0, inv1], seed=1)
    tables = TableBuilder(sequence_length=100.0)
    active = sim._initial_lineages(tables)
    classes = _classes_of_lineage(active[0])
    # inv 0: S0; inv 1: P; flanks: P
    assert classes[1] == (10.0, 30.0, 'S0')
    assert classes[3] == (50.0, 80.0, 'P')


def test_wrong_length_raises():
    """Mismatched karyotype tuple length raises ValueError."""
    inv0 = InversionSpec(bp_left=10.0, bp_right=30.0,
                          p_inv=0.5, t_inv=1000.0)
    inv1 = InversionSpec(bp_left=50.0, bp_right=80.0,
                          p_inv=0.5, t_inv=1000.0)
    sim = HullSimulator(
        sample_config={(('S', 'I', 'S'), 0): 1},  # 3 entries for 2 invs
        population_size=1000, sequence_length=100.0,
        inversions=[inv0, inv1], seed=1)
    tables = TableBuilder(sequence_length=100.0)
    with pytest.raises(ValueError, match="entries but there are"):
        sim._initial_lineages(tables)


# ---------------------------------------------------------------------------
# Class barrier behaviour with independent karyotype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_independent_karyotype_each_inv_barrier_independent(seed):
    """Two invs with their own t_inv. Samples ('S', 'I') vs ('S', 'S')
    should have:
      - inv 0: same class (S/S) → no class barrier inside inv 0.
      - inv 1: different class (I vs S) → barrier inside inv 1.

    In our model the barrier only matters for cross-class pairs. We
    test by setting up two samples with karyotypes ('S', 'S') and
    ('S', 'I'), then checking that their MRCA inside inv 1 is
    >= inv 1's t_inv.
    """
    Ne = 1000
    L = 100.0
    inv0 = InversionSpec(bp_left=10.0, bp_right=40.0,
                          p_inv=0.5, t_inv=2000.0)
    inv1 = InversionSpec(bp_left=60.0, bp_right=90.0,
                          p_inv=0.5, t_inv=4000.0)

    # 4 samples, all 'S' at inv 0, but split S/I at inv 1.
    sim = HullSimulator(
        sample_config={
            (('S', 'S'), 0): 2,   # SS samples
            (('S', 'I'), 0): 2,   # SI samples (S at inv 0, I at inv 1)
        },
        population_size=Ne, sequence_length=L,
        inversions=[inv0, inv1], seed=seed)
    ts = sim.simulate()
    samples = list(ts.samples())
    SS = samples[:2]
    SI = samples[2:]

    # Inside inv 1: SS samples are S<1>, SI samples are I<1>. Cross-
    # class T_MRCA must be >= inv 1's t_inv = 4000.
    inv1_violations = 0
    for tree in ts.trees():
        l, r = tree.interval.left, tree.interval.right
        if l < inv1.bp_left or r > inv1.bp_right:
            continue
        for s in SS:
            for i in SI:
                tmrca = tree.time(tree.mrca(s, i))
                if tmrca < inv1.t_inv - 1e-6:
                    inv1_violations += 1
    assert inv1_violations == 0


def test_independent_karyotype_inv0_panmictic_when_all_S():
    """When all samples are 'S' at inv 0, inv 0's barrier is moot —
    cross-class T_MRCA at inv 0 should be ≪ inv 0's t_inv."""
    Ne = 1000
    L = 100.0
    inv0 = InversionSpec(bp_left=10.0, bp_right=40.0,
                          p_inv=0.5, t_inv=10_000.0)  # very deep
    inv1 = InversionSpec(bp_left=60.0, bp_right=90.0,
                          p_inv=0.5, t_inv=2000.0)

    sim = HullSimulator(
        sample_config={
            (('S', 'S'), 0): 3,
            (('S', 'I'), 0): 3,
        },
        population_size=Ne, sequence_length=L,
        inversions=[inv0, inv1], seed=42)
    ts = sim.simulate()
    samples = list(ts.samples())
    # Inside inv 0, ALL samples are S → SS pair MRCA should follow
    # standard structured-coal expectation (~2·Ne·p_std = 1000), NOT
    # inv 0's t_inv = 10000.
    SS_pairs_mrca = []
    for tree in ts.trees():
        l, r = tree.interval.left, tree.interval.right
        if l < inv0.bp_left or r > inv0.bp_right:
            continue
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                SS_pairs_mrca.append(tree.time(tree.mrca(samples[i], samples[j])))
    if SS_pairs_mrca:
        median_t = np.median(SS_pairs_mrca)
        assert median_t < inv0.t_inv / 2, (
            f"Inv 0 with all-S samples should NOT impose t_inv barrier; "
            f"median T_MRCA = {median_t:.0f}, t_inv = {inv0.t_inv}")
