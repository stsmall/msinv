"""Phase-1 sanity tests for the hull simulator skeleton.

Verifies the ARG bookkeeping for the simplest case (panmictic, no
recombination): tskit can build a valid tree sequence with the
expected number of trees (1 tree spanning the chromosome) and the
expected number of internal nodes (samples - 1).
"""

import pytest
import tskit

from msinv.hull import HullSimulator


@pytest.mark.parametrize("n", [2, 5, 10, 20])
def test_panmictic_no_recomb_gives_single_tree(n):
    sim = HullSimulator(samples=n, population_size=1.0,
                         sequence_length=10000.0,
                         recombination_rate=1e-8, seed=42)
    ts = sim.simulate()
    assert ts.num_trees >= 1
    assert ts.num_samples == n
    tree = ts.first()
    assert tree.num_roots == 1


def test_segment_split_at_boundary():
    """Splitting at an exact segment boundary does not duplicate."""
    from msinv.hull.segment import Segment, split_segment_list
    s1 = Segment(0.0, 1.0, 0)
    s2 = Segment(1.0, 2.0, 1)
    s1.next = s2; s2.prev = s1
    (lh, lt), (rh, rt) = split_segment_list(s1, s2, 1.0)
    assert lh.left == 0.0 and lh.right == 1.0
    assert rh.left == 1.0 and rh.right == 2.0
    assert lh.next is None
    assert rh.prev is None


def test_segment_split_inside_segment():
    """Splitting inside a segment yields two new segments."""
    from msinv.hull.segment import Segment, split_segment_list
    s = Segment(0.0, 10.0, 0)
    (lh, lt), (rh, rt) = split_segment_list(s, s, 4.0)
    assert lh.left == 0.0 and lh.right == 4.0
    assert rh.left == 4.0 and rh.right == 10.0
    assert lh.node_id == rh.node_id == 0
