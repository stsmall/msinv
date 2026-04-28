"""Unit tests for msinv.hull._event_log helpers.

Pure-Python tests on synthetic dict-list event logs; no simulator
dependency.
"""
import numpy as np
import pytest

from msinv.hull._event_log import (
    filter_cmig, filter_flux, tract_lengths, survival_curve,
    coverage_count,
)


def test_require_log_raises_on_none():
    """Helpers raise ValueError with a clear message if the log is None
    (i.e., record_events was off when the sim ran)."""
    with pytest.raises(ValueError, match="record_events=True"):
        filter_cmig(None)
    with pytest.raises(ValueError, match="record_events=True"):
        filter_flux(None)


def test_filter_cmig_keeps_only_cmig_kind():
    log = [
        {"kind": "cmig", "src": 1, "dst": 0},
        {"kind": "flux", "tract_left": 0, "tract_right": 1},
    ]
    out = filter_cmig(log)
    assert len(out) == 1
    assert out[0]["src"] == 1


def test_filter_flux_inv_id_filter():
    log = [
        {"kind": "flux", "inv_id": 0, "tract_left": 0, "tract_right": 1},
        {"kind": "flux", "inv_id": 1, "tract_left": 0, "tract_right": 1},
    ]
    out = filter_flux(log, inv_id=0)
    assert len(out) == 1
    assert out[0]["inv_id"] == 0


def test_tract_lengths_returns_array():
    recs = [
        {"tract_left": 100, "tract_right": 250},
        {"tract_left": 0,   "tract_right": 50},
    ]
    lens = tract_lengths(recs)
    np.testing.assert_array_equal(lens, [150, 50])


def test_survival_curve_decreases_monotonically():
    """For uniform values on [0, 100), S(d) should decrease ~linearly."""
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 100, size=10_000)
    s = survival_curve(values, [25, 50, 75])
    assert s[0] > s[1] > s[2]
    np.testing.assert_allclose(s, [0.75, 0.5, 0.25], atol=0.02)


def test_coverage_count_inclusive_bounds():
    recs = [
        {"tract_left": 100, "tract_right": 200},
        {"tract_left": 150, "tract_right": 250},
        {"tract_left": 300, "tract_right": 400},
    ]
    assert coverage_count(recs, 175) == 2  # both first two cover 175
    assert coverage_count(recs, 100) == 1  # only the first (boundary inclusive)
    assert coverage_count(recs, 350) == 1  # only the last
    assert coverage_count(recs,  50) == 0


def test_samples_converted_at_empty_log_returns_zero():
    """No flux records → fraction == 0.0 regardless of ts."""
    import msprime
    from msinv.hull._event_log import samples_converted_at
    ts = msprime.sim_ancestry(samples=4, sequence_length=100,
                              recombination_rate=0, random_seed=1)
    assert samples_converted_at([], ts, 50.0) == 0.0


def test_samples_converted_at_root_node_returns_one():
    """A single record pointing at the root → all samples converted."""
    import msprime
    from msinv.hull._event_log import samples_converted_at
    ts = msprime.sim_ancestry(samples=4, sequence_length=100,
                              recombination_rate=0, random_seed=2)
    tree = ts.at(50.0)
    root = tree.root
    rec = {
        "kind": "flux",
        "tract_left": 0.0,
        "tract_right": 100.0,
        "tract_segments": [{"seg_left": 0.0, "seg_right": 100.0,
                            "node_id": int(root)}],
    }
    assert samples_converted_at([rec], ts, 50.0) == 1.0


def test_samples_converted_at_specific_descendants_match():
    """A record pointing at a non-root internal node → exactly its
    descendant-leaf set."""
    import msprime
    from msinv.hull._event_log import samples_converted_at
    ts = msprime.sim_ancestry(samples=8, sequence_length=100,
                              recombination_rate=0, random_seed=3)
    tree = ts.at(50.0)
    # Pick an internal node that is not the root and has at least 2
    # leaves below it.
    chosen = None
    for u in tree.nodes():
        if tree.is_internal(u) and u != tree.root:
            leaves = list(tree.samples(u))
            if len(leaves) >= 2:
                chosen = u
                break
    assert chosen is not None, "expected an internal non-root node"
    rec = {
        "kind": "flux",
        "tract_left": 0.0,
        "tract_right": 100.0,
        "tract_segments": [{"seg_left": 0.0, "seg_right": 100.0,
                            "node_id": int(chosen)}],
    }
    expected_frac = len(list(tree.samples(chosen))) / ts.num_samples
    assert samples_converted_at([rec], ts, 50.0) == expected_frac
