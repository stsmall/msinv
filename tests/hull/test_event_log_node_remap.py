"""Event-log node IDs must name nodes of the *simplified* tree sequence.

`rust_simulate` simplifies the raw tables before returning them. Any node
that simplify filters (an orphan node from a no-overlap partial merge,
nodes above the root) renumbers every later node, so raw event-log IDs
would silently point at the wrong node. The bridge remaps them through
simplify's node map; this test feeds it raw tables with an orphan node.
"""

from types import SimpleNamespace

import numpy as np

from msinv.hull import _rust_bridge
from msinv.hull._event_log import (
    filter_flux,
    remap_event_log_nodes,
    samples_converted_at,
)

L = 100.0


def _raw_with_orphan():
    # 0, 1, 2: samples. 3: orphan (no edges). 4 = parent(0, 1) at t=1,
    # 5 = parent(4, 2) at t=2. simplify drops 3: old 4 -> 3, old 5 -> 4.
    return {
        "sequence_length": L,
        "num_populations": 1,
        "node_flags": np.array([1, 1, 1, 0, 0, 0], dtype=np.uint32),
        "node_time": np.array([0.0, 0.0, 0.0, 0.5, 1.0, 2.0]),
        "node_population": np.zeros(6, dtype=np.int32),
        # tskit canonical order: time[parent], parent, child, left.
        "edge_left": np.zeros(4),
        "edge_right": np.full(4, L),
        "edge_parent": np.array([4, 4, 5, 5], dtype=np.int32),
        "edge_child": np.array([0, 1, 2, 4], dtype=np.int32),
        "sweep_a_count": 0,
    }


def _log():
    return [
        {"kind": "cmig", "t": 0.1, "src": 0, "dst": 0, "kary": "I",
         "inv_id": 0, "n_eligible": 1, "n_moved": 0},
        {"kind": "flux", "t": 0.7, "lineage_uid": 9, "position": 25.0,
         "tract_left": 0.0, "tract_right": L, "inv_id": 0,
         "tract_segments": [
             # raw node 4 = MRCA of samples {0, 1}
             {"seg_left": 0.0, "seg_right": 50.0, "node_id": 4},
             # raw node 3 = orphan, absent from the simplified ts
             {"seg_left": 50.0, "seg_right": L, "node_id": 3},
         ]},
    ]


def _stub_simulator():
    demo = SimpleNamespace(n_pops=1, pop_sizes=[1000.0],
                           migration_matrix=[[0.0]], events=[])
    return SimpleNamespace(sample_config={(None, 0): 3}, inversions=[],
                           sweeps=[], demography=demo, L=L, r=1e-8,
                           _record_events=True)


def test_event_log_ids_remapped_after_simplify(monkeypatch):
    log = _log()
    monkeypatch.setattr(_rust_bridge, "_simulate_raw",
                        lambda **kw: (_raw_with_orphan(), log),
                        raising=False)
    ts, out_log, _ = _rust_bridge.rust_simulate(_stub_simulator())

    assert ts.num_nodes == 5  # the orphan was filtered
    segs = filter_flux(out_log)[0]["tract_segments"]
    # Raw 4 -> 3; the orphan's entry is dropped.
    assert [s["node_id"] for s in segs] == [3]
    tree = ts.at(25.0)
    assert sorted(tree.samples(segs[0]["node_id"])) == [0, 1]
    # Without the remap, raw ID 4 is the root in the simplified ts (3/3)
    # and raw ID 3 is the {0, 1} ancestor at position 75 (2/3).
    flux = filter_flux(out_log)
    assert samples_converted_at(flux, ts, 25.0) == 2 / 3
    assert samples_converted_at(flux, ts, 75.0) == 0.0
    # Non-flux records are untouched.
    assert out_log[0] == _log()[0]


def test_remap_identity_and_out_of_range():
    log = _log()
    remap_event_log_nodes(log, np.arange(6))
    assert [s["node_id"] for s in log[1]["tract_segments"]] == [4, 3]
    log = _log()
    log[1]["tract_segments"][1]["node_id"] = 99
    remap_event_log_nodes(log, np.arange(6))
    assert [s["node_id"] for s in log[1]["tract_segments"]] == [4]
    assert remap_event_log_nodes(None, np.arange(6)) is None
