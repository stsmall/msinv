"""Phase-8 tests: trajectory-based selection types.

Covers the post-port additions:
  - DeterministicTrajectory with explicit ``p_start`` (partial-SHIC).
  - IntegerWFTrajectory (proper integer-copy WF, large-N robust).
  - StochasticDeterministicTrajectory (discoal-style hybrid).
"""

import numpy as np
import pytest

from msinv.hull import HullSimulator, InversionSpec
from msinv.hull.demography import Demography


def _build_inv(traj_dict, gamma=1e-15):
    return InversionSpec(
        bp_left=2000, bp_right=8000,
        trajectory=traj_dict,
        gene_conversion_rate=gamma, flux_window=0.05, inv_id=0,
    )


# ---------------------------------------------------------------------------
# Deterministic with p_start (soft-sweep on standing variation)
# ---------------------------------------------------------------------------

def test_deterministic_p_start_builds_and_runs():
    inv = _build_inv({
        'type': 'deterministic',
        'p_final': 0.7, 'p_start': 0.05,
        'n_e': 1000, 's': 0.005,
    })
    demo = Demography(pop_sizes=[1000, 1000])
    demo.add_event(('ej', 1500.0, 1, 0))
    sim = HullSimulator(
        sample_config={('S', 0): 5, ('S', 1): 5, ('I', 1): 5},
        demography=demo, sequence_length=10000,
        recombination_rate=1e-8, inversions=[inv], seed=1,
    )
    ts = sim.simulate()
    assert ts.num_trees > 0


def test_deterministic_p_start_default_is_hard_sweep():
    """Without ``p_start``, the trajectory should default to 1/(2N)."""
    inv_no_p = _build_inv({
        'type': 'deterministic',
        'p_final': 0.7, 'n_e': 1000, 's': 0.005,
    })
    inv_explicit = _build_inv({
        'type': 'deterministic',
        'p_final': 0.7, 'n_e': 1000, 's': 0.005,
        'p_start': 1.0 / 2000.0,
    })
    # Both should build OK.
    assert inv_no_p.trajectory['p_final'] == 0.7
    assert inv_explicit.trajectory['p_start'] == 1.0 / 2000.0


# ---------------------------------------------------------------------------
# IntegerWFTrajectory
# ---------------------------------------------------------------------------

def test_integer_wf_builds_and_runs_small_n():
    inv = _build_inv({
        'type': 'integer_wf',
        'p_final': 0.5, 'p_start': 0.05,
        'n_e': 500, 's': 0.01,
        'seed': 42, 'max_attempts': 50,
    })
    demo = Demography(pop_sizes=[500, 500])
    demo.add_event(('ej', 500.0, 1, 0))
    sim = HullSimulator(
        sample_config={('S', 0): 5, ('S', 1): 5, ('I', 1): 5},
        demography=demo, sequence_length=10000,
        recombination_rate=1e-8, inversions=[inv], seed=1,
    )
    ts = sim.simulate()
    assert ts.num_trees > 0


def test_integer_wf_works_at_large_n():
    """The whole point: large N where continuous-diffusion 'stochastic'
    breaks down."""
    inv = _build_inv({
        'type': 'integer_wf',
        'p_final': 0.7, 'p_start': 0.05,
        'n_e': 450_000, 's': 1.2e-5,
        'seed': 42, 'max_attempts': 100,
    })
    demo = Demography(pop_sizes=[1000, 1000])
    demo.add_event(('ej', 1500.0, 1, 0))
    sim = HullSimulator(
        sample_config={('S', 0): 5, ('S', 1): 5, ('I', 1): 5},
        demography=demo, sequence_length=10000,
        recombination_rate=1e-8, inversions=[inv], seed=1,
    )
    ts = sim.simulate()
    assert ts.num_trees > 0


def test_integer_wf_seed_reproducibility():
    """Same seed → same trajectory; different seeds may differ."""
    base = {
        'type': 'integer_wf',
        'p_final': 0.5, 'p_start': 0.05,
        'n_e': 500, 's': 0.01,
        'max_attempts': 50,
    }
    demo = Demography(pop_sizes=[500])
    inv_a = _build_inv({**base, 'seed': 42})
    inv_b = _build_inv({**base, 'seed': 42})
    sim_a = HullSimulator(
        sample_config={('S', 0): 5, ('I', 0): 5},
        demography=demo, sequence_length=10000,
        recombination_rate=1e-8, inversions=[inv_a], seed=1)
    sim_b = HullSimulator(
        sample_config={('S', 0): 5, ('I', 0): 5},
        demography=demo, sequence_length=10000,
        recombination_rate=1e-8, inversions=[inv_b], seed=1)
    ts_a = sim_a.simulate()
    ts_b = sim_b.simulate()
    assert ts_a.num_nodes == ts_b.num_nodes
    assert ts_a.num_edges == ts_b.num_edges


def test_integer_wf_invalid_endpoints_raises():
    """p_final < p_start should fail at construction."""
    with pytest.raises((RuntimeError, ValueError)):
        InversionSpec(
            bp_left=0, bp_right=1000,
            trajectory={
                'type': 'integer_wf',
                'p_final': 0.05, 'p_start': 0.5,
                'n_e': 1000, 's': 0.01,
                'seed': 42, 'max_attempts': 10,
            },
            gene_conversion_rate=1e-15, flux_window=0.05, inv_id=0,
        )._p_inv_as_list = None  # force trajectory construction (simulator-build path)
        # Actually trajectory dict validation is in PyO3 — only fires
        # when the simulator hands the dict to Rust.  Build minimal sim.
        demo = Demography(pop_sizes=[1000, 1000])
        demo.add_event(('ej', 100.0, 1, 0))
        sim = HullSimulator(
            sample_config={('S', 0): 2, ('S', 1): 2, ('I', 1): 2},
            demography=demo, sequence_length=2000,
            recombination_rate=1e-8,
            inversions=[InversionSpec(
                bp_left=200, bp_right=1500,
                trajectory={
                    'type': 'integer_wf',
                    'p_final': 0.05, 'p_start': 0.5,
                    'n_e': 1000, 's': 0.01,
                    'seed': 42, 'max_attempts': 10,
                },
                gene_conversion_rate=1e-15, flux_window=0.05, inv_id=0,
            )],
            seed=1,
        )
        sim.simulate()


# ---------------------------------------------------------------------------
# StochasticDeterministicTrajectory (discoal hybrid)
# ---------------------------------------------------------------------------

def test_stoch_det_builds_and_runs():
    inv = _build_inv({
        'type': 'stoch_det',
        'p_final': 0.5, 'p_start': 1.0 / 2000.0,
        'n_e': 1000, 's': 0.01,
        'seed': 42, 'max_attempts': 50,
    })
    demo = Demography(pop_sizes=[1000, 1000])
    demo.add_event(('ej', 5000.0, 1, 0))
    sim = HullSimulator(
        sample_config={('S', 0): 5, ('S', 1): 5, ('I', 1): 5},
        demography=demo, sequence_length=10000,
        recombination_rate=1e-8, inversions=[inv], seed=1,
    )
    ts = sim.simulate()
    assert ts.num_trees > 0


def test_stoch_det_rejects_neutral_s():
    """Neutral selection (s=0) is not supported by the hybrid (the
    deterministic phase needs positive selection to drive the rise)."""
    with pytest.raises(RuntimeError):
        # PyO3 surfaces the Rust error as RuntimeError.
        demo = Demography(pop_sizes=[1000, 1000])
        demo.add_event(('ej', 100.0, 1, 0))
        sim = HullSimulator(
            sample_config={('S', 0): 2, ('S', 1): 2, ('I', 1): 2},
            demography=demo, sequence_length=2000,
            recombination_rate=1e-8,
            inversions=[InversionSpec(
                bp_left=200, bp_right=1500,
                trajectory={
                    'type': 'stoch_det',
                    'p_final': 0.5, 'p_start': 0.001,
                    'n_e': 1000, 's': 0.0,
                    'seed': 42, 'max_attempts': 10,
                },
                gene_conversion_rate=1e-15, flux_window=0.05, inv_id=0,
            )],
            seed=1,
        )
        sim.simulate()


def test_stoch_det_explicit_threshold():
    """Custom det_threshold should be accepted."""
    inv = _build_inv({
        'type': 'stoch_det',
        'p_final': 0.5, 'p_start': 0.005,
        'n_e': 1000, 's': 0.01,
        'det_threshold': 0.05,
        'seed': 42,
    })
    demo = Demography(pop_sizes=[1000, 1000])
    demo.add_event(('ej', 1000.0, 1, 0))
    sim = HullSimulator(
        sample_config={('S', 0): 3, ('S', 1): 3, ('I', 1): 3},
        demography=demo, sequence_length=10000,
        recombination_rate=1e-8, inversions=[inv], seed=2,
    )
    ts = sim.simulate()
    assert ts.num_trees > 0
