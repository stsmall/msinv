"""Smoke tests for HullSimulator.event_log API end-to-end wiring.

Three cases:
  - record_events=False: sim.event_log is None.
  - record_events=True with no events: sim.event_log == [].
  - record_events=True with cmig event: sim.event_log has one cmig record.
"""

import pytest

from msinv.hull import HullSimulator, InversionSpec
from msinv.hull.demography import Demography

from .conftest import NEGLIGIBLE_GAMMA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_minimal_sim(record_events: bool):
    """Smallest valid HullSimulator: panmictic, no inversion, fast run."""
    return HullSimulator(
        samples=6,
        population_size=500.0,
        sequence_length=10000.0,
        recombination_rate=1e-8,
        seed=42,
        record_events=record_events,
    )


def _build_inv(t_inv=5000):
    """Single inversion for cmig tests. Mirrors test_phase4b_class_migration."""
    return InversionSpec(
        bp_left=2000,
        bp_right=8000,
        p_inv={0: 0.0, 1: 0.5},
        t_inv=t_inv,
        gene_conversion_rate=NEGLIGIBLE_GAMMA,
        mean_tract_length=300.0,
        tract_distribution="fixed",
        inv_id=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_event_log_none_when_flag_off():
    sim = _build_minimal_sim(record_events=False)
    sim.simulate()
    assert sim.event_log is None


def test_event_log_empty_list_when_flag_on_no_events():
    """With record_events=True and no scheduled cmig + no flux (gamma~0),
    the log should be allocated but empty."""
    sim = _build_minimal_sim(record_events=True)
    sim.simulate()
    assert sim.event_log == [], (
        f"expected empty list with no events, got {sim.event_log!r}"
    )


def test_event_log_records_cmig():
    """2-pop sim with one scheduled cmig event; confirm it shows up in
    the event log with the expected fields."""
    inv = _build_inv(t_inv=5000)

    d = Demography([1000, 1000])
    d.add_class_migration(
        time=100.0, source=1, dest=0, karyotype="S", inv_id=0, proportion=0.5
    )
    # Safety ej so all lineages eventually coalesce.
    d.add_event(("ej", 10000.0, 1, 0))

    sim = HullSimulator(
        sample_config={("S", 0): 4, ("S", 1): 4, ("I", 1): 4},
        demography=d,
        sequence_length=10000.0,
        recombination_rate=1e-8,
        inversions=[inv],
        seed=7,
        record_events=True,
    )
    sim.simulate()

    assert sim.event_log is not None, "event_log should not be None"
    cmig_recs = [r for r in sim.event_log if r["kind"] == "cmig"]
    assert len(cmig_recs) == 1, (
        f"expected 1 cmig record, got {len(cmig_recs)}: {sim.event_log}"
    )
    r = cmig_recs[0]
    assert r["src"] == 1
    assert r["dst"] == 0
    assert r["inv_id"] == 0
    assert r["n_eligible"] >= 0
    assert 0 <= r["n_moved"] <= r["n_eligible"]
