import numpy as np
import pytest

from illex import demography, theory


def test_growth_schedule_endpoints():
    """Schedule must start at N0 and reach N_ANC at T_GROW."""
    t, ne = demography.growth_ne_schedule(t_max=2.0e6)
    assert ne[0] == pytest.approx(theory.N0, rel=0.01)
    i = int(np.argmin(np.abs(t - theory.T_GROW)))
    assert ne[i] == pytest.approx(theory.N_ANC, rel=0.02)


def test_growth_schedule_flat_before_growth():
    """Before T_GROW (deeper in the past) Ne is constant at N_ANC."""
    t, ne = demography.growth_ne_schedule(t_max=2.0e6)
    deep = ne[t > theory.T_GROW * 1.2]
    assert np.allclose(deep, theory.N_ANC, rtol=0.02)


def test_growth_schedule_matches_theory_N_growth():
    """The schedule handed to msinv must be the same N(t) theory integrates."""
    t, ne = demography.growth_ne_schedule(t_max=2.0e6)
    assert np.allclose(ne, theory.N_growth(t), rtol=1e-6)


def test_builders_return_msinv_demography():
    from msinv import Demography
    assert isinstance(demography.growth_demography(), Demography)
    assert isinstance(demography.constant_demography(), Demography)


def test_constant_demography_present_size():
    assert demography.PRESENT_NE_CONST == pytest.approx(775_000.0)


def test_growth_demography_trajectory_matches_theory():
    """The encoded eg/en events must reproduce theory.N_growth(t).

    Demography.size_at() is a LIVE state query -- it does not replay
    events past t, only what has already been applied via
    apply_event_at(). So events must be applied progressively, forward
    in backward-time order, exactly as the simulator would:
      - apply_event_at(0.0, ...) activates the eg growth-rate event.
      - apply_event_at(T_GROW, ...) activates the en size-pin event,
        which also resets the growth rate to 0 (see
        rust/msinv-core/src/demography.rs apply_events_at), flattening
        the trajectory for all deeper times.

    A flipped ALPHA sign, swapped eg/en order, or wrong pop index would
    all fail this test while still passing
    test_builders_return_msinv_demography's bare isinstance() check.
    """
    d = demography.growth_demography()

    d.apply_event_at(0.0, [])
    for t in (0.0, theory.T_GROW / 2.0):
        assert d.size_at(0, t) == pytest.approx(theory.N_growth(t), rel=1e-9)

    d.apply_event_at(theory.T_GROW, [])
    for t in (theory.T_GROW, theory.T_GROW * 1.5):
        assert d.size_at(0, t) == pytest.approx(theory.N_growth(t), rel=1e-9)


def test_constant_demography_trajectory_matches_theory():
    """constant_demography() has no events; size stays NE_CONST at all t."""
    d = demography.constant_demography()

    d.apply_event_at(0.0, [])
    for t in (0.0, 1.0e5, 1.0e6):
        assert d.size_at(0, t) == pytest.approx(theory.NE_CONST, rel=1e-9)
