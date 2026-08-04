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
