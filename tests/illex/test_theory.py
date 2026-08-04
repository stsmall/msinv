"""Theory tests. The closed-form <-> numerical agreement test is the important
one: it is what catches algebra errors in the E[T] derivations."""
import numpy as np
import pytest

from illex import theory


def test_panmictic_constant_ne_is_2ne():
    """For constant Ne, mean pairwise coalescence time is exactly 2*Ne."""
    got = theory.expected_times(theory.N_const, t_inv=1.0e6)["panmictic"]
    assert got == pytest.approx(2 * theory.NE_CONST, rel=0.002)


def test_growth_panmictic_reproduces_observed_pi():
    """The moments growth model must be self-consistent with observed pi."""
    et = theory.expected_times(theory.N_growth, t_inv=1.0e6)["panmictic"]
    pi = 2 * theory.MU * et
    assert pi == pytest.approx(0.00930, abs=0.0002)


@pytest.mark.parametrize("t_inv", [3.0e5, 9.0e5, 1.5e6, 3.0e6])
def test_closed_form_matches_numerical_integration(t_inv):
    """Independent implementations of the same quantity must agree.

    This is the regression test for the E[T_S] algebra slip: the integral
    contributes -t*exp(-t/tau_S), which cancels the +t carried in by survivors
    entering the ancestral population.
    """
    num = theory.expected_times(theory.N_const, t_inv)
    closed = theory.const_closed_form(theory.NE_CONST, t_inv)
    for key in ("within_i", "within_s", "between"):
        assert num[key] == pytest.approx(closed[key], rel=0.01), key


def test_young_inversion_gives_large_dxy_ratio():
    """Single-origin bottleneck drives pi_I -> 0, so young inversions give a
    LARGE dxy/pi_I, not a small one."""
    r = theory.ratios(theory.N_growth, t_inv=2.0e5)
    assert r["dxy_over_pi_i"] > 7.0


def test_old_inversion_pi_ratio_approaches_frequency_ratio():
    """As t_inv -> infinity both classes equilibrate, so pi_I/pi_S -> p_I/p_S."""
    r = theory.ratios(theory.N_const, t_inv=2.0e7)
    assert r["pi_i_over_pi_s"] == pytest.approx(0.626 / 0.374, rel=0.05)


def test_dxy_floor_values_match_spec():
    """The floors the spec commits to, and that harness test 1 checks msinv against."""
    g_floor, g_at = theory.dxy_floor(theory.N_growth)
    c_floor, c_at = theory.dxy_floor(theory.N_const)
    assert g_floor == pytest.approx(2.563, abs=0.02)
    assert c_floor == pytest.approx(3.978, abs=0.02)
    assert g_at == pytest.approx(1.136e6, rel=0.05)
    assert c_at == pytest.approx(1.344e6, rel=0.05)
    assert g_floor < c_floor, "growth must lower the floor vs constant Ne"


def test_solve_t_inv_matches_spec():
    """t_inv implied by the observed pi_I/pi_S = 0.744."""
    got_g = theory.solve_t_inv(theory.N_growth, 0.744)
    assert got_g == pytest.approx(952_984, rel=0.03)
    got_c = theory.solve_t_inv(theory.N_const, 0.744)
    assert got_c == pytest.approx(896_340, rel=0.03)


def test_observed_is_below_growth_floor():
    """The flux claim: observed 1.846 sits below the growth floor, by 1.39x."""
    floor, _ = theory.dxy_floor(theory.N_growth)
    assert 1.846 < floor
    assert floor / 1.846 == pytest.approx(1.39, abs=0.05)
