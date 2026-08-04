"""Theory tests. The closed-form <-> numerical agreement test is the important
one: it is what catches algebra errors in the E[T] derivations."""
import numpy as np
import pytest

from illex import empirical, theory


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
    """Regression check on the closed-form floor computation itself.

    These are the floors of theory.py's single-origin-monophyly model
    (see its module docstring's warning) -- retained here purely as a
    numeric regression guard on dxy_floor()'s implementation. Harness
    test 1 (tests/illex/test_floor_harness.py) no longer checks msinv
    against these values: msinv's trajectory family is larger than this
    model (see I1, task-final-fixes-report.md), so these floors are not
    an msinv acceptance criterion.
    """
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


def test_observed_below_single_founder_bound():
    """Observed dxy/pi_I=1.846 sits below the strict single-founder floor.

    This does NOT imply gene flux (that conclusion was retracted -- see
    progress.md's FINAL WHOLE-BRANCH REVIEW and I1/task-final-fixes-report
    .md). What this test actually shows: the growth-arm floor from
    theory.py's strict single-origin-monophyly model is a CONSERVATIVE
    LOWER BOUND on the true single-founder floor (see theory.py's module
    docstring), and the observed ratio sits below even that conservative
    bound. So this excludes the strict k=1 single-origin-monophyly origin
    specifically -- msinv's own hard-sweep limit (p_start=1/(2*Ne)) in
    fact gives dxy/pi_I=4.75-5.33, well above 1.846 too. It says nothing
    about softer origins (intermediate p_start), which msinv can express
    and theory.py cannot, and which do reproduce 1.846 with zero flux
    (gamma~0) -- see progress.md Task 4's reconnaissance and the
    growth-arm result in the final whole-branch review.
    """
    floor, _ = theory.dxy_floor(theory.N_growth)
    assert empirical.DXY_OVER_PI_I < floor
    assert floor / empirical.DXY_OVER_PI_I == pytest.approx(1.39, abs=0.05)
