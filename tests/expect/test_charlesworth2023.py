"""Tier 0 against Charlesworth (2023), Genetics 224:iyad116.

Charlesworth's panmictic model: an inversion (In, index 1) at constant
frequency x, the standard arrangement (St, index 2) at y = 1 - x, and
exchange between arrangements at rate r per site per generation in
heterokaryotypes. Backward, an In lineage moves to St at rate r*y and an St
lineage to In at rate r*x (Appendix, Equations A1-A2 with m = 0). In msinv
that is the per-lineage switching hazard ``switching_coefficient(y) *
f_other``, so r = ``switching_coefficient``. Times are in units of 2N and
rho = 4*N*r.

Results checked:
- Equations 5a-5c, equilibrium T12, T11, T22, and Equation 3,
  x*T11 + y*T22 = 1 (Maruyama invariance).
- Equation 9, rho from X = T11/T22, on the paper's In(3R)P example
  (x = 0.11, X = 0.81 gives rho = 34).
- The approach to equilibrium after an instantaneous sweep of a new
  inversion, against the paper's per-generation recursion (Equation A2,
  m = 0) iterated from T11 = 0, T22 = T12 = 1.
- With no exchange (Supplementary File 1, section 6): T12 grows linearly
  and T11, T22 relax exponentially towards x and y. The exponential forms
  here are derived for msinv's model, not transcribed from the supplement.

TODO: check against Supplementary File 1 when it is available (it was not
retrievable when these tests were written). Confirm (a) the no-exchange
forms, where the paper calls x and y the "rate constants" but these decay
at 1/x and 1/y in units of 2N, and (b) Equation 9, transcribed here from
the Europe PMC full-text XML, against the published PDF.
"""

import math

import pytest

from msinv.expect import ExpectModel, flux_coverage
from msinv.hull import Demography, InversionSpec

BL, BR, Y, ELL = 0.0, 100_000.0, 50_000.0, 100.0


def _model(n, x, r, t_inv, origin):
    """Panmictic model whose switching coefficient at Y is exactly r."""
    kw = {"mean_tract_length": 0.0}  # no exchange
    if r > 0:
        cov = flux_coverage(Y, BL, BR, ELL, "geometric")
        kw = {"gene_conversion_rate": r / cov, "mean_tract_length": ELL}
    inv = InversionSpec(bp_left=BL, bp_right=BR, p_inv=x, t_inv=t_inv, **kw)
    m = ExpectModel(Demography([n]), inv, 1e-8, origin=origin)
    if r > 0:
        assert m.switching_coefficient(Y) == pytest.approx(r, rel=1e-12)
    return m


def _scaled_times(m, n):
    def t(a, b):
        return m.pairwise_times(Y, (0, a), (0, b)).mean / (2 * n)

    return t("I", "I"), t("S", "S"), t("I", "S")


def _eq5(x, rho):
    y = 1 - x
    t12 = 1 + 2 / rho
    t11 = (x * (1 + 2 * y) + rho * x * y) / (1 + rho * x * y)
    t22 = (y * (1 + 2 * x) + rho * x * y) / (1 + rho * x * y)
    return t11, t22, t12


@pytest.mark.parametrize("x", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("rho", [0.4, 4.0, 40.0])
def test_equilibrium_matches_equation_5(x, rho):
    n = 1000.0
    m = _model(n, x, rho / (4 * n), 1e9, "barrier")
    t11, t22, t12 = _scaled_times(m, n)
    e11, e22, e12 = _eq5(x, rho)
    assert t11 == pytest.approx(e11, rel=1e-8)
    assert t22 == pytest.approx(e22, rel=1e-8)
    assert t12 == pytest.approx(e12, rel=1e-8)
    # Equation 3: x*T11 + y*T22 = 1.
    assert x * t11 + (1 - x) * t22 == pytest.approx(1.0, rel=1e-8)


def test_in3rp_example_equation_9():
    """The paper's In(3R)P example: x = 0.11, X = T11/T22 = 0.81.

    Equation 9 as published, rho = X(1 - 2x) / ((1 - X) x y), gives
    rho = 34, but it is not the exact inverse of Equations 5b-5c: at
    rho = 34 they give X = 0.823. Solving 5b-5c exactly,
    rho = (X y (1 + 2x) - x (1 + 2y)) / ((1 - X) x y) = 30.8, about 10%
    lower. Tier 0 reproduces X = 0.81 at the exact inverse.
    """
    x, xx = 0.11, 0.81
    y = 1 - x
    rho_eq9 = xx * (1 - 2 * x) / ((1 - xx) * x * y)
    assert rho_eq9 == pytest.approx(34.0, abs=0.05)
    rho = (xx * y * (1 + 2 * x) - x * (1 + 2 * y)) / ((1 - xx) * x * y)
    assert rho == pytest.approx(30.8, abs=0.05)
    n = 1000.0
    t11, t22, _ = _scaled_times(_model(n, x, rho / (4 * n), 1e9, "barrier"), n)
    assert t11 / t22 == pytest.approx(xx, rel=1e-8)
    t11, t22, _ = _scaled_times(_model(n, x, rho_eq9 / (4 * n), 1e9, "barrier"), n)
    assert t11 / t22 == pytest.approx(0.823, abs=5e-4)


def _recursion(n, x, r, gens):
    """Equation A2 with m = 0, from an instantaneous sweep (generations)."""
    y = 1 - x
    t11, t22, t12 = 0.0, 2.0 * n, 2.0 * n
    for _ in range(gens):
        t11, t22, t12 = (
            1 + (1 - 1 / (2 * n * x) - 2 * r * y) * t11 + 2 * r * y * t12,
            1 + (1 - 1 / (2 * n * y) - 2 * r * x) * t22 + 2 * r * x * t12,
            1 + r * x * t11 + r * y * t22 + (1 - r) * t12,
        )
    return t11 / (2 * n), t22 / (2 * n), t12 / (2 * n)


@pytest.mark.parametrize("rho", [4.0, 40.0])
@pytest.mark.parametrize("tau", [0.05, 0.5, 2.0])
def test_approach_to_equilibrium_matches_recursion(rho, tau):
    """A new inversion swept instantly to x = 0.1, tau * 2N generations
    ago (origin='single'). The recursion is discrete and first order in
    r and 1/N, so agreement is to O(1/N) with N = 10^4."""
    n, x = 10_000.0, 0.1
    r = rho / (4 * n)
    gens = int(tau * 2 * n)
    got = _scaled_times(_model(n, x, r, float(gens), "single"), n)
    want = _recursion(n, x, r, gens)
    for g, w in zip(got, want):
        assert g == pytest.approx(w, rel=2e-3)


@pytest.mark.parametrize("tau", [0.1, 1.0, 3.0])
def test_no_exchange_after_sweep(tau):
    """r = 0: T11 = x(1 - e^{-tau/x}), T22 = y + (1 - y) e^{-tau/y},
    T12 = tau + 1, all in units of 2N."""
    n, x = 1000.0, 0.3
    y = 1 - x
    m = _model(n, x, 0.0, tau * 2 * n, "single")
    t11, t22, t12 = _scaled_times(m, n)
    assert t11 == pytest.approx(x * (1 - math.exp(-tau / x)), rel=1e-6)
    assert t22 == pytest.approx(y + (1 - y) * math.exp(-tau / y), rel=1e-6)
    assert t12 == pytest.approx(tau + 1, rel=1e-6)
